import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.biasq = bq
        self.biask = bk
        self.biasv = bv

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        
        self.head_dim = hidden_size // num_heads
        assert num_heads % num_kv_heads == 0
        
        self.rope = RoPE(self.head_dim, max_seq_len, theta)
        self.scale = mx.rsqrt(self.head_dim)
        

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape
        q_proj = linear(x, self.wq, self.biasq)   
        k_proj = linear(x, self.wk, self.biask) 
        v_proj = linear(x, self.wv, self.biasv) 
        
        # q 有 num_heads 个头
        '''
            q (batch, len, hidden_size)
            k (batch, len, )
        '''
        q_proj = q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k_proj = k_proj.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v_proj = v_proj.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        
        # 位置编码
        # [batch, H_q ,len, head_dim]
        # [batch, H ,len, head_dim]
        q_proj = self.rope(q_proj, offset=slice(0, seq_len))
        k_proj = self.rope(k_proj, offset=slice(0, seq_len))
        
        q_proj = q_proj.transpose(0, 2, 1, 3)
        k_proj = k_proj.transpose(0, 2, 1, 3)
        v_proj = v_proj.transpose(0, 2, 1, 3)
        
        # [batch, H_q ,len, head_dim]
        o = scaled_dot_product_attention_grouped(q_proj, k_proj, v_proj, scale = self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3)
        o = o.reshape(batch_size, seq_len, -1)
        o = linear(o, self.wo)
        return o

class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        '''
            x:       shape=(BATCH_SIZE, SEQ_LEN, DIM)
            w_gate:  shape=(HIDDEN_DIM, DIM)
            w_up:    shape=(HIDDEN_DIM, DIM)
            w_down:  shape=(DIM, HIDDEN_DIM)
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down


    def __call__(self, x: mx.array) -> mx.array:
        gate_x = linear(x, self.w_gate)
        gate_x = silu(gate_x)
        
        up_x = linear(x, self.w_up)
        
        hidden_x = up_x * gate_x
        
        return linear(hidden_x, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,        
        hidden_size: int,               # d_model
        intermediate_size: int,         # d_ff
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.attn = Qwen2MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads,
                                            wq, wk, wv, wo, bq, bk, bv, max_seq_len, theta)
        
        self.input_layer_norm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.post_attention_layer_norm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)

        

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        residual = x
        x = self.input_layer_norm(x)
        x = self.attn(x, mask)
        x = x + residual
        
        residual = x
        x = self.post_attention_layer_norm(x)
        x = self.mlp(x)
        x = x + residual
        
        return x
    
    

class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        '''
            quantize::dequantize_linear 需要用来反量化参数
        '''
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        precision = mx.float16
        self.precision = precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj)

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq.astype(precision),
                wk=wk.astype(precision),
                wv=wv.astype(precision),
                wo=wo.astype(precision),
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate.astype(precision),
                w_up=w_up.astype(precision),
                w_down=w_down.astype(precision),
                w_input_layernorm=mlx_model.model.layers[
                    i
                ].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model


    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, mask="causal")
        h = self.norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
