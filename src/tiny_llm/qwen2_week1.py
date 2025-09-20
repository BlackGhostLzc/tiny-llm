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
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
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
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
