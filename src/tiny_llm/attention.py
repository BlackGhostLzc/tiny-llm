import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # query,key,value接受的参数是(*batch, DIM_L, DIM_D), 比如(2,3,4,5)
    d_k = query.shape[-1]
    k = key.swapaxes(-1, -2)
    factor = 0
    if scale is None:
        factor = mx.rsqrt(d_k)
    else:
        factor = scale
    scores = mx.matmul(query, k) * factor
    if mask is not None:
        scores = scores + mask
    p_attn = mx.softmax(scores, axis=-1)
    return mx.matmul(p_attn, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        执行多头注意力计算
        
        参数:
        query: 查询张量, 形状为 [batch_size, seq_len_q, hidden_size]
        key: 键张量, 形状为 [batch_size, seq_len_k, hidden_size]
        value: 值张量, 形状为 [batch_size, seq_len_v, hidden_size]
        mask: 可选的掩码张量, 形状为 [batch_size, seq_len_q, seq_len_k]
        
        返回:
        注意力输出张量, 形状为 [batch_size, seq_len_q, hidden_size]
        """
        batch_size = query.shape[0]
        len = query.shape[1]
        
        # 1. 线性投影
        Q_proj = linear(query, self.wq)  # [batch_size, seq_len_q, hidden_size]
        K_proj = linear(key, self.wk)    # [batch_size, seq_len_k, hidden_size]
        V_proj = linear(value, self.wv)  # [batch_size, seq_len_v, hidden_size]
        
        # 2. 分割多头
        # reshape 重塑为 [batch_size, seq_len, num_heads, head_dim]
        # transpose 重塑为 [batch_size, num_heads, seq_len, head_dim]
        Q_proj = Q_proj.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K_proj = K_proj.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V_proj = V_proj.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        x = scaled_dot_product_attention_simple(
            Q_proj,
            K_proj,
            V_proj,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, len, self.hidden_size)
        return linear(x, self.wo)




def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
