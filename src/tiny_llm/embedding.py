import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        if weight is None:
            # 正态分布初始化
            weight = mx.random.normal(shape=(vocab_size, embedding_dim)) * (1.0 / embedding_dim**0.5)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight  # shape: [vocab_size, embedding_dim]
        

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]
    

    def as_linear(self, x: mx.array) -> mx.array:
        return x @ self.weight.T

