import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        '''
            data:     shape=(SIZE, SIZE_Y)
            weight:   shape=(SIZE_Y,)
        '''
        self.weight = weight
        self.dim = dim
        self.eps = eps


    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        x_squared = mx.square(x)
        mean_squared = mx.mean(x_squared, axis=-1, keepdims=True)
        rms = mx.sqrt(mean_squared + self.eps)
        
        x =  x / rms
        return x * self.weight
    