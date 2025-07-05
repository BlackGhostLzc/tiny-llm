import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    o = mx.matmul(x, mx.transpose(w, (-1, -2)))
    if bias is not None:
        o = o + bias
    return o


def silu(x: mx.array) -> mx.array:
    pass
