from typing import Sequence

import torch as th


def unsqueeze_as(x, y) -> th.Tensor:
    if isinstance(y, th.Tensor):
        d = y.dim()
    else:
        d = len(y)
    return x.view(list(x.shape) + [1] * (d - x.dim()))


def cumprod(x: Sequence):
    out = 1
    for v in x:
        out *= v
    return out
