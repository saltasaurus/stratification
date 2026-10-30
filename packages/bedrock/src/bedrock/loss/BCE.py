import numpy as np

from bedrock.autograd.tensaur import Tensaur, as_tensaur, collapse

EPS = 1e-12


def bce_mean(p: Tensaur, y: Tensaur) -> Tensaur:

    y = as_tensaur(y)

    p_clamp = p.clamp(EPS, 1 - EPS)
    y_clamp = y.clamp(EPS, 1 - EPS)
    n = p.data.shape[0]  # Check batch

    # loss = -mean(y*log(p) + (1-y)(log(1-p)))
    loss_val = -np.mean(y_clamp.data * np.log(p_clamp.data) + (1 - y_clamp.data) * np.log(1 - p_clamp.data))
    out = Tensaur(loss_val, (p, y), _op="bce_mean")

    def _backward():
        # dL/dp_i = (p - y) / (n * p * (1-p))
        denom = n * p.data * (1 - p.data) + EPS
        p.grad += collapse(out.grad * ((p.data - y.data) / denom), p.grad.shape)

    out._backward = _backward
    return out
