import numpy as np

from bedrock.autograd.tensaur import Tensaur, as_tensaur, collapse

EPS = 1e-12


def bce_mean(p: Tensaur, y: Tensaur) -> Tensaur:

    y = as_tensaur(y)

    p_clamp = p.clamp(EPS, 1 - EPS)
    y_data = y.data
    n = p.data.size

    # loss = -mean(y*log(p) + (1-y)(log(1-p)))
    loss_val = -np.mean(y_data * np.log(p_clamp.data) + (1 - y_data) * np.log(1 - p_clamp.data))
    out = Tensaur(loss_val, (p,), _op="bce_mean")

    def _backward():
        pc = p_clamp.data
        inside = (p.data >= EPS) & (p.data <= 1 - EPS)

        # dL/dp_i = (p - y) / (n * p * (1-p))
        local_grad = (pc - y_data) / (n * pc * (1 - pc))
        p.grad += collapse(out.grad * inside * local_grad, p.grad.shape)

    out._backward = _backward
    return out
