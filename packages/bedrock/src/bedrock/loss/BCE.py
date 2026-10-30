import numpy as np
from bedrock.autograd.tensaur import Tensaur, as_tensaur, collapse

EPS = 1e-12


def bce_mean(p: Tensaur, y: Tensaur) -> Tensaur:

    y = as_tensaur(y)

    p_data = Tensaur(np.clip(p.data, EPS, 1 - EPS), (p,), _op="clip")
    y_data = Tensaur(np.clip(y.data, EPS, 1 - EPS), (y,), _op="clip")
    n = p.data.size  # Check batch

    # loss = -mean(y*log(p) + (1-y)(log(1-p)))
    term = (y_data * p_data.log()) + (1 - y) * (1 - p_data).log()
    loss = -term.sum() / n

    def _backward():
        # dL/dp_i = (p - y) / (n * p * (1-p))
        denom = n * p.data * (1 - p.data) + EPS
        p.grad += collapse(loss.grad * ((p.data - y.data) / denom), p.grad.shape)

    loss._backward = _backward

    return loss
