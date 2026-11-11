import numpy as np
import pytest
from bedrock.autograd.tensaur import Tensaur
from bedrock.optim import SGD


def make_param(value):
    return Tensaur(np.array([value], dtype=float), requires_grad=True)


def apply_gradients(optimizer, param, gradients):
    """Run one step per gradient, zeroing between steps like a real loop."""
    history = []
    for g in gradients:
        optimizer.zero_grad()
        param.grad[...] = g
        optimizer.step()
        history.append(param.data.item())
    return history


def test_momentum_matches_hand_calculation():
    # v1 = 2         -> p = 1.0 - 0.1*2   = 0.8
    # v2 = 0.9*2 + 2 -> p = 0.8 - 0.1*3.8 = 0.42
    # The second value also checks that velocity survives zero_grad.
    p = make_param(1.0)
    opt = SGD([p], lr=0.1, momentum=0.9)

    assert apply_gradients(opt, p, [2.0, 2.0]) == pytest.approx([0.8, 0.42])


def test_zero_momentum_is_plain_sgd():
    p = make_param(1.0)
    opt = SGD([p], lr=0.1, momentum=0.0)

    assert apply_gradients(opt, p, [2.0, 2.0]) == pytest.approx([0.8, 0.6])


def test_each_parameter_keeps_its_own_velocity():
    a = make_param(1.0)
    b = make_param(1.0)
    opt = SGD([a, b], lr=0.1, momentum=0.9)

    for grad_a, grad_b in [(2.0, 0.0), (2.0, 0.0)]:
        opt.zero_grad()
        a.grad[...] = grad_a
        b.grad[...] = grad_b
        opt.step()

    assert a.data.item() == pytest.approx(0.42)
    assert b.data.item() == pytest.approx(1.0)  # never got a gradient
