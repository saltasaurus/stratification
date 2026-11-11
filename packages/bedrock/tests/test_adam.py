import numpy as np
import pytest
from bedrock.autograd.tensaur import Tensaur
from bedrock.optim import Adam


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


def test_first_step_bias_correction_matches_hand_values():
    # g = 2: m1 = 0.2, v1 = 0.004 -> m_hat = 2, v_hat = 4
    # step = lr * 2 / sqrt(4) = lr, so p = 1.0 - 0.1 = 0.9
    # Without bias correction the step would be ~3.16 * lr (p ~ 0.684).
    p = make_param(1.0)
    opt = Adam([p], lr=0.1)

    assert apply_gradients(opt, p, [2.0]) == pytest.approx([0.9])
    assert opt.m[0] / (1 - 0.9) == pytest.approx([2.0])
    assert opt.v[0] / (1 - 0.999) == pytest.approx([4.0])


@pytest.mark.parametrize("g", [0.001, 2.0, 1000.0])
def test_first_step_size_is_lr_regardless_of_gradient_scale(g):
    # Adam normalizes by sqrt(v_hat), so step 1 moves exactly lr * sign(g).
    p = make_param(1.0)
    opt = Adam([p], lr=0.1)

    assert apply_gradients(opt, p, [g]) == pytest.approx([0.9], abs=1e-6)


def test_constant_gradient_keeps_step_size_lr_after_many_steps():
    # Catches correction using (1 - beta) instead of (1 - beta**t):
    # with a constant gradient, m_hat / sqrt(v_hat) stays 1 every step.
    p = make_param(0.0)
    opt = Adam([p], lr=0.01)

    history = apply_gradients(opt, p, [3.0] * 5)
    assert history == pytest.approx([-0.01, -0.02, -0.03, -0.04, -0.05], abs=1e-6)


def test_each_parameter_keeps_its_own_moments():
    a, b = make_param(1.0), make_param(1.0)
    opt = Adam([a, b], lr=0.1)

    opt.zero_grad()
    a.grad[...] = 2.0
    b.grad[...] = -5.0
    opt.step()

    assert a.data.item() == pytest.approx(0.9)
    assert b.data.item() == pytest.approx(1.1)
    assert opt.m[0] == pytest.approx([0.2])
    assert opt.m[1] == pytest.approx([-0.5])


def test_matches_torch_adam_over_several_steps():
    torch = pytest.importorskip("torch")
    gradients = [0.5, -1.0, 2.0, 0.1, -0.3]

    p = make_param(1.0)
    ours = apply_gradients(Adam([p], lr=0.1), p, gradients)

    tp = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    topt = torch.optim.Adam([tp], lr=0.1)
    theirs = []
    for g in gradients:
        topt.zero_grad()
        tp.grad = torch.tensor([g], dtype=torch.float64)
        topt.step()
        theirs.append(tp.item())

    assert ours == pytest.approx(theirs, rel=1e-9)