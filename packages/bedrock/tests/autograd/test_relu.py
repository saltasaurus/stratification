"""Tests for Tensaur.relu forward and backward."""

import numpy as np
import pytest
from bedrock.autograd.tensaur import Tensaur


def test_relu_hand_values_and_grad_at_zero():
    """ReLU([-2, 0, 3]) is [0, 0, 3]; ones upstream give grad [0, 0, 1].

    The slope at exactly 0 is defined as 0, matching PyTorch. Catches a
    backward mask of ``(z >= 0)``, which would send grad 1 through the zero.
    """
    x = Tensaur([-2.0, 0.0, 3.0], requires_grad=True)
    out = x.relu()

    assert out.requires_grad is True
    np.testing.assert_array_equal(out.data, [0.0, 0.0, 3.0])

    out.backward()
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 1.0])


def test_relu_keeps_2d_shape():
    """Elementwise ReLU must not reduce axes the way ``np.max(data, 0)`` does."""
    x = Tensaur([[-2.0, 3.0], [1.0, -4.0]], requires_grad=True)
    out = x.relu()

    assert out.data.shape == (2, 2)
    np.testing.assert_array_equal(out.data, [[0.0, 3.0], [1.0, 0.0]])

    out.backward()
    np.testing.assert_array_equal(x.grad, [[0.0, 1.0], [1.0, 0.0]])


def test_relu_scales_by_upstream_grad():
    """Local slope is 0 or 1, so the incoming grad should pass through unchanged on positives."""
    x = Tensaur([-2.0, 0.0, 3.0], requires_grad=True)
    out = x.relu()
    out.backward(grad=np.array([2.0, 5.0, 7.0]))
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 7.0])


def test_relu_frozen_input_writes_no_grad():
    """``requires_grad=False`` must not crash and must leave ``.grad`` at zeros."""
    values = np.array([-2.0, 0.0, 3.0])
    x = Tensaur(values, requires_grad=False)
    out = x.relu()

    assert out.requires_grad is False
    np.testing.assert_array_equal(out.data, [0.0, 0.0, 3.0])

    # Exercise the local guard without defining frozen-root backward policy.
    out.grad = np.ones_like(out.data)
    out._backward()
    np.testing.assert_array_equal(x.grad, np.zeros_like(values))


def test_relu_matches_torch():
    """Forward and sum-backward should match ``torch.relu`` on a random 2D array."""
    torch = pytest.importorskip("torch")
    values = np.random.default_rng(0).normal(size=(4, 5))

    x = Tensaur(values, requires_grad=True)
    out = x.relu()
    out.sum().backward()

    tx = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    tout = torch.relu(tx)
    tout.sum().backward()

    np.testing.assert_allclose(out.data, tout.detach().numpy())
    np.testing.assert_allclose(x.grad, tx.grad.numpy())
