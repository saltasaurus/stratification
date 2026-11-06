"""requires_grad integration tests for bce_mean.

Imports the real bedrock package. Complements (does not replace) the earlier
normalization/clipping regressions. Labels are deliberately excluded from the
loss graph regardless of y.requires_grad.
"""

from __future__ import annotations

import numpy as np
import pytest
from bedrock.autograd.tensaur import Tensaur
from bedrock.loss.BCE import bce_mean


def T(data, *, requires_grad=True):
    return Tensaur(data, requires_grad=requires_grad)


@pytest.mark.parametrize(
    "p_rg,y_rg,expected",
    [
        (False, False, False),
        (False, True, False),  # labels never force loss.requires_grad
        (True, False, True),
        (True, True, True),
    ],
)
def test_bce_output_flag_follows_predictions_only(p_rg, y_rg, expected):
    p = T([[0.2], [0.8], [0.4]], requires_grad=p_rg)
    y = T([[0.0], [1.0], [1.0]], requires_grad=y_rg)
    loss = bce_mean(p, y)
    assert loss.requires_grad is expected
    # Labels are excluded from the loss graph regardless of y.requires_grad.
    assert y not in loss._prev


def test_bce_labels_and_their_source_graph_receive_no_grad():
    """y comes from a real sigmoid graph; BCE must not touch y or src."""
    p = T([[0.25], [0.75]], requires_grad=True)
    src = T([[0.5], [-0.5]], requires_grad=True)  # moderate logits
    y = src.sigmoid()  # soft targets in (0, 1); real connected graph
    assert y.requires_grad is True
    sentinel = np.array([[3.0], [4.0]])
    y.grad = sentinel.copy()

    bce_mean(p, y).backward()
    np.testing.assert_array_equal(y.grad, sentinel)
    np.testing.assert_array_equal(src.grad, np.zeros_like(src.data))
    assert not np.allclose(p.grad, 0.0)


def test_bce_frozen_predictions_all_frozen_output_flag():
    """p frozen + y trainable => loss.requires_grad False (no public-backward policy)."""
    p = T([[0.3], [0.6]], requires_grad=False)
    y = T([[1.0], [0.0]], requires_grad=True)
    loss = bce_mean(p, y)
    assert loss.requires_grad is False
    np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))


def test_bce_local_backward_guard_when_predictions_frozen():
    """Seed loss.grad and call the private closure; frozen p must stay untouched."""
    p = T([[0.2], [0.9]], requires_grad=False)
    y = T([[1.0], [0.0]], requires_grad=False)
    sentinel = np.array([[5.0], [6.0]])
    p.grad = sentinel.copy()
    loss = bce_mean(p, y)
    assert loss.requires_grad is False
    loss.grad = np.ones_like(loss.data)
    loss._backward()
    np.testing.assert_array_equal(p.grad, sentinel)


def test_bce_through_sigmoid_matches_logistic_grad():
    """For moderate logits z: d/dz mean BCE(sigmoid(z), y) = (sigmoid(z) - y) / n."""
    z_data = np.array([[0.5], [-1.0], [2.0], [0.0]])
    y_data = np.array([[1.0], [0.0], [1.0], [0.0]])
    z = T(z_data, requires_grad=True)
    y = T(y_data, requires_grad=False)
    bce_mean(z.sigmoid(), y).backward()
    s = 1.0 / (1.0 + np.exp(-z_data))
    expected = (s - y_data) / z_data.size
    np.testing.assert_allclose(z.grad, expected, rtol=1e-6, atol=1e-9)


def test_bce_ndarray_labels_via_as_tensaur_stay_frozen():
    p = T([[0.4], [0.6]], requires_grad=True)
    y_raw = np.array([[1.0], [0.0]])
    loss = bce_mean(p, y_raw)  # type: ignore
    assert loss.requires_grad is True
    loss.backward()
    assert not np.allclose(p.grad, 0.0)
