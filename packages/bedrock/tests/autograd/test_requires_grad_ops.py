"""Regression tests for Tensaur.requires_grad on public autograd ops.

Import the real bedrock package. These tests encode the taught requires_grad
contract (flag OR on binaries, unary mirrors input, per-input grad guards,
as_tensaur wraps constants as frozen). They do not prove an unseen
implementation is fully correct; run the rest of the suite too.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from bedrock.autograd.tensaur import Tensaur, as_tensaur


def T(data, *, requires_grad=True):
    return Tensaur(data, requires_grad=requires_grad)


def _graph_leaves(root):
    seen, leaves = set(), []
    stack = [root]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        if not v._prev:
            leaves.append(v)
        else:
            stack.extend(v._prev)
    return leaves


# ---------------------------------------------------------------------------
# as_tensaur / constructor
# ---------------------------------------------------------------------------


def test_as_tensaur_preserves_existing_object_and_flag():
    x = T([1.0, 2.0], requires_grad=False)
    y = as_tensaur(x)
    assert y is x
    assert y.requires_grad is False


def test_as_tensaur_wraps_python_and_ndarray_as_frozen():
    for raw in (3.0, [1.0, 2.0], np.array([[1.0, 2.0]])):
        t = as_tensaur(raw)
        assert isinstance(t, Tensaur)
        assert t.requires_grad is False
        np.testing.assert_allclose(t.data, np.asarray(raw, dtype=float))


def test_constructor_default_requires_grad_true_and_zero_grad():
    x = Tensaur([1.0, -2.0])
    assert x.requires_grad is True
    np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))


# ---------------------------------------------------------------------------
# Binary flag OR + per-input guards (add / mul / matmul)
# ---------------------------------------------------------------------------

BINARY_FLAG_CASES = list(itertools.product([False, True], repeat=2))


@pytest.mark.parametrize("a_rg,b_rg", BINARY_FLAG_CASES)
def test_add_output_flag_is_or(a_rg, b_rg):
    a = T([[1.0, 2.0], [3.0, 4.0]], requires_grad=a_rg)
    b = T([[0.5, -1.0], [2.0, 0.0]], requires_grad=b_rg)
    assert (a + b).requires_grad is (a_rg or b_rg)


@pytest.mark.parametrize("a_rg,b_rg", BINARY_FLAG_CASES)
def test_mul_output_flag_is_or(a_rg, b_rg):
    a = T([[1.0, 2.0], [3.0, 4.0]], requires_grad=a_rg)
    b = T([[0.5, -1.0], [2.0, 0.5]], requires_grad=b_rg)
    assert (a * b).requires_grad is (a_rg or b_rg)


@pytest.mark.parametrize("a_rg,b_rg", BINARY_FLAG_CASES)
def test_matmul_output_flag_is_or(a_rg, b_rg):
    a = T([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=a_rg)
    b = T([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=b_rg)
    assert (a @ b).requires_grad is (a_rg or b_rg)


@pytest.mark.parametrize("a_rg,b_rg", BINARY_FLAG_CASES)
def test_add_grad_guards_and_values(a_rg, b_rg):
    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b_data = np.array([[0.5, -1.0, 2.0]])  # broadcast (1, 3)
    weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a, b = T(a_data, requires_grad=a_rg), T(b_data, requires_grad=b_rg)
    out = a + b
    if not out.requires_grad:
        assert out.requires_grad is False
        return
    out.backward(grad=weights)
    if a_rg:
        np.testing.assert_allclose(a.grad, weights)
    else:
        np.testing.assert_array_equal(a.grad, np.zeros_like(a.data))
    if b_rg:
        np.testing.assert_allclose(b.grad, weights.sum(axis=0, keepdims=True))
    else:
        np.testing.assert_array_equal(b.grad, np.zeros_like(b.data))


@pytest.mark.parametrize("a_rg,b_rg", BINARY_FLAG_CASES)
def test_mul_grad_guards_and_values(a_rg, b_rg):
    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b_data = np.array([[0.5, -1.0, 2.0]])  # broadcast (1, 3)
    weights = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    a, b = T(a_data, requires_grad=a_rg), T(b_data, requires_grad=b_rg)
    out = a * b
    if not out.requires_grad:
        assert out.requires_grad is False
        return
    out.backward(grad=weights)
    if a_rg:
        np.testing.assert_allclose(a.grad, np.broadcast_to(b_data, a_data.shape) * weights)
    else:
        np.testing.assert_array_equal(a.grad, np.zeros_like(a.data))
    if b_rg:
        np.testing.assert_allclose(b.grad, (a_data * weights).sum(axis=0, keepdims=True))
    else:
        np.testing.assert_array_equal(b.grad, np.zeros_like(b.data))


@pytest.mark.parametrize("a_rg,b_rg", BINARY_FLAG_CASES)
def test_matmul_grad_guards_and_values(a_rg, b_rg):
    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # (3, 2)
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    a, b = T(a_data, requires_grad=a_rg), T(b_data, requires_grad=b_rg)
    out = a @ b
    if not out.requires_grad:
        assert out.requires_grad is False
        return
    out.backward(grad=weights)
    if a_rg:
        np.testing.assert_allclose(a.grad, weights @ b_data.T)
    else:
        np.testing.assert_array_equal(a.grad, np.zeros_like(a.data))
    if b_rg:
        np.testing.assert_allclose(b.grad, a_data.T @ weights)
    else:
        np.testing.assert_array_equal(b.grad, np.zeros_like(b.data))


def test_add_wrong_second_operand_guard_would_fail_here():
    """Frozen second operand must not receive grad even when first is trainable."""
    a = T([2.0, 3.0], requires_grad=True)
    b = T([4.0, 5.0], requires_grad=False)
    sentinel = np.array([7.0, 8.0])
    b.grad = sentinel.copy()
    (a + b).backward(grad=np.ones_like(a.data))
    np.testing.assert_array_equal(b.grad, sentinel)
    np.testing.assert_allclose(a.grad, np.ones_like(a.data))


# ---------------------------------------------------------------------------
# Scalar / reflected ops: both flags, frozen constant leaves, numerical grads
# ---------------------------------------------------------------------------

SCALAR_CASES = [
    ("radd", lambda x: 2.0 + x, lambda x, w: w),
    ("rmul", lambda x: 3.0 * x, lambda x, w: 3.0 * w),
    ("rsub", lambda x: 5.0 - x, lambda x, w: -w),
    ("div_scalar", lambda x: x / 2.0, lambda x, w: w / 2.0),
    ("rdiv", lambda x: 4.0 / x, lambda x, w: -4.0 / (x**2) * w),
    ("sub_scalar", lambda x: x - 1.5, lambda x, w: w),
    ("pow3", lambda x: x**3, lambda x, w: 3.0 * (x**2) * w),
    ("neg", lambda x: -x, lambda x, w: -w),
]


@pytest.mark.parametrize("rg", [False, True])
@pytest.mark.parametrize("name,make_out,d_dx", SCALAR_CASES, ids=[c[0] for c in SCALAR_CASES])
def test_scalar_reflected_flag_leaves_and_grads(rg, name, make_out, d_dx):
    x_data = np.array([2.0, 3.0])
    weights = np.array([0.5, -1.0])
    x = T(x_data, requires_grad=rg)
    out = make_out(x)
    assert out.requires_grad is rg
    for leaf in _graph_leaves(out):
        if leaf is not x:
            assert leaf.requires_grad is False
    if not rg:
        return
    out.backward(grad=weights)
    np.testing.assert_allclose(x.grad, d_dx(x_data, weights), rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Unary ops: flag mirroring, enabled numerical grads, frozen local guards
# ---------------------------------------------------------------------------

# Interior clamp points only — avoid exact thresholds where the mask is undefined.
UNARY_CASES = [
    (
        "neg",
        lambda x: -x,
        np.array([1.0, -2.0, 0.5]),
        np.array([0.5, 1.0, -1.0]),
        lambda x, w: -w,
    ),
    (
        "pow3",
        lambda x: x**3,
        np.array([1.0, -2.0, 0.5]),
        np.array([0.5, 1.0, -1.0]),
        lambda x, w: 3.0 * (x**2) * w,
    ),
    (
        "sigmoid",
        lambda x: x.sigmoid(),
        np.array([0.0, 1.0, -1.0]),
        np.array([1.0, 2.0, 0.5]),
        lambda x, w: (lambda s: s * (1.0 - s) * w)(1.0 / (1.0 + np.exp(-x))),
    ),
    (
        "log",
        lambda x: x.log(),
        np.array([0.5, 1.0, 2.0]),
        np.array([1.0, 0.5, 2.0]),
        lambda x, w: (1.0 / x) * w,
    ),
    (
        "clamp",
        lambda x: x.clamp(-0.5, 0.5),
        np.array([-0.25, 0.0, 0.25]),  # strictly inside (-0.5, 0.5)
        np.array([1.0, 2.0, 3.0]),
        lambda x, w: w,  # interior => identity mask
    ),
    (
        "sum",
        lambda x: x.sum(),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array(2.0),
        lambda x, w: np.ones_like(x) * w,
    ),
]


@pytest.mark.parametrize("name,builder,data,weights,d_dx", UNARY_CASES, ids=[u[0] for u in UNARY_CASES])
@pytest.mark.parametrize("rg", [False, True])
def test_unary_output_flag_mirrors_input(name, builder, data, weights, d_dx, rg):
    out = builder(T(data, requires_grad=rg))
    assert out.requires_grad is rg


@pytest.mark.parametrize("name,builder,data,weights,d_dx", UNARY_CASES, ids=[u[0] for u in UNARY_CASES])
def test_unary_enabled_numerical_grad(name, builder, data, weights, d_dx):
    """Catch inverted/missing write when requires_grad=True (guards alone miss this)."""
    x = T(data, requires_grad=True)
    out = builder(x)
    out.backward(grad=weights)
    np.testing.assert_allclose(x.grad, d_dx(data, weights), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("name,builder,data,weights,d_dx", UNARY_CASES, ids=[u[0] for u in UNARY_CASES])
def test_unary_local_backward_guard_when_input_frozen(name, builder, data, weights, d_dx):
    """Seed out.grad and call the private closure; frozen input must stay untouched.

    A public frozen-root backward alone is not enough to prove the per-input guard:
    a mixed-chain frozen branch is what normally stays zero, while a missing guard
    only shows up when a nonzero upstream still reaches this op's _backward.
    """
    x = T(data, requires_grad=False)
    sentinel = np.full_like(x.data, 9.0)
    x.grad = sentinel.copy()
    out = builder(x)
    assert out.requires_grad is False
    out.grad = np.ones_like(out.data)
    out._backward()
    np.testing.assert_array_equal(x.grad, sentinel)


# ---------------------------------------------------------------------------
# Mixed graphs
# ---------------------------------------------------------------------------


def test_mixed_chain_frozen_branch_does_not_drop_trainable_grad():
    """Reusable frozen intermediate must not zero out a trainable sibling path."""
    frozen = T([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    train = T([[0.5, -1.0], [2.0, 0.0]], requires_grad=True)
    out = (frozen * 2.0) + (train * 3.0)
    assert out.requires_grad is True
    out.sum().backward()
    np.testing.assert_array_equal(frozen.grad, np.zeros_like(frozen.data))
    np.testing.assert_allclose(train.grad, np.full_like(train.data, 3.0))


def test_shared_node_accumulation_x_times_x_plus_x():
    x = T([2.0, 3.0], requires_grad=True)
    out = x * x + x
    out.sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)


def test_all_frozen_graph_output_flag_false_without_mandating_backward_policy():
    a = T([1.0, 2.0], requires_grad=False)
    b = T([3.0, 4.0], requires_grad=False)
    out = ((a + b) * a).sigmoid().sum()
    assert out.requires_grad is False
