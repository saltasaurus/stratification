import numpy as np
import numpy.testing as npt
import pytest
from bedrock.autograd import Tensaur, collapse


class TestAddTensaur:
    def test_add(self):

        a = Tensaur([1, 2, 3])
        b = Tensaur([4, 5, 6])

        c = a + b
        c.grad = np.ones_like(c.data, dtype=float)

        c._backward()

        npt.assert_allclose(a.grad, np.ones_like(c.data, dtype=float))
        npt.assert_allclose(b.grad, np.ones_like(c.data, dtype=float))

    def test_add_broadcast_scalar(self):
        # bias on the right: x + b

        x = Tensaur([10.0, 20.0, 30.0])
        b = Tensaur(1.0)
        c = x + b
        c.grad = np.array([0.5, 2.0, 1.0])
        c._backward()

        npt.assert_allclose(x.grad, [0.5, 2.0, 1.0])  # Same shape -> gradients passed through
        npt.assert_allclose(b.grad, [3.5])  # broadcast -> summed
        assert b.grad.shape == b.data.shape  # gradient must keep b's shape

    def test_add_broadcast_scalar_left(self):
        # bias on the left: b + x

        x = Tensaur([10.0, 20.0, 30.0])
        b = Tensaur(1.0)
        c = b + x
        c.grad = np.array([0.5, 2.0, 1.0])
        c._backward()

        npt.assert_allclose(x.grad, [0.5, 2.0, 1.0])  # Same shape -> gradients passed through
        npt.assert_allclose(b.grad, [3.5])  # broadcast -> summed
        assert b.grad.shape == b.data.shape  # gradient must keep b's shape

    def test_add_broadcast_bias_row(self):
        # inputs (x): 6 samples x 3 features, bias (b): one per feature
        x = Tensaur(np.ones((6, 3)))
        b = Tensaur(np.ones(3))
        c = x + b
        c.grad = np.ones((6, 3))
        c._backward()

        assert x.grad.shape == (6, 3)
        npt.assert_allclose(x.grad, np.ones((6, 3)))
        assert b.grad.shape == (3,)
        npt.assert_allclose(b.grad, np.full(3, 6.0))  # summed over 6 samples

    def test_add_broadcast_keepdims(self):
        x = Tensaur(np.ones((6, 3)))
        b = Tensaur(np.ones((1, 3)))
        c = b + x  # bias on the left
        c.grad = np.ones((6, 3))
        c._backward()

        assert b.grad.shape == (1, 3)
        npt.assert_allclose(b.grad, np.full((1, 3), 6.0))


class TestMultiplyTensaur:
    def test_multiply(self):

        a = Tensaur([1, 2, 3])
        b = Tensaur([4, 5, 6])

        c = a * b
        c.grad = np.ones_like(c.data, dtype=float)

        c._backward()

        npt.assert_allclose(a.grad, np.array([4.0, 5.0, 6.0]))
        npt.assert_allclose(b.grad, np.array([1.0, 2.0, 3.0]))

    def test_mul_broadcast_row(self):
        # different values per row, so the sum is visible
        x_data = np.arange(18, dtype=float).reshape(6, 3)
        x = Tensaur(x_data)
        w = Tensaur(np.array([1.0, 2.0, 3.0]))
        c = x * w
        c.grad = np.ones((6, 3))
        c._backward()

        assert x.grad.shape == (6, 3)
        npt.assert_allclose(x.grad, np.tile([1.0, 2.0, 3.0], (6, 1)))  # w copied to each row
        assert w.grad.shape == (3,)
        npt.assert_allclose(w.grad, x_data.sum(axis=0))  # column sums of x

    def test_mul_broadcast_scalar(self):
        x = Tensaur([10.0, 20.0, 30.0])
        w = Tensaur(2.0)
        c = x * w
        c.grad = np.ones(3)
        c._backward()

        npt.assert_allclose(x.grad, [2.0, 2.0, 2.0])
        assert w.grad.shape == ()
        npt.assert_allclose(w.grad, 60.0)  # the nudge check from earlier


class TestTensaurUtils:
    @pytest.mark.parametrize(
        "grad_shape, target_shape, expected_value",
        [
            ((3,), (3,), 1.0),  # same shape: unchanged
            ((6, 3), (3,), 6.0),  # extra front axis removed
            ((6, 3), (1, 3), 6.0),  # size-1 axis stretched, kept as 1
            ((6, 3), (6, 1), 3.0),  # stretched along the last axis
            ((6, 3), (), 18.0),  # scalar: sum everything
            ((6, 3), (1,), 18.0),  # shape (1,): remove front axis, then keep size 1
            ((2, 6, 3), (3,), 12.0),  # two extra front axes
        ],
    )
    def test_collapse(grad_shape, target_shape, expected_value):
        grad = np.ones(grad_shape)
        out = collapse(grad, target_shape)

        assert out.shape == target_shape
        npt.assert_allclose(out, np.full(target_shape, expected_value))

    def test_collapse_does_not_modify_input():
        grad = np.ones((6, 3))
        collapse(grad, (3,))
        npt.assert_allclose(grad, np.ones((6, 3)))
