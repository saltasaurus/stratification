import numpy as np
import numpy.testing as npt
from bedrock.autograd import Tensaur


class TestTensaur:
    def test_add(self):

        a = Tensaur([1, 2, 3])
        b = Tensaur([4, 5, 6])

        c = a + b
        c.grad = np.ones_like(c.data, dtype=float)

        c._backward()

        npt.assert_allclose(a.grad, np.ones_like(c.data, dtype=float))
        npt.assert_allclose(b.grad, np.ones_like(c.data, dtype=float))

    def test_multiply(self):

        a = Tensaur([1, 2, 3])
        b = Tensaur([4, 5, 6])

        c = a * b
        c.grad = np.ones_like(c.data, dtype=float)

        c._backward()

        npt.assert_allclose(a.grad, np.array([4.0, 5.0, 6.0]))
        npt.assert_allclose(b.grad, np.array([1.0, 2.0, 3.0]))

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
