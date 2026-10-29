import numpy as np
from bedrock.autograd.tensaur import Tensaur


def test_matmul_forward_matches_numpy():
    x = Tensaur(np.array([[1.0, 2.0], [3.0, 4.0]]))
    w = Tensaur(np.array([[0.5, -1.0], [1.5, 2.0]]))
    y = x @ w
    expected = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[0.5, -1.0], [1.5, 2.0]])
    np.testing.assert_allclose(y.data, expected)


def test_matmul_grad_shapes():
    x = Tensaur(np.ones((2, 3)))
    w = Tensaur(np.ones((3, 4)))
    y = x @ w
    y.backward()
    assert x.grad.shape == (2, 3)
    assert w.grad.shape == (3, 4)


def test_matmul_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    x_np = rng.normal(size=(3, 2))
    w_np = rng.normal(size=(2, 4))
    g_np = rng.normal(size=(3, 4))
    eps = 1e-5

    def loss_np(xn, wn):
        return float((xn @ wn * g_np).sum())

    x = Tensaur(x_np.copy())
    w = Tensaur(w_np.copy())
    g = Tensaur(g_np.copy())
    L = (x @ w * g).sum()
    L.backward()

    x_fd = np.zeros_like(x_np)
    w_fd = np.zeros_like(w_np)
    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            xp, xm = x_np.copy(), x_np.copy()
            xp[i, j] += eps
            xm[i, j] -= eps
            x_fd[i, j] = (loss_np(xp, w_np) - loss_np(xm, w_np)) / (2 * eps)
    for i in range(w_np.shape[0]):
        for j in range(w_np.shape[1]):
            wp, wm = w_np.copy(), w_np.copy()
            wp[i, j] += eps
            wm[i, j] -= eps
            w_fd[i, j] = (loss_np(x_np, wp) - loss_np(x_np, wm)) / (2 * eps)

    np.testing.assert_allclose(x.grad, x_fd, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(w.grad, w_fd, rtol=1e-4, atol=1e-5)
