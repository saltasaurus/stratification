import numpy as np
from bedrock.autograd.tensaur import Tensaur
from bedrock.loss import bce_mean
from bedrock.models.logistic_regression import LogisticRegression


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def _bce_mean_np(p, y):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_bce_mean_forward_matches_numpy():
    rng = np.random.default_rng(0)
    p_np = rng.uniform(0.05, 0.95, size=(8,))
    y_np = rng.integers(0, 2, size=(8,)).astype(float)
    p = Tensaur(p_np.copy())
    y = Tensaur(y_np.copy())
    out = bce_mean(p, y)
    np.testing.assert_allclose(out.data, _bce_mean_np(p_np, y_np), rtol=1e-6, atol=1e-8)


def test_bce_mean_backward_matches_finite_difference():
    rng = np.random.default_rng(1)
    p_np = rng.uniform(0.1, 0.9, size=(6,))
    y_np = rng.integers(0, 2, size=(6,)).astype(float)
    eps = 1e-5

    p = Tensaur(p_np.copy())
    y = Tensaur(y_np.copy())
    loss = bce_mean(p, y)
    loss.backward()

    def loss_np(pn):
        return _bce_mean_np(pn, y_np)

    p_fd = np.zeros_like(p_np)
    for i in range(p_np.shape[0]):
        pp, pm = p_np.copy(), p_np.copy()
        pp[i] += eps
        pm[i] -= eps
        p_fd[i] = (loss_np(pp) - loss_np(pm)) / (2 * eps)

    np.testing.assert_allclose(p.grad, p_fd, rtol=1e-4, atol=1e-5)


def test_sigmoid_bce_grads_match_handwritten():
    rng = np.random.default_rng(2)
    X_np = rng.normal(size=(10, 3))
    y_np = rng.integers(0, 2, size=(10, 1)).astype(float)  # (n, 1)
    w_np = rng.normal(size=(3, 1)) * 0.1  # (d, 1)
    b_np = np.zeros((1, 1))
    z = X_np @ w_np + b_np
    p = _sigmoid(z)
    n = y_np.shape[0]
    dz = (p - y_np) / n
    dw = X_np.T @ dz
    db = dz.sum() / n  # scalar
    X = Tensaur(X_np.copy())
    y = Tensaur(y_np.copy())
    w = Tensaur(w_np.copy())
    b = Tensaur(b_np.copy())
    logits = X @ w + b
    probs = logits.sigmoid()
    loss = bce_mean(probs, y)
    loss.backward()
    np.testing.assert_allclose(w.grad, dw, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(np.asarray(b.grad).sum()), db, rtol=1e-5, atol=1e-6)


def test_fit_tensaur_loss_decreases():
    rng = np.random.default_rng(3)
    X_np = np.vstack(
        [
            rng.normal(loc=-1, size=(20, 2)),
            rng.normal(loc=1, size=(20, 2)),
        ]
    )
    y_np = np.array([[0.0]] * 20 + [[1.0]] * 20)
    model = LogisticRegression(lr=0.5, epochs=200)
    # fit_tensaur should init w as (d, 1), b as (1, 1)
    model.fit_tensaur(Tensaur(X_np.copy()), Tensaur(y_np.copy()))
    w0 = np.zeros((2, 1))
    b0 = np.zeros((1, 1))
    loss0 = _bce_mean_np(_sigmoid(X_np @ w0 + b0), y_np)
    w = np.asarray(model.w.data)
    b = np.asarray(model.b)
    loss1 = _bce_mean_np(_sigmoid(X_np @ w + b), y_np)
    assert loss1 < loss0


def test_fit_tensaur_grads_match_numpy_fit_one_step():
    rng = np.random.default_rng(4)
    X_np = rng.normal(size=(12, 4))
    y_np = rng.integers(0, 2, size=(12, 1)).astype(float)
    w0 = rng.normal(size=(4, 1)) * 0.05
    b0 = np.array([[0.1]])
    z = X_np @ w0 + b0
    p = _sigmoid(z)
    n = y_np.shape[0]
    dz = (p - y_np) / n
    dw_ref = X_np.T @ dz
    db_ref = float(dz.sum() / n)
    X = Tensaur(X_np.copy())
    y = Tensaur(y_np.copy())
    w = Tensaur(w0.copy())
    b = Tensaur(b0.copy())
    logits = X @ w + b
    probs = logits.sigmoid()
    loss = bce_mean(probs, y)
    loss.backward()
    np.testing.assert_allclose(w.grad, dw_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(np.asarray(b.grad).sum()), db_ref, rtol=1e-5, atol=1e-6)
