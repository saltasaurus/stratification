import numpy as np
import torch
from bedrock.autograd.tensaur import Tensaur
from bedrock.loss import bce_mean


def test_mul_gradients_match_torch():
    a = np.array([2.0, -3.0, 0.5])
    b = np.array([-4.0, 2.0, 3.0])
    upstream = np.array([1.0, -2.0, 0.5])

    x = Tensaur(a, requires_grad=True)
    y = Tensaur(b, requires_grad=True)
    out = x * y
    out.backward(grad=upstream)

    tx = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    tout = tx * ty
    tout.backward(torch.tensor(upstream, dtype=torch.float64))

    np.testing.assert_allclose(out.data, tout.detach().numpy())
    np.testing.assert_allclose(x.grad, tx.grad.numpy())
    np.testing.assert_allclose(y.grad, ty.grad.numpy())


def test_broadcast_mul_gradients_match_torch():
    a = np.array([[2.0, -3.0], [0.5, 4.0]])
    b = np.array([[-4.0, 2.0]])
    upstream = np.array([[1.0, -2.0], [0.5, 3.0]])

    x = Tensaur(a, requires_grad=True)
    y = Tensaur(b, requires_grad=True)
    out = x * y
    out.backward(grad=upstream)

    tx = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    tout = tx * ty
    tout.backward(torch.tensor(upstream, dtype=torch.float64))

    np.testing.assert_allclose(out.data, tout.detach().numpy())
    np.testing.assert_allclose(x.grad, tx.grad.numpy())
    np.testing.assert_allclose(y.grad, ty.grad.numpy())


def test_linear_layer_gradients_match_torch():
    inputs = np.array([[2.0, -3.0], [0.5, 4.0]])
    weights = np.array([[-4.0], [2.0]])
    bias = np.array([0.5])
    upstream = np.array([[1.0], [-2.0]])

    x = Tensaur(inputs, requires_grad=False)
    w = Tensaur(weights, requires_grad=True)
    b = Tensaur(bias, requires_grad=True)
    out = x @ w + b
    out.backward(grad=upstream)

    tx = torch.tensor(inputs, dtype=torch.float64)
    tw = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    tout = tx @ tw + tb
    tout.backward(torch.tensor(upstream, dtype=torch.float64))

    np.testing.assert_allclose(out.data, tout.detach().numpy())
    np.testing.assert_allclose(w.grad, tw.grad.numpy())
    np.testing.assert_allclose(b.grad, tb.grad.numpy())
    np.testing.assert_array_equal(x.grad, np.zeros_like(inputs))


def test_linear_sigmoid_gradients_match_torch():
    inputs = np.array([[1.0, -0.5], [0.5, 2.0]])
    weights = np.array([[0.4], [-0.2]])
    bias = np.array([0.1])
    upstream = np.array([[1.0], [-2.0]])

    x = Tensaur(inputs, requires_grad=False)
    w = Tensaur(weights, requires_grad=True)
    b = Tensaur(bias, requires_grad=True)
    out = (x @ w + b).sigmoid()
    out.backward(grad=upstream)

    tx = torch.tensor(inputs, dtype=torch.float64)
    tw = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    tout = torch.sigmoid(tx @ tw + tb)
    tout.backward(torch.tensor(upstream, dtype=torch.float64))

    np.testing.assert_allclose(out.data, tout.detach().numpy())
    np.testing.assert_allclose(w.grad, tw.grad.numpy())
    np.testing.assert_allclose(b.grad, tb.grad.numpy())
    np.testing.assert_array_equal(x.grad, np.zeros_like(inputs))


def test_logistic_bce_gradients_match_torch():
    inputs = np.array([[1.0, -0.5], [0.5, 2.0]])
    weights = np.array([[0.4], [-0.2]])
    bias = np.array([0.1])
    labels = np.array([[1.0], [0.0]])

    x = Tensaur(inputs, requires_grad=False)
    w = Tensaur(weights, requires_grad=True)
    b = Tensaur(bias, requires_grad=True)
    y = Tensaur(labels, requires_grad=False)

    loss = bce_mean((x @ w + b).sigmoid(), y)
    loss.backward()

    tx = torch.tensor(inputs, dtype=torch.float64)
    tw = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(labels, dtype=torch.float64)

    probabilities = torch.sigmoid(tx @ tw + tb)
    torch_loss = torch.nn.functional.binary_cross_entropy(probabilities, ty, reduction="mean")
    torch_loss.backward()

    np.testing.assert_allclose(loss.data, torch_loss.detach().numpy())
    np.testing.assert_allclose(w.grad, tw.grad.numpy())
    np.testing.assert_allclose(b.grad, tb.grad.numpy())
    np.testing.assert_array_equal(x.grad, np.zeros_like(inputs))
    np.testing.assert_array_equal(y.grad, np.zeros_like(labels))


def test_tanh_gradients_match_torch():
    values = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    upstream = np.array([0.5, -2.0, 1.0, 3.0, -0.5])

    x = Tensaur(values, requires_grad=True)
    out = x.tanh()
    assert out.requires_grad
    out.backward(grad=upstream)

    tx = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    tout = tx.tanh()
    tout.backward(torch.tensor(upstream, dtype=torch.float64))

    np.testing.assert_allclose(out.data, tout.detach().numpy())
    np.testing.assert_allclose(x.grad, tx.grad.numpy())


def test_tanh_frozen_input():
    values = np.array([-0.5, 0.0, 0.5])
    x = Tensaur(values, requires_grad=False)
    out = x.tanh()

    assert not out.requires_grad
    np.testing.assert_allclose(out.data, np.tanh(values))

    # Exercise the local guard without defining frozen-root backward policy.
    out.grad = np.ones_like(out.data)
    out._backward()
    np.testing.assert_array_equal(x.grad, np.zeros_like(values))


def test_mlp_training_step_matches_torch():
    rng = np.random.default_rng(0)
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([[0.0], [1.0], [1.0], [0.0]])
    initial = [
        rng.normal(scale=0.5, size=(2, 8)),
        np.zeros(8),
        rng.normal(scale=0.5, size=(8, 1)),
        np.zeros(1),
    ]

    params = [Tensaur(a.copy(), requires_grad=True) for a in initial]
    tparams = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in initial]
    w1, b1, w2, b2 = params
    tw1, tb1, tw2, tb2 = tparams

    x = Tensaur(inputs, requires_grad=False)
    y = Tensaur(labels, requires_grad=False)
    probabilities = (((x @ w1 + b1).tanh()) @ w2 + b2).sigmoid()
    loss = bce_mean(probabilities, y)
    loss.backward()

    tx = torch.tensor(inputs, dtype=torch.float64)
    ty = torch.tensor(labels, dtype=torch.float64)
    logits = torch.tanh(tx @ tw1 + tb1) @ tw2 + tb2
    tloss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty)
    tloss.backward()

    np.testing.assert_allclose(loss.data, tloss.detach().numpy())
    for p, tp in zip(params, tparams, strict=False):
        np.testing.assert_allclose(p.grad, tp.grad.numpy(), rtol=1e-7, atol=1e-10)

    with torch.no_grad():
        for p, tp in zip(params, tparams, strict=False):
            p.data -= 0.1 * p.grad
            tp -= 0.1 * tp.grad
            np.testing.assert_allclose(p.data, tp.numpy(), rtol=1e-7, atol=1e-10)
