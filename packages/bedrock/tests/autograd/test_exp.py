import numpy as np
from bedrock.autograd import Tensaur


def test_exp_forward():
    x = Tensaur([0.0, 1.0, -2.0])
    np.testing.assert_allclose(x.exp().data, np.exp([0.0, 1.0, -2.0]))


def test_exp_grad_is_output():
    x = Tensaur([1.5], requires_grad=True)
    out = x.exp()
    out.backward()
    np.testing.assert_allclose(x.grad, np.exp([1.5]))


def test_exp_grad_matches_numerical():
    v, h = 0.7, 1e-6
    x = Tensaur([v], requires_grad=True)
    x.exp().backward()
    numerical = (np.exp(v + h) - np.exp(v - h)) / (2 * h)
    np.testing.assert_allclose(x.grad, [numerical], rtol=1e-5)


def test_exp_chain_with_mul():
    # d/dx exp(3x) = 3·exp(3x)
    x = Tensaur([0.2], requires_grad=True)
    (x * 3).exp().backward()
    np.testing.assert_allclose(x.grad, [3 * np.exp(0.6)])
