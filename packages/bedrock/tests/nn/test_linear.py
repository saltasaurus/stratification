import numpy as np
from bedrock.autograd.tensaur import Tensaur
from bedrock.nn import Linear


def test_linear_forward_and_parameters():
    layer = Linear(2, 3, np.random.default_rng(0))
    layer.weight.data[:] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    layer.bias.data[:] = [0.1, 0.2, 0.3]
    x = Tensaur([[2.0, -1.0]], requires_grad=False)

    out = layer(x)

    assert layer.weight.data.shape == (2, 3)
    assert layer.bias.data.shape == (3,)
    assert layer.weight.requires_grad
    assert layer.bias.requires_grad
    np.testing.assert_allclose(out.data, [[-1.9, -0.8, 0.3]])
    params = layer.parameters()
    assert len(params) == 2
    assert params[0] is layer.weight
    assert params[1] is layer.bias


def test_linear_backward_and_zero_grad():
    layer = Linear(2, 3, np.random.default_rng(0))
    x = Tensaur([[2.0, -1.0], [1.0, 3.0]], requires_grad=False)

    layer(x).sum().backward()

    np.testing.assert_allclose(layer.weight.grad, [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(layer.bias.grad, [2.0, 2.0, 2.0])
    before = [p.data.copy() for p in layer.parameters()]

    layer.zero_grad()

    for p, original in zip(layer.parameters(), before, strict=False):
        np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))
        np.testing.assert_array_equal(p.data, original)
