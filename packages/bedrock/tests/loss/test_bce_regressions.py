import numpy as np
from bedrock.autograd.tensaur import Tensaur
from bedrock.loss.BCE import EPS, bce_mean


def finite_diff_grad(p_data, y_data, h):
    """Estimate each gradient by slightly changing one prediction."""
    grad = np.zeros_like(p_data)

    for idx in np.ndindex(p_data.shape):
        plus = p_data.copy()
        minus = p_data.copy()
        plus[idx] += h
        minus[idx] -= h

        loss_plus = bce_mean(Tensaur(plus), Tensaur(y_data)).data
        loss_minus = bce_mean(Tensaur(minus), Tensaur(y_data)).data
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

    return grad


def test_bce_gradient_averages_over_all_elements():
    # Six predictions: backward must divide by six, not two rows.
    p_data = np.array(
        [
            [0.2, 0.7, 0.4],
            [0.9, 0.3, 0.6],
        ]
    )
    y_data = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    p = Tensaur(p_data.copy())
    bce_mean(p, Tensaur(y_data)).backward()

    expected = finite_diff_grad(p_data, y_data, h=1e-6)
    np.testing.assert_allclose(p.grad, expected, rtol=1e-5, atol=1e-8)


def test_bce_gradient_is_zero_where_prediction_is_clipped():
    p_data = np.array(
        [
            [EPS / 4],
            [1 - EPS / 4],
        ]
    )
    y_data = np.array([[1.0], [0.0]])

    p = Tensaur(p_data.copy())
    bce_mean(p, Tensaur(y_data)).backward()

    # Both perturbations stay in the flat regions of clipping:
    # changing p slightly does not change the forward loss.
    expected = finite_diff_grad(p_data, y_data, h=EPS / 8)
    np.testing.assert_array_equal(expected, np.zeros_like(p_data))
    np.testing.assert_array_equal(p.grad, expected)
