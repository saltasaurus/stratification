import numpy as np
from bedrock.autograd import Tensaur
from strata.calibration.ece import ece
from strata.calibration.temperature import fit_temperature


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_data(scale, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    true_z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < sigmoid(true_z)).astype(float)
    return true_z * scale, y  # model logits = scaled truth


def fit(z, y, **kw):
    return fit_temperature(Tensaur(z), Tensaur(y), **kw)


def test_overconfident_gives_T_above_1():
    z, y = make_data(scale=3.0)
    T = fit(z, y, epochs=500, lr=0.05)
    assert 2.5 < T < 3.5  # should recover roughly 3


def test_underconfident_gives_T_below_1():
    z, y = make_data(scale=0.4)
    T = fit(z, y, epochs=500, lr=0.05)
    assert 0.3 < T < 0.5


def test_calibrated_stays_near_1():
    z, y = make_data(scale=1.0)
    T = fit(z, y, epochs=500, lr=0.05)
    assert 0.85 < T < 1.15


def test_ece_drops_after_scaling():
    z, y = make_data(scale=3.0)
    T = fit(z, y, epochs=500, lr=0.05)
    assert ece(sigmoid(z / T), y) < ece(sigmoid(z), y)


def test_ranking_unchanged():
    z, y = make_data(scale=3.0)
    T = fit(z, y, epochs=500, lr=0.05)
    np.testing.assert_array_equal(np.argsort(z), np.argsort(z / T))


def test_returns_positive_float():
    z, y = make_data(scale=3.0)
    T = fit(z, y, epochs=50, lr=0.05)
    assert isinstance(T, float) and T > 0
