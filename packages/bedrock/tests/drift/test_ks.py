import numpy as np
import pytest
from bedrock.drift.ks import empirical_cdf, ks_statistic

# empirical_cdf(sample, t) -> fraction of sample values <= t (t can be a scalar or an array)


def test_cdf_basic_steps():
    s = np.array([3.0, 1.0, 2.0, 4.0])
    assert empirical_cdf(s, 0.5) == 0.0
    assert empirical_cdf(s, 1.0) == 0.25  # "at or below" includes ties
    assert empirical_cdf(s, 2.5) == 0.5
    assert empirical_cdf(s, 10.0) == 1.0


def test_cdf_vectorized_t():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([0.0, 2.0, 4.0])
    np.testing.assert_allclose(empirical_cdf(s, t), [0.0, 0.5, 1.0])


def test_cdf_duplicates():
    s = np.array([1.0, 1.0, 1.0, 5.0])
    assert empirical_cdf(s, 1.0) == 0.75


# ks_statistic(a, b) -> largest vertical gap between the two CDFs, in [0, 1]


def test_ks_identical_is_zero():
    a = np.random.default_rng(0).normal(size=500)
    assert ks_statistic(a, a.copy()) == 0.0


def test_ks_no_overlap_is_one():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 11.0, 12.0])
    assert ks_statistic(a, b) == 1.0


def test_ks_small_hand_example():
    # a CDF: 1/2 at 1, 1 at 2. b CDF: 1/2 at 2, 1 at 3. Biggest gap is 0.5.
    assert ks_statistic(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx(0.5)


def test_ks_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=200), rng.normal(0.5, 1, size=300)
    assert ks_statistic(a, b) == pytest.approx(ks_statistic(b, a))


def test_ks_shift_grows_gap():
    a = np.random.default_rng(2).normal(size=1000)
    small, big = ks_statistic(a, a + 0.1), ks_statistic(a, a + 2.0)
    assert small < big
    assert big > 0.6


def test_ks_catches_spread_with_same_mean():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, size=2000)
    b = rng.normal(0, 2, size=2000)  # same mean, twice the spread
    assert abs(a.mean() - b.mean()) < 0.15  # a mean check would miss this
    assert ks_statistic(a, b) > 0.1


def test_ks_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=300), rng.normal(0.3, 1.2, size=250)
    assert ks_statistic(a, b) == pytest.approx(scipy_stats.ks_2samp(a, b).statistic)
