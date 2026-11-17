import numpy as np
import pytest
from strata.calibration.ece import ece


def test_hand_example():
    assert ece(np.full(4, 0.9), np.array([1, 1, 1, 0])) == pytest.approx(0.15)


def test_perfect_calibration_is_zero():
    probs = np.array([0.8] * 5 + [0.2] * 5)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    assert ece(probs, y) == pytest.approx(0.0)


def test_negative_class_confidence_mapping():
    # p=0.1 means 90% confident in class 0
    assert ece(np.full(4, 0.1), np.array([0, 0, 0, 1])) == pytest.approx(0.15)


def test_confidence_one_lands_in_last_bin():
    assert ece(np.array([1.0, 1.0]), np.array([1, 1])) == pytest.approx(0.0)
    assert ece(np.array([1.0, 1.0]), np.array([1, 0])) == pytest.approx(0.5)


def test_two_bins_weighted():
    probs = np.array([0.6, 0.6, 0.9, 0.9])
    y = np.array([1, 0, 1, 1])
    assert ece(probs, y) == pytest.approx(0.5 * 0.1 + 0.5 * 0.1)


def test_n_bins_one_matches_global_gap():
    probs = np.array([0.6, 0.9, 0.7, 0.95])
    y = np.array([1, 0, 1, 1])
    assert ece(probs, y, n_bins=1) == pytest.approx(abs(0.75 - probs.mean()))


def test_returns_python_float():
    assert isinstance(ece(np.full(4, 0.9), np.array([1, 1, 1, 0])), float)
