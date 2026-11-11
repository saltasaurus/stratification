"""Tests for Linear weight initialization (normal, Xavier, and He)."""

import numpy as np
import pytest
from bedrock.autograd.tensaur import Tensaur
from bedrock.nn import Linear
from bedrock.random import manual_seed

INITS = ("normal", "xavier", "he")
LARGE_IN, LARGE_OUT = 2000, 500


def _expected_std(init: str, in_features: int) -> float:
    if init == "xavier":
        return float(np.sqrt(1 / in_features))
    if init == "he":
        return float(np.sqrt(2 / in_features))
    if init == "normal":
        return 0.5
    raise AssertionError(f"unexpected init: {init!r}")


@pytest.fixture(autouse=True)
def _reseed():
    """Start each test from a known shared generator state."""
    manual_seed(0)


@pytest.mark.parametrize("init", INITS)
def test_weight_and_bias_shapes_requires_grad_and_zero_bias(init):
    """Each init must wrap weights in a trainable Tensaur and leave bias at 0.

    Catches Xavier/He branches that assign a raw ndarray instead of
    ``Tensaur(..., requires_grad=True)``.
    """
    in_features, out_features = 6, 4
    layer = Linear(in_features, out_features, init=init)

    assert isinstance(layer.weight, Tensaur)
    assert isinstance(layer.bias, Tensaur)
    assert layer.weight.requires_grad is True
    assert layer.bias.requires_grad is True
    assert layer.weight.data.shape == (in_features, out_features)
    assert layer.bias.data.shape == (out_features,)
    np.testing.assert_array_equal(layer.bias.data, np.zeros(out_features))


@pytest.mark.parametrize("init", INITS)
def test_large_layer_std_matches_init_formula(init):
    """Sample std of a large layer should match the formula within 2%.

    Catches dividing by σ instead of multiplying, which inflates weights by ~1/σ².
    """
    layer = Linear(LARGE_IN, LARGE_OUT, init=init)
    expected = _expected_std(init, LARGE_IN)
    np.testing.assert_allclose(layer.weight.data.std(), expected, rtol=0.02)


@pytest.mark.parametrize("init", INITS)
def test_large_layer_mean_is_near_zero(init):
    """Weights should be zero-mean, not shifted like uniform draws on [0, 1).

    Catches ``rng.random`` (mean 0.5, or 0.5σ if then scaled) instead of a
    zero-mean normal.
    """
    weights = Linear(LARGE_IN, LARGE_OUT, init=init).weight.data
    # Uniform[0, 1) has |mean|/std ≈ 1.73 whether or not it is later scaled.
    # A zero-mean normal sits near 0 even when someone divides by σ (std test
    # catches that). Compare to the measured std so extra variance is not a fail.
    assert abs(weights.mean()) < 0.15 * weights.std()


def test_he_std_for_50_inputs_is_point_two():
    """Lesson hand value: He with 50 inputs has σ = sqrt(2/50) = 0.2."""
    assert np.sqrt(2 / 50) == pytest.approx(0.2)
    layer = Linear(50, 4000, init="he")
    np.testing.assert_allclose(layer.weight.data.std(), 0.2, rtol=0.02)


def test_unknown_init_raises_value_error():
    """A typo must raise instead of silently falling through to the default."""
    with pytest.raises(ValueError, match="unknown init: 'xaiver'"):
        Linear(4, 3, init="xaiver")


def test_manual_seed_reproduces_and_changes_with_seed():
    """The same seed must replay the same weights; a different seed must not."""
    manual_seed(0)
    first = Linear(8, 8, init="xavier")
    manual_seed(0)
    replayed = Linear(8, 8, init="xavier")
    np.testing.assert_array_equal(first.weight.data, replayed.weight.data)

    manual_seed(1)
    other = Linear(8, 8, init="xavier")
    assert not np.array_equal(first.weight.data, other.weight.data)


def test_two_layers_after_one_seed_are_not_identical():
    """Sequential draws from the shared generator must give each layer its own weights.

    Catches storing a per-layer seed (or reseeding inside ``__init__``) so every
    ``Linear(8, 8)`` starts as a clone.
    """
    first = Linear(8, 8, init="xavier")
    second = Linear(8, 8, init="xavier")
    assert not np.array_equal(first.weight.data, second.weight.data)


def test_default_init_is_normal():
    """Omitting ``init`` must match ``init='normal'`` (std 0.5)."""
    manual_seed(0)
    default = Linear(30, 20)
    manual_seed(0)
    explicit = Linear(30, 20, init="normal")
    np.testing.assert_array_equal(default.weight.data, explicit.weight.data)
