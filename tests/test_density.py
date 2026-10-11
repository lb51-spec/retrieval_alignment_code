"""Tests for density-ratio weight utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.density import (
    clip_density_weights,
    effective_sample_size,
    make_density_weights,
    positive_density_weights,
    self_normalize_weights,
    weight_entropy,
)


def test_softplus_output_is_positive() -> None:
    weights = positive_density_weights(torch.tensor([-100.0, 0.0, 2.0]))

    assert torch.all(weights > 0)


def test_shape_handling_for_vector_and_column() -> None:
    vector = positive_density_weights(torch.tensor([0.0, 1.0]))
    column = positive_density_weights(torch.tensor([[0.0], [1.0]]))

    assert vector.shape == (2,)
    assert column.shape == (2,)


def test_clipping_respects_max_weight() -> None:
    clipped = clip_density_weights(torch.tensor([1.0, 5.0, 10.0]), max_weight=4.0)

    assert clipped.tolist() == pytest.approx([1.0, 4.0, 4.0])


def test_max_weight_validation() -> None:
    with pytest.raises(ValueError, match="max_weight"):
        clip_density_weights(torch.tensor([1.0]), max_weight=0.0)


def test_self_normalized_weights_have_mean_one() -> None:
    normalized = self_normalize_weights(torch.tensor([1.0, 2.0, 3.0]))

    assert normalized.mean().item() == pytest.approx(1.0)


def test_all_zero_weights_raise_in_self_normalization() -> None:
    with pytest.raises(ValueError, match="all zero"):
        self_normalize_weights(torch.tensor([0.0, 0.0]))


def test_negative_raw_weights_raise_value_error() -> None:
    for fn in (self_normalize_weights, effective_sample_size, weight_entropy):
        with pytest.raises(ValueError, match="nonnegative"):
            fn(torch.tensor([1.0, -0.1]))


def test_ess_uniform_equals_batch_size() -> None:
    ess = effective_sample_size(torch.ones(4))

    assert ess.item() == pytest.approx(4.0)


def test_ess_decreases_for_skewed_weights() -> None:
    uniform = effective_sample_size(torch.ones(4))
    skewed = effective_sample_size(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    assert skewed < uniform


def test_entropy_is_finite_and_larger_for_uniform_than_skewed() -> None:
    uniform = weight_entropy(torch.ones(4))
    skewed = weight_entropy(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    assert torch.isfinite(uniform)
    assert torch.isfinite(skewed)
    assert uniform > skewed


def test_make_density_weights_combines_positivity_clipping_and_normalization() -> None:
    weights = make_density_weights(
        torch.tensor([-1.0, 0.0, 10.0]),
        max_weight=2.0,
        normalize=True,
    )

    assert weights.shape == (3,)
    assert torch.all(weights > 0)
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights.max().item() <= 2.1


def test_non_finite_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        positive_density_weights(torch.tensor([0.0, float("nan")]))

    with pytest.raises(ValueError, match="finite"):
        self_normalize_weights(torch.tensor([1.0, float("inf")]))


def test_empty_tensors_raise_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        positive_density_weights(torch.empty(0))


def test_unsupported_dimensions_raise_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        positive_density_weights(torch.zeros(2, 2))
