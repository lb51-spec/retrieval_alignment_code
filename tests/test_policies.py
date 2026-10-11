"""Tests for masked candidate-policy utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.policies import (
    candidate_policy_probs,
    greedy_candidate_action,
    masked_log_softmax,
    masked_softmax,
    sample_candidate_action,
)


def test_masked_softmax_output_shape() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    mask = torch.tensor([[True, True, False], [True, False, True]])

    probs = masked_softmax(scores, mask)

    assert probs.shape == scores.shape


def test_invalid_candidates_get_zero_probability() -> None:
    scores = torch.tensor([[1.0, 2.0, 100.0]])
    mask = torch.tensor([[True, True, False]])

    probs = masked_softmax(scores, mask)

    assert probs[0, 2].item() == 0.0


def test_probabilities_sum_to_one_over_valid_candidates() -> None:
    scores = torch.tensor([[1.0, 2.0, 100.0], [3.0, -1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [False, True, True]])

    probs = masked_softmax(scores, mask)

    assert torch.allclose(probs.sum(dim=1), torch.ones(2))


def test_temperature_changes_distribution_sharpness() -> None:
    scores = torch.tensor([[0.0, 2.0]])
    mask = torch.tensor([[True, True]])

    cold = masked_softmax(scores, mask, temperature=0.5)
    warm = masked_softmax(scores, mask, temperature=2.0)

    assert cold[0, 1] > warm[0, 1]


def test_greedy_action_ignores_invalid_high_score_candidates() -> None:
    scores = torch.tensor([[1.0, 2.0, 999.0], [5.0, 3.0, 1000.0]])
    candidate_indices = torch.tensor([[10, 11, 12], [20, 21, 22]])
    mask = torch.tensor([[True, True, False], [True, False, False]])

    selected = greedy_candidate_action(scores, candidate_indices, mask)

    assert selected.tolist() == [11, 20]


def test_stochastic_sampling_returns_only_valid_candidate_indices() -> None:
    scores = torch.tensor([[1.0, 2.0, 999.0], [5.0, 3.0, 1000.0]])
    candidate_indices = torch.tensor([[10, 11, 12], [20, 21, 22]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    generator = torch.Generator().manual_seed(7)

    for _ in range(20):
        sampled = sample_candidate_action(scores, candidate_indices, mask, generator=generator)
        assert sampled[0].item() in {10, 11}
        assert sampled[1].item() == 20


def test_all_invalid_rows_raise_clear_error() -> None:
    scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="at least one valid candidate"):
        masked_softmax(scores, mask)


def test_log_probs_are_finite_for_valid_candidates() -> None:
    scores = torch.tensor([[1.0, 2.0, 100.0]])
    mask = torch.tensor([[True, True, False]])

    log_probs = masked_log_softmax(scores, mask)

    assert torch.isfinite(log_probs[mask]).all()


def test_log_probs_are_masked_for_invalid_candidates() -> None:
    scores = torch.tensor([[1.0, 2.0, 100.0]])
    mask = torch.tensor([[True, True, False]])

    log_probs = masked_log_softmax(scores, mask)

    assert torch.isfinite(log_probs).all()
    assert log_probs[0, 2] == torch.finfo(log_probs.dtype).min


def test_deterministic_sampling_with_fixed_generator_seed() -> None:
    scores = torch.tensor([[0.1, 1.0, 2.0], [3.0, 0.5, -1.0]])
    candidate_indices = torch.tensor([[10, 11, 12], [20, 21, 22]])
    mask = torch.tensor([[True, True, True], [True, True, False]])

    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)

    first = sample_candidate_action(scores, candidate_indices, mask, generator=first_generator)
    second = sample_candidate_action(scores, candidate_indices, mask, generator=second_generator)

    assert torch.equal(first, second)


def test_candidate_policy_output_contains_probs_and_logs() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])

    output = candidate_policy_probs(scores, mask, temperature=0.75)

    assert output.probs.shape == scores.shape
    assert output.log_probs.shape == scores.shape
    assert output.mask.dtype == torch.bool
    assert output.temperature == 0.75
