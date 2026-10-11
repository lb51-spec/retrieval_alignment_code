"""Tests for retrieval metrics used by exact-search audits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.metrics import (
    best_value_gap,
    candidate_constrained_ranking_metrics,
    candidate_overlap_count,
    candidate_overlap_fraction,
    compute_support_correction_loss,
    expected_value_gap,
    fixed_reference_retrieval_metrics,
    normalized_policy_mass_recall,
    pmr_missing_mass,
    policy_mass_recall,
    recall_at_k,
    single_positive_ranking_metrics,
    uniform_policy_mass_baseline,
)


def test_recall_at_k_exact_match_equals_one() -> None:
    recall = recall_at_k(
        retrieved_ids=torch.tensor([[1, 2, 3]]),
        exact_ids=torch.tensor([[1, 2, 3]]),
    )

    assert recall.tolist() == pytest.approx([1.0])


def test_recall_at_k_partial_match() -> None:
    recall = recall_at_k(
        retrieved_ids=torch.tensor([[1, 9, 8]]),
        exact_ids=torch.tensor([[1, 2, 3]]),
    )

    assert recall.tolist() == pytest.approx([1.0 / 3.0])


def test_recall_at_k_ignores_duplicates() -> None:
    recall = recall_at_k(
        retrieved_ids=torch.tensor([[1, 1, 1]]),
        exact_ids=torch.tensor([[1, 2, 3]]),
    )

    assert recall.tolist() == pytest.approx([1.0 / 3.0])


def test_recall_at_k_respects_retrieved_mask() -> None:
    recall = recall_at_k(
        retrieved_ids=torch.tensor([[1, 2, 3]]),
        exact_ids=torch.tensor([[1, 2, 3]]),
        mask=torch.tensor([[True, False, False]]),
    )

    assert recall.tolist() == pytest.approx([1.0 / 3.0])


def test_overlap_count_and_fraction() -> None:
    retrieved = torch.tensor([[1, 2, 9], [4, 5, 5]])
    exact = torch.tensor([[1, 2, 3], [5, 6, 7]])

    count = candidate_overlap_count(retrieved, exact)
    fraction = candidate_overlap_fraction(retrieved, exact)

    assert count.tolist() == pytest.approx([2.0, 1.0])
    assert fraction.tolist() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])


def test_policy_mass_recall_sums_captured_exact_policy_mass() -> None:
    pmr = policy_mass_recall(
        retrieved_ids=torch.tensor([[10, 30]]),
        exact_ids=torch.tensor([[10, 20, 30]]),
        exact_policy_probs=torch.tensor([[0.2, 0.3, 0.5]]),
    )

    assert pmr.tolist() == pytest.approx([0.7])


def test_policy_mass_recall_ignores_duplicates() -> None:
    pmr = policy_mass_recall(
        retrieved_ids=torch.tensor([[10, 10, 10]]),
        exact_ids=torch.tensor([[10, 10, 20]]),
        exact_policy_probs=torch.tensor([[0.2, 0.4, 0.4]]),
    )

    assert pmr.tolist() == pytest.approx([0.2])


def test_policy_mass_recall_respects_mask() -> None:
    pmr = policy_mass_recall(
        retrieved_ids=torch.tensor([[10, 20, 30]]),
        exact_ids=torch.tensor([[10, 20, 30]]),
        exact_policy_probs=torch.tensor([[0.2, 0.3, 0.5]]),
        mask=torch.tensor([[True, False, True]]),
    )

    assert pmr.tolist() == pytest.approx([0.7])


def test_policy_mass_recall_respects_exact_mask() -> None:
    pmr = policy_mass_recall(
        retrieved_ids=torch.tensor([[10, 20, 30]]),
        exact_ids=torch.tensor([[10, 20, 30]]),
        exact_policy_probs=torch.tensor([[0.25, 0.25, 0.50]]),
        exact_mask=torch.tensor([[True, False, True]]),
        validate_normalized=False,
    )

    assert pmr.tolist() == pytest.approx([0.75])


def test_uniform_policy_mass_baseline_equals_k_over_catalogue_size() -> None:
    assert uniform_policy_mass_baseline(5, 100) == pytest.approx(0.05)


def test_normalized_policy_mass_recall_uses_uniform_baseline() -> None:
    assert normalized_policy_mass_recall(0.05, 5, 100) == pytest.approx(1.0)
    assert normalized_policy_mass_recall(0.15, 5, 100) > 1.0
    tensor = normalized_policy_mass_recall(torch.tensor([0.05, 0.10]), 5, 100)
    assert tensor.tolist() == pytest.approx([1.0, 2.0])


def test_pmr_missing_mass_is_one_minus_pmr() -> None:
    assert pmr_missing_mass(0.25) == pytest.approx(0.75)
    assert pmr_missing_mass(torch.tensor([0.25, 1.0])).tolist() == pytest.approx([0.75, 0.0])


def test_compute_support_correction_loss_missing_mass() -> None:
    loss = compute_support_correction_loss(torch.tensor([0.25, 0.75]), metric="missing_mass")

    assert loss.item() == pytest.approx(0.5)


def test_compute_support_correction_loss_normalized_pmr_margin() -> None:
    loss = compute_support_correction_loss(
        torch.tensor([0.5, 0.7]),
        normalized_pmr=torch.tensor([1.0, 3.0]),
        metric="normalized_pmr_margin",
        target=2.0,
    )

    assert loss.item() == pytest.approx(0.5)


def test_compute_support_correction_loss_rejects_invalid_metric() -> None:
    with pytest.raises(ValueError, match="support correction metric"):
        compute_support_correction_loss(torch.tensor([0.5]), metric="bad")


def test_pmr_normalization_rejects_invalid_catalogue_values() -> None:
    with pytest.raises(ValueError, match="k"):
        uniform_policy_mass_baseline(0, 10)
    with pytest.raises(ValueError, match="num_actions"):
        uniform_policy_mass_baseline(1, 0)
    with pytest.raises(ValueError, match="exceed"):
        uniform_policy_mass_baseline(11, 10)


def test_policy_mass_recall_validate_normalized_rejects_bad_probs() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        policy_mass_recall(
            retrieved_ids=torch.tensor([[10, 20]]),
            exact_ids=torch.tensor([[10, 20]]),
            exact_policy_probs=torch.tensor([[0.2, 0.2]]),
            validate_normalized=True,
        )


def test_best_value_gap() -> None:
    gap = best_value_gap(
        exact_values=torch.tensor([[1.0, 5.0, 3.0]]),
        retrieved_values=torch.tensor([[2.0, 4.0]]),
    )

    assert gap.tolist() == pytest.approx([1.0])


def test_best_value_gap_ignores_invalid_retrieved_values() -> None:
    gap = best_value_gap(
        exact_values=torch.tensor([[1.0, 5.0, 3.0]]),
        retrieved_values=torch.tensor([[100.0, 4.0]]),
        retrieved_mask=torch.tensor([[False, True]]),
    )

    assert gap.tolist() == pytest.approx([1.0])


def test_expected_value_gap() -> None:
    gap = expected_value_gap(
        exact_values=torch.tensor([[1.0, 3.0]]),
        exact_probs=torch.tensor([[0.25, 0.75]]),
        retrieved_values=torch.tensor([[2.0, 4.0]]),
        retrieved_probs=torch.tensor([[0.5, 0.5]]),
    )

    assert gap.tolist() == pytest.approx([-0.5])


def test_shape_mismatch_errors() -> None:
    with pytest.raises(ValueError, match="shape|batch"):
        recall_at_k(torch.tensor([[1, 2]]), torch.tensor([[1, 2], [3, 4]]))

    with pytest.raises(ValueError, match="shape"):
        expected_value_gap(
            exact_values=torch.tensor([[1.0, 2.0]]),
            exact_probs=torch.tensor([[1.0]]),
            retrieved_values=torch.tensor([[1.0, 2.0]]),
            retrieved_probs=torch.tensor([[0.5, 0.5]]),
        )


def test_non_finite_input_errors() -> None:
    with pytest.raises(ValueError, match="finite"):
        best_value_gap(
            exact_values=torch.tensor([[1.0, float("nan")]]),
            retrieved_values=torch.tensor([[1.0, 2.0]]),
        )


def test_negative_probability_errors() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        policy_mass_recall(
            retrieved_ids=torch.tensor([[1, 2]]),
            exact_ids=torch.tensor([[1, 2]]),
            exact_policy_probs=torch.tensor([[1.0, -0.1]]),
        )

    with pytest.raises(ValueError, match="nonnegative"):
        expected_value_gap(
            exact_values=torch.tensor([[1.0, 2.0]]),
            exact_probs=torch.tensor([[0.5, 0.5]]),
            retrieved_values=torch.tensor([[1.0, 2.0]]),
            retrieved_probs=torch.tensor([[1.0, -0.1]]),
        )


def test_all_invalid_retrieved_rows_raise_when_max_required() -> None:
    with pytest.raises(ValueError, match="retrieved_values row"):
        best_value_gap(
            exact_values=torch.tensor([[1.0, 2.0]]),
            retrieved_values=torch.tensor([[1.0, 2.0]]),
            retrieved_mask=torch.tensor([[False, False]]),
        )


def test_single_positive_ranking_metrics_match_hand_computed_values() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.4, 0.3]])
    positives = torch.tensor([1, 2])

    result = single_positive_ranking_metrics(scores, positives, k=2)

    assert result["hit_rate_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(0.5)
    assert result["ndcg_at_k"] == pytest.approx(0.5)
    assert result["mean_reciprocal_rank"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert result["mean_positive_rank"] == pytest.approx(2.0)


def test_candidate_constrained_metrics_count_missing_positive_as_miss() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.4, 0.3]])
    positives = torch.tensor([1, 2])
    candidates = torch.tensor([[1, 0], [0, 1]])

    result = candidate_constrained_ranking_metrics(scores, positives, candidates, k=2)

    assert result["candidate_hit_rate_at_k"] == pytest.approx(0.5)
    assert result["candidate_ndcg_at_k"] == pytest.approx(0.5)
    assert result["positive_retrieval_rate"] == pytest.approx(0.5)


def test_fixed_reference_metrics_use_reference_scores() -> None:
    reference = torch.tensor([[4.0, 1.0, 0.0]])
    candidates = torch.tensor([[1, 2]])

    result = fixed_reference_retrieval_metrics(reference, candidates, temperature=1.0)

    expected_mass = torch.softmax(reference, dim=1)[0, 1:].sum().item()
    assert result["reference_pmr_at_k"] == pytest.approx(expected_mass)
    assert result["reference_best_value_gap"] == pytest.approx(3.0)
