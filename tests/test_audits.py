"""Tests for exact-search audit wrapper utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.audits import ExactSearchAuditResult, audit_candidate_sets, compute_candidate_policy_mass_recall


def test_audit_wrapper_returns_expected_fields() -> None:
    result = audit_candidate_sets(
        retrieved_ids=torch.tensor([[1, 3, 9]]),
        exact_ids=torch.tensor([[1, 2, 3]]),
        exact_policy_probs=torch.tensor([[0.2, 0.3, 0.5]]),
        exact_values=torch.tensor([[1.0, 5.0, 3.0]]),
        retrieved_values=torch.tensor([[1.0, 3.0, 0.0]]),
        exact_value_probs=torch.tensor([[0.2, 0.3, 0.5]]),
        retrieved_value_probs=torch.tensor([[0.5, 0.5, 0.0]]),
    )

    assert isinstance(result, ExactSearchAuditResult)
    assert result.recall_at_k.tolist() == pytest.approx([2.0 / 3.0])
    assert result.overlap_count.tolist() == pytest.approx([2.0])
    assert result.overlap_fraction.tolist() == pytest.approx([2.0 / 3.0])
    assert result.pmr_at_k is not None
    assert result.pmr_at_k.tolist() == pytest.approx([0.7])
    assert result.best_value_gap is not None
    assert result.best_value_gap.tolist() == pytest.approx([2.0])
    assert result.expected_value_gap is not None
    assert result.expected_value_gap.tolist() == pytest.approx([1.2])


def test_audit_wrapper_optional_fields_are_none_when_inputs_missing() -> None:
    result = audit_candidate_sets(
        retrieved_ids=torch.tensor([[1, 3, 9]]),
        exact_ids=torch.tensor([[1, 2, 3]]),
    )

    assert result.pmr_at_k is None
    assert result.best_value_gap is None
    assert result.expected_value_gap is None


def test_compute_candidate_policy_mass_recall_exact_match_is_one() -> None:
    pmr = compute_candidate_policy_mass_recall(
        retrieved_ids=torch.tensor([[1, 2, 3]]),
        reference_ids=torch.tensor([[1, 2, 3]]),
        reference_scores=torch.tensor([[1.0, 2.0, 3.0]]),
    )

    assert pmr.tolist() == pytest.approx([1.0])


def test_compute_candidate_policy_mass_recall_partial_match_from_probs() -> None:
    pmr = compute_candidate_policy_mass_recall(
        retrieved_ids=torch.tensor([[1, 3, 9]]),
        reference_ids=torch.tensor([[1, 2, 3]]),
        reference_probs=torch.tensor([[0.2, 0.3, 0.5]]),
    )

    assert pmr.tolist() == pytest.approx([0.7])


def test_compute_candidate_policy_mass_recall_ignores_duplicates_and_masks() -> None:
    pmr = compute_candidate_policy_mass_recall(
        retrieved_ids=torch.tensor([[1, 1, 2, 3]]),
        reference_ids=torch.tensor([[1, 2, 3, 4]]),
        reference_probs=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        retrieved_mask=torch.tensor([[True, True, False, True]]),
        reference_mask=torch.tensor([[True, True, True, False]]),
        validate_normalized=False,
    )

    assert pmr.tolist() == pytest.approx([0.4])


def test_compute_candidate_policy_mass_recall_rejects_invalid_probs() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        compute_candidate_policy_mass_recall(
            retrieved_ids=torch.tensor([[1, 2]]),
            reference_ids=torch.tensor([[1, 2]]),
            reference_probs=torch.tensor([[1.1, -0.1]]),
        )


def test_compute_candidate_policy_mass_recall_temperature_changes_mass() -> None:
    retrieved = torch.tensor([[1]])
    reference = torch.tensor([[1, 2]])
    scores = torch.tensor([[1.0, 3.0]])

    cold = compute_candidate_policy_mass_recall(
        retrieved,
        reference,
        reference_scores=scores,
        temperature=0.5,
    )
    warm = compute_candidate_policy_mass_recall(
        retrieved,
        reference,
        reference_scores=scores,
        temperature=2.0,
    )

    assert cold.item() < warm.item()


def test_compute_candidate_policy_mass_recall_can_return_normalized_details() -> None:
    details = compute_candidate_policy_mass_recall(
        retrieved_ids=torch.tensor([[1, 3]]),
        reference_ids=torch.tensor([[1, 2, 3]]),
        reference_probs=torch.tensor([[0.2, 0.3, 0.5]]),
        num_actions=10,
        return_details=True,
    )

    assert details["pmr"].tolist() == pytest.approx([0.7])
    assert details["missing_mass"].tolist() == pytest.approx([0.3])
    assert details["uniform_baseline"] == pytest.approx(0.2)
    assert details["normalized_pmr"].tolist() == pytest.approx([3.5])


def test_compute_candidate_policy_mass_recall_details_require_catalogue_size() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        compute_candidate_policy_mass_recall(
            retrieved_ids=torch.tensor([[1, 2]]),
            reference_ids=torch.tensor([[1, 2]]),
            reference_probs=torch.tensor([[0.5, 0.5]]),
            return_details=True,
        )
