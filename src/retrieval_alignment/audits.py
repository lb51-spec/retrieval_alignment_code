"""Exact-search audit utilities for retrieval-aware DW-MQL.

Exact-search audits compare a production candidate set against an exact top-k
candidate set. These diagnostics measure ANN-induced truncation error through
recall@k, PMR@k, candidate overlap, and value gaps. The functions here are
independent of FAISS and training loops; they operate on tensors produced by
exact or approximate retrieval backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]

from .metrics import (
    best_value_gap,
    candidate_overlap_count,
    candidate_overlap_fraction,
    expected_value_gap,
    normalized_policy_mass_recall,
    pmr_missing_mass,
    policy_mass_recall,
    recall_at_k,
    uniform_policy_mass_baseline,
)
from .policies import masked_softmax


@dataclass(slots=True)
class ExactSearchAuditResult:
    """Container for exact-versus-retrieved candidate diagnostics."""

    recall_at_k: "torch.Tensor | Any"
    overlap_count: "torch.Tensor | Any"
    overlap_fraction: "torch.Tensor | Any"
    pmr_at_k: "torch.Tensor | Any | None" = None
    best_value_gap: "torch.Tensor | Any | None" = None
    expected_value_gap: "torch.Tensor | Any | None" = None


def audit_candidate_sets(
    retrieved_ids: Any,
    exact_ids: Any,
    *,
    retrieved_mask: Any | None = None,
    exact_policy_probs: Any | None = None,
    exact_values: Any | None = None,
    retrieved_values: Any | None = None,
    exact_value_probs: Any | None = None,
    retrieved_value_probs: Any | None = None,
    exact_value_mask: Any | None = None,
    retrieved_value_mask: Any | None = None,
) -> ExactSearchAuditResult:
    """Compare retrieved candidates against exact top-k candidates.

    Optional PMR and value-gap fields are computed only when their required
    tensors are provided.
    """

    recall = recall_at_k(retrieved_ids, exact_ids, mask=retrieved_mask)
    count = candidate_overlap_count(retrieved_ids, exact_ids, mask=retrieved_mask)
    fraction = candidate_overlap_fraction(retrieved_ids, exact_ids, mask=retrieved_mask)

    pmr = None
    if exact_policy_probs is not None:
        pmr = policy_mass_recall(
            retrieved_ids,
            exact_ids,
            exact_policy_probs,
            mask=retrieved_mask,
        )

    best_gap = None
    if exact_values is not None and retrieved_values is not None:
        best_gap = best_value_gap(
            exact_values,
            retrieved_values,
            exact_mask=exact_value_mask,
            retrieved_mask=retrieved_value_mask,
        )

    expected_gap = None
    if (
        exact_values is not None
        and retrieved_values is not None
        and exact_value_probs is not None
        and retrieved_value_probs is not None
    ):
        expected_gap = expected_value_gap(
            exact_values,
            exact_value_probs,
            retrieved_values,
            retrieved_value_probs,
            exact_mask=exact_value_mask,
            retrieved_mask=retrieved_value_mask,
        )

    return ExactSearchAuditResult(
        recall_at_k=recall,
        overlap_count=count,
        overlap_fraction=fraction,
        pmr_at_k=pmr,
        best_value_gap=best_gap,
        expected_value_gap=expected_gap,
    )


def compute_candidate_policy_mass_recall(
    retrieved_ids: Any,
    reference_ids: Any,
    *,
    reference_scores: Any | None = None,
    reference_probs: Any | None = None,
    retrieved_mask: Any | None = None,
    reference_mask: Any | None = None,
    temperature: float = 1.0,
    validate_normalized: bool = True,
    num_actions: int | None = None,
    return_details: bool = False,
) -> Any:
    """Compute PMR@k under a fresh/reference candidate policy.

    The retrieved ids are usually stale or approximate candidates. The
    reference ids are fresh or exact candidates. If ``reference_probs`` are not
    provided, ``reference_scores`` are converted into an ANN-truncated policy
    with masked softmax. The resulting policy-mass recall measures how much
    reference policy support is covered by the retrieved set.
    """

    if reference_probs is None:
        if reference_scores is None:
            raise ValueError("Either reference_probs or reference_scores must be provided.")
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required to compute policy-mass recall.")
        score_tensor = torch.as_tensor(reference_scores)
        if reference_mask is None:
            reference_mask = torch.ones_like(score_tensor, dtype=torch.bool)
        reference_probs = masked_softmax(score_tensor, reference_mask, temperature=temperature)
    elif reference_scores is not None:
        raise ValueError("Provide either reference_probs or reference_scores, not both.")

    pmr = policy_mass_recall(
        retrieved_ids,
        reference_ids,
        reference_probs,
        mask=retrieved_mask,
        exact_mask=reference_mask,
        validate_normalized=validate_normalized,
    )
    if not return_details:
        return pmr
    if num_actions is None:
        raise ValueError("num_actions is required when return_details=True.")
    k = int(torch.as_tensor(retrieved_ids).shape[1]) if torch is not None else len(retrieved_ids[0])
    baseline = uniform_policy_mass_baseline(k, int(num_actions))
    normalized = normalized_policy_mass_recall(pmr, k, int(num_actions))
    missing = pmr_missing_mass(pmr)
    return {
        "pmr": pmr,
        "missing_mass": missing,
        "uniform_baseline": baseline,
        "normalized_pmr": normalized,
    }
