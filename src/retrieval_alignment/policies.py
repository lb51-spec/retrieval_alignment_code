"""Masked candidate-policy utilities for ANN-truncated policies.

The formal policy in ``docs/equations.md`` is the full policy restricted to a
retrieved candidate set and renormalized:

``pi_k(a | s) = pi(a | s) 1[a in C_k(s)] / Z_k(s)``.

This module implements the ANN-independent tensor utilities needed for that
renormalization over fixed-size padded candidate sets. It does not know about
FAISS, replay buffers, critics, density ratios, or training loops. See
``docs/algorithm_pseudocode.md`` and ``docs/code_architecture.md`` for how
these utilities fit into Bellman-target construction later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class CandidatePolicyOutput:
    """Masked policy probabilities and log-probabilities over candidates."""

    probs: "torch.Tensor | Any"
    log_probs: "torch.Tensor | Any"
    mask: "torch.Tensor | Any"
    temperature: float


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to compute candidate policies.")


def _validate_scores_and_mask(
    scores: "torch.Tensor | Any",
    mask: "torch.Tensor | Any",
    temperature: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Validate and normalize score/mask inputs."""

    _require_torch()
    score_tensor = torch.as_tensor(scores)
    if not torch.is_floating_point(score_tensor):
        score_tensor = score_tensor.float()
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=score_tensor.device)

    if score_tensor.shape != mask_tensor.shape:
        raise ValueError(
            f"scores shape {tuple(score_tensor.shape)} must match mask shape {tuple(mask_tensor.shape)}."
        )
    if score_tensor.ndim != 2:
        raise ValueError("scores and mask must be 2D tensors shaped [batch, candidates].")
    if not isinstance(temperature, int | float) or temperature <= 0:
        raise ValueError("temperature must be a positive finite scalar.")
    if not torch.isfinite(torch.tensor(float(temperature), dtype=score_tensor.dtype)):
        raise ValueError("temperature must be a positive finite scalar.")
    if torch.any(~mask_tensor.any(dim=1)):
        raise ValueError("Each batch row must contain at least one valid candidate.")
    if torch.any(~torch.isfinite(score_tensor[mask_tensor])):
        raise ValueError("Valid candidate scores must be finite.")

    return score_tensor, mask_tensor


def masked_softmax(
    scores: "torch.Tensor | Any",
    mask: "torch.Tensor | Any",
    temperature: float = 1.0,
) -> "torch.Tensor":
    """Compute a stable softmax over valid candidates only.

    Invalid candidates receive probability exactly zero. Valid probabilities
    are renormalized per row, implementing the candidate-supported distribution
    used by the ANN-truncated policy.
    """

    score_tensor, mask_tensor = _validate_scores_and_mask(scores, mask, temperature)
    scaled = score_tensor / float(temperature)
    neg_large = -torch.finfo(scaled.dtype).max
    masked_scores = torch.where(mask_tensor, scaled, torch.full_like(scaled, neg_large))
    shifted = masked_scores - masked_scores.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(shifted) * mask_tensor.to(dtype=shifted.dtype)
    denom = exp_scores.sum(dim=1, keepdim=True)
    probs = exp_scores / denom
    return torch.where(mask_tensor, probs, torch.zeros_like(probs))


def masked_log_softmax(
    scores: "torch.Tensor | Any",
    mask: "torch.Tensor | Any",
    temperature: float = 1.0,
) -> "torch.Tensor":
    """Return safe log-probabilities for a masked candidate policy.

    Valid candidates receive finite log-probabilities. Invalid candidates are
    assigned the most negative finite value for the tensor dtype rather than
    ``-inf`` so downstream diagnostics can avoid infinities while still treating
    invalid positions as impossible actions.
    """

    probs = masked_softmax(scores, mask, temperature)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=probs.device)
    tiny = torch.finfo(probs.dtype).tiny
    invalid_log_prob = torch.finfo(probs.dtype).min
    log_probs = torch.log(probs.clamp_min(tiny))
    return torch.where(mask_tensor, log_probs, torch.full_like(log_probs, invalid_log_prob))


def candidate_policy_probs(
    scores: "torch.Tensor | Any",
    mask: "torch.Tensor | Any",
    temperature: float = 1.0,
) -> CandidatePolicyOutput:
    """Return probabilities and log-probabilities for the truncated policy."""

    probs = masked_softmax(scores, mask, temperature)
    log_probs = masked_log_softmax(scores, mask, temperature)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=probs.device)
    return CandidatePolicyOutput(
        probs=probs,
        log_probs=log_probs,
        mask=mask_tensor,
        temperature=float(temperature),
    )


def greedy_candidate_action(
    scores: "torch.Tensor | Any",
    candidate_indices: "torch.Tensor | Any",
    mask: "torch.Tensor | Any",
) -> "torch.Tensor":
    """Select the highest-scoring valid candidate id for each batch row."""

    score_tensor, mask_tensor = _validate_scores_and_mask(scores, mask, temperature=1.0)
    candidate_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=score_tensor.device)
    if candidate_tensor.shape != score_tensor.shape:
        raise ValueError(
            "candidate_indices must have the same shape as scores and mask: "
            f"got {tuple(candidate_tensor.shape)} and {tuple(score_tensor.shape)}."
        )

    neg_large = -torch.finfo(score_tensor.dtype).max
    masked_scores = torch.where(mask_tensor, score_tensor, torch.full_like(score_tensor, neg_large))
    positions = masked_scores.argmax(dim=1)
    return candidate_tensor.gather(1, positions.unsqueeze(1)).squeeze(1)


def sample_candidate_action(
    scores: "torch.Tensor | Any",
    candidate_indices: "torch.Tensor | Any",
    mask: "torch.Tensor | Any",
    temperature: float = 1.0,
    generator: "torch.Generator | None" = None,
) -> "torch.Tensor":
    """Sample valid candidate ids from the ANN-truncated policy.

    Sampling is deterministic when a seeded ``torch.Generator`` is supplied.
    """

    probs = masked_softmax(scores, mask, temperature)
    candidate_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=probs.device)
    if candidate_tensor.shape != probs.shape:
        raise ValueError(
            "candidate_indices must have the same shape as scores and mask: "
            f"got {tuple(candidate_tensor.shape)} and {tuple(probs.shape)}."
        )

    positions = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator)
    return candidate_tensor.gather(1, positions).squeeze(1)
