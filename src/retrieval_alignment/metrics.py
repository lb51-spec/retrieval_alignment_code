"""Retrieval and diagnostic metrics for DW-MQL+ANN.

The experiment plan requires metrics that separate learning error from
retrieval-induced truncation error: recall@k, policy-mass recall, candidate
overlap, and exact-versus-retrieved value gaps. These utilities are pure tensor
functions and do not depend on FAISS, networks, replay buffers, or training
loops.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to compute tensor metrics.")


def _as_2d_long(values: Any, *, name: str) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, k].")
    if tensor.numel() == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    return tensor


def _as_2d_float(values: Any, *, name: str, device: "torch.device | None" = None) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values, device=device)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, k].")
    if tensor.numel() == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _optional_mask(mask: Any | None, *, like: "torch.Tensor", name: str) -> "torch.Tensor":
    if mask is None:
        return torch.ones_like(like, dtype=torch.bool)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=like.device)
    if mask_tensor.shape != like.shape:
        raise ValueError(f"{name} shape {tuple(mask_tensor.shape)} must match {tuple(like.shape)}.")
    return mask_tensor


def _validate_retrieval_inputs(
    retrieved_ids: Any,
    exact_ids: Any,
    mask: Any | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    retrieved = _as_2d_long(retrieved_ids, name="retrieved_ids")
    exact = _as_2d_long(exact_ids, name="exact_ids").to(device=retrieved.device)
    if exact.shape[0] != retrieved.shape[0]:
        raise ValueError(
            f"exact_ids batch size {exact.shape[0]} must match retrieved_ids batch size {retrieved.shape[0]}."
        )
    retrieved_mask = _optional_mask(mask, like=retrieved, name="mask")
    return retrieved, exact, retrieved_mask


def _unique_valid_row(values: "torch.Tensor", mask: "torch.Tensor | None" = None) -> set[int]:
    if mask is None:
        return {int(v) for v in values.tolist()}
    return {int(v) for v, valid in zip(values.tolist(), mask.tolist(), strict=True) if bool(valid)}


def candidate_overlap_count(retrieved_ids: Any, exact_ids: Any, mask: Any | None = None) -> "torch.Tensor":
    """Count unique exact candidates recovered by retrieved candidates per row."""

    retrieved, exact, retrieved_mask = _validate_retrieval_inputs(retrieved_ids, exact_ids, mask)
    counts: list[int] = []
    for row in range(retrieved.shape[0]):
        retrieved_set = _unique_valid_row(retrieved[row], retrieved_mask[row])
        exact_set = _unique_valid_row(exact[row])
        counts.append(len(retrieved_set.intersection(exact_set)))
    return torch.tensor(counts, dtype=torch.float32, device=retrieved.device)


def candidate_overlap_fraction(retrieved_ids: Any, exact_ids: Any, mask: Any | None = None) -> "torch.Tensor":
    """Return overlap count divided by the number of unique exact ids."""

    retrieved, exact, retrieved_mask = _validate_retrieval_inputs(retrieved_ids, exact_ids, mask)
    fractions: list[float] = []
    for row in range(retrieved.shape[0]):
        retrieved_set = _unique_valid_row(retrieved[row], retrieved_mask[row])
        exact_set = _unique_valid_row(exact[row])
        if len(exact_set) == 0:
            raise ValueError("exact_ids rows must contain at least one unique id.")
        fractions.append(len(retrieved_set.intersection(exact_set)) / len(exact_set))
    return torch.tensor(fractions, dtype=torch.float32, device=retrieved.device)


def recall_at_k(retrieved_ids: Any, exact_ids: Any, mask: Any | None = None) -> "torch.Tensor":
    """Compute set-style recall@k against exact top-k ids.

    Duplicates in either retrieved or exact ids do not inflate recall.
    """

    return candidate_overlap_fraction(retrieved_ids, exact_ids, mask=mask)


def policy_mass_recall(
    retrieved_ids: Any,
    exact_ids: Any,
    exact_policy_probs: Any,
    mask: Any | None = None,
    exact_mask: Any | None = None,
    *,
    validate_normalized: bool = False,
    atol: float = 1e-5,
) -> "torch.Tensor":
    """Compute policy-mass recall over exact candidates.

    For each row, this sums probabilities of unique exact ids that appear in
    the retrieved set. Duplicate exact ids are counted only once, using their
    first occurrence probability.
    """

    retrieved, exact, retrieved_mask = _validate_retrieval_inputs(retrieved_ids, exact_ids, mask)
    probs = _as_2d_float(exact_policy_probs, name="exact_policy_probs", device=exact.device)
    if probs.shape != exact.shape:
        raise ValueError(
            f"exact_policy_probs shape {tuple(probs.shape)} must match exact_ids shape {tuple(exact.shape)}."
        )
    if torch.any(probs < 0):
        raise ValueError("exact_policy_probs must be nonnegative.")
    exact_valid = _optional_mask(exact_mask, like=exact, name="exact_mask")
    masked_probs = torch.where(exact_valid, probs, torch.zeros_like(probs))
    if validate_normalized and not torch.allclose(
        masked_probs.sum(dim=1),
        torch.ones(probs.shape[0], device=probs.device),
        atol=atol,
    ):
        raise ValueError("exact_policy_probs must sum to one per row.")

    masses: list[float] = []
    for row in range(retrieved.shape[0]):
        retrieved_set = _unique_valid_row(retrieved[row], retrieved_mask[row])
        seen_exact: set[int] = set()
        mass = 0.0
        for action_id, prob, valid in zip(
            exact[row].tolist(),
            probs[row].tolist(),
            exact_valid[row].tolist(),
            strict=True,
        ):
            if not bool(valid):
                continue
            action_int = int(action_id)
            if action_int in seen_exact:
                continue
            seen_exact.add(action_int)
            if action_int in retrieved_set:
                mass += float(prob)
        masses.append(mass)
    return torch.tensor(masses, dtype=probs.dtype, device=probs.device)


def uniform_policy_mass_baseline(k: int, num_actions: int) -> float:
    """Return the PMR expected from a uniform policy over a catalogue.

    This baseline helps prevent raw PMR from being confused with the trivial
    catalogue fraction ``k / num_actions``. It is a normalization reference,
    not a performance target.
    """

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(num_actions, int) or num_actions <= 0:
        raise ValueError("num_actions must be a positive integer.")
    if k > num_actions:
        raise ValueError("k must not exceed num_actions.")
    return float(k) / float(num_actions)


def normalized_policy_mass_recall(
    pmr: Any,
    k: int,
    num_actions: int,
    eps: float = 1e-12,
) -> Any:
    """Normalize PMR by the uniform-policy mass baseline ``k / num_actions``."""

    if not isinstance(eps, int | float) or float(eps) <= 0:
        raise ValueError("eps must be positive.")
    baseline = max(uniform_policy_mass_baseline(k, num_actions), float(eps))
    if torch is None:
        return float(pmr) / baseline
    if isinstance(pmr, torch.Tensor):
        if torch.any(~torch.isfinite(pmr)):
            raise ValueError("pmr must contain only finite values.")
        return pmr / baseline
    pmr_float = float(pmr)
    if not torch.isfinite(torch.tensor(pmr_float)):
        raise ValueError("pmr must be finite.")
    return pmr_float / baseline


def pmr_missing_mass(pmr: Any) -> Any:
    """Return missing policy mass ``1 - PMR``."""

    if torch is not None and isinstance(pmr, torch.Tensor):
        if torch.any(~torch.isfinite(pmr)):
            raise ValueError("pmr must contain only finite values.")
        return 1.0 - pmr
    pmr_float = float(pmr)
    if torch is not None and not torch.isfinite(torch.tensor(pmr_float)):
        raise ValueError("pmr must be finite.")
    return 1.0 - pmr_float


def compute_support_correction_loss(
    pmr: Any,
    normalized_pmr: Any | None = None,
    *,
    metric: str = "missing_mass",
    target: float | None = None,
) -> "torch.Tensor":
    """Return a scalar PMR/support-loss correction term.

    ``missing_mass`` penalizes uncaptured policy mass directly. The
    ``normalized_pmr_margin`` variant penalizes rows whose normalized PMR falls
    below a positive target. The function is tensor-native so callers can pass a
    differentiable PMR proxy when one is available.
    """

    _require_torch()
    pmr_tensor = torch.as_tensor(pmr)
    if not torch.is_floating_point(pmr_tensor):
        pmr_tensor = pmr_tensor.float()
    if pmr_tensor.numel() == 0:
        raise ValueError("pmr must be non-empty.")
    if torch.any(~torch.isfinite(pmr_tensor)):
        raise ValueError("pmr must contain only finite values.")
    if torch.any(pmr_tensor < -1e-6) or torch.any(pmr_tensor > 1.0 + 1e-6):
        raise ValueError("pmr values must lie in [0, 1].")
    pmr_tensor = pmr_tensor.clamp(0.0, 1.0)

    if metric == "missing_mass":
        return (1.0 - pmr_tensor).mean()
    if metric == "normalized_pmr_margin":
        if normalized_pmr is None:
            raise ValueError("normalized_pmr is required for normalized_pmr_margin.")
        if target is None:
            raise ValueError("target is required for normalized_pmr_margin.")
        target_float = float(target)
        if not torch.isfinite(torch.tensor(target_float, device=pmr_tensor.device)) or target_float <= 0:
            raise ValueError("target must be finite and positive.")
        normalized_tensor = torch.as_tensor(normalized_pmr, device=pmr_tensor.device)
        if not torch.is_floating_point(normalized_tensor):
            normalized_tensor = normalized_tensor.float()
        if normalized_tensor.shape != pmr_tensor.shape:
            raise ValueError("normalized_pmr shape must match pmr shape.")
        if torch.any(~torch.isfinite(normalized_tensor)):
            raise ValueError("normalized_pmr must contain only finite values.")
        return torch.relu(torch.as_tensor(target_float, dtype=normalized_tensor.dtype, device=normalized_tensor.device) - normalized_tensor).mean()
    raise ValueError("support correction metric must be one of {'missing_mass', 'normalized_pmr_margin'}.")


def best_value_gap(
    exact_values: Any,
    retrieved_values: Any,
    exact_mask: Any | None = None,
    retrieved_mask: Any | None = None,
) -> "torch.Tensor":
    """Return ``max(exact_values) - max(retrieved_values)`` per row."""

    exact = _as_2d_float(exact_values, name="exact_values")
    retrieved = _as_2d_float(retrieved_values, name="retrieved_values", device=exact.device)
    if exact.shape[0] != retrieved.shape[0]:
        raise ValueError("exact_values and retrieved_values must have the same batch size.")
    exact_valid = _optional_mask(exact_mask, like=exact, name="exact_mask")
    retrieved_valid = _optional_mask(retrieved_mask, like=retrieved, name="retrieved_mask")
    if torch.any(~exact_valid.any(dim=1)):
        raise ValueError("Each exact_values row must contain at least one valid candidate.")
    if torch.any(~retrieved_valid.any(dim=1)):
        raise ValueError("Each retrieved_values row must contain at least one valid candidate.")

    neg_large = -torch.finfo(exact.dtype).max
    exact_max = torch.where(exact_valid, exact, torch.full_like(exact, neg_large)).max(dim=1).values
    retrieved_max = torch.where(
        retrieved_valid,
        retrieved,
        torch.full_like(retrieved, neg_large),
    ).max(dim=1).values
    return exact_max - retrieved_max


def expected_value_gap(
    exact_values: Any,
    exact_probs: Any,
    retrieved_values: Any,
    retrieved_probs: Any,
    exact_mask: Any | None = None,
    retrieved_mask: Any | None = None,
) -> "torch.Tensor":
    """Return expected exact value minus expected retrieved value per row."""

    exact = _as_2d_float(exact_values, name="exact_values")
    exact_p = _as_2d_float(exact_probs, name="exact_probs", device=exact.device)
    retrieved = _as_2d_float(retrieved_values, name="retrieved_values", device=exact.device)
    retrieved_p = _as_2d_float(retrieved_probs, name="retrieved_probs", device=exact.device)
    if exact_p.shape != exact.shape:
        raise ValueError("exact_probs shape must match exact_values shape.")
    if retrieved_p.shape != retrieved.shape:
        raise ValueError("retrieved_probs shape must match retrieved_values shape.")
    if exact.shape[0] != retrieved.shape[0]:
        raise ValueError("exact and retrieved value tensors must have the same batch size.")
    if torch.any(exact_p < 0) or torch.any(retrieved_p < 0):
        raise ValueError("probabilities must be nonnegative.")

    exact_valid = _optional_mask(exact_mask, like=exact, name="exact_mask").to(dtype=exact.dtype)
    retrieved_valid = _optional_mask(retrieved_mask, like=retrieved, name="retrieved_mask").to(dtype=retrieved.dtype)
    exact_expected = (exact * exact_p * exact_valid).sum(dim=1)
    retrieved_expected = (retrieved * retrieved_p * retrieved_valid).sum(dim=1)
    return exact_expected - retrieved_expected


def _validate_ranking_inputs(
    scores: Any,
    positive_ids: Any,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    score_tensor = _as_2d_float(scores, name="scores")
    positive_tensor = torch.as_tensor(positive_ids, dtype=torch.long, device=score_tensor.device)
    if positive_tensor.ndim != 1 or positive_tensor.shape[0] != score_tensor.shape[0]:
        raise ValueError("positive_ids must have shape [batch].")
    if torch.any(positive_tensor < 0) or torch.any(positive_tensor >= score_tensor.shape[1]):
        raise ValueError("positive_ids must contain valid action indices.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    return score_tensor, positive_tensor, min(k, int(score_tensor.shape[1]))


def single_positive_ranking_metrics(
    scores: Any,
    positive_ids: Any,
    k: int,
) -> dict[str, float | int]:
    """Return full-catalogue ranking metrics for one positive action per row."""

    score_tensor, positive_tensor, safe_k = _validate_ranking_inputs(scores, positive_ids, k=k)
    target_scores = score_tensor.gather(1, positive_tensor[:, None]).squeeze(1)
    ranks = (score_tensor > target_scores[:, None]).sum(dim=1) + 1
    hits = ranks <= safe_k
    ndcg = torch.where(
        hits,
        1.0 / torch.log2(ranks.to(dtype=torch.float32) + 1.0),
        torch.zeros_like(ranks, dtype=torch.float32),
    )
    return {
        "num_eval_rows": int(score_tensor.shape[0]),
        "catalogue_size": int(score_tensor.shape[1]),
        "k": safe_k,
        "hit_rate_at_k": float(hits.to(dtype=torch.float32).mean().item()),
        "recall_at_k": float(hits.to(dtype=torch.float32).mean().item()),
        "ndcg_at_k": float(ndcg.mean().item()),
        "mean_reciprocal_rank": float((1.0 / ranks.to(dtype=torch.float32)).mean().item()),
        "mean_positive_rank": float(ranks.to(dtype=torch.float32).mean().item()),
    }


def candidate_constrained_ranking_metrics(
    scores: Any,
    positive_ids: Any,
    candidate_ids: Any,
    k: int,
) -> dict[str, float | int]:
    """Return held-out ranking metrics after restricting actions to candidates."""

    score_tensor, positive_tensor, _ = _validate_ranking_inputs(scores, positive_ids, k=k)
    candidates = _as_2d_long(candidate_ids, name="candidate_ids").to(device=score_tensor.device)
    if candidates.shape[0] != score_tensor.shape[0]:
        raise ValueError("candidate_ids batch size must match scores batch size.")
    if torch.any(candidates < 0) or torch.any(candidates >= score_tensor.shape[1]):
        raise ValueError("candidate_ids must contain valid action indices.")
    safe_k = min(k, int(candidates.shape[1]))
    candidate_scores = score_tensor.gather(1, candidates)
    order = torch.argsort(candidate_scores, dim=1, descending=True)
    ranked_ids = candidates.gather(1, order)
    matches = ranked_ids == positive_tensor[:, None]
    retrieved = matches.any(dim=1)
    first_positions = matches.to(dtype=torch.int64).argmax(dim=1) + 1
    missing_rank = torch.full_like(first_positions, candidates.shape[1] + 1)
    ranks = torch.where(retrieved, first_positions, missing_rank)
    hits = retrieved & (ranks <= safe_k)
    ndcg = torch.where(
        hits,
        1.0 / torch.log2(ranks.to(dtype=torch.float32) + 1.0),
        torch.zeros_like(ranks, dtype=torch.float32),
    )
    reciprocal_rank = torch.where(
        retrieved,
        1.0 / ranks.to(dtype=torch.float32),
        torch.zeros_like(ranks, dtype=torch.float32),
    )
    return {
        "num_eval_rows": int(score_tensor.shape[0]),
        "candidate_set_size": int(candidates.shape[1]),
        "k": safe_k,
        "positive_retrieval_rate": float(retrieved.to(dtype=torch.float32).mean().item()),
        "candidate_hit_rate_at_k": float(hits.to(dtype=torch.float32).mean().item()),
        "candidate_recall_at_k": float(hits.to(dtype=torch.float32).mean().item()),
        "candidate_ndcg_at_k": float(ndcg.mean().item()),
        "candidate_mean_reciprocal_rank": float(reciprocal_rank.mean().item()),
    }


def fixed_reference_retrieval_metrics(
    reference_scores: Any,
    candidate_ids: Any,
    temperature: float = 1.0,
) -> dict[str, float | int]:
    """Measure candidate quality under one scorer shared across methods."""

    scores = _as_2d_float(reference_scores, name="reference_scores")
    candidates = _as_2d_long(candidate_ids, name="candidate_ids").to(device=scores.device)
    if candidates.shape[0] != scores.shape[0]:
        raise ValueError("candidate_ids batch size must match reference_scores batch size.")
    if torch.any(candidates < 0) or torch.any(candidates >= scores.shape[1]):
        raise ValueError("candidate_ids must contain valid action indices.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")

    duplicate = candidates[:, :, None] == candidates[:, None, :]
    earlier = torch.tril(duplicate, diagonal=-1).any(dim=2)
    unique_mask = ~earlier
    full_probs = torch.softmax(scores / float(temperature), dim=1)
    candidate_scores = scores.gather(1, candidates)
    candidate_mass = full_probs.gather(1, candidates) * unique_mask.to(dtype=full_probs.dtype)
    pmr = candidate_mass.sum(dim=1)
    conditional_probs = candidate_mass / pmr[:, None].clamp_min(torch.finfo(full_probs.dtype).tiny)
    full_expected = (full_probs * scores).sum(dim=1)
    retrieved_expected = (conditional_probs * candidate_scores).sum(dim=1)
    best_gap = scores.max(dim=1).values - torch.where(
        unique_mask,
        candidate_scores,
        torch.full_like(candidate_scores, -torch.inf),
    ).max(dim=1).values

    safe_k = min(int(candidates.shape[1]), int(scores.shape[1]))
    reference_topk = torch.topk(scores, k=safe_k, dim=1).indices
    overlap = candidate_overlap_fraction(candidates, reference_topk)
    return {
        "num_eval_rows": int(scores.shape[0]),
        "candidate_set_size": int(candidates.shape[1]),
        "reference_pmr_at_k": float(pmr.mean().item()),
        "reference_missing_mass_at_k": float((1.0 - pmr).mean().item()),
        "reference_best_value_gap": float(best_gap.mean().item()),
        "reference_expected_value_gap": float((full_expected - retrieved_expected).mean().item()),
        "reference_topk_overlap_at_k": float(overlap.mean().item()),
    }


def effective_sample_size(weights: "torch.Tensor") -> "torch.Tensor":
    """Return the standard ESS for a one-dimensional weight tensor."""

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to compute tensor metrics.")
    numerator = weights.sum().square()
    denominator = weights.square().sum().clamp_min(torch.finfo(weights.dtype).eps)
    return numerator / denominator
