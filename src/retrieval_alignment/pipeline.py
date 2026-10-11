"""Forward-only pipeline smoke step for DW-MQL+ANN components.

This module wires existing pieces together for integration smoke tests:

``synthetic data -> replay batch -> action index -> candidates -> policy ->
Bellman residual -> density weights -> stationarity moment -> audit metrics``.

It is not a training loop, not a benchmark, and not an optimizer. The function
uses ``torch.no_grad()`` and performs no backward pass or parameter updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]

from .ann_index import make_action_index
from .audits import audit_candidate_sets
from .candidate_sets import build_next_state_candidates
from .density import effective_sample_size, make_density_weights, weight_entropy
from .losses import (
    ann_bellman_residual,
    bellman_target,
    masked_expected_q_value,
    weighted_squared_bellman_loss,
)
from .networks import DWMQLNetworks
from .policies import masked_softmax
from .stationarity import squared_stationarity_loss, stationarity_moment
from .synthetic import SyntheticOfflineData


@dataclass(slots=True)
class PipelineSmokeResult:
    """Outputs from one forward-only pipeline smoke step."""

    batch_size: int
    num_candidates: int
    candidate_ids: "torch.Tensor | Any"
    candidate_mask: "torch.Tensor | Any"
    candidate_probs: "torch.Tensor | Any"
    current_q: "torch.Tensor | Any"
    next_candidate_q: "torch.Tensor | Any"
    bellman_target: "torch.Tensor | Any"
    bellman_residual: "torch.Tensor | Any"
    bellman_loss: "torch.Tensor | Any"
    density_weights: "torch.Tensor | Any"
    ess: "torch.Tensor | Any"
    weight_entropy: "torch.Tensor | Any"
    stationarity_moment: "torch.Tensor | Any"
    stationarity_loss: "torch.Tensor | Any"
    recall_at_k: "torch.Tensor | Any"
    pmr_at_k: "torch.Tensor | Any | None" = None
    best_value_gap: "torch.Tensor | Any | None" = None
    exact_backend_recall_at_k: "torch.Tensor | Any | None" = None
    exact_backend_overlap_fraction: "torch.Tensor | Any | None" = None
    exact_backend_best_value_gap: "torch.Tensor | Any | None" = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run the pipeline smoke step.")


def _validate_inputs(
    data: SyntheticOfflineData,
    batch_size: int,
    k: int,
    gamma: float,
    temperature: float,
    index_backend: str,
    index_metric: str,
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> None:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > int(data.action_features.shape[0]):
        raise ValueError("k must not exceed the number of actions.")
    if not isinstance(gamma, int | float) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of 'exact', 'faiss', 'faiss_ivf', or 'faiss_hnsw'.")
    if index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be either 'dot' or 'cosine'.")
    for name, value in {"index_nlist": index_nlist, "index_nprobe": index_nprobe}.items():
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be None or a positive integer.")


def _gather_action_features(action_features: "torch.Tensor", candidate_ids: "torch.Tensor") -> "torch.Tensor":
    safe_ids = candidate_ids.clamp_min(0)
    return action_features[safe_ids]


def _encode_candidate_actions(model: DWMQLNetworks, candidate_action_features: "torch.Tensor") -> "torch.Tensor":
    batch_size, num_candidates, action_dim = candidate_action_features.shape
    flat = candidate_action_features.reshape(batch_size * num_candidates, action_dim)
    encoded = model.encode_actions(flat)
    return encoded.reshape(batch_size, num_candidates, -1)


def run_pipeline_smoke_step(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    batch_size: int,
    k: int,
    gamma: float,
    generator: "torch.Generator | None" = None,
    temperature: float = 1.0,
    density_max_weight: float | None = None,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
    audit_exact_backend: bool = True,
) -> PipelineSmokeResult:
    """Run one no-optimization forward pass through the prototype pipeline.

    The action index is built from encoded action catalogue embeddings. The
    default backend remains exact dot-product retrieval; optional FAISS support
    uses the same index interface when installed. When ``audit_exact_backend``
    is true, a second exact index is used as a smoke-scale retrieval reference.
    """

    _require_torch()
    if not isinstance(audit_exact_backend, bool):
        raise TypeError("audit_exact_backend must be a bool.")
    _validate_inputs(data, batch_size, k, gamma, temperature, index_backend, index_metric, index_nlist, index_nprobe)
    model.eval()

    with torch.no_grad():
        batch = data.dataset.sample(batch_size, generator=generator)
        action_features = data.action_features.to(device=batch.states.device)
        action_ids = data.action_ids.to(device=batch.states.device)

        catalogue_embeddings = model.encode_actions(action_features)
        selected_index = make_action_index(
            catalogue_embeddings,
            action_ids=action_ids,
            backend=index_backend,
            metric=index_metric,
            normalize=index_metric == "cosine",
            nlist=index_nlist,
            nprobe=index_nprobe,
        )
        next_state_embeddings = model.encode_states(batch.next_states)
        selected_search = selected_index.search(next_state_embeddings, k=k)
        if audit_exact_backend:
            exact_index = make_action_index(
                catalogue_embeddings,
                action_ids=action_ids,
                backend="exact",
                metric=index_metric,
                normalize=index_metric == "cosine",
            )
            exact_search = exact_index.search(next_state_embeddings, k=k)
        else:
            exact_search = selected_search

        candidates = build_next_state_candidates(
            selected_search.ids,
            logged_action_indices=batch.next_logged_action_ids,
        )
        if torch.any(~candidates.mask.any(dim=1)):
            raise ValueError("Each candidate row must contain at least one valid candidate.")

        current_state_embeddings = model.encode_states(batch.states)
        current_action_embeddings = model.encode_actions(batch.actions)
        current_q = model.q_values(current_state_embeddings, current_action_embeddings)

        candidate_action_features = _gather_action_features(action_features, candidates.ids)
        candidate_action_embeddings = _encode_candidate_actions(model, candidate_action_features)
        next_candidate_q = model.q_values(next_state_embeddings, candidate_action_embeddings)
        candidate_probs = masked_softmax(next_candidate_q, candidates.mask, temperature=temperature)
        if torch.any(~torch.isfinite(candidate_probs)):
            raise ValueError("candidate_probs must be finite.")

        expected_next_q = masked_expected_q_value(next_candidate_q, candidate_probs, candidates.mask)
        target = bellman_target(batch.rewards, batch.dones, expected_next_q, gamma=float(gamma))
        residual = ann_bellman_residual(
            current_q,
            batch.rewards,
            batch.dones,
            next_candidate_q,
            candidate_probs,
            candidates.mask,
            gamma=float(gamma),
        )

        density_logits = model.density_logits(current_state_embeddings, current_action_embeddings)
        density_weights = make_density_weights(
            density_logits,
            max_weight=density_max_weight,
            normalize=True,
        )
        bellman_loss = weighted_squared_bellman_loss(residual, density_weights)
        ess = effective_sample_size(density_weights)
        entropy = weight_entropy(density_weights)

        current_dual = model.dual_values(current_state_embeddings, current_action_embeddings)
        next_candidate_dual = model.dual_values(next_state_embeddings, candidate_action_embeddings)
        stat_moment = stationarity_moment(
            density_weights,
            current_dual,
            next_candidate_dual,
            candidate_probs,
            candidates.mask,
            gamma=float(gamma),
        )
        stat_loss = squared_stationarity_loss(stat_moment)

        exact_candidate_features = _gather_action_features(action_features, exact_search.ids)
        exact_candidate_embeddings = _encode_candidate_actions(model, exact_candidate_features)
        exact_candidate_q = model.q_values(next_state_embeddings, exact_candidate_embeddings)
        exact_mask = torch.ones_like(exact_search.ids, dtype=torch.bool)
        exact_probs = masked_softmax(exact_candidate_q, exact_mask, temperature=temperature)
        selected_topk_features = _gather_action_features(action_features, selected_search.ids)
        selected_topk_embeddings = _encode_candidate_actions(model, selected_topk_features)
        selected_topk_q = model.q_values(next_state_embeddings, selected_topk_embeddings)
        selected_topk_mask = torch.ones_like(selected_search.ids, dtype=torch.bool)
        exact_backend_audit = audit_candidate_sets(
            selected_search.ids,
            exact_search.ids,
            retrieved_mask=selected_topk_mask,
            exact_values=exact_candidate_q,
            retrieved_values=selected_topk_q,
            exact_value_mask=exact_mask,
            retrieved_value_mask=selected_topk_mask,
        )
        audit = audit_candidate_sets(
            candidates.ids,
            exact_search.ids,
            retrieved_mask=candidates.mask,
            exact_policy_probs=exact_probs,
            exact_values=exact_candidate_q,
            retrieved_values=next_candidate_q,
            exact_value_probs=exact_probs,
            retrieved_value_probs=candidate_probs,
            exact_value_mask=exact_mask,
            retrieved_value_mask=candidates.mask,
        )

        tensors_to_check = {
            "bellman_loss": bellman_loss,
            "stationarity_moment": stat_moment,
            "stationarity_loss": stat_loss,
            "ess": ess,
            "weight_entropy": entropy,
            "exact_backend_recall_mean": exact_backend_audit.recall_at_k.mean(),
            "exact_backend_overlap_fraction_mean": exact_backend_audit.overlap_fraction.mean(),
        }
        if exact_backend_audit.best_value_gap is not None:
            tensors_to_check["exact_backend_best_value_gap_mean"] = exact_backend_audit.best_value_gap.mean()
        for name, value in tensors_to_check.items():
            if torch.any(~torch.isfinite(value)):
                raise ValueError(f"{name} must be finite.")

        diagnostics = {
            "temperature": float(temperature),
            "gamma": float(gamma),
            "k": int(k),
            "exact_candidate_width": int(exact_search.ids.shape[1]),
            "selected_candidate_width": int(selected_search.ids.shape[1]),
            "candidate_width": int(candidates.ids.shape[1]),
            "used_exact_retrieval": index_backend == "exact",
            "index_backend": index_backend,
            "index_metric": index_metric,
            "index_nlist": index_nlist,
            "index_nprobe": index_nprobe,
            "selected_index_backend": index_backend,
            "selected_index_metric": index_metric,
            "audit_exact_backend": audit_exact_backend,
            "exact_backend_recall_mean": float(exact_backend_audit.recall_at_k.mean().item()),
            "exact_backend_overlap_fraction_mean": float(exact_backend_audit.overlap_fraction.mean().item()),
            "exact_backend_best_value_gap_mean": (
                None
                if exact_backend_audit.best_value_gap is None
                else float(exact_backend_audit.best_value_gap.mean().item())
            ),
        }
        return PipelineSmokeResult(
            batch_size=batch.batch_size,
            num_candidates=candidates.width,
            candidate_ids=candidates.ids,
            candidate_mask=candidates.mask,
            candidate_probs=candidate_probs,
            current_q=current_q,
            next_candidate_q=next_candidate_q,
            bellman_target=target,
            bellman_residual=residual,
            bellman_loss=bellman_loss,
            density_weights=density_weights,
            ess=ess,
            weight_entropy=entropy,
            stationarity_moment=stat_moment,
            stationarity_loss=stat_loss,
            recall_at_k=audit.recall_at_k,
            pmr_at_k=audit.pmr_at_k,
            best_value_gap=audit.best_value_gap,
            exact_backend_recall_at_k=exact_backend_audit.recall_at_k,
            exact_backend_overlap_fraction=exact_backend_audit.overlap_fraction,
            exact_backend_best_value_gap=exact_backend_audit.best_value_gap,
            diagnostics=diagnostics,
        )
