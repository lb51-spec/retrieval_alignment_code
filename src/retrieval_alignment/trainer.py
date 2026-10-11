"""Minimal differentiable train-step smoke utility.

This module implements exactly one optimization step on synthetic data to check
that the model, candidate policy, Bellman utilities, density weights, and
stationarity moment are differentiably wired. It is not a full benchmark trainer
and does not include checkpointing, required FAISS, or benchmark logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]

from .ann_index import ActionIndex, make_action_index
from .audits import audit_candidate_sets, compute_candidate_policy_mass_recall
from .candidate_sets import build_next_state_candidates
from .config import DWMQLObjectiveConfig, validate_dwmql_objective_config
from .density import effective_sample_size, make_density_weights, weight_entropy
from .losses import bellman_residual, bellman_target, masked_expected_q_value, weighted_squared_bellman_loss
from .metrics import (
    compute_support_correction_loss,
    normalized_policy_mass_recall,
    pmr_missing_mass,
    uniform_policy_mass_baseline,
)
from .networks import DWMQLNetworks
from .policies import masked_softmax
from .stationarity import squared_stationarity_loss, stationarity_moment
from .synthetic import SyntheticOfflineData
from .target_networks import clone_target_network, soft_update_target


@dataclass(slots=True)
class TrainStepResult:
    """Diagnostics from one minimal optimization step."""

    total_loss: "torch.Tensor | Any"
    bellman_loss: "torch.Tensor | Any"
    stationarity_loss: "torch.Tensor | Any"
    density_regularization: "torch.Tensor | Any"
    grad_norm: "torch.Tensor | Any"
    parameter_delta_norm: "torch.Tensor | Any"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectiveUpdateResult:
    """Diagnostics from one explicit synthetic objective update."""

    loss: "torch.Tensor | Any"
    objective_name: str
    grad_norm: "torch.Tensor | Any"
    parameter_delta_norm: "torch.Tensor | Any"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SyntheticTrainingBatchTerms:
    """Reusable tensors computed from one synthetic replay batch."""

    current_q: "torch.Tensor | Any"
    next_candidate_q: "torch.Tensor | Any"
    candidate_probs: "torch.Tensor | Any"
    bellman_target: "torch.Tensor | Any"
    bellman_residual: "torch.Tensor | Any"
    density_logits: "torch.Tensor | Any"
    density_weights: "torch.Tensor | Any"
    current_dual_f: "torch.Tensor | Any"
    next_candidate_dual_f: "torch.Tensor | Any"
    stationarity_moment: "torch.Tensor | Any"
    bellman_loss: "torch.Tensor | Any"
    support_weighted_bellman_loss: "torch.Tensor | Any"
    stationarity_loss: "torch.Tensor | Any"
    density_regularization: "torch.Tensor | Any"
    support_correction_loss: "torch.Tensor | Any"
    bellman_moment_loss: "torch.Tensor | Any"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DWMQLBatchTerms:
    """Structured tensors from one DW-MQL experiment batch.

    Produced by :func:`compute_dwmql_batch_terms`. Fields are live tensors;
    diagnostics are detached scalar floats.
    """

    current_q: "torch.Tensor | Any"
    next_candidate_q: "torch.Tensor | Any"
    candidate_probs: "torch.Tensor | Any"
    bellman_target_val: "torch.Tensor | Any"
    bellman_residual: "torch.Tensor | Any"
    density_logits: "torch.Tensor | Any"
    density_weights: "torch.Tensor | Any"
    ess: "torch.Tensor | Any"
    weight_entropy: "torch.Tensor | Any"
    current_dual_f: "torch.Tensor | Any"
    next_candidate_dual_f: "torch.Tensor | Any"
    stationarity_moment: "torch.Tensor | Any"
    bellman_loss: "torch.Tensor | Any"
    bellman_moment_loss: "torch.Tensor | Any"
    stationarity_loss: "torch.Tensor | Any"
    density_regularization: "torch.Tensor | Any"
    support_correction_loss: "torch.Tensor | Any"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DWMQLStepResult:
    """Diagnostics from one DW-MQL minimax experiment step."""

    bellman_loss: float
    stationarity_loss: float
    density_regularization: float
    ess: float
    critic_loss: float
    density_loss: float
    dual_loss: float
    extragradient_used: bool
    extragradient_groups: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainLoopSmokeResult:
    """Diagnostics from a tiny synthetic-only multi-step smoke loop.

    This records whether repeated differentiable steps and optional soft target
    updates are numerically wired. It is not an experiment runner, benchmark, or
    full training implementation; see ``docs/algorithm_pseudocode.md`` and
    ``docs/testing_plan.md`` for the intended staged algorithm path.
    """

    num_steps: int
    initial_total_loss: float
    final_total_loss: float
    losses: list[float]
    grad_norms: list[float]
    parameter_delta_norms: list[float]
    target_update_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingHistory:
    """History collected by the synthetic trainer abstraction."""

    losses: list[float]
    bellman_losses: list[float]
    stationarity_losses: list[float]
    density_regularizations: list[float]
    grad_norms: list[float]
    parameter_delta_norms: list[float]
    target_update_steps: list[int]
    index_ages: list[int]
    index_refresh_steps: list[int]
    index_refresh_count: int
    stale_index_recall_means: list[float | None]
    stale_index_overlap_fraction_means: list[float | None]
    stale_index_best_value_gap_means: list[float | None]
    stale_index_pmr_means: list[float | None]
    stale_index_pmr_mins: list[float | None]
    stale_index_pmr_maxs: list[float | None]
    stale_index_pmr_normalized_means: list[float | None]
    stale_index_missing_mass_means: list[float | None]
    critic_total_losses: list[float | None]
    support_correction_losses: list[float | None]
    support_correction_contributions: list[float | None]
    density_ess_values: list[float | None]
    weight_entropy_values: list[float | None]
    index_ema_update_count: int
    index_encoder_drift_mean_l2: list[float | None]
    index_encoder_drift_max_l2: list[float | None]
    index_encoder_drift_mean_cosine_distance: list[float | None]
    index_encoder_drift_max_cosine_distance: list[float | None]
    target_vs_online_recall_means: list[float | None]
    target_vs_online_overlap_fraction_means: list[float | None]
    ema_vs_online_recall_means: list[float | None]
    ema_vs_online_overlap_fraction_means: list[float | None]
    ema_vs_target_recall_means: list[float | None]
    ema_vs_target_overlap_fraction_means: list[float | None]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    support_weighted_bellman_losses: list[float | None] = field(default_factory=list)
    support_weight_means: list[float | None] = field(default_factory=list)
    support_weight_mins: list[float | None] = field(default_factory=list)
    support_weight_maxs: list[float | None] = field(default_factory=list)
    q_abs_max_values: list[float | None] = field(default_factory=list)


@dataclass(slots=True)
class SyntheticIndexCache:
    """Synthetic-only persistent action-index cache scaffold.

    The cached index is built from no-grad encoded action catalogue embeddings.
    It is intended for stale-index wiring diagnostics, not for benchmark-grade
    indexing or checkpointed index persistence.
    """

    index: ActionIndex
    index_backend: str
    index_metric: str
    age: int = 0
    refresh_count: int = 0
    last_refresh_step: int = 0


@dataclass(slots=True)
class OptimizerBundle:
    """Small optimizer grouping scaffold for smoke-only alternating updates.

    The bundle makes optimizer ownership explicit for shared encoders and the
    three scalar heads. It does not split losses, implement extragradient, or
    encode the final DW-MQL update schedule.
    """

    encoder_optimizer: "torch.optim.Optimizer | Any | None" = None
    critic_optimizer: "torch.optim.Optimizer | Any | None" = None
    density_optimizer: "torch.optim.Optimizer | Any | None" = None
    dual_optimizer: "torch.optim.Optimizer | Any | None" = None
    joint_optimizer: "torch.optim.Optimizer | Any | None" = None


@dataclass(slots=True)
class LossComponentConfig:
    """Controls active loss terms and smoke-step detach boundaries.

    The defaults are conservative for Q-learning-style smoke updates: density
    weights and candidate policy probabilities are detached before they enter
    the Bellman loss path. This is still a single-backward scaffold, not the
    final DW-MQL minimax optimization scheme.
    """

    use_bellman_loss: bool = True
    use_stationarity_loss: bool = True
    use_density_regularization: bool = True
    detach_density_weights_for_bellman: bool = True
    detach_policy_probs_for_bellman: bool = True


RETRIEVAL_CORRECTION_MODES: tuple[str, ...] = (
    "aware",
    "naive",
    "uniform_weights",
    "no_density",
)

SUPPORT_CORRECTION_METRICS: tuple[str, ...] = ("missing_mass", "normalized_pmr_margin")
SUPPORT_WEIGHTING_MODES: tuple[str, ...] = (
    "none",
    "missing_mass",
    "inverse_pmr",
    "normalized_pmr_margin",
)


def validate_retrieval_correction_mode(mode: str) -> str:
    """Validate retrieval-correction ablation mode names."""

    if mode not in RETRIEVAL_CORRECTION_MODES:
        raise ValueError(
            "retrieval_correction_mode must be one of "
            f"{', '.join(RETRIEVAL_CORRECTION_MODES)}."
        )
    return mode


def validate_support_correction_config(
    *,
    weight: float,
    target: float | None,
    metric: str,
) -> tuple[float, float | None, str]:
    """Validate PMR/support-loss correction settings."""

    if metric not in SUPPORT_CORRECTION_METRICS:
        raise ValueError(
            "support_correction_metric must be one of "
            f"{', '.join(SUPPORT_CORRECTION_METRICS)}."
        )
    if not isinstance(weight, int | float) or float(weight) < 0:
        raise ValueError("support_correction_weight must be nonnegative.")
    target_float = None
    if target is not None:
        target_float = float(target)
        if torch is not None:
            finite = bool(torch.isfinite(torch.tensor(target_float)).item())
        else:
            finite = target_float == target_float and target_float not in {float("inf"), float("-inf")}
        if not finite or target_float <= 0:
            raise ValueError("support_correction_target must be None or finite and positive.")
    if metric == "normalized_pmr_margin" and target_float is None:
        raise ValueError("support_correction_target is required for normalized_pmr_margin.")
    return float(weight), target_float, metric


def validate_support_weighting_config(
    *,
    mode: str,
    scale: float,
    min_weight: float,
    max_weight: float,
    eps: float,
) -> tuple[str, float, float, float, float]:
    """Validate support-aware Bellman weighting settings."""

    if mode not in SUPPORT_WEIGHTING_MODES:
        raise ValueError(
            "support_weighting_mode must be one of "
            f"{', '.join(SUPPORT_WEIGHTING_MODES)}."
        )
    if not isinstance(scale, int | float) or float(scale) < 0:
        raise ValueError("support_weight_scale must be nonnegative.")
    if not isinstance(min_weight, int | float) or float(min_weight) <= 0:
        raise ValueError("support_weight_min must be positive.")
    if not isinstance(max_weight, int | float) or float(max_weight) < float(min_weight):
        raise ValueError("support_weight_max must be greater than or equal to support_weight_min.")
    if not isinstance(eps, int | float) or float(eps) <= 0:
        raise ValueError("support_pmr_eps must be positive.")
    return mode, float(scale), float(min_weight), float(max_weight), float(eps)


def compute_support_weights(
    pmr: Any,
    normalized_pmr: Any | None = None,
    missing_mass: Any | None = None,
    mode: str = "none",
    scale: float = 1.0,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """Compute per-sample Bellman weights from PMR/support-loss diagnostics.

    The returned weights are intended for the Bellman residual reduction. They
    are diagnostics/prototype weights, not a finalized theorem-backed
    correction.
    """

    _require_torch()
    mode, scale, min_weight, max_weight, eps = validate_support_weighting_config(
        mode=mode,
        scale=scale,
        min_weight=min_weight,
        max_weight=max_weight,
        eps=eps,
    )
    pmr_tensor = torch.as_tensor(pmr)
    if not torch.is_floating_point(pmr_tensor):
        pmr_tensor = pmr_tensor.float()
    if pmr_tensor.numel() == 0:
        raise ValueError("pmr must be non-empty.")
    tolerance = 1e-6
    if torch.any(~torch.isfinite(pmr_tensor)) or torch.any((pmr_tensor < -tolerance) | (pmr_tensor > 1 + tolerance)):
        raise ValueError("pmr values must be finite and in [0, 1].")
    pmr_tensor = pmr_tensor.clamp(0.0, 1.0)

    if mode == "none":
        weights = torch.ones_like(pmr_tensor, dtype=pmr_tensor.dtype)
    elif mode == "missing_mass":
        if missing_mass is None:
            missing_tensor = 1.0 - pmr_tensor
        else:
            missing_tensor = torch.as_tensor(missing_mass, dtype=pmr_tensor.dtype, device=pmr_tensor.device)
        if missing_tensor.shape != pmr_tensor.shape:
            raise ValueError("missing_mass shape must match pmr shape.")
        if torch.any(~torch.isfinite(missing_tensor)) or torch.any((missing_tensor < -tolerance) | (missing_tensor > 1 + tolerance)):
            raise ValueError("missing_mass values must be finite and in [0, 1].")
        missing_tensor = missing_tensor.clamp(0.0, 1.0)
        weights = 1.0 + scale * missing_tensor
    elif mode == "inverse_pmr":
        weights = 1.0 / pmr_tensor.clamp_min(eps)
    else:
        if normalized_pmr is None:
            raise ValueError("normalized_pmr is required for normalized_pmr_margin support weighting.")
        normalized_tensor = torch.as_tensor(normalized_pmr, dtype=pmr_tensor.dtype, device=pmr_tensor.device)
        if normalized_tensor.shape != pmr_tensor.shape:
            raise ValueError("normalized_pmr shape must match pmr shape.")
        if torch.any(~torch.isfinite(normalized_tensor)) or torch.any(normalized_tensor < 0):
            raise ValueError("normalized_pmr values must be finite and nonnegative.")
        weights = 1.0 + scale * torch.relu(1.0 - normalized_tensor)

    weights = weights.clamp(min=min_weight, max=max_weight)
    if torch.any(~torch.isfinite(weights)):
        raise ValueError("support weights must be finite.")
    return weights


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run the minimal train step.")


def _json_serializable(value: Any) -> Any:
    """Convert common smoke diagnostic objects into JSON-safe values."""

    if torch is not None and isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return detached.item()
        return detached.tolist()
    if isinstance(value, dict):
        return {str(key): _json_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_serializable(asdict(value))
    return value


def train_step_result_to_dict(result: TrainStepResult) -> dict[str, Any]:
    """Convert a ``TrainStepResult`` into a JSON-serializable dictionary."""

    if not isinstance(result, TrainStepResult):
        raise TypeError("result must be a TrainStepResult.")
    return _json_serializable(asdict(result))


def training_history_to_dict(history: TrainingHistory) -> dict[str, Any]:
    """Convert ``TrainingHistory`` into a JSON-serializable dictionary."""

    if not isinstance(history, TrainingHistory):
        raise TypeError("history must be a TrainingHistory.")
    return _json_serializable(asdict(history))


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a UTF-8 JSON diagnostics report, creating parents as needed."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("JSON output path must be a file, not a directory.")
    if output_path.name == "":
        raise ValueError("JSON output path must include a file name.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_serializable(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def compute_action_embedding_drift(
    reference_embeddings: "torch.Tensor",
    comparison_embeddings: "torch.Tensor",
) -> dict[str, float]:
    """Compute synthetic-only action-encoder drift diagnostics.

    The returned values compare two action-catalogue embedding tensors with the
    same shape. They are retrieval-index diagnostics only and do not imply that
    lower or higher drift is preferable.
    """

    _require_torch()
    if reference_embeddings.shape != comparison_embeddings.shape:
        raise ValueError("reference_embeddings and comparison_embeddings must have the same shape.")
    if reference_embeddings.numel() == 0:
        raise ValueError("embedding tensors must be non-empty.")
    if reference_embeddings.ndim < 2:
        raise ValueError("embedding tensors must have at least two dimensions.")
    if torch.any(~torch.isfinite(reference_embeddings)) or torch.any(~torch.isfinite(comparison_embeddings)):
        raise ValueError("embedding tensors must contain only finite values.")

    difference = reference_embeddings - comparison_embeddings
    per_item_l2 = torch.linalg.vector_norm(difference.reshape(difference.shape[0], -1), dim=1)
    reference_flat = reference_embeddings.reshape(reference_embeddings.shape[0], -1)
    comparison_flat = comparison_embeddings.reshape(comparison_embeddings.shape[0], -1)
    reference_norm = torch.linalg.vector_norm(reference_flat, dim=1)
    comparison_norm = torch.linalg.vector_norm(comparison_flat, dim=1)
    cosine_similarity = torch.zeros_like(reference_norm)
    valid = (reference_norm > 0) & (comparison_norm > 0)
    cosine_similarity[valid] = (
        reference_flat[valid] * comparison_flat[valid]
    ).sum(dim=1) / (reference_norm[valid] * comparison_norm[valid])
    cosine_distance = 1.0 - cosine_similarity.clamp(-1.0, 1.0)
    if torch.any(~torch.isfinite(per_item_l2)) or torch.any(~torch.isfinite(cosine_distance)):
        raise ValueError("drift diagnostics must be finite.")
    return {
        "mean_l2": float(per_item_l2.mean().item()),
        "max_l2": float(per_item_l2.max().item()),
        "mean_cosine_distance": float(cosine_distance.mean().item()),
        "max_cosine_distance": float(cosine_distance.max().item()),
    }


def compute_encoder_policy_retrieval_overlap(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    query_states: "torch.Tensor",
    k: int,
    index_backend: str,
    index_metric: str,
    target_model: DWMQLNetworks | None = None,
    ema_index_model: DWMQLNetworks | None = None,
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> dict[str, float]:
    """Compare fresh top-k retrieval from online, target, and EMA encoders.

    The online encoder is the reference for target-vs-online and
    EMA-vs-online diagnostics. These values are synthetic-only retrieval
    overlap diagnostics and make no claim about policy or learning quality.
    """

    _require_torch()
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if query_states.numel() == 0:
        raise ValueError("query_states must be non-empty.")
    if torch.any(~torch.isfinite(query_states)):
        raise ValueError("query_states must contain only finite values.")

    def search_with_encoder(encoder_model: DWMQLNetworks) -> "torch.Tensor":
        with torch.no_grad():
            action_features = data.action_features.to(device=query_states.device)
            action_ids = data.action_ids.to(device=query_states.device)
            action_embeddings = encoder_model.encode_actions(action_features).detach()
            query_embeddings = encoder_model.encode_states(query_states).detach()
            index = make_action_index(
                action_embeddings,
                action_ids=action_ids,
                backend=index_backend,
                metric=index_metric,
                normalize=index_metric == "cosine",
                nlist=index_nlist,
                nprobe=index_nprobe,
            )
            return index.search(query_embeddings, k=k).ids

    online_ids = search_with_encoder(model)
    retrieved_by_policy: dict[str, "torch.Tensor"] = {"online": online_ids}
    if target_model is not None:
        retrieved_by_policy["target"] = search_with_encoder(target_model)
    if ema_index_model is not None:
        retrieved_by_policy["ema"] = search_with_encoder(ema_index_model)

    diagnostics: dict[str, float] = {}

    def add_pair(prefix: str, retrieved: "torch.Tensor", reference: "torch.Tensor") -> None:
        audit = audit_candidate_sets(retrieved, reference)
        diagnostics[f"{prefix}_recall_mean"] = float(audit.recall_at_k.mean().item())
        diagnostics[f"{prefix}_overlap_fraction_mean"] = float(audit.overlap_fraction.mean().item())

    if "target" in retrieved_by_policy:
        add_pair("target_vs_online", retrieved_by_policy["target"], retrieved_by_policy["online"])
    if "ema" in retrieved_by_policy:
        add_pair("ema_vs_online", retrieved_by_policy["ema"], retrieved_by_policy["online"])
    if "ema" in retrieved_by_policy and "target" in retrieved_by_policy:
        add_pair("ema_vs_target", retrieved_by_policy["ema"], retrieved_by_policy["target"])

    for name, value in diagnostics.items():
        if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    return diagnostics


def _validate_inputs(
    data: SyntheticOfflineData,
    batch_size: int,
    k: int,
    gamma: float,
    temperature: float,
    index_backend: str = "exact",
    index_metric: str = "dot",
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


def _pmr_support_diagnostics(
    pmr_values: Any,
    *,
    k: int,
    num_actions: int,
) -> dict[str, float]:
    normalized = normalized_policy_mass_recall(pmr_values, k, num_actions)
    missing = pmr_missing_mass(pmr_values)
    return {
        "stale_index_pmr_normalized_mean": float(normalized.mean().item()),
        "stale_index_missing_mass_mean": float(missing.mean().item()),
        "stale_index_uniform_pmr_baseline": uniform_policy_mass_baseline(k, num_actions),
    }


def _mean_optional(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _candidate_membership_pmr(
    *,
    model: DWMQLNetworks,
    next_states: "torch.Tensor",
    action_features: "torch.Tensor",
    retrieved_ids: "torch.Tensor",
    reference_ids: "torch.Tensor",
    temperature: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Differentiable PMR proxy over reference candidates.

    Candidate ids are discrete and therefore non-differentiable, but the
    reference softmax mass remains differentiable with respect to Q scores.
    """

    membership = (reference_ids.unsqueeze(-1) == retrieved_ids.unsqueeze(1)).any(dim=-1)
    reference_features = _gather_action_features(action_features, reference_ids)
    reference_state_embeddings = model.encode_states(next_states)
    reference_action_embeddings = _encode_candidate_actions(model, reference_features)
    reference_q = model.q_values(reference_state_embeddings, reference_action_embeddings)
    reference_mask = torch.ones_like(reference_ids, dtype=torch.bool)
    reference_probs = masked_softmax(reference_q, reference_mask, temperature=temperature)
    pmr = (reference_probs * membership.to(dtype=reference_probs.dtype)).sum(dim=1)
    return pmr, normalized_policy_mass_recall(
        pmr,
        k=int(retrieved_ids.shape[1]),
        num_actions=int(action_features.shape[0]),
    )


def _gather_action_features(action_features: "torch.Tensor", candidate_ids: "torch.Tensor") -> "torch.Tensor":
    return action_features[candidate_ids.clamp_min(0)]


def _encode_candidate_actions(model: DWMQLNetworks, candidate_action_features: "torch.Tensor") -> "torch.Tensor":
    batch_size, num_candidates, action_dim = candidate_action_features.shape
    flat = candidate_action_features.reshape(batch_size * num_candidates, action_dim)
    encoded = model.encode_actions(flat)
    return encoded.reshape(batch_size, num_candidates, -1)


def _parameter_vector_norm(parameters: list["torch.Tensor"]) -> "torch.Tensor":
    if not parameters:
        return torch.tensor(0.0)
    total = sum(param.detach().square().sum() for param in parameters)
    return torch.sqrt(total)


def _parameter_delta_norm(before: list["torch.Tensor"], after_params: list["torch.Tensor"]) -> "torch.Tensor":
    if not before:
        return torch.tensor(0.0)
    total = sum((after.detach() - before_param).square().sum() for before_param, after in zip(before, after_params, strict=True))
    return torch.sqrt(total)


def _model_group_parameters(model: DWMQLNetworks) -> dict[str, list["torch.Tensor"]]:
    return {
        "encoder": [
            *model.state_encoder.parameters(),
            *model.action_encoder.parameters(),
        ],
        "critic": list(model.q_head.parameters()),
        "density": list(model.density_head.parameters()),
        "dual": list(model.dual_head.parameters()),
    }


def _norm_from_tensors(tensors: list["torch.Tensor"], *, name: str) -> float:
    if not tensors:
        return 0.0
    total = None
    for tensor in tensors:
        detached = tensor.detach()
        if torch.any(~torch.isfinite(detached)):
            raise ValueError(f"{name} must contain only finite values.")
        squared_sum = detached.square().sum()
        total = squared_sum if total is None else total + squared_sum
    norm = torch.sqrt(total)
    if torch.any(~torch.isfinite(norm)):
        raise ValueError(f"{name} norm must be finite.")
    return float(norm.item())


def _snapshot_parameter_groups(model: DWMQLNetworks) -> dict[str, list["torch.Tensor"]]:
    return {
        group: [parameter.detach().clone() for parameter in parameters]
        for group, parameters in _model_group_parameters(model).items()
    }


def gradient_group_norms(model: DWMQLNetworks) -> dict[str, float]:
    """Return per-group gradient L2 norms for encoders and scalar heads."""

    norms: dict[str, float] = {}
    for group, parameters in _model_group_parameters(model).items():
        grads = []
        for parameter in parameters:
            if parameter.grad is None:
                grads.append(torch.zeros_like(parameter.detach()))
            else:
                grads.append(parameter.grad.detach())
        norms[group] = _norm_from_tensors(grads, name=f"{group} gradients")
    return norms


def parameter_group_norms(model: DWMQLNetworks) -> dict[str, float]:
    """Return per-group parameter L2 norms for encoders and scalar heads."""

    return {
        group: _norm_from_tensors(parameters, name=f"{group} parameters")
        for group, parameters in _model_group_parameters(model).items()
    }


def parameter_delta_group_norms(
    before_state: dict[str, list["torch.Tensor"]],
    model: DWMQLNetworks,
) -> dict[str, float]:
    """Return per-group parameter delta norms against a pre-step snapshot."""

    deltas: dict[str, float] = {}
    current_groups = _model_group_parameters(model)
    expected_groups = {"encoder", "critic", "density", "dual"}
    if set(before_state) != expected_groups:
        raise ValueError("before_state must contain encoder, critic, density, and dual groups.")
    for group, before_parameters in before_state.items():
        current_parameters = current_groups[group]
        if len(before_parameters) != len(current_parameters):
            raise ValueError(f"before_state group {group!r} does not match current parameter count.")
        delta_tensors = []
        for before, current in zip(before_parameters, current_parameters, strict=True):
            if before.shape != current.shape:
                raise ValueError(f"before_state group {group!r} has mismatched parameter shapes.")
            delta_tensors.append(current.detach() - before.to(device=current.device))
        deltas[group] = _norm_from_tensors(delta_tensors, name=f"{group} parameter deltas")
    return deltas


def expected_groups_for_update_parts(update_parts: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    """Map optimizer update parts to model groups expected to receive gradients."""

    if not isinstance(update_parts, list | tuple | set):
        raise TypeError("update_parts must be a list, tuple, or set.")
    mapping = {
        "encoder": "encoder",
        "critic": "critic",
        "density": "density",
        "dual": "dual",
        "joint": None,
    }
    expected: set[str] = set()
    unknown = set(update_parts) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown update part(s): {sorted(unknown)}.")
    if "joint" in update_parts:
        expected.update({"encoder", "critic", "density", "dual"})
    for part in update_parts:
        group = mapping[part]
        if group is not None:
            expected.add(group)
    return expected


def assert_expected_gradient_flow(
    gradient_group_norms: dict[str, float],
    expected_positive_groups: list[str] | tuple[str, ...] | set[str],
    min_norm: float = 1e-12,
) -> bool:
    """Assert expected model groups have nonzero smoke gradients."""

    if not isinstance(min_norm, int | float) or float(min_norm) < 0:
        raise ValueError("min_norm must be a nonnegative finite scalar.")
    min_norm_float = float(min_norm)
    if not torch.isfinite(torch.tensor(min_norm_float)):
        raise ValueError("min_norm must be a nonnegative finite scalar.")
    valid_groups = {"encoder", "critic", "density", "dual"}
    expected_groups = set(expected_positive_groups)
    unknown_expected = expected_groups - valid_groups
    if unknown_expected:
        raise ValueError(f"Unknown expected group(s): {sorted(unknown_expected)}.")
    for group, norm in gradient_group_norms.items():
        if group not in valid_groups:
            raise ValueError(f"Unknown gradient group: {group!r}.")
        norm_float = float(norm)
        if norm_float < 0 or not torch.isfinite(torch.tensor(norm_float)):
            raise ValueError(f"Gradient norm for group {group!r} must be finite and nonnegative.")
    for group in expected_groups:
        if group not in gradient_group_norms:
            raise AssertionError(f"Expected gradient group {group!r} is missing.")
        if float(gradient_group_norms[group]) <= min_norm_float:
            raise AssertionError(
                f"Expected gradient group {group!r} to have norm > {min_norm_float}, "
                f"got {gradient_group_norms[group]}."
            )
    return True


def _grad_norm(parameters: list["torch.Tensor"]) -> "torch.Tensor":
    grads = [param.grad.detach() for param in parameters if param.grad is not None]
    if not grads:
        return torch.tensor(0.0, device=parameters[0].device if parameters else None)
    total = sum(grad.square().sum() for grad in grads)
    norm = torch.sqrt(total)
    if torch.any(~torch.isfinite(norm)):
        raise ValueError("Gradient norm must be finite.")
    return norm


def _validate_loss_config(loss_config: LossComponentConfig | None) -> LossComponentConfig:
    config = loss_config or LossComponentConfig()
    if not isinstance(config, LossComponentConfig):
        raise TypeError("loss_config must be a LossComponentConfig instance.")
    for field_name in (
        "use_bellman_loss",
        "use_stationarity_loss",
        "use_density_regularization",
        "detach_density_weights_for_bellman",
        "detach_policy_probs_for_bellman",
    ):
        if not isinstance(getattr(config, field_name), bool):
            raise TypeError(f"loss_config.{field_name} must be a bool.")
    if not (
        config.use_bellman_loss
        or config.use_stationarity_loss
        or config.use_density_regularization
    ):
        raise ValueError("At least one loss component must be active.")
    return config


def _loss_config_diagnostics(config: LossComponentConfig) -> dict[str, bool]:
    return {
        "use_bellman_loss": config.use_bellman_loss,
        "use_stationarity_loss": config.use_stationarity_loss,
        "use_density_regularization": config.use_density_regularization,
        "detach_density_weights_for_bellman": config.detach_density_weights_for_bellman,
        "detach_policy_probs_for_bellman": config.detach_policy_probs_for_bellman,
    }


def critic_loss_config() -> LossComponentConfig:
    """Return a Bellman-only smoke loss config for critic updates."""

    return LossComponentConfig(
        use_bellman_loss=True,
        use_stationarity_loss=False,
        use_density_regularization=False,
        detach_density_weights_for_bellman=True,
        detach_policy_probs_for_bellman=True,
    )


def density_loss_config() -> LossComponentConfig:
    """Return a stationarity-plus-density-regularization smoke config."""

    return LossComponentConfig(
        use_bellman_loss=False,
        use_stationarity_loss=True,
        use_density_regularization=True,
    )


def dual_loss_config() -> LossComponentConfig:
    """Return a stationarity-only smoke loss config for dual updates."""

    return LossComponentConfig(
        use_bellman_loss=False,
        use_stationarity_loss=True,
        use_density_regularization=False,
    )


def _validate_update_parts(parts: Any) -> list[str]:
    if not isinstance(parts, list | tuple):
        raise TypeError("schedule step parts must be a non-empty list or tuple.")
    update_parts = list(parts)
    if not update_parts:
        raise ValueError("schedule step parts must be non-empty.")
    unknown = set(update_parts) - {"encoder", "critic", "density", "dual", "joint"}
    if unknown:
        raise ValueError(f"Unknown update part(s): {sorted(unknown)}.")
    return update_parts


def parse_update_schedule_step(step: Any) -> tuple[list[str], LossComponentConfig | None]:
    """Parse an alternating-loop schedule step.

    A step may be the legacy list/tuple of optimizer parts, or a dict with
    ``"parts"`` and optional ``"loss_config"``. The returned loss config is
    ``None`` for legacy steps, which preserves the default train-step behavior.
    """

    if isinstance(step, dict):
        if "parts" not in step:
            raise ValueError("Structured schedule steps must include 'parts'.")
        update_parts = _validate_update_parts(step["parts"])
        loss_config = step.get("loss_config")
        if loss_config is not None and not isinstance(loss_config, LossComponentConfig):
            raise TypeError("Structured schedule step loss_config must be a LossComponentConfig.")
        return update_parts, loss_config
    return _validate_update_parts(step), None


def default_smoke_alternating_schedule(include_encoder_on_critic: bool = True) -> list[dict[str, Any]]:
    """Return the named default alternating schedule for synthetic smoke tests.

    The order is dual, density, then critic. This is inspired by alternating
    saddle-style updates, but it is only a smoke-test schedule for checking that
    optimizer groups and loss-component controls are wired. It is not the final
    DW-MQL training algorithm.
    """

    if not isinstance(include_encoder_on_critic, bool):
        raise TypeError("include_encoder_on_critic must be a bool.")
    critic_parts = ["encoder", "critic"] if include_encoder_on_critic else ["critic"]
    return [
        {"parts": ["dual"], "loss_config": dual_loss_config()},
        {"parts": ["density"], "loss_config": density_loss_config()},
        {"parts": critic_parts, "loss_config": critic_loss_config()},
    ]


def make_optimizer_bundle(
    model: DWMQLNetworks,
    lr: float = 3e-4,
    encoder_lr: float | None = None,
    weight_decay: float = 0.0,
    mode: str = "joint",
) -> OptimizerBundle:
    """Create optimizers for joint or head-separated smoke updates.

    Supported modes are ``"joint"`` and ``"separate_heads"``. The separated
    mode provides one optimizer for the shared encoders and one optimizer per
    head, which is enough to smoke-test alternating schedules without changing
    the current single-loss backward pass.
    """

    _require_torch()
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if encoder_lr is not None and encoder_lr <= 0:
        raise ValueError("encoder_lr must be positive when provided.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative.")

    if mode == "joint":
        return OptimizerBundle(
            joint_optimizer=torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=lr,
                weight_decay=weight_decay,
            )
        )
    if mode == "separate_heads":
        shared_encoder_params = [
            *model.state_encoder.parameters(),
            *model.action_encoder.parameters(),
        ]
        # B2: retrieval projection heads (when present) train alongside the
        # encoder group. They receive gradients only from the listwise KL loss;
        # the detach in encode_*_query/index keeps them off the Q/Bellman path.
        if getattr(model, "retrieval_action_proj", None) is not None:
            shared_encoder_params.extend(model.retrieval_state_proj.parameters())
            shared_encoder_params.extend(model.retrieval_action_proj.parameters())
        return OptimizerBundle(
            encoder_optimizer=torch.optim.AdamW(
                [parameter for parameter in shared_encoder_params if parameter.requires_grad],
                lr=encoder_lr if encoder_lr is not None else lr,
                weight_decay=weight_decay,
            ),
            critic_optimizer=torch.optim.AdamW(
                [parameter for parameter in model.q_head.parameters() if parameter.requires_grad],
                lr=lr,
                weight_decay=weight_decay,
            ),
            density_optimizer=torch.optim.AdamW(
                [parameter for parameter in model.density_head.parameters() if parameter.requires_grad],
                lr=lr,
                weight_decay=weight_decay,
            ),
            dual_optimizer=torch.optim.AdamW(
                [parameter for parameter in model.dual_head.parameters() if parameter.requires_grad],
                lr=lr,
                weight_decay=weight_decay,
            ),
        )
    raise ValueError("mode must be 'joint' or 'separate_heads'.")


def _bundle_optimizers(bundle: OptimizerBundle) -> dict[str, "torch.optim.Optimizer"]:
    optimizers = {
        "encoder": bundle.encoder_optimizer,
        "critic": bundle.critic_optimizer,
        "density": bundle.density_optimizer,
        "dual": bundle.dual_optimizer,
        "joint": bundle.joint_optimizer,
    }
    return {name: optimizer for name, optimizer in optimizers.items() if optimizer is not None}


def zero_grad_bundle(bundle: OptimizerBundle) -> None:
    """Zero gradients for every optimizer present in ``bundle``."""

    for optimizer in _bundle_optimizers(bundle).values():
        optimizer.zero_grad(set_to_none=True)


def step_bundle(bundle: OptimizerBundle, parts: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Step selected optimizers in ``bundle`` and return stepped part names.

    ``parts=None`` steps all optimizers present. For a joint bundle, explicit
    part selection only steps the joint optimizer when ``"joint"`` is requested;
    head-level selection is intended for ``mode="separate_heads"`` bundles.
    """

    optimizers = _bundle_optimizers(bundle)
    if parts is None:
        selected_parts = list(optimizers)
    else:
        selected_parts = list(parts)
        unknown = set(selected_parts) - {"encoder", "critic", "density", "dual", "joint"}
        if unknown:
            raise ValueError(f"Unknown update part(s): {sorted(unknown)}.")

    stepped: list[str] = []
    for part in selected_parts:
        optimizer = optimizers.get(part)
        if optimizer is not None:
            optimizer.step()
            stepped.append(part)
    return stepped


def compute_synthetic_training_batch_terms(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    batch_size: int,
    k: int,
    gamma: float,
    generator: "torch.Generator | None" = None,
    temperature: float = 1.0,
    density_max_weight: float | None = None,
    target_model: DWMQLNetworks | None = None,
    detach_bellman_target: bool = True,
    detach_policy_probs_for_bellman: bool = True,
    detach_density_weights_for_bellman: bool = True,
    detach_density_weights_for_stationarity: bool = False,
    detach_dual_for_stationarity: bool = False,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
    action_index: ActionIndex | None = None,
    audit_stale_index: bool = False,
    retrieval_correction_mode: str = "aware",
    support_correction_weight: float = 0.0,
    support_correction_target: float | None = None,
    support_correction_metric: str = "missing_mass",
    support_weighting_mode: str = "none",
    support_weight_scale: float = 1.0,
    support_weight_min: float = 0.1,
    support_weight_max: float = 10.0,
    support_pmr_eps: float = 1e-6,
    reward_scale: float = 1.0,
    q_value_clip: float | None = None,
) -> SyntheticTrainingBatchTerms:
    """Compute reusable synthetic DW-MQL+ANN objective terms for one batch.

    This is a synthetic-only training primitive. Retrieval itself is
    non-differentiable, while selected raw action features are re-encoded through
    the online model so objective losses remain differentiable where intended.
    """

    _require_torch()
    retrieval_correction_mode = validate_retrieval_correction_mode(retrieval_correction_mode)
    support_correction_weight, support_correction_target, support_correction_metric = validate_support_correction_config(
        weight=support_correction_weight,
        target=support_correction_target,
        metric=support_correction_metric,
    )
    support_weighting_mode, support_weight_scale, support_weight_min, support_weight_max, support_pmr_eps = (
        validate_support_weighting_config(
            mode=support_weighting_mode,
            scale=support_weight_scale,
            min_weight=support_weight_min,
            max_weight=support_weight_max,
            eps=support_pmr_eps,
        )
    )
    _validate_inputs(data, batch_size, k, gamma, temperature, index_backend, index_metric, index_nlist, index_nprobe)
    if target_model is not None:
        if not isinstance(target_model, DWMQLNetworks):
            raise TypeError("target_model must be a DWMQLNetworks instance.")
        if any(parameter.requires_grad for parameter in target_model.parameters()):
            raise ValueError("target_model parameters must be frozen before use.")
        target_model.eval()
    if action_index is not None:
        if not hasattr(action_index, "search") or not hasattr(action_index, "embedding_dim"):
            raise TypeError("action_index must follow the ActionIndex interface.")
        if int(action_index.embedding_dim) != int(model.config.embedding_dim):
            raise ValueError("action_index embedding_dim must match model embedding_dim.")

    batch = data.dataset.sample(batch_size, generator=generator)
    # Move the mini-batch to the model's device (GPU when device="cuda"). The full
    # replay buffer stays on CPU to avoid GPU OOM on large datasets; only the sampled
    # batch (batch_size rows) is moved per step.
    compute_device = next(model.parameters()).device
    if batch.states.device != compute_device:
        batch = batch.to(compute_device)
    action_features = data.action_features.to(device=compute_device)
    action_ids = data.action_ids.to(device=compute_device)

    with torch.no_grad():
        # Retrieval-aware critic: the ANN truncation that defines the Bellman
        # target's candidate support uses the retrieval embeddings (B2 projection
        # when present, else the base encoders). The Q values of the selected
        # actions below still use the base encoders. encode_*_index/query fall
        # back to base encoders when no projection exists, so non-B2 runs are
        # byte-identical.
        catalogue_embeddings = model.encode_action_index(action_features).detach()
        query_embeddings = model.encode_state_query(batch.next_states).detach()
        index = action_index
        if index is None:
            index = make_action_index(
                catalogue_embeddings,
                action_ids=action_ids,
                backend=index_backend,
                metric=index_metric,
                normalize=index_metric == "cosine",
                nlist=index_nlist,
                nprobe=index_nprobe,
            )
        search = index.search(query_embeddings, k=k)
        fresh_search = None
        if audit_stale_index and action_index is not None:
            fresh_index = make_action_index(
                catalogue_embeddings,
                action_ids=action_ids,
                backend=index_backend,
                metric=index_metric,
                normalize=index_metric == "cosine",
                nlist=index_nlist,
                nprobe=index_nprobe,
            )
            fresh_search = fresh_index.search(query_embeddings, k=k)

    candidates = build_next_state_candidates(
        search.ids,
        logged_action_indices=batch.next_logged_action_ids,
    )
    if torch.any(~candidates.mask.any(dim=1)):
        raise ValueError("Each candidate row must contain at least one valid candidate.")

    current_state_embeddings = model.encode_states(batch.states)
    current_action_embeddings = model.encode_actions(batch.actions)
    current_q = model.q_values(current_state_embeddings, current_action_embeddings)

    next_state_embeddings = model.encode_states(batch.next_states)
    stale_index_recall_mean = None
    stale_index_overlap_fraction_mean = None
    stale_index_best_value_gap_mean = None
    stale_index_pmr_mean = None
    stale_index_pmr_min = None
    stale_index_pmr_max = None
    stale_index_pmr_normalized_mean = None
    stale_index_missing_mass_mean = None
    stale_index_uniform_pmr_baseline = uniform_policy_mass_baseline(k, int(data.action_features.shape[0]))
    if fresh_search is not None:
        with torch.no_grad():
            fresh_next_state_embeddings = model.encode_states(batch.next_states)
            cached_topk_features = _gather_action_features(action_features, search.ids)
            fresh_topk_features = _gather_action_features(action_features, fresh_search.ids)
            cached_topk_embeddings = _encode_candidate_actions(model, cached_topk_features)
            fresh_topk_embeddings = _encode_candidate_actions(model, fresh_topk_features)
            cached_topk_q = model.q_values(fresh_next_state_embeddings, cached_topk_embeddings)
            fresh_topk_q = model.q_values(fresh_next_state_embeddings, fresh_topk_embeddings)
            topk_mask = torch.ones_like(search.ids, dtype=torch.bool)
            stale_audit = audit_candidate_sets(
                search.ids,
                fresh_search.ids,
                retrieved_mask=topk_mask,
                exact_values=fresh_topk_q,
                retrieved_values=cached_topk_q,
                exact_value_mask=topk_mask,
                retrieved_value_mask=topk_mask,
            )
            stale_index_recall_mean = float(stale_audit.recall_at_k.mean().item())
            stale_index_overlap_fraction_mean = float(stale_audit.overlap_fraction.mean().item())
            stale_index_best_value_gap_mean = (
                None
                if stale_audit.best_value_gap is None
                else float(stale_audit.best_value_gap.mean().item())
            )
            stale_pmr = compute_candidate_policy_mass_recall(
                search.ids,
                fresh_search.ids,
                reference_scores=fresh_topk_q,
                retrieved_mask=topk_mask,
                reference_mask=topk_mask,
                temperature=temperature,
            )
            stale_index_pmr_mean = float(stale_pmr.mean().item())
            stale_index_pmr_min = float(stale_pmr.min().item())
            stale_index_pmr_max = float(stale_pmr.max().item())
            pmr_diagnostics = _pmr_support_diagnostics(
                stale_pmr,
                k=k,
                num_actions=int(action_features.shape[0]),
            )
            stale_index_pmr_normalized_mean = pmr_diagnostics["stale_index_pmr_normalized_mean"]
            stale_index_missing_mass_mean = pmr_diagnostics["stale_index_missing_mass_mean"]
            stale_index_uniform_pmr_baseline = pmr_diagnostics["stale_index_uniform_pmr_baseline"]

    candidate_action_features = _gather_action_features(action_features, candidates.ids)
    candidate_action_embeddings = _encode_candidate_actions(model, candidate_action_features)
    next_candidate_q = model.q_values(next_state_embeddings, candidate_action_embeddings)
    candidate_probs = masked_softmax(next_candidate_q, candidates.mask, temperature=temperature)

    if target_model is None:
        bellman_candidate_probs = candidate_probs.detach() if detach_policy_probs_for_bellman else candidate_probs
        expected_next_q = masked_expected_q_value(next_candidate_q, bellman_candidate_probs, candidates.mask)
    else:
        with torch.no_grad():
            target_next_state_embeddings = target_model.encode_states(batch.next_states)
            target_candidate_action_embeddings = _encode_candidate_actions(target_model, candidate_action_features)
            target_next_candidate_q = target_model.q_values(
                target_next_state_embeddings,
                target_candidate_action_embeddings,
            )
            target_candidate_probs = masked_softmax(
                target_next_candidate_q,
                candidates.mask,
                temperature=temperature,
            )
            expected_next_q = masked_expected_q_value(
                target_next_candidate_q,
                target_candidate_probs,
                candidates.mask,
            )
    bellman_expected_next_q = expected_next_q.detach() if (detach_bellman_target or target_model is not None) else expected_next_q
    if q_value_clip is not None:
        bellman_expected_next_q = bellman_expected_next_q.clamp(-float(q_value_clip), float(q_value_clip))
    scaled_rewards = batch.rewards * float(reward_scale) if reward_scale != 1.0 else batch.rewards
    target = bellman_target(scaled_rewards, batch.dones, bellman_expected_next_q, gamma=float(gamma))
    residual = bellman_residual(target, current_q)

    density_logits = model.density_logits(current_state_embeddings, current_action_embeddings)
    density_weights = make_density_weights(density_logits, max_weight=density_max_weight, normalize=True)
    density_ess = effective_sample_size(density_weights.detach())
    density_entropy = weight_entropy(density_weights.detach())
    objective_density_weights = (
        torch.ones_like(density_weights)
        if retrieval_correction_mode == "no_density"
        else density_weights
    )
    if retrieval_correction_mode in {"uniform_weights", "no_density"}:
        bellman_weights = torch.ones_like(density_weights)
    else:
        bellman_weights = (
            objective_density_weights.detach()
            if detach_density_weights_for_bellman
            else objective_density_weights
        )
    bellman_loss = weighted_squared_bellman_loss(residual, bellman_weights)
    bellman_moment = (bellman_weights.detach() * residual).mean()
    bellman_moment_loss = bellman_moment.square()
    density_regularization = (
        density_weights.sum() * 0.0
        if retrieval_correction_mode == "no_density"
        else (density_weights - 1.0).square().mean()
    )

    support_correction_available = fresh_search is not None
    proxy_pmr = None
    proxy_normalized_pmr = None
    support_correction_active = (
        retrieval_correction_mode == "aware"
        and support_correction_weight > 0.0
        and support_correction_available
    )
    support_weighting_requested = retrieval_correction_mode == "aware" and support_weighting_mode != "none"
    support_weighting_active = support_weighting_requested and support_correction_available
    support_correction_loss = bellman_loss.detach().new_zeros(())
    support_weights = torch.ones_like(bellman_weights)
    support_correction_differentiable_proxy = False
    if support_correction_available:
        proxy_pmr, proxy_normalized_pmr = _candidate_membership_pmr(
            model=model,
            next_states=batch.next_states,
            action_features=action_features,
            retrieved_ids=search.ids,
            reference_ids=fresh_search.ids,
            temperature=temperature,
        )
        proxy_loss = compute_support_correction_loss(
            proxy_pmr,
            proxy_normalized_pmr,
            metric=support_correction_metric,
            target=support_correction_target,
        )
        support_correction_loss = proxy_loss if support_correction_active else proxy_loss.detach()
        support_correction_differentiable_proxy = support_correction_active
        effective_support_weighting_mode = support_weighting_mode if support_weighting_active else "none"
        support_weights = compute_support_weights(
            proxy_pmr,
            normalized_pmr=proxy_normalized_pmr,
            missing_mass=1.0 - proxy_pmr,
            mode=effective_support_weighting_mode,
            scale=support_weight_scale,
            min_weight=support_weight_min,
            max_weight=support_weight_max,
            eps=support_pmr_eps,
        )
    else:
        effective_support_weighting_mode = "none"
    support_weighted_bellman_loss = weighted_squared_bellman_loss(
        residual,
        bellman_weights * support_weights.detach(),
    )
    support_correction_contribution = (
        support_correction_weight * support_correction_loss
        if support_correction_active
        else support_correction_loss.detach().new_zeros(())
    )
    critic_total_loss = support_weighted_bellman_loss + support_correction_contribution

    current_dual = model.dual_values(current_state_embeddings, current_action_embeddings)
    next_candidate_dual = model.dual_values(next_state_embeddings, candidate_action_embeddings)
    stationarity_density_weights = (
        objective_density_weights.detach() if detach_density_weights_for_stationarity else objective_density_weights
    )
    stationarity_current_dual = current_dual.detach() if detach_dual_for_stationarity else current_dual
    stationarity_next_candidate_dual = (
        next_candidate_dual.detach() if detach_dual_for_stationarity else next_candidate_dual
    )
    stat_moment = stationarity_moment(
        stationarity_density_weights,
        stationarity_current_dual,
        stationarity_next_candidate_dual,
        candidate_probs,
        candidates.mask,
        gamma=float(gamma),
    )
    stat_loss = squared_stationarity_loss(stat_moment)

    return SyntheticTrainingBatchTerms(
        current_q=current_q,
        next_candidate_q=next_candidate_q,
        candidate_probs=candidate_probs,
        bellman_target=target,
        bellman_residual=residual,
        density_logits=density_logits,
        density_weights=density_weights,
        current_dual_f=current_dual,
        next_candidate_dual_f=next_candidate_dual,
        stationarity_moment=stat_moment,
        bellman_loss=bellman_loss,
        support_weighted_bellman_loss=support_weighted_bellman_loss,
        stationarity_loss=stat_loss,
        density_regularization=density_regularization,
        support_correction_loss=support_correction_loss,
        bellman_moment_loss=bellman_moment_loss,
        diagnostics={
            "batch_size": int(batch_size),
            "k": int(k),
            "gamma": float(gamma),
            "num_candidates": int(candidates.width),
            "bellman_target_requires_grad": bool(target.requires_grad),
            "density_weights_detached_for_bellman": detach_density_weights_for_bellman,
            "policy_probs_detached_for_bellman": detach_policy_probs_for_bellman or target_model is not None,
            "target_model_used": target_model is not None,
            "bellman_moment": float(bellman_moment.detach().item()),
            "bellman_moment_loss": float(bellman_moment_loss.detach().item()),
            "detach_density_weights_for_stationarity": detach_density_weights_for_stationarity,
            "detach_dual_for_stationarity": detach_dual_for_stationarity,
            "stale_index_recall_mean": stale_index_recall_mean,
            "stale_index_overlap_fraction_mean": stale_index_overlap_fraction_mean,
            "stale_index_best_value_gap_mean": stale_index_best_value_gap_mean,
            "stale_index_pmr_mean": stale_index_pmr_mean,
            "stale_index_pmr_min": stale_index_pmr_min,
            "stale_index_pmr_max": stale_index_pmr_max,
            "stale_index_pmr_normalized_mean": stale_index_pmr_normalized_mean,
            "stale_index_missing_mass_mean": stale_index_missing_mass_mean,
            "stale_index_uniform_pmr_baseline": stale_index_uniform_pmr_baseline,
            "density_ess": float(density_ess.item()),
            "weight_entropy": float(density_entropy.item()),
            "retrieval_correction_mode": retrieval_correction_mode,
            "pmr_support_correction_used": support_correction_active,
            "raw_bellman_loss": float(bellman_loss.detach().item()),
            "support_weighted_bellman_loss": float(support_weighted_bellman_loss.detach().item()),
            "support_weighting_mode": effective_support_weighting_mode,
            "support_weighting_requested_mode": support_weighting_mode,
            "support_weighting_active": support_weighting_active,
            "support_weight_mean": float(support_weights.detach().mean().item()),
            "support_weight_min": float(support_weights.detach().min().item()),
            "support_weight_max": float(support_weights.detach().max().item()),
            "support_weight_scale": float(support_weight_scale),
            "support_correction_loss": float(support_correction_loss.detach().item()),
            "support_correction_weight": float(support_correction_weight),
            "support_correction_contribution": float(support_correction_contribution.detach().item()),
            "critic_total_loss": float(critic_total_loss.detach().item()),
            "support_correction_metric": support_correction_metric,
            "support_correction_target": support_correction_target,
            "support_correction_active": support_correction_active,
            "support_correction_available": support_correction_available,
            "support_correction_differentiable_proxy": support_correction_differentiable_proxy,
            "bellman_weights_forced_uniform": retrieval_correction_mode in {"uniform_weights", "no_density"},
            "density_objective_enabled": retrieval_correction_mode != "no_density",
        },
    )


def compute_dwmql_batch_terms(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    obj_config: DWMQLObjectiveConfig,
    batch_size: int,
    k: int,
    *,
    generator: "torch.Generator | None" = None,
    temperature: float = 1.0,
    target_model: DWMQLNetworks | None = None,
    action_index: ActionIndex | None = None,
    detach_density_weights_for_stationarity: bool = False,
    detach_dual_for_stationarity: bool = False,
) -> DWMQLBatchTerms:
    """Compute experiment-ready DW-MQL objective terms for one batch.

    Delegates retrieval, Bellman target, density weights, and stationarity
    computation to :func:`compute_synthetic_training_batch_terms` with
    parameters drawn from ``obj_config``. Returns a :class:`DWMQLBatchTerms`
    with ESS and weight entropy as live tensors (detached from the graph).
    """

    validate_dwmql_objective_config(obj_config)
    raw = compute_synthetic_training_batch_terms(
        data=data,
        model=model,
        batch_size=batch_size,
        k=k,
        gamma=obj_config.gamma,
        generator=generator,
        temperature=temperature,
        density_max_weight=obj_config.density_cap,
        target_model=target_model,
        detach_bellman_target=obj_config.detach_bellman_target,
        detach_density_weights_for_bellman=obj_config.detach_density_weights_for_critic,
        detach_policy_probs_for_bellman=True,
        detach_density_weights_for_stationarity=detach_density_weights_for_stationarity,
        detach_dual_for_stationarity=detach_dual_for_stationarity,
        action_index=action_index,
        retrieval_correction_mode="aware",
        support_correction_weight=obj_config.support_correction_weight,
    )

    ess_tensor = effective_sample_size(raw.density_weights.detach())
    entropy_tensor = weight_entropy(raw.density_weights.detach())

    return DWMQLBatchTerms(
        current_q=raw.current_q,
        next_candidate_q=raw.next_candidate_q,
        candidate_probs=raw.candidate_probs,
        bellman_target_val=raw.bellman_target,
        bellman_residual=raw.bellman_residual,
        density_logits=raw.density_logits,
        density_weights=raw.density_weights,
        ess=ess_tensor,
        weight_entropy=entropy_tensor,
        current_dual_f=raw.current_dual_f,
        next_candidate_dual_f=raw.next_candidate_dual_f,
        stationarity_moment=raw.stationarity_moment,
        bellman_loss=raw.bellman_loss,
        bellman_moment_loss=raw.bellman_moment_loss,
        stationarity_loss=raw.stationarity_loss,
        density_regularization=raw.density_regularization,
        support_correction_loss=raw.support_correction_loss,
        diagnostics={
            **raw.diagnostics,
            "objective_config_gamma": float(obj_config.gamma),
            "objective_config_bellman_weight": float(obj_config.bellman_weight),
            "objective_config_stationarity_weight": float(obj_config.stationarity_weight),
            "objective_config_density_regularization_weight": float(obj_config.density_regularization_weight),
            "objective_config_support_correction_weight": float(obj_config.support_correction_weight),
            "objective_config_extragradient": bool(obj_config.extragradient),
            "objective_config_dual_ascent": bool(obj_config.dual_ascent),
            "ess": float(ess_tensor.item()),
            "weight_entropy": float(entropy_tensor.item()),
            "detach_density_weights_for_stationarity": detach_density_weights_for_stationarity,
            "detach_dual_for_stationarity": detach_dual_for_stationarity,
        },
    )


def run_minimal_train_step(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    optimizer: "torch.optim.Optimizer | None",
    batch_size: int,
    k: int,
    gamma: float,
    generator: "torch.Generator | None" = None,
    temperature: float = 1.0,
    density_max_weight: float | None = None,
    stationarity_weight: float = 1.0,
    density_reg_weight: float = 1e-4,
    detach_bellman_target: bool = True,
    target_model: DWMQLNetworks | None = None,
    optimizer_bundle: OptimizerBundle | None = None,
    update_parts: list[str] | tuple[str, ...] | None = None,
    loss_config: LossComponentConfig | None = None,
    split_backward: bool = False,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
    action_index: ActionIndex | None = None,
    audit_stale_index: bool = False,
    index_encoder_policy: str = "online",
    index_ema_tau: float | None = None,
    retrieval_correction_mode: str = "aware",
    support_correction_weight: float = 0.0,
    support_correction_target: float | None = None,
    support_correction_metric: str = "missing_mass",
    support_weighting_mode: str = "none",
    support_weight_scale: float = 1.0,
    support_weight_min: float = 0.1,
    support_weight_max: float = 10.0,
    support_pmr_eps: float = 1e-6,
) -> TrainStepResult:
    """Run one differentiable optimization step on synthetic smoke data.

    By default, the Bellman bootstrap target is detached. If ``target_model`` is
    provided, it is used only for the Bellman bootstrap under ``torch.no_grad()``
    and the Bellman target is detached regardless of ``detach_bellman_target``.
    The online model still computes current Q, density weights, and stationarity
    terms. This is still a one-step smoke trainer, not a full target-network
    training loop or refresh schedule.
    """

    _require_torch()
    retrieval_correction_mode = validate_retrieval_correction_mode(retrieval_correction_mode)
    support_correction_weight, support_correction_target, support_correction_metric = validate_support_correction_config(
        weight=support_correction_weight,
        target=support_correction_target,
        metric=support_correction_metric,
    )
    support_weighting_mode, support_weight_scale, support_weight_min, support_weight_max, support_pmr_eps = (
        validate_support_weighting_config(
            mode=support_weighting_mode,
            scale=support_weight_scale,
            min_weight=support_weight_min,
            max_weight=support_weight_max,
            eps=support_pmr_eps,
        )
    )
    if not isinstance(detach_bellman_target, bool):
        raise TypeError("detach_bellman_target must be a bool.")
    if not isinstance(split_backward, bool):
        raise TypeError("split_backward must be a bool.")
    if not isinstance(audit_stale_index, bool):
        raise TypeError("audit_stale_index must be a bool.")
    if index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of 'online', 'target', or 'ema'.")
    if index_ema_tau is not None:
        if not isinstance(index_ema_tau, int | float) or not 0.0 <= float(index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    loss_config = _validate_loss_config(loss_config)
    _validate_inputs(data, batch_size, k, gamma, temperature, index_backend, index_metric, index_nlist, index_nprobe)
    if stationarity_weight < 0:
        raise ValueError("stationarity_weight must be nonnegative.")
    if density_reg_weight < 0:
        raise ValueError("density_reg_weight must be nonnegative.")
    if optimizer is not None and optimizer_bundle is not None:
        raise ValueError("Provide either optimizer or optimizer_bundle, not both.")
    if optimizer is None and optimizer_bundle is None:
        raise ValueError("An optimizer or optimizer_bundle is required.")
    if split_backward and optimizer_bundle is None:
        raise ValueError("split_backward=True requires optimizer_bundle.")
    if action_index is not None:
        if not hasattr(action_index, "search") or not hasattr(action_index, "embedding_dim"):
            raise TypeError("action_index must follow the ActionIndex interface.")
        expected_embedding_dim = int(model.config.embedding_dim)
        if int(action_index.embedding_dim) != expected_embedding_dim:
            raise ValueError(
                f"action_index embedding_dim {action_index.embedding_dim} does not match model embedding_dim {expected_embedding_dim}."
            )
    target_model_used = target_model is not None
    if target_model is not None:
        if not isinstance(target_model, DWMQLNetworks):
            raise TypeError("target_model must be a DWMQLNetworks instance.")
        if any(parameter.requires_grad for parameter in target_model.parameters()):
            raise ValueError("target_model parameters must be frozen before use.")
        target_model.eval()

    model.train()
    batch = data.dataset.sample(batch_size, generator=generator)
    compute_device = next(model.parameters()).device
    if batch.states.device != compute_device:
        batch = batch.to(compute_device)
    action_features = data.action_features.to(device=compute_device)
    action_ids = data.action_ids.to(device=compute_device)

    catalogue_embeddings = model.encode_actions(action_features).detach()
    with torch.no_grad():
        query_embeddings = model.encode_states(batch.next_states).detach()
        if action_index is None:
            index = make_action_index(
                catalogue_embeddings,
                action_ids=action_ids,
                backend=index_backend,
                metric=index_metric,
                normalize=index_metric == "cosine",
                nlist=index_nlist,
                nprobe=index_nprobe,
            )
        else:
            index = action_index
        exact_search = index.search(query_embeddings, k=k)
        fresh_search = None
        if audit_stale_index and action_index is not None:
            fresh_index = make_action_index(
                catalogue_embeddings,
                action_ids=action_ids,
                backend=index_backend,
                metric=index_metric,
                normalize=index_metric == "cosine",
                nlist=index_nlist,
                nprobe=index_nprobe,
            )
            fresh_search = fresh_index.search(query_embeddings, k=k)

    candidates = build_next_state_candidates(
        exact_search.ids,
        logged_action_indices=batch.next_logged_action_ids,
    )
    if torch.any(~candidates.mask.any(dim=1)):
        raise ValueError("Each candidate row must contain at least one valid candidate.")

    current_state_embeddings = model.encode_states(batch.states)
    current_action_embeddings = model.encode_actions(batch.actions)
    current_q = model.q_values(current_state_embeddings, current_action_embeddings)

    next_state_embeddings = model.encode_states(batch.next_states)
    stale_index_recall_mean = None
    stale_index_overlap_fraction_mean = None
    stale_index_best_value_gap_mean = None
    stale_index_pmr_mean = None
    stale_index_pmr_min = None
    stale_index_pmr_max = None
    stale_index_pmr_normalized_mean = None
    stale_index_missing_mass_mean = None
    stale_index_uniform_pmr_baseline = uniform_policy_mass_baseline(k, int(data.action_features.shape[0]))
    if fresh_search is not None:
        with torch.no_grad():
            fresh_next_state_embeddings = model.encode_states(batch.next_states)
            cached_topk_features = _gather_action_features(action_features, exact_search.ids)
            fresh_topk_features = _gather_action_features(action_features, fresh_search.ids)
            cached_topk_embeddings = _encode_candidate_actions(model, cached_topk_features)
            fresh_topk_embeddings = _encode_candidate_actions(model, fresh_topk_features)
            cached_topk_q = model.q_values(fresh_next_state_embeddings, cached_topk_embeddings)
            fresh_topk_q = model.q_values(fresh_next_state_embeddings, fresh_topk_embeddings)
            topk_mask = torch.ones_like(exact_search.ids, dtype=torch.bool)
            stale_audit = audit_candidate_sets(
                exact_search.ids,
                fresh_search.ids,
                retrieved_mask=topk_mask,
                exact_values=fresh_topk_q,
                retrieved_values=cached_topk_q,
                exact_value_mask=topk_mask,
                retrieved_value_mask=topk_mask,
            )
            stale_index_recall_mean = float(stale_audit.recall_at_k.mean().item())
            stale_index_overlap_fraction_mean = float(stale_audit.overlap_fraction.mean().item())
            stale_index_best_value_gap_mean = (
                None
                if stale_audit.best_value_gap is None
                else float(stale_audit.best_value_gap.mean().item())
            )
            stale_pmr = compute_candidate_policy_mass_recall(
                exact_search.ids,
                fresh_search.ids,
                reference_scores=fresh_topk_q,
                retrieved_mask=topk_mask,
                reference_mask=topk_mask,
                temperature=temperature,
            )
            stale_index_pmr_mean = float(stale_pmr.mean().item())
            stale_index_pmr_min = float(stale_pmr.min().item())
            stale_index_pmr_max = float(stale_pmr.max().item())
            pmr_diagnostics = _pmr_support_diagnostics(
                stale_pmr,
                k=k,
                num_actions=int(action_features.shape[0]),
            )
            stale_index_pmr_normalized_mean = pmr_diagnostics["stale_index_pmr_normalized_mean"]
            stale_index_missing_mass_mean = pmr_diagnostics["stale_index_missing_mass_mean"]
            stale_index_uniform_pmr_baseline = pmr_diagnostics["stale_index_uniform_pmr_baseline"]
    candidate_action_features = _gather_action_features(action_features, candidates.ids)
    candidate_action_embeddings = _encode_candidate_actions(model, candidate_action_features)
    next_candidate_q = model.q_values(next_state_embeddings, candidate_action_embeddings)
    candidate_probs = masked_softmax(next_candidate_q, candidates.mask, temperature=temperature)
    if target_model is None:
        bellman_candidate_probs = (
            candidate_probs.detach()
            if loss_config.detach_policy_probs_for_bellman
            else candidate_probs
        )
        expected_next_q = masked_expected_q_value(next_candidate_q, bellman_candidate_probs, candidates.mask)
    else:
        with torch.no_grad():
            target_next_state_embeddings = target_model.encode_states(batch.next_states)
            target_candidate_action_embeddings = _encode_candidate_actions(
                target_model,
                candidate_action_features,
            )
            target_next_candidate_q = target_model.q_values(
                target_next_state_embeddings,
                target_candidate_action_embeddings,
            )
            target_candidate_probs = masked_softmax(
                target_next_candidate_q,
                candidates.mask,
                temperature=temperature,
            )
            expected_next_q = masked_expected_q_value(
                target_next_candidate_q,
                target_candidate_probs,
                candidates.mask,
            )
    expected_next_q_requires_grad = bool(expected_next_q.requires_grad)
    bellman_expected_next_q = (
        expected_next_q.detach()
        if (detach_bellman_target or target_model is not None)
        else expected_next_q
    )
    target = bellman_target(batch.rewards, batch.dones, bellman_expected_next_q, gamma=float(gamma))
    residual = bellman_residual(target, current_q)

    density_logits = model.density_logits(current_state_embeddings, current_action_embeddings)
    density_weights = make_density_weights(
        density_logits,
        max_weight=density_max_weight,
        normalize=True,
    )
    density_ess = effective_sample_size(density_weights.detach())
    density_entropy = weight_entropy(density_weights.detach())
    objective_density_weights = (
        torch.ones_like(density_weights)
        if retrieval_correction_mode == "no_density"
        else density_weights
    )
    if retrieval_correction_mode in {"uniform_weights", "no_density"}:
        bellman_weights = torch.ones_like(density_weights)
    else:
        bellman_weights = (
            objective_density_weights.detach()
            if loss_config.detach_density_weights_for_bellman
            else objective_density_weights
        )
    bellman_loss = weighted_squared_bellman_loss(residual, bellman_weights)
    density_regularization = (
        density_weights.sum() * 0.0
        if retrieval_correction_mode == "no_density"
        else (density_weights - 1.0).square().mean()
    )
    support_correction_available = fresh_search is not None
    proxy_pmr = None
    proxy_normalized_pmr = None
    support_correction_active = (
        retrieval_correction_mode == "aware"
        and support_correction_weight > 0.0
        and support_correction_available
    )
    support_weighting_requested = retrieval_correction_mode == "aware" and support_weighting_mode != "none"
    support_weighting_active = support_weighting_requested and support_correction_available
    support_correction_loss = bellman_loss.detach().new_zeros(())
    support_weights = torch.ones_like(bellman_weights)
    support_correction_differentiable_proxy = False
    if support_correction_available:
        proxy_pmr, proxy_normalized_pmr = _candidate_membership_pmr(
            model=model,
            next_states=batch.next_states,
            action_features=action_features,
            retrieved_ids=exact_search.ids,
            reference_ids=fresh_search.ids,
            temperature=temperature,
        )
        proxy_loss = compute_support_correction_loss(
            proxy_pmr,
            proxy_normalized_pmr,
            metric=support_correction_metric,
            target=support_correction_target,
        )
        support_correction_loss = proxy_loss if support_correction_active else proxy_loss.detach()
        support_correction_differentiable_proxy = support_correction_active
        effective_support_weighting_mode = support_weighting_mode if support_weighting_active else "none"
        support_weights = compute_support_weights(
            proxy_pmr,
            normalized_pmr=proxy_normalized_pmr,
            missing_mass=1.0 - proxy_pmr,
            mode=effective_support_weighting_mode,
            scale=support_weight_scale,
            min_weight=support_weight_min,
            max_weight=support_weight_max,
            eps=support_pmr_eps,
        )
    else:
        effective_support_weighting_mode = "none"
    support_weighted_bellman_loss = weighted_squared_bellman_loss(
        residual,
        bellman_weights * support_weights.detach(),
    )
    support_correction_contribution = (
        support_correction_weight * support_correction_loss
        if support_correction_active
        else support_correction_loss.detach().new_zeros(())
    )
    critic_total_loss = support_weighted_bellman_loss + support_correction_contribution

    current_dual = model.dual_values(current_state_embeddings, current_action_embeddings)
    next_candidate_dual = model.dual_values(next_state_embeddings, candidate_action_embeddings)
    stat_moment = stationarity_moment(
        objective_density_weights,
        current_dual,
        next_candidate_dual,
        candidate_probs,
        candidates.mask,
        gamma=float(gamma),
    )
    stat_loss = squared_stationarity_loss(stat_moment)
    total_loss = bellman_loss.new_zeros(())
    active_losses: list[str] = []
    if loss_config.use_bellman_loss:
        total_loss = total_loss + critic_total_loss
        active_losses.append("bellman")
        if support_correction_active:
            active_losses.append("support_correction")
    if loss_config.use_stationarity_loss:
        total_loss = total_loss + stationarity_weight * stat_loss
        active_losses.append("stationarity")
    if loss_config.use_density_regularization:
        total_loss = total_loss + density_reg_weight * density_regularization
        active_losses.append("density_regularization")
    if torch.any(~torch.isfinite(total_loss)):
        raise ValueError("total_loss must be finite.")

    parameters = [param for param in model.parameters() if param.requires_grad]
    before = [param.detach().clone() for param in parameters]
    before_groups = _snapshot_parameter_groups(model)
    backward_blocks_run: list[str] = []
    if split_backward:
        zero_grad_bundle(optimizer_bundle)
        split_losses: list[tuple[str, "torch.Tensor"]] = []
        if loss_config.use_bellman_loss:
            split_losses.append(
                (
                    "bellman",
                    critic_total_loss,
                )
            )
        if loss_config.use_stationarity_loss:
            split_losses.append(("stationarity", stationarity_weight * stat_loss))
        if loss_config.use_density_regularization:
            split_losses.append(("density_regularization", density_reg_weight * density_regularization))
        for index, (name, loss_value) in enumerate(split_losses):
            loss_value.backward(retain_graph=index < len(split_losses) - 1)
            backward_blocks_run.append(name)
    elif optimizer_bundle is None:
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
    else:
        zero_grad_bundle(optimizer_bundle)
        total_loss.backward()
    grad_norm = _grad_norm(parameters)
    group_grad_norms = gradient_group_norms(model)
    if optimizer_bundle is None:
        optimizer.step()
        stepped_parts = ["optimizer"]
    else:
        stepped_parts = step_bundle(optimizer_bundle, update_parts)
    delta_norm = _parameter_delta_norm(before, parameters)
    group_delta_norms = parameter_delta_group_norms(before_groups, model)

    if not torch.isfinite(grad_norm) or not torch.isfinite(delta_norm):
        raise ValueError("Gradient and parameter delta norms must be finite.")

    return TrainStepResult(
        total_loss=total_loss.detach(),
        bellman_loss=bellman_loss.detach(),
        stationarity_loss=stat_loss.detach(),
        density_regularization=density_regularization.detach(),
        grad_norm=grad_norm.detach(),
        parameter_delta_norm=delta_norm.detach(),
        diagnostics={
            "batch_size": int(batch_size),
            "k": int(k),
            "gamma": float(gamma),
            "temperature": float(temperature),
            "num_candidates": int(candidates.width),
            "parameter_norm_before": _parameter_vector_norm(before).item(),
            "detach_bellman_target": detach_bellman_target,
            "target_model_used": target_model_used,
            "expected_next_q_requires_grad_before_detach": expected_next_q_requires_grad,
            "bellman_target_requires_grad": bool(target.requires_grad),
            "update_parts": None if update_parts is None else list(update_parts),
            "stepped_parts": stepped_parts,
            "loss_config": _loss_config_diagnostics(loss_config),
            "active_losses": active_losses,
            "split_backward": split_backward,
            "index_backend": index_backend,
            "index_metric": index_metric,
            "index_nlist": index_nlist,
            "index_nprobe": index_nprobe,
            "index_age": None,
            "index_refresh_every": None,
            "index_refresh_count": None,
            "index_refreshed_this_step": None,
            "persistent_index_used": action_index is not None,
            "audit_stale_index": audit_stale_index and action_index is not None,
            "index_encoder_policy": index_encoder_policy,
            "index_ema_tau": None if index_ema_tau is None else float(index_ema_tau),
            "index_ema_update_count": None,
            "stale_index_recall_mean": stale_index_recall_mean,
            "stale_index_overlap_fraction_mean": stale_index_overlap_fraction_mean,
            "stale_index_best_value_gap_mean": stale_index_best_value_gap_mean,
            "stale_index_pmr_mean": stale_index_pmr_mean,
            "stale_index_pmr_min": stale_index_pmr_min,
            "stale_index_pmr_max": stale_index_pmr_max,
            "stale_index_pmr_normalized_mean": stale_index_pmr_normalized_mean,
            "stale_index_missing_mass_mean": stale_index_missing_mass_mean,
            "stale_index_uniform_pmr_baseline": stale_index_uniform_pmr_baseline,
            "density_ess": float(density_ess.item()),
            "weight_entropy": float(density_entropy.item()),
            "retrieval_correction_mode": retrieval_correction_mode,
            "pmr_support_correction_used": support_correction_active,
            "raw_bellman_loss": float(bellman_loss.detach().item()),
            "support_weighted_bellman_loss": float(support_weighted_bellman_loss.detach().item()),
            "support_weighting_mode": effective_support_weighting_mode,
            "support_weighting_requested_mode": support_weighting_mode,
            "support_weighting_active": support_weighting_active,
            "support_weight_mean": float(support_weights.detach().mean().item()),
            "support_weight_min": float(support_weights.detach().min().item()),
            "support_weight_max": float(support_weights.detach().max().item()),
            "support_weight_scale": float(support_weight_scale),
            "support_correction_loss": float(support_correction_loss.detach().item()),
            "support_correction_weight": float(support_correction_weight),
            "support_correction_contribution": float(support_correction_contribution.detach().item()),
            "critic_total_loss": float(critic_total_loss.detach().item()),
            "support_correction_metric": support_correction_metric,
            "support_correction_target": support_correction_target,
            "support_correction_active": support_correction_active,
            "support_correction_available": support_correction_available,
            "support_correction_differentiable_proxy": support_correction_differentiable_proxy,
            "bellman_weights_forced_uniform": retrieval_correction_mode in {"uniform_weights", "no_density"},
            "density_objective_enabled": retrieval_correction_mode != "no_density",
            "backward_blocks_run": backward_blocks_run if split_backward else ["joint"],
            "bellman_weights_detached": loss_config.detach_density_weights_for_bellman,
            "bellman_policy_probs_detached": loss_config.detach_policy_probs_for_bellman or target_model is not None,
            "gradient_group_norms": group_grad_norms,
            "parameter_delta_group_norms": group_delta_norms,
            "parameter_group_norms": parameter_group_norms(model),
        },
    )


def run_minimal_train_loop(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    optimizer: "torch.optim.Optimizer",
    num_steps: int,
    batch_size: int,
    k: int,
    gamma: float,
    generator: "torch.Generator | None" = None,
    target_model: DWMQLNetworks | None = None,
    target_tau: float | None = None,
    target_update_every: int = 1,
    temperature: float = 1.0,
    density_max_weight: float | None = None,
    stationarity_weight: float = 1.0,
    density_reg_weight: float = 1e-4,
    detach_bellman_target: bool = True,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> TrainLoopSmokeResult:
    """Run a tiny synthetic-only smoke training loop.

    The loop repeatedly invokes :func:`run_minimal_train_step` and optionally
    soft-updates a provided frozen target model after optimization steps. This
    is only a differentiability and wiring sanity check; it deliberately omits
    FAISS, checkpointing, YAML config loading, benchmark evaluation, and
    alternating update schedules.
    """

    _require_torch()
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    if not isinstance(target_update_every, int) or target_update_every <= 0:
        raise ValueError("target_update_every must be a positive integer.")
    if target_tau is not None:
        if target_model is None:
            raise ValueError("target_tau requires target_model.")
        if not isinstance(target_tau, int | float) or not 0.0 <= float(target_tau) <= 1.0:
            raise ValueError("target_tau must be in [0, 1].")

    losses: list[float] = []
    grad_norms: list[float] = []
    parameter_delta_norms: list[float] = []
    gradient_group_norms_by_step: list[dict[str, float]] = []
    parameter_delta_group_norms_by_step: list[dict[str, float]] = []
    target_update_count = 0

    for step_index in range(num_steps):
        step_result = run_minimal_train_step(
            data=data,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            k=k,
            gamma=gamma,
            generator=generator,
            temperature=temperature,
            density_max_weight=density_max_weight,
            stationarity_weight=stationarity_weight,
            density_reg_weight=density_reg_weight,
            detach_bellman_target=detach_bellman_target,
            target_model=target_model,
            index_backend=index_backend,
            index_metric=index_metric,
            index_nlist=index_nlist,
            index_nprobe=index_nprobe,
        )

        total_loss = float(step_result.total_loss.item())
        grad_norm = float(step_result.grad_norm.item())
        parameter_delta_norm = float(step_result.parameter_delta_norm.item())
        scalar_tensor = torch.tensor([total_loss, grad_norm, parameter_delta_norm])
        if torch.any(~torch.isfinite(scalar_tensor)):
            raise ValueError("Collected loop diagnostics must be finite.")

        losses.append(total_loss)
        grad_norms.append(grad_norm)
        parameter_delta_norms.append(parameter_delta_norm)
        gradient_group_norms_by_step.append(step_result.diagnostics["gradient_group_norms"])
        parameter_delta_group_norms_by_step.append(step_result.diagnostics["parameter_delta_group_norms"])

        if target_model is not None and target_tau is not None and (step_index + 1) % target_update_every == 0:
            soft_update_target(target_model, model, float(target_tau))
            target_update_count += 1

    return TrainLoopSmokeResult(
        num_steps=num_steps,
        initial_total_loss=losses[0],
        final_total_loss=losses[-1],
        losses=losses,
        grad_norms=grad_norms,
        parameter_delta_norms=parameter_delta_norms,
        target_update_count=target_update_count,
        diagnostics={
            "batch_size": int(batch_size),
            "k": int(k),
            "gamma": float(gamma),
            "temperature": float(temperature),
            "target_model_used": target_model is not None,
            "target_tau": None if target_tau is None else float(target_tau),
            "target_update_every": int(target_update_every),
            "detach_bellman_target": detach_bellman_target,
            "index_backend": index_backend,
            "index_metric": index_metric,
            "index_nlist": index_nlist,
            "index_nprobe": index_nprobe,
            "gradient_group_norms_by_step": gradient_group_norms_by_step,
            "parameter_delta_group_norms_by_step": parameter_delta_group_norms_by_step,
        },
    )


def run_minimal_alternating_loop(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    optimizer_bundle: OptimizerBundle,
    num_steps: int,
    batch_size: int,
    k: int,
    gamma: float,
    update_schedule: list[Any] | tuple[Any, ...],
    generator: "torch.Generator | None" = None,
    target_model: DWMQLNetworks | None = None,
    target_tau: float | None = None,
    target_update_every: int = 1,
    temperature: float = 1.0,
    density_max_weight: float | None = None,
    stationarity_weight: float = 1.0,
    density_reg_weight: float = 1e-4,
    detach_bellman_target: bool = True,
    assert_gradient_flow: bool = False,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> TrainLoopSmokeResult:
    """Run a tiny alternating-update smoke loop with an optimizer bundle.

    The schedule selects which optimizers step after the single joint backward
    pass at each iteration. This is scaffolding for future staged updates, not
    the final minimax training algorithm.
    """

    _require_torch()
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    if not update_schedule:
        raise ValueError("update_schedule must contain at least one update part list.")
    if not isinstance(target_update_every, int) or target_update_every <= 0:
        raise ValueError("target_update_every must be a positive integer.")
    if target_tau is not None:
        if target_model is None:
            raise ValueError("target_tau requires target_model.")
        if not isinstance(target_tau, int | float) or not 0.0 <= float(target_tau) <= 1.0:
            raise ValueError("target_tau must be in [0, 1].")
    if not isinstance(assert_gradient_flow, bool):
        raise TypeError("assert_gradient_flow must be a bool.")

    losses: list[float] = []
    grad_norms: list[float] = []
    parameter_delta_norms: list[float] = []
    gradient_group_norms_by_step: list[dict[str, float]] = []
    parameter_delta_group_norms_by_step: list[dict[str, float]] = []
    target_update_count = 0
    applied_schedule: list[list[str]] = []
    active_losses_by_step: list[list[str]] = []

    for step_index in range(num_steps):
        update_parts, step_loss_config = parse_update_schedule_step(
            update_schedule[step_index % len(update_schedule)]
        )
        step_result = run_minimal_train_step(
            data=data,
            model=model,
            optimizer=None,
            optimizer_bundle=optimizer_bundle,
            update_parts=update_parts,
            batch_size=batch_size,
            k=k,
            gamma=gamma,
            generator=generator,
            temperature=temperature,
            density_max_weight=density_max_weight,
            stationarity_weight=stationarity_weight,
            density_reg_weight=density_reg_weight,
            detach_bellman_target=detach_bellman_target,
            target_model=target_model,
            loss_config=step_loss_config,
            index_backend=index_backend,
            index_metric=index_metric,
            index_nlist=index_nlist,
            index_nprobe=index_nprobe,
        )

        total_loss = float(step_result.total_loss.item())
        grad_norm = float(step_result.grad_norm.item())
        parameter_delta_norm = float(step_result.parameter_delta_norm.item())
        scalar_tensor = torch.tensor([total_loss, grad_norm, parameter_delta_norm])
        if torch.any(~torch.isfinite(scalar_tensor)):
            raise ValueError("Collected loop diagnostics must be finite.")

        losses.append(total_loss)
        grad_norms.append(grad_norm)
        parameter_delta_norms.append(parameter_delta_norm)
        applied_schedule.append(update_parts)
        active_losses_by_step.append(list(step_result.diagnostics["active_losses"]))
        gradient_group_norms_by_step.append(step_result.diagnostics["gradient_group_norms"])
        parameter_delta_group_norms_by_step.append(step_result.diagnostics["parameter_delta_group_norms"])
        if assert_gradient_flow:
            expected_groups = expected_groups_for_update_parts(update_parts)
            assert_expected_gradient_flow(
                step_result.diagnostics["gradient_group_norms"],
                expected_groups,
            )

        if target_model is not None and target_tau is not None and (step_index + 1) % target_update_every == 0:
            soft_update_target(target_model, model, float(target_tau))
            target_update_count += 1

    return TrainLoopSmokeResult(
        num_steps=num_steps,
        initial_total_loss=losses[0],
        final_total_loss=losses[-1],
        losses=losses,
        grad_norms=grad_norms,
        parameter_delta_norms=parameter_delta_norms,
        target_update_count=target_update_count,
        diagnostics={
            "batch_size": int(batch_size),
            "k": int(k),
            "gamma": float(gamma),
            "temperature": float(temperature),
            "target_model_used": target_model is not None,
            "target_tau": None if target_tau is None else float(target_tau),
            "target_update_every": int(target_update_every),
            "detach_bellman_target": detach_bellman_target,
            "update_schedule": applied_schedule,
            "active_losses_by_step": active_losses_by_step,
            "gradient_group_norms_by_step": gradient_group_norms_by_step,
            "parameter_delta_group_norms_by_step": parameter_delta_group_norms_by_step,
            "assert_gradient_flow": assert_gradient_flow,
            "index_backend": index_backend,
            "index_metric": index_metric,
            "index_nlist": index_nlist,
            "index_nprobe": index_nprobe,
        },
    )


class SyntheticDWMQLTrainer:
    """Stateful synthetic-only trainer with minimax gradient structure.

    Implements alternating critic/density/dual updates with correct gradient
    isolation: the dual update detaches density weights from the stationarity
    computation; the density update detaches dual f-values; the critic update
    adds the Bellman moment term ``(E[w δ])^2`` alongside the stabilizing
    weighted TD loss. This matches the joint saddle objective in
    ``docs/equations.md`` and the pseudocode in ``docs/algorithm_pseudocode.md``.

    This is a synthetic-only scaffold. It does not include FAISS, checkpointing,
    benchmarks, or the full extragradient update rule.
    """

    def __init__(
        self,
        data: SyntheticOfflineData,
        model: DWMQLNetworks,
        optimizer: "torch.optim.Optimizer | None" = None,
        optimizer_bundle: OptimizerBundle | None = None,
        batch_size: int = 8,
        k: int = 5,
        gamma: float = 0.99,
        generator: "torch.Generator | None" = None,
        target_model: DWMQLNetworks | None = None,
        target_tau: float | None = None,
        target_update_every: int = 1,
        temperature: float = 1.0,
        density_max_weight: float | None = None,
        stationarity_weight: float = 1.0,
        density_reg_weight: float = 1e-4,
        detach_bellman_target: bool = True,
        loss_config: LossComponentConfig | None = None,
        split_backward: bool = False,
        index_backend: str = "exact",
        index_metric: str = "dot",
        index_nlist: int | None = None,
        index_nprobe: int | None = None,
        index_refresh_every: int = 1,
        use_persistent_index: bool = False,
        audit_stale_index: bool = False,
        audit_encoder_policy_overlap: bool = False,
        index_encoder_policy: str = "online",
        index_ema_tau: float | None = None,
        retrieval_correction_mode: str = "aware",
        support_correction_weight: float = 0.0,
        support_correction_target: float | None = None,
        support_correction_metric: str = "missing_mass",
        support_weighting_mode: str = "none",
        support_weight_scale: float = 1.0,
        support_weight_min: float = 0.1,
        support_weight_max: float = 10.0,
        support_pmr_eps: float = 1e-6,
        bellman_moment_weight: float = 1.0,
        max_grad_norm: float | None = None,
        q_l2_regularization_weight: float = 0.0,
        q_value_clip: float | None = None,
        reward_scale: float = 1.0,
        ranking_margin_weight: float = 0.0,
        ranking_margin: float = 1.0,
        ranking_negative_samples: int = 20,
        retrieval_alignment_weight: float = 0.0,
        retrieval_alignment_margin: float = 1.0,
        retrieval_alignment_warmup_steps: int = 0,
        retrieval_alignment_loss_type: str = "listwise_kl",
        listwise_teacher_temperature: float = 0.2,
        listwise_retrieval_temperature: float = 0.07,
        cql_weight: float = 0.0,
        cql_num_samples: int = 64,
        bcq_weight: float = 0.0,
        wolpertinger_bc_weight: float = 0.0,
        device: str = "cpu",
    ) -> None:
        _require_torch()
        retrieval_correction_mode = validate_retrieval_correction_mode(retrieval_correction_mode)
        support_correction_weight, support_correction_target, support_correction_metric = validate_support_correction_config(
            weight=support_correction_weight,
            target=support_correction_target,
            metric=support_correction_metric,
        )
        support_weighting_mode, support_weight_scale, support_weight_min, support_weight_max, support_pmr_eps = (
            validate_support_weighting_config(
                mode=support_weighting_mode,
                scale=support_weight_scale,
                min_weight=support_weight_min,
                max_weight=support_weight_max,
                eps=support_pmr_eps,
            )
        )
        _validate_inputs(data, batch_size, k, gamma, temperature, index_backend, index_metric, index_nlist, index_nprobe)
        if not isinstance(target_update_every, int) or target_update_every <= 0:
            raise ValueError("target_update_every must be a positive integer.")
        if not isinstance(index_refresh_every, int) or index_refresh_every <= 0:
            raise ValueError("index_refresh_every must be a positive integer.")
        if not isinstance(use_persistent_index, bool):
            raise TypeError("use_persistent_index must be a bool.")
        if not isinstance(audit_stale_index, bool):
            raise TypeError("audit_stale_index must be a bool.")
        if not isinstance(audit_encoder_policy_overlap, bool):
            raise TypeError("audit_encoder_policy_overlap must be a bool.")
        if audit_stale_index and not use_persistent_index:
            raise ValueError("audit_stale_index requires use_persistent_index=True.")
        if audit_encoder_policy_overlap and not use_persistent_index:
            raise ValueError("audit_encoder_policy_overlap requires use_persistent_index=True.")
        if index_encoder_policy not in {"online", "target", "ema"}:
            raise ValueError("index_encoder_policy must be one of 'online', 'target', or 'ema'.")
        if index_encoder_policy == "target" and target_model is None:
            raise ValueError("index_encoder_policy='target' requires target_model.")
        if index_ema_tau is not None:
            if not isinstance(index_ema_tau, int | float) or not 0.0 <= float(index_ema_tau) <= 1.0:
                raise ValueError("index_ema_tau must be None or in [0, 1].")
        if index_encoder_policy == "ema" and index_ema_tau is None:
            raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
        if target_tau is not None and (not isinstance(target_tau, int | float) or not 0.0 <= float(target_tau) <= 1.0):
            raise ValueError("target_tau must be None or in [0, 1].")
        if optimizer is not None and optimizer_bundle is not None:
            raise ValueError("Provide either optimizer or optimizer_bundle, not both.")
        if optimizer is None and optimizer_bundle is None:
            raise ValueError("An optimizer or optimizer_bundle is required.")
        if loss_config is not None:
            _validate_loss_config(loss_config)
        if not isinstance(split_backward, bool):
            raise TypeError("split_backward must be a bool.")
        if split_backward and optimizer_bundle is None:
            raise ValueError("split_backward=True requires optimizer_bundle.")
        if not isinstance(bellman_moment_weight, int | float) or float(bellman_moment_weight) < 0:
            raise ValueError("bellman_moment_weight must be a nonnegative float.")
        if max_grad_norm is not None:
            if not isinstance(max_grad_norm, int | float) or float(max_grad_norm) <= 0:
                raise ValueError("max_grad_norm must be None or a positive float.")
        if not isinstance(q_l2_regularization_weight, int | float) or float(q_l2_regularization_weight) < 0:
            raise ValueError("q_l2_regularization_weight must be a nonnegative float.")
        if q_value_clip is not None:
            if not isinstance(q_value_clip, int | float) or float(q_value_clip) <= 0:
                raise ValueError("q_value_clip must be None or a positive float.")
        if not isinstance(reward_scale, int | float) or float(reward_scale) <= 0:
            raise ValueError("reward_scale must be a positive float.")
        if not isinstance(ranking_margin_weight, int | float) or float(ranking_margin_weight) < 0:
            raise ValueError("ranking_margin_weight must be a nonnegative float.")
        if not isinstance(ranking_margin, int | float) or float(ranking_margin) <= 0:
            raise ValueError("ranking_margin must be a positive float.")
        if not isinstance(ranking_negative_samples, int) or ranking_negative_samples <= 0:
            raise ValueError("ranking_negative_samples must be a positive integer.")
        if not isinstance(retrieval_alignment_weight, int | float) or float(retrieval_alignment_weight) < 0:
            raise ValueError("retrieval_alignment_weight must be a nonnegative float.")
        if not isinstance(retrieval_alignment_margin, int | float) or float(retrieval_alignment_margin) <= 0:
            raise ValueError("retrieval_alignment_margin must be a positive float.")
        if not isinstance(retrieval_alignment_warmup_steps, int) or retrieval_alignment_warmup_steps < 0:
            raise ValueError("retrieval_alignment_warmup_steps must be a nonnegative int.")
        if retrieval_alignment_loss_type not in {"hinge", "listwise_kl"}:
            raise ValueError("retrieval_alignment_loss_type must be 'hinge' or 'listwise_kl'.")
        if not isinstance(listwise_teacher_temperature, int | float) or float(listwise_teacher_temperature) <= 0:
            raise ValueError("listwise_teacher_temperature must be a positive float.")
        if not isinstance(listwise_retrieval_temperature, int | float) or float(listwise_retrieval_temperature) <= 0:
            raise ValueError("listwise_retrieval_temperature must be a positive float.")

        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.optimizer_bundle = optimizer_bundle
        self.batch_size = batch_size
        self.k = k
        self.gamma = gamma
        self.generator = generator
        self.target_model = target_model
        self.target_tau = target_tau
        self.target_update_every = target_update_every
        self.temperature = temperature
        self.density_max_weight = density_max_weight
        self.stationarity_weight = stationarity_weight
        self.density_reg_weight = density_reg_weight
        self.detach_bellman_target = detach_bellman_target
        self.loss_config = loss_config
        self.split_backward = split_backward
        self.index_backend = index_backend
        self.index_metric = index_metric
        self.index_nlist = index_nlist
        self.index_nprobe = index_nprobe
        self.index_refresh_every = index_refresh_every
        self.use_persistent_index = use_persistent_index
        self.audit_stale_index = audit_stale_index
        self.audit_encoder_policy_overlap = audit_encoder_policy_overlap
        self.index_encoder_policy = index_encoder_policy
        self.index_ema_tau = None if index_ema_tau is None else float(index_ema_tau)
        self.retrieval_correction_mode = retrieval_correction_mode
        self.support_correction_weight = support_correction_weight
        self.support_correction_target = support_correction_target
        self.support_correction_metric = support_correction_metric
        self.support_weighting_mode = support_weighting_mode
        self.support_weight_scale = support_weight_scale
        self.support_weight_min = support_weight_min
        self.support_weight_max = support_weight_max
        self.support_pmr_eps = support_pmr_eps
        self.bellman_moment_weight = float(bellman_moment_weight)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.q_l2_regularization_weight = float(q_l2_regularization_weight)
        self.q_value_clip = None if q_value_clip is None else float(q_value_clip)
        self.reward_scale = float(reward_scale)
        self.ranking_margin_weight = float(ranking_margin_weight)
        self.ranking_margin = float(ranking_margin)
        self.ranking_negative_samples = int(ranking_negative_samples)
        self.retrieval_alignment_weight = float(retrieval_alignment_weight)
        self.retrieval_alignment_margin = float(retrieval_alignment_margin)
        self.retrieval_alignment_warmup_steps = int(retrieval_alignment_warmup_steps)
        self.retrieval_alignment_loss_type = retrieval_alignment_loss_type
        self.listwise_teacher_temperature = float(listwise_teacher_temperature)
        self.listwise_retrieval_temperature = float(listwise_retrieval_temperature)
        self.cql_weight = float(cql_weight)
        self.cql_num_samples = max(1, int(cql_num_samples))
        self.bcq_weight = float(bcq_weight)
        self.wolpertinger_bc_weight = float(wolpertinger_bc_weight)
        self.index_ema_model = clone_target_network(model) if index_encoder_policy == "ema" else None
        # Device placement. Default "cpu" leaves all existing behavior byte-identical.
        # When "cuda": model weights go to GPU; the replay buffer and action catalogue
        # stay on CPU (large replay buffers — e.g. 20M transitions at dim 64 — exceed
        # GPU VRAM). Sampled mini-batches are moved to GPU on each step. The ANN/FAISS
        # path stays on CPU. Sampler RNG draws on CPU then moves indices, preserving
        # the RNG stream regardless of device.
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        if self.target_model is not None:
            self.target_model = self.target_model.to(self.device)
        if self.index_ema_model is not None:
            self.index_ema_model = self.index_ema_model.to(self.device)
        # Replay buffer and catalogue always stay on CPU — move per-batch instead.
        self.index_cache: SyntheticIndexCache | None = None
        self._step = 0
        self._target_update_count = 0
        self._target_update_steps: list[int] = []
        self._index_ema_update_count = 0
        self._index_age = 0
        self._index_refresh_count = 0
        self._index_refresh_steps: list[int] = []
        if self.use_persistent_index:
            self.index_cache = self._build_index_cache(last_refresh_step=0, refresh_count=0, age=0)

    @property
    def step(self) -> int:
        """Number of completed optimization steps."""

        return self._step

    @property
    def target_update_count(self) -> int:
        """Number of target soft updates applied by this trainer."""

        return self._target_update_count

    @property
    def index_ema_update_count(self) -> int:
        """Number of EMA index-encoder updates applied by this trainer."""

        return self._index_ema_update_count

    @property
    def index_age(self) -> int:
        """Number of completed steps since the last diagnostic index refresh.

        The current train step still rebuilds its retrieval index internally.
        This age is bookkeeping for future persistent-index reuse: after a
        completed step, a cadence hit records a refresh and resets age to zero;
        otherwise age increments by one.
        """

        return self._index_age

    @property
    def index_refresh_count(self) -> int:
        """Number of diagnostic index-refresh events recorded."""

        return self._index_refresh_count

    @property
    def index_refresh_steps(self) -> list[int]:
        """Completed trainer steps at which diagnostic refreshes were recorded."""

        return list(self._index_refresh_steps)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable synthetic trainer counters.

        Persistent ANN/FAISS index objects are deliberately excluded. When this
        state is loaded, a persistent synthetic index is rebuilt from the
        current configured encoder if needed.
        """

        return {
            "step": self._step,
            "target_update_count": self._target_update_count,
            "target_update_steps": list(self._target_update_steps),
            "index_age": self._index_age,
            "index_refresh_count": self._index_refresh_count,
            "index_refresh_steps": list(self._index_refresh_steps),
            "index_ema_update_count": self._index_ema_update_count,
            "index_encoder_policy": self.index_encoder_policy,
            "retrieval_correction_mode": self.retrieval_correction_mode,
            "support_correction_weight": self.support_correction_weight,
            "support_correction_target": self.support_correction_target,
            "support_correction_metric": self.support_correction_metric,
            "support_weighting_mode": self.support_weighting_mode,
            "support_weight_scale": self.support_weight_scale,
            "support_weight_min": self.support_weight_min,
            "support_weight_max": self.support_weight_max,
            "support_pmr_eps": self.support_pmr_eps,
            "bellman_moment_weight": self.bellman_moment_weight,
            "use_persistent_index": self.use_persistent_index,
            "index_refresh_every": self.index_refresh_every,
            "persistent_index_last_refresh_step": None if self.index_cache is None else self.index_cache.last_refresh_step,
            "diagnostics": {
                "synthetic_only": True,
                "persistent_index_serialized": False,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore serializable trainer counters from ``state_dict`` output."""

        if not isinstance(state, dict):
            raise TypeError("trainer state must be a dictionary.")
        self._step = int(state.get("step", 0))
        self._target_update_count = int(state.get("target_update_count", 0))
        self._target_update_steps = [int(step) for step in state.get("target_update_steps", [])]
        self._index_age = int(state.get("index_age", 0))
        self._index_refresh_count = int(state.get("index_refresh_count", 0))
        self._index_refresh_steps = [int(step) for step in state.get("index_refresh_steps", [])]
        self._index_ema_update_count = int(state.get("index_ema_update_count", 0))
        if self.use_persistent_index:
            last_refresh_step = state.get("persistent_index_last_refresh_step")
            if last_refresh_step is None:
                last_refresh_step = self._index_refresh_steps[-1] if self._index_refresh_steps else 0
            self.index_cache = self._build_index_cache(
                last_refresh_step=int(last_refresh_step),
                refresh_count=self._index_refresh_count,
                age=self._index_age,
            )
        else:
            self.index_cache = None

    def _build_index_cache(
        self,
        *,
        last_refresh_step: int,
        refresh_count: int,
        age: int,
    ) -> SyntheticIndexCache:
        """Build a synthetic persistent index from the configured encoder policy."""

        with torch.no_grad():
            # Encoding runs on the model's device; move features there then move
            # embeddings back to CPU for FAISS (which is CPU-only).
            action_features = self.data.action_features.to(self.device)
            action_ids = self.data.action_ids.to(device=action_features.device)
            if self.index_encoder_policy == "target":
                encoder_model = self.target_model
            elif self.index_encoder_policy == "ema":
                encoder_model = self.index_ema_model
            else:
                encoder_model = self.model
            if encoder_model is None:
                raise ValueError(f"index_encoder_policy={self.index_encoder_policy!r} requires an encoder model.")
            # Persistent index uses retrieval embeddings (B2 projection when
            # present, else base) so the cached training-time truncation matches
            # the eval-time ANN retrieval (no train/inference gap).
            catalogue_embeddings = encoder_model.encode_action_index(action_features).detach()
            index = make_action_index(
                catalogue_embeddings,
                action_ids=action_ids,
                backend=self.index_backend,
                metric=self.index_metric,
                normalize=self.index_metric == "cosine",
                nlist=self.index_nlist,
                nprobe=self.index_nprobe,
            )
        return SyntheticIndexCache(
            index=index,
            index_backend=self.index_backend,
            index_metric=self.index_metric,
            age=age,
            refresh_count=refresh_count,
            last_refresh_step=last_refresh_step,
        )

    def _refresh_persistent_index(self) -> None:
        """Refresh the synthetic persistent index cache after a completed step."""

        self.index_cache = self._build_index_cache(
            last_refresh_step=self._step,
            refresh_count=self._index_refresh_count,
            age=self._index_age,
        )

    def _compute_index_encoder_drift(self) -> dict[str, float] | None:
        """Compare online action embeddings to the configured index encoder."""

        if not self.use_persistent_index:
            return None
        with torch.no_grad():
            action_features = self.data.action_features.to(self.device)
            online_embeddings = self.model.encode_actions(action_features).detach()
            if self.index_encoder_policy == "online":
                comparison_embeddings = online_embeddings
            elif self.index_encoder_policy == "target":
                if self.target_model is None:
                    raise ValueError("index_encoder_policy='target' requires target_model.")
                comparison_embeddings = self.target_model.encode_actions(action_features).detach()
            elif self.index_encoder_policy == "ema":
                if self.index_ema_model is None:
                    raise ValueError("index_encoder_policy='ema' requires index_ema_model.")
                comparison_embeddings = self.index_ema_model.encode_actions(action_features).detach()
            else:  # pragma: no cover - constructor validation owns this.
                raise ValueError(f"Unknown index_encoder_policy: {self.index_encoder_policy!r}.")
        return compute_action_embedding_drift(online_embeddings, comparison_embeddings)

    def _compute_encoder_policy_retrieval_overlap(self) -> dict[str, float]:
        """Compute fresh retrieval overlap between available encoder policies."""

        if not self.audit_encoder_policy_overlap:
            return {}
        query_batch_size = min(self.batch_size, len(self.data.dataset))
        query_batch = self.data.dataset.sample(query_batch_size, generator=self.generator)
        return compute_encoder_policy_retrieval_overlap(
            data=self.data,
            model=self.model,
            query_states=query_batch.next_states,
            k=self.k,
            index_backend=self.index_backend,
            index_metric=self.index_metric,
            target_model=self.target_model,
            ema_index_model=self.index_ema_model,
            index_nlist=self.index_nlist,
            index_nprobe=self.index_nprobe,
        )

    def _require_objective_bundle(self) -> OptimizerBundle:
        if self.optimizer_bundle is None:
            raise ValueError("Explicit objective updates require optimizer_bundle.")
        return self.optimizer_bundle

    def _compute_objective_terms(
        self,
        *,
        detach_bellman_target: bool | None = None,
        detach_density_weights_for_bellman: bool = True,
        detach_policy_probs_for_bellman: bool = True,
        detach_density_weights_for_stationarity: bool = False,
        detach_dual_for_stationarity: bool = False,
    ) -> SyntheticTrainingBatchTerms:
        return compute_synthetic_training_batch_terms(
            data=self.data,
            model=self.model,
            batch_size=self.batch_size,
            k=self.k,
            gamma=self.gamma,
            generator=self.generator,
            temperature=self.temperature,
            density_max_weight=self.density_max_weight,
            target_model=self.target_model,
            detach_bellman_target=self.detach_bellman_target if detach_bellman_target is None else detach_bellman_target,
            detach_density_weights_for_bellman=detach_density_weights_for_bellman,
            detach_policy_probs_for_bellman=detach_policy_probs_for_bellman,
            detach_density_weights_for_stationarity=detach_density_weights_for_stationarity,
            detach_dual_for_stationarity=detach_dual_for_stationarity,
            index_backend=self.index_backend,
            index_metric=self.index_metric,
            index_nlist=self.index_nlist,
            index_nprobe=self.index_nprobe,
            action_index=None if self.index_cache is None else self.index_cache.index,
            audit_stale_index=self.audit_stale_index,
            retrieval_correction_mode=self.retrieval_correction_mode,
            support_correction_weight=self.support_correction_weight,
            support_correction_target=self.support_correction_target,
            support_correction_metric=self.support_correction_metric,
            support_weighting_mode=self.support_weighting_mode,
            support_weight_scale=self.support_weight_scale,
            support_weight_min=self.support_weight_min,
            support_weight_max=self.support_weight_max,
            support_pmr_eps=self.support_pmr_eps,
            reward_scale=self.reward_scale,
            q_value_clip=self.q_value_clip,
        )

    def _retrieval_projection_param_ids(self) -> set[int]:
        """Return id()s of B2 retrieval projection and Wolpertinger proto-action parameters.

        These are clipped separately from the core encoder/Q-head params so that
        large alignment or BC gradients cannot scale down the Bellman gradients
        through a shared global clip factor.
        """
        ids: set[int] = set()
        if getattr(self.model, "has_retrieval_projection", False):
            for module in (self.model.retrieval_state_proj, self.model.retrieval_action_proj):
                if module is not None:
                    ids.update(id(p) for p in module.parameters())
        if getattr(self.model, "has_proto_action", False) and self.model.proto_action_head is not None:
            ids.update(id(p) for p in self.model.proto_action_head.parameters())
        return ids

    def _apply_objective_update(
        self,
        *,
        loss: "torch.Tensor",
        objective_name: str,
        update_parts: list[str],
        diagnostics: dict[str, Any],
    ) -> ObjectiveUpdateResult:
        bundle = self._require_objective_bundle()
        optimizers = _bundle_optimizers(bundle)
        missing = [part for part in update_parts if part not in optimizers]
        if missing:
            raise ValueError(f"optimizer_bundle is missing optimizer(s) for objective update part(s): {missing}.")
        if torch.any(~torch.isfinite(loss)):
            raise ValueError(f"{objective_name} objective loss must be finite.")

        parameters = [param for param in self.model.parameters() if param.requires_grad]
        before = [param.detach().clone() for param in parameters]
        zero_grad_bundle(bundle)
        loss.backward()
        grad_norm_before = _grad_norm(parameters)
        if self.max_grad_norm is not None:
            # B2: clip the retrieval projection heads separately from the core
            # params. A shared global clip would let the (large, sharp) KL
            # projection gradients scale down the Bellman gradients, re-coupling
            # the alignment loss into Q learning through the clip factor.
            proj_param_ids = self._retrieval_projection_param_ids()
            if proj_param_ids:
                core_params = [p for p in parameters if id(p) not in proj_param_ids]
                proj_params = [p for p in parameters if id(p) in proj_param_ids]
                torch.nn.utils.clip_grad_norm_(core_params, self.max_grad_norm)
                if proj_params:
                    torch.nn.utils.clip_grad_norm_(proj_params, self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        grad_norm_after = _grad_norm(parameters)
        stepped_parts = step_bundle(bundle, update_parts)
        delta_norm = _parameter_delta_norm(before, parameters)
        if not torch.isfinite(grad_norm_before) or not torch.isfinite(delta_norm):
            raise ValueError("Objective gradient and parameter delta norms must be finite.")

        return ObjectiveUpdateResult(
            loss=loss.detach(),
            objective_name=objective_name,
            grad_norm=grad_norm_after.detach(),
            parameter_delta_norm=delta_norm.detach(),
            diagnostics={
                **diagnostics,
                "objective_name": objective_name,
                "update_parts": list(update_parts),
                "stepped_parts": stepped_parts,
                "gradient_group_norms": gradient_group_norms(self.model),
                "parameter_delta_norm": float(delta_norm.item()),
                "grad_norm_before_clip": float(grad_norm_before.item()),
                "grad_norm_after_clip": float(grad_norm_after.item()),
            },
        )

    def _compute_ranking_margin_loss(self) -> "torch.Tensor":
        """Auxiliary ranking-margin loss pushing positive Q above negative Q by a margin.

        Loss = mean(relu(margin - Q(s, pos) + Q(s, neg))) over a random batch.
        Only called when ranking_margin_weight > 0. Auxiliary stabilization only,
        not a core objective component.
        """

        batch = self.data.dataset.sample(self.batch_size, generator=self.generator)
        if batch.states.device != self.device:
            batch = batch.to(self.device)
        state_emb = self.model.encode_states(batch.states)  # [B, D]
        # Positive: the logged action features from the batch [B, D]
        pos_emb = self.model.encode_actions(batch.actions)  # [B, D]
        pos_q = self.model.q_values(state_emb, pos_emb)  # [B]

        # Negative: random catalogue items [B, K, action_dim]
        action_features = self.data.action_features.to(device=state_emb.device)
        num_actions = action_features.shape[0]
        num_neg = min(self.ranking_negative_samples, num_actions)
        neg_ids = torch.randint(0, num_actions, (self.batch_size, num_neg), generator=self.generator).to(state_emb.device)
        neg_feats = action_features[neg_ids]  # [B, K, action_dim]
        neg_emb = self.model.encode_actions(neg_feats.reshape(-1, neg_feats.shape[-1])).reshape(
            self.batch_size, num_neg, -1
        )  # [B, K, D]
        neg_q = self.model.q_values(state_emb, neg_emb)  # [B, K] — pair builder broadcasts [B,D] to [B,K,D]

        # Hinge loss: push pos_q >= neg_q + margin
        margin_loss = torch.relu(self.ranking_margin - pos_q.unsqueeze(1) + neg_q).mean()
        return margin_loss

    def _effective_alignment_weight(self) -> float:
        """Return alignment weight with linear warmup applied."""
        if self.retrieval_alignment_warmup_steps <= 0:
            return self.retrieval_alignment_weight
        ramp = min(1.0, (self._step + 1) / self.retrieval_alignment_warmup_steps)
        return self.retrieval_alignment_weight * ramp

    def _compute_retrieval_alignment_loss(self) -> "torch.Tensor":
        """Hinge triplet: mean(relu(margin - dot(s,pos) + dot(s,neg))).

        Retained for backward compatibility. Use listwise_kl for better alignment.
        """
        batch = self.data.dataset.sample(self.batch_size, generator=self.generator)
        if batch.states.device != self.device:
            batch = batch.to(self.device)
        state_emb = self.model.encode_states(batch.states)  # [B, D]
        pos_emb = self.model.encode_actions(batch.actions)  # [B, D]
        pos_dot = (state_emb * pos_emb).sum(dim=-1)  # [B]

        action_features = self.data.action_features.to(device=state_emb.device)
        num_actions = action_features.shape[0]
        num_neg = min(self.ranking_negative_samples, num_actions)
        neg_ids = torch.randint(0, num_actions, (self.batch_size, num_neg), generator=self.generator).to(state_emb.device)
        neg_feats = action_features[neg_ids]  # [B, K, action_dim]
        neg_emb = self.model.encode_actions(neg_feats.reshape(-1, neg_feats.shape[-1])).reshape(
            self.batch_size, num_neg, -1
        )  # [B, K, D]
        neg_dot = (state_emb.unsqueeze(1) * neg_emb).sum(dim=-1)  # [B, K]
        return torch.relu(self.retrieval_alignment_margin - pos_dot.unsqueeze(1) + neg_dot).mean()

    def _compute_listwise_kl_loss(self) -> "torch.Tensor":
        """Listwise KL distillation: KL(pi_Q || pi_R) over a candidate set.

        Distills the Q-ranker's full ordering into the embedding dot-product space
        that ANN uses. pi_Q = softmax(Q/T_Q), pi_R = softmax(dot/T_R) over the
        positive plus ranking_negative_samples random negatives.
        """
        import torch.nn.functional as F  # noqa: PLC0415

        batch = self.data.dataset.sample(self.batch_size, generator=self.generator)
        if batch.states.device != self.device:
            batch = batch.to(self.device)
        states = batch.states
        device = self.device

        action_features = self.data.action_features.to(device=device)
        num_actions = action_features.shape[0]
        num_neg = min(self.ranking_negative_samples, num_actions - 1)

        neg_ids = torch.randint(0, num_actions, (self.batch_size, num_neg), generator=self.generator).to(device)
        neg_feats = action_features[neg_ids]  # [B, K, action_dim]

        # Candidate set: [positive] + [negatives] → [B, K+1, action_dim]
        pos_feats = batch.actions.to(device=device).unsqueeze(1)
        cand_feats = torch.cat([pos_feats, neg_feats], dim=1)
        B, C, Adim = cand_feats.shape

        # Teacher: Q ordering over candidates — base encoders, fully detached so
        # no KL gradient touches the Q/Bellman path.
        with torch.no_grad():
            state_emb_base = self.model.encode_states(states)  # [B, D]
            cand_emb_base = self.model.encode_actions(cand_feats.reshape(B * C, Adim)).reshape(B, C, -1)
            q_scores = self.model.q_values(state_emb_base, cand_emb_base)  # [B, C]

        # Student: retrieval dot product. With the B2 projection, encode_*_query/
        # index apply trainable projections to DETACHED base embeddings, so the
        # KL gradient reaches only the projection heads. Without projection, fall
        # back to the shared encoders with the state side detached (E3 behavior).
        # When the index uses cosine, train the student on L2-normalized
        # (direction-only) similarity so the learned ranking is norm-invariant
        # and matches what the cosine ANN index retrieves. Otherwise use raw dot.
        cosine_space = self.index_metric == "cosine"
        if self.model.has_retrieval_projection:
            state_q = self.model.encode_state_query(states)  # [B, D]
            cand_idx = self.model.encode_action_index(cand_feats.reshape(B * C, Adim)).reshape(B, C, -1)
            if cosine_space:
                state_q = F.normalize(state_q, dim=-1)
                cand_idx = F.normalize(cand_idx, dim=-1)
            dot_scores = (state_q.unsqueeze(1) * cand_idx).sum(dim=-1)  # [B, C]
        else:
            state_emb = self.model.encode_states(states).detach()
            cand_emb = self.model.encode_actions(cand_feats.reshape(B * C, Adim)).reshape(B, C, -1)
            if cosine_space:
                state_emb = F.normalize(state_emb, dim=-1)
                cand_emb = F.normalize(cand_emb, dim=-1)
            dot_scores = (state_emb.unsqueeze(1) * cand_emb).sum(dim=-1)  # [B, C]

        pi_Q = F.softmax(q_scores / self.listwise_teacher_temperature, dim=-1)  # [B, C]
        log_pi_R = F.log_softmax(dot_scores / self.listwise_retrieval_temperature, dim=-1)  # [B, C]

        # KL(pi_Q || pi_R) — Q is teacher, embedding dot is student
        return F.kl_div(log_pi_R, pi_Q, reduction="batchmean")

    def _compute_cql_loss(self) -> "torch.Tensor":
        """Conservative Q-Learning penalty (CQL, Kumar et al. 2020).

        Adds logsumexp(Q over random catalogue actions) - Q(logged action) to
        penalise overestimation of out-of-distribution Q-values.
        Only called when cql_weight > 0.
        """
        import torch.nn.functional as F  # noqa: PLC0415

        batch = self.data.dataset.sample(self.batch_size, generator=self.generator)
        if batch.states.device != self.device:
            batch = batch.to(self.device)
        action_features = self.data.action_features.to(device=self.device)
        num_actions = action_features.shape[0]
        num_samples = min(self.cql_num_samples, num_actions)

        state_emb = self.model.encode_states(batch.states)  # [B, D]
        log_action_emb = self.model.encode_actions(batch.actions)  # [B, D]
        q_logged = self.model.q_values(state_emb, log_action_emb)  # [B]

        rand_ids = torch.randint(0, num_actions, (self.batch_size, num_samples), generator=self.generator).to(self.device)
        rand_feats = action_features[rand_ids]  # [B, num_samples, action_dim]
        rand_emb = self.model.encode_actions(rand_feats.reshape(-1, rand_feats.shape[-1])).reshape(
            self.batch_size, num_samples, -1
        )
        q_rand = self.model.q_values(state_emb, rand_emb)  # [B, num_samples]

        logsumexp_q = q_rand.logsumexp(dim=1)  # [B]
        return (logsumexp_q - q_logged).mean()

    def _compute_bcq_loss(self) -> "torch.Tensor":
        """BCQ behavioral cloning constraint (Fujimoto et al. 2019, discrete variant).

        Constrains the policy to the behavioral support: treats the logged action as
        the target class and random catalogue actions as negatives, minimising the
        cross-entropy loss so Q ranks logged actions above OOD samples.
        Only called when bcq_weight > 0.
        """
        import torch.nn.functional as F  # noqa: PLC0415

        batch = self.data.dataset.sample(self.batch_size, generator=self.generator)
        if batch.states.device != self.device:
            batch = batch.to(self.device)
        action_features = self.data.action_features.to(device=self.device)
        num_actions = action_features.shape[0]

        state_emb = self.model.encode_states(batch.states)  # [B, D]
        log_action_emb = self.model.encode_actions(batch.actions)  # [B, D]
        q_logged = self.model.q_values(state_emb, log_action_emb)  # [B]

        rand_ids = torch.randint(0, num_actions, (self.batch_size,), generator=self.generator).to(self.device)
        rand_feats = action_features[rand_ids]  # [B, action_dim]
        rand_emb = self.model.encode_actions(rand_feats)  # [B, D]
        q_rand = self.model.q_values(state_emb, rand_emb)  # [B]

        # CE: logged action = class 0, random action = class 1
        logits = torch.stack([q_logged, q_rand], dim=1)  # [B, 2]
        labels = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)

    def _compute_wolpertinger_bc_loss(self) -> "torch.Tensor":
        """Wolpertinger BC: train proto-action head to predict logged action embedding.

        Dulac-Arnold et al. 2015 (offline variant): the proto-action head maps a
        detached state embedding to action embedding space via MSE on logged actions.
        The proto-action output is used as the ANN query key (encode_state_query),
        so the ANN retrieves actions near the behavior policy's logged action.
        Only called when wolpertinger_bc_weight > 0 and model.has_proto_action.
        """
        import torch.nn.functional as F  # noqa: PLC0415

        batch = self.data.dataset.sample(self.batch_size, generator=self.generator)
        if batch.states.device != self.device:
            batch = batch.to(self.device)
        # Proto-action reads DETACHED state embedding (encode_state_query fires
        # proto_action_head branch). No gradient to base state encoder.
        state_proto = self.model.encode_state_query(batch.states)  # [B, D]
        with torch.no_grad():
            logged_action_emb = self.model.encode_actions(batch.actions)  # [B, D]
        return F.mse_loss(state_proto, logged_action_emb)

    def update_critic(self, *, update_encoder: bool = True) -> ObjectiveUpdateResult:
        """Run one critic-only synthetic DW-MQL+ANN update.

        The critic minimizes the stabilizing weighted TD loss plus the Bellman
        moment term ``(E[w δ])^2`` (scaled by ``bellman_moment_weight``). Density
        weights and the Bellman target are detached so that gradients flow only
        through the Q-head and encoder.
        """

        if not isinstance(update_encoder, bool):
            raise TypeError("update_encoder must be a bool.")
        self.model.train()
        terms = self._compute_objective_terms(
            detach_bellman_target=True,
            detach_density_weights_for_bellman=True,
            detach_policy_probs_for_bellman=True,
        )
        update_parts = ["critic"]
        if update_encoder:
            update_parts.insert(0, "encoder")
        support_correction_active = bool(terms.diagnostics.get("support_correction_active"))
        support_weighting_active = bool(terms.diagnostics.get("support_weighting_active"))
        critic_loss = terms.support_weighted_bellman_loss
        loss_source = "support_weighted_bellman_loss" if support_weighting_active else "bellman_loss"
        if support_correction_active:
            critic_loss = critic_loss + self.support_correction_weight * terms.support_correction_loss
            loss_source = f"{loss_source}_plus_support_correction"
        if self.bellman_moment_weight > 0.0:
            critic_loss = critic_loss + self.bellman_moment_weight * terms.bellman_moment_loss
            loss_source = f"{loss_source}_plus_bellman_moment"
        q_abs_mean = float(terms.current_q.detach().abs().mean().item())
        q_abs_max = float(terms.current_q.detach().abs().max().item())
        q_l2_reg_tensor = self.q_l2_regularization_weight * terms.current_q.pow(2).mean()
        q_l2_reg_val = float(q_l2_reg_tensor.detach().item())
        if self.q_l2_regularization_weight > 0.0:
            critic_loss = critic_loss + q_l2_reg_tensor
            loss_source = f"{loss_source}_plus_q_l2_regularization"
        ranking_margin_loss_val = 0.0
        if self.ranking_margin_weight > 0.0:
            ranking_margin_tensor = self._compute_ranking_margin_loss()
            ranking_margin_loss_val = float(ranking_margin_tensor.detach().item())
            critic_loss = critic_loss + self.ranking_margin_weight * ranking_margin_tensor
            loss_source = f"{loss_source}_plus_ranking_margin"
        retrieval_alignment_loss_val = 0.0
        effective_alignment_weight = self._effective_alignment_weight()
        if effective_alignment_weight > 0.0:
            if self.retrieval_alignment_loss_type == "listwise_kl":
                retrieval_alignment_tensor = self._compute_listwise_kl_loss()
            else:
                retrieval_alignment_tensor = self._compute_retrieval_alignment_loss()
            retrieval_alignment_loss_val = float(retrieval_alignment_tensor.detach().item())
            critic_loss = critic_loss + effective_alignment_weight * retrieval_alignment_tensor
            loss_source = f"{loss_source}_plus_retrieval_alignment"
        cql_loss_val = 0.0
        if self.cql_weight > 0.0:
            cql_tensor = self._compute_cql_loss()
            cql_loss_val = float(cql_tensor.detach().item())
            critic_loss = critic_loss + self.cql_weight * cql_tensor
            loss_source = f"{loss_source}_plus_cql"
        bcq_loss_val = 0.0
        if self.bcq_weight > 0.0:
            bcq_tensor = self._compute_bcq_loss()
            bcq_loss_val = float(bcq_tensor.detach().item())
            critic_loss = critic_loss + self.bcq_weight * bcq_tensor
            loss_source = f"{loss_source}_plus_bcq"
        wolpertinger_bc_loss_val = 0.0
        if self.wolpertinger_bc_weight > 0.0 and getattr(self.model, "has_proto_action", False):
            wbc_tensor = self._compute_wolpertinger_bc_loss()
            wolpertinger_bc_loss_val = float(wbc_tensor.detach().item())
            critic_loss = critic_loss + self.wolpertinger_bc_weight * wbc_tensor
            loss_source = f"{loss_source}_plus_wolpertinger_bc"
        return self._apply_objective_update(
            loss=critic_loss,
            objective_name="critic",
            update_parts=update_parts,
            diagnostics={
                **terms.diagnostics,
                "loss_source": loss_source,
                "bellman_target_detached": True,
                "density_weights_detached": True,
                "bellman_moment_weight": float(self.bellman_moment_weight),
                "q_abs_mean": q_abs_mean,
                "q_abs_max": q_abs_max,
                "q_l2_regularization": q_l2_reg_val,
                "ranking_margin_loss": ranking_margin_loss_val,
                "ranking_margin_weight": self.ranking_margin_weight,
                "retrieval_alignment_loss": retrieval_alignment_loss_val,
                "retrieval_alignment_weight": effective_alignment_weight,
                "retrieval_alignment_loss_type": self.retrieval_alignment_loss_type,
                "cql_loss": cql_loss_val,
                "cql_weight": self.cql_weight,
                "bcq_loss": bcq_loss_val,
                "bcq_weight": self.bcq_weight,
                "wolpertinger_bc_loss": wolpertinger_bc_loss_val,
                "wolpertinger_bc_weight": self.wolpertinger_bc_weight,
                # Full quantity actually optimized (bellman + correction + moment +
                # q_l2 + ranking + retrieval + cql + bcq). Distinct key so it does not
                # clobber the `critic_total_loss` decomposition key that
                # _compute_objective_terms defines as (bellman + correction) and that
                # tests assert against.
                "critic_optimized_loss": float(critic_loss.detach().item()),
                "reward_scale": self.reward_scale,
            },
        )

    def update_density(self, *, update_encoder: bool = False) -> ObjectiveUpdateResult:
        """Run one density objective update using stationarity plus regularization.

        Dual f-values are detached from the stationarity computation so that
        gradients flow only through the density head, matching the minimax
        structure where the density player minimizes over w with f fixed.
        """

        if not isinstance(update_encoder, bool):
            raise TypeError("update_encoder must be a bool.")
        self.model.train()
        terms = self._compute_objective_terms(detach_dual_for_stationarity=True)
        if self.retrieval_correction_mode == "no_density":
            zero = terms.density_regularization.detach().new_zeros(())
            return ObjectiveUpdateResult(
                loss=zero,
                objective_name="density",
                grad_norm=zero,
                parameter_delta_norm=zero,
                diagnostics={
                    **terms.diagnostics,
                    "objective_name": "density",
                    "loss_source": "density_disabled_by_retrieval_correction_mode",
                    "retrieval_correction_mode": self.retrieval_correction_mode,
                    "update_parts": [],
                    "stepped_parts": [],
                    "gradient_group_norms": gradient_group_norms(self.model),
                    "parameter_delta_norm": 0.0,
                },
            )
        density_loss = self.stationarity_weight * terms.stationarity_loss + self.density_reg_weight * terms.density_regularization
        update_parts = ["density"]
        if update_encoder:
            update_parts.insert(0, "encoder")
        return self._apply_objective_update(
            loss=density_loss,
            objective_name="density",
            update_parts=update_parts,
            diagnostics={
                **terms.diagnostics,
                "loss_source": "stationarity_loss_plus_density_regularization",
                "stationarity_weight": float(self.stationarity_weight),
                "density_reg_weight": float(self.density_reg_weight),
                "dual_detached_for_stationarity": True,
            },
        )

    def update_dual(self) -> ObjectiveUpdateResult:
        """Run one dual/test-function ascent-style objective update.

        Density weights are detached from the stationarity computation so that
        gradients flow only through the dual head (f-function), matching the
        minimax structure where the dual player maximizes over f with w fixed.
        """

        self.model.train()
        terms = self._compute_objective_terms(detach_density_weights_for_stationarity=True)
        dual_loss = -terms.stationarity_loss
        return self._apply_objective_update(
            loss=dual_loss,
            objective_name="dual",
            update_parts=["dual"],
            diagnostics={
                **terms.diagnostics,
                "loss_source": "negative_stationarity_loss",
                "dual_update_sign": "ascent_via_minimize_negative_stationarity_loss",
                "maximized_quantity": "stationarity_loss",
                "density_weights_detached_for_stationarity": True,
            },
        )

    def train_objective_step(self, schedule_step: str) -> ObjectiveUpdateResult:
        """Run one named objective update: critic, density, or dual."""

        if schedule_step == "critic":
            return self.update_critic()
        if schedule_step == "density":
            return self.update_density()
        if schedule_step == "dual":
            return self.update_dual()
        raise ValueError("schedule_step must be one of 'critic', 'density', or 'dual'.")

    # ------------------------------------------------------------------ #
    # DW-MQL experiment-ready update methods                               #
    # ------------------------------------------------------------------ #

    def _compute_dwmql_terms(
        self,
        obj_config: DWMQLObjectiveConfig,
        *,
        detach_density_weights_for_stationarity: bool = False,
        detach_dual_for_stationarity: bool = False,
    ) -> DWMQLBatchTerms:
        """Compute DWMQLBatchTerms using the trainer's current index cache."""

        return compute_dwmql_batch_terms(
            data=self.data,
            model=self.model,
            obj_config=obj_config,
            batch_size=self.batch_size,
            k=self.k,
            generator=self.generator,
            temperature=self.temperature,
            target_model=self.target_model,
            action_index=None if self.index_cache is None else self.index_cache.index,
            detach_density_weights_for_stationarity=detach_density_weights_for_stationarity,
            detach_dual_for_stationarity=detach_dual_for_stationarity,
        )

    def update_dwmql_critic(self, obj_config: DWMQLObjectiveConfig) -> ObjectiveUpdateResult:
        """DW-MQL critic update: weighted Bellman loss + optional moment + correction.

        Bellman target and density weights are detached so gradients reach only
        the Q-head and encoder.
        """

        validate_dwmql_objective_config(obj_config)
        if self.optimizer_bundle is None:
            raise ValueError("update_dwmql_critic requires optimizer_bundle.")
        self.model.train()
        terms = self._compute_dwmql_terms(obj_config)
        critic_loss = obj_config.bellman_weight * terms.bellman_loss
        critic_loss = critic_loss + self.bellman_moment_weight * terms.bellman_moment_loss
        if obj_config.support_correction_weight > 0:
            critic_loss = critic_loss + obj_config.support_correction_weight * terms.support_correction_loss
        if obj_config.max_grad_norm is not None:
            with torch.no_grad():
                pass  # grad clipping applied after backward in _apply_objective_update
        return self._apply_objective_update(
            loss=critic_loss,
            objective_name="critic",
            update_parts=["encoder", "critic"],
            diagnostics={
                **terms.diagnostics,
                "loss_source": "dwmql_bellman_plus_moment",
                "bellman_weight": float(obj_config.bellman_weight),
                "bellman_moment_weight": float(self.bellman_moment_weight),
                "max_grad_norm": obj_config.max_grad_norm,
                "update_method": "update_dwmql_critic",
            },
        )

    def update_dwmql_density(self, obj_config: DWMQLObjectiveConfig) -> ObjectiveUpdateResult:
        """DW-MQL density update: stationarity + regularization with dual detached.

        Dual f-values are detached so gradients reach only the density head.
        """

        validate_dwmql_objective_config(obj_config)
        if self.optimizer_bundle is None:
            raise ValueError("update_dwmql_density requires optimizer_bundle.")
        self.model.train()
        terms = self._compute_dwmql_terms(obj_config, detach_dual_for_stationarity=True)
        density_loss = (
            obj_config.stationarity_weight * terms.stationarity_loss
            + obj_config.density_regularization_weight * terms.density_regularization
        )
        return self._apply_objective_update(
            loss=density_loss,
            objective_name="density",
            update_parts=["density"],
            diagnostics={
                **terms.diagnostics,
                "loss_source": "dwmql_stationarity_plus_regularization",
                "stationarity_weight": float(obj_config.stationarity_weight),
                "density_regularization_weight": float(obj_config.density_regularization_weight),
                "dual_detached_for_stationarity": True,
                "max_grad_norm": obj_config.max_grad_norm,
                "update_method": "update_dwmql_density",
            },
        )

    def update_dwmql_dual(self, obj_config: DWMQLObjectiveConfig) -> ObjectiveUpdateResult:
        """DW-MQL dual ascent: maximize stationarity loss with density weights detached.

        Density weights are detached so gradients reach only the dual head.
        Sign convention: minimizing negative stationarity loss implements ascent.
        """

        validate_dwmql_objective_config(obj_config)
        if self.optimizer_bundle is None:
            raise ValueError("update_dwmql_dual requires optimizer_bundle.")
        self.model.train()
        terms = self._compute_dwmql_terms(obj_config, detach_density_weights_for_stationarity=True)
        dual_loss = -obj_config.stationarity_weight * terms.stationarity_loss
        return self._apply_objective_update(
            loss=dual_loss,
            objective_name="dual",
            update_parts=["dual"],
            diagnostics={
                **terms.diagnostics,
                "loss_source": "dwmql_negative_stationarity",
                "dual_update_sign": "ascent_via_minimize_negative_stationarity_loss",
                "maximized_quantity": "stationarity_loss",
                "stationarity_weight": float(obj_config.stationarity_weight),
                "density_weights_detached_for_stationarity": True,
                "max_grad_norm": obj_config.max_grad_norm,
                "update_method": "update_dwmql_dual",
            },
        )

    def _extragradient_dual_density_update(
        self, obj_config: DWMQLObjectiveConfig
    ) -> dict[str, Any]:
        """Two-step stochastic extragradient for dual+density heads.

        Step 1 (lookahead): compute gradient at θ, take a manual SGD step
        to θ_la = θ - step_size * ∇L(θ).
        Step 2 (final): compute gradient at θ_la, restore θ, apply
        the final gradient: θ_new = θ - step_size * ∇L(θ_la).

        Uses separate batches for lookahead and final (stochastic EG).
        Gradient isolation is maintained: dual update detaches density weights;
        density update detaches dual f-values.
        """

        bundle = self._require_objective_bundle()
        step_size = float(obj_config.extragradient_step_size)

        dual_params = [p for p in self.model.dual_head.parameters() if p.requires_grad]
        density_params = [p for p in self.model.density_head.parameters() if p.requires_grad]
        all_eg_params = dual_params + density_params

        # Save original parameters
        saved = [p.data.clone() for p in all_eg_params]

        # --- Lookahead step: compute ∇L(θ) ---
        zero_grad_bundle(bundle)
        # Dual lookahead gradient (detach density weights)
        la_dual_terms = self._compute_dwmql_terms(obj_config, detach_density_weights_for_stationarity=True)
        la_dual_loss = -obj_config.stationarity_weight * la_dual_terms.stationarity_loss
        la_dual_loss.backward()
        # Density lookahead gradient (detach dual f-values)
        la_density_terms = self._compute_dwmql_terms(obj_config, detach_dual_for_stationarity=True)
        la_density_loss = (
            obj_config.stationarity_weight * la_density_terms.stationarity_loss
            + obj_config.density_regularization_weight * la_density_terms.density_regularization
        )
        la_density_loss.backward()
        lookahead_loss = float(la_dual_loss.detach().item() + la_density_loss.detach().item())

        # Manual lookahead step to θ_la
        with torch.no_grad():
            for p in all_eg_params:
                if p.grad is not None:
                    p.data -= step_size * p.grad.data

        # --- Final step: compute ∇L(θ_la) ---
        zero_grad_bundle(bundle)
        final_dual_terms = self._compute_dwmql_terms(obj_config, detach_density_weights_for_stationarity=True)
        final_dual_loss = -obj_config.stationarity_weight * final_dual_terms.stationarity_loss
        final_dual_loss.backward()
        final_density_terms = self._compute_dwmql_terms(obj_config, detach_dual_for_stationarity=True)
        final_density_loss = (
            obj_config.stationarity_weight * final_density_terms.stationarity_loss
            + obj_config.density_regularization_weight * final_density_terms.density_regularization
        )
        final_density_loss.backward()
        final_loss = float(final_dual_loss.detach().item() + final_density_loss.detach().item())

        # Restore original θ
        with torch.no_grad():
            for p, s in zip(all_eg_params, saved):
                p.data.copy_(s)

        # Apply ∇L(θ_la) starting from original θ
        with torch.no_grad():
            for p in all_eg_params:
                if p.grad is not None:
                    p.data -= step_size * p.grad.data

        return {
            "extragradient_used": True,
            "extragradient_groups": ["dual", "density"],
            "lookahead_loss": lookahead_loss,
            "final_loss": final_loss,
            "extragradient_step_size": step_size,
        }

    def train_dwmql_step(self, obj_config: DWMQLObjectiveConfig) -> DWMQLStepResult:
        """Run one DW-MQL minimax step: dual → density → critic.

        If ``obj_config.extragradient`` is True, the dual+density updates use
        the stochastic extragradient scheme (:meth:`_extragradient_dual_density_update`).
        Otherwise, standard alternating updates are applied.

        Applies ``obj_config.dual_steps``, ``obj_config.density_steps``, and
        ``obj_config.critic_steps`` inner repetitions per component.
        """

        validate_dwmql_objective_config(obj_config)
        if self.optimizer_bundle is None:
            raise ValueError("train_dwmql_step requires optimizer_bundle.")
        self.model.train()

        eg_diag: dict[str, Any] = {}
        if obj_config.extragradient:
            for _ in range(max(obj_config.dual_steps, obj_config.density_steps)):
                eg_diag = self._extragradient_dual_density_update(obj_config)
            dual_loss_val = eg_diag.get("final_loss", 0.0)
            density_loss_val = eg_diag.get("final_loss", 0.0)
        else:
            dual_result = self.update_dwmql_dual(obj_config)
            for _ in range(obj_config.dual_steps - 1):
                dual_result = self.update_dwmql_dual(obj_config)
            density_result = self.update_dwmql_density(obj_config)
            for _ in range(obj_config.density_steps - 1):
                density_result = self.update_dwmql_density(obj_config)
            dual_loss_val = float(dual_result.loss.item())
            density_loss_val = float(density_result.loss.item())

        critic_result = self.update_dwmql_critic(obj_config)
        for _ in range(obj_config.critic_steps - 1):
            critic_result = self.update_dwmql_critic(obj_config)

        # Sample diagnostics batch (detached, no grad)
        with torch.no_grad():
            diag_terms = self._compute_dwmql_terms(obj_config)

        return DWMQLStepResult(
            bellman_loss=float(diag_terms.bellman_loss.item()),
            stationarity_loss=float(diag_terms.stationarity_loss.item()),
            density_regularization=float(diag_terms.density_regularization.item()),
            ess=float(diag_terms.ess.item()),
            critic_loss=float(critic_result.loss.item()),
            density_loss=density_loss_val,
            dual_loss=dual_loss_val,
            extragradient_used=bool(obj_config.extragradient),
            extragradient_groups=["dual", "density"] if obj_config.extragradient else [],
            diagnostics={
                **diag_terms.diagnostics,
                **eg_diag,
                "critic_steps": int(obj_config.critic_steps),
                "density_steps": int(obj_config.density_steps),
                "dual_steps": int(obj_config.dual_steps),
                "dual_ascent": bool(obj_config.dual_ascent),
                "benchmark_claim": False,
            },
        )

    def train_dwmql_schedule(
        self,
        num_steps: int,
        obj_config: DWMQLObjectiveConfig,
    ) -> dict[str, Any]:
        """Run ``num_steps`` DW-MQL minimax steps and return a summary dict.

        Each step calls :meth:`train_dwmql_step`. Post-step bookkeeping (target
        updates, index refresh) is handled the same as for the scaffold schedule.
        """

        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        validate_dwmql_objective_config(obj_config)
        if self.optimizer_bundle is None:
            raise ValueError("train_dwmql_schedule requires optimizer_bundle.")

        results: list[DWMQLStepResult] = []
        for _ in range(num_steps):
            step_result = self.train_dwmql_step(obj_config)
            results.append(step_result)
            self._apply_post_update_bookkeeping(step_result.diagnostics)

        bellman_losses = [r.bellman_loss for r in results]
        stationarity_losses = [r.stationarity_loss for r in results]
        ess_values = [r.ess for r in results]

        def _mean(vals: list[float]) -> float | None:
            return sum(vals) / len(vals) if vals else None

        return {
            "num_steps": num_steps,
            "results": results,
            "bellman_losses": bellman_losses,
            "stationarity_losses": stationarity_losses,
            "ess_values": ess_values,
            "final_bellman_loss": bellman_losses[-1] if bellman_losses else None,
            "final_stationarity_loss": stationarity_losses[-1] if stationarity_losses else None,
            "final_density_regularization": results[-1].density_regularization if results else None,
            "final_ess": ess_values[-1] if ess_values else None,
            "mean_bellman_loss": _mean(bellman_losses),
            "mean_stationarity_loss": _mean(stationarity_losses),
            "mean_ess": _mean(ess_values),
            "extragradient_used": bool(obj_config.extragradient),
            "extragradient_groups": ["dual", "density"] if obj_config.extragradient else [],
            "index_refresh_count": self.index_refresh_count,
            "index_refresh_steps": self.index_refresh_steps,
            "index_ages": [r.diagnostics.get("index_age") for r in results],
            "target_update_count": self.target_update_count,
            "benchmark_claim": False,
            "diagnostics": {
                "final_step": self._step,
                "target_update_count": self.target_update_count,
                "persistent_index_used": self.use_persistent_index,
                "final_index_age": self._index_age,
            },
        }

    def _apply_post_update_bookkeeping(self, diagnostics: dict[str, Any]) -> None:
        """Apply target/index refresh bookkeeping after one optimizer update."""

        self._step += 1
        index_refreshed_this_step = self._step % self.index_refresh_every == 0
        if index_refreshed_this_step:
            self._index_refresh_count += 1
            self._index_refresh_steps.append(self._step)
            self._index_age = 0
        else:
            self._index_age += 1
        diagnostics.update(
            {
                "index_age": self._index_age,
                "index_refresh_every": self.index_refresh_every,
                "index_refresh_count": self.index_refresh_count,
                "index_refreshed_this_step": index_refreshed_this_step,
                "persistent_index_used": self.use_persistent_index,
                "audit_stale_index": self.audit_stale_index,
                "audit_encoder_policy_overlap": self.audit_encoder_policy_overlap,
                "index_encoder_policy": self.index_encoder_policy,
                "index_ema_tau": self.index_ema_tau,
                "index_ema_update_count": self.index_ema_update_count,
                "retrieval_correction_mode": self.retrieval_correction_mode,
                "persistent_index_last_refresh_step": None if self.index_cache is None else self.index_cache.last_refresh_step,
            }
        )
        if self.target_model is not None and self.target_tau is not None and self._step % self.target_update_every == 0:
            soft_update_target(self.target_model, self.model, float(self.target_tau))
            self._target_update_count += 1
            self._target_update_steps.append(self._step)
        if self.index_ema_model is not None:
            soft_update_target(self.index_ema_model, self.model, float(self.index_ema_tau))
            self._index_ema_update_count += 1
        if self.use_persistent_index:
            if index_refreshed_this_step:
                self._refresh_persistent_index()
            elif self.index_cache is not None:
                self.index_cache.age = self._index_age
                self.index_cache.refresh_count = self._index_refresh_count
            diagnostics["persistent_index_last_refresh_step"] = (
                None if self.index_cache is None else self.index_cache.last_refresh_step
            )
        drift = self._compute_index_encoder_drift()
        retrieval_overlap = self._compute_encoder_policy_retrieval_overlap()
        diagnostics.update(
            {
                "index_encoder_drift_mean_l2": None if drift is None else drift["mean_l2"],
                "index_encoder_drift_max_l2": None if drift is None else drift["max_l2"],
                "index_encoder_drift_mean_cosine_distance": None if drift is None else drift["mean_cosine_distance"],
                "index_encoder_drift_max_cosine_distance": None if drift is None else drift["max_cosine_distance"],
                "target_vs_online_recall_mean": retrieval_overlap.get("target_vs_online_recall_mean"),
                "target_vs_online_overlap_fraction_mean": retrieval_overlap.get("target_vs_online_overlap_fraction_mean"),
                "ema_vs_online_recall_mean": retrieval_overlap.get("ema_vs_online_recall_mean"),
                "ema_vs_online_overlap_fraction_mean": retrieval_overlap.get("ema_vs_online_overlap_fraction_mean"),
                "ema_vs_target_recall_mean": retrieval_overlap.get("ema_vs_target_recall_mean"),
                "ema_vs_target_overlap_fraction_mean": retrieval_overlap.get("ema_vs_target_overlap_fraction_mean"),
                "target_update_count": self.target_update_count,
                "index_ema_update_count": self.index_ema_update_count,
            }
        )

    def train_objective_schedule(
        self,
        num_steps: int,
        schedule: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Run a synthetic objective-specific schedule.

        The default schedule is dual, density, critic. This remains
        synthetic-only training scaffolding, not the finalized algorithm.
        """

        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        objective_schedule = tuple(schedule or ("dual", "density", "critic"))
        if not objective_schedule:
            raise ValueError("schedule must be non-empty.")
        results: list[ObjectiveUpdateResult] = []
        index_ages: list[int] = []
        stale_index_recall_means: list[float | None] = []
        stale_index_overlap_fraction_means: list[float | None] = []
        stale_index_best_value_gap_means: list[float | None] = []
        stale_index_pmr_means: list[float | None] = []
        stale_index_pmr_mins: list[float | None] = []
        stale_index_pmr_maxs: list[float | None] = []
        stale_index_pmr_normalized_means: list[float | None] = []
        stale_index_missing_mass_means: list[float | None] = []
        support_weighted_bellman_losses: list[float | None] = []
        support_weight_means: list[float | None] = []
        support_weight_mins: list[float | None] = []
        support_weight_maxs: list[float | None] = []
        q_abs_max_values: list[float | None] = []
        critic_total_losses: list[float | None] = []
        support_correction_losses: list[float | None] = []
        support_correction_contributions: list[float | None] = []
        support_correction_active_by_step: list[bool] = []
        density_ess_values: list[float | None] = []
        weight_entropy_values: list[float | None] = []
        index_encoder_drift_mean_l2: list[float | None] = []
        index_encoder_drift_max_l2: list[float | None] = []
        index_encoder_drift_mean_cosine_distance: list[float | None] = []
        index_encoder_drift_max_cosine_distance: list[float | None] = []
        target_vs_online_recall_means: list[float | None] = []
        target_vs_online_overlap_fraction_means: list[float | None] = []
        ema_vs_online_recall_means: list[float | None] = []
        ema_vs_online_overlap_fraction_means: list[float | None] = []
        ema_vs_target_recall_means: list[float | None] = []
        ema_vs_target_overlap_fraction_means: list[float | None] = []
        for step_index in range(num_steps):
            objective_name = objective_schedule[step_index % len(objective_schedule)]
            result = self.train_objective_step(objective_name)
            self._apply_post_update_bookkeeping(result.diagnostics)
            results.append(result)
            index_ages.append(int(result.diagnostics["index_age"]))
            stale_index_recall_means.append(result.diagnostics["stale_index_recall_mean"])
            stale_index_overlap_fraction_means.append(result.diagnostics["stale_index_overlap_fraction_mean"])
            stale_index_best_value_gap_means.append(result.diagnostics["stale_index_best_value_gap_mean"])
            stale_index_pmr_means.append(result.diagnostics["stale_index_pmr_mean"])
            stale_index_pmr_mins.append(result.diagnostics["stale_index_pmr_min"])
            stale_index_pmr_maxs.append(result.diagnostics["stale_index_pmr_max"])
            stale_index_pmr_normalized_means.append(result.diagnostics["stale_index_pmr_normalized_mean"])
            stale_index_missing_mass_means.append(result.diagnostics["stale_index_missing_mass_mean"])
            support_weighted_bellman_losses.append(result.diagnostics.get("support_weighted_bellman_loss"))
            support_weight_means.append(result.diagnostics.get("support_weight_mean"))
            support_weight_mins.append(result.diagnostics.get("support_weight_min"))
            support_weight_maxs.append(result.diagnostics.get("support_weight_max"))
            q_abs_max_values.append(result.diagnostics.get("q_abs_max"))
            critic_total_losses.append(result.diagnostics.get("critic_total_loss"))
            support_correction_losses.append(result.diagnostics.get("support_correction_loss"))
            support_correction_contributions.append(result.diagnostics.get("support_correction_contribution"))
            support_correction_active_by_step.append(bool(result.diagnostics.get("support_correction_active")))
            density_ess_values.append(result.diagnostics["density_ess"])
            weight_entropy_values.append(result.diagnostics["weight_entropy"])
            index_encoder_drift_mean_l2.append(result.diagnostics["index_encoder_drift_mean_l2"])
            index_encoder_drift_max_l2.append(result.diagnostics["index_encoder_drift_max_l2"])
            index_encoder_drift_mean_cosine_distance.append(
                result.diagnostics["index_encoder_drift_mean_cosine_distance"]
            )
            index_encoder_drift_max_cosine_distance.append(
                result.diagnostics["index_encoder_drift_max_cosine_distance"]
            )
            target_vs_online_recall_means.append(result.diagnostics["target_vs_online_recall_mean"])
            target_vs_online_overlap_fraction_means.append(
                result.diagnostics["target_vs_online_overlap_fraction_mean"]
            )
            ema_vs_online_recall_means.append(result.diagnostics["ema_vs_online_recall_mean"])
            ema_vs_online_overlap_fraction_means.append(
                result.diagnostics["ema_vs_online_overlap_fraction_mean"]
            )
            ema_vs_target_recall_means.append(result.diagnostics["ema_vs_target_recall_mean"])
            ema_vs_target_overlap_fraction_means.append(
                result.diagnostics["ema_vs_target_overlap_fraction_mean"]
            )
        losses = [float(result.loss.item()) for result in results]
        if torch.any(~torch.isfinite(torch.tensor(losses))):
            raise ValueError("Objective schedule losses must be finite.")
        critic_step_indices = [
            index for index, result in enumerate(results) if result.objective_name == "critic"
        ]

        def critic_values(values: list[float | None]) -> list[float | None]:
            return [values[index] for index in critic_step_indices]

        raw_bellman_losses = [
            result.diagnostics.get("raw_bellman_loss") for result in results
        ]
        critic_raw_bellman_losses = critic_values(raw_bellman_losses)
        critic_support_weighted_losses = critic_values(support_weighted_bellman_losses)
        critic_support_weight_means = critic_values(support_weight_means)
        critic_support_weight_mins = critic_values(support_weight_mins)
        critic_support_weight_maxs = critic_values(support_weight_maxs)
        critic_total_losses_only = critic_values(critic_total_losses)
        critic_support_correction_losses = critic_values(support_correction_losses)
        critic_support_correction_contributions = critic_values(
            support_correction_contributions
        )
        return {
            "num_steps": num_steps,
            "schedule": list(objective_schedule),
            "objective_names": [result.objective_name for result in results],
            "losses": losses,
            "results": results,
            "target_update_count": self.target_update_count,
            "target_update_steps": list(self._target_update_steps),
            "index_ages": index_ages,
            "index_refresh_count": self.index_refresh_count,
            "index_refresh_steps": list(self._index_refresh_steps),
            "stale_index_recall_means": stale_index_recall_means,
            "stale_index_overlap_fraction_means": stale_index_overlap_fraction_means,
            "stale_index_best_value_gap_means": stale_index_best_value_gap_means,
            "stale_index_pmr_means": stale_index_pmr_means,
            "stale_index_pmr_mins": stale_index_pmr_mins,
            "stale_index_pmr_maxs": stale_index_pmr_maxs,
            "stale_index_pmr_normalized_means": stale_index_pmr_normalized_means,
            "stale_index_missing_mass_means": stale_index_missing_mass_means,
            "support_weighted_bellman_losses": support_weighted_bellman_losses,
            "support_weight_means": support_weight_means,
            "support_weight_mins": support_weight_mins,
            "support_weight_maxs": support_weight_maxs,
            "q_abs_max_values": q_abs_max_values,
            "critic_total_losses": critic_total_losses,
            "support_correction_losses": support_correction_losses,
            "support_correction_contributions": support_correction_contributions,
            "support_correction_active_by_step": support_correction_active_by_step,
            "density_ess_values": density_ess_values,
            "weight_entropy_values": weight_entropy_values,
            "index_ema_update_count": self.index_ema_update_count,
            "index_encoder_drift_mean_l2": index_encoder_drift_mean_l2,
            "index_encoder_drift_max_l2": index_encoder_drift_max_l2,
            "index_encoder_drift_mean_cosine_distance": index_encoder_drift_mean_cosine_distance,
            "index_encoder_drift_max_cosine_distance": index_encoder_drift_max_cosine_distance,
            "target_vs_online_recall_means": target_vs_online_recall_means,
            "target_vs_online_overlap_fraction_means": target_vs_online_overlap_fraction_means,
            "ema_vs_online_recall_means": ema_vs_online_recall_means,
            "ema_vs_online_overlap_fraction_means": ema_vs_online_overlap_fraction_means,
            "ema_vs_target_recall_means": ema_vs_target_recall_means,
            "ema_vs_target_overlap_fraction_means": ema_vs_target_overlap_fraction_means,
            "diagnostics": {
                "synthetic_only": True,
                "objective_specific_updates": True,
                "num_steps": num_steps,
                "final_step": self.step,
                "target_update_count": self.target_update_count,
                "index_backend": self.index_backend,
                "index_metric": self.index_metric,
                "index_nlist": self.index_nlist,
                "index_nprobe": self.index_nprobe,
                "index_refresh_every": self.index_refresh_every,
                "index_refresh_count": self.index_refresh_count,
                "final_index_age": self.index_age,
                "persistent_index_used": self.use_persistent_index,
                "audit_stale_index": self.audit_stale_index,
                "audit_encoder_policy_overlap": self.audit_encoder_policy_overlap,
                "index_encoder_policy": self.index_encoder_policy,
                "retrieval_correction_mode": self.retrieval_correction_mode,
                "support_correction_weight": self.support_correction_weight,
                "support_correction_metric": self.support_correction_metric,
                "support_correction_target": self.support_correction_target,
                "support_correction_active": any(support_correction_active_by_step),
                "support_weighting_mode": self.support_weighting_mode,
                "support_weight_scale": self.support_weight_scale,
                "support_weight_min": self.support_weight_min,
                "support_weight_max": self.support_weight_max,
                "support_pmr_eps": self.support_pmr_eps,
                "final_raw_bellman_loss": next(
                    (value for value in reversed(critic_raw_bellman_losses) if value is not None),
                    None,
                ),
                "mean_raw_bellman_loss": _mean_optional(critic_raw_bellman_losses),
                "final_support_weighted_bellman_loss": next(
                    (value for value in reversed(critic_support_weighted_losses) if value is not None),
                    None,
                ),
                "mean_support_weighted_bellman_loss": _mean_optional(
                    critic_support_weighted_losses
                ),
                "final_support_weight_mean": next(
                    (value for value in reversed(critic_support_weight_means) if value is not None),
                    None,
                ),
                "mean_support_weight_mean": _mean_optional(critic_support_weight_means),
                "final_support_weight_min": next(
                    (value for value in reversed(critic_support_weight_mins) if value is not None),
                    None,
                ),
                "final_support_weight_max": next(
                    (value for value in reversed(critic_support_weight_maxs) if value is not None),
                    None,
                ),
                "final_critic_total_loss": next(
                    (value for value in reversed(critic_total_losses_only) if value is not None),
                    None,
                ),
                "mean_critic_total_loss": _mean_optional(critic_total_losses_only),
                "final_support_correction_loss": next(
                    (value for value in reversed(critic_support_correction_losses) if value is not None),
                    None,
                ),
                "mean_support_correction_loss": _mean_optional(
                    critic_support_correction_losses
                ),
                "final_support_correction_contribution": next(
                    (
                        value
                        for value in reversed(critic_support_correction_contributions)
                        if value is not None
                    ),
                    None,
                ),
                "mean_support_correction_contribution": _mean_optional(
                    critic_support_correction_contributions
                ),
                "index_ema_tau": self.index_ema_tau,
                "index_ema_update_count": self.index_ema_update_count,
                "final_index_encoder_drift_mean_l2": index_encoder_drift_mean_l2[-1],
                "final_index_encoder_drift_max_l2": index_encoder_drift_max_l2[-1],
                "final_index_encoder_drift_mean_cosine_distance": index_encoder_drift_mean_cosine_distance[-1],
                "final_index_encoder_drift_max_cosine_distance": index_encoder_drift_max_cosine_distance[-1],
                "final_target_vs_online_recall_mean": target_vs_online_recall_means[-1],
                "final_target_vs_online_overlap_fraction_mean": target_vs_online_overlap_fraction_means[-1],
                "final_ema_vs_online_recall_mean": ema_vs_online_recall_means[-1],
                "final_ema_vs_online_overlap_fraction_mean": ema_vs_online_overlap_fraction_means[-1],
                "final_ema_vs_target_recall_mean": ema_vs_target_recall_means[-1],
                "final_ema_vs_target_overlap_fraction_mean": ema_vs_target_overlap_fraction_means[-1],
                "final_stale_index_pmr_mean": stale_index_pmr_means[-1],
                "final_stale_index_pmr_min": stale_index_pmr_mins[-1],
                "final_stale_index_pmr_max": stale_index_pmr_maxs[-1],
                "final_stale_index_pmr_normalized_mean": stale_index_pmr_normalized_means[-1],
                "final_stale_index_missing_mass_mean": stale_index_missing_mass_means[-1],
                "final_stale_index_uniform_pmr_baseline": uniform_policy_mass_baseline(self.k, int(self.data.action_features.shape[0])),
                "final_density_ess": density_ess_values[-1],
                "final_weight_entropy": weight_entropy_values[-1],
                "persistent_index_last_refresh_step": None if self.index_cache is None else self.index_cache.last_refresh_step,
            },
        }

    def train_step(
        self,
        update_parts: list[str] | tuple[str, ...] | None = None,
        loss_config: LossComponentConfig | None = None,
    ) -> TrainStepResult:
        """Run one synthetic optimization step and optional target update."""

        result = run_minimal_train_step(
            data=self.data,
            model=self.model,
            optimizer=self.optimizer,
            optimizer_bundle=self.optimizer_bundle,
            update_parts=update_parts,
            batch_size=self.batch_size,
            k=self.k,
            gamma=self.gamma,
            generator=self.generator,
            temperature=self.temperature,
            density_max_weight=self.density_max_weight,
            stationarity_weight=self.stationarity_weight,
            density_reg_weight=self.density_reg_weight,
            detach_bellman_target=self.detach_bellman_target,
            target_model=self.target_model,
            loss_config=loss_config if loss_config is not None else self.loss_config,
            split_backward=self.split_backward,
            index_backend=self.index_backend,
            index_metric=self.index_metric,
            index_nlist=self.index_nlist,
            index_nprobe=self.index_nprobe,
            action_index=None if self.index_cache is None else self.index_cache.index,
            audit_stale_index=self.audit_stale_index,
            index_encoder_policy=self.index_encoder_policy,
            index_ema_tau=self.index_ema_tau,
            retrieval_correction_mode=self.retrieval_correction_mode,
            support_correction_weight=self.support_correction_weight,
            support_correction_target=self.support_correction_target,
            support_correction_metric=self.support_correction_metric,
            support_weighting_mode=self.support_weighting_mode,
            support_weight_scale=self.support_weight_scale,
            support_weight_min=self.support_weight_min,
            support_weight_max=self.support_weight_max,
            support_pmr_eps=self.support_pmr_eps,
        )
        self._apply_post_update_bookkeeping(result.diagnostics)
        return result

    def train(
        self,
        num_steps: int,
        update_schedule: list[Any] | tuple[Any, ...] | None = None,
    ) -> TrainingHistory:
        """Run repeated synthetic optimization and return scalar history."""

        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")

        losses: list[float] = []
        bellman_losses: list[float] = []
        stationarity_losses: list[float] = []
        density_regularizations: list[float] = []
        grad_norms: list[float] = []
        parameter_delta_norms: list[float] = []
        active_losses_by_step: list[list[str]] = []
        update_parts_by_step: list[list[str] | None] = []
        index_ages: list[int] = []
        stale_index_recall_means: list[float | None] = []
        stale_index_overlap_fraction_means: list[float | None] = []
        stale_index_best_value_gap_means: list[float | None] = []
        stale_index_pmr_means: list[float | None] = []
        stale_index_pmr_mins: list[float | None] = []
        stale_index_pmr_maxs: list[float | None] = []
        stale_index_pmr_normalized_means: list[float | None] = []
        stale_index_missing_mass_means: list[float | None] = []
        support_weighted_bellman_losses: list[float | None] = []
        support_weight_means: list[float | None] = []
        support_weight_mins: list[float | None] = []
        support_weight_maxs: list[float | None] = []
        q_abs_max_values: list[float | None] = []
        critic_total_losses: list[float | None] = []
        support_correction_losses: list[float | None] = []
        support_correction_contributions: list[float | None] = []
        support_correction_active_by_step: list[bool] = []
        density_ess_values: list[float | None] = []
        weight_entropy_values: list[float | None] = []
        index_encoder_drift_mean_l2: list[float | None] = []
        index_encoder_drift_max_l2: list[float | None] = []
        index_encoder_drift_mean_cosine_distance: list[float | None] = []
        index_encoder_drift_max_cosine_distance: list[float | None] = []
        target_vs_online_recall_means: list[float | None] = []
        target_vs_online_overlap_fraction_means: list[float | None] = []
        ema_vs_online_recall_means: list[float | None] = []
        ema_vs_online_overlap_fraction_means: list[float | None] = []
        ema_vs_target_recall_means: list[float | None] = []
        ema_vs_target_overlap_fraction_means: list[float | None] = []

        for index in range(num_steps):
            if update_schedule is None:
                update_parts = None
                step_loss_config = None
            else:
                update_parts, step_loss_config = parse_update_schedule_step(
                    update_schedule[index % len(update_schedule)]
                )
            result = self.train_step(update_parts=update_parts, loss_config=step_loss_config)

            scalars = {
                "loss": float(result.total_loss.item()),
                "bellman_loss": float(result.bellman_loss.item()),
                "stationarity_loss": float(result.stationarity_loss.item()),
                "density_regularization": float(result.density_regularization.item()),
                "grad_norm": float(result.grad_norm.item()),
                "parameter_delta_norm": float(result.parameter_delta_norm.item()),
            }
            if torch.any(~torch.isfinite(torch.tensor(list(scalars.values())))):
                raise ValueError("Training history scalars must be finite.")

            losses.append(scalars["loss"])
            bellman_losses.append(scalars["bellman_loss"])
            stationarity_losses.append(scalars["stationarity_loss"])
            density_regularizations.append(scalars["density_regularization"])
            grad_norms.append(scalars["grad_norm"])
            parameter_delta_norms.append(scalars["parameter_delta_norm"])
            active_losses_by_step.append(list(result.diagnostics["active_losses"]))
            update_parts_by_step.append(None if update_parts is None else list(update_parts))
            index_ages.append(int(result.diagnostics["index_age"]))
            stale_index_recall_means.append(result.diagnostics["stale_index_recall_mean"])
            stale_index_overlap_fraction_means.append(result.diagnostics["stale_index_overlap_fraction_mean"])
            stale_index_best_value_gap_means.append(result.diagnostics["stale_index_best_value_gap_mean"])
            stale_index_pmr_means.append(result.diagnostics["stale_index_pmr_mean"])
            stale_index_pmr_mins.append(result.diagnostics["stale_index_pmr_min"])
            stale_index_pmr_maxs.append(result.diagnostics["stale_index_pmr_max"])
            stale_index_pmr_normalized_means.append(result.diagnostics["stale_index_pmr_normalized_mean"])
            stale_index_missing_mass_means.append(result.diagnostics["stale_index_missing_mass_mean"])
            support_weighted_bellman_losses.append(result.diagnostics.get("support_weighted_bellman_loss"))
            support_weight_means.append(result.diagnostics.get("support_weight_mean"))
            support_weight_mins.append(result.diagnostics.get("support_weight_min"))
            support_weight_maxs.append(result.diagnostics.get("support_weight_max"))
            q_abs_max_values.append(result.diagnostics.get("q_abs_max"))
            critic_total_losses.append(result.diagnostics.get("critic_total_loss"))
            support_correction_losses.append(result.diagnostics.get("support_correction_loss"))
            support_correction_contributions.append(result.diagnostics.get("support_correction_contribution"))
            support_correction_active_by_step.append(bool(result.diagnostics.get("support_correction_active")))
            density_ess_values.append(result.diagnostics["density_ess"])
            weight_entropy_values.append(result.diagnostics["weight_entropy"])
            index_encoder_drift_mean_l2.append(result.diagnostics["index_encoder_drift_mean_l2"])
            index_encoder_drift_max_l2.append(result.diagnostics["index_encoder_drift_max_l2"])
            index_encoder_drift_mean_cosine_distance.append(
                result.diagnostics["index_encoder_drift_mean_cosine_distance"]
            )
            index_encoder_drift_max_cosine_distance.append(
                result.diagnostics["index_encoder_drift_max_cosine_distance"]
            )
            target_vs_online_recall_means.append(result.diagnostics["target_vs_online_recall_mean"])
            target_vs_online_overlap_fraction_means.append(
                result.diagnostics["target_vs_online_overlap_fraction_mean"]
            )
            ema_vs_online_recall_means.append(result.diagnostics["ema_vs_online_recall_mean"])
            ema_vs_online_overlap_fraction_means.append(
                result.diagnostics["ema_vs_online_overlap_fraction_mean"]
            )
            ema_vs_target_recall_means.append(result.diagnostics["ema_vs_target_recall_mean"])
            ema_vs_target_overlap_fraction_means.append(
                result.diagnostics["ema_vs_target_overlap_fraction_mean"]
            )

        return TrainingHistory(
            losses=losses,
            bellman_losses=bellman_losses,
            stationarity_losses=stationarity_losses,
            density_regularizations=density_regularizations,
            grad_norms=grad_norms,
            parameter_delta_norms=parameter_delta_norms,
            target_update_steps=list(self._target_update_steps),
            index_ages=index_ages,
            index_refresh_steps=list(self._index_refresh_steps),
            index_refresh_count=self.index_refresh_count,
            stale_index_recall_means=stale_index_recall_means,
            stale_index_overlap_fraction_means=stale_index_overlap_fraction_means,
            stale_index_best_value_gap_means=stale_index_best_value_gap_means,
            stale_index_pmr_means=stale_index_pmr_means,
            stale_index_pmr_mins=stale_index_pmr_mins,
            stale_index_pmr_maxs=stale_index_pmr_maxs,
            stale_index_pmr_normalized_means=stale_index_pmr_normalized_means,
            stale_index_missing_mass_means=stale_index_missing_mass_means,
            support_weighted_bellman_losses=support_weighted_bellman_losses,
            support_weight_means=support_weight_means,
            support_weight_mins=support_weight_mins,
            support_weight_maxs=support_weight_maxs,
            q_abs_max_values=q_abs_max_values,
            critic_total_losses=critic_total_losses,
            support_correction_losses=support_correction_losses,
            support_correction_contributions=support_correction_contributions,
            density_ess_values=density_ess_values,
            weight_entropy_values=weight_entropy_values,
            index_ema_update_count=self.index_ema_update_count,
            index_encoder_drift_mean_l2=index_encoder_drift_mean_l2,
            index_encoder_drift_max_l2=index_encoder_drift_max_l2,
            index_encoder_drift_mean_cosine_distance=index_encoder_drift_mean_cosine_distance,
            index_encoder_drift_max_cosine_distance=index_encoder_drift_max_cosine_distance,
            target_vs_online_recall_means=target_vs_online_recall_means,
            target_vs_online_overlap_fraction_means=target_vs_online_overlap_fraction_means,
            ema_vs_online_recall_means=ema_vs_online_recall_means,
            ema_vs_online_overlap_fraction_means=ema_vs_online_overlap_fraction_means,
            ema_vs_target_recall_means=ema_vs_target_recall_means,
            ema_vs_target_overlap_fraction_means=ema_vs_target_overlap_fraction_means,
            diagnostics={
                "num_steps": num_steps,
                "final_step": self.step,
                "target_update_count": self.target_update_count,
                "split_backward": self.split_backward,
                "index_backend": self.index_backend,
                "index_metric": self.index_metric,
                "index_nlist": self.index_nlist,
                "index_nprobe": self.index_nprobe,
                "index_refresh_every": self.index_refresh_every,
                "index_refresh_count": self.index_refresh_count,
                "final_index_age": self.index_age,
                "persistent_index_used": self.use_persistent_index,
                "audit_stale_index": self.audit_stale_index,
                "audit_encoder_policy_overlap": self.audit_encoder_policy_overlap,
                "index_encoder_policy": self.index_encoder_policy,
                "retrieval_correction_mode": self.retrieval_correction_mode,
                "support_correction_weight": self.support_correction_weight,
                "support_correction_metric": self.support_correction_metric,
                "support_correction_target": self.support_correction_target,
                "support_correction_active": any(support_correction_active_by_step),
                "support_weighting_mode": self.support_weighting_mode,
                "support_weight_scale": self.support_weight_scale,
                "support_weight_min": self.support_weight_min,
                "support_weight_max": self.support_weight_max,
                "support_pmr_eps": self.support_pmr_eps,
                "support_correction_losses": support_correction_losses,
                "support_correction_contributions": support_correction_contributions,
                "support_correction_active_by_step": support_correction_active_by_step,
                "final_raw_bellman_loss": bellman_losses[-1],
                "mean_raw_bellman_loss": _mean_optional(bellman_losses),
                "final_support_weighted_bellman_loss": support_weighted_bellman_losses[-1],
                "mean_support_weighted_bellman_loss": _mean_optional(support_weighted_bellman_losses),
                "final_support_weight_mean": support_weight_means[-1],
                "mean_support_weight_mean": _mean_optional(support_weight_means),
                "final_support_weight_min": support_weight_mins[-1],
                "final_support_weight_max": support_weight_maxs[-1],
                "final_critic_total_loss": critic_total_losses[-1],
                "mean_critic_total_loss": _mean_optional(critic_total_losses),
                "final_support_correction_loss": support_correction_losses[-1],
                "mean_support_correction_loss": _mean_optional(support_correction_losses),
                "final_support_correction_contribution": support_correction_contributions[-1],
                "mean_support_correction_contribution": _mean_optional(support_correction_contributions),
                "index_ema_tau": self.index_ema_tau,
                "index_ema_update_count": self.index_ema_update_count,
                "final_index_encoder_drift_mean_l2": index_encoder_drift_mean_l2[-1],
                "final_index_encoder_drift_max_l2": index_encoder_drift_max_l2[-1],
                "final_index_encoder_drift_mean_cosine_distance": index_encoder_drift_mean_cosine_distance[-1],
                "final_index_encoder_drift_max_cosine_distance": index_encoder_drift_max_cosine_distance[-1],
                "final_target_vs_online_recall_mean": target_vs_online_recall_means[-1],
                "final_target_vs_online_overlap_fraction_mean": target_vs_online_overlap_fraction_means[-1],
                "final_ema_vs_online_recall_mean": ema_vs_online_recall_means[-1],
                "final_ema_vs_online_overlap_fraction_mean": ema_vs_online_overlap_fraction_means[-1],
                "final_ema_vs_target_recall_mean": ema_vs_target_recall_means[-1],
                "final_ema_vs_target_overlap_fraction_mean": ema_vs_target_overlap_fraction_means[-1],
                "final_stale_index_pmr_mean": stale_index_pmr_means[-1],
                "final_stale_index_pmr_min": stale_index_pmr_mins[-1],
                "final_stale_index_pmr_max": stale_index_pmr_maxs[-1],
                "final_stale_index_pmr_normalized_mean": stale_index_pmr_normalized_means[-1],
                "final_stale_index_missing_mass_mean": stale_index_missing_mass_means[-1],
                "final_stale_index_uniform_pmr_baseline": uniform_policy_mass_baseline(self.k, int(self.data.action_features.shape[0])),
                "final_density_ess": density_ess_values[-1],
                "final_weight_entropy": weight_entropy_values[-1],
                "persistent_index_last_refresh_step": None if self.index_cache is None else self.index_cache.last_refresh_step,
                "active_losses_by_step": active_losses_by_step,
                "update_parts_by_step": update_parts_by_step,
                "synthetic_only": True,
            },
        )
