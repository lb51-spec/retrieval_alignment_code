"""Synthetic evaluation schema for benchmark scaffold reports.

This module runs repeated forward-only pipeline diagnostics over synthetic
offline data. It is synthetic-only and claim-free: the outputs are numerical
sanity and retrieval-support diagnostics, not benchmark performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - dependency guard.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .pipeline import run_pipeline_smoke_step


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for synthetic forward-only evaluation diagnostics."""

    num_batches: int = 4
    batch_size: int | None = None
    k: int | None = None
    temperature: float = 1.0
    index_backend: str | None = None
    index_metric: str | None = None
    audit_exact_backend: bool = True


@dataclass(slots=True)
class EvaluationSummary:
    """Aggregated synthetic evaluation diagnostics."""

    task_name: str
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    num_batches: int = 0
    synthetic_only: bool = True
    benchmark_claim: bool = False


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for synthetic evaluation.")


def validate_evaluation_config(config: EvaluationConfig) -> EvaluationConfig:
    """Validate synthetic evaluation config values."""

    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig.")
    if not isinstance(config.num_batches, int) or config.num_batches <= 0:
        raise ValueError("num_batches must be a positive integer.")
    if config.batch_size is not None and (not isinstance(config.batch_size, int) or config.batch_size <= 0):
        raise ValueError("batch_size must be None or a positive integer.")
    if config.k is not None and (not isinstance(config.k, int) or config.k <= 0):
        raise ValueError("k must be None or a positive integer.")
    if not isinstance(config.temperature, int | float) or float(config.temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if config.index_backend is not None and config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be None or one of 'exact', 'faiss', 'faiss_ivf', or 'faiss_hnsw'.")
    if config.index_metric is not None and config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be None or either 'dot' or 'cosine'.")
    if not isinstance(config.audit_exact_backend, bool):
        raise TypeError("audit_exact_backend must be a bool.")
    return config


def _config_value(source: Any, name: str, default: Any = None) -> Any:
    return getattr(source, name, default) if source is not None else default


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError("Evaluation metrics must be finite.")
    return float(tensor.mean().item())


def evaluate_synthetic_model(
    data: Any,
    model: Any,
    config: EvaluationConfig,
    base_profile_or_experiment_config: Any,
    task_name: str = "synthetic_debug",
) -> EvaluationSummary:
    """Run repeated synthetic forward-only diagnostics and aggregate metrics."""

    _require_torch()
    config = validate_evaluation_config(config)
    batch_size = config.batch_size or int(_config_value(base_profile_or_experiment_config, "batch_size", 8))
    k = config.k or int(_config_value(base_profile_or_experiment_config, "k", 5))
    gamma = float(_config_value(base_profile_or_experiment_config, "gamma", 0.99))
    index_backend = config.index_backend or str(_config_value(base_profile_or_experiment_config, "index_backend", "exact"))
    index_metric = config.index_metric or str(_config_value(base_profile_or_experiment_config, "index_metric", "dot"))
    index_nlist = _config_value(base_profile_or_experiment_config, "index_nlist", None)
    index_nprobe = _config_value(base_profile_or_experiment_config, "index_nprobe", None)
    if hasattr(base_profile_or_experiment_config, "ann"):
        index_nlist = getattr(base_profile_or_experiment_config.ann, "nlist", index_nlist)
        index_nprobe = getattr(base_profile_or_experiment_config.ann, "nprobe", index_nprobe)
    seed = int(_config_value(base_profile_or_experiment_config, "seed", 0))
    generator = torch.Generator().manual_seed(seed + 10_000)

    collected: dict[str, list[float]] = {
        "bellman_loss": [],
        "stationarity_loss": [],
        "ess": [],
        "recall_at_k": [],
        "exact_backend_recall": [],
        "exact_backend_overlap_fraction": [],
        "exact_backend_best_value_gap": [],
    }
    widths: list[int] = []
    for _ in range(config.num_batches):
        result = run_pipeline_smoke_step(
            data=data,
            model=model,
            batch_size=batch_size,
            k=k,
            gamma=gamma,
            generator=generator,
            temperature=float(config.temperature),
            index_backend=index_backend,
            index_metric=index_metric,
            index_nlist=index_nlist,
            index_nprobe=index_nprobe,
            audit_exact_backend=config.audit_exact_backend,
        )
        collected["bellman_loss"].append(float(result.bellman_loss.detach().item()))
        collected["stationarity_loss"].append(float(result.stationarity_loss.detach().item()))
        collected["ess"].append(float(result.ess.detach().item()))
        collected["recall_at_k"].append(float(result.recall_at_k.float().mean().item()))
        if result.exact_backend_recall_at_k is not None:
            collected["exact_backend_recall"].append(float(result.exact_backend_recall_at_k.float().mean().item()))
        if result.exact_backend_overlap_fraction is not None:
            collected["exact_backend_overlap_fraction"].append(
                float(result.exact_backend_overlap_fraction.float().mean().item())
            )
        if result.exact_backend_best_value_gap is not None:
            collected["exact_backend_best_value_gap"].append(float(result.exact_backend_best_value_gap.mean().item()))
        widths.append(int(result.num_candidates))

    metrics = {
        "mean_bellman_loss": _mean(collected["bellman_loss"]),
        "mean_stationarity_loss": _mean(collected["stationarity_loss"]),
        "mean_ess": _mean(collected["ess"]),
        "mean_recall_at_k": _mean(collected["recall_at_k"]),
        "mean_exact_backend_recall": _mean(collected["exact_backend_recall"]),
        "mean_exact_backend_overlap_fraction": _mean(collected["exact_backend_overlap_fraction"]),
        "mean_exact_backend_best_value_gap": _mean(collected["exact_backend_best_value_gap"]),
    }
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return EvaluationSummary(
        task_name=task_name,
        metrics=metrics,
        diagnostics={
            "synthetic_only": True,
            "benchmark_claim": False,
            "evaluation_loop": "synthetic_forward_pipeline",
            "batch_size": batch_size,
            "k": k,
            "gamma": gamma,
            "temperature": float(config.temperature),
            "index_backend": index_backend,
            "index_metric": index_metric,
            "audit_exact_backend": config.audit_exact_backend,
            "mean_candidate_width": _mean([float(width) for width in widths]),
        },
        num_batches=config.num_batches,
        synthetic_only=True,
        benchmark_claim=False,
    )
