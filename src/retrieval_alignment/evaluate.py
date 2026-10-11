"""Synthetic smoke evaluation helpers for DW-MQL wiring checks.

This module intentionally provides only before/after synthetic smoke diagnostics.
It is not benchmark evaluation, does not split datasets, and makes no
performance claims. The helper exists to verify that forward-only smoke metrics
can be collected around the tiny synthetic smoke-training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]

from .config import ExperimentConfig
from .networks import DWMQLNetworks
from .pipeline import PipelineSmokeResult, run_pipeline_smoke_step
from .synthetic import SyntheticOfflineData
from .trainer import (
    OptimizerBundle,
    TrainLoopSmokeResult,
    make_optimizer_bundle,
    run_minimal_alternating_loop,
    run_minimal_train_loop,
)


@dataclass(slots=True)
class BeforeAfterSmokeResult:
    """Container for before/train/after synthetic smoke diagnostics."""

    before: PipelineSmokeResult
    train_loop: TrainLoopSmokeResult
    after: PipelineSmokeResult
    deltas: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run smoke evaluation.")


def _validate_inputs(data: SyntheticOfflineData, num_steps: int, batch_size: int, k: int, gamma: float) -> None:
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > int(data.action_features.shape[0]):
        raise ValueError("k must not exceed the number of actions.")
    if not isinstance(gamma, int | float) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")


def _scalar(value: Any, *, name: str) -> float:
    tensor = torch.as_tensor(value).detach()
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite.")
    return float(tensor.item())


def _mean_scalar(value: Any, *, name: str) -> float:
    tensor = torch.as_tensor(value).detach()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    mean = tensor.float().mean()
    if torch.any(~torch.isfinite(mean)):
        raise ValueError(f"{name} mean must be finite.")
    return float(mean.item())


def _smoke_scalars(result: PipelineSmokeResult, *, prefix: str) -> dict[str, float]:
    diagnostics = result.diagnostics
    return {
        f"{prefix}_bellman_loss": _scalar(result.bellman_loss, name=f"{prefix}.bellman_loss"),
        f"{prefix}_stationarity_loss": _scalar(result.stationarity_loss, name=f"{prefix}.stationarity_loss"),
        f"{prefix}_ess": _scalar(result.ess, name=f"{prefix}.ess"),
        f"{prefix}_mean_recall": _mean_scalar(result.recall_at_k, name=f"{prefix}.recall_at_k"),
        f"{prefix}_exact_backend_recall_mean": float(diagnostics["exact_backend_recall_mean"]),
        f"{prefix}_exact_backend_overlap_fraction_mean": float(
            diagnostics["exact_backend_overlap_fraction_mean"]
        ),
        f"{prefix}_exact_backend_best_value_gap_mean": float(
            diagnostics["exact_backend_best_value_gap_mean"]
        ),
    }


def _optimizer_bundle_from_optimizer(model: DWMQLNetworks, optimizer: Any) -> OptimizerBundle:
    if isinstance(optimizer, OptimizerBundle):
        return optimizer
    if not hasattr(optimizer, "param_groups") or not optimizer.param_groups:
        raise TypeError("optimizer must be a torch optimizer or OptimizerBundle.")
    lr = float(optimizer.param_groups[0].get("lr", 3e-4))
    weight_decay = float(optimizer.param_groups[0].get("weight_decay", 0.0))
    return make_optimizer_bundle(model, lr=lr, weight_decay=weight_decay, mode="separate_heads")


def run_before_after_smoke_evaluation(
    data: SyntheticOfflineData,
    model: DWMQLNetworks,
    optimizer: "torch.optim.Optimizer | OptimizerBundle | Any",
    num_steps: int,
    batch_size: int,
    k: int,
    gamma: float,
    generator: "torch.Generator | None" = None,
    target_model: DWMQLNetworks | None = None,
    target_tau: float | None = None,
    target_update_every: int = 1,
    alternating_schedule: list[Any] | tuple[Any, ...] | None = None,
    assert_gradient_flow: bool = False,
    temperature: float = 1.0,
    density_max_weight: float | None = None,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> BeforeAfterSmokeResult:
    """Run forward-only smoke diagnostics before and after synthetic smoke training.

    No loss decrease is expected or asserted. The returned deltas are simple
    scalar differences in existing smoke diagnostics, useful only for numerical
    and wiring sanity checks.
    """

    _require_torch()
    _validate_inputs(data, num_steps, batch_size, k, gamma)

    before = run_pipeline_smoke_step(
        data=data,
        model=model,
        batch_size=batch_size,
        k=k,
        gamma=gamma,
        generator=generator,
        temperature=temperature,
        density_max_weight=density_max_weight,
        index_backend=index_backend,
        index_metric=index_metric,
        index_nlist=index_nlist,
        index_nprobe=index_nprobe,
    )

    if alternating_schedule is None:
        train_loop = run_minimal_train_loop(
            data=data,
            model=model,
            optimizer=optimizer,
            num_steps=num_steps,
            batch_size=batch_size,
            k=k,
            gamma=gamma,
            generator=generator,
            target_model=target_model,
            target_tau=target_tau,
            target_update_every=target_update_every,
            temperature=temperature,
            density_max_weight=density_max_weight,
            index_backend=index_backend,
            index_metric=index_metric,
            index_nlist=index_nlist,
            index_nprobe=index_nprobe,
        )
    else:
        train_loop = run_minimal_alternating_loop(
            data=data,
            model=model,
            optimizer_bundle=_optimizer_bundle_from_optimizer(model, optimizer),
            num_steps=num_steps,
            batch_size=batch_size,
            k=k,
            gamma=gamma,
            update_schedule=alternating_schedule,
            generator=generator,
            target_model=target_model,
            target_tau=target_tau,
            target_update_every=target_update_every,
            temperature=temperature,
            density_max_weight=density_max_weight,
            assert_gradient_flow=assert_gradient_flow,
            index_backend=index_backend,
            index_metric=index_metric,
            index_nlist=index_nlist,
            index_nprobe=index_nprobe,
        )

    after = run_pipeline_smoke_step(
        data=data,
        model=model,
        batch_size=batch_size,
        k=k,
        gamma=gamma,
        generator=generator,
        temperature=temperature,
        density_max_weight=density_max_weight,
        index_backend=index_backend,
        index_metric=index_metric,
        index_nlist=index_nlist,
        index_nprobe=index_nprobe,
    )

    before_scalars = _smoke_scalars(before, prefix="before")
    after_scalars = _smoke_scalars(after, prefix="after")
    deltas = {
        "bellman_loss_delta": after_scalars["after_bellman_loss"] - before_scalars["before_bellman_loss"],
        "stationarity_loss_delta": after_scalars["after_stationarity_loss"] - before_scalars["before_stationarity_loss"],
        "ess_delta": after_scalars["after_ess"] - before_scalars["before_ess"],
        "mean_recall_delta": after_scalars["after_mean_recall"] - before_scalars["before_mean_recall"],
        "exact_backend_recall_mean_delta": (
            after_scalars["after_exact_backend_recall_mean"]
            - before_scalars["before_exact_backend_recall_mean"]
        ),
        "exact_backend_overlap_fraction_mean_delta": (
            after_scalars["after_exact_backend_overlap_fraction_mean"]
            - before_scalars["before_exact_backend_overlap_fraction_mean"]
        ),
        "exact_backend_best_value_gap_mean_delta": (
            after_scalars["after_exact_backend_best_value_gap_mean"]
            - before_scalars["before_exact_backend_best_value_gap_mean"]
        ),
    }
    for name, value in deltas.items():
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{name} must be finite.")

    return BeforeAfterSmokeResult(
        before=before,
        train_loop=train_loop,
        after=after,
        deltas=deltas,
        diagnostics={
            "synthetic_smoke_only": True,
            "benchmark_evaluation": False,
            "alternating_schedule_used": alternating_schedule is not None,
            "index_backend": index_backend,
            "index_metric": index_metric,
            "index_nlist": index_nlist,
            "index_nprobe": index_nprobe,
            **before_scalars,
            **after_scalars,
        },
    )


def evaluate(config: ExperimentConfig | None = None) -> None:
    """Placeholder for future benchmark evaluation."""

    _ = config or ExperimentConfig()
    raise NotImplementedError("Benchmark evaluation is intentionally unimplemented. Use smoke helpers only.")
