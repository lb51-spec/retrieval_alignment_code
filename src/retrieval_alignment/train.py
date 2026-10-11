"""Sanity-check entry points for DW-MQL+ANN.

Real training is intentionally not implemented yet. This module provides
synthetic smoke, diagnostics, and scaffold runners. Only the synthetic
experiment scaffold has minimal checkpoint support; none of these paths perform
benchmark evaluation. Optional FAISS flat-index retrieval can be selected
explicitly when FAISS is installed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]

from .config import (
    DWMQLObjectiveConfig,
    ExperimentConfig,
    ModelConfig,
    SmokeProfile,
    build_smoke_components,
    default_smoke_profile,
    load_d4rl_optional_protocol_config,
    load_experiment_config_yaml,
    load_kuairand_chrono_protocol_config,
    load_movielens_chrono_protocol_config,
    load_recsim_optional_protocol_config,
    smoke_profile_from_args,
    synthetic_config_from_smoke_profile,
    validate_d4rl_optional_protocol_config,
    validate_dwmql_objective_config,
    validate_experiment_config,
    validate_movielens_chrono_protocol_config,
    validate_recsim_optional_protocol_config,
)
from .evaluate import BeforeAfterSmokeResult, run_before_after_smoke_evaluation
from .networks import build_networks
from .pipeline import PipelineSmokeResult, run_pipeline_smoke_step
from .synthetic import make_synthetic_offline_data
from .trainer import (
    DWMQLStepResult,
    ObjectiveUpdateResult,
    SyntheticDWMQLTrainer,
    TrainLoopSmokeResult,
    TrainingHistory,
    default_smoke_alternating_schedule,
    make_optimizer_bundle,
    run_minimal_alternating_loop,
    run_minimal_train_loop,
    training_history_to_dict,
    write_json_report,
)


_GROUP_NAMES = ("encoder", "critic", "density", "dual")
_OBJECTIVE_NAMES = ("critic", "density", "dual")
DIAGNOSTICS_SCHEMA_NAME = "dwmql_synthetic_diagnostics"
DIAGNOSTICS_SCHEMA_VERSION = 1


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run smoke checks.")


def run_smoke_from_config(config: ExperimentConfig | None = None) -> PipelineSmokeResult:
    """Run one forward-only pipeline smoke step from config."""

    _require_torch()
    config = config or ExperimentConfig()
    if config.algorithm.candidate_k <= 0:
        raise ValueError("candidate_k must be positive.")
    if not 0.0 <= config.algorithm.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if config.ann.backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("ann.backend must be one of 'exact', 'faiss', 'faiss_ivf', or 'faiss_hnsw'.")
    if config.ann.metric not in {"dot", "cosine"}:
        raise ValueError("ann.metric must be either 'dot' or 'cosine'.")
    for name, value in {"ann.nlist": config.ann.nlist, "ann.nprobe": config.ann.nprobe}.items():
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be None or a positive integer.")

    data = make_synthetic_offline_data(config.synthetic)
    torch.manual_seed(config.seed)
    model = build_networks(config.model)
    generator = torch.Generator().manual_seed(config.seed)
    return run_pipeline_smoke_step(
        data=data,
        model=model,
        batch_size=min(config.synthetic.num_transitions, max(1, int(config.algorithm.candidate_k))),
        k=config.algorithm.candidate_k,
        gamma=config.algorithm.gamma,
        generator=generator,
        density_max_weight=config.density.max_weight,
        index_backend=config.ann.backend,
        index_metric=config.ann.metric,
        index_nlist=config.ann.nlist,
        index_nprobe=config.ann.nprobe,
    )


def _config_from_profile(profile: SmokeProfile) -> ExperimentConfig:
    config = ExperimentConfig(
        model=ModelConfig(
            state_dim=profile.state_dim,
            action_dim=profile.action_dim,
            embedding_dim=profile.embedding_dim,
        ),
        synthetic=synthetic_config_from_smoke_profile(profile),
        seed=profile.seed,
    )
    config.algorithm.candidate_k = profile.k
    config.algorithm.gamma = profile.gamma
    config.ann.backend = profile.index_backend
    config.ann.metric = profile.index_metric
    config.ann.nlist = profile.index_nlist
    config.ann.nprobe = profile.index_nprobe
    config.retrieval_correction_mode = profile.retrieval_correction_mode
    config.support_correction_weight = profile.support_correction_weight
    config.support_correction_target = profile.support_correction_target
    config.support_correction_metric = profile.support_correction_metric
    config.support_weighting_mode = profile.support_weighting_mode
    config.support_weight_scale = profile.support_weight_scale
    config.support_weight_min = profile.support_weight_min
    config.support_weight_max = profile.support_weight_max
    config.support_pmr_eps = profile.support_pmr_eps
    return config


def _print_smoke_summary(result: PipelineSmokeResult) -> None:
    mean_recall = result.recall_at_k.mean().item()
    print("DW-MQL smoke check passed")
    print(f"batch_size: {result.batch_size}")
    print(f"num_candidates: {result.num_candidates}")
    print(f"selected_index_backend: {result.diagnostics['selected_index_backend']}")
    print(f"selected_index_metric: {result.diagnostics['selected_index_metric']}")
    print(f"bellman_loss: {result.bellman_loss.item():.6g}")
    print(f"ess: {result.ess.item():.6g}")
    print(f"stationarity_loss: {result.stationarity_loss.item():.6g}")
    print(f"mean_recall_at_k: {mean_recall:.6g}")
    print(f"exact_backend_recall_mean: {result.diagnostics['exact_backend_recall_mean']:.6g}")
    print(
        "exact_backend_overlap_fraction_mean: "
        f"{result.diagnostics['exact_backend_overlap_fraction_mean']:.6g}"
    )
    if result.diagnostics["exact_backend_best_value_gap_mean"] is not None:
        print(
            "exact_backend_best_value_gap_mean: "
            f"{result.diagnostics['exact_backend_best_value_gap_mean']:.6g}"
        )


def summarize_group_norms_by_step(norms_by_step: list[dict[str, float]]) -> dict[str, float]:
    """Return finite mean group norms from per-step diagnostics."""

    if not norms_by_step:
        raise ValueError("norms_by_step must be non-empty.")
    totals = {group: 0.0 for group in _GROUP_NAMES}
    for step_norms in norms_by_step:
        missing = set(_GROUP_NAMES) - set(step_norms)
        if missing:
            raise ValueError(f"Missing group norm key(s): {sorted(missing)}.")
        for group in _GROUP_NAMES:
            value = float(step_norms[group])
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError("Group norm values must be finite.")
            totals[group] += value
    return {group: totals[group] / len(norms_by_step) for group in _GROUP_NAMES}


def _print_group_norm_summary(result: TrainLoopSmokeResult) -> None:
    try:
        grad_means = summarize_group_norms_by_step(result.diagnostics["gradient_group_norms_by_step"])
        delta_means = summarize_group_norms_by_step(result.diagnostics["parameter_delta_group_norms_by_step"])
    except KeyError as exc:
        raise ValueError("Missing group diagnostics in train-loop result.") from exc

    print("group_grad_norms_mean:")
    for group in _GROUP_NAMES:
        print(f"  {group}: {grad_means[group]:.6g}")
    print("group_delta_norms_mean:")
    for group in _GROUP_NAMES:
        print(f"  {group}: {delta_means[group]:.6g}")


def _print_smoke_train_summary(result: TrainLoopSmokeResult) -> None:
    mean_grad_norm = sum(result.grad_norms) / len(result.grad_norms)
    mean_delta_norm = sum(result.parameter_delta_norms) / len(result.parameter_delta_norms)
    print("DW-MQL synthetic smoke train only passed")
    print(f"num_steps: {result.num_steps}")
    print(f"initial_total_loss: {result.initial_total_loss:.6g}")
    print(f"final_total_loss: {result.final_total_loss:.6g}")
    print(f"mean_grad_norm: {mean_grad_norm:.6g}")
    print(f"mean_parameter_delta_norm: {mean_delta_norm:.6g}")
    print(f"target_update_count: {result.target_update_count}")
    _print_group_norm_summary(result)


def _print_alternating_smoke_train_summary(result: TrainLoopSmokeResult) -> None:
    mean_grad_norm = sum(result.grad_norms) / len(result.grad_norms)
    mean_delta_norm = sum(result.parameter_delta_norms) / len(result.parameter_delta_norms)
    print("DW-MQL synthetic alternating smoke train only passed")
    print(f"num_steps: {result.num_steps}")
    print(f"initial_total_loss: {result.initial_total_loss:.6g}")
    print(f"final_total_loss: {result.final_total_loss:.6g}")
    print(f"mean_grad_norm: {mean_grad_norm:.6g}")
    print(f"mean_parameter_delta_norm: {mean_delta_norm:.6g}")
    print(f"target_update_count: {result.target_update_count}")
    print("schedule_name: default_smoke_alternating_schedule")
    print(f"schedule_length: {len(result.diagnostics['update_schedule']) if result.diagnostics.get('update_schedule') else 0}")
    _print_group_norm_summary(result)


def _print_smoke_eval_summary(result: BeforeAfterSmokeResult) -> None:
    print("DW-MQL synthetic smoke evaluation only")
    print("retrieval_diagnostics_only: true")
    print("benchmark_claim: false")
    print(f"num_steps: {result.train_loop.num_steps}")
    print(f"before_bellman_loss: {result.diagnostics['before_bellman_loss']:.6g}")
    print(f"after_bellman_loss: {result.diagnostics['after_bellman_loss']:.6g}")
    print(f"bellman_loss_delta: {result.deltas['bellman_loss_delta']:.6g}")
    print(f"before_stationarity_loss: {result.diagnostics['before_stationarity_loss']:.6g}")
    print(f"after_stationarity_loss: {result.diagnostics['after_stationarity_loss']:.6g}")
    print(f"stationarity_loss_delta: {result.deltas['stationarity_loss_delta']:.6g}")
    print(f"before_ess: {result.diagnostics['before_ess']:.6g}")
    print(f"after_ess: {result.diagnostics['after_ess']:.6g}")
    print(f"ess_delta: {result.deltas['ess_delta']:.6g}")
    print(f"before_mean_recall_at_k: {result.diagnostics['before_mean_recall']:.6g}")
    print(f"after_mean_recall_at_k: {result.diagnostics['after_mean_recall']:.6g}")
    print(f"mean_recall_delta: {result.deltas['mean_recall_delta']:.6g}")
    print(
        "before_exact_backend_recall_mean: "
        f"{result.diagnostics['before_exact_backend_recall_mean']:.6g}"
    )
    print(
        "after_exact_backend_recall_mean: "
        f"{result.diagnostics['after_exact_backend_recall_mean']:.6g}"
    )
    print(
        "exact_backend_recall_mean_delta: "
        f"{result.deltas['exact_backend_recall_mean_delta']:.6g}"
    )
    print(
        "before_exact_backend_overlap_fraction_mean: "
        f"{result.diagnostics['before_exact_backend_overlap_fraction_mean']:.6g}"
    )
    print(
        "after_exact_backend_overlap_fraction_mean: "
        f"{result.diagnostics['after_exact_backend_overlap_fraction_mean']:.6g}"
    )
    print(
        "exact_backend_overlap_fraction_mean_delta: "
        f"{result.deltas['exact_backend_overlap_fraction_mean_delta']:.6g}"
    )
    print(
        "before_exact_backend_best_value_gap_mean: "
        f"{result.diagnostics['before_exact_backend_best_value_gap_mean']:.6g}"
    )
    print(
        "after_exact_backend_best_value_gap_mean: "
        f"{result.diagnostics['after_exact_backend_best_value_gap_mean']:.6g}"
    )
    print(
        "exact_backend_best_value_gap_mean_delta: "
        f"{result.deltas['exact_backend_best_value_gap_mean_delta']:.6g}"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def add_report_metadata(report: dict[str, object], mode: str) -> dict[str, object]:
    """Attach stable JSON diagnostics schema metadata to a report."""

    report.update(
        {
            "schema_name": DIAGNOSTICS_SCHEMA_NAME,
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "mode": mode,
            "synthetic_only": True,
            "benchmark_claim": False,
        }
    )
    return report


def _report_base(mode: str, profile: SmokeProfile) -> dict[str, object]:
    return add_report_metadata(
        {
            "profile": asdict(profile),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        mode,
    )


def _pipeline_smoke_report(result: PipelineSmokeResult, profile: SmokeProfile) -> dict[str, object]:
    report = _report_base("smoke", profile)
    report.update(
        {
            "summary": {
                "batch_size": result.batch_size,
                "num_candidates": result.num_candidates,
                "bellman_loss": result.bellman_loss,
                "ess": result.ess,
                "stationarity_loss": result.stationarity_loss,
                "mean_recall_at_k": result.recall_at_k.mean(),
            },
            "diagnostics": result.diagnostics,
        }
    )
    return report


def _train_loop_report(result: TrainLoopSmokeResult, profile: SmokeProfile, *, mode: str) -> dict[str, object]:
    report = _report_base(mode, profile)
    report.update(
        {
            "summary": {
                "num_steps": result.num_steps,
                "initial_total_loss": result.initial_total_loss,
                "final_total_loss": result.final_total_loss,
                "mean_grad_norm": _mean(result.grad_norms),
                "mean_parameter_delta_norm": _mean(result.parameter_delta_norms),
                "target_update_count": result.target_update_count,
            },
            "losses": result.losses,
            "grad_norms": result.grad_norms,
            "parameter_delta_norms": result.parameter_delta_norms,
            "diagnostics": result.diagnostics,
        }
    )
    return report


def _smoke_eval_report(result: BeforeAfterSmokeResult, profile: SmokeProfile) -> dict[str, object]:
    report = _report_base("smoke_eval", profile)
    report.update(
        {
            "summary": {
                "num_steps": result.train_loop.num_steps,
                "before_bellman_loss": result.diagnostics["before_bellman_loss"],
                "after_bellman_loss": result.diagnostics["after_bellman_loss"],
                "bellman_loss_delta": result.deltas["bellman_loss_delta"],
                "before_stationarity_loss": result.diagnostics["before_stationarity_loss"],
                "after_stationarity_loss": result.diagnostics["after_stationarity_loss"],
                "stationarity_loss_delta": result.deltas["stationarity_loss_delta"],
                "before_ess": result.diagnostics["before_ess"],
                "after_ess": result.diagnostics["after_ess"],
                "ess_delta": result.deltas["ess_delta"],
            },
            "deltas": result.deltas,
            "diagnostics": result.diagnostics,
        }
    )
    return report


def _synthetic_training_report(history: TrainingHistory, profile: SmokeProfile) -> dict[str, object]:
    report = _report_base("train_synthetic", profile)
    report.update(
        {
            "summary": {
                "num_steps": history.diagnostics["num_steps"],
                "initial_total_loss": history.losses[0],
                "final_total_loss": history.losses[-1],
                "mean_total_loss": _mean(history.losses),
                "mean_bellman_loss": _mean(history.bellman_losses),
                "final_raw_bellman_loss": history.diagnostics.get("final_raw_bellman_loss"),
                "mean_raw_bellman_loss": history.diagnostics.get("mean_raw_bellman_loss"),
                "final_support_weighted_bellman_loss": history.diagnostics.get("final_support_weighted_bellman_loss"),
                "mean_support_weighted_bellman_loss": history.diagnostics.get("mean_support_weighted_bellman_loss"),
                "final_critic_total_loss": history.diagnostics.get("final_critic_total_loss"),
                "mean_critic_total_loss": history.diagnostics.get("mean_critic_total_loss"),
                "mean_stationarity_loss": _mean(history.stationarity_losses),
                "mean_density_regularization": _mean(history.density_regularizations),
                "mean_grad_norm": _mean(history.grad_norms),
                "mean_parameter_delta_norm": _mean(history.parameter_delta_norms),
                "target_update_count": history.diagnostics["target_update_count"],
                "index_refresh_count": history.index_refresh_count,
                "final_stale_index_pmr_mean": history.diagnostics.get("final_stale_index_pmr_mean"),
                "final_stale_index_pmr_normalized_mean": history.diagnostics.get("final_stale_index_pmr_normalized_mean"),
                "final_stale_index_missing_mass_mean": history.diagnostics.get("final_stale_index_missing_mass_mean"),
                "uniform_pmr_baseline": history.diagnostics.get("final_stale_index_uniform_pmr_baseline"),
                "final_density_ess": history.diagnostics.get("final_density_ess"),
                "final_weight_entropy": history.diagnostics.get("final_weight_entropy"),
                "retrieval_correction_mode": history.diagnostics.get("retrieval_correction_mode"),
                "support_correction_loss": history.diagnostics.get("final_support_correction_loss"),
                "mean_support_correction_loss": history.diagnostics.get("mean_support_correction_loss"),
                "final_support_correction_loss": history.diagnostics.get("final_support_correction_loss"),
                "final_support_correction_contribution": history.diagnostics.get("final_support_correction_contribution"),
                "mean_support_correction_contribution": history.diagnostics.get("mean_support_correction_contribution"),
                "support_correction_active": history.diagnostics.get("support_correction_active"),
                "support_correction_weight": history.diagnostics.get("support_correction_weight"),
                "support_correction_metric": history.diagnostics.get("support_correction_metric"),
                "support_correction_target": history.diagnostics.get("support_correction_target"),
                "support_weighting_mode": history.diagnostics.get("support_weighting_mode"),
                "support_weight_scale": history.diagnostics.get("support_weight_scale"),
                "final_support_weight_mean": history.diagnostics.get("final_support_weight_mean"),
                "mean_support_weight_mean": history.diagnostics.get("mean_support_weight_mean"),
                "final_support_weight_min": history.diagnostics.get("final_support_weight_min"),
                "final_support_weight_max": history.diagnostics.get("final_support_weight_max"),
            },
            "history": training_history_to_dict(history),
            "diagnostics": history.diagnostics,
        }
    )
    return report


def _objective_loss_summary(
    objective_names: Sequence[str],
    losses: Sequence[float],
) -> dict[str, float | None]:
    grouped: dict[str, list[float]] = {name: [] for name in _OBJECTIVE_NAMES}
    for objective_name, loss in zip(objective_names, losses, strict=True):
        if objective_name not in grouped:
            raise ValueError(f"Unknown objective name in result: {objective_name!r}.")
        grouped[objective_name].append(float(loss))
    summary: dict[str, float | None] = {}
    for objective_name, objective_losses in grouped.items():
        summary[f"mean_{objective_name}_loss"] = _mean(objective_losses) if objective_losses else None
        summary[f"final_{objective_name}_loss"] = objective_losses[-1] if objective_losses else None
    return summary


def _format_optional_float(value: float | None) -> str:
    return "None" if value is None else f"{value:.6g}"


def _objective_update_to_dict(result: ObjectiveUpdateResult) -> dict[str, object]:
    return {
        "objective_name": result.objective_name,
        "loss": result.loss,
        "grad_norm": result.grad_norm,
        "parameter_delta_norm": result.parameter_delta_norm,
        "diagnostics": result.diagnostics,
    }


def _objective_training_report(
    schedule_result: dict[str, object],
    profile: SmokeProfile,
    trainer: SyntheticDWMQLTrainer,
) -> dict[str, object]:
    objective_names = list(schedule_result["objective_names"])  # type: ignore[arg-type]
    losses = [float(loss) for loss in schedule_result["losses"]]  # type: ignore[index]
    loss_summary = _objective_loss_summary(objective_names, losses)
    objective_results = [
        _objective_update_to_dict(result)
        for result in schedule_result["results"]  # type: ignore[index]
    ]
    report = _report_base("train_objective_synthetic", profile)
    report.update(
        {
            "summary": {
                "num_steps": schedule_result["num_steps"],
                "schedule": schedule_result["schedule"],
                **loss_summary,
                "final_raw_bellman_loss": schedule_result["diagnostics"].get("final_raw_bellman_loss"),  # type: ignore[union-attr]
                "mean_raw_bellman_loss": schedule_result["diagnostics"].get("mean_raw_bellman_loss"),  # type: ignore[union-attr]
                "final_support_weighted_bellman_loss": schedule_result["diagnostics"].get("final_support_weighted_bellman_loss"),  # type: ignore[union-attr]
                "mean_support_weighted_bellman_loss": schedule_result["diagnostics"].get("mean_support_weighted_bellman_loss"),  # type: ignore[union-attr]
                "final_critic_total_loss": schedule_result["diagnostics"].get("final_critic_total_loss"),  # type: ignore[union-attr]
                "mean_critic_total_loss": schedule_result["diagnostics"].get("mean_critic_total_loss"),  # type: ignore[union-attr]
                "target_update_count": schedule_result["target_update_count"],
                "persistent_index_used": profile.use_persistent_index,
                "index_encoder_policy": profile.index_encoder_policy,
                "index_refresh_count": schedule_result["index_refresh_count"],
                "final_index_age": schedule_result["diagnostics"]["final_index_age"],  # type: ignore[index]
                "final_stale_index_pmr_mean": schedule_result["diagnostics"].get("final_stale_index_pmr_mean"),  # type: ignore[union-attr]
                "final_stale_index_pmr_normalized_mean": schedule_result["diagnostics"].get("final_stale_index_pmr_normalized_mean"),  # type: ignore[union-attr]
                "final_stale_index_missing_mass_mean": schedule_result["diagnostics"].get("final_stale_index_missing_mass_mean"),  # type: ignore[union-attr]
                "uniform_pmr_baseline": schedule_result["diagnostics"].get("final_stale_index_uniform_pmr_baseline"),  # type: ignore[union-attr]
                "final_density_ess": schedule_result["diagnostics"].get("final_density_ess"),  # type: ignore[union-attr]
                "final_weight_entropy": schedule_result["diagnostics"].get("final_weight_entropy"),  # type: ignore[union-attr]
                "retrieval_correction_mode": schedule_result["diagnostics"].get("retrieval_correction_mode"),  # type: ignore[union-attr]
                "support_correction_loss": schedule_result["diagnostics"].get("final_support_correction_loss"),  # type: ignore[union-attr]
                "mean_support_correction_loss": schedule_result["diagnostics"].get("mean_support_correction_loss"),  # type: ignore[union-attr]
                "final_support_correction_loss": schedule_result["diagnostics"].get("final_support_correction_loss"),  # type: ignore[union-attr]
                "final_support_correction_contribution": schedule_result["diagnostics"].get("final_support_correction_contribution"),  # type: ignore[union-attr]
                "mean_support_correction_contribution": schedule_result["diagnostics"].get("mean_support_correction_contribution"),  # type: ignore[union-attr]
                "support_correction_active": schedule_result["diagnostics"].get("support_correction_active"),  # type: ignore[union-attr]
                "support_correction_weight": schedule_result["diagnostics"].get("support_correction_weight"),  # type: ignore[union-attr]
                "support_correction_metric": schedule_result["diagnostics"].get("support_correction_metric"),  # type: ignore[union-attr]
                "support_correction_target": schedule_result["diagnostics"].get("support_correction_target"),  # type: ignore[union-attr]
                "support_weighting_mode": schedule_result["diagnostics"].get("support_weighting_mode"),  # type: ignore[union-attr]
                "support_weight_scale": schedule_result["diagnostics"].get("support_weight_scale"),  # type: ignore[union-attr]
            },
            "objective_schedule": schedule_result["schedule"],
            "objective_names": objective_names,
            "objective_losses": losses,
            "objective_updates": objective_results,
            "target_update_steps": schedule_result["target_update_steps"],
            "index_ages": schedule_result["index_ages"],
            "index_refresh_steps": schedule_result["index_refresh_steps"],
            "stale_index_recall_means": schedule_result["stale_index_recall_means"],
            "stale_index_overlap_fraction_means": schedule_result["stale_index_overlap_fraction_means"],
            "stale_index_best_value_gap_means": schedule_result["stale_index_best_value_gap_means"],
            "stale_index_pmr_means": schedule_result["stale_index_pmr_means"],
            "stale_index_pmr_mins": schedule_result["stale_index_pmr_mins"],
            "stale_index_pmr_maxs": schedule_result["stale_index_pmr_maxs"],
            "stale_index_pmr_normalized_means": schedule_result["stale_index_pmr_normalized_means"],
            "stale_index_missing_mass_means": schedule_result["stale_index_missing_mass_means"],
            "critic_total_losses": schedule_result.get("critic_total_losses", []),
            "support_correction_losses": schedule_result.get("support_correction_losses", []),
            "support_correction_contributions": schedule_result.get("support_correction_contributions", []),
            "support_weighted_bellman_losses": schedule_result.get("support_weighted_bellman_losses", []),
            "support_correction_active_by_step": schedule_result.get("support_correction_active_by_step", []),
            "density_ess_values": schedule_result["density_ess_values"],
            "weight_entropy_values": schedule_result["weight_entropy_values"],
            "index_encoder_drift_mean_l2": schedule_result["index_encoder_drift_mean_l2"],
            "index_encoder_drift_max_l2": schedule_result["index_encoder_drift_max_l2"],
            "index_encoder_drift_mean_cosine_distance": schedule_result["index_encoder_drift_mean_cosine_distance"],
            "index_encoder_drift_max_cosine_distance": schedule_result["index_encoder_drift_max_cosine_distance"],
            "target_vs_online_recall_means": schedule_result["target_vs_online_recall_means"],
            "target_vs_online_overlap_fraction_means": schedule_result["target_vs_online_overlap_fraction_means"],
            "ema_vs_online_recall_means": schedule_result["ema_vs_online_recall_means"],
            "ema_vs_online_overlap_fraction_means": schedule_result["ema_vs_online_overlap_fraction_means"],
            "ema_vs_target_recall_means": schedule_result["ema_vs_target_recall_means"],
            "ema_vs_target_overlap_fraction_means": schedule_result["ema_vs_target_overlap_fraction_means"],
            "diagnostics": {
                **schedule_result["diagnostics"],  # type: ignore[arg-type]
                "persistent_index_used": profile.use_persistent_index,
                "index_encoder_policy": profile.index_encoder_policy,
                "audit_stale_index": profile.audit_stale_index,
            },
        }
    )
    return report


def _maybe_write_json(report: dict[str, object], json_out: str | None) -> None:
    if json_out is not None:
        write_json_report(report, json_out)


def parse_comma_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list for diagnostics sweep CLI flags."""

    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integers, got {value!r}.") from exc
    if not parsed:
        raise ValueError("Expected at least one integer.")
    return parsed


def parse_comma_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float list for diagnostics sweep CLI flags."""

    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated floats, got {value!r}.") from exc
    if not parsed:
        raise ValueError("Expected at least one float.")
    return parsed


def parse_comma_strings(value: str) -> tuple[str, ...]:
    """Parse a comma-separated string list for diagnostics sweep CLI flags."""

    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("Expected at least one value.")
    return parsed


def _print_synthetic_training_summary(history: TrainingHistory) -> None:
    print("DW-MQL synthetic training scaffold only")
    print(f"num_steps: {history.diagnostics['num_steps']}")
    print(f"split_backward: {str(history.diagnostics.get('split_backward', False)).lower()}")
    print(f"initial_total_loss: {history.losses[0]:.6g}")
    print(f"final_total_loss: {history.losses[-1]:.6g}")
    print(f"mean_total_loss: {_mean(history.losses):.6g}")
    print(f"mean_bellman_loss: {_mean(history.bellman_losses):.6g}")
    print(f"mean_critic_total_loss: {_format_optional_float(history.diagnostics.get('mean_critic_total_loss'))}")
    print(f"mean_stationarity_loss: {_mean(history.stationarity_losses):.6g}")
    print(f"mean_density_regularization: {_mean(history.density_regularizations):.6g}")
    print(f"mean_grad_norm: {_mean(history.grad_norms):.6g}")
    print(f"mean_parameter_delta_norm: {_mean(history.parameter_delta_norms):.6g}")
    print(f"target_update_count: {history.diagnostics['target_update_count']}")
    print(f"index_refresh_every: {history.diagnostics['index_refresh_every']}")
    print(f"index_refresh_count: {history.index_refresh_count}")
    print(f"final_index_age: {history.diagnostics['final_index_age']}")
    print(f"persistent_index_used: {str(history.diagnostics['persistent_index_used']).lower()}")
    print(f"index_encoder_policy: {history.diagnostics['index_encoder_policy']}")
    print(f"retrieval_correction_mode: {history.diagnostics['retrieval_correction_mode']}")
    print(f"support_correction_active: {str(history.diagnostics.get('support_correction_active', False)).lower()}")
    print(f"final_support_correction_loss: {_format_optional_float(history.diagnostics.get('final_support_correction_loss'))}")
    print(f"final_support_correction_contribution: {_format_optional_float(history.diagnostics.get('final_support_correction_contribution'))}")
    print(f"index_ema_tau: {history.diagnostics['index_ema_tau']}")
    print(f"index_ema_update_count: {history.diagnostics['index_ema_update_count']}")
    if history.diagnostics["persistent_index_used"]:
        print(f"final_index_encoder_drift_mean_l2: {history.diagnostics['final_index_encoder_drift_mean_l2']:.6g}")
        print(f"final_index_encoder_drift_max_l2: {history.diagnostics['final_index_encoder_drift_max_l2']:.6g}")
        print(
            "final_index_encoder_drift_mean_cosine_distance: "
            f"{history.diagnostics['final_index_encoder_drift_mean_cosine_distance']:.6g}"
        )
        print(
            "final_index_encoder_drift_max_cosine_distance: "
            f"{history.diagnostics['final_index_encoder_drift_max_cosine_distance']:.6g}"
        )
    if history.diagnostics.get("audit_encoder_policy_overlap"):
        for key in (
            "final_target_vs_online_recall_mean",
            "final_target_vs_online_overlap_fraction_mean",
            "final_ema_vs_online_recall_mean",
            "final_ema_vs_online_overlap_fraction_mean",
            "final_ema_vs_target_recall_mean",
            "final_ema_vs_target_overlap_fraction_mean",
        ):
            value = history.diagnostics[key]
            if value is not None:
                print(f"{key}: {value:.6g}")
    if history.diagnostics.get("audit_stale_index"):
        print(f"stale_index_recall_mean_final: {history.stale_index_recall_means[-1]:.6g}")
        print(
            "stale_index_overlap_fraction_mean_final: "
            f"{history.stale_index_overlap_fraction_means[-1]:.6g}"
        )
        if history.stale_index_best_value_gap_means[-1] is not None:
            print(f"stale_index_best_value_gap_mean_final: {history.stale_index_best_value_gap_means[-1]:.6g}")
        print(f"final_stale_index_pmr_mean: {_format_optional_float(history.stale_index_pmr_means[-1])}")
        print(f"final_stale_index_pmr_normalized_mean: {_format_optional_float(history.stale_index_pmr_normalized_means[-1])}")
        print(f"final_stale_index_missing_mass_mean: {_format_optional_float(history.stale_index_missing_mass_means[-1])}")
        print(f"final_stale_index_pmr_min: {_format_optional_float(history.stale_index_pmr_mins[-1])}")
        print(f"final_stale_index_pmr_max: {_format_optional_float(history.stale_index_pmr_maxs[-1])}")


def _parse_objective_schedule(value: str) -> tuple[str, ...]:
    schedule = parse_comma_strings(value)
    invalid = [name for name in schedule if name not in _OBJECTIVE_NAMES]
    if invalid:
        raise ValueError("objective schedule entries must be one of 'dual', 'density', or 'critic'.")
    return schedule


def _print_objective_training_summary(
    schedule_result: dict[str, object],
    profile: SmokeProfile,
    trainer: SyntheticDWMQLTrainer,
) -> None:
    objective_names = list(schedule_result["objective_names"])  # type: ignore[arg-type]
    losses = [float(loss) for loss in schedule_result["losses"]]  # type: ignore[index]
    summary = _objective_loss_summary(objective_names, losses)
    schedule = ",".join(schedule_result["schedule"])  # type: ignore[arg-type]

    print("DW-MQL synthetic objective training scaffold only")
    print(f"num_steps: {schedule_result['num_steps']}")
    print(f"schedule: {schedule}")
    print(f"mean_critic_loss: {_format_optional_float(summary['mean_critic_loss'])}")
    print(f"mean_density_loss: {_format_optional_float(summary['mean_density_loss'])}")
    print(f"mean_dual_objective_loss: {_format_optional_float(summary['mean_dual_loss'])}")
    print(f"final_critic_loss: {_format_optional_float(summary['final_critic_loss'])}")
    print(f"mean_critic_total_loss: {_format_optional_float(schedule_result['diagnostics'].get('mean_critic_total_loss'))}")  # type: ignore[union-attr]
    print(f"final_density_loss: {_format_optional_float(summary['final_density_loss'])}")
    print(f"final_dual_loss: {_format_optional_float(summary['final_dual_loss'])}")
    print(f"target_update_count: {schedule_result['target_update_count']}")
    print(f"persistent_index_used: {str(profile.use_persistent_index).lower()}")
    print(f"index_refresh_count: {schedule_result['index_refresh_count']}")
    print(f"final_index_age: {schedule_result['diagnostics']['final_index_age']}")  # type: ignore[index]
    print(f"index_encoder_policy: {profile.index_encoder_policy}")
    print(f"retrieval_correction_mode: {schedule_result['diagnostics'].get('retrieval_correction_mode')}")  # type: ignore[union-attr]
    print(f"support_correction_active: {str(schedule_result['diagnostics'].get('support_correction_active', False)).lower()}")  # type: ignore[union-attr]
    print(f"final_support_correction_loss: {_format_optional_float(schedule_result['diagnostics'].get('final_support_correction_loss'))}")  # type: ignore[union-attr]
    print(f"final_support_correction_contribution: {_format_optional_float(schedule_result['diagnostics'].get('final_support_correction_contribution'))}")  # type: ignore[union-attr]
    diagnostics = schedule_result["diagnostics"]  # type: ignore[assignment]
    if profile.use_persistent_index:
        print(f"final_index_encoder_drift_mean_l2: {_format_optional_float(diagnostics['final_index_encoder_drift_mean_l2'])}")  # type: ignore[index]
        print(f"final_index_encoder_drift_max_l2: {_format_optional_float(diagnostics['final_index_encoder_drift_max_l2'])}")  # type: ignore[index]
        print(
            "final_index_encoder_drift_mean_cosine_distance: "
            f"{_format_optional_float(diagnostics['final_index_encoder_drift_mean_cosine_distance'])}"  # type: ignore[index]
        )
        print(
            "final_index_encoder_drift_max_cosine_distance: "
            f"{_format_optional_float(diagnostics['final_index_encoder_drift_max_cosine_distance'])}"  # type: ignore[index]
        )
    if profile.audit_stale_index:
        print(f"stale_index_recall_mean_final: {_format_optional_float(schedule_result['stale_index_recall_means'][-1])}")  # type: ignore[index]
        print(
            "stale_index_overlap_fraction_mean_final: "
            f"{_format_optional_float(schedule_result['stale_index_overlap_fraction_means'][-1])}"  # type: ignore[index]
        )
        print(
            "stale_index_best_value_gap_mean_final: "
            f"{_format_optional_float(schedule_result['stale_index_best_value_gap_means'][-1])}"  # type: ignore[index]
        )
        print(f"final_stale_index_pmr_mean: {_format_optional_float(schedule_result['stale_index_pmr_means'][-1])}")  # type: ignore[index]
        print(f"final_stale_index_pmr_normalized_mean: {_format_optional_float(schedule_result['stale_index_pmr_normalized_means'][-1])}")  # type: ignore[index]
        print(f"final_stale_index_missing_mass_mean: {_format_optional_float(schedule_result['stale_index_missing_mass_means'][-1])}")  # type: ignore[index]
        print(f"final_stale_index_pmr_min: {_format_optional_float(schedule_result['stale_index_pmr_mins'][-1])}")  # type: ignore[index]
        print(f"final_stale_index_pmr_max: {_format_optional_float(schedule_result['stale_index_pmr_maxs'][-1])}")  # type: ignore[index]
    if profile.audit_encoder_policy_overlap:
        for key in (
            "final_target_vs_online_recall_mean",
            "final_target_vs_online_overlap_fraction_mean",
            "final_ema_vs_online_recall_mean",
            "final_ema_vs_online_overlap_fraction_mean",
            "final_ema_vs_target_recall_mean",
            "final_ema_vs_target_overlap_fraction_mean",
        ):
            value = diagnostics[key]  # type: ignore[index]
            if value is not None:
                print(f"{key}: {value:.6g}")


def _print_dwmql_training_summary(
    schedule_result: dict[str, object],
    obj_config: DWMQLObjectiveConfig,
) -> None:
    print("DW-MQL synthetic minimax training scaffold only")
    print(f"num_steps: {schedule_result['num_steps']}")
    print(f"extragradient_used: {str(schedule_result['extragradient_used']).lower()}")
    print(f"extragradient_groups: {','.join(schedule_result['extragradient_groups'])}")  # type: ignore[arg-type]
    print(f"final_bellman_loss: {_format_optional_float(schedule_result['final_bellman_loss'])}")
    print(f"final_stationarity_loss: {_format_optional_float(schedule_result['final_stationarity_loss'])}")
    print(f"final_density_regularization: {_format_optional_float(schedule_result['final_density_regularization'])}")
    print(f"final_ess: {_format_optional_float(schedule_result['final_ess'])}")
    print(f"mean_bellman_loss: {_format_optional_float(schedule_result['mean_bellman_loss'])}")
    print(f"mean_stationarity_loss: {_format_optional_float(schedule_result['mean_stationarity_loss'])}")
    print(f"mean_ess: {_format_optional_float(schedule_result['mean_ess'])}")
    print(f"target_update_count: {schedule_result['target_update_count']}")
    print(f"index_refresh_count: {schedule_result['index_refresh_count']}")
    print(f"final_index_age: {schedule_result['diagnostics']['final_index_age']}")  # type: ignore[index]
    print(f"persistent_index_used: {str(schedule_result['diagnostics']['persistent_index_used']).lower()}")  # type: ignore[index]
    print(f"bellman_weight: {obj_config.bellman_weight}")
    print(f"stationarity_weight: {obj_config.stationarity_weight}")
    print(f"density_regularization_weight: {obj_config.density_regularization_weight}")
    print(f"support_correction_weight: {obj_config.support_correction_weight}")
    print("benchmark_claim: false")


def _dwmql_training_report(
    schedule_result: dict[str, object],
    obj_config: DWMQLObjectiveConfig,
    profile: SmokeProfile,
) -> dict[str, object]:
    from dataclasses import asdict as _asdict
    report = add_report_metadata(
        {
            "benchmark_claim": False,
            "synthetic_only": True,
            "profile": asdict(profile),
            "obj_config": _asdict(obj_config),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "train_dwmql_synthetic",
    )
    report.update(
        {
            "summary": {
                "num_steps": schedule_result["num_steps"],
                "final_bellman_loss": schedule_result["final_bellman_loss"],
                "final_stationarity_loss": schedule_result["final_stationarity_loss"],
                "final_density_regularization": schedule_result["final_density_regularization"],
                "final_ess": schedule_result["final_ess"],
                "mean_bellman_loss": schedule_result["mean_bellman_loss"],
                "mean_stationarity_loss": schedule_result["mean_stationarity_loss"],
                "mean_ess": schedule_result["mean_ess"],
                "extragradient_used": schedule_result["extragradient_used"],
                "extragradient_groups": schedule_result["extragradient_groups"],
                "target_update_count": schedule_result["target_update_count"],
                "index_refresh_count": schedule_result["index_refresh_count"],
                "final_index_age": schedule_result["diagnostics"]["final_index_age"],  # type: ignore[index]
                "persistent_index_used": schedule_result["diagnostics"]["persistent_index_used"],  # type: ignore[index]
            },
            "bellman_losses": schedule_result["bellman_losses"],
            "stationarity_losses": schedule_result["stationarity_losses"],
            "ess_values": schedule_result["ess_values"],
            "index_ages": schedule_result["index_ages"],
            "index_refresh_steps": schedule_result["index_refresh_steps"],
            "diagnostics": schedule_result["diagnostics"],
        }
    )
    return report


def _format_table_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_grouped_sweep_summary(summary: dict[str, object], group_keys: tuple[str, ...]) -> None:
    from .sweeps import grouped_sweep_diagnostics

    groups = grouped_sweep_diagnostics(summary, group_keys=group_keys)
    print(f"grouped_by: {','.join(group_keys)}")
    print(
        "  "
        + " ".join(group_keys)
        + " count recall_mean recall_std pmr_mean pmr_std overlap_mean overlap_std drift_l2_mean drift_l2_std"
    )
    if not groups:
        print("  None")
        return
    for row in groups:
        group_values = " ".join(_format_table_value(row.get(key)) for key in group_keys)
        print(
            "  "
            f"{group_values} "
            f"{_format_table_value(row.get('count'))} "
            f"{_format_table_value(row.get('final_stale_index_recall_mean_mean'))} "
            f"{_format_table_value(row.get('final_stale_index_recall_mean_std'))} "
            f"{_format_table_value(row.get('final_stale_index_pmr_mean_mean'))} "
            f"{_format_table_value(row.get('final_stale_index_pmr_mean_std'))} "
            f"{_format_table_value(row.get('final_stale_index_overlap_fraction_mean_mean'))} "
            f"{_format_table_value(row.get('final_stale_index_overlap_fraction_mean_std'))} "
            f"{_format_table_value(row.get('final_index_encoder_drift_mean_l2_mean'))} "
            f"{_format_table_value(row.get('final_index_encoder_drift_mean_l2_std'))}"
        )


def _support_loss_group_keys(summary: dict[str, object]) -> tuple[str, ...]:
    runs = summary.get("runs", [])
    include_tau = (
        isinstance(runs, list)
        and any(isinstance(run, dict) and run.get("index_ema_tau") is not None for run in runs)
    )
    keys = [
        "retrieval_correction_mode",
        "support_correction_weight",
        "support_weighting_mode",
        "support_weight_scale",
        "k",
        "index_refresh_every",
        "index_encoder_policy",
    ]
    if include_tau:
        keys.append("index_ema_tau")
    return tuple(keys)


def _print_support_loss_ablation_groups(summary: dict[str, object]) -> None:
    from .sweeps import grouped_sweep_diagnostics

    group_keys = _support_loss_group_keys(summary)
    metric_keys = (
        "final_stale_index_pmr_mean",
        "final_stale_index_pmr_normalized_mean",
        "final_stale_index_missing_mass_mean",
        "uniform_pmr_baseline",
        "mean_raw_bellman_loss",
        "mean_support_weighted_bellman_loss",
        "mean_support_weight_mean",
        "mean_critic_total_loss",
        "mean_support_correction_contribution",
        "mean_bellman_loss",
        "mean_stationarity_loss",
        "mean_density_regularization",
        "final_density_ess",
        "final_weight_entropy",
    )
    runs = summary.get("runs", [])
    if not isinstance(runs, list) or any(
        not isinstance(run, dict) or any(key not in run for key in group_keys)
        for run in runs
    ):
        print("support_loss_ablation_grouped_diagnostics:")
        print(f"  grouped_by: {','.join(group_keys)}")
        print("  unavailable: summary runs do not include the support-loss grouping keys")
        return
    groups = grouped_sweep_diagnostics(summary, group_keys=group_keys, metric_keys=metric_keys)
    print("support_loss_ablation_grouped_diagnostics:")
    print(f"  grouped_by: {','.join(group_keys)}")
    print(
        "  "
        + " ".join(group_keys)
        + " count pmr_mean norm_pmr_mean missing_mass_mean uniform_baseline raw_bellman_mean support_weighted_bellman_mean support_weight_mean critic_total_mean support_contrib_mean stationarity_mean density_reg_mean ess_mean entropy_mean"
    )
    if not groups:
        print("  None")
        return
    for row in groups:
        group_values = " ".join(_format_table_value(row.get(key)) for key in group_keys)
        print(
            "  "
            f"{group_values} "
            f"{_format_table_value(row.get('count'))} "
            f"{_format_table_value(row.get('final_stale_index_pmr_mean_mean'))} "
            f"{_format_table_value(row.get('final_stale_index_pmr_normalized_mean_mean'))} "
            f"{_format_table_value(row.get('final_stale_index_missing_mass_mean_mean'))} "
            f"{_format_table_value(row.get('uniform_pmr_baseline_mean'))} "
            f"{_format_table_value(row.get('mean_raw_bellman_loss_mean'))} "
            f"{_format_table_value(row.get('mean_support_weighted_bellman_loss_mean'))} "
            f"{_format_table_value(row.get('mean_support_weight_mean_mean'))} "
            f"{_format_table_value(row.get('mean_critic_total_loss_mean'))} "
            f"{_format_table_value(row.get('mean_support_correction_contribution_mean'))} "
            f"{_format_table_value(row.get('mean_stationarity_loss_mean'))} "
            f"{_format_table_value(row.get('mean_density_regularization_mean'))} "
            f"{_format_table_value(row.get('final_density_ess_mean'))} "
            f"{_format_table_value(row.get('final_weight_entropy_mean'))}"
        )


def _print_sweep_summary_inspection(
    summary: dict[str, object],
    *,
    grouped: bool = False,
    group_keys: tuple[str, ...] = ("index_encoder_policy", "index_refresh_every"),
    associations: bool = False,
    support_associations: bool = False,
    paired_correction_analysis: bool = False,
    reference_mode: str = "aware",
    comparison_modes: tuple[str, ...] = ("naive", "uniform_weights"),
    support_bins: bool = False,
) -> None:
    from .sweeps import summarize_sweep_diagnostics_table

    rows = summarize_sweep_diagnostics_table(summary)
    top_rows = rows[:5]
    bottom_rows = list(reversed(rows))[:5]
    axes = summary.get("sweep_axes", {})

    print("DW-MQL synthetic diagnostics summary inspection only")
    print("benchmark_claim: false")
    print(f"num_runs: {summary.get('num_runs')}")
    print(f"sweep_axes: {axes}")
    if summary.get("mode") == "support_loss_ablation":
        _print_support_loss_ablation_groups(summary)
    if grouped:
        _print_grouped_sweep_summary(summary, group_keys)
        if associations:
            _print_sweep_associations(summary)
        if support_associations:
            _print_support_loss_associations(summary)
        if paired_correction_analysis:
            _print_paired_correction_analysis(
                summary,
                reference_mode=reference_mode,
                comparison_modes=comparison_modes,
                support_bins=support_bins,
            )
        return

    def print_rows(label: str, selected_rows: list[dict[str, object]]) -> None:
        print(f"{label}:")
        print(
            "  seed refresh policy recall overlap drift_l2 final_age"
        )
        if not selected_rows:
            print("  None")
            return
        for row in selected_rows:
            print(
                "  "
                f"{_format_table_value(row.get('seed'))} "
                f"{_format_table_value(row.get('index_refresh_every'))} "
                f"{_format_table_value(row.get('index_encoder_policy'))} "
                f"{_format_table_value(row.get('final_stale_index_recall_mean'))} "
                f"{_format_table_value(row.get('final_stale_index_overlap_fraction_mean'))} "
                f"{_format_table_value(row.get('final_index_encoder_drift_mean_l2'))} "
                f"{_format_table_value(row.get('final_index_age'))}"
            )

    print_rows("top_by_final_stale_index_recall_mean", top_rows)
    print_rows("bottom_by_final_stale_index_recall_mean", bottom_rows)
    if associations:
        _print_sweep_associations(summary)
    if support_associations:
        _print_support_loss_associations(summary)
    if paired_correction_analysis:
        _print_paired_correction_analysis(
            summary,
            reference_mode=reference_mode,
            comparison_modes=comparison_modes,
            support_bins=support_bins,
        )


def _print_sweep_associations(summary: dict[str, object]) -> None:
    from .sweeps import compute_sweep_metric_associations

    result = compute_sweep_metric_associations(summary)
    associations = result["associations"]
    print("synthetic diagnostics associations only")
    print("benchmark_claim: false")
    print("causality_implied: false")
    printed = 0
    for key, value in associations.items():
        if printed >= 12:
            break
        if not isinstance(value, dict):
            continue
        print(
            "  "
            f"{key}: count={_format_table_value(value.get('count'))} "
            f"pearson_r={_format_table_value(value.get('pearson_r'))}"
        )
        printed += 1


def _print_support_loss_associations(summary: dict[str, object]) -> None:
    from .sweeps import compute_support_loss_objective_associations

    result = compute_support_loss_objective_associations(summary)
    print("synthetic diagnostics associations only")
    print("support_loss_associations: true")
    print("benchmark_claim: false")
    print("causality_implied: false")
    printed = 0
    for key, value in result["associations"].items():
        if printed >= 12:
            break
        if not isinstance(value, dict):
            continue
        print(
            "  "
            f"{key}: count={_format_table_value(value.get('count'))} "
            f"pearson_r={_format_table_value(value.get('pearson_r'))}"
        )
        printed += 1


def _print_paired_correction_analysis(
    summary: dict[str, object],
    *,
    reference_mode: str,
    comparison_modes: tuple[str, ...],
    support_bins: bool = False,
) -> None:
    from .sweeps import binned_support_loss_contrasts, paired_correction_mode_contrasts

    result = paired_correction_mode_contrasts(
        summary,
        reference_mode=reference_mode,
        comparison_modes=comparison_modes,
    )
    print("synthetic paired correction diagnostics only")
    print("benchmark_claim: false")
    print("causality_implied: false")
    print(f"reference_mode: {result['reference_mode']}")
    print(f"comparison_modes: {','.join(str(mode) for mode in result['comparison_modes'])}")
    for comparison_mode, comparison in result["comparisons"].items():
        if not isinstance(comparison, dict):
            continue
        print(f"comparison: {comparison_mode}")
        print(f"  paired_count: {_format_table_value(comparison.get('paired_count'))}")
        metrics = comparison.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        printed = 0
        for metric_key, metric in metrics.items():
            if printed >= 20:
                break
            if not isinstance(metric, dict):
                continue
            print(
                "  "
                f"{metric_key}: count={_format_table_value(metric.get('count'))} "
                f"reference_mean={_format_table_value(metric.get('reference_mean'))} "
                f"comparison_mean={_format_table_value(metric.get('comparison_mean'))} "
                f"mean_delta={_format_table_value(metric.get('mean_delta'))} "
                f"std_delta={_format_table_value(metric.get('std_delta'))} "
                f"frac_neg={_format_table_value(metric.get('fraction_delta_negative'))} "
                f"frac_pos={_format_table_value(metric.get('fraction_delta_positive'))}"
            )
            printed += 1
    if not support_bins:
        return
    binned = binned_support_loss_contrasts(summary)
    print("support_bins: true")
    print(f"support_metric: {binned['support_metric']}")
    for bin_row in binned["bins"]:
        if not isinstance(bin_row, dict):
            continue
        print(
            "  "
            f"bin={bin_row.get('bin')} count={_format_table_value(bin_row.get('count'))} "
            f"support_min={_format_table_value(bin_row.get('support_min'))} "
            f"support_max={_format_table_value(bin_row.get('support_max'))}"
        )
        contrasts = bin_row.get("contrasts", {})
        if not isinstance(contrasts, dict):
            continue
        comparisons = contrasts.get("comparisons", {})
        if not isinstance(comparisons, dict):
            continue
        for comparison_mode in comparison_modes:
            comparison = comparisons.get(comparison_mode, {})
            if not isinstance(comparison, dict):
                continue
            metrics = comparison.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            raw_bellman = metrics.get("mean_raw_bellman_loss", {})
            critic_total = metrics.get("mean_critic_total_loss", {})
            pmr = metrics.get("final_stale_index_pmr_normalized_mean", {})
            print(
                "    "
                f"{comparison_mode}: paired_count={_format_table_value(comparison.get('paired_count'))} "
                f"mean_raw_bellman_delta={_format_table_value(raw_bellman.get('mean_delta') if isinstance(raw_bellman, dict) else None)} "
                f"mean_critic_total_delta={_format_table_value(critic_total.get('mean_delta') if isinstance(critic_total, dict) else None)} "
                f"norm_pmr_delta={_format_table_value(pmr.get('mean_delta') if isinstance(pmr, dict) else None)}"
            )


def _print_experiment_sweep_inspection(
    summary: dict[str, object],
    *,
    grouped: bool = False,
    group_keys: tuple[str, ...] = ("index_encoder_policy", "index_refresh_every"),
) -> None:
    print("synthetic experiment sweep diagnostics only")
    print("benchmark_claim: false")
    print(f"num_runs: {summary.get('num_runs')}")
    print(f"sweep_axes: {summary.get('sweep_axes')}")
    if not grouped:
        print(f"run_reports: {len(summary.get('run_reports', []))}")
        return
    from .experiment import grouped_experiment_sweep_metrics

    rows = grouped_experiment_sweep_metrics(summary, group_keys=group_keys)
    print("grouped_metrics:")
    for row in rows:
        group_label = " ".join(f"{key}={row[key]}" for key in group_keys)
        parts = [group_label, f"count={row['count']}"]
        for metric in (
            "mean_bellman_loss",
            "mean_stationarity_loss",
            "mean_ess",
            "mean_exact_backend_recall",
        ):
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in row:
                parts.append(f"{metric}={_format_table_value(row[mean_key])}")
                parts.append(f"{metric}_std={_format_table_value(row[std_key])}")
        print("  " + " ".join(parts))


def _experiment_sweep_table_from_summary(summary: dict[str, object]):
    from .reporting import (
        TableReport,
        experiment_sweep_summary_to_rows,
        grouped_statistics_to_table,
    )

    if isinstance(summary.get("statistical_summaries"), dict):
        return grouped_statistics_to_table(summary["statistical_summaries"])  # type: ignore[arg-type]
    rows = experiment_sweep_summary_to_rows(summary)  # type: ignore[arg-type]
    columns = list(rows[0].keys()) if rows else [
        "experiment_name",
        "seed",
        "k",
        "index_refresh_every",
        "index_encoder_policy",
        "index_ema_tau",
    ]
    return TableReport(
        title="Synthetic Experiment Sweep Runs",
        columns=columns,
        rows=rows,
        metadata={
            "synthetic_only": True,
            "benchmark_claim": False,
            "source": "runs",
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the smoke runner."""

    defaults = default_smoke_profile()
    parser = argparse.ArgumentParser(description="DW-MQL sanity runner")
    parser.add_argument("--smoke", action="store_true", help="Run the forward-only smoke check.")
    parser.add_argument("--smoke-train", action="store_true", help="Run the tiny synthetic backward-pass smoke loop.")
    parser.add_argument("--smoke-eval", action="store_true", help="Run before/after synthetic smoke evaluation.")
    parser.add_argument("--train-synthetic", action="store_true", help="Run the structured synthetic trainer scaffold.")
    parser.add_argument("--train-objective-synthetic", action="store_true", help="Run objective-scheduled synthetic trainer scaffold.")
    parser.add_argument("--sweep-synthetic-diagnostics", action="store_true", help="Run a diagnostics-only synthetic sweep.")
    parser.add_argument("--inspect-sweep-summary", action="store_true", help="Inspect a diagnostics-only sweep summary JSON.")
    parser.add_argument("--run-synthetic-experiment", action="store_true", help="Run the benchmark-ready synthetic experiment scaffold.")
    parser.add_argument("--run-synthetic-experiment-sweep", action="store_true", help="Run a synthetic experiment-level sweep scaffold.")
    parser.add_argument("--inspect-experiment-sweep", action="store_true", help="Inspect a synthetic experiment-level sweep summary JSON.")
    parser.add_argument("--validate-adapter", action="store_true", help="Validate a benchmark adapter scaffold without training.")
    parser.add_argument("--report-movielens-scaffold", action="store_true", help="Write a local MovieLens scaffold diagnostics JSON report.")
    parser.add_argument("--run-movielens-local-experiment", action="store_true", help="Run a claim-free local MovieLens experiment scaffold.")
    parser.add_argument("--run-d4rl-mock-experiment", action="store_true", help="Run a claim-free D4RL mock large-action experiment scaffold.")
    parser.add_argument("--run-recsim-mock-experiment", action="store_true", help="Run a claim-free RecSim mock slate experiment scaffold.")
    parser.add_argument("--inspect-external-experiment-report", action="store_true", help="Inspect a claim-free external scaffold experiment report.")
    parser.add_argument("--run-external-experiment-bundle", action="store_true", help="Run a claim-free bundle of external scaffold experiments.")
    parser.add_argument("--inspect-external-experiment-bundle", action="store_true", help="Inspect a claim-free external scaffold experiment bundle manifest.")
    parser.add_argument("--run-external-support-protocol", action="store_true", help="Run a claim-free external support diagnostics protocol.")
    parser.add_argument("--inspect-external-support-protocol", action="store_true", help="Inspect an external support diagnostics protocol summary.")
    parser.add_argument("--run-support-loss-ablation", action="store_true", help="Run a synthetic support-loss ablation grid.")
    parser.add_argument("--train-dwmql-synthetic", action="store_true", help="Run the experiment-ready DW-MQL minimax training mode on synthetic data.")
    parser.add_argument("--inspect-movielens-report", action="store_true", help="Inspect a local MovieLens scaffold report JSON.")
    parser.add_argument("--report-external-adapter", action="store_true", help="Write a unified external adapter scaffold report.")
    parser.add_argument("--inspect-external-adapter-report", action="store_true", help="Inspect a unified external adapter scaffold report.")
    parser.add_argument("--report-external-adapter-bundle", action="store_true", help="Write a bundle of external adapter scaffold reports.")
    parser.add_argument("--inspect-external-adapter-bundle", action="store_true", help="Inspect an external adapter scaffold report bundle manifest.")
    parser.add_argument("--run-movielens-chrono-protocol", action="store_true", help="Run the claim-free MovieLens chronological strict support protocol.")
    parser.add_argument("--run-real-data-experiment", action="store_true", help="Run a corrected seed-scoped real-data experiment.")
    parser.add_argument("--evaluate-real-data-artifacts", action="store_true", help="Inventory and aggregate real-data evaluation artifacts.")
    parser.add_argument("--real-data-dataset", choices=("movielens", "kuairand"), default=None)
    parser.add_argument("--real-data-config", type=str, default=None)
    parser.add_argument("--real-data-seeds", type=str, default="0,1,2")
    parser.add_argument("--real-data-output-root", type=str, default="runs/real_data_experiments")
    parser.add_argument("--run-d4rl-optional-protocol", action="store_true", help="Run the claim-free optional D4RL/Minari support protocol (fails clearly if deps absent).")
    parser.add_argument("--run-recsim-optional-protocol", action="store_true", help="Run the claim-free optional RecSim support protocol (mock scaffold only).")
    parser.add_argument("--run-benchmark-protocol", action="store_true", help="Run the first benchmark-style multi-seed protocol with baselines and grouped summaries.")
    parser.add_argument("--inspect-benchmark-protocol", action="store_true", help="Inspect a benchmark protocol summary JSON.")
    parser.add_argument("--benchmark-summary-json", type=str, default=None, help="Path to benchmark protocol summary JSON for --inspect-benchmark-protocol.")
    parser.add_argument("--routes", type=str, default=None, help="Comma-separated route names for --run-benchmark-protocol (e.g. movielens_chrono).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for --run-benchmark-protocol (e.g. 0,1,2).")
    parser.add_argument("--index-backend-values", type=str, default=None, help="Comma-separated index backends for --run-benchmark-protocol (e.g. exact).")
    parser.add_argument("--index-refresh-values", type=str, default=None, help="Comma-separated refresh cadences for --run-benchmark-protocol (e.g. 5,20).")
    parser.add_argument("--baselines", type=str, default=None, help="Comma-separated baseline names (e.g. random,nearest) for --run-benchmark-protocol or --run-synthetic-experiment.")
    parser.add_argument("--config", type=str, default=None, help="Load smoke-only profile YAML.")
    parser.add_argument("--json-out", type=str, default=None, help="Write a JSON diagnostics report.")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--eval-num-batches", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--adapter-name", choices=("movielens", "d4rl", "recsim", "mock_external"), default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None, help="Optional scaffold task selector for external adapter reports.")
    parser.add_argument("--dataset-name", type=str, default=None, help="D4RL/Minari dataset name for --run-d4rl-optional-protocol.")
    parser.add_argument("--external-adapter-report-json", type=str, default=None)
    parser.add_argument("--external-experiment-report-json", type=str, default=None)
    parser.add_argument("--external-experiment-bundle-manifest", type=str, default=None)
    parser.add_argument("--external-support-summary-json", type=str, default=None)
    parser.add_argument("--bundle-output-dir", type=str, default=None)
    parser.add_argument("--external-adapter-bundle-manifest", type=str, default=None)
    parser.add_argument("--movielens-data-path", type=str, default=None)
    parser.add_argument("--include-movielens", action="store_true", default=None)
    parser.add_argument("--exclude-d4rl-mock", action="store_true", default=None)
    parser.add_argument("--exclude-recsim-mock", action="store_true", default=None)
    parser.add_argument("--movielens-report-json", type=str, default=None)
    parser.add_argument("--split-mode", choices=("none", "chronological", "session", "chronological_strict"), default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--session-gap-seconds", type=int, default=None)
    parser.add_argument("--min-rating-threshold", type=float, default=4.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-transitions", type=int, default=None)
    parser.add_argument("--num-actions", type=int, default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--num-items", type=int, default=None)
    parser.add_argument("--user-feature-dim", type=int, default=None)
    parser.add_argument("--item-feature-dim", type=int, default=None)
    parser.add_argument("--slate-size", type=int, default=None)
    parser.add_argument("--num-interactions", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--k-values", type=str, default=None)
    parser.add_argument("--train-k", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target-tau", type=float, default=defaults.target_tau)
    parser.add_argument("--target-update-every", type=int, default=None)
    parser.add_argument("--index-backend", choices=("exact", "faiss", "faiss_ivf", "faiss_hnsw"), default=None)
    parser.add_argument("--index-metric", choices=("dot", "cosine"), default=None)
    parser.add_argument("--index-nlist", type=int, default=None)
    parser.add_argument("--index-nprobe", type=int, default=None)
    parser.add_argument("--index-refresh-every", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping max norm for stability controls.")
    parser.add_argument("--q-l2-regularization-weight", type=float, default=None, help="Q-value L2 regularization weight for stability controls.")
    parser.add_argument("--q-value-clip", type=float, default=None, help="Clip next-Q values in Bellman targets to [-q_value_clip, q_value_clip].")
    parser.add_argument("--reward-scale", type=float, default=None, help="Scale rewards before Bellman target construction.")
    parser.add_argument("--ranking-margin-weight", type=float, default=None, help="Weight for auxiliary ranking-margin loss (0 disables it).")
    parser.add_argument("--ranking-margin", type=float, default=None, help="Margin for auxiliary ranking-margin hinge loss.")
    parser.add_argument("--ranking-negative-samples", type=int, default=None, help="Number of negative samples for auxiliary ranking-margin loss.")
    parser.add_argument("--retrieval-alignment-weight", type=float, default=None, help="Weight for retrieval alignment loss aligning embedding dot products with relevance (0 disables).")
    parser.add_argument("--retrieval-alignment-margin", type=float, default=None, help="Margin for retrieval alignment hinge loss.")
    parser.add_argument("--alternating-smoke", action="store_true", default=None, help="Use the default alternating smoke schedule with --smoke-train.")
    parser.add_argument("--no-encoder-on-critic", action="store_true", default=None, help="Exclude shared encoders from the alternating critic smoke step.")
    parser.add_argument("--assert-gradient-flow", action="store_true", default=None, help="Fail alternating smoke if stepped groups have zero gradients.")
    parser.add_argument("--split-backward", action="store_true", help="Use experimental split-backward scaffold with --train-synthetic --alternating-smoke.")
    parser.add_argument("--objective-schedule", type=str, default="dual,density,critic", help="Comma-separated objective schedule for --train-objective-synthetic.")
    parser.add_argument("--extragradient", action="store_true", default=None, help="Use stochastic extragradient for dual+density updates in --train-dwmql-synthetic.")
    parser.add_argument("--extragradient-step-size", type=float, default=None, help="Step size for extragradient lookahead and final steps in --train-dwmql-synthetic.")
    parser.add_argument("--bellman-weight", type=float, default=None, help="Weight for the Bellman loss term in DWMQLObjectiveConfig.")
    parser.add_argument("--stationarity-weight", type=float, default=None, help="Weight for the stationarity moment term in DWMQLObjectiveConfig.")
    parser.add_argument("--density-regularization-weight", type=float, default=None, help="Weight for the density regularization term in DWMQLObjectiveConfig.")
    parser.add_argument("--critic-steps", type=int, default=None, help="Number of critic steps per DW-MQL minimax step.")
    parser.add_argument("--density-steps", type=int, default=None, help="Number of density steps per DW-MQL minimax step.")
    parser.add_argument("--dual-steps", type=int, default=None, help="Number of dual steps per DW-MQL minimax step.")
    parser.add_argument("--use-persistent-index", action="store_true", default=None, help="Use synthetic-only persistent action-index cache with --train-synthetic.")
    parser.add_argument("--audit-stale-index", action="store_true", default=None, help="Audit cached retrieval against freshly rebuilt retrieval with --train-synthetic --use-persistent-index.")
    parser.add_argument("--audit-encoder-policy-overlap", action="store_true", default=None, help="Compare retrieval overlap between online, target, and EMA index encoders.")
    parser.add_argument("--index-encoder-policy", choices=("online", "target", "ema"), default=None)
    parser.add_argument("--index-ema-tau", type=float, default=None)
    parser.add_argument(
        "--retrieval-correction-mode",
        choices=("aware", "naive", "uniform_weights", "no_density"),
        default=None,
        help="Synthetic ablation mode for retrieval/density correction scaffolds.",
    )
    parser.add_argument("--support-correction-weight", type=float, default=None)
    parser.add_argument("--support-correction-target", type=float, default=None)
    parser.add_argument(
        "--support-correction-metric",
        choices=("missing_mass", "normalized_pmr_margin"),
        default=None,
    )
    parser.add_argument(
        "--support-weighting-mode",
        choices=("none", "missing_mass", "inverse_pmr", "normalized_pmr_margin"),
        default=None,
        help="Support-aware Bellman weighting mode for aware retrieval correction.",
    )
    parser.add_argument("--support-weight-scale", type=float, default=None)
    parser.add_argument("--support-weight-min", type=float, default=None)
    parser.add_argument("--support-weight-max", type=float, default=None)
    parser.add_argument("--support-pmr-eps", type=float, default=None)
    parser.add_argument("--sweep-output-dir", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--grouped", action="store_true", help="Print grouped sweep diagnostics with --inspect-sweep-summary.")
    parser.add_argument("--group-keys", type=str, default="index_encoder_policy,index_refresh_every", help="Comma-separated run-summary keys for grouped sweep inspection.")
    parser.add_argument("--associations", action="store_true", help="Print diagnostics-only metric associations with --inspect-sweep-summary.")
    parser.add_argument("--support-associations", action="store_true", help="Print PMR/support-loss objective associations for sweep or ablation summaries.")
    parser.add_argument("--paired-correction-analysis", action="store_true", help="Print paired retrieval-correction mode contrasts for support-loss ablation summaries.")
    parser.add_argument("--reference-mode", type=str, default="aware", help="Reference retrieval correction mode for paired analysis.")
    parser.add_argument("--comparison-modes", type=str, default="naive,uniform_weights", help="Comma-separated comparison retrieval correction modes for paired analysis.")
    parser.add_argument("--support-bins", action="store_true", help="Print low/medium/high support-bin paired correction contrasts.")
    parser.add_argument("--sweep-refresh-values", type=str, default="1,5")
    parser.add_argument("--sweep-encoder-policies", type=str, default="online")
    parser.add_argument("--sweep-seeds", type=str, default="0")
    parser.add_argument("--sweep-k-values", type=str, default=None)
    parser.add_argument("--sweep-index-ema-tau-values", type=str, default=None)
    parser.add_argument("--support-ablation-output-dir", type=str, default=None)
    parser.add_argument("--support-ablation-modes", type=str, default=None)
    parser.add_argument("--support-ablation-seeds", type=str, default=None)
    parser.add_argument("--support-ablation-k-values", type=str, default=None)
    parser.add_argument("--support-ablation-refresh-values", type=str, default=None)
    parser.add_argument("--support-ablation-encoder-policies", type=str, default=None)
    parser.add_argument("--support-ablation-ema-tau-values", type=str, default=None)
    parser.add_argument("--support-correction-weight-values", type=str, default=None)
    parser.add_argument("--support-weighting-modes", type=str, default=None)
    parser.add_argument("--support-weight-scale-values", type=str, default=None)
    parser.add_argument("--experiment-sweep-output-dir", type=str, default=None)
    parser.add_argument("--experiment-sweep-seeds", type=str, default="0")
    parser.add_argument("--experiment-sweep-k-values", type=str, default=None)
    parser.add_argument("--experiment-sweep-refresh-values", type=str, default=None)
    parser.add_argument("--experiment-sweep-encoder-policies", type=str, default=None)
    parser.add_argument("--experiment-sweep-index-ema-tau-values", type=str, default=None)
    parser.add_argument("--experiment-summary-json", type=str, default=None)
    parser.add_argument("--experiment-grouped", action="store_true")
    parser.add_argument("--experiment-group-keys", type=str, default="index_encoder_policy,index_refresh_every")
    parser.add_argument("--write-markdown-table", type=str, default=None, help="Write an experiment-sweep markdown table during --inspect-experiment-sweep.")
    parser.add_argument("--write-csv-table", type=str, default=None, help="Write an experiment-sweep CSV table during --inspect-experiment-sweep.")
    parser.add_argument("--plot-metric", type=str, default=None, help="Plot one experiment-sweep metric during --inspect-experiment-sweep.")
    parser.add_argument("--plot-x-key", type=str, default="index_refresh_every")
    parser.add_argument("--plot-group-key", type=str, default=None)
    parser.add_argument("--plot-out", type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Either ``--smoke`` or ``--smoke-train`` may be used. The latter is still
    synthetic-only smoke training, not a real training entry point.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    selected_modes = [
        args.smoke,
        args.smoke_train,
        args.smoke_eval,
        args.train_synthetic,
        args.train_objective_synthetic,
        args.sweep_synthetic_diagnostics,
        args.inspect_sweep_summary,
        args.run_synthetic_experiment,
        args.run_synthetic_experiment_sweep,
        args.inspect_experiment_sweep,
        args.validate_adapter,
        args.report_movielens_scaffold,
        args.run_movielens_local_experiment,
        args.run_d4rl_mock_experiment,
        args.run_recsim_mock_experiment,
        args.inspect_external_experiment_report,
        args.run_external_experiment_bundle,
        args.inspect_external_experiment_bundle,
        args.run_external_support_protocol,
        args.inspect_external_support_protocol,
        args.run_support_loss_ablation,
        args.train_dwmql_synthetic,
        args.inspect_movielens_report,
        args.report_external_adapter,
        args.inspect_external_adapter_report,
        args.report_external_adapter_bundle,
        args.inspect_external_adapter_bundle,
        args.run_movielens_chrono_protocol,
        args.run_real_data_experiment,
        args.evaluate_real_data_artifacts,
        args.run_d4rl_optional_protocol,
        args.run_recsim_optional_protocol,
        args.run_benchmark_protocol,
        args.inspect_benchmark_protocol,
    ]
    if sum(bool(mode) for mode in selected_modes) > 1:
        parser.error("--smoke, --smoke-train, --smoke-eval, --train-synthetic, --train-objective-synthetic, --sweep-synthetic-diagnostics, --inspect-sweep-summary, --run-synthetic-experiment, --run-synthetic-experiment-sweep, --inspect-experiment-sweep, --validate-adapter, --report-movielens-scaffold, --run-movielens-local-experiment, --run-d4rl-mock-experiment, --run-recsim-mock-experiment, --inspect-external-experiment-report, --run-external-experiment-bundle, --inspect-external-experiment-bundle, --run-external-support-protocol, --inspect-external-support-protocol, --run-support-loss-ablation, --train-dwmql-synthetic, --inspect-movielens-report, --report-external-adapter, --inspect-external-adapter-report, --report-external-adapter-bundle, --inspect-external-adapter-bundle, --run-movielens-chrono-protocol, --run-d4rl-optional-protocol, --run-recsim-optional-protocol, --run-benchmark-protocol, and --inspect-benchmark-protocol are mutually exclusive.")
    if args.alternating_smoke and not (args.smoke_train or args.smoke_eval or args.train_synthetic):
        parser.error("--alternating-smoke requires --smoke-train, --smoke-eval, or --train-synthetic.")
    if args.no_encoder_on_critic and not args.alternating_smoke:
        parser.error("--no-encoder-on-critic requires --alternating-smoke.")
    if args.assert_gradient_flow and not args.alternating_smoke:
        parser.error("--assert-gradient-flow requires --alternating-smoke.")
    if args.split_backward and not args.train_synthetic:
        parser.error("--split-backward requires --train-synthetic.")
    if args.split_backward and not args.alternating_smoke:
        parser.error("--split-backward requires --alternating-smoke.")
    if args.run_real_data_experiment and (
        args.real_data_dataset is None or args.real_data_config is None
    ):
        parser.error(
            "--run-real-data-experiment requires --real-data-dataset and --real-data-config."
        )
    synthetic_train_mode = (
        args.train_synthetic
        or args.train_objective_synthetic
        or args.train_dwmql_synthetic
        or args.run_movielens_local_experiment
        or args.run_d4rl_mock_experiment
        or args.run_recsim_mock_experiment
        or args.run_movielens_chrono_protocol
        or args.run_d4rl_optional_protocol
        or args.run_recsim_optional_protocol
    )
    if args.use_persistent_index and not synthetic_train_mode:
        parser.error("--use-persistent-index requires --train-synthetic, --train-objective-synthetic, or --run-movielens-local-experiment.")
    if args.audit_stale_index and not (synthetic_train_mode and args.use_persistent_index):
        parser.error("--audit-stale-index requires --train-synthetic/--train-objective-synthetic with --use-persistent-index.")
    if args.audit_encoder_policy_overlap and not (synthetic_train_mode and args.use_persistent_index):
        parser.error("--audit-encoder-policy-overlap requires --train-synthetic/--train-objective-synthetic with --use-persistent-index.")
    if args.sweep_synthetic_diagnostics and args.sweep_output_dir is None:
        parser.error("--sweep-synthetic-diagnostics requires --sweep-output-dir.")
    if args.inspect_sweep_summary and args.summary_json is None:
        parser.error("--inspect-sweep-summary requires --summary-json.")
    if args.support_bins and not (args.inspect_sweep_summary and args.paired_correction_analysis):
        parser.error("--support-bins requires --inspect-sweep-summary --paired-correction-analysis.")
    if args.inspect_experiment_sweep and args.experiment_summary_json is None:
        parser.error("--inspect-experiment-sweep requires --experiment-summary-json.")
    if args.validate_adapter and args.adapter_name is None:
        parser.error("--validate-adapter requires --adapter-name.")
    if args.validate_adapter and args.adapter_name == "movielens" and args.data_path is None:
        parser.error("--adapter-name movielens requires --data-path.")
    if args.report_movielens_scaffold and args.data_path is None:
        parser.error("--report-movielens-scaffold requires --data-path.")
    if args.report_movielens_scaffold and args.json_out is None:
        parser.error("--report-movielens-scaffold requires --json-out.")
    if args.run_movielens_local_experiment and args.config is None and args.data_path is None:
        parser.error("--run-movielens-local-experiment requires --config or --data-path.")
    if args.inspect_external_experiment_report and args.external_experiment_report_json is None:
        parser.error("--inspect-external-experiment-report requires --external-experiment-report-json.")
    if args.run_external_experiment_bundle and args.bundle_output_dir is None:
        parser.error("--run-external-experiment-bundle requires --bundle-output-dir.")
    if args.inspect_external_experiment_bundle and args.external_experiment_bundle_manifest is None:
        parser.error("--inspect-external-experiment-bundle requires --external-experiment-bundle-manifest.")
    if args.run_external_support_protocol and args.config is None:
        parser.error("--run-external-support-protocol requires --config.")
    if args.inspect_external_support_protocol and args.external_support_summary_json is None:
        parser.error("--inspect-external-support-protocol requires --external-support-summary-json.")
    if args.run_support_loss_ablation and args.support_ablation_output_dir is None:
        parser.error("--run-support-loss-ablation requires --support-ablation-output-dir.")
    if args.inspect_movielens_report and args.movielens_report_json is None:
        parser.error("--inspect-movielens-report requires --movielens-report-json.")
    if args.report_external_adapter and args.adapter_name is None:
        parser.error("--report-external-adapter requires --adapter-name.")
    if args.report_external_adapter and args.json_out is None:
        parser.error("--report-external-adapter requires --json-out.")
    if args.report_external_adapter and args.adapter_name == "movielens" and args.data_path is None:
        parser.error("--adapter-name movielens requires --data-path.")
    if args.inspect_external_adapter_report and args.external_adapter_report_json is None:
        parser.error("--inspect-external-adapter-report requires --external-adapter-report-json.")
    if args.report_external_adapter_bundle and args.bundle_output_dir is None:
        parser.error("--report-external-adapter-bundle requires --bundle-output-dir.")
    if args.inspect_external_adapter_bundle and args.external_adapter_bundle_manifest is None:
        parser.error("--inspect-external-adapter-bundle requires --external-adapter-bundle-manifest.")
    table_export_mode = args.inspect_experiment_sweep or args.inspect_external_support_protocol
    if args.write_markdown_table and not table_export_mode:
        parser.error("--write-markdown-table requires --inspect-experiment-sweep or --inspect-external-support-protocol.")
    if args.write_csv_table and not table_export_mode:
        parser.error("--write-csv-table requires --inspect-experiment-sweep or --inspect-external-support-protocol.")
    if bool(args.plot_metric) != bool(args.plot_out):
        parser.error("--plot-metric and --plot-out must be provided together.")
    if (args.plot_metric or args.plot_out) and not args.inspect_experiment_sweep:
        parser.error("--plot-metric/--plot-out require --inspect-experiment-sweep.")
    if args.run_movielens_chrono_protocol and args.config is None and args.data_path is None:
        parser.error("--run-movielens-chrono-protocol requires --config or --data-path.")
    if args.run_d4rl_optional_protocol and args.config is None and args.dataset_name is None:
        parser.error("--run-d4rl-optional-protocol requires --config or --dataset-name.")
    if not any(selected_modes):
        print("Training is not implemented yet. Re-run with --smoke, --smoke-train, --smoke-eval, --train-synthetic, --train-objective-synthetic, --sweep-synthetic-diagnostics, --inspect-sweep-summary, --run-synthetic-experiment, --run-synthetic-experiment-sweep, --inspect-experiment-sweep, --validate-adapter, --report-movielens-scaffold, --run-movielens-local-experiment, --run-d4rl-mock-experiment, --run-recsim-mock-experiment, --inspect-external-experiment-report, --run-external-experiment-bundle, --inspect-external-experiment-bundle, --run-external-support-protocol, --inspect-external-support-protocol, --run-support-loss-ablation, --train-dwmql-synthetic, --inspect-movielens-report, --report-external-adapter, --inspect-external-adapter-report, --report-external-adapter-bundle, --inspect-external-adapter-bundle, --run-movielens-chrono-protocol, --run-d4rl-optional-protocol, --run-recsim-optional-protocol, --run-benchmark-protocol, or --inspect-benchmark-protocol.")
        return 0
    if args.inspect_external_support_protocol:
        from .experiment import (
            external_support_protocol_summary_to_table,
            load_external_support_protocol_summary,
            summarize_external_support_protocol,
        )

        try:
            summary = load_external_support_protocol_summary(args.external_support_summary_json)
            rows = summarize_external_support_protocol(summary)
            table = external_support_protocol_summary_to_table(summary)
            if args.write_markdown_table is not None:
                from .reporting import write_markdown_table

                write_markdown_table(table, args.write_markdown_table)
            if args.write_csv_table is not None:
                from .reporting import write_csv_table

                write_csv_table(table, args.write_csv_table)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external support diagnostics protocol")
        print(f"protocol_name: {summary['protocol_name']}")
        print(f"included_routes: {','.join(summary['included_routes'])}")
        print(f"k_values: {','.join(str(value) for value in summary['k_values'])}")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        for row in rows:
            print(
                "support_row: "
                f"route={row['route']} | k={row['k']} | "
                f"pmr_at_k_scaffold={row.get('pmr_at_k_scaffold')} | "
                f"missing_policy_mass_mean={row.get('missing_policy_mass_mean')} | "
                f"logged_or_positive_in_topk_rate={row.get('logged_or_positive_in_topk_rate')} | "
                f"ann_candidate_pmr_mean={row.get('ann_candidate_pmr_mean')} | "
                f"ann_candidate_recall_mean={row.get('ann_candidate_recall_mean')}"
            )
        return 0
    if args.run_external_support_protocol:
        from .experiment import (
            load_external_support_protocol_config,
            run_external_support_protocol,
            validate_external_support_protocol_config,
        )

        try:
            config = load_external_support_protocol_config(args.config)
            if args.output_dir is not None:
                config.output_dir = args.output_dir
            if args.movielens_data_path is not None:
                config.movielens_data_path = args.movielens_data_path
            if args.include_movielens:
                config.include_movielens = True
            if args.exclude_d4rl_mock:
                config.include_d4rl_mock = False
            if args.exclude_recsim_mock:
                config.include_recsim_mock = False
            if args.num_steps is not None:
                config.num_steps = args.num_steps
            if args.seed is not None:
                config.seed = args.seed
            if args.batch_size is not None:
                config.batch_size = args.batch_size
            if args.train_k is not None:
                config.train_k = args.train_k
            if args.k_values is not None:
                config.k_values = parse_comma_ints(args.k_values)
            if args.embedding_dim is not None:
                config.embedding_dim = args.embedding_dim
            if args.gamma is not None:
                config.gamma = args.gamma
            if args.lr is not None:
                config.lr = args.lr
            if args.index_backend is not None:
                config.index_backend = args.index_backend
            if args.index_metric is not None:
                config.index_metric = args.index_metric
            if args.use_persistent_index is not None:
                config.use_persistent_index = bool(args.use_persistent_index)
            if args.index_refresh_every is not None:
                config.index_refresh_every = args.index_refresh_every
            if args.index_encoder_policy is not None:
                config.index_encoder_policy = args.index_encoder_policy
            if args.index_ema_tau is not None:
                config.index_ema_tau = args.index_ema_tau
            if args.target_tau is not None:
                config.target_tau = args.target_tau
            if args.target_update_every is not None:
                config.target_update_every = args.target_update_every
            result = run_external_support_protocol(validate_external_support_protocol_config(config))
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external support diagnostics protocol only")
        print(f"protocol_name: {result['summary']['protocol_name']}")
        print(f"output_dir: {result['output_dir']}")
        print(f"summary: {result['summary_file']}")
        print(f"included_routes: {','.join(result['included_routes'])}")
        print(f"k_values: {','.join(str(value) for value in result['summary']['k_values'])}")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0
    if args.run_support_loss_ablation:
        from .sweeps import run_support_loss_ablation, support_loss_ablation_profile_from_yaml

        try:
            if args.config is not None:
                base_profile, options = support_loss_ablation_profile_from_yaml(args.config)
            else:
                base_profile = default_smoke_profile()
                base_profile.use_persistent_index = True
                base_profile.audit_stale_index = True
                base_profile.index_ema_tau = 0.05
                options = {
                    "retrieval_correction_modes": ("aware", "naive"),
                    "seeds": (0,),
                    "k_values": (base_profile.k,),
                    "index_refresh_values": (base_profile.index_refresh_every,),
                    "index_encoder_policies": ("online",),
                    "index_ema_tau_values": None,
                }
            if args.num_steps is not None:
                base_profile.num_steps = args.num_steps
            if args.batch_size is not None:
                base_profile.batch_size = args.batch_size
            if args.num_transitions is not None:
                base_profile.num_transitions = args.num_transitions
            if args.num_actions is not None:
                base_profile.num_actions = args.num_actions
            if args.state_dim is not None:
                base_profile.state_dim = args.state_dim
            if args.action_dim is not None:
                base_profile.action_dim = args.action_dim
            if args.embedding_dim is not None:
                base_profile.embedding_dim = args.embedding_dim
            if args.gamma is not None:
                base_profile.gamma = args.gamma
            if args.lr is not None:
                base_profile.lr = args.lr
            if args.index_backend is not None:
                base_profile.index_backend = args.index_backend
            if args.index_metric is not None:
                base_profile.index_metric = args.index_metric
            if args.target_tau is not None:
                base_profile.target_tau = args.target_tau
            if args.target_update_every is not None:
                base_profile.target_update_every = args.target_update_every
            if args.index_ema_tau is not None:
                base_profile.index_ema_tau = args.index_ema_tau
            if args.support_correction_weight is not None:
                base_profile.support_correction_weight = args.support_correction_weight
            if args.support_correction_target is not None:
                base_profile.support_correction_target = args.support_correction_target
            if args.support_correction_metric is not None:
                base_profile.support_correction_metric = args.support_correction_metric
            if args.support_weighting_mode is not None:
                base_profile.support_weighting_mode = args.support_weighting_mode
            if args.support_weight_scale is not None:
                base_profile.support_weight_scale = args.support_weight_scale
            if args.support_weight_min is not None:
                base_profile.support_weight_min = args.support_weight_min
            if args.support_weight_max is not None:
                base_profile.support_weight_max = args.support_weight_max
            if args.support_pmr_eps is not None:
                base_profile.support_pmr_eps = args.support_pmr_eps
            modes = (
                parse_comma_strings(args.support_ablation_modes)
                if args.support_ablation_modes is not None
                else options["retrieval_correction_modes"]
            )
            seeds = (
                parse_comma_ints(args.support_ablation_seeds)
                if args.support_ablation_seeds is not None
                else options["seeds"]
            )
            k_values = (
                parse_comma_ints(args.support_ablation_k_values)
                if args.support_ablation_k_values is not None
                else options["k_values"]
            )
            refresh_values = (
                parse_comma_ints(args.support_ablation_refresh_values)
                if args.support_ablation_refresh_values is not None
                else options["index_refresh_values"]
            )
            policies = (
                parse_comma_strings(args.support_ablation_encoder_policies)
                if args.support_ablation_encoder_policies is not None
                else options["index_encoder_policies"]
            )
            tau_values = (
                parse_comma_floats(args.support_ablation_ema_tau_values)
                if args.support_ablation_ema_tau_values is not None
                else options["index_ema_tau_values"]
            )
            support_weight_values = (
                parse_comma_floats(args.support_correction_weight_values)
                if args.support_correction_weight_values is not None
                else options.get("support_correction_weight_values")
            )
            support_weighting_modes = (
                parse_comma_strings(args.support_weighting_modes)
                if args.support_weighting_modes is not None
                else options.get("support_weighting_modes")
            )
            support_weight_scale_values = (
                parse_comma_floats(args.support_weight_scale_values)
                if args.support_weight_scale_values is not None
                else options.get("support_weight_scale_values")
            )
            result = run_support_loss_ablation(
                base_profile=base_profile,
                output_dir=args.support_ablation_output_dir,
                retrieval_correction_modes=modes,
                seeds=seeds,
                k_values=k_values,
                index_refresh_values=refresh_values,
                index_encoder_policies=policies,
                index_ema_tau_values=tau_values,
                support_correction_weight_values=support_weight_values,
                support_weighting_modes=support_weighting_modes,
                support_weight_scale_values=support_weight_scale_values,
                num_steps=base_profile.num_steps,
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL synthetic support-loss ablation only")
        print(f"num_runs: {result.num_runs}")
        print(f"output_dir: {result.output_dir}")
        print(f"summary_file: {result.summary_file}")
        print(f"retrieval_correction_modes: {','.join(result.diagnostics['retrieval_correction_modes'])}")
        print("synthetic_only: true")
        print("benchmark_claim: false")
        return 0
    if args.inspect_external_experiment_bundle:
        from .experiment import inspect_external_scaffold_experiment_bundle

        try:
            result = inspect_external_scaffold_experiment_bundle(args.external_experiment_bundle_manifest)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external scaffold experiment bundle")
        print(f"manifest_path: {result['manifest_path']}")
        print(f"output_dir: {result['output_dir']}")
        print(f"report_count: {result['report_count']}")
        print(f"included_experiments: {','.join(result['included_experiments'])}")
        for summary in result["summaries"]:
            parts = [
                f"report_type={summary['report_type']}",
                f"experiment_name={summary['experiment_name']}",
                f"scaffold_only={str(summary['scaffold_only']).lower()}",
                f"benchmark_claim={str(summary['benchmark_claim']).lower()}",
            ]
            for key in (
                "num_users",
                "num_items",
                "num_interactions",
                "num_transitions",
                "codebook_size",
                "slate_size",
                "positive_rate",
                "pmr_at_k_scaffold",
                "ann_candidate_pmr_mean",
                "ann_candidate_recall_mean",
                "observed_item_coverage",
                "avg_reward",
            ):
                if summary.get(key) is not None:
                    parts.append(f"{key}={summary[key]}")
            print("experiment_summary: " + " | ".join(parts))
        print("benchmark_claim: false")
        return 0
    if args.run_external_experiment_bundle:
        from .experiment import run_external_scaffold_experiment_bundle

        try:
            result = run_external_scaffold_experiment_bundle(
                args.bundle_output_dir,
                movielens_data_path=args.movielens_data_path,
                num_steps=2 if args.num_steps is None else args.num_steps,
                seed=0 if args.seed is None else args.seed,
            )
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external scaffold experiment bundle only")
        print(f"output_dir: {result['output_dir']}")
        print(f"manifest_path: {result['manifest_file']}")
        print(f"report_count: {len(result['report_files'])}")
        print(f"included_experiments: {','.join(result['included_experiments'])}")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0
    if args.inspect_external_experiment_report:
        from .experiment import (
            load_external_scaffold_experiment_report,
            summarize_external_scaffold_experiment_report,
        )

        try:
            report = load_external_scaffold_experiment_report(args.external_experiment_report_json)
            summary = summarize_external_scaffold_experiment_report(report)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external scaffold experiment report")
        for key in (
            "report_type",
            "experiment_name",
            "synthetic_only",
            "scaffold_only",
            "benchmark_claim",
            "num_users",
            "num_items",
            "num_interactions",
            "num_transitions",
            "state_dim",
            "action_dim",
            "codebook_size",
            "slate_size",
            "num_steps",
            "final_total_loss",
            "final_bellman_loss",
            "final_stationarity_loss",
            "final_density_regularization",
            "positive_rate",
            "candidate_hit_rate_at_k",
            "pmr_at_k_scaffold",
            "missing_policy_mass_mean",
            "logged_action_in_topk_rate",
            "logged_item_in_topk_rate",
            "ann_candidate_pmr_mean",
            "ann_candidate_recall_mean",
            "ann_candidate_overlap_fraction_mean",
            "ann_candidate_best_value_gap_mean",
            "observed_item_coverage",
            "avg_reward",
        ):
            if key in summary and summary[key] is not None:
                value = summary[key]
                if isinstance(value, bool):
                    value = str(value).lower()
                print(f"{key}: {value}")
        return 0
    if args.inspect_external_adapter_bundle:
        from .benchmarks import inspect_external_adapter_scaffold_bundle

        try:
            result = inspect_external_adapter_scaffold_bundle(args.external_adapter_bundle_manifest)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external adapter scaffold bundle")
        print(f"manifest_path: {result['manifest_path']}")
        print(f"output_dir: {result['output_dir']}")
        print(f"report_count: {result['report_count']}")
        print(f"included_adapters: {','.join(result['included_adapters'])}")
        for summary in result["summaries"]:
            parts = [
                f"adapter_name={summary['adapter_name']}",
                f"task_name={summary['task_name']}",
                f"task_type={summary['task_type']}",
            ]
            for key in (
                "dataset_size",
                "num_actions",
                "positive_rate",
                "ann_candidate_recall_mean",
                "observed_item_coverage",
                "num_transitions",
            ):
                if summary.get(key) is not None:
                    parts.append(f"{key}={summary[key]}")
            print("adapter_summary: " + " | ".join(parts))
        print("benchmark_claim: false")
        return 0
    if args.report_external_adapter_bundle:
        from .benchmarks import run_external_adapter_scaffold_report_bundle

        try:
            result = run_external_adapter_scaffold_report_bundle(
                args.bundle_output_dir,
                movielens_data_path=args.movielens_data_path,
            )
        except (AssertionError, FileNotFoundError, ImportError, NotImplementedError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external adapter scaffold report bundle only")
        print(f"output_dir: {result['output_dir']}")
        print(f"report_count: {len(result['report_files'])}")
        print(f"manifest_path: {result['manifest_file']}")
        print(f"included_adapters: {','.join(result['included_adapters'])}")
        print("benchmark_claim: false")
        return 0
    if args.inspect_external_adapter_report:
        from .benchmarks import (
            load_external_adapter_scaffold_report,
            summarize_external_adapter_scaffold_report,
        )

        try:
            report = load_external_adapter_scaffold_report(args.external_adapter_report_json)
            summary = summarize_external_adapter_scaffold_report(report)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external adapter scaffold report")
        for key in (
            "adapter_name",
            "task_name",
            "task_type",
            "synthetic_only",
            "scaffold_only",
            "benchmark_claim",
            "dataset_size",
            "state_dim",
            "action_dim",
            "num_actions",
            "positive_rate",
            "catalogue_coverage_observed",
            "ann_candidate_recall_mean",
            "ann_candidate_overlap_fraction_mean",
            "observed_item_coverage",
            "avg_reward",
            "num_transitions",
            "contract_num_actions",
        ):
            if key in summary and summary[key] is not None:
                value = summary[key]
                if isinstance(value, bool):
                    value = str(value).lower()
                print(f"{key}: {value}")
        return 0
    if args.report_external_adapter:
        from .benchmarks import (
            D4RLAdapter,
            MockExternalBenchmarkTask,
            MovieLensAdapter,
            RecSimAdapter,
            build_external_adapter_scaffold_report,
            write_external_adapter_scaffold_report,
        )

        try:
            if args.adapter_name == "mock_external":
                task = MockExternalBenchmarkTask(seed=0 if args.seed is None else args.seed)
            elif args.adapter_name == "movielens":
                task = MovieLensAdapter(
                    data_path=args.data_path,
                    min_rating_threshold=args.min_rating_threshold,
                    max_rows=args.max_rows,
                    seed=0 if args.seed is None else args.seed,
                    split_mode=args.split_mode or "none",
                    validation_fraction=args.validation_fraction,
                    test_fraction=args.test_fraction,
                    session_gap_seconds=args.session_gap_seconds,
                ).make_task()
            elif args.adapter_name == "recsim":
                task_name = args.task_name or "mock_slate"
                task = RecSimAdapter().make_task(task_name=task_name)
            elif args.adapter_name == "d4rl":
                task_name = args.task_name or "mock_arrays"
                if task_name != "mock_arrays":
                    raise NotImplementedError("D4RL CLI reporting supports only --task-name mock_arrays for now.")
                observations = torch.tensor(
                    [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0], [1.0, 1.0, 0.0]],
                    dtype=torch.float32,
                )
                actions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
                rewards = torch.tensor([1.0, 0.0, 0.5, 1.5], dtype=torch.float32)
                next_observations = observations + 0.1
                terminals = torch.tensor([False, False, False, True])
                task = D4RLAdapter(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    terminals=terminals,
                ).make_task()
            else:  # pragma: no cover - argparse choices own this.
                raise ValueError(f"Unsupported adapter: {args.adapter_name}")
            report = build_external_adapter_scaffold_report(
                task,
                adapter_name=args.adapter_name,
                extra_metadata={"task_name_arg": args.task_name},
            )
            write_external_adapter_scaffold_report(report, args.json_out)
        except (AssertionError, FileNotFoundError, ImportError, NotImplementedError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL external adapter scaffold report only")
        print(f"adapter_name: {args.adapter_name}")
        print(f"task_name: {report['task_name']}")
        print(f"task_type: {report['task_type']}")
        print(f"report_path: {args.json_out}")
        print("benchmark_claim: false")
        print("scaffold_only: true")
        return 0
    if args.inspect_movielens_report:
        try:
            report_path = Path(args.movielens_report_json)
            if not report_path.exists():
                raise FileNotFoundError(f"MovieLens report not found: {report_path}")
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            if not isinstance(report, dict):
                raise ValueError("MovieLens report must contain a JSON object.")
            if report.get("benchmark_claim") is not False:
                raise ValueError("MovieLens report benchmark_claim must be false.")
            if report.get("scaffold_only") is not True:
                raise ValueError("MovieLens report scaffold_only must be true.")
            report_type = report.get("report_type")
            if report_type not in {"movielens_local_scaffold", "movielens_local_experiment_scaffold"}:
                raise ValueError("Unsupported MovieLens report_type.")
            metadata = report.get("dataset_metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("MovieLens report dataset_metadata must be a dictionary.")
            metrics = report.get("scaffold_metrics")
            if metrics is None:
                metrics = report.get("scaffold_evaluation", {}).get("metrics", {})
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL MovieLens local report inspection scaffold only")
        print(f"report_type: {report_type}")
        print("synthetic_only: false")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        print(f"num_users: {metadata.get('num_users')}")
        print(f"num_items: {metadata.get('num_items')}")
        print(f"num_interactions: {metadata.get('num_rows', metadata.get('num_interactions'))}")
        print(f"num_train_rows: {metadata.get('num_train_rows')}")
        print(f"num_validation_rows: {metadata.get('num_validation_rows')}")
        print(f"num_test_rows: {metadata.get('num_test_rows')}")
        if isinstance(metrics, dict) and "positive_rate" in metrics:
            print(f"positive_rate: {metrics['positive_rate']}")
        ranking = report.get("ranking_scaffold_diagnostics", {})
        support = report.get("candidate_support_diagnostics", {})
        pmr = report.get("movielens_pmr_scaffold_diagnostics", {})
        ann_audit = report.get("movielens_ann_candidate_audit", {})
        if ranking:
            print("ranking_scaffold_diagnostics:")
            for split, payload in ranking.items():
                print(f"  {split}: {payload.get('metrics', {})}")
        if support:
            print("candidate_support_diagnostics:")
            for split, payload in support.items():
                print(f"  {split}: {payload.get('metrics', {})}")
        if pmr:
            print("movielens_pmr_scaffold_diagnostics:")
            for split, payload in pmr.items():
                metrics_payload = payload.get("metrics", {})
                print(f"  {split}: {metrics_payload}")
                if "pmr_at_k_scaffold" in metrics_payload:
                    print(f"  {split}_pmr_at_k_scaffold: {metrics_payload.get('pmr_at_k_scaffold')}")
                    print(f"  {split}_missing_policy_mass_mean: {metrics_payload.get('missing_policy_mass_mean')}")
                    print(f"  {split}_positive_item_in_topk_rate: {metrics_payload.get('positive_item_in_topk_rate')}")
                    print(f"  {split}_positive_item_rank_mean: {metrics_payload.get('positive_item_rank_mean')}")
        if ann_audit:
            print("movielens_ann_candidate_audit:")
            for split, payload in ann_audit.items():
                metrics_payload = payload.get("metrics", {})
                print(f"  {split}: {metrics_payload}")
                print(f"  {split}_index_backend: {metrics_payload.get('index_backend')}")
                print(f"  {split}_index_metric: {metrics_payload.get('index_metric')}")
                print(f"  {split}_ann_candidate_recall_mean: {metrics_payload.get('ann_candidate_recall_mean')}")
                print(f"  {split}_ann_candidate_overlap_fraction_mean: {metrics_payload.get('ann_candidate_overlap_fraction_mean')}")
                print(f"  {split}_ann_candidate_pmr_mean: {metrics_payload.get('ann_candidate_pmr_mean')}")
                print(f"  {split}_ann_candidate_best_value_gap_mean: {metrics_payload.get('ann_candidate_best_value_gap_mean')}")
        return 0
    if args.run_d4rl_mock_experiment:
        from .config import (
            D4RLMockExperimentConfig,
            load_d4rl_mock_experiment_config,
            validate_d4rl_mock_experiment_config,
        )
        from .experiment import run_d4rl_mock_experiment

        try:
            config = (
                load_d4rl_mock_experiment_config(args.config)
                if args.config is not None
                else D4RLMockExperimentConfig()
            )
            if args.experiment_name is not None:
                config.experiment_name = args.experiment_name
            if args.output_dir is not None:
                config.output_dir = args.output_dir
            if args.seed is not None:
                config.seed = args.seed
            if args.num_steps is not None:
                config.num_steps = args.num_steps
            if args.batch_size is not None:
                config.batch_size = args.batch_size
            if args.k is not None:
                config.k = args.k
            if args.embedding_dim is not None:
                config.embedding_dim = args.embedding_dim
            if args.num_transitions is not None:
                config.num_transitions = args.num_transitions
            if args.state_dim is not None:
                config.state_dim = args.state_dim
            if args.action_dim is not None:
                config.action_dim = args.action_dim
            if args.codebook_size is not None:
                config.codebook_size = args.codebook_size
            if args.gamma is not None:
                config.gamma = args.gamma
            if args.lr is not None:
                config.lr = args.lr
            if args.index_backend is not None:
                config.index_backend = args.index_backend
            if args.index_metric is not None:
                config.index_metric = args.index_metric
            if args.use_persistent_index is not None:
                config.use_persistent_index = bool(args.use_persistent_index)
            if args.index_refresh_every is not None:
                config.index_refresh_every = args.index_refresh_every
            if args.index_encoder_policy is not None:
                config.index_encoder_policy = args.index_encoder_policy
            if args.index_ema_tau is not None:
                config.index_ema_tau = args.index_ema_tau
            if args.target_tau is not None:
                config.target_tau = args.target_tau
            if args.target_update_every is not None:
                config.target_update_every = args.target_update_every
            result = run_d4rl_mock_experiment(validate_d4rl_mock_experiment_config(config))
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        metrics = result.evaluation_result.metrics
        print("DW-MQL D4RL mock large-action experiment scaffold only")
        print(f"experiment_name: {result.config.experiment_name}")
        print(f"output_dir: {result.config.output_dir}")
        print(f"report: {result.output_files['report']}")
        print(f"manifest: {result.output_files['manifest']}")
        print(f"num_steps: {result.config.num_steps}")
        print(f"num_transitions: {result.config.num_transitions}")
        print(f"codebook_size: {metrics.get('codebook_size')}")
        print(f"ann_candidate_recall_mean: {metrics.get('ann_candidate_recall_mean')}")
        print(f"ann_candidate_overlap_fraction_mean: {metrics.get('ann_candidate_overlap_fraction_mean')}")
        print(f"pmr_at_k_scaffold: {result.diagnostics.get('pmr_at_k_scaffold')}")
        print(f"ann_candidate_pmr_mean: {result.diagnostics.get('ann_candidate_pmr_mean')}")
        print("synthetic_only: false")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0
    if args.run_recsim_mock_experiment:
        from .config import (
            RecSimMockExperimentConfig,
            load_recsim_mock_experiment_config,
            validate_recsim_mock_experiment_config,
        )
        from .experiment import run_recsim_mock_experiment

        try:
            config = (
                load_recsim_mock_experiment_config(args.config)
                if args.config is not None
                else RecSimMockExperimentConfig()
            )
            if args.experiment_name is not None:
                config.experiment_name = args.experiment_name
            if args.output_dir is not None:
                config.output_dir = args.output_dir
            if args.seed is not None:
                config.seed = args.seed
            if args.num_steps is not None:
                config.num_steps = args.num_steps
            if args.batch_size is not None:
                config.batch_size = args.batch_size
            if args.k is not None:
                config.k = args.k
            if args.embedding_dim is not None:
                config.embedding_dim = args.embedding_dim
            if args.num_users is not None:
                config.num_users = args.num_users
            if args.num_items is not None:
                config.num_items = args.num_items
            if args.user_feature_dim is not None:
                config.user_feature_dim = args.user_feature_dim
            if args.item_feature_dim is not None:
                config.item_feature_dim = args.item_feature_dim
            if args.slate_size is not None:
                config.slate_size = args.slate_size
            if args.num_interactions is not None:
                config.num_interactions = args.num_interactions
            if args.gamma is not None:
                config.gamma = args.gamma
            if args.lr is not None:
                config.lr = args.lr
            if args.index_backend is not None:
                config.index_backend = args.index_backend
            if args.index_metric is not None:
                config.index_metric = args.index_metric
            if args.use_persistent_index is not None:
                config.use_persistent_index = bool(args.use_persistent_index)
            if args.index_refresh_every is not None:
                config.index_refresh_every = args.index_refresh_every
            if args.index_encoder_policy is not None:
                config.index_encoder_policy = args.index_encoder_policy
            if args.index_ema_tau is not None:
                config.index_ema_tau = args.index_ema_tau
            if args.target_tau is not None:
                config.target_tau = args.target_tau
            if args.target_update_every is not None:
                config.target_update_every = args.target_update_every
            result = run_recsim_mock_experiment(validate_recsim_mock_experiment_config(config))
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        metrics = result.evaluation_result.metrics
        print("DW-MQL RecSim mock slate experiment scaffold only")
        print(f"experiment_name: {result.config.experiment_name}")
        print(f"output_dir: {result.config.output_dir}")
        print(f"report: {result.output_files['report']}")
        print(f"manifest: {result.output_files['manifest']}")
        print(f"num_steps: {result.config.num_steps}")
        print(f"num_users: {metrics.get('num_users')}")
        print(f"num_items: {metrics.get('num_items')}")
        print(f"num_interactions: {metrics.get('num_interactions')}")
        print(f"slate_size: {metrics.get('slate_size')}")
        print(f"observed_item_coverage: {metrics.get('observed_item_coverage')}")
        print(f"avg_reward: {metrics.get('avg_reward')}")
        print(f"pmr_at_k_scaffold: {result.diagnostics.get('pmr_at_k_scaffold')}")
        print(f"ann_candidate_pmr_mean: {result.diagnostics.get('ann_candidate_pmr_mean')}")
        print("synthetic_only: false")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0
    if args.run_movielens_local_experiment:
        from .config import (
            MovieLensLocalExperimentConfig,
            load_movielens_local_experiment_config,
            validate_movielens_local_experiment_config,
        )
        from .experiment import run_movielens_local_experiment

        try:
            config = (
                load_movielens_local_experiment_config(args.config)
                if args.config is not None
                else MovieLensLocalExperimentConfig(data_path=args.data_path)
            )
            if args.experiment_name is not None:
                config.experiment_name = args.experiment_name
            if args.data_path is not None:
                config.data_path = args.data_path
            if args.output_dir is not None:
                config.output_dir = args.output_dir
            if args.json_out is not None:
                config.report_path = args.json_out
            if args.seed is not None:
                config.seed = args.seed
            if args.num_steps is not None:
                config.num_steps = args.num_steps
            if args.batch_size is not None:
                config.batch_size = args.batch_size
            if args.k is not None:
                config.k = args.k
            if args.embedding_dim is not None:
                config.embedding_dim = args.embedding_dim
            if args.gamma is not None:
                config.gamma = args.gamma
            if args.lr is not None:
                config.lr = args.lr
            if args.split_mode is not None:
                config.split_mode = args.split_mode
            if args.validation_fraction is not None:
                config.validation_fraction = args.validation_fraction
            if args.test_fraction is not None:
                config.test_fraction = args.test_fraction
            if args.session_gap_seconds is not None:
                config.session_gap_seconds = args.session_gap_seconds
            if args.min_rating_threshold is not None:
                config.min_rating_threshold = args.min_rating_threshold
            if args.max_rows is not None:
                config.max_rows = args.max_rows
            if args.index_backend is not None:
                config.index_backend = args.index_backend
            if args.index_metric is not None:
                config.index_metric = args.index_metric
            if args.use_persistent_index is not None:
                config.use_persistent_index = bool(args.use_persistent_index)
            if args.index_refresh_every is not None:
                config.index_refresh_every = args.index_refresh_every
            if args.index_encoder_policy is not None:
                config.index_encoder_policy = args.index_encoder_policy
            if args.index_ema_tau is not None:
                config.index_ema_tau = args.index_ema_tau
            if args.target_tau is not None:
                config.target_tau = args.target_tau
            if args.target_update_every is not None:
                config.target_update_every = args.target_update_every
            if args.baselines is not None:
                config.baseline_names = parse_comma_strings(args.baselines)
            result = run_movielens_local_experiment(validate_movielens_local_experiment_config(config))
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        metrics = result.evaluation_result.metrics
        print("DW-MQL MovieLens local experiment scaffold only")
        print(f"experiment_name: {result.config.experiment_name}")
        print(f"output_dir: {result.config.output_dir}")
        print(f"report: {result.output_files['report']}")
        print(f"manifest: {result.output_files['manifest']}")
        print(f"num_steps: {result.config.num_steps}")
        print(f"num_users: {metrics.get('num_users')}")
        print(f"num_items: {metrics.get('num_items')}")
        print(f"num_interactions: {metrics.get('num_interactions')}")
        print(f"positive_rate: {metrics.get('positive_rate'):.6g}")
        if result.config.baseline_names:
            print(f"baselines: {','.join(result.config.baseline_names)}")
            print("baseline_diagnostics_only: true")
        print("synthetic_only: false")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0
    if args.report_movielens_scaffold:
        from .benchmarks import (
            MovieLensAdapter,
            assert_benchmark_task_contract,
            build_movielens_scaffold_report,
            write_movielens_scaffold_report,
        )

        try:
            adapter = MovieLensAdapter(
                data_path=args.data_path,
                min_rating_threshold=args.min_rating_threshold,
                max_rows=args.max_rows,
                seed=0 if args.seed is None else args.seed,
                split_mode=args.split_mode or "none",
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                session_gap_seconds=args.session_gap_seconds,
            )
            task = adapter.make_task()
            assert_benchmark_task_contract(task)
            report = build_movielens_scaffold_report(task)
            write_movielens_scaffold_report(report, args.json_out)
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL MovieLens local scaffold report only")
        print(f"report_path: {args.json_out}")
        print(f"num_users: {report['dataset_metadata']['num_users']}")
        print(f"num_items: {report['dataset_metadata']['num_items']}")
        print(f"num_interactions: {report['dataset_metadata']['num_interactions']}")
        print(f"split_mode: {report['dataset_metadata']['split_mode']}")
        print(f"positive_rate: {report['scaffold_metrics']['positive_rate']:.6g}")
        print("benchmark_claim: false")
        print("scaffold_only: true")
        return 0
    if args.validate_adapter:
        from .benchmarks import (
            MockExternalBenchmarkTask,
            MovieLensAdapter,
            assert_benchmark_task_contract,
            get_external_adapter,
        )

        try:
            if args.adapter_name == "mock_external":
                availability = {"available": True, "message": "mock external task is dependency-free"}
                task = MockExternalBenchmarkTask(seed=0 if args.seed is None else args.seed)
            elif args.adapter_name == "movielens":
                adapter = MovieLensAdapter(
                    data_path=args.data_path,
                    min_rating_threshold=args.min_rating_threshold,
                    max_rows=args.max_rows,
                    seed=0 if args.seed is None else args.seed,
                    split_mode=args.split_mode or "none",
                    validation_fraction=args.validation_fraction,
                    test_fraction=args.test_fraction,
                    session_gap_seconds=args.session_gap_seconds,
                )
                adapter_availability = adapter.is_available()
                availability = {
                    "available": adapter_availability.available,
                    "message": adapter_availability.message,
                }
                task = adapter.make_task()
            else:
                adapter = get_external_adapter(args.adapter_name)
                adapter_availability = adapter.is_available()
                availability = {
                    "available": adapter_availability.available,
                    "message": adapter_availability.message,
                }
                task = adapter.make_task()
            assert_benchmark_task_contract(task)
            bundle = task.make_dataset_bundle()
            metadata = task.metadata
        except (AssertionError, FileNotFoundError, ImportError, NotImplementedError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL adapter validation scaffold only")
        print(f"adapter_name: {args.adapter_name}")
        print(f"available: {availability['available']}")
        print(f"task_name: {task.name}")
        print(f"task_type: {task.task_type}")
        print(f"dataset_size: {len(bundle.dataset)}")
        print(f"num_actions: {bundle.action_catalogue.num_actions}")
        print(f"state_dim: {task.state_dim}")
        print(f"action_dim: {task.action_dim}")
        print("benchmark_claim: false")
        if "scaffold_only" in metadata:
            print(f"scaffold_only: {str(metadata['scaffold_only']).lower()}")
        return 0
    if args.inspect_sweep_summary:
        from .sweeps import load_support_loss_ablation_summary, load_sweep_summary

        try:
            try:
                summary = load_sweep_summary(args.summary_json)
            except ValueError:
                summary = load_support_loss_ablation_summary(args.summary_json)
            group_keys = parse_comma_strings(args.group_keys)
            comparison_modes = parse_comma_strings(args.comparison_modes)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        try:
            _print_sweep_summary_inspection(
                summary,
                grouped=args.grouped,
                group_keys=group_keys,
                associations=args.associations,
                support_associations=args.support_associations,
                paired_correction_analysis=args.paired_correction_analysis,
                reference_mode=args.reference_mode,
                comparison_modes=comparison_modes,
                support_bins=args.support_bins,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return 0
    if args.inspect_experiment_sweep:
        from .experiment import load_experiment_sweep_summary

        try:
            summary = load_experiment_sweep_summary(args.experiment_summary_json)
            group_keys = parse_comma_strings(args.experiment_group_keys)
            _print_experiment_sweep_inspection(
                summary,
                grouped=args.experiment_grouped,
                group_keys=group_keys,
            )
            if args.write_markdown_table is not None:
                from .reporting import write_markdown_table

                table = _experiment_sweep_table_from_summary(summary)
                write_markdown_table(table, args.write_markdown_table)
            if args.write_csv_table is not None:
                from .reporting import write_csv_table

                table = _experiment_sweep_table_from_summary(summary)
                write_csv_table(table, args.write_csv_table)
            if args.plot_metric is not None:
                from .reporting import plot_experiment_sweep_metric

                plot_experiment_sweep_metric(
                    summary,
                    x_key=args.plot_x_key,
                    y_key=args.plot_metric,
                    group_key=args.plot_group_key,
                    output_path=args.plot_out,
                )
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        return 0
    if args.run_synthetic_experiment_sweep:
        from .experiment import ExperimentSweepConfig, run_synthetic_experiment_sweep

        try:
            base_config = (
                load_experiment_config_yaml(args.config)
                if args.config is not None
                else ExperimentConfig()
            )
            if args.num_steps is not None:
                base_config.num_steps = args.num_steps
            if args.batch_size is not None:
                base_config.batch_size = args.batch_size
            if args.k is not None:
                base_config.k = args.k
                base_config.algorithm.candidate_k = args.k
            if args.gamma is not None:
                base_config.gamma = args.gamma
                base_config.algorithm.gamma = args.gamma
                base_config.synthetic.gamma = args.gamma
            if args.eval_num_batches is not None:
                base_config.eval_num_batches = args.eval_num_batches
            if args.eval_batch_size is not None:
                base_config.eval_batch_size = args.eval_batch_size
            if args.index_backend is not None:
                base_config.index_backend = args.index_backend
                base_config.ann.backend = args.index_backend
            if args.index_metric is not None:
                base_config.index_metric = args.index_metric
                base_config.ann.metric = args.index_metric
            if args.target_tau is not None:
                base_config.target_tau = args.target_tau
            if args.index_ema_tau is not None:
                base_config.index_ema_tau = args.index_ema_tau
            seeds = parse_comma_ints(args.experiment_sweep_seeds)
            k_values = None if args.experiment_sweep_k_values is None else parse_comma_ints(args.experiment_sweep_k_values)
            refresh_values = (
                None
                if args.experiment_sweep_refresh_values is None
                else parse_comma_ints(args.experiment_sweep_refresh_values)
            )
            policies = (
                None
                if args.experiment_sweep_encoder_policies is None
                else parse_comma_strings(args.experiment_sweep_encoder_policies)
            )
            tau_values = (
                None
                if args.experiment_sweep_index_ema_tau_values is None
                else parse_comma_floats(args.experiment_sweep_index_ema_tau_values)
            )
            sweep_config = ExperimentSweepConfig(
                base_config=validate_experiment_config(base_config),
                seeds=seeds,
                k_values=k_values,
                index_refresh_values=refresh_values,
                index_encoder_policies=policies,
                index_ema_tau_values=tau_values,
                output_dir=args.experiment_sweep_output_dir,
            )
            result = run_synthetic_experiment_sweep(sweep_config)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL synthetic experiment sweep scaffold only")
        print(f"num_runs: {result.num_runs}")
        print(f"output_dir: {result.output_dir}")
        print(f"summary_file: {result.summary_file}")
        print("benchmark_claim: false")
        return 0
    if args.run_synthetic_experiment:
        from .experiment import run_synthetic_experiment

        try:
            experiment_config = (
                load_experiment_config_yaml(args.config)
                if args.config is not None
                else ExperimentConfig()
            )
            if args.experiment_name is not None:
                experiment_config.experiment_name = args.experiment_name
            if args.output_dir is not None:
                experiment_config.output_dir = args.output_dir
            if args.checkpoint_every is not None:
                experiment_config.checkpoint_every = args.checkpoint_every
            if args.resume_from is not None:
                experiment_config.resume_from = args.resume_from
            if args.eval_num_batches is not None:
                experiment_config.eval_num_batches = args.eval_num_batches
            if args.eval_batch_size is not None:
                experiment_config.eval_batch_size = args.eval_batch_size
            if args.baselines is not None:
                experiment_config.baseline_names = parse_comma_strings(args.baselines)
            if args.num_steps is not None:
                experiment_config.num_steps = args.num_steps
            if args.batch_size is not None:
                experiment_config.batch_size = args.batch_size
            if args.k is not None:
                experiment_config.k = args.k
                experiment_config.algorithm.candidate_k = args.k
            if args.gamma is not None:
                experiment_config.gamma = args.gamma
                experiment_config.algorithm.gamma = args.gamma
                experiment_config.synthetic.gamma = args.gamma
            if args.index_backend is not None:
                experiment_config.index_backend = args.index_backend
                experiment_config.ann.backend = args.index_backend
            if args.index_metric is not None:
                experiment_config.index_metric = args.index_metric
                experiment_config.ann.metric = args.index_metric
            if args.retrieval_correction_mode is not None:
                experiment_config.retrieval_correction_mode = args.retrieval_correction_mode
            experiment_config = validate_experiment_config(experiment_config)
            result = run_synthetic_experiment(experiment_config)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL synthetic benchmark scaffold only")
        print(f"experiment_name: {result.config.experiment_name}")
        print(f"num_steps: {result.config.num_steps}")
        print(f"output_dir: {result.config.output_dir}")
        print(f"report: {result.output_files['report']}")
        print(f"final_checkpoint: {result.output_files['final_checkpoint']}")
        if result.config.checkpoint_every is not None:
            print(f"checkpoint_every: {result.config.checkpoint_every}")
        if result.config.resume_from is not None:
            print(f"resume_from: {result.config.resume_from}")
        print(f"eval_num_batches: {result.config.eval_num_batches}")
        print(f"mean_eval_bellman_loss: {result.evaluation_summary.metrics.get('mean_bellman_loss')}")
        if result.config.baseline_names:
            print(f"baselines: {','.join(result.config.baseline_names)}")
            print("baseline_diagnostics_only: true")
        print("benchmark_claim: false")
        return 0
    try:
        # Benchmark protocol dispatches use --config for their own YAML schema.
        # Don't load it as a smoke profile — pass config=None so the profile
        # falls back to defaults until the benchmark dispatch runs and returns.
        import copy as _copy
        _profile_args = args
        if getattr(args, "run_benchmark_protocol", False) or getattr(args, "inspect_benchmark_protocol", False):
            _profile_args = _copy.copy(args)
            _profile_args.config = None
        profile = smoke_profile_from_args(_profile_args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if profile.alternating_smoke and not (args.smoke_train or args.smoke_eval or args.train_synthetic):
        parser.error("--alternating-smoke requires --smoke-train, --smoke-eval, or --train-synthetic.")
    synthetic_profile_train_mode = (
        args.train_synthetic
        or args.train_objective_synthetic
        or args.train_dwmql_synthetic
        or args.sweep_synthetic_diagnostics
    )
    if profile.audit_stale_index and not (synthetic_profile_train_mode and profile.use_persistent_index):
        parser.error("--audit-stale-index requires synthetic training with --use-persistent-index.")
    if profile.audit_encoder_policy_overlap and not (synthetic_profile_train_mode and profile.use_persistent_index):
        parser.error("--audit-encoder-policy-overlap requires synthetic training with --use-persistent-index.")

    if args.sweep_synthetic_diagnostics:
        try:
            refresh_values = parse_comma_ints(args.sweep_refresh_values)
            sweep_seeds = parse_comma_ints(args.sweep_seeds)
            sweep_policies = parse_comma_strings(args.sweep_encoder_policies)
            sweep_k_values = None if args.sweep_k_values is None else parse_comma_ints(args.sweep_k_values)
            sweep_index_ema_tau_values = (
                None
                if args.sweep_index_ema_tau_values is None
                else parse_comma_floats(args.sweep_index_ema_tau_values)
            )
        except ValueError as exc:
            parser.error(str(exc))
        from .sweeps import run_synthetic_diagnostics_sweep

        try:
            result = run_synthetic_diagnostics_sweep(
                base_profile=profile,
                output_dir=args.sweep_output_dir,
                index_refresh_values=refresh_values,
                index_encoder_policies=sweep_policies,
                seeds=sweep_seeds,
                num_steps=profile.num_steps,
                use_persistent_index=True,
                audit_stale_index=True,
                audit_encoder_policy_overlap=profile.audit_encoder_policy_overlap,
                k_values=sweep_k_values,
                index_ema_tau_values=sweep_index_ema_tau_values,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL synthetic diagnostics sweep only")
        print(f"num_runs: {result.num_runs}")
        print(f"output_dir: {result.output_dir}")
        print("benchmark_claim: false")
        return 0

    config = _config_from_profile(profile)
    components = build_smoke_components(profile)
    data = components.data
    model = components.model
    generator = components.generator
    target_model = components.target_model
    if args.train_objective_synthetic:
        try:
            objective_schedule = _parse_objective_schedule(args.objective_schedule)
        except ValueError as exc:
            parser.error(str(exc))
        optimizer_bundle = make_optimizer_bundle(model, lr=profile.lr, mode="separate_heads")
        trainer = SyntheticDWMQLTrainer(
            data=data,
            model=model,
            optimizer_bundle=optimizer_bundle,
            batch_size=profile.batch_size,
            k=profile.k,
            gamma=profile.gamma,
            generator=generator,
            target_model=target_model,
            target_tau=profile.target_tau,
            target_update_every=profile.target_update_every,
            density_max_weight=config.density.max_weight,
            index_backend=profile.index_backend,
            index_metric=profile.index_metric,
            index_nlist=profile.index_nlist,
            index_nprobe=profile.index_nprobe,
            index_refresh_every=profile.index_refresh_every,
            use_persistent_index=profile.use_persistent_index,
            audit_stale_index=profile.audit_stale_index,
            audit_encoder_policy_overlap=profile.audit_encoder_policy_overlap,
            index_encoder_policy=profile.index_encoder_policy,
            index_ema_tau=profile.index_ema_tau,
            retrieval_correction_mode=profile.retrieval_correction_mode,
            support_correction_weight=profile.support_correction_weight,
            support_correction_target=profile.support_correction_target,
            support_correction_metric=profile.support_correction_metric,
            support_weighting_mode=profile.support_weighting_mode,
            support_weight_scale=profile.support_weight_scale,
            support_weight_min=profile.support_weight_min,
            support_weight_max=profile.support_weight_max,
            support_pmr_eps=profile.support_pmr_eps,
        )
        try:
            schedule_result = trainer.train_objective_schedule(profile.num_steps, schedule=objective_schedule)
        except ValueError as exc:
            parser.error(str(exc))
        _print_objective_training_summary(schedule_result, profile, trainer)
        _maybe_write_json(_objective_training_report(schedule_result, profile, trainer), args.json_out)
        return 0

    if args.train_dwmql_synthetic:
        obj_config = DWMQLObjectiveConfig(
            gamma=profile.gamma,
            bellman_weight=args.bellman_weight if args.bellman_weight is not None else 1.0,
            stationarity_weight=args.stationarity_weight if args.stationarity_weight is not None else 1.0,
            density_regularization_weight=(
                args.density_regularization_weight if args.density_regularization_weight is not None else 1e-4
            ),
            support_correction_weight=(
                profile.support_correction_weight if profile.support_correction_weight is not None else 0.0
            ),
            extragradient=bool(args.extragradient),
            extragradient_step_size=(
                args.extragradient_step_size if args.extragradient_step_size is not None else 1.0
            ),
            critic_steps=args.critic_steps if args.critic_steps is not None else 1,
            density_steps=args.density_steps if args.density_steps is not None else 1,
            dual_steps=args.dual_steps if args.dual_steps is not None else 1,
        )
        try:
            validate_dwmql_objective_config(obj_config)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        optimizer_bundle = make_optimizer_bundle(model, lr=profile.lr, mode="separate_heads")
        trainer = SyntheticDWMQLTrainer(
            data=data,
            model=model,
            optimizer_bundle=optimizer_bundle,
            batch_size=profile.batch_size,
            k=profile.k,
            gamma=profile.gamma,
            generator=generator,
            target_model=target_model,
            target_tau=profile.target_tau,
            target_update_every=profile.target_update_every,
            density_max_weight=config.density.max_weight,
            index_backend=profile.index_backend,
            index_metric=profile.index_metric,
            index_nlist=profile.index_nlist,
            index_nprobe=profile.index_nprobe,
            index_refresh_every=profile.index_refresh_every,
            use_persistent_index=profile.use_persistent_index,
            audit_stale_index=profile.audit_stale_index,
            audit_encoder_policy_overlap=profile.audit_encoder_policy_overlap,
            index_encoder_policy=profile.index_encoder_policy,
            index_ema_tau=profile.index_ema_tau,
        )
        try:
            schedule_result = trainer.train_dwmql_schedule(profile.num_steps, obj_config)
        except ValueError as exc:
            parser.error(str(exc))
        _print_dwmql_training_summary(schedule_result, obj_config)
        _maybe_write_json(_dwmql_training_report(schedule_result, obj_config, profile), args.json_out)
        return 0

    if args.run_real_data_experiment:
        from .real_data_experiments import run_real_data_experiment

        if args.real_data_dataset == "movielens":
            real_data_config = load_movielens_chrono_protocol_config(
                args.real_data_config
            )
        else:
            real_data_config = load_kuairand_chrono_protocol_config(
                args.real_data_config
            )
        try:
            real_data_result = run_real_data_experiment(
                dataset=args.real_data_dataset,
                config=real_data_config,
                seeds=parse_comma_ints(args.real_data_seeds),
                output_root=args.real_data_output_root,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(real_data_result, indent=2, sort_keys=True))
        return 0

    if args.evaluate_real_data_artifacts:
        from .real_data_experiments import (
            aggregate_real_data_results,
            inventory_reports,
            write_real_data_tables,
        )

        real_data_root = Path(args.real_data_output_root)
        report_paths = list(real_data_root.rglob("real_data_evaluation.json"))
        records = inventory_reports(report_paths)
        real_data_summary = aggregate_real_data_results(
            records,
            (
                ("b2_only", "base"),
                ("density_only", "base"),
                ("full", "b2_only"),
                ("cql_b2", "cql"),
                ("bcq_b2", "bcq"),
            ),
        )
        outputs = write_real_data_tables(real_data_summary, real_data_root)
        inventory_path = real_data_root / "inventory.json"
        inventory_path.write_text(
            json.dumps(
                real_data_summary["inventory"],
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "num_records": real_data_summary["num_records"],
                    "num_valid_records": real_data_summary["num_valid_records"],
                    "inventory": str(inventory_path),
                    **outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.run_movielens_chrono_protocol:
        from .experiment import run_movielens_chrono_protocol

        try:
            if args.config is not None:
                cfg = load_movielens_chrono_protocol_config(args.config)
            else:
                from .config import MovieLensChronoProtocolConfig

                cfg = MovieLensChronoProtocolConfig(data_path=args.data_path or "")
            if args.output_dir is not None:
                cfg.output_dir = args.output_dir
            if args.data_path is not None:
                cfg.data_path = args.data_path
            if args.num_steps is not None:
                cfg.num_steps = args.num_steps
            if args.seed is not None:
                cfg.seed = args.seed
            if args.batch_size is not None:
                cfg.batch_size = args.batch_size
            if args.k_values is not None:
                cfg.k_values = parse_comma_ints(args.k_values)
            if args.train_k is not None:
                cfg.train_k = args.train_k
            if args.max_rows is not None:
                cfg.max_rows = args.max_rows
            if args.index_backend is not None:
                cfg.index_backend = args.index_backend
            if args.index_metric is not None:
                cfg.index_metric = args.index_metric
            validate_movielens_chrono_protocol_config(cfg)
            result = run_movielens_chrono_protocol(cfg)
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL MovieLens chronological support protocol only")
        print(f"protocol_name: {result.config.protocol_name}")
        print(f"output_dir: {result.config.output_dir}")
        print(f"leakage_check_passed: {result.chrono_support.get('leakage_check_passed')}")
        print(f"num_train_rows: {result.chrono_support.get('num_train_rows')}")
        print(f"num_test_rows: {result.chrono_support.get('num_test_rows')}")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0

    if args.run_d4rl_optional_protocol:
        from .experiment import run_d4rl_optional_protocol

        try:
            if args.config is not None:
                cfg = load_d4rl_optional_protocol_config(args.config)
            else:
                from .config import D4RLOptionalProtocolConfig

                dataset_name = args.dataset_name or "hopper-medium-v2"
                cfg = D4RLOptionalProtocolConfig(dataset_name=dataset_name)
            if args.output_dir is not None:
                cfg.output_dir = args.output_dir
            if args.dataset_name is not None:
                cfg.dataset_name = args.dataset_name
            if args.num_steps is not None:
                cfg.num_steps = args.num_steps
            if args.seed is not None:
                cfg.seed = args.seed
            if args.batch_size is not None:
                cfg.batch_size = args.batch_size
            if args.k_values is not None:
                cfg.k_values = parse_comma_ints(args.k_values)
            if args.codebook_size is not None:
                cfg.codebook_size = args.codebook_size
            if args.index_backend is not None:
                cfg.index_backend = args.index_backend
            if args.index_metric is not None:
                cfg.index_metric = args.index_metric
            validate_d4rl_optional_protocol_config(cfg)
            result = run_d4rl_optional_protocol(cfg)
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL D4RL optional support protocol only")
        print(f"protocol_name: {result.config.protocol_name}")
        print(f"dataset_name: {result.config.dataset_name}")
        if result.import_error is not None:
            print(f"status: dependency_not_available")
            print(f"import_error: {result.import_error}")
        else:
            print("status: completed")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0

    if args.run_recsim_optional_protocol:
        from .experiment import run_recsim_optional_protocol

        try:
            if args.config is not None:
                cfg = load_recsim_optional_protocol_config(args.config)
            else:
                from .config import RecSimOptionalProtocolConfig

                cfg = RecSimOptionalProtocolConfig()
            if args.output_dir is not None:
                cfg.output_dir = args.output_dir
            if args.num_steps is not None:
                cfg.num_steps = args.num_steps
            if args.seed is not None:
                cfg.seed = args.seed
            if args.batch_size is not None:
                cfg.batch_size = args.batch_size
            if args.k_values is not None:
                cfg.k_values = parse_comma_ints(args.k_values)
            if args.num_users is not None:
                cfg.num_users = args.num_users
            if args.num_items is not None:
                cfg.num_items = args.num_items
            if args.index_backend is not None:
                cfg.index_backend = args.index_backend
            if args.index_metric is not None:
                cfg.index_metric = args.index_metric
            validate_recsim_optional_protocol_config(cfg)
            result = run_recsim_optional_protocol(cfg)
        except (AssertionError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL RecSim optional support protocol only")
        print(f"protocol_name: {result.config.protocol_name}")
        print(f"not_implemented: {result.not_implemented}")
        print("status: mock_scaffold_only")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0

    if args.inspect_benchmark_protocol:
        if args.benchmark_summary_json is None:
            parser.error("--inspect-benchmark-protocol requires --benchmark-summary-json.")
        from .experiment import inspect_benchmark_protocol_summary

        try:
            summary_path = Path(args.benchmark_summary_json)
            if not summary_path.exists():
                raise FileNotFoundError(f"Benchmark summary not found: {summary_path}")
            with summary_path.open("r", encoding="utf-8") as handle:
                bench_summary = json.load(handle)
            if not isinstance(bench_summary, dict):
                raise ValueError("Benchmark summary must be a JSON object.")
            if bench_summary.get("benchmark_claim") is not False:
                raise ValueError("Benchmark summary benchmark_claim must be false.")
            inspection = inspect_benchmark_protocol_summary(bench_summary)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print("DW-MQL benchmark protocol summary inspection")
        print(inspection)
        return 0

    if args.run_benchmark_protocol:
        from .config import (
            BenchmarkProtocolConfig,
            _parse_benchmark_protocol_config,
            validate_benchmark_protocol_config,
        )
        from .experiment import run_benchmark_protocol

        try:
            if args.config is not None:
                cfg = _parse_benchmark_protocol_config(args.config)
            else:
                cfg = BenchmarkProtocolConfig()
            if args.output_dir is not None:
                cfg.output_dir = args.output_dir
            if args.data_path is not None:
                cfg.data_path = args.data_path
            if args.num_steps is not None:
                cfg.num_steps = args.num_steps
            if args.batch_size is not None:
                cfg.batch_size = args.batch_size
            if args.routes is not None:
                cfg.routes = parse_comma_strings(args.routes)
            if args.seeds is not None:
                cfg.seeds = parse_comma_ints(args.seeds)
            if args.k_values is not None:
                cfg.k_values = parse_comma_ints(args.k_values)
            if args.train_k is not None:
                cfg.train_k = args.train_k
            if args.index_backend_values is not None:
                cfg.index_backend_values = parse_comma_strings(args.index_backend_values)
            if args.index_refresh_values is not None:
                cfg.index_refresh_values = parse_comma_ints(args.index_refresh_values)
            if args.baselines is not None:
                cfg.baselines = parse_comma_strings(args.baselines)
            if args.embedding_dim is not None:
                cfg.embedding_dim = args.embedding_dim
            if args.gamma is not None:
                cfg.gamma = args.gamma
            if args.lr is not None:
                cfg.lr = args.lr
            if getattr(args, "max_grad_norm", None) is not None:
                cfg.max_grad_norm = args.max_grad_norm
            if getattr(args, "q_l2_regularization_weight", None) is not None:
                cfg.q_l2_regularization_weight = args.q_l2_regularization_weight
            if getattr(args, "q_value_clip", None) is not None:
                cfg.q_value_clip = args.q_value_clip
            if getattr(args, "reward_scale", None) is not None:
                cfg.reward_scale = args.reward_scale
            if getattr(args, "ranking_margin_weight", None) is not None:
                cfg.ranking_margin_weight = args.ranking_margin_weight
            if getattr(args, "ranking_margin", None) is not None:
                cfg.ranking_margin = args.ranking_margin
            if getattr(args, "ranking_negative_samples", None) is not None:
                cfg.ranking_negative_samples = args.ranking_negative_samples
            if getattr(args, "retrieval_alignment_weight", None) is not None:
                cfg.retrieval_alignment_weight = args.retrieval_alignment_weight
            if getattr(args, "retrieval_alignment_margin", None) is not None:
                cfg.retrieval_alignment_margin = args.retrieval_alignment_margin
            validate_benchmark_protocol_config(cfg)
            result = run_benchmark_protocol(cfg)
        except (AssertionError, FileNotFoundError, NotImplementedError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        num_actual_runs = result.summary.get("num_runs", len(result.run_results))
        print("DW-MQL benchmark protocol scaffold only")
        print(f"protocol_name: {result.config.protocol_name}")
        print(f"routes: {list(result.config.routes)}")
        print(f"seeds: {list(result.config.seeds)}")
        print(f"k_values: {list(result.config.k_values)}")
        print(f"baselines: {list(result.config.baselines)}")
        print(f"num_runs: {num_actual_runs}")
        print(f"total_run_rows: {len(result.run_results)}")
        print(f"output_dir: {result.output_dir}")
        print(f"summary: {result.summary_file}")
        print(f"markdown_table: {result.markdown_table_file}")
        print(f"csv_table: {result.csv_table_file}")
        print(f"manifest: {result.manifest_file}")
        print("scaffold_only: true")
        print("benchmark_claim: false")
        return 0

    if args.train_synthetic:
        if profile.alternating_smoke:
            optimizer_bundle = make_optimizer_bundle(model, lr=profile.lr, mode="separate_heads")
            optimizer = None
            schedule = default_smoke_alternating_schedule(
                include_encoder_on_critic=not profile.no_encoder_on_critic
            )
        else:
            optimizer_bundle = None
            optimizer = components.optimizer
            schedule = None
        trainer = SyntheticDWMQLTrainer(
            data=data,
            model=model,
            optimizer=optimizer,
            optimizer_bundle=optimizer_bundle,
            batch_size=profile.batch_size,
            k=profile.k,
            gamma=profile.gamma,
            generator=generator,
            target_model=target_model,
            target_tau=profile.target_tau,
            target_update_every=profile.target_update_every,
            density_max_weight=config.density.max_weight,
            split_backward=args.split_backward,
            index_backend=profile.index_backend,
            index_metric=profile.index_metric,
            index_nlist=profile.index_nlist,
            index_nprobe=profile.index_nprobe,
            index_refresh_every=profile.index_refresh_every,
            use_persistent_index=profile.use_persistent_index,
            audit_stale_index=profile.audit_stale_index,
            audit_encoder_policy_overlap=profile.audit_encoder_policy_overlap,
            index_encoder_policy=profile.index_encoder_policy,
            index_ema_tau=profile.index_ema_tau,
            retrieval_correction_mode=profile.retrieval_correction_mode,
            support_correction_weight=profile.support_correction_weight,
            support_correction_target=profile.support_correction_target,
            support_correction_metric=profile.support_correction_metric,
            support_weighting_mode=profile.support_weighting_mode,
            support_weight_scale=profile.support_weight_scale,
            support_weight_min=profile.support_weight_min,
            support_weight_max=profile.support_weight_max,
            support_pmr_eps=profile.support_pmr_eps,
        )
        history = trainer.train(profile.num_steps, update_schedule=schedule)
        _print_synthetic_training_summary(history)
        _maybe_write_json(_synthetic_training_report(history, profile), args.json_out)
        return 0

    if args.smoke_eval:
        schedule = (
            default_smoke_alternating_schedule(
                include_encoder_on_critic=not profile.no_encoder_on_critic
            )
            if profile.alternating_smoke
            else None
        )
        result = run_before_after_smoke_evaluation(
            data=data,
            model=model,
            optimizer=components.optimizer,
            num_steps=profile.num_steps,
            batch_size=profile.batch_size,
            k=profile.k,
            gamma=profile.gamma,
            generator=generator,
            target_model=target_model,
            target_tau=profile.target_tau,
            target_update_every=profile.target_update_every,
            alternating_schedule=schedule,
            assert_gradient_flow=profile.assert_gradient_flow,
            density_max_weight=config.density.max_weight,
            index_backend=profile.index_backend,
            index_metric=profile.index_metric,
            index_nlist=profile.index_nlist,
            index_nprobe=profile.index_nprobe,
        )
        _print_smoke_eval_summary(result)
        _maybe_write_json(_smoke_eval_report(result, profile), args.json_out)
        return 0

    if args.smoke_train:
        if profile.alternating_smoke:
            bundle = make_optimizer_bundle(model, lr=profile.lr, mode="separate_heads")
            schedule = default_smoke_alternating_schedule(
                include_encoder_on_critic=not profile.no_encoder_on_critic
            )
            result = run_minimal_alternating_loop(
                data=data,
                model=model,
                optimizer_bundle=bundle,
                num_steps=profile.num_steps,
                batch_size=profile.batch_size,
                k=profile.k,
                gamma=profile.gamma,
                update_schedule=schedule,
                generator=generator,
                target_model=target_model,
                target_tau=profile.target_tau,
                target_update_every=profile.target_update_every,
                density_max_weight=config.density.max_weight,
                assert_gradient_flow=profile.assert_gradient_flow,
                index_backend=profile.index_backend,
                index_metric=profile.index_metric,
                index_nlist=profile.index_nlist,
                index_nprobe=profile.index_nprobe,
            )
            _print_alternating_smoke_train_summary(result)
            _maybe_write_json(_train_loop_report(result, profile, mode="smoke_train"), args.json_out)
            return 0

        result = run_minimal_train_loop(
            data=data,
            model=model,
            optimizer=components.optimizer,
            num_steps=profile.num_steps,
            batch_size=profile.batch_size,
            k=profile.k,
            gamma=profile.gamma,
            generator=generator,
            target_model=target_model,
            target_tau=profile.target_tau,
            target_update_every=profile.target_update_every,
            density_max_weight=config.density.max_weight,
            index_backend=profile.index_backend,
            index_metric=profile.index_metric,
            index_nlist=profile.index_nlist,
            index_nprobe=profile.index_nprobe,
        )
        _print_smoke_train_summary(result)
        _maybe_write_json(_train_loop_report(result, profile, mode="smoke_train"), args.json_out)
        return 0

    result = run_pipeline_smoke_step(
        data=data,
        model=model,
        batch_size=profile.batch_size,
        k=profile.k,
        gamma=profile.gamma,
        generator=generator,
        density_max_weight=config.density.max_weight,
        index_backend=profile.index_backend,
        index_metric=profile.index_metric,
        index_nlist=profile.index_nlist,
        index_nprobe=profile.index_nprobe,
    )
    _print_smoke_summary(result)
    _maybe_write_json(_pipeline_smoke_report(result, profile), args.json_out)
    return 0


def train(config: ExperimentConfig | None = None) -> None:
    """Placeholder for future training."""

    _ = config or ExperimentConfig()
    raise NotImplementedError("Training is not implemented yet. Use --smoke for the sanity check.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
