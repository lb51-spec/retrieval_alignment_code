"""Synthetic-only benchmark experiment runner scaffold.

The runner prepares the shape of future paper-grade experiments while staying
controlled: no external benchmark dependencies, no baselines, and no benchmark
claims. Checkpointing here is minimal synthetic-experiment state persistence.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

try:  # pragma: no cover - dependency guard.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .baselines import BaselineResult, evaluate_synthetic_baseline, get_baseline
from .benchmarks import (
    D4RLActionCodebook,
    D4RLAdapter,
    D4RLDatasetTask,
    D4RLLargeActionTask,
    EvaluationResult,
    KuaiRandLocalTask,
    MovieLensAdapter,
    MovieLensLocalTask,
    RecSimAdapter,
    RecSimMockSlateTask,
    SyntheticBenchmarkTask,
    assert_benchmark_task_contract,
    build_d4rl_action_codebook_from_actions,
    diagnose_movielens_q_degeneracy,
    evaluate_d4rl_action_codebook_audit,
    evaluate_d4rl_ann_candidate_audit,
    evaluate_d4rl_pmr_scaffold,
    evaluate_movielens_ann_candidate_audit,
    evaluate_movielens_candidate_support_scaffold,
    evaluate_movielens_chrono_support_protocol,
    evaluate_movielens_pmr_scaffold,
    evaluate_movielens_ranking_scaffold,
    evaluate_movielens_scaffold_task,
    evaluate_recsim_ann_candidate_audit,
    evaluate_recsim_pmr_scaffold,
    evaluate_recsim_slate_scaffold,
)
from .checkpointing import (
    CheckpointMetadata,
    load_checkpoint,
    restore_model_from_checkpoint,
    restore_optimizer_from_checkpoint,
    restore_target_model_from_checkpoint,
    save_checkpoint,
)
from .config import (
    BenchmarkProtocolConfig,
    component_identity_from_config,
    D4RLMockExperimentConfig,
    D4RLOptionalProtocolConfig,
    ExperimentConfig,
    ModelConfig,
    MovieLensChronoProtocolConfig,
    MovieLensLocalExperimentConfig,
    RecSimMockExperimentConfig,
    RecSimOptionalProtocolConfig,
    validate_benchmark_protocol_config,
    validate_d4rl_mock_experiment_config,
    validate_d4rl_optional_protocol_config,
    validate_experiment_config,
    validate_movielens_chrono_protocol_config,
    validate_movielens_local_experiment_config,
    validate_recsim_mock_experiment_config,
    validate_recsim_optional_protocol_config,
    KuaiRandChronoProtocolConfig,
    validate_kuairand_chrono_protocol_config,
)
from .datasets import InMemoryReplayDataset
from .evaluation import EvaluationConfig, EvaluationSummary, evaluate_synthetic_model
from .manifest import RunManifest, write_run_manifest
from .real_data_evaluation import stable_action_id_hash
from .statistics import group_metric_summary, summarize_experiment_sweep_statistics
from .target_networks import clone_target_network
from .trainer import (
    SyntheticDWMQLTrainer,
    TrainingHistory,
    make_optimizer_bundle,
    training_history_to_dict,
    write_json_report,
)


@dataclass(slots=True)
class ExperimentResult:
    """Result object returned by synthetic benchmark scaffold runs."""

    config: ExperimentConfig
    training_history: TrainingHistory
    evaluation_result: EvaluationResult
    evaluation_summary: EvaluationSummary
    baseline_results: list[BaselineResult] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentSweepConfig:
    """Config for synthetic experiment-level sweeps."""

    base_config: ExperimentConfig
    seeds: tuple[int, ...] = (0,)
    k_values: tuple[int, ...] | None = None
    index_refresh_values: tuple[int, ...] | None = None
    index_encoder_policies: tuple[str, ...] | None = None
    index_ema_tau_values: tuple[float, ...] | None = None
    output_dir: str | Path | None = None
    sweep_name: str = "synthetic_experiment_sweep"


@dataclass(slots=True)
class ExperimentSweepResult:
    """Aggregate result for synthetic experiment-level sweeps."""

    sweep_config: ExperimentSweepConfig
    experiment_results: list[ExperimentResult]
    summary_file: str
    manifest_file: str
    summary: dict[str, Any]
    output_dir: str
    num_runs: int


@dataclass(slots=True)
class MovieLensTrainerData:
    """Trainer-compatible local MovieLens data view with compact action ids."""

    dataset: Any
    action_features: Any
    action_ids: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MovieLensLocalExperimentResult:
    """Result object for claim-free local MovieLens experiment scaffolds."""

    config: MovieLensLocalExperimentConfig
    training_history: TrainingHistory
    evaluation_result: EvaluationResult
    baseline_results: list[BaselineResult] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: MovieLensLocalTask | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)


@dataclass(slots=True)
class D4RLTrainerData:
    """Trainer-compatible D4RL mock data view."""

    dataset: Any
    action_features: Any
    action_ids: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class D4RLMockExperimentResult:
    """Result object for claim-free D4RL mock large-action experiments."""

    config: D4RLMockExperimentConfig
    training_history: TrainingHistory
    evaluation_result: EvaluationResult
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: D4RLLargeActionTask | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)


@dataclass(slots=True)
class RecSimMockExperimentResult:
    """Result object for claim-free RecSim mock slate experiments."""

    config: RecSimMockExperimentConfig
    training_history: TrainingHistory
    evaluation_result: EvaluationResult
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: RecSimMockSlateTask | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)


@dataclass(slots=True)
class ExternalSupportProtocolConfig:
    """Config for claim-free external-route support diagnostics."""

    protocol_name: str = "external_support_protocol"
    output_dir: str = "runs/external_support_protocol"
    seed: int = 0
    num_steps: int = 20
    batch_size: int = 8
    k_values: tuple[int, ...] = (5, 10, 20)
    train_k: int = 10
    embedding_dim: int = 8
    gamma: float = 0.99
    lr: float = 3e-4
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    target_tau: float | None = 0.05
    target_update_every: int = 1
    include_movielens: bool = False
    movielens_data_path: str | None = None
    include_d4rl_mock: bool = True
    include_recsim_mock: bool = True
    scaffold_only: bool = True
    benchmark_claim: bool = False


EXTERNAL_SCAFFOLD_EXPERIMENT_REPORT_TYPES = {
    "movielens_local_experiment_scaffold",
    "d4rl_mock_experiment_scaffold",
    "recsim_mock_experiment_scaffold",
}


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run synthetic experiments.")


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_split_metric(split_payload: Any, metric_name: str) -> Any:
    if not isinstance(split_payload, dict):
        return None
    for split in ("test", "validation", "train"):
        metrics = _nested_get(split_payload, split, "metrics")
        if isinstance(metrics, dict) and metric_name in metrics:
            return metrics[metric_name]
    for payload in split_payload.values():
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if isinstance(metrics, dict) and metric_name in metrics:
            return metrics[metric_name]
    return None


def load_external_scaffold_experiment_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a claim-free external scaffold experiment report."""

    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"External scaffold experiment report not found: {report_path}")
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid external scaffold experiment report JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("External scaffold experiment report must contain a JSON object.")
    if report.get("benchmark_claim") is not False:
        raise ValueError("External scaffold experiment report benchmark_claim must be false.")
    if report.get("scaffold_only") is not True:
        raise ValueError("External scaffold experiment report scaffold_only must be true.")
    report_type = report.get("report_type")
    if report_type not in EXTERNAL_SCAFFOLD_EXPERIMENT_REPORT_TYPES:
        raise ValueError("Unsupported external scaffold experiment report_type.")
    if not isinstance(report.get("experiment_name"), str) or not report["experiment_name"]:
        raise ValueError("External scaffold experiment report must include experiment_name.")
    return report


def summarize_external_scaffold_experiment_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact metadata and diagnostics from an external scaffold experiment report."""

    validated = load_external_scaffold_experiment_report_dict(report)
    report_type = validated["report_type"]
    config = validated.get("config", {}) if isinstance(validated.get("config"), dict) else {}
    history = (
        validated.get("training_history_summary", {})
        if isinstance(validated.get("training_history_summary"), dict)
        else {}
    )
    summary: dict[str, Any] = {
        "report_type": report_type,
        "experiment_name": validated["experiment_name"],
        "synthetic_only": validated.get("synthetic_only"),
        "scaffold_only": validated.get("scaffold_only"),
        "benchmark_claim": validated.get("benchmark_claim"),
        "num_steps": history.get("num_steps", config.get("num_steps")),
        "final_total_loss": history.get("final_total_loss"),
        "final_bellman_loss": history.get("final_bellman_loss"),
        "final_stationarity_loss": history.get("final_stationarity_loss"),
        "final_density_regularization": history.get("final_density_regularization"),
    }
    if report_type == "movielens_local_experiment_scaffold":
        metadata = validated.get("dataset_metadata", {})
        metrics = _nested_get(validated, "scaffold_evaluation", "metrics") or {}
        summary.update(
            {
                "num_users": metadata.get("num_users") if isinstance(metadata, dict) else None,
                "num_items": metadata.get("num_items") if isinstance(metadata, dict) else None,
                "num_interactions": (
                    metadata.get("num_interactions", metadata.get("num_rows")) if isinstance(metadata, dict) else None
                ),
                "positive_rate": metrics.get("positive_rate") if isinstance(metrics, dict) else None,
                "candidate_hit_rate_at_k": _first_split_metric(
                    validated.get("candidate_support_diagnostics"),
                    "candidate_hit_rate_at_k",
                ),
                "pmr_at_k_scaffold": _first_split_metric(
                    validated.get("movielens_pmr_scaffold_diagnostics"),
                    "pmr_at_k_scaffold",
                ),
                "ann_candidate_pmr_mean": _first_split_metric(
                    validated.get("movielens_ann_candidate_audit"),
                    "ann_candidate_pmr_mean",
                ),
            }
        )
    elif report_type == "d4rl_mock_experiment_scaffold":
        audit_metrics = _nested_get(validated, "d4rl_codebook_audit", "metrics") or {}
        pmr_metrics = _nested_get(validated, "d4rl_pmr_scaffold_diagnostics", "metrics") or {}
        ann_metrics = _nested_get(validated, "d4rl_ann_candidate_audit", "metrics") or {}
        task_metadata = validated.get("task_metadata", {})
        summary.update(
            {
                "num_transitions": config.get("num_transitions", task_metadata.get("num_transitions") if isinstance(task_metadata, dict) else None),
                "state_dim": config.get("state_dim", task_metadata.get("state_dim") if isinstance(task_metadata, dict) else None),
                "action_dim": config.get("action_dim", task_metadata.get("action_dim") if isinstance(task_metadata, dict) else None),
                "codebook_size": (
                    audit_metrics.get("codebook_size", config.get("codebook_size"))
                    if isinstance(audit_metrics, dict)
                    else config.get("codebook_size")
                ),
                "ann_candidate_recall_mean": (
                    ann_metrics.get("ann_candidate_recall_mean", audit_metrics.get("ann_candidate_recall_mean"))
                    if isinstance(ann_metrics, dict) and isinstance(audit_metrics, dict)
                    else None
                ),
                "ann_candidate_overlap_fraction_mean": (
                    ann_metrics.get(
                        "ann_candidate_overlap_fraction_mean",
                        audit_metrics.get("ann_candidate_overlap_fraction_mean"),
                    )
                    if isinstance(ann_metrics, dict) and isinstance(audit_metrics, dict)
                    else None
                ),
                "ann_candidate_best_value_gap_mean": (
                    ann_metrics.get(
                        "ann_candidate_best_value_gap_mean",
                        audit_metrics.get("ann_candidate_best_value_gap_mean"),
                    )
                    if isinstance(ann_metrics, dict) and isinstance(audit_metrics, dict)
                    else None
                ),
                "pmr_at_k_scaffold": pmr_metrics.get("pmr_at_k_scaffold") if isinstance(pmr_metrics, dict) else None,
                "missing_policy_mass_mean": (
                    pmr_metrics.get("missing_policy_mass_mean") if isinstance(pmr_metrics, dict) else None
                ),
                "logged_action_in_topk_rate": (
                    pmr_metrics.get("logged_action_in_topk_rate") if isinstance(pmr_metrics, dict) else None
                ),
                "ann_candidate_pmr_mean": (
                    ann_metrics.get("ann_candidate_pmr_mean") if isinstance(ann_metrics, dict) else None
                ),
            }
        )
    elif report_type == "recsim_mock_experiment_scaffold":
        metrics = _nested_get(validated, "recsim_scaffold_evaluation", "metrics") or {}
        pmr_metrics = _nested_get(validated, "recsim_pmr_scaffold_diagnostics", "metrics") or {}
        ann_metrics = _nested_get(validated, "recsim_ann_candidate_audit", "metrics") or {}
        slate_metadata = validated.get("slate_metadata", {})
        summary.update(
            {
                "num_users": metrics.get("num_users") if isinstance(metrics, dict) else config.get("num_users"),
                "num_items": metrics.get("num_items") if isinstance(metrics, dict) else config.get("num_items"),
                "num_interactions": (
                    metrics.get("num_interactions") if isinstance(metrics, dict) else config.get("num_interactions")
                ),
                "slate_size": (
                    metrics.get("slate_size")
                    if isinstance(metrics, dict)
                    else slate_metadata.get("slate_size") if isinstance(slate_metadata, dict) else config.get("slate_size")
                ),
                "observed_item_coverage": (
                    metrics.get("observed_item_coverage") if isinstance(metrics, dict) else None
                ),
                "avg_reward": metrics.get("avg_reward") if isinstance(metrics, dict) else None,
                "pmr_at_k_scaffold": pmr_metrics.get("pmr_at_k_scaffold") if isinstance(pmr_metrics, dict) else None,
                "missing_policy_mass_mean": (
                    pmr_metrics.get("missing_policy_mass_mean") if isinstance(pmr_metrics, dict) else None
                ),
                "logged_item_in_topk_rate": (
                    pmr_metrics.get("logged_item_in_topk_rate") if isinstance(pmr_metrics, dict) else None
                ),
                "ann_candidate_pmr_mean": (
                    ann_metrics.get("ann_candidate_pmr_mean") if isinstance(ann_metrics, dict) else None
                ),
                "ann_candidate_recall_mean": (
                    ann_metrics.get("ann_candidate_recall_mean") if isinstance(ann_metrics, dict) else None
                ),
                "ann_candidate_overlap_fraction_mean": (
                    ann_metrics.get("ann_candidate_overlap_fraction_mean") if isinstance(ann_metrics, dict) else None
                ),
                "ann_candidate_best_value_gap_mean": (
                    ann_metrics.get("ann_candidate_best_value_gap_mean") if isinstance(ann_metrics, dict) else None
                ),
            }
        )
    return summary


def load_external_scaffold_experiment_report_dict(report: dict[str, Any]) -> dict[str, Any]:
    """Validate an already parsed external scaffold experiment report dictionary."""

    if not isinstance(report, dict):
        raise ValueError("External scaffold experiment report must contain a JSON object.")
    if report.get("benchmark_claim") is not False:
        raise ValueError("External scaffold experiment report benchmark_claim must be false.")
    if report.get("scaffold_only") is not True:
        raise ValueError("External scaffold experiment report scaffold_only must be true.")
    if report.get("report_type") not in EXTERNAL_SCAFFOLD_EXPERIMENT_REPORT_TYPES:
        raise ValueError("Unsupported external scaffold experiment report_type.")
    if not isinstance(report.get("experiment_name"), str) or not report["experiment_name"]:
        raise ValueError("External scaffold experiment report must include experiment_name.")
    return report


def validate_external_support_protocol_config(
    config: ExternalSupportProtocolConfig,
) -> ExternalSupportProtocolConfig:
    """Validate a claim-free external support protocol config."""

    if not isinstance(config, ExternalSupportProtocolConfig):
        raise TypeError("config must be an ExternalSupportProtocolConfig.")
    if not isinstance(config.protocol_name, str) or not config.protocol_name:
        raise ValueError("protocol_name must be a non-empty string.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in ("num_steps", "batch_size", "train_k", "embedding_dim", "index_refresh_every", "target_update_every"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.k_values, tuple) or not config.k_values:
        raise ValueError("k_values must be a non-empty tuple.")
    if any(not isinstance(value, int) or value <= 0 for value in config.k_values):
        raise ValueError("k_values must contain positive integers.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of: exact, faiss, faiss_ivf, faiss_hnsw.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of: dot, cosine.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of: online, target, ema.")
    if config.index_encoder_policy == "target" and config.target_tau is None:
        raise ValueError("target index encoder policy requires target_tau.")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("ema index encoder policy requires index_ema_tau.")
    if config.index_ema_tau is not None and not 0.0 <= float(config.index_ema_tau) <= 1.0:
        raise ValueError("index_ema_tau must be in [0, 1].")
    if config.target_tau is not None and not 0.0 <= float(config.target_tau) <= 1.0:
        raise ValueError("target_tau must be None or in [0, 1].")
    for name in ("include_movielens", "include_d4rl_mock", "include_recsim_mock"):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"{name} must be bool.")
    if not (config.include_movielens or config.include_d4rl_mock or config.include_recsim_mock):
        raise ValueError("At least one external support protocol route must be included.")
    if config.include_movielens:
        if config.movielens_data_path is None:
            raise ValueError("include_movielens requires movielens_data_path.")
        if not Path(config.movielens_data_path).exists():
            raise FileNotFoundError(f"MovieLens CSV not found: {config.movielens_data_path}")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false.")
    return config


def load_external_support_protocol_config(path: str | Path) -> ExternalSupportProtocolConfig:
    """Load a claim-free external support protocol config from YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"External support protocol config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("External support protocol config YAML must contain a mapping.")
    allowed = set(ExternalSupportProtocolConfig.__dataclass_fields__)
    unknown = sorted(set(loaded) - allowed)
    if unknown:
        raise ValueError(f"Unknown external support protocol config field(s): {unknown}.")
    if "k_values" in loaded and loaded["k_values"] is not None:
        loaded["k_values"] = tuple(int(value) for value in loaded["k_values"])
    return validate_external_support_protocol_config(ExternalSupportProtocolConfig(**loaded))


def _result_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "task_name": result.task_name,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "synthetic_only": result.synthetic_only,
        "benchmark_claim": result.benchmark_claim,
        "scaffold_only": True,
    }


def _external_support_row(route: str, k: int, pmr: EvaluationResult, ann: EvaluationResult) -> dict[str, Any]:
    pmr_metrics = pmr.metrics
    ann_metrics = ann.metrics
    row = {
        "route": route,
        "k": k,
        "pmr_at_k_scaffold": pmr_metrics.get("pmr_at_k_scaffold"),
        "missing_policy_mass_mean": pmr_metrics.get("missing_policy_mass_mean"),
        "logged_or_positive_in_topk_rate": (
            pmr_metrics.get("positive_item_in_topk_rate")
            if "positive_item_in_topk_rate" in pmr_metrics
            else pmr_metrics.get("logged_action_in_topk_rate", pmr_metrics.get("logged_item_in_topk_rate"))
        ),
        "ann_candidate_pmr_mean": ann_metrics.get("ann_candidate_pmr_mean"),
        "ann_candidate_recall_mean": ann_metrics.get("ann_candidate_recall_mean"),
        "ann_candidate_overlap_fraction_mean": ann_metrics.get("ann_candidate_overlap_fraction_mean"),
        "ann_candidate_best_value_gap_mean": ann_metrics.get("ann_candidate_best_value_gap_mean"),
        "scaffold_only": True,
        "benchmark_claim": False,
    }
    return row


def _external_support_protocol_summary_table(summary: dict[str, Any]):
    from .reporting import TableReport

    rows = summarize_external_support_protocol(summary)
    columns = [
        "route",
        "k",
        "pmr_at_k_scaffold",
        "missing_policy_mass_mean",
        "logged_or_positive_in_topk_rate",
        "ann_candidate_pmr_mean",
        "ann_candidate_recall_mean",
        "ann_candidate_overlap_fraction_mean",
        "ann_candidate_best_value_gap_mean",
        "scaffold_only",
        "benchmark_claim",
    ]
    return TableReport(
        title="External Support Protocol Diagnostics",
        columns=columns,
        rows=rows,
        synthetic_only=False,
        benchmark_claim=False,
        metadata={"scaffold_only": True, "source": "external_support_protocol_summary"},
    )


def run_external_support_protocol(config: ExternalSupportProtocolConfig) -> dict[str, Any]:
    """Run a unified claim-free support protocol across external scaffold routes."""

    _require_torch()
    config = validate_external_support_protocol_config(config)
    root = Path(config.output_dir)
    if root.exists() and root.is_file():
        raise ValueError("output_dir must not be an existing file.")
    root.mkdir(parents=True, exist_ok=True)
    route_reports: dict[str, str] = {}
    route_summaries: dict[str, dict[str, Any]] = {}
    support_metrics: list[dict[str, Any]] = []
    included_routes: list[str] = []

    if config.include_d4rl_mock:
        result = run_d4rl_mock_experiment(
            D4RLMockExperimentConfig(
                experiment_name="d4rl_mock_support_protocol",
                output_dir=str(root / "d4rl_mock"),
                seed=config.seed,
                num_steps=config.num_steps,
                batch_size=config.batch_size,
                k=config.train_k,
                gamma=config.gamma,
                lr=config.lr,
                embedding_dim=config.embedding_dim,
                index_backend=config.index_backend,
                index_metric=config.index_metric,
                use_persistent_index=config.use_persistent_index,
                index_refresh_every=config.index_refresh_every,
                index_encoder_policy=config.index_encoder_policy,
                index_ema_tau=config.index_ema_tau,
                target_tau=config.target_tau,
                target_update_every=config.target_update_every,
            )
        )
        included_routes.append("d4rl_mock")
        route_reports["d4rl_mock"] = result.output_files["report"]
        route_summaries["d4rl_mock"] = summarize_external_scaffold_experiment_report(
            load_external_scaffold_experiment_report(result.output_files["report"])
        )
        for k in config.k_values:
            pmr = evaluate_d4rl_pmr_scaffold(result.task, result.model, k=k)
            ann = evaluate_d4rl_ann_candidate_audit(
                result.task,
                result.model,
                k=k,
                index_backend=config.index_backend,
                index_metric=config.index_metric,
            )
            support_metrics.append(_external_support_row("d4rl_mock", k, pmr, ann))

    if config.include_recsim_mock:
        result = run_recsim_mock_experiment(
            RecSimMockExperimentConfig(
                experiment_name="recsim_mock_support_protocol",
                output_dir=str(root / "recsim_mock"),
                seed=config.seed,
                num_steps=config.num_steps,
                batch_size=config.batch_size,
                k=config.train_k,
                gamma=config.gamma,
                lr=config.lr,
                embedding_dim=config.embedding_dim,
                index_backend=config.index_backend,
                index_metric=config.index_metric,
                use_persistent_index=config.use_persistent_index,
                index_refresh_every=config.index_refresh_every,
                index_encoder_policy=config.index_encoder_policy,
                index_ema_tau=config.index_ema_tau,
                target_tau=config.target_tau,
                target_update_every=config.target_update_every,
            )
        )
        included_routes.append("recsim_mock")
        route_reports["recsim_mock"] = result.output_files["report"]
        route_summaries["recsim_mock"] = summarize_external_scaffold_experiment_report(
            load_external_scaffold_experiment_report(result.output_files["report"])
        )
        for k in config.k_values:
            pmr = evaluate_recsim_pmr_scaffold(result.task, result.model, k=k)
            ann = evaluate_recsim_ann_candidate_audit(
                result.task,
                result.model,
                k=k,
                index_backend=config.index_backend,
                index_metric=config.index_metric,
            )
            support_metrics.append(_external_support_row("recsim_mock", k, pmr, ann))

    if config.include_movielens:
        result = run_movielens_local_experiment(
            MovieLensLocalExperimentConfig(
                experiment_name="movielens_local_support_protocol",
                data_path=str(config.movielens_data_path),
                output_dir=str(root / "movielens_local"),
                seed=config.seed,
                num_steps=config.num_steps,
                batch_size=config.batch_size,
                k=config.train_k,
                gamma=config.gamma,
                lr=config.lr,
                embedding_dim=config.embedding_dim,
                index_backend=config.index_backend,
                index_metric=config.index_metric,
                use_persistent_index=config.use_persistent_index,
                index_refresh_every=config.index_refresh_every,
                index_encoder_policy=config.index_encoder_policy,
                index_ema_tau=config.index_ema_tau,
                target_tau=config.target_tau,
                target_update_every=config.target_update_every,
                baseline_names=(),
            )
        )
        included_routes.append("movielens_local")
        route_reports["movielens_local"] = result.output_files["report"]
        route_summaries["movielens_local"] = summarize_external_scaffold_experiment_report(
            load_external_scaffold_experiment_report(result.output_files["report"])
        )
        split = "test" if result.task.get_split_rows("test") else "validation"
        if not result.task.get_split_rows(split):
            split = "train"
        for k in config.k_values:
            pmr = evaluate_movielens_pmr_scaffold(result.task, result.model, split=split, k=k)
            ann = evaluate_movielens_ann_candidate_audit(
                result.task,
                result.model,
                split=split,
                k=k,
                index_backend=config.index_backend,
                index_metric=config.index_metric,
            )
            support_metrics.append(_external_support_row("movielens_local", k, pmr, ann))

    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "external_support_protocol_summary",
        "protocol_name": config.protocol_name,
        "included_routes": included_routes,
        "k_values": list(config.k_values),
        "train_k": config.train_k,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "route_reports": route_reports,
        "route_summaries": route_summaries,
        "support_metrics_by_route_and_k": support_metrics,
        "metadata": {
            "no_performance_ranking": True,
            "no_benchmark_claims": True,
            "index_backend": config.index_backend,
            "index_metric": config.index_metric,
            "num_steps": config.num_steps,
            "seed": config.seed,
        },
    }
    summary_path = root / "summary.json"
    write_json_report(summary, summary_path)
    return {
        "output_dir": str(root),
        "summary_file": str(summary_path),
        "included_routes": included_routes,
        "route_reports": route_reports,
        "summary": summary,
        "scaffold_only": True,
        "benchmark_claim": False,
    }


def load_external_support_protocol_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate an external support protocol summary."""

    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"External support protocol summary not found: {summary_path}")
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid external support protocol summary JSON: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError("External support protocol summary must contain a JSON object.")
    if summary.get("report_type") != "external_support_protocol_summary":
        raise ValueError("External support protocol summary report_type is unsupported.")
    if summary.get("scaffold_only") is not True:
        raise ValueError("External support protocol summary scaffold_only must be true.")
    if summary.get("benchmark_claim") is not False:
        raise ValueError("External support protocol summary benchmark_claim must be false.")
    if not isinstance(summary.get("route_summaries"), dict):
        raise ValueError("External support protocol summary must include route_summaries.")
    if not isinstance(summary.get("support_metrics_by_route_and_k"), list):
        raise ValueError("External support protocol summary must include support_metrics_by_route_and_k.")
    return summary


def summarize_external_support_protocol(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact table rows from an external support protocol summary."""

    validated = summary if summary.get("report_type") == "external_support_protocol_summary" else None
    if validated is None:
        raise ValueError("summary must be an external_support_protocol_summary.")
    rows = validated.get("support_metrics_by_route_and_k")
    if not isinstance(rows, list):
        raise ValueError("summary must include support_metrics_by_route_and_k.")
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("support metric rows must be dictionaries.")
        compact_rows.append(
            {
                "route": row.get("route"),
                "k": row.get("k"),
                "pmr_at_k_scaffold": row.get("pmr_at_k_scaffold"),
                "missing_policy_mass_mean": row.get("missing_policy_mass_mean"),
                "logged_or_positive_in_topk_rate": row.get("logged_or_positive_in_topk_rate"),
                "ann_candidate_pmr_mean": row.get("ann_candidate_pmr_mean"),
                "ann_candidate_recall_mean": row.get("ann_candidate_recall_mean"),
                "ann_candidate_overlap_fraction_mean": row.get("ann_candidate_overlap_fraction_mean"),
                "ann_candidate_best_value_gap_mean": row.get("ann_candidate_best_value_gap_mean"),
                "scaffold_only": row.get("scaffold_only"),
                "benchmark_claim": row.get("benchmark_claim"),
            }
        )
    return compact_rows


def external_support_protocol_summary_to_table(summary: dict[str, Any]):
    """Convert an external support protocol summary to a claim-free table."""

    return _external_support_protocol_summary_table(summary)


def _write_external_experiment_bundle_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("External scaffold experiment bundle manifest path must be a file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_external_scaffold_experiment_bundle(
    output_dir: str | Path,
    movielens_data_path: str | Path | None = None,
    num_steps: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a tiny claim-free bundle of external scaffold experiments."""

    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    root = Path(output_dir)
    if root.exists() and root.is_file():
        raise ValueError("output_dir must not be an existing file.")
    root.mkdir(parents=True, exist_ok=True)

    d4rl_result = run_d4rl_mock_experiment(
        D4RLMockExperimentConfig(
            experiment_name="d4rl_mock_bundle",
            output_dir=str(root / "d4rl_mock"),
            seed=seed,
            num_steps=num_steps,
            batch_size=4,
            k=3,
            embedding_dim=8,
            num_transitions=16,
            state_dim=4,
            action_dim=3,
            codebook_size=8,
            target_tau=0.05,
        )
    )
    recsim_result = run_recsim_mock_experiment(
        RecSimMockExperimentConfig(
            experiment_name="recsim_mock_bundle",
            output_dir=str(root / "recsim_mock"),
            seed=seed,
            num_steps=num_steps,
            batch_size=4,
            k=3,
            embedding_dim=8,
            num_users=3,
            num_items=8,
            user_feature_dim=4,
            item_feature_dim=4,
            slate_size=2,
            num_interactions=16,
            target_tau=0.05,
        )
    )
    included = ["d4rl_mock", "recsim_mock"]
    report_files = [d4rl_result.output_files["report"], recsim_result.output_files["report"]]
    manifest_files = [d4rl_result.output_files["manifest"], recsim_result.output_files["manifest"]]

    if movielens_data_path is not None:
        movie_path = Path(movielens_data_path)
        if not movie_path.exists():
            raise FileNotFoundError(f"MovieLens CSV not found: {movie_path}")
        movielens_result = run_movielens_local_experiment(
            MovieLensLocalExperimentConfig(
                experiment_name="movielens_local_bundle",
                data_path=str(movie_path),
                output_dir=str(root / "movielens_local"),
                seed=seed,
                num_steps=num_steps,
                batch_size=3,
                k=2,
                embedding_dim=8,
                split_mode="chronological",
                validation_fraction=0.2,
                test_fraction=0.2,
                target_tau=0.05,
                baseline_names=(),
            )
        )
        included.append("movielens_local")
        report_files.append(movielens_result.output_files["report"])
        manifest_files.append(movielens_result.output_files["manifest"])

    manifest_path = root / "external_scaffold_experiment_bundle_manifest.json"
    manifest = {
        "schema_name": "dwmql_external_scaffold_experiment_bundle_manifest",
        "schema_version": 1,
        "run_type": "external_scaffold_experiment_bundle",
        "scaffold_only": True,
        "benchmark_claim": False,
        "output_dir": str(root),
        "included_experiments": included,
        "report_files": report_files,
        "manifest_files": manifest_files,
        "metadata": {
            "num_steps": num_steps,
            "seed": seed,
            "movielens_included": movielens_data_path is not None,
            "no_real_d4rl_loading": True,
            "no_real_recsim_loading": True,
            "no_benchmark_claims": True,
        },
    }
    _write_external_experiment_bundle_manifest(manifest_path, manifest)
    return {
        "output_dir": str(root),
        "manifest_file": str(manifest_path),
        "report_files": report_files,
        "manifest_files": manifest_files,
        "included_experiments": included,
        "benchmark_claim": False,
        "scaffold_only": True,
    }


def load_external_scaffold_experiment_bundle_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate an external scaffold experiment bundle manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"External scaffold experiment bundle manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid external scaffold experiment bundle manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("External scaffold experiment bundle manifest must contain a JSON object.")
    if manifest.get("run_type") != "external_scaffold_experiment_bundle":
        raise ValueError("External scaffold experiment bundle manifest run_type is unsupported.")
    if manifest.get("scaffold_only") is not True:
        raise ValueError("External scaffold experiment bundle manifest scaffold_only must be true.")
    if manifest.get("benchmark_claim") is not False:
        raise ValueError("External scaffold experiment bundle manifest benchmark_claim must be false.")
    if not isinstance(manifest.get("report_files"), list):
        raise ValueError("External scaffold experiment bundle manifest must include report_files as a list.")
    if not isinstance(manifest.get("included_experiments"), list):
        raise ValueError("External scaffold experiment bundle manifest must include included_experiments as a list.")
    return manifest


def inspect_external_scaffold_experiment_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """Validate and summarize every report in an external scaffold experiment bundle."""

    manifest_file = Path(manifest_path)
    manifest = load_external_scaffold_experiment_bundle_manifest(manifest_file)
    base_dir = manifest_file.parent
    summaries: list[dict[str, Any]] = []
    for report_file in manifest["report_files"]:
        report_path = Path(report_file)
        if not report_path.is_absolute() and not report_path.exists():
            report_path = base_dir / report_path
        report = load_external_scaffold_experiment_report(report_path)
        summaries.append(summarize_external_scaffold_experiment_report(report))
    return {
        "manifest_path": str(manifest_file),
        "output_dir": manifest.get("output_dir"),
        "report_count": len(summaries),
        "included_experiments": list(manifest["included_experiments"]),
        "summaries": summaries,
        "scaffold_only": True,
        "benchmark_claim": False,
    }


def _safe_float_label(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def validate_experiment_sweep_config(config: ExperimentSweepConfig) -> ExperimentSweepConfig:
    """Validate a synthetic experiment sweep config."""

    if not isinstance(config, ExperimentSweepConfig):
        raise TypeError("config must be an ExperimentSweepConfig.")
    validate_experiment_config(config.base_config)
    if not isinstance(config.seeds, tuple) or not config.seeds:
        raise ValueError("seeds must be a non-empty tuple.")
    if any(not isinstance(seed, int) or seed < 0 for seed in config.seeds):
        raise ValueError("seeds must contain nonnegative integers.")
    if config.k_values is not None:
        if not isinstance(config.k_values, tuple) or not config.k_values:
            raise ValueError("k_values must be None or a non-empty tuple.")
        for value in config.k_values:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("k_values must contain positive integers.")
            if value > config.base_config.synthetic.num_actions:
                raise ValueError("k_values must not exceed synthetic.num_actions.")
    if config.index_refresh_values is not None:
        if not isinstance(config.index_refresh_values, tuple) or not config.index_refresh_values:
            raise ValueError("index_refresh_values must be None or a non-empty tuple.")
        if any(not isinstance(value, int) or value <= 0 for value in config.index_refresh_values):
            raise ValueError("index_refresh_values must contain positive integers.")
    if config.index_encoder_policies is not None:
        if not isinstance(config.index_encoder_policies, tuple) or not config.index_encoder_policies:
            raise ValueError("index_encoder_policies must be None or a non-empty tuple.")
        if any(policy not in {"online", "target", "ema"} for policy in config.index_encoder_policies):
            raise ValueError("index_encoder_policies must contain only online, target, or ema.")
    if config.index_ema_tau_values is not None:
        if not isinstance(config.index_ema_tau_values, tuple) or not config.index_ema_tau_values:
            raise ValueError("index_ema_tau_values must be None or a non-empty tuple.")
        for value in config.index_ema_tau_values:
            if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("index_ema_tau_values must contain floats in [0, 1].")
    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
        if output_dir.exists() and output_dir.is_file():
            raise ValueError("output_dir must not be an existing file.")
    if not isinstance(config.sweep_name, str) or not config.sweep_name:
        raise ValueError("sweep_name must be a non-empty string.")
    policies = config.index_encoder_policies or (config.base_config.index_encoder_policy,)
    if "ema" in policies and config.index_ema_tau_values is None and config.base_config.index_ema_tau is None:
        raise ValueError("EMA sweep policy requires index_ema_tau_values or base_config.index_ema_tau.")
    return config


def _report_for_experiment(
    config: ExperimentConfig,
    history: TrainingHistory,
    evaluation: EvaluationResult,
    evaluation_summary: EvaluationSummary,
    evaluation_config: EvaluationConfig,
    baseline_results: list[BaselineResult],
    checkpoint_files: list[str],
) -> dict[str, Any]:
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    history_dict = training_history_to_dict(history)
    return {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_experiment",
        "experiment_name": config.experiment_name,
        "synthetic_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "checkpoint_files": checkpoint_files,
        "resume_from": config.resume_from,
        "evaluation_config": asdict(evaluation_config),
        "summary": {
            "num_steps": config.num_steps,
            "final_total_loss": history.losses[-1],
            "final_bellman_loss": history.bellman_losses[-1],
            "final_stationarity_loss": history.stationarity_losses[-1],
            "final_density_regularization": history.density_regularizations[-1],
            "target_update_count": history.diagnostics.get("target_update_count"),
            "index_refresh_count": history.index_refresh_count,
            "final_stale_index_pmr_mean": history.diagnostics.get("final_stale_index_pmr_mean"),
            "final_stale_index_pmr_normalized_mean": history.diagnostics.get("final_stale_index_pmr_normalized_mean"),
            "final_stale_index_missing_mass_mean": history.diagnostics.get("final_stale_index_missing_mass_mean"),
            "retrieval_correction_mode": config.retrieval_correction_mode,
            "checkpoint_count": len(checkpoint_files),
        },
        "training_history": history_dict,
        "final_training_diagnostics": history.diagnostics,
        "evaluation": asdict(evaluation),
        "evaluation_summary": asdict(evaluation_summary),
        "baseline_diagnostics": [asdict(result) for result in baseline_results],
        "baseline_diagnostics_only": bool(baseline_results),
    }


def _movielens_data_for_trainer(task: MovieLensLocalTask | KuaiRandLocalTask) -> MovieLensTrainerData:
    """Build a trainer-compatible MovieLens data view.

    The public action catalogue preserves MovieLens movie ids, while the ANN
    trainer expects candidate ids to be compact row indices into
    ``action_features``. This view keeps the local-file contract intact and
    supplies compact ids only to the training scaffold.
    """

    bundle = task.make_dataset_bundle()
    dataset = bundle.dataset
    catalogue = bundle.action_catalogue
    actual_ids = [int(value) for value in catalogue.action_ids.tolist()]
    id_to_row = {movie_id: index for index, movie_id in enumerate(actual_ids)}

    def compact(values: Any) -> Any:
        converted = [id_to_row[int(value)] for value in values.tolist()]
        return torch.tensor(converted, dtype=torch.long, device=dataset.batch.states.device)

    compact_dataset = InMemoryReplayDataset(
        states=dataset.batch.states,
        actions=dataset.batch.actions,
        rewards=dataset.batch.rewards,
        next_states=dataset.batch.next_states,
        dones=dataset.batch.dones,
        action_ids=compact(dataset.batch.action_ids),
        next_logged_action_ids=compact(dataset.batch.next_logged_action_ids),
        metadata={**dataset.batch.metadata, "compact_action_ids_for_training": True},
    )
    return MovieLensTrainerData(
        dataset=compact_dataset,
        action_features=catalogue.action_features,
        action_ids=torch.arange(catalogue.num_actions, dtype=torch.long, device=catalogue.action_features.device),
        metadata={
            **bundle.metadata,
            "compact_action_ids_for_training": True,
            "original_action_ids_preserved_in_task_catalogue": True,
        },
    )


def summarize_critic_steps(
    history: Mapping[str, Any],
    max_grad_norm: float | None,
) -> dict[str, Any]:
    """Summarize only steps that actually optimized the critic."""

    diagnostics = history.get("diagnostics", {})
    objective_names = list(diagnostics.get("objective_names", []))
    critic_indices = [index for index, name in enumerate(objective_names) if name == "critic"]

    def selected(key: str) -> list[float]:
        values = list(history.get(key, []))
        return [
            float(values[index])
            for index in critic_indices
            if index < len(values) and values[index] is not None
        ]

    def final(values: list[float]) -> float | None:
        return values[-1] if values else None

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    optimized = selected("losses")
    critic_total = selected("critic_total_losses")
    support_weighted = selected("support_weighted_bellman_losses")
    raw_bellman = selected("bellman_losses")
    grad_norms = selected("grad_norms")
    clip_fraction = None
    if max_grad_norm is not None and grad_norms:
        threshold = float(max_grad_norm)
        clip_fraction = sum(value >= threshold for value in grad_norms) / len(grad_norms)
    return {
        "num_critic_updates": len(critic_indices),
        "final_critic_optimized_loss": final(optimized),
        "mean_critic_optimized_loss": mean(optimized),
        "final_critic_total_loss": final(critic_total),
        "mean_critic_total_loss": mean(critic_total),
        "final_support_weighted_bellman_loss": final(support_weighted),
        "mean_support_weighted_bellman_loss": mean(support_weighted),
        "final_raw_bellman_loss": final(raw_bellman),
        "mean_raw_bellman_loss": mean(raw_bellman),
        "final_critic_grad_norm": final(grad_norms),
        "mean_critic_grad_norm": mean(grad_norms),
        "critic_grad_clip_fraction": clip_fraction,
        "critic_total_loss_available": bool(critic_total),
        "support_weighted_bellman_loss_available": bool(support_weighted),
        "raw_bellman_loss_available": bool(raw_bellman),
    }


def _movielens_history_summary(
    history: TrainingHistory,
    *,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    history_dict = training_history_to_dict(history)
    return {
        "num_steps": len(history.losses),
        "final_stationarity_loss": history.stationarity_losses[-1] if history.stationarity_losses else None,
        "final_density_regularization": (
            history.density_regularizations[-1] if history.density_regularizations else None
        ),
        "target_update_count": history.diagnostics.get("target_update_count"),
        "index_refresh_count": history.index_refresh_count,
        "final_index_age": history.diagnostics.get("final_index_age"),
        "final_stale_index_pmr_mean": history.diagnostics.get("final_stale_index_pmr_mean"),
        **summarize_critic_steps(history_dict, max_grad_norm=max_grad_norm),
    }


def _movielens_baseline_diagnostics(
    config: MovieLensLocalExperimentConfig,
    task: MovieLensLocalTask,
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    dataset = task.make_dataset()
    action_features, action_ids = task.make_action_catalogue()
    for name in config.baseline_names:
        baseline = get_baseline(name, seed=config.seed)
        baseline.fit(dataset, action_features=action_features, action_ids=action_ids)
        evaluation = evaluate_movielens_scaffold_task(task, model_or_baseline=baseline, k=config.k)
        results.append(
            BaselineResult(
                baseline_name=baseline.name,
                metrics=dict(evaluation.metrics),
                diagnostics={
                    **evaluation.diagnostics,
                    "scaffold_only": True,
                    "baseline_diagnostics_only": True,
                    "synthetic_only": False,
                    "benchmark_claim": False,
                },
                synthetic_only=False,
                benchmark_claim=False,
            )
        )
    return results


def _movielens_split_diagnostics(
    task: MovieLensLocalTask,
    *,
    k: int,
    model_or_baseline: Any = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for split in ("validation", "test"):
        if not task.get_split_rows(split):
            continue
        ranking = evaluate_movielens_ranking_scaffold(
            task,
            model_or_baseline=model_or_baseline,
            split=split,
            k=k,
        )
        support = evaluate_movielens_candidate_support_scaffold(
            task,
            model_or_baseline=model_or_baseline,
            split=split,
            k=k,
        )
        diagnostics[split] = {
            "ranking": {
                "metrics": ranking.metrics,
                "diagnostics": ranking.diagnostics,
                "synthetic_only": ranking.synthetic_only,
                "benchmark_claim": ranking.benchmark_claim,
                "scaffold_only": True,
            },
            "candidate_support": {
                "metrics": support.metrics,
                "diagnostics": support.diagnostics,
                "synthetic_only": support.synthetic_only,
                "benchmark_claim": support.benchmark_claim,
                "scaffold_only": True,
            },
        }
    return diagnostics


def _movielens_pmr_diagnostics(
    task: MovieLensLocalTask,
    model: Any,
    *,
    k: int,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for split in ("validation", "test"):
        if not task.get_split_rows(split):
            continue
        result = evaluate_movielens_pmr_scaffold(task, model, split=split, k=k)
        diagnostics[split] = {
            "metrics": result.metrics,
            "diagnostics": result.diagnostics,
            "synthetic_only": result.synthetic_only,
            "benchmark_claim": result.benchmark_claim,
            "scaffold_only": True,
        }
    return diagnostics


def _movielens_ann_candidate_audit_diagnostics(
    task: MovieLensLocalTask,
    model: Any,
    config: MovieLensLocalExperimentConfig,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for split in ("validation", "test"):
        if not task.get_split_rows(split):
            continue
        result = evaluate_movielens_ann_candidate_audit(
            task,
            model,
            split=split,
            k=config.k,
            index_backend=config.index_backend,
            index_metric=config.index_metric,
        )
        diagnostics[split] = {
            "metrics": result.metrics,
            "diagnostics": result.diagnostics,
            "synthetic_only": result.synthetic_only,
            "benchmark_claim": result.benchmark_claim,
            "scaffold_only": True,
        }
    return diagnostics


def _movielens_baseline_ranking_diagnostics(
    config: MovieLensLocalExperimentConfig,
    task: MovieLensLocalTask,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    dataset = task.make_dataset()
    action_features, action_ids = task.make_action_catalogue()
    for name in config.baseline_names:
        baseline = get_baseline(name, seed=config.seed)
        baseline.fit(dataset, action_features=action_features, action_ids=action_ids)
        diagnostics[baseline.name] = _movielens_split_diagnostics(task, k=config.k, model_or_baseline=baseline)
    return diagnostics


def _write_movielens_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("MovieLens manifest path must be a file, not a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def make_d4rl_mock_arrays(config: D4RLMockExperimentConfig) -> dict[str, Any]:
    """Create deterministic finite continuous-action mock arrays."""

    _require_torch()
    config = validate_d4rl_mock_experiment_config(config)
    generator = torch.Generator().manual_seed(int(config.seed))
    observations = torch.randn(config.num_transitions, config.state_dim, generator=generator)
    action_projection = torch.randn(config.state_dim, config.action_dim, generator=generator)
    action_noise = 0.05 * torch.randn(config.num_transitions, config.action_dim, generator=generator)
    actions = torch.tanh(observations @ action_projection + action_noise)
    transition_projection = torch.randn(config.action_dim, config.state_dim, generator=generator)
    next_observations = observations + 0.1 * (actions @ transition_projection)
    reward_projection = torch.randn(config.action_dim, generator=generator)
    state_reward_projection = torch.randn(config.state_dim, generator=generator)
    rewards = torch.tanh(actions @ reward_projection + 0.25 * (observations @ state_reward_projection))
    terminals = ((torch.arange(config.num_transitions) + 1) % max(2, config.batch_size) == 0)
    arrays = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminals": terminals,
    }
    for name, tensor in arrays.items():
        if name != "terminals" and torch.any(~torch.isfinite(tensor)):
            raise ValueError(f"{name} must contain only finite values.")
    return arrays


def _d4rl_data_for_trainer(task: D4RLLargeActionTask) -> D4RLTrainerData:
    bundle = task.make_dataset_bundle()
    catalogue = bundle.action_catalogue
    return D4RLTrainerData(
        dataset=bundle.dataset,
        action_features=catalogue.action_features,
        action_ids=catalogue.action_ids,
        metadata={**bundle.metadata, "d4rl_mock_trainer_view": True},
    )


def _d4rl_history_summary(history: TrainingHistory) -> dict[str, Any]:
    return {
        "num_steps": len(history.losses),
        "final_total_loss": history.losses[-1] if history.losses else None,
        "final_bellman_loss": history.bellman_losses[-1] if history.bellman_losses else None,
        "final_stationarity_loss": history.stationarity_losses[-1] if history.stationarity_losses else None,
        "final_density_regularization": (
            history.density_regularizations[-1] if history.density_regularizations else None
        ),
        "target_update_count": history.diagnostics.get("target_update_count"),
        "index_refresh_count": history.index_refresh_count,
        "final_index_age": history.diagnostics.get("final_index_age"),
        "final_stale_index_pmr_mean": history.diagnostics.get("final_stale_index_pmr_mean"),
    }


def _write_d4rl_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("D4RL mock manifest path must be a file, not a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _d4rl_mock_experiment_report(
    config: D4RLMockExperimentConfig,
    task: D4RLLargeActionTask,
    history: TrainingHistory,
    audit: EvaluationResult,
    pmr: EvaluationResult,
    ann_audit: EvaluationResult,
    output_files: dict[str, str],
) -> dict[str, Any]:
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    return {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "d4rl_mock_experiment_scaffold",
        "mode": "d4rl_mock_experiment_scaffold",
        "experiment_name": config.experiment_name,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "task_metadata": task.metadata,
        "training_history_summary": _d4rl_history_summary(history),
        "training_history": training_history_to_dict(history),
        "d4rl_codebook_audit": {
            "task_name": audit.task_name,
            "metrics": audit.metrics,
            "diagnostics": audit.diagnostics,
            "synthetic_only": audit.synthetic_only,
            "benchmark_claim": audit.benchmark_claim,
            "scaffold_only": True,
        },
        "d4rl_pmr_scaffold_diagnostics": {
            "task_name": pmr.task_name,
            "metrics": pmr.metrics,
            "diagnostics": pmr.diagnostics,
            "synthetic_only": pmr.synthetic_only,
            "benchmark_claim": pmr.benchmark_claim,
            "scaffold_only": True,
        },
        "d4rl_ann_candidate_audit": {
            "task_name": ann_audit.task_name,
            "metrics": ann_audit.metrics,
            "diagnostics": ann_audit.diagnostics,
            "synthetic_only": ann_audit.synthetic_only,
            "benchmark_claim": ann_audit.benchmark_claim,
            "scaffold_only": True,
        },
        "output_files": output_files,
        "limitations": [
            "mock continuous-action arrays only",
            "no real D4RL loading",
            "no Gymnasium or Minari dependency",
            "no offline-control benchmark claims",
            "objective training machinery is reused as a controlled scaffold",
        ],
    }


def run_d4rl_mock_experiment(config: D4RLMockExperimentConfig) -> D4RLMockExperimentResult:
    """Run a claim-free D4RL mock large-action experiment scaffold."""

    _require_torch()
    config = validate_d4rl_mock_experiment_config(config)
    arrays = make_d4rl_mock_arrays(config)
    codebook: D4RLActionCodebook = build_d4rl_action_codebook_from_actions(
        arrays["actions"],
        max_actions=config.codebook_size,
        seed=config.seed,
    )
    task = D4RLAdapter(
        observations=arrays["observations"],
        actions=arrays["actions"],
        rewards=arrays["rewards"],
        next_observations=arrays["next_observations"],
        terminals=arrays["terminals"],
        codebook=codebook,
    ).make_task()
    if not isinstance(task, D4RLLargeActionTask):
        raise TypeError("D4RL mock adapter did not return D4RLLargeActionTask.")
    assert_benchmark_task_contract(task)
    trainer_data = _d4rl_data_for_trainer(task)

    from .networks import build_networks

    torch.manual_seed(config.seed)
    model = build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=config.embedding_dim,
        )
    )
    optimizer_bundle = make_optimizer_bundle(model, lr=config.lr, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    trainer = SyntheticDWMQLTrainer(
        data=trainer_data,  # type: ignore[arg-type]
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=config.k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
    )
    history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))
    audit = evaluate_d4rl_action_codebook_audit(
        task,
        model_or_scores=model,
        k=config.k,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
    )
    pmr = evaluate_d4rl_pmr_scaffold(task, model, k=config.k)
    ann_audit = evaluate_d4rl_ann_candidate_audit(
        task,
        model,
        k=config.k,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.experiment_name}.json"
    manifest_path = output_dir / "manifest.json"
    output_files = {"report": str(report_path), "manifest": str(manifest_path)}
    report = _d4rl_mock_experiment_report(config, task, history, audit, pmr, ann_audit, output_files)
    write_json_report(report, report_path)
    _write_d4rl_manifest(
        manifest_path,
        {
            "schema_name": "dwmql_d4rl_mock_experiment_manifest",
            "schema_version": 1,
            "run_type": "d4rl_mock_experiment_scaffold",
            "output_dir": str(output_dir),
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "report_files": [str(report_path)],
            "metadata": {
                "experiment_name": config.experiment_name,
                "task_name": task.name,
                "task_type": task.task_type,
                "codebook_size": task.num_actions,
                "no_real_d4rl_loading": True,
            },
        },
    )
    return D4RLMockExperimentResult(
        config=config,
        training_history=history,
        evaluation_result=audit,
        output_files=output_files,
        diagnostics={
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "task_name": task.name,
            "task_type": task.task_type,
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "pmr_at_k_scaffold": pmr.metrics.get("pmr_at_k_scaffold"),
            "ann_candidate_pmr_mean": ann_audit.metrics.get("ann_candidate_pmr_mean"),
            "ann_candidate_recall_mean": ann_audit.metrics.get("ann_candidate_recall_mean"),
        },
        task=task,
        model=model,
    )


def _recsim_history_summary(history: TrainingHistory) -> dict[str, Any]:
    return {
        "num_steps": len(history.losses),
        "final_total_loss": history.losses[-1] if history.losses else None,
        "final_bellman_loss": history.bellman_losses[-1] if history.bellman_losses else None,
        "final_stationarity_loss": history.stationarity_losses[-1] if history.stationarity_losses else None,
        "final_density_regularization": (
            history.density_regularizations[-1] if history.density_regularizations else None
        ),
        "target_update_count": history.diagnostics.get("target_update_count"),
        "index_refresh_count": history.index_refresh_count,
        "final_index_age": history.diagnostics.get("final_index_age"),
        "final_stale_index_pmr_mean": history.diagnostics.get("final_stale_index_pmr_mean"),
    }


def _recsim_candidate_support_scaffold(task: RecSimMockSlateTask, model: Any, k: int) -> dict[str, Any]:
    dataset = task.make_dataset()
    action_features, _ = task.make_action_catalogue()
    safe_k = min(k, task.num_actions)
    with torch.no_grad():
        state_embeddings = model.encode_states(dataset.batch.states)
        action_embeddings = model.encode_actions(action_features)
        expanded_actions = action_embeddings.unsqueeze(0).expand(state_embeddings.shape[0], -1, -1)
        scores = model.q_values(state_embeddings, expanded_actions)
    topk_ids = torch.topk(scores, k=safe_k, dim=1).indices
    logged_ids = dataset.batch.action_ids.view(-1, 1)
    hits = (topk_ids == logged_ids).any(dim=1).float()
    observed = torch.unique(dataset.batch.action_ids)
    return {
        "candidate_hit_rate_at_k": float(hits.mean().item()),
        "catalogue_size": task.num_actions,
        "k": safe_k,
        "num_eval_rows": len(dataset),
        "candidate_coverage_fraction": float(observed.numel() / task.num_actions),
        "benchmark_claim": False,
        "scaffold_only": True,
    }


def _recsim_mock_experiment_report(
    config: RecSimMockExperimentConfig,
    task: RecSimMockSlateTask,
    history: TrainingHistory,
    evaluation: EvaluationResult,
    pmr: EvaluationResult,
    ann_audit: EvaluationResult,
    candidate_support: dict[str, Any],
    output_files: dict[str, str],
) -> dict[str, Any]:
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    return {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "recsim_mock_experiment_scaffold",
        "mode": "recsim_mock_experiment_scaffold",
        "experiment_name": config.experiment_name,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "task_metadata": task.metadata,
        "slate_metadata": asdict(task.slate_metadata),
        "training_history_summary": _recsim_history_summary(history),
        "training_history": training_history_to_dict(history),
        "recsim_scaffold_evaluation": {
            "task_name": evaluation.task_name,
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "synthetic_only": evaluation.synthetic_only,
            "benchmark_claim": evaluation.benchmark_claim,
            "scaffold_only": True,
        },
        "recsim_pmr_scaffold_diagnostics": {
            "task_name": pmr.task_name,
            "metrics": pmr.metrics,
            "diagnostics": pmr.diagnostics,
            "synthetic_only": pmr.synthetic_only,
            "benchmark_claim": pmr.benchmark_claim,
            "scaffold_only": True,
        },
        "recsim_ann_candidate_audit": {
            "task_name": ann_audit.task_name,
            "metrics": ann_audit.metrics,
            "diagnostics": ann_audit.diagnostics,
            "synthetic_only": ann_audit.synthetic_only,
            "benchmark_claim": ann_audit.benchmark_claim,
            "scaffold_only": True,
        },
        "candidate_support_scaffold_diagnostics": candidate_support,
        "output_files": output_files,
        "limitations": [
            "mock slate data only",
            "no real RecSim or RecSim NG loading",
            "no long-horizon simulator evaluation",
            "no NDCG/CTR evaluation",
            "no slate RL training claims",
        ],
    }


def run_recsim_mock_experiment(config: RecSimMockExperimentConfig) -> RecSimMockExperimentResult:
    """Run a claim-free RecSim mock slate experiment scaffold."""

    _require_torch()
    config = validate_recsim_mock_experiment_config(config)
    task = RecSimMockSlateTask(
        num_users=config.num_users,
        num_items=config.num_items,
        user_feature_dim=config.user_feature_dim,
        item_feature_dim=config.item_feature_dim,
        slate_size=config.slate_size,
        num_interactions=config.num_interactions,
        seed=config.seed,
    )
    assert_benchmark_task_contract(task)
    bundle = task.make_dataset_bundle()

    from .networks import build_networks

    torch.manual_seed(config.seed)
    model = build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=config.embedding_dim,
        )
    )
    optimizer_bundle = make_optimizer_bundle(model, lr=config.lr, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    trainer_data = D4RLTrainerData(
        dataset=bundle.dataset,
        action_features=bundle.action_catalogue.action_features,
        action_ids=bundle.action_catalogue.action_ids,
        metadata={**bundle.metadata, "recsim_mock_trainer_view": True},
    )
    trainer = SyntheticDWMQLTrainer(
        data=trainer_data,  # type: ignore[arg-type]
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=config.k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
    )
    history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))
    evaluation = evaluate_recsim_slate_scaffold(task, model_or_baseline=None, k=config.k)
    pmr = evaluate_recsim_pmr_scaffold(task, model, k=config.k)
    ann_audit = evaluate_recsim_ann_candidate_audit(
        task,
        model,
        k=config.k,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
    )
    candidate_support = _recsim_candidate_support_scaffold(task, model, config.k)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.experiment_name}.json"
    manifest_path = output_dir / "manifest.json"
    output_files = {"report": str(report_path), "manifest": str(manifest_path)}
    report = _recsim_mock_experiment_report(
        config,
        task,
        history,
        evaluation,
        pmr,
        ann_audit,
        candidate_support,
        output_files,
    )
    write_json_report(report, report_path)
    _write_d4rl_manifest(
        manifest_path,
        {
            "schema_name": "dwmql_recsim_mock_experiment_manifest",
            "schema_version": 1,
            "run_type": "recsim_mock_experiment_scaffold",
            "output_dir": str(output_dir),
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "report_files": [str(report_path)],
            "metadata": {
                "experiment_name": config.experiment_name,
                "task_name": task.name,
                "task_type": task.task_type,
                "slate_size": task.slate_size,
                "num_items": task.num_actions,
                "no_real_recsim_loading": True,
            },
        },
    )
    return RecSimMockExperimentResult(
        config=config,
        training_history=history,
        evaluation_result=evaluation,
        output_files=output_files,
        diagnostics={
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "task_name": task.name,
            "task_type": task.task_type,
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "pmr_at_k_scaffold": pmr.metrics.get("pmr_at_k_scaffold"),
            "ann_candidate_pmr_mean": ann_audit.metrics.get("ann_candidate_pmr_mean"),
            "ann_candidate_recall_mean": ann_audit.metrics.get("ann_candidate_recall_mean"),
        },
        task=task,
        model=model,
    )


def _movielens_experiment_report(
    config: MovieLensLocalExperimentConfig,
    task: MovieLensLocalTask,
    history: TrainingHistory,
    evaluation: EvaluationResult,
    split_diagnostics: dict[str, Any],
    pmr_diagnostics: dict[str, Any],
    ann_candidate_audit: dict[str, Any],
    baseline_ranking_diagnostics: dict[str, Any],
    baseline_results: list[BaselineResult],
    output_files: dict[str, str],
) -> dict[str, Any]:
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    return {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "movielens_local_experiment_scaffold",
        "mode": "movielens_local_experiment_scaffold",
        "experiment_name": config.experiment_name,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "dataset_metadata": task.metadata,
        "training_history_summary": _movielens_history_summary(
            history,
            max_grad_norm=getattr(config, "max_grad_norm", None),
        ),
        "training_history": training_history_to_dict(history),
        "scaffold_evaluation": {
            "task_name": evaluation.task_name,
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "synthetic_only": evaluation.synthetic_only,
            "benchmark_claim": evaluation.benchmark_claim,
            "scaffold_only": evaluation.diagnostics.get("scaffold_evaluation", False),
        },
        "evaluation_summary": {
            "task_name": evaluation.task_name,
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "synthetic_only": evaluation.synthetic_only,
            "benchmark_claim": evaluation.benchmark_claim,
            "scaffold_only": evaluation.diagnostics.get("scaffold_evaluation", False),
        },
        "ranking_scaffold_diagnostics": {
            split: payload["ranking"] for split, payload in split_diagnostics.items()
        },
        "candidate_support_diagnostics": {
            split: payload["candidate_support"] for split, payload in split_diagnostics.items()
        },
        "movielens_pmr_scaffold_diagnostics": pmr_diagnostics,
        "movielens_ann_candidate_audit": ann_candidate_audit,
        "baseline_diagnostics": [asdict(result) for result in baseline_results],
        "baseline_ranking_scaffold_diagnostics": baseline_ranking_diagnostics,
        "output_files": output_files,
        "limitations": [
            "local-file scaffold only",
            "no recommender benchmark claims",
            "no NDCG/CTR evaluation",
            "no exposure-bias correction",
            "objective training machinery is reused as a controlled scaffold",
        ],
    }


def run_movielens_local_experiment(
    config: MovieLensLocalExperimentConfig,
) -> MovieLensLocalExperimentResult:
    """Run a claim-free local MovieLens experiment scaffold."""

    _require_torch()
    config = validate_movielens_local_experiment_config(config)
    adapter = MovieLensAdapter(
        data_path=config.data_path,
        min_rating_threshold=config.min_rating_threshold,
        max_rows=config.max_rows,
        row_sampling_mode=config.row_sampling_mode,
        seed=config.seed,
        split_mode=config.split_mode,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        session_gap_seconds=config.session_gap_seconds,
    )
    task = adapter.make_task()
    assert_benchmark_task_contract(task)
    if config.k > task.num_actions:
        raise ValueError("k must not exceed the MovieLens action catalogue size.")

    from .networks import build_networks

    torch.manual_seed(config.seed)
    trainer_data = _movielens_data_for_trainer(task)
    model = build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=config.embedding_dim,
        )
    )
    optimizer_bundle = make_optimizer_bundle(model, lr=config.lr, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    trainer = SyntheticDWMQLTrainer(
        data=trainer_data,
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=config.k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
    )
    history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))
    evaluation = evaluate_movielens_scaffold_task(task, k=config.k)
    split_diagnostics = _movielens_split_diagnostics(task, k=config.k)
    pmr_diagnostics = _movielens_pmr_diagnostics(task, model, k=config.k)
    ann_candidate_audit = _movielens_ann_candidate_audit_diagnostics(task, model, config)
    baseline_results = _movielens_baseline_diagnostics(config, task)
    baseline_ranking_diagnostics = _movielens_baseline_ranking_diagnostics(config, task)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(config.report_path) if config.report_path is not None else output_dir / f"{config.experiment_name}.json"
    manifest_path = output_dir / "manifest.json"
    output_files = {
        "report": str(report_path),
        "manifest": str(manifest_path),
    }
    report = _movielens_experiment_report(
        config,
        task,
        history,
        evaluation,
        split_diagnostics,
        pmr_diagnostics,
        ann_candidate_audit,
        baseline_ranking_diagnostics,
        baseline_results,
        output_files,
    )
    write_json_report(report, report_path)
    _write_movielens_manifest(
        manifest_path,
        {
            "schema_name": "dwmql_movielens_local_manifest",
            "schema_version": 1,
            "run_type": "movielens_local_experiment_scaffold",
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "output_dir": str(output_dir),
            "report_files": [str(report_path)],
            "checkpoint_files": [],
            "summary_files": [],
            "metadata": {
                "experiment_name": config.experiment_name,
                "data_path": config.data_path,
                "baseline_names": list(config.baseline_names),
                "dataset_metadata": task.metadata,
                "checkpointing": "not_written_for_movielens_local_scaffold",
            },
        },
    )
    return MovieLensLocalExperimentResult(
        config=config,
        training_history=history,
        evaluation_result=evaluation,
        baseline_results=baseline_results,
        output_files=output_files,
        diagnostics={
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "task_name": task.name,
            "task_type": task.task_type,
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "baseline_names": list(config.baseline_names),
        },
        task=task,
        model=model,
    )


def _history_from_objective_schedule(history: dict[str, Any]) -> TrainingHistory:
    losses = [float(value) for value in history["losses"]]
    objective_names = list(history["objective_names"])
    bellman_losses = [
        float(
            result.diagnostics.get(
                "raw_bellman_loss",
                result.diagnostics.get("bellman_loss", abs(float(result.loss.item()))),
            )
        )
        if hasattr(result, "diagnostics")
        else abs(loss)
        for result, loss in zip(history["results"], losses, strict=True)
    ]
    stationarity_losses = [
        float(result.diagnostics.get("stationarity_loss", abs(float(result.loss.item()))))
        if hasattr(result, "diagnostics")
        else abs(loss)
        for result, loss in zip(history["results"], losses, strict=True)
    ]
    density_regs = [
        float(result.diagnostics.get("density_regularization", 0.0))
        if hasattr(result, "diagnostics")
        else 0.0
        for result in history["results"]
    ]
    critic_total_losses = history.get("critic_total_losses", [None] * len(losses))
    support_weighted_bellman_losses = history.get(
        "support_weighted_bellman_losses",
        [None] * len(losses),
    )
    support_correction_losses = history.get("support_correction_losses", [None] * len(losses))
    support_correction_contributions = history.get("support_correction_contributions", [None] * len(losses))
    q_abs_max_values = history.get("q_abs_max_values", [None] * len(losses))
    return TrainingHistory(
        losses=losses,
        bellman_losses=bellman_losses,
        stationarity_losses=stationarity_losses,
        density_regularizations=density_regs,
        grad_norms=[float(result.grad_norm.item()) for result in history["results"]],
        parameter_delta_norms=[float(result.parameter_delta_norm.item()) for result in history["results"]],
        target_update_steps=history["target_update_steps"],
        index_ages=history["index_ages"],
        index_refresh_steps=history["index_refresh_steps"],
        index_refresh_count=history["index_refresh_count"],
        stale_index_recall_means=history["stale_index_recall_means"],
        stale_index_overlap_fraction_means=history["stale_index_overlap_fraction_means"],
        stale_index_best_value_gap_means=history["stale_index_best_value_gap_means"],
        stale_index_pmr_means=history["stale_index_pmr_means"],
        stale_index_pmr_mins=history["stale_index_pmr_mins"],
        stale_index_pmr_maxs=history["stale_index_pmr_maxs"],
        stale_index_pmr_normalized_means=history.get("stale_index_pmr_normalized_means", [None] * len(losses)),
        stale_index_missing_mass_means=history.get("stale_index_missing_mass_means", [None] * len(losses)),
        support_weighted_bellman_losses=support_weighted_bellman_losses,
        critic_total_losses=critic_total_losses,
        support_correction_losses=support_correction_losses,
        support_correction_contributions=support_correction_contributions,
        density_ess_values=history["density_ess_values"],
        weight_entropy_values=history["weight_entropy_values"],
        index_ema_update_count=history["index_ema_update_count"],
        index_encoder_drift_mean_l2=history["index_encoder_drift_mean_l2"],
        index_encoder_drift_max_l2=history["index_encoder_drift_max_l2"],
        index_encoder_drift_mean_cosine_distance=history["index_encoder_drift_mean_cosine_distance"],
        index_encoder_drift_max_cosine_distance=history["index_encoder_drift_max_cosine_distance"],
        target_vs_online_recall_means=history["target_vs_online_recall_means"],
        target_vs_online_overlap_fraction_means=history["target_vs_online_overlap_fraction_means"],
        ema_vs_online_recall_means=history["ema_vs_online_recall_means"],
        ema_vs_online_overlap_fraction_means=history["ema_vs_online_overlap_fraction_means"],
        ema_vs_target_recall_means=history["ema_vs_target_recall_means"],
        ema_vs_target_overlap_fraction_means=history["ema_vs_target_overlap_fraction_means"],
        diagnostics={**history["diagnostics"], "objective_names": objective_names},
        q_abs_max_values=q_abs_max_values,
    )


def run_synthetic_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run the synthetic benchmark-ready experiment scaffold."""

    _require_torch()
    config = validate_experiment_config(config)
    task = SyntheticBenchmarkTask(config.synthetic, name=config.experiment_name)
    torch.manual_seed(config.seed)
    from .networks import build_networks

    model = build_networks(config.model)
    optimizer_bundle = make_optimizer_bundle(model, lr=3e-4, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    checkpoint = None
    if config.resume_from is not None:
        checkpoint = load_checkpoint(config.resume_from, map_location="cpu")
        restore_model_from_checkpoint(model, checkpoint)
        restore_optimizer_from_checkpoint(optimizer_bundle, checkpoint)
        if target_model is not None and checkpoint.get("target_model_state_dict") is not None:
            restore_target_model_from_checkpoint(target_model, checkpoint)
    trainer = SyntheticDWMQLTrainer(
        data=task.data,
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=config.k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
        retrieval_correction_mode=config.retrieval_correction_mode,
        support_correction_weight=config.support_correction_weight,
        support_correction_target=config.support_correction_target,
        support_correction_metric=config.support_correction_metric,
    )
    if checkpoint is not None:
        trainer.load_state_dict(checkpoint.get("trainer_state", {}))
    training_history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))
    evaluation_config = EvaluationConfig(
        num_batches=config.eval_num_batches,
        batch_size=config.eval_batch_size,
        k=config.k,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
    )
    evaluation_summary = evaluate_synthetic_model(
        task.data,
        model,
        evaluation_config,
        config,
        task_name=task.name,
    )
    evaluation = EvaluationResult(
        task_name=evaluation_summary.task_name,
        metrics=evaluation_summary.metrics,
        diagnostics=evaluation_summary.diagnostics,
        synthetic_only=evaluation_summary.synthetic_only,
        benchmark_claim=evaluation_summary.benchmark_claim,
    )
    baseline_results = [
        evaluate_synthetic_baseline(
            get_baseline(name, seed=config.seed),
            task,
            evaluation_config,
        )
        for name in config.baseline_names
    ]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = output_dir / "final_checkpoint.pt"
    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer_bundle,
        target_model=target_model,
        trainer_state=trainer.state_dict(),
        config=asdict(config),
        metadata=CheckpointMetadata(
            step=trainer.step,
            experiment_name=config.experiment_name,
            diagnostics={
                "synthetic_experiment": True,
                "checkpoint_every": config.checkpoint_every,
                "midrun_checkpointing": False,
                "resume_from": config.resume_from,
            },
        ),
    )
    checkpoint_files = [str(final_checkpoint_path)]
    report = _report_for_experiment(
        config,
        training_history,
        evaluation,
        evaluation_summary,
        evaluation_config,
        baseline_results,
        checkpoint_files,
    )
    report_path = output_dir / f"{config.experiment_name}.json"
    write_json_report(report, report_path)
    manifest_path = output_dir / "manifest.json"
    write_run_manifest(
        RunManifest(
            run_type="synthetic_experiment",
            output_dir=str(output_dir),
            report_files=[str(report_path)],
            checkpoint_files=checkpoint_files,
            config_path=None,
            metadata={
                "experiment_name": config.experiment_name,
                "baseline_names": list(config.baseline_names),
                "evaluation_config": asdict(evaluation_config),
            },
        ),
        manifest_path,
    )
    return ExperimentResult(
        config=config,
        training_history=training_history,
        evaluation_result=evaluation,
        evaluation_summary=evaluation_summary,
        baseline_results=baseline_results,
        output_files={
            "report": str(report_path),
            "final_checkpoint": str(final_checkpoint_path),
            "manifest": str(manifest_path),
        },
        diagnostics={
            "synthetic_only": True,
            "benchmark_claim": False,
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "checkpoint_files": checkpoint_files,
            "resume_from": config.resume_from,
            "task_name": task.name,
            "baseline_names": list(config.baseline_names),
            "baseline_diagnostics_only": bool(baseline_results),
        },
    )


def _experiment_run_summary(result: ExperimentResult) -> dict[str, Any]:
    diagnostics = result.training_history.diagnostics
    metrics = result.evaluation_summary.metrics
    return {
        "experiment_name": result.config.experiment_name,
        "seed": result.config.seed,
        "k": result.config.k,
        "index_refresh_every": result.config.index_refresh_every,
        "index_encoder_policy": result.config.index_encoder_policy,
        "index_ema_tau": result.config.index_ema_tau,
        "report_file": result.output_files.get("report"),
        "final_checkpoint": result.output_files.get("final_checkpoint"),
        "final_total_loss": result.training_history.losses[-1],
        "final_bellman_loss": result.training_history.bellman_losses[-1],
        "final_stationarity_loss": result.training_history.stationarity_losses[-1],
        "final_density_regularization": result.training_history.density_regularizations[-1],
        "target_update_count": diagnostics.get("target_update_count"),
        "index_refresh_count": diagnostics.get("index_refresh_count"),
        "final_index_age": diagnostics.get("final_index_age"),
        "mean_bellman_loss": metrics.get("mean_bellman_loss"),
        "mean_stationarity_loss": metrics.get("mean_stationarity_loss"),
        "mean_ess": metrics.get("mean_ess"),
        "mean_exact_backend_recall": metrics.get("mean_exact_backend_recall"),
        "mean_exact_backend_overlap_fraction": metrics.get("mean_exact_backend_overlap_fraction"),
    }


def run_synthetic_experiment_sweep(sweep_config: ExperimentSweepConfig) -> ExperimentSweepResult:
    """Run a synthetic experiment-level sweep over selected config axes."""

    _require_torch()
    sweep_config = validate_experiment_sweep_config(sweep_config)
    base = sweep_config.base_config
    root = Path(sweep_config.output_dir) if sweep_config.output_dir is not None else Path(base.output_dir) / sweep_config.sweep_name
    root.mkdir(parents=True, exist_ok=True)
    seeds = sweep_config.seeds
    k_values = sweep_config.k_values or (base.k,)
    refresh_values = sweep_config.index_refresh_values or (base.index_refresh_every,)
    policies = sweep_config.index_encoder_policies or (base.index_encoder_policy,)
    results: list[ExperimentResult] = []
    run_summaries: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    run_reports: list[str] = []

    for seed, k, refresh, policy in itertools.product(seeds, k_values, refresh_values, policies):
        tau_values: tuple[float | None, ...]
        if policy == "ema":
            tau_values = sweep_config.index_ema_tau_values or (base.index_ema_tau,)
        else:
            tau_values = (None,)
        for tau in tau_values:
            if policy == "ema" and tau is None:
                raise ValueError("EMA policy requires an index EMA tau.")
            config = copy.deepcopy(base)
            config.seed = int(seed)
            config.synthetic.seed = int(seed)
            config.k = int(k)
            config.algorithm.candidate_k = int(k)
            config.index_refresh_every = int(refresh)
            config.index_encoder_policy = str(policy)
            config.index_ema_tau = None if tau is None else float(tau)
            if policy == "target" and config.target_tau is None:
                raise ValueError("target index encoder policy requires target_tau.")
            run_name = f"{sweep_config.sweep_name}_seed{seed}_k{k}_refresh{refresh}_policy{policy}"
            if tau is not None:
                run_name += f"_tau{_safe_float_label(float(tau))}"
            config.experiment_name = run_name
            config.output_dir = str(root / run_name)
            config.resume_from = None
            result = run_synthetic_experiment(validate_experiment_config(config))
            results.append(result)
            run_dirs.append(config.output_dir)
            run_reports.append(result.output_files["report"])
            run_summaries.append(_experiment_run_summary(result))

    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_experiment_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": len(results),
        "sweep_name": sweep_config.sweep_name,
        "output_dir": str(root),
        "sweep_axes": {
            "seeds": list(seeds),
            "k_values": list(k_values),
            "index_refresh_values": list(refresh_values),
            "index_encoder_policies": list(policies),
            "index_ema_tau_values": (
                None if sweep_config.index_ema_tau_values is None else list(sweep_config.index_ema_tau_values)
            ),
        },
        "run_dirs": run_dirs,
        "run_reports": run_reports,
        "runs": run_summaries,
    }
    summary["statistical_summaries"] = summarize_experiment_sweep_statistics(summary)
    summary_file = root / "summary.json"
    write_json_report(summary, summary_file)
    manifest_file = root / "manifest.json"
    write_run_manifest(
        RunManifest(
            run_type="synthetic_experiment_sweep",
            output_dir=str(root),
            report_files=run_reports,
            checkpoint_files=[
                result.output_files["final_checkpoint"]
                for result in results
                if "final_checkpoint" in result.output_files
            ],
            summary_files=[str(summary_file)],
            extra_files=run_dirs,
            metadata={
                "sweep_name": sweep_config.sweep_name,
                "sweep_axes": summary["sweep_axes"],
                "num_runs": len(results),
            },
        ),
        manifest_file,
    )
    return ExperimentSweepResult(
        sweep_config=sweep_config,
        experiment_results=results,
        summary_file=str(summary_file),
        manifest_file=str(manifest_file),
        summary=summary,
        output_dir=str(root),
        num_runs=len(results),
    )


def load_experiment_sweep_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate a synthetic experiment-level sweep summary."""

    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Experiment sweep summary not found: {summary_path}")
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in experiment sweep summary: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise ValueError("Experiment sweep summary must contain a JSON object.")
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    if summary.get("schema_name") != DIAGNOSTICS_SCHEMA_NAME:
        raise ValueError("Experiment sweep summary schema_name is unsupported.")
    if summary.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("Experiment sweep summary schema_version is unsupported.")
    if summary.get("mode") != "synthetic_experiment_sweep":
        raise ValueError("Experiment sweep summary mode must be synthetic_experiment_sweep.")
    if summary.get("synthetic_only") is not True:
        raise ValueError("Experiment sweep summary synthetic_only must be true.")
    if summary.get("benchmark_claim") is not False:
        raise ValueError("Experiment sweep summary benchmark_claim must be false.")
    if not isinstance(summary.get("runs"), list):
        raise ValueError("Experiment sweep summary must include runs as a list.")
    if not isinstance(summary.get("run_reports"), list):
        raise ValueError("Experiment sweep summary must include run_reports as a list.")
    return summary


def _numeric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        if not isinstance(value, int | float):
            continue
        values.append(float(value))
    return values


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty.")
    tensor = torch.tensor(values, dtype=torch.float32)
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError("Grouped metric values must be finite.")
    mean = float(tensor.mean().item())
    if len(values) == 1:
        return mean, 0.0
    return mean, float(tensor.std(unbiased=False).item())


def grouped_experiment_sweep_metrics(
    summary: dict[str, Any],
    group_keys: tuple[str, ...] = ("index_encoder_policy", "index_refresh_every"),
    metric_keys: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Group synthetic experiment sweep run summaries and aggregate metrics."""

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    if not group_keys:
        raise ValueError("group_keys must be non-empty.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("summary must include runs as a list.")
    metrics = metric_keys or (
        "final_total_loss",
        "final_bellman_loss",
        "final_stationarity_loss",
        "final_density_regularization",
        "mean_bellman_loss",
        "mean_stationarity_loss",
        "mean_ess",
        "mean_exact_backend_recall",
        "mean_exact_backend_overlap_fraction",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Each run summary must be a dictionary.")
        missing = [key for key in group_keys if key not in run]
        if missing:
            raise ValueError(f"Run summary is missing group key(s): {missing}.")
        group = tuple(run[key] for key in group_keys)
        grouped.setdefault(group, []).append(run)
    rows: list[dict[str, Any]] = []
    for group, group_runs in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row = {key: value for key, value in zip(group_keys, group, strict=True)}
        row["count"] = len(group_runs)
        for metric in metrics:
            values = _numeric_values(group_runs, metric)
            if not values:
                continue
            mean, std = _mean_std(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# MovieLens chronological strict protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MovieLensChronoProtocolResult:
    """Result from a claim-free MovieLens chronological support protocol run."""

    config: MovieLensChronoProtocolConfig
    training_history: TrainingHistory
    chrono_support: dict[str, Any]
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: MovieLensLocalTask | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)


def restore_movielens_chrono_model(checkpoint_path: str | Path) -> Any:
    """Rebuild a DW-MQL model from a chrono-protocol checkpoint.

    Loads a checkpoint written by ``run_movielens_chrono_protocol`` with
    ``save_model_checkpoint=True`` and returns the model with weights restored
    (in eval mode), so it can be re-evaluated under different ANN indexes without
    retraining. The matching task is rebuilt deterministically from the data path.
    """

    from .config import ModelConfig
    from .networks import build_networks

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_name") != "dwmql_movielens_chrono_checkpoint":
        raise ValueError("Not a movielens chrono model checkpoint.")
    mc = checkpoint["model_config"]
    model = build_networks(
        ModelConfig(
            state_dim=int(mc["state_dim"]),
            action_dim=int(mc["action_dim"]),
            embedding_dim=int(mc["embedding_dim"]),
            retrieval_projection=bool(mc["retrieval_projection"]),
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_movielens_chrono_protocol(
    config: MovieLensChronoProtocolConfig,
) -> MovieLensChronoProtocolResult:
    """Run the claim-free MovieLens chronological support protocol.

    Uses chronological_strict global timestamp ordering to create a leakage-free
    train/validation/test split. Trains a DW-MQL scaffold model on train rows and
    computes PMR/ranking/ANN support diagnostics on all splits.

    This is scaffold evidence only. No recommender benchmark claims are permitted.
    """

    _require_torch()
    config = validate_movielens_chrono_protocol_config(config)
    adapter = MovieLensAdapter(
        data_path=config.data_path,
        min_rating_threshold=config.min_rating_threshold,
        max_rows=config.max_rows,
        row_sampling_mode=config.row_sampling_mode,
        seed=config.seed,
        split_mode="chronological_strict",
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    task = adapter.make_task()
    assert_benchmark_task_contract(task)

    train_k = min(config.train_k, task.num_actions)
    for k in config.k_values:
        if k > task.num_actions:
            raise ValueError(f"k={k} in k_values exceeds the MovieLens action catalogue size ({task.num_actions}).")

    from .networks import build_networks

    torch.manual_seed(config.seed)
    trainer_data = _movielens_data_for_trainer(task)
    model = build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=config.embedding_dim,
            retrieval_projection=(
                config.retrieval_alignment_loss_type == "listwise_kl"
                and config.retrieval_alignment_weight > 0.0
            ),
            proto_action=float(getattr(config, "wolpertinger_bc_weight", 0.0)) > 0.0,
        )
    )
    optimizer_bundle = make_optimizer_bundle(model, lr=config.lr, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    trainer = SyntheticDWMQLTrainer(
        data=trainer_data,
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=train_k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
        stationarity_weight=config.stationarity_weight,
        density_reg_weight=config.density_regularization_weight,
        bellman_moment_weight=config.bellman_weight,
        max_grad_norm=config.max_grad_norm,
        q_l2_regularization_weight=config.q_l2_regularization_weight,
        q_value_clip=config.q_value_clip,
        reward_scale=config.reward_scale,
        ranking_margin_weight=config.ranking_margin_weight,
        ranking_margin=config.ranking_margin,
        ranking_negative_samples=config.ranking_negative_samples,
        retrieval_alignment_weight=config.retrieval_alignment_weight,
        retrieval_alignment_margin=config.retrieval_alignment_margin,
        retrieval_alignment_warmup_steps=config.retrieval_alignment_warmup_steps,
        retrieval_alignment_loss_type=config.retrieval_alignment_loss_type,
        listwise_teacher_temperature=config.listwise_teacher_temperature,
        listwise_retrieval_temperature=config.listwise_retrieval_temperature,
        cql_weight=config.cql_weight,
        cql_num_samples=config.cql_num_samples,
        bcq_weight=config.bcq_weight,
        wolpertinger_bc_weight=float(getattr(config, "wolpertinger_bc_weight", 0.0)),
        retrieval_correction_mode=config.retrieval_correction_mode,
        device=config.device,
    )
    history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))
    # Evaluation/diagnostics run on CPU: move the trained model (and target) back
    # so the many encode/eval call sites built on CPU tensors stay device-consistent.
    if str(config.device) != "cpu":
        model = model.to("cpu")
        if target_model is not None:
            target_model = target_model.to("cpu")
    chrono_support = evaluate_movielens_chrono_support_protocol(
        task,
        model,
        k_values=config.k_values,
        temperature=config.temperature,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        max_eval_rows=config.max_eval_rows,
        seed=config.seed,
    )

    from .benchmarks import diagnose_movielens_retrieval_alignment

    retrieval_alignment = diagnose_movielens_retrieval_alignment(
        task,
        model,
        split="test",
        k=min(train_k, task.num_actions),
        index_backend=config.index_backend,
        index_metric=config.index_metric,
    )
    q_degeneracy_diagnostics = diagnose_movielens_q_degeneracy(
        task,
        model,
        split="test",
        k=min(train_k, task.num_actions),
        temperature=config.temperature,
        q_degeneracy_std_threshold=config.q_degeneracy_std_threshold,
        q_degeneracy_margin_threshold=config.q_degeneracy_margin_threshold,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.protocol_name}.json"
    manifest_path = output_dir / "manifest.json"
    output_files = {"report": str(report_path), "manifest": str(manifest_path)}

    if config.save_model_checkpoint:
        # Honest model checkpoint (scaffold, benchmark_claim: false): saves trained
        # weights + the dims needed to rebuild the network, so the model can be
        # re-evaluated under different ANN indexes without retraining.
        checkpoint_path = output_dir / "model_checkpoint.pt"
        _, catalogue_action_ids = task.make_action_catalogue()
        torch.save(
            {
                "schema_name": "dwmql_movielens_chrono_checkpoint",
                "schema_version": 2,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "state_dim": int(task.state_dim),
                    "action_dim": int(task.action_dim),
                    "embedding_dim": int(config.embedding_dim),
                    "retrieval_projection": bool(model.has_retrieval_projection),
                },
                "seed": int(config.seed),
                "dataset_name": task.name,
                "catalogue_size": int(task.num_actions),
                "action_id_hash": stable_action_id_hash(catalogue_action_ids),
                "component_identity": component_identity_from_config(config),
                "config": asdict(config),
                "scaffold_only": True,
                "benchmark_claim": False,
            },
            checkpoint_path,
        )
        output_files["model_checkpoint"] = str(checkpoint_path)

    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    report = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "movielens_chrono_support_protocol",
        "mode": "movielens_chrono_support_protocol",
        "protocol_name": config.protocol_name,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "component_identity": component_identity_from_config(config),
        "dataset_metadata": task.metadata,
        "leakage_check_passed": chrono_support.get("leakage_check_passed"),
        "training_history_summary": _movielens_history_summary(
            history,
            max_grad_norm=config.max_grad_norm,
        ),
        "training_history": training_history_to_dict(history),
        "chrono_support_protocol": chrono_support,
        "retrieval_alignment_diagnostics": retrieval_alignment,
        "q_degeneracy_diagnostics": q_degeneracy_diagnostics,
        "limitations": [
            "chronological_strict split is global not per-user",
            "local-file scaffold only",
            "no recommender benchmark claims",
            "no NDCG/CTR evaluation",
        ],
    }
    write_json_report(report, report_path)
    _write_movielens_manifest(
        manifest_path,
        {
            "schema_name": "dwmql_movielens_chrono_manifest",
            "schema_version": 1,
            "run_type": "movielens_chrono_support_protocol",
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "output_dir": str(output_dir),
            "report_files": [str(report_path)],
            "checkpoint_files": (
                [output_files["model_checkpoint"]]
                if "model_checkpoint" in output_files
                else []
            ),
            "summary_files": [],
            "metadata": {
                "protocol_name": config.protocol_name,
                "data_path": config.data_path,
                "leakage_check_passed": chrono_support.get("leakage_check_passed"),
                "dataset_metadata": task.metadata,
                "component_identity": component_identity_from_config(config),
            },
        },
    )
    return MovieLensChronoProtocolResult(
        config=config,
        training_history=history,
        chrono_support=chrono_support,
        output_files=output_files,
        diagnostics={
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "protocol_name": config.protocol_name,
            "leakage_check_passed": chrono_support.get("leakage_check_passed"),
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "retrieval_alignment_diagnostics": retrieval_alignment,
            "q_degeneracy_diagnostics": q_degeneracy_diagnostics,
        },
        task=task,
        model=model,
    )


# ---------------------------------------------------------------------------
# Optional D4RL/Minari protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class D4RLOptionalProtocolResult:
    """Result from a claim-free optional D4RL/Minari support protocol run."""

    config: D4RLOptionalProtocolConfig
    training_history: TrainingHistory | None
    evaluation_result: EvaluationResult | None
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: D4RLDatasetTask | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)
    import_error: str | None = None


def run_d4rl_optional_protocol(
    config: D4RLOptionalProtocolConfig,
) -> D4RLOptionalProtocolResult:
    """Run the claim-free optional D4RL/Minari support protocol.

    Attempts to load a real D4RL or Minari dataset. If neither dependency is
    installed, returns a result with import_error set rather than raising.

    This is scaffold evidence only. No offline RL benchmark claims are permitted.
    """

    _require_torch()
    config = validate_d4rl_optional_protocol_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.protocol_name}.json"
    manifest_path = output_dir / "manifest.json"
    output_files = {"report": str(report_path), "manifest": str(manifest_path)}

    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    try:
        task = D4RLDatasetTask(
            dataset_name=config.dataset_name,
            codebook_size=config.codebook_size,
            seed=config.seed,
        )
    except ImportError as exc:
        import_message = str(exc)
        report = {
            "schema_name": DIAGNOSTICS_SCHEMA_NAME,
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "report_type": "d4rl_optional_support_protocol",
            "protocol_name": config.protocol_name,
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "config": asdict(config),
            "import_error": import_message,
            "status": "dependency_not_available",
        }
        write_json_report(report, report_path)
        return D4RLOptionalProtocolResult(
            config=config,
            training_history=None,
            evaluation_result=None,
            output_files=output_files,
            diagnostics={"import_error": import_message, "status": "dependency_not_available"},
            import_error=import_message,
        )

    assert_benchmark_task_contract(task)
    train_k = min(config.train_k, task.num_actions)

    from .networks import build_networks

    torch.manual_seed(config.seed)
    model = build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=config.embedding_dim,
        )
    )
    optimizer_bundle = make_optimizer_bundle(model, lr=config.lr, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    trainer = SyntheticDWMQLTrainer(
        data=task.make_dataset(),
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=train_k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
    )
    history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))

    codebook_audit = evaluate_d4rl_action_codebook_audit(
        task,  # type: ignore[arg-type]
        model,
        k=train_k,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
    )

    k_results: dict[str, Any] = {}
    for k in config.k_values:
        safe_k = min(k, task.num_actions)
        pmr = evaluate_d4rl_pmr_scaffold(task, model, k=safe_k)  # type: ignore[arg-type]
        ann = evaluate_d4rl_ann_candidate_audit(task, model, k=safe_k, index_backend=config.index_backend, index_metric=config.index_metric)  # type: ignore[arg-type]
        k_results[f"k_{k}"] = {
            "pmr": asdict(pmr),
            "ann": asdict(ann),
            "summary": {
                "k": safe_k,
                "pmr_at_k_scaffold": pmr.metrics.get("pmr_at_k_scaffold"),
                "ann_candidate_recall_mean": ann.metrics.get("ann_candidate_recall_mean"),
                "ann_candidate_pmr_mean": ann.metrics.get("ann_candidate_pmr_mean"),
            },
        }

    report = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "d4rl_optional_support_protocol",
        "protocol_name": config.protocol_name,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "dataset_metadata": task.metadata,
        "training_history_summary": {
            "num_steps": len(history.losses),
            "final_bellman_loss": history.bellman_losses[-1] if history.bellman_losses else None,
        },
        "codebook_audit": asdict(codebook_audit),
        "k_results": k_results,
        "status": "completed",
        "limitations": [
            "scaffold-only codebook built from random sample of dataset actions",
            "no offline RL benchmark claims",
            "no policy quality evaluation",
        ],
    }
    write_json_report(report, report_path)
    return D4RLOptionalProtocolResult(
        config=config,
        training_history=history,
        evaluation_result=codebook_audit,
        output_files=output_files,
        diagnostics={
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "dataset_name": config.dataset_name,
            "loaded_via": task.metadata.get("loaded_via"),
            "num_transitions": task.metadata.get("num_transitions"),
            "codebook_size": task.num_actions,
            "report_path": str(report_path),
        },
        task=task,
        model=model,
        import_error=None,
    )


# ---------------------------------------------------------------------------
# Optional RecSim/RecSim NG protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RecSimOptionalProtocolResult:
    """Result from a claim-free optional RecSim support protocol run."""

    config: RecSimOptionalProtocolConfig
    training_history: TrainingHistory | None
    evaluation_result: EvaluationResult | None
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: RecSimMockSlateTask | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)
    not_implemented: bool = False


def run_recsim_optional_protocol(
    config: RecSimOptionalProtocolConfig,
) -> RecSimOptionalProtocolResult:
    """Run the claim-free optional RecSim support protocol.

    Currently falls back to the mock slate scaffold. Real RecSim/RecSim NG
    loading is not yet implemented; this function will raise NotImplementedError
    if the API requires it. Uses the existing mock scaffold for diagnostics.

    This is scaffold evidence only. No recommender benchmark claims are permitted.
    """

    _require_torch()
    config = validate_recsim_optional_protocol_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.protocol_name}.json"
    manifest_path = output_dir / "manifest.json"
    output_files = {"report": str(report_path), "manifest": str(manifest_path)}

    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    # Real RecSim/RecSim NG loading is not yet implemented. Run mock scaffold instead.
    adapter = RecSimAdapter(mock=True)
    task = adapter.make_task(
        task_name="mock_slate",
        num_users=config.num_users,
        num_items=config.num_items,
        user_feature_dim=config.user_feature_dim,
        item_feature_dim=config.item_feature_dim,
        slate_size=config.slate_size,
        num_interactions=config.num_interactions,
        seed=config.seed,
    )
    assert_benchmark_task_contract(task)

    train_k = min(config.train_k, task.num_actions)

    from .networks import build_networks

    torch.manual_seed(config.seed)
    bundle = task.make_dataset_bundle()
    action_features = bundle.action_catalogue.action_features
    action_ids = bundle.action_catalogue.action_ids
    trainer_data = D4RLTrainerData(
        dataset=bundle.dataset,
        action_features=action_features,
        action_ids=action_ids,
        metadata={**bundle.metadata, "recsim_optional_trainer_view": True},
    )
    model = build_networks(
        ModelConfig(
            state_dim=bundle.dataset.batch.states.shape[1],
            action_dim=action_features.shape[1],
            embedding_dim=config.embedding_dim,
        )
    )
    optimizer_bundle = make_optimizer_bundle(model, lr=config.lr, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    trainer = SyntheticDWMQLTrainer(
        data=trainer_data,  # type: ignore[arg-type]
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=train_k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
    )
    history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))

    evaluation = evaluate_recsim_slate_scaffold(task, model, k=train_k)

    k_results: dict[str, Any] = {}
    for k in config.k_values:
        safe_k = min(k, task.num_actions)
        pmr = evaluate_recsim_pmr_scaffold(task, model, k=safe_k)
        ann = evaluate_recsim_ann_candidate_audit(task, model, k=safe_k, index_backend=config.index_backend, index_metric=config.index_metric)
        k_results[f"k_{k}"] = {
            "pmr": asdict(pmr),
            "ann": asdict(ann),
            "summary": {
                "k": safe_k,
                "pmr_at_k_scaffold": pmr.metrics.get("pmr_at_k_scaffold"),
                "ann_candidate_recall_mean": ann.metrics.get("ann_candidate_recall_mean"),
                "ann_candidate_pmr_mean": ann.metrics.get("ann_candidate_pmr_mean"),
            },
        }

    report = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "recsim_optional_support_protocol",
        "protocol_name": config.protocol_name,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "task_metadata": task.metadata,
        "training_history_summary": {
            "num_steps": len(history.losses),
            "final_bellman_loss": history.bellman_losses[-1] if history.bellman_losses else None,
        },
        "recsim_evaluation": asdict(evaluation),
        "k_results": k_results,
        "status": "mock_scaffold_only",
        "not_implemented": True,
        "limitations": [
            "real RecSim/RecSim NG loading is not yet implemented",
            "mock scaffold only",
            "no recommender benchmark claims",
        ],
    }
    write_json_report(report, report_path)
    return RecSimOptionalProtocolResult(
        config=config,
        training_history=history,
        evaluation_result=evaluation,
        output_files=output_files,
        diagnostics={
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "protocol_name": config.protocol_name,
            "not_implemented": True,
            "status": "mock_scaffold_only",
            "report_path": str(report_path),
        },
        task=task,
        model=model,
        not_implemented=True,
    )


# ---------------------------------------------------------------------------
# First benchmark protocol runner
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkRunResult:
    """Result for one run inside a benchmark protocol (one seed/backend/refresh/policy combo)."""

    route: str
    seed: int
    index_backend: str
    index_refresh_every: int
    index_encoder_policy: str
    index_ema_tau: float | None
    dwmql_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_dir: str = ""
    report_path: str = ""
    benchmark_claim: bool = False


@dataclass(slots=True)
class BenchmarkProtocolResult:
    """Aggregate result for a full benchmark protocol run."""

    config: BenchmarkProtocolConfig
    run_results: list[BenchmarkRunResult]
    summary: dict[str, Any]
    output_dir: str
    summary_file: str
    manifest_file: str
    markdown_table_file: str | None = None
    csv_table_file: str | None = None
    benchmark_claim: bool = False


def _probe_movielens_catalogue_size(
    data_path: str,
    min_rating_threshold: float,
    max_rows: int | None,
    seed: int,
) -> int:
    """Build a minimal task to read the MovieLens catalogue size without training."""

    adapter = MovieLensAdapter(
        data_path=data_path,
        min_rating_threshold=min_rating_threshold,
        max_rows=max_rows,
        seed=seed,
        split_mode="chronological_strict",
    )
    return adapter.make_task().num_actions


def _benchmark_capped_k_values(
    config: BenchmarkProtocolConfig,
    route: str,
) -> tuple[tuple[int, ...], int]:
    """Return (capped_k_values, capped_train_k) for a route.

    Probes catalogue size for MovieLens to avoid k > num_actions errors.
    Falls back to uncapped on any error.
    """

    if route == "movielens_chrono" and config.data_path is not None:
        try:
            cat_size = _probe_movielens_catalogue_size(
                data_path=config.data_path,
                min_rating_threshold=config.min_rating_threshold,
                max_rows=config.max_rows,
                seed=config.seeds[0] if config.seeds else 0,
            )
            capped = tuple(k for k in config.k_values if k <= cat_size)
            if not capped:
                capped = (min(config.train_k, cat_size),)
            return capped, min(config.train_k, cat_size)
        except Exception:
            pass
    return config.k_values, config.train_k


# Bellman loss threshold above which q_divergence_warning fires.
# Conservative: seed-0 stabilized run reached 36.4; seed-1 reached 12.1.
_Q_DIVERGENCE_BELLMAN_THRESHOLD: float = 50.0


def _extract_chrono_dwmql_metrics(
    result: MovieLensChronoProtocolResult,
    train_k: int,
) -> dict[str, Any]:
    """Extract flat support metrics and warning flags from a MovieLens chrono protocol result."""

    history = result.training_history
    final_bellman = history.bellman_losses[-1] if history.bellman_losses else None
    final_total = history.losses[-1] if history.losses else None
    chrono = result.chrono_support
    splits = chrono.get("splits", {})
    k_key = f"k_{train_k}"
    train_pmr = splits.get("train", {}).get("k_results", {}).get(k_key, {}).get("pmr", {}).get("metrics", {})
    train_ann = splits.get("train", {}).get("k_results", {}).get(k_key, {}).get("ann", {}).get("metrics", {})
    test_pmr = splits.get("test", {}).get("k_results", {}).get(k_key, {}).get("pmr", {}).get("metrics", {})
    test_ann = splits.get("test", {}).get("k_results", {}).get(k_key, {}).get("ann", {}).get("metrics", {})

    # Warning flags — diagnostic only, not benchmark claims
    test_q_embedding_alignment_recall = test_ann.get("ann_candidate_recall_mean")
    retrieval_collapse_warning = (
        test_q_embedding_alignment_recall is not None and test_q_embedding_alignment_recall == 0.0 and train_k > 0
    )

    q_value_clip = getattr(result.config, "q_value_clip", None) or float("inf")
    q_abs_max_hit_clip = any(
        v is not None and v >= q_value_clip for v in history.q_abs_max_values
    )
    q_divergence_warning = (
        (final_bellman is not None and final_bellman > _Q_DIVERGENCE_BELLMAN_THRESHOLD)
        or q_abs_max_hit_clip
    )

    retrieval_alignment = result.diagnostics.get("retrieval_alignment_diagnostics", {})
    id_alignment_warning = not retrieval_alignment.get("id_space_alignment_passed", True)

    q_degeneracy_diag = result.diagnostics.get("q_degeneracy_diagnostics", {})
    q_degeneracy_warning = bool(q_degeneracy_diag.get("q_degeneracy_warning", False))

    return {
        "final_bellman_loss": final_bellman,
        "final_total_loss": final_total,
        "train_pmr_at_k": train_pmr.get("pmr_at_k_scaffold"),
        "train_q_embedding_alignment_recall": train_ann.get("ann_candidate_recall_mean"),
        "train_ann_pmr": train_ann.get("ann_candidate_pmr_mean"),
        "test_pmr_at_k": test_pmr.get("pmr_at_k_scaffold"),
        "test_q_embedding_alignment_recall": test_ann.get("ann_candidate_recall_mean"),
        "test_ann_pmr": test_ann.get("ann_candidate_pmr_mean"),
        "leakage_check_passed": chrono.get("leakage_check_passed"),
        "retrieval_collapse_warning": retrieval_collapse_warning,
        "q_divergence_warning": q_divergence_warning,
        "id_alignment_warning": id_alignment_warning,
        "q_degeneracy_warning": q_degeneracy_warning,
        "q_score_std": q_degeneracy_diag.get("q_score_std"),
        "q_topk_margin_mean": q_degeneracy_diag.get("q_topk_margin_mean"),
        "positive_negative_score_gap_mean": q_degeneracy_diag.get("positive_negative_score_gap_mean"),
        "benchmark_claim": False,
    }


def _extract_d4rl_dwmql_metrics(
    result: D4RLOptionalProtocolResult,
    train_k: int,
) -> dict[str, Any]:
    """Extract flat metrics from a D4RL optional protocol result."""

    del train_k
    if result.import_error is not None:
        return {"status": "dependency_not_available", "import_error": result.import_error, "benchmark_claim": False}
    history = result.training_history
    final_bellman = history.bellman_losses[-1] if (history and history.bellman_losses) else None
    final_total = history.losses[-1] if (history and history.losses) else None
    return {"final_bellman_loss": final_bellman, "final_total_loss": final_total, "benchmark_claim": False}


def _extract_recsim_dwmql_metrics(
    result: RecSimOptionalProtocolResult,
    train_k: int,
) -> dict[str, Any]:
    """Extract flat metrics from a RecSim optional protocol result."""

    del train_k
    history = result.training_history
    final_bellman = history.bellman_losses[-1] if (history and history.bellman_losses) else None
    final_total = history.losses[-1] if (history and history.losses) else None
    return {
        "final_bellman_loss": final_bellman,
        "final_total_loss": final_total,
        "not_implemented": result.not_implemented,
        "benchmark_claim": False,
    }


def _run_benchmark_baseline_single(baseline_name: str, task: Any) -> dict[str, Any]:
    """Fit and evaluate one baseline on a task; gracefully handles NotImplementedError."""

    from .evaluation import EvaluationConfig

    try:
        baseline = get_baseline(baseline_name)
        result = evaluate_synthetic_baseline(baseline, task, EvaluationConfig(num_batches=1))
        return {
            "baseline_name": baseline_name,
            "implemented": True,
            "metrics": result.metrics,
            "benchmark_claim": False,
        }
    except NotImplementedError as exc:
        return {
            "baseline_name": baseline_name,
            "implemented": False,
            "error": str(exc),
            "benchmark_claim": False,
        }
    except Exception as exc:
        return {
            "baseline_name": baseline_name,
            "implemented": True,
            "error": str(exc),
            "benchmark_claim": False,
        }


def _make_benchmark_per_run_row(
    route: str,
    seed: int,
    backend: str,
    refresh: int,
    encoder_policy: str,
    ema_tau: float | None,
    method: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a flat per-run summary row for the benchmark table."""

    return {
        "route": route,
        "seed": seed,
        "method": method,
        "index_backend": backend,
        "index_refresh_every": refresh,
        "index_encoder_policy": encoder_policy,
        "index_ema_tau": ema_tau,
        "train_pmr_at_k": metrics.get("train_pmr_at_k"),
        "train_q_embedding_alignment_recall": metrics.get("train_q_embedding_alignment_recall"),
        "test_pmr_at_k": metrics.get("test_pmr_at_k"),
        "test_q_embedding_alignment_recall": metrics.get("test_q_embedding_alignment_recall"),
        "final_bellman_loss": metrics.get("final_bellman_loss"),
        "retrieval_collapse_warning": metrics.get("retrieval_collapse_warning"),
        "q_divergence_warning": metrics.get("q_divergence_warning"),
        "id_alignment_warning": metrics.get("id_alignment_warning"),
        "q_degeneracy_warning": metrics.get("q_degeneracy_warning"),
        "benchmark_claim": False,
    }


def _run_benchmark_dwmql_single(
    config: BenchmarkProtocolConfig,
    route: str,
    seed: int,
    backend: str,
    refresh: int,
    encoder_policy: str,
    ema_tau: float | None,
    capped_k_values: tuple[int, ...],
    capped_train_k: int,
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Run one DW-MQL protocol for one seed/parameter combination.

    Returns ``(task, metrics_dict)`` where ``task`` can be used for baselines.
    """

    task = None
    metrics: dict[str, Any] = {}

    if route == "movielens_chrono":
        per_run_cfg = MovieLensChronoProtocolConfig(
            protocol_name=f"benchmark_movielens_chrono_s{seed}",
            data_path=config.data_path or "",
            output_dir=str(run_dir),
            seed=seed,
            num_steps=config.num_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            k_values=capped_k_values,
            train_k=capped_train_k,
            lr=config.lr,
            embedding_dim=config.embedding_dim,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            min_rating_threshold=config.min_rating_threshold,
            max_rows=config.max_rows,
            temperature=config.temperature,
            index_backend=backend,
            index_metric=config.index_metric,
            use_persistent_index=config.use_persistent_index,
            index_refresh_every=refresh,
            index_encoder_policy=encoder_policy,
            index_ema_tau=ema_tau,
            target_tau=config.target_tau,
            target_update_every=config.target_update_every,
            max_grad_norm=config.max_grad_norm,
            q_l2_regularization_weight=config.q_l2_regularization_weight,
            q_value_clip=config.q_value_clip,
            reward_scale=config.reward_scale,
            density_regularization_weight=config.density_regularization_weight,
            stationarity_weight=config.stationarity_weight,
            bellman_weight=config.bellman_weight,
            scaffold_only=True,
            benchmark_claim=False,
            ranking_margin_weight=config.ranking_margin_weight,
            ranking_margin=config.ranking_margin,
            ranking_negative_samples=config.ranking_negative_samples,
            retrieval_alignment_weight=config.retrieval_alignment_weight,
            retrieval_alignment_margin=config.retrieval_alignment_margin,
            retrieval_alignment_warmup_steps=config.retrieval_alignment_warmup_steps,
            retrieval_alignment_loss_type=config.retrieval_alignment_loss_type,
            listwise_teacher_temperature=config.listwise_teacher_temperature,
            listwise_retrieval_temperature=config.listwise_retrieval_temperature,
            q_degeneracy_std_threshold=config.q_degeneracy_std_threshold,
            q_degeneracy_margin_threshold=config.q_degeneracy_margin_threshold,
        )
        result = run_movielens_chrono_protocol(per_run_cfg)
        task = result.task
        metrics = _extract_chrono_dwmql_metrics(result, capped_train_k)

    elif route == "d4rl_optional":
        per_run_cfg_d4rl = D4RLOptionalProtocolConfig(
            protocol_name=f"benchmark_d4rl_optional_s{seed}",
            dataset_name=config.dataset_name or "hopper-medium-v2",
            output_dir=str(run_dir),
            seed=seed,
            num_steps=config.num_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            k_values=capped_k_values,
            train_k=capped_train_k,
            lr=config.lr,
            embedding_dim=config.embedding_dim,
            codebook_size=config.codebook_size,
            index_backend=backend,
            index_metric=config.index_metric,
            use_persistent_index=config.use_persistent_index,
            index_refresh_every=refresh,
            index_encoder_policy=encoder_policy,
            index_ema_tau=ema_tau,
            target_tau=config.target_tau,
            target_update_every=config.target_update_every,
            scaffold_only=True,
            benchmark_claim=False,
        )
        d4rl_result = run_d4rl_optional_protocol(per_run_cfg_d4rl)
        task = d4rl_result.task
        metrics = _extract_d4rl_dwmql_metrics(d4rl_result, capped_train_k)

    elif route == "recsim_optional":
        per_run_cfg_rs = RecSimOptionalProtocolConfig(
            protocol_name=f"benchmark_recsim_optional_s{seed}",
            output_dir=str(run_dir),
            seed=seed,
            num_steps=config.num_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            k_values=capped_k_values,
            train_k=capped_train_k,
            lr=config.lr,
            embedding_dim=config.embedding_dim,
            num_users=config.num_users,
            num_items=config.num_items,
            user_feature_dim=config.user_feature_dim,
            item_feature_dim=config.item_feature_dim,
            slate_size=config.slate_size,
            num_interactions=config.num_interactions,
            index_backend=backend,
            index_metric=config.index_metric,
            use_persistent_index=config.use_persistent_index,
            index_refresh_every=refresh,
            index_encoder_policy=encoder_policy,
            index_ema_tau=ema_tau,
            target_tau=config.target_tau,
            target_update_every=config.target_update_every,
            scaffold_only=True,
            benchmark_claim=False,
        )
        rs_result = run_recsim_optional_protocol(per_run_cfg_rs)
        task = rs_result.task
        metrics = _extract_recsim_dwmql_metrics(rs_result, capped_train_k)

    return task, metrics


def run_benchmark_protocol(config: BenchmarkProtocolConfig) -> BenchmarkProtocolResult:
    """Run the first benchmark-style multi-seed experiment protocol.

    Iterates over routes × seeds × index backends × refresh cadences ×
    encoder policies, runs DW-MQL and trivial baselines, and writes per-run
    reports plus a grouped summary with descriptive statistics.

    All results are claim-free. ``benchmark_claim`` is always ``False``.
    """

    _require_torch()
    config = validate_benchmark_protocol_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .reporting import (
        benchmark_grouped_statistics_to_table,
        benchmark_protocol_rows_to_table,
        write_csv_table,
        write_markdown_table,
    )
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    run_results: list[BenchmarkRunResult] = []
    all_per_run_rows: list[dict[str, Any]] = []

    for route in config.routes:
        capped_k_values, capped_train_k = _benchmark_capped_k_values(config, route)

        for seed, backend, refresh, encoder_policy, ema_tau in itertools.product(
            config.seeds,
            config.index_backend_values,
            config.index_refresh_values,
            config.index_encoder_policies,
            config.index_ema_tau_values,
        ):
            if encoder_policy == "ema" and ema_tau is None:
                continue

            tau_tag = f"_tau{ema_tau}" if ema_tau is not None else ""
            run_slug = f"{route}_s{seed}_{backend}_r{refresh}_{encoder_policy}{tau_tag}"
            run_dir = output_dir / run_slug
            run_dir.mkdir(parents=True, exist_ok=True)

            dwmql_metrics: dict[str, Any] = {}
            baseline_metrics_map: dict[str, dict[str, Any]] = {}
            task_for_baselines: Any = None

            if config.enable_dwmql:
                task_for_baselines, dwmql_metrics = _run_benchmark_dwmql_single(
                    config=config,
                    route=route,
                    seed=seed,
                    backend=backend,
                    refresh=refresh,
                    encoder_policy=encoder_policy,
                    ema_tau=ema_tau,
                    capped_k_values=capped_k_values,
                    capped_train_k=capped_train_k,
                    run_dir=run_dir,
                )

            if task_for_baselines is not None:
                for bl_name in config.baselines:
                    baseline_metrics_map[bl_name] = _run_benchmark_baseline_single(bl_name, task_for_baselines)

            run_report: dict[str, Any] = {
                "schema_name": DIAGNOSTICS_SCHEMA_NAME,
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "route": route,
                "seed": seed,
                "index_backend": backend,
                "index_refresh_every": refresh,
                "index_encoder_policy": encoder_policy,
                "index_ema_tau": ema_tau,
                "benchmark_claim": False,
                "scaffold_only": True,
                "dwmql_metrics": dwmql_metrics,
                "baseline_metrics": baseline_metrics_map,
                "warning_flags": {
                    "retrieval_collapse_warning": dwmql_metrics.get("retrieval_collapse_warning"),
                    "q_divergence_warning": dwmql_metrics.get("q_divergence_warning"),
                    "id_alignment_warning": dwmql_metrics.get("id_alignment_warning"),
                    "q_degeneracy_warning": dwmql_metrics.get("q_degeneracy_warning"),
                },
            }
            report_path = run_dir / "run_report.json"
            write_json_report(run_report, report_path)

            run_results.append(
                BenchmarkRunResult(
                    route=route,
                    seed=seed,
                    index_backend=backend,
                    index_refresh_every=refresh,
                    index_encoder_policy=encoder_policy,
                    index_ema_tau=ema_tau,
                    dwmql_metrics=dwmql_metrics,
                    baseline_metrics=baseline_metrics_map,
                    run_dir=str(run_dir),
                    report_path=str(report_path),
                )
            )

            dwmql_row = _make_benchmark_per_run_row(route, seed, backend, refresh, encoder_policy, ema_tau, "dwmql", dwmql_metrics)
            all_per_run_rows.append(dwmql_row)
            for bl_name, bl_data in baseline_metrics_map.items():
                bl_row = _make_benchmark_per_run_row(route, seed, backend, refresh, encoder_policy, ema_tau, bl_name, bl_data.get("metrics", {}))
                all_per_run_rows.append(bl_row)

    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    md_path = output_dir / "benchmark_table.md"
    csv_path = output_dir / "benchmark_table.csv"

    group_keys = ["route", "index_backend", "index_refresh_every", "index_encoder_policy", "method"]
    metric_keys = ["train_pmr_at_k", "train_q_embedding_alignment_recall", "test_pmr_at_k", "test_q_embedding_alignment_recall", "final_bellman_loss"]
    grouped_stats_rows = group_metric_summary(all_per_run_rows, group_keys, metric_keys)
    grouped_stats: dict[str, Any] = {
        "benchmark_claim": False,
        "scaffold_only": True,
        "group_keys": group_keys,
        "metric_keys": metric_keys,
        "groups": grouped_stats_rows,
    }

    summary: dict[str, Any] = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "benchmark_protocol_summary",
        "protocol_name": config.protocol_name,
        "benchmark_claim": False,
        "scaffold_only": True,
        "num_runs": len(run_results),
        "routes": list(config.routes),
        "seeds": list(config.seeds),
        "config": asdict(config),
        "per_run_rows": all_per_run_rows,
        "grouped_statistics": grouped_stats,
        "limitations": [
            "all results are claim-free scaffold diagnostics",
            "no benchmark performance claims",
            "no statistical significance tests",
            "baselines are trivial (random, nearest) — not paper baselines",
        ],
    }
    write_json_report(summary, summary_path)

    table = benchmark_protocol_rows_to_table(all_per_run_rows)
    write_markdown_table(table, md_path)
    write_csv_table(table, csv_path)

    try:
        grouped_table = benchmark_grouped_statistics_to_table(grouped_stats)
        grouped_md_path = output_dir / "benchmark_grouped_table.md"
        grouped_csv_path = output_dir / "benchmark_grouped_table.csv"
        write_markdown_table(grouped_table, grouped_md_path)
        write_csv_table(grouped_table, grouped_csv_path)
    except Exception:
        grouped_md_path = None
        grouped_csv_path = None

    manifest: dict[str, Any] = {
        "schema_name": "dwmql_benchmark_protocol_manifest",
        "schema_version": 1,
        "protocol_name": config.protocol_name,
        "benchmark_claim": False,
        "scaffold_only": True,
        "output_dir": str(output_dir),
        "summary_file": str(summary_path),
        "report_files": [r.report_path for r in run_results],
        "run_dirs": [r.run_dir for r in run_results],
        "table_files": [str(md_path), str(csv_path)],
    }
    write_json_report(manifest, manifest_path)

    return BenchmarkProtocolResult(
        config=config,
        run_results=run_results,
        summary=summary,
        output_dir=str(output_dir),
        summary_file=str(summary_path),
        manifest_file=str(manifest_path),
        markdown_table_file=str(md_path),
        csv_table_file=str(csv_path),
    )


def inspect_benchmark_protocol_summary(summary: dict[str, Any]) -> str:
    """Return a human-readable inspection of a benchmark protocol summary."""

    lines: list[str] = []
    lines.append(f"Protocol: {summary.get('protocol_name', '(unknown)')}")
    lines.append(f"benchmark_claim: {str(summary.get('benchmark_claim', False)).lower()}")
    lines.append(f"scaffold_only: {str(summary.get('scaffold_only', True)).lower()}")
    lines.append(f"Routes: {summary.get('routes', [])}")
    lines.append(f"Seeds: {summary.get('seeds', [])}")
    lines.append(f"Total runs (DW-MQL + baseline rows): {summary.get('num_runs', 0)}")

    grouped = summary.get("grouped_statistics", {})
    groups = grouped.get("groups", [])
    if groups:
        lines.append("")
        lines.append("Grouped support diagnostics (descriptive only, no significance tests):")
        for group_row in groups:
            group = group_row.get("group", {})
            count = group_row.get("count", 0)
            metrics = group_row.get("metrics", {})
            group_label = ", ".join(f"{k}={v}" for k, v in sorted(group.items()))
            lines.append(f"  [{group_label}] n={count}")
            for metric_name, ms in sorted(metrics.items()):
                if isinstance(ms, dict):
                    mean_val = ms.get("mean")
                    stderr_val = ms.get("stderr")
                    if mean_val is not None:
                        se_str = f" ± {stderr_val:.4f}" if stderr_val is not None else ""
                        lines.append(f"    {metric_name}: {mean_val:.4f}{se_str}")

    per_run_rows = summary.get("per_run_rows", [])
    dwmql_rows = [r for r in per_run_rows if r.get("method") == "dwmql"]
    if dwmql_rows:
        lines.append("")
        lines.append("Warning flags (DW-MQL runs only, diagnostic — not benchmark claims):")
        collapse_count = sum(1 for r in dwmql_rows if r.get("retrieval_collapse_warning"))
        diverge_count = sum(1 for r in dwmql_rows if r.get("q_divergence_warning"))
        id_align_count = sum(1 for r in dwmql_rows if r.get("id_alignment_warning"))
        degeneracy_count = sum(1 for r in dwmql_rows if r.get("q_degeneracy_warning"))
        lines.append(f"  retrieval_collapse_warning: {collapse_count}/{len(dwmql_rows)} runs")
        lines.append(f"  q_divergence_warning:       {diverge_count}/{len(dwmql_rows)} runs")
        lines.append(f"  id_alignment_warning:       {id_align_count}/{len(dwmql_rows)} runs")
        lines.append(f"  q_degeneracy_warning:       {degeneracy_count}/{len(dwmql_rows)} runs")
        if collapse_count:
            lines.append("  [!] Index collapse detected: ANN recall=0 on some seeds.")
        if diverge_count:
            lines.append(
                f"  [!] Q divergence detected: Bellman loss > {_Q_DIVERGENCE_BELLMAN_THRESHOLD} or Q clipped."
            )
        if id_align_count:
            lines.append("  [!] ID-space alignment failed: index returned out-of-bounds ids.")
        if degeneracy_count:
            lines.append("  [!] Q degeneracy detected: near-uniform Q scores on some seeds.")

    lines.append("")
    lines.append("Limitations:")
    for lim in summary.get("limitations", []):
        lines.append(f"  - {lim}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# KuaiRand-Pure chronological support protocol (Phase 4)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KuaiRandChronoProtocolResult:
    """Result from a claim-free KuaiRand-Pure chronological support protocol run."""

    config: KuaiRandChronoProtocolConfig
    training_history: TrainingHistory
    chrono_support: dict[str, Any]
    output_files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: KuaiRandLocalTask | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)


def run_kuairand_chrono_protocol(
    config: KuaiRandChronoProtocolConfig,
) -> KuaiRandChronoProtocolResult:
    """Run the claim-free KuaiRand-Pure chronological support protocol.

    Structurally identical to run_movielens_chrono_protocol but uses
    KuaiRandLocalTask with short-video implicit-feedback data. Phase 4
    generalization gate: does DW-MQL ANN recall >> baselines on KuaiRand
    as it does on MovieLens?

    All results carry benchmark_claim: False. Scaffold diagnostics only.
    """
    from .benchmarks import diagnose_movielens_retrieval_alignment, diagnose_movielens_q_degeneracy
    from .networks import build_networks
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    _require_torch()
    config = validate_kuairand_chrono_protocol_config(config)
    task = KuaiRandLocalTask(
        data_dir=config.data_dir,
        max_rows=config.max_rows,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )

    train_k = min(config.train_k, task.num_actions)
    for k in config.k_values:
        if k > task.num_actions:
            raise ValueError(f"k={k} in k_values exceeds the KuaiRand action catalogue size ({task.num_actions}).")

    torch.manual_seed(config.seed)
    trainer_data = _movielens_data_for_trainer(task)
    model = build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=config.embedding_dim,
            retrieval_projection=(
                config.retrieval_alignment_loss_type == "listwise_kl"
                and config.retrieval_alignment_weight > 0.0
            ),
            proto_action=float(getattr(config, "wolpertinger_bc_weight", 0.0)) > 0.0,
        )
    )
    optimizer_bundle = make_optimizer_bundle(model, lr=config.lr, mode="separate_heads")
    target_model = clone_target_network(model) if config.target_tau is not None else None
    generator = torch.Generator().manual_seed(config.seed)
    trainer = SyntheticDWMQLTrainer(
        data=trainer_data,
        model=model,
        optimizer_bundle=optimizer_bundle,
        batch_size=config.batch_size,
        k=train_k,
        gamma=config.gamma,
        generator=generator,
        target_model=target_model,
        target_tau=config.target_tau,
        target_update_every=config.target_update_every,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        index_refresh_every=config.index_refresh_every,
        use_persistent_index=config.use_persistent_index,
        audit_stale_index=config.use_persistent_index,
        audit_encoder_policy_overlap=False,
        index_encoder_policy=config.index_encoder_policy,
        index_ema_tau=config.index_ema_tau,
        stationarity_weight=config.stationarity_weight,
        density_reg_weight=config.density_regularization_weight,
        bellman_moment_weight=config.bellman_weight,
        max_grad_norm=config.max_grad_norm,
        q_l2_regularization_weight=config.q_l2_regularization_weight,
        q_value_clip=config.q_value_clip,
        reward_scale=config.reward_scale,
        ranking_margin_weight=config.ranking_margin_weight,
        ranking_margin=config.ranking_margin,
        ranking_negative_samples=config.ranking_negative_samples,
        retrieval_alignment_weight=config.retrieval_alignment_weight,
        retrieval_alignment_margin=config.retrieval_alignment_margin,
        retrieval_alignment_warmup_steps=config.retrieval_alignment_warmup_steps,
        retrieval_alignment_loss_type=config.retrieval_alignment_loss_type,
        listwise_teacher_temperature=config.listwise_teacher_temperature,
        listwise_retrieval_temperature=config.listwise_retrieval_temperature,
        cql_weight=config.cql_weight,
        cql_num_samples=config.cql_num_samples,
        bcq_weight=config.bcq_weight,
        wolpertinger_bc_weight=float(getattr(config, "wolpertinger_bc_weight", 0.0)),
        retrieval_correction_mode=config.retrieval_correction_mode,
        device=config.device,
    )
    history = _history_from_objective_schedule(trainer.train_objective_schedule(config.num_steps))

    if str(config.device) != "cpu":
        model = model.to("cpu")
        if target_model is not None:
            target_model = target_model.to("cpu")

    chrono_support = evaluate_movielens_chrono_support_protocol(
        task,
        model,
        k_values=config.k_values,
        temperature=config.temperature,
        index_backend=config.index_backend,
        index_metric=config.index_metric,
        max_eval_rows=config.max_eval_rows,
        seed=config.seed,
    )
    retrieval_alignment = diagnose_movielens_retrieval_alignment(
        task,
        model,
        split="test",
        k=min(train_k, task.num_actions),
        index_backend=config.index_backend,
        index_metric=config.index_metric,
    )
    q_degeneracy_diagnostics = diagnose_movielens_q_degeneracy(
        task,
        model,
        split="test",
        k=min(train_k, task.num_actions),
        temperature=config.temperature,
        q_degeneracy_std_threshold=config.q_degeneracy_std_threshold,
        q_degeneracy_margin_threshold=config.q_degeneracy_margin_threshold,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.protocol_name}.json"
    manifest_path = output_dir / "manifest.json"
    output_files = {"report": str(report_path), "manifest": str(manifest_path)}

    if config.save_model_checkpoint:
        checkpoint_path = output_dir / "model_checkpoint.pt"
        _, catalogue_action_ids = task.make_action_catalogue()
        torch.save(
            {
                "schema_name": "dwmql_kuairand_chrono_checkpoint",
                "schema_version": 2,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "state_dim": int(task.state_dim),
                    "action_dim": int(task.action_dim),
                    "embedding_dim": int(config.embedding_dim),
                    "retrieval_projection": bool(model.has_retrieval_projection),
                },
                "seed": int(config.seed),
                "dataset_name": task.name,
                "catalogue_size": int(task.num_actions),
                "action_id_hash": stable_action_id_hash(catalogue_action_ids),
                "component_identity": component_identity_from_config(config),
                "config": asdict(config),
                "scaffold_only": True,
                "benchmark_claim": False,
            },
            checkpoint_path,
        )
        output_files["model_checkpoint"] = str(checkpoint_path)

    report = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "report_type": "kuairand_chrono_support_protocol",
        "mode": "kuairand_chrono_support_protocol",
        "protocol_name": config.protocol_name,
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "config": asdict(config),
        "component_identity": component_identity_from_config(config),
        "dataset_metadata": task.metadata,
        "leakage_check_passed": chrono_support.get("leakage_check_passed"),
        "training_history_summary": _movielens_history_summary(
            history,
            max_grad_norm=config.max_grad_norm,
        ),
        "training_history": training_history_to_dict(history),
        "chrono_support_protocol": chrono_support,
        "retrieval_alignment_diagnostics": retrieval_alignment,
        "q_degeneracy_diagnostics": q_degeneracy_diagnostics,
        "limitations": [
            "chronological_strict split is global not per-user",
            "local-file scaffold only",
            "no recommender benchmark claims",
            "no NDCG/CTR evaluation",
            "implicit feedback only (is_click)",
        ],
    }
    write_json_report(report, report_path)
    _write_movielens_manifest(
        manifest_path,
        {
            "schema_name": "dwmql_kuairand_chrono_manifest",
            "schema_version": 1,
            "run_type": "kuairand_chrono_support_protocol",
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "output_dir": str(output_dir),
            "report_files": [str(report_path)],
            "checkpoint_files": (
                [output_files["model_checkpoint"]]
                if "model_checkpoint" in output_files
                else []
            ),
            "summary_files": [],
            "metadata": {
                "protocol_name": config.protocol_name,
                "data_dir": config.data_dir,
                "leakage_check_passed": chrono_support.get("leakage_check_passed"),
                "dataset_metadata": task.metadata,
                "component_identity": component_identity_from_config(config),
            },
        },
    )
    return KuaiRandChronoProtocolResult(
        config=config,
        training_history=history,
        chrono_support=chrono_support,
        output_files=output_files,
        diagnostics={
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "protocol_name": config.protocol_name,
            "leakage_check_passed": chrono_support.get("leakage_check_passed"),
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "retrieval_alignment_diagnostics": retrieval_alignment,
            "q_degeneracy_diagnostics": q_degeneracy_diagnostics,
        },
        task=task,
        model=model,
    )
