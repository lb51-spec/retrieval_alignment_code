"""Diagnostics-only synthetic sweep helpers.

This module runs small synthetic trainer diagnostics across a few configuration
knobs and writes one JSON report per run. It is not benchmark evaluation, does
not checkpoint models, and makes no performance claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import SmokeProfile, build_smoke_components, default_smoke_profile, load_smoke_profile_yaml, validate_smoke_profile
from .manifest import RunManifest, write_run_manifest
from .metrics import uniform_policy_mass_baseline
from .trainer import SyntheticDWMQLTrainer, TrainingHistory, write_json_report


@dataclass(slots=True)
class SyntheticSweepResult:
    """Summary of a diagnostics-only synthetic sweep."""

    run_reports: list[dict[str, Any]]
    output_dir: Path
    num_runs: int
    summary_file: Path
    manifest_file: Path
    summary: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class SupportLossAblationResult:
    """Summary of a synthetic retrieval-correction ablation grid."""

    run_reports: list[dict[str, Any]]
    output_dir: Path
    num_runs: int
    summary_file: Path
    manifest_file: Path
    summary: dict[str, Any]
    diagnostics: dict[str, Any]


def _validate_non_empty_positive_ints(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple of integers.")
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must contain only positive integers.")
    return values


def _validate_non_empty_ints(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple of integers.")
    for value in values:
        if not isinstance(value, int):
            raise ValueError(f"{name} must contain only integers.")
    return values


def _validate_encoder_policies(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("index_encoder_policies must be a non-empty tuple.")
    valid = {"online", "target", "ema"}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown index encoder policy value(s): {sorted(unknown)}.")
    return values


def _format_tau_for_filename(tau: float) -> str:
    return f"{tau:g}".replace(".", "p").replace("-", "m")


def _safe_run_filename(
    seed: int,
    refresh: int,
    policy: str,
    k: int | None = None,
    tau: float | None = None,
) -> str:
    suffix = "" if k is None else f"_k{k}"
    if tau is not None:
        suffix += f"_tau{_format_tau_for_filename(tau)}"
    return f"seed{seed}_refresh{refresh}_policy{policy}{suffix}.json"


def _safe_ablation_filename(
    seed: int,
    refresh: int,
    policy: str,
    k: int,
    mode: str,
    tau: float | None = None,
    support_weight: float | None = None,
    support_weighting_mode: str | None = None,
    support_weight_scale: float | None = None,
) -> str:
    suffix = f"_mode{mode}_k{k}"
    if tau is not None:
        suffix += f"_tau{_format_tau_for_filename(tau)}"
    if support_weight is not None:
        suffix += f"_scw{_format_tau_for_filename(float(support_weight))}"
    if support_weighting_mode is not None:
        suffix += f"_swm{support_weighting_mode}"
    if support_weight_scale is not None:
        suffix += f"_sws{_format_tau_for_filename(float(support_weight_scale))}"
    return f"seed{seed}_refresh{refresh}_policy{policy}{suffix}.json"


def _mean_numeric(values: list[Any]) -> float | None:
    numeric_values = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _build_trainer(profile: SmokeProfile) -> tuple[SyntheticDWMQLTrainer, Any]:
    components = build_smoke_components(profile)
    trainer = SyntheticDWMQLTrainer(
        data=components.data,
        model=components.model,
        optimizer=components.optimizer,
        batch_size=profile.batch_size,
        k=profile.k,
        gamma=profile.gamma,
        generator=components.generator,
        target_model=components.target_model,
        target_tau=profile.target_tau,
        target_update_every=profile.target_update_every,
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
    return trainer, components


def _report_for_history(history: TrainingHistory, profile: SmokeProfile) -> dict[str, Any]:
    from .train import _synthetic_training_report

    report = _synthetic_training_report(history, profile)
    report["mode"] = "synthetic_diagnostics_sweep_run"
    report["diagnostics_only"] = True
    return report


def _load_support_ablation_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Support-loss ablation config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        import yaml

        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Support-loss ablation config YAML must contain a mapping.")
    return loaded


def summarize_sweep_run(report: dict[str, Any], json_file: str | Path) -> dict[str, Any]:
    """Extract a compact per-run summary from a sweep run report."""

    sweep = report.get("sweep", {})
    summary = report.get("summary", {})
    diagnostics = report.get("diagnostics", {})
    history = report.get("history", {})
    bellman_losses = history.get("bellman_losses", [])
    support_weighted_bellman_losses = history.get("support_weighted_bellman_losses", [])
    support_weight_means = history.get("support_weight_means", [])
    support_weight_mins = history.get("support_weight_mins", [])
    support_weight_maxs = history.get("support_weight_maxs", [])
    critic_total_losses = history.get("critic_total_losses", [])
    stationarity_losses = history.get("stationarity_losses", [])
    density_regularizations = history.get("density_regularizations", [])
    stale_recall = history.get("stale_index_recall_means", [])
    stale_overlap = history.get("stale_index_overlap_fraction_means", [])
    stale_best_gap = history.get("stale_index_best_value_gap_means", [])
    stale_pmr = history.get("stale_index_pmr_means", [])
    stale_pmr_normalized = history.get("stale_index_pmr_normalized_means", [])
    stale_missing_mass = history.get("stale_index_missing_mass_means", [])
    support_correction_losses = history.get("support_correction_losses", report.get("support_correction_losses", []))
    support_correction_contributions = history.get(
        "support_correction_contributions",
        report.get("support_correction_contributions", []),
    )
    density_ess = history.get("density_ess_values", [])
    weight_entropy = history.get("weight_entropy_values", [])
    profile = report.get("profile", {})
    num_actions = profile.get("num_actions") if isinstance(profile, dict) else None
    candidate_k = sweep.get("k")
    uniform_baseline = diagnostics.get("final_stale_index_uniform_pmr_baseline")
    if uniform_baseline is None and isinstance(candidate_k, int) and isinstance(num_actions, int):
        uniform_baseline = uniform_policy_mass_baseline(candidate_k, num_actions)

    return {
        "seed": sweep.get("seed"),
        "index_refresh_every": sweep.get("index_refresh_every"),
        "index_encoder_policy": sweep.get("index_encoder_policy"),
        "k": sweep.get("k"),
        "index_ema_tau": sweep.get("index_ema_tau"),
        "retrieval_correction_mode": sweep.get("retrieval_correction_mode", summary.get("retrieval_correction_mode")),
        "support_correction_loss": summary.get(
            "support_correction_loss",
            support_correction_losses[-1] if support_correction_losses else diagnostics.get("final_support_correction_loss"),
        ),
        "final_support_correction_loss": summary.get(
            "final_support_correction_loss",
            support_correction_losses[-1] if support_correction_losses else diagnostics.get("final_support_correction_loss"),
        ),
        "mean_support_correction_loss": summary.get(
            "mean_support_correction_loss",
            _mean_numeric(support_correction_losses) if support_correction_losses else diagnostics.get("mean_support_correction_loss"),
        ),
        "final_support_correction_contribution": summary.get(
            "final_support_correction_contribution",
            support_correction_contributions[-1]
            if support_correction_contributions
            else diagnostics.get("final_support_correction_contribution"),
        ),
        "mean_support_correction_contribution": summary.get(
            "mean_support_correction_contribution",
            _mean_numeric(support_correction_contributions)
            if support_correction_contributions
            else diagnostics.get("mean_support_correction_contribution"),
        ),
        "support_correction_active": summary.get("support_correction_active", diagnostics.get("support_correction_active")),
        "support_correction_weight": summary.get("support_correction_weight", diagnostics.get("support_correction_weight")),
        "support_correction_metric": summary.get("support_correction_metric", diagnostics.get("support_correction_metric")),
        "support_correction_target": summary.get("support_correction_target", diagnostics.get("support_correction_target")),
        "support_weighting_mode": summary.get("support_weighting_mode", diagnostics.get("support_weighting_mode")),
        "support_weight_scale": summary.get("support_weight_scale", diagnostics.get("support_weight_scale")),
        "final_support_weighted_bellman_loss": summary.get(
            "final_support_weighted_bellman_loss",
            support_weighted_bellman_losses[-1]
            if support_weighted_bellman_losses
            else diagnostics.get("final_support_weighted_bellman_loss"),
        ),
        "mean_support_weighted_bellman_loss": summary.get(
            "mean_support_weighted_bellman_loss",
            _mean_numeric(support_weighted_bellman_losses)
            if support_weighted_bellman_losses
            else diagnostics.get("mean_support_weighted_bellman_loss"),
        ),
        "final_support_weight_mean": summary.get(
            "final_support_weight_mean",
            support_weight_means[-1] if support_weight_means else diagnostics.get("final_support_weight_mean"),
        ),
        "mean_support_weight_mean": summary.get(
            "mean_support_weight_mean",
            _mean_numeric(support_weight_means) if support_weight_means else diagnostics.get("mean_support_weight_mean"),
        ),
        "final_support_weight_min": summary.get(
            "final_support_weight_min",
            support_weight_mins[-1] if support_weight_mins else diagnostics.get("final_support_weight_min"),
        ),
        "final_support_weight_max": summary.get(
            "final_support_weight_max",
            support_weight_maxs[-1] if support_weight_maxs else diagnostics.get("final_support_weight_max"),
        ),
        "json_file": str(json_file),
        "num_steps": summary.get("num_steps"),
        "final_total_loss": summary.get("final_total_loss"),
        "final_bellman_loss": bellman_losses[-1] if bellman_losses else None,
        "mean_bellman_loss": _mean_numeric(bellman_losses),
        "final_raw_bellman_loss": summary.get("final_raw_bellman_loss", bellman_losses[-1] if bellman_losses else None),
        "mean_raw_bellman_loss": summary.get("mean_raw_bellman_loss", _mean_numeric(bellman_losses)),
        "final_critic_total_loss": summary.get(
            "final_critic_total_loss",
            critic_total_losses[-1] if critic_total_losses else diagnostics.get("final_critic_total_loss"),
        ),
        "mean_critic_total_loss": summary.get(
            "mean_critic_total_loss",
            _mean_numeric(critic_total_losses) if critic_total_losses else diagnostics.get("mean_critic_total_loss"),
        ),
        "final_stationarity_loss": stationarity_losses[-1] if stationarity_losses else None,
        "mean_stationarity_loss": _mean_numeric(stationarity_losses),
        "final_density_regularization": density_regularizations[-1] if density_regularizations else None,
        "mean_density_regularization": _mean_numeric(density_regularizations),
        "final_index_age": diagnostics.get("final_index_age"),
        "index_refresh_count": diagnostics.get("index_refresh_count"),
        "final_stale_index_recall_mean": stale_recall[-1] if stale_recall else None,
        "final_stale_index_overlap_fraction_mean": stale_overlap[-1] if stale_overlap else None,
        "final_stale_index_best_value_gap_mean": stale_best_gap[-1] if stale_best_gap else None,
        "final_stale_index_pmr_mean": stale_pmr[-1] if stale_pmr else None,
        "final_stale_index_pmr_normalized_mean": (
            stale_pmr_normalized[-1]
            if stale_pmr_normalized
            else diagnostics.get("final_stale_index_pmr_normalized_mean")
        ),
        "final_stale_index_missing_mass_mean": (
            stale_missing_mass[-1]
            if stale_missing_mass
            else diagnostics.get("final_stale_index_missing_mass_mean")
        ),
        "uniform_pmr_baseline": uniform_baseline,
        "final_density_ess": density_ess[-1] if density_ess else diagnostics.get("final_density_ess"),
        "final_weight_entropy": weight_entropy[-1] if weight_entropy else diagnostics.get("final_weight_entropy"),
        "final_index_encoder_drift_mean_l2": diagnostics.get("final_index_encoder_drift_mean_l2"),
        "final_index_encoder_drift_mean_cosine_distance": diagnostics.get(
            "final_index_encoder_drift_mean_cosine_distance"
        ),
    }


def _summary_mode_is_supported(summary: dict[str, Any]) -> bool:
    return summary.get("mode") in {"synthetic_diagnostics_sweep", "support_loss_ablation"}


def write_sweep_summary(summary: dict[str, Any], output_dir: str | Path) -> Path:
    """Write the aggregate sweep summary JSON and return its path."""

    summary_file = Path(output_dir) / "summary.json"
    write_json_report(summary, summary_file)
    return summary_file


def _read_json_file(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        try:
            loaded = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON report must contain an object: {json_path}")
    return loaded


def _validate_report_metadata(report: dict[str, Any], *, expected_mode: str | None = None) -> None:
    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    if report.get("schema_name") != DIAGNOSTICS_SCHEMA_NAME:
        raise ValueError("Report schema_name does not match DW-MQL diagnostics schema.")
    if report.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("Report schema_version does not match the supported diagnostics schema version.")
    if expected_mode is not None and report.get("mode") != expected_mode:
        raise ValueError(f"Report mode must be {expected_mode!r}.")
    if report.get("synthetic_only") is not True:
        raise ValueError("Report must have synthetic_only=true.")
    if report.get("benchmark_claim") is not False:
        raise ValueError("Report must have benchmark_claim=false.")


def load_sweep_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate a diagnostics-only sweep ``summary.json`` file."""

    summary = _read_json_file(path)
    _validate_report_metadata(summary, expected_mode="synthetic_diagnostics_sweep")
    for field_name in ("runs", "run_files"):
        if field_name not in summary:
            raise ValueError(f"Sweep summary is missing required field {field_name!r}.")
        if not isinstance(summary[field_name], list):
            raise ValueError(f"Sweep summary field {field_name!r} must be a list.")
    return summary


def load_support_loss_ablation_summary(path: str | Path) -> dict[str, Any]:
    """Load a synthetic support-loss ablation summary."""

    summary = _read_json_file(path)
    _validate_report_metadata(summary, expected_mode="support_loss_ablation")
    if summary.get("benchmark_claim") is not False or summary.get("synthetic_only") is not True:
        raise ValueError("Support-loss ablation summary must be synthetic_only=true and benchmark_claim=false.")
    if not isinstance(summary.get("runs"), list):
        raise ValueError("Support-loss ablation summary must include runs as a list.")
    return summary


def load_sweep_run_reports(
    summary: dict[str, Any],
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate per-run JSON reports referenced by a sweep summary."""

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    _validate_report_metadata(summary, expected_mode="synthetic_diagnostics_sweep")
    run_files = summary.get("run_files")
    if not isinstance(run_files, list):
        raise ValueError("Sweep summary field 'run_files' must be present and must be a list.")
    if base_dir is None:
        output_dir = summary.get("output_dir")
        if not output_dir:
            raise ValueError("base_dir is required when summary does not contain output_dir.")
        base_path = Path(output_dir)
    else:
        base_path = Path(base_dir)

    reports: list[dict[str, Any]] = []
    for run_file in run_files:
        if not isinstance(run_file, str):
            raise ValueError("run_files entries must be strings.")
        report = _read_json_file(base_path / run_file)
        _validate_report_metadata(report)
        reports.append(report)
    return reports


def summarize_sweep_diagnostics_table(
    summary: dict[str, Any],
    sort_key: str = "final_stale_index_recall_mean",
) -> list[dict[str, Any]]:
    """Return compact sweep rows sorted by a diagnostics key.

    This is diagnostics-only summary inspection. Missing optional diagnostics
    are preserved as ``None`` and sorted after rows with numeric values.
    """

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Sweep summary field 'runs' must be present and must be a list.")
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Sweep summary runs must contain dictionaries.")
        rows.append(
            {
                "seed": run.get("seed"),
                "index_refresh_every": run.get("index_refresh_every"),
                "index_encoder_policy": run.get("index_encoder_policy"),
                "k": run.get("k"),
                "index_ema_tau": run.get("index_ema_tau"),
                "final_stale_index_recall_mean": run.get("final_stale_index_recall_mean"),
                "final_stale_index_pmr_mean": run.get("final_stale_index_pmr_mean"),
                "final_stale_index_overlap_fraction_mean": run.get(
                    "final_stale_index_overlap_fraction_mean"
                ),
                "final_index_encoder_drift_mean_l2": run.get("final_index_encoder_drift_mean_l2"),
                "final_index_age": run.get("final_index_age"),
                "json_file": run.get("json_file"),
            }
        )

    def sort_value(row: dict[str, Any]) -> tuple[int, float]:
        value = row.get(sort_key)
        if isinstance(value, int | float):
            return (0, -float(value))
        return (1, 0.0)

    return sorted(rows, key=sort_value)


def grouped_sweep_diagnostics(
    summary: dict[str, Any],
    group_keys: tuple[str, ...] = ("index_encoder_policy", "index_refresh_every"),
    metric_keys: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate sweep diagnostics by group keys.

    Means and population standard deviations are computed over available
    numeric values only. Missing metrics are skipped for that group.
    """

    if metric_keys is None:
        metric_keys = (
            "final_stale_index_recall_mean",
            "final_stale_index_pmr_mean",
            "final_stale_index_pmr_normalized_mean",
            "final_stale_index_missing_mass_mean",
            "uniform_pmr_baseline",
            "final_stale_index_overlap_fraction_mean",
            "final_stale_index_best_value_gap_mean",
            "final_raw_bellman_loss",
            "mean_raw_bellman_loss",
            "final_support_weighted_bellman_loss",
            "mean_support_weighted_bellman_loss",
            "final_support_weight_mean",
            "mean_support_weight_mean",
            "final_critic_total_loss",
            "mean_critic_total_loss",
            "final_support_correction_loss",
            "mean_support_correction_loss",
            "final_support_correction_contribution",
            "mean_support_correction_contribution",
            "final_bellman_loss",
            "mean_bellman_loss",
            "final_raw_bellman_loss",
            "mean_raw_bellman_loss",
            "final_support_weighted_bellman_loss",
            "mean_support_weighted_bellman_loss",
            "final_critic_total_loss",
            "mean_critic_total_loss",
            "final_stationarity_loss",
            "mean_stationarity_loss",
            "final_density_regularization",
            "mean_density_regularization",
            "final_density_ess",
            "final_weight_entropy",
            "final_index_encoder_drift_mean_l2",
            "final_index_encoder_drift_mean_cosine_distance",
            "final_index_age",
        )
    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    if not isinstance(group_keys, tuple) or not group_keys:
        raise ValueError("group_keys must be a non-empty tuple.")
    if not isinstance(metric_keys, tuple) or not metric_keys:
        raise ValueError("metric_keys must be a non-empty tuple.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Sweep summary field 'runs' must be present and must be a list.")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Sweep summary runs must contain dictionaries.")
        missing_group_keys = [key for key in group_keys if key not in run]
        if missing_group_keys:
            raise ValueError(f"Sweep run is missing requested group key(s): {missing_group_keys}.")
        group = tuple(run.get(key) for key in group_keys)
        groups.setdefault(group, []).append(run)

    aggregated: list[dict[str, Any]] = []
    for group, group_runs in groups.items():
        row = {key: value for key, value in zip(group_keys, group, strict=True)}
        row["count"] = len(group_runs)
        for metric_key in metric_keys:
            values = [
                float(run[metric_key])
                for run in group_runs
                if isinstance(run.get(metric_key), int | float)
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            row[f"{metric_key}_mean"] = mean
            row[f"{metric_key}_std"] = variance**0.5 if len(values) > 1 else 0.0
        aggregated.append(row)

    return sorted(aggregated, key=lambda row: tuple(row.get(key) for key in group_keys))


def _pearson_or_none(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    count = len(pairs)
    if count < 2:
        return {"count": count, "pearson_r": None}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    var_x = sum(value * value for value in centered_x)
    var_y = sum(value * value for value in centered_y)
    if var_x <= 0.0 or var_y <= 0.0:
        return {"count": count, "pearson_r": None}
    cov = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    return {"count": count, "pearson_r": cov / ((var_x * var_y) ** 0.5)}


def compute_sweep_metric_associations(
    summary: dict[str, Any],
    x_metrics: tuple[str, ...] | None = None,
    y_metrics: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute diagnostics-only Pearson associations between sweep metrics.

    The result is descriptive only: no p-values are computed and no causality is
    implied. Missing metrics, insufficient data, or zero variance produce
    ``pearson_r=None`` for that metric pair.
    """

    if x_metrics is None:
        x_metrics = (
            "final_stale_index_recall_mean",
            "final_stale_index_pmr_mean",
            "final_stale_index_pmr_normalized_mean",
            "final_stale_index_overlap_fraction_mean",
            "final_stale_index_best_value_gap_mean",
            "final_index_encoder_drift_mean_l2",
            "final_index_encoder_drift_mean_cosine_distance",
        )
    if y_metrics is None:
        y_metrics = (
            "final_bellman_loss",
            "mean_bellman_loss",
            "final_stationarity_loss",
            "mean_stationarity_loss",
            "final_density_regularization",
            "mean_density_regularization",
            "final_density_ess",
            "final_weight_entropy",
        )
    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Sweep summary field 'runs' must be present and must be a list.")

    associations: dict[str, Any] = {}
    for x_metric in x_metrics:
        for y_metric in y_metrics:
            pairs: list[tuple[float, float]] = []
            for run in runs:
                if not isinstance(run, dict):
                    raise ValueError("Sweep summary runs must contain dictionaries.")
                x_value = run.get(x_metric)
                y_value = run.get(y_metric)
                if isinstance(x_value, int | float) and isinstance(y_value, int | float):
                    pairs.append((float(x_value), float(y_value)))
            associations[f"{x_metric}__vs__{y_metric}"] = {
                "x_metric": x_metric,
                "y_metric": y_metric,
                **_pearson_or_none(pairs),
            }
    return {
        "synthetic_only": True,
        "benchmark_claim": False,
        "causality_implied": False,
        "associations": associations,
    }


def compute_support_loss_objective_associations(
    summary: dict[str, Any],
    x_metrics: tuple[str, ...] | None = None,
    y_metrics: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute descriptive associations between support loss and objectives."""

    if x_metrics is None:
        x_metrics = (
            "final_stale_index_pmr_mean",
            "final_stale_index_pmr_normalized_mean",
            "final_stale_index_recall_mean",
            "final_stale_index_overlap_fraction_mean",
            "final_stale_index_best_value_gap_mean",
            "final_index_encoder_drift_mean_l2",
        )
    if y_metrics is None:
        y_metrics = (
            "final_bellman_loss",
            "mean_bellman_loss",
            "final_stationarity_loss",
            "mean_stationarity_loss",
            "final_density_regularization",
            "mean_density_regularization",
            "final_density_ess",
            "final_weight_entropy",
        )
    result = compute_sweep_metric_associations(summary, x_metrics=x_metrics, y_metrics=y_metrics)
    result["diagnostic_association_only"] = True
    result["support_loss_associations"] = True
    return result


def _default_correction_pairing_keys() -> tuple[str, ...]:
    return ("seed", "k", "index_refresh_every", "index_encoder_policy", "index_ema_tau")


def _default_correction_metric_keys() -> tuple[str, ...]:
    return (
        "final_raw_bellman_loss",
        "mean_raw_bellman_loss",
        "final_support_weighted_bellman_loss",
        "mean_support_weighted_bellman_loss",
        "final_support_weight_mean",
        "mean_support_weight_mean",
        "final_critic_total_loss",
        "mean_critic_total_loss",
        "final_support_correction_loss",
        "mean_support_correction_loss",
        "final_support_correction_contribution",
        "mean_support_correction_contribution",
        "final_bellman_loss",
        "mean_bellman_loss",
        "final_stationarity_loss",
        "mean_stationarity_loss",
        "final_density_regularization",
        "mean_density_regularization",
        "final_density_ess",
        "final_weight_entropy",
        "final_stale_index_pmr_mean",
        "final_stale_index_pmr_normalized_mean",
        "final_stale_index_missing_mass_mean",
    )


def _delta_summary(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    count = len(pairs)
    if count == 0:
        return {
            "count": 0,
            "reference_mean": None,
            "comparison_mean": None,
            "mean_delta": None,
            "std_delta": None,
            "fraction_delta_negative": None,
            "fraction_delta_positive": None,
        }
    reference_values = [pair[0] for pair in pairs]
    comparison_values = [pair[1] for pair in pairs]
    deltas = [reference - comparison for reference, comparison in pairs]
    mean_delta = sum(deltas) / count
    variance = sum((delta - mean_delta) ** 2 for delta in deltas) / count
    return {
        "count": count,
        "reference_mean": sum(reference_values) / count,
        "comparison_mean": sum(comparison_values) / count,
        "mean_delta": mean_delta,
        "std_delta": variance**0.5 if count > 1 else 0.0,
        "fraction_delta_negative": sum(1 for delta in deltas if delta < 0.0) / count,
        "fraction_delta_positive": sum(1 for delta in deltas if delta > 0.0) / count,
    }


def paired_correction_mode_contrasts(
    summary: dict[str, Any],
    reference_mode: str = "aware",
    comparison_modes: tuple[str, ...] = ("naive", "uniform_weights"),
    group_keys: tuple[str, ...] | None = None,
    metric_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute paired correction-mode contrasts for support-loss ablations.

    Pairing is by seed, k, refresh cadence, encoder policy, and EMA tau by
    default. Deltas are reference minus comparison. This is descriptive only:
    no p-values, significance claims, or causal claims are produced.
    """

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Sweep summary field 'runs' must be present and must be a list.")
    if not isinstance(reference_mode, str) or not reference_mode:
        raise ValueError("reference_mode must be a non-empty string.")
    if not isinstance(comparison_modes, tuple) or not comparison_modes:
        raise ValueError("comparison_modes must be a non-empty tuple.")
    if group_keys is None:
        group_keys = _default_correction_pairing_keys()
    if metric_keys is None:
        metric_keys = _default_correction_metric_keys()
    if not isinstance(group_keys, tuple) or not group_keys:
        raise ValueError("group_keys must be a non-empty tuple.")
    if not isinstance(metric_keys, tuple) or not metric_keys:
        raise ValueError("metric_keys must be a non-empty tuple.")

    reference_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    comparison_by_mode: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {
        mode: {} for mode in comparison_modes
    }
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Sweep summary runs must contain dictionaries.")
        if "retrieval_correction_mode" not in run:
            raise ValueError("Sweep run is missing retrieval_correction_mode.")
        missing_group_keys = [key for key in group_keys if key not in run]
        if missing_group_keys:
            raise ValueError(f"Sweep run is missing requested pairing key(s): {missing_group_keys}.")
        pairing_key = tuple(run.get(key) for key in group_keys)
        mode = run.get("retrieval_correction_mode")
        if mode == reference_mode:
            reference_by_key.setdefault(pairing_key, []).append(run)
        elif mode in comparison_by_mode:
            comparison_by_mode[str(mode)][pairing_key] = run

    comparisons: dict[str, Any] = {}
    for mode in comparison_modes:
        matched_keys = sorted(
            set(reference_by_key).intersection(comparison_by_mode[mode]),
            key=lambda values: tuple("" if value is None else str(value) for value in values),
        )
        metrics: dict[str, Any] = {}
        for metric_key in metric_keys:
            pairs: list[tuple[float, float]] = []
            for pairing_key in matched_keys:
                comparison_value = comparison_by_mode[mode][pairing_key].get(metric_key)
                if not isinstance(comparison_value, int | float):
                    continue
                for reference_run in reference_by_key[pairing_key]:
                    reference_value = reference_run.get(metric_key)
                    if isinstance(reference_value, int | float):
                        pairs.append((float(reference_value), float(comparison_value)))
            metrics[metric_key] = _delta_summary(pairs)
        comparisons[mode] = {
            "paired_count": sum(len(reference_by_key[key]) for key in matched_keys),
            "metrics": metrics,
        }

    return {
        "synthetic_only": True,
        "benchmark_claim": False,
        "causality_implied": False,
        "significance_claim": False,
        "reference_mode": reference_mode,
        "comparison_modes": list(comparison_modes),
        "pairing_keys": list(group_keys),
        "metric_keys": list(metric_keys),
        "comparisons": comparisons,
    }


def binned_support_loss_contrasts(
    summary: dict[str, Any],
    support_metric: str = "final_stale_index_pmr_normalized_mean",
) -> dict[str, Any]:
    """Compute paired correction-mode contrasts in low/medium/high support bins."""

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Sweep summary field 'runs' must be present and must be a list.")
    pairing_keys = _default_correction_pairing_keys()
    reference_rows: list[tuple[tuple[Any, ...], float]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Sweep summary runs must contain dictionaries.")
        if run.get("retrieval_correction_mode") != "aware":
            continue
        missing_group_keys = [key for key in pairing_keys if key not in run]
        if missing_group_keys:
            raise ValueError(f"Sweep run is missing requested pairing key(s): {missing_group_keys}.")
        value = run.get(support_metric)
        if isinstance(value, int | float):
            reference_rows.append((tuple(run.get(key) for key in pairing_keys), float(value)))
    reference_rows.sort(key=lambda item: item[1])
    count = len(reference_rows)
    if count == 0:
        bins = [
            {"bin": "low", "count": 0, "support_min": None, "support_max": None, "contrasts": paired_correction_mode_contrasts({"runs": []})},
            {"bin": "medium", "count": 0, "support_min": None, "support_max": None, "contrasts": paired_correction_mode_contrasts({"runs": []})},
            {"bin": "high", "count": 0, "support_min": None, "support_max": None, "contrasts": paired_correction_mode_contrasts({"runs": []})},
        ]
    else:
        low_end = count // 3
        medium_end = (2 * count) // 3
        slices = {
            "low": reference_rows[:low_end],
            "medium": reference_rows[low_end:medium_end],
            "high": reference_rows[medium_end:],
        }
        bins = []
        for name in ("low", "medium", "high"):
            selected = slices[name]
            selected_keys = {key for key, _ in selected}
            selected_runs = [
                run for run in runs
                if isinstance(run, dict)
                and tuple(run.get(key) for key in pairing_keys) in selected_keys
            ]
            values = [value for _, value in selected]
            bins.append(
                {
                    "bin": name,
                    "count": len(selected),
                    "support_min": min(values) if values else None,
                    "support_max": max(values) if values else None,
                    "contrasts": paired_correction_mode_contrasts({"runs": selected_runs}),
                }
            )

    return {
        "synthetic_only": True,
        "benchmark_claim": False,
        "causality_implied": False,
        "support_metric": support_metric,
        "bins": bins,
    }


def run_support_loss_ablation(
    base_profile: SmokeProfile,
    output_dir: str | Path,
    retrieval_correction_modes: tuple[str, ...] = ("aware", "naive"),
    seeds: tuple[int, ...] = (0,),
    k_values: tuple[int, ...] = (10, 40),
    index_refresh_values: tuple[int, ...] = (10, 50),
    index_encoder_policies: tuple[str, ...] = ("online", "ema"),
    index_ema_tau_values: tuple[float, ...] | None = None,
    support_correction_weight_values: tuple[float, ...] | None = None,
    support_weighting_modes: tuple[str, ...] | None = None,
    support_weight_scale_values: tuple[float, ...] | None = None,
    num_steps: int | None = None,
) -> SupportLossAblationResult:
    """Run a synthetic support-loss ablation grid.

    This is a diagnostics-only runner. It compares explicit retrieval
    correction modes without asserting performance or causality.
    """

    from .trainer import (
        SUPPORT_WEIGHTING_MODES,
        RETRIEVAL_CORRECTION_MODES,
        validate_retrieval_correction_mode,
        validate_support_weighting_config,
    )

    if not isinstance(base_profile, SmokeProfile):
        raise TypeError("base_profile must be a SmokeProfile.")
    output_path = Path(output_dir)
    if output_path.exists() and output_path.is_file():
        raise ValueError("output_dir must be a directory path, not an existing file.")
    modes = tuple(validate_retrieval_correction_mode(mode) for mode in retrieval_correction_modes)
    if not modes:
        raise ValueError("retrieval_correction_modes must be non-empty.")
    seed_values = _validate_non_empty_ints(tuple(seeds), name="seeds")
    candidate_k_values = _validate_non_empty_positive_ints(tuple(k_values), name="k_values")
    refresh_values = _validate_non_empty_positive_ints(tuple(index_refresh_values), name="index_refresh_values")
    policies = _validate_encoder_policies(tuple(index_encoder_policies))
    for candidate_k in candidate_k_values:
        if candidate_k > base_profile.num_actions:
            raise ValueError("k_values entries must be <= base_profile.num_actions.")
    tau_values: tuple[float, ...] | None = None
    if index_ema_tau_values is not None:
        if not isinstance(index_ema_tau_values, tuple) or not index_ema_tau_values:
            raise ValueError("index_ema_tau_values must be None or a non-empty tuple.")
        tau_values = tuple(float(value) for value in index_ema_tau_values)
        for tau in tau_values:
            if not 0.0 <= tau <= 1.0:
                raise ValueError("index_ema_tau_values entries must be in [0, 1].")
    if "ema" in policies and base_profile.index_ema_tau is None and tau_values is None:
        raise ValueError("EMA ablation policy requires base_profile.index_ema_tau or index_ema_tau_values.")
    if "target" in policies and base_profile.target_tau is None:
        raise ValueError("target ablation policy requires base_profile.target_tau.")
    support_weight_values: tuple[float, ...] | None = None
    if support_correction_weight_values is not None:
        if not isinstance(support_correction_weight_values, tuple) or not support_correction_weight_values:
            raise ValueError("support_correction_weight_values must be None or a non-empty tuple.")
        support_weight_values = tuple(float(value) for value in support_correction_weight_values)
        if any(value < 0.0 for value in support_weight_values):
            raise ValueError("support_correction_weight_values entries must be nonnegative.")
    weighting_modes: tuple[str, ...] | None = None
    if support_weighting_modes is not None:
        if not isinstance(support_weighting_modes, tuple) or not support_weighting_modes:
            raise ValueError("support_weighting_modes must be None or a non-empty tuple.")
        unknown = set(support_weighting_modes) - set(SUPPORT_WEIGHTING_MODES)
        if unknown:
            raise ValueError(f"Unknown support weighting mode value(s): {sorted(unknown)}.")
        weighting_modes = tuple(str(value) for value in support_weighting_modes)
    weight_scale_values: tuple[float, ...] | None = None
    if support_weight_scale_values is not None:
        if not isinstance(support_weight_scale_values, tuple) or not support_weight_scale_values:
            raise ValueError("support_weight_scale_values must be None or a non-empty tuple.")
        weight_scale_values = tuple(float(value) for value in support_weight_scale_values)
        if any(value < 0.0 for value in weight_scale_values):
            raise ValueError("support_weight_scale_values entries must be nonnegative.")
    validate_support_weighting_config(
        mode=base_profile.support_weighting_mode,
        scale=base_profile.support_weight_scale,
        min_weight=base_profile.support_weight_min,
        max_weight=base_profile.support_weight_max,
        eps=base_profile.support_pmr_eps,
    )
    if num_steps is not None and (not isinstance(num_steps, int) or num_steps <= 0):
        raise ValueError("num_steps must be None or a positive integer.")

    output_path.mkdir(parents=True, exist_ok=True)
    run_reports: list[dict[str, Any]] = []
    run_files: list[str] = []
    run_summaries: list[dict[str, Any]] = []

    for mode in modes:
        mode_support_weight_values: tuple[float, ...] = (
            support_weight_values
            if mode == "aware" and support_weight_values is not None
            else (0.0,)
        )
        mode_support_weighting_modes: tuple[str, ...] = (
            weighting_modes
            if mode == "aware" and weighting_modes is not None
            else ((base_profile.support_weighting_mode,) if mode == "aware" else ("none",))
        )
        mode_support_weight_scale_values: tuple[float, ...] = (
            weight_scale_values
            if mode == "aware" and weight_scale_values is not None
            else (base_profile.support_weight_scale,)
        )
        for seed in seed_values:
            for refresh in refresh_values:
                for policy in policies:
                    policy_tau_values: tuple[float | None, ...] = (
                        tau_values if policy == "ema" and tau_values is not None else
                        ((base_profile.index_ema_tau,) if policy == "ema" else (None,))
                    )
                    for candidate_k in candidate_k_values:
                        for tau in policy_tau_values:
                            for support_weight in mode_support_weight_values:
                                for support_weighting_mode in mode_support_weighting_modes:
                                    for support_weight_scale in mode_support_weight_scale_values:
                                        effective_support_weight = float(support_weight) if mode == "aware" else 0.0
                                        effective_support_weighting_mode = (
                                            support_weighting_mode if mode == "aware" else "none"
                                        )
                                        effective_support_weight_scale = (
                                            float(support_weight_scale)
                                            if mode == "aware"
                                            else base_profile.support_weight_scale
                                        )
                                        profile = replace(
                                            base_profile,
                                            seed=seed,
                                            k=candidate_k,
                                            index_refresh_every=refresh,
                                            index_encoder_policy=policy,
                                            index_ema_tau=None if policy != "ema" else float(tau),
                                            retrieval_correction_mode=mode,
                                            support_correction_weight=effective_support_weight,
                                            support_weighting_mode=effective_support_weighting_mode,
                                            support_weight_scale=effective_support_weight_scale,
                                            num_steps=base_profile.num_steps if num_steps is None else num_steps,
                                            use_persistent_index=True,
                                            audit_stale_index=True,
                                        )
                                        validate_smoke_profile(profile)
                                        trainer, _ = _build_trainer(profile)
                                        history = trainer.train(profile.num_steps)
                                        report = _report_for_history(history, profile)
                                        report["mode"] = "support_loss_ablation_run"
                                        report["support_loss_ablation"] = {
                                            "seed": seed,
                                            "k": candidate_k,
                                            "index_refresh_every": refresh,
                                            "index_encoder_policy": policy,
                                            "index_ema_tau": None if policy != "ema" else float(tau),
                                            "retrieval_correction_mode": mode,
                                            "support_correction_weight": effective_support_weight,
                                            "support_weighting_mode": effective_support_weighting_mode,
                                            "support_weight_scale": effective_support_weight_scale,
                                        }
                                        report_path = output_path / _safe_ablation_filename(
                                            seed,
                                            refresh,
                                            policy,
                                            candidate_k,
                                            mode,
                                            None if policy != "ema" or tau_values is None else float(tau),
                                            effective_support_weight if support_weight_values is not None else None,
                                            effective_support_weighting_mode if weighting_modes is not None else None,
                                            effective_support_weight_scale if weight_scale_values is not None else None,
                                        )
                                        write_json_report(report, report_path)
                                        run_files.append(report_path.name)
                                        compact = summarize_sweep_run(report, report_path.name)
                                        compact.update(report["support_loss_ablation"])
                                        run_summaries.append(compact)
                                        run_reports.append(report)

    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "support_loss_ablation",
        "synthetic_only": True,
        "benchmark_claim": False,
        "diagnostic_association_only": True,
        "num_runs": len(run_reports),
        "output_dir": str(output_path),
        "run_files": run_files,
        "sweep_axes": {
            "seeds": list(seed_values),
            "k_values": list(candidate_k_values),
            "index_refresh_values": list(refresh_values),
            "index_encoder_policies": list(policies),
            "retrieval_correction_modes": list(modes),
            **({} if tau_values is None else {"index_ema_tau_values": list(tau_values)}),
            **({} if support_weight_values is None else {"support_correction_weight_values": list(support_weight_values)}),
            **({} if weighting_modes is None else {"support_weighting_modes": list(weighting_modes)}),
            **({} if weight_scale_values is None else {"support_weight_scale_values": list(weight_scale_values)}),
        },
        "runs": run_summaries,
    }
    summary["support_loss_objective_associations"] = compute_support_loss_objective_associations(summary)
    summary_file = write_sweep_summary(summary, output_path)
    manifest_file = output_path / "manifest.json"
    write_run_manifest(
        RunManifest(
            run_type="support_loss_ablation",
            output_dir=str(output_path),
            report_files=[str(output_path / file_name) for file_name in run_files],
            summary_files=[str(summary_file)],
            metadata={
                "sweep_axes": summary["sweep_axes"],
                "num_runs": len(run_reports),
                "diagnostics_only": True,
                "benchmark_claim": False,
            },
        ),
        manifest_file,
    )
    return SupportLossAblationResult(
        run_reports=run_reports,
        output_dir=output_path,
        num_runs=len(run_reports),
        summary_file=summary_file,
        manifest_file=manifest_file,
        summary=summary,
        diagnostics={
            "synthetic_only": True,
            "benchmark_claim": False,
            "retrieval_correction_modes": list(modes),
            "manifest_file": str(manifest_file),
        },
    )


def support_loss_ablation_profile_from_yaml(path: str | Path) -> tuple[SmokeProfile, dict[str, Any]]:
    """Load conservative support-loss ablation defaults from YAML."""

    loaded = _load_support_ablation_yaml(path)
    profile = default_smoke_profile()
    smoke_key_map = {
        "seed": "seed",
        "batch_size": "batch_size",
        "num_transitions": "num_transitions",
        "num_actions": "num_actions",
        "state_dim": "state_dim",
        "action_dim": "action_dim",
        "embedding_dim": "embedding_dim",
        "gamma": "gamma",
        "num_steps": "num_steps",
        "lr": "lr",
        "target_tau": "target_tau",
        "target_update_every": "target_update_every",
        "index_backend": "index_backend",
        "index_metric": "index_metric",
        "index_refresh_every": "index_refresh_every",
        "index_encoder_policy": "index_encoder_policy",
        "index_ema_tau": "index_ema_tau",
        "support_correction_weight": "support_correction_weight",
        "support_correction_target": "support_correction_target",
        "support_correction_metric": "support_correction_metric",
        "support_weighting_mode": "support_weighting_mode",
        "support_weight_scale": "support_weight_scale",
        "support_weight_min": "support_weight_min",
        "support_weight_max": "support_weight_max",
        "support_pmr_eps": "support_pmr_eps",
    }
    for source, target in smoke_key_map.items():
        if source in loaded and loaded[source] is not None:
            setattr(profile, target, loaded[source])
    profile.use_persistent_index = True
    profile.audit_stale_index = True
    if profile.index_encoder_policy == "ema" and profile.index_ema_tau is None:
        profile.index_ema_tau = 0.05
    options = {
        "synthetic_only": bool(loaded.get("synthetic_only", True)),
        "benchmark_claim": bool(loaded.get("benchmark_claim", False)),
        "seeds": tuple(int(value) for value in loaded.get("seeds", [profile.seed])),
        "k_values": tuple(int(value) for value in loaded.get("k_values", [profile.k])),
        "index_refresh_values": tuple(int(value) for value in loaded.get("refresh_values", [profile.index_refresh_every])),
        "index_encoder_policies": tuple(str(value) for value in loaded.get("encoder_policies", [profile.index_encoder_policy])),
        "retrieval_correction_modes": tuple(str(value) for value in loaded.get("retrieval_correction_modes", [profile.retrieval_correction_mode])),
        "index_ema_tau_values": (
            None
            if loaded.get("ema_tau_values") is None
            else tuple(float(value) for value in loaded.get("ema_tau_values"))
        ),
        "support_correction_weight_values": (
            None
            if loaded.get("support_correction_weight_values") is None
            else tuple(float(value) for value in loaded.get("support_correction_weight_values"))
        ),
        "support_weighting_modes": (
            None
            if loaded.get("support_weighting_modes") is None
            else tuple(str(value) for value in loaded.get("support_weighting_modes"))
        ),
        "support_weight_scale_values": (
            None
            if loaded.get("support_weight_scale_values") is None
            else tuple(float(value) for value in loaded.get("support_weight_scale_values"))
        ),
    }
    if options["synthetic_only"] is not True:
        raise ValueError("support-loss ablation config must have synthetic_only: true.")
    if options["benchmark_claim"] is not False:
        raise ValueError("support-loss ablation config must have benchmark_claim: false.")
    return validate_smoke_profile(profile), options


def run_synthetic_diagnostics_sweep(
    base_profile: SmokeProfile,
    output_dir: str | Path,
    index_refresh_values: tuple[int, ...] = (1, 5),
    index_encoder_policies: tuple[str, ...] = ("online",),
    seeds: tuple[int, ...] = (0,),
    num_steps: int | None = None,
    use_persistent_index: bool = True,
    audit_stale_index: bool = True,
    audit_encoder_policy_overlap: bool = False,
    k_values: tuple[int, ...] | None = None,
    index_ema_tau_values: tuple[float, ...] | None = None,
) -> SyntheticSweepResult:
    """Run a tiny synthetic diagnostics sweep and write JSON reports.

    The sweep varies seed, index refresh cadence, and index encoder policy. It
    is for diagnostics and plotting only; it is not a benchmark runner.
    """

    if not isinstance(base_profile, SmokeProfile):
        raise TypeError("base_profile must be a SmokeProfile.")
    output_path = Path(output_dir)
    if output_path.exists() and output_path.is_file():
        raise ValueError("output_dir must be a directory path, not an existing file.")
    refresh_values = _validate_non_empty_positive_ints(tuple(index_refresh_values), name="index_refresh_values")
    seed_values = _validate_non_empty_ints(tuple(seeds), name="seeds")
    policies = _validate_encoder_policies(tuple(index_encoder_policies))
    candidate_k_values = (
        (base_profile.k,)
        if k_values is None
        else _validate_non_empty_positive_ints(tuple(k_values), name="k_values")
    )
    for candidate_k in candidate_k_values:
        if candidate_k > base_profile.num_actions:
            raise ValueError("k_values entries must be <= base_profile.num_actions.")
    if index_ema_tau_values is not None:
        if not isinstance(index_ema_tau_values, tuple) or not index_ema_tau_values:
            raise ValueError("index_ema_tau_values must be None or a non-empty tuple.")
        for tau in index_ema_tau_values:
            if not isinstance(tau, int | float) or not 0.0 <= float(tau) <= 1.0:
                raise ValueError("index_ema_tau_values entries must be floats in [0, 1].")
        tau_values = tuple(float(tau) for tau in index_ema_tau_values)
    else:
        tau_values = None
    if num_steps is not None and (not isinstance(num_steps, int) or num_steps <= 0):
        raise ValueError("num_steps must be None or a positive integer.")
    if "target" in policies and base_profile.target_tau is None:
        raise ValueError("index_encoder_policy='target' requires target_tau on base_profile.")
    if "ema" in policies and base_profile.index_ema_tau is None and tau_values is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau on base_profile.")

    output_path.mkdir(parents=True, exist_ok=True)
    run_reports: list[dict[str, Any]] = []
    run_files: list[str] = []
    run_summaries: list[dict[str, Any]] = []

    for seed in seed_values:
        for refresh in refresh_values:
            for policy in policies:
                policy_tau_values: tuple[float | None, ...]
                if policy == "ema":
                    policy_tau_values = tau_values if tau_values is not None else (base_profile.index_ema_tau,)
                else:
                    policy_tau_values = (None,)
                for candidate_k in candidate_k_values:
                    for tau in policy_tau_values:
                        if policy == "ema" and tau is None:
                            raise ValueError("EMA sweep policy requires an index_ema_tau value.")
                        run_index_ema_tau = None if policy != "ema" else float(tau)
                        profile = replace(
                            base_profile,
                            seed=seed,
                            index_refresh_every=refresh,
                            index_encoder_policy=policy,
                            index_ema_tau=run_index_ema_tau,
                            k=candidate_k,
                            num_steps=base_profile.num_steps if num_steps is None else num_steps,
                            use_persistent_index=use_persistent_index,
                            audit_stale_index=audit_stale_index,
                            audit_encoder_policy_overlap=audit_encoder_policy_overlap,
                        )
                        validate_smoke_profile(profile)
                        trainer, _ = _build_trainer(profile)
                        history = trainer.train(profile.num_steps)
                        report = _report_for_history(history, profile)
                        report["sweep"] = {
                            "seed": seed,
                            "index_refresh_every": refresh,
                            "index_encoder_policy": policy,
                            "k": candidate_k,
                            "index_ema_tau": run_index_ema_tau,
                            "retrieval_correction_mode": profile.retrieval_correction_mode,
                        }
                        report_path = output_path / _safe_run_filename(
                            seed,
                            refresh,
                            policy,
                            None if k_values is None else candidate_k,
                            None if not (policy == "ema" and tau_values is not None) else run_index_ema_tau,
                        )
                        write_json_report(report, report_path)
                        run_files.append(report_path.name)
                        run_summaries.append(summarize_sweep_run(report, report_path.name))
                        run_reports.append(report)

    from .train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION

    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": len(run_reports),
        "output_dir": str(output_path),
        "run_files": run_files,
        "sweep_axes": {
            "seeds": list(seed_values),
            "index_refresh_values": list(refresh_values),
            "index_encoder_policies": list(policies),
            **({} if k_values is None else {"k_values": list(candidate_k_values)}),
            **({} if tau_values is None else {"index_ema_tau_values": list(tau_values)}),
        },
        "runs": run_summaries,
    }
    summary_file = write_sweep_summary(summary, output_path)
    manifest_file = output_path / "manifest.json"
    write_run_manifest(
        RunManifest(
            run_type="synthetic_diagnostics_sweep",
            output_dir=str(output_path),
            report_files=[str(output_path / file_name) for file_name in run_files],
            summary_files=[str(summary_file)],
            metadata={
                "sweep_axes": summary["sweep_axes"],
                "num_runs": len(run_reports),
                "diagnostics_only": True,
            },
        ),
        manifest_file,
    )

    return SyntheticSweepResult(
        run_reports=run_reports,
        output_dir=output_path,
        num_runs=len(run_reports),
        summary_file=summary_file,
        manifest_file=manifest_file,
        summary=summary,
        diagnostics={
            "synthetic_only": True,
            "benchmark_claim": False,
            "manifest_file": str(manifest_file),
            "index_refresh_values": list(refresh_values),
            "index_encoder_policies": list(policies),
            "seeds": list(seed_values),
            "k_values": None if k_values is None else list(candidate_k_values),
            "index_ema_tau_values": None if tau_values is None else list(tau_values),
        },
    )
