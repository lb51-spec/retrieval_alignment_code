"""Validation and aggregation for real_data-facing experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .statistics import bootstrap_ci_mean, paired_mean_delta, summarize_values


@dataclass(slots=True)
class ArtifactRecord:
    """Normalized inventory entry for one experiment report."""

    path: Path
    dataset: str
    seed: int
    method: str
    component_identity: dict[str, Any]
    config_hash: str
    checkpoint_path: Path | None
    valid: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, float]


def _config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _legacy_component_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    training = report.get("training_history", {})
    diagnostics = training.get("diagnostics", {}) if isinstance(training, Mapping) else {}
    mode = diagnostics.get(
        "retrieval_correction_mode",
        config.get("retrieval_correction_mode", "aware"),
    )
    alignment_weight = float(config.get("retrieval_alignment_weight", 0.0) or 0.0)
    return {
        "critic_method": str(config.get("critic_method", "dwmql")),
        "density_enabled": mode != "no_density",
        "density_affects_shared_encoder": False,
        "retrieval_alignment_enabled": alignment_weight > 0.0,
        "retrieval_projection_enabled": (
            config.get("retrieval_alignment_loss_type", "listwise_kl") == "listwise_kl"
            and alignment_weight > 0.0
        ),
        "index_backend": str(config.get("index_backend", "unknown")),
        "index_metric": str(config.get("index_metric", "unknown")),
    }


def _flatten_finite_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(_flatten_finite_metrics(child, child_prefix))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            metrics[prefix] = number
    return metrics


def _dataset_name(report: Mapping[str, Any], path: Path) -> str:
    metadata = report.get("dataset_metadata", {})
    config = report.get("config", {})
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if source:
        return str(source)
    report_type = str(report.get("report_type", "")).lower()
    data_hint = str(config.get("data_path", config.get("data_dir", path))).lower()
    if "kuairand" in report_type or "kuairand" in data_hint:
        return "kuairand"
    if "movielens" in report_type or "movie" in data_hint:
        return "movielens"
    return "unknown"


def _method_name(config: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    explicit = config.get("real_data_method") or config.get("method")
    if explicit:
        return str(explicit)
    critic = str(identity.get("critic_method", "dwmql"))
    density = bool(identity.get("density_enabled"))
    alignment = bool(identity.get("retrieval_alignment_enabled"))
    if critic == "dwmql" and density and alignment:
        return "full"
    if not density and alignment:
        return "b2_only" if critic in {"dqn", "dwmql"} else f"{critic}_b2"
    if density and not alignment:
        return "density_only"
    if not density and not alignment and critic in {"dqn", "dwmql"}:
        return "base"
    return critic


def artifact_record_from_report(
    report: Mapping[str, Any],
    path: Path,
) -> ArtifactRecord:
    """Normalize one loaded report without silently relabelling legacy runs."""

    config = report.get("config", {})
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(config, Mapping):
        errors.append("config must be a mapping")
        config = {}
    explicit_identity = report.get("component_identity")
    if isinstance(explicit_identity, Mapping):
        identity = dict(explicit_identity)
    else:
        identity = _legacy_component_identity(report)
        warnings.append("legacy_missing_explicit_component_identity")
    try:
        seed = int(config.get("seed", -1))
    except (TypeError, ValueError):
        seed = -1
        errors.append("seed must be an integer")

    metric_source = report.get("real_data_metrics")
    if metric_source is None:
        metric_source = report.get("metrics", {})
    metrics = _flatten_finite_metrics(metric_source)
    if not metrics:
        warnings.append("no_real_data_metrics")

    output_files = report.get("output_files", {})
    checkpoint_value = (
        output_files.get("model_checkpoint")
        if isinstance(output_files, Mapping)
        else None
    )
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    return ArtifactRecord(
        path=Path(path),
        dataset=_dataset_name(report, Path(path)),
        seed=seed,
        method=_method_name(config, identity),
        component_identity=identity,
        config_hash=_config_hash(config),
        checkpoint_path=checkpoint_path,
        valid=not errors,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def inventory_reports(paths: Sequence[Path]) -> list[ArtifactRecord]:
    """Load every requested report, preserving invalid entries in the inventory."""

    records: list[ArtifactRecord] = []
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, Mapping):
                raise ValueError("report root must be a JSON object")
            records.append(artifact_record_from_report(report, path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            records.append(
                ArtifactRecord(
                    path=path,
                    dataset="unknown",
                    seed=-1,
                    method="unknown",
                    component_identity={},
                    config_hash="",
                    checkpoint_path=None,
                    valid=False,
                    errors=[f"{type(exc).__name__}: {exc}"],
                    warnings=[],
                    metrics={},
                )
            )
    return records


def validate_artifact_identity(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[str]:
    """Return exact identity mismatches between a report and run expectation."""

    config = report.get("config", {})
    identity = report.get("component_identity", {})
    errors: list[str] = []
    for key, expected_value in expected.items():
        if key in config:
            found = config[key]
        elif isinstance(identity, Mapping) and key in identity:
            found = identity[key]
        else:
            errors.append(f"{key} missing: expected {expected_value}")
            continue
        if found != expected_value:
            errors.append(f"{key} mismatch: expected {expected_value}, found {found}")
    return errors


def aggregate_real_data_results(
    records: Sequence[ArtifactRecord],
    comparisons: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Aggregate valid artifacts and compute paired deltas on shared seeds."""

    valid_records = [record for record in records if record.valid]
    grouped: dict[tuple[Any, ...], list[ArtifactRecord]] = {}
    for record in valid_records:
        identity = record.component_identity
        key = (
            record.dataset,
            record.method,
            identity.get("critic_method"),
            identity.get("density_enabled"),
            identity.get("retrieval_alignment_enabled"),
            identity.get("index_backend"),
        )
        grouped.setdefault(key, []).append(record)

    grouped_metrics: list[dict[str, Any]] = []
    for group_key, group_records in sorted(grouped.items(), key=lambda item: str(item[0])):
        metric_names = sorted(
            {name for record in group_records for name in record.metrics}
        )
        for metric in metric_names:
            values = [
                record.metrics[metric]
                for record in group_records
                if metric in record.metrics
            ]
            if not values:
                continue
            stats = summarize_values(metric, values)
            ci_low, ci_high = bootstrap_ci_mean(values, seed=0)
            grouped_metrics.append(
                {
                    "dataset": group_key[0],
                    "method": group_key[1],
                    "critic_method": group_key[2],
                    "density_enabled": group_key[3],
                    "retrieval_alignment_enabled": group_key[4],
                    "index_backend": group_key[5],
                    "metric": metric,
                    "n": stats.count,
                    "missing_count": len(group_records) - stats.count,
                    "mean": stats.mean,
                    "std": stats.std,
                    "stderr": stats.stderr,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    paired_comparisons: list[dict[str, Any]] = []
    for method_a, method_b in comparisons:
        datasets = sorted(
            {
                record.dataset
                for record in valid_records
                if record.method in {method_a, method_b}
            }
        )
        for dataset in datasets:
            records_a = {
                record.seed: record
                for record in valid_records
                if record.dataset == dataset and record.method == method_a
            }
            records_b = {
                record.seed: record
                for record in valid_records
                if record.dataset == dataset and record.method == method_b
            }
            common_seeds = sorted(set(records_a) & set(records_b))
            if not common_seeds:
                continue
            metric_names = sorted(
                {
                    metric
                    for seed in common_seeds
                    for metric in set(records_a[seed].metrics) & set(records_b[seed].metrics)
                }
            )
            mean_delta: dict[str, float] = {}
            delta_ci: dict[str, dict[str, float]] = {}
            for metric in metric_names:
                metric_seeds = [
                    seed
                    for seed in common_seeds
                    if metric in records_a[seed].metrics and metric in records_b[seed].metrics
                ]
                deltas = [
                    records_a[seed].metrics[metric] - records_b[seed].metrics[metric]
                    for seed in metric_seeds
                ]
                mean_delta[metric] = paired_mean_delta(
                    [records_a[seed].metrics[metric] for seed in metric_seeds],
                    [records_b[seed].metrics[metric] for seed in metric_seeds],
                )
                ci_low, ci_high = bootstrap_ci_mean(deltas, seed=0)
                delta_ci[metric] = {
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": len(deltas),
                }
            paired_comparisons.append(
                {
                    "dataset": dataset,
                    "method_a": method_a,
                    "method_b": method_b,
                    "valid_paired_seeds": common_seeds,
                    "paired_keys": [
                        {"dataset": dataset, "seed": seed} for seed in common_seeds
                    ],
                    "mean_delta": mean_delta,
                    "paired_delta_ci": delta_ci,
                }
            )

    return {
        "num_records": len(records),
        "num_valid_records": len(valid_records),
        "num_invalid_records": len(records) - len(valid_records),
        "inventory": [
            {
                **asdict(record),
                "path": str(record.path),
                "checkpoint_path": (
                    str(record.checkpoint_path) if record.checkpoint_path else None
                ),
            }
            for record in records
        ],
        "grouped_metrics": grouped_metrics,
        "paired_comparisons": paired_comparisons,
    }


def write_real_data_tables(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, CSV, and Markdown views from the same aggregate rows."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "summary.json"
    csv_path = directory / "summary.csv"
    markdown_path = directory / "summary.md"
    rows = list(summary.get("grouped_metrics", []))

    json_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "dataset",
        "method",
        "critic_method",
        "density_enabled",
        "retrieval_alignment_enabled",
        "index_backend",
        "metric",
        "n",
        "missing_count",
        "mean",
        "std",
        "stderr",
        "ci_low",
        "ci_high",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |"
        for row in rows
    ]
    markdown_path.write_text(
        "\n".join(["# RealData experiment summary", "", header, separator, *body]) + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }


def run_real_data_experiment(
    *,
    dataset: str,
    config: Any,
    seeds: Sequence[int],
    output_root: str | Path,
) -> dict[str, Any]:
    """Run seed-scoped training followed by exact and HNSW held-out audits."""

    from .config import (
        KuaiRandChronoProtocolConfig,
        MovieLensChronoProtocolConfig,
        component_identity_from_config,
        validate_kuairand_chrono_protocol_config,
        validate_movielens_chrono_protocol_config,
    )
    from .experiment import (
        run_kuairand_chrono_protocol,
        run_movielens_chrono_protocol,
    )
    from .real_data_evaluation import (
        RealDataEvaluationConfig,
        evaluate_real_data_metrics,
    )

    if dataset not in {"movielens", "kuairand"}:
        raise ValueError("dataset must be 'movielens' or 'kuairand'.")
    if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain nonnegative integers.")
    expected_type = (
        MovieLensChronoProtocolConfig
        if dataset == "movielens"
        else KuaiRandChronoProtocolConfig
    )
    if not isinstance(config, expected_type):
        raise TypeError(f"config must be a {expected_type.__name__} for {dataset}.")

    root = Path(output_root)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        identity = component_identity_from_config(config)
        method = _method_name(asdict(config), identity)
        run_dir = root / dataset / method / f"seed_{seed}"
        run_config = replace(config, seed=seed, output_dir=str(run_dir))
        if dataset == "movielens":
            run_config = validate_movielens_chrono_protocol_config(run_config)
        else:
            run_config = validate_kuairand_chrono_protocol_config(run_config)
        config_dict = asdict(run_config)
        config_digest = _config_hash(config_dict)
        identity_path = run_dir / "real_data_run_identity.json"
        if identity_path.exists():
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != config_digest:
                raise ValueError(
                    f"refusing to overwrite non-matching real_data run: {run_dir}"
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "config_hash": config_digest,
                    "component_identity": component_identity_from_config(run_config),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        native_result = (
            run_movielens_chrono_protocol(run_config)
            if dataset == "movielens"
            else run_kuairand_chrono_protocol(run_config)
        )
        evaluation: dict[str, Any] = {}
        for backend in ("exact", "faiss_hnsw"):
            evaluation[backend] = evaluate_real_data_metrics(
                native_result.task,
                native_result.model,
                RealDataEvaluationConfig(
                    split="test",
                    k=min(int(run_config.train_k), int(native_result.task.num_actions)),
                    index_backend=backend,
                    index_metric=run_config.index_metric,
                    temperature=run_config.temperature,
                    max_eval_rows=run_config.max_eval_rows,
                    seed=seed,
                ),
            )
        evaluation_payload = {
            "schema_name": "dwmql_real_data_evaluation",
            "schema_version": 1,
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "config": config_dict,
            "component_identity": component_identity_from_config(run_config),
            "native_report": native_result.output_files["report"],
            "checkpoint": native_result.output_files.get("model_checkpoint"),
            "real_data_metrics": evaluation,
        }
        evaluation_path = run_dir / "real_data_evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest_path = Path(native_result.output_files["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("summary_files", [])
        if str(evaluation_path) not in manifest["summary_files"]:
            manifest["summary_files"].append(str(evaluation_path))
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        runs.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "status": "completed",
                "config_hash": config_digest,
                "report": native_result.output_files["report"],
                "checkpoint": native_result.output_files.get("model_checkpoint"),
                "real_data_evaluation": str(evaluation_path),
                "manifest": str(manifest_path),
            }
        )
    return {"dataset": dataset, "output_root": str(root), "runs": runs}
