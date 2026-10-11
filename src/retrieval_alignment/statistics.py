"""Descriptive statistics helpers for synthetic experiment sweeps.

These utilities summarize repeated-seed diagnostics only. They intentionally do
not compute p-values, hypothesis tests, or benchmark conclusions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any


DEFAULT_EXPERIMENT_SWEEP_STAT_METRICS: tuple[str, ...] = (
    "mean_bellman_loss",
    "mean_stationarity_loss",
    "mean_ess",
    "mean_exact_backend_recall",
    "mean_exact_backend_overlap_fraction",
    "final_total_loss",
    "final_bellman_loss",
    "final_stationarity_loss",
    "final_density_regularization",
)


@dataclass(slots=True)
class MetricSummary:
    """Descriptive summary for one scalar diagnostic metric."""

    metric_name: str
    count: int
    mean: float | None
    std: float | None
    stderr: float | None
    min: float | None
    max: float | None
    synthetic_only: bool = True
    benchmark_claim: bool = False


def _as_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must contain numeric values.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must contain only finite values.")
    return converted


def _finite_values(values: Any, *, label: str) -> list[float]:
    if not isinstance(values, list | tuple):
        raise TypeError(f"{label} must be a list or tuple.")
    converted = [_as_finite_float(value, label=label) for value in values]
    if not converted:
        raise ValueError(f"{label} must be non-empty.")
    return converted


def summarize_values(metric_name: str, values: list[float | int | None] | tuple[float | int | None, ...]) -> MetricSummary:
    """Summarize finite values for one metric, ignoring ``None`` entries."""

    if not isinstance(metric_name, str) or not metric_name:
        raise ValueError("metric_name must be a non-empty string.")
    if not isinstance(values, list | tuple):
        raise TypeError("values must be a list or tuple.")
    finite_values = [
        _as_finite_float(value, label=f"values for {metric_name}")
        for value in values
        if value is not None
    ]
    count = len(finite_values)
    if count == 0:
        return MetricSummary(
            metric_name=metric_name,
            count=0,
            mean=None,
            std=None,
            stderr=None,
            min=None,
            max=None,
        )
    mean = sum(finite_values) / count
    if count == 1:
        std = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in finite_values) / (count - 1)
        std = math.sqrt(variance)
    stderr = std / math.sqrt(count)
    return MetricSummary(
        metric_name=metric_name,
        count=count,
        mean=mean,
        std=std,
        stderr=stderr,
        min=min(finite_values),
        max=max(finite_values),
    )


def summarize_runs_by_group(
    runs: list[dict[str, Any]],
    group_keys: tuple[str, ...] | list[str],
    metric_keys: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    """Group compact run summaries and compute descriptive metric summaries."""

    if not isinstance(runs, list):
        raise TypeError("runs must be a list.")
    if not isinstance(group_keys, list | tuple) or not group_keys:
        raise ValueError("group_keys must be a non-empty list or tuple.")
    if not isinstance(metric_keys, list | tuple) or not metric_keys:
        raise ValueError("metric_keys must be a non-empty list or tuple.")
    for key in (*group_keys, *metric_keys):
        if not isinstance(key, str) or not key:
            raise ValueError("group and metric keys must be non-empty strings.")

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
        metric_summaries: dict[str, dict[str, Any]] = {}
        for metric in metric_keys:
            values = [run[metric] for run in group_runs if metric in run and run[metric] is not None]
            if not values:
                continue
            metric_summaries[metric] = asdict(summarize_values(metric, values))
        rows.append(
            {
                "group": {key: value for key, value in zip(group_keys, group, strict=True)},
                "count": len(group_runs),
                "metrics": metric_summaries,
            }
        )
    return rows


def summarize_experiment_sweep_statistics(
    summary: dict[str, Any],
    group_keys: tuple[str, ...] = ("index_encoder_policy", "index_refresh_every"),
    metric_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute descriptive grouped summaries for an experiment sweep summary."""

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("summary must include runs as a list.")
    metrics = metric_keys or DEFAULT_EXPERIMENT_SWEEP_STAT_METRICS
    groups = summarize_runs_by_group(runs, group_keys, metrics)
    return {
        "synthetic_only": True,
        "benchmark_claim": False,
        "group_keys": list(group_keys),
        "metric_keys": list(metrics),
        "groups": groups,
    }


def interquartile_mean(values: list[float | int] | tuple[float | int, ...]) -> float:
    """Return a lightweight rliable-style interquartile mean.

    This trims the lowest and highest quartiles by rank and averages the
    remaining values. It is descriptive only and does not imply significance.
    """

    finite = sorted(_finite_values(values, label="values"))
    n = len(finite)
    lo = int(math.floor(0.25 * n))
    hi = int(math.ceil(0.75 * n))
    middle = finite[lo:hi]
    if not middle:
        middle = finite
    return sum(middle) / len(middle)


def bootstrap_confidence_interval(
    values: list[float | int] | tuple[float | int, ...],
    statistic_fn: Any,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a statistic."""

    finite = _finite_values(values, label="values")
    if not callable(statistic_fn):
        raise TypeError("statistic_fn must be callable.")
    if not isinstance(num_resamples, int) or num_resamples <= 0:
        raise ValueError("num_resamples must be a positive integer.")
    if not isinstance(confidence, int | float) or not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(num_resamples):
        sample = [finite[rng.randrange(len(finite))] for _ in finite]
        estimates.append(_as_finite_float(statistic_fn(sample), label="bootstrap statistic"))
    estimates.sort()
    alpha = 1.0 - float(confidence)
    low_index = min(len(estimates) - 1, max(0, int(math.floor((alpha / 2.0) * len(estimates)))))
    high_index = min(len(estimates) - 1, max(0, int(math.ceil((1.0 - alpha / 2.0) * len(estimates))) - 1))
    return estimates[low_index], estimates[high_index]


def probability_of_improvement(
    values_a: list[float | int] | tuple[float | int, ...],
    values_b: list[float | int] | tuple[float | int, ...],
) -> float:
    """Return P(sample from A > sample from B), with ties counted as 0.5."""

    a_values = _finite_values(values_a, label="values_a")
    b_values = _finite_values(values_b, label="values_b")
    total = 0.0
    comparisons = 0
    for a_value in a_values:
        for b_value in b_values:
            if a_value > b_value:
                total += 1.0
            elif a_value == b_value:
                total += 0.5
            comparisons += 1
    return total / comparisons


def paired_mean_delta(
    values_a: list[float | int] | tuple[float | int, ...],
    values_b: list[float | int] | tuple[float | int, ...],
) -> float:
    """Return the mean of paired differences (a_i - b_i).

    Descriptive only. Not a hypothesis test. Requires equal-length sequences.
    """

    a = _finite_values(values_a, label="values_a")
    b = _finite_values(values_b, label="values_b")
    if len(a) != len(b):
        raise ValueError(
            f"values_a and values_b must have the same length for paired differences "
            f"(got {len(a)} and {len(b)})."
        )
    return sum(ai - bi for ai, bi in zip(a, b)) / len(a)


def bootstrap_ci_mean(
    values: list[float | int] | tuple[float | int, ...],
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    Convenience wrapper around ``bootstrap_confidence_interval``.
    Descriptive only; not a significance test.
    """

    return bootstrap_confidence_interval(
        values,
        lambda sample: sum(sample) / len(sample),
        num_resamples=num_resamples,
        confidence=confidence,
        seed=seed,
    )


def group_metric_summary(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...] | list[str],
    metric_keys: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    """Group rows by keys and compute descriptive metric summaries.

    Convenience alias for ``summarize_runs_by_group`` for benchmark protocol use.
    """

    return summarize_runs_by_group(rows, group_keys, metric_keys)


def performance_profile_points(
    values_by_task_or_group: dict[str, list[float | int] | tuple[float | int, ...]],
    thresholds: list[float | int] | tuple[float | int, ...],
) -> list[dict[str, Any]]:
    """Return fractions of groups meeting each threshold.

    This is a small descriptive scaffold inspired by rliable-style performance
    profiles. Higher values are treated as meeting a threshold.
    """

    if not isinstance(values_by_task_or_group, dict) or not values_by_task_or_group:
        raise ValueError("values_by_task_or_group must be a non-empty dictionary.")
    threshold_values = _finite_values(thresholds, label="thresholds")
    group_values = {
        str(group): _finite_values(values, label=f"values for {group}")
        for group, values in values_by_task_or_group.items()
    }
    rows: list[dict[str, Any]] = []
    total_groups = len(group_values)
    for threshold in threshold_values:
        groups_meeting = 0
        for values in group_values.values():
            if sum(value >= threshold for value in values) / len(values) >= 0.5:
                groups_meeting += 1
        rows.append(
            {
                "threshold": threshold,
                "fraction_groups_at_or_above": groups_meeting / total_groups,
                "num_groups": total_groups,
                "synthetic_only": True,
                "benchmark_claim": False,
            }
        )
    return rows
