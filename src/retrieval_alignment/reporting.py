"""Reporting helpers for synthetic experiment sweep summaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from io import StringIO
import math
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_SWEEP_TABLE_METRICS: tuple[str, ...] = (
    "mean_bellman_loss",
    "mean_stationarity_loss",
    "mean_ess",
    "mean_exact_backend_recall",
    "mean_exact_backend_overlap_fraction",
    "final_total_loss",
    "final_bellman_loss",
    "final_stationarity_loss",
)

EXPERIMENT_SWEEP_CONFIG_COLUMNS: tuple[str, ...] = (
    "experiment_name",
    "seed",
    "k",
    "index_refresh_every",
    "index_encoder_policy",
    "index_ema_tau",
)


@dataclass(slots=True)
class TableReport:
    """A claim-free table representation for synthetic diagnostics."""

    title: str
    columns: list[str]
    rows: list[dict[str, Any]]
    synthetic_only: bool = True
    benchmark_claim: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.synthetic_only, bool):
            raise TypeError("TableReport synthetic_only must be a bool.")
        if self.benchmark_claim is not False:
            raise ValueError("TableReport benchmark_claim must be false.")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("TableReport title must be a non-empty string.")
        if not isinstance(self.columns, list) or not self.columns:
            raise ValueError("TableReport columns must be a non-empty list.")
        if not all(isinstance(column, str) and column for column in self.columns):
            raise ValueError("TableReport columns must contain non-empty strings.")
        if not isinstance(self.rows, list):
            raise TypeError("TableReport rows must be a list.")


@dataclass(slots=True)
class PlotReport:
    """A claim-free diagnostic plot report."""

    title: str
    output_path: str | None
    x_key: str
    y_key: str
    group_key: str | None
    num_points: int
    synthetic_only: bool = True
    benchmark_claim: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.synthetic_only is not True:
            raise ValueError("PlotReport synthetic_only must be true.")
        if self.benchmark_claim is not False:
            raise ValueError("PlotReport benchmark_claim must be false.")
        for name, value in {"title": self.title, "x_key": self.x_key, "y_key": self.y_key}.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"PlotReport {name} must be a non-empty string.")
        if self.group_key is not None and (not isinstance(self.group_key, str) or not self.group_key):
            raise ValueError("PlotReport group_key must be None or a non-empty string.")
        if not isinstance(self.num_points, int) or self.num_points < 0:
            raise ValueError("PlotReport num_points must be a nonnegative integer.")


def _finite_numeric(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} values must be numeric when present.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{key} values must be finite when present.")
    return converted


def experiment_sweep_summary_to_rows(
    summary: dict[str, Any],
    metric_keys: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten compact experiment sweep run summaries into table rows."""

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("summary must include runs as a list.")
    metrics = tuple(metric_keys or DEFAULT_EXPERIMENT_SWEEP_TABLE_METRICS)
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Each run summary must be a dictionary.")
        row = {column: run.get(column) for column in EXPERIMENT_SWEEP_CONFIG_COLUMNS}
        for metric in metrics:
            row[metric] = run.get(metric)
        rows.append(row)
    return rows


def grouped_statistics_to_table(statistical_summaries: dict[str, Any]) -> TableReport:
    """Convert grouped statistical summaries into a flat table report."""

    if not isinstance(statistical_summaries, dict):
        raise TypeError("statistical_summaries must be a dictionary.")
    if statistical_summaries.get("synthetic_only") is not True:
        raise ValueError("statistical_summaries synthetic_only must be true.")
    if statistical_summaries.get("benchmark_claim") is not False:
        raise ValueError("statistical_summaries benchmark_claim must be false.")
    group_keys = statistical_summaries.get("group_keys")
    groups = statistical_summaries.get("groups")
    if not isinstance(group_keys, list) or not group_keys:
        raise ValueError("statistical_summaries must include non-empty group_keys.")
    if not isinstance(groups, list):
        raise ValueError("statistical_summaries must include groups as a list.")

    rows: list[dict[str, Any]] = []
    metric_columns: set[str] = set()
    for group_row in groups:
        if not isinstance(group_row, dict):
            raise ValueError("Each grouped statistics row must be a dictionary.")
        group = group_row.get("group")
        metrics = group_row.get("metrics", {})
        if not isinstance(group, dict):
            raise ValueError("Each grouped statistics row must include a group dictionary.")
        if not isinstance(metrics, dict):
            raise ValueError("Each grouped statistics row metrics must be a dictionary.")
        row: dict[str, Any] = {key: group.get(key) for key in group_keys}
        row["count"] = group_row.get("count")
        for metric_name, metric_summary in metrics.items():
            if not isinstance(metric_summary, dict):
                continue
            for suffix in ("mean", "std", "stderr"):
                column = f"{metric_name}_{suffix}"
                row[column] = metric_summary.get(suffix)
                metric_columns.add(column)
        rows.append(row)

    columns = [*group_keys, "count", *sorted(metric_columns)]
    return TableReport(
        title="Synthetic Experiment Sweep Descriptive Statistics",
        columns=columns,
        rows=rows,
        metadata={
            "synthetic_only": True,
            "benchmark_claim": False,
            "source": "statistical_summaries",
        },
    )


def _stringify_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table_to_markdown(table: TableReport) -> str:
    """Render a ``TableReport`` as a GitHub-flavored markdown table."""

    if not isinstance(table, TableReport):
        raise TypeError("table must be a TableReport.")
    lines = [
        f"# {table.title}",
        "",
        f"synthetic_only: {str(table.synthetic_only).lower()}",
        "benchmark_claim: false",
        "",
        "| " + " | ".join(_stringify_markdown_value(column) for column in table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.rows:
        lines.append(
            "| "
            + " | ".join(_stringify_markdown_value(row.get(column)) for column in table.columns)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown_table(table: TableReport, path: str | Path) -> None:
    """Write a markdown table report to disk."""

    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("Markdown table path must be a file, not a directory.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_to_markdown(table), encoding="utf-8")


def _stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def table_to_csv_text(table: TableReport) -> str:
    """Render a ``TableReport`` as UTF-8-compatible CSV text."""

    if not isinstance(table, TableReport):
        raise TypeError("table must be a TableReport.")
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(table.columns)
    for row in table.rows:
        writer.writerow([_stringify_csv_value(row.get(column)) for column in table.columns])
    return buffer.getvalue()


def write_csv_table(table: TableReport, path: str | Path) -> None:
    """Write a table report to a CSV file."""

    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("CSV table path must be a file, not a directory.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_to_csv_text(table), encoding="utf-8", newline="")


BENCHMARK_PROTOCOL_TABLE_COLUMNS: tuple[str, ...] = (
    "route",
    "seed",
    "method",
    "index_backend",
    "index_refresh_every",
    "index_encoder_policy",
    "index_ema_tau",
    "train_pmr_at_k",
    "train_q_embedding_alignment_recall",
    "test_pmr_at_k",
    "test_q_embedding_alignment_recall",
    "final_bellman_loss",
    "benchmark_claim",
)


def benchmark_protocol_rows_to_table(rows: list[dict[str, Any]]) -> TableReport:
    """Convert benchmark protocol per-run rows into a claim-free diagnostic table."""

    if not isinstance(rows, list):
        raise TypeError("rows must be a list.")
    return TableReport(
        title="Benchmark Protocol Diagnostic Summary",
        columns=list(BENCHMARK_PROTOCOL_TABLE_COLUMNS),
        rows=rows,
        synthetic_only=False,
        benchmark_claim=False,
        metadata={"benchmark_claim": False, "scaffold_only": True, "source": "benchmark_protocol"},
    )


def benchmark_grouped_statistics_to_table(statistical_summaries: dict[str, Any]) -> TableReport:
    """Convert benchmark protocol grouped statistics into a flat diagnostic table.

    Accepts summaries with ``benchmark_claim: false`` regardless of
    ``synthetic_only`` value (unlike the synthetic-only counterpart).
    """

    if not isinstance(statistical_summaries, dict):
        raise TypeError("statistical_summaries must be a dictionary.")
    if statistical_summaries.get("benchmark_claim") is not False:
        raise ValueError("statistical_summaries benchmark_claim must be false.")
    group_keys = statistical_summaries.get("group_keys")
    groups = statistical_summaries.get("groups")
    if not isinstance(group_keys, list) or not group_keys:
        raise ValueError("statistical_summaries must include non-empty group_keys.")
    if not isinstance(groups, list):
        raise ValueError("statistical_summaries must include groups as a list.")

    rows: list[dict[str, Any]] = []
    metric_columns: set[str] = set()
    for group_row in groups:
        if not isinstance(group_row, dict):
            raise ValueError("Each grouped statistics row must be a dictionary.")
        group = group_row.get("group")
        metrics = group_row.get("metrics", {})
        if not isinstance(group, dict):
            raise ValueError("Each group row must include a group dict.")
        row: dict[str, Any] = {key: group.get(key) for key in group_keys}
        row["count"] = group_row.get("count")
        for metric_name, metric_summary in metrics.items():
            if not isinstance(metric_summary, dict):
                continue
            for suffix in ("mean", "std", "stderr"):
                column = f"{metric_name}_{suffix}"
                row[column] = metric_summary.get(suffix)
                metric_columns.add(column)
        rows.append(row)

    columns = [*group_keys, "count", *sorted(metric_columns)]
    return TableReport(
        title="Benchmark Protocol Grouped Diagnostic Statistics",
        columns=columns,
        rows=rows,
        synthetic_only=False,
        benchmark_claim=False,
        metadata={"benchmark_claim": False, "scaffold_only": True, "source": "benchmark_grouped_statistics"},
    )


def plot_experiment_sweep_metric(
    summary: dict[str, Any],
    x_key: str,
    y_key: str,
    group_key: str | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> PlotReport:
    """Plot one synthetic experiment sweep metric against a run-summary key."""

    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary.")
    if not isinstance(x_key, str) or not x_key:
        raise ValueError("x_key must be a non-empty string.")
    if not isinstance(y_key, str) or not y_key:
        raise ValueError("y_key must be a non-empty string.")
    if group_key is not None and (not isinstance(group_key, str) or not group_key):
        raise ValueError("group_key must be None or a non-empty string.")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("summary must include runs as a list.")
    resolved_output_path: Path | None = None
    if output_path is not None:
        resolved_output_path = Path(output_path)
        if resolved_output_path.exists() and resolved_output_path.is_dir():
            raise ValueError("Plot output path must be a file, not a directory.")

    grouped_points: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Each run summary must be a dictionary.")
        if run.get(x_key) is None or run.get(y_key) is None:
            continue
        x_value = _finite_numeric(run[x_key], key=x_key)
        y_value = _finite_numeric(run[y_key], key=y_key)
        label = "all" if group_key is None else str(run.get(group_key, "missing"))
        grouped_points.setdefault(label, []).append((x_value, y_value))

    num_points = sum(len(points) for points in grouped_points.values())
    if num_points == 0:
        raise ValueError("No valid points are available for the requested plot.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise ImportError("matplotlib is required for diagnostic plot export.") from exc

    plot_title = title or f"{y_key} vs {x_key}"
    figure, axis = plt.subplots()
    for label, points in sorted(grouped_points.items(), key=lambda item: item[0]):
        points = sorted(points, key=lambda point: point[0])
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if group_key is None:
            axis.plot(xs, ys, marker="o")
        else:
            axis.plot(xs, ys, marker="o", label=label)
    axis.set_title(plot_title)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    if group_key is not None:
        axis.legend(title=group_key)
    figure.tight_layout()
    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_output_path)
    plt.close(figure)
    return PlotReport(
        title=plot_title,
        output_path=None if resolved_output_path is None else str(resolved_output_path),
        x_key=x_key,
        y_key=y_key,
        group_key=group_key,
        num_points=num_points,
        metadata={
            "synthetic_only": True,
            "benchmark_claim": False,
            "groups": sorted(grouped_points),
        },
    )
