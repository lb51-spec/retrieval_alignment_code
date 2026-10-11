"""Tests for synthetic sweep table reporting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.reporting import (  # noqa: E402
    PlotReport,
    TableReport,
    benchmark_grouped_statistics_to_table,
    benchmark_protocol_rows_to_table,
    experiment_sweep_summary_to_rows,
    grouped_statistics_to_table,
    plot_experiment_sweep_metric,
    table_to_csv_text,
    table_to_markdown,
    write_csv_table,
    write_markdown_table,
)
from retrieval_alignment.statistics import summarize_experiment_sweep_statistics  # noqa: E402


def _summary() -> dict:
    return {
        "runs": [
            {
                "experiment_name": "run0",
                "seed": 0,
                "k": 3,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "index_ema_tau": None,
                "mean_bellman_loss": 2.0,
                "mean_stationarity_loss": 3.0,
                "mean_ess": 4.0,
                "final_total_loss": 5.0,
                "final_bellman_loss": 6.0,
                "final_stationarity_loss": 7.0,
            },
            {
                "experiment_name": "run1",
                "seed": 1,
                "k": 3,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "index_ema_tau": None,
                "mean_bellman_loss": 4.0,
                "mean_stationarity_loss": 5.0,
                "mean_ess": 6.0,
                "final_total_loss": 7.0,
                "final_bellman_loss": 8.0,
                "final_stationarity_loss": 9.0,
            },
        ]
    }


def test_experiment_sweep_summary_to_rows_returns_rows() -> None:
    rows = experiment_sweep_summary_to_rows(_summary())

    assert len(rows) == 2
    assert rows[0]["experiment_name"] == "run0"
    assert rows[0]["seed"] == 0
    assert rows[0]["mean_bellman_loss"] == 2.0


def test_experiment_sweep_summary_to_rows_missing_metrics_are_none() -> None:
    rows = experiment_sweep_summary_to_rows(
        {"runs": [{"experiment_name": "run0", "seed": 0}]},
        metric_keys=("mean_bellman_loss", "missing_metric"),
    )

    assert rows[0]["mean_bellman_loss"] is None
    assert rows[0]["missing_metric"] is None


def test_grouped_statistics_to_table_returns_table_report() -> None:
    stats = summarize_experiment_sweep_statistics(_summary())

    table = grouped_statistics_to_table(stats)

    assert isinstance(table, TableReport)
    assert table.synthetic_only is True
    assert table.benchmark_claim is False
    assert "index_encoder_policy" in table.columns
    assert "mean_bellman_loss_mean" in table.columns
    assert table.rows[0]["mean_bellman_loss_mean"] == pytest.approx(3.0)


def test_table_to_markdown_includes_header_rows_and_empty_none() -> None:
    table = TableReport(
        title="Synthetic Table",
        columns=["name", "value", "missing"],
        rows=[{"name": "a|b", "value": 1.0, "missing": None}],
    )

    markdown = table_to_markdown(table)

    assert "# Synthetic Table" in markdown
    assert "benchmark_claim: false" in markdown
    assert "| name | value | missing |" in markdown
    assert "| a\\|b | 1.0 |  |" in markdown


def test_table_to_csv_text_includes_header_and_empty_none() -> None:
    table = TableReport(
        title="Synthetic Table",
        columns=["name", "value", "missing"],
        rows=[{"name": "run", "value": 1.0, "missing": None}],
    )

    csv_text = table_to_csv_text(table)

    assert csv_text.splitlines()[0] == "name,value,missing"
    assert csv_text.splitlines()[1] == "run,1.0,"


def test_table_to_csv_text_escapes_commas_and_quotes() -> None:
    table = TableReport(
        title="Synthetic Table",
        columns=["name", "note"],
        rows=[{"name": "a,b", "note": 'quoted "value"'}],
    )

    csv_text = table_to_csv_text(table)

    assert '"a,b"' in csv_text
    assert '"quoted ""value"""' in csv_text


def test_write_markdown_table_writes_file_and_rejects_directory(tmp_path: Path) -> None:
    table = TableReport(title="Synthetic Table", columns=["name"], rows=[{"name": "run"}])
    output_path = tmp_path / "tables" / "table.md"

    write_markdown_table(table, output_path)

    assert output_path.exists()
    assert "# Synthetic Table" in output_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="file"):
        write_markdown_table(table, tmp_path)


def test_write_csv_table_writes_file_and_rejects_directory(tmp_path: Path) -> None:
    table = TableReport(title="Synthetic Table", columns=["name"], rows=[{"name": "run"}])
    output_path = tmp_path / "tables" / "table.csv"

    write_csv_table(table, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").splitlines() == ["name", "run"]
    with pytest.raises(ValueError, match="file"):
        write_csv_table(table, tmp_path)


def test_table_report_rejects_benchmark_claim_true() -> None:
    with pytest.raises(ValueError, match="benchmark_claim"):
        TableReport(title="Bad", columns=["x"], rows=[], benchmark_claim=True)


def test_plot_experiment_sweep_metric_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output_path = tmp_path / "plots" / "bellman.png"

    report = plot_experiment_sweep_metric(
        _summary(),
        x_key="index_refresh_every",
        y_key="mean_bellman_loss",
        group_key="index_encoder_policy",
        output_path=output_path,
    )

    assert isinstance(report, PlotReport)
    assert report.benchmark_claim is False
    assert report.synthetic_only is True
    assert report.num_points == 2
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plot_experiment_sweep_metric_skips_missing_y_values(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output_path = tmp_path / "plot.png"
    summary = {
        "runs": [
            {"index_refresh_every": 1, "policy": "online", "metric": None},
            {"index_refresh_every": 2, "policy": "online", "metric": 3.0},
        ]
    }

    report = plot_experiment_sweep_metric(
        summary,
        x_key="index_refresh_every",
        y_key="metric",
        group_key="policy",
        output_path=output_path,
    )

    assert report.num_points == 1
    assert output_path.exists()


def test_plot_experiment_sweep_metric_rejects_directory_output(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    with pytest.raises(ValueError, match="file"):
        plot_experiment_sweep_metric(
            _summary(),
            x_key="index_refresh_every",
            y_key="mean_bellman_loss",
            output_path=tmp_path,
        )


def test_plot_experiment_sweep_metric_no_valid_points_raises(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    with pytest.raises(ValueError, match="No valid points"):
        plot_experiment_sweep_metric(
            {"runs": [{"index_refresh_every": 1, "metric": None}]},
            x_key="index_refresh_every",
            y_key="metric",
            output_path=tmp_path / "plot.png",
        )


def test_benchmark_protocol_rows_to_table_basic() -> None:
    rows = [
        {
            "route": "movielens_chrono",
            "seed": 0,
            "method": "retrieval_alignment",
            "index_backend": "exact",
            "index_refresh_every": 5,
            "index_encoder_policy": "online",
            "index_ema_tau": None,
            "train_pmr_at_k": 0.5,
            "train_q_embedding_alignment_recall": 0.8,
            "test_pmr_at_k": 0.4,
            "test_q_embedding_alignment_recall": 0.7,
            "final_bellman_loss": 0.12,
            "benchmark_claim": False,
        }
    ]
    table = benchmark_protocol_rows_to_table(rows)
    assert isinstance(table, TableReport)
    assert table.benchmark_claim is False
    assert table.synthetic_only is False
    assert len(table.rows) == 1


def test_benchmark_protocol_rows_to_table_rejects_non_list() -> None:
    with pytest.raises(TypeError, match="list"):
        benchmark_protocol_rows_to_table("not a list")  # type: ignore[arg-type]


def test_benchmark_grouped_statistics_to_table_basic() -> None:
    summaries = {
        "benchmark_claim": False,
        "group_keys": ["route", "method"],
        "groups": [
            {
                "group": {"route": "movielens_chrono", "method": "retrieval_alignment"},
                "count": 3,
                "metrics": {
                    "final_bellman_loss": {"mean": 0.2, "std": 0.05, "stderr": 0.03}
                },
            }
        ],
    }
    table = benchmark_grouped_statistics_to_table(summaries)
    assert isinstance(table, TableReport)
    assert table.benchmark_claim is False
    assert table.synthetic_only is False
    assert len(table.rows) == 1
    assert "final_bellman_loss_mean" in table.columns


def test_benchmark_grouped_statistics_to_table_rejects_benchmark_claim_true() -> None:
    with pytest.raises(ValueError, match="benchmark_claim"):
        benchmark_grouped_statistics_to_table({"benchmark_claim": True, "group_keys": ["x"], "groups": []})
