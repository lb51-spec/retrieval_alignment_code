"""Tests for diagnostics-only synthetic sweep helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import default_smoke_profile
from retrieval_alignment.manifest import load_run_manifest
from retrieval_alignment.sweeps import (
    SyntheticSweepResult,
    binned_support_loss_contrasts,
    compute_support_loss_objective_associations,
    compute_sweep_metric_associations,
    grouped_sweep_diagnostics,
    load_support_loss_ablation_summary,
    paired_correction_mode_contrasts,
    load_sweep_run_reports,
    load_sweep_summary,
    run_support_loss_ablation,
    run_synthetic_diagnostics_sweep,
    support_loss_ablation_profile_from_yaml,
    summarize_sweep_diagnostics_table,
    summarize_sweep_run,
)
from retrieval_alignment.train import DIAGNOSTICS_SCHEMA_NAME, DIAGNOSTICS_SCHEMA_VERSION


def _small_profile():
    return replace(
        default_smoke_profile(),
        batch_size=4,
        num_transitions=16,
        num_actions=8,
        embedding_dim=8,
        k=3,
        num_steps=1,
    )


def test_run_synthetic_diagnostics_sweep_writes_expected_json_files(tmp_path: Path) -> None:
    result = run_synthetic_diagnostics_sweep(
        base_profile=_small_profile(),
        output_dir=tmp_path,
        index_refresh_values=(1, 2),
        index_encoder_policies=("online",),
        seeds=(0, 1),
        num_steps=1,
    )

    files = sorted(path for path in tmp_path.glob("*.json") if path.name not in {"summary.json", "manifest.json"})
    assert isinstance(result, SyntheticSweepResult)
    assert result.num_runs == 4
    assert len(files) == 4
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["schema_name"] == DIAGNOSTICS_SCHEMA_NAME
        assert isinstance(report["schema_version"], int)
        assert report["benchmark_claim"] is False
        assert report["synthetic_only"] is True
        assert report["mode"] == "synthetic_diagnostics_sweep_run"


def test_run_synthetic_diagnostics_sweep_writes_summary_json(tmp_path: Path) -> None:
    result = run_synthetic_diagnostics_sweep(
        base_profile=replace(_small_profile(), index_ema_tau=0.05),
        output_dir=tmp_path,
        index_refresh_values=(1, 2),
        index_encoder_policies=("online",),
        seeds=(0,),
        num_steps=1,
    )

    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result.summary_file == summary_path
    assert result.manifest_file == tmp_path / "manifest.json"
    assert result.summary == summary
    assert summary["schema_name"] == DIAGNOSTICS_SCHEMA_NAME
    assert isinstance(summary["schema_version"], int)
    assert summary["mode"] == "synthetic_diagnostics_sweep"
    assert summary["synthetic_only"] is True
    assert summary["benchmark_claim"] is False
    assert summary["num_runs"] == 2
    assert len(summary["run_files"]) == 2
    assert len(summary["runs"]) == 2
    assert summary["sweep_axes"]["seeds"] == [0]
    assert summary["sweep_axes"]["index_refresh_values"] == [1, 2]
    assert summary["sweep_axes"]["index_encoder_policies"] == ["online"]
    for run in summary["runs"]:
        assert set(
            [
                "seed",
                "index_refresh_every",
                "index_encoder_policy",
                "json_file",
            ]
        ).issubset(run)
        assert run["json_file"] in summary["run_files"]
    manifest = load_run_manifest(result.manifest_file)
    assert manifest["summary_files"] == [str(summary_path)]
    assert len(manifest["report_files"]) == 2
    assert manifest["metadata"]["diagnostics_only"] is True


def test_run_synthetic_diagnostics_sweep_with_k_values_writes_expected_files(tmp_path: Path) -> None:
    result = run_synthetic_diagnostics_sweep(
        base_profile=replace(_small_profile(), index_ema_tau=0.05),
        output_dir=tmp_path,
        index_refresh_values=(10,),
        index_encoder_policies=("online", "ema"),
        seeds=(0, 1),
        num_steps=1,
        k_values=(2, 3),
    )

    files = sorted(path.name for path in tmp_path.glob("*.json") if path.name not in {"summary.json", "manifest.json"})

    assert result.num_runs == 8
    assert len(files) == 8
    assert "seed0_refresh10_policyema_k2.json" in files
    assert "seed1_refresh10_policyonline_k3.json" in files
    assert result.summary["sweep_axes"]["k_values"] == [2, 3]
    assert all("k" in run for run in result.summary["runs"])
    assert {run["k"] for run in result.summary["runs"]} == {2, 3}


def test_run_synthetic_diagnostics_sweep_with_ema_tau_values_writes_expected_files(tmp_path: Path) -> None:
    result = run_synthetic_diagnostics_sweep(
        base_profile=_small_profile(),
        output_dir=tmp_path,
        index_refresh_values=(50,),
        index_encoder_policies=("ema",),
        seeds=(0, 1),
        num_steps=1,
        k_values=(2,),
        index_ema_tau_values=(0.01, 0.05),
    )

    files = sorted(path.name for path in tmp_path.glob("*.json") if path.name != "summary.json")

    assert result.num_runs == 4
    assert "seed0_refresh50_policyema_k2_tau0p01.json" in files
    assert "seed1_refresh50_policyema_k2_tau0p05.json" in files
    assert result.summary["sweep_axes"]["index_ema_tau_values"] == [0.01, 0.05]
    assert {run["index_ema_tau"] for run in result.summary["runs"]} == {0.01, 0.05}
    assert all("index_ema_tau" in run for run in result.summary["runs"])


def test_load_sweep_summary_works_on_produced_summary(tmp_path: Path) -> None:
    result = run_synthetic_diagnostics_sweep(
        base_profile=_small_profile(),
        output_dir=tmp_path,
        index_refresh_values=(1,),
        index_encoder_policies=("online",),
        seeds=(0,),
        num_steps=1,
    )

    summary = load_sweep_summary(result.summary_file)

    assert summary["schema_name"] == DIAGNOSTICS_SCHEMA_NAME
    assert summary["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert summary["mode"] == "synthetic_diagnostics_sweep"
    assert len(summary["runs"]) == 1


def test_load_sweep_run_reports_loads_all_runs(tmp_path: Path) -> None:
    result = run_synthetic_diagnostics_sweep(
        base_profile=_small_profile(),
        output_dir=tmp_path,
        index_refresh_values=(1, 2),
        index_encoder_policies=("online",),
        seeds=(0,),
        num_steps=1,
    )
    summary = load_sweep_summary(result.summary_file)

    reports = load_sweep_run_reports(summary)

    assert len(reports) == 2
    assert all(report["benchmark_claim"] is False for report in reports)
    assert all(report["synthetic_only"] is True for report in reports)


def test_load_sweep_summary_wrong_schema_name_raises(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = {
        "schema_name": "bad",
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "runs": [],
        "run_files": [],
    }
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_name"):
        load_sweep_summary(path)


def test_load_sweep_summary_wrong_schema_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION + 1,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "runs": [],
        "run_files": [],
    }
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_sweep_summary(path)


def test_load_sweep_summary_wrong_mode_raises(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep_run",
        "synthetic_only": True,
        "benchmark_claim": False,
        "runs": [],
        "run_files": [],
    }
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="mode"):
        load_sweep_summary(path)


def test_load_sweep_summary_missing_runs_raises(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "run_files": [],
    }
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="runs"):
        load_sweep_summary(path)


def test_load_sweep_summary_invalid_json_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_sweep_summary(path)


def test_load_sweep_summary_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sweep_summary(tmp_path / "missing.json")


def test_load_sweep_run_reports_requires_base_dir_when_missing_output_dir(tmp_path: Path) -> None:
    result = run_synthetic_diagnostics_sweep(
        base_profile=_small_profile(),
        output_dir=tmp_path,
        index_refresh_values=(1,),
        index_encoder_policies=("online",),
        seeds=(0,),
        num_steps=1,
    )
    summary = load_sweep_summary(result.summary_file)
    summary.pop("output_dir")

    with pytest.raises(ValueError, match="base_dir"):
        load_sweep_run_reports(summary)

    reports = load_sweep_run_reports(summary, base_dir=tmp_path)
    assert len(reports) == 1


def test_summarize_sweep_run_handles_missing_optional_diagnostics() -> None:
    report = {
        "sweep": {"seed": 0, "index_refresh_every": 1, "index_encoder_policy": "online"},
        "summary": {"num_steps": 1, "final_total_loss": 1.25},
        "diagnostics": {},
        "history": {},
    }

    summary = summarize_sweep_run(report, "run.json")

    assert summary["seed"] == 0
    assert summary["index_refresh_every"] == 1
    assert summary["index_encoder_policy"] == "online"
    assert summary["json_file"] == "run.json"
    assert summary["final_total_loss"] == 1.25
    assert summary["final_bellman_loss"] is None
    assert summary["mean_bellman_loss"] is None
    assert summary["final_stale_index_recall_mean"] is None
    assert summary["final_stale_index_pmr_mean"] is None


def test_summarize_sweep_run_includes_pmr_and_objective_summary_metrics() -> None:
    report = {
        "sweep": {"seed": 0, "index_refresh_every": 10, "index_encoder_policy": "ema", "k": 4},
        "summary": {"num_steps": 2, "final_total_loss": 1.25},
        "diagnostics": {"final_density_ess": 4.0, "final_weight_entropy": 1.2},
        "history": {
            "bellman_losses": [1.0, 2.0],
            "stationarity_losses": [3.0, 5.0],
            "density_regularizations": [0.1, 0.3],
            "stale_index_recall_means": [0.5, 0.75],
            "stale_index_overlap_fraction_means": [0.4, 0.8],
            "stale_index_best_value_gap_means": [0.2, 0.1],
            "stale_index_pmr_means": [0.6, 0.9],
            "density_ess_values": [3.0, 4.0],
            "weight_entropy_values": [1.0, 1.2],
        },
    }

    summary = summarize_sweep_run(report, "run.json")

    assert summary["final_bellman_loss"] == 2.0
    assert summary["mean_bellman_loss"] == pytest.approx(1.5)
    assert summary["final_stationarity_loss"] == 5.0
    assert summary["mean_stationarity_loss"] == pytest.approx(4.0)
    assert summary["final_density_regularization"] == 0.3
    assert summary["mean_density_regularization"] == pytest.approx(0.2)
    assert summary["final_stale_index_recall_mean"] == 0.75
    assert summary["final_stale_index_overlap_fraction_mean"] == 0.8
    assert summary["final_stale_index_best_value_gap_mean"] == 0.1
    assert summary["final_stale_index_pmr_mean"] == 0.9
    assert summary["final_density_ess"] == 4.0
    assert summary["final_weight_entropy"] == 1.2


def test_summarize_sweep_diagnostics_table_sorts_by_key() -> None:
    summary = {
        "runs": [
            {"seed": 0, "index_refresh_every": 1, "index_encoder_policy": "online", "final_stale_index_recall_mean": 0.7},
            {"seed": 1, "index_refresh_every": 5, "index_encoder_policy": "ema", "final_stale_index_recall_mean": 0.9},
            {"seed": 2, "index_refresh_every": 10, "index_encoder_policy": "online"},
        ]
    }

    rows = summarize_sweep_diagnostics_table(summary)

    assert [row["seed"] for row in rows] == [1, 0, 2]
    assert rows[-1]["final_stale_index_recall_mean"] is None


def test_summarize_sweep_diagnostics_table_handles_missing_optional_diagnostics() -> None:
    summary = {"runs": [{"seed": 0, "index_refresh_every": 1, "index_encoder_policy": "online"}]}

    rows = summarize_sweep_diagnostics_table(summary)

    assert rows == [
        {
            "seed": 0,
            "index_refresh_every": 1,
            "index_encoder_policy": "online",
            "k": None,
            "index_ema_tau": None,
            "final_stale_index_recall_mean": None,
            "final_stale_index_pmr_mean": None,
            "final_stale_index_overlap_fraction_mean": None,
            "final_index_encoder_drift_mean_l2": None,
            "final_index_age": None,
            "json_file": None,
        }
    ]


def test_grouped_sweep_diagnostics_groups_by_policy_and_refresh() -> None:
    summary = {
        "runs": [
            {
                "seed": 0,
                "index_encoder_policy": "online",
                "index_refresh_every": 1,
                "final_stale_index_recall_mean": 1.0,
                "final_stale_index_overlap_fraction_mean": 1.0,
                "final_index_encoder_drift_mean_l2": 0.0,
                "final_index_age": 0,
            },
            {
                "seed": 1,
                "index_encoder_policy": "online",
                "index_refresh_every": 1,
                "final_stale_index_recall_mean": 0.8,
                "final_stale_index_overlap_fraction_mean": 0.9,
                "final_index_encoder_drift_mean_l2": 0.2,
                "final_index_age": 0,
            },
            {
                "seed": 0,
                "index_encoder_policy": "ema",
                "index_refresh_every": 5,
                "final_stale_index_recall_mean": 0.5,
                "final_stale_index_overlap_fraction_mean": 0.6,
                "final_index_encoder_drift_mean_l2": 0.4,
                "final_index_age": 2,
            },
        ]
    }

    rows = grouped_sweep_diagnostics(summary)

    assert len(rows) == 2
    online = next(row for row in rows if row["index_encoder_policy"] == "online")
    ema = next(row for row in rows if row["index_encoder_policy"] == "ema")
    assert online["count"] == 2
    assert online["final_stale_index_recall_mean_mean"] == pytest.approx(0.9)
    assert online["final_stale_index_recall_mean_std"] == pytest.approx(0.1)
    assert online["final_stale_index_overlap_fraction_mean_mean"] == pytest.approx(0.95)
    assert online["final_index_encoder_drift_mean_l2_mean"] == pytest.approx(0.1)
    assert ema["count"] == 1
    assert ema["final_stale_index_recall_mean_mean"] == pytest.approx(0.5)
    assert ema["final_stale_index_recall_mean_std"] == 0.0


def test_grouped_sweep_diagnostics_groups_by_k() -> None:
    summary = {
        "runs": [
            {
                "seed": 0,
                "index_encoder_policy": "online",
                "index_refresh_every": 10,
                "k": 5,
                "final_stale_index_recall_mean": 0.8,
            },
            {
                "seed": 1,
                "index_encoder_policy": "online",
                "index_refresh_every": 10,
                "k": 5,
                "final_stale_index_recall_mean": 1.0,
            },
            {
                "seed": 0,
                "index_encoder_policy": "online",
                "index_refresh_every": 10,
                "k": 10,
                "final_stale_index_recall_mean": 0.7,
            },
        ]
    }

    rows = grouped_sweep_diagnostics(summary, group_keys=("index_encoder_policy", "index_refresh_every", "k"))

    assert len(rows) == 2
    k5 = next(row for row in rows if row["k"] == 5)
    k10 = next(row for row in rows if row["k"] == 10)
    assert k5["count"] == 2
    assert k5["final_stale_index_recall_mean_mean"] == pytest.approx(0.9)
    assert k10["count"] == 1
    assert k10["final_stale_index_recall_mean_std"] == 0.0


def test_grouped_sweep_diagnostics_groups_by_index_ema_tau() -> None:
    summary = {
        "runs": [
            {
                "seed": 0,
                "index_refresh_every": 50,
                "k": 10,
                "index_ema_tau": 0.01,
                "final_stale_index_recall_mean": 0.7,
            },
            {
                "seed": 1,
                "index_refresh_every": 50,
                "k": 10,
                "index_ema_tau": 0.01,
                "final_stale_index_recall_mean": 0.9,
            },
            {
                "seed": 0,
                "index_refresh_every": 50,
                "k": 10,
                "index_ema_tau": 0.05,
                "final_stale_index_recall_mean": 0.8,
            },
        ]
    }

    rows = grouped_sweep_diagnostics(summary, group_keys=("index_refresh_every", "k", "index_ema_tau"))

    assert len(rows) == 2
    tau001 = next(row for row in rows if row["index_ema_tau"] == 0.01)
    tau005 = next(row for row in rows if row["index_ema_tau"] == 0.05)
    assert tau001["count"] == 2
    assert tau001["final_stale_index_recall_mean_mean"] == pytest.approx(0.8)
    assert tau005["count"] == 1
    assert tau005["final_stale_index_recall_mean_std"] == 0.0


def test_grouped_sweep_diagnostics_missing_group_key_raises() -> None:
    summary = {"runs": [{"seed": 0, "index_encoder_policy": "online"}]}

    with pytest.raises(ValueError, match="group key"):
        grouped_sweep_diagnostics(summary, group_keys=("index_encoder_policy", "k"))


def test_grouped_sweep_diagnostics_skips_missing_metrics() -> None:
    summary = {
        "runs": [
            {"seed": 0, "index_encoder_policy": "online", "index_refresh_every": 1},
            {"seed": 1, "index_encoder_policy": "online", "index_refresh_every": 1},
        ]
    }

    rows = grouped_sweep_diagnostics(summary)

    assert rows == [{"index_encoder_policy": "online", "index_refresh_every": 1, "count": 2}]


def test_compute_sweep_metric_associations_returns_pearson_values() -> None:
    summary = {
        "runs": [
            {"final_stale_index_pmr_mean": 0.2, "final_bellman_loss": 4.0},
            {"final_stale_index_pmr_mean": 0.5, "final_bellman_loss": 2.0},
            {"final_stale_index_pmr_mean": 0.8, "final_bellman_loss": 1.0},
        ]
    }

    result = compute_sweep_metric_associations(
        summary,
        x_metrics=("final_stale_index_pmr_mean",),
        y_metrics=("final_bellman_loss",),
    )
    association = result["associations"]["final_stale_index_pmr_mean__vs__final_bellman_loss"]

    assert result["benchmark_claim"] is False
    assert result["causality_implied"] is False
    assert association["count"] == 3
    assert association["pearson_r"] < 0.0


def test_compute_sweep_metric_associations_handles_missing_or_constant_metrics() -> None:
    summary = {
        "runs": [
            {"final_stale_index_pmr_mean": 0.5, "final_bellman_loss": 1.0},
            {"final_stale_index_pmr_mean": 0.5, "final_bellman_loss": 2.0},
        ]
    }

    result = compute_sweep_metric_associations(
        summary,
        x_metrics=("final_stale_index_pmr_mean", "missing_metric"),
        y_metrics=("final_bellman_loss",),
    )

    assert result["associations"]["final_stale_index_pmr_mean__vs__final_bellman_loss"]["pearson_r"] is None
    assert result["associations"]["missing_metric__vs__final_bellman_loss"]["count"] == 0


def test_compute_support_loss_objective_associations_returns_descriptive_rows() -> None:
    summary = {
        "runs": [
            {"final_stale_index_pmr_normalized_mean": 1.0, "final_bellman_loss": 3.0},
            {"final_stale_index_pmr_normalized_mean": 2.0, "final_bellman_loss": 2.0},
            {"final_stale_index_pmr_normalized_mean": 3.0, "final_bellman_loss": 1.0},
        ]
    }

    result = compute_support_loss_objective_associations(
        summary,
        x_metrics=("final_stale_index_pmr_normalized_mean", "missing_metric"),
        y_metrics=("final_bellman_loss",),
    )

    key = "final_stale_index_pmr_normalized_mean__vs__final_bellman_loss"
    assert result["benchmark_claim"] is False
    assert result["causality_implied"] is False
    assert result["associations"][key]["pearson_r"] == pytest.approx(-1.0)
    assert result["associations"]["missing_metric__vs__final_bellman_loss"]["count"] == 0
    assert result["associations"]["missing_metric__vs__final_bellman_loss"]["pearson_r"] is None


def test_paired_correction_mode_contrasts_match_rows_and_deltas() -> None:
    summary = {
        "runs": [
            {
                "seed": 0,
                "k": 2,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "index_ema_tau": None,
                "retrieval_correction_mode": "aware",
                "mean_bellman_loss": 1.0,
                "final_stale_index_pmr_mean": 0.8,
            },
            {
                "seed": 0,
                "k": 2,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "index_ema_tau": None,
                "retrieval_correction_mode": "naive",
                "mean_bellman_loss": 1.5,
                "final_stale_index_pmr_mean": 0.7,
            },
            {
                "seed": 1,
                "k": 2,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "index_ema_tau": None,
                "retrieval_correction_mode": "aware",
                "mean_bellman_loss": 2.0,
                "final_stale_index_pmr_mean": 0.6,
            },
            {
                "seed": 99,
                "k": 2,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "index_ema_tau": None,
                "retrieval_correction_mode": "naive",
                "mean_bellman_loss": 9.0,
                "final_stale_index_pmr_mean": 0.1,
            },
        ]
    }

    result = paired_correction_mode_contrasts(
        summary,
        comparison_modes=("naive",),
        metric_keys=("mean_bellman_loss", "final_stale_index_pmr_mean"),
    )

    assert result["benchmark_claim"] is False
    comparison = result["comparisons"]["naive"]
    assert comparison["paired_count"] == 1
    bellman = comparison["metrics"]["mean_bellman_loss"]
    assert bellman["mean_delta"] == pytest.approx(-0.5)
    assert bellman["fraction_delta_negative"] == 1.0
    pmr = comparison["metrics"]["final_stale_index_pmr_mean"]
    assert pmr["mean_delta"] == pytest.approx(0.1)
    assert pmr["fraction_delta_positive"] == 1.0


def test_binned_support_loss_contrasts_produce_three_bins() -> None:
    runs = []
    for seed, support in enumerate((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
        base = {
            "seed": seed,
            "k": 2,
            "index_refresh_every": 1,
            "index_encoder_policy": "online",
            "index_ema_tau": None,
            "final_stale_index_pmr_normalized_mean": support,
            "mean_bellman_loss": float(seed),
        }
        runs.append({**base, "retrieval_correction_mode": "aware"})
        runs.append({**base, "retrieval_correction_mode": "naive", "mean_bellman_loss": float(seed) + 1.0})
    summary = {"runs": runs}

    result = binned_support_loss_contrasts(summary)

    assert result["benchmark_claim"] is False
    assert [row["bin"] for row in result["bins"]] == ["low", "medium", "high"]
    assert [row["count"] for row in result["bins"]] == [2, 2, 2]
    low_naive = result["bins"][0]["contrasts"]["comparisons"]["naive"]
    assert low_naive["paired_count"] == 2
    assert low_naive["metrics"]["mean_bellman_loss"]["mean_delta"] == pytest.approx(-1.0)
    assert "mean_raw_bellman_loss" in low_naive["metrics"]
    assert "mean_support_weighted_bellman_loss" in low_naive["metrics"]
    assert "mean_critic_total_loss" in low_naive["metrics"]


def test_run_support_loss_ablation_writes_reports_and_summary(tmp_path: Path) -> None:
    profile = replace(
        _small_profile(),
        index_ema_tau=0.05,
        num_steps=1,
    )

    result = run_support_loss_ablation(
        profile,
        tmp_path,
        retrieval_correction_modes=("aware", "uniform_weights"),
        seeds=(0,),
        k_values=(2,),
        index_refresh_values=(1,),
        index_encoder_policies=("online",),
        num_steps=1,
    )

    assert result.num_runs == 2
    assert result.summary["benchmark_claim"] is False
    assert result.summary["mode"] == "support_loss_ablation"
    assert {run["retrieval_correction_mode"] for run in result.summary["runs"]} == {"aware", "uniform_weights"}
    for run in result.summary["runs"]:
        assert "final_stale_index_pmr_mean" in run
        assert "final_raw_bellman_loss" in run
        assert "final_support_weighted_bellman_loss" in run
        assert "final_critic_total_loss" in run
        assert "final_stale_index_pmr_normalized_mean" in run
        assert "final_stale_index_missing_mass_mean" in run
        assert "uniform_pmr_baseline" in run
        assert "support_correction_loss" in run
        assert "final_support_correction_contribution" in run
        assert "support_correction_active" in run
        assert "support_weighting_mode" in run
        assert "support_weight_scale" in run
    assert result.summary_file.exists()
    assert result.manifest_file.exists()
    loaded = load_support_loss_ablation_summary(result.summary_file)
    assert loaded["num_runs"] == 2


def test_support_loss_ablation_support_weight_sweep_applies_only_to_aware(tmp_path: Path) -> None:
    profile = replace(_small_profile(), index_ema_tau=0.05, num_steps=1)

    result = run_support_loss_ablation(
        profile,
        tmp_path,
        retrieval_correction_modes=("aware", "naive"),
        seeds=(0,),
        k_values=(2,),
        index_refresh_values=(1,),
        index_encoder_policies=("online",),
        support_correction_weight_values=(0.0, 0.1),
        num_steps=1,
    )

    assert result.num_runs == 3
    assert result.summary["sweep_axes"]["support_correction_weight_values"] == [0.0, 0.1]
    weights_by_mode = {
        run["retrieval_correction_mode"]: sorted(
            candidate["support_correction_weight"]
            for candidate in result.summary["runs"]
            if candidate["retrieval_correction_mode"] == run["retrieval_correction_mode"]
        )
        for run in result.summary["runs"]
    }
    assert weights_by_mode["aware"] == [0.0, 0.1]
    assert weights_by_mode["naive"] == [0.0]


def test_support_loss_ablation_support_weighting_sweep_applies_only_to_aware(tmp_path: Path) -> None:
    profile = replace(_small_profile(), index_ema_tau=0.05, num_steps=1)

    result = run_support_loss_ablation(
        profile,
        tmp_path,
        retrieval_correction_modes=("aware", "naive"),
        seeds=(0,),
        k_values=(2,),
        index_refresh_values=(1,),
        index_encoder_policies=("online",),
        support_weighting_modes=("none", "missing_mass"),
        support_weight_scale_values=(0.5, 1.0),
        num_steps=1,
    )

    assert result.num_runs == 5
    assert result.summary["sweep_axes"]["support_weighting_modes"] == ["none", "missing_mass"]
    assert result.summary["sweep_axes"]["support_weight_scale_values"] == [0.5, 1.0]
    aware_runs = [run for run in result.summary["runs"] if run["retrieval_correction_mode"] == "aware"]
    naive_run = next(run for run in result.summary["runs"] if run["retrieval_correction_mode"] == "naive")
    assert {run["support_weighting_mode"] for run in aware_runs} == {"none", "missing_mass"}
    assert {run["support_weight_scale"] for run in aware_runs} == {0.5, 1.0}
    assert naive_run["support_weighting_mode"] == "none"


def test_support_loss_ablation_rejects_invalid_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_correction_mode"):
        run_support_loss_ablation(
            _small_profile(),
            tmp_path,
            retrieval_correction_modes=("bad",),
            seeds=(0,),
            k_values=(2,),
            index_refresh_values=(1,),
            index_encoder_policies=("online",),
        )


def test_support_loss_ablation_yaml_loader(tmp_path: Path) -> None:
    config_path = tmp_path / "support.yaml"
    config_path.write_text(
        "\n".join(
            [
                "synthetic_only: true",
                "benchmark_claim: false",
                "num_steps: 2",
                "num_actions: 8",
                "k_values: [2]",
                "refresh_values: [1]",
                "encoder_policies: [online]",
                "retrieval_correction_modes: [aware, naive]",
                "support_correction_weight: 0.25",
                "support_correction_weight_values: [0.0, 0.25]",
                "support_correction_metric: missing_mass",
                "seeds: [0]",
            ]
        ),
        encoding="utf-8",
    )

    profile, options = support_loss_ablation_profile_from_yaml(config_path)

    assert profile.num_steps == 2
    assert profile.support_correction_weight == pytest.approx(0.25)
    assert profile.support_correction_metric == "missing_mass"
    assert options["retrieval_correction_modes"] == ("aware", "naive")
    assert options["support_correction_weight_values"] == (0.0, 0.25)


def test_support_loss_ablation_full_config_loads() -> None:
    config_path = ROOT / "configs" / "support_loss_ablation_full.yaml"

    profile, options = support_loss_ablation_profile_from_yaml(config_path)

    assert profile.num_steps == 150
    assert profile.num_actions >= max(options["k_values"])
    assert options["seeds"] == tuple(range(10))
    assert options["k_values"] == (10, 40, 100)
    assert options["retrieval_correction_modes"] == ("aware", "naive", "uniform_weights")
    assert profile.support_correction_weight > 0.0
    assert options["support_correction_weight_values"] == (0.0, 0.01, 0.1, 1.0)
    assert options["support_weighting_modes"] == ("none", "missing_mass", "inverse_pmr")
    assert options["support_weight_scale_values"] == (0.1, 1.0)
    assert options["benchmark_claim"] is False




def test_run_synthetic_diagnostics_sweep_rejects_existing_file_output_dir(tmp_path: Path) -> None:
    output_file = tmp_path / "not_a_dir"
    output_file.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir"):
        run_synthetic_diagnostics_sweep(_small_profile(), output_file)


def test_run_synthetic_diagnostics_sweep_rejects_invalid_refresh_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="index_refresh_values"):
        run_synthetic_diagnostics_sweep(_small_profile(), tmp_path, index_refresh_values=(0,))


def test_run_synthetic_diagnostics_sweep_rejects_invalid_k_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="k_values"):
        run_synthetic_diagnostics_sweep(_small_profile(), tmp_path, k_values=(0,))

    with pytest.raises(ValueError, match="num_actions"):
        run_synthetic_diagnostics_sweep(_small_profile(), tmp_path, k_values=(99,))


def test_run_synthetic_diagnostics_sweep_rejects_invalid_ema_tau_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="index_ema_tau_values"):
        run_synthetic_diagnostics_sweep(_small_profile(), tmp_path, index_ema_tau_values=())

    with pytest.raises(ValueError, match="index_ema_tau_values"):
        run_synthetic_diagnostics_sweep(_small_profile(), tmp_path, index_ema_tau_values=(-0.1,))

    with pytest.raises(ValueError, match="index_ema_tau_values"):
        run_synthetic_diagnostics_sweep(_small_profile(), tmp_path, index_ema_tau_values=(1.1,))


def test_run_synthetic_diagnostics_sweep_rejects_invalid_encoder_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="encoder policy"):
        run_synthetic_diagnostics_sweep(
            _small_profile(),
            tmp_path,
            index_encoder_policies=("online", "bad"),
        )


def test_run_synthetic_diagnostics_sweep_target_policy_requires_target_tau(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target_tau"):
        run_synthetic_diagnostics_sweep(
            _small_profile(),
            tmp_path,
            index_encoder_policies=("target",),
        )


def test_run_synthetic_diagnostics_sweep_ema_policy_requires_index_ema_tau(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="index_ema_tau"):
        run_synthetic_diagnostics_sweep(
            _small_profile(),
            tmp_path,
            index_encoder_policies=("ema",),
        )
