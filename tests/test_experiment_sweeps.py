"""Tests for synthetic experiment-level sweeps."""

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

from retrieval_alignment.config import (  # noqa: E402
    ExperimentConfig,
    default_smoke_profile,
    experiment_config_from_smoke_profile,
)
from retrieval_alignment.experiment import (  # noqa: E402
    ExperimentSweepConfig,
    ExperimentSweepResult,
    grouped_experiment_sweep_metrics,
    load_experiment_sweep_summary,
    run_synthetic_experiment_sweep,
    validate_experiment_sweep_config,
)
from retrieval_alignment.manifest import load_run_manifest  # noqa: E402


def _base_config(tmp_path: Path) -> ExperimentConfig:
    profile = replace(
        default_smoke_profile(),
        seed=0,
        batch_size=4,
        num_transitions=16,
        num_actions=8,
        state_dim=4,
        action_dim=3,
        embedding_dim=8,
        k=3,
        num_steps=1,
        use_persistent_index=True,
        index_refresh_every=1,
        index_encoder_policy="online",
    )
    config = experiment_config_from_smoke_profile(profile, experiment_name="sweep_base")
    config.output_dir = str(tmp_path / "base")
    config.eval_num_batches = 1
    config.eval_batch_size = 4
    return config


def test_experiment_sweep_config_validation(tmp_path: Path) -> None:
    config = ExperimentSweepConfig(base_config=_base_config(tmp_path), seeds=(0, 1), k_values=(3,))

    assert validate_experiment_sweep_config(config).seeds == (0, 1)

    with pytest.raises(ValueError, match="seeds"):
        validate_experiment_sweep_config(ExperimentSweepConfig(base_config=_base_config(tmp_path), seeds=()))
    with pytest.raises(ValueError, match="k_values"):
        validate_experiment_sweep_config(ExperimentSweepConfig(base_config=_base_config(tmp_path), k_values=(0,)))
    with pytest.raises(ValueError, match="index_refresh_values"):
        validate_experiment_sweep_config(
            ExperimentSweepConfig(base_config=_base_config(tmp_path), index_refresh_values=(0,))
        )
    with pytest.raises(ValueError, match="index_encoder_policies"):
        validate_experiment_sweep_config(
            ExperimentSweepConfig(base_config=_base_config(tmp_path), index_encoder_policies=("bad",))
        )
    with pytest.raises(ValueError, match="index_ema_tau_values"):
        validate_experiment_sweep_config(
            ExperimentSweepConfig(base_config=_base_config(tmp_path), index_ema_tau_values=(1.5,))
        )


def test_run_synthetic_experiment_sweep_writes_summary_and_run_dirs(tmp_path: Path) -> None:
    output_dir = tmp_path / "experiment_sweep"
    sweep = ExperimentSweepConfig(
        base_config=_base_config(tmp_path),
        seeds=(0, 1),
        k_values=(3,),
        index_refresh_values=(1,),
        index_encoder_policies=("online",),
        output_dir=output_dir,
    )

    result = run_synthetic_experiment_sweep(sweep)
    summary = json.loads(Path(result.summary_file).read_text(encoding="utf-8"))

    assert isinstance(result, ExperimentSweepResult)
    assert result.num_runs == 2
    assert Path(result.summary_file).exists()
    assert Path(result.manifest_file).exists()
    assert summary["mode"] == "synthetic_experiment_sweep"
    assert summary["benchmark_claim"] is False
    assert summary["num_runs"] == 2
    assert summary["statistical_summaries"]["benchmark_claim"] is False
    assert summary["statistical_summaries"]["group_keys"] == [
        "index_encoder_policy",
        "index_refresh_every",
    ]
    assert len(summary["run_dirs"]) == 2
    assert len(summary["run_reports"]) == 2
    assert len(summary["runs"]) == 2
    for run in summary["runs"]:
        assert Path(run["report_file"]).exists()
        assert Path(run["final_checkpoint"]).exists()
        assert "mean_bellman_loss" in run
        assert "mean_exact_backend_recall" in run
    manifest = load_run_manifest(result.manifest_file)
    assert manifest["summary_files"] == [str(result.summary_file)]
    assert len(manifest["report_files"]) == 2
    assert len(manifest["checkpoint_files"]) == 2


def test_run_synthetic_experiment_sweep_supports_ema_tau_axis(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    base.index_encoder_policy = "ema"
    base.index_ema_tau = 0.05
    sweep = ExperimentSweepConfig(
        base_config=base,
        seeds=(0,),
        k_values=(3,),
        index_refresh_values=(1,),
        index_encoder_policies=("ema",),
        index_ema_tau_values=(0.01, 0.05),
        output_dir=tmp_path / "ema_sweep",
    )

    result = run_synthetic_experiment_sweep(sweep)

    assert result.num_runs == 2
    assert result.summary["sweep_axes"]["index_ema_tau_values"] == [0.01, 0.05]
    assert {run["index_ema_tau"] for run in result.summary["runs"]} == {0.01, 0.05}


def test_load_experiment_sweep_summary_validates_summary(tmp_path: Path) -> None:
    result = run_synthetic_experiment_sweep(
        ExperimentSweepConfig(
            base_config=_base_config(tmp_path),
            seeds=(0,),
            k_values=(3,),
            index_refresh_values=(1,),
            index_encoder_policies=("online",),
            output_dir=tmp_path / "load_sweep",
        )
    )

    loaded = load_experiment_sweep_summary(result.summary_file)

    assert loaded["mode"] == "synthetic_experiment_sweep"
    assert loaded["benchmark_claim"] is False

    wrong_mode = Path(result.summary_file).with_name("wrong_mode.json")
    payload = dict(loaded)
    payload["mode"] = "synthetic_diagnostics_sweep"
    wrong_mode.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mode"):
        load_experiment_sweep_summary(wrong_mode)

    invalid = Path(result.summary_file).with_name("invalid.json")
    invalid.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_experiment_sweep_summary(invalid)


def test_grouped_experiment_sweep_metrics_computes_means_and_skips_missing() -> None:
    summary = {
        "runs": [
            {
                "index_encoder_policy": "online",
                "index_refresh_every": 1,
                "final_total_loss": 1.0,
                "mean_bellman_loss": 2.0,
            },
            {
                "index_encoder_policy": "online",
                "index_refresh_every": 1,
                "final_total_loss": 3.0,
                "mean_bellman_loss": 4.0,
            },
            {
                "index_encoder_policy": "ema",
                "index_refresh_every": 5,
                "final_total_loss": 10.0,
            },
        ]
    }

    rows = grouped_experiment_sweep_metrics(
        summary,
        group_keys=("index_encoder_policy", "index_refresh_every"),
        metric_keys=("final_total_loss", "mean_bellman_loss", "missing_metric"),
    )

    online = next(row for row in rows if row["index_encoder_policy"] == "online")
    ema = next(row for row in rows if row["index_encoder_policy"] == "ema")
    assert online["count"] == 2
    assert online["final_total_loss_mean"] == pytest.approx(2.0)
    assert online["final_total_loss_std"] == pytest.approx(1.0)
    assert online["mean_bellman_loss_mean"] == pytest.approx(3.0)
    assert "missing_metric_mean" not in online
    assert ema["final_total_loss_std"] == 0.0


def test_grouped_experiment_sweep_metrics_missing_group_key_raises() -> None:
    with pytest.raises(ValueError, match="group key"):
        grouped_experiment_sweep_metrics({"runs": [{"index_encoder_policy": "online"}]})
