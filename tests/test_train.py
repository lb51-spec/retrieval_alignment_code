"""Tests for the smoke-runner CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import ANNConfig, AlgorithmConfig, ExperimentConfig, ModelConfig, SyntheticDatasetConfig
from retrieval_alignment.pipeline import PipelineSmokeResult
from retrieval_alignment.train import (
    DIAGNOSTICS_SCHEMA_NAME,
    DIAGNOSTICS_SCHEMA_VERSION,
    main,
    parse_comma_floats,
    parse_comma_ints,
    parse_comma_strings,
    run_smoke_from_config,
    summarize_group_norms_by_step,
)


def _config() -> ExperimentConfig:
    config = ExperimentConfig(
        model=ModelConfig(state_dim=4, action_dim=3, embedding_dim=8, hidden_dims=(16,)),
        synthetic=SyntheticDatasetConfig(
            num_transitions=24,
            state_dim=4,
            action_dim=3,
            num_actions=9,
            seed=123,
        ),
        algorithm=AlgorithmConfig(gamma=0.95, candidate_k=4),
        seed=123,
    )
    return config


def _assert_json_report_metadata(report: dict, mode: str) -> None:
    assert report["schema_name"] == DIAGNOSTICS_SCHEMA_NAME
    assert report["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert isinstance(report["schema_version"], int)
    assert report["mode"] == mode
    assert report["synthetic_only"] is True
    assert report["benchmark_claim"] is False


def test_parse_comma_ints_handles_sweep_values() -> None:
    assert parse_comma_ints("1,5") == (1, 5)
    assert parse_comma_ints("0,1") == (0, 1)


def test_parse_comma_strings_handles_sweep_values() -> None:
    assert parse_comma_strings("online,target") == ("online", "target")


def test_parse_comma_floats_handles_sweep_values() -> None:
    assert parse_comma_floats("0.01, 0.05,0.1") == (0.01, 0.05, 0.1)


def test_run_smoke_from_config_returns_result() -> None:
    result = run_smoke_from_config(_config())

    assert isinstance(result, PipelineSmokeResult)


def test_smoke_result_has_finite_scalar_diagnostics() -> None:
    result = run_smoke_from_config(_config())

    assert torch.isfinite(result.bellman_loss)
    assert torch.isfinite(result.ess)
    assert torch.isfinite(result.stationarity_loss)
    assert result.bellman_loss.ndim == 0
    assert result.ess.ndim == 0
    assert result.stationarity_loss.ndim == 0


def test_cli_smoke_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DW-MQL smoke check passed" in output
    assert "bellman_loss" in output
    assert "selected_index_backend" in output
    assert "selected_index_metric" in output
    assert "exact_backend_recall_mean" in output
    assert "exact_backend_overlap_fraction_mean" in output
    assert "exact_backend_best_value_gap_mean" in output


def test_cli_smoke_json_out_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "reports" / "smoke.json"

    exit_code = main(
        [
            "--smoke",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--json-out",
            str(output_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    _assert_json_report_metadata(report, "smoke")
    assert "diagnostics" in report


def test_cli_json_out_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file"):
        main(
            [
                "--smoke",
                "--seed",
                "7",
                "--batch-size",
                "4",
                "--num-transitions",
                "16",
                "--num-actions",
                "8",
                "--state-dim",
                "4",
                "--action-dim",
                "3",
                "--embedding-dim",
                "8",
                "--k",
                "3",
                "--gamma",
                "0.9",
                "--json-out",
                str(tmp_path),
            ]
        )


def test_cli_synthetic_diagnostics_sweep_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "sweep"

    exit_code = main(
        [
            "--sweep-synthetic-diagnostics",
            "--sweep-output-dir",
            str(output_dir),
            "--sweep-refresh-values",
            "1,2",
            "--sweep-seeds",
            "0,1",
            "--sweep-encoder-policies",
            "online",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--num-steps",
            "1",
        ]
    )
    output = capsys.readouterr().out
    reports = sorted(path for path in output_dir.glob("*.json") if path.name not in {"summary.json", "manifest.json"})

    assert exit_code == 0
    assert "synthetic diagnostics sweep only" in output
    assert "num_runs: 4" in output
    assert "output_dir:" in output
    assert len(reports) == 4
    assert (output_dir / "summary.json").exists()
    loaded = json.loads(reports[0].read_text(encoding="utf-8"))
    _assert_json_report_metadata(loaded, "synthetic_diagnostics_sweep_run")


def test_cli_synthetic_diagnostics_sweep_k_values_writes_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "k_sweep"

    exit_code = main(
        [
            "--sweep-synthetic-diagnostics",
            "--sweep-output-dir",
            str(output_dir),
            "--sweep-refresh-values",
            "10",
            "--sweep-seeds",
            "0,1",
            "--sweep-encoder-policies",
            "online,ema",
            "--sweep-k-values",
            "2,3",
            "--index-ema-tau",
            "0.05",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--embedding-dim",
            "8",
            "--num-steps",
            "1",
        ]
    )
    output = capsys.readouterr().out
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "num_runs: 8" in output
    assert summary["sweep_axes"]["k_values"] == [2, 3]
    assert all("k" in run for run in summary["runs"])
    assert (output_dir / "seed0_refresh10_policyema_k2.json").exists()


def test_cli_synthetic_diagnostics_sweep_index_ema_tau_values_writes_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "ema_tau_sweep"

    exit_code = main(
        [
            "--sweep-synthetic-diagnostics",
            "--sweep-output-dir",
            str(output_dir),
            "--sweep-refresh-values",
            "10",
            "--sweep-seeds",
            "0,1",
            "--sweep-encoder-policies",
            "ema",
            "--sweep-k-values",
            "2",
            "--sweep-index-ema-tau-values",
            "0.01,0.05",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--embedding-dim",
            "8",
            "--num-steps",
            "1",
        ]
    )
    output = capsys.readouterr().out
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "num_runs: 4" in output
    assert summary["sweep_axes"]["index_ema_tau_values"] == [0.01, 0.05]
    assert {run["index_ema_tau"] for run in summary["runs"]} == {0.01, 0.05}
    assert (output_dir / "seed0_refresh10_policyema_k2_tau0p01.json").exists()
    assert (output_dir / "seed1_refresh10_policyema_k2_tau0p05.json").exists()


def test_cli_synthetic_diagnostics_sweep_mutual_exclusion() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke", "--sweep-synthetic-diagnostics", "--sweep-output-dir", "runs/sweep"])


def test_cli_synthetic_diagnostics_sweep_requires_output_dir() -> None:
    with pytest.raises(SystemExit):
        main(["--sweep-synthetic-diagnostics"])


def test_cli_run_synthetic_experiment_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "experiment_debug"

    exit_code = main(
        [
            "--run-synthetic-experiment",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            "cli_experiment",
            "--num-steps",
            "2",
            "--batch-size",
            "4",
            "--k",
            "3",
            "--checkpoint-every",
            "1",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
            "--baselines",
            "random,nearest",
        ]
    )
    output = capsys.readouterr().out
    report_path = output_dir / "cli_experiment.json"
    checkpoint_path = output_dir / "final_checkpoint.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "synthetic benchmark scaffold only" in output
    assert "final_checkpoint:" in output
    assert "checkpoint_every: 1" in output
    assert "eval_num_batches: 1" in output
    assert "mean_eval_bellman_loss:" in output
    assert "baselines: random,nearest" in output
    assert "baseline_diagnostics_only: true" in output
    assert "benchmark_claim: false" in output
    assert checkpoint_path.exists()
    assert report["experiment_name"] == "cli_experiment"
    assert report["synthetic_only"] is True
    assert report["benchmark_claim"] is False
    assert report["checkpoint_files"] == [str(checkpoint_path)]
    assert report["evaluation_summary"]["num_batches"] == 1
    assert [item["baseline_name"] for item in report["baseline_diagnostics"]] == ["random", "nearest"]
    assert all(item["benchmark_claim"] is False for item in report["baseline_diagnostics"])


def test_cli_run_synthetic_experiment_invalid_baseline_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-synthetic-experiment",
                "--config",
                str(ROOT / "configs" / "benchmark_synthetic.yaml"),
                "--output-dir",
                str(tmp_path),
                "--baselines",
                "cql",
            ]
        )


def test_cli_validate_movielens_adapter_local_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "\n".join(
            [
                "userId,movieId,rating,timestamp",
                "1,10,5.0,100",
                "1,11,3.0,101",
                "2,10,4.5,102",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--validate-adapter",
            "--adapter-name",
            "movielens",
            "--data-path",
            str(ratings_path),
            "--split-mode",
            "chronological",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "adapter validation scaffold only" in output
    assert "adapter_name: movielens" in output
    assert "available: True" in output
    assert "task_type: movielens_local_scaffold" in output
    assert "dataset_size:" in output
    assert "num_actions:" in output
    assert "benchmark_claim: false" in output
    assert "scaffold_only: true" in output


def test_cli_validate_movielens_adapter_missing_csv_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--validate-adapter",
                "--adapter-name",
                "movielens",
                "--data-path",
                str(tmp_path / "missing.csv"),
            ]
        )


def test_cli_validate_placeholder_adapter_fails_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--validate-adapter", "--adapter-name", "d4rl"])


def test_cli_report_movielens_scaffold_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    report_path = tmp_path / "reports" / "movielens_scaffold.json"
    ratings_path.write_text(
        "\n".join(
            [
                "userId,movieId,rating,timestamp",
                "1,10,5.0,100",
                "1,11,3.0,101",
                "2,10,4.5,102",
                "2,12,2.0,103",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--report-movielens-scaffold",
            "--data-path",
            str(ratings_path),
            "--split-mode",
            "chronological",
            "--json-out",
            str(report_path),
        ]
    )
    output = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "MovieLens local scaffold report only" in output
    assert "benchmark_claim: false" in output
    assert "scaffold_only: true" in output
    assert report["benchmark_claim"] is False
    assert report["scaffold_only"] is True
    assert report["dataset_metadata"]["split_mode"] == "chronological"
    assert "positive_rate" in report["scaffold_metrics"]


def test_cli_report_movielens_scaffold_session_fields_in_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    report_path = tmp_path / "movielens_session.json"
    ratings_path.write_text(
        "\n".join(
            [
                "userId,movieId,rating,timestamp",
                "1,10,5.0,0",
                "1,11,5.0,10",
                "1,12,5.0,100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--report-movielens-scaffold",
            "--data-path",
            str(ratings_path),
            "--split-mode",
            "session",
            "--session-gap-seconds",
            "30",
            "--json-out",
            str(report_path),
        ]
    )
    capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["dataset_metadata"]["split_mode"] == "session"
    assert report["dataset_metadata"]["num_sessions"] == 2


def test_cli_report_movielens_scaffold_directory_json_out_fails(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,0\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "--report-movielens-scaffold",
                "--data-path",
                str(ratings_path),
                "--json-out",
                str(tmp_path),
            ]
        )


def test_cli_report_external_adapter_mock_external_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "mock_external.json"

    exit_code = main(
        [
            "--report-external-adapter",
            "--adapter-name",
            "mock_external",
            "--json-out",
            str(report_path),
        ]
    )
    output = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "external adapter scaffold report only" in output
    assert "benchmark_claim: false" in output
    assert report["report_type"] == "external_adapter_scaffold"
    assert report["adapter_name"] == "mock_external"
    assert report["benchmark_claim"] is False
    assert "action_catalogue_metadata" in report
    assert "dataset_metadata" in report


def test_cli_report_external_adapter_movielens_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    report_path = tmp_path / "movielens_report.json"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,0\n2,11,3.0,1\n", encoding="utf-8")

    exit_code = main(
        [
            "--report-external-adapter",
            "--adapter-name",
            "movielens",
            "--data-path",
            str(ratings_path),
            "--json-out",
            str(report_path),
        ]
    )
    capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["adapter_name"] == "movielens"
    assert report["benchmark_claim"] is False
    assert "movielens_scaffold" in report["evaluation_diagnostics"]


def test_cli_report_external_adapter_recsim_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "recsim_report.json"

    exit_code = main(
        [
            "--report-external-adapter",
            "--adapter-name",
            "recsim",
            "--task-name",
            "mock_slate",
            "--json-out",
            str(report_path),
        ]
    )
    capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["adapter_name"] == "recsim"
    assert report["task_type"] == "recsim_mock_slate"
    assert "recsim_slate_scaffold" in report["evaluation_diagnostics"]
    assert report["benchmark_claim"] is False


def test_cli_report_external_adapter_d4rl_mock_arrays_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "d4rl_report.json"

    exit_code = main(
        [
            "--report-external-adapter",
            "--adapter-name",
            "d4rl",
            "--task-name",
            "mock_arrays",
            "--json-out",
            str(report_path),
        ]
    )
    capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["adapter_name"] == "d4rl"
    assert report["task_type"] == "d4rl_large_action_scaffold"
    assert "d4rl_action_codebook_audit" in report["evaluation_diagnostics"]
    assert report["benchmark_claim"] is False


def test_cli_report_external_adapter_directory_json_out_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--report-external-adapter",
                "--adapter-name",
                "mock_external",
                "--json-out",
                str(tmp_path),
            ]
        )


def test_cli_inspect_external_adapter_report_mock_external(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "mock_external.json"
    assert main(["--report-external-adapter", "--adapter-name", "mock_external", "--json-out", str(report_path)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "--inspect-external-adapter-report",
            "--external-adapter-report-json",
            str(report_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "external adapter scaffold report" in output
    assert "adapter_name: mock_external" in output
    assert "task_type: mock_external" in output
    assert "dataset_size: 6" in output
    assert "num_actions: 4" in output
    assert "benchmark_claim: false" in output


def test_cli_inspect_external_adapter_report_movielens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    report_path = tmp_path / "movielens_report.json"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,0\n2,11,3.0,1\n", encoding="utf-8")
    assert (
        main(
            [
                "--report-external-adapter",
                "--adapter-name",
                "movielens",
                "--data-path",
                str(ratings_path),
                "--json-out",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(["--inspect-external-adapter-report", "--external-adapter-report-json", str(report_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "adapter_name: movielens" in output
    assert "positive_rate:" in output
    assert "catalogue_coverage_observed:" in output
    assert "benchmark_claim: false" in output


def test_cli_inspect_external_adapter_report_missing_path_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--inspect-external-adapter-report",
                "--external-adapter-report-json",
                str(tmp_path / "missing.json"),
            ]
        )


def test_cli_report_external_adapter_bundle_without_movielens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "adapter_bundle"

    exit_code = main(["--report-external-adapter-bundle", "--bundle-output-dir", str(output_dir)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "external adapter scaffold report bundle only" in output
    assert "report_count: 3" in output
    assert "included_adapters: mock_external,d4rl,recsim" in output
    assert "benchmark_claim: false" in output
    assert (output_dir / "manifest.json").exists()
    for filename in ("mock_external_report.json", "d4rl_mock_arrays_report.json", "recsim_mock_slate_report.json"):
        assert (output_dir / filename).exists()


def test_cli_report_external_adapter_bundle_with_movielens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    output_dir = tmp_path / "adapter_bundle"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,0\n2,11,3.0,1\n", encoding="utf-8")

    exit_code = main(
        [
            "--report-external-adapter-bundle",
            "--bundle-output-dir",
            str(output_dir),
            "--movielens-data-path",
            str(ratings_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "report_count: 4" in output
    assert "included_adapters: mock_external,d4rl,recsim,movielens" in output
    assert (output_dir / "movielens_local_report.json").exists()


def test_cli_report_external_adapter_bundle_missing_movielens_path_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--report-external-adapter-bundle",
                "--bundle-output-dir",
                str(tmp_path / "adapter_bundle"),
                "--movielens-data-path",
                str(tmp_path / "missing.csv"),
            ]
        )


def test_cli_inspect_external_adapter_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "adapter_bundle"
    assert main(["--report-external-adapter-bundle", "--bundle-output-dir", str(output_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "--inspect-external-adapter-bundle",
            "--external-adapter-bundle-manifest",
            str(output_dir / "manifest.json"),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "external adapter scaffold bundle" in output
    assert "report_count: 3" in output
    assert "included_adapters: mock_external,d4rl,recsim" in output
    assert "adapter_summary: adapter_name=mock_external" in output
    assert "adapter_summary: adapter_name=d4rl" in output
    assert "adapter_summary: adapter_name=recsim" in output
    assert "benchmark_claim: false" in output


def test_cli_inspect_external_adapter_bundle_missing_manifest_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--inspect-external-adapter-bundle",
                "--external-adapter-bundle-manifest",
                str(tmp_path / "missing_manifest.json"),
            ]
        )


def test_cli_run_movielens_local_experiment_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    output_dir = tmp_path / "movielens_exp"
    ratings_path.write_text(
        "\n".join(
            [
                "userId,movieId,rating,timestamp",
                "1,10,5.0,100",
                "1,11,3.0,101",
                "2,10,4.5,102",
                "2,12,2.0,103",
                "3,11,5.0,104",
                "3,12,1.0,105",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-movielens-local-experiment",
            "--config",
            str(ROOT / "configs" / "movielens_local_experiment.yaml"),
            "--data-path",
            str(ratings_path),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            "cli_movielens",
            "--num-steps",
            "2",
            "--batch-size",
            "2",
            "--k",
            "2",
            "--embedding-dim",
            "8",
            "--baselines",
            "random,nearest",
        ]
    )
    output = capsys.readouterr().out
    report = json.loads((output_dir / "cli_movielens.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "MovieLens local experiment scaffold only" in output
    assert "synthetic_only: false" in output
    assert "benchmark_claim: false" in output
    assert "scaffold_only: true" in output
    assert "baselines: random,nearest" in output
    assert report["report_type"] == "movielens_local_experiment_scaffold"
    assert report["benchmark_claim"] is False
    assert report["scaffold_only"] is True
    assert "training_history_summary" in report
    assert "evaluation_summary" in report
    assert [item["baseline_name"] for item in report["baseline_diagnostics"]] == ["random", "nearest"]
    assert (output_dir / "manifest.json").exists()


def test_cli_inspect_movielens_report_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    output_dir = tmp_path / "movielens_exp"
    ratings_path.write_text(
        "\n".join(
            [
                "userId,movieId,rating,timestamp",
                "1,10,5.0,0",
                "1,11,5.0,1",
                "1,12,5.0,2",
                "1,13,5.0,3",
                "1,14,5.0,4",
                "1,15,5.0,5",
                "1,16,5.0,6",
                "1,17,5.0,7",
                "1,18,5.0,8",
                "1,19,5.0,9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    main(
        [
            "--run-movielens-local-experiment",
            "--data-path",
            str(ratings_path),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            "inspectable",
            "--num-steps",
            "1",
            "--batch-size",
            "2",
            "--k",
            "2",
            "--split-mode",
            "chronological",
        ]
    )
    report_path = output_dir / "inspectable.json"

    exit_code = main(["--inspect-movielens-report", "--movielens-report-json", str(report_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "MovieLens local report inspection scaffold only" in output
    assert "benchmark_claim: false" in output
    assert "num_train_rows:" in output
    assert "num_validation_rows:" in output
    assert "num_test_rows:" in output
    assert "ranking_scaffold_diagnostics:" in output
    assert "candidate_support_diagnostics:" in output
    assert "movielens_pmr_scaffold_diagnostics:" in output
    assert "pmr_at_k_scaffold" in output
    assert "missing_policy_mass_mean" in output
    assert "positive_item_in_topk_rate" in output
    assert "positive_item_rank_mean" in output
    assert "movielens_ann_candidate_audit:" in output
    assert "ann_candidate_recall_mean" in output
    assert "ann_candidate_overlap_fraction_mean" in output
    assert "ann_candidate_pmr_mean" in output
    assert "ann_candidate_best_value_gap_mean" in output
    assert "index_backend" in output
    assert "index_metric" in output


def test_cli_inspect_movielens_report_missing_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--inspect-movielens-report", "--movielens-report-json", str(tmp_path / "missing.json")])


def test_cli_inspect_movielens_report_wrong_type_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"report_type": "other", "scaffold_only": true, "benchmark_claim": false}', encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["--inspect-movielens-report", "--movielens-report-json", str(path)])


def test_cli_run_movielens_local_experiment_mutual_exclusion(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,0\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["--run-movielens-local-experiment", "--smoke", "--data-path", str(ratings_path)])


def test_cli_run_d4rl_mock_experiment_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "d4rl_mock"

    exit_code = main(
        [
            "--run-d4rl-mock-experiment",
            "--config",
            str(ROOT / "configs" / "d4rl_mock_experiment.yaml"),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            "cli_d4rl_mock",
            "--num-steps",
            "2",
            "--batch-size",
            "4",
            "--k",
            "3",
            "--num-transitions",
            "16",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--codebook-size",
            "8",
            "--embedding-dim",
            "8",
        ]
    )
    output = capsys.readouterr().out
    report_path = output_dir / "cli_d4rl_mock.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "D4RL mock large-action experiment scaffold only" in output
    assert "benchmark_claim: false" in output
    assert report["report_type"] == "d4rl_mock_experiment_scaffold"
    assert report["benchmark_claim"] is False
    assert report["scaffold_only"] is True
    assert "d4rl_codebook_audit" in report
    assert "d4rl_pmr_scaffold_diagnostics" in report
    assert "d4rl_ann_candidate_audit" in report
    assert (output_dir / "manifest.json").exists()


def test_cli_run_d4rl_mock_experiment_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-d4rl-mock-experiment",
                "--smoke",
                "--output-dir",
                str(tmp_path / "d4rl"),
                "--num-steps",
                "1",
            ]
        )


def test_cli_run_recsim_mock_experiment_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "recsim_mock"

    exit_code = main(
        [
            "--run-recsim-mock-experiment",
            "--config",
            str(ROOT / "configs" / "recsim_mock_experiment.yaml"),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            "cli_recsim_mock",
            "--num-steps",
            "2",
            "--batch-size",
            "4",
            "--k",
            "3",
            "--num-users",
            "3",
            "--num-items",
            "8",
            "--user-feature-dim",
            "4",
            "--item-feature-dim",
            "3",
            "--slate-size",
            "2",
            "--num-interactions",
            "16",
            "--embedding-dim",
            "8",
        ]
    )
    output = capsys.readouterr().out
    report_path = output_dir / "cli_recsim_mock.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "RecSim mock slate experiment scaffold only" in output
    assert "benchmark_claim: false" in output
    assert report["report_type"] == "recsim_mock_experiment_scaffold"
    assert report["benchmark_claim"] is False
    assert report["scaffold_only"] is True
    assert "recsim_scaffold_evaluation" in report
    assert "recsim_pmr_scaffold_diagnostics" in report
    assert "recsim_ann_candidate_audit" in report
    assert "slate_metadata" in report
    assert (output_dir / "manifest.json").exists()


def test_cli_run_recsim_mock_experiment_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-recsim-mock-experiment",
                "--smoke",
                "--output-dir",
                str(tmp_path / "recsim"),
                "--num-steps",
                "1",
            ]
        )


def test_cli_inspect_external_experiment_report_works_for_recsim(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "recsim"
    assert (
        main(
            [
                "--run-recsim-mock-experiment",
                "--config",
                str(ROOT / "configs" / "recsim_mock_experiment.yaml"),
                "--output-dir",
                str(output_dir),
                "--experiment-name",
                "recsim_inspectable",
                "--num-steps",
                "2",
                "--batch-size",
                "4",
                "--num-items",
                "8",
                "--num-interactions",
                "16",
                "--k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    report_path = output_dir / "recsim_inspectable.json"

    exit_code = main(
        [
            "--inspect-external-experiment-report",
            "--external-experiment-report-json",
            str(report_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "external scaffold experiment report" in output
    assert "report_type: recsim_mock_experiment_scaffold" in output
    assert "scaffold_only: true" in output
    assert "benchmark_claim: false" in output
    assert "observed_item_coverage:" in output
    assert "pmr_at_k_scaffold:" in output
    assert "ann_candidate_pmr_mean:" in output


def test_cli_inspect_external_experiment_report_works_for_d4rl_pmr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "d4rl"
    assert (
        main(
            [
                "--run-d4rl-mock-experiment",
                "--config",
                str(ROOT / "configs" / "d4rl_mock_experiment.yaml"),
                "--output-dir",
                str(output_dir),
                "--experiment-name",
                "d4rl_inspectable",
                "--num-steps",
                "2",
                "--batch-size",
                "4",
                "--num-transitions",
                "12",
                "--codebook-size",
                "6",
                "--k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        [
            "--inspect-external-experiment-report",
            "--external-experiment-report-json",
            str(output_dir / "d4rl_inspectable.json"),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "report_type: d4rl_mock_experiment_scaffold" in output
    assert "pmr_at_k_scaffold:" in output
    assert "logged_action_in_topk_rate:" in output
    assert "ann_candidate_pmr_mean:" in output
    assert "ann_candidate_recall_mean:" in output


def test_cli_inspect_external_experiment_report_missing_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--inspect-external-experiment-report",
                "--external-experiment-report-json",
                str(tmp_path / "missing.json"),
            ]
        )


def test_cli_inspect_external_experiment_report_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--smoke",
                "--inspect-external-experiment-report",
                "--external-experiment-report-json",
                str(tmp_path / "report.json"),
            ]
        )


def test_cli_run_external_experiment_bundle_without_movielens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "external_bundle"

    exit_code = main(
        [
            "--run-external-experiment-bundle",
            "--bundle-output-dir",
            str(output_dir),
            "--num-steps",
            "2",
        ]
    )
    output = capsys.readouterr().out
    manifest_path = output_dir / "external_scaffold_experiment_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "external scaffold experiment bundle only" in output
    assert "report_count: 2" in output
    assert "included_experiments: d4rl_mock,recsim_mock" in output
    assert "benchmark_claim: false" in output
    assert manifest["benchmark_claim"] is False
    assert len(manifest["report_files"]) == 2


def test_cli_run_external_experiment_bundle_with_movielens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n1,10,5.0,0\n1,11,3.0,1\n2,10,4.5,2\n2,12,2.0,3\n3,11,5.0,4\n3,12,1.0,5\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "external_bundle"

    exit_code = main(
        [
            "--run-external-experiment-bundle",
            "--bundle-output-dir",
            str(output_dir),
            "--movielens-data-path",
            str(ratings_path),
            "--num-steps",
            "2",
        ]
    )
    output = capsys.readouterr().out
    manifest = json.loads(
        (output_dir / "external_scaffold_experiment_bundle_manifest.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert "report_count: 3" in output
    assert "included_experiments: d4rl_mock,recsim_mock,movielens_local" in output
    assert manifest["included_experiments"] == ["d4rl_mock", "recsim_mock", "movielens_local"]


def test_cli_inspect_external_experiment_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "external_bundle"
    assert (
        main(
            [
                "--run-external-experiment-bundle",
                "--bundle-output-dir",
                str(output_dir),
                "--num-steps",
                "2",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest_path = output_dir / "external_scaffold_experiment_bundle_manifest.json"

    exit_code = main(
        [
            "--inspect-external-experiment-bundle",
            "--external-experiment-bundle-manifest",
            str(manifest_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "external scaffold experiment bundle" in output
    assert "report_count: 2" in output
    assert "included_experiments: d4rl_mock,recsim_mock" in output
    assert "experiment_summary: report_type=d4rl_mock_experiment_scaffold" in output
    assert "experiment_summary: report_type=recsim_mock_experiment_scaffold" in output
    assert "benchmark_claim: false" in output


def test_cli_inspect_external_experiment_bundle_missing_manifest_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--inspect-external-experiment-bundle",
                "--external-experiment-bundle-manifest",
                str(tmp_path / "missing.json"),
            ]
        )


def test_cli_external_experiment_bundle_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--smoke",
                "--run-external-experiment-bundle",
                "--bundle-output-dir",
                str(tmp_path / "bundle"),
            ]
        )


def test_cli_run_external_support_protocol_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "support"

    exit_code = main(
        [
            "--run-external-support-protocol",
            "--config",
            str(ROOT / "configs" / "external_support_protocol.yaml"),
            "--output-dir",
            str(output_dir),
            "--num-steps",
            "2",
            "--batch-size",
            "4",
            "--train-k",
            "3",
            "--k-values",
            "2,3",
        ]
    )
    output = capsys.readouterr().out
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "external support diagnostics protocol only" in output
    assert "included_routes: d4rl_mock,recsim_mock" in output
    assert "benchmark_claim: false" in output
    assert summary["report_type"] == "external_support_protocol_summary"
    assert summary["benchmark_claim"] is False
    assert len(summary["support_metrics_by_route_and_k"]) == 4


def test_cli_run_external_support_protocol_with_movielens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n1,10,5.0,0\n1,11,3.0,1\n2,10,4.5,2\n2,12,2.0,3\n3,11,5.0,4\n3,12,1.0,5\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "support"

    exit_code = main(
        [
            "--run-external-support-protocol",
            "--config",
            str(ROOT / "configs" / "external_support_protocol.yaml"),
            "--output-dir",
            str(output_dir),
            "--include-movielens",
            "--movielens-data-path",
            str(ratings_path),
            "--exclude-d4rl-mock",
            "--exclude-recsim-mock",
            "--num-steps",
            "2",
            "--batch-size",
            "3",
            "--train-k",
            "2",
            "--k-values",
            "2",
        ]
    )
    output = capsys.readouterr().out
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "included_routes: movielens_local" in output
    assert summary["included_routes"] == ["movielens_local"]


def test_cli_inspect_external_support_protocol_writes_tables(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "support"
    markdown_path = tmp_path / "support.md"
    csv_path = tmp_path / "support.csv"
    assert (
        main(
            [
                "--run-external-support-protocol",
                "--config",
                str(ROOT / "configs" / "external_support_protocol.yaml"),
                "--output-dir",
                str(output_dir),
                "--num-steps",
                "2",
                "--batch-size",
                "4",
                "--train-k",
                "3",
                "--k-values",
                "2",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        [
            "--inspect-external-support-protocol",
            "--external-support-summary-json",
            str(output_dir / "summary.json"),
            "--write-markdown-table",
            str(markdown_path),
            "--write-csv-table",
            str(csv_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "external support diagnostics protocol" in output
    assert "support_row: route=d4rl_mock" in output
    assert "benchmark_claim: false" in output
    assert markdown_path.exists()
    assert csv_path.exists()
    assert "synthetic_only: false" in markdown_path.read_text(encoding="utf-8")
    assert "route,k,pmr_at_k_scaffold" in csv_path.read_text(encoding="utf-8")


def test_cli_inspect_external_support_protocol_missing_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--inspect-external-support-protocol",
                "--external-support-summary-json",
                str(tmp_path / "missing.json"),
            ]
        )


def test_cli_external_support_protocol_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--smoke",
                "--run-external-support-protocol",
                "--config",
                str(ROOT / "configs" / "external_support_protocol.yaml"),
                "--output-dir",
                str(tmp_path / "support"),
            ]
        )


def test_cli_run_synthetic_experiment_resume_from_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first_dir = tmp_path / "first"
    main(
        [
            "--run-synthetic-experiment",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--output-dir",
            str(first_dir),
            "--experiment-name",
            "first",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--k",
            "3",
        ]
    )
    checkpoint_path = first_dir / "final_checkpoint.pt"

    resumed_dir = tmp_path / "resumed"
    exit_code = main(
        [
            "--run-synthetic-experiment",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--output-dir",
            str(resumed_dir),
            "--experiment-name",
            "resumed",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--k",
            "3",
            "--resume-from",
            str(checkpoint_path),
        ]
    )
    output = capsys.readouterr().out
    report = json.loads((resumed_dir / "resumed.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "resume_from:" in output
    assert report["resume_from"] == str(checkpoint_path)


def test_cli_run_synthetic_experiment_invalid_eval_num_batches(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-synthetic-experiment",
                "--config",
                str(ROOT / "configs" / "benchmark_synthetic.yaml"),
                "--output-dir",
                str(tmp_path),
                "--eval-num-batches",
                "0",
            ]
        )


def test_cli_run_synthetic_experiment_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-synthetic-experiment",
                "--smoke",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_cli_run_synthetic_experiment_sweep_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "experiment_sweep"

    exit_code = main(
        [
            "--run-synthetic-experiment-sweep",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--experiment-sweep-output-dir",
            str(output_dir),
            "--experiment-sweep-seeds",
            "0,1",
            "--experiment-sweep-k-values",
            "3",
            "--experiment-sweep-refresh-values",
            "1",
            "--experiment-sweep-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
        ]
    )
    output = capsys.readouterr().out
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "synthetic experiment sweep scaffold only" in output
    assert "num_runs: 2" in output
    assert "benchmark_claim: false" in output
    assert summary["benchmark_claim"] is False
    assert summary["num_runs"] == 2


def test_cli_run_synthetic_experiment_sweep_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-synthetic-experiment-sweep",
                "--run-synthetic-experiment",
                "--experiment-sweep-output-dir",
                str(tmp_path),
            ]
        )


def test_cli_run_synthetic_experiment_sweep_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-synthetic-experiment-sweep",
                "--config",
                str(ROOT / "configs" / "benchmark_synthetic.yaml"),
                "--experiment-sweep-output-dir",
                str(tmp_path),
                "--experiment-sweep-k-values",
                "0",
            ]
        )


def test_cli_inspect_experiment_sweep_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "experiment_sweep"
    main(
        [
            "--run-synthetic-experiment-sweep",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--experiment-sweep-output-dir",
            str(output_dir),
            "--experiment-sweep-seeds",
            "0",
            "--experiment-sweep-k-values",
            "3",
            "--experiment-sweep-refresh-values",
            "1",
            "--experiment-sweep-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
        ]
    )

    exit_code = main(
        [
            "--inspect-experiment-sweep",
            "--experiment-summary-json",
            str(output_dir / "summary.json"),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic experiment sweep diagnostics only" in output
    assert "benchmark_claim: false" in output
    assert "num_runs: 1" in output


def test_cli_inspect_experiment_sweep_grouped_works(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "experiment_sweep"
    main(
        [
            "--run-synthetic-experiment-sweep",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--experiment-sweep-output-dir",
            str(output_dir),
            "--experiment-sweep-seeds",
            "0,1",
            "--experiment-sweep-k-values",
            "3",
            "--experiment-sweep-refresh-values",
            "1",
            "--experiment-sweep-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
        ]
    )

    exit_code = main(
        [
            "--inspect-experiment-sweep",
            "--experiment-summary-json",
            str(output_dir / "summary.json"),
            "--experiment-grouped",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "grouped_metrics:" in output
    assert "index_encoder_policy=online" in output
    assert "mean_bellman_loss=" in output


def test_cli_inspect_experiment_sweep_writes_markdown_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "experiment_sweep"
    table_path = output_dir / "table.md"
    main(
        [
            "--run-synthetic-experiment-sweep",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--experiment-sweep-output-dir",
            str(output_dir),
            "--experiment-sweep-seeds",
            "0,1",
            "--experiment-sweep-k-values",
            "3",
            "--experiment-sweep-refresh-values",
            "1",
            "--experiment-sweep-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
        ]
    )

    exit_code = main(
        [
            "--inspect-experiment-sweep",
            "--experiment-summary-json",
            str(output_dir / "summary.json"),
            "--experiment-grouped",
            "--write-markdown-table",
            str(table_path),
        ]
    )
    capsys.readouterr()
    markdown = table_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert table_path.exists()
    assert "# Synthetic Experiment Sweep Descriptive Statistics" in markdown
    assert "benchmark_claim: false" in markdown
    assert "mean_bellman_loss_mean" in markdown


def test_cli_inspect_experiment_sweep_writes_csv_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "experiment_sweep"
    csv_path = output_dir / "table.csv"
    main(
        [
            "--run-synthetic-experiment-sweep",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--experiment-sweep-output-dir",
            str(output_dir),
            "--experiment-sweep-seeds",
            "0,1",
            "--experiment-sweep-k-values",
            "3",
            "--experiment-sweep-refresh-values",
            "1",
            "--experiment-sweep-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
        ]
    )

    exit_code = main(
        [
            "--inspect-experiment-sweep",
            "--experiment-summary-json",
            str(output_dir / "summary.json"),
            "--write-csv-table",
            str(csv_path),
        ]
    )
    capsys.readouterr()
    csv_text = csv_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert csv_path.exists()
    assert "index_encoder_policy,index_refresh_every" in csv_text
    assert "mean_bellman_loss_mean" in csv_text


def test_cli_inspect_experiment_sweep_writes_markdown_and_csv_tables(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "experiment_sweep"
    markdown_path = output_dir / "table.md"
    csv_path = output_dir / "table.csv"
    main(
        [
            "--run-synthetic-experiment-sweep",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--experiment-sweep-output-dir",
            str(output_dir),
            "--experiment-sweep-seeds",
            "0",
            "--experiment-sweep-k-values",
            "3",
            "--experiment-sweep-refresh-values",
            "1",
            "--experiment-sweep-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
        ]
    )

    exit_code = main(
        [
            "--inspect-experiment-sweep",
            "--experiment-summary-json",
            str(output_dir / "summary.json"),
            "--write-markdown-table",
            str(markdown_path),
            "--write-csv-table",
            str(csv_path),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert markdown_path.exists()
    assert csv_path.exists()


def test_cli_inspect_experiment_sweep_writes_plot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "experiment_sweep"
    plot_path = output_dir / "bellman.png"
    main(
        [
            "--run-synthetic-experiment-sweep",
            "--config",
            str(ROOT / "configs" / "benchmark_synthetic.yaml"),
            "--experiment-sweep-output-dir",
            str(output_dir),
            "--experiment-sweep-seeds",
            "0,1",
            "--experiment-sweep-k-values",
            "3",
            "--experiment-sweep-refresh-values",
            "1",
            "--experiment-sweep-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--eval-num-batches",
            "1",
            "--eval-batch-size",
            "4",
        ]
    )

    exit_code = main(
        [
            "--inspect-experiment-sweep",
            "--experiment-summary-json",
            str(output_dir / "summary.json"),
            "--plot-metric",
            "mean_bellman_loss",
            "--plot-x-key",
            "index_refresh_every",
            "--plot-group-key",
            "index_encoder_policy",
            "--plot-out",
            str(plot_path),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert plot_path.exists()
    assert plot_path.read_bytes().startswith(b"\x89PNG")


def test_cli_inspect_experiment_sweep_plot_args_must_pair() -> None:
    with pytest.raises(SystemExit):
        main(["--inspect-experiment-sweep", "--experiment-summary-json", "missing.json", "--plot-metric", "x"])

    with pytest.raises(SystemExit):
        main(["--plot-metric", "x", "--plot-out", "plot.png"])


def test_cli_inspect_experiment_sweep_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--inspect-experiment-sweep",
                "--run-synthetic-experiment",
                "--experiment-summary-json",
                str(tmp_path / "summary.json"),
            ]
        )


def test_cli_inspect_sweep_summary_prints_compact_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "sweep"
    main(
        [
            "--sweep-synthetic-diagnostics",
            "--sweep-output-dir",
            str(output_dir),
            "--sweep-refresh-values",
            "1,2",
            "--sweep-seeds",
            "0",
            "--sweep-encoder-policies",
            "online",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--num-steps",
            "1",
        ]
    )
    capsys.readouterr()

    exit_code = main(["--inspect-sweep-summary", "--summary-json", str(output_dir / "summary.json")])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic diagnostics summary inspection only" in output
    assert "benchmark_claim: false" in output
    assert "num_runs: 2" in output
    assert "sweep_axes:" in output
    assert "top_by_final_stale_index_recall_mean" in output
    assert "bottom_by_final_stale_index_recall_mean" in output
    assert "seed refresh policy recall overlap drift_l2 final_age" in output


def test_cli_inspect_sweep_summary_grouped_prints_grouped_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": 3,
        "output_dir": str(tmp_path),
        "run_files": [],
        "sweep_axes": {"seeds": [0, 1], "index_refresh_values": [1], "index_encoder_policies": ["online"]},
        "runs": [
            {
                "seed": 0,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "final_stale_index_recall_mean": 1.0,
                "final_stale_index_overlap_fraction_mean": 1.0,
                "final_index_encoder_drift_mean_l2": 0.0,
            },
            {
                "seed": 1,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "final_stale_index_recall_mean": 0.8,
                "final_stale_index_overlap_fraction_mean": 0.9,
                "final_index_encoder_drift_mean_l2": 0.2,
            },
            {
                "seed": 0,
                "index_refresh_every": 5,
                "index_encoder_policy": "ema",
                "final_stale_index_recall_mean": 0.5,
                "final_stale_index_overlap_fraction_mean": 0.6,
                "final_index_encoder_drift_mean_l2": 0.4,
            },
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main(["--inspect-sweep-summary", "--summary-json", str(summary_path), "--grouped"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic diagnostics summary inspection only" in output
    assert "grouped_by: index_encoder_policy,index_refresh_every" in output
    assert "index_encoder_policy index_refresh_every count recall_mean recall_std pmr_mean pmr_std overlap_mean overlap_std drift_l2_mean drift_l2_std" in output
    assert "online 1 2 0.9 0.1 None None 0.95 0.05 0.1 0.1" in output
    assert "benchmark_claim: false" in output


def test_cli_inspect_sweep_summary_group_keys_include_k(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": 2,
        "output_dir": str(tmp_path),
        "run_files": [],
        "sweep_axes": {
            "seeds": [0],
            "index_refresh_values": [10],
            "index_encoder_policies": ["online"],
            "k_values": [5, 10],
        },
        "runs": [
            {
                "seed": 0,
                "index_refresh_every": 10,
                "index_encoder_policy": "online",
                "k": 5,
                "final_stale_index_recall_mean": 0.8,
            },
            {
                "seed": 0,
                "index_refresh_every": 10,
                "index_encoder_policy": "online",
                "k": 10,
                "final_stale_index_recall_mean": 0.9,
            },
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main(
        [
            "--inspect-sweep-summary",
            "--summary-json",
            str(summary_path),
            "--grouped",
            "--group-keys",
            "index_encoder_policy, index_refresh_every, k",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "grouped_by: index_encoder_policy,index_refresh_every,k" in output
    assert "online 10 5 1 0.8 0 None None" in output
    assert "online 10 10 1 0.9 0 None None" in output


def test_cli_inspect_sweep_summary_group_keys_include_index_ema_tau(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": 2,
        "output_dir": str(tmp_path),
        "run_files": [],
        "sweep_axes": {
            "seeds": [0],
            "index_refresh_values": [50],
            "index_encoder_policies": ["ema"],
            "k_values": [10],
            "index_ema_tau_values": [0.01, 0.05],
        },
        "runs": [
            {
                "seed": 0,
                "index_refresh_every": 50,
                "index_encoder_policy": "ema",
                "k": 10,
                "index_ema_tau": 0.01,
                "final_stale_index_recall_mean": 0.7,
            },
            {
                "seed": 1,
                "index_refresh_every": 50,
                "index_encoder_policy": "ema",
                "k": 10,
                "index_ema_tau": 0.05,
                "final_stale_index_recall_mean": 0.9,
            },
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main(
        [
            "--inspect-sweep-summary",
            "--summary-json",
            str(summary_path),
            "--grouped",
            "--group-keys",
            "index_refresh_every,k,index_ema_tau",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "grouped_by: index_refresh_every,k,index_ema_tau" in output
    assert "50 10 0.01 1 0.7 0 None None" in output
    assert "50 10 0.05 1 0.9 0 None None" in output


def test_cli_inspect_sweep_summary_missing_group_key_fails(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": 1,
        "output_dir": str(tmp_path),
        "run_files": [],
        "sweep_axes": {"seeds": [0]},
        "runs": [{"seed": 0}],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "--inspect-sweep-summary",
                "--summary-json",
                str(summary_path),
                "--grouped",
                "--group-keys",
                "index_encoder_policy,k",
            ]
        )


def test_cli_inspect_sweep_summary_handles_missing_optional_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": 1,
        "output_dir": str(tmp_path),
        "run_files": [],
        "sweep_axes": {"seeds": [0], "index_refresh_values": [1], "index_encoder_policies": ["online"]},
        "runs": [{"seed": 0, "index_refresh_every": 1, "index_encoder_policy": "online"}],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main(["--inspect-sweep-summary", "--summary-json", str(summary_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "None" in output
    assert "benchmark_claim: false" in output


def test_cli_inspect_sweep_summary_associations_prints_diagnostics_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "synthetic_diagnostics_sweep",
        "synthetic_only": True,
        "benchmark_claim": False,
        "num_runs": 3,
        "output_dir": str(tmp_path),
        "run_files": [],
        "sweep_axes": {"seeds": [0, 1, 2]},
        "runs": [
            {"final_stale_index_pmr_mean": 0.2, "final_bellman_loss": 4.0},
            {"final_stale_index_pmr_mean": 0.5, "final_bellman_loss": 2.0},
            {"final_stale_index_pmr_mean": 0.8, "final_bellman_loss": 1.0},
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main(["--inspect-sweep-summary", "--summary-json", str(summary_path), "--associations"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic diagnostics associations only" in output
    assert "benchmark_claim: false" in output
    assert "causality_implied: false" in output
    assert "final_stale_index_pmr_mean__vs__final_bellman_loss" in output


def test_cli_inspect_sweep_summary_invalid_path_fails_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--inspect-sweep-summary", "--summary-json", "missing/summary.json"])


def test_cli_smoke_exact_backend_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--index-backend",
            "exact",
            "--index-metric",
            "cosine",
        ]
    )

    assert exit_code == 0
    assert "DW-MQL smoke check passed" in capsys.readouterr().out


def test_cli_smoke_train_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke-train",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
            "--lr",
            "0.001",
            "--target-tau",
            "0.5",
            "--target-update-every",
            "1",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DW-MQL synthetic smoke train only passed" in output
    assert "num_steps" in output
    assert "initial_total_loss" in output
    assert "final_total_loss" in output
    assert "mean_grad_norm" in output
    assert "mean_parameter_delta_norm" in output
    assert "target_update_count" in output
    assert "group_grad_norms_mean" in output
    assert "group_delta_norms_mean" in output
    for group in ("encoder", "critic", "density", "dual"):
        assert f"  {group}:" in output


def test_cli_smoke_train_json_out_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "smoke_train.json"

    exit_code = main(
        [
            "--smoke-train",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
            "--json-out",
            str(output_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    _assert_json_report_metadata(report, "smoke_train")
    assert report["summary"]["num_steps"] == 2


def test_cli_alternating_smoke_train_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke-train",
            "--alternating-smoke",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
            "--lr",
            "0.001",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DW-MQL synthetic alternating smoke train only passed" in output
    assert "num_steps" in output
    assert "initial_total_loss" in output
    assert "final_total_loss" in output
    assert "mean_grad_norm" in output
    assert "mean_parameter_delta_norm" in output
    assert "target_update_count" in output
    assert "schedule_name" in output
    assert "schedule_length" in output
    assert "group_grad_norms_mean" in output
    assert "group_delta_norms_mean" in output
    for group in ("encoder", "critic", "density", "dual"):
        assert f"  {group}:" in output


def test_cli_smoke_eval_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke-eval",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
            "--lr",
            "0.001",
            "--index-refresh-every",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic smoke evaluation only" in output
    assert "before_bellman_loss" in output
    assert "after_bellman_loss" in output
    assert "bellman_loss_delta" in output
    assert "stationarity_loss_delta" in output
    assert "ess_delta" in output
    assert "mean_recall_delta" in output
    assert "retrieval_diagnostics_only: true" in output
    assert "benchmark_claim: false" in output
    assert "before_exact_backend_recall_mean" in output
    assert "after_exact_backend_recall_mean" in output
    assert "exact_backend_recall_mean_delta" in output
    assert "before_exact_backend_overlap_fraction_mean" in output
    assert "after_exact_backend_overlap_fraction_mean" in output
    assert "exact_backend_overlap_fraction_mean_delta" in output
    assert "before_exact_backend_best_value_gap_mean" in output
    assert "after_exact_backend_best_value_gap_mean" in output
    assert "exact_backend_best_value_gap_mean_delta" in output


def test_cli_smoke_eval_json_out_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "smoke_eval.json"

    exit_code = main(
        [
            "--smoke-eval",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
            "--json-out",
            str(output_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    _assert_json_report_metadata(report, "smoke_eval")
    assert "deltas" in report
    assert "bellman_loss_delta" in report["deltas"]


def test_cli_smoke_eval_alternating_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke-eval",
            "--alternating-smoke",
            "--assert-gradient-flow",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
            "--lr",
            "0.001",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic smoke evaluation only" in output
    assert "before_bellman_loss" in output
    assert "after_bellman_loss" in output


def test_cli_train_synthetic_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
            "--lr",
            "0.001",
            "--index-refresh-every",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic training scaffold only" in output
    assert "num_steps" in output
    assert "initial_total_loss" in output
    assert "final_total_loss" in output
    assert "mean_total_loss" in output
    assert "mean_bellman_loss" in output
    assert "mean_stationarity_loss" in output
    assert "mean_density_regularization" in output
    assert "mean_grad_norm" in output
    assert "mean_parameter_delta_norm" in output
    assert "target_update_count" in output
    assert "index_refresh_every: 2" in output
    assert "index_refresh_count: 1" in output
    assert "final_index_age: 0" in output
    assert "persistent_index_used: false" in output


def test_cli_train_synthetic_json_out_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "synthetic_train.json"

    exit_code = main(
        [
            "--train-synthetic",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
            "--json-out",
            str(output_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    _assert_json_report_metadata(report, "train_synthetic")
    assert report["summary"]["num_steps"] == 2
    assert "history" in report


def test_cli_train_objective_synthetic_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-objective-synthetic",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
            "--lr",
            "0.001",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DW-MQL synthetic objective training scaffold only" in output
    assert "num_steps: 3" in output
    assert "schedule: dual,density,critic" in output
    assert "mean_critic_loss" in output
    assert "mean_density_loss" in output
    assert "mean_dual_objective_loss" in output
    assert "final_critic_loss" in output
    assert "final_density_loss" in output
    assert "final_dual_loss" in output
    assert "persistent_index_used: false" in output


def test_cli_train_objective_synthetic_custom_schedule(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-objective-synthetic",
            "--objective-schedule",
            "critic,density,dual",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--num-steps",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "schedule: critic,density,dual" in output


def test_cli_train_objective_synthetic_invalid_schedule_fails() -> None:
    with pytest.raises(SystemExit):
        main(["--train-objective-synthetic", "--objective-schedule", "dual,bad,critic"])


def test_cli_train_objective_synthetic_json_out_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "objective_train.json"

    exit_code = main(
        [
            "--train-objective-synthetic",
            "--objective-schedule",
            "dual,density,critic",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--num-steps",
            "3",
            "--json-out",
            str(output_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    _assert_json_report_metadata(report, "train_objective_synthetic")
    assert report["objective_schedule"] == ["dual", "density", "critic"]
    assert report["objective_names"] == ["dual", "density", "critic"]
    assert len(report["objective_losses"]) == 3
    assert len(report["objective_updates"]) == 3
    assert report["summary"]["num_steps"] == 3
    assert report["synthetic_only"] is True
    assert report["benchmark_claim"] is False


def test_cli_train_objective_synthetic_with_persistent_stale_audit_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--train-objective-synthetic",
            "--use-persistent-index",
            "--audit-stale-index",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--num-steps",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "persistent_index_used: true" in output
    assert "index_refresh_count" in output
    assert "final_index_age" in output
    assert "stale_index_recall_mean_final" in output
    assert "stale_index_overlap_fraction_mean_final" in output
    assert "final_stale_index_pmr_mean" in output


def test_cli_train_objective_synthetic_loads_ready_config_with_tiny_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = ROOT / "configs" / "objective_synthetic.yaml"
    output_path = tmp_path / "objective_synthetic.json"

    exit_code = main(
        [
            "--train-objective-synthetic",
            "--config",
            str(config_path),
            "--num-steps",
            "3",
            "--json-out",
            str(output_path),
        ]
    )
    output = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "DW-MQL synthetic objective training scaffold only" in output
    assert "num_steps: 3" in output
    assert "schedule: dual,density,critic" in output
    assert "persistent_index_used: true" in output
    assert "index_encoder_policy: ema" in output
    assert "index_refresh_count" in output
    assert "stale_index_recall_mean_final" in output
    assert "final_stale_index_pmr_mean" in output
    assert "final_ema_vs_online_recall_mean" in output
    _assert_json_report_metadata(report, "train_objective_synthetic")
    assert report["summary"]["persistent_index_used"] is True
    assert report["summary"]["index_refresh_count"] == 0
    assert report["summary"]["final_index_age"] == 3
    assert "final_stale_index_pmr_mean" in report["summary"]
    assert len(report["stale_index_recall_means"]) == 3
    assert len(report["stale_index_pmr_means"]) == 3
    assert len(report["ema_vs_online_recall_means"]) == 3
    assert report["diagnostics"]["audit_stale_index"] is True


def test_cli_train_synthetic_with_persistent_index_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--use-persistent-index",
            "--index-refresh-every",
            "2",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic training scaffold only" in output
    assert "persistent_index_used: true" in output
    assert "index_encoder_policy: online" in output
    assert "index_refresh_every: 2" in output
    assert "index_refresh_count: 1" in output
    assert "final_index_age: 1" in output


def test_cli_train_synthetic_with_stale_index_audit_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--use-persistent-index",
            "--audit-stale-index",
            "--index-refresh-every",
            "2",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "persistent_index_used: true" in output
    assert "stale_index_recall_mean_final" in output
    assert "stale_index_overlap_fraction_mean_final" in output
    assert "stale_index_best_value_gap_mean_final" in output
    assert "final_stale_index_pmr_mean" in output
    assert "final_stale_index_pmr_min" in output
    assert "final_stale_index_pmr_max" in output


def test_cli_train_synthetic_accepts_target_index_encoder_policy(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--use-persistent-index",
            "--target-tau",
            "0.5",
            "--target-update-every",
            "1",
            "--index-encoder-policy",
            "target",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "persistent_index_used: true" in output
    assert "index_encoder_policy: target" in output
    assert "final_index_encoder_drift_mean_l2" in output
    assert "final_index_encoder_drift_max_l2" in output


def test_cli_train_synthetic_with_encoder_policy_overlap_audit(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--use-persistent-index",
            "--target-tau",
            "0.5",
            "--index-encoder-policy",
            "target",
            "--audit-encoder-policy-overlap",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "final_target_vs_online_recall_mean" in output
    assert "final_target_vs_online_overlap_fraction_mean" in output


def test_cli_target_index_encoder_policy_requires_target_tau() -> None:
    with pytest.raises(SystemExit):
        main(["--train-synthetic", "--use-persistent-index", "--index-encoder-policy", "target"])


def test_cli_train_synthetic_accepts_ema_index_encoder_policy(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--use-persistent-index",
            "--index-encoder-policy",
            "ema",
            "--index-ema-tau",
            "0.05",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "persistent_index_used: true" in output
    assert "index_encoder_policy: ema" in output
    assert "index_ema_tau: 0.05" in output
    assert "index_ema_update_count: 2" in output
    assert "final_index_encoder_drift_mean_l2" in output
    assert "final_index_encoder_drift_max_l2" in output
    assert "final_index_encoder_drift_mean_cosine_distance" in output


def test_cli_ema_index_encoder_policy_requires_index_ema_tau() -> None:
    with pytest.raises(SystemExit):
        main(["--train-synthetic", "--use-persistent-index", "--index-encoder-policy", "ema"])


def test_cli_train_synthetic_exact_backend_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
            "--index-backend",
            "exact",
        ]
    )

    assert exit_code == 0
    assert "synthetic training scaffold only" in capsys.readouterr().out


def test_cli_train_synthetic_alternating_path_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--alternating-smoke",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert "synthetic training scaffold only" in capsys.readouterr().out


def test_cli_train_synthetic_alternating_split_backward_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--alternating-smoke",
            "--split-backward",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic training scaffold only" in output
    assert "split_backward: true" in output


def test_cli_train_synthetic_with_target_tau_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--train-synthetic",
            "--target-tau",
            "0.5",
            "--target-update-every",
            "1",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "target_update_count: 2" in output


def test_cli_config_works_for_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "batch_size: 4\nnum_transitions: 16\nnum_actions: 8\nk: 3\nembedding_dim: 8\n",
        encoding="utf-8",
    )

    exit_code = main(["--smoke", "--config", str(config_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DW-MQL smoke check passed" in output


def test_cli_config_works_for_smoke_train(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "batch_size: 4\nnum_transitions: 16\nnum_actions: 8\nk: 3\nembedding_dim: 8\nnum_steps: 2\n",
        encoding="utf-8",
    )

    exit_code = main(["--smoke-train", "--config", str(config_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic smoke train only" in output


def test_cli_config_works_for_smoke_eval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "batch_size: 4\nnum_transitions: 16\nnum_actions: 8\nk: 3\nembedding_dim: 8\nnum_steps: 2\n",
        encoding="utf-8",
    )

    exit_code = main(["--smoke-eval", "--config", str(config_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic smoke evaluation only" in output


def test_cli_config_values_are_overridden_by_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "batch_size: 2\nnum_transitions: 16\nnum_actions: 8\nk: 3\nembedding_dim: 8\n",
        encoding="utf-8",
    )

    exit_code = main(["--smoke", "--config", str(config_path), "--batch-size", "4"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "batch_size: 4" in output


def test_cli_config_index_values_are_overridden_by_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "batch_size: 4\nnum_transitions: 16\nnum_actions: 8\nk: 3\nembedding_dim: 8\nindex_metric: cosine\n",
        encoding="utf-8",
    )

    exit_code = main(["--smoke", "--config", str(config_path), "--index-backend", "exact", "--index-metric", "dot"])

    assert exit_code == 0
    assert "DW-MQL smoke check passed" in capsys.readouterr().out


def test_cli_faiss_backend_missing_raises_import_error_or_runs(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "--smoke",
        "--seed",
        "7",
        "--batch-size",
        "4",
        "--num-transitions",
        "16",
        "--num-actions",
        "8",
        "--state-dim",
        "4",
        "--action-dim",
        "3",
        "--embedding-dim",
        "8",
        "--k",
        "3",
        "--index-backend",
        "faiss",
    ]
    try:
        __import__("faiss")
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match="FAISS"):
            main(args)
    else:
        assert main(args) == 0
        assert "DW-MQL smoke check passed" in capsys.readouterr().out


def test_cli_faiss_ivf_backend_missing_raises_import_error_or_runs(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "--smoke",
        "--seed",
        "7",
        "--batch-size",
        "4",
        "--num-transitions",
        "16",
        "--num-actions",
        "8",
        "--state-dim",
        "4",
        "--action-dim",
        "3",
        "--embedding-dim",
        "8",
        "--k",
        "3",
        "--index-backend",
        "faiss_ivf",
        "--index-nlist",
        "2",
        "--index-nprobe",
        "2",
    ]
    try:
        __import__("faiss")
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match="FAISS"):
            main(args)
    else:
        assert main(args) == 0
        assert "DW-MQL smoke check passed" in capsys.readouterr().out


def test_cli_smoke_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke", "--smoke-train"])


def test_cli_run_support_loss_ablation_tiny_grid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "support_ablation"

    exit_code = main(
        [
            "--run-support-loss-ablation",
            "--support-ablation-output-dir",
            str(output_dir),
            "--support-ablation-modes",
            "aware,uniform_weights",
            "--support-ablation-seeds",
            "0",
            "--support-ablation-k-values",
            "2",
            "--support-ablation-refresh-values",
            "1",
            "--support-ablation-encoder-policies",
            "online",
            "--num-steps",
            "1",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--support-correction-weight",
            "0.1",
            "--support-correction-weight-values",
            "0,0.1",
            "--support-weighting-modes",
            "none,missing_mass",
            "--support-weight-scale-values",
            "0.5",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DW-MQL synthetic support-loss ablation only" in output
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_claim"] is False
    assert summary["num_runs"] == 5
    assert summary["sweep_axes"]["support_correction_weight_values"] == [0.0, 0.1]
    assert summary["sweep_axes"]["support_weighting_modes"] == ["none", "missing_mass"]
    assert summary["sweep_axes"]["support_weight_scale_values"] == [0.5]
    assert {run["retrieval_correction_mode"] for run in summary["runs"]} == {"aware", "uniform_weights"}
    aware_weights = sorted(run["support_correction_weight"] for run in summary["runs"] if run["retrieval_correction_mode"] == "aware")
    uniform_run = next(run for run in summary["runs"] if run["retrieval_correction_mode"] == "uniform_weights")
    assert aware_weights == [0.0, 0.0, 0.1, 0.1]
    assert {run["support_weighting_mode"] for run in summary["runs"] if run["retrieval_correction_mode"] == "aware"} == {"none", "missing_mass"}
    assert any(run["support_correction_active"] in {True, False} for run in summary["runs"] if run["retrieval_correction_mode"] == "aware")
    assert uniform_run["support_correction_active"] is False


def test_cli_inspect_support_loss_ablation_associations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_name": DIAGNOSTICS_SCHEMA_NAME,
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "mode": "support_loss_ablation",
                "synthetic_only": True,
                "benchmark_claim": False,
                "num_runs": 3,
                "run_files": [],
                "runs": [
                    {"final_stale_index_pmr_mean": 0.1, "final_bellman_loss": 3.0},
                    {"final_stale_index_pmr_mean": 0.2, "final_bellman_loss": 2.0},
                    {"final_stale_index_pmr_mean": 0.3, "final_bellman_loss": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--inspect-sweep-summary",
            "--summary-json",
            str(summary_path),
            "--support-associations",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "support_loss_associations: true" in output
    assert "benchmark_claim: false" in output
    assert "support_loss_ablation_grouped_diagnostics:" in output
    assert "grouped_by: retrieval_correction_mode" in output


def test_cli_inspect_support_loss_ablation_paired_analysis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    runs = []
    for mode, loss in (("aware", 1.0), ("naive", 1.5), ("uniform_weights", 2.0)):
        runs.append(
            {
                "seed": 0,
                "k": 2,
                "index_refresh_every": 1,
                "index_encoder_policy": "online",
                "index_ema_tau": None,
                "retrieval_correction_mode": mode,
                "mean_bellman_loss": loss,
                "final_bellman_loss": loss,
                "final_stale_index_pmr_normalized_mean": 2.0,
                "final_stale_index_pmr_mean": 0.5,
                "final_stale_index_missing_mass_mean": 0.5,
            }
        )
    summary_path.write_text(
        json.dumps(
            {
                "schema_name": DIAGNOSTICS_SCHEMA_NAME,
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "mode": "support_loss_ablation",
                "synthetic_only": True,
                "benchmark_claim": False,
                "num_runs": 3,
                "run_files": [],
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--inspect-sweep-summary",
            "--summary-json",
            str(summary_path),
            "--paired-correction-analysis",
            "--support-bins",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "synthetic paired correction diagnostics only" in output
    assert "reference_mode: aware" in output
    assert "comparison: naive" in output
    assert "mean_bellman_loss: count=1" in output
    assert "support_bins: true" in output


def test_cli_support_loss_ablation_mutual_exclusion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--run-support-loss-ablation",
                "--smoke",
                "--support-ablation-output-dir",
                str(tmp_path / "out"),
            ]
        )

    with pytest.raises(SystemExit):
        main(["--smoke", "--smoke-eval"])

    with pytest.raises(SystemExit):
        main(["--smoke-train", "--smoke-eval"])

    with pytest.raises(SystemExit):
        main(["--smoke", "--train-synthetic"])

    with pytest.raises(SystemExit):
        main(["--smoke-train", "--train-synthetic"])

    with pytest.raises(SystemExit):
        main(["--smoke-eval", "--train-synthetic"])

    with pytest.raises(SystemExit):
        main(["--smoke", "--train-objective-synthetic"])

    with pytest.raises(SystemExit):
        main(["--train-synthetic", "--train-objective-synthetic"])

    with pytest.raises(SystemExit):
        main(["--sweep-synthetic-diagnostics", "--train-objective-synthetic", "--sweep-output-dir", "runs/sweep"])

    with pytest.raises(SystemExit):
        main(["--smoke", "--inspect-sweep-summary", "--summary-json", "runs/sweep/summary.json"])


def test_cli_use_persistent_index_requires_train_synthetic() -> None:
    with pytest.raises(SystemExit):
        main(["--use-persistent-index"])

    with pytest.raises(SystemExit):
        main(["--smoke", "--use-persistent-index"])

    with pytest.raises(SystemExit):
        main(["--smoke-train", "--use-persistent-index"])

    with pytest.raises(SystemExit):
        main(["--smoke-eval", "--use-persistent-index"])

    assert main(["--train-objective-synthetic", "--use-persistent-index", "--num-steps", "1"]) == 0


def test_cli_audit_stale_index_requires_persistent_index() -> None:
    with pytest.raises(SystemExit):
        main(["--audit-stale-index"])

    with pytest.raises(SystemExit):
        main(["--train-synthetic", "--audit-stale-index"])


def test_cli_audit_encoder_policy_overlap_requires_persistent_index() -> None:
    with pytest.raises(SystemExit):
        main(["--audit-encoder-policy-overlap"])

    with pytest.raises(SystemExit):
        main(["--train-synthetic", "--audit-encoder-policy-overlap"])


def test_alternating_smoke_requires_smoke_train() -> None:
    with pytest.raises(SystemExit):
        main(["--alternating-smoke"])


def test_no_encoder_on_critic_requires_alternating_smoke() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke-train", "--no-encoder-on-critic"])

    with pytest.raises(SystemExit):
        main(["--smoke-eval", "--no-encoder-on-critic"])


def test_missing_smoke_does_not_start_training(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Training is not implemented yet" in output


def test_invalid_k_raises_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke", "--k", "0"])

    config = _config()
    config.algorithm.candidate_k = 0
    with pytest.raises(ValueError, match="candidate_k"):
        run_smoke_from_config(config)


def test_invalid_gamma_raises_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke", "--gamma", "1.5"])

    config = _config()
    config.algorithm.gamma = 1.5
    with pytest.raises(ValueError, match="gamma"):
        run_smoke_from_config(config)


def test_invalid_index_backend_in_config_raises_cleanly() -> None:
    config = _config()
    config.ann = ANNConfig(backend="bad", metric="dot")

    with pytest.raises(ValueError, match="ann.backend"):
        run_smoke_from_config(config)


def test_invalid_index_metric_in_config_raises_cleanly() -> None:
    config = _config()
    config.ann = ANNConfig(backend="exact", metric="ip")

    with pytest.raises(ValueError, match="ann.metric"):
        run_smoke_from_config(config)


def test_invalid_index_nlist_in_config_raises_cleanly() -> None:
    config = _config()
    config.ann = ANNConfig(backend="exact", metric="dot", nlist=0)

    with pytest.raises(ValueError, match="ann.nlist"):
        run_smoke_from_config(config)


def test_invalid_index_nprobe_in_config_raises_cleanly() -> None:
    config = _config()
    config.ann = ANNConfig(backend="exact", metric="dot", nprobe=0)

    with pytest.raises(ValueError, match="ann.nprobe"):
        run_smoke_from_config(config)


def test_invalid_smoke_train_num_steps_raises_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke-train", "--num-steps", "0"])


def test_invalid_smoke_train_lr_raises_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke-train", "--lr", "0"])


def test_invalid_smoke_train_target_tau_raises_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke-train", "--target-tau", "1.5"])


def test_invalid_smoke_train_target_update_every_raises_cleanly() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke-train", "--target-update-every", "0"])


def test_cli_alternating_smoke_without_encoder_on_critic_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke-train",
            "--alternating-smoke",
            "--no-encoder-on-critic",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert "synthetic alternating smoke train only" in capsys.readouterr().out


def test_cli_alternating_smoke_accepts_assert_gradient_flow(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke-train",
            "--alternating-smoke",
            "--assert-gradient-flow",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
            "--num-steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert "synthetic alternating smoke train only" in capsys.readouterr().out


def test_cli_assert_gradient_flow_requires_alternating_smoke() -> None:
    with pytest.raises(SystemExit):
        main(["--smoke-train", "--assert-gradient-flow"])

    with pytest.raises(SystemExit):
        main(["--smoke-eval", "--assert-gradient-flow"])


def test_cli_split_backward_requires_train_synthetic_and_alternating() -> None:
    with pytest.raises(SystemExit):
        main(["--split-backward"])

    with pytest.raises(SystemExit):
        main(["--smoke-train", "--alternating-smoke", "--split-backward"])

    with pytest.raises(SystemExit):
        main(["--train-synthetic", "--split-backward"])


def test_summarize_group_norms_by_step_returns_finite_means() -> None:
    summary = summarize_group_norms_by_step(
        [
            {"encoder": 1.0, "critic": 2.0, "density": 3.0, "dual": 4.0},
            {"encoder": 3.0, "critic": 4.0, "density": 5.0, "dual": 6.0},
        ]
    )

    assert summary == {"encoder": 2.0, "critic": 3.0, "density": 4.0, "dual": 5.0}
    assert all(torch.isfinite(torch.tensor(value)) for value in summary.values())


def test_summarize_group_norms_by_step_missing_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Missing group"):
        summarize_group_norms_by_step([{"encoder": 1.0, "critic": 2.0, "density": 3.0}])


# ---------------------------------------------------------------------------
# Part F: --train-dwmql-synthetic CLI tests
# ---------------------------------------------------------------------------


def _retrieval_alignment_base_args(num_steps: int = 3) -> list[str]:
    return [
        "--train-dwmql-synthetic",
        "--seed", "7",
        "--batch-size", "4",
        "--num-transitions", "16",
        "--num-actions", "8",
        "--state-dim", "4",
        "--action-dim", "3",
        "--embedding-dim", "8",
        "--k", "3",
        "--gamma", "0.95",
        "--num-steps", str(num_steps),
        "--lr", "0.001",
    ]


def test_cli_train_dwmql_synthetic_runs_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(_retrieval_alignment_base_args())
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DW-MQL synthetic minimax training scaffold only" in output
    assert "num_steps: 3" in output
    assert "final_bellman_loss" in output
    assert "final_stationarity_loss" in output
    assert "final_ess" in output
    assert "benchmark_claim: false" in output
    assert "extragradient_used: false" in output


def test_cli_train_dwmql_synthetic_extragradient_mode(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([*_retrieval_alignment_base_args(num_steps=2), "--extragradient", "--extragradient-step-size", "0.01"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "extragradient_used: true" in output
    assert "benchmark_claim: false" in output


def test_cli_train_dwmql_synthetic_custom_weights(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([
        *_retrieval_alignment_base_args(num_steps=2),
        "--bellman-weight", "2.0",
        "--stationarity-weight", "0.5",
        "--density-regularization-weight", "0.001",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "bellman_weight: 2.0" in output
    assert "stationarity_weight: 0.5" in output
    assert "density_regularization_weight: 0.001" in output


def test_cli_train_dwmql_synthetic_writes_json_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "retrieval_alignment_train.json"

    exit_code = main([*_retrieval_alignment_base_args(num_steps=3), "--json-out", str(output_path)])
    capsys.readouterr()

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    _assert_json_report_metadata(report, "train_dwmql_synthetic")
    assert report["benchmark_claim"] is False
    assert report["summary"]["num_steps"] == 3
    assert report["summary"]["extragradient_used"] is False
    assert len(report["bellman_losses"]) == 3
    assert len(report["stationarity_losses"]) == 3
    assert len(report["ess_values"]) == 3


def test_cli_train_dwmql_synthetic_json_report_with_extragradient(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "retrieval_alignment_eg.json"

    exit_code = main([
        *_retrieval_alignment_base_args(num_steps=2),
        "--extragradient",
        "--extragradient-step-size", "0.01",
        "--json-out", str(output_path),
    ])
    capsys.readouterr()

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["extragradient_used"] is True
    assert report["summary"]["extragradient_groups"] == ["dual", "density"]


def test_cli_train_dwmql_synthetic_with_persistent_index(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([
        *_retrieval_alignment_base_args(num_steps=3),
        "--use-persistent-index",
        "--index-refresh-every", "2",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "persistent_index_used: true" in output
    assert "index_refresh_count" in output


def test_cli_train_dwmql_synthetic_mutually_exclusive_with_train_objective_synthetic() -> None:
    with pytest.raises(SystemExit):
        main(["--train-dwmql-synthetic", "--train-objective-synthetic"])


# ---------------------------------------------------------------------------
# New benchmark protocol CLI tests
# ---------------------------------------------------------------------------


def test_cli_run_movielens_chrono_protocol_runs_on_tiny_csv(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    lines = ["userId,movieId,rating,timestamp"]
    for i in range(12):
        uid = (i % 3) + 1
        mid = 10 + i
        ts = 100 * (i + 1)
        rating = 4.5 if i % 2 == 0 else 3.0
        lines.append(f"{uid},{mid},{rating},{ts}")
    ratings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    exit_code = main([
        "--run-movielens-chrono-protocol",
        "--data-path", str(ratings_path),
        "--output-dir", str(output_dir),
        "--num-steps", "2",
        "--batch-size", "4",
        "--k-values", "3,5",
    ])

    assert exit_code == 0


def test_cli_run_movielens_chrono_protocol_report_is_claim_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json
    ratings_path = tmp_path / "ratings.csv"
    lines = ["userId,movieId,rating,timestamp"]
    for i in range(12):
        uid = (i % 3) + 1
        mid = 10 + i
        ts = 100 * (i + 1)
        rating = 4.5 if i % 2 == 0 else 3.0
        lines.append(f"{uid},{mid},{rating},{ts}")
    ratings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    main([
        "--run-movielens-chrono-protocol",
        "--data-path", str(ratings_path),
        "--output-dir", str(output_dir),
        "--num-steps", "2",
        "--batch-size", "4",
        "--k-values", "3,5",
    ])
    output = capsys.readouterr().out

    assert "benchmark_claim: false" in output
    assert "scaffold_only: true" in output
    report_file = output_dir / "movielens_chrono_support_protocol.json"
    loaded = json.loads(report_file.read_text(encoding="utf-8"))
    assert loaded["benchmark_claim"] is False
    assert loaded["scaffold_only"] is True
    assert "chrono_support_protocol" in loaded
    assert loaded["leakage_check_passed"] is True


def test_cli_run_movielens_chrono_protocol_requires_data_path() -> None:
    with pytest.raises(SystemExit):
        main(["--run-movielens-chrono-protocol"])


def test_cli_run_d4rl_optional_protocol_fails_clearly_without_deps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "output"
    exit_code = main([
        "--run-d4rl-optional-protocol",
        "--dataset-name", "hopper-medium-v2",
        "--output-dir", str(output_dir),
        "--num-steps", "2",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "scaffold_only: true" in output
    assert "benchmark_claim: false" in output
    # When d4rl and minari are not installed, the status should show dependency_not_available
    # The report file should still exist and be valid
    report_file = output_dir / "d4rl_optional_support_protocol.json"
    if report_file.exists():
        import json
        loaded = json.loads(report_file.read_text(encoding="utf-8"))
        assert loaded["benchmark_claim"] is False
        assert loaded["scaffold_only"] is True


def test_cli_run_recsim_optional_protocol_mock_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "output"
    exit_code = main([
        "--run-recsim-optional-protocol",
        "--output-dir", str(output_dir),
        "--num-steps", "2",
        "--batch-size", "4",
        "--num-users", "4",
        "--num-items", "8",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "scaffold_only: true" in output
    assert "benchmark_claim: false" in output
    assert "not_implemented: True" in output


def test_cli_new_protocols_mutually_exclusive(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,100\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main([
            "--run-movielens-chrono-protocol",
            "--run-d4rl-optional-protocol",
            "--data-path", str(ratings_path),
        ])


def _make_tiny_ratings_csv(tmp_path: Path) -> Path:
    ratings_path = tmp_path / "ratings.csv"
    lines = ["userId,movieId,rating,timestamp"]
    for i in range(15):
        uid = (i % 3) + 1
        mid = 10 + i
        ts = 100 * (i + 1)
        rating = 4.5 if i % 2 == 0 else 3.0
        lines.append(f"{uid},{mid},{rating},{ts}")
    ratings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ratings_path


def test_cli_run_benchmark_protocol_movielens_is_claim_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    ratings_path = _make_tiny_ratings_csv(tmp_path)
    output_dir = tmp_path / "bench_output"

    exit_code = main([
        "--run-benchmark-protocol",
        "--routes", "movielens_chrono",
        "--seeds", "0",
        "--index-backend-values", "exact",
        "--index-refresh-values", "2",
        "--baselines", "random",
        "--data-path", str(ratings_path),
        "--output-dir", str(output_dir),
        "--num-steps", "2",
        "--batch-size", "4",
        "--k-values", "3",
        "--train-k", "3",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "benchmark_claim: false" in output
    assert "scaffold_only: true" in output

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["benchmark_claim"] is False
    assert loaded["scaffold_only"] is True
    assert loaded["num_runs"] >= 1

    md_path = output_dir / "benchmark_table.md"
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "benchmark_claim: false" in md_text


def test_cli_inspect_benchmark_protocol_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    summary_path = tmp_path / "summary.json"
    summary = {
        "protocol_name": "first_benchmark_protocol",
        "benchmark_claim": False,
        "scaffold_only": True,
        "routes": ["movielens_chrono"],
        "seeds": [0],
        "num_runs": 1,
        "grouped_statistics": {
            "benchmark_claim": False,
            "group_keys": ["route", "method"],
            "groups": [],
        },
        "limitations": ["scaffold only"],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    exit_code = main([
        "--inspect-benchmark-protocol",
        "--benchmark-summary-json", str(summary_path),
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "benchmark_claim" in output
    assert "first_benchmark_protocol" in output


def test_cli_inspect_benchmark_protocol_requires_summary_json() -> None:
    with pytest.raises(SystemExit):
        main(["--inspect-benchmark-protocol"])


def test_cli_run_benchmark_protocol_requires_data_path_for_movielens_route() -> None:
    with pytest.raises(SystemExit):
        main(["--run-benchmark-protocol", "--routes", "movielens_chrono"])


def test_cli_run_benchmark_protocol_config_with_cli_data_path_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--config with null data_path + --data-path CLI override must work (critical bug regression)."""
    import json

    ratings_path = _make_tiny_ratings_csv(tmp_path)
    output_dir = tmp_path / "bench_out"

    config_yaml = tmp_path / "bench.yaml"
    config_yaml.write_text(
        "protocol_name: test_bench\n"
        "routes:\n  - movielens_chrono\n"
        "seeds:\n  - 0\n"
        "k_values:\n  - 3\n"
        "train_k: 3\n"
        "index_backend_values:\n  - exact\n"
        "index_refresh_values:\n  - 2\n"
        "index_encoder_policies:\n  - online\n"
        "index_ema_tau_values:\n  - null\n"
        "baselines:\n  - random\n"
        "num_steps: 2\n"
        "batch_size: 4\n"
        "benchmark_claim: false\n",
        encoding="utf-8",
    )

    exit_code = main([
        "--run-benchmark-protocol",
        "--config", str(config_yaml),
        "--data-path", str(ratings_path),
        "--output-dir", str(output_dir),
    ])
    output = capsys.readouterr().out

    assert exit_code == 0, output
    assert "benchmark_claim: false" in output
    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["benchmark_claim"] is False


def test_cli_run_benchmark_protocol_wrong_csv_columns_fails(tmp_path: Path) -> None:
    """A CSV with wrong columns must raise a clear SystemExit before training."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("user_id,item_id,score\n1,10,5\n", encoding="utf-8")
    output_dir = tmp_path / "bench_out"
    with pytest.raises(SystemExit):
        main([
            "--run-benchmark-protocol",
            "--routes", "movielens_chrono",
            "--seeds", "0",
            "--data-path", str(bad_csv),
            "--output-dir", str(output_dir),
            "--num-steps", "2",
            "--k-values", "3",
            "--train-k", "3",
            "--index-refresh-values", "2",
        ])


def test_cli_run_benchmark_protocol_placeholder_baseline_fails(tmp_path: Path) -> None:
    """Placeholder baseline names must raise a clear SystemExit at validation time."""
    ratings_path = _make_tiny_ratings_csv(tmp_path)
    output_dir = tmp_path / "bench_out"
    with pytest.raises(SystemExit):
        main([
            "--run-benchmark-protocol",
            "--routes", "movielens_chrono",
            "--seeds", "0",
            "--data-path", str(ratings_path),
            "--output-dir", str(output_dir),
            "--num-steps", "2",
            "--k-values", "3",
            "--train-k", "3",
            "--index-refresh-values", "2",
            "--baselines", "cql",
        ])


def test_cli_run_benchmark_protocol_prints_routes_and_seeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI summary must include routes, seeds, k_values, baselines, and num_runs."""
    ratings_path = _make_tiny_ratings_csv(tmp_path)
    output_dir = tmp_path / "bench_out"

    exit_code = main([
        "--run-benchmark-protocol",
        "--routes", "movielens_chrono",
        "--seeds", "0",
        "--index-backend-values", "exact",
        "--index-refresh-values", "2",
        "--baselines", "random",
        "--data-path", str(ratings_path),
        "--output-dir", str(output_dir),
        "--num-steps", "2",
        "--batch-size", "4",
        "--k-values", "3",
        "--train-k", "3",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "routes:" in output
    assert "seeds:" in output
    assert "k_values:" in output
    assert "baselines:" in output
    assert "num_runs:" in output


def test_cli_benchmark_protocol_accepts_stability_controls(
    tmp_path: Path,
) -> None:
    """Benchmark protocol must accept stability control CLI args without error."""
    ratings_path = _make_tiny_ratings_csv(tmp_path)
    output_dir = tmp_path / "bench_stability"

    exit_code = main([
        "--run-benchmark-protocol",
        "--routes", "movielens_chrono",
        "--seeds", "0",
        "--index-backend-values", "exact",
        "--index-refresh-values", "2",
        "--baselines", "random",
        "--data-path", str(ratings_path),
        "--output-dir", str(output_dir),
        "--num-steps", "2",
        "--batch-size", "4",
        "--k-values", "3",
        "--train-k", "3",
        "--max-grad-norm", "10.0",
        "--q-l2-regularization-weight", "0.001",
        "--q-value-clip", "20.0",
        "--reward-scale", "1.0",
    ])
    assert exit_code == 0


def test_benchmark_protocol_config_accepts_stability_fields() -> None:
    """BenchmarkProtocolConfig stability fields must parse and validate correctly."""
    from retrieval_alignment.config import BenchmarkProtocolConfig, validate_benchmark_protocol_config

    cfg = BenchmarkProtocolConfig(
        routes=("movielens_chrono",),
        seeds=(0,),
        data_path=".",
        max_grad_norm=5.0,
        q_l2_regularization_weight=0.01,
        q_value_clip=15.0,
        reward_scale=2.0,
        density_regularization_weight=0.05,
        stationarity_weight=0.5,
        bellman_weight=0.8,
    )
    assert cfg.max_grad_norm == 5.0
    assert cfg.q_l2_regularization_weight == 0.01
    assert cfg.q_value_clip == 15.0
    assert cfg.reward_scale == 2.0
    assert cfg.density_regularization_weight == 0.05
    assert cfg.stationarity_weight == 0.5
    assert cfg.bellman_weight == 0.8


def test_benchmark_protocol_config_stability_fields_validate_errors() -> None:
    """BenchmarkProtocolConfig stability fields must reject invalid values."""
    import pytest
    from retrieval_alignment.config import BenchmarkProtocolConfig, validate_benchmark_protocol_config

    with pytest.raises(ValueError, match="max_grad_norm"):
        validate_benchmark_protocol_config(
            BenchmarkProtocolConfig(routes=("movielens_chrono",), seeds=(0,), data_path=".", max_grad_norm=-1.0)
        )

    with pytest.raises(ValueError, match="q_l2_regularization_weight"):
        validate_benchmark_protocol_config(
            BenchmarkProtocolConfig(routes=("movielens_chrono",), seeds=(0,), data_path=".", q_l2_regularization_weight=-0.5)
        )

    with pytest.raises(ValueError, match="reward_scale"):
        validate_benchmark_protocol_config(
            BenchmarkProtocolConfig(routes=("movielens_chrono",), seeds=(0,), data_path=".", reward_scale=0.0)
        )
