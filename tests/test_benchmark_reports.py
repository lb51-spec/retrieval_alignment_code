"""Tests for unified external adapter scaffold reports."""

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

from retrieval_alignment.benchmarks import (  # noqa: E402
    EXTERNAL_ADAPTER_REPORT_SCHEMA_NAME,
    EXTERNAL_ADAPTER_REPORT_SCHEMA_VERSION,
    MockExternalBenchmarkTask,
    build_external_adapter_scaffold_report,
    inspect_external_adapter_scaffold_bundle,
    load_external_adapter_bundle_manifest,
    load_external_adapter_scaffold_report,
    run_external_adapter_scaffold_report_bundle,
    summarize_external_adapter_scaffold_report,
    write_external_adapter_scaffold_report,
)


def test_build_external_adapter_scaffold_report_mock_external() -> None:
    task = MockExternalBenchmarkTask(seed=0)

    report = build_external_adapter_scaffold_report(task, adapter_name="mock_external")

    assert report["schema_name"] == EXTERNAL_ADAPTER_REPORT_SCHEMA_NAME
    assert report["schema_version"] == EXTERNAL_ADAPTER_REPORT_SCHEMA_VERSION
    assert report["report_type"] == "external_adapter_scaffold"
    assert report["adapter_name"] == "mock_external"
    assert report["task_name"] == task.name
    assert report["benchmark_claim"] is False
    assert report["dataset_metadata"]["dataset_size"] == 6
    assert report["action_catalogue_metadata"]["num_actions"] == 4
    assert "contract" in report["evaluation_diagnostics"]


def test_write_external_adapter_scaffold_report_creates_valid_json(tmp_path: Path) -> None:
    report = build_external_adapter_scaffold_report(MockExternalBenchmarkTask(seed=0), "mock_external")
    output_path = tmp_path / "reports" / "mock.json"

    write_external_adapter_scaffold_report(report, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["schema_name"] == EXTERNAL_ADAPTER_REPORT_SCHEMA_NAME
    assert loaded["benchmark_claim"] is False

    with pytest.raises(ValueError, match="file"):
        write_external_adapter_scaffold_report(report, tmp_path)


def test_load_external_adapter_scaffold_report_validates_schema(tmp_path: Path) -> None:
    report = build_external_adapter_scaffold_report(MockExternalBenchmarkTask(seed=0), "mock_external")
    output_path = tmp_path / "mock.json"
    write_external_adapter_scaffold_report(report, output_path)

    loaded = load_external_adapter_scaffold_report(output_path)

    assert loaded["adapter_name"] == "mock_external"
    assert loaded["benchmark_claim"] is False


def test_load_external_adapter_scaffold_report_rejects_wrong_schema(tmp_path: Path) -> None:
    report = build_external_adapter_scaffold_report(MockExternalBenchmarkTask(seed=0), "mock_external")
    report["schema_name"] = "wrong"
    output_path = tmp_path / "wrong_schema.json"
    output_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_name"):
        load_external_adapter_scaffold_report(output_path)


def test_load_external_adapter_scaffold_report_rejects_wrong_report_type(tmp_path: Path) -> None:
    report = build_external_adapter_scaffold_report(MockExternalBenchmarkTask(seed=0), "mock_external")
    report["report_type"] = "wrong"
    output_path = tmp_path / "wrong_type.json"
    output_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="report_type"):
        load_external_adapter_scaffold_report(output_path)


def test_load_external_adapter_scaffold_report_rejects_benchmark_claim(tmp_path: Path) -> None:
    report = build_external_adapter_scaffold_report(MockExternalBenchmarkTask(seed=0), "mock_external")
    report["benchmark_claim"] = True
    output_path = tmp_path / "claim.json"
    output_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark_claim"):
        load_external_adapter_scaffold_report(output_path)


def test_load_external_adapter_scaffold_report_rejects_invalid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "bad.json"
    output_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid external adapter"):
        load_external_adapter_scaffold_report(output_path)


def test_load_external_adapter_scaffold_report_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_external_adapter_scaffold_report(tmp_path / "missing.json")


def test_summarize_external_adapter_scaffold_report_returns_expected_keys() -> None:
    report = build_external_adapter_scaffold_report(MockExternalBenchmarkTask(seed=0), "mock_external")

    summary = summarize_external_adapter_scaffold_report(report)

    assert summary["adapter_name"] == "mock_external"
    assert summary["task_name"] == "mock_external_debug"
    assert summary["task_type"] == "mock_external"
    assert summary["benchmark_claim"] is False
    assert summary["dataset_size"] == 6
    assert summary["num_actions"] == 4
    assert summary["num_transitions"] == 6


def test_run_external_adapter_scaffold_report_bundle_writes_core_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"

    result = run_external_adapter_scaffold_report_bundle(output_dir)

    assert result["benchmark_claim"] is False
    assert result["included_adapters"] == ["mock_external", "d4rl", "recsim"]
    assert len(result["report_files"]) == 3
    assert Path(result["manifest_file"]).exists()
    manifest = json.loads(Path(result["manifest_file"]).read_text(encoding="utf-8"))
    assert manifest["run_type"] == "external_adapter_scaffold_report_bundle"
    assert manifest["synthetic_only"] is False
    assert manifest["scaffold_only"] is True
    assert manifest["benchmark_claim"] is False
    for report_file in result["report_files"]:
        loaded = load_external_adapter_scaffold_report(report_file)
        assert loaded["benchmark_claim"] is False


def test_run_external_adapter_scaffold_report_bundle_optionally_writes_movielens(
    tmp_path: Path,
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,0\n2,11,3.0,1\n", encoding="utf-8")

    result = run_external_adapter_scaffold_report_bundle(
        tmp_path / "bundle",
        movielens_data_path=ratings_path,
    )

    assert result["included_adapters"] == ["mock_external", "d4rl", "recsim", "movielens"]
    assert len(result["report_files"]) == 4
    movielens_report = load_external_adapter_scaffold_report(tmp_path / "bundle" / "movielens_local_report.json")
    assert movielens_report["adapter_name"] == "movielens"


def test_run_external_adapter_scaffold_report_bundle_missing_movielens_path_fails(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        run_external_adapter_scaffold_report_bundle(
            tmp_path / "bundle",
            movielens_data_path=tmp_path / "missing.csv",
        )


def test_load_external_adapter_bundle_manifest_works_on_generated_bundle(tmp_path: Path) -> None:
    result = run_external_adapter_scaffold_report_bundle(tmp_path / "bundle")

    manifest = load_external_adapter_bundle_manifest(result["manifest_file"])

    assert manifest["run_type"] == "external_adapter_scaffold_report_bundle"
    assert manifest["scaffold_only"] is True
    assert manifest["benchmark_claim"] is False
    assert manifest["included_adapters"] == ["mock_external", "d4rl", "recsim"]


def test_load_external_adapter_bundle_manifest_rejects_benchmark_claim(tmp_path: Path) -> None:
    result = run_external_adapter_scaffold_report_bundle(tmp_path / "bundle")
    manifest_path = Path(result["manifest_file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_claim"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark_claim"):
        load_external_adapter_bundle_manifest(manifest_path)


def test_load_external_adapter_bundle_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid external adapter bundle"):
        load_external_adapter_bundle_manifest(manifest_path)


def test_inspect_external_adapter_scaffold_bundle_summarizes_reports(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,10,5.0,0\n2,11,3.0,1\n", encoding="utf-8")
    result = run_external_adapter_scaffold_report_bundle(tmp_path / "bundle", movielens_data_path=ratings_path)

    inspection = inspect_external_adapter_scaffold_bundle(result["manifest_file"])

    assert inspection["benchmark_claim"] is False
    assert inspection["scaffold_only"] is True
    assert inspection["report_count"] == 4
    assert [summary["adapter_name"] for summary in inspection["summaries"]] == [
        "mock_external",
        "d4rl",
        "recsim",
        "movielens",
    ]
    assert all("task_name" in summary for summary in inspection["summaries"])


def test_inspect_external_adapter_scaffold_bundle_broken_report_path_raises(
    tmp_path: Path,
) -> None:
    result = run_external_adapter_scaffold_report_bundle(tmp_path / "bundle")
    manifest_path = Path(result["manifest_file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report_files"][0] = "missing_report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        inspect_external_adapter_scaffold_bundle(manifest_path)
