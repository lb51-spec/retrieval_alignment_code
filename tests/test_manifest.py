"""Tests for run manifest helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.manifest import (  # noqa: E402
    MANIFEST_SCHEMA_NAME,
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    load_run_manifest,
    manifest_to_dict,
    write_run_manifest,
)


def test_write_and_load_run_manifest(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_type="synthetic_experiment",
        output_dir=str(tmp_path),
        report_files=["report.json"],
        checkpoint_files=["checkpoint.pt"],
        summary_files=[],
        metadata={"experiment_name": "unit"},
    )
    path = tmp_path / "manifest.json"

    write_run_manifest(manifest, path)
    loaded = load_run_manifest(path)

    assert path.exists()
    assert loaded["schema_name"] == MANIFEST_SCHEMA_NAME
    assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert loaded["synthetic_only"] is True
    assert loaded["benchmark_claim"] is False
    assert loaded["report_files"] == ["report.json"]
    assert loaded["metadata"]["experiment_name"] == "unit"


def test_manifest_to_dict_rejects_benchmark_claim_true(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_type="synthetic_experiment",
        output_dir=str(tmp_path),
        benchmark_claim=True,
    )

    with pytest.raises(ValueError, match="benchmark_claim"):
        manifest_to_dict(manifest)


def test_load_run_manifest_wrong_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = manifest_to_dict(RunManifest(run_type="synthetic_experiment", output_dir=str(tmp_path)))
    payload["schema_name"] = "bad"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_name"):
        load_run_manifest(path)


def test_load_run_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run_manifest(tmp_path / "missing.json")
