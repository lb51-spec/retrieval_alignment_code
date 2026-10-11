"""Run manifest helpers for synthetic DW-MQL scaffold outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_NAME = "dwmql_run_manifest"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(slots=True)
class RunManifest:
    """Machine-readable path manifest for synthetic runs and sweeps."""

    run_type: str
    output_dir: str
    schema_name: str = MANIFEST_SCHEMA_NAME
    schema_version: int = MANIFEST_SCHEMA_VERSION
    synthetic_only: bool = True
    benchmark_claim: bool = False
    created_at: str | None = None
    command: str | None = None
    config_path: str | None = None
    report_files: list[str] = field(default_factory=list)
    checkpoint_files: list[str] = field(default_factory=list)
    summary_files: list[str] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_manifest_dict(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_name") != MANIFEST_SCHEMA_NAME:
        raise ValueError("Manifest schema_name is unsupported.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Manifest schema_version is unsupported.")
    if manifest.get("synthetic_only") is not True:
        raise ValueError("Manifest synthetic_only must be true.")
    if manifest.get("benchmark_claim") is not False:
        raise ValueError("Manifest benchmark_claim must be false.")
    for field_name in ("report_files", "checkpoint_files", "summary_files", "extra_files"):
        if not isinstance(manifest.get(field_name), list):
            raise ValueError(f"Manifest {field_name} must be a list.")
    if not isinstance(manifest.get("output_dir"), str) or not manifest["output_dir"]:
        raise ValueError("Manifest output_dir must be a non-empty string.")
    if not isinstance(manifest.get("run_type"), str) or not manifest["run_type"]:
        raise ValueError("Manifest run_type must be a non-empty string.")
    return manifest


def manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    """Convert and validate a ``RunManifest`` as a JSON-safe dictionary."""

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest.")
    payload = asdict(manifest)
    if payload["created_at"] is None:
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
    return _validate_manifest_dict(payload)


def write_run_manifest(manifest: RunManifest, path: str | Path) -> None:
    """Write a run manifest JSON file, creating parents as needed."""

    manifest_path = Path(path)
    if manifest_path.exists() and manifest_path.is_dir():
        raise ValueError("Manifest path must be a file, not a directory.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_to_dict(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a run manifest JSON file."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in run manifest: {manifest_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Run manifest must contain a JSON object.")
    return _validate_manifest_dict(loaded)
