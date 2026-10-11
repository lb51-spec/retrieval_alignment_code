"""Import tests for the research implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_all_modules_import() -> None:
    modules = [
        "retrieval_alignment",
        "retrieval_alignment.config",
        "retrieval_alignment.datasets",
        "retrieval_alignment.density",
        "retrieval_alignment.networks",
        "retrieval_alignment.policies",
        "retrieval_alignment.pipeline",
        "retrieval_alignment.trainer",
        "retrieval_alignment.target_networks",
        "retrieval_alignment.losses",
        "retrieval_alignment.stationarity",
        "retrieval_alignment.ann_index",
        "retrieval_alignment.audits",
        "retrieval_alignment.synthetic",
        "retrieval_alignment.candidate_sets",
        "retrieval_alignment.metrics",
        "retrieval_alignment.train",
        "retrieval_alignment.evaluate",
        "retrieval_alignment.utils",
    ]

    for module in modules:
        importlib.import_module(module)
