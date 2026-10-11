"""Tests for real_data-facing held-out and retrieval evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.benchmarks import MovieLensAdapter  # noqa: E402
from retrieval_alignment.config import ModelConfig  # noqa: E402
from retrieval_alignment.networks import build_networks  # noqa: E402
from retrieval_alignment.real_data_evaluation import (  # noqa: E402
    RealDataEvaluationConfig,
    evaluate_real_data_metrics,
)


def _tiny_task_and_model(tmp_path: Path):
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,5.0,100\n"
        "1,11,3.0,101\n"
        "2,10,4.5,102\n"
        "2,12,2.0,103\n"
        "3,11,5.0,104\n"
        "3,12,5.0,105\n",
        encoding="utf-8",
    )
    task = MovieLensAdapter(
        data_path=ratings_path,
        split_mode="chronological_strict",
        validation_fraction=0.2,
        test_fraction=0.2,
    ).make_task()
    torch.manual_seed(0)
    model = build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=4,
            retrieval_projection=True,
        )
    )
    return task, model


def test_real_data_evaluator_reports_full_and_candidate_metrics(tmp_path: Path) -> None:
    task, model = _tiny_task_and_model(tmp_path)

    result = evaluate_real_data_metrics(
        task,
        model,
        RealDataEvaluationConfig(split="test", k=2, index_backend="exact", seed=7),
    )

    assert result["full_catalogue"]["num_eval_rows"] > 0
    assert "ndcg_at_k" in result["full_catalogue"]
    assert "candidate_ndcg_at_k" in result["deployed_candidates"]
    assert result["metadata"]["index_backend"] == "exact"
    assert result["metadata"]["action_id_hash"]


def test_fixed_reference_metrics_are_reported_separately(tmp_path: Path) -> None:
    task, model = _tiny_task_and_model(tmp_path)
    torch.manual_seed(11)
    reference_model = build_networks(model.config)

    result = evaluate_real_data_metrics(
        task,
        model,
        RealDataEvaluationConfig(split="test", k=2, index_backend="exact", seed=7),
        reference_model=reference_model,
    )

    assert "fixed_reference" in result
    assert "reference_pmr_at_k" in result["fixed_reference"]
    assert result["metadata"]["fixed_reference_used"] is True


def test_exact_and_hnsw_keep_catalogue_action_ids(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    task, model = _tiny_task_and_model(tmp_path)

    exact = evaluate_real_data_metrics(
        task,
        model,
        RealDataEvaluationConfig(split="test", k=2, index_backend="exact", seed=0),
    )
    hnsw = evaluate_real_data_metrics(
        task,
        model,
        RealDataEvaluationConfig(
            split="test",
            k=2,
            index_backend="faiss_hnsw",
            ef_search=64,
            seed=0,
        ),
    )

    assert exact["metadata"]["action_id_hash"] == hnsw["metadata"]["action_id_hash"]
    assert hnsw["metadata"]["index_parameters"]["ef_search"] == 64
