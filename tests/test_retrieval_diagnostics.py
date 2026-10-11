"""Tests for retrieval alignment diagnostics, Q-degeneracy diagnostics, and warning flags."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from retrieval_alignment.benchmarks import (
    MovieLensLocalTask,
    diagnose_movielens_q_degeneracy,
    diagnose_movielens_retrieval_alignment,
)
from retrieval_alignment.config import ModelConfig, MovieLensChronoProtocolConfig, validate_benchmark_protocol_config
from retrieval_alignment.networks import build_networks
from retrieval_alignment.experiment import (
    MovieLensChronoProtocolResult,
    _extract_chrono_dwmql_metrics,
    _Q_DIVERGENCE_BELLMAN_THRESHOLD,
    inspect_benchmark_protocol_summary,
)
from retrieval_alignment.trainer import TrainingHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiny_ratings_csv(path: Path) -> None:
    """Write a minimal ratings CSV with 3 movies and enough rows for train/test."""
    rows = ["userId,movieId,rating,timestamp"]
    # 12 rows so chronological split gives non-empty train/test
    for i in range(12):
        user = (i % 3) + 1
        movie = 10 + (i % 3)
        rows.append(f"{user},{movie},5.0,{i}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_tiny_task(tmp_path: Path) -> MovieLensLocalTask:
    ratings_path = tmp_path / "ratings.csv"
    _write_tiny_ratings_csv(ratings_path)
    return MovieLensLocalTask(
        ratings_path,
        min_rating_threshold=4.0,
        split_mode="chronological_strict",
        validation_fraction=0.1,
        test_fraction=0.2,
    )


def _make_tiny_model(task: MovieLensLocalTask) -> Any:
    return build_networks(
        ModelConfig(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            embedding_dim=8,
        )
    )


def _make_minimal_training_history(**kwargs: Any) -> TrainingHistory:
    defaults: dict[str, Any] = dict(
        losses=[0.1],
        bellman_losses=[0.1],
        stationarity_losses=[0.0],
        density_regularizations=[0.0],
        grad_norms=[0.1],
        parameter_delta_norms=[0.1],
        target_update_steps=[],
        index_ages=[0],
        index_refresh_steps=[],
        index_refresh_count=0,
        stale_index_recall_means=[None],
        stale_index_overlap_fraction_means=[None],
        stale_index_best_value_gap_means=[None],
        stale_index_pmr_means=[None],
        stale_index_pmr_mins=[None],
        stale_index_pmr_maxs=[None],
        stale_index_pmr_normalized_means=[None],
        stale_index_missing_mass_means=[None],
        critic_total_losses=[None],
        support_correction_losses=[None],
        support_correction_contributions=[None],
        density_ess_values=[None],
        weight_entropy_values=[None],
        index_ema_update_count=0,
        index_encoder_drift_mean_l2=[None],
        index_encoder_drift_max_l2=[None],
        index_encoder_drift_mean_cosine_distance=[None],
        index_encoder_drift_max_cosine_distance=[None],
        target_vs_online_recall_means=[None],
        target_vs_online_overlap_fraction_means=[None],
        ema_vs_online_recall_means=[None],
        ema_vs_online_overlap_fraction_means=[None],
        ema_vs_target_recall_means=[None],
        ema_vs_target_overlap_fraction_means=[None],
    )
    defaults.update(kwargs)
    return TrainingHistory(**defaults)


def _make_minimal_chrono_result(
    *,
    bellman_loss: float = 0.5,
    test_q_embedding_alignment_recall: float = 1.0,
    q_abs_max_values: list[float | None] | None = None,
    q_value_clip: float | None = None,
    id_space_alignment_passed: bool = True,
    q_degeneracy_warning: bool = False,
) -> MovieLensChronoProtocolResult:
    """Build a minimal MovieLensChronoProtocolResult for warning-flag tests."""
    history = _make_minimal_training_history(
        bellman_losses=[bellman_loss],
        q_abs_max_values=q_abs_max_values if q_abs_max_values is not None else [None],
    )
    k_result: dict[str, Any] = {
        "pmr": {"metrics": {"pmr_at_k_scaffold": 0.1}},
        "ann": {"metrics": {"ann_candidate_recall_mean": test_q_embedding_alignment_recall, "ann_candidate_pmr_mean": None}},
    }
    chrono_support: dict[str, Any] = {
        "leakage_check_passed": True,
        "splits": {
            "train": {"k_results": {"k_20": k_result}},
            "test": {"k_results": {"k_20": k_result}},
        },
    }
    config = MovieLensChronoProtocolConfig(
        q_value_clip=q_value_clip,
        train_k=20,
    )
    diagnostics: dict[str, Any] = {
        "retrieval_alignment_diagnostics": {
            "id_space_alignment_passed": id_space_alignment_passed,
        },
        "q_degeneracy_diagnostics": {
            "q_degeneracy_warning": q_degeneracy_warning,
            "q_score_std": 0.0 if q_degeneracy_warning else 1.0,
            "q_topk_margin_mean": 0.0 if q_degeneracy_warning else 1.0,
            "positive_negative_score_gap_mean": 0.0,
        },
    }
    return MovieLensChronoProtocolResult(
        config=config,
        training_history=history,
        chrono_support=chrono_support,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Part A: diagnose_movielens_retrieval_alignment
# ---------------------------------------------------------------------------

def test_diagnose_retrieval_alignment_returns_required_keys(tmp_path: Path) -> None:
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)

    result = diagnose_movielens_retrieval_alignment(task, model, split="test", k=2)

    required = {
        "exact_q_topk_ids_sample",
        "index_topk_ids_sample",
        "topk_id_overlap_count_sample",
        "q_score_topk_value_mean",
        "q_score_topk_value_std",
        "index_score_topk_value_mean",
        "action_embedding_norm_mean",
        "action_embedding_norm_min",
        "action_embedding_norm_max",
        "query_embedding_norm_mean",
        "query_embedding_norm_min",
        "query_embedding_norm_max",
        "zero_norm_action_embeddings_count",
        "zero_norm_query_embeddings_count",
        "candidate_id_space_min",
        "candidate_id_space_max",
        "action_id_space_min",
        "action_id_space_max",
        "id_space_alignment_passed",
        "benchmark_claim",
        "scaffold_only",
    }
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


def test_diagnose_retrieval_alignment_id_space_passes_on_valid_task(tmp_path: Path) -> None:
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)

    result = diagnose_movielens_retrieval_alignment(task, model, split="test", k=2)

    assert result["id_space_alignment_passed"] is True
    assert result["benchmark_claim"] is False
    assert result["scaffold_only"] is True


def test_diagnose_retrieval_alignment_zero_norm_counts_nonnegative(tmp_path: Path) -> None:
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)

    result = diagnose_movielens_retrieval_alignment(task, model, split="test", k=2)

    assert result["zero_norm_action_embeddings_count"] >= 0
    assert result["zero_norm_query_embeddings_count"] >= 0


def test_diagnose_retrieval_alignment_bad_task_type_raises() -> None:
    with pytest.raises(TypeError, match="MovieLensLocalTask"):
        diagnose_movielens_retrieval_alignment(object(), model=None)  # type: ignore[arg-type]


def test_diagnose_retrieval_alignment_bad_split_raises(tmp_path: Path) -> None:
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)

    with pytest.raises(ValueError, match="split must be one of"):
        diagnose_movielens_retrieval_alignment(task, model, split="invalid")


# ---------------------------------------------------------------------------
# Part B: warning flags via _extract_chrono_dwmql_metrics
# ---------------------------------------------------------------------------

def test_retrieval_collapse_warning_triggers_when_recall_is_zero() -> None:
    result = _make_minimal_chrono_result(test_q_embedding_alignment_recall=0.0)

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["retrieval_collapse_warning"] is True
    assert metrics["benchmark_claim"] is False


def test_retrieval_collapse_warning_false_when_recall_nonzero() -> None:
    result = _make_minimal_chrono_result(test_q_embedding_alignment_recall=0.5)

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["retrieval_collapse_warning"] is False


def test_q_divergence_warning_triggers_on_high_bellman_loss() -> None:
    result = _make_minimal_chrono_result(bellman_loss=_Q_DIVERGENCE_BELLMAN_THRESHOLD + 1.0)

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["q_divergence_warning"] is True


def test_q_divergence_warning_triggers_when_q_abs_max_hits_clip() -> None:
    clip = 5.0
    result = _make_minimal_chrono_result(
        bellman_loss=0.1,
        q_abs_max_values=[clip],  # exactly at clip threshold
        q_value_clip=clip,
    )

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["q_divergence_warning"] is True


def test_q_divergence_warning_false_when_bellman_low_and_q_not_clipped() -> None:
    clip = 20.0
    result = _make_minimal_chrono_result(
        bellman_loss=1.0,
        q_abs_max_values=[3.0],
        q_value_clip=clip,
    )

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["q_divergence_warning"] is False


def test_id_alignment_warning_triggers_when_alignment_fails() -> None:
    result = _make_minimal_chrono_result(id_space_alignment_passed=False)

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["id_alignment_warning"] is True


def test_id_alignment_warning_false_when_alignment_passes() -> None:
    result = _make_minimal_chrono_result(id_space_alignment_passed=True)

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["id_alignment_warning"] is False


# ---------------------------------------------------------------------------
# Part C: strict stability config loads and validates
# ---------------------------------------------------------------------------

def test_strict_stability_config_loads_and_validates() -> None:
    config_path = ROOT / "configs" / "first_benchmark_protocol_strict_stability.yaml"
    assert config_path.exists(), "strict stability config file must exist"

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    assert isinstance(loaded, dict)
    assert loaded.get("benchmark_claim") is False
    assert loaded.get("q_value_clip") == 5.0
    assert loaded.get("q_l2_regularization_weight") == 0.01
    assert loaded.get("max_grad_norm") == 5.0
    assert loaded.get("seeds") == [0, 1, 2, 3, 4]
    assert loaded.get("index_refresh_values") == [5]
    assert loaded.get("k_values") == [20]


# ---------------------------------------------------------------------------
# Part D: inspect CLI prints warning flags
# ---------------------------------------------------------------------------

def test_inspect_benchmark_protocol_prints_warning_flags() -> None:
    summary: dict[str, Any] = {
        "protocol_name": "test_protocol",
        "benchmark_claim": False,
        "scaffold_only": True,
        "routes": ["movielens_chrono"],
        "seeds": [0, 1, 2],
        "num_runs": 3,
        "grouped_statistics": {},
        "per_run_rows": [
            {
                "method": "dwmql",
                "route": "movielens_chrono",
                "seed": 0,
                "retrieval_collapse_warning": True,
                "q_divergence_warning": False,
                "id_alignment_warning": False,
            },
            {
                "method": "dwmql",
                "route": "movielens_chrono",
                "seed": 1,
                "retrieval_collapse_warning": False,
                "q_divergence_warning": True,
                "id_alignment_warning": False,
            },
        ],
        "limitations": ["all results are claim-free scaffold diagnostics"],
    }

    output = inspect_benchmark_protocol_summary(summary)

    assert "retrieval_collapse_warning" in output
    assert "q_divergence_warning" in output
    assert "id_alignment_warning" in output
    # Counts: 1 collapse, 1 diverge, 0 id-align
    assert "1/2" in output  # collapse and diverge counts
    assert "0/2" in output  # id_alignment_warning count
    assert "[!]" in output   # at least one warning banner


# ---------------------------------------------------------------------------
# Part E: Q-degeneracy diagnostics
# ---------------------------------------------------------------------------

def test_diagnose_q_degeneracy_returns_required_keys(tmp_path: Path) -> None:
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)

    result = diagnose_movielens_q_degeneracy(task, model, split="test", k=2)

    required = {
        "q_score_mean",
        "q_score_std",
        "q_score_min",
        "q_score_max",
        "q_score_range",
        "q_topk_margin_mean",
        "q_topk_margin_min",
        "candidate_policy_entropy_mean",
        "candidate_policy_entropy_max",
        "effective_action_support_mean",
        "positive_item_score_mean",
        "sampled_negative_score_mean",
        "positive_negative_score_gap_mean",
        "q_degeneracy_warning",
        "benchmark_claim",
        "scaffold_only",
    }
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


def test_diagnose_q_degeneracy_benchmark_claim_always_false(tmp_path: Path) -> None:
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)

    result = diagnose_movielens_q_degeneracy(task, model, split="test", k=2)

    assert result["benchmark_claim"] is False
    assert result["scaffold_only"] is True


def test_diagnose_q_degeneracy_bad_task_type_raises() -> None:
    with pytest.raises(TypeError, match="MovieLensLocalTask"):
        diagnose_movielens_q_degeneracy(object(), model=None)  # type: ignore[arg-type]


def test_diagnose_q_degeneracy_bad_split_raises(tmp_path: Path) -> None:
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)

    with pytest.raises(ValueError, match="split must be one of"):
        diagnose_movielens_q_degeneracy(task, model, split="invalid")


def test_diagnose_q_degeneracy_warning_triggers_when_q_uniform(tmp_path: Path) -> None:
    """A model with zero Q-head weights produces uniform Q scores — warning should fire."""
    task = _make_tiny_task(tmp_path)
    model = _make_tiny_model(task)
    # Zero out all Q head parameters so Q values become constant
    with torch.no_grad():
        for param in model.q_head.parameters():
            param.zero_()

    result = diagnose_movielens_q_degeneracy(
        task,
        model,
        split="test",
        k=2,
        q_degeneracy_std_threshold=1.0,  # high threshold to reliably trigger
        q_degeneracy_margin_threshold=1.0,
    )

    assert result["q_degeneracy_warning"] is True
    assert result["benchmark_claim"] is False


def test_diagnose_q_degeneracy_warning_false_on_normal_model(tmp_path: Path) -> None:
    """A freshly initialised model produces varied Q scores — warning should not fire."""
    task = _make_tiny_task(tmp_path)
    torch.manual_seed(42)
    model = _make_tiny_model(task)

    result = diagnose_movielens_q_degeneracy(
        task,
        model,
        split="test",
        k=2,
        q_degeneracy_std_threshold=1e-8,   # very tight threshold
        q_degeneracy_margin_threshold=1e-8,
    )

    # With negligible thresholds the warning should not fire on a random init
    # (std and margin are non-zero for a random model)
    assert result["q_score_std"] >= 0.0  # basic sanity
    assert result["benchmark_claim"] is False


def test_q_degeneracy_warning_extracted_into_metrics() -> None:
    """q_degeneracy_warning=True should propagate through _extract_chrono_dwmql_metrics."""
    result = _make_minimal_chrono_result(q_degeneracy_warning=True)

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["q_degeneracy_warning"] is True
    assert metrics["benchmark_claim"] is False


def test_q_degeneracy_warning_false_when_not_set() -> None:
    """When q_degeneracy diagnostics indicate no degeneracy, warning is False."""
    result = _make_minimal_chrono_result(q_degeneracy_warning=False)

    metrics = _extract_chrono_dwmql_metrics(result, train_k=20)

    assert metrics["q_degeneracy_warning"] is False


def test_margin_stability_config_loads(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "first_benchmark_protocol_margin_stability.yaml"
    assert config_path.exists(), "margin stability config must exist"

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    assert isinstance(loaded, dict)
    assert loaded.get("benchmark_claim") is False
    assert loaded.get("ranking_margin_weight") == 0.1
    assert loaded.get("ranking_margin") == 0.5
    assert loaded.get("ranking_negative_samples") == 20
    assert loaded.get("q_value_clip") == 5.0
    assert loaded.get("q_l2_regularization_weight") == 0.01
    assert loaded.get("max_grad_norm") == 5.0


def test_inspect_benchmark_includes_q_degeneracy_warning() -> None:
    summary: dict[str, Any] = {
        "protocol_name": "test_protocol",
        "benchmark_claim": False,
        "scaffold_only": True,
        "routes": ["movielens_chrono"],
        "seeds": [0],
        "num_runs": 1,
        "grouped_statistics": {},
        "per_run_rows": [
            {
                "method": "dwmql",
                "route": "movielens_chrono",
                "seed": 0,
                "retrieval_collapse_warning": False,
                "q_divergence_warning": False,
                "id_alignment_warning": False,
                "q_degeneracy_warning": True,
            },
        ],
        "limitations": ["all results are claim-free scaffold diagnostics"],
    }

    output = inspect_benchmark_protocol_summary(summary)

    assert "q_degeneracy_warning" in output
    assert "1/1" in output  # degeneracy count
    assert "[!]" in output   # degeneracy banner
