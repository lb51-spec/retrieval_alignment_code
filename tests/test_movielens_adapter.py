"""Tests for the dependency-free MovieLens local CSV scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.benchmarks import (  # noqa: E402
    MovieLensAdapter,
    MovieLensLocalTask,
    assert_benchmark_task_contract,
    build_movielens_scaffold_report,
    build_movielens_split_metadata,
    evaluate_movielens_ann_candidate_audit,
    evaluate_movielens_candidate_support_scaffold,
    evaluate_movielens_chrono_support_protocol,
    evaluate_movielens_pmr_scaffold,
    evaluate_movielens_ranking_scaffold,
    evaluate_movielens_scaffold_task,
    load_movielens_rows,
    score_movielens_catalogue_with_model,
    write_movielens_scaffold_report,
)
from retrieval_alignment.baselines import NearestActionBaseline
from retrieval_alignment.config import ModelConfig
from retrieval_alignment.networks import build_networks

import json


def _write_ratings_csv(path: Path) -> None:
    path.write_text(
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


def test_movielens_adapter_reads_tiny_local_csv(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    adapter = MovieLensAdapter(data_path=ratings_path)

    availability = adapter.is_available()
    task = adapter.make_task()

    assert availability.available is True
    assert isinstance(task, MovieLensLocalTask)
    assert assert_benchmark_task_contract(task) is True
    assert task.metadata["source"] == "movielens_local_csv"
    assert task.metadata["synthetic_only"] is False
    assert task.metadata["benchmark_claim"] is False
    assert task.metadata["scaffold_only"] is True


def test_movielens_reservoir_sampling_is_bounded_and_deterministic(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    lines = ["userId,movieId,rating,timestamp"]
    for index in range(100):
        lines.append(f"{index % 7},{10 + index},{float(index % 5)}, {1000 + index}")
    ratings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    first = load_movielens_rows(
        ratings_path,
        max_rows=12,
        sampling_mode="reservoir",
        seed=7,
    )
    second = load_movielens_rows(
        ratings_path,
        max_rows=12,
        sampling_mode="reservoir",
        seed=7,
    )

    assert len(first) == 12
    assert first == second
    assert {row["movie_id"] for row in first} != set(range(10, 22))


def test_movielens_local_task_dataset_bundle_non_empty(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)

    task = MovieLensLocalTask(ratings_path, min_rating_threshold=4.0)
    bundle = task.make_dataset_bundle()

    assert len(bundle.dataset) == 4
    assert bundle.action_catalogue.num_actions == 3
    assert bundle.action_catalogue.action_dim == 3
    assert set(bundle.action_catalogue.action_ids.tolist()) == {10, 11, 12}
    assert torch.equal(bundle.dataset.batch.rewards, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_movielens_adapter_missing_file_raises(tmp_path: Path) -> None:
    adapter = MovieLensAdapter(data_path=tmp_path / "missing.csv")

    assert adapter.is_available().available is False
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.make_task()


def test_movielens_local_task_missing_required_columns_raises(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating\n1,10,5.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        MovieLensLocalTask(ratings_path)


def test_movielens_local_task_empty_rows_raises(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no valid rows"):
        MovieLensLocalTask(ratings_path)


def test_movielens_local_task_no_pandas_dependency() -> None:
    source = (ROOT / "src" / "retrieval_alignment" / "benchmarks.py").read_text(encoding="utf-8")

    assert "import pandas" not in source
    assert "from pandas" not in source


def test_movielens_local_task_evaluation_placeholder_claim_free(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path, max_rows=2)

    result = task.evaluate_policy_or_model(model=None)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.metrics["num_transitions"] == 2
    assert result.metrics["num_actions"] == 2
    assert result.diagnostics["scaffold_only"] is True


def test_movielens_chronological_split_metadata_counts_rows(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )

    task = MovieLensLocalTask(
        ratings_path,
        split_mode="chronological",
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    assert task.metadata["split_mode"] == "chronological"
    assert task.metadata["num_train_rows"] == 6
    assert task.metadata["num_validation_rows"] == 2
    assert task.metadata["num_test_rows"] == 2
    assert len(task.make_dataset()) == 6
    assert assert_benchmark_task_contract(task) is True
    assert len(task.get_split_rows("train")) == 6
    assert len(task.get_split_rows("validation")) == 2
    assert len(task.get_split_rows("test")) == 2
    assert task.available_splits() == ("train", "validation", "test")


def test_movielens_split_mode_none_returns_only_train_rows(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)

    task = MovieLensLocalTask(ratings_path, split_mode="none")

    assert len(task.get_split_rows("train")) == 4
    assert task.get_split_rows("validation") == []
    assert task.get_split_rows("test") == []
    assert task.available_splits() == ("train",)
    with pytest.raises(ValueError, match="split"):
        task.get_split_rows("holdout")


def test_movielens_session_mode_creates_session_metadata(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "\n".join(
            [
                "userId,movieId,rating,timestamp",
                "1,10,5.0,0",
                "1,11,5.0,10",
                "1,12,5.0,100",
                "2,10,4.0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    task = MovieLensLocalTask(ratings_path, split_mode="session", session_gap_seconds=30)

    assert task.metadata["split_mode"] == "session"
    assert task.metadata["num_sessions"] == 3
    assert task.metadata["min_timestamp"] == 0.0
    assert task.metadata["max_timestamp"] == 100.0


def test_movielens_invalid_split_args_raise(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)

    with pytest.raises(ValueError, match="split_mode"):
        MovieLensLocalTask(ratings_path, split_mode="bad")
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        MovieLensLocalTask(ratings_path, validation_fraction=1.0)
    with pytest.raises(ValueError, match="< 1"):
        MovieLensLocalTask(ratings_path, validation_fraction=0.6, test_fraction=0.4)
    with pytest.raises(ValueError, match="session_gap_seconds"):
        MovieLensLocalTask(ratings_path, split_mode="session")


def test_movielens_scaffold_evaluation_metrics_are_finite_and_correct(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path, min_rating_threshold=4.0)

    result = evaluate_movielens_scaffold_task(task)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.metrics["num_users"] == 2
    assert result.metrics["num_items"] == 3
    assert result.metrics["num_interactions"] == 4
    assert result.metrics["positive_rate"] == pytest.approx(0.5)
    assert result.metrics["avg_user_interactions"] == pytest.approx(2.0)
    assert result.metrics["catalogue_coverage_observed"] == pytest.approx(1.0)
    assert all(torch.isfinite(torch.tensor(float(value))) for value in result.metrics.values())


def test_movielens_scaffold_evaluation_topk_shape_diagnostics(tmp_path: Path) -> None:
    class TrivialBaseline:
        def predict_action_scores(self, states, action_features):
            return states[:, :1] @ action_features[:, :1].T

    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path)

    result = evaluate_movielens_scaffold_task(task, model_or_baseline=TrivialBaseline(), k=2)

    assert result.diagnostics["score_shape"] == [4, 3]
    assert result.diagnostics["topk_shape"] == [4, 2]


def test_movielens_ranking_scaffold_no_model_returns_split_diagnostics(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    task = MovieLensLocalTask(ratings_path, split_mode="chronological", validation_fraction=0.2, test_fraction=0.2)

    result = evaluate_movielens_ranking_scaffold(task, split="test", k=3)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.metrics["num_eval_rows"] == 2
    assert result.metrics["hit_rate_at_k"] is None
    assert result.diagnostics["scoring_available"] is False


def test_movielens_ranking_scaffold_with_nearest_baseline_is_bounded(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    task = MovieLensLocalTask(ratings_path, split_mode="chronological", validation_fraction=0.2, test_fraction=0.2)
    baseline = NearestActionBaseline(seed=0).fit(
        task.make_dataset(),
        action_features=task.make_action_catalogue()[0],
        action_ids=task.make_action_catalogue()[1],
    )

    result = evaluate_movielens_ranking_scaffold(task, model_or_baseline=baseline, split="test", k=3)

    assert result.benchmark_claim is False
    assert 0.0 <= result.metrics["hit_rate_at_k"] <= 1.0
    assert 0.0 <= result.metrics["recall_at_k_scaffold"] <= 1.0
    assert result.metrics["mean_rank_of_positive"] >= 1.0


def test_movielens_ranking_scaffold_empty_split_returns_zero_count(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path, split_mode="none")

    result = evaluate_movielens_ranking_scaffold(task, split="test", k=2)

    assert result.metrics["num_eval_rows"] == 0
    assert result.metrics["hit_rate_at_k"] is None
    assert result.diagnostics["reason"] == "no positive held-out rows"


def test_movielens_candidate_support_scaffold_is_finite_and_bounded(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    task = MovieLensLocalTask(ratings_path, split_mode="chronological", validation_fraction=0.2, test_fraction=0.2)

    result = evaluate_movielens_candidate_support_scaffold(task, split="test", k=999)

    assert result.benchmark_claim is False
    assert result.metrics["k"] == task.num_actions
    assert 0.0 <= result.metrics["candidate_hit_rate_at_k"] <= 1.0
    assert 0.0 <= result.metrics["candidate_coverage_fraction"] <= 1.0
    assert torch.isfinite(torch.tensor(float(result.metrics["candidate_hit_rate_at_k"])))


def test_score_movielens_catalogue_with_model_returns_finite_scores(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path)
    rows = task.get_split_rows("train")
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    scores = score_movielens_catalogue_with_model(task, model, rows)

    assert scores.shape == (len(rows), task.num_actions)
    assert torch.all(torch.isfinite(scores))


def test_movielens_pmr_scaffold_is_claim_free_and_bounded(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    task = MovieLensLocalTask(ratings_path, split_mode="chronological", validation_fraction=0.2, test_fraction=0.2)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_movielens_pmr_scaffold(task, model, split="test", k=3, temperature=1.0)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.diagnostics["scaffold_only"] is True
    assert 0.0 <= result.metrics["pmr_at_k_scaffold"] <= 1.0
    assert 0.0 <= result.metrics["missing_policy_mass_mean"] <= 1.0
    assert 0.0 <= result.metrics["positive_item_in_topk_rate"] <= 1.0
    assert result.metrics["positive_item_rank_mean"] >= 1.0


def test_movielens_pmr_scaffold_empty_split_returns_zero_count(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path, split_mode="none")
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_movielens_pmr_scaffold(task, model, split="test", k=2)

    assert result.metrics["num_eval_rows"] == 0
    assert result.metrics["pmr_at_k_scaffold"] is None
    assert result.diagnostics["reason"] == "split has no rows"


def test_movielens_ann_candidate_audit_exact_backend_is_bounded(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    task = MovieLensLocalTask(ratings_path, split_mode="chronological", validation_fraction=0.2, test_fraction=0.2)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_movielens_ann_candidate_audit(task, model, split="test", k=3, index_backend="exact")

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.diagnostics["scaffold_only"] is True
    assert result.metrics["index_backend"] == "exact"
    assert result.metrics["index_metric"] == "dot"
    assert 0.0 <= result.metrics["ann_candidate_recall_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_overlap_fraction_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_pmr_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_pmr_min"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_pmr_max"] <= 1.0
    assert torch.isfinite(torch.tensor(float(result.metrics["ann_candidate_best_value_gap_mean"])))


def test_movielens_ann_candidate_audit_empty_split_returns_zero_count(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path, split_mode="none")
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_movielens_ann_candidate_audit(task, model, split="test", k=2)

    assert result.metrics["num_eval_rows"] == 0
    assert result.metrics["ann_candidate_recall_mean"] is None
    assert result.diagnostics["reason"] == "split has no rows"


def test_movielens_ann_candidate_audit_faiss_missing_or_runs(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    task = MovieLensLocalTask(ratings_path, split_mode="chronological", validation_fraction=0.2, test_fraction=0.2)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))
    try:
        __import__("faiss")
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match="FAISS"):
            evaluate_movielens_ann_candidate_audit(task, model, split="test", k=2, index_backend="faiss")
    else:
        result = evaluate_movielens_ann_candidate_audit(task, model, split="test", k=2, index_backend="faiss")
        assert result.metrics["index_backend"] == "faiss"
        assert 0.0 <= result.metrics["ann_candidate_recall_mean"] <= 1.0


def test_build_movielens_scaffold_report_required_keys(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    task = MovieLensLocalTask(ratings_path, split_mode="chronological")

    report = build_movielens_scaffold_report(task)

    assert report["schema_name"] == "dwmql_movielens_scaffold_report"
    assert report["schema_version"] == 1
    assert report["report_type"] == "movielens_local_scaffold"
    assert report["adapter_name"] == "movielens"
    assert report["synthetic_only"] is False
    assert report["benchmark_claim"] is False
    assert report["scaffold_only"] is True
    assert report["dataset_metadata"]["num_users"] == 2
    assert report["dataset_metadata"]["num_items"] == 3
    assert report["dataset_metadata"]["split_mode"] == "chronological"
    assert report["action_catalogue_metadata"]["num_actions"] == 3
    assert "positive_rate" in report["scaffold_metrics"]
    assert "no training" in report["limitations"]


def test_write_movielens_scaffold_report_creates_json(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings_csv(ratings_path)
    report = build_movielens_scaffold_report(MovieLensLocalTask(ratings_path))
    output_path = tmp_path / "reports" / "movielens.json"

    write_movielens_scaffold_report(report, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["benchmark_claim"] is False
    assert loaded["scaffold_only"] is True
    with pytest.raises(ValueError, match="file"):
        write_movielens_scaffold_report(report, tmp_path)


# ---------------------------------------------------------------------------
# chronological_strict split tests
# ---------------------------------------------------------------------------


def _make_rows_for_strict_split() -> list[dict]:
    return [
        {"user_id": 1, "movie_id": 10, "rating": 5.0, "timestamp": 100},
        {"user_id": 2, "movie_id": 11, "rating": 4.0, "timestamp": 200},
        {"user_id": 1, "movie_id": 12, "rating": 3.0, "timestamp": 300},
        {"user_id": 2, "movie_id": 13, "rating": 5.0, "timestamp": 400},
        {"user_id": 1, "movie_id": 14, "rating": 4.0, "timestamp": 500},
        {"user_id": 2, "movie_id": 15, "rating": 2.0, "timestamp": 600},
        {"user_id": 1, "movie_id": 16, "rating": 4.5, "timestamp": 700},
        {"user_id": 2, "movie_id": 17, "rating": 1.0, "timestamp": 800},
        {"user_id": 1, "movie_id": 18, "rating": 5.0, "timestamp": 900},
        {"user_id": 2, "movie_id": 19, "rating": 4.0, "timestamp": 1000},
    ]


def test_chronological_strict_split_leakage_check_passes() -> None:
    rows = _make_rows_for_strict_split()
    meta = build_movielens_split_metadata(
        rows,
        split_mode="chronological_strict",
        validation_fraction=0.1,
        test_fraction=0.2,
    )
    assert meta["split_mode"] == "chronological_strict"
    assert meta["leakage_check_passed"] is True
    assert meta["num_train_rows"] + meta["num_validation_rows"] + meta["num_test_rows"] == len(rows)


def test_chronological_strict_split_train_excludes_held_out_rows() -> None:
    rows = _make_rows_for_strict_split()
    build_movielens_split_metadata(
        rows,
        split_mode="chronological_strict",
        validation_fraction=0.1,
        test_fraction=0.2,
    )
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    assert len(train_rows) > 0
    assert len(test_rows) > 0
    max_train_ts = max(r["timestamp"] for r in train_rows)
    min_test_ts = min(r["timestamp"] for r in test_rows)
    assert max_train_ts <= min_test_ts


def test_chronological_strict_split_metadata_fields() -> None:
    rows = _make_rows_for_strict_split()
    meta = build_movielens_split_metadata(
        rows,
        split_mode="chronological_strict",
        validation_fraction=0.1,
        test_fraction=0.2,
    )
    for key in ("leakage_check_passed", "min_train_timestamp", "max_train_timestamp", "min_test_timestamp", "max_test_timestamp"):
        assert key in meta


def test_chronological_strict_split_via_adapter(tmp_path: Path) -> None:
    path = tmp_path / "ratings.csv"
    lines = ["userId,movieId,rating,timestamp"]
    for i, (uid, mid, ts) in enumerate([(1, 10, 100), (2, 11, 200), (1, 12, 300), (2, 13, 400), (1, 14, 500), (2, 15, 600)]):
        lines.append(f"{uid},{mid},{4.5 if i % 2 == 0 else 3.0},{ts}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    adapter = MovieLensAdapter(
        data_path=path,
        split_mode="chronological_strict",
        validation_fraction=0.1,
        test_fraction=0.2,
    )
    task = adapter.make_task()
    meta = task.metadata
    assert meta.get("leakage_check_passed") is True
    assert meta.get("split_mode") == "chronological_strict"


def test_evaluate_movielens_chrono_support_protocol_returns_correct_shape(tmp_path: Path) -> None:
    path = tmp_path / "ratings.csv"
    lines = ["userId,movieId,rating,timestamp"]
    for i in range(12):
        uid = (i % 3) + 1
        mid = 10 + i
        ts = 100 * (i + 1)
        rating = 4.5 if i % 2 == 0 else 3.0
        lines.append(f"{uid},{mid},{rating},{ts}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    task = MovieLensAdapter(
        data_path=path,
        split_mode="chronological_strict",
        validation_fraction=0.1,
        test_fraction=0.2,
    ).make_task()
    model = build_networks(ModelConfig(state_dim=3, action_dim=3, embedding_dim=4))
    result = evaluate_movielens_chrono_support_protocol(
        task,
        model,
        k_values=(3, 5),
        temperature=1.0,
        index_backend="exact",
        index_metric="dot",
    )
    assert result["benchmark_claim"] is False
    assert result["scaffold_only"] is True
    assert result["split_mode"] == "chronological_strict"
    assert result["leakage_check_passed"] is True
    assert "train" in result["splits"]
    assert "test" in result["splits"]
    for split in ("train", "test"):
        for k_key in ("k_3", "k_5"):
            assert k_key in result["splits"][split]["k_results"]
            summary = result["splits"][split]["k_results"][k_key]["summary"]
            assert "hit_rate_at_k" in summary
            assert "pmr_at_k_scaffold" in summary


def test_evaluate_movielens_chrono_support_protocol_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        evaluate_movielens_chrono_support_protocol("not_a_task", None)  # type: ignore[arg-type]


def test_evaluate_movielens_chrono_support_protocol_rejects_empty_k_values(tmp_path: Path) -> None:
    path = tmp_path / "ratings.csv"
    _write_ratings_csv(path)
    task = MovieLensAdapter(data_path=path, split_mode="chronological_strict").make_task()
    model = build_networks(ModelConfig(state_dim=3, action_dim=3, embedding_dim=4))
    with pytest.raises(ValueError, match="k_values"):
        evaluate_movielens_chrono_support_protocol(task, model, k_values=())
