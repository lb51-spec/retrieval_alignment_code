"""Tests for real_data experiment artifact validation and aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import (  # noqa: E402
    MovieLensChronoProtocolConfig,
    component_identity_from_config,
    load_movielens_chrono_protocol_config,
)
from retrieval_alignment.real_data_experiments import (  # noqa: E402
    ArtifactRecord,
    aggregate_real_data_results,
    artifact_record_from_report,
    inventory_reports,
    run_real_data_experiment,
    validate_artifact_identity,
    write_real_data_tables,
)


def test_legacy_zero_regularizer_is_not_labelled_density_off() -> None:
    report = {
        "config": {"density_regularization_weight": 0.0},
        "training_history": {"diagnostics": {"retrieval_correction_mode": "aware"}},
    }

    record = artifact_record_from_report(report, Path("legacy.json"))

    assert record.component_identity["density_enabled"] is True
    assert "legacy_missing_explicit_component_identity" in record.warnings


def test_real_data_b2_only_config_has_true_no_density_identity(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings(ratings_path)
    source_path = ROOT / "configs" / "real_data_ml25m_b2_only.yaml"
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload["data_path"] = str(ratings_path)
    config_path = tmp_path / "real_data_ml25m_b2_only.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_movielens_chrono_protocol_config(config_path)

    identity = component_identity_from_config(config)

    assert identity["critic_method"] == "dqn"
    assert identity["density_enabled"] is False
    assert identity["retrieval_alignment_enabled"] is True


def test_artifact_seed_mismatch_is_invalid() -> None:
    errors = validate_artifact_identity(
        {"config": {"seed": 2}},
        {"seed": 1},
    )

    assert "seed mismatch: expected 1, found 2" in errors


def test_inventory_keeps_invalid_json_visible(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    invalid_path = tmp_path / "invalid.json"
    valid_path.write_text(
        json.dumps(
            {
                "config": {"seed": 0, "critic_method": "dqn"},
                "component_identity": {"density_enabled": False},
                "real_data_metrics": {"ndcg_at_k": 0.5},
            }
        ),
        encoding="utf-8",
    )
    invalid_path.write_text("{", encoding="utf-8")

    records = inventory_reports([valid_path, invalid_path])
    by_name = {record.path.name: record for record in records}

    assert len(records) == 2
    assert by_name["valid.json"].valid is True
    assert by_name["invalid.json"].valid is False
    assert by_name["invalid.json"].errors


def _record(
    method: str,
    seed: int,
    ndcg: float,
    dataset: str = "movielens",
) -> ArtifactRecord:
    return ArtifactRecord(
        path=Path(f"{method}-{seed}.json"),
        dataset=dataset,
        seed=seed,
        method=method,
        component_identity={},
        config_hash="same",
        checkpoint_path=None,
        valid=True,
        errors=[],
        warnings=[],
        metrics={"ndcg_at_k": ndcg},
    )


def _write_ratings(path: Path) -> None:
    path.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,5.0,100\n"
        "1,11,4.0,101\n"
        "1,12,2.0,102\n"
        "2,10,4.5,103\n"
        "2,11,3.0,104\n"
        "2,12,5.0,105\n"
        "3,10,2.0,106\n"
        "3,11,4.5,107\n"
        "3,12,5.0,108\n"
        "3,13,5.0,109\n",
        encoding="utf-8",
    )


def test_real_data_runner_writes_seed_scoped_output(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_ratings(ratings_path)
    config = MovieLensChronoProtocolConfig(
        data_path=str(ratings_path),
        output_dir=str(tmp_path / "native"),
        num_steps=3,
        batch_size=2,
        embedding_dim=4,
        k_values=(2,),
        train_k=2,
        max_eval_rows=1,
        device="cpu",
        critic_method="dqn",
        retrieval_correction_mode="no_density",
        retrieval_alignment_weight=0.1,
        save_model_checkpoint=True,
    )

    result = run_real_data_experiment(
        dataset="movielens",
        config=config,
        seeds=(0,),
        output_root=tmp_path / "real_data",
    )

    run = result["runs"][0]
    assert Path(run["report"]).parent.name == "seed_0"
    assert Path(run["real_data_evaluation"]).exists()
    assert run["status"] == "completed"


def test_aggregation_pairs_only_matching_valid_seeds() -> None:
    records = [
        _record("full", 0, ndcg=0.4),
        _record("full", 1, ndcg=0.5),
        _record("b2_only", 0, ndcg=0.3),
        _record("b2_only", 2, ndcg=0.2),
    ]

    result = aggregate_real_data_results(records, [("full", "b2_only")])
    comparison = result["paired_comparisons"][0]

    assert comparison["valid_paired_seeds"] == [0]
    assert comparison["mean_delta"]["ndcg_at_k"] == pytest.approx(0.1)
    assert comparison["paired_delta_ci"]["ndcg_at_k"]["n"] == 1


def test_aggregation_never_pairs_or_pools_across_datasets() -> None:
    records = [
        _record("b2_only", 0, ndcg=0.2, dataset="movielens"),
        _record("base", 0, ndcg=0.1, dataset="movielens"),
        _record("b2_only", 0, ndcg=0.8, dataset="kuairand"),
        _record("base", 0, ndcg=0.5, dataset="kuairand"),
    ]

    result = aggregate_real_data_results(records, [("b2_only", "base")])
    comparisons = {item["dataset"]: item for item in result["paired_comparisons"]}

    assert comparisons["movielens"]["mean_delta"]["ndcg_at_k"] == pytest.approx(0.1)
    assert comparisons["kuairand"]["mean_delta"]["ndcg_at_k"] == pytest.approx(0.3)


def test_real_data_tables_share_aggregate_rows(tmp_path: Path) -> None:
    summary = aggregate_real_data_results(
        [_record("full", 0, 0.4), _record("full", 1, 0.6)],
        [],
    )

    outputs = write_real_data_tables(summary, tmp_path)

    saved = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    csv_text = Path(outputs["csv"]).read_text(encoding="utf-8")
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert saved["grouped_metrics"] == summary["grouped_metrics"]
    assert "ndcg_at_k" in csv_text
    assert "ndcg_at_k" in markdown
