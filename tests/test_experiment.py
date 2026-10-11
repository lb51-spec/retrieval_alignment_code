"""Tests for the synthetic benchmark experiment scaffold."""

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

from retrieval_alignment.config import (
    D4RLMockExperimentConfig,
    ExperimentConfig,
    KuaiRandChronoProtocolConfig,
    MovieLensChronoProtocolConfig,
    MovieLensLocalExperimentConfig,
    RecSimMockExperimentConfig,
    default_smoke_profile,
    experiment_config_from_smoke_profile,
    load_d4rl_mock_experiment_config,
    load_experiment_config_yaml,
    load_movielens_local_experiment_config,
    load_recsim_mock_experiment_config,
    validate_d4rl_mock_experiment_config,
    validate_experiment_config,
    validate_movielens_local_experiment_config,
    validate_recsim_mock_experiment_config,
)
from retrieval_alignment.experiment import (
    D4RLMockExperimentResult,
    ExternalSupportProtocolConfig,
    ExperimentResult,
    MovieLensLocalExperimentResult,
    RecSimMockExperimentResult,
    inspect_external_scaffold_experiment_bundle,
    load_external_support_protocol_config,
    load_external_support_protocol_summary,
    load_external_scaffold_experiment_bundle_manifest,
    load_external_scaffold_experiment_report,
    make_d4rl_mock_arrays,
    run_external_scaffold_experiment_bundle,
    run_external_support_protocol,
    run_d4rl_mock_experiment,
    run_kuairand_chrono_protocol,
    run_movielens_local_experiment,
    run_movielens_chrono_protocol,
    run_recsim_mock_experiment,
    run_synthetic_experiment,
    summarize_critic_steps,
    summarize_external_scaffold_experiment_report,
    summarize_external_support_protocol,
    validate_external_support_protocol_config,
)
from retrieval_alignment.manifest import load_run_manifest


def _write_movielens_ratings(path: Path) -> None:
    path.write_text(
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


def _write_kuairand_fixture(data_dir: Path) -> None:
    data_dir.mkdir()
    header = "user_id,video_id,time_ms,is_click\n"
    rows = [
        f"{index % 3 + 1},{index % 4 + 10},{100 + index},{int(index % 2 == 1)}"
        for index in range(12)
    ]
    (data_dir / "log_standard_4_08_to_4_21_pure.csv").write_text(
        header + "\n".join(rows),
        encoding="utf-8",
    )
    (data_dir / "log_standard_4_22_to_5_08_pure.csv").write_text(
        header,
        encoding="utf-8",
    )
    (data_dir / "video_features_basic_pure.csv").write_text(
        "video_id,video_duration,server_width,server_height,music_type,tag,upload_type\n"
        "10,5000,720,1280,1,a,ShortImport\n"
        "11,6000,720,1280,2,b,ShortImport\n"
        "12,7000,720,1280,3,c,LongImport\n"
        "13,8000,720,1280,4,d,LongImport\n",
        encoding="utf-8",
    )


def test_critic_summary_ignores_trailing_density_step() -> None:
    history = {
        "diagnostics": {
            "objective_names": ["dual", "density", "critic", "dual", "density"],
        },
        "losses": [7.0, 0.0, 3.0, 8.0, 0.0],
        "critic_total_losses": [None, None, 2.5, None, None],
        "support_weighted_bellman_losses": [None, None, 2.0, None, None],
        "bellman_losses": [7.0, 0.0, 1.5, 8.0, 0.0],
        "grad_norms": [1.0, 0.0, 5.0, 1.0, 0.0],
    }

    result = summarize_critic_steps(history, max_grad_norm=5.0)

    assert result["num_critic_updates"] == 1
    assert result["final_critic_optimized_loss"] == pytest.approx(3.0)
    assert result["final_critic_total_loss"] == pytest.approx(2.5)
    assert result["final_support_weighted_bellman_loss"] == pytest.approx(2.0)
    assert result["final_raw_bellman_loss"] == pytest.approx(1.5)
    assert result["critic_grad_clip_fraction"] == pytest.approx(1.0)


def test_movielens_chrono_report_records_forwarded_component_identity(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_movielens_ratings(ratings_path)
    result = run_movielens_chrono_protocol(
        MovieLensChronoProtocolConfig(
            data_path=str(ratings_path),
            output_dir=str(tmp_path / "run"),
            num_steps=3,
            batch_size=2,
            embedding_dim=4,
            k_values=(2,),
            train_k=2,
            critic_method="dqn",
            retrieval_correction_mode="no_density",
        )
    )
    report = json.loads(Path(result.output_files["report"]).read_text(encoding="utf-8"))

    assert report["component_identity"]["critic_method"] == "dqn"
    assert report["component_identity"]["density_enabled"] is False
    assert report["training_history"]["diagnostics"]["retrieval_correction_mode"] == "no_density"
    assert report["training_history_summary"]["final_raw_bellman_loss"] == pytest.approx(
        report["training_history_summary"]["final_support_weighted_bellman_loss"]
    )


def test_kuairand_chrono_checkpoint_records_catalogue_and_component_identity(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "kuairand"
    _write_kuairand_fixture(data_dir)

    result = run_kuairand_chrono_protocol(
        KuaiRandChronoProtocolConfig(
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "run"),
            num_steps=3,
            batch_size=2,
            embedding_dim=4,
            k_values=(2,),
            train_k=2,
            critic_method="dqn",
            retrieval_correction_mode="no_density",
            save_model_checkpoint=True,
        )
    )

    checkpoint_path = Path(result.output_files["model_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(result.output_files["manifest"]).read_text(encoding="utf-8"))

    assert checkpoint["dataset_name"] == "kuairand_pure_local"
    assert checkpoint["catalogue_size"] == 4
    assert len(checkpoint["action_id_hash"]) == 64
    assert checkpoint["component_identity"]["density_enabled"] is False
    assert str(checkpoint_path) in manifest["checkpoint_files"]


def _small_experiment_config(tmp_path: Path) -> ExperimentConfig:
    profile = replace(
        default_smoke_profile(),
        seed=11,
        batch_size=4,
        num_transitions=16,
        num_actions=8,
        state_dim=4,
        action_dim=3,
        embedding_dim=8,
        k=3,
        num_steps=2,
        target_tau=0.05,
        use_persistent_index=True,
        index_refresh_every=2,
        index_encoder_policy="ema",
        index_ema_tau=0.05,
    )
    config = experiment_config_from_smoke_profile(profile, experiment_name="unit_experiment")
    config.output_dir = str(tmp_path / "experiment")
    return validate_experiment_config(config)


def test_experiment_config_from_smoke_profile() -> None:
    profile = replace(default_smoke_profile(), num_steps=3, batch_size=4, num_actions=8, k=3)

    config = experiment_config_from_smoke_profile(profile, experiment_name="from_smoke")

    assert config.experiment_name == "from_smoke"
    assert config.num_steps == 3
    assert config.batch_size == 4
    assert config.k == 3
    assert config.synthetic_only is True


def test_validate_experiment_config_rejects_invalid_values(tmp_path: Path) -> None:
    config = _small_experiment_config(tmp_path)

    with pytest.raises(ValueError, match="num_steps"):
        validate_experiment_config(replace(config, num_steps=0))

    with pytest.raises(ValueError, match="gamma"):
        validate_experiment_config(replace(config, gamma=1.5))

    with pytest.raises(ValueError, match="synthetic_only"):
        validate_experiment_config(replace(config, synthetic_only=False))

    with pytest.raises(ValueError, match="eval_num_batches"):
        validate_experiment_config(replace(config, eval_num_batches=0))

    with pytest.raises(ValueError, match="eval_batch_size"):
        validate_experiment_config(replace(config, eval_batch_size=0))

    assert validate_experiment_config(replace(config, baseline_names=())).baseline_names == ()
    assert validate_experiment_config(replace(config, baseline_names=("random", "nearest"))).baseline_names == (
        "random",
        "nearest",
    )
    with pytest.raises(ValueError, match="Unknown baseline"):
        validate_experiment_config(replace(config, baseline_names=("cql",)))


def test_load_experiment_config_yaml_loads_benchmark_config() -> None:
    config = load_experiment_config_yaml(ROOT / "configs" / "benchmark_synthetic.yaml")

    assert config.experiment_name == "synthetic_debug"
    assert config.synthetic_only is True
    assert config.index_backend == "exact"
    assert config.index_encoder_policy == "ema"


def test_load_experiment_config_yaml_loads_baseline_names(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: baseline_yaml",
                "num_steps: 1",
                "batch_size: 4",
                "num_transitions: 16",
                "num_actions: 8",
                "state_dim: 4",
                "action_dim: 3",
                "embedding_dim: 8",
                "k: 3",
                "gamma: 0.99",
                "index_encoder_policy: online",
                "baseline_names:",
                "  - random",
                "  - nearest",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config_yaml(config_path)

    assert config.baseline_names == ("random", "nearest")


def test_movielens_local_experiment_config_validates(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_movielens_ratings(ratings_path)
    config = MovieLensLocalExperimentConfig(
        data_path=str(ratings_path),
        output_dir=str(tmp_path / "movielens"),
        num_steps=1,
        batch_size=2,
        k=2,
        baseline_names=("random", "nearest"),
    )

    validated = validate_movielens_local_experiment_config(config)

    assert validated.benchmark_claim is False
    assert validated.scaffold_only is True
    assert validated.baseline_names == ("random", "nearest")


def test_movielens_local_experiment_config_invalid_data_path_raises(tmp_path: Path) -> None:
    config = MovieLensLocalExperimentConfig(data_path=str(tmp_path / "missing.csv"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_movielens_local_experiment_config(config)


def test_load_movielens_local_experiment_config_allows_placeholder_path() -> None:
    config = load_movielens_local_experiment_config(ROOT / "configs" / "movielens_local_experiment.yaml")

    assert config.experiment_name == "movielens_local_scaffold_debug"
    assert config.data_path == "path/to/ratings.csv"
    assert config.benchmark_claim is False
    assert config.scaffold_only is True
    assert config.baseline_names == ("random", "nearest")


def test_d4rl_mock_experiment_config_validates(tmp_path: Path) -> None:
    config = D4RLMockExperimentConfig(
        output_dir=str(tmp_path / "d4rl"),
        num_steps=2,
        batch_size=4,
        k=3,
        num_transitions=16,
        state_dim=4,
        action_dim=3,
        codebook_size=8,
    )

    validated = validate_d4rl_mock_experiment_config(config)

    assert validated.benchmark_claim is False
    assert validated.scaffold_only is True
    assert validated.codebook_size == 8


def test_d4rl_mock_experiment_config_invalid_values_raise(tmp_path: Path) -> None:
    valid = D4RLMockExperimentConfig(output_dir=str(tmp_path / "d4rl"))

    with pytest.raises(ValueError, match="num_steps"):
        validate_d4rl_mock_experiment_config(replace(valid, num_steps=0))

    with pytest.raises(ValueError, match="codebook_size"):
        validate_d4rl_mock_experiment_config(replace(valid, codebook_size=0))

    with pytest.raises(ValueError, match="k"):
        validate_d4rl_mock_experiment_config(replace(valid, k=valid.codebook_size + 1))

    with pytest.raises(ValueError, match="benchmark_claim"):
        validate_d4rl_mock_experiment_config(replace(valid, benchmark_claim=True))


def test_load_d4rl_mock_experiment_config_loads_yaml() -> None:
    config = load_d4rl_mock_experiment_config(ROOT / "configs" / "d4rl_mock_experiment.yaml")

    assert config.experiment_name == "d4rl_mock_large_action_debug"
    assert config.scaffold_only is True
    assert config.benchmark_claim is False
    assert config.index_backend == "exact"


def test_make_d4rl_mock_arrays_is_deterministic(tmp_path: Path) -> None:
    config = D4RLMockExperimentConfig(
        output_dir=str(tmp_path / "d4rl"),
        seed=5,
        num_transitions=12,
        state_dim=4,
        action_dim=3,
        codebook_size=6,
        k=3,
    )

    first = make_d4rl_mock_arrays(config)
    second = make_d4rl_mock_arrays(config)

    for key in ("observations", "actions", "rewards", "next_observations", "terminals"):
        assert torch.equal(first[key], second[key])
    assert first["observations"].shape == (12, 4)
    assert first["actions"].shape == (12, 3)
    assert first["rewards"].shape == (12,)
    assert first["terminals"].shape == (12,)
    assert torch.isfinite(first["rewards"]).all()


def test_run_d4rl_mock_experiment_writes_claim_free_report(tmp_path: Path) -> None:
    config = D4RLMockExperimentConfig(
        experiment_name="d4rl_unit",
        output_dir=str(tmp_path / "d4rl_exp"),
        num_steps=2,
        batch_size=4,
        k=3,
        num_transitions=16,
        state_dim=4,
        action_dim=3,
        codebook_size=8,
        embedding_dim=8,
    )

    result = run_d4rl_mock_experiment(config)
    report = json.loads(Path(result.output_files["report"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result.output_files["manifest"]).read_text(encoding="utf-8"))

    assert isinstance(result, D4RLMockExperimentResult)
    assert report["report_type"] == "d4rl_mock_experiment_scaffold"
    assert report["synthetic_only"] is False
    assert report["scaffold_only"] is True
    assert report["benchmark_claim"] is False
    assert "training_history_summary" in report
    assert "d4rl_codebook_audit" in report
    assert "d4rl_pmr_scaffold_diagnostics" in report
    assert "d4rl_ann_candidate_audit" in report
    assert report["d4rl_codebook_audit"]["benchmark_claim"] is False
    assert report["d4rl_pmr_scaffold_diagnostics"]["benchmark_claim"] is False
    assert report["d4rl_ann_candidate_audit"]["benchmark_claim"] is False
    assert manifest["run_type"] == "d4rl_mock_experiment_scaffold"
    assert manifest["benchmark_claim"] is False


def test_recsim_mock_experiment_config_validates(tmp_path: Path) -> None:
    config = RecSimMockExperimentConfig(
        output_dir=str(tmp_path / "recsim"),
        num_steps=2,
        batch_size=4,
        k=3,
        num_users=3,
        num_items=8,
        user_feature_dim=4,
        item_feature_dim=3,
        slate_size=2,
        num_interactions=16,
    )

    validated = validate_recsim_mock_experiment_config(config)

    assert validated.benchmark_claim is False
    assert validated.scaffold_only is True
    assert validated.slate_size == 2


def test_recsim_mock_experiment_config_invalid_values_raise(tmp_path: Path) -> None:
    valid = RecSimMockExperimentConfig(output_dir=str(tmp_path / "recsim"))

    with pytest.raises(ValueError, match="num_steps"):
        validate_recsim_mock_experiment_config(replace(valid, num_steps=0))

    with pytest.raises(ValueError, match="slate_size"):
        validate_recsim_mock_experiment_config(replace(valid, slate_size=valid.num_items + 1))

    with pytest.raises(ValueError, match="k"):
        validate_recsim_mock_experiment_config(replace(valid, k=valid.num_items + 1))

    with pytest.raises(ValueError, match="benchmark_claim"):
        validate_recsim_mock_experiment_config(replace(valid, benchmark_claim=True))


def test_load_recsim_mock_experiment_config_loads_yaml() -> None:
    config = load_recsim_mock_experiment_config(ROOT / "configs" / "recsim_mock_experiment.yaml")

    assert config.experiment_name == "recsim_mock_slate_debug"
    assert config.scaffold_only is True
    assert config.benchmark_claim is False
    assert config.index_backend == "exact"


def test_run_recsim_mock_experiment_writes_claim_free_report(tmp_path: Path) -> None:
    config = RecSimMockExperimentConfig(
        experiment_name="recsim_unit",
        output_dir=str(tmp_path / "recsim_exp"),
        num_steps=2,
        batch_size=4,
        k=3,
        num_users=3,
        num_items=8,
        user_feature_dim=4,
        item_feature_dim=3,
        slate_size=2,
        num_interactions=16,
        embedding_dim=8,
    )

    result = run_recsim_mock_experiment(config)
    report = json.loads(Path(result.output_files["report"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result.output_files["manifest"]).read_text(encoding="utf-8"))

    assert isinstance(result, RecSimMockExperimentResult)
    assert report["report_type"] == "recsim_mock_experiment_scaffold"
    assert report["synthetic_only"] is False
    assert report["scaffold_only"] is True
    assert report["benchmark_claim"] is False
    assert "training_history_summary" in report
    assert "recsim_scaffold_evaluation" in report
    assert "recsim_pmr_scaffold_diagnostics" in report
    assert "recsim_ann_candidate_audit" in report
    assert "slate_metadata" in report
    assert report["recsim_scaffold_evaluation"]["benchmark_claim"] is False
    assert report["recsim_pmr_scaffold_diagnostics"]["benchmark_claim"] is False
    assert report["recsim_ann_candidate_audit"]["benchmark_claim"] is False
    assert manifest["run_type"] == "recsim_mock_experiment_scaffold"
    assert manifest["benchmark_claim"] is False


def test_run_movielens_local_experiment_writes_claim_free_report(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n"
        + "\n".join(f"1,{100 + index},5.0,{index}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    config = MovieLensLocalExperimentConfig(
        experiment_name="movielens_unit",
        data_path=str(ratings_path),
        output_dir=str(tmp_path / "movielens_exp"),
        num_steps=2,
        batch_size=2,
        k=2,
        embedding_dim=8,
        split_mode="chronological",
        validation_fraction=0.2,
        test_fraction=0.2,
        baseline_names=("random", "nearest"),
    )

    result = run_movielens_local_experiment(config)
    report = json.loads(Path(result.output_files["report"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result.output_files["manifest"]).read_text(encoding="utf-8"))

    assert isinstance(result, MovieLensLocalExperimentResult)
    assert report["report_type"] == "movielens_local_experiment_scaffold"
    assert report["synthetic_only"] is False
    assert report["scaffold_only"] is True
    assert report["benchmark_claim"] is False
    assert "training_history_summary" in report
    assert "evaluation_summary" in report
    assert "scaffold_evaluation" in report
    assert "ranking_scaffold_diagnostics" in report
    assert "candidate_support_diagnostics" in report
    assert "movielens_pmr_scaffold_diagnostics" in report
    assert "movielens_ann_candidate_audit" in report
    assert "test" in report["ranking_scaffold_diagnostics"]
    assert "test" in report["candidate_support_diagnostics"]
    assert "test" in report["movielens_pmr_scaffold_diagnostics"]
    assert "test" in report["movielens_ann_candidate_audit"]
    assert report["movielens_pmr_scaffold_diagnostics"]["test"]["benchmark_claim"] is False
    assert report["movielens_ann_candidate_audit"]["test"]["benchmark_claim"] is False
    assert "baseline_ranking_scaffold_diagnostics" in report
    assert [item["baseline_name"] for item in report["baseline_diagnostics"]] == ["random", "nearest"]
    assert all(item["benchmark_claim"] is False for item in report["baseline_diagnostics"])
    assert manifest["run_type"] == "movielens_local_experiment_scaffold"
    assert manifest["synthetic_only"] is False
    assert manifest["benchmark_claim"] is False


def test_load_external_scaffold_experiment_report_works_for_movielens(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_movielens_ratings(ratings_path)
    result = run_movielens_local_experiment(
        MovieLensLocalExperimentConfig(
            data_path=str(ratings_path),
            output_dir=str(tmp_path / "movielens"),
            experiment_name="movielens_inspectable",
            num_steps=2,
            batch_size=3,
            k=2,
            embedding_dim=6,
            split_mode="chronological",
            target_tau=0.05,
        )
    )
    report = load_external_scaffold_experiment_report(result.output_files["report"])
    summary = summarize_external_scaffold_experiment_report(report)

    assert report["report_type"] == "movielens_local_experiment_scaffold"
    assert summary["report_type"] == "movielens_local_experiment_scaffold"
    assert summary["benchmark_claim"] is False
    assert summary["num_users"] == 3
    assert "positive_rate" in summary


def test_load_external_scaffold_experiment_report_works_for_d4rl(tmp_path: Path) -> None:
    result = run_d4rl_mock_experiment(
        D4RLMockExperimentConfig(
            output_dir=str(tmp_path / "d4rl"),
            experiment_name="d4rl_inspectable",
            num_steps=2,
            batch_size=4,
            num_transitions=12,
            state_dim=4,
            action_dim=3,
            codebook_size=6,
            k=3,
            embedding_dim=6,
        )
    )
    report = load_external_scaffold_experiment_report(result.output_files["report"])
    summary = summarize_external_scaffold_experiment_report(report)

    assert report["report_type"] == "d4rl_mock_experiment_scaffold"
    assert summary["codebook_size"] == 6
    assert summary["benchmark_claim"] is False
    assert "ann_candidate_recall_mean" in summary
    assert "pmr_at_k_scaffold" in summary
    assert "ann_candidate_pmr_mean" in summary


def test_load_external_scaffold_experiment_report_works_for_recsim(tmp_path: Path) -> None:
    result = run_recsim_mock_experiment(
        RecSimMockExperimentConfig(
            output_dir=str(tmp_path / "recsim"),
            experiment_name="recsim_inspectable",
            num_steps=2,
            batch_size=4,
            num_users=3,
            num_items=8,
            user_feature_dim=4,
            item_feature_dim=4,
            slate_size=2,
            num_interactions=16,
            k=3,
            embedding_dim=6,
        )
    )
    report = load_external_scaffold_experiment_report(result.output_files["report"])
    summary = summarize_external_scaffold_experiment_report(report)

    assert report["report_type"] == "recsim_mock_experiment_scaffold"
    assert summary["slate_size"] == 2
    assert summary["benchmark_claim"] is False
    assert "observed_item_coverage" in summary
    assert "pmr_at_k_scaffold" in summary
    assert "ann_candidate_pmr_mean" in summary


def test_external_scaffold_experiment_report_rejects_wrong_type(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(
        json.dumps(
            {
                "report_type": "unsupported",
                "experiment_name": "bad",
                "scaffold_only": True,
                "benchmark_claim": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported"):
        load_external_scaffold_experiment_report(path)


def test_external_scaffold_experiment_report_rejects_benchmark_claim(tmp_path: Path) -> None:
    path = tmp_path / "claim.json"
    path.write_text(
        json.dumps(
            {
                "report_type": "d4rl_mock_experiment_scaffold",
                "experiment_name": "bad",
                "scaffold_only": True,
                "benchmark_claim": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark_claim"):
        load_external_scaffold_experiment_report(path)


def test_external_scaffold_experiment_bundle_without_movielens(tmp_path: Path) -> None:
    result = run_external_scaffold_experiment_bundle(tmp_path / "bundle", num_steps=2, seed=3)
    manifest = load_external_scaffold_experiment_bundle_manifest(result["manifest_file"])

    assert result["benchmark_claim"] is False
    assert result["scaffold_only"] is True
    assert manifest["benchmark_claim"] is False
    assert manifest["included_experiments"] == ["d4rl_mock", "recsim_mock"]
    assert len(manifest["report_files"]) == 2
    for report_file in manifest["report_files"]:
        report = load_external_scaffold_experiment_report(report_file)
        assert report["benchmark_claim"] is False


def test_external_scaffold_experiment_bundle_with_movielens(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_movielens_ratings(ratings_path)

    result = run_external_scaffold_experiment_bundle(
        tmp_path / "bundle",
        movielens_data_path=ratings_path,
        num_steps=2,
        seed=4,
    )
    manifest = load_external_scaffold_experiment_bundle_manifest(result["manifest_file"])

    assert manifest["included_experiments"] == ["d4rl_mock", "recsim_mock", "movielens_local"]
    assert len(manifest["report_files"]) == 3
    assert any("movielens_local" in report_file for report_file in manifest["report_files"])


def test_external_scaffold_experiment_bundle_inspector_summarizes_reports(tmp_path: Path) -> None:
    result = run_external_scaffold_experiment_bundle(tmp_path / "bundle", num_steps=2, seed=5)

    inspection = inspect_external_scaffold_experiment_bundle(result["manifest_file"])

    assert inspection["benchmark_claim"] is False
    assert inspection["scaffold_only"] is True
    assert inspection["report_count"] == 2
    assert inspection["included_experiments"] == ["d4rl_mock", "recsim_mock"]
    assert {summary["report_type"] for summary in inspection["summaries"]} == {
        "d4rl_mock_experiment_scaffold",
        "recsim_mock_experiment_scaffold",
    }


def test_external_scaffold_experiment_bundle_broken_report_path_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_type": "external_scaffold_experiment_bundle",
                "scaffold_only": True,
                "benchmark_claim": False,
                "included_experiments": ["d4rl_mock"],
                "report_files": ["missing_report.json"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        inspect_external_scaffold_experiment_bundle(manifest_path)


def test_external_scaffold_experiment_bundle_manifest_rejects_benchmark_claim(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_type": "external_scaffold_experiment_bundle",
                "scaffold_only": True,
                "benchmark_claim": True,
                "included_experiments": [],
                "report_files": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark_claim"):
        load_external_scaffold_experiment_bundle_manifest(manifest_path)


def test_external_support_protocol_config_validates(tmp_path: Path) -> None:
    config = ExternalSupportProtocolConfig(output_dir=str(tmp_path / "support"), num_steps=2, k_values=(2, 3))

    validated = validate_external_support_protocol_config(config)

    assert validated.scaffold_only is True
    assert validated.benchmark_claim is False
    assert validated.k_values == (2, 3)


def test_external_support_protocol_config_invalid_values_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        validate_external_support_protocol_config(
            ExternalSupportProtocolConfig(
                output_dir=str(tmp_path / "support"),
                include_d4rl_mock=False,
                include_recsim_mock=False,
                include_movielens=False,
            )
        )
    with pytest.raises(ValueError, match="k_values"):
        validate_external_support_protocol_config(
            ExternalSupportProtocolConfig(output_dir=str(tmp_path / "support"), k_values=())
        )
    with pytest.raises(ValueError, match="movielens_data_path"):
        validate_external_support_protocol_config(
            ExternalSupportProtocolConfig(output_dir=str(tmp_path / "support"), include_movielens=True)
        )


def test_load_external_support_protocol_config_loads_yaml() -> None:
    config = load_external_support_protocol_config(ROOT / "configs" / "external_support_protocol.yaml")

    assert config.include_d4rl_mock is True
    assert config.include_recsim_mock is True
    assert config.include_movielens is False
    assert config.k_values == (5, 10, 20)


def test_run_external_support_protocol_writes_summary(tmp_path: Path) -> None:
    result = run_external_support_protocol(
        ExternalSupportProtocolConfig(
            output_dir=str(tmp_path / "support"),
            num_steps=2,
            batch_size=4,
            train_k=3,
            k_values=(2, 3),
            include_d4rl_mock=True,
            include_recsim_mock=True,
            include_movielens=False,
        )
    )
    summary = load_external_support_protocol_summary(result["summary_file"])
    rows = summarize_external_support_protocol(summary)

    assert summary["report_type"] == "external_support_protocol_summary"
    assert summary["scaffold_only"] is True
    assert summary["benchmark_claim"] is False
    assert summary["included_routes"] == ["d4rl_mock", "recsim_mock"]
    assert len(rows) == 4
    assert all(row["benchmark_claim"] is False for row in rows)
    assert all(0.0 <= row["pmr_at_k_scaffold"] <= 1.0 for row in rows)


def test_run_external_support_protocol_with_movielens_writes_summary(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_movielens_ratings(ratings_path)
    result = run_external_support_protocol(
        ExternalSupportProtocolConfig(
            output_dir=str(tmp_path / "support"),
            num_steps=2,
            batch_size=3,
            train_k=2,
            k_values=(2,),
            include_d4rl_mock=False,
            include_recsim_mock=False,
            include_movielens=True,
            movielens_data_path=str(ratings_path),
        )
    )
    summary = load_external_support_protocol_summary(result["summary_file"])
    rows = summarize_external_support_protocol(summary)

    assert summary["included_routes"] == ["movielens_local"]
    assert len(rows) == 1
    assert rows[0]["route"] == "movielens_local"
    assert rows[0]["scaffold_only"] is True


def test_load_external_support_protocol_summary_rejects_benchmark_claim(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "report_type": "external_support_protocol_summary",
                "scaffold_only": True,
                "benchmark_claim": True,
                "route_summaries": {},
                "support_metrics_by_route_and_k": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark_claim"):
        load_external_support_protocol_summary(path)


def test_run_synthetic_experiment_returns_result_and_writes_json(tmp_path: Path) -> None:
    config = _small_experiment_config(tmp_path)

    result = run_synthetic_experiment(config)

    report_path = Path(result.output_files["report"])
    checkpoint_path = Path(result.output_files["final_checkpoint"])
    manifest_path = Path(result.output_files["manifest"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = load_run_manifest(manifest_path)

    assert isinstance(result, ExperimentResult)
    assert report_path.exists()
    assert checkpoint_path.exists()
    assert manifest_path.exists()
    assert report["mode"] == "synthetic_experiment"
    assert report["experiment_name"] == "unit_experiment"
    assert report["synthetic_only"] is True
    assert report["benchmark_claim"] is False
    assert report["evaluation"]["benchmark_claim"] is False
    assert report["evaluation_summary"]["benchmark_claim"] is False
    assert report["evaluation_summary"]["num_batches"] == config.eval_num_batches
    assert "mean_bellman_loss" in report["evaluation_summary"]["metrics"]
    assert report["evaluation_config"]["num_batches"] == config.eval_num_batches
    assert report["checkpoint_files"] == [str(checkpoint_path)]
    assert report["summary"]["checkpoint_count"] == 1
    assert result.evaluation_result.benchmark_claim is False
    assert result.evaluation_summary.benchmark_claim is False
    assert report["baseline_diagnostics"] == []
    assert report["baseline_diagnostics_only"] is False
    assert manifest["report_files"] == [str(report_path)]
    assert manifest["checkpoint_files"] == [str(checkpoint_path)]
    assert manifest["metadata"]["experiment_name"] == "unit_experiment"


def test_run_synthetic_experiment_includes_optional_baseline_diagnostics(tmp_path: Path) -> None:
    config = _small_experiment_config(tmp_path)
    config.baseline_names = ("random", "nearest")

    result = run_synthetic_experiment(validate_experiment_config(config))
    report = json.loads(Path(result.output_files["report"]).read_text(encoding="utf-8"))

    assert len(result.baseline_results) == 2
    assert report["baseline_diagnostics_only"] is True
    assert [item["baseline_name"] for item in report["baseline_diagnostics"]] == ["random", "nearest"]
    assert all(item["benchmark_claim"] is False for item in report["baseline_diagnostics"])
    assert "baseline_diagnostics" not in report["evaluation_summary"]


def test_run_synthetic_experiment_resume_from_checkpoint(tmp_path: Path) -> None:
    first_config = _small_experiment_config(tmp_path)
    first_config.num_steps = 1
    first_result = run_synthetic_experiment(first_config)

    resumed_config = _small_experiment_config(tmp_path)
    resumed_config.experiment_name = "resumed_experiment"
    resumed_config.resume_from = first_result.output_files["final_checkpoint"]
    resumed_config.checkpoint_every = 1
    resumed_result = run_synthetic_experiment(validate_experiment_config(resumed_config))
    report = json.loads(Path(resumed_result.output_files["report"]).read_text(encoding="utf-8"))

    assert Path(resumed_result.output_files["final_checkpoint"]).exists()
    assert report["resume_from"] == first_result.output_files["final_checkpoint"]
    assert report["checkpoint_files"]
    assert resumed_result.diagnostics["resume_from"] == first_result.output_files["final_checkpoint"]


def _make_tiny_ratings_csv_for_experiment(tmp_path: Path) -> Path:
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


def test_run_benchmark_protocol_movielens_produces_summary_and_tables(tmp_path: Path) -> None:
    from retrieval_alignment.config import BenchmarkProtocolConfig, validate_benchmark_protocol_config
    from retrieval_alignment.experiment import run_benchmark_protocol

    ratings_path = _make_tiny_ratings_csv_for_experiment(tmp_path)
    cfg = BenchmarkProtocolConfig(
        output_dir=str(tmp_path / "bench_output"),
        routes=("movielens_chrono",),
        seeds=(0,),
        num_steps=2,
        batch_size=4,
        k_values=(3,),
        train_k=3,
        index_backend_values=("exact",),
        index_refresh_values=(2,),
        index_encoder_policies=("online",),
        index_ema_tau_values=(None,),
        baselines=("random",),
        data_path=str(ratings_path),
    )
    result = run_benchmark_protocol(validate_benchmark_protocol_config(cfg))

    assert result.summary["benchmark_claim"] is False
    assert result.summary["scaffold_only"] is True
    assert result.summary["num_runs"] >= 1
    assert Path(result.summary_file).exists()
    assert Path(result.markdown_table_file).exists()
    assert Path(result.csv_table_file).exists()

    md_text = Path(result.markdown_table_file).read_text(encoding="utf-8")
    assert "benchmark_claim: false" in md_text


def test_run_benchmark_protocol_benchmark_claim_is_always_false(tmp_path: Path) -> None:
    from retrieval_alignment.config import BenchmarkProtocolConfig, validate_benchmark_protocol_config
    from retrieval_alignment.experiment import run_benchmark_protocol
    import json

    ratings_path = _make_tiny_ratings_csv_for_experiment(tmp_path)
    cfg = BenchmarkProtocolConfig(
        output_dir=str(tmp_path / "bench_output2"),
        routes=("movielens_chrono",),
        seeds=(0, 1),
        num_steps=2,
        batch_size=4,
        k_values=(3,),
        train_k=3,
        index_backend_values=("exact",),
        index_refresh_values=(2,),
        index_encoder_policies=("online",),
        index_ema_tau_values=(None,),
        baselines=("random", "nearest"),
        data_path=str(ratings_path),
    )
    result = run_benchmark_protocol(validate_benchmark_protocol_config(cfg))

    loaded = json.loads(Path(result.summary_file).read_text(encoding="utf-8"))
    assert loaded["benchmark_claim"] is False
    for row in loaded.get("per_run_rows", []):
        assert row.get("benchmark_claim") is False


def test_inspect_benchmark_protocol_summary_returns_string() -> None:
    from retrieval_alignment.experiment import inspect_benchmark_protocol_summary

    summary = {
        "protocol_name": "first_benchmark_protocol",
        "benchmark_claim": False,
        "scaffold_only": True,
        "routes": ["movielens_chrono"],
        "seeds": [0],
        "num_runs": 2,
        "grouped_statistics": {
            "benchmark_claim": False,
            "group_keys": ["route", "method"],
            "groups": [
                {
                    "group": {"route": "movielens_chrono", "method": "retrieval_alignment"},
                    "count": 2,
                    "metrics": {
                        "final_bellman_loss": {"mean": 0.5, "std": 0.1, "stderr": 0.07}
                    },
                }
            ],
        },
        "limitations": ["scaffold only"],
    }
    text = inspect_benchmark_protocol_summary(summary)

    assert isinstance(text, str)
    assert "first_benchmark_protocol" in text
    assert "benchmark_claim: false" in text
    assert "scaffold only" in text
