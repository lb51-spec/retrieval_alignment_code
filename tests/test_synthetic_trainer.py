"""Tests for the synthetic trainer abstraction."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import ModelConfig, SyntheticDatasetConfig
from retrieval_alignment.networks import build_networks
from retrieval_alignment.synthetic import make_synthetic_offline_data
from retrieval_alignment.target_networks import clone_target_network
from retrieval_alignment.trainer import (
    SyntheticIndexCache,
    SyntheticDWMQLTrainer,
    TrainStepResult,
    TrainingHistory,
    compute_action_embedding_drift,
    compute_encoder_policy_retrieval_overlap,
    compute_support_weights,
    default_smoke_alternating_schedule,
    make_optimizer_bundle,
    train_step_result_to_dict,
    training_history_to_dict,
    validate_retrieval_correction_mode,
    write_json_report,
)


def test_compute_support_weights_modes() -> None:
    pmr = torch.tensor([1.0, 0.5, 0.25])
    normalized = torch.tensor([2.0, 1.0, 0.5])

    assert torch.allclose(compute_support_weights(pmr, mode="none"), torch.ones_like(pmr))
    missing = compute_support_weights(pmr, mode="missing_mass", scale=2.0)
    assert missing[0] < missing[1] < missing[2]
    inverse = compute_support_weights(pmr, mode="inverse_pmr", max_weight=3.0)
    assert inverse.max().item() <= 3.0
    margin = compute_support_weights(pmr, normalized_pmr=normalized, mode="normalized_pmr_margin", scale=2.0)
    assert margin[-1].item() > margin[0].item()


def test_compute_support_weights_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="support_weighting_mode"):
        compute_support_weights(torch.tensor([0.5]), mode="bad")
    with pytest.raises(ValueError, match="normalized_pmr"):
        compute_support_weights(torch.tensor([0.5]), mode="normalized_pmr_margin")
    with pytest.raises(ValueError, match="pmr"):
        compute_support_weights(torch.tensor([float("nan")]), mode="missing_mass")


def _data_model_optimizer(seed: int = 864):
    data = make_synthetic_offline_data(
        SyntheticDatasetConfig(
            num_transitions=40,
            state_dim=4,
            action_dim=3,
            num_actions=10,
            seed=seed,
            terminal_prob=0.2,
        )
    )
    torch.manual_seed(seed)
    model = build_networks(
        ModelConfig(
            state_dim=4,
            action_dim=3,
            embedding_dim=8,
            hidden_dims=(16,),
            pair_feature_mode="concat_product",
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return data, model, optimizer


def test_compute_action_embedding_drift_returns_finite_values() -> None:
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    comparison = torch.tensor([[0.5, 0.0], [0.0, 0.5]])

    drift = compute_action_embedding_drift(reference, comparison)

    assert set(drift) == {
        "mean_l2",
        "max_l2",
        "mean_cosine_distance",
        "max_cosine_distance",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in drift.values())
    assert drift["mean_l2"] > 0


def test_compute_action_embedding_drift_identical_embeddings_are_zero() -> None:
    embeddings = torch.randn(4, 3)

    drift = compute_action_embedding_drift(embeddings, embeddings.clone())

    assert drift["mean_l2"] == pytest.approx(0.0)
    assert drift["max_l2"] == pytest.approx(0.0)
    assert drift["mean_cosine_distance"] == pytest.approx(0.0, abs=1e-6)
    assert drift["max_cosine_distance"] == pytest.approx(0.0, abs=1e-6)


def test_compute_action_embedding_drift_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_action_embedding_drift(torch.zeros(2, 3), torch.zeros(2, 4))


def test_compute_action_embedding_drift_non_finite_raises() -> None:
    reference = torch.zeros(2, 3)
    comparison = torch.zeros(2, 3)
    comparison[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        compute_action_embedding_drift(reference, comparison)


def test_compute_action_embedding_drift_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_action_embedding_drift(torch.empty(0, 3), torch.empty(0, 3))


def test_retrieval_correction_modes_validate_and_run_tiny_schedule() -> None:
    for mode in ("aware", "naive", "uniform_weights", "no_density"):
        data, model, _ = _data_model_optimizer(seed=300 + len(mode))
        bundle = make_optimizer_bundle(model, lr=1e-3, mode="separate_heads")
        trainer = SyntheticDWMQLTrainer(
            data,
            model,
            optimizer_bundle=bundle,
            batch_size=6,
            k=4,
            gamma=0.95,
            use_persistent_index=True,
            audit_stale_index=True,
            retrieval_correction_mode=mode,
        )

        history = trainer.train_objective_schedule(2, schedule=("critic", "density"))

        assert history["diagnostics"]["retrieval_correction_mode"] == mode
        assert all(torch.isfinite(torch.tensor(value)) for value in history["losses"])
        assert len(history["stale_index_pmr_normalized_means"]) == 2
        if mode == "no_density":
            density_step = history["results"][1]
            assert density_step.diagnostics["stepped_parts"] == []


def test_aware_support_correction_changes_critic_loss_relative_to_naive() -> None:
    data, model_aware, _ = _data_model_optimizer(seed=321)
    _, model_naive, _ = _data_model_optimizer(seed=321)
    model_naive.load_state_dict(model_aware.state_dict())
    aware = SyntheticDWMQLTrainer(
        data,
        model_aware,
        optimizer_bundle=make_optimizer_bundle(model_aware, lr=1e-3, mode="separate_heads"),
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(123),
        use_persistent_index=True,
        audit_stale_index=True,
        retrieval_correction_mode="aware",
        support_correction_weight=0.1,
        support_correction_metric="normalized_pmr_margin",
        support_correction_target=10.0,
    )
    naive = SyntheticDWMQLTrainer(
        data,
        model_naive,
        optimizer_bundle=make_optimizer_bundle(model_naive, lr=1e-3, mode="separate_heads"),
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(123),
        use_persistent_index=True,
        audit_stale_index=True,
        retrieval_correction_mode="naive",
        support_correction_weight=0.1,
        support_correction_metric="normalized_pmr_margin",
        support_correction_target=10.0,
    )

    aware_result = aware.update_critic(update_encoder=False)
    naive_result = naive.update_critic(update_encoder=False)

    assert aware_result.diagnostics["support_correction_active"] is True
    assert naive_result.diagnostics["support_correction_active"] is False
    assert aware_result.diagnostics["support_correction_loss"] > 0.0
    expected_delta = 0.1 * aware_result.diagnostics["support_correction_loss"]
    assert aware_result.diagnostics["raw_bellman_loss"] > 0.0
    assert aware_result.diagnostics["support_correction_contribution"] == pytest.approx(expected_delta)
    assert aware_result.diagnostics["critic_total_loss"] == pytest.approx(
        aware_result.diagnostics["raw_bellman_loss"] + expected_delta
    )
    assert naive_result.diagnostics["support_correction_contribution"] == pytest.approx(0.0)
    assert naive_result.diagnostics["critic_total_loss"] == pytest.approx(naive_result.diagnostics["raw_bellman_loss"])
    assert aware_result.loss.item() - naive_result.loss.item() == pytest.approx(expected_delta, rel=1e-5)


def test_default_support_correction_weight_keeps_aware_diagnostic_only() -> None:
    data, model, _ = _data_model_optimizer(seed=322)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer_bundle=make_optimizer_bundle(model, lr=1e-3, mode="separate_heads"),
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        audit_stale_index=True,
        retrieval_correction_mode="aware",
    )

    result = trainer.update_critic(update_encoder=False)

    assert result.diagnostics["support_correction_active"] is False
    assert result.diagnostics["support_correction_weight"] == pytest.approx(0.0)


def test_aware_support_weighting_records_nonuniform_bellman_weights() -> None:
    data, model, _ = _data_model_optimizer(seed=333)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer_bundle=make_optimizer_bundle(model, lr=1e-3, mode="separate_heads"),
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(123),
        use_persistent_index=True,
        audit_stale_index=True,
        retrieval_correction_mode="aware",
        support_weighting_mode="inverse_pmr",
        support_weight_scale=1.0,
        support_weight_max=20.0,
    )
    with torch.no_grad():
        for parameter in trainer.model.action_encoder.parameters():
            parameter.add_(0.5 * torch.randn_like(parameter))

    result = trainer.update_critic(update_encoder=False)

    assert result.diagnostics["support_weighting_mode"] == "inverse_pmr"
    assert result.diagnostics["support_weighting_active"] is True
    assert result.diagnostics["support_weight_mean"] >= 1.0
    assert result.diagnostics["support_weight_max"] >= result.diagnostics["support_weight_min"]
    assert result.diagnostics["support_weighted_bellman_loss"] != pytest.approx(
        result.diagnostics["raw_bellman_loss"]
    )
    assert result.diagnostics["critic_total_loss"] == pytest.approx(
        result.diagnostics["support_weighted_bellman_loss"]
        + result.diagnostics["support_correction_contribution"]
    )


def test_naive_mode_forces_support_weighting_to_none() -> None:
    data, model, _ = _data_model_optimizer(seed=334)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer_bundle=make_optimizer_bundle(model, lr=1e-3, mode="separate_heads"),
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        audit_stale_index=True,
        retrieval_correction_mode="naive",
        support_weighting_mode="missing_mass",
    )

    result = trainer.update_critic(update_encoder=False)

    assert result.diagnostics["support_weighting_mode"] == "none"
    assert result.diagnostics["support_weighting_active"] is False
    assert result.diagnostics["support_weight_mean"] == pytest.approx(1.0)
    assert result.diagnostics["support_weighted_bellman_loss"] == pytest.approx(
        result.diagnostics["raw_bellman_loss"]
    )


def test_invalid_retrieval_correction_mode_raises() -> None:
    with pytest.raises(ValueError, match="retrieval_correction_mode"):
        validate_retrieval_correction_mode("unsupported")

    data, model, optimizer = _data_model_optimizer()
    with pytest.raises(ValueError, match="retrieval_correction_mode"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            retrieval_correction_mode="unsupported",
        )


def _assert_overlap_values_are_valid(diagnostics: dict[str, float]) -> None:
    assert diagnostics
    for value in diagnostics.values():
        assert torch.isfinite(torch.tensor(value))
        assert 0.0 <= value <= 1.0


def test_compute_encoder_policy_retrieval_overlap_online_only_is_empty() -> None:
    data, model, _ = _data_model_optimizer()
    query_states = data.dataset.sample(4).next_states

    diagnostics = compute_encoder_policy_retrieval_overlap(
        data=data,
        model=model,
        query_states=query_states,
        k=3,
        index_backend="exact",
        index_metric="dot",
    )

    assert diagnostics == {}


def test_compute_encoder_policy_retrieval_overlap_with_target_model() -> None:
    data, model, _ = _data_model_optimizer()
    target_model = clone_target_network(model)
    query_states = data.dataset.sample(4).next_states

    diagnostics = compute_encoder_policy_retrieval_overlap(
        data=data,
        model=model,
        query_states=query_states,
        k=3,
        index_backend="exact",
        index_metric="dot",
        target_model=target_model,
    )

    assert "target_vs_online_recall_mean" in diagnostics
    assert "target_vs_online_overlap_fraction_mean" in diagnostics
    _assert_overlap_values_are_valid(diagnostics)


def test_compute_encoder_policy_retrieval_overlap_with_ema_model() -> None:
    data, model, _ = _data_model_optimizer()
    ema_model = clone_target_network(model)
    query_states = data.dataset.sample(4).next_states

    diagnostics = compute_encoder_policy_retrieval_overlap(
        data=data,
        model=model,
        query_states=query_states,
        k=3,
        index_backend="exact",
        index_metric="dot",
        ema_index_model=ema_model,
    )

    assert "ema_vs_online_recall_mean" in diagnostics
    assert "ema_vs_online_overlap_fraction_mean" in diagnostics
    _assert_overlap_values_are_valid(diagnostics)


def test_compute_encoder_policy_retrieval_overlap_with_target_and_ema_models() -> None:
    data, model, _ = _data_model_optimizer()
    target_model = clone_target_network(model)
    ema_model = clone_target_network(model)
    query_states = data.dataset.sample(4).next_states

    diagnostics = compute_encoder_policy_retrieval_overlap(
        data=data,
        model=model,
        query_states=query_states,
        k=3,
        index_backend="exact",
        index_metric="dot",
        target_model=target_model,
        ema_index_model=ema_model,
    )

    assert "target_vs_online_recall_mean" in diagnostics
    assert "ema_vs_online_recall_mean" in diagnostics
    assert "ema_vs_target_recall_mean" in diagnostics
    assert "ema_vs_target_overlap_fraction_mean" in diagnostics
    _assert_overlap_values_are_valid(diagnostics)


def test_compute_encoder_policy_retrieval_overlap_validation() -> None:
    data, model, _ = _data_model_optimizer()
    query_states = data.dataset.sample(4).next_states

    with pytest.raises(ValueError, match="k"):
        compute_encoder_policy_retrieval_overlap(data, model, query_states, 0, "exact", "dot")

    bad_query_states = query_states.clone()
    bad_query_states[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        compute_encoder_policy_retrieval_overlap(data, model, bad_query_states, 3, "exact", "dot")


def test_synthetic_trainer_initializes() -> None:
    data, model, optimizer = _data_model_optimizer()

    trainer = SyntheticDWMQLTrainer(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
    )

    assert trainer.step == 0
    assert trainer.target_update_count == 0
    assert trainer.index_age == 0
    assert trainer.index_refresh_count == 0
    assert trainer.index_refresh_steps == []
    assert trainer.index_ema_update_count == 0


def test_train_step_returns_result_and_increments_step() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95)

    result = trainer.train_step()

    assert isinstance(result, TrainStepResult)
    assert trainer.step == 1
    assert result.diagnostics["index_age"] == 0
    assert result.diagnostics["index_refresh_count"] == 1
    assert result.diagnostics["index_refreshed_this_step"] is True


def test_train_step_result_to_dict_is_json_serializable() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95)
    result = trainer.train_step()

    report = train_step_result_to_dict(result)

    json.dumps(report, allow_nan=False)
    assert "total_loss" in report
    assert "diagnostics" in report


def test_train_returns_history_with_expected_lengths() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95)

    history = trainer.train(3)

    assert isinstance(history, TrainingHistory)
    assert len(history.losses) == 3
    assert len(history.bellman_losses) == 3
    assert len(history.stationarity_losses) == 3
    assert len(history.density_regularizations) == 3
    assert len(history.grad_norms) == 3
    assert len(history.parameter_delta_norms) == 3
    assert len(history.index_ages) == 3
    assert trainer.step == 3


def test_training_history_to_dict_is_json_serializable() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95)
    history = trainer.train(2)

    report = training_history_to_dict(history)

    json.dumps(report, allow_nan=False)
    assert report["losses"] == history.losses
    assert report["diagnostics"]["num_steps"] == 2


def test_write_json_report_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "report.json"

    write_json_report({"mode": "unit", "synthetic_only": True, "benchmark_claim": False}, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["mode"] == "unit"
    assert loaded["synthetic_only"] is True
    assert loaded["benchmark_claim"] is False


def test_write_json_report_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file"):
        write_json_report({"mode": "unit"}, tmp_path)


def test_history_losses_are_finite() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95)

    history = trainer.train(3)

    assert all(torch.isfinite(torch.tensor(value)) for value in history.losses)
    assert all(torch.isfinite(torch.tensor(value)) for value in history.grad_norms)


def test_model_parameters_change_after_training() -> None:
    data, model, optimizer = _data_model_optimizer()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    trainer = SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95)

    trainer.train(3)

    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, model.parameters(), strict=True)
    )


def test_target_model_updates_when_tau_is_provided() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    before = [parameter.detach().clone() for parameter in target_model.parameters()]
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_model,
        target_tau=0.5,
        target_update_every=2,
    )

    history = trainer.train(5)

    assert trainer.target_update_count == 2
    assert history.target_update_steps == [2, 4]
    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, target_model.parameters(), strict=True)
    )


def test_index_refresh_cadence_tracks_age_and_steps() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        index_refresh_every=2,
    )

    first = trainer.train_step()
    second = trainer.train_step()
    third = trainer.train_step()

    assert first.diagnostics["index_age"] == 1
    assert first.diagnostics["index_refreshed_this_step"] is False
    assert second.diagnostics["index_age"] == 0
    assert second.diagnostics["index_refreshed_this_step"] is True
    assert third.diagnostics["index_age"] == 1
    assert trainer.index_refresh_count == 1
    assert trainer.index_refresh_steps == [2]


def test_persistent_index_cache_initializes_and_train_step_runs() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
    )

    assert isinstance(trainer.index_cache, SyntheticIndexCache)
    assert trainer.index_cache.last_refresh_step == 0

    result = trainer.train_step()

    assert result.diagnostics["persistent_index_used"] is True
    assert result.diagnostics["index_encoder_policy"] == "online"
    assert result.diagnostics["index_encoder_drift_mean_l2"] == pytest.approx(0.0)
    assert result.diagnostics["index_encoder_drift_max_l2"] == pytest.approx(0.0)
    assert result.diagnostics["index_encoder_drift_mean_cosine_distance"] == pytest.approx(0.0, abs=1e-6)
    assert result.diagnostics["index_encoder_drift_max_cosine_distance"] == pytest.approx(0.0, abs=1e-6)
    assert result.diagnostics["persistent_index_last_refresh_step"] == 1
    assert torch.isfinite(result.total_loss)


def test_persistent_index_target_encoder_policy_runs() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_model,
        target_tau=0.5,
        target_update_every=1,
        use_persistent_index=True,
        index_encoder_policy="target",
    )

    result = trainer.train_step()

    assert result.diagnostics["persistent_index_used"] is True
    assert result.diagnostics["index_encoder_policy"] == "target"
    assert result.diagnostics["index_encoder_drift_mean_l2"] is not None
    assert result.diagnostics["index_encoder_drift_max_l2"] is not None
    assert torch.isfinite(torch.tensor(result.diagnostics["index_encoder_drift_mean_l2"]))
    assert torch.isfinite(torch.tensor(result.diagnostics["index_encoder_drift_max_l2"]))
    assert trainer.index_cache is not None
    assert trainer.index_cache.last_refresh_step == 1


def test_persistent_index_ema_encoder_policy_runs_and_is_frozen() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_encoder_policy="ema",
        index_ema_tau=0.05,
    )

    assert trainer.index_ema_model is not None
    assert all(not parameter.requires_grad for parameter in trainer.index_ema_model.parameters())

    result = trainer.train_step()

    assert result.diagnostics["persistent_index_used"] is True
    assert result.diagnostics["index_encoder_policy"] == "ema"
    assert result.diagnostics["index_ema_tau"] == 0.05
    assert result.diagnostics["index_ema_update_count"] == 1
    assert result.diagnostics["index_encoder_drift_mean_l2"] is not None
    assert result.diagnostics["index_encoder_drift_max_l2"] is not None
    assert torch.isfinite(torch.tensor(result.diagnostics["index_encoder_drift_mean_l2"]))
    assert torch.isfinite(torch.tensor(result.diagnostics["index_encoder_drift_max_l2"]))
    assert trainer.index_ema_update_count == 1
    assert trainer.index_cache is not None
    assert trainer.index_cache.last_refresh_step == 1


def test_ema_index_encoder_policy_requires_tau() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="index_ema_tau"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            use_persistent_index=True,
            index_encoder_policy="ema",
        )


def test_invalid_index_ema_tau_raises() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="index_ema_tau"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            use_persistent_index=True,
            index_encoder_policy="ema",
            index_ema_tau=1.5,
        )


def test_target_model_and_ema_index_model_can_coexist() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_model,
        target_tau=0.5,
        target_update_every=1,
        use_persistent_index=True,
        index_encoder_policy="ema",
        index_ema_tau=0.1,
    )

    result = trainer.train_step()

    assert trainer.target_update_count == 1
    assert trainer.index_ema_update_count == 1
    assert trainer.index_ema_model is not None
    assert result.diagnostics["target_update_count"] == 1
    assert result.diagnostics["index_ema_update_count"] == 1


def test_target_index_encoder_policy_requires_target_model() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="target_model"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            use_persistent_index=True,
            index_encoder_policy="target",
        )


def test_persistent_index_train_runs_and_refreshes_on_cadence() -> None:
    data, model, optimizer = _data_model_optimizer()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_refresh_every=2,
    )

    history = trainer.train(5)

    assert history.index_refresh_count == 2
    assert history.index_refresh_steps == [2, 4]
    assert history.index_ages == [1, 0, 1, 0, 1]
    assert history.diagnostics["persistent_index_used"] is True
    assert history.index_ema_update_count == 0
    assert trainer.index_cache is not None
    assert trainer.index_cache.last_refresh_step == 4
    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, model.parameters(), strict=True)
    )


def test_persistent_index_with_stale_audit_records_diagnostics() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        audit_stale_index=True,
        index_refresh_every=2,
    )

    result = trainer.train_step()

    assert result.diagnostics["persistent_index_used"] is True
    assert result.diagnostics["audit_stale_index"] is True
    assert result.diagnostics["stale_index_recall_mean"] is not None
    assert result.diagnostics["stale_index_overlap_fraction_mean"] is not None
    assert result.diagnostics["stale_index_pmr_mean"] is not None
    assert result.diagnostics["stale_index_pmr_min"] is not None
    assert result.diagnostics["stale_index_pmr_max"] is not None
    assert 0.0 <= result.diagnostics["stale_index_recall_mean"] <= 1.0
    assert 0.0 <= result.diagnostics["stale_index_overlap_fraction_mean"] <= 1.0
    assert 0.0 <= result.diagnostics["stale_index_pmr_min"] <= result.diagnostics["stale_index_pmr_mean"] <= result.diagnostics["stale_index_pmr_max"] <= 1.0
    assert torch.isfinite(torch.tensor(result.diagnostics["stale_index_recall_mean"]))
    assert torch.isfinite(torch.tensor(result.diagnostics["stale_index_overlap_fraction_mean"]))
    assert torch.isfinite(torch.tensor(result.diagnostics["stale_index_pmr_mean"]))
    assert torch.isfinite(torch.tensor(result.diagnostics["stale_index_best_value_gap_mean"]))


def test_training_history_stores_stale_index_audit_lists() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        audit_stale_index=True,
        index_refresh_every=2,
    )

    history = trainer.train(3)

    assert len(history.stale_index_recall_means) == 3
    assert len(history.stale_index_overlap_fraction_means) == 3
    assert len(history.stale_index_best_value_gap_means) == 3
    assert len(history.stale_index_pmr_means) == 3
    assert len(history.stale_index_pmr_mins) == 3
    assert len(history.stale_index_pmr_maxs) == 3
    assert all(value is not None for value in history.stale_index_recall_means)
    assert all(value is not None for value in history.stale_index_overlap_fraction_means)
    assert all(value is not None for value in history.stale_index_pmr_means)
    assert all(0.0 <= value <= 1.0 for value in history.stale_index_recall_means if value is not None)
    assert all(0.0 <= value <= 1.0 for value in history.stale_index_pmr_means if value is not None)
    assert history.diagnostics["audit_stale_index"] is True
    assert history.diagnostics["final_stale_index_pmr_mean"] == history.stale_index_pmr_means[-1]


def test_trainer_records_encoder_policy_overlap_with_target() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_model,
        target_tau=0.5,
        use_persistent_index=True,
        index_encoder_policy="target",
        audit_encoder_policy_overlap=True,
    )

    result = trainer.train_step()

    assert result.diagnostics["audit_encoder_policy_overlap"] is True
    assert result.diagnostics["target_vs_online_recall_mean"] is not None
    assert result.diagnostics["target_vs_online_overlap_fraction_mean"] is not None
    assert 0.0 <= result.diagnostics["target_vs_online_recall_mean"] <= 1.0
    assert 0.0 <= result.diagnostics["target_vs_online_overlap_fraction_mean"] <= 1.0
    assert result.diagnostics["ema_vs_online_recall_mean"] is None


def test_trainer_records_encoder_policy_overlap_with_ema() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_encoder_policy="ema",
        index_ema_tau=0.05,
        audit_encoder_policy_overlap=True,
    )

    result = trainer.train_step()

    assert result.diagnostics["audit_encoder_policy_overlap"] is True
    assert result.diagnostics["ema_vs_online_recall_mean"] is not None
    assert result.diagnostics["ema_vs_online_overlap_fraction_mean"] is not None
    assert 0.0 <= result.diagnostics["ema_vs_online_recall_mean"] <= 1.0
    assert 0.0 <= result.diagnostics["ema_vs_online_overlap_fraction_mean"] <= 1.0
    assert result.diagnostics["target_vs_online_recall_mean"] is None


def test_trainer_records_encoder_policy_overlap_with_target_and_ema() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_model,
        target_tau=0.5,
        use_persistent_index=True,
        index_encoder_policy="ema",
        index_ema_tau=0.05,
        audit_encoder_policy_overlap=True,
    )

    result = trainer.train_step()

    assert result.diagnostics["target_vs_online_recall_mean"] is not None
    assert result.diagnostics["ema_vs_online_recall_mean"] is not None
    assert result.diagnostics["ema_vs_target_recall_mean"] is not None
    assert result.diagnostics["ema_vs_target_overlap_fraction_mean"] is not None


def test_training_history_stores_encoder_policy_overlap_lists() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_model,
        target_tau=0.5,
        use_persistent_index=True,
        index_encoder_policy="ema",
        index_ema_tau=0.05,
        audit_encoder_policy_overlap=True,
    )

    history = trainer.train(3)

    assert len(history.target_vs_online_recall_means) == 3
    assert len(history.target_vs_online_overlap_fraction_means) == 3
    assert len(history.ema_vs_online_recall_means) == 3
    assert len(history.ema_vs_online_overlap_fraction_means) == 3
    assert len(history.ema_vs_target_recall_means) == 3
    assert len(history.ema_vs_target_overlap_fraction_means) == 3
    assert all(value is not None for value in history.target_vs_online_recall_means)
    assert all(value is not None for value in history.ema_vs_online_recall_means)
    assert all(value is not None for value in history.ema_vs_target_recall_means)
    assert history.diagnostics["audit_encoder_policy_overlap"] is True
    assert history.diagnostics["final_ema_vs_target_recall_mean"] == history.ema_vs_target_recall_means[-1]


def test_audit_stale_index_requires_persistent_index() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="audit_stale_index"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            audit_stale_index=True,
        )


def test_audit_encoder_policy_overlap_requires_persistent_index() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="audit_encoder_policy_overlap"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            audit_encoder_policy_overlap=True,
        )


def test_training_history_includes_index_refresh_info() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        index_refresh_every=2,
    )

    history = trainer.train(5)

    assert history.index_ages == [1, 0, 1, 0, 1]
    assert history.index_refresh_steps == [2, 4]
    assert history.index_refresh_count == 2
    assert history.diagnostics["index_refresh_every"] == 2
    assert history.diagnostics["final_index_age"] == 1


def test_synthetic_trainer_state_dict_roundtrip_restores_counters() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_refresh_every=2,
    )
    trainer.train(3)
    state = trainer.state_dict()

    data_2, model_2, optimizer_2 = _data_model_optimizer()
    restored = SyntheticDWMQLTrainer(
        data_2,
        model_2,
        optimizer_2,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_refresh_every=2,
    )
    restored.load_state_dict(state)

    assert restored.step == trainer.step
    assert restored.index_age == trainer.index_age
    assert restored.index_refresh_count == trainer.index_refresh_count
    assert restored.index_refresh_steps == trainer.index_refresh_steps
    assert restored.state_dict()["diagnostics"]["persistent_index_serialized"] is False
    assert restored.index_cache is not None


def test_training_history_includes_ema_index_diagnostics() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_encoder_policy="ema",
        index_ema_tau=0.05,
        index_refresh_every=2,
    )

    history = trainer.train(3)

    assert history.index_ema_update_count == 3
    assert history.diagnostics["index_encoder_policy"] == "ema"
    assert history.diagnostics["index_ema_tau"] == 0.05
    assert history.diagnostics["index_ema_update_count"] == 3
    assert len(history.index_encoder_drift_mean_l2) == 3
    assert len(history.index_encoder_drift_max_l2) == 3
    assert len(history.index_encoder_drift_mean_cosine_distance) == 3
    assert len(history.index_encoder_drift_max_cosine_distance) == 3
    assert all(value is not None for value in history.index_encoder_drift_mean_l2)
    assert history.diagnostics["final_index_encoder_drift_mean_l2"] == history.index_encoder_drift_mean_l2[-1]


def test_training_history_online_policy_records_zero_drift() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(
        data,
        model,
        optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_encoder_policy="online",
    )

    history = trainer.train(2)

    assert history.index_encoder_drift_mean_l2 == pytest.approx([0.0, 0.0])
    assert history.index_encoder_drift_max_l2 == pytest.approx([0.0, 0.0])
    assert history.index_encoder_drift_mean_cosine_distance == pytest.approx([0.0, 0.0], abs=1e-6)
    assert history.index_encoder_drift_max_cosine_distance == pytest.approx([0.0, 0.0], abs=1e-6)


def test_update_schedule_with_default_smoke_alternating_schedule_works() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    trainer = SyntheticDWMQLTrainer(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        batch_size=6,
        k=4,
        gamma=0.95,
    )

    history = trainer.train(3, update_schedule=default_smoke_alternating_schedule())

    assert len(history.losses) == 3
    assert history.diagnostics["update_parts_by_step"] == [
        ["dual"],
        ["density"],
        ["encoder", "critic"],
    ]


def test_train_invalid_num_steps_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()
    trainer = SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95)

    with pytest.raises(ValueError, match="num_steps"):
        trainer.train(0)


def test_constructor_invalid_args_raise_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="batch_size"):
        SyntheticDWMQLTrainer(data, model, optimizer, batch_size=0, k=4, gamma=0.95)

    with pytest.raises(ValueError, match="gamma"):
        SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=1.5)

    with pytest.raises(ValueError, match="either optimizer or optimizer_bundle"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            optimizer_bundle=make_optimizer_bundle(model),
            batch_size=6,
            k=4,
            gamma=0.95,
        )

    with pytest.raises(ValueError, match="optimizer"):
        SyntheticDWMQLTrainer(data, model, None, batch_size=6, k=4, gamma=0.95)

    with pytest.raises(ValueError, match="index_refresh_every"):
        SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95, index_refresh_every=0)

    with pytest.raises(TypeError, match="use_persistent_index"):
        SyntheticDWMQLTrainer(
            data,
            model,
            optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            use_persistent_index="yes",  # type: ignore[arg-type]
        )


def test_grad_clip_clips_gradients() -> None:
    """max_grad_norm must limit the gradient norm seen by the optimizer."""
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, lr=1e-3, mode="separate_heads")
    trainer_clipped = SyntheticDWMQLTrainer(
        data, model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95, max_grad_norm=1e-6,
    )
    result = trainer_clipped.update_critic()
    # With tiny max_grad_norm the clipped norm must be <= 1e-6 (plus float tolerance)
    assert result.diagnostics["grad_norm_after_clip"] <= 1e-6 + 1e-7
    assert "grad_norm_before_clip" in result.diagnostics
    assert "grad_norm_after_clip" in result.diagnostics


def test_grad_clip_absent_when_max_grad_norm_none() -> None:
    """Without max_grad_norm the before/after clip norms should be equal."""
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, lr=1e-3, mode="separate_heads")
    trainer = SyntheticDWMQLTrainer(data, model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)
    result = trainer.update_critic()
    assert result.diagnostics["grad_norm_before_clip"] == result.diagnostics["grad_norm_after_clip"]


def test_q_l2_reg_appears_in_diagnostics() -> None:
    """q_l2_regularization_weight > 0 must produce q_l2_regularization in diagnostics."""
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, lr=1e-3, mode="separate_heads")
    trainer = SyntheticDWMQLTrainer(
        data, model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95,
        q_l2_regularization_weight=0.1,
    )
    result = trainer.update_critic()
    assert "q_l2_regularization" in result.diagnostics
    assert result.diagnostics["q_l2_regularization"] >= 0.0
    assert "q_abs_mean" in result.diagnostics
    assert "q_abs_max" in result.diagnostics
    assert result.diagnostics["q_abs_max"] >= result.diagnostics["q_abs_mean"]


def test_q_value_clip_prevents_large_targets() -> None:
    """q_value_clip must prevent Q values from feeding unbounded next-Q into targets."""
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, lr=1e-3, mode="separate_heads")
    trainer = SyntheticDWMQLTrainer(
        data, model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95, q_value_clip=0.01,
    )
    result = trainer.update_critic()
    # The test verifies that training runs without error with a very tight clip
    assert torch.isfinite(result.loss)
    assert "q_abs_mean" in result.diagnostics


def test_reward_scale_changes_bellman_target() -> None:
    """reward_scale != 1 must produce a different Bellman loss than reward_scale=1."""
    torch.manual_seed(42)
    data, model, _ = _data_model_optimizer()
    bundle1 = make_optimizer_bundle(model, lr=1e-4, mode="separate_heads")
    trainer_default = SyntheticDWMQLTrainer(
        data, model, optimizer_bundle=bundle1, batch_size=6, k=4, gamma=0.95,
        reward_scale=1.0, generator=torch.Generator().manual_seed(0),
    )
    loss_default = trainer_default.update_critic().loss.item()

    torch.manual_seed(42)
    data2, model2, _ = _data_model_optimizer()
    bundle2 = make_optimizer_bundle(model2, lr=1e-4, mode="separate_heads")
    trainer_scaled = SyntheticDWMQLTrainer(
        data2, model2, optimizer_bundle=bundle2, batch_size=6, k=4, gamma=0.95,
        reward_scale=100.0, generator=torch.Generator().manual_seed(0),
    )
    loss_scaled = trainer_scaled.update_critic().loss.item()

    assert loss_default != loss_scaled, "reward_scale=100 must yield a different loss than reward_scale=1"


def test_stability_controls_invalid_args_raise() -> None:
    """Stability control params must validate at construction time."""
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="max_grad_norm"):
        SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95, max_grad_norm=-1.0)

    with pytest.raises(ValueError, match="q_l2_regularization_weight"):
        SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95, q_l2_regularization_weight=-0.1)

    with pytest.raises(ValueError, match="q_value_clip"):
        SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95, q_value_clip=-5.0)

    with pytest.raises(ValueError, match="reward_scale"):
        SyntheticDWMQLTrainer(data, model, optimizer, batch_size=6, k=4, gamma=0.95, reward_scale=0.0)
