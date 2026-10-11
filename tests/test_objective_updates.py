"""Tests for explicit synthetic objective update primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import DWMQLObjectiveConfig, ModelConfig, SyntheticDatasetConfig, validate_dwmql_objective_config
from retrieval_alignment.networks import build_networks
from retrieval_alignment.synthetic import make_synthetic_offline_data
from retrieval_alignment.target_networks import clone_target_network
from retrieval_alignment.trainer import (
    DWMQLBatchTerms,
    DWMQLStepResult,
    ObjectiveUpdateResult,
    SyntheticDWMQLTrainer,
    SyntheticTrainingBatchTerms,
    compute_dwmql_batch_terms,
    compute_synthetic_training_batch_terms,
    make_optimizer_bundle,
)


def _data_model_bundle(seed: int = 1234):
    data = make_synthetic_offline_data(
        SyntheticDatasetConfig(
            num_transitions=48,
            state_dim=4,
            action_dim=3,
            num_actions=12,
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
    bundle = make_optimizer_bundle(model, lr=1e-3, mode="separate_heads")
    return data, model, bundle


def _params(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module) -> bool:
    return any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, module.parameters(), strict=True)
    )


def test_compute_synthetic_training_batch_terms_returns_finite_terms() -> None:
    data, model, _ = _data_model_bundle()

    terms = compute_synthetic_training_batch_terms(
        data=data,
        model=model,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(5),
    )

    assert isinstance(terms, SyntheticTrainingBatchTerms)
    assert terms.current_q.shape == (6,)
    assert terms.next_candidate_q.shape[0] == 6
    assert terms.candidate_probs.shape == terms.next_candidate_q.shape
    assert terms.bellman_residual.shape == (6,)
    for scalar in (terms.bellman_loss, terms.stationarity_loss, terms.density_regularization):
        assert scalar.ndim == 0
        assert torch.isfinite(scalar)


def test_update_critic_changes_q_head_parameters() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)
    before = _params(model.q_head)

    result = trainer.update_critic(update_encoder=False)

    assert isinstance(result, ObjectiveUpdateResult)
    assert result.objective_name == "critic"
    assert torch.isfinite(result.loss)
    assert _changed(before, model.q_head)
    assert result.diagnostics["loss_source"] == "bellman_loss_plus_bellman_moment"


def test_update_density_changes_density_head_parameters() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)
    before = _params(model.density_head)

    result = trainer.update_density()

    assert result.objective_name == "density"
    assert torch.isfinite(result.loss)
    assert _changed(before, model.density_head)
    assert result.diagnostics["loss_source"] == "stationarity_loss_plus_density_regularization"


def test_update_dual_changes_dual_head_parameters_and_records_ascent_sign() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)
    before = _params(model.dual_head)

    result = trainer.update_dual()

    assert result.objective_name == "dual"
    assert torch.isfinite(result.loss)
    assert _changed(before, model.dual_head)
    assert result.diagnostics["dual_update_sign"] == "ascent_via_minimize_negative_stationarity_loss"
    assert result.diagnostics["maximized_quantity"] == "stationarity_loss"


def test_train_objective_schedule_runs_multiple_steps() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.train_objective_schedule(5)

    assert result["num_steps"] == 5
    assert result["schedule"] == ["dual", "density", "critic"]
    assert result["objective_names"] == ["dual", "density", "critic", "dual", "density"]
    assert len(result["losses"]) == 5
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result["losses"])
    assert all(isinstance(item, ObjectiveUpdateResult) for item in result["results"])


def test_objective_schedule_final_critic_diagnostics_ignore_trailing_steps() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        batch_size=6,
        k=4,
        gamma=0.95,
    )

    result = trainer.train_objective_schedule(5)
    critic_diagnostics = result["results"][2].diagnostics

    assert result["objective_names"] == ["dual", "density", "critic", "dual", "density"]
    assert result["diagnostics"]["final_raw_bellman_loss"] == pytest.approx(
        critic_diagnostics["raw_bellman_loss"]
    )
    assert result["diagnostics"]["final_support_weighted_bellman_loss"] == pytest.approx(
        critic_diagnostics["support_weighted_bellman_loss"]
    )
    assert result["diagnostics"]["final_critic_total_loss"] == pytest.approx(
        critic_diagnostics["critic_total_loss"]
    )


def test_train_objective_schedule_records_persistent_index_refresh_count() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        index_refresh_every=2,
    )

    result = trainer.train_objective_schedule(5)

    assert result["index_refresh_count"] == 2
    assert result["index_refresh_steps"] == [2, 4]
    assert result["index_ages"] == [1, 0, 1, 0, 1]
    assert result["diagnostics"]["persistent_index_used"] is True
    assert result["diagnostics"]["final_index_age"] == 1


def test_train_objective_schedule_records_stale_index_diagnostics() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        batch_size=6,
        k=4,
        gamma=0.95,
        use_persistent_index=True,
        audit_stale_index=True,
        index_refresh_every=2,
    )

    result = trainer.train_objective_schedule(3)

    assert len(result["stale_index_recall_means"]) == 3
    assert len(result["stale_index_overlap_fraction_means"]) == 3
    assert len(result["stale_index_best_value_gap_means"]) == 3
    for value in result["stale_index_recall_means"] + result["stale_index_overlap_fraction_means"]:
        assert value is not None
        assert 0.0 <= value <= 1.0
    assert result["results"][-1].diagnostics["stale_index_recall_mean"] is not None


def test_train_objective_schedule_records_encoder_policy_overlap_diagnostics() -> None:
    data, model, bundle = _data_model_bundle()
    target_model = clone_target_network(model)
    trainer = SyntheticDWMQLTrainer(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_model,
        target_tau=0.5,
        use_persistent_index=True,
        audit_encoder_policy_overlap=True,
        index_encoder_policy="ema",
        index_ema_tau=0.5,
    )

    result = trainer.train_objective_schedule(3)

    assert len(result["ema_vs_online_recall_means"]) == 3
    assert len(result["target_vs_online_recall_means"]) == 3
    assert len(result["ema_vs_target_recall_means"]) == 3
    for key in (
        "final_ema_vs_online_recall_mean",
        "final_target_vs_online_recall_mean",
        "final_ema_vs_target_recall_mean",
    ):
        value = result["diagnostics"][key]
        assert value is not None
        assert 0.0 <= value <= 1.0
    assert result["diagnostics"]["index_ema_update_count"] == 3


def test_train_and_objective_schedule_share_post_update_bookkeeping() -> None:
    data_a, model_a, bundle_a = _data_model_bundle(seed=99)
    target_a = clone_target_network(model_a)
    trainer_a = SyntheticDWMQLTrainer(
        data=data_a,
        model=model_a,
        optimizer_bundle=bundle_a,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_a,
        target_tau=0.5,
        target_update_every=1,
        use_persistent_index=True,
        audit_stale_index=True,
        audit_encoder_policy_overlap=True,
        index_encoder_policy="ema",
        index_ema_tau=0.5,
        index_refresh_every=2,
    )

    data_b, model_b, bundle_b = _data_model_bundle(seed=100)
    target_b = clone_target_network(model_b)
    trainer_b = SyntheticDWMQLTrainer(
        data=data_b,
        model=model_b,
        optimizer_bundle=bundle_b,
        batch_size=6,
        k=4,
        gamma=0.95,
        target_model=target_b,
        target_tau=0.5,
        target_update_every=1,
        use_persistent_index=True,
        audit_stale_index=True,
        audit_encoder_policy_overlap=True,
        index_encoder_policy="ema",
        index_ema_tau=0.5,
        index_refresh_every=2,
    )

    train_history = trainer_a.train(3, update_schedule=[["dual"], ["density"], ["critic"]])
    objective_history = trainer_b.train_objective_schedule(3)

    assert train_history.diagnostics["final_step"] == objective_history["diagnostics"]["final_step"] == 3
    assert train_history.diagnostics["target_update_count"] == objective_history["target_update_count"] == 3
    assert train_history.index_refresh_count == objective_history["index_refresh_count"] == 1
    assert train_history.index_refresh_steps == objective_history["index_refresh_steps"] == [2]
    assert train_history.index_ages == objective_history["index_ages"] == [1, 0, 1]
    assert train_history.index_ema_update_count == objective_history["index_ema_update_count"] == 3
    assert len(train_history.stale_index_recall_means) == len(objective_history["stale_index_recall_means"]) == 3
    assert train_history.stale_index_recall_means[-1] is not None
    assert objective_history["stale_index_recall_means"][-1] is not None
    assert train_history.ema_vs_online_recall_means[-1] is not None
    assert objective_history["ema_vs_online_recall_means"][-1] is not None
    assert train_history.index_encoder_drift_mean_l2[-1] is not None
    assert objective_history["index_encoder_drift_mean_l2"][-1] is not None


def test_train_objective_step_invalid_name_raises_value_error() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    with pytest.raises(ValueError, match="critic"):
        trainer.train_objective_step("unknown")


def test_objective_updates_require_optimizer_bundle() -> None:
    data, model, _ = _data_model_bundle()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer=optimizer, batch_size=6, k=4, gamma=0.95)

    with pytest.raises(ValueError, match="optimizer_bundle"):
        trainer.update_critic()


def test_batch_terms_includes_bellman_moment_loss() -> None:
    data, model, _ = _data_model_bundle()

    terms = compute_synthetic_training_batch_terms(
        data=data,
        model=model,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(7),
    )

    assert isinstance(terms, SyntheticTrainingBatchTerms)
    assert hasattr(terms, "bellman_moment_loss")
    assert terms.bellman_moment_loss.ndim == 0
    assert torch.isfinite(terms.bellman_moment_loss)
    assert terms.bellman_moment_loss.item() >= 0.0
    assert "bellman_moment" in terms.diagnostics
    assert "bellman_moment_loss" in terms.diagnostics


def test_update_critic_with_zero_bellman_moment_weight_uses_bellman_only() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(
        data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95, bellman_moment_weight=0.0
    )

    result = trainer.update_critic(update_encoder=False)

    assert result.diagnostics["loss_source"] == "bellman_loss"
    assert result.diagnostics["bellman_moment_weight"] == 0.0


def test_update_critic_with_positive_bellman_moment_weight_includes_moment_term() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(
        data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95, bellman_moment_weight=2.0
    )

    result = trainer.update_critic(update_encoder=False)

    assert result.diagnostics["loss_source"] == "bellman_loss_plus_bellman_moment"
    assert result.diagnostics["bellman_moment_weight"] == 2.0


def test_update_dual_density_head_receives_no_gradient() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.update_dual()

    grad_norms = result.diagnostics["gradient_group_norms"]
    assert grad_norms["density"] == 0.0
    assert result.diagnostics["density_weights_detached_for_stationarity"] is True


def test_update_density_dual_head_receives_no_gradient() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.update_density()

    grad_norms = result.diagnostics["gradient_group_norms"]
    assert grad_norms["dual"] == 0.0
    assert result.diagnostics["dual_detached_for_stationarity"] is True


def test_update_dual_records_density_weights_detached_diagnostic() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.update_dual()

    assert result.diagnostics.get("detach_density_weights_for_stationarity") is True
    assert result.diagnostics.get("density_weights_detached_for_stationarity") is True


def test_update_density_records_dual_detached_diagnostic() -> None:
    data, model, bundle = _data_model_bundle()
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.update_density()

    assert result.diagnostics.get("detach_dual_for_stationarity") is True
    assert result.diagnostics.get("dual_detached_for_stationarity") is True


# ---------------------------------------------------------------------------
# Part F: DWMQLObjectiveConfig validation
# ---------------------------------------------------------------------------


def test_validate_dwmql_objective_config_accepts_valid_defaults() -> None:
    config = DWMQLObjectiveConfig()
    result = validate_dwmql_objective_config(config)
    assert result is config


def test_validate_dwmql_objective_config_rejects_negative_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        validate_dwmql_objective_config(DWMQLObjectiveConfig(gamma=-0.1))


def test_validate_dwmql_objective_config_rejects_negative_bellman_weight() -> None:
    with pytest.raises(ValueError, match="bellman_weight"):
        validate_dwmql_objective_config(DWMQLObjectiveConfig(bellman_weight=-1.0))


def test_validate_dwmql_objective_config_rejects_zero_critic_steps() -> None:
    with pytest.raises(ValueError, match="critic_steps"):
        validate_dwmql_objective_config(DWMQLObjectiveConfig(critic_steps=0))


def test_validate_dwmql_objective_config_rejects_negative_extragradient_step_size() -> None:
    with pytest.raises(ValueError, match="extragradient_step_size"):
        validate_dwmql_objective_config(DWMQLObjectiveConfig(extragradient_step_size=0.0))


# ---------------------------------------------------------------------------
# Part F: compute_dwmql_batch_terms
# ---------------------------------------------------------------------------


def test_compute_dwmql_batch_terms_returns_finite_outputs() -> None:
    data, model, _ = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)

    terms = compute_dwmql_batch_terms(
        data=data,
        model=model,
        obj_config=obj_config,
        batch_size=6,
        k=4,
        generator=torch.Generator().manual_seed(42),
    )

    assert isinstance(terms, DWMQLBatchTerms)
    assert terms.bellman_loss.ndim == 0
    assert torch.isfinite(terms.bellman_loss)
    assert terms.stationarity_loss.ndim == 0
    assert torch.isfinite(terms.stationarity_loss)
    assert terms.density_regularization.ndim == 0
    assert torch.isfinite(terms.density_regularization)
    assert terms.ess.ndim == 0
    assert torch.isfinite(terms.ess)
    assert terms.ess.item() > 0.0
    assert terms.weight_entropy.ndim == 0
    assert torch.isfinite(terms.weight_entropy)


def test_compute_dwmql_batch_terms_ess_is_detached_tensor() -> None:
    data, model, _ = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)

    terms = compute_dwmql_batch_terms(
        data=data,
        model=model,
        obj_config=obj_config,
        batch_size=6,
        k=4,
    )

    assert isinstance(terms.ess, torch.Tensor)
    assert isinstance(terms.weight_entropy, torch.Tensor)
    assert not terms.ess.requires_grad
    assert not terms.weight_entropy.requires_grad


# ---------------------------------------------------------------------------
# Part F: DW-MQL update methods
# ---------------------------------------------------------------------------


def test_update_dwmql_critic_changes_q_head_parameters() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)
    before = _params(model.q_head)

    result = trainer.update_dwmql_critic(obj_config)

    assert isinstance(result, ObjectiveUpdateResult)
    assert result.objective_name == "critic"
    assert torch.isfinite(result.loss)
    assert _changed(before, model.q_head)


def test_update_dwmql_density_changes_density_head_parameters() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)
    before = _params(model.density_head)

    result = trainer.update_dwmql_density(obj_config)

    assert isinstance(result, ObjectiveUpdateResult)
    assert result.objective_name == "density"
    assert torch.isfinite(result.loss)
    assert _changed(before, model.density_head)


def test_update_dwmql_dual_changes_dual_head_parameters() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)
    before = _params(model.dual_head)

    result = trainer.update_dwmql_dual(obj_config)

    assert isinstance(result, ObjectiveUpdateResult)
    assert result.objective_name == "dual"
    assert torch.isfinite(result.loss)
    assert _changed(before, model.dual_head)


def test_update_dwmql_density_dual_head_receives_no_gradient() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.update_dwmql_density(obj_config)

    grad_norms = result.diagnostics.get("gradient_group_norms", {})
    assert grad_norms.get("dual", 0.0) == 0.0


def test_update_dwmql_dual_density_head_receives_no_gradient() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.update_dwmql_dual(obj_config)

    grad_norms = result.diagnostics.get("gradient_group_norms", {})
    assert grad_norms.get("density", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Part F: train_dwmql_step and train_dwmql_schedule
# ---------------------------------------------------------------------------


def test_train_dwmql_step_returns_retrieval_alignment_step_result() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.train_dwmql_step(obj_config)

    assert isinstance(result, DWMQLStepResult)
    assert torch.isfinite(torch.tensor(result.bellman_loss))
    assert torch.isfinite(torch.tensor(result.stationarity_loss))
    assert torch.isfinite(torch.tensor(result.ess))
    assert result.ess > 0.0
    assert result.extragradient_used is False


def test_train_dwmql_step_extragradient_mode_sets_flag() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95, extragradient=True, extragradient_step_size=0.01)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.train_dwmql_step(obj_config)

    assert isinstance(result, DWMQLStepResult)
    assert result.extragradient_used is True
    assert result.extragradient_groups == ["dual", "density"]
    assert torch.isfinite(torch.tensor(result.bellman_loss))


def test_train_dwmql_schedule_returns_finite_losses() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.train_dwmql_schedule(4, obj_config)

    assert result["num_steps"] == 4
    assert len(result["bellman_losses"]) == 4
    assert len(result["stationarity_losses"]) == 4
    assert len(result["ess_values"]) == 4
    assert all(torch.isfinite(torch.tensor(v)) for v in result["bellman_losses"])
    assert all(torch.isfinite(torch.tensor(v)) for v in result["stationarity_losses"])
    assert result["benchmark_claim"] is False


def test_train_dwmql_schedule_extragradient_runs_without_error() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95, extragradient=True, extragradient_step_size=0.01)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.train_dwmql_schedule(3, obj_config)

    assert result["num_steps"] == 3
    assert result["extragradient_used"] is True
    assert result["extragradient_groups"] == ["dual", "density"]
    assert all(torch.isfinite(torch.tensor(v)) for v in result["bellman_losses"])


def test_train_dwmql_schedule_requires_optimizer_bundle() -> None:
    data, model, _ = _data_model_bundle()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer=optimizer, batch_size=6, k=4, gamma=0.95)

    with pytest.raises(ValueError, match="optimizer_bundle"):
        trainer.train_dwmql_schedule(2, obj_config)


def test_train_dwmql_schedule_invalid_num_steps_raises_value_error() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    with pytest.raises(ValueError, match="num_steps"):
        trainer.train_dwmql_schedule(0, obj_config)


def test_train_dwmql_schedule_with_support_correction_weight_runs() -> None:
    data, model, bundle = _data_model_bundle()
    obj_config = DWMQLObjectiveConfig(gamma=0.95, support_correction_weight=0.1)
    trainer = SyntheticDWMQLTrainer(data=data, model=model, optimizer_bundle=bundle, batch_size=6, k=4, gamma=0.95)

    result = trainer.train_dwmql_schedule(2, obj_config)

    assert result["num_steps"] == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in result["bellman_losses"])
