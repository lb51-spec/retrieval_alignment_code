from __future__ import annotations

import numpy as np

from retrieval_alignment.known_bandit import (
    KnownBanditConfig,
    fit_bilinear_critic,
    generate_known_bandit,
    ope_metrics,
    policy_from_candidates,
    retrieval_candidates,
    run_known_bandit_seed,
)


def test_known_bandit_has_full_support_propensities() -> None:
    train, evaluation = generate_known_bandit(KnownBanditConfig(num_train=50, num_eval=20), seed=3)
    assert train.propensities.min() > 0
    assert evaluation.propensities.min() > 0
    assert train.all_rewards.shape == (50, 64)


def test_aligned_retrieval_has_perfect_critic_topk_overlap() -> None:
    config = KnownBanditConfig(num_train=100, num_eval=20, ks=(5,))
    train, evaluation = generate_known_bandit(config, seed=4)
    critic = fit_bilinear_critic(train, config.ridge)
    candidates = retrieval_candidates(
        evaluation.contexts, evaluation.action_features, critic, 5, aligned=True
    )
    values = np.einsum("nd,df,af->na", evaluation.contexts, critic, evaluation.action_features)
    topk = np.argsort(-values, axis=1)[:, :5]
    assert np.array_equal(candidates, topk)


def test_ips_and_snips_are_finite() -> None:
    config = KnownBanditConfig(num_train=100, num_eval=100, bootstrap_samples=20)
    train, evaluation = generate_known_bandit(config, seed=5)
    critic = fit_bilinear_critic(train, config.ridge)
    values = np.einsum("nd,df,af->na", evaluation.contexts, critic, evaluation.action_features)
    policy = policy_from_candidates(values, None, config.target_temperature)
    metrics = ope_metrics(policy, evaluation, bootstrap_samples=20, seed=5)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["ess"] > 0


def test_known_bandit_report_contains_value_and_ope_metrics() -> None:
    report = run_known_bandit_seed(
        KnownBanditConfig(num_train=100, num_eval=100, ks=(5,), bootstrap_samples=20), 1
    )
    assert report["known_propensities"] is True
    assert report["policies"]["exact"]["true_value"] > 0
    assert "ips" in report["policies"]["b2_k5_exact_ef0"]
    assert "snips_ci_low" in report["policies"]["b2_k5_hnsw_ef16"]
    assert 0.0 <= report["policies"]["base_k5_exact_ef0"]["pmr"] <= 1.0
