"""Controlled contextual-bandit validation with exact behavior propensities.

This module is intentionally independent of the MovieLens/KuaiRand loaders. It
provides ground-truth rewards and a known logging policy so IPS/SNIPS can be
checked without fitting an unvalidated propensity model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KnownBanditConfig:
    num_train: int = 10_000
    num_eval: int = 2_000
    num_actions: int = 64
    context_dim: int = 8
    action_dim: int = 8
    behavior_temperature: float = 1.0
    target_temperature: float = 0.25
    reward_noise_std: float = 0.05
    ridge: float = 1e-2
    bootstrap_samples: int = 500
    ks: tuple[int, ...] = (5, 10, 20)
    projection_dim: int = 4
    projection_steps: int = 250
    projection_lr: float = 0.05
    hnsw_ef_search_values: tuple[int, ...] = (16, 64)


@dataclass
class KnownBanditData:
    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    all_rewards: np.ndarray
    action_features: np.ndarray
    true_matrix: np.ndarray


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    values = np.exp(np.clip(scaled, -60.0, 60.0))
    return values / np.sum(values, axis=1, keepdims=True)


def _validate_config(config: KnownBanditConfig) -> None:
    for name in ("num_train", "num_eval", "num_actions", "context_dim", "action_dim"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.reward_noise_std < 0 or config.ridge < 0:
        raise ValueError("reward_noise_std and ridge must be nonnegative")
    if config.projection_dim <= 0 or config.projection_steps <= 0 or config.projection_lr <= 0:
        raise ValueError("projection settings must be positive")
    if not config.ks or any(k <= 0 or k > config.num_actions for k in config.ks):
        raise ValueError("ks must contain valid positive candidate sizes")
    if not config.hnsw_ef_search_values or any(v <= 0 for v in config.hnsw_ef_search_values):
        raise ValueError("hnsw_ef_search_values must be positive")


def _all_rewards(contexts: np.ndarray, actions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    scores = np.einsum("nd,df,af->na", contexts, matrix, actions)
    return np.tanh(scores)


def generate_known_bandit(
    config: KnownBanditConfig, seed: int
) -> tuple[KnownBanditData, KnownBanditData]:
    """Generate independent training and evaluation logs with known propensities."""

    _validate_config(config)
    rng = np.random.default_rng(seed)
    actions = rng.normal(size=(config.num_actions, config.action_dim))
    actions /= np.maximum(np.linalg.norm(actions, axis=1, keepdims=True), 1e-8)
    matrix = rng.normal(
        scale=1.0 / np.sqrt(config.context_dim),
        size=(config.context_dim, config.action_dim),
    )

    def sample(n: int) -> KnownBanditData:
        contexts = rng.normal(size=(n, config.context_dim))
        rewards_all = _all_rewards(contexts, actions, matrix)
        behavior = _softmax(rewards_all, config.behavior_temperature)
        sampled = np.array(
            [rng.choice(config.num_actions, p=row) for row in behavior], dtype=np.int64
        )
        propensities = behavior[np.arange(n), sampled]
        noise = rng.normal(scale=config.reward_noise_std, size=n)
        rewards = rewards_all[np.arange(n), sampled] + noise
        return KnownBanditData(
            contexts=contexts,
            actions=sampled,
            rewards=rewards,
            propensities=propensities,
            all_rewards=rewards_all,
            action_features=actions,
            true_matrix=matrix,
        )

    return sample(config.num_train), sample(config.num_eval)


def fit_bilinear_critic(data: KnownBanditData, ridge: float) -> np.ndarray:
    """Fit a bilinear reward critic using ridge regression on logged rows."""

    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    features = np.einsum("nd,nf->ndf", data.contexts, data.action_features[data.actions]).reshape(
        data.contexts.shape[0], -1
    )
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    weights = np.linalg.solve(gram, features.T @ data.rewards)
    return weights.reshape(data.contexts.shape[1], data.action_features.shape[1])


def critic_values(contexts: np.ndarray, actions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("nd,df,af->na", contexts, matrix, actions)


def fit_b2_projection(
    data: KnownBanditData,
    critic_matrix: np.ndarray,
    projection_dim: int,
    steps: int,
    learning_rate: float,
    seed: int,
    teacher_temperature: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit detached query/action projections by listwise-KL distillation."""

    import torch

    torch.manual_seed(seed)
    contexts = torch.as_tensor(data.contexts, dtype=torch.float32)
    actions = torch.as_tensor(data.action_features, dtype=torch.float32)
    teacher = torch.as_tensor(
        critic_values(data.contexts, data.action_features, critic_matrix), dtype=torch.float32
    )
    q_u, singular, v_h = np.linalg.svd(critic_matrix, full_matrices=False)
    dim = min(projection_dim, singular.shape[0])
    q_init = q_u[:, :dim] * np.sqrt(singular[:dim])[None, :]
    a_init = v_h[:dim, :].T * np.sqrt(singular[:dim])[None, :]
    q_weight = torch.nn.Parameter(torch.as_tensor(q_init, dtype=torch.float32))
    a_weight = torch.nn.Parameter(torch.as_tensor(a_init, dtype=torch.float32))
    optimizer = torch.optim.Adam((q_weight, a_weight), lr=learning_rate)
    teacher_log_probs = torch.log_softmax(teacher / teacher_temperature, dim=1).detach()
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(steps):
        student = (contexts @ q_weight) @ (actions @ a_weight).T
        student_log_probs = torch.log_softmax(student / teacher_temperature, dim=1)
        loss = torch.nn.functional.kl_div(
            student_log_probs, teacher_log_probs.exp(), reduction="batchmean", log_target=False
        )
        if step == 0:
            initial_loss = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return (
        q_weight.detach().cpu().numpy(),
        a_weight.detach().cpu().numpy(),
        {"initial_listwise_kl": initial_loss, "final_listwise_kl": final_loss},
    )


def unaligned_query_matrix(context_dim: int, action_dim: int) -> np.ndarray:
    """Return a deterministic state encoder with no critic-ranking objective."""

    rng = np.random.default_rng(17_071)
    return rng.normal(size=(context_dim, action_dim)) / np.sqrt(context_dim)


def retrieve_with_embeddings(
    queries: np.ndarray,
    keys: np.ndarray,
    k: int,
    backend: str = "exact",
    ef_search: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retrieve candidates using exact NumPy search or FAISS HNSW."""

    if k <= 0 or k > keys.shape[0]:
        raise ValueError("k must be in [1, num_actions]")
    import time

    start = time.perf_counter()
    if backend == "exact":
        candidates = np.argsort(-(queries @ keys.T), axis=1, kind="stable")[:, :k]
        used_backend = "exact"
    elif backend == "hnsw":
        try:
            import faiss
        except ModuleNotFoundError:
            candidates = np.argsort(-(queries @ keys.T), axis=1, kind="stable")[:, :k]
            used_backend = "exact_fallback_no_faiss"
        else:
            index = faiss.IndexHNSWFlat(keys.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = int(ef_search)
            index.add(np.asarray(keys, dtype=np.float32))
            _, candidates = index.search(np.asarray(queries, dtype=np.float32), k)
            used_backend = "faiss_hnsw"
    else:
        raise ValueError("backend must be exact or hnsw")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return candidates.astype(np.int64), {
        "backend": used_backend,
        "ef_search": int(ef_search),
        "query_latency_ms": float(elapsed_ms),
    }


def mask_policy(policy: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Restrict and renormalize one fixed target policy to candidate support."""

    masked = np.zeros_like(policy)
    rows = np.arange(policy.shape[0])[:, None]
    masked[rows, candidates] = policy[rows, candidates]
    normalizer = np.sum(masked, axis=1, keepdims=True)
    if np.any(normalizer <= 0):
        raise ValueError("candidate support has zero target-policy mass")
    return masked / normalizer


def retrieval_candidates(
    contexts: np.ndarray,
    action_features: np.ndarray,
    critic_matrix: np.ndarray,
    k: int,
    aligned: bool,
) -> np.ndarray:
    """Return top-k candidates from aligned or deliberately unaligned keys."""

    query = contexts @ critic_matrix
    if aligned:
        keys = action_features
    else:
        rng = np.random.default_rng(17_071)
        rotation = rng.normal(size=(action_features.shape[1], action_features.shape[1]))
        keys = action_features @ rotation / np.sqrt(action_features.shape[1])
    scores = query @ keys.T
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def policy_from_candidates(
    values: np.ndarray, candidates: np.ndarray | None, temperature: float
) -> np.ndarray:
    logits = values.copy()
    if candidates is not None:
        mask = np.ones_like(logits, dtype=bool)
        rows = np.arange(logits.shape[0])[:, None]
        mask[rows, candidates] = False
        logits[mask] = -np.inf
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probs = np.exp(np.where(np.isfinite(scaled), np.clip(scaled, -60.0, 60.0), -np.inf))
    return probs / np.sum(probs, axis=1, keepdims=True)


def policy_metrics(
    policy: np.ndarray,
    true_rewards: np.ndarray,
    critic_values_all: np.ndarray,
    candidates: np.ndarray | None,
    reference_policy: np.ndarray | None = None,
) -> dict[str, float]:
    width = candidates.shape[1] if candidates is not None else 20
    greedy = np.argsort(-critic_values_all, axis=1, kind="stable")[:, :width]
    overlap = 0.0
    pmr = 0.0
    if candidates is not None:
        overlap = float(
            np.mean(
                [
                    len(set(a).intersection(b)) / len(a)
                    for a, b in zip(greedy, candidates, strict=True)
                ]
            )
        )
        reference = policy if reference_policy is None else reference_policy
        pmr = float(
            np.mean(np.sum(reference[np.arange(reference.shape[0])[:, None], candidates], axis=1))
        )
    return {
        "true_value": float(np.mean(np.sum(policy * true_rewards, axis=1))),
        "critic_topk_overlap": overlap,
        "pmr": pmr,
    }


def ope_metrics(
    policy: np.ndarray, data: KnownBanditData, bootstrap_samples: int, seed: int
) -> dict[str, float]:
    """Compute IPS, SNIPS, ESS, and a percentile bootstrap interval."""

    n = data.actions.shape[0]
    target = policy[np.arange(n), data.actions]
    weights = target / np.maximum(data.propensities, 1e-12)
    ips_terms = weights * data.rewards
    ips = float(np.mean(ips_terms))
    snips = float(np.sum(ips_terms) / np.maximum(np.sum(weights), 1e-12))
    ess = float(np.sum(weights) ** 2 / np.maximum(np.sum(weights**2), 1e-12))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(max(bootstrap_samples, 1), n))
    boot_ips = np.mean(ips_terms[draws], axis=1)
    sampled_weights = weights[draws]
    sampled_terms = ips_terms[draws]
    boot_snips = np.sum(sampled_terms, axis=1) / np.maximum(
        np.sum(sampled_weights, axis=1), 1e-12
    )
    return {
        "ips": ips,
        "snips": snips,
        "ips_ci_low": float(np.percentile(boot_ips, 2.5)),
        "ips_ci_high": float(np.percentile(boot_ips, 97.5)),
        "snips_ci_low": float(np.percentile(boot_snips, 2.5)),
        "snips_ci_high": float(np.percentile(boot_snips, 97.5)),
        "ess": ess,
        "max_weight": float(np.max(weights)),
    }


def run_known_bandit_seed(config: KnownBanditConfig, seed: int) -> dict[str, Any]:
    train, evaluation = generate_known_bandit(config, seed)
    critic = fit_bilinear_critic(train, config.ridge)
    values_eval = critic_values(evaluation.contexts, evaluation.action_features, critic)
    exact_policy = policy_from_candidates(values_eval, None, config.target_temperature)
    b2_query_weight, b2_action_weight, projection_metrics = fit_b2_projection(
        train,
        critic,
        config.projection_dim,
        config.projection_steps,
        config.projection_lr,
        seed + 7000,
    )
    b2_queries = evaluation.contexts @ b2_query_weight
    b2_keys = evaluation.action_features @ b2_action_weight
    base_query_matrix = unaligned_query_matrix(
        evaluation.contexts.shape[1], evaluation.action_features.shape[1]
    )
    base_queries = evaluation.contexts @ base_query_matrix
    base_keys = evaluation.action_features
    report: dict[str, Any] = {
        "seed": seed,
        "config": asdict(config),
        "known_propensities": True,
        "projection_metrics": projection_metrics,
        "policies": {},
    }
    report["policies"]["exact"] = {
        **policy_metrics(exact_policy, evaluation.all_rewards, values_eval, None),
        **ope_metrics(exact_policy, evaluation, config.bootstrap_samples, seed + 1000),
    }
    for k in config.ks:
        for name, queries, keys in (("base", base_queries, base_keys), ("b2", b2_queries, b2_keys)):
            for backend in ("exact", "hnsw"):
                ef_values = config.hnsw_ef_search_values if backend == "hnsw" else (0,)
                for ef_search in ef_values:
                    candidates, retrieval_meta = retrieve_with_embeddings(
                        queries, keys, k, backend=backend, ef_search=ef_search
                    )
                    policy = mask_policy(exact_policy, candidates)
                    metrics = policy_metrics(
                        policy, evaluation.all_rewards, values_eval, candidates, exact_policy
                    )
                    metrics.update(
                        ope_metrics(policy, evaluation, config.bootstrap_samples, seed + 2000 + k)
                    )
                    metrics["value_gap_vs_exact"] = (
                        report["policies"]["exact"]["true_value"] - metrics["true_value"]
                    )
                    metrics.update(retrieval_meta)
                    suffix = f"{name}_k{k}_{backend}_ef{ef_search}"
                    report["policies"][suffix] = metrics
    return report
