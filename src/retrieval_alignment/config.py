"""Dataclass configuration objects for the DW-MQL implementation.

See ``docs/code_architecture.md`` for the intended config boundaries and
``docs/open_math_questions.md`` for assumptions that should stay explicit in
future configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    """Neural-network dimensions for the minimal prototype."""

    state_dim: int = 4
    action_dim: int = 3
    embedding_dim: int = 16
    hidden_dim: int = 32
    hidden_dims: tuple[int, ...] | None = None
    activation: str = "relu"
    pair_feature_mode: str = "concat_product"
    # Retrieval projection (B2 firewall): when True, build separate retrieval
    # projection heads on detached encoder outputs, trained only by the listwise
    # KL alignment loss and used only for the ANN index — so the alignment loss
    # cannot perturb the Q/Bellman path. Default off preserves existing behavior.
    retrieval_projection: bool = False
    # Wolpertinger proto-action head: when True, build a separate MLP that maps
    # detached state embeddings to action embedding space, trained by BC loss.
    # Used as ANN query instead of the raw state embedding.
    proto_action: bool = False


@dataclass(slots=True)
class AlgorithmConfig:
    """Core algorithm knobs that are safe to expose before implementation."""

    gamma: float = 0.99
    candidate_k: int = 10
    weight_cap: float = 10.0


@dataclass(slots=True)
class ANNConfig:
    """ANN index options for the package interface."""

    backend: str = "exact"
    metric: str = "dot"
    nlist: int | None = None
    nprobe: int | None = None


@dataclass(slots=True)
class CandidateConfig:
    """Candidate-set construction options.

    These settings correspond to the candidate-union design in
    ``docs/algorithm_pseudocode.md``:
    logged action, ANN top-k, random negatives, and hard negatives.
    """

    max_candidates: int | None = None
    pad_value: int = -1
    include_logged: bool = True


@dataclass(slots=True)
class DensityConfig:
    """Density-ratio weight options.

    The first implementation follows ``docs/algorithm_notes.md``:
    ``softplus(logits) + eps``, optional clipping, and optional batch
    self-normalisation.
    """

    max_weight: float | None = 10.0
    normalize: bool = True
    eps: float = 1e-8


@dataclass(slots=True)
class DataConfig:
    """Offline replay dataset options."""

    replacement: bool = True


@dataclass(slots=True)
class SyntheticDatasetConfig:
    """Tiny synthetic offline data options for smoke tests."""

    num_transitions: int = 128
    state_dim: int = 4
    action_dim: int = 3
    num_actions: int = 16
    gamma: float = 0.99
    seed: int = 0
    reward_noise_std: float = 0.0
    terminal_prob: float = 0.1


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment config used by future training/evaluation code."""

    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    ann: ANNConfig = field(default_factory=ANNConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)
    seed: int = 0
    experiment_name: str = "synthetic_debug"
    num_steps: int = 100
    batch_size: int = 16
    gamma: float = 0.99
    k: int = 10
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 1
    index_encoder_policy: str = "online"
    target_tau: float | None = None
    target_update_every: int = 1
    index_ema_tau: float | None = None
    output_dir: str = "runs/benchmark_synthetic"
    checkpoint_every: int | None = None
    resume_from: str | None = None
    eval_num_batches: int = 4
    eval_batch_size: int | None = None
    baseline_names: tuple[str, ...] = ()
    retrieval_correction_mode: str = "aware"
    support_correction_weight: float = 0.0
    support_correction_target: float | None = None
    support_correction_metric: str = "missing_mass"
    support_weighting_mode: str = "none"
    support_weight_scale: float = 1.0
    support_weight_min: float = 0.1
    support_weight_max: float = 10.0
    support_pmr_eps: float = 1e-6
    synthetic_only: bool = True


@dataclass(slots=True)
class MovieLensLocalExperimentConfig:
    """Claim-free local MovieLens experiment scaffold config.

    This config is deliberately separate from ``ExperimentConfig`` because the
    local MovieLens scaffold is not synthetic, but it is still scaffold-only
    and must not be treated as a benchmark result.
    """

    experiment_name: str = "movielens_local_scaffold_debug"
    data_path: str = "path/to/ratings.csv"
    output_dir: str = "runs/movielens_local_experiment"
    seed: int = 0
    num_steps: int = 20
    batch_size: int = 8
    gamma: float = 0.99
    k: int = 5
    lr: float = 3e-4
    embedding_dim: int = 8
    split_mode: str = "chronological"
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    session_gap_seconds: int | None = None
    min_rating_threshold: float = 4.0
    max_rows: int | None = None
    row_sampling_mode: str = "head"
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    retrieval_correction_mode: str = "aware"
    support_correction_weight: float = 0.0
    support_correction_target: float | None = None
    support_correction_metric: str = "missing_mass"
    support_weighting_mode: str = "none"
    support_weight_scale: float = 1.0
    support_weight_min: float = 0.1
    support_weight_max: float = 10.0
    support_pmr_eps: float = 1e-6
    target_tau: float | None = 0.05
    target_update_every: int = 1
    baseline_names: tuple[str, ...] = ()
    report_path: str | None = None
    scaffold_only: bool = True
    benchmark_claim: bool = False


@dataclass(slots=True)
class D4RLMockExperimentConfig:
    """Claim-free D4RL-style mock large-action experiment scaffold config."""

    experiment_name: str = "d4rl_mock_large_action_debug"
    output_dir: str = "runs/d4rl_mock_experiment"
    seed: int = 0
    num_steps: int = 20
    batch_size: int = 8
    gamma: float = 0.99
    k: int = 5
    lr: float = 3e-4
    embedding_dim: int = 8
    num_transitions: int = 64
    state_dim: int = 6
    action_dim: int = 4
    codebook_size: int = 16
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    target_tau: float | None = 0.05
    target_update_every: int = 1
    scaffold_only: bool = True
    benchmark_claim: bool = False


@dataclass(slots=True)
class RecSimMockExperimentConfig:
    """Claim-free RecSim mock slate experiment scaffold config."""

    experiment_name: str = "recsim_mock_slate_debug"
    output_dir: str = "runs/recsim_mock_experiment"
    seed: int = 0
    num_steps: int = 20
    batch_size: int = 8
    gamma: float = 0.99
    k: int = 5
    lr: float = 3e-4
    embedding_dim: int = 8
    num_users: int = 4
    num_items: int = 16
    user_feature_dim: int = 4
    item_feature_dim: int = 4
    slate_size: int = 3
    num_interactions: int = 64
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    target_tau: float | None = 0.05
    target_update_every: int = 1
    scaffold_only: bool = True
    benchmark_claim: bool = False


@dataclass(slots=True)
class SmokeProfile:
    """Flat defaults for synthetic smoke CLI modes.

    This is intentionally lightweight and local to smoke checks. It is not a
    replacement for future YAML experiment configs.
    """

    seed: int = 0
    batch_size: int = 8
    num_transitions: int = 64
    num_actions: int = 32
    state_dim: int = 4
    action_dim: int = 3
    embedding_dim: int = 8
    k: int = 5
    gamma: float = 0.99
    num_steps: int = 5
    lr: float = 3e-4
    target_tau: float | None = None
    target_update_every: int = 1
    alternating_smoke: bool = False
    assert_gradient_flow: bool = False
    no_encoder_on_critic: bool = False
    index_backend: str = "exact"
    index_metric: str = "dot"
    index_nlist: int | None = None
    index_nprobe: int | None = None
    index_refresh_every: int = 1
    use_persistent_index: bool = False
    audit_stale_index: bool = False
    audit_encoder_policy_overlap: bool = False
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    retrieval_correction_mode: str = "aware"
    support_correction_weight: float = 0.0
    support_correction_target: float | None = None
    support_correction_metric: str = "missing_mass"
    support_weighting_mode: str = "none"
    support_weight_scale: float = 1.0
    support_weight_min: float = 0.1
    support_weight_max: float = 10.0
    support_pmr_eps: float = 1e-6


@dataclass(slots=True)
class SmokeComponents:
    """Constructed objects for smoke CLI modes."""

    profile: SmokeProfile
    data: Any
    model: Any
    optimizer: Any
    target_model: Any | None = None
    generator: Any | None = None


def default_smoke_profile() -> SmokeProfile:
    """Return the default synthetic smoke profile."""

    return SmokeProfile()


def validate_smoke_profile(profile: SmokeProfile) -> SmokeProfile:
    """Validate a synthetic smoke profile and return it."""

    if not isinstance(profile, SmokeProfile):
        raise TypeError("profile must be a SmokeProfile.")
    if not isinstance(profile.seed, int) or profile.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    positive_int_fields = {
        "batch_size": profile.batch_size,
        "num_transitions": profile.num_transitions,
        "num_actions": profile.num_actions,
        "state_dim": profile.state_dim,
        "action_dim": profile.action_dim,
        "embedding_dim": profile.embedding_dim,
        "k": profile.k,
        "num_steps": profile.num_steps,
        "target_update_every": profile.target_update_every,
        "index_refresh_every": profile.index_refresh_every,
    }
    for name, value in positive_int_fields.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if profile.k > profile.num_actions:
        raise ValueError("k must not exceed num_actions.")
    if not isinstance(profile.gamma, int | float) or not 0.0 <= float(profile.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(profile.lr, int | float) or float(profile.lr) <= 0:
        raise ValueError("lr must be positive.")
    if profile.target_tau is not None:
        if not isinstance(profile.target_tau, int | float) or not 0.0 <= float(profile.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    for name in (
        "alternating_smoke",
        "assert_gradient_flow",
        "no_encoder_on_critic",
        "use_persistent_index",
        "audit_stale_index",
        "audit_encoder_policy_overlap",
    ):
        if not isinstance(getattr(profile, name), bool):
            raise TypeError(f"{name} must be a bool.")
    if profile.assert_gradient_flow and not profile.alternating_smoke:
        raise ValueError("assert_gradient_flow requires alternating_smoke.")
    if profile.no_encoder_on_critic and not profile.alternating_smoke:
        raise ValueError("no_encoder_on_critic requires alternating_smoke.")
    if profile.audit_encoder_policy_overlap and not profile.use_persistent_index:
        raise ValueError("audit_encoder_policy_overlap requires use_persistent_index.")
    if profile.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if profile.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if profile.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if profile.index_encoder_policy == "target" and profile.target_tau is None:
        raise ValueError("index_encoder_policy='target' requires target_tau.")
    if profile.index_ema_tau is not None:
        if not isinstance(profile.index_ema_tau, int | float) or not 0.0 <= float(profile.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if profile.index_encoder_policy == "ema" and profile.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if profile.retrieval_correction_mode not in {"aware", "naive", "uniform_weights", "no_density"}:
        raise ValueError("retrieval_correction_mode must be one of {'aware', 'naive', 'uniform_weights', 'no_density'}.")
    if not isinstance(profile.support_correction_weight, int | float) or float(profile.support_correction_weight) < 0:
        raise ValueError("support_correction_weight must be nonnegative.")
    if profile.support_correction_metric not in {"missing_mass", "normalized_pmr_margin"}:
        raise ValueError("support_correction_metric must be one of {'missing_mass', 'normalized_pmr_margin'}.")
    if profile.support_correction_target is not None:
        target = float(profile.support_correction_target)
        if target != target or target in {float("inf"), float("-inf")} or target <= 0:
            raise ValueError("support_correction_target must be None or finite and positive.")
    if profile.support_correction_metric == "normalized_pmr_margin" and profile.support_correction_target is None:
        raise ValueError("support_correction_target is required for normalized_pmr_margin.")
    if profile.support_weighting_mode not in {"none", "missing_mass", "inverse_pmr", "normalized_pmr_margin"}:
        raise ValueError(
            "support_weighting_mode must be one of {'none', 'missing_mass', 'inverse_pmr', 'normalized_pmr_margin'}."
        )
    if not isinstance(profile.support_weight_scale, int | float) or float(profile.support_weight_scale) < 0:
        raise ValueError("support_weight_scale must be nonnegative.")
    if not isinstance(profile.support_weight_min, int | float) or float(profile.support_weight_min) <= 0:
        raise ValueError("support_weight_min must be positive.")
    if not isinstance(profile.support_weight_max, int | float) or float(profile.support_weight_max) < float(profile.support_weight_min):
        raise ValueError("support_weight_max must be greater than or equal to support_weight_min.")
    if not isinstance(profile.support_pmr_eps, int | float) or float(profile.support_pmr_eps) <= 0:
        raise ValueError("support_pmr_eps must be positive.")
    for name in ("index_nlist", "index_nprobe"):
        value = getattr(profile, name)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be None or a positive integer.")
    return profile


def smoke_profile_from_args(args: Any) -> SmokeProfile:
    """Build a validated smoke profile from argparse-style attributes."""

    profile = load_smoke_profile_yaml(args.config) if getattr(args, "config", None) else default_smoke_profile()
    for field_name in SmokeProfile.__dataclass_fields__:
        if hasattr(args, field_name):
            value = getattr(args, field_name)
            if value is not None:
                setattr(profile, field_name, value)
    return validate_smoke_profile(profile)


def synthetic_config_from_smoke_profile(profile: SmokeProfile) -> SyntheticDatasetConfig:
    """Build synthetic data config from a validated smoke profile."""

    validate_smoke_profile(profile)
    return SyntheticDatasetConfig(
        num_transitions=profile.num_transitions,
        state_dim=profile.state_dim,
        action_dim=profile.action_dim,
        num_actions=profile.num_actions,
        gamma=profile.gamma,
        seed=profile.seed,
    )


def experiment_config_from_smoke_profile(
    profile: SmokeProfile,
    experiment_name: str = "synthetic_debug",
) -> ExperimentConfig:
    """Convert a smoke profile into the benchmark-ready experiment scaffold config."""

    validate_smoke_profile(profile)
    config = ExperimentConfig(
        model=ModelConfig(
            state_dim=profile.state_dim,
            action_dim=profile.action_dim,
            embedding_dim=profile.embedding_dim,
        ),
        algorithm=AlgorithmConfig(gamma=profile.gamma, candidate_k=profile.k),
        ann=ANNConfig(
            backend=profile.index_backend,
            metric=profile.index_metric,
            nlist=profile.index_nlist,
            nprobe=profile.index_nprobe,
        ),
        synthetic=synthetic_config_from_smoke_profile(profile),
        seed=profile.seed,
        experiment_name=experiment_name,
        num_steps=profile.num_steps,
        batch_size=profile.batch_size,
        gamma=profile.gamma,
        k=profile.k,
        index_backend=profile.index_backend,
        index_metric=profile.index_metric,
        use_persistent_index=profile.use_persistent_index,
        index_refresh_every=profile.index_refresh_every,
        index_encoder_policy=profile.index_encoder_policy,
        target_tau=profile.target_tau,
        target_update_every=profile.target_update_every,
        index_ema_tau=profile.index_ema_tau,
        retrieval_correction_mode=profile.retrieval_correction_mode,
        support_correction_weight=profile.support_correction_weight,
        support_correction_target=profile.support_correction_target,
        support_correction_metric=profile.support_correction_metric,
        support_weighting_mode=profile.support_weighting_mode,
        support_weight_scale=profile.support_weight_scale,
        support_weight_min=profile.support_weight_min,
        support_weight_max=profile.support_weight_max,
        support_pmr_eps=profile.support_pmr_eps,
        synthetic_only=True,
    )
    return validate_experiment_config(config)


def validate_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate the synthetic-only benchmark experiment scaffold config."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    if not isinstance(config.experiment_name, str) or not config.experiment_name:
        raise ValueError("experiment_name must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in ("num_steps", "batch_size", "k", "index_refresh_every", "target_update_every"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config.k > config.synthetic.num_actions:
        raise ValueError("k must not exceed synthetic.num_actions.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be a bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "target" and config.target_tau is None:
        raise ValueError("index_encoder_policy='target' requires target_tau.")
    if config.index_ema_tau is not None:
        if not isinstance(config.index_ema_tau, int | float) or not 0.0 <= float(config.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if config.retrieval_correction_mode not in {"aware", "naive", "uniform_weights", "no_density"}:
        raise ValueError("retrieval_correction_mode must be one of {'aware', 'naive', 'uniform_weights', 'no_density'}.")
    if not isinstance(config.support_correction_weight, int | float) or float(config.support_correction_weight) < 0:
        raise ValueError("support_correction_weight must be nonnegative.")
    if config.support_correction_metric not in {"missing_mass", "normalized_pmr_margin"}:
        raise ValueError("support_correction_metric must be one of {'missing_mass', 'normalized_pmr_margin'}.")
    if config.support_correction_target is not None:
        target = float(config.support_correction_target)
        if target != target or target in {float("inf"), float("-inf")} or target <= 0:
            raise ValueError("support_correction_target must be None or finite and positive.")
    if config.support_correction_metric == "normalized_pmr_margin" and config.support_correction_target is None:
        raise ValueError("support_correction_target is required for normalized_pmr_margin.")
    if config.support_weighting_mode not in {"none", "missing_mass", "inverse_pmr", "normalized_pmr_margin"}:
        raise ValueError(
            "support_weighting_mode must be one of {'none', 'missing_mass', 'inverse_pmr', 'normalized_pmr_margin'}."
        )
    if not isinstance(config.support_weight_scale, int | float) or float(config.support_weight_scale) < 0:
        raise ValueError("support_weight_scale must be nonnegative.")
    if not isinstance(config.support_weight_min, int | float) or float(config.support_weight_min) <= 0:
        raise ValueError("support_weight_min must be positive.")
    if not isinstance(config.support_weight_max, int | float) or float(config.support_weight_max) < float(config.support_weight_min):
        raise ValueError("support_weight_max must be greater than or equal to support_weight_min.")
    if not isinstance(config.support_pmr_eps, int | float) or float(config.support_pmr_eps) <= 0:
        raise ValueError("support_pmr_eps must be positive.")
    if config.checkpoint_every is not None:
        if not isinstance(config.checkpoint_every, int) or config.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be None or a positive integer.")
    if config.resume_from is not None and not isinstance(config.resume_from, str):
        raise TypeError("resume_from must be None or a string.")
    if not isinstance(config.eval_num_batches, int) or config.eval_num_batches <= 0:
        raise ValueError("eval_num_batches must be a positive integer.")
    if config.eval_batch_size is not None and (
        not isinstance(config.eval_batch_size, int) or config.eval_batch_size <= 0
    ):
        raise ValueError("eval_batch_size must be None or a positive integer.")
    if not isinstance(config.baseline_names, tuple):
        raise TypeError("baseline_names must be a tuple of strings.")
    if config.baseline_names:
        from .baselines import PLACEHOLDER_BASELINE_NAMES, get_baseline

        for name in config.baseline_names:
            if not isinstance(name, str) or not name:
                raise ValueError("baseline_names must contain non-empty strings.")
            if name in PLACEHOLDER_BASELINE_NAMES:
                raise ValueError(
                    f"Unknown baseline '{name}': placeholder baselines are not implemented for use in experiments."
                )
            get_baseline(name)
    if config.synthetic_only is not True:
        raise ValueError("synthetic_only must be true for the current benchmark scaffold.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    return config


def validate_movielens_local_experiment_config(
    config: MovieLensLocalExperimentConfig,
) -> MovieLensLocalExperimentConfig:
    """Validate the local MovieLens experiment scaffold config."""

    if not isinstance(config, MovieLensLocalExperimentConfig):
        raise TypeError("config must be a MovieLensLocalExperimentConfig.")
    if not isinstance(config.experiment_name, str) or not config.experiment_name:
        raise ValueError("experiment_name must be a non-empty string.")
    if not isinstance(config.data_path, str) or not config.data_path:
        raise ValueError("data_path must be a non-empty string.")
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"MovieLens local CSV file does not exist: {config.data_path}")
    if not Path(config.data_path).is_file():
        raise ValueError("data_path must point to a local ratings CSV file.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in ("num_steps", "batch_size", "k", "embedding_dim", "index_refresh_every", "target_update_every"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    if config.split_mode not in {"none", "chronological", "session", "chronological_strict"}:
        raise ValueError("split_mode must be one of {'none', 'chronological', 'session', 'chronological_strict'}.")
    for name in ("validation_fraction", "test_fraction"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if float(config.validation_fraction) + float(config.test_fraction) >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1.")
    if config.session_gap_seconds is not None and (
        not isinstance(config.session_gap_seconds, int) or config.session_gap_seconds <= 0
    ):
        raise ValueError("session_gap_seconds must be None or a positive integer.")
    if config.split_mode == "session" and config.session_gap_seconds is None:
        raise ValueError("session split_mode requires session_gap_seconds.")
    if not isinstance(config.min_rating_threshold, int | float):
        raise ValueError("min_rating_threshold must be numeric.")
    if config.max_rows is not None and (not isinstance(config.max_rows, int) or config.max_rows <= 0):
        raise ValueError("max_rows must be None or a positive integer.")
    if config.row_sampling_mode not in {"head", "reservoir"}:
        raise ValueError("row_sampling_mode must be one of {'head', 'reservoir'}.")
    if config.row_sampling_mode == "reservoir" and config.max_rows is None:
        raise ValueError("row_sampling_mode='reservoir' requires max_rows to be set.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be a bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "target" and config.target_tau is None:
        raise ValueError("index_encoder_policy='target' requires target_tau.")
    if config.index_ema_tau is not None:
        if not isinstance(config.index_ema_tau, int | float) or not 0.0 <= float(config.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if not isinstance(config.baseline_names, tuple):
        raise TypeError("baseline_names must be a tuple of strings.")
    if config.baseline_names:
        from .baselines import PLACEHOLDER_BASELINE_NAMES, get_baseline

        for name in config.baseline_names:
            if not isinstance(name, str) or not name:
                raise ValueError("baseline_names must contain non-empty strings.")
            if name in PLACEHOLDER_BASELINE_NAMES:
                raise ValueError(
                    f"Unknown baseline '{name}': placeholder baselines are not implemented for use in experiments."
                )
            get_baseline(name)
    if config.report_path is not None and not isinstance(config.report_path, str):
        raise TypeError("report_path must be None or a string.")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true for the MovieLens local scaffold.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false for the MovieLens local scaffold.")
    return config


def validate_d4rl_mock_experiment_config(config: D4RLMockExperimentConfig) -> D4RLMockExperimentConfig:
    """Validate the D4RL mock large-action experiment scaffold config."""

    if not isinstance(config, D4RLMockExperimentConfig):
        raise TypeError("config must be a D4RLMockExperimentConfig.")
    if not isinstance(config.experiment_name, str) or not config.experiment_name:
        raise ValueError("experiment_name must be a non-empty string.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in (
        "num_steps",
        "batch_size",
        "k",
        "embedding_dim",
        "num_transitions",
        "state_dim",
        "action_dim",
        "codebook_size",
        "index_refresh_every",
        "target_update_every",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config.codebook_size > config.num_transitions:
        raise ValueError("codebook_size must not exceed num_transitions in the mock scaffold.")
    if config.k > config.codebook_size:
        raise ValueError("k must not exceed codebook_size.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be a bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "target" and config.target_tau is None:
        raise ValueError("index_encoder_policy='target' requires target_tau.")
    if config.index_ema_tau is not None:
        if not isinstance(config.index_ema_tau, int | float) or not 0.0 <= float(config.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true for the D4RL mock scaffold.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false for the D4RL mock scaffold.")
    return config


def load_d4rl_mock_experiment_config(path: str | Path) -> D4RLMockExperimentConfig:
    """Load a D4RL mock large-action experiment scaffold YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"D4RL mock experiment config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("D4RL mock experiment config YAML must contain a mapping.")
    allowed = set(D4RLMockExperimentConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown D4RL mock experiment config key(s): {sorted(unknown)}.")
    config = D4RLMockExperimentConfig()
    for key, value in loaded.items():
        setattr(config, key, value)
    return validate_d4rl_mock_experiment_config(config)


def validate_recsim_mock_experiment_config(config: RecSimMockExperimentConfig) -> RecSimMockExperimentConfig:
    """Validate the RecSim mock slate experiment scaffold config."""

    if not isinstance(config, RecSimMockExperimentConfig):
        raise TypeError("config must be a RecSimMockExperimentConfig.")
    if not isinstance(config.experiment_name, str) or not config.experiment_name:
        raise ValueError("experiment_name must be a non-empty string.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in (
        "num_steps",
        "batch_size",
        "k",
        "embedding_dim",
        "num_users",
        "num_items",
        "user_feature_dim",
        "item_feature_dim",
        "slate_size",
        "num_interactions",
        "index_refresh_every",
        "target_update_every",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config.slate_size > config.num_items:
        raise ValueError("slate_size must not exceed num_items.")
    if config.k > config.num_items:
        raise ValueError("k must not exceed num_items.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be a bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "target" and config.target_tau is None:
        raise ValueError("index_encoder_policy='target' requires target_tau.")
    if config.index_ema_tau is not None:
        if not isinstance(config.index_ema_tau, int | float) or not 0.0 <= float(config.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true for the RecSim mock scaffold.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false for the RecSim mock scaffold.")
    return config


def load_recsim_mock_experiment_config(path: str | Path) -> RecSimMockExperimentConfig:
    """Load a RecSim mock slate experiment scaffold YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"RecSim mock experiment config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("RecSim mock experiment config YAML must contain a mapping.")
    allowed = set(RecSimMockExperimentConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown RecSim mock experiment config key(s): {sorted(unknown)}.")
    config = RecSimMockExperimentConfig()
    for key, value in loaded.items():
        setattr(config, key, value)
    return validate_recsim_mock_experiment_config(config)


@dataclass(slots=True)
class MovieLensChronoProtocolConfig:
    """Config for the claim-free MovieLens chronological support protocol.

    Uses chronological_strict global timestamp ordering for the held-out split.
    Results are diagnostic-only; no recommender benchmark claims are permitted.
    """

    protocol_name: str = "movielens_chrono_support_protocol"
    data_path: str = "path/to/ratings.csv"
    output_dir: str = "runs/movielens_chrono_protocol"
    seed: int = 0
    num_steps: int = 20
    batch_size: int = 8
    gamma: float = 0.99
    k_values: tuple[int, ...] = (5, 10, 20)
    train_k: int = 10
    lr: float = 3e-4
    embedding_dim: int = 8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    min_rating_threshold: float = 4.0
    max_rows: int | None = None
    row_sampling_mode: str = "head"
    temperature: float = 1.0
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    target_tau: float | None = 0.05
    target_update_every: int = 1
    # Stability controls
    max_grad_norm: float | None = None
    q_l2_regularization_weight: float = 0.0
    q_value_clip: float | None = None
    reward_scale: float = 1.0
    critic_method: str = "dwmql"
    retrieval_correction_mode: str = "aware"
    density_regularization_weight: float = 1e-4
    stationarity_weight: float = 1.0
    bellman_weight: float = 1.0
    scaffold_only: bool = True
    benchmark_claim: bool = False
    # When True, save the trained model state + dims to output_dir so the model
    # can be re-evaluated (e.g. under different ANN indexes) without retraining.
    save_model_checkpoint: bool = False
    # Training device: "cpu" (default) or "cuda"/"cuda:N". Training runs on this
    # device; evaluation/diagnostics run on CPU.
    device: str = "cpu"
    # Ranking-margin auxiliary loss (disabled by default)
    ranking_margin_weight: float = 0.0
    ranking_margin: float = 1.0
    ranking_negative_samples: int = 20
    # Retrieval alignment auxiliary loss (disabled by default)
    retrieval_alignment_weight: float = 0.0
    retrieval_alignment_margin: float = 1.0
    retrieval_alignment_warmup_steps: int = 0
    retrieval_alignment_loss_type: str = "listwise_kl"
    listwise_teacher_temperature: float = 0.2
    listwise_retrieval_temperature: float = 0.07
    # Q-degeneracy diagnostic thresholds
    q_degeneracy_std_threshold: float = 1e-3
    q_degeneracy_margin_threshold: float = 1e-4
    # Cap evaluation rows per split. None = evaluate all rows (slow on 5M+ datasets).
    max_eval_rows: int | None = None
    # CQL conservative penalty (Kumar et al. 2020): penalises OOD Q-value overestimation.
    cql_weight: float = 0.0
    cql_num_samples: int = 64
    # BCQ behavioral constraint (Fujimoto et al. 2019): constrains policy to behavioral support.
    bcq_weight: float = 0.0
    # Wolpertinger BC proto-action (Dulac-Arnold et al. 2015): trains a state→action-space
    # projection with behavior cloning loss; output used as ANN query key.
    wolpertinger_bc_weight: float = 0.0


def component_identity_from_config(config: Any) -> dict[str, bool | str]:
    """Return explicit critic, density, retrieval, and index identities."""

    required = (
        "critic_method",
        "retrieval_correction_mode",
        "retrieval_alignment_weight",
        "retrieval_alignment_loss_type",
        "index_backend",
        "index_metric",
    )
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise TypeError(f"config is missing component identity field(s): {missing}.")
    return {
        "critic_method": str(config.critic_method),
        "density_enabled": config.retrieval_correction_mode != "no_density",
        "density_affects_shared_encoder": False,
        "retrieval_alignment_enabled": float(config.retrieval_alignment_weight) > 0.0,
        "retrieval_projection_enabled": (
            config.retrieval_alignment_loss_type == "listwise_kl"
            and float(config.retrieval_alignment_weight) > 0.0
        ),
        "index_backend": str(config.index_backend),
        "index_metric": str(config.index_metric),
    }


def _validate_component_identity_fields(config: Any) -> None:
    if config.critic_method not in {"dqn", "cql", "bcq", "dwmql"}:
        raise ValueError("critic_method must be one of {'dqn', 'cql', 'bcq', 'dwmql'}.")
    if config.retrieval_correction_mode not in {"aware", "naive", "uniform_weights", "no_density"}:
        raise ValueError(
            "retrieval_correction_mode must be one of "
            "{'aware', 'naive', 'uniform_weights', 'no_density'}."
        )
    if config.critic_method == "dqn" and (
        float(config.cql_weight) > 0.0 or float(config.bcq_weight) > 0.0
    ):
        raise ValueError("critic_method='dqn' requires cql_weight=0 and bcq_weight=0.")
    if config.critic_method == "cql" and float(config.cql_weight) <= 0.0:
        raise ValueError("critic_method='cql' requires a positive cql_weight.")
    if config.critic_method == "bcq" and float(config.bcq_weight) <= 0.0:
        raise ValueError("critic_method='bcq' requires a positive bcq_weight.")


def validate_movielens_chrono_protocol_config(
    config: MovieLensChronoProtocolConfig,
) -> MovieLensChronoProtocolConfig:
    """Validate the MovieLens chronological support protocol config."""

    if not isinstance(config, MovieLensChronoProtocolConfig):
        raise TypeError("config must be a MovieLensChronoProtocolConfig.")
    if not isinstance(config.protocol_name, str) or not config.protocol_name:
        raise ValueError("protocol_name must be a non-empty string.")
    if not isinstance(config.data_path, str) or not config.data_path:
        raise ValueError("data_path must be a non-empty string.")
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"MovieLens local CSV file does not exist: {config.data_path}")
    if not Path(config.data_path).is_file():
        raise ValueError("data_path must point to a local ratings CSV file.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in ("num_steps", "batch_size", "train_k", "embedding_dim", "index_refresh_every", "target_update_every"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.k_values, tuple) or not config.k_values:
        raise TypeError("k_values must be a non-empty tuple of positive integers.")
    for k in config.k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Each k in k_values must be a positive integer.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    for name in ("validation_fraction", "test_fraction"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if float(config.validation_fraction) + float(config.test_fraction) >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1.")
    if not isinstance(config.min_rating_threshold, int | float):
        raise ValueError("min_rating_threshold must be numeric.")
    if config.max_rows is not None and (not isinstance(config.max_rows, int) or config.max_rows <= 0):
        raise ValueError("max_rows must be None or a positive integer.")
    if config.row_sampling_mode not in {"head", "reservoir"}:
        raise ValueError("row_sampling_mode must be one of {'head', 'reservoir'}.")
    if config.row_sampling_mode == "reservoir" and config.max_rows is None:
        raise ValueError("row_sampling_mode='reservoir' requires max_rows to be set.")
    if config.max_eval_rows is not None and (not isinstance(config.max_eval_rows, int) or config.max_eval_rows <= 0):
        raise ValueError("max_eval_rows must be None or a positive integer.")
    if not isinstance(config.cql_weight, int | float) or float(config.cql_weight) < 0:
        raise ValueError("cql_weight must be a nonnegative float.")
    if not isinstance(config.cql_num_samples, int) or config.cql_num_samples <= 0:
        raise ValueError("cql_num_samples must be a positive integer.")
    if not isinstance(config.bcq_weight, int | float) or float(config.bcq_weight) < 0:
        raise ValueError("bcq_weight must be a nonnegative float.")
    _validate_component_identity_fields(config)
    if not isinstance(config.temperature, int | float) or float(config.temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be a bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.index_ema_tau is not None:
        if not isinstance(config.index_ema_tau, int | float) or not 0.0 <= float(config.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if config.max_grad_norm is not None:
        if not isinstance(config.max_grad_norm, int | float) or float(config.max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be None or a positive float.")
    if not isinstance(config.q_l2_regularization_weight, int | float) or float(config.q_l2_regularization_weight) < 0:
        raise ValueError("q_l2_regularization_weight must be a nonnegative float.")
    if config.q_value_clip is not None:
        if not isinstance(config.q_value_clip, int | float) or float(config.q_value_clip) <= 0:
            raise ValueError("q_value_clip must be None or a positive float.")
    if not isinstance(config.reward_scale, int | float) or float(config.reward_scale) <= 0:
        raise ValueError("reward_scale must be a positive float.")
    for name in ("density_regularization_weight", "stationarity_weight", "bellman_weight"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or float(value) < 0:
            raise ValueError(f"{name} must be a nonnegative float.")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true for the MovieLens chrono protocol.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false for the MovieLens chrono protocol.")
    if not isinstance(config.ranking_margin_weight, int | float) or float(config.ranking_margin_weight) < 0:
        raise ValueError("ranking_margin_weight must be a nonnegative float.")
    if not isinstance(config.ranking_margin, int | float) or float(config.ranking_margin) <= 0:
        raise ValueError("ranking_margin must be a positive float.")
    if not isinstance(config.ranking_negative_samples, int) or config.ranking_negative_samples <= 0:
        raise ValueError("ranking_negative_samples must be a positive integer.")
    if not isinstance(config.retrieval_alignment_weight, int | float) or float(config.retrieval_alignment_weight) < 0:
        raise ValueError("retrieval_alignment_weight must be a nonnegative float.")
    if not isinstance(config.retrieval_alignment_margin, int | float) or float(config.retrieval_alignment_margin) <= 0:
        raise ValueError("retrieval_alignment_margin must be a positive float.")
    if not isinstance(config.retrieval_alignment_warmup_steps, int) or config.retrieval_alignment_warmup_steps < 0:
        raise ValueError("retrieval_alignment_warmup_steps must be a nonnegative int.")
    if config.retrieval_alignment_loss_type not in {"hinge", "listwise_kl"}:
        raise ValueError("retrieval_alignment_loss_type must be 'hinge' or 'listwise_kl'.")
    if not isinstance(config.listwise_teacher_temperature, int | float) or float(config.listwise_teacher_temperature) <= 0:
        raise ValueError("listwise_teacher_temperature must be a positive float.")
    if not isinstance(config.listwise_retrieval_temperature, int | float) or float(config.listwise_retrieval_temperature) <= 0:
        raise ValueError("listwise_retrieval_temperature must be a positive float.")
    if not isinstance(config.q_degeneracy_std_threshold, int | float) or float(config.q_degeneracy_std_threshold) < 0:
        raise ValueError("q_degeneracy_std_threshold must be a nonnegative float.")
    if not isinstance(config.q_degeneracy_margin_threshold, int | float) or float(config.q_degeneracy_margin_threshold) < 0:
        raise ValueError("q_degeneracy_margin_threshold must be a nonnegative float.")
    return config


def load_movielens_chrono_protocol_config(path: str | Path) -> MovieLensChronoProtocolConfig:
    """Load a MovieLens chronological support protocol YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"MovieLens chrono protocol config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("MovieLens chrono protocol config YAML must contain a mapping.")
    allowed = set(MovieLensChronoProtocolConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown MovieLens chrono protocol config key(s): {sorted(unknown)}.")
    config = MovieLensChronoProtocolConfig()
    for key, value in loaded.items():
        if key == "k_values":
            if not isinstance(value, list | tuple):
                raise ValueError("k_values must be a YAML list.")
            value = tuple(int(v) for v in value)
        setattr(config, key, value)
    return validate_movielens_chrono_protocol_config(config)


@dataclass(slots=True)
class D4RLOptionalProtocolConfig:
    """Config for the claim-free optional D4RL/Minari offline RL support protocol.

    Loads real datasets behind lazy imports (d4rl or minari). When neither
    dependency is available, the protocol fails clearly with an ImportError.
    Results are diagnostic-only; no offline RL benchmark claims are permitted.
    """

    protocol_name: str = "d4rl_optional_support_protocol"
    dataset_name: str = "hopper-medium-v2"
    output_dir: str = "runs/d4rl_optional_protocol"
    seed: int = 0
    num_steps: int = 20
    batch_size: int = 8
    gamma: float = 0.99
    k_values: tuple[int, ...] = (5, 10, 20)
    train_k: int = 10
    lr: float = 3e-4
    embedding_dim: int = 8
    codebook_size: int = 16
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    target_tau: float | None = 0.05
    target_update_every: int = 1
    scaffold_only: bool = True
    benchmark_claim: bool = False


def validate_d4rl_optional_protocol_config(
    config: D4RLOptionalProtocolConfig,
) -> D4RLOptionalProtocolConfig:
    """Validate the optional D4RL support protocol config."""

    if not isinstance(config, D4RLOptionalProtocolConfig):
        raise TypeError("config must be a D4RLOptionalProtocolConfig.")
    if not isinstance(config.protocol_name, str) or not config.protocol_name:
        raise ValueError("protocol_name must be a non-empty string.")
    if not isinstance(config.dataset_name, str) or not config.dataset_name:
        raise ValueError("dataset_name must be a non-empty string.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in ("num_steps", "batch_size", "train_k", "embedding_dim", "codebook_size", "index_refresh_every", "target_update_every"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.k_values, tuple) or not config.k_values:
        raise TypeError("k_values must be a non-empty tuple of positive integers.")
    for k in config.k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Each k in k_values must be a positive integer.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be a bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.index_ema_tau is not None:
        if not isinstance(config.index_ema_tau, int | float) or not 0.0 <= float(config.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true for the D4RL optional protocol.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false for the D4RL optional protocol.")
    return config


def load_d4rl_optional_protocol_config(path: str | Path) -> D4RLOptionalProtocolConfig:
    """Load an optional D4RL support protocol YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"D4RL optional protocol config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("D4RL optional protocol config YAML must contain a mapping.")
    allowed = set(D4RLOptionalProtocolConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown D4RL optional protocol config key(s): {sorted(unknown)}.")
    config = D4RLOptionalProtocolConfig()
    for key, value in loaded.items():
        if key == "k_values":
            if not isinstance(value, list | tuple):
                raise ValueError("k_values must be a YAML list.")
            value = tuple(int(v) for v in value)
        setattr(config, key, value)
    return validate_d4rl_optional_protocol_config(config)


@dataclass(slots=True)
class RecSimOptionalProtocolConfig:
    """Config for the claim-free optional RecSim/RecSim NG support protocol.

    The RecSim loading shell may raise NotImplementedError until the API is
    finalized. Results are diagnostic-only; no recommender benchmark claims.
    """

    protocol_name: str = "recsim_optional_support_protocol"
    output_dir: str = "runs/recsim_optional_protocol"
    seed: int = 0
    num_steps: int = 20
    batch_size: int = 8
    gamma: float = 0.99
    k_values: tuple[int, ...] = (5, 10, 20)
    train_k: int = 10
    lr: float = 3e-4
    embedding_dim: int = 8
    num_users: int = 4
    num_items: int = 16
    user_feature_dim: int = 4
    item_feature_dim: int = 4
    slate_size: int = 3
    num_interactions: int = 64
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: float | None = None
    target_tau: float | None = 0.05
    target_update_every: int = 1
    scaffold_only: bool = True
    benchmark_claim: bool = False


def validate_recsim_optional_protocol_config(
    config: RecSimOptionalProtocolConfig,
) -> RecSimOptionalProtocolConfig:
    """Validate the optional RecSim support protocol config."""

    if not isinstance(config, RecSimOptionalProtocolConfig):
        raise TypeError("config must be a RecSimOptionalProtocolConfig.")
    if not isinstance(config.protocol_name, str) or not config.protocol_name:
        raise ValueError("protocol_name must be a non-empty string.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in (
        "num_steps", "batch_size", "train_k", "embedding_dim",
        "num_users", "num_items", "user_feature_dim", "item_feature_dim",
        "slate_size", "num_interactions", "index_refresh_every", "target_update_every",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.k_values, tuple) or not config.k_values:
        raise TypeError("k_values must be a non-empty tuple of positive integers.")
    for k in config.k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Each k in k_values must be a positive integer.")
    if config.slate_size > config.num_items:
        raise ValueError("slate_size must not exceed num_items.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if not isinstance(config.use_persistent_index, bool):
        raise TypeError("use_persistent_index must be a bool.")
    if config.index_encoder_policy not in {"online", "target", "ema"}:
        raise ValueError("index_encoder_policy must be one of {'online', 'target', 'ema'}.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.index_ema_tau is not None:
        if not isinstance(config.index_ema_tau, int | float) or not 0.0 <= float(config.index_ema_tau) <= 1.0:
            raise ValueError("index_ema_tau must be None or in [0, 1].")
    if config.index_encoder_policy == "ema" and config.index_ema_tau is None:
        raise ValueError("index_encoder_policy='ema' requires index_ema_tau.")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true for the RecSim optional protocol.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false for the RecSim optional protocol.")
    return config


def load_recsim_optional_protocol_config(path: str | Path) -> RecSimOptionalProtocolConfig:
    """Load an optional RecSim support protocol YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"RecSim optional protocol config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("RecSim optional protocol config YAML must contain a mapping.")
    allowed = set(RecSimOptionalProtocolConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown RecSim optional protocol config key(s): {sorted(unknown)}.")
    config = RecSimOptionalProtocolConfig()
    for key, value in loaded.items():
        if key == "k_values":
            if not isinstance(value, list | tuple):
                raise ValueError("k_values must be a YAML list.")
            value = tuple(int(v) for v in value)
        setattr(config, key, value)
    return validate_recsim_optional_protocol_config(config)


@dataclass(slots=True)
class BenchmarkProtocolConfig:
    """Config for the first benchmark-style multi-seed experiment protocol.

    Runs repeated-seed real/local protocol runs, collects support diagnostics,
    evaluates trivial baselines, and produces grouped summary tables.
    All results are claim-free; benchmark_claim must remain false until real
    datasets, implemented baselines, and statistical tests are in place.
    """

    protocol_name: str = "first_benchmark_protocol"
    output_dir: str = "runs/benchmark_protocol"
    # Route and sweep axes
    routes: tuple[str, ...] = ("movielens_chrono",)
    seeds: tuple[int, ...] = (0, 1, 2)
    num_steps: int = 50
    batch_size: int = 8
    gamma: float = 0.99
    k_values: tuple[int, ...] = (20, 50, 100)
    train_k: int = 20
    index_backend_values: tuple[str, ...] = ("exact",)
    index_metric: str = "dot"
    index_refresh_values: tuple[int, ...] = (5, 20)
    index_encoder_policies: tuple[str, ...] = ("online",)
    index_ema_tau_values: tuple[float | None, ...] = (None,)
    use_persistent_index: bool = True
    target_tau: float | None = 0.05
    target_update_every: int = 1
    # Baseline names — "random" and "nearest" are implemented; others raise NotImplementedError
    baselines: tuple[str, ...] = ("random", "nearest")
    enable_dwmql: bool = True
    enable_exact_backend_control: bool = True
    enable_ann_backend: bool = False
    # MovieLens-specific
    data_path: str | None = None
    embedding_dim: int = 8
    lr: float = 3e-4
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    min_rating_threshold: float = 4.0
    max_rows: int | None = None
    temperature: float = 1.0
    # D4RL-specific
    dataset_name: str | None = None
    codebook_size: int = 64
    # RecSim-specific
    num_users: int = 100
    num_items: int = 500
    user_feature_dim: int = 16
    item_feature_dim: int = 16
    slate_size: int = 5
    num_interactions: int = 1000
    # Stability controls
    max_grad_norm: float | None = None
    q_l2_regularization_weight: float = 0.0
    q_value_clip: float | None = None
    reward_scale: float = 1.0
    density_regularization_weight: float = 1e-4
    stationarity_weight: float = 1.0
    bellman_weight: float = 1.0
    benchmark_claim: bool = False
    # Ranking-margin auxiliary loss (disabled by default)
    ranking_margin_weight: float = 0.0
    ranking_margin: float = 1.0
    ranking_negative_samples: int = 20
    # Retrieval alignment auxiliary loss (disabled by default)
    retrieval_alignment_weight: float = 0.0
    retrieval_alignment_margin: float = 1.0
    retrieval_alignment_warmup_steps: int = 0
    retrieval_alignment_loss_type: str = "listwise_kl"
    listwise_teacher_temperature: float = 0.2
    listwise_retrieval_temperature: float = 0.07
    # Q-degeneracy diagnostic thresholds
    q_degeneracy_std_threshold: float = 1e-3
    q_degeneracy_margin_threshold: float = 1e-4


_BENCHMARK_KNOWN_ROUTES: frozenset[str] = frozenset({"movielens_chrono", "d4rl_optional", "recsim_optional"})
_BENCHMARK_KNOWN_BACKENDS: frozenset[str] = frozenset({"exact", "faiss", "faiss_ivf", "faiss_hnsw"})
_BENCHMARK_KNOWN_ENCODER_POLICIES: frozenset[str] = frozenset({"online", "target", "ema"})


def validate_benchmark_protocol_config(config: BenchmarkProtocolConfig) -> BenchmarkProtocolConfig:
    """Validate a BenchmarkProtocolConfig."""

    if not isinstance(config, BenchmarkProtocolConfig):
        raise TypeError("config must be a BenchmarkProtocolConfig.")
    if not isinstance(config.protocol_name, str) or not config.protocol_name:
        raise ValueError("protocol_name must be a non-empty string.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.routes, tuple) or not config.routes:
        raise ValueError("routes must be a non-empty tuple.")
    for route in config.routes:
        if route not in _BENCHMARK_KNOWN_ROUTES:
            raise ValueError(f"Unknown route '{route}'. Supported: {sorted(_BENCHMARK_KNOWN_ROUTES)}.")
    if not isinstance(config.seeds, tuple) or not config.seeds:
        raise ValueError("seeds must be a non-empty tuple of non-negative integers.")
    for seed in config.seeds:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Each seed must be a non-negative integer.")
    for name in ("num_steps", "batch_size", "train_k", "embedding_dim", "target_update_every"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.k_values, tuple) or not config.k_values:
        raise ValueError("k_values must be a non-empty tuple of positive integers.")
    for k in config.k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Each k in k_values must be a positive integer.")
    if not isinstance(config.index_backend_values, tuple) or not config.index_backend_values:
        raise ValueError("index_backend_values must be a non-empty tuple.")
    for backend in config.index_backend_values:
        if backend not in _BENCHMARK_KNOWN_BACKENDS:
            raise ValueError(f"index_backend '{backend}' must be one of {sorted(_BENCHMARK_KNOWN_BACKENDS)}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be 'dot' or 'cosine'.")
    if not isinstance(config.index_refresh_values, tuple) or not config.index_refresh_values:
        raise ValueError("index_refresh_values must be a non-empty tuple of positive integers.")
    for refresh in config.index_refresh_values:
        if not isinstance(refresh, int) or refresh <= 0:
            raise ValueError("Each value in index_refresh_values must be a positive integer.")
    if not isinstance(config.index_encoder_policies, tuple) or not config.index_encoder_policies:
        raise ValueError("index_encoder_policies must be a non-empty tuple.")
    for policy in config.index_encoder_policies:
        if policy not in _BENCHMARK_KNOWN_ENCODER_POLICIES:
            raise ValueError(f"index_encoder_policy '{policy}' must be one of {sorted(_BENCHMARK_KNOWN_ENCODER_POLICIES)}.")
    if not isinstance(config.index_ema_tau_values, tuple) or not config.index_ema_tau_values:
        raise ValueError("index_ema_tau_values must be a non-empty tuple (use (None,) for no EMA).")
    for tau in config.index_ema_tau_values:
        if tau is not None:
            if not isinstance(tau, int | float) or not 0.0 <= float(tau) <= 1.0:
                raise ValueError("Each tau in index_ema_tau_values must be None or in [0, 1].")
    if "ema" in config.index_encoder_policies:
        non_none_taus = [t for t in config.index_ema_tau_values if t is not None]
        if not non_none_taus:
            raise ValueError("index_encoder_policies includes 'ema' but index_ema_tau_values has no non-None entries.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    if not isinstance(config.baselines, tuple):
        raise TypeError("baselines must be a tuple of baseline names.")
    from .baselines import PLACEHOLDER_BASELINE_NAMES as _PLACEHOLDER_NAMES, get_baseline as _get_baseline
    for bname in config.baselines:
        if bname in _PLACEHOLDER_NAMES:
            raise ValueError(
                f"Baseline '{bname}' is a placeholder and not implemented for use. "
                f"Implemented baselines: random, nearest."
            )
        try:
            _get_baseline(bname)
        except ValueError:
            raise ValueError(
                f"Unknown baseline '{bname}'. Implemented baselines: random, nearest."
            )
    for name in ("enable_dwmql", "enable_exact_backend_control", "enable_ann_backend", "use_persistent_index"):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"{name} must be a bool.")
    for name in ("validation_fraction", "test_fraction"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if float(config.validation_fraction) + float(config.test_fraction) >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1.")
    if not isinstance(config.temperature, int | float) or float(config.temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if config.target_tau is not None:
        if not isinstance(config.target_tau, int | float) or not 0.0 <= float(config.target_tau) <= 1.0:
            raise ValueError("target_tau must be None or in [0, 1].")
    if config.max_rows is not None and (not isinstance(config.max_rows, int) or config.max_rows <= 0):
        raise ValueError("max_rows must be None or a positive integer.")
    if "movielens_chrono" in config.routes:
        if config.data_path is None:
            raise ValueError("data_path is required for route 'movielens_chrono'.")
        if not Path(config.data_path).exists():
            raise FileNotFoundError(f"MovieLens data_path does not exist: {config.data_path}")
    if config.max_grad_norm is not None:
        if not isinstance(config.max_grad_norm, int | float) or float(config.max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be None or a positive float.")
    if not isinstance(config.q_l2_regularization_weight, int | float) or float(config.q_l2_regularization_weight) < 0:
        raise ValueError("q_l2_regularization_weight must be a nonnegative float.")
    if config.q_value_clip is not None:
        if not isinstance(config.q_value_clip, int | float) or float(config.q_value_clip) <= 0:
            raise ValueError("q_value_clip must be None or a positive float.")
    if not isinstance(config.reward_scale, int | float) or float(config.reward_scale) <= 0:
        raise ValueError("reward_scale must be a positive float.")
    for name in ("density_regularization_weight", "stationarity_weight", "bellman_weight"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or float(value) < 0:
            raise ValueError(f"{name} must be a nonnegative float.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false.")
    if not isinstance(config.ranking_margin_weight, int | float) or float(config.ranking_margin_weight) < 0:
        raise ValueError("ranking_margin_weight must be a nonnegative float.")
    if not isinstance(config.ranking_margin, int | float) or float(config.ranking_margin) <= 0:
        raise ValueError("ranking_margin must be a positive float.")
    if not isinstance(config.ranking_negative_samples, int) or config.ranking_negative_samples <= 0:
        raise ValueError("ranking_negative_samples must be a positive integer.")
    if not isinstance(config.retrieval_alignment_weight, int | float) or float(config.retrieval_alignment_weight) < 0:
        raise ValueError("retrieval_alignment_weight must be a nonnegative float.")
    if not isinstance(config.retrieval_alignment_margin, int | float) or float(config.retrieval_alignment_margin) <= 0:
        raise ValueError("retrieval_alignment_margin must be a positive float.")
    if not isinstance(config.retrieval_alignment_warmup_steps, int) or config.retrieval_alignment_warmup_steps < 0:
        raise ValueError("retrieval_alignment_warmup_steps must be a nonnegative int.")
    if config.retrieval_alignment_loss_type not in {"hinge", "listwise_kl"}:
        raise ValueError("retrieval_alignment_loss_type must be 'hinge' or 'listwise_kl'.")
    if not isinstance(config.listwise_teacher_temperature, int | float) or float(config.listwise_teacher_temperature) <= 0:
        raise ValueError("listwise_teacher_temperature must be a positive float.")
    if not isinstance(config.listwise_retrieval_temperature, int | float) or float(config.listwise_retrieval_temperature) <= 0:
        raise ValueError("listwise_retrieval_temperature must be a positive float.")
    if not isinstance(config.q_degeneracy_std_threshold, int | float) or float(config.q_degeneracy_std_threshold) < 0:
        raise ValueError("q_degeneracy_std_threshold must be a nonnegative float.")
    if not isinstance(config.q_degeneracy_margin_threshold, int | float) or float(config.q_degeneracy_margin_threshold) < 0:
        raise ValueError("q_degeneracy_margin_threshold must be a nonnegative float.")
    return config


def _parse_benchmark_protocol_config(path: str | Path) -> BenchmarkProtocolConfig:
    """Parse a BenchmarkProtocolConfig from YAML without running validation.

    Use this when CLI overrides (e.g. --data-path) must be applied before
    validate_benchmark_protocol_config is called. The public loader
    load_benchmark_protocol_config calls this then validates immediately.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark protocol config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Benchmark protocol config YAML must contain a mapping.")
    allowed = set(BenchmarkProtocolConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark protocol config key(s): {sorted(unknown)}.")
    config = BenchmarkProtocolConfig()
    _tuple_int_fields = {"seeds", "k_values", "index_refresh_values"}
    _tuple_str_fields = {"routes", "index_backend_values", "index_encoder_policies", "baselines"}
    for key, value in loaded.items():
        if key in _tuple_int_fields:
            if not isinstance(value, list | tuple):
                raise ValueError(f"{key} must be a YAML list.")
            value = tuple(int(v) for v in value)
        elif key in _tuple_str_fields:
            if not isinstance(value, list | tuple):
                raise ValueError(f"{key} must be a YAML list.")
            value = tuple(value)
        elif key == "index_ema_tau_values":
            if not isinstance(value, list | tuple):
                raise ValueError("index_ema_tau_values must be a YAML list.")
            value = tuple(None if v is None else float(v) for v in value)
        setattr(config, key, value)
    return config


def load_benchmark_protocol_config(path: str | Path) -> BenchmarkProtocolConfig:
    """Load and validate a BenchmarkProtocolConfig from a YAML file.

    Requires all fields to be fully specified in the YAML (e.g. data_path must
    be set). When CLI overrides need to be applied before validation, use
    _parse_benchmark_protocol_config instead.
    """
    config = _parse_benchmark_protocol_config(path)
    return validate_benchmark_protocol_config(config)


def load_movielens_local_experiment_config(path: str | Path) -> MovieLensLocalExperimentConfig:
    """Load a local MovieLens experiment scaffold YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"MovieLens local experiment config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("MovieLens local experiment config YAML must contain a mapping.")
    allowed = set(MovieLensLocalExperimentConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown MovieLens local experiment config key(s): {sorted(unknown)}.")
    config = MovieLensLocalExperimentConfig()
    for key, value in loaded.items():
        if key == "baseline_names":
            if not isinstance(value, list | tuple):
                raise ValueError("baseline_names must be a YAML list or tuple.")
            value = tuple(value)
        setattr(config, key, value)
    return config


def load_experiment_config_yaml(path: str | Path) -> ExperimentConfig:
    """Load a synthetic-only experiment scaffold YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Experiment config YAML must contain a mapping.")
    allowed = {
        "experiment_name",
        "seed",
        "num_steps",
        "batch_size",
        "gamma",
        "k",
        "index_backend",
        "index_metric",
        "use_persistent_index",
        "index_refresh_every",
        "index_encoder_policy",
        "target_tau",
        "target_update_every",
        "index_ema_tau",
        "output_dir",
        "checkpoint_every",
        "resume_from",
        "eval_num_batches",
        "eval_batch_size",
        "baseline_names",
        "retrieval_correction_mode",
        "support_correction_weight",
        "support_correction_target",
        "support_correction_metric",
        "support_weighting_mode",
        "support_weight_scale",
        "support_weight_min",
        "support_weight_max",
        "support_pmr_eps",
        "synthetic_only",
        "num_transitions",
        "num_actions",
        "state_dim",
        "action_dim",
        "embedding_dim",
    }
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown experiment config key(s): {sorted(unknown)}.")
    profile = default_smoke_profile()
    for key, value in loaded.items():
        if key in SmokeProfile.__dataclass_fields__:
            setattr(profile, key, value)
    config = experiment_config_from_smoke_profile(
        validate_smoke_profile(profile),
        experiment_name=str(loaded.get("experiment_name", "synthetic_debug")),
    )
    for key, value in loaded.items():
        if hasattr(config, key):
            if key == "baseline_names":
                if not isinstance(value, list | tuple):
                    raise ValueError("baseline_names must be a YAML list or tuple.")
                value = tuple(value)
            setattr(config, key, value)
    config.synthetic_only = bool(loaded.get("synthetic_only", True))
    config.synthetic = SyntheticDatasetConfig(
        num_transitions=int(loaded.get("num_transitions", config.synthetic.num_transitions)),
        state_dim=int(loaded.get("state_dim", config.synthetic.state_dim)),
        action_dim=int(loaded.get("action_dim", config.synthetic.action_dim)),
        num_actions=int(loaded.get("num_actions", config.synthetic.num_actions)),
        gamma=float(config.gamma),
        seed=int(config.seed),
    )
    config.model = ModelConfig(
        state_dim=config.synthetic.state_dim,
        action_dim=config.synthetic.action_dim,
        embedding_dim=int(loaded.get("embedding_dim", config.model.embedding_dim)),
    )
    return validate_experiment_config(config)


def build_smoke_components(profile: SmokeProfile) -> SmokeComponents:
    """Construct synthetic data, model, optimizer, target, and generator.

    This helper is smoke-only. Imports are local to avoid making ``config.py``
    depend on training modules during ordinary config-object imports.
    """

    validated = validate_smoke_profile(profile)
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
        raise ModuleNotFoundError("PyTorch is required to build smoke components.") from exc

    from .networks import build_networks
    from .synthetic import make_synthetic_offline_data
    from .target_networks import clone_target_network

    data = make_synthetic_offline_data(synthetic_config_from_smoke_profile(validated))
    torch.manual_seed(validated.seed)
    model = build_networks(
        ModelConfig(
            state_dim=validated.state_dim,
            action_dim=validated.action_dim,
            embedding_dim=validated.embedding_dim,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=validated.lr)
    target_model = clone_target_network(model) if validated.target_tau is not None else None
    generator = torch.Generator().manual_seed(validated.seed)
    return SmokeComponents(
        profile=validated,
        data=data,
        model=model,
        optimizer=optimizer,
        target_model=target_model,
        generator=generator,
    )


@dataclass(slots=True)
class DWMQLObjectiveConfig:
    """Experiment-oriented objective configuration for DW-MQL minimax training.

    Bundles all objective hyperparameters for the minimax critic/density/dual
    update cycle. Separate from SmokeProfile and ExperimentConfig so experiment
    scripts can swap objectives without changing data or network configs.
    """

    gamma: float = 0.99
    bellman_weight: float = 1.0
    stationarity_weight: float = 1.0
    density_regularization_weight: float = 1e-4
    support_correction_weight: float = 0.0
    density_cap: float | None = None
    normalize_density_weights: bool = True
    detach_bellman_target: bool = True
    detach_density_weights_for_critic: bool = True
    dual_ascent: bool = True
    extragradient: bool = False
    extragradient_step_size: float = 1.0
    critic_steps: int = 1
    density_steps: int = 1
    dual_steps: int = 1
    max_grad_norm: float | None = None


def validate_dwmql_objective_config(config: DWMQLObjectiveConfig) -> DWMQLObjectiveConfig:
    """Validate and return a DWMQLObjectiveConfig."""

    if not isinstance(config, DWMQLObjectiveConfig):
        raise TypeError("config must be a DWMQLObjectiveConfig.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    for name in ("bellman_weight", "stationarity_weight", "density_regularization_weight", "support_correction_weight"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or float(value) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if config.density_cap is not None:
        if not isinstance(config.density_cap, int | float) or float(config.density_cap) <= 0:
            raise ValueError("density_cap must be None or positive.")
    for name in ("normalize_density_weights", "detach_bellman_target", "detach_density_weights_for_critic", "dual_ascent", "extragradient"):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"{name} must be a bool.")
    if not isinstance(config.extragradient_step_size, int | float) or float(config.extragradient_step_size) <= 0:
        raise ValueError("extragradient_step_size must be positive.")
    for name in ("critic_steps", "density_steps", "dual_steps"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config.max_grad_norm is not None:
        if not isinstance(config.max_grad_norm, int | float) or float(config.max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be None or positive.")
    if float(config.bellman_weight) == 0 and float(config.stationarity_weight) == 0 and float(config.density_regularization_weight) == 0:
        raise ValueError("At least one of bellman_weight, stationarity_weight, density_regularization_weight must be positive.")
    return config


def load_smoke_profile_yaml(path: str | Path) -> SmokeProfile:
    """Load a smoke-only YAML profile.

    The YAML file may contain any ``SmokeProfile`` field. Missing fields keep
    their defaults and unknown keys are rejected. This is intentionally not a
    full experiment config system.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Smoke profile YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Smoke profile YAML must contain a mapping.")
    allowed = set(SmokeProfile.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown smoke profile key(s): {sorted(unknown)}.")
    profile = default_smoke_profile()
    for key, value in loaded.items():
        setattr(profile, key, value)
    return validate_smoke_profile(profile)


# ---------------------------------------------------------------------------
# KuaiRand-Pure chronological support protocol config (Phase 4)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KuaiRandChronoProtocolConfig:
    """Config for the claim-free KuaiRand-Pure chronological support protocol.

    Mirrors MovieLensChronoProtocolConfig but loads from KuaiRand-Pure CSVs.
    data_dir must point to the directory containing log_standard_*_pure.csv and
    video_features_basic_pure.csv. Results are diagnostic-only; no recommender
    benchmark claims are permitted.
    """

    protocol_name: str = "kuairand_chrono_support_protocol"
    data_dir: str = "data/kuairand/KuaiRand-Pure/data"
    output_dir: str = "runs/kuairand_chrono_protocol"
    seed: int = 0
    num_steps: int = 5000
    batch_size: int = 64
    gamma: float = 0.99
    k_values: tuple[int, ...] = (5, 10, 20)
    train_k: int = 10
    lr: float = 3e-4
    embedding_dim: int = 32
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    max_rows: "int | None" = None
    temperature: float = 1.0
    index_backend: str = "exact"
    index_metric: str = "dot"
    use_persistent_index: bool = True
    index_refresh_every: int = 5
    index_encoder_policy: str = "online"
    index_ema_tau: "float | None" = None
    target_tau: "float | None" = 0.05
    target_update_every: int = 1
    max_grad_norm: "float | None" = 5.0
    q_l2_regularization_weight: float = 0.0
    q_value_clip: "float | None" = None
    reward_scale: float = 1.0
    critic_method: str = "dwmql"
    retrieval_correction_mode: str = "aware"
    density_regularization_weight: float = 1e-4
    stationarity_weight: float = 1.0
    bellman_weight: float = 1.0
    scaffold_only: bool = True
    benchmark_claim: bool = False
    save_model_checkpoint: bool = False
    device: str = "cpu"
    ranking_margin_weight: float = 0.0
    ranking_margin: float = 1.0
    ranking_negative_samples: int = 20
    retrieval_alignment_weight: float = 0.0
    retrieval_alignment_margin: float = 1.0
    retrieval_alignment_warmup_steps: int = 0
    retrieval_alignment_loss_type: str = "listwise_kl"
    listwise_teacher_temperature: float = 0.2
    listwise_retrieval_temperature: float = 0.07
    q_degeneracy_std_threshold: float = 1e-3
    q_degeneracy_margin_threshold: float = 1e-4
    max_eval_rows: "int | None" = None
    cql_weight: float = 0.0
    cql_num_samples: int = 64
    bcq_weight: float = 0.0


def validate_kuairand_chrono_protocol_config(
    config: KuaiRandChronoProtocolConfig,
) -> KuaiRandChronoProtocolConfig:
    """Validate the KuaiRand chronological support protocol config."""
    from pathlib import Path as _Path
    if not isinstance(config, KuaiRandChronoProtocolConfig):
        raise TypeError("config must be a KuaiRandChronoProtocolConfig.")
    if not isinstance(config.protocol_name, str) or not config.protocol_name:
        raise ValueError("protocol_name must be a non-empty string.")
    if not isinstance(config.data_dir, str) or not config.data_dir:
        raise ValueError("data_dir must be a non-empty string.")
    if not _Path(config.data_dir).exists():
        raise FileNotFoundError(f"KuaiRand data_dir does not exist: {config.data_dir}")
    if not _Path(config.data_dir).is_dir():
        raise ValueError("data_dir must be a directory.")
    if not isinstance(config.output_dir, str) or not config.output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name in ("num_steps", "batch_size", "train_k", "embedding_dim", "index_refresh_every", "target_update_every"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.k_values, tuple) or not config.k_values:
        raise TypeError("k_values must be a non-empty tuple of positive integers.")
    for k in config.k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Each k in k_values must be a positive integer.")
    if not isinstance(config.gamma, int | float) or not 0.0 <= float(config.gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isinstance(config.lr, int | float) or float(config.lr) <= 0:
        raise ValueError("lr must be positive.")
    for name in ("validation_fraction", "test_fraction"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if float(config.validation_fraction) + float(config.test_fraction) >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1.")
    if config.max_rows is not None and (not isinstance(config.max_rows, int) or config.max_rows <= 0):
        raise ValueError("max_rows must be None or a positive integer.")
    if config.max_eval_rows is not None and (not isinstance(config.max_eval_rows, int) or config.max_eval_rows <= 0):
        raise ValueError("max_eval_rows must be None or a positive integer.")
    if not isinstance(config.cql_weight, int | float) or float(config.cql_weight) < 0:
        raise ValueError("cql_weight must be a nonnegative float.")
    if not isinstance(config.cql_num_samples, int) or config.cql_num_samples <= 0:
        raise ValueError("cql_num_samples must be a positive integer.")
    if not isinstance(config.bcq_weight, int | float) or float(config.bcq_weight) < 0:
        raise ValueError("bcq_weight must be a nonnegative float.")
    _validate_component_identity_fields(config)
    if not isinstance(config.temperature, int | float) or float(config.temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if config.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if config.index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if config.scaffold_only is not True:
        raise ValueError("scaffold_only must be true for the KuaiRand chrono protocol.")
    if config.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false for the KuaiRand chrono protocol.")
    if not isinstance(config.retrieval_alignment_weight, int | float) or float(config.retrieval_alignment_weight) < 0:
        raise ValueError("retrieval_alignment_weight must be a nonnegative float.")
    if config.retrieval_alignment_loss_type not in {"hinge", "listwise_kl"}:
        raise ValueError("retrieval_alignment_loss_type must be 'hinge' or 'listwise_kl'.")
    return config


def load_kuairand_chrono_protocol_config(path: "str | Path") -> KuaiRandChronoProtocolConfig:
    """Load a KuaiRand chronological support protocol YAML config."""
    from pathlib import Path as _Path
    config_path = _Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"KuaiRand chrono protocol config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("KuaiRand chrono protocol config YAML must contain a mapping.")
    allowed = set(KuaiRandChronoProtocolConfig.__dataclass_fields__)
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown KuaiRand chrono protocol config key(s): {sorted(unknown)}.")
    config = KuaiRandChronoProtocolConfig()
    for key, value in loaded.items():
        if key == "k_values":
            if not isinstance(value, list | tuple):
                raise ValueError("k_values must be a YAML list.")
            value = tuple(int(v) for v in value)
        setattr(config, key, value)
    return validate_kuairand_chrono_protocol_config(config)
