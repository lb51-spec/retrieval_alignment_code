"""Tests for lightweight smoke profile configs."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import (
    MovieLensChronoProtocolConfig,
    SmokeComponents,
    SmokeProfile,
    build_smoke_components,
    component_identity_from_config,
    default_smoke_profile,
    load_smoke_profile_yaml,
    smoke_profile_from_args,
    synthetic_config_from_smoke_profile,
    validate_movielens_chrono_protocol_config,
    validate_smoke_profile,
)
from retrieval_alignment.train import build_arg_parser


def test_default_smoke_profile_returns_valid_profile() -> None:
    profile = default_smoke_profile()

    assert isinstance(profile, SmokeProfile)
    assert profile.index_backend == "exact"
    assert profile.index_metric == "dot"
    assert profile.index_nlist is None
    assert profile.index_nprobe is None
    assert profile.index_refresh_every == 1
    assert profile.use_persistent_index is False
    assert profile.audit_stale_index is False
    assert profile.audit_encoder_policy_overlap is False
    assert profile.index_encoder_policy == "online"
    assert profile.index_ema_tau is None
    assert validate_smoke_profile(profile) is profile


def _write_real_data_ratings(path: Path) -> None:
    path.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,5.0,100\n"
        "1,11,3.0,101\n"
        "2,10,4.5,102\n"
        "2,12,2.0,103\n",
        encoding="utf-8",
    )


def test_zero_density_regularizer_does_not_imply_density_off(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_real_data_ratings(ratings_path)
    config = MovieLensChronoProtocolConfig(
        data_path=str(ratings_path),
        critic_method="dqn",
        retrieval_correction_mode="aware",
        density_regularization_weight=0.0,
    )

    identity = component_identity_from_config(config)

    assert identity["density_enabled"] is True
    assert identity["critic_method"] == "dqn"


def test_no_density_mode_has_unit_weight_component_identity(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_real_data_ratings(ratings_path)
    config = MovieLensChronoProtocolConfig(
        data_path=str(ratings_path),
        critic_method="dqn",
        retrieval_correction_mode="no_density",
        retrieval_alignment_weight=0.1,
    )

    validated = validate_movielens_chrono_protocol_config(config)
    identity = component_identity_from_config(validated)

    assert identity["density_enabled"] is False
    assert identity["retrieval_alignment_enabled"] is True
    assert identity["retrieval_projection_enabled"] is True


def test_cql_component_identity_requires_positive_cql_weight(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    _write_real_data_ratings(ratings_path)
    config = MovieLensChronoProtocolConfig(
        data_path=str(ratings_path),
        critic_method="cql",
        cql_weight=0.0,
    )

    with pytest.raises(ValueError, match="cql_weight"):
        validate_movielens_chrono_protocol_config(config)


def test_validate_smoke_profile_catches_invalid_k() -> None:
    with pytest.raises(ValueError, match="k"):
        validate_smoke_profile(replace(default_smoke_profile(), k=0))


def test_validate_smoke_profile_catches_k_greater_than_num_actions() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        validate_smoke_profile(replace(default_smoke_profile(), k=99, num_actions=8))


def test_validate_smoke_profile_catches_invalid_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        validate_smoke_profile(replace(default_smoke_profile(), gamma=1.5))


def test_validate_smoke_profile_catches_invalid_lr() -> None:
    with pytest.raises(ValueError, match="lr"):
        validate_smoke_profile(replace(default_smoke_profile(), lr=0.0))


def test_validate_smoke_profile_catches_invalid_target_tau() -> None:
    with pytest.raises(ValueError, match="target_tau"):
        validate_smoke_profile(replace(default_smoke_profile(), target_tau=1.5))


def test_validate_smoke_profile_catches_invalid_index_backend() -> None:
    with pytest.raises(ValueError, match="index_backend"):
        validate_smoke_profile(replace(default_smoke_profile(), index_backend="bad"))


def test_validate_smoke_profile_accepts_faiss_ivf_backend() -> None:
    profile = replace(default_smoke_profile(), index_backend="faiss_ivf", index_nlist=2, index_nprobe=1)

    assert validate_smoke_profile(profile) is profile


def test_validate_smoke_profile_catches_invalid_index_metric() -> None:
    with pytest.raises(ValueError, match="index_metric"):
        validate_smoke_profile(replace(default_smoke_profile(), index_metric="ip"))


def test_validate_smoke_profile_catches_invalid_index_nlist() -> None:
    with pytest.raises(ValueError, match="index_nlist"):
        validate_smoke_profile(replace(default_smoke_profile(), index_nlist=0))


def test_validate_smoke_profile_catches_invalid_index_nprobe() -> None:
    with pytest.raises(ValueError, match="index_nprobe"):
        validate_smoke_profile(replace(default_smoke_profile(), index_nprobe=0))


def test_validate_smoke_profile_catches_invalid_index_refresh_every() -> None:
    with pytest.raises(ValueError, match="index_refresh_every"):
        validate_smoke_profile(replace(default_smoke_profile(), index_refresh_every=0))


def test_validate_smoke_profile_catches_invalid_use_persistent_index_type() -> None:
    with pytest.raises(TypeError, match="use_persistent_index"):
        validate_smoke_profile(replace(default_smoke_profile(), use_persistent_index="yes"))  # type: ignore[arg-type]


def test_validate_smoke_profile_catches_invalid_audit_stale_index_type() -> None:
    with pytest.raises(TypeError, match="audit_stale_index"):
        validate_smoke_profile(replace(default_smoke_profile(), audit_stale_index="yes"))  # type: ignore[arg-type]


def test_validate_smoke_profile_catches_invalid_audit_encoder_policy_overlap_type() -> None:
    with pytest.raises(TypeError, match="audit_encoder_policy_overlap"):
        validate_smoke_profile(replace(default_smoke_profile(), audit_encoder_policy_overlap="yes"))  # type: ignore[arg-type]


def test_validate_smoke_profile_encoder_policy_overlap_requires_persistent_index() -> None:
    with pytest.raises(ValueError, match="audit_encoder_policy_overlap"):
        validate_smoke_profile(replace(default_smoke_profile(), audit_encoder_policy_overlap=True))

    profile = replace(
        default_smoke_profile(),
        use_persistent_index=True,
        audit_encoder_policy_overlap=True,
    )
    assert validate_smoke_profile(profile) is profile


def test_validate_smoke_profile_catches_invalid_index_encoder_policy() -> None:
    with pytest.raises(ValueError, match="index_encoder_policy"):
        validate_smoke_profile(replace(default_smoke_profile(), index_encoder_policy="bad"))


def test_validate_smoke_profile_target_index_encoder_policy_requires_target_tau() -> None:
    with pytest.raises(ValueError, match="target_tau"):
        validate_smoke_profile(replace(default_smoke_profile(), index_encoder_policy="target"))

    profile = replace(default_smoke_profile(), index_encoder_policy="target", target_tau=0.5)
    assert validate_smoke_profile(profile) is profile


def test_validate_smoke_profile_accepts_ema_index_encoder_policy() -> None:
    profile = replace(default_smoke_profile(), index_encoder_policy="ema", index_ema_tau=0.05)

    assert validate_smoke_profile(profile) is profile


def test_validate_smoke_profile_ema_policy_requires_index_ema_tau() -> None:
    with pytest.raises(ValueError, match="index_ema_tau"):
        validate_smoke_profile(replace(default_smoke_profile(), index_encoder_policy="ema"))


def test_validate_smoke_profile_catches_invalid_index_ema_tau() -> None:
    with pytest.raises(ValueError, match="index_ema_tau"):
        validate_smoke_profile(replace(default_smoke_profile(), index_ema_tau=1.5))


def test_validate_smoke_profile_enforces_assert_gradient_flow_requires_alternating() -> None:
    with pytest.raises(ValueError, match="assert_gradient_flow"):
        validate_smoke_profile(
            replace(default_smoke_profile(), assert_gradient_flow=True, alternating_smoke=False)
        )


def test_validate_smoke_profile_enforces_no_encoder_on_critic_requires_alternating() -> None:
    with pytest.raises(ValueError, match="no_encoder_on_critic"):
        validate_smoke_profile(
            replace(default_smoke_profile(), no_encoder_on_critic=True, alternating_smoke=False)
        )


def test_smoke_profile_from_args_works_with_argparse_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])

    profile = smoke_profile_from_args(args)

    assert isinstance(profile, SmokeProfile)
    assert profile == default_smoke_profile()


def test_load_smoke_profile_yaml_loads_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "seed: 3\nbatch_size: 4\nk: 2\nnum_actions: 5\nindex_backend: faiss_ivf\nindex_metric: cosine\nindex_nlist: 2\nindex_nprobe: 1\n",
        encoding="utf-8",
    )

    profile = load_smoke_profile_yaml(config_path)

    assert profile.seed == 3
    assert profile.batch_size == 4
    assert profile.k == 2
    assert profile.num_actions == 5
    assert profile.index_backend == "faiss_ivf"
    assert profile.index_metric == "cosine"
    assert profile.index_nlist == 2
    assert profile.index_nprobe == 1


def test_load_smoke_profile_yaml_overrides_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("num_steps: 2\nlr: 0.001\n", encoding="utf-8")

    profile = load_smoke_profile_yaml(config_path)

    assert profile.num_steps == 2
    assert profile.lr == 0.001
    assert profile.batch_size == default_smoke_profile().batch_size


def test_load_smoke_profile_yaml_unknown_key_raises_value_error(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown smoke profile"):
        load_smoke_profile_yaml(config_path)


def test_load_smoke_profile_yaml_invalid_value_raises_value_error(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("k: 99\nnum_actions: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="num_actions"):
        load_smoke_profile_yaml(config_path)


def test_load_smoke_profile_yaml_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_smoke_profile_yaml("does-not-exist.yaml")


def test_smoke_profile_from_args_loads_yaml_then_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("batch_size: 3\nnum_steps: 2\nindex_metric: cosine\n", encoding="utf-8")
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--config", str(config_path), "--batch-size", "5", "--index-backend", "exact", "--index-metric", "dot"]
    )

    profile = smoke_profile_from_args(args)

    assert profile.batch_size == 5
    assert profile.num_steps == 2
    assert profile.index_backend == "exact"
    assert profile.index_metric == "dot"


def test_smoke_profile_from_args_accepts_ema_index_encoder_policy() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--index-encoder-policy", "ema", "--index-ema-tau", "0.05"])

    profile = smoke_profile_from_args(args)

    assert profile.index_encoder_policy == "ema"
    assert profile.index_ema_tau == 0.05


def test_synthetic_config_from_smoke_profile_creates_expected_fields() -> None:
    profile = default_smoke_profile()

    config = synthetic_config_from_smoke_profile(profile)

    assert config.num_transitions == profile.num_transitions
    assert config.state_dim == profile.state_dim
    assert config.action_dim == profile.action_dim
    assert config.num_actions == profile.num_actions
    assert config.gamma == profile.gamma
    assert config.seed == profile.seed


def test_build_smoke_components_returns_expected_objects() -> None:
    profile = default_smoke_profile()
    profile.num_transitions = 16
    profile.num_actions = 8
    profile.k = 3

    components = build_smoke_components(profile)

    assert isinstance(components, SmokeComponents)
    assert components.profile is profile
    assert len(components.data.dataset) == profile.num_transitions
    assert components.model.config.state_dim == profile.state_dim
    assert components.optimizer is not None
    assert components.generator is not None
    assert components.target_model is None


def test_build_smoke_components_target_model_is_frozen_when_tau_set() -> None:
    profile = default_smoke_profile()
    profile.num_transitions = 16
    profile.num_actions = 8
    profile.k = 3
    profile.target_tau = 0.5

    components = build_smoke_components(profile)

    assert components.target_model is not None
    assert all(not parameter.requires_grad for parameter in components.target_model.parameters())


def test_build_smoke_components_invalid_profile_raises_value_error() -> None:
    profile = default_smoke_profile()
    profile.k = 0

    with pytest.raises(ValueError, match="k"):
        build_smoke_components(profile)
