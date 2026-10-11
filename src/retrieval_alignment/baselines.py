"""Dependency-free baseline interfaces and trivial synthetic baselines.

These baselines are scaffolding only. They exist to exercise future benchmark
plumbing without adding real offline RL baselines or making performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

try:  # pragma: no cover - dependency guard.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .evaluation import EvaluationConfig, validate_evaluation_config


class BaselineAlgorithm(Protocol):
    """Protocol for future baseline algorithms."""

    name: str
    family: str
    synthetic_only: bool
    metadata: dict[str, Any]

    def fit(self, dataset: Any, action_features: Any | None = None, action_ids: Any | None = None) -> "BaselineAlgorithm":
        """Fit or initialize the baseline from an offline dataset."""

    def predict_action_scores(self, states: Any, action_features: Any) -> Any:
        """Return scores with shape ``[batch, num_actions]``."""

    def select_actions(self, states: Any, action_features: Any, k: int | None = None) -> Any:
        """Return selected action row indices with shape ``[batch, k]``."""

    def evaluate(self, task: Any, evaluation_config: EvaluationConfig | None = None) -> "BaselineResult":
        """Evaluate the baseline on a synthetic task scaffold."""


@dataclass(slots=True)
class BaselineResult:
    """Result container for trivial synthetic baseline diagnostics."""

    baseline_name: str
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    synthetic_only: bool = True
    benchmark_claim: bool = False
    implemented: bool = True


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for baseline scaffolds.")


def _validate_state_action_inputs(states: "torch.Tensor", action_features: "torch.Tensor") -> None:
    if states.ndim != 2:
        raise ValueError("states must have shape [batch, state_dim].")
    if action_features.ndim != 2:
        raise ValueError("action_features must have shape [num_actions, action_dim].")
    if states.shape[0] == 0 or action_features.shape[0] == 0:
        raise ValueError("states and action_features must be non-empty.")
    if torch.any(~torch.isfinite(states)) or torch.any(~torch.isfinite(action_features)):
        raise ValueError("states and action_features must be finite.")


def _validate_k(k: int | None, num_actions: int) -> int:
    selected_k = num_actions if k is None else k
    if not isinstance(selected_k, int) or selected_k <= 0:
        raise ValueError("k must be None or a positive integer.")
    if selected_k > num_actions:
        raise ValueError("k must not exceed the number of actions.")
    return selected_k


@dataclass(slots=True)
class RandomCandidateBaseline:
    """No-learning random-score baseline for synthetic plumbing tests."""

    seed: int = 0
    name: str = "random"
    family: str = "trivial_synthetic"
    synthetic_only: bool = True
    metadata: dict[str, Any] = field(default_factory=lambda: {"learned": False, "benchmark_claim": False})

    def fit(self, dataset: Any, action_features: Any | None = None, action_ids: Any | None = None) -> "RandomCandidateBaseline":
        del action_features, action_ids
        self.metadata["num_fit_transitions"] = len(dataset) if hasattr(dataset, "__len__") else None
        return self

    def predict_action_scores(self, states: "torch.Tensor", action_features: "torch.Tensor") -> "torch.Tensor":
        _require_torch()
        _validate_state_action_inputs(states, action_features)
        generator = torch.Generator(device=states.device).manual_seed(int(self.seed))
        return torch.rand(
            (states.shape[0], action_features.shape[0]),
            generator=generator,
            device=states.device,
            dtype=states.dtype,
        )

    def select_actions(self, states: "torch.Tensor", action_features: "torch.Tensor", k: int | None = None) -> "torch.Tensor":
        scores = self.predict_action_scores(states, action_features)
        selected_k = _validate_k(k, int(action_features.shape[0]))
        return torch.topk(scores, k=selected_k, dim=1).indices

    def evaluate(self, task: Any, evaluation_config: EvaluationConfig | None = None) -> BaselineResult:
        return evaluate_synthetic_baseline(self, task, evaluation_config)


@dataclass(slots=True)
class NearestActionBaseline:
    """No-learning state-action dot-product baseline.

    State vectors are deterministically truncated or zero-padded to action
    dimension before scoring actions by dot product.
    """

    seed: int = 0
    name: str = "nearest"
    family: str = "trivial_synthetic"
    synthetic_only: bool = True
    metadata: dict[str, Any] = field(default_factory=lambda: {"learned": False, "benchmark_claim": False})

    def fit(self, dataset: Any, action_features: Any | None = None, action_ids: Any | None = None) -> "NearestActionBaseline":
        del action_features, action_ids
        self.metadata["num_fit_transitions"] = len(dataset) if hasattr(dataset, "__len__") else None
        return self

    def _project_states(self, states: "torch.Tensor", action_dim: int) -> "torch.Tensor":
        if states.shape[1] == action_dim:
            return states
        if states.shape[1] > action_dim:
            return states[:, :action_dim]
        pad_width = action_dim - states.shape[1]
        return torch.nn.functional.pad(states, (0, pad_width))

    def predict_action_scores(self, states: "torch.Tensor", action_features: "torch.Tensor") -> "torch.Tensor":
        _require_torch()
        _validate_state_action_inputs(states, action_features)
        projected_states = self._project_states(states, int(action_features.shape[1]))
        scores = projected_states @ action_features.T
        if torch.any(~torch.isfinite(scores)):
            raise ValueError("NearestActionBaseline scores must be finite.")
        return scores

    def select_actions(self, states: "torch.Tensor", action_features: "torch.Tensor", k: int | None = None) -> "torch.Tensor":
        scores = self.predict_action_scores(states, action_features)
        selected_k = _validate_k(k, int(action_features.shape[0]))
        return torch.topk(scores, k=selected_k, dim=1).indices

    def evaluate(self, task: Any, evaluation_config: EvaluationConfig | None = None) -> BaselineResult:
        return evaluate_synthetic_baseline(self, task, evaluation_config)


_NOT_IMPLEMENTED_MSG = (
    "{cls} is a placeholder and has not been implemented yet. "
    "Implement the algorithm before using this baseline in experiments."
)


@dataclass(slots=True)
class BehaviorCloningBaseline:
    """Placeholder for a future behavior cloning baseline. Not yet implemented."""

    seed: int = 0
    name: str = "behavior_cloning"
    family: str = "imitation_learning"
    synthetic_only: bool = False
    metadata: dict[str, Any] = field(default_factory=lambda: {"learned": True, "implemented": False, "benchmark_claim": False})

    def fit(self, dataset: Any, action_features: Any | None = None, action_ids: Any | None = None) -> "BehaviorCloningBaseline":
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="BehaviorCloningBaseline"))

    def predict_action_scores(self, states: Any, action_features: Any) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="BehaviorCloningBaseline"))

    def select_actions(self, states: Any, action_features: Any, k: int | None = None) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="BehaviorCloningBaseline"))

    def evaluate(self, task: Any, evaluation_config: EvaluationConfig | None = None) -> BaselineResult:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="BehaviorCloningBaseline"))


@dataclass(slots=True)
class CQLBaseline:
    """Placeholder for a future Conservative Q-Learning baseline. Not yet implemented."""

    seed: int = 0
    name: str = "cql"
    family: str = "offline_rl"
    synthetic_only: bool = False
    metadata: dict[str, Any] = field(default_factory=lambda: {"learned": True, "implemented": False, "benchmark_claim": False})

    def fit(self, dataset: Any, action_features: Any | None = None, action_ids: Any | None = None) -> "CQLBaseline":
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="CQLBaseline"))

    def predict_action_scores(self, states: Any, action_features: Any) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="CQLBaseline"))

    def select_actions(self, states: Any, action_features: Any, k: int | None = None) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="CQLBaseline"))

    def evaluate(self, task: Any, evaluation_config: EvaluationConfig | None = None) -> BaselineResult:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="CQLBaseline"))


@dataclass(slots=True)
class IQLBaseline:
    """Placeholder for a future Implicit Q-Learning baseline. Not yet implemented."""

    seed: int = 0
    name: str = "iql"
    family: str = "offline_rl"
    synthetic_only: bool = False
    metadata: dict[str, Any] = field(default_factory=lambda: {"learned": True, "implemented": False, "benchmark_claim": False})

    def fit(self, dataset: Any, action_features: Any | None = None, action_ids: Any | None = None) -> "IQLBaseline":
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="IQLBaseline"))

    def predict_action_scores(self, states: Any, action_features: Any) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="IQLBaseline"))

    def select_actions(self, states: Any, action_features: Any, k: int | None = None) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="IQLBaseline"))

    def evaluate(self, task: Any, evaluation_config: EvaluationConfig | None = None) -> BaselineResult:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="IQLBaseline"))


@dataclass(slots=True)
class TD3BCBaseline:
    """Placeholder for a future TD3+BC baseline. Not yet implemented."""

    seed: int = 0
    name: str = "td3_bc"
    family: str = "offline_rl"
    synthetic_only: bool = False
    metadata: dict[str, Any] = field(default_factory=lambda: {"learned": True, "implemented": False, "benchmark_claim": False})

    def fit(self, dataset: Any, action_features: Any | None = None, action_ids: Any | None = None) -> "TD3BCBaseline":
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="TD3BCBaseline"))

    def predict_action_scores(self, states: Any, action_features: Any) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="TD3BCBaseline"))

    def select_actions(self, states: Any, action_features: Any, k: int | None = None) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="TD3BCBaseline"))

    def evaluate(self, task: Any, evaluation_config: EvaluationConfig | None = None) -> BaselineResult:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(cls="TD3BCBaseline"))


PLACEHOLDER_BASELINE_NAMES: frozenset[str] = frozenset({"behavior_cloning", "cql", "iql", "td3_bc"})


def get_baseline(name: str, **kwargs: Any) -> BaselineAlgorithm:
    """Return a baseline by name.

    Implemented: random, nearest.
    Placeholders (raise NotImplementedError on use): behavior_cloning, cql, iql, td3_bc.
    """

    normalized = name.lower()
    if normalized == "random":
        return RandomCandidateBaseline(**kwargs)
    if normalized == "nearest":
        return NearestActionBaseline(**kwargs)
    if normalized == "behavior_cloning":
        return BehaviorCloningBaseline(**kwargs)
    if normalized in ("cql", "conservative_q_learning"):
        return CQLBaseline(**kwargs)
    if normalized in ("iql", "implicit_q_learning"):
        return IQLBaseline(**kwargs)
    if normalized in ("td3_bc", "td3bc"):
        return TD3BCBaseline(**kwargs)
    raise ValueError(
        f"Unknown baseline '{name}'. "
        f"Implemented: random, nearest. "
        f"Placeholders (raise NotImplementedError on use): {sorted(PLACEHOLDER_BASELINE_NAMES)}."
    )


def evaluate_synthetic_baseline(
    baseline: BaselineAlgorithm,
    task: Any,
    evaluation_config: EvaluationConfig | None = None,
) -> BaselineResult:
    """Run minimal synthetic baseline diagnostics.

    This helper checks that a baseline can fit, score, and select actions on the
    synthetic task interface. It does not compare against DW-MQL or claim
    policy performance.
    """

    _require_torch()
    config = validate_evaluation_config(evaluation_config or EvaluationConfig(num_batches=1))
    dataset = task.make_dataset()
    action_features, action_ids = task.make_action_catalogue()
    batch_size = config.batch_size or min(8, len(dataset))
    k = config.k or min(5, int(action_features.shape[0]))
    generator = torch.Generator().manual_seed(int(getattr(baseline, "seed", 0)))
    batch = dataset.sample(batch_size, generator=generator)
    baseline.fit(dataset, action_features=action_features, action_ids=action_ids)
    scores = baseline.predict_action_scores(batch.states, action_features)
    selected = baseline.select_actions(batch.states, action_features, k=k)
    if scores.shape != (batch.batch_size, action_features.shape[0]):
        raise ValueError("Baseline scores must have shape [batch, num_actions].")
    if selected.shape != (batch.batch_size, k):
        raise ValueError("Selected actions must have shape [batch, k].")
    if torch.any(~torch.isfinite(scores)):
        raise ValueError("Baseline scores must be finite.")
    return BaselineResult(
        baseline_name=baseline.name,
        metrics={
            "mean_score": float(scores.mean().item()),
            "max_score": float(scores.max().item()),
            "num_selected": int(selected.numel()),
        },
        diagnostics={
            "score_shape": tuple(int(dim) for dim in scores.shape),
            "selected_shape": tuple(int(dim) for dim in selected.shape),
            "action_ids_shape": tuple(int(dim) for dim in action_ids.shape),
            "synthetic_only": True,
            "benchmark_claim": False,
        },
        synthetic_only=True,
        benchmark_claim=False,
    )
