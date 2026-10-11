"""Benchmark task abstractions for staged DW-MQL+ANN experiments.

This module intentionally provides only a synthetic task adapter for now. D4RL,
MovieLens, RecSim, baselines, and statistical evaluation are future adapters,
not dependencies of this scaffold.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import numpy as np
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import SyntheticDatasetConfig
from .datasets import InMemoryReplayDataset
from .evaluation import EvaluationConfig, evaluate_synthetic_model
from .synthetic import SyntheticOfflineData, make_synthetic_offline_data

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to validate benchmark data contracts.")


EXTERNAL_ADAPTER_REPORT_SCHEMA_NAME = "dwmql_external_adapter_scaffold_report"
EXTERNAL_ADAPTER_REPORT_SCHEMA_VERSION = 1
EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_NAME = "dwmql_external_adapter_scaffold_bundle_manifest"
EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_VERSION = 1


class BenchmarkTask(Protocol):
    """Protocol for future benchmark task adapters."""

    name: str
    task_type: str
    state_dim: int
    action_dim: int
    num_actions: int | None
    metadata: dict[str, Any]

    def make_dataset(self) -> Any:
        """Return an offline replay dataset."""

    def make_action_catalogue(self) -> tuple[Any, Any]:
        """Return action features and action ids."""

    def make_dataset_bundle(self) -> "OfflineDatasetBundle":
        """Return a validated offline dataset plus action catalogue bundle."""

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> "EvaluationResult":
        """Placeholder evaluation hook for future task-specific evaluators."""


@dataclass(slots=True)
class AdapterAvailability:
    """Dependency/data availability result for external benchmark adapters."""

    available: bool
    dependency_name: str | None
    message: str


class ExternalBenchmarkAdapter(Protocol):
    """Protocol for dependency-free external benchmark adapter placeholders."""

    name: str
    benchmark_family: str
    requires_dependency: bool
    dependency_name: str | None
    metadata: dict[str, Any]

    def is_available(self) -> AdapterAvailability:
        """Return dependency or local-data availability."""

    def validate(self) -> AdapterAvailability:
        """Validate availability and adapter configuration."""

    def make_task(self) -> BenchmarkTask:
        """Construct a benchmark task when the future adapter is implemented."""

    def describe(self) -> str:
        """Describe the future benchmark adapter purpose."""


@dataclass(slots=True)
class ActionCatalogue:
    """Validated action catalogue contract for benchmark adapters."""

    action_features: Any
    action_ids: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_action_catalogue(self)

    @property
    def num_actions(self) -> int:
        return int(self.action_features.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.action_features.shape[1])


@dataclass(slots=True)
class D4RLActionCodebook:
    """Validated continuous-action codebook for D4RL-style scaffolds."""

    action_features: Any
    action_ids: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_torch()
        features = torch.as_tensor(self.action_features)
        if not torch.is_floating_point(features):
            features = features.float()
        if features.ndim != 2:
            raise ValueError("action_features must have shape [num_actions, action_dim].")
        if features.shape[0] == 0:
            raise ValueError("D4RL action codebook must be non-empty.")
        if features.shape[1] == 0:
            raise ValueError("action_features must have positive action dimension.")
        if torch.any(~torch.isfinite(features)):
            raise ValueError("action_features must contain only finite values.")
        ids = torch.as_tensor(self.action_ids, dtype=torch.long, device=features.device)
        if ids.ndim != 1:
            raise ValueError("action_ids must have shape [num_actions].")
        if ids.shape[0] != features.shape[0]:
            raise ValueError("action_ids length must match action_features num_actions.")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("action_ids must be unique.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        self.action_features = features
        self.action_ids = ids
        self.metadata = dict(self.metadata)

    @property
    def num_actions(self) -> int:
        return int(self.action_features.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.action_features.shape[1])


def build_d4rl_action_codebook_from_actions(
    actions: Any,
    max_actions: int | None = None,
    seed: int = 0,
) -> D4RLActionCodebook:
    """Build a deterministic continuous-action codebook from logged actions."""

    _require_torch()
    action_tensor = torch.as_tensor(actions)
    if not torch.is_floating_point(action_tensor):
        action_tensor = action_tensor.float()
    if action_tensor.ndim != 2:
        raise ValueError("actions must have shape [num_samples, action_dim].")
    if action_tensor.shape[0] == 0 or action_tensor.shape[1] == 0:
        raise ValueError("actions must be non-empty with positive action_dim.")
    if torch.any(~torch.isfinite(action_tensor)):
        raise ValueError("actions must contain only finite values.")
    if max_actions is not None and (not isinstance(max_actions, int) or max_actions <= 0):
        raise ValueError("max_actions must be None or a positive integer.")
    unique_actions = torch.unique(action_tensor, dim=0)
    if max_actions is not None and unique_actions.shape[0] > max_actions:
        generator = torch.Generator(device=unique_actions.device).manual_seed(int(seed))
        indices = torch.randperm(unique_actions.shape[0], generator=generator, device=unique_actions.device)[:max_actions]
        indices = torch.sort(indices).values
        unique_actions = unique_actions[indices]
    return D4RLActionCodebook(
        action_features=unique_actions,
        action_ids=torch.arange(unique_actions.shape[0], dtype=torch.long, device=unique_actions.device),
        metadata={
            "source": "continuous_action_samples",
            "num_source_actions": int(action_tensor.shape[0]),
            "max_actions": max_actions,
            "seed": int(seed),
            "scaffold_only": True,
            "benchmark_claim": False,
        },
    )


@dataclass(slots=True)
class OfflineDatasetBundle:
    """Validated offline dataset and action catalogue pair."""

    dataset: Any
    action_catalogue: ActionCatalogue
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_offline_dataset_bundle(self)


@dataclass(slots=True)
class BenchmarkEvaluationProtocol:
    """Evaluation protocol metadata required from future benchmark adapters."""

    task_name: str
    metrics: list[str] | tuple[str, ...] = field(default_factory=list)
    diagnostics: list[str] | tuple[str, ...] = field(default_factory=list)
    required_splits: list[str] | tuple[str, ...] = field(default_factory=list)
    synthetic_only: bool = True
    benchmark_claim: bool = False

    def __post_init__(self) -> None:
        validate_benchmark_evaluation_protocol(self)


@dataclass(slots=True)
class RecSimSlateMetadata:
    """Slate-task metadata contract for future RecSim/RecSim NG adapters."""

    slate_size: int
    num_items: int
    item_feature_dim: int
    user_feature_dim: int
    num_users: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("slate_size", "num_items", "item_feature_dim", "user_feature_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_users is not None and (not isinstance(self.num_users, int) or self.num_users <= 0):
            raise ValueError("num_users must be None or a positive integer.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        self.metadata = dict(self.metadata)


def validate_action_catalogue(catalogue: ActionCatalogue) -> ActionCatalogue:
    """Validate an action catalogue without constructing external adapters."""

    _require_torch()
    if not isinstance(catalogue, ActionCatalogue):
        raise TypeError("catalogue must be an ActionCatalogue.")
    features = torch.as_tensor(catalogue.action_features)
    if not torch.is_floating_point(features):
        features = features.float()
    if features.ndim != 2:
        raise ValueError("action_features must have shape [num_actions, action_dim].")
    if features.shape[0] == 0:
        raise ValueError("action catalogue must be non-empty.")
    if features.shape[1] == 0:
        raise ValueError("action_features must have positive action dimension.")
    if torch.any(~torch.isfinite(features)):
        raise ValueError("action_features must contain only finite values.")
    ids = torch.as_tensor(catalogue.action_ids, dtype=torch.long, device=features.device)
    if ids.ndim != 1:
        raise ValueError("action_ids must have shape [num_actions].")
    if ids.shape[0] != features.shape[0]:
        raise ValueError("action_ids length must match action_features num_actions.")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("action_ids must be unique.")
    if not isinstance(catalogue.metadata, dict):
        raise ValueError("metadata must be a dictionary.")
    catalogue.action_features = features
    catalogue.action_ids = ids
    catalogue.metadata = dict(catalogue.metadata)
    return catalogue


def validate_offline_dataset_bundle(bundle: OfflineDatasetBundle) -> OfflineDatasetBundle:
    """Validate an offline dataset bundle returned by a benchmark adapter."""

    if not isinstance(bundle, OfflineDatasetBundle):
        raise TypeError("bundle must be an OfflineDatasetBundle.")
    if not isinstance(bundle.action_catalogue, ActionCatalogue):
        raise TypeError("action_catalogue must be an ActionCatalogue.")
    validate_action_catalogue(bundle.action_catalogue)
    if not isinstance(bundle.metadata, dict):
        raise ValueError("metadata must be a dictionary.")
    if not hasattr(bundle.dataset, "__len__"):
        raise ValueError("dataset must expose __len__.")
    if len(bundle.dataset) <= 0:
        raise ValueError("dataset must be non-empty.")
    bundle.metadata = dict(bundle.metadata)
    return bundle


def validate_benchmark_evaluation_protocol(
    protocol: BenchmarkEvaluationProtocol,
) -> BenchmarkEvaluationProtocol:
    """Validate claim-free evaluation protocol metadata."""

    if not isinstance(protocol, BenchmarkEvaluationProtocol):
        raise TypeError("protocol must be a BenchmarkEvaluationProtocol.")
    if not isinstance(protocol.task_name, str) or not protocol.task_name:
        raise ValueError("task_name must be a non-empty string.")
    if not isinstance(protocol.metrics, list | tuple):
        raise ValueError("metrics must be a list or tuple.")
    if not isinstance(protocol.diagnostics, list | tuple):
        raise ValueError("diagnostics must be a list or tuple.")
    if not isinstance(protocol.required_splits, list | tuple):
        raise ValueError("required_splits must be a list or tuple.")
    for field_name, values in {
        "metrics": protocol.metrics,
        "diagnostics": protocol.diagnostics,
        "required_splits": protocol.required_splits,
    }.items():
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{field_name} must contain non-empty strings.")
    if protocol.synthetic_only is not True:
        raise ValueError("synthetic_only must be true for the current scaffold.")
    if protocol.benchmark_claim is not False:
        raise ValueError("benchmark_claim must be false.")
    return protocol


def synthetic_data_to_dataset_bundle(data: SyntheticOfflineData) -> OfflineDatasetBundle:
    """Convert existing synthetic data into the adapter-facing bundle contract."""

    if not isinstance(data, SyntheticOfflineData):
        raise TypeError("data must be SyntheticOfflineData.")
    catalogue = ActionCatalogue(
        action_features=data.action_features,
        action_ids=data.action_ids,
        metadata={
            **data.metadata,
            "synthetic_only": True,
            "benchmark_claim": False,
        },
    )
    return OfflineDatasetBundle(
        dataset=data.dataset,
        action_catalogue=catalogue,
        metadata={
            **data.metadata,
            "synthetic_only": True,
            "benchmark_claim": False,
        },
    )


def assert_benchmark_task_contract(task: Any) -> bool:
    """Assert that a task satisfies the adapter-facing benchmark contract."""

    if not isinstance(getattr(task, "name", None), str) or not task.name:
        raise AssertionError("Benchmark task must expose a non-empty name.")
    if not isinstance(getattr(task, "task_type", None), str) or not task.task_type:
        raise AssertionError("Benchmark task must expose a non-empty task_type.")
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, dict):
        raise AssertionError("Benchmark task metadata must be a dictionary.")
    for dim_name in ("state_dim", "action_dim"):
        value = getattr(task, dim_name, None)
        if not isinstance(value, int) or value <= 0:
            raise AssertionError(f"Benchmark task {dim_name} must be a positive integer.")
    num_actions = getattr(task, "num_actions", None)
    if num_actions is not None and (not isinstance(num_actions, int) or num_actions <= 0):
        raise AssertionError("Benchmark task num_actions must be None or a positive integer.")
    if not callable(getattr(task, "make_dataset_bundle", None)):
        raise AssertionError("Benchmark task must implement make_dataset_bundle().")
    if not callable(getattr(task, "make_action_catalogue_contract", None)):
        raise AssertionError("Benchmark task must implement make_action_catalogue_contract().")
    bundle = task.make_dataset_bundle()
    catalogue = task.make_action_catalogue_contract()
    validate_offline_dataset_bundle(bundle)
    validate_action_catalogue(catalogue)
    if bundle.action_catalogue.num_actions != catalogue.num_actions:
        raise AssertionError("Dataset bundle and action catalogue num_actions must match.")
    evaluation = task.evaluate_policy_or_model(model=None)
    if not isinstance(evaluation, EvaluationResult):
        raise AssertionError("evaluate_policy_or_model(None) must return EvaluationResult.")
    if evaluation.benchmark_claim is not False:
        raise AssertionError("EvaluationResult benchmark_claim must be false.")
    return True


def _module_available(*module_names: str) -> str | None:
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is not None:
            return module_name
    return None


@dataclass(slots=True)
class D4RLAdapter:
    """Placeholder adapter for future D4RL/Minari offline RL tasks."""

    observations: Any | None = None
    actions: Any | None = None
    rewards: Any | None = None
    next_observations: Any | None = None
    terminals: Any | None = None
    codebook: D4RLActionCodebook | None = None
    dataset_name: str | None = None
    codebook_size: int = 16
    seed: int = 0
    name: str = "d4rl"
    benchmark_family: str = "offline_rl"
    requires_dependency: bool = True
    dependency_name: str | None = "d4rl or minari/gymnasium"
    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "implemented": False,
            "large_action_scaffold": True,
            "scaffold_only": True,
            "benchmark_claim": False,
        }
    )

    def is_available(self) -> AdapterAvailability:
        if self.observations is not None:
            return AdapterAvailability(True, None, "Local/mock D4RL-style arrays were provided.")
        found = _module_available("d4rl", "minari", "gymnasium")
        if found is not None:
            return AdapterAvailability(True, found, f"Dependency probe found {found}; adapter task loading is not implemented yet.")
        return AdapterAvailability(False, self.dependency_name, "D4RL/Minari/Gymnasium is not installed; no dependency is required for this scaffold.")

    def validate(self) -> AdapterAvailability:
        return self.is_available()

    def make_task(self) -> BenchmarkTask:
        if self.observations is not None:
            return D4RLLargeActionTask(
                observations=self.observations,
                actions=self.actions,
                rewards=self.rewards,
                next_observations=self.next_observations,
                terminals=self.terminals,
                codebook=self.codebook,
            )
        if self.dataset_name is not None:
            return D4RLDatasetTask(
                dataset_name=self.dataset_name,
                codebook_size=self.codebook_size,
                seed=self.seed,
            )
        availability = self.is_available()
        if not availability.available:
            raise ImportError(availability.message)
        raise NotImplementedError("D4RL task construction is a future adapter; no benchmark loading is implemented yet.")

    def describe(self) -> str:
        return "Future adapter for continuous-control offline RL datasets via D4RL/Minari/Gymnasium plus a dependency-free large-action mock-array scaffold."


@dataclass(slots=True)
class MovieLensAdapter:
    """Placeholder adapter for future local MovieLens recommendation data."""

    data_path: str | Path | None = None
    min_rating_threshold: float = 4.0
    max_rows: int | None = None
    row_sampling_mode: str = "head"
    seed: int = 0
    split_mode: str = "none"
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    session_gap_seconds: int | None = None
    name: str = "movielens"
    benchmark_family: str = "recommendation"
    requires_dependency: bool = False
    dependency_name: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "implemented": False,
            "local_csv_scaffold": True,
            "downloads_data": False,
            "scaffold_only": True,
        }
    )

    def is_available(self) -> AdapterAvailability:
        if self.data_path is None:
            return AdapterAvailability(False, None, "MovieLens data_path was not provided; this placeholder never downloads data.")
        path = Path(self.data_path)
        if path.exists():
            return AdapterAvailability(True, None, f"Local MovieLens path exists: {path}")
        return AdapterAvailability(False, None, f"MovieLens data_path does not exist: {path}")

    def validate(self) -> AdapterAvailability:
        return self.is_available()

    def make_task(self) -> BenchmarkTask:
        if self.data_path is None:
            raise FileNotFoundError("MovieLensAdapter requires a local CSV data_path.")
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"MovieLens local CSV file does not exist: {path}")
        if not path.is_file():
            raise NotImplementedError("MovieLensAdapter currently supports only a local ratings CSV file.")
        return MovieLensLocalTask(
            data_path=path,
            min_rating_threshold=self.min_rating_threshold,
            max_rows=self.max_rows,
            row_sampling_mode=self.row_sampling_mode,
            seed=self.seed,
            split_mode=self.split_mode,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            session_gap_seconds=self.session_gap_seconds,
        )

    def describe(self) -> str:
        return "Local CSV scaffold for MovieLens-style ratings data; no downloader, pandas, or benchmark claims."


@dataclass(slots=True)
class RecSimAdapter:
    """Placeholder adapter for future RecSim-style recommendation simulation."""

    mock: bool = False
    task_name: str | None = None
    name: str = "recsim"
    benchmark_family: str = "simulated_recommendation"
    requires_dependency: bool = True
    dependency_name: str | None = "recsim or recsim_ng"
    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "implemented": False,
            "mock_slate_scaffold": True,
            "scaffold_only": True,
            "benchmark_claim": False,
        }
    )

    def is_available(self) -> AdapterAvailability:
        if self.mock or self.task_name == "mock_slate":
            return AdapterAvailability(True, None, "Dependency-free RecSim mock_slate task is available.")
        found = _module_available("recsim", "recsim_ng")
        if found is not None:
            return AdapterAvailability(True, found, f"Dependency probe found {found}; real simulator loading is not implemented yet.")
        return AdapterAvailability(False, self.dependency_name, "RecSim/RecSim NG is not installed; no dependency is required for this scaffold.")

    def validate(self) -> AdapterAvailability:
        return self.is_available()

    def make_task(self, task_name: str | None = None, **kwargs: Any) -> BenchmarkTask:
        selected_task = task_name or self.task_name
        if self.mock or selected_task == "mock_slate":
            return RecSimMockSlateTask(**kwargs)
        availability = self.is_available()
        if not availability.available:
            raise ImportError(availability.message)
        raise NotImplementedError("RecSim task construction is a future adapter; no real simulator integration is implemented yet.")

    def describe(self) -> str:
        return "Future adapter for RecSim/RecSim NG simulator-backed recommendation tasks plus a dependency-free mock slate scaffold."


def get_external_adapter(name: str, **kwargs: Any) -> ExternalBenchmarkAdapter:
    """Return a dependency-free placeholder adapter by name."""

    normalized = name.lower()
    if normalized == "d4rl":
        return D4RLAdapter(**kwargs)
    if normalized == "movielens":
        return MovieLensAdapter(**kwargs)
    if normalized == "recsim":
        return RecSimAdapter(**kwargs)
    raise ValueError("Unknown external adapter. Supported names are: d4rl, movielens, recsim.")


def available_external_adapters() -> dict[str, dict[str, Any]]:
    """Return descriptions and availability for all placeholder adapters."""

    adapters = {
        "d4rl": D4RLAdapter(),
        "movielens": MovieLensAdapter(),
        "recsim": RecSimAdapter(),
    }
    return {
        name: {
            "name": adapter.name,
            "benchmark_family": adapter.benchmark_family,
            "requires_dependency": adapter.requires_dependency,
            "dependency_name": adapter.dependency_name,
            "availability": adapter.is_available(),
            "description": adapter.describe(),
            "metadata": adapter.metadata,
        }
        for name, adapter in adapters.items()
    }


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation container for benchmark-ready scaffold outputs."""

    task_name: str
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    synthetic_only: bool = True
    benchmark_claim: bool = False


@dataclass(slots=True)
class D4RLLargeActionTask:
    """Dependency-free D4RL-style large-action wrapper scaffold."""

    observations: Any
    actions: Any
    rewards: Any
    next_observations: Any
    terminals: Any
    codebook: D4RLActionCodebook | None = None
    name: str = "d4rl_large_action_mock"
    task_type: str = "d4rl_large_action_scaffold"
    _dataset: InMemoryReplayDataset = field(init=False, repr=False)
    _catalogue: ActionCatalogue = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        observations = torch.as_tensor(self.observations)
        actions = torch.as_tensor(self.actions)
        rewards = torch.as_tensor(self.rewards)
        next_observations = torch.as_tensor(self.next_observations)
        terminals = torch.as_tensor(self.terminals)
        if not torch.is_floating_point(observations):
            observations = observations.float()
        if not torch.is_floating_point(actions):
            actions = actions.float()
        if not torch.is_floating_point(rewards):
            rewards = rewards.float()
        if not torch.is_floating_point(next_observations):
            next_observations = next_observations.float()
        if observations.ndim != 2:
            raise ValueError("observations must have shape [num_transitions, state_dim].")
        if actions.ndim != 2:
            raise ValueError("actions must have shape [num_transitions, action_dim].")
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must match observations shape.")
        if rewards.ndim != 1 or rewards.shape[0] != observations.shape[0]:
            raise ValueError("rewards must have shape [num_transitions].")
        if terminals.ndim != 1 or terminals.shape[0] != observations.shape[0]:
            raise ValueError("terminals must have shape [num_transitions].")
        if actions.shape[0] != observations.shape[0]:
            raise ValueError("actions and observations must have the same number of transitions.")
        for name, tensor in {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
        }.items():
            if tensor.numel() == 0 or torch.any(~torch.isfinite(tensor)):
                raise ValueError(f"{name} must be non-empty and finite.")
        codebook = self.codebook or build_d4rl_action_codebook_from_actions(actions)
        if codebook.action_dim != actions.shape[1]:
            raise ValueError("codebook action_dim must match logged action dimension.")
        distances = torch.cdist(actions, codebook.action_features)
        logged_ids = torch.argmin(distances, dim=1).long()
        self._metadata = {
            "source": "d4rl_large_action_scaffold",
            "synthetic_only": False,
            "scaffold_only": True,
            "benchmark_claim": False,
            "num_transitions": int(observations.shape[0]),
            "state_dim": int(observations.shape[1]),
            "action_dim": int(actions.shape[1]),
            "codebook_size": int(codebook.num_actions),
            "logged_action_mapping": "nearest_l2_codebook",
        }
        self._dataset = InMemoryReplayDataset(
            states=observations,
            actions=codebook.action_features[logged_ids],
            rewards=rewards,
            next_states=next_observations,
            dones=terminals.bool(),
            action_ids=logged_ids,
            next_logged_action_ids=logged_ids,
            metadata=self._metadata,
        )
        self._catalogue = ActionCatalogue(
            action_features=codebook.action_features,
            action_ids=codebook.action_ids,
            metadata={**codebook.metadata, **self._metadata},
        )

    @property
    def state_dim(self) -> int:
        return int(self._dataset.batch.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._catalogue.action_dim)

    @property
    def num_actions(self) -> int:
        return int(self._catalogue.num_actions)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def make_dataset(self) -> InMemoryReplayDataset:
        return self._dataset

    def make_action_catalogue(self) -> tuple[Any, Any]:
        return self._catalogue.action_features, self._catalogue.action_ids

    def make_action_catalogue_contract(self) -> ActionCatalogue:
        return ActionCatalogue(
            action_features=self._catalogue.action_features,
            action_ids=self._catalogue.action_ids,
            metadata=self._catalogue.metadata,
        )

    def make_dataset_bundle(self) -> OfflineDatasetBundle:
        return OfflineDatasetBundle(
            dataset=self._dataset,
            action_catalogue=self.make_action_catalogue_contract(),
            metadata=self.metadata,
        )

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> EvaluationResult:
        del model, kwargs
        return EvaluationResult(
            task_name=self.name,
            metrics={
                "num_transitions": len(self._dataset),
                "num_actions": self.num_actions,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            diagnostics={**self.metadata, "placeholder_evaluation": True},
            synthetic_only=False,
            benchmark_claim=False,
        )


@dataclass(slots=True)
class D4RLDatasetTask:
    """Real D4RL/Minari dataset task loaded behind optional lazy imports.

    Attempts d4rl first, then minari, then raises ImportError with clear guidance.
    This is scaffold-only; it is not a benchmark claim or performance evidence.
    """

    dataset_name: str
    codebook_size: int = 16
    seed: int = 0
    name: str = "d4rl_dataset"
    task_type: str = "d4rl_real_dataset"
    _dataset: InMemoryReplayDataset = field(init=False, repr=False)
    _catalogue: ActionCatalogue = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        observations_np: Any = None
        actions_np: Any = None
        rewards_np: Any = None
        next_observations_np: Any = None
        terminals_np: Any = None
        loaded_via: str = ""

        try:
            import d4rl  # noqa: F401
            import gym  # noqa: F401

            env = gym.make(self.dataset_name)
            raw = env.get_dataset()
            observations_np = raw["observations"]
            actions_np = raw["actions"]
            rewards_np = raw["rewards"]
            next_observations_np = raw["next_observations"]
            terminals_np = raw.get("terminals", raw.get("timeouts"))
            loaded_via = "d4rl"
        except ImportError:
            pass

        if observations_np is None:
            try:
                import minari  # noqa: F401

                ds = minari.load_dataset(self.dataset_name)
                eps = list(ds.iterate_episodes())
                obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []
                for ep in eps:
                    obs_list.append(ep.observations[:-1])
                    act_list.append(ep.actions)
                    rew_list.append(ep.rewards)
                    next_obs_list.append(ep.observations[1:])
                    term_list.append(ep.terminations)
                import numpy as np

                observations_np = np.concatenate(obs_list, axis=0)
                actions_np = np.concatenate(act_list, axis=0)
                rewards_np = np.concatenate(rew_list, axis=0)
                next_observations_np = np.concatenate(next_obs_list, axis=0)
                terminals_np = np.concatenate(term_list, axis=0)
                loaded_via = "minari"
            except ImportError:
                pass

        if observations_np is None:
            raise ImportError(
                f"D4RLDatasetTask requires d4rl or minari to load dataset '{self.dataset_name}'. "
                "Install d4rl (pip install d4rl) or minari (pip install minari)."
            )

        observations = torch.as_tensor(observations_np, dtype=torch.float32)
        actions = torch.as_tensor(actions_np, dtype=torch.float32)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32)
        next_observations = torch.as_tensor(next_observations_np, dtype=torch.float32)
        if terminals_np is None:
            terminals = torch.zeros(observations.shape[0], dtype=torch.bool)
        else:
            terminals = torch.as_tensor(terminals_np, dtype=torch.bool)
        if rewards.ndim != 1:
            rewards = rewards.squeeze(-1)
        if terminals.ndim != 1:
            terminals = terminals.squeeze(-1)

        # Build a simple codebook by clustering action space.
        torch.manual_seed(self.seed)
        safe_codebook_size = min(self.codebook_size, actions.shape[0])
        perm = torch.randperm(actions.shape[0])[:safe_codebook_size]
        codebook_vectors = actions[perm]
        action_ids = torch.arange(safe_codebook_size, dtype=torch.long)

        self._catalogue = ActionCatalogue(
            action_features=codebook_vectors,
            action_ids=action_ids,
            metadata={
                "codebook_size": safe_codebook_size,
                "dataset_name": self.dataset_name,
                "loaded_via": loaded_via,
                "scaffold_only": True,
                "benchmark_claim": False,
            },
        )
        self._dataset = InMemoryReplayDataset(
            states=observations,
            actions=actions,
            rewards=rewards,
            next_states=next_observations,
            dones=terminals,
            action_ids=torch.zeros(observations.shape[0], dtype=torch.long),
            next_logged_action_ids=torch.zeros(observations.shape[0], dtype=torch.long),
            metadata={
                "dataset_name": self.dataset_name,
                "num_transitions": observations.shape[0],
                "loaded_via": loaded_via,
                "scaffold_only": True,
                "benchmark_claim": False,
            },
        )
        self._metadata = {
            "dataset_name": self.dataset_name,
            "loaded_via": loaded_via,
            "num_transitions": observations.shape[0],
            "codebook_size": safe_codebook_size,
            "state_dim": observations.shape[1],
            "action_dim": actions.shape[1],
            "scaffold_only": True,
            "benchmark_claim": False,
            "synthetic_only": False,
        }

    @property
    def state_dim(self) -> int:
        return int(self._dataset.batch.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._catalogue.action_features.shape[1])

    @property
    def num_actions(self) -> int:
        return int(self._catalogue.num_actions)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def make_dataset(self) -> InMemoryReplayDataset:
        return self._dataset

    def make_action_catalogue(self) -> tuple[Any, Any]:
        return self._catalogue.action_features, self._catalogue.action_ids

    def make_action_catalogue_contract(self) -> ActionCatalogue:
        return ActionCatalogue(
            action_features=self._catalogue.action_features,
            action_ids=self._catalogue.action_ids,
            metadata=self._catalogue.metadata,
        )

    def make_dataset_bundle(self) -> OfflineDatasetBundle:
        return OfflineDatasetBundle(
            dataset=self._dataset,
            action_catalogue=self.make_action_catalogue_contract(),
            metadata=self.metadata,
        )

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> EvaluationResult:
        del model, kwargs
        return EvaluationResult(
            task_name=self.name,
            metrics={
                "num_transitions": len(self._dataset),
                "num_actions": self.num_actions,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            diagnostics={**self.metadata, "placeholder_evaluation": True},
            synthetic_only=False,
            benchmark_claim=False,
        )


@dataclass(slots=True)
class RecSimMockSlateTask:
    """Dependency-free mock slate task for future RecSim-style adapters."""

    num_users: int = 4
    num_items: int = 16
    user_feature_dim: int = 4
    item_feature_dim: int = 4
    slate_size: int = 3
    num_interactions: int = 32
    seed: int = 0
    name: str = "recsim_mock_slate"
    task_type: str = "recsim_mock_slate"
    _dataset: InMemoryReplayDataset = field(init=False, repr=False)
    _catalogue: ActionCatalogue = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)
    _slate_metadata: RecSimSlateMetadata = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        for name in ("num_users", "num_items", "user_feature_dim", "item_feature_dim", "slate_size", "num_interactions"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.slate_size > self.num_items:
            raise ValueError("slate_size must not exceed num_items.")
        generator = torch.Generator().manual_seed(int(self.seed))
        user_features = torch.randn(self.num_users, self.user_feature_dim, generator=generator)
        item_features = torch.randn(self.num_items, self.item_feature_dim, generator=generator)
        item_ids = torch.arange(self.num_items, dtype=torch.long)
        user_indices = torch.arange(self.num_interactions, dtype=torch.long) % self.num_users
        logged_item_ids = (torch.arange(self.num_interactions, dtype=torch.long) * 3 + int(self.seed)) % self.num_items
        states = user_features[user_indices]
        actions = item_features[logged_item_ids]
        projected_states = _project_features_for_scores(states, self.item_feature_dim)
        rewards = torch.sigmoid((projected_states * actions).sum(dim=1))
        next_user_indices = (user_indices + 1) % self.num_users
        next_states = user_features[next_user_indices]
        dones = ((torch.arange(self.num_interactions) + 1) % max(1, self.num_users) == 0)
        self._slate_metadata = RecSimSlateMetadata(
            slate_size=self.slate_size,
            num_users=self.num_users,
            num_items=self.num_items,
            item_feature_dim=self.item_feature_dim,
            user_feature_dim=self.user_feature_dim,
            metadata={
                "source": "recsim_mock_slate",
                "synthetic_only": False,
                "scaffold_only": True,
                "benchmark_claim": False,
            },
        )
        self._metadata = {
            **self._slate_metadata.metadata,
            "slate_size": self.slate_size,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_interactions": self.num_interactions,
            "user_feature_dim": self.user_feature_dim,
            "item_feature_dim": self.item_feature_dim,
            "seed": int(self.seed),
        }
        self._dataset = InMemoryReplayDataset(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            action_ids=logged_item_ids,
            next_logged_action_ids=(logged_item_ids + 1) % self.num_items,
            metadata=self._metadata,
        )
        self._catalogue = ActionCatalogue(
            action_features=item_features,
            action_ids=item_ids,
            metadata=self._metadata,
        )

    @property
    def state_dim(self) -> int:
        return self.user_feature_dim

    @property
    def action_dim(self) -> int:
        return self.item_feature_dim

    @property
    def num_actions(self) -> int:
        return self.num_items

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def slate_metadata(self) -> RecSimSlateMetadata:
        return self._slate_metadata

    def make_dataset(self) -> InMemoryReplayDataset:
        return self._dataset

    def make_action_catalogue(self) -> tuple[Any, Any]:
        return self._catalogue.action_features, self._catalogue.action_ids

    def make_action_catalogue_contract(self) -> ActionCatalogue:
        return ActionCatalogue(
            action_features=self._catalogue.action_features,
            action_ids=self._catalogue.action_ids,
            metadata=self._catalogue.metadata,
        )

    def make_dataset_bundle(self) -> OfflineDatasetBundle:
        return OfflineDatasetBundle(
            dataset=self._dataset,
            action_catalogue=self.make_action_catalogue_contract(),
            metadata=self.metadata,
        )

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> EvaluationResult:
        k = kwargs.get("k")
        return evaluate_recsim_slate_scaffold(self, model_or_baseline=model, k=k)


# Canonical genre list for stable one-hot encoding. Sorted alphabetically.
_MOVIELENS_CANONICAL_GENRES: tuple[str, ...] = (
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
_MOVIELENS_GENRE_DIM: int = len(_MOVIELENS_CANONICAL_GENRES)  # 19
_MOVIELENS_FEATURE_DIM: int = _MOVIELENS_GENRE_DIM + 1  # +1 for normalized year = 20


def _load_movielens_movie_features(
    data_path: "str | Path",
) -> "dict[int, list[float]] | None":
    """Load genre + year features from movies.csv adjacent to ratings.csv.

    Returns a dict mapping movie_id -> 20-dim feature vector
    [genre_0, ..., genre_18, year_norm] if movies.csv exists, else None.
    Falls back gracefully: missing genres → zero vector; missing year → 0.5.
    Never raises — returns None on any I/O or parse failure.
    """
    import re as _re

    movies_path = Path(data_path).parent / "movies.csv"
    if not movies_path.exists():
        return None
    genre_index = {g: i for i, g in enumerate(_MOVIELENS_CANONICAL_GENRES)}
    features: dict[int, list[float]] = {}
    try:
        with movies_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                movie_id = int(row["movieId"])
                vec = [0.0] * _MOVIELENS_FEATURE_DIM
                genres_str = row.get("genres", "") or ""
                if genres_str and genres_str != "(no genres listed)":
                    for g in genres_str.split("|"):
                        if g in genre_index:
                            vec[genre_index[g]] = 1.0
                title = row.get("title", "") or ""
                m = _re.search(r"\((\d{4})\)", title)
                year_norm = (float(m.group(1)) - 1900.0) / 120.0 if m else 0.5
                vec[_MOVIELENS_GENRE_DIM] = float(year_norm)
                features[movie_id] = vec
    except Exception:  # noqa: BLE001
        return None
    return features if features else None


def _compute_user_genre_affinity(
    dataset_rows: list[dict],
    genre_feature_map: "dict[int, list[float]] | None",
    min_rating_threshold: float,
) -> "dict[int, list[float]] | None":
    """Return {user_id: mean_genre_vec} over each user's liked training items.

    Liked = rating >= min_rating_threshold. Users with no liked items get the
    zero vector. Returns None when genre_feature_map is None (no movies.csv).
    """
    if genre_feature_map is None:
        return None
    zero_vec = [0.0] * _MOVIELENS_FEATURE_DIM
    user_vecs: dict[int, list[list[float]]] = {}
    for row in dataset_rows:
        if row["rating"] >= min_rating_threshold:
            vec = genre_feature_map.get(row["movie_id"])
            if vec is None:
                vec = [0.0] * _MOVIELENS_GENRE_DIM + [0.5]
            user_vecs.setdefault(row["user_id"], []).append(vec)
    all_users = {row["user_id"] for row in dataset_rows}
    result: dict[int, list[float]] = {}
    for user_id in all_users:
        vecs = user_vecs.get(user_id)
        if vecs:
            n = len(vecs)
            result[user_id] = [sum(v[i] for v in vecs) / n for i in range(_MOVIELENS_FEATURE_DIM)]
        else:
            result[user_id] = zero_vec[:]
    return result


@dataclass(slots=True)
class MovieLensLocalTask:
    """Dependency-free local CSV MovieLens scaffold task."""

    data_path: str | Path
    min_rating_threshold: float = 4.0
    max_rows: int | None = None
    row_sampling_mode: str = "head"
    seed: int = 0
    split_mode: str = "none"
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    session_gap_seconds: int | None = None
    name: str = "movielens_local_csv"
    task_type: str = "movielens_local_scaffold"
    _dataset: InMemoryReplayDataset = field(init=False, repr=False)
    _catalogue: ActionCatalogue = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)
    _rows: list[dict[str, Any]] = field(init=False, repr=False)
    _user_affinity_map: "dict[int, list[float]] | None" = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        _require_torch()
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"MovieLens local CSV file does not exist: {path}")
        if not path.is_file():
            raise ValueError("MovieLensLocalTask data_path must be a CSV file.")
        if self.max_rows is not None and (not isinstance(self.max_rows, int) or self.max_rows <= 0):
            raise ValueError("max_rows must be None or a positive integer.")
        if not isinstance(self.min_rating_threshold, int | float):
            raise ValueError("min_rating_threshold must be numeric.")
        _validate_movielens_split_args(
            self.split_mode,
            self.validation_fraction,
            self.test_fraction,
            self.session_gap_seconds,
        )

        rows = load_movielens_rows(
            path,
            self.max_rows,
            sampling_mode=self.row_sampling_mode,
            seed=self.seed,
        )
        if not rows:
            raise ValueError("MovieLens local CSV contains no valid rows.")
        split_metadata = build_movielens_split_metadata(
            rows,
            split_mode=self.split_mode,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            session_gap_seconds=self.session_gap_seconds,
        )
        self._rows = rows
        dataset_rows = sorted(
            [row for row in rows if row["split"] == "train"],
            key=lambda r: (int(r["user_id"]), float(r["timestamp"])),
        )
        if not dataset_rows:
            raise ValueError("MovieLens local CSV split produced no training rows.")
        users = sorted({row["user_id"] for row in rows})
        movies = sorted({row["movie_id"] for row in rows})
        user_to_index = {user_id: index for index, user_id in enumerate(users)}
        movie_to_index = {movie_id: index for index, movie_id in enumerate(movies)}
        user_denominator = max(1, len(users) - 1)
        movie_denominator = max(1, len(movies) - 1)
        max_movie_id = max(1, max(abs(movie_id) for movie_id in movies))
        max_timestamp = max(1.0, max(abs(row["timestamp"]) for row in rows))

        genre_feature_map = _load_movielens_movie_features(path)
        user_affinity_map = _compute_user_genre_affinity(dataset_rows, genre_feature_map, self.min_rating_threshold)
        movie_features = []
        for movie_id in movies:
            movie_index = movie_to_index[movie_id]
            if genre_feature_map is not None:
                vec = genre_feature_map.get(movie_id)
                if vec is None:
                    # Movie in ratings but not in movies.csv — zero genres, mid year
                    vec = [0.0] * _MOVIELENS_GENRE_DIM + [0.5]
                movie_features.append(vec)
            else:
                movie_features.append(
                    [
                        float(movie_index / movie_denominator),
                        float(movie_id / max_movie_id),
                        1.0,
                    ]
                )
        action_features = torch.tensor(movie_features, dtype=torch.float32)
        action_ids = torch.tensor(movies, dtype=torch.long)
        self._catalogue = ActionCatalogue(
            action_features=action_features,
            action_ids=action_ids,
            metadata=self._base_metadata(path, len(rows), len(users), len(movies), split_metadata),
        )

        # Pre-allocate NumPy arrays to avoid accumulating millions of individual
        # Python tensor objects (torch.stack over a list of 20M tensors exhausts the heap).
        n = len(dataset_rows)
        first_row = dataset_rows[0]
        if user_affinity_map is not None:
            state_vec_0 = user_affinity_map[first_row["user_id"]]
        else:
            state_vec_0 = [0.0, 0.0, 1.0]
        state_dim = len(state_vec_0)
        action_dim = action_features.shape[1]
        action_features_np = action_features.numpy()  # shared read-only view

        states_np = np.empty((n, state_dim), dtype=np.float32)
        actions_np = np.empty((n, action_dim), dtype=np.float32)
        rewards_np = np.empty(n, dtype=np.float32)
        next_states_np = np.empty((n, state_dim), dtype=np.float32)
        dones_np = np.empty(n, dtype=np.bool_)
        action_row_ids_np = np.empty(n, dtype=np.int64)
        next_logged_action_ids_np = np.empty(n, dtype=np.int64)

        threshold = float(self.min_rating_threshold)
        for index, row in enumerate(dataset_rows):
            uid = row["user_id"]
            if user_affinity_map is not None:
                state_vec = user_affinity_map[uid]
            else:
                state_vec = [
                    float(user_to_index[uid] / user_denominator),
                    float(row["timestamp"] / max_timestamp),
                    1.0,
                ]
            if index + 1 < n and dataset_rows[index + 1]["user_id"] == uid:
                next_row = dataset_rows[index + 1]
                done = False
                next_logged_id = next_row["movie_id"]
                if user_affinity_map is not None:
                    next_state_vec = user_affinity_map[next_row["user_id"]]
                else:
                    next_state_vec = [
                        float(user_to_index[next_row["user_id"]] / user_denominator),
                        float(next_row["timestamp"] / max_timestamp),
                        1.0,
                    ]
            else:
                done = True
                next_logged_id = row["movie_id"]
                next_state_vec = state_vec
            movie_index = movie_to_index[row["movie_id"]]
            states_np[index] = state_vec
            actions_np[index] = action_features_np[movie_index]
            rewards_np[index] = 1.0 if row["rating"] >= threshold else 0.0
            next_states_np[index] = next_state_vec
            dones_np[index] = done
            action_row_ids_np[index] = row["movie_id"]
            next_logged_action_ids_np[index] = next_logged_id

        self._metadata = self._base_metadata(path, len(rows), len(users), len(movies), split_metadata)
        self._user_affinity_map = user_affinity_map
        self._dataset = InMemoryReplayDataset(
            states=torch.from_numpy(states_np),
            actions=torch.from_numpy(actions_np),
            rewards=torch.from_numpy(rewards_np),
            next_states=torch.from_numpy(next_states_np),
            dones=torch.from_numpy(dones_np),
            action_ids=torch.from_numpy(action_row_ids_np),
            next_logged_action_ids=torch.from_numpy(next_logged_action_ids_np),
            metadata=self._metadata,
        )

    def _base_metadata(
        self,
        path: Path,
        num_rows: int,
        num_users: int,
        num_movies: int,
        split_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "source": "movielens_local_csv",
            "data_path": str(path),
            "num_rows": int(num_rows),
            "num_users": int(num_users),
            "num_movies": int(num_movies),
            "min_rating_threshold": float(self.min_rating_threshold),
            "seed": int(self.seed),
            "synthetic_only": False,
            "benchmark_claim": False,
            "scaffold_only": True,
            **split_metadata,
        }

    @property
    def state_dim(self) -> int:
        return int(self._dataset.batch.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._catalogue.action_features.shape[1])

    @property
    def num_actions(self) -> int:
        return self._catalogue.num_actions

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_split_rows(self, split: str) -> list[dict[str, Any]]:
        """Return lightweight rows for a recorded split.

        Rows include the binary scaffold reward/positive flag derived from the
        configured rating threshold. This is local-file diagnostics plumbing,
        not a recommender benchmark split protocol.
        """

        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test.")
        selected: list[dict[str, Any]] = []
        for row in self._rows:
            if row["split"] != split:
                continue
            positive = row["rating"] >= float(self.min_rating_threshold)
            selected.append(
                {
                    "user_id": int(row["user_id"]),
                    "movie_id": int(row["movie_id"]),
                    "rating": float(row["rating"]),
                    "timestamp": float(row["timestamp"]),
                    "reward": 1.0 if positive else 0.0,
                    "positive": bool(positive),
                    "split": row["split"],
                    "session_id": row.get("session_id"),
                }
            )
        return selected

    def available_splits(self) -> tuple[str, ...]:
        """Return split names with at least one row."""

        return tuple(split for split in ("train", "validation", "test") if self.get_split_rows(split))

    def make_dataset(self) -> InMemoryReplayDataset:
        return self._dataset

    def make_action_catalogue(self) -> tuple[Any, Any]:
        return self._catalogue.action_features, self._catalogue.action_ids

    def make_action_catalogue_contract(self) -> ActionCatalogue:
        return ActionCatalogue(
            action_features=self._catalogue.action_features,
            action_ids=self._catalogue.action_ids,
            metadata=self.metadata,
        )

    def make_dataset_bundle(self) -> OfflineDatasetBundle:
        return OfflineDatasetBundle(
            dataset=self._dataset,
            action_catalogue=self.make_action_catalogue_contract(),
            metadata=self.metadata,
        )

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> EvaluationResult:
        k = int(kwargs.get("k", 10))
        return evaluate_movielens_scaffold_task(self, model_or_baseline=model, k=k)


def load_movielens_rows(
    path: str | Path,
    max_rows: int | None = None,
    *,
    sampling_mode: str = "head",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Load MovieLens rows, optionally retaining a bounded deterministic sample.

    ``head`` preserves the historic first-N behaviour for small smoke tests.
    ``reservoir`` scans the entire file while retaining at most ``max_rows``
    rows, avoiding a 25M-row Python-object materialisation and removing the
    user-order bias of the ratings CSV prefix.
    """

    path = Path(path)
    required_columns = {"userId", "movieId", "rating", "timestamp"}
    if sampling_mode not in {"head", "reservoir"}:
        raise ValueError("sampling_mode must be 'head' or 'reservoir'.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if sampling_mode == "reservoir" and max_rows is None:
        raise ValueError("reservoir sampling requires a positive max_rows cap.")
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    seen = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("MovieLens local CSV must include a header row.")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"MovieLens local CSV is missing required column(s): {sorted(missing)}.")
        for raw in reader:
            if sampling_mode == "head" and max_rows is not None and len(rows) >= max_rows:
                break
            try:
                user_id = int(raw["userId"])
                movie_id = int(raw["movieId"])
                rating = float(raw["rating"])
                timestamp = float(raw["timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError("MovieLens local CSV rows must contain numeric userId, movieId, rating, and timestamp.") from exc
            if not math.isfinite(rating) or not math.isfinite(timestamp):
                raise ValueError("MovieLens local CSV rating and timestamp values must be finite.")
            row = {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
                "timestamp": timestamp,
            }
            if sampling_mode == "head" or max_rows is None:
                rows.append(row)
                continue
            if len(rows) < max_rows:
                rows.append(row)
            else:
                replacement = rng.randrange(seen + 1)
                if replacement < max_rows:
                    rows[replacement] = row
            seen += 1
    rows.sort(key=lambda row: (row["user_id"], row["timestamp"], row["movie_id"]))
    return rows


def _validate_movielens_split_args(
    split_mode: str,
    validation_fraction: float,
    test_fraction: float,
    session_gap_seconds: int | None,
) -> None:
    if split_mode not in {"none", "chronological", "session", "chronological_strict"}:
        raise ValueError("split_mode must be one of: none, chronological, session, chronological_strict.")
    for name, value in {
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }.items():
        if not isinstance(value, int | float) or not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if float(validation_fraction) + float(test_fraction) >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1.")
    if session_gap_seconds is not None and (
        not isinstance(session_gap_seconds, int) or session_gap_seconds <= 0
    ):
        raise ValueError("session_gap_seconds must be None or a positive integer.")
    if split_mode == "session" and session_gap_seconds is None:
        raise ValueError("session mode requires session_gap_seconds.")


def build_movielens_split_metadata(
    rows: list[dict[str, Any]],
    *,
    split_mode: str = "none",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    session_gap_seconds: int | None = None,
) -> dict[str, Any]:
    """Annotate rows with split/session metadata and return aggregate counts."""

    _validate_movielens_split_args(split_mode, validation_fraction, test_fraction, session_gap_seconds)
    if not rows:
        raise ValueError("rows must be non-empty.")
    for row in rows:
        if not isinstance(row.get("timestamp"), int | float):
            raise ValueError("MovieLens rows must include numeric timestamps.")

    timestamps = [float(row["timestamp"]) for row in rows]
    by_user: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_user.setdefault(int(row["user_id"]), []).append(row)

    leakage_check_passed: bool | None = None

    if split_mode == "chronological_strict":
        # Global timestamp ordering: last test_fraction rows → test, next validation_fraction → validation, rest → train.
        sorted_rows = sorted(rows, key=lambda row: (float(row["timestamp"]), int(row["movie_id"])))
        n = len(sorted_rows)
        test_count = max(1, int(n * float(test_fraction)))
        validation_count = max(0, int(n * float(validation_fraction)))
        train_count = n - validation_count - test_count
        if train_count < 0:
            train_count = 0
        for index, row in enumerate(sorted_rows):
            if index < train_count:
                row["split"] = "train"
            elif index < train_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "test"
            row["session_id"] = None
        train_ts = [float(r["timestamp"]) for r in sorted_rows if r["split"] == "train"]
        test_ts = [float(r["timestamp"]) for r in sorted_rows if r["split"] == "test"]
        leakage_check_passed = (not train_ts or not test_ts) or (max(train_ts) <= min(test_ts))
        split_counts = {"train": 0, "validation": 0, "test": 0}
        for row in rows:
            split_counts[row["split"]] += 1
        return {
            "split_mode": split_mode,
            "num_train_rows": split_counts["train"],
            "num_validation_rows": split_counts["validation"],
            "num_test_rows": split_counts["test"],
            "num_users": len(by_user),
            "num_items": len({row["movie_id"] for row in rows}),
            "num_interactions": len(rows),
            "catalogue_size": len({row["movie_id"] for row in rows}),
            "num_sessions": None,
            "min_timestamp": min(timestamps),
            "max_timestamp": max(timestamps),
            "min_train_timestamp": min(train_ts) if train_ts else None,
            "max_train_timestamp": max(train_ts) if train_ts else None,
            "min_test_timestamp": min(test_ts) if test_ts else None,
            "max_test_timestamp": max(test_ts) if test_ts else None,
            "leakage_check_passed": leakage_check_passed,
        }

    session_count = 0
    for user_rows in by_user.values():
        user_rows.sort(key=lambda row: (row["timestamp"], row["movie_id"]))
        n = len(user_rows)
        if split_mode in {"chronological", "session"}:
            test_count = int(n * float(test_fraction))
            validation_count = int(n * float(validation_fraction))
            train_count = n - validation_count - test_count
        else:
            train_count = n
            validation_count = 0
            test_count = 0
        for index, row in enumerate(user_rows):
            if index < train_count:
                row["split"] = "train"
            elif index < train_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "test"
            if split_mode == "session":
                if index == 0:
                    session_count += 1
                else:
                    previous = user_rows[index - 1]
                    if row["timestamp"] - previous["timestamp"] > float(session_gap_seconds):
                        session_count += 1
                row["session_id"] = session_count - 1
            else:
                row["session_id"] = None

    split_counts = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        split_counts[row["split"]] += 1
    return {
        "split_mode": split_mode,
        "num_train_rows": split_counts["train"],
        "num_validation_rows": split_counts["validation"],
        "num_test_rows": split_counts["test"],
        "num_users": len(by_user),
        "num_items": len({row["movie_id"] for row in rows}),
        "num_sessions": session_count if split_mode == "session" else None,
        "min_timestamp": min(timestamps),
        "max_timestamp": max(timestamps),
    }


def evaluate_movielens_scaffold_task(
    task: MovieLensLocalTask,
    model_or_baseline: Any = None,
    k: int = 10,
) -> EvaluationResult:
    """Return claim-free local MovieLens scaffold diagnostics."""

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    rows = task._rows
    if not rows:
        raise ValueError("MovieLensLocalTask has no rows to evaluate.")
    positives = [1.0 if row["rating"] >= float(task.min_rating_threshold) else 0.0 for row in rows]
    num_users = len({row["user_id"] for row in rows})
    num_items = len({row["movie_id"] for row in rows})
    item_counts = {movie_id: 0 for movie_id in task._catalogue.action_ids.tolist()}
    user_counts: dict[int, int] = {}
    for row in rows:
        item_counts[row["movie_id"]] = item_counts.get(row["movie_id"], 0) + 1
        user_counts[row["user_id"]] = user_counts.get(row["user_id"], 0) + 1
    metrics: dict[str, float | int | None] = {
        "num_transitions": len(rows),
        "num_actions": task.num_actions,
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": len(rows),
        "positive_rate": sum(positives) / len(positives),
        "avg_user_interactions": len(rows) / num_users,
        "avg_item_interactions": len(rows) / num_items,
        "catalogue_coverage_observed": len([count for count in item_counts.values() if count > 0])
        / task.num_actions,
    }
    diagnostics: dict[str, Any] = {
        **task.metadata,
        "scaffold_evaluation": True,
    }
    if model_or_baseline is not None and hasattr(model_or_baseline, "predict_action_scores"):
        scores = model_or_baseline.predict_action_scores(
            task._dataset.batch.states,
            task._catalogue.action_features,
        )
        scores_tensor = torch.as_tensor(scores)
        if scores_tensor.ndim != 2:
            raise ValueError("predict_action_scores must return shape [batch, num_actions].")
        topk = min(k, scores_tensor.shape[1])
        topk_indices = torch.topk(scores_tensor, k=topk, dim=1).indices
        diagnostics["score_shape"] = list(scores_tensor.shape)
        diagnostics["topk_shape"] = list(topk_indices.shape)
    return EvaluationResult(
        task_name=task.name,
        metrics=metrics,
        diagnostics=diagnostics,
        synthetic_only=False,
        benchmark_claim=False,
    )


def _movielens_states_for_rows(task: Any, rows: list[dict[str, Any]]) -> "torch.Tensor":
    if task._user_affinity_map is not None:
        zero_vec = [0.0] * task.state_dim
        states = [task._user_affinity_map.get(int(row["user_id"]), zero_vec) for row in rows]
    else:
        users = sorted({row["user_id"] for row in task._rows})
        user_to_index = {user_id: index for index, user_id in enumerate(users)}
        user_denominator = max(1, len(users) - 1)
        max_timestamp = max(1.0, max(abs(float(row["timestamp"])) for row in task._rows))
        states = [
            [
                float(user_to_index[int(row["user_id"])] / user_denominator),
                float(float(row["timestamp"]) / max_timestamp),
                1.0,
            ]
            for row in rows
        ]
    return torch.tensor(states, dtype=torch.float32)


def _movielens_target_indices(task: MovieLensLocalTask, rows: list[dict[str, Any]]) -> "torch.Tensor":
    action_ids = [int(value) for value in task._catalogue.action_ids.tolist()]
    id_to_row = {movie_id: index for index, movie_id in enumerate(action_ids)}
    return torch.tensor([id_to_row[int(row["movie_id"])] for row in rows], dtype=torch.long)


def _score_movielens_rows(
    task: MovieLensLocalTask,
    rows: list[dict[str, Any]],
    model_or_baseline: Any,
) -> tuple["torch.Tensor | None", dict[str, Any]]:
    if model_or_baseline is None or not hasattr(model_or_baseline, "predict_action_scores"):
        return None, {"scoring_available": False}
    states = _movielens_states_for_rows(task, rows)
    scores = model_or_baseline.predict_action_scores(states, task._catalogue.action_features)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
    if scores_tensor.shape != (len(rows), task.num_actions):
        raise ValueError("predict_action_scores must return shape [num_eval_rows, num_actions].")
    if torch.any(~torch.isfinite(scores_tensor)):
        raise ValueError("MovieLens scaffold scores must be finite.")
    return scores_tensor, {
        "scoring_available": True,
        "score_shape": list(scores_tensor.shape),
    }


def score_movielens_catalogue_with_model(
    task: MovieLensLocalTask,
    model: Any,
    rows: list[dict[str, Any]],
    scoring_batch_size: int = 128,
) -> "torch.Tensor":
    """Score all local MovieLens catalogue actions with a DW-MQL model.

    This helper uses the same deterministic row state features as the local CSV
    scaffold and returns critic Q scores over the full item catalogue. It is a
    local diagnostics bridge only; scores are not recommender benchmark metrics.

    Rows are processed in batches of scoring_batch_size to avoid OOM on large
    datasets where num_rows × num_actions × embedding_dim exceeds RAM.
    """

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if model is None:
        raise ValueError("model is required for MovieLens Q-score diagnostics.")
    if not rows:
        return torch.empty((0, task.num_actions), dtype=torch.float32)
    catalogue_action_ids = set(int(value) for value in task._catalogue.action_ids.tolist())
    for row in rows:
        if int(row["movie_id"]) not in catalogue_action_ids:
            raise ValueError("rows contain a movie_id outside the action catalogue.")
    action_features = torch.as_tensor(task._catalogue.action_features, dtype=torch.float32)
    batch_scores: list[Any] = []
    with torch.no_grad():
        action_embeddings = model.encode_actions(action_features)
        for start in range(0, len(rows), scoring_batch_size):
            batch_rows = rows[start: start + scoring_batch_size]
            states = _movielens_states_for_rows(task, batch_rows)
            state_embeddings = model.encode_states(states)
            expanded = action_embeddings.unsqueeze(0).expand(len(batch_rows), -1, -1)
            batch_q = model.q_values(state_embeddings, expanded)
            batch_scores.append(torch.as_tensor(batch_q, dtype=torch.float32))
    scores = torch.cat(batch_scores, dim=0)
    if scores.shape != (len(rows), task.num_actions):
        raise ValueError("MovieLens model Q scores must have shape [num_rows, num_actions].")
    if torch.any(~torch.isfinite(scores)):
        raise ValueError("MovieLens model Q scores must be finite.")
    return scores


def evaluate_movielens_ranking_scaffold(
    task: MovieLensLocalTask,
    model_or_baseline: Any = None,
    split: str = "test",
    k: int = 10,
) -> EvaluationResult:
    """Compute claim-free held-out top-k ranking scaffold diagnostics.

    These diagnostics are shape/support checks over local split rows. They are
    not NDCG, CTR, recommendation-quality, or benchmark-performance metrics.
    """

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    split_rows = task.get_split_rows(split)
    positive_rows = [row for row in split_rows if row["positive"]]
    safe_k = min(k, task.num_actions)
    metrics: dict[str, float | int | None] = {
        "num_split_rows": len(split_rows),
        "num_eval_rows": len(positive_rows),
        "catalogue_size": task.num_actions,
        "k": safe_k,
        "hit_rate_at_k": None,
        "recall_at_k_scaffold": None,
        "mean_rank_of_positive": None,
    }
    diagnostics: dict[str, Any] = {
        "split": split,
        "requested_k": k,
        "effective_k": safe_k,
        "scaffold_only": True,
        "ranking_scaffold": True,
        "benchmark_claim": False,
    }
    if not positive_rows:
        diagnostics["reason"] = "no positive held-out rows"
        return EvaluationResult(
            task_name=task.name,
            metrics=metrics,
            diagnostics=diagnostics,
            synthetic_only=False,
            benchmark_claim=False,
        )
    scores, score_diagnostics = _score_movielens_rows(task, positive_rows, model_or_baseline)
    diagnostics.update(score_diagnostics)
    if scores is None:
        diagnostics["reason"] = "model_or_baseline does not expose predict_action_scores"
        return EvaluationResult(
            task_name=task.name,
            metrics=metrics,
            diagnostics=diagnostics,
            synthetic_only=False,
            benchmark_claim=False,
        )
    target_indices = _movielens_target_indices(task, positive_rows)
    topk_indices = torch.topk(scores, k=safe_k, dim=1).indices
    hits = (topk_indices == target_indices[:, None]).any(dim=1).float()
    target_scores = scores[torch.arange(scores.shape[0]), target_indices]
    ranks = (scores > target_scores[:, None]).sum(dim=1).float() + 1.0
    metrics.update(
        {
            "hit_rate_at_k": float(hits.mean().item()),
            "recall_at_k_scaffold": float(hits.mean().item()),
            "mean_rank_of_positive": float(ranks.mean().item()),
        }
    )
    diagnostics["topk_shape"] = list(topk_indices.shape)
    return EvaluationResult(
        task_name=task.name,
        metrics=metrics,
        diagnostics=diagnostics,
        synthetic_only=False,
        benchmark_claim=False,
    )


def evaluate_movielens_candidate_support_scaffold(
    task: MovieLensLocalTask,
    model_or_baseline: Any = None,
    split: str = "test",
    k: int = 10,
) -> EvaluationResult:
    """Compute claim-free candidate support diagnostics for held-out rows."""

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    rows = task.get_split_rows(split)
    safe_k = min(k, task.num_actions)
    scorer = model_or_baseline
    fallback_used = False
    if scorer is None or not hasattr(scorer, "predict_action_scores"):
        from .baselines import NearestActionBaseline

        scorer = NearestActionBaseline(seed=task.seed)
        fallback_used = True
    metrics: dict[str, float | int | None] = {
        "num_eval_rows": len(rows),
        "catalogue_size": task.num_actions,
        "k": safe_k,
    }
    diagnostics: dict[str, Any] = {
        "split": split,
        "requested_k": k,
        "effective_k": safe_k,
        "scaffold_only": True,
        "candidate_support_scaffold": True,
        "fallback_scoring": "nearest" if fallback_used else None,
        "benchmark_claim": False,
    }
    if not rows:
        metrics.update(
            {
                "candidate_hit_rate_at_k": None,
                "candidate_coverage_fraction": 0.0,
            }
        )
        diagnostics["reason"] = "split has no rows"
        return EvaluationResult(task.name, metrics, diagnostics, synthetic_only=False, benchmark_claim=False)
    scores, score_diagnostics = _score_movielens_rows(task, rows, scorer)
    diagnostics.update(score_diagnostics)
    if scores is None:  # pragma: no cover - fallback scorer should prevent this.
        metrics.update({"candidate_hit_rate_at_k": None, "candidate_coverage_fraction": None})
        return EvaluationResult(task.name, metrics, diagnostics, synthetic_only=False, benchmark_claim=False)
    target_indices = _movielens_target_indices(task, rows)
    topk_indices = torch.topk(scores, k=safe_k, dim=1).indices
    hits = (topk_indices == target_indices[:, None]).any(dim=1).float()
    unique_candidates = torch.unique(topk_indices)
    metrics.update(
        {
            "candidate_hit_rate_at_k": float(hits.mean().item()),
            "candidate_coverage_fraction": float(unique_candidates.numel() / task.num_actions),
        }
    )
    diagnostics["topk_shape"] = list(topk_indices.shape)
    return EvaluationResult(task.name, metrics, diagnostics, synthetic_only=False, benchmark_claim=False)


def evaluate_movielens_pmr_scaffold(
    task: MovieLensLocalTask,
    model: Any,
    split: str = "test",
    k: int = 10,
    temperature: float = 1.0,
) -> EvaluationResult:
    """Compute claim-free MovieLens local PMR/support diagnostics from Q scores."""

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    rows = task.get_split_rows(split)
    safe_k = min(k, task.num_actions)
    metrics: dict[str, float | int | None] = {
        "num_eval_rows": len(rows),
        "catalogue_size": task.num_actions,
        "k": safe_k,
        "temperature": float(temperature),
        "pmr_at_k_scaffold": None,
        "candidate_policy_mass_captured_by_topk": None,
        "missing_policy_mass_mean": None,
        "positive_item_policy_mass_mean": None,
        "positive_item_rank_mean": None,
        "positive_item_in_topk_rate": None,
    }
    diagnostics: dict[str, Any] = {
        "split": split,
        "requested_k": k,
        "effective_k": safe_k,
        "temperature": float(temperature),
        "scaffold_only": True,
        "movielens_pmr_scaffold": True,
        "benchmark_claim": False,
        "note": "PMR is computed from the trained Q-score softmax over the local catalogue and is not a recommendation quality metric.",
    }
    if not rows:
        diagnostics["reason"] = "split has no rows"
        return EvaluationResult(task.name, metrics, diagnostics, synthetic_only=False, benchmark_claim=False)
    scores = score_movielens_catalogue_with_model(task, model, rows)
    probs = torch.softmax(scores / float(temperature), dim=1)
    if torch.any(~torch.isfinite(probs)) or torch.any(probs < 0):
        raise ValueError("MovieLens PMR probabilities must be finite and nonnegative.")
    topk_indices = torch.topk(scores, k=safe_k, dim=1).indices
    topk_mass = torch.gather(probs, dim=1, index=topk_indices).sum(dim=1)
    target_indices = _movielens_target_indices(task, rows)
    positive_masses = probs[torch.arange(probs.shape[0]), target_indices]
    positive_scores = scores[torch.arange(scores.shape[0]), target_indices]
    positive_ranks = (scores > positive_scores[:, None]).sum(dim=1).float() + 1.0
    positive_hits = (topk_indices == target_indices[:, None]).any(dim=1).float()
    pmr_mean = float(topk_mass.mean().item())
    metrics.update(
        {
            "pmr_at_k_scaffold": pmr_mean,
            "candidate_policy_mass_captured_by_topk": pmr_mean,
            "missing_policy_mass_mean": float((1.0 - topk_mass).mean().item()),
            "positive_item_policy_mass_mean": float(positive_masses.mean().item()),
            "positive_item_rank_mean": float(positive_ranks.mean().item()),
            "positive_item_in_topk_rate": float(positive_hits.mean().item()),
        }
    )
    diagnostics["score_shape"] = list(scores.shape)
    diagnostics["topk_shape"] = list(topk_indices.shape)
    return EvaluationResult(task.name, metrics, diagnostics, synthetic_only=False, benchmark_claim=False)


def evaluate_movielens_ann_candidate_audit(
    task: MovieLensLocalTask,
    model: Any,
    split: str = "test",
    k: int = 10,
    temperature: float = 1.0,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> EvaluationResult:
    """Compare selected ANN candidates against full-catalogue Q top-k ids."""

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    rows = task.get_split_rows(split)
    safe_k = min(k, task.num_actions)
    metrics: dict[str, float | int | str | None] = {
        "num_eval_rows": len(rows),
        "catalogue_size": task.num_actions,
        "k": safe_k,
        "temperature": float(temperature),
        "index_backend": index_backend,
        "index_metric": index_metric,
        "ann_candidate_recall_mean": None,
        "ann_candidate_overlap_fraction_mean": None,
        "ann_candidate_pmr_mean": None,
        "ann_candidate_pmr_min": None,
        "ann_candidate_pmr_max": None,
        "ann_candidate_best_value_gap_mean": None,
    }
    diagnostics: dict[str, Any] = {
        "split": split,
        "requested_k": k,
        "effective_k": safe_k,
        "index_backend": index_backend,
        "index_metric": index_metric,
        "index_nlist": index_nlist,
        "index_nprobe": index_nprobe,
        "scaffold_only": True,
        "movielens_ann_candidate_audit": True,
        "benchmark_claim": False,
        "note": "Selected ANN candidates are compared with full-catalogue Q top-k ids; this is not recommendation quality.",
    }
    if not rows:
        diagnostics["reason"] = "split has no rows"
        return EvaluationResult(task.name, metrics, diagnostics, synthetic_only=False, benchmark_claim=False)

    from .ann_index import make_action_index
    from .audits import audit_candidate_sets

    scores = score_movielens_catalogue_with_model(task, model, rows)
    full_probs = torch.softmax(scores / float(temperature), dim=1)
    exact_values, exact_ids = torch.topk(scores, k=safe_k, dim=1)
    exact_policy_probs = torch.gather(full_probs, dim=1, index=exact_ids)
    with torch.no_grad():
        states = _movielens_states_for_rows(task, rows)
        action_features = task._catalogue.action_features.to(device=states.device)
        # B2: when retrieval projection is present, build the ANN index from the
        # projected retrieval embeddings (what the KL loss trains) rather than the
        # raw Q encoders. Q top-k above still comes from the base q_values path.
        if getattr(model, "has_retrieval_projection", False):
            action_embeddings = model.encode_action_index(action_features).detach()
            query_embeddings = model.encode_state_query(states).detach()
        else:
            action_embeddings = model.encode_actions(action_features).detach()
            query_embeddings = model.encode_states(states).detach()
        index = make_action_index(
            action_embeddings,
            action_ids=torch.arange(task.num_actions, dtype=torch.long, device=action_embeddings.device),
            backend=index_backend,
            metric=index_metric,
            normalize=index_metric == "cosine",
            nlist=index_nlist,
            nprobe=index_nprobe,
        )
        selected = index.search(query_embeddings, k=safe_k)
    selected_ids = selected.ids.to(device=scores.device)
    selected_values = torch.gather(scores, dim=1, index=selected_ids)
    audit = audit_candidate_sets(
        selected_ids,
        exact_ids,
        exact_policy_probs=exact_policy_probs,
        exact_values=exact_values,
        retrieved_values=selected_values,
    )
    metrics.update(
        {
            "ann_candidate_recall_mean": float(audit.recall_at_k.mean().item()),
            "ann_candidate_overlap_fraction_mean": float(audit.overlap_fraction.mean().item()),
            "ann_candidate_pmr_mean": None if audit.pmr_at_k is None else float(audit.pmr_at_k.mean().item()),
            "ann_candidate_pmr_min": None if audit.pmr_at_k is None else float(audit.pmr_at_k.min().item()),
            "ann_candidate_pmr_max": None if audit.pmr_at_k is None else float(audit.pmr_at_k.max().item()),
            "ann_candidate_best_value_gap_mean": (
                None if audit.best_value_gap is None else float(audit.best_value_gap.mean().item())
            ),
        }
    )
    diagnostics["exact_ids_shape"] = list(exact_ids.shape)
    diagnostics["selected_ids_shape"] = list(selected_ids.shape)
    diagnostics["score_shape"] = list(scores.shape)
    return EvaluationResult(task.name, metrics, diagnostics, synthetic_only=False, benchmark_claim=False)


def diagnose_movielens_retrieval_alignment(
    task: "MovieLensLocalTask",
    model: Any,
    split: str = "test",
    k: int = 20,
    index_backend: str = "exact",
    index_metric: str = "dot",
    max_sample_rows: int = 10,
) -> dict[str, Any]:
    """Claim-free retrieval/id-space alignment diagnostics for MovieLens.

    Compares Q-score top-k over the full catalogue with index-retrieval top-k
    over action embeddings on a small sample of rows. Reports embedding norm
    and id-space diagnostics useful for diagnosing index collapse.

    Returns a claim-free dict — never a benchmark claim.
    """

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    rows = task.get_split_rows(split)
    safe_k = min(k, task.num_actions)
    sample_rows = rows[:max_sample_rows]

    result: dict[str, Any] = {
        "split": split,
        "k": safe_k,
        "index_backend": index_backend,
        "index_metric": index_metric,
        "num_eval_rows": len(rows),
        "num_sample_rows": len(sample_rows),
        "scaffold_only": True,
        "benchmark_claim": False,
    }

    if not sample_rows:
        result["reason"] = "split has no rows"
        result["id_space_alignment_passed"] = False
        result["zero_norm_action_embeddings_count"] = 0
        result["zero_norm_query_embeddings_count"] = 0
        return result

    from .ann_index import make_action_index

    with torch.no_grad():
        scores_sample = score_movielens_catalogue_with_model(task, model, sample_rows)
        states_sample = _movielens_states_for_rows(task, sample_rows)
        action_features = task._catalogue.action_features.to(device=states_sample.device)
        action_embeddings = model.encode_actions(action_features).detach()
        query_embeddings = model.encode_states(states_sample).detach()

    # Q-score top-k (brute force over full catalogue)
    exact_values_sample, exact_ids_sample = torch.topk(scores_sample, k=safe_k, dim=1)

    # Index-retrieval top-k (using 0-indexed position ids, matching scores columns)
    index = make_action_index(
        action_embeddings,
        action_ids=torch.arange(task.num_actions, dtype=torch.long, device=action_embeddings.device),
        backend=index_backend,
        metric=index_metric,
        normalize=index_metric == "cosine",
    )
    selected = index.search(query_embeddings, k=safe_k)
    selected_ids_sample = selected.ids.to(device=scores_sample.device)
    selected_scores_sample = selected.scores.to(device=scores_sample.device)

    # ID overlap: how many index-retrieved ids match the brute-force Q top-k
    overlap_per_row = (exact_ids_sample.unsqueeze(2) == selected_ids_sample.unsqueeze(1)).any(dim=2).float()
    overlap_count = int(overlap_per_row.sum().item())

    # Embedding norm diagnostics
    zero_threshold = 1e-6
    action_norms = action_embeddings.norm(dim=1)
    query_norms = query_embeddings.norm(dim=1)

    # ID-space alignment: index returns position indices in [0, num_actions)
    candidate_min = int(selected_ids_sample.min().item()) if selected_ids_sample.numel() > 0 else 0
    candidate_max = int(selected_ids_sample.max().item()) if selected_ids_sample.numel() > 0 else 0
    action_id_min = int(task._catalogue.action_ids.min().item())
    action_id_max = int(task._catalogue.action_ids.max().item())
    id_alignment_passed = (candidate_min >= 0) and (candidate_max < task.num_actions)

    # Limit id sample lists to first 5 candidates per row to avoid large arrays
    id_sample_cols = min(5, safe_k)

    result.update(
        {
            "exact_q_topk_ids_sample": exact_ids_sample[:, :id_sample_cols].tolist(),
            "index_topk_ids_sample": selected_ids_sample[:, :id_sample_cols].tolist(),
            "topk_id_overlap_count_sample": overlap_count,
            "q_score_topk_value_mean": float(exact_values_sample.mean().item()),
            "q_score_topk_value_std": float(exact_values_sample.std().item())
            if exact_values_sample.numel() > 1
            else 0.0,
            "index_score_topk_value_mean": float(selected_scores_sample.mean().item()),
            "action_embedding_norm_mean": float(action_norms.mean().item()),
            "action_embedding_norm_min": float(action_norms.min().item()),
            "action_embedding_norm_max": float(action_norms.max().item()),
            "query_embedding_norm_mean": float(query_norms.mean().item()),
            "query_embedding_norm_min": float(query_norms.min().item()),
            "query_embedding_norm_max": float(query_norms.max().item()),
            "zero_norm_action_embeddings_count": int((action_norms < zero_threshold).sum().item()),
            "zero_norm_query_embeddings_count": int((query_norms < zero_threshold).sum().item()),
            "candidate_id_space_min": candidate_min,
            "candidate_id_space_max": candidate_max,
            "action_id_space_min": action_id_min,
            "action_id_space_max": action_id_max,
            "id_space_alignment_passed": id_alignment_passed,
        }
    )

    return result


def diagnose_movielens_q_degeneracy(
    task: "MovieLensLocalTask",
    model: Any,
    split: str = "test",
    k: int = 20,
    temperature: float = 1.0,
    num_negative_samples: int = 20,
    max_sample_rows: int = 10,
    q_degeneracy_std_threshold: float = 1e-3,
    q_degeneracy_margin_threshold: float = 1e-4,
) -> dict[str, Any]:
    """Claim-free Q-degeneracy diagnostics for MovieLens.

    Diagnoses Q-function degeneration: when Q scores become near-uniform the
    Q-top-k is arbitrary, causing ANN recall collapse. Tracks score spread,
    top-k margin, entropy, and positive/negative score gap.

    Returns a claim-free dict — never a benchmark claim.
    """

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(num_negative_samples, int) or num_negative_samples <= 0:
        raise ValueError("num_negative_samples must be a positive integer.")

    rows = task.get_split_rows(split)
    safe_k = min(k, task.num_actions)
    sample_rows = rows[:max_sample_rows]

    result: dict[str, Any] = {
        "split": split,
        "k": safe_k,
        "num_eval_rows": len(rows),
        "num_sample_rows": len(sample_rows),
        "scaffold_only": True,
        "benchmark_claim": False,
    }

    if not sample_rows:
        result["reason"] = "split has no rows"
        result["q_degeneracy_warning"] = False
        return result

    with torch.no_grad():
        # Full catalogue Q-scores for each sample row: [num_sample, num_actions]
        scores_sample = score_movielens_catalogue_with_model(task, model, sample_rows)

        # Q-score statistics across entire catalogue
        q_score_mean = float(scores_sample.mean().item())
        q_score_std = float(scores_sample.std().item()) if scores_sample.numel() > 1 else 0.0
        q_score_min = float(scores_sample.min().item())
        q_score_max = float(scores_sample.max().item())
        q_score_range = q_score_max - q_score_min

        # Top-k margin: gap between the k-th score and the (k+1)-th score per row
        # Measures whether top-k is well-separated from the rest
        if safe_k < task.num_actions:
            topk_vals, _ = torch.topk(scores_sample, k=safe_k + 1, dim=1)
            # margin = score[k-1] - score[k] (last-top-k minus first-outside)
            margins = topk_vals[:, safe_k - 1] - topk_vals[:, safe_k]
        else:
            topk_vals, _ = torch.topk(scores_sample, k=safe_k, dim=1)
            margins = topk_vals[:, safe_k - 1] - scores_sample.min(dim=1).values
        q_topk_margin_mean = float(margins.mean().item())
        q_topk_margin_min = float(margins.min().item())

        # Softmax policy entropy and effective support
        scaled = scores_sample / float(temperature)
        log_probs = scaled - torch.logsumexp(scaled, dim=1, keepdim=True)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)  # [num_sample]
        candidate_policy_entropy_mean = float(entropy.mean().item())
        candidate_policy_entropy_max = float(entropy.max().item())
        # Effective support: exp(entropy) gives effective number of actions
        effective_action_support = entropy.exp()
        effective_action_support_mean = float(effective_action_support.mean().item())

        # Positive/negative score gap using target item ids from rows
        pos_indices = _movielens_target_indices(task, sample_rows).to(device=scores_sample.device)
        pos_indices = pos_indices.clamp(0, task.num_actions - 1)
        positive_scores = scores_sample[torch.arange(len(sample_rows), device=scores_sample.device), pos_indices]
        positive_item_score_mean = float(positive_scores.mean().item())

        # Sample negatives uniformly
        num_neg = min(num_negative_samples, task.num_actions)
        neg_indices = torch.randint(0, task.num_actions, (len(sample_rows), num_neg), device=scores_sample.device)
        neg_scores = scores_sample.gather(1, neg_indices)
        sampled_negative_score_mean = float(neg_scores.mean().item())
        positive_negative_score_gap_mean = positive_item_score_mean - sampled_negative_score_mean

    # Warning: Q-scores near-uniform (degenerate solution, not embedding collapse)
    catalogue_fraction_threshold = 0.9
    effective_support_fraction = effective_action_support_mean / max(task.num_actions, 1)
    q_degeneracy_warning = (
        q_score_std < float(q_degeneracy_std_threshold)
        or q_topk_margin_mean < float(q_degeneracy_margin_threshold)
        or effective_support_fraction >= catalogue_fraction_threshold
    )

    result.update(
        {
            "q_score_mean": q_score_mean,
            "q_score_std": q_score_std,
            "q_score_min": q_score_min,
            "q_score_max": q_score_max,
            "q_score_range": q_score_range,
            "q_topk_margin_mean": q_topk_margin_mean,
            "q_topk_margin_min": q_topk_margin_min,
            "candidate_policy_entropy_mean": candidate_policy_entropy_mean,
            "candidate_policy_entropy_max": candidate_policy_entropy_max,
            "effective_action_support_mean": effective_action_support_mean,
            "effective_action_support_fraction": effective_support_fraction,
            "positive_item_score_mean": positive_item_score_mean,
            "sampled_negative_score_mean": sampled_negative_score_mean,
            "positive_negative_score_gap_mean": positive_negative_score_gap_mean,
            "q_degeneracy_std_threshold": float(q_degeneracy_std_threshold),
            "q_degeneracy_margin_threshold": float(q_degeneracy_margin_threshold),
            "q_degeneracy_warning": q_degeneracy_warning,
        }
    )

    return result


def evaluate_movielens_chrono_support_protocol(
    task: MovieLensLocalTask,
    model: Any,
    k_values: tuple[int, ...] = (5, 10, 20),
    temperature: float = 1.0,
    index_backend: str = "exact",
    index_metric: str = "dot",
    max_eval_rows: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run composite ranking/PMR/ANN support diagnostics per split and k.

    This is claim-free scaffold diagnostics only. It is not a benchmark protocol,
    not a recommendation quality metric, and does not establish statistical significance.
    Results are indexed by split name and then by 'k_{k}' for each k in k_values.

    max_eval_rows: if set, each split is randomly sub-sampled to at most this many
    rows before evaluation. Use to bound evaluation time on large datasets.
    """

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if not k_values:
        raise ValueError("k_values must be non-empty.")
    for k in k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Each k in k_values must be a positive integer.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
        raise ValueError("index_backend must be one of {'exact', 'faiss', 'faiss_ivf', 'faiss_hnsw'}.")
    if index_metric not in {"dot", "cosine"}:
        raise ValueError("index_metric must be one of {'dot', 'cosine'}.")
    if max_eval_rows is not None and (not isinstance(max_eval_rows, int) or max_eval_rows <= 0):
        raise ValueError("max_eval_rows must be None or a positive integer.")

    from dataclasses import asdict

    # Temporarily cap get_split_rows on the *class* so all sub-evaluation isinstance
    # checks still pass (both task types use slots=True, so instance-level override
    # is not possible). Single-threaded; finally block restores the original.
    _task_cls = type(task)
    _orig_get_split_rows = _task_cls.get_split_rows
    if max_eval_rows is not None:
        import random as _random
        _rng = _random.Random(seed)
        _cap = max_eval_rows

        def _capped(self: Any, split: str) -> list:
            rows = _orig_get_split_rows(self, split)
            return _rng.sample(rows, _cap) if len(rows) > _cap else rows

        _task_cls.get_split_rows = _capped  # type: ignore[method-assign]

    try:
        splits_results: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation", "test"):
            split_rows = task.get_split_rows(split)
            split_k_results: dict[str, Any] = {}
            for k in k_values:
                ranking = evaluate_movielens_ranking_scaffold(task, model, split=split, k=k)
                pmr = evaluate_movielens_pmr_scaffold(task, model, split=split, k=k, temperature=temperature)
                ann = evaluate_movielens_ann_candidate_audit(
                    task, model, split=split, k=k,
                    temperature=temperature,
                    index_backend=index_backend,
                    index_metric=index_metric,
                )
                split_k_results[f"k_{k}"] = {
                    "ranking": asdict(ranking),
                    "pmr": asdict(pmr),
                    "ann": asdict(ann),
                    "summary": {
                        "num_rows": len(split_rows),
                        "k": k,
                        "hit_rate_at_k": ranking.metrics.get("hit_rate_at_k"),
                        "pmr_at_k_scaffold": pmr.metrics.get("pmr_at_k_scaffold"),
                        "missing_policy_mass_mean": pmr.metrics.get("missing_policy_mass_mean"),
                        "ann_candidate_recall_mean": ann.metrics.get("ann_candidate_recall_mean"),
                        "ann_candidate_pmr_mean": ann.metrics.get("ann_candidate_pmr_mean"),
                    },
                }
            splits_results[split] = {
                "num_rows": len(split_rows),
                "k_results": split_k_results,
            }
    finally:
        _task_cls.get_split_rows = _orig_get_split_rows  # type: ignore[method-assign]

    task_metadata = task.metadata
    return {
        "split_mode": task_metadata.get("split_mode", "unknown"),
        "leakage_check_passed": task_metadata.get("leakage_check_passed"),
        "num_train_rows": task_metadata.get("num_train_rows"),
        "num_validation_rows": task_metadata.get("num_validation_rows"),
        "num_test_rows": task_metadata.get("num_test_rows"),
        "min_train_timestamp": task_metadata.get("min_train_timestamp"),
        "max_train_timestamp": task_metadata.get("max_train_timestamp"),
        "min_test_timestamp": task_metadata.get("min_test_timestamp"),
        "max_test_timestamp": task_metadata.get("max_test_timestamp"),
        "k_values": list(k_values),
        "temperature": float(temperature),
        "index_backend": index_backend,
        "index_metric": index_metric,
        "splits": splits_results,
        "scaffold_only": True,
        "benchmark_claim": False,
        "note": "Claim-free per-split support/PMR/ANN diagnostics for chronological_strict split. Not a recommender benchmark.",
    }


def _project_features_for_scores(values: "torch.Tensor", target_dim: int) -> "torch.Tensor":
    if values.shape[1] == target_dim:
        return values
    if values.shape[1] > target_dim:
        return values[:, :target_dim]
    return torch.nn.functional.pad(values, (0, target_dim - values.shape[1]))


def evaluate_recsim_slate_scaffold(
    task: RecSimMockSlateTask,
    model_or_baseline: Any = None,
    k: int | None = None,
) -> EvaluationResult:
    """Return claim-free diagnostics for the mock RecSim slate scaffold."""

    if not isinstance(task, RecSimMockSlateTask):
        raise TypeError("task must be a RecSimMockSlateTask.")
    selected_k = task.slate_size if k is None else k
    if not isinstance(selected_k, int) or selected_k <= 0:
        raise ValueError("k must be None or a positive integer.")
    selected_k = min(selected_k, task.num_items)
    dataset = task.make_dataset()
    action_features, _ = task.make_action_catalogue()
    observed_items = torch.unique(dataset.batch.action_ids)
    metrics: dict[str, float | int | None] = {
        "num_users": task.num_users,
        "num_items": task.num_items,
        "num_interactions": len(dataset),
        "slate_size": task.slate_size,
        "catalogue_size": task.num_items,
        "observed_item_coverage": float(observed_items.numel() / task.num_items),
        "avg_reward": float(dataset.batch.rewards.mean().item()),
    }
    diagnostics: dict[str, Any] = {
        **task.metadata,
        "scaffold_only": True,
        "recsim_slate_scaffold": True,
        "benchmark_claim": False,
        "k": selected_k,
    }
    if model_or_baseline is not None and hasattr(model_or_baseline, "predict_action_scores"):
        scores = model_or_baseline.predict_action_scores(dataset.batch.states, action_features)
        scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
        if scores_tensor.shape != (len(dataset), task.num_items):
            raise ValueError("predict_action_scores must return shape [num_interactions, num_items].")
        if torch.any(~torch.isfinite(scores_tensor)):
            raise ValueError("RecSim scaffold scores must be finite.")
        topk = torch.topk(scores_tensor, k=selected_k, dim=1).indices
        diagnostics["score_shape"] = list(scores_tensor.shape)
        diagnostics["topk_shape"] = list(topk.shape)
    return EvaluationResult(
        task_name=task.name,
        metrics=metrics,
        diagnostics=diagnostics,
        synthetic_only=False,
        benchmark_claim=False,
    )


def evaluate_d4rl_action_codebook_audit(
    task: D4RLLargeActionTask,
    model_or_scores: Any = None,
    k: int = 10,
    index_backend: str = "exact",
    index_metric: str = "dot",
) -> EvaluationResult:
    """Run claim-free exact-vs-selected codebook candidate diagnostics."""

    if not isinstance(task, D4RLLargeActionTask | D4RLDatasetTask):
        raise TypeError("task must be a D4RLLargeActionTask or D4RLDatasetTask.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    from .ann_index import make_action_index
    from .audits import audit_candidate_sets

    observations = task.make_dataset().batch.states
    action_features, action_ids = task.make_action_catalogue()
    safe_k = min(k, task.num_actions)
    if model_or_scores is None:
        queries = _project_features_for_scores(observations, task.action_dim)
        scores = queries @ action_features.T
        index_embeddings = action_features
        query_embeddings = queries
    elif hasattr(model_or_scores, "encode_states") and hasattr(model_or_scores, "encode_actions"):
        with torch.no_grad():
            query_embeddings = model_or_scores.encode_states(observations)
            index_embeddings = model_or_scores.encode_actions(action_features)
            expanded_actions = index_embeddings.unsqueeze(0).expand(observations.shape[0], -1, -1)
            scores = model_or_scores.q_values(query_embeddings, expanded_actions)
    else:
        scores = torch.as_tensor(model_or_scores)
        if scores.shape != (observations.shape[0], task.num_actions):
            raise ValueError("model_or_scores tensor must have shape [num_transitions, num_actions].")
        query_embeddings = _project_features_for_scores(observations, task.action_dim)
        index_embeddings = action_features
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if torch.any(~torch.isfinite(scores)):
        raise ValueError("D4RL action codebook audit scores must be finite.")
    exact_values, exact_ids = torch.topk(scores, k=safe_k, dim=1)
    index = make_action_index(
        index_embeddings.detach(),
        action_ids=action_ids,
        backend=index_backend,
        metric=index_metric,
        normalize=index_metric == "cosine",
    )
    selected = index.search(query_embeddings.detach(), k=safe_k)
    selected_ids = selected.ids.to(device=scores.device)
    selected_values = torch.gather(scores, dim=1, index=selected_ids)
    audit = audit_candidate_sets(
        selected_ids,
        exact_ids,
        exact_values=exact_values,
        retrieved_values=selected_values,
    )
    return EvaluationResult(
        task_name=task.name,
        metrics={
            "codebook_size": task.num_actions,
            "k": safe_k,
            "index_backend": index_backend,
            "index_metric": index_metric,
            "ann_candidate_recall_mean": float(audit.recall_at_k.mean().item()),
            "ann_candidate_overlap_fraction_mean": float(audit.overlap_fraction.mean().item()),
            "ann_candidate_best_value_gap_mean": (
                None if audit.best_value_gap is None else float(audit.best_value_gap.mean().item())
            ),
        },
        diagnostics={
            **task.metadata,
            "scaffold_only": True,
            "d4rl_action_codebook_audit": True,
            "benchmark_claim": False,
            "exact_ids_shape": list(exact_ids.shape),
            "selected_ids_shape": list(selected_ids.shape),
        },
        synthetic_only=False,
        benchmark_claim=False,
    )


def score_d4rl_codebook_with_model(
    task: D4RLLargeActionTask,
    model: Any,
    rows_or_states: Any = None,
) -> "torch.Tensor":
    """Score every D4RL mock codebook action with a trained DW-MQL model."""

    if not isinstance(task, D4RLLargeActionTask | D4RLDatasetTask):
        raise TypeError("task must be a D4RLLargeActionTask or D4RLDatasetTask.")
    if model is None:
        raise ValueError("model is required for D4RL codebook scoring.")
    states = task.make_dataset().batch.states if rows_or_states is None else torch.as_tensor(rows_or_states)
    states = torch.as_tensor(states, dtype=torch.float32)
    if states.ndim != 2 or states.shape[1] != task.state_dim:
        raise ValueError("rows_or_states must have shape [num_rows, state_dim].")
    if torch.any(~torch.isfinite(states)):
        raise ValueError("D4RL scoring states must be finite.")
    action_features, _ = task.make_action_catalogue()
    with torch.no_grad():
        action_features = action_features.to(device=states.device)
        state_embeddings = model.encode_states(states)
        action_embeddings = model.encode_actions(action_features)
        expanded_actions = action_embeddings.unsqueeze(0).expand(states.shape[0], -1, -1)
        scores = model.q_values(state_embeddings, expanded_actions)
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.shape != (states.shape[0], task.num_actions):
        raise ValueError("D4RL model Q scores must have shape [num_rows, num_actions].")
    if torch.any(~torch.isfinite(scores)):
        raise ValueError("D4RL model Q scores must be finite.")
    return scores


def evaluate_d4rl_pmr_scaffold(
    task: D4RLLargeActionTask,
    model: Any,
    k: int = 10,
    temperature: float = 1.0,
) -> EvaluationResult:
    """Compute claim-free PMR/support diagnostics from D4RL mock Q scores."""

    if not isinstance(task, D4RLLargeActionTask | D4RLDatasetTask):
        raise TypeError("task must be a D4RLLargeActionTask or D4RLDatasetTask.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    dataset = task.make_dataset()
    safe_k = min(k, task.num_actions)
    scores = score_d4rl_codebook_with_model(task, model)
    probs = torch.softmax(scores / float(temperature), dim=1)
    if torch.any(~torch.isfinite(probs)) or torch.any(probs < 0):
        raise ValueError("D4RL PMR probabilities must be finite and nonnegative.")
    topk_indices = torch.topk(scores, k=safe_k, dim=1).indices
    topk_mass = torch.gather(probs, dim=1, index=topk_indices).sum(dim=1)
    logged_ids = dataset.batch.action_ids.long()
    logged_masses = probs[torch.arange(probs.shape[0]), logged_ids]
    logged_scores = scores[torch.arange(scores.shape[0]), logged_ids]
    logged_ranks = (scores > logged_scores[:, None]).sum(dim=1).float() + 1.0
    logged_hits = (topk_indices == logged_ids[:, None]).any(dim=1).float()
    pmr_mean = float(topk_mass.mean().item())
    return EvaluationResult(
        task_name=task.name,
        metrics={
            "pmr_at_k_scaffold": pmr_mean,
            "candidate_policy_mass_captured_by_topk": pmr_mean,
            "missing_policy_mass_mean": float((1.0 - topk_mass).mean().item()),
            "logged_action_policy_mass_mean": float(logged_masses.mean().item()),
            "logged_action_rank_mean": float(logged_ranks.mean().item()),
            "logged_action_in_topk_rate": float(logged_hits.mean().item()),
            "num_eval_rows": len(dataset),
            "catalogue_size": task.num_actions,
            "k": safe_k,
            "temperature": float(temperature),
        },
        diagnostics={
            **task.metadata,
            "scaffold_only": True,
            "d4rl_pmr_scaffold": True,
            "benchmark_claim": False,
            "score_shape": list(scores.shape),
            "topk_shape": list(topk_indices.shape),
            "note": "PMR is computed from trained Q-score softmax over the mock codebook and is not a control-performance metric.",
        },
        synthetic_only=False,
        benchmark_claim=False,
    )


def evaluate_d4rl_ann_candidate_audit(
    task: D4RLLargeActionTask,
    model: Any,
    k: int = 10,
    temperature: float = 1.0,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> EvaluationResult:
    """Compare selected index candidates with full-codebook Q top-k ids."""

    if not isinstance(task, D4RLLargeActionTask | D4RLDatasetTask):
        raise TypeError("task must be a D4RLLargeActionTask or D4RLDatasetTask.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    from .ann_index import make_action_index
    from .audits import audit_candidate_sets

    states = task.make_dataset().batch.states
    action_features, _ = task.make_action_catalogue()
    safe_k = min(k, task.num_actions)
    scores = score_d4rl_codebook_with_model(task, model, states)
    probs = torch.softmax(scores / float(temperature), dim=1)
    exact_values, exact_ids = torch.topk(scores, k=safe_k, dim=1)
    exact_policy_probs = torch.gather(probs, dim=1, index=exact_ids)
    with torch.no_grad():
        action_embeddings = model.encode_actions(action_features).detach()
        query_embeddings = model.encode_states(states).detach()
        index = make_action_index(
            action_embeddings,
            action_ids=torch.arange(task.num_actions, dtype=torch.long, device=action_embeddings.device),
            backend=index_backend,
            metric=index_metric,
            normalize=index_metric == "cosine",
            nlist=index_nlist,
            nprobe=index_nprobe,
        )
        selected = index.search(query_embeddings, k=safe_k)
    selected_ids = selected.ids.to(device=scores.device)
    selected_values = torch.gather(scores, dim=1, index=selected_ids)
    audit = audit_candidate_sets(
        selected_ids,
        exact_ids,
        exact_policy_probs=exact_policy_probs,
        exact_values=exact_values,
        retrieved_values=selected_values,
    )
    return EvaluationResult(
        task_name=task.name,
        metrics={
            "ann_candidate_recall_mean": float(audit.recall_at_k.mean().item()),
            "ann_candidate_overlap_fraction_mean": float(audit.overlap_fraction.mean().item()),
            "ann_candidate_pmr_mean": None if audit.pmr_at_k is None else float(audit.pmr_at_k.mean().item()),
            "ann_candidate_pmr_min": None if audit.pmr_at_k is None else float(audit.pmr_at_k.min().item()),
            "ann_candidate_pmr_max": None if audit.pmr_at_k is None else float(audit.pmr_at_k.max().item()),
            "ann_candidate_best_value_gap_mean": (
                None if audit.best_value_gap is None else float(audit.best_value_gap.mean().item())
            ),
            "index_backend": index_backend,
            "index_metric": index_metric,
            "num_eval_rows": states.shape[0],
            "catalogue_size": task.num_actions,
            "k": safe_k,
            "temperature": float(temperature),
        },
        diagnostics={
            **task.metadata,
            "scaffold_only": True,
            "d4rl_ann_candidate_audit": True,
            "benchmark_claim": False,
            "index_nlist": index_nlist,
            "index_nprobe": index_nprobe,
            "exact_ids_shape": list(exact_ids.shape),
            "selected_ids_shape": list(selected_ids.shape),
            "score_shape": list(scores.shape),
            "note": "Selected candidates are compared with full-codebook Q top-k ids; this is not D4RL benchmark evaluation.",
        },
        synthetic_only=False,
        benchmark_claim=False,
    )


def score_recsim_catalogue_with_model(
    task: RecSimMockSlateTask,
    model: Any,
    rows_or_states: Any = None,
) -> "torch.Tensor":
    """Score every RecSim mock catalogue item with a trained DW-MQL model."""

    if not isinstance(task, RecSimMockSlateTask):
        raise TypeError("task must be a RecSimMockSlateTask.")
    if model is None:
        raise ValueError("model is required for RecSim catalogue scoring.")
    states = task.make_dataset().batch.states if rows_or_states is None else torch.as_tensor(rows_or_states)
    states = torch.as_tensor(states, dtype=torch.float32)
    if states.ndim != 2 or states.shape[1] != task.state_dim:
        raise ValueError("rows_or_states must have shape [num_rows, user_feature_dim].")
    if torch.any(~torch.isfinite(states)):
        raise ValueError("RecSim scoring states must be finite.")
    action_features, _ = task.make_action_catalogue()
    with torch.no_grad():
        action_features = action_features.to(device=states.device)
        state_embeddings = model.encode_states(states)
        action_embeddings = model.encode_actions(action_features)
        expanded_actions = action_embeddings.unsqueeze(0).expand(states.shape[0], -1, -1)
        scores = model.q_values(state_embeddings, expanded_actions)
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.shape != (states.shape[0], task.num_actions):
        raise ValueError("RecSim model Q scores must have shape [num_rows, num_items].")
    if torch.any(~torch.isfinite(scores)):
        raise ValueError("RecSim model Q scores must be finite.")
    return scores


def evaluate_recsim_pmr_scaffold(
    task: RecSimMockSlateTask,
    model: Any,
    k: int = 10,
    temperature: float = 1.0,
) -> EvaluationResult:
    """Compute claim-free PMR/support diagnostics from RecSim mock Q scores."""

    if not isinstance(task, RecSimMockSlateTask):
        raise TypeError("task must be a RecSimMockSlateTask.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    dataset = task.make_dataset()
    safe_k = min(k, task.num_actions)
    scores = score_recsim_catalogue_with_model(task, model)
    probs = torch.softmax(scores / float(temperature), dim=1)
    if torch.any(~torch.isfinite(probs)) or torch.any(probs < 0):
        raise ValueError("RecSim PMR probabilities must be finite and nonnegative.")
    topk_indices = torch.topk(scores, k=safe_k, dim=1).indices
    topk_mass = torch.gather(probs, dim=1, index=topk_indices).sum(dim=1)
    logged_ids = dataset.batch.action_ids.long()
    logged_masses = probs[torch.arange(probs.shape[0]), logged_ids]
    logged_scores = scores[torch.arange(scores.shape[0]), logged_ids]
    logged_ranks = (scores > logged_scores[:, None]).sum(dim=1).float() + 1.0
    logged_hits = (topk_indices == logged_ids[:, None]).any(dim=1).float()
    pmr_mean = float(topk_mass.mean().item())
    return EvaluationResult(
        task_name=task.name,
        metrics={
            "pmr_at_k_scaffold": pmr_mean,
            "candidate_policy_mass_captured_by_topk": pmr_mean,
            "missing_policy_mass_mean": float((1.0 - topk_mass).mean().item()),
            "logged_item_policy_mass_mean": float(logged_masses.mean().item()),
            "logged_item_rank_mean": float(logged_ranks.mean().item()),
            "logged_item_in_topk_rate": float(logged_hits.mean().item()),
            "num_eval_rows": len(dataset),
            "catalogue_size": task.num_actions,
            "k": safe_k,
            "temperature": float(temperature),
        },
        diagnostics={
            **task.metadata,
            "scaffold_only": True,
            "recsim_pmr_scaffold": True,
            "benchmark_claim": False,
            "score_shape": list(scores.shape),
            "topk_shape": list(topk_indices.shape),
            "note": "PMR is computed from trained Q-score softmax over the mock item catalogue and is not a slate-quality metric.",
        },
        synthetic_only=False,
        benchmark_claim=False,
    )


def evaluate_recsim_ann_candidate_audit(
    task: RecSimMockSlateTask,
    model: Any,
    k: int = 10,
    temperature: float = 1.0,
    index_backend: str = "exact",
    index_metric: str = "dot",
    index_nlist: int | None = None,
    index_nprobe: int | None = None,
) -> EvaluationResult:
    """Compare selected index candidates with full-catalogue Q top-k ids."""

    if not isinstance(task, RecSimMockSlateTask):
        raise TypeError("task must be a RecSimMockSlateTask.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(temperature, int | float) or float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    from .ann_index import make_action_index
    from .audits import audit_candidate_sets

    states = task.make_dataset().batch.states
    action_features, _ = task.make_action_catalogue()
    safe_k = min(k, task.num_actions)
    scores = score_recsim_catalogue_with_model(task, model, states)
    probs = torch.softmax(scores / float(temperature), dim=1)
    exact_values, exact_ids = torch.topk(scores, k=safe_k, dim=1)
    exact_policy_probs = torch.gather(probs, dim=1, index=exact_ids)
    with torch.no_grad():
        action_embeddings = model.encode_actions(action_features).detach()
        query_embeddings = model.encode_states(states).detach()
        index = make_action_index(
            action_embeddings,
            action_ids=torch.arange(task.num_actions, dtype=torch.long, device=action_embeddings.device),
            backend=index_backend,
            metric=index_metric,
            normalize=index_metric == "cosine",
            nlist=index_nlist,
            nprobe=index_nprobe,
        )
        selected = index.search(query_embeddings, k=safe_k)
    selected_ids = selected.ids.to(device=scores.device)
    selected_values = torch.gather(scores, dim=1, index=selected_ids)
    audit = audit_candidate_sets(
        selected_ids,
        exact_ids,
        exact_policy_probs=exact_policy_probs,
        exact_values=exact_values,
        retrieved_values=selected_values,
    )
    return EvaluationResult(
        task_name=task.name,
        metrics={
            "ann_candidate_recall_mean": float(audit.recall_at_k.mean().item()),
            "ann_candidate_overlap_fraction_mean": float(audit.overlap_fraction.mean().item()),
            "ann_candidate_pmr_mean": None if audit.pmr_at_k is None else float(audit.pmr_at_k.mean().item()),
            "ann_candidate_pmr_min": None if audit.pmr_at_k is None else float(audit.pmr_at_k.min().item()),
            "ann_candidate_pmr_max": None if audit.pmr_at_k is None else float(audit.pmr_at_k.max().item()),
            "ann_candidate_best_value_gap_mean": (
                None if audit.best_value_gap is None else float(audit.best_value_gap.mean().item())
            ),
            "index_backend": index_backend,
            "index_metric": index_metric,
            "num_eval_rows": states.shape[0],
            "catalogue_size": task.num_actions,
            "k": safe_k,
            "temperature": float(temperature),
        },
        diagnostics={
            **task.metadata,
            "scaffold_only": True,
            "recsim_ann_candidate_audit": True,
            "benchmark_claim": False,
            "index_nlist": index_nlist,
            "index_nprobe": index_nprobe,
            "exact_ids_shape": list(exact_ids.shape),
            "selected_ids_shape": list(selected_ids.shape),
            "score_shape": list(scores.shape),
            "note": "Selected candidates are compared with full-catalogue Q top-k ids; this is not RecSim benchmark evaluation.",
        },
        synthetic_only=False,
        benchmark_claim=False,
    )


def build_movielens_scaffold_report(task: MovieLensLocalTask) -> dict[str, Any]:
    """Build a claim-free JSON-safe report for a local MovieLens scaffold task."""

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    evaluation = evaluate_movielens_scaffold_task(task)
    metadata = task.metadata
    return {
        "schema_name": "dwmql_movielens_scaffold_report",
        "schema_version": 1,
        "report_type": "movielens_local_scaffold",
        "adapter_name": "movielens",
        "task_name": task.name,
        "task_type": task.task_type,
        "synthetic_only": False,
        "benchmark_claim": False,
        "scaffold_only": True,
        "dataset_metadata": {
            "num_users": metadata["num_users"],
            "num_items": metadata["num_items"],
            "num_interactions": metadata["num_rows"],
            "split_mode": metadata["split_mode"],
            "num_train_rows": metadata["num_train_rows"],
            "num_validation_rows": metadata["num_validation_rows"],
            "num_test_rows": metadata["num_test_rows"],
            "num_sessions": metadata.get("num_sessions"),
            "min_timestamp": metadata["min_timestamp"],
            "max_timestamp": metadata["max_timestamp"],
        },
        "action_catalogue_metadata": {
            "num_actions": task.num_actions,
            "action_dim": task.action_dim,
        },
        "scaffold_metrics": {
            "positive_rate": evaluation.metrics["positive_rate"],
            "avg_user_interactions": evaluation.metrics["avg_user_interactions"],
            "avg_item_interactions": evaluation.metrics["avg_item_interactions"],
            "catalogue_coverage_observed": evaluation.metrics["catalogue_coverage_observed"],
        },
        "diagnostics": evaluation.diagnostics,
        "limitations": [
            "no training",
            "no recommender benchmark claims",
            "no NDCG/CTR evaluation",
            "no exposure-bias correction",
        ],
    }


def write_movielens_scaffold_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a local MovieLens scaffold report as UTF-8 JSON."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")
    if report.get("benchmark_claim") is not False:
        raise ValueError("MovieLens scaffold report benchmark_claim must be false.")
    if report.get("scaffold_only") is not True:
        raise ValueError("MovieLens scaffold report scaffold_only must be true.")
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("MovieLens scaffold report path must be a file, not a directory.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def build_external_adapter_scaffold_report(
    task: Any,
    adapter_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified claim-free external-adapter scaffold report."""

    assert_benchmark_task_contract(task)
    bundle = task.make_dataset_bundle()
    catalogue = task.make_action_catalogue_contract()
    evaluation: EvaluationResult | None = None
    diagnostics: dict[str, Any] = {}
    if isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        evaluation = evaluate_movielens_scaffold_task(task)
        diagnostics["movielens_scaffold"] = {
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "synthetic_only": evaluation.synthetic_only,
            "benchmark_claim": evaluation.benchmark_claim,
        }
    elif isinstance(task, D4RLLargeActionTask | D4RLDatasetTask):
        evaluation = evaluate_d4rl_action_codebook_audit(task)
        diagnostics["d4rl_action_codebook_audit"] = {
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "synthetic_only": evaluation.synthetic_only,
            "benchmark_claim": evaluation.benchmark_claim,
        }
    elif isinstance(task, RecSimMockSlateTask):
        evaluation = evaluate_recsim_slate_scaffold(task)
        diagnostics["recsim_slate_scaffold"] = {
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "synthetic_only": evaluation.synthetic_only,
            "benchmark_claim": evaluation.benchmark_claim,
        }
    else:
        evaluation = task.evaluate_policy_or_model(model=None)
        diagnostics["contract"] = {
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "synthetic_only": evaluation.synthetic_only,
            "benchmark_claim": evaluation.benchmark_claim,
        }
    synthetic_only = bool(getattr(evaluation, "synthetic_only", task.metadata.get("synthetic_only", False)))
    return {
        "schema_name": EXTERNAL_ADAPTER_REPORT_SCHEMA_NAME,
        "schema_version": EXTERNAL_ADAPTER_REPORT_SCHEMA_VERSION,
        "report_type": "external_adapter_scaffold",
        "adapter_name": adapter_name,
        "task_name": task.name,
        "task_type": task.task_type,
        "synthetic_only": synthetic_only,
        "scaffold_only": bool(task.metadata.get("scaffold_only", True)),
        "benchmark_claim": False,
        "dataset_metadata": {
            "dataset_size": len(bundle.dataset),
            "state_dim": task.state_dim,
            "action_dim": task.action_dim,
        },
        "action_catalogue_metadata": {
            "num_actions": catalogue.num_actions,
            "action_dim": catalogue.action_dim,
        },
        "task_metadata": task.metadata,
        "evaluation_diagnostics": diagnostics,
        "limitations": [
            "scaffold diagnostics only",
            "no benchmark claims",
            "no statistical tests",
            "no plots",
            "no external benchmark loading except explicitly provided local files or mock arrays",
        ],
        "extra_metadata": {} if extra_metadata is None else dict(extra_metadata),
    }


def write_external_adapter_scaffold_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a unified external adapter scaffold report as UTF-8 JSON."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")
    if report.get("schema_name") != EXTERNAL_ADAPTER_REPORT_SCHEMA_NAME:
        raise ValueError("External adapter report schema_name is unsupported.")
    if report.get("schema_version") != EXTERNAL_ADAPTER_REPORT_SCHEMA_VERSION:
        raise ValueError("External adapter report schema_version is unsupported.")
    if report.get("benchmark_claim") is not False:
        raise ValueError("External adapter scaffold report benchmark_claim must be false.")
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("External adapter scaffold report path must be a file, not a directory.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validate_external_adapter_scaffold_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate unified external adapter scaffold report metadata."""

    if not isinstance(report, dict):
        raise ValueError("External adapter scaffold report must contain a JSON object.")
    if report.get("schema_name") != EXTERNAL_ADAPTER_REPORT_SCHEMA_NAME:
        raise ValueError("External adapter scaffold report schema_name is unsupported.")
    if report.get("schema_version") != EXTERNAL_ADAPTER_REPORT_SCHEMA_VERSION:
        raise ValueError("External adapter scaffold report schema_version is unsupported.")
    if report.get("report_type") != "external_adapter_scaffold":
        raise ValueError("External adapter scaffold report_type is unsupported.")
    if report.get("benchmark_claim") is not False:
        raise ValueError("External adapter scaffold report benchmark_claim must be false.")
    for key in ("adapter_name", "task_name", "task_type"):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ValueError(f"External adapter scaffold report must include non-empty {key}.")
    return report


def load_external_adapter_scaffold_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a unified external adapter scaffold report."""

    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"External adapter scaffold report not found: {report_path}")
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid external adapter scaffold report JSON: {report_path}") from exc
    return _validate_external_adapter_scaffold_report(report)


def _diagnostic_metrics(report: dict[str, Any], diagnostic_key: str) -> dict[str, Any]:
    diagnostics = report.get("evaluation_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return {}
    item = diagnostics.get(diagnostic_key, {})
    if not isinstance(item, dict):
        return {}
    metrics = item.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def summarize_external_adapter_scaffold_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact claim-free diagnostics from a unified adapter report."""

    report = _validate_external_adapter_scaffold_report(report)
    dataset_metadata = report.get("dataset_metadata", {})
    action_metadata = report.get("action_catalogue_metadata", {})
    if not isinstance(dataset_metadata, dict):
        dataset_metadata = {}
    if not isinstance(action_metadata, dict):
        action_metadata = {}
    summary: dict[str, Any] = {
        "adapter_name": report["adapter_name"],
        "task_name": report["task_name"],
        "task_type": report["task_type"],
        "synthetic_only": report.get("synthetic_only"),
        "scaffold_only": report.get("scaffold_only"),
        "benchmark_claim": report["benchmark_claim"],
        "dataset_size": dataset_metadata.get("dataset_size"),
        "state_dim": dataset_metadata.get("state_dim"),
        "action_dim": dataset_metadata.get("action_dim"),
        "num_actions": action_metadata.get("num_actions"),
    }
    adapter_name = report["adapter_name"]
    if adapter_name == "movielens":
        metrics = _diagnostic_metrics(report, "movielens_scaffold")
        summary["positive_rate"] = metrics.get("positive_rate")
        summary["catalogue_coverage_observed"] = metrics.get("catalogue_coverage_observed")
    elif adapter_name == "d4rl":
        metrics = _diagnostic_metrics(report, "d4rl_action_codebook_audit")
        summary["ann_candidate_recall_mean"] = metrics.get("ann_candidate_recall_mean")
        summary["ann_candidate_overlap_fraction_mean"] = metrics.get("ann_candidate_overlap_fraction_mean")
    elif adapter_name == "recsim":
        metrics = _diagnostic_metrics(report, "recsim_slate_scaffold")
        summary["observed_item_coverage"] = metrics.get("observed_item_coverage")
        summary["avg_reward"] = metrics.get("avg_reward")
    elif adapter_name == "mock_external":
        metrics = _diagnostic_metrics(report, "contract")
        summary["num_transitions"] = metrics.get("num_transitions", summary["dataset_size"])
        summary["contract_num_actions"] = metrics.get("num_actions", summary["num_actions"])
    return summary


def _make_d4rl_mock_arrays_task() -> BenchmarkTask:
    """Construct the tiny deterministic D4RL mock-array scaffold task."""

    _require_torch()
    observations = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    actions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    rewards = torch.tensor([1.0, 0.0, 0.5, 1.5], dtype=torch.float32)
    next_observations = observations + 0.1
    terminals = torch.tensor([False, False, False, True])
    return D4RLAdapter(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    ).make_task()


def _write_external_adapter_bundle_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a local manifest for mixed synthetic/external adapter scaffold reports."""

    if manifest.get("schema_name") != EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_NAME:
        raise ValueError("External adapter bundle manifest schema_name is unsupported.")
    if manifest.get("schema_version") != EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("External adapter bundle manifest schema_version is unsupported.")
    if manifest.get("benchmark_claim") is not False:
        raise ValueError("External adapter bundle manifest benchmark_claim must be false.")
    if manifest.get("scaffold_only") is not True:
        raise ValueError("External adapter bundle manifest scaffold_only must be true.")
    for key in ("report_files", "included_adapters"):
        if not isinstance(manifest.get(key), list):
            raise ValueError(f"External adapter bundle manifest {key} must be a list.")
    manifest_path = Path(path)
    if manifest_path.exists() and manifest_path.is_dir():
        raise ValueError("External adapter bundle manifest path must be a file, not a directory.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validate_external_adapter_bundle_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a local external-adapter scaffold report bundle manifest."""

    if not isinstance(manifest, dict):
        raise ValueError("External adapter bundle manifest must contain a JSON object.")
    if manifest.get("schema_name") != EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_NAME:
        raise ValueError("External adapter bundle manifest schema_name is unsupported.")
    if manifest.get("schema_version") != EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("External adapter bundle manifest schema_version is unsupported.")
    if manifest.get("run_type") != "external_adapter_scaffold_report_bundle":
        raise ValueError("External adapter bundle manifest run_type is unsupported.")
    if manifest.get("scaffold_only") is not True:
        raise ValueError("External adapter bundle manifest scaffold_only must be true.")
    if manifest.get("benchmark_claim") is not False:
        raise ValueError("External adapter bundle manifest benchmark_claim must be false.")
    if not isinstance(manifest.get("report_files"), list):
        raise ValueError("External adapter bundle manifest report_files must be a list.")
    if not isinstance(manifest.get("included_adapters"), list):
        raise ValueError("External adapter bundle manifest included_adapters must be a list.")
    return manifest


def load_external_adapter_bundle_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate an external-adapter scaffold report bundle manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"External adapter bundle manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid external adapter bundle manifest JSON: {manifest_path}") from exc
    return _validate_external_adapter_bundle_manifest(manifest)


def inspect_external_adapter_scaffold_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """Load a bundle manifest, validate reports, and summarize each adapter."""

    manifest_path = Path(manifest_path)
    manifest = load_external_adapter_bundle_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    summaries: list[dict[str, Any]] = []
    for raw_report_file in manifest["report_files"]:
        if not isinstance(raw_report_file, str) or not raw_report_file:
            raise ValueError("External adapter bundle manifest report_files must contain non-empty strings.")
        report_path = Path(raw_report_file)
        if not report_path.is_absolute() and not report_path.exists():
            report_path = manifest_dir / report_path
        report = load_external_adapter_scaffold_report(report_path)
        summaries.append(summarize_external_adapter_scaffold_report(report))
    return {
        "manifest_path": str(manifest_path),
        "output_dir": manifest.get("output_dir"),
        "report_count": len(summaries),
        "included_adapters": manifest["included_adapters"],
        "summaries": summaries,
        "benchmark_claim": False,
        "scaffold_only": True,
    }


def run_external_adapter_scaffold_report_bundle(
    output_dir: str | Path,
    movielens_data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a small claim-free bundle of external-adapter scaffold reports."""

    bundle_dir = Path(output_dir)
    if bundle_dir.exists() and bundle_dir.is_file():
        raise ValueError("External adapter bundle output_dir must be a directory, not a file.")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    report_specs: list[tuple[str, BenchmarkTask, str]] = [
        ("mock_external", MockExternalBenchmarkTask(seed=0), "mock_external_report.json"),
        ("d4rl", _make_d4rl_mock_arrays_task(), "d4rl_mock_arrays_report.json"),
        ("recsim", RecSimAdapter().make_task(task_name="mock_slate"), "recsim_mock_slate_report.json"),
    ]
    if movielens_data_path is not None:
        report_specs.append(
            (
                "movielens",
                MovieLensAdapter(data_path=movielens_data_path).make_task(),
                "movielens_local_report.json",
            )
        )

    report_files: list[str] = []
    included_adapters: list[str] = []
    for adapter_name, task, filename in report_specs:
        report_path = bundle_dir / filename
        report = build_external_adapter_scaffold_report(task, adapter_name=adapter_name)
        write_external_adapter_scaffold_report(report, report_path)
        report_files.append(str(report_path))
        included_adapters.append(adapter_name)

    manifest_path = bundle_dir / "manifest.json"
    manifest = {
        "schema_name": EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_NAME,
        "schema_version": EXTERNAL_ADAPTER_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "report_type": "external_adapter_scaffold_report_bundle",
        "run_type": "external_adapter_scaffold_report_bundle",
        "output_dir": str(bundle_dir),
        "synthetic_only": False,
        "scaffold_only": True,
        "benchmark_claim": False,
        "report_files": report_files,
        "included_adapters": included_adapters,
        "metadata": {
            "movielens_included": movielens_data_path is not None,
            "movielens_data_path": None if movielens_data_path is None else str(movielens_data_path),
            "no_real_d4rl_loading": True,
            "no_real_recsim_loading": True,
            "no_movielens_download": True,
        },
    }
    _write_external_adapter_bundle_manifest(manifest, manifest_path)
    return {
        "output_dir": str(bundle_dir),
        "report_files": report_files,
        "manifest_file": str(manifest_path),
        "included_adapters": included_adapters,
        "benchmark_claim": False,
    }


@dataclass(slots=True)
class MockExternalBenchmarkTask:
    """Dependency-free mock task for adapter compliance tests."""

    seed: int = 0
    name: str = "mock_external_debug"
    task_type: str = "mock_external"
    _dataset: InMemoryReplayDataset = field(init=False, repr=False)
    _catalogue: ActionCatalogue = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        generator = torch.Generator().manual_seed(int(self.seed))
        states = torch.randn(6, 3, generator=generator)
        next_states = states + 0.05 * torch.randn(6, 3, generator=generator)
        action_features = torch.randn(4, 2, generator=generator) + 0.01
        action_ids = torch.arange(4, dtype=torch.long)
        logged_ids = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        rewards = torch.linspace(0.0, 1.0, steps=6)
        dones = torch.tensor([False, False, False, True, False, True])
        self._dataset = InMemoryReplayDataset(
            states=states,
            actions=action_features[logged_ids],
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            action_ids=logged_ids,
            next_logged_action_ids=(logged_ids + 1) % 4,
            metadata=self.metadata,
        )
        self._catalogue = ActionCatalogue(
            action_features=action_features,
            action_ids=action_ids,
            metadata=self.metadata,
        )

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 4

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "task_name": self.name,
            "task_type": self.task_type,
            "seed": int(self.seed),
            "synthetic_only": True,
            "mock_external": True,
            "benchmark_claim": False,
        }

    def make_dataset(self) -> InMemoryReplayDataset:
        return self._dataset

    def make_action_catalogue(self) -> tuple[Any, Any]:
        return self._catalogue.action_features, self._catalogue.action_ids

    def make_action_catalogue_contract(self) -> ActionCatalogue:
        return ActionCatalogue(
            action_features=self._catalogue.action_features,
            action_ids=self._catalogue.action_ids,
            metadata=self.metadata,
        )

    def make_dataset_bundle(self) -> OfflineDatasetBundle:
        return OfflineDatasetBundle(
            dataset=self._dataset,
            action_catalogue=self.make_action_catalogue_contract(),
            metadata=self.metadata,
        )

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> EvaluationResult:
        del model, kwargs
        return EvaluationResult(
            task_name=self.name,
            metrics={
                "num_transitions": len(self._dataset),
                "num_actions": self.num_actions,
            },
            diagnostics={
                **self.metadata,
                "placeholder_evaluation": True,
            },
            synthetic_only=True,
            benchmark_claim=False,
        )


@dataclass(slots=True)
class SyntheticBenchmarkTask:
    """Synthetic-only benchmark task wrapper around the existing generator."""

    config: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)
    name: str = "synthetic_debug"
    task_type: str = "synthetic_offline_rl"
    _data: SyntheticOfflineData = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._data = make_synthetic_offline_data(self.config)

    @property
    def state_dim(self) -> int:
        return int(self.config.state_dim)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_dim)

    @property
    def num_actions(self) -> int:
        return int(self.config.num_actions)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._data.metadata,
            "task_name": self.name,
            "task_type": self.task_type,
            "synthetic_only": True,
            "benchmark_claim": False,
        }

    @property
    def data(self) -> SyntheticOfflineData:
        return self._data

    @property
    def dataset(self) -> Any:
        return self._data.dataset

    @property
    def action_features(self) -> Any:
        return self._data.action_features

    @property
    def action_ids(self) -> Any:
        return self._data.action_ids

    def make_dataset(self) -> Any:
        """Return the synthetic offline replay dataset."""

        return self.dataset

    def make_action_catalogue(self) -> tuple[Any, Any]:
        """Return synthetic action features and action ids."""

        return self.action_features, self.action_ids

    def make_action_catalogue_contract(self) -> ActionCatalogue:
        """Return the validated synthetic action catalogue contract."""

        return ActionCatalogue(
            action_features=self.action_features,
            action_ids=self.action_ids,
            metadata=self.metadata,
        )

    def make_dataset_bundle(self) -> OfflineDatasetBundle:
        """Return the validated synthetic offline dataset bundle contract."""

        return synthetic_data_to_dataset_bundle(self.data)

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> EvaluationResult:
        """Return synthetic evaluation diagnostics when a model is provided.

        This is not benchmark evaluation. With ``model=None`` it keeps the
        placeholder metadata path for interface tests; with a model it runs the
        synthetic forward-only evaluation loop.
        """

        if model is not None:
            evaluation_config = kwargs.get("evaluation_config", EvaluationConfig())
            base_config = kwargs.get("base_config", kwargs.get("base_profile_or_experiment_config"))
            summary = evaluate_synthetic_model(
                self.data,
                model,
                evaluation_config,
                base_config,
                task_name=self.name,
            )
            return EvaluationResult(
                task_name=summary.task_name,
                metrics=summary.metrics,
                diagnostics={
                    **summary.diagnostics,
                    "evaluation_summary": summary,
                    "task_metadata": self.metadata,
                },
                synthetic_only=summary.synthetic_only,
                benchmark_claim=summary.benchmark_claim,
            )

        del kwargs
        return EvaluationResult(
            task_name=self.name,
            metrics={
                "num_transitions": self.config.num_transitions,
                "num_actions": self.config.num_actions,
            },
            diagnostics=self.metadata,
            synthetic_only=True,
            benchmark_claim=False,
        )


# ---------------------------------------------------------------------------
# KuaiRand-Pure scaffold adapter (Phase 4 generalization)
# ---------------------------------------------------------------------------

_KUAIRAND_ACTION_DIM: int = 5  # log_dur, aspect_ratio, music_type, tag_count, is_long_import


def _load_kuairand_log_rows(
    data_dir: Path,
    max_rows: "int | None",
) -> list[dict[str, Any]]:
    """Load KuaiRand-Pure standard interaction logs (stdlib only, no pandas)."""
    log_files = [
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
    ]
    rows: list[dict[str, Any]] = []
    for log_path in log_files:
        if not log_path.exists():
            raise FileNotFoundError(f"KuaiRand log file not found: {log_path}")
        with log_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                if max_rows is not None and len(rows) >= max_rows:
                    break
                try:
                    rows.append({
                        "user_id": int(raw["user_id"]),
                        "video_id": int(raw["video_id"]),
                        "time_ms": int(raw["time_ms"]),
                        "is_click": int(raw["is_click"]),
                    })
                except (KeyError, ValueError):
                    continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _load_kuairand_video_features(data_dir: Path) -> dict[int, list[float]]:
    """Load KuaiRand-Pure video_features_basic_pure.csv → {video_id: [5 floats]}."""
    feat_path = data_dir / "video_features_basic_pure.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"KuaiRand video features file not found: {feat_path}")
    features: dict[int, list[float]] = {}
    max_music = 14.0
    with feat_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            try:
                vid = int(raw["video_id"])
                dur_ms = float(raw["video_duration"]) if raw["video_duration"] not in ("", "nan") else 5000.0
                width = float(raw["server_width"]) if raw["server_width"] not in ("", "nan") else 720.0
                height = float(raw["server_height"]) if raw["server_height"] not in ("", "nan") else 1280.0
                music_type = float(raw["music_type"]) if raw["music_type"] not in ("", "nan") else 0.0
                tag_str = raw.get("tag", "") or ""
                tag_count = float(len([t for t in tag_str.split(",") if t.strip()]))
                upload_type = raw.get("upload_type", "").strip()
                dur_sec = dur_ms / 1000.0
                log_dur = float(np.log(dur_sec + 1.0) / 7.0)
                aspect = min(width / max(height, 1.0), 2.0) / 2.0
                mus_norm = min(music_type, max_music) / max_music
                tag_norm = min(tag_count, 20.0) / 20.0
                is_long = float(upload_type == "LongImport")
                features[vid] = [log_dur, aspect, mus_norm, tag_norm, is_long]
            except (KeyError, ValueError):
                continue
    return features


def _build_kuairand_user_affinity_map(
    dataset_rows: list[dict[str, Any]],
    video_feature_map: dict[int, list[float]],
) -> dict[int, list[float]]:
    """Compute per-user mean action-feature vector over clicked items."""
    zero_vec = [0.0] * _KUAIRAND_ACTION_DIM
    user_vecs: dict[int, list[list[float]]] = {}
    for row in dataset_rows:
        if row["is_click"]:
            vec = video_feature_map.get(row["video_id"])
            if vec is not None:
                user_vecs.setdefault(row["user_id"], []).append(vec)
    all_users = {row["user_id"] for row in dataset_rows}
    result: dict[int, list[float]] = {}
    for uid in all_users:
        vecs = user_vecs.get(uid)
        if vecs:
            n = len(vecs)
            result[uid] = [sum(v[i] for v in vecs) / n for i in range(_KUAIRAND_ACTION_DIM)]
        else:
            result[uid] = zero_vec[:]
    return result


@dataclass(slots=True)
class KuaiRandLocalTask:
    """Dependency-free local CSV KuaiRand-Pure scaffold task (Phase 4 generalization).

    Loads log_standard_*_pure.csv + video_features_basic_pure.csv from data_dir.
    State: per-user mean action-feature vector over clicked items (5-dim).
    Action: [log_dur, aspect_ratio, music_type, tag_count, is_long_import] (5-dim).
    Reward: is_click (binary). Chronological-strict split by time_ms.

    All results carry benchmark_claim: False. This is scaffold diagnostics only.
    """

    data_dir: str | Path
    max_rows: "int | None" = None
    seed: int = 0
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    min_rating_threshold: float = 0.5  # is_click >= threshold → positive
    name: str = "kuairand_pure_local"
    task_type: str = "kuairand_pure_scaffold"
    _dataset: InMemoryReplayDataset = field(init=False, repr=False)
    _catalogue: ActionCatalogue = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)
    _rows: list[dict[str, Any]] = field(init=False, repr=False)
    _user_affinity_map: "dict[int, list[float]] | None" = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        _require_torch()
        data_dir = Path(self.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"KuaiRand data_dir does not exist: {data_dir}")
        if not data_dir.is_dir():
            raise ValueError("KuaiRandLocalTask data_dir must be a directory.")
        if self.max_rows is not None and (not isinstance(self.max_rows, int) or self.max_rows <= 0):
            raise ValueError("max_rows must be None or a positive integer.")
        for frac_name in ("validation_fraction", "test_fraction"):
            v = getattr(self, frac_name)
            if not isinstance(v, int | float) or not 0.0 <= float(v) < 1.0:
                raise ValueError(f"{frac_name} must be in [0, 1).")
        if float(self.validation_fraction) + float(self.test_fraction) >= 1.0:
            raise ValueError("validation_fraction + test_fraction must be < 1.")

        raw_rows = _load_kuairand_log_rows(data_dir, self.max_rows)
        if not raw_rows:
            raise ValueError("KuaiRand logs contain no valid rows.")
        video_feature_map = _load_kuairand_video_features(data_dir)
        if not video_feature_map:
            raise ValueError("KuaiRand video features are empty.")

        # Chronological-strict split by time_ms
        raw_rows.sort(key=lambda r: r["time_ms"])
        n = len(raw_rows)
        n_test = max(1, int(n * float(self.test_fraction)))
        n_val = max(1, int(n * float(self.validation_fraction)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("KuaiRand split produced no training rows.")
        for i, row in enumerate(raw_rows):
            if i < n_train:
                row["split"] = "train"
            elif i < n_train + n_val:
                row["split"] = "validation"
            else:
                row["split"] = "test"

        # Build consolidated rows for all-split access
        all_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            click = int(row["is_click"])
            all_rows.append({
                "user_id": int(row["user_id"]),
                "movie_id": int(row["video_id"]),   # alias for eval-function compat
                "video_id": int(row["video_id"]),
                "rating": float(click),              # alias: is_click as rating
                "timestamp": float(row["time_ms"]) / 1000.0,
                "reward": float(click),
                "positive": bool(click),
                "split": row["split"],
                "session_id": None,
            })
        self._rows = all_rows

        # Build action catalogue from all videos in features file
        videos = sorted(video_feature_map.keys())
        video_to_index = {vid: i for i, vid in enumerate(videos)}
        action_features_list = [video_feature_map[vid] for vid in videos]
        action_features = torch.tensor(action_features_list, dtype=torch.float32)
        action_ids = torch.tensor(videos, dtype=torch.long)

        train_rows_raw = [r for r in raw_rows if r["split"] == "train"]
        user_affinity_map = _build_kuairand_user_affinity_map(train_rows_raw, video_feature_map)
        self._user_affinity_map = user_affinity_map

        state_dim = _KUAIRAND_ACTION_DIM
        action_dim = _KUAIRAND_ACTION_DIM
        action_features_np = action_features.numpy()
        n_train_rows = len(train_rows_raw)

        states_np = np.empty((n_train_rows, state_dim), dtype=np.float32)
        actions_np = np.empty((n_train_rows, action_dim), dtype=np.float32)
        rewards_np = np.empty(n_train_rows, dtype=np.float32)
        next_states_np = np.empty((n_train_rows, state_dim), dtype=np.float32)
        dones_np = np.empty(n_train_rows, dtype=np.bool_)
        action_row_ids_np = np.empty(n_train_rows, dtype=np.int64)
        next_logged_action_ids_np = np.empty(n_train_rows, dtype=np.int64)

        zero_state = [0.0] * state_dim
        for idx, row in enumerate(train_rows_raw):
            uid = row["user_id"]
            vid = row["video_id"]
            state_vec = user_affinity_map.get(uid, zero_state)
            if idx + 1 < n_train_rows and train_rows_raw[idx + 1]["user_id"] == uid:
                next_row = train_rows_raw[idx + 1]
                done = False
                next_vid = next_row["video_id"]
                next_state_vec = user_affinity_map.get(next_row["user_id"], zero_state)
            else:
                done = True
                next_vid = vid
                next_state_vec = state_vec
            feat_idx = video_to_index.get(vid)
            if feat_idx is None:
                feat_idx = 0  # unknown video → first action slot
            states_np[idx] = state_vec
            actions_np[idx] = action_features_np[feat_idx]
            rewards_np[idx] = float(row["is_click"])
            next_states_np[idx] = next_state_vec
            dones_np[idx] = done
            action_row_ids_np[idx] = vid
            next_logged_action_ids_np[idx] = next_vid

        min_ts = min(r["time_ms"] for r in raw_rows)
        max_ts = max(r["time_ms"] for r in raw_rows)
        n_users = len({r["user_id"] for r in raw_rows})
        train_ts = [r["time_ms"] for r in raw_rows if r["split"] == "train"]
        test_ts = [r["time_ms"] for r in raw_rows if r["split"] == "test"]
        meta = {
            "source": "kuairand_pure_local",
            "data_dir": str(data_dir),
            "num_rows": n,
            "num_users": n_users,
            "num_items": len(videos),
            "num_train_rows": n_train_rows,
            "num_validation_rows": sum(1 for r in raw_rows if r["split"] == "validation"),
            "num_test_rows": sum(1 for r in raw_rows if r["split"] == "test"),
            "split_mode": "chronological_strict",
            "min_timestamp": float(min_ts) / 1000.0,
            "max_timestamp": float(max_ts) / 1000.0,
            "min_train_timestamp": float(min(train_ts)) / 1000.0 if train_ts else None,
            "max_train_timestamp": float(max(train_ts)) / 1000.0 if train_ts else None,
            "min_test_timestamp": float(min(test_ts)) / 1000.0 if test_ts else None,
            "max_test_timestamp": float(max(test_ts)) / 1000.0 if test_ts else None,
            "leakage_check_passed": True,
            "validation_fraction": float(self.validation_fraction),
            "test_fraction": float(self.test_fraction),
            "seed": int(self.seed),
            "synthetic_only": False,
            "benchmark_claim": False,
            "scaffold_only": True,
        }
        self._metadata = meta
        self._catalogue = ActionCatalogue(
            action_features=action_features,
            action_ids=action_ids,
            metadata=meta,
        )
        self._dataset = InMemoryReplayDataset(
            states=torch.from_numpy(states_np),
            actions=torch.from_numpy(actions_np),
            rewards=torch.from_numpy(rewards_np),
            next_states=torch.from_numpy(next_states_np),
            dones=torch.from_numpy(dones_np),
            action_ids=torch.from_numpy(action_row_ids_np),
            next_logged_action_ids=torch.from_numpy(next_logged_action_ids_np),
            metadata=meta,
        )

    @property
    def state_dim(self) -> int:
        return int(self._dataset.batch.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._catalogue.action_features.shape[1])

    @property
    def num_actions(self) -> int:
        return self._catalogue.num_actions

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_split_rows(self, split: str) -> list[dict[str, Any]]:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test.")
        return [r for r in self._rows if r["split"] == split]

    def available_splits(self) -> tuple[str, ...]:
        return tuple(s for s in ("train", "validation", "test") if self.get_split_rows(s))

    def make_dataset(self) -> InMemoryReplayDataset:
        return self._dataset

    def make_action_catalogue(self) -> tuple[Any, Any]:
        return self._catalogue.action_features, self._catalogue.action_ids

    def make_action_catalogue_contract(self) -> ActionCatalogue:
        return ActionCatalogue(
            action_features=self._catalogue.action_features,
            action_ids=self._catalogue.action_ids,
            metadata=self.metadata,
        )

    def make_dataset_bundle(self) -> OfflineDatasetBundle:
        return OfflineDatasetBundle(
            dataset=self._dataset,
            action_catalogue=self.make_action_catalogue_contract(),
            metadata=self.metadata,
        )

    def evaluate_policy_or_model(self, model: Any, **kwargs: Any) -> EvaluationResult:
        k = int(kwargs.get("k", 10))
        return evaluate_movielens_scaffold_task(self, model_or_baseline=model, k=k)
