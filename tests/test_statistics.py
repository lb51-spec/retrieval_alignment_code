"""Tests for descriptive synthetic sweep statistics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.statistics import (  # noqa: E402
    MetricSummary,
    bootstrap_ci_mean,
    bootstrap_confidence_interval,
    group_metric_summary,
    interquartile_mean,
    paired_mean_delta,
    performance_profile_points,
    probability_of_improvement,
    summarize_experiment_sweep_statistics,
    summarize_runs_by_group,
    summarize_values,
)


def test_summarize_values_basic_mean_std_stderr() -> None:
    summary = summarize_values("metric", [1.0, 2.0, 3.0])

    assert isinstance(summary, MetricSummary)
    assert summary.metric_name == "metric"
    assert summary.count == 3
    assert summary.mean == pytest.approx(2.0)
    assert summary.std == pytest.approx(1.0)
    assert summary.stderr == pytest.approx(1.0 / math.sqrt(3.0))
    assert summary.min == 1.0
    assert summary.max == 3.0
    assert summary.synthetic_only is True
    assert summary.benchmark_claim is False


def test_summarize_values_count_one_std_zero() -> None:
    summary = summarize_values("metric", [7.0])

    assert summary.count == 1
    assert summary.mean == 7.0
    assert summary.std == 0.0
    assert summary.stderr == 0.0


def test_summarize_values_ignores_none_and_handles_empty() -> None:
    summary = summarize_values("metric", [None, 1.0, None, 3.0])
    empty = summarize_values("empty", [None])

    assert summary.count == 2
    assert summary.mean == pytest.approx(2.0)
    assert empty.count == 0
    assert empty.mean is None
    assert empty.std is None
    assert empty.stderr is None
    assert empty.min is None
    assert empty.max is None


def test_summarize_values_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        summarize_values("metric", [1.0, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        summarize_values("metric", [float("inf")])


def test_summarize_runs_by_group_groups_and_skips_missing_metrics() -> None:
    runs = [
        {"policy": "online", "refresh": 1, "loss": 1.0, "recall": 0.9},
        {"policy": "online", "refresh": 1, "loss": 3.0},
        {"policy": "ema", "refresh": 5, "loss": 10.0, "recall": 0.5},
    ]

    rows = summarize_runs_by_group(
        runs,
        group_keys=("policy", "refresh"),
        metric_keys=("loss", "recall", "missing"),
    )

    online = next(row for row in rows if row["group"]["policy"] == "online")
    ema = next(row for row in rows if row["group"]["policy"] == "ema")
    assert online["count"] == 2
    assert online["metrics"]["loss"]["mean"] == pytest.approx(2.0)
    assert online["metrics"]["loss"]["std"] == pytest.approx(math.sqrt(2.0))
    assert online["metrics"]["recall"]["count"] == 1
    assert "missing" not in online["metrics"]
    assert ema["metrics"]["loss"]["std"] == 0.0


def test_summarize_runs_by_group_missing_group_key_raises() -> None:
    with pytest.raises(ValueError, match="group key"):
        summarize_runs_by_group([{"policy": "online", "loss": 1.0}], ("policy", "refresh"), ("loss",))


def test_summarize_experiment_sweep_statistics_uses_defaults() -> None:
    summary = {
        "runs": [
            {
                "index_encoder_policy": "online",
                "index_refresh_every": 1,
                "mean_bellman_loss": 2.0,
                "final_total_loss": 5.0,
            },
            {
                "index_encoder_policy": "online",
                "index_refresh_every": 1,
                "mean_bellman_loss": 4.0,
                "final_total_loss": 7.0,
            },
        ]
    }

    stats = summarize_experiment_sweep_statistics(summary)

    assert stats["synthetic_only"] is True
    assert stats["benchmark_claim"] is False
    assert stats["group_keys"] == ["index_encoder_policy", "index_refresh_every"]
    assert len(stats["groups"]) == 1
    metrics = stats["groups"][0]["metrics"]
    assert metrics["mean_bellman_loss"]["mean"] == pytest.approx(3.0)
    assert metrics["final_total_loss"]["mean"] == pytest.approx(6.0)


def test_interquartile_mean_uses_middle_mass() -> None:
    assert interquartile_mean([1.0, 2.0, 3.0, 100.0]) == pytest.approx(2.5)


def test_bootstrap_confidence_interval_contains_mean_for_simple_values() -> None:
    low, high = bootstrap_confidence_interval(
        [1.0, 2.0, 3.0],
        lambda sample: sum(sample) / len(sample),
        num_resamples=100,
        seed=0,
    )

    assert low <= 2.0 <= high


def test_probability_of_improvement_counts_pairwise_wins_and_ties() -> None:
    assert probability_of_improvement([2.0, 3.0], [1.0, 3.0]) == pytest.approx(0.625)


def test_performance_profile_points_returns_claim_free_rows() -> None:
    rows = performance_profile_points({"a": [1.0, 2.0], "b": [3.0, 4.0]}, [2.0, 4.0])

    assert rows[0]["fraction_groups_at_or_above"] == pytest.approx(1.0)
    assert rows[1]["fraction_groups_at_or_above"] == pytest.approx(0.5)
    assert rows[0]["benchmark_claim"] is False


def test_robust_statistics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        interquartile_mean([])
    with pytest.raises(ValueError, match="finite"):
        probability_of_improvement([1.0], [float("nan")])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_confidence_interval([1.0], sum, confidence=1.5)


def test_paired_mean_delta_basic() -> None:
    assert paired_mean_delta([3.0, 4.0], [1.0, 2.0]) == pytest.approx(2.0)


def test_paired_mean_delta_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_mean_delta([1.0, 2.0], [1.0])


def test_bootstrap_ci_mean_deterministic_with_seed() -> None:
    low1, high1 = bootstrap_ci_mean([1.0, 2.0, 3.0, 4.0], num_resamples=200, seed=7)
    low2, high2 = bootstrap_ci_mean([1.0, 2.0, 3.0, 4.0], num_resamples=200, seed=7)
    assert low1 == pytest.approx(low2)
    assert high1 == pytest.approx(high2)
    assert low1 <= 2.5 <= high1


def test_group_metric_summary_is_alias_for_summarize_runs_by_group() -> None:
    rows = [
        {"route": "ml", "method": "retrieval_alignment", "final_bellman_loss": 0.5},
        {"route": "ml", "method": "retrieval_alignment", "final_bellman_loss": 0.3},
        {"route": "ml", "method": "random", "final_bellman_loss": 1.0},
    ]
    result = group_metric_summary(rows, ["route", "method"], ["final_bellman_loss"])
    expected = summarize_runs_by_group(rows, ["route", "method"], ["final_bellman_loss"])
    assert len(result) == len(expected)
