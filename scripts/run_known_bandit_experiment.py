"""Run the controlled known-propensity contextual-bandit appendix experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from retrieval_alignment.known_bandit import KnownBanditConfig, run_known_bandit_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/known_bandit"))
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-train", type=int, default=10_000)
    parser.add_argument("--num-eval", type=int, default=2_000)
    parser.add_argument("--num-actions", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error("--seeds must be a non-empty comma-separated list of nonnegative integers")
    config = KnownBanditConfig(
        num_train=args.num_train,
        num_eval=args.num_eval,
        num_actions=args.num_actions,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [run_known_bandit_seed(config, seed) for seed in seeds]
    (args.output / "reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    summary: dict[str, dict[str, float]] = {}
    for name in reports[0]["policies"]:
        keys = [k for k, v in reports[0]["policies"][name].items() if isinstance(v, (int, float))]
        summary[name] = {}
        for key in keys:
            values = [r["policies"][name][key] for r in reports]
            summary[name][key] = float(sum(values) / len(values))
            variance = sum((v - summary[name][key]) ** 2 for v in values) / max(len(values) - 1, 1)
            summary[name][f"{key}_std"] = float(variance**0.5)
    for k in config.ks:
        for backend, ef_search in (("exact", 0), ("hnsw", max(config.hnsw_ef_search_values))):
            base = [r["policies"][f"base_k{k}_{backend}_ef{ef_search}"] for r in reports]
            b2 = [r["policies"][f"b2_k{k}_{backend}_ef{ef_search}"] for r in reports]
            for metric in ("true_value", "ips", "snips"):
                deltas = [b2[i][metric] - base[i][metric] for i in range(len(reports))]
                mean = sum(deltas) / len(deltas)
                summary[f"b2_minus_base_k{k}_{backend}_ef{ef_search}_{metric}"] = {
                    "mean": float(mean),
                    "std": float(
                        (sum((v - mean) ** 2 for v in deltas) / max(len(deltas) - 1, 1)) ** 0.5
                    ),
                    "all_positive": bool(all(v > 0 for v in deltas)),
                }
    payload = {
        "config": asdict(config),
        "seeds": seeds,
        "known_propensities": True,
        "summary": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
