"""Run or inspect the predeclared full-scale real_data experiment matrix.

The runner is intentionally sequential and resumable: it leaves parallel GPU
scheduling to the caller while refusing to mix configurations in one artifact
directory.  Use ``--dry-run`` before a multi-hour launch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from retrieval_alignment.config import (
    load_kuairand_chrono_protocol_config,
    load_movielens_chrono_protocol_config,
)
from retrieval_alignment.real_data_experiments import run_real_data_experiment


DEFAULT_CONFIGS = {
    "movielens": (
        "real_data_ml25m_base.yaml",
        "real_data_ml25m_b2_only.yaml",
        "real_data_ml25m_cql.yaml",
        "real_data_ml25m_cql_b2.yaml",
        "real_data_ml25m_bcq.yaml",
        "real_data_ml25m_bcq_b2.yaml",
    ),
    "kuairand": (
        "real_data_kuairand_base.yaml",
        "real_data_kuairand_b2_only.yaml",
        "real_data_kuairand_density_only.yaml",
        "real_data_kuairand_full.yaml",
        "real_data_kuairand_cql.yaml",
        "real_data_kuairand_cql_b2.yaml",
        "real_data_kuairand_bcq.yaml",
        "real_data_kuairand_bcq_b2.yaml",
    ),
}


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in text.split(",") if part.strip())
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be a non-empty comma-separated list of nonnegative integers.")
    return seeds


def _completed_seeds(
    root: Path,
    dataset: str,
    protocol_name: str,
    num_steps: int,
    max_eval_rows: int,
) -> set[int]:
    """Find only artifacts produced by this exact high-level protocol."""

    completed: set[int] = set()
    for path in (root / dataset).glob("*/seed_*/real_data_evaluation.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload.get("config", {})
            if (
                config.get("protocol_name") == protocol_name
                and config.get("num_steps") == num_steps
                and config.get("max_eval_rows") == max_eval_rows
            ):
                completed.add(int(config["seed"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            continue
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("movielens", "kuairand", "all"), default="all")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=tuple(range(10)))
    parser.add_argument("--num-steps", type=int, default=5_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2_000_000,
        help="MovieLens reservoir size; use a positive cap to keep preprocessing memory-bounded.",
    )
    parser.add_argument("--max-eval-rows", type=int, default=10_000)
    parser.add_argument("--alignment-warmup-steps", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if (
        args.num_steps <= 0
        or args.max_rows <= 0
        or args.max_eval_rows <= 0
        or args.alignment_warmup_steps < 0
    ):
        parser.error("steps, row cap, and evaluation rows must be positive; warm-up must be nonnegative.")

    repo_root = args.data_root.resolve()
    config_dir = repo_root / "configs"
    root = args.output_root.resolve()
    datasets = tuple(DEFAULT_CONFIGS) if args.dataset == "all" else (args.dataset,)
    planned: list[dict[str, object]] = []
    for dataset in datasets:
        for filename in DEFAULT_CONFIGS[dataset]:
            for seed in args.seeds:
                planned.append({"dataset": dataset, "config": filename, "seed": seed})
    print(json.dumps({"planned_runs": planned, "count": len(planned)}, indent=2))
    if args.dry_run:
        return 0

    for dataset in datasets:
        for filename in DEFAULT_CONFIGS[dataset]:
            config_path = config_dir / filename
            if dataset == "movielens":
                config = load_movielens_chrono_protocol_config(config_path)
                config = replace(
                    config,
                    data_path=str(repo_root / "data/movielens/ml-25m/ratings.csv"),
                    num_steps=args.num_steps,
                    max_rows=args.max_rows,
                    row_sampling_mode="reservoir",
                    max_eval_rows=args.max_eval_rows,
                    retrieval_alignment_warmup_steps=args.alignment_warmup_steps,
                )
            else:
                config = load_kuairand_chrono_protocol_config(config_path)
                config = replace(
                    config,
                    data_dir=str(repo_root / "data/kuairand/KuaiRand-Pure/data"),
                    num_steps=args.num_steps,
                    max_rows=None,
                    max_eval_rows=args.max_eval_rows,
                    retrieval_alignment_warmup_steps=args.alignment_warmup_steps,
                )
            if args.skip_completed:
                completed = _completed_seeds(
                    root,
                    dataset,
                    config.protocol_name,
                    args.num_steps,
                    args.max_eval_rows,
                )
                missing = [seed for seed in args.seeds if seed not in completed]
                if not missing:
                    continue
                run_seeds = tuple(missing)
            else:
                run_seeds = args.seeds
            print(
                f"starting dataset={dataset} config={filename} "
                f"seeds={','.join(str(seed) for seed in run_seeds)}",
                flush=True,
            )
            result = run_real_data_experiment(
                dataset=dataset,
                config=config,
                seeds=run_seeds,
                output_root=root,
            )
            print(
                f"completed dataset={dataset} config={filename} "
                f"runs={len(result['runs'])}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
