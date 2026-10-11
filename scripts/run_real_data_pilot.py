"""Run a bounded real_data experiment from a predeclared configuration."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from retrieval_alignment.config import (
    load_kuairand_chrono_protocol_config,
    load_movielens_chrono_protocol_config,
)
from retrieval_alignment.real_data_experiments import run_real_data_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("movielens", "kuairand"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Training-row cap; pass 0 to retain the configured full dataset.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=1_000,
        help="Held-out-row cap; pass 0 to retain the configured full split.",
    )
    parser.add_argument("--alignment-warmup-steps", type=int, default=0)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    max_rows = None if args.max_rows == 0 else args.max_rows
    max_eval_rows = None if args.max_eval_rows == 0 else args.max_eval_rows
    if max_rows is not None and max_rows <= 0:
        parser.error("--max-rows must be positive or 0 for no override.")
    if max_eval_rows is not None and max_eval_rows <= 0:
        parser.error("--max-eval-rows must be positive or 0 for no override.")
    os.chdir(data_root)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if args.dataset == "movielens":
        config = load_movielens_chrono_protocol_config(config_path)
        config = replace(
            config,
            data_path=str(data_root / "data/movielens/ml-25m/ratings.csv"),
            num_steps=args.num_steps,
            max_rows=max_rows,
            max_eval_rows=max_eval_rows,
            retrieval_alignment_warmup_steps=args.alignment_warmup_steps,
        )
    else:
        config = load_kuairand_chrono_protocol_config(config_path)
        config = replace(
            config,
            data_dir=str(data_root / "data/kuairand/KuaiRand-Pure/data"),
            num_steps=args.num_steps,
            max_rows=max_rows,
            max_eval_rows=max_eval_rows,
            retrieval_alignment_warmup_steps=args.alignment_warmup_steps,
        )
    result = run_real_data_experiment(
        dataset=args.dataset,
        config=config,
        seeds=seeds,
        output_root=output_root,
    )
    for run in result["runs"]:
        print(
            f"{run['dataset']} {run['method']} seed={run['seed']} "
            f"status={run['status']} evaluation={run['real_data_evaluation']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
