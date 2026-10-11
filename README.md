# Retrieval Alignment for ANN-Deployed Offline Reinforcement Learning: A Diagnostic Study

The release contains the code for the manuscript experiments. It evaluates how approximate nearest-neighbour action retrieval changes offline reinforcement-learning behaviour, with retrieval-aware critics, density correction, and exact-search audits.

## Install

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests
```

Run the synthetic wiring check:

```bash
python -m retrieval_alignment.train --smoke
```

The `src/retrieval_alignment` package contains the implementation. The `real_data_*` configuration files describe the MovieLens and KuaiRand experiment matrix. The `run_real_data_*` scripts run or inspect those experiments. `run_known_bandit_experiment.py` runs the controlled known-propensity validation.

No manuscript, dataset, result artifact, or generated binary is included. Supply local data paths when running the real-data commands.

Install FAISS for HNSW retrieval audits:

```bash
python -m pip install faiss-cpu
```

Preview the MovieLens and KuaiRand experiment matrix:

```bash
python scripts/run_real_data_matrix.py --dataset all --data-root /path/to/dataset_root --output-root runs/real_data --dry-run
```

The dataset root must contain `data/movielens/ml-25m/ratings.csv` and
`data/kuairand/KuaiRand-Pure/data`. Remove `--dry-run` to run the matrix.
Use `--help` for each script to see its options.

