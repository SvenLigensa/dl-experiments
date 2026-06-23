#!/usr/bin/env python
"""Run an experiment over several seeds and report mean +/- std.

The real lesson of the practical: **one seed lies**. A single run is a sample from a
distribution; report the distribution. This script runs ``src.train`` once per seed
(in a subprocess, so each gets a clean interpreter/RNG state), reads each run's final
metrics from its ``run_metadata.json`` + the metrics it logged, and aggregates.

Usage (on a branch that ships a task, e.g. ``example``)::

    python scripts/run_seeds.py +experiment=fashion_mnist_vit --seeds 0 1 2 3 4

Any extra args are forwarded verbatim to ``src.train`` as Hydra overrides. We capture
each run's final ``val/*`` metrics from the JSON the trainer writes (see below) and
print a table of mean +/- std.

Reference: Bouthillier et al., "Accounting for Variance in Machine Learning
Benchmarks", MLSys 2021.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds to run (default: 0 1 2 3 4).",
    )
    return parser.parse_known_args()


def run_one(seed: int, run_dir: Path, overrides: list[str]) -> dict[str, float]:
    """Run training for a single seed and return its final metrics."""
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"trainer.seed={seed}",
        f"hydra.run.dir={run_dir}",
        # Final metrics are dumped to <run_dir>/final_metrics.json by the trainer.
        *overrides,
    ]
    print(f"\n=== seed {seed} ===\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    metrics_file = run_dir / "final_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"{metrics_file} not found. Ensure the trainer writes final_metrics.json "
            "(see src/train.py)."
        )
    with open(metrics_file) as f:
        return json.load(f)


def aggregate(per_seed: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    keys = sorted({k for m in per_seed for k in m})
    summary = {}
    for key in keys:
        values = [m[key] for m in per_seed if key in m]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[key] = (mean, std)
    return summary


def main() -> None:
    args, overrides = parse_args()
    base = Path("multirun/seeds")
    base.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        run_dir = base / f"seed_{seed}"
        per_seed.append(run_one(seed, run_dir, overrides))

    summary = aggregate(per_seed)
    print("\n" + "=" * 60)
    print(f"Results over {len(args.seeds)} seeds: {args.seeds}")
    print("=" * 60)
    for key, (mean, std) in summary.items():
        print(f"{key:30s} {mean:10.4f} +/- {std:.4f}")
    print("=" * 60)
    print("Report the mean +/- std, not the best single run.")


if __name__ == "__main__":
    main()
