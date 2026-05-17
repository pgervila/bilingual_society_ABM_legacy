"""Top-level entrypoint for running a bilangsim simulation.

Usage:
    PYTHONHASHSEED=0 uv run python run.py [--people N] [--clusters N] [--steps N] ...

PYTHONHASHSEED must be set in the environment BEFORE Python starts (Python reads
it during interpreter initialization, so setting it from inside Python is too
late). The VSCode launch configurations in .vscode/launch.json take care of that.
"""
from __future__ import annotations

import argparse

from bilangsim import BiLangModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bilangsim simulation.")
    parser.add_argument("--people", type=int, default=400,
                        help="Initial population size.")
    parser.add_argument("--clusters", type=int, default=2,
                        help="Number of city clusters.")
    parser.add_argument("--steps", type=int, default=50,
                        help="Steps to simulate (after warmup).")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Warmup steps before data collection begins. "
                             "Currently a fallback while the proper IC system is "
                             "being redesigned; leave at 0 once that lands.")
    parser.add_argument("--save-dir", default="",
                        help="Directory for Parquet result parts. Omit to skip saving.")
    parser.add_argument("--save-data-freq", type=int, default=50,
                        help="Flush results to disk every N steps.")
    args = parser.parse_args()

    model = BiLangModel(
        args.people,
        num_clusters=args.clusters,
        warmup_steps=args.warmup,
    )
    model.run_model(args.steps,
                    save_data_freq=args.save_data_freq,
                    save_dir=args.save_dir)


if __name__ == "__main__":
    main()
