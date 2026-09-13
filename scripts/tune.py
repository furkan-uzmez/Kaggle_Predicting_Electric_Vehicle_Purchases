#!/usr/bin/env python3
"""CLI script for bounded hyperparameter optimization of GBDT models."""

import argparse
import sys
from pathlib import Path

from ev_purchase.tuning import run_hpo_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded hyperparameter optimization for GBDT models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tuning YAML configuration file.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Optional override for total target trials count.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke optimization on deterministic sample subset.",
    )

    args = parser.parse_args()

    if args.trials is not None and args.trials <= 0:
        print("Error: --trials must be a positive integer", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        results = run_hpo_study(
            config_path=config_path,
            n_trials=args.trials,
            smoke=args.smoke,
        )

        meta = results["metadata"]
        print("=" * 60)
        print("HPO STUDY COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Study Name:          {results['study_name']}")
        print(f"Model Family:        {meta['model_name']}")
        print(f"Smoke Mode:          {meta['smoke']}")
        print(f"Trials Target:       {meta['n_trials_target']}")
        print(f"Trials Executed Tot: {meta['n_trials_executed_total']}")
        print(f"Trials Executed Run: {meta['n_trials_executed_run']}")
        print(f"Search Fold IDs:     {meta['provenance']['effective_search_fold_ids']}")
        print(f"Finalist Fold IDs:   {meta['provenance']['effective_finalist_fold_ids']}")
        print(f"Trials Parquet:      {results['trials_path']}")
        print(f"Finalists Parquet:   {results['finalists_path']}")
        print(f"Finalist Folds Pq:   {results['finalist_fold_metrics_path']}")
        print(f"Metadata JSON:       {results['metadata_path']}")
        print("=" * 60)

    except Exception as exc:
        print(f"Error executing HPO study: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
