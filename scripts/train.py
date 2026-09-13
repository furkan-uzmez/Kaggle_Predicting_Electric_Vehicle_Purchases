#!/usr/bin/env python3
"""CLI script to run repeated cross-validation model training."""

import argparse
import sys
from pathlib import Path

# Ensure src directory is in sys.path when script is executed directly
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ev_purchase.training import run_cross_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated CV training for EV purchase prediction."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in fast smoke mode on deterministic sample subset.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Optional path to split manifest parquet file.",
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    manifest_path = Path(args.split_manifest) if args.split_manifest else None

    result = run_cross_validation(
        config_path=cfg_path,
        project_root=project_root,
        split_manifest_path=manifest_path,
        smoke=args.smoke,
    )

    agg = result["aggregate_metrics"]
    print("=== CV Training Complete ===")
    print(f"Run ID: {result['run_id']}")
    print(f"Status: {result['status']}")
    print(f"ROC-AUC: {agg['roc_auc']:.6f}")
    print(f"PR-AUC:  {agg['average_precision']:.6f}")
    print(f"LogLoss: {agg['log_loss']:.6f}")
    print(f"Saved metrics artifact to: {result['metrics_path']}")


if __name__ == "__main__":
    main()
