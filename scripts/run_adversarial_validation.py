"""Leakage-safe adversarial validation runner for train-vs-test domain shift diagnostic."""

import argparse
import sys
from pathlib import Path

# Ensure src is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from ev_purchase.adversarial_validation import (  # noqa: E402
    compute_file_sha256,
    run_adversarial_validation,
)
from ev_purchase.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train-vs-test adversarial validation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "base.yaml"),
        help="Path to base configuration YAML.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run tiny smoke test with a small sample for rapid verification.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "diagnostics"),
        help="Directory to save diagnostic artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, project_root=PROJECT_ROOT)

    train_path = config.data.train
    test_path = config.data.test

    print(f"Loading data from:\n  train: {train_path}\n  test:  {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Compute data sha256 for audit trail
    train_sha = compute_file_sha256(train_path)
    test_sha = compute_file_sha256(test_path)
    data_sha256 = {
        "train_sha256": train_sha,
        "test_sha256": test_sha,
    }

    if args.smoke:
        print("\n--- Running SMOKE test (tiny sample) ---")
        train_df = train_df.head(1000).copy()
        test_df = test_df.head(500).copy()
        lgb_params = {
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "verbose": -1,
            "n_jobs": -1,
        }
        artifacts_dir = None
    else:
        print(
            f"\n--- Running FULL 5-fold adversarial validation on "
            f"{len(train_df)} train and {len(test_df)} test rows ---"
        )
        lgb_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        artifacts_dir = Path(args.artifacts_dir)

    result = run_adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        id_column=config.data.id_column,
        target_column=config.data.target_column,
        n_splits=5,
        random_state=42,
        lgb_params=lgb_params,
        artifacts_dir=artifacts_dir,
        data_sha256=data_sha256,
        source_generator_paths=[
            "src/ev_purchase/adversarial_validation.py",
            "scripts/run_adversarial_validation.py",
            "scripts/build_notebooks.py",
        ],
    )

    print("\nAdversarial validation overall metrics:")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\nPer-fold metrics:")
    for f in result.fold_metrics:
        print(
            f"  fold {f['fold']}: roc_auc={f['roc_auc']:.6f}, "
            f"avg_prec={f['average_precision']:.6f}, log_loss={f['log_loss']:.6f}"
        )

    print("\nTop 10 shift features (fold-averaged importance):")
    print(result.feature_importance.head(10).to_string(index=False))

    print(f"\nInterpretation band: {result.interpretation['shift_band']}")
    print(f"Summary: {result.interpretation['summary']}")
    print(f"Action: {result.interpretation['action']}")
    print(f"Caveats: {result.interpretation['caveats']}")

    if result.artifact_paths:
        print("\nSaved artifacts:")
        for k, p in result.artifact_paths.items():
            print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
