"""Run the canonical LightGBM three-seed bagging experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ev_purchase.seed_bagging import run_seed_bagging_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse experiment paths and smoke-mode selection."""
    parser = argparse.ArgumentParser(description="Run LightGBM 3-seed bagged 5x3 CV experiment")
    parser.add_argument(
        "--train-data",
        default=PROJECT_ROOT / "data" / "playground-series-s6e9" / "train.csv",
        type=Path,
    )
    parser.add_argument(
        "--test-data",
        default=PROJECT_ROOT / "data" / "playground-series-s6e9" / "test.csv",
        type=Path,
    )
    parser.add_argument(
        "--splits-manifest",
        default=PROJECT_ROOT / "outputs" / "splits" / "repeated_stratified_5x3.parquet",
        type=Path,
    )
    parser.add_argument(
        "--baseline-oof",
        default=(
            PROJECT_ROOT
            / "outputs"
            / "ensemble"
            / "candidates"
            / "lightgbm_hpo_trial35"
            / "oof_predictions.parquet"
        ),
        type=Path,
    )
    parser.add_argument(
        "--final-config", default=PROJECT_ROOT / "configs" / "final.yaml", type=Path
    )
    parser.add_argument(
        "--frozen-slices",
        default=(
            PROJECT_ROOT / "outputs" / "diagnostics" / "severe_fn_audit" / "frozen_slices.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "outputs" / "experiments" / "seed_bagging" / "lightgbm_3seed",
        type=Path,
    )
    parser.add_argument("--smoke", action="store_true", help="Train only repeat 0, fold 0")
    return parser.parse_args()


def main() -> None:
    """Execute the experiment and print its exact summary."""
    args = parse_args()
    result = run_seed_bagging_experiment(
        train_path=args.train_data,
        test_path=args.test_data,
        manifest_path=args.splits_manifest,
        baseline_oof_path=args.baseline_oof,
        final_config_path=args.final_config,
        frozen_slices_path=args.frozen_slices,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
    if args.smoke:
        print(f"Smoke fold: {result['fold']}")
        print(f"Rows: {result['val_rows']:,}")
        print(f"Baseline ROC-AUC: {result['baseline_auc']:.9f}")
        print(f"Bagged ROC-AUC: {result['candidate_auc']:.9f}")
        print(f"Delta ROC-AUC: {result['candidate_auc_delta']:+.9f}")
        print(f"Runtime seconds: {result['runtime_seconds']:.3f}")
        return

    global_metrics = result["metrics"]["global_metrics"]
    corrected = result["metrics"]["nadeau_bengio"]
    diversity = result["metrics"]["diversity"]
    print(f"Gate decision: {result['decision'].decision}")
    print(f"Global ROC-AUC delta: {global_metrics['delta_roc_auc']:+.9f}")
    print(f"Repeat ROC-AUC deltas: {global_metrics['repeat_auc_deltas']}")
    print(
        f"Nadeau-Bengio 95% CI: [{corrected['ci_nb_lower']:+.9f}, {corrected['ci_nb_upper']:+.9f}]"
    )
    print(f"Nadeau-Bengio p-value: {corrected['p_value_nb_raw']:.8g}")
    print(f"Mean seed Pearson correlation: {diversity['mean_pairwise_pearson']:.9f}")
    print(f"Frozen slice/cohort rows: {len(result['slice_df'])}")
    print(f"Runtime seconds: {result['runtime_seconds']:.3f}")
    print(f"Artifacts: {result['output_dir']}")


if __name__ == "__main__":
    main()
