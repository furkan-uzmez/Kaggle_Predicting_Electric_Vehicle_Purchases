"""Execution runner for targeted single-feature experiment: Concern × Subsidy.

Executes a controlled single-feature candidate training against the canonical
lightgbm_hpo_trial35 baseline, computes fold/repeat/global metrics and
Nadeau-Bengio corrected resampled hypothesis tests, evaluates frozen slices,
makes dynamic gate decisions, and serializes artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ev_purchase.targeted_feature_experiment import run_targeted_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted single-feature experiment: concern_subsidy, "
        "charging_access, commute_anxiety, income_per_vehicle, age_income."
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="concern_subsidy",
        choices=[
            "concern_subsidy",
            "charging_access",
            "commute_anxiety",
            "income_per_vehicle",
            "age_income",
        ],
        help="Targeted feature policy to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Artifacts directory (defaults to outputs/.../<policy>).",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "playground-series-s6e9" / "train.csv"),
        help="Path to training data CSV.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "playground-series-s6e9" / "test.csv"),
        help="Path to test data CSV.",
    )
    parser.add_argument(
        "--final-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "final.yaml"),
        help="Path to final production config.",
    )
    parser.add_argument(
        "--splits-manifest",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "splits" / "repeated_stratified_5x3.parquet"),
        help="Path to 5x3 repeated stratified splits parquet.",
    )
    parser.add_argument(
        "--baseline-oof",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "ensemble"
            / "candidates"
            / "lightgbm_hpo_trial35"
            / "oof_predictions.parquet"
        ),
        help="Path to baseline OOF predictions parquet.",
    )
    parser.add_argument(
        "--frozen-slices",
        type=str,
        default=str(
            PROJECT_ROOT / "outputs" / "diagnostics" / "severe_fn_audit" / "frozen_slices.json"
        ),
        help="Path to frozen slices definitions JSON.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test on one fold only.",
    )
    parser.add_argument(
        "--reuse-candidate-oof",
        action="store_true",
        help="Reuse existing candidate OOF if already generated in output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "experiments" / "targeted_features" / args.policy
    )

    print("=" * 70)
    print(f"TARGETED FEATURE EXPERIMENT: {args.policy}")
    print("=" * 70)
    print(f"Policy:           {args.policy}")
    print(f"Output directory: {out_dir}")
    print(f"Smoke mode:       {args.smoke}")
    print(f"Reuse cand OOF:   {args.reuse_candidate_oof}")

    result = run_targeted_experiment(
        train_path=args.train_data,
        test_path=args.test_data,
        manifest_path=args.splits_manifest,
        baseline_oof_path=args.baseline_oof,
        final_config_path=args.final_config,
        frozen_slices_path=args.frozen_slices,
        output_dir=out_dir,
        smoke=args.smoke,
        reuse_candidate_oof=args.reuse_candidate_oof,
        feature_policy=args.policy,
    )

    if args.smoke:
        print("\n--- SMOKE EXECUTION FINISHED ---")
        print(f"Validation rows evaluated: {result['val_rows']:,}")
        print(f"Baseline AUC:             {result['baseline_auc']:.6f}")
        print(f"Candidate AUC:            {result['candidate_auc']:.6f}")
        print(f"Delta AUC:                {result['delta_auc']:+.6f}")
        return

    decision = result["decision"]
    metrics = result["metrics"]["global_metrics"]
    nb = result["metrics"]["nadeau_bengio"]

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"Gate Decision:            {decision.decision}")
    print(f"Decision Reason:          {decision.reason}")
    print(f"Global Baseline ROC-AUC:  {metrics['baseline_roc_auc']:.6f}")
    print(f"Global Candidate ROC-AUC: {metrics['candidate_roc_auc']:.6f}")
    print(f"Global Delta ROC-AUC:     {metrics['delta_roc_auc']:+.6f}")
    print(f"Global Delta PR-AUC:      {metrics['delta_pr_auc']:+.6f}")
    print(f"Global Delta LogLoss:     {metrics['delta_log_loss']:+.6f}")
    print(f"Repeat Deltas:            {[round(d, 6) for d in metrics['repeat_auc_deltas']]}")
    print(f"NB Mean Difference:       {nb['mean_difference']:+.6f}")
    print(f"NB 95% CI:                [{nb['ci_nb_lower']:+.6f}, {nb['ci_nb_upper']:+.6f}]")
    print(f"NB p-value:               {nb['p_value_nb_raw']:.4e}")
    print(f"NB Conclusion:            {nb['conclusion']}")
    print(f"\nArtifacts saved to:       {out_dir}")


if __name__ == "__main__":
    main()
