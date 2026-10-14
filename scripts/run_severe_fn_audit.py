"""Execution runner for severe false-negative and learnability audit.

Reconstructs canonical and exploratory out-of-fold predictions from candidate artifacts:
- Canonical selected policy: LightGBM finalist
- Exploratory sensitivity policy: 70% LightGBM + 30% CatBoost
Verifies manifest checksums, row-key alignment, produces deterministic audit
cohorts, frozen slice definitions, and audit reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from ev_purchase.adversarial_validation import compute_file_sha256  # noqa: E402
from ev_purchase.ensemble import compute_dataset_sha256  # noqa: E402
from ev_purchase.severe_fn_audit import run_severe_fn_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run severe false-negative and learnability audit on the canonical policy."
    )
    parser.add_argument(
        "--blend-metrics",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "ensemble" / "blend_metrics.json"),
        help="Selected ensemble-policy manifest used to anchor the canonical baseline.",
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
        "--lgb-candidate-meta",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "ensemble"
            / "candidates"
            / "lightgbm_hpo_trial35"
            / "candidate_metadata.json"
        ),
        help="Path to LightGBM candidate metadata JSON.",
    )
    parser.add_argument(
        "--cb-candidate-meta",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "ensemble"
            / "candidates"
            / "catboost_hpo_trial22"
            / "candidate_metadata.json"
        ),
        help="Path to CatBoost candidate metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "diagnostics" / "severe_fn_audit"),
        help="Directory to save audit artifacts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke audit on small sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    lgb_meta_path = Path(args.lgb_candidate_meta)
    cb_meta_path = Path(args.cb_candidate_meta)
    blend_metrics_path = Path(args.blend_metrics)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("STARTING SEVERE FALSE-NEGATIVE & LEARNABILITY AUDIT")
    print("=" * 70)
    print(f"Train data:       {train_path}")
    print(f"Test data:        {test_path}")
    print(f"LightGBM meta:    {lgb_meta_path}")
    print(f"CatBoost meta:    {cb_meta_path}")
    print(f"Output directory: {output_dir}")

    # 1. Load candidate metadata and verify source checksums
    with open(lgb_meta_path) as f:
        lgb_meta = json.load(f)
    with open(cb_meta_path) as f:
        cb_meta = json.load(f)
    with open(blend_metrics_path) as f:
        blend_metrics = json.load(f)

    expected_policy = "single_model"
    expected_weights = {
        cb_meta["candidate_id"]: 0.0,
        lgb_meta["candidate_id"]: 1.0,
    }
    if blend_metrics.get("selected_policy") != expected_policy:
        raise ValueError(
            "Severe-FN audit currently requires the selected single-model policy; "
            f"got {blend_metrics.get('selected_policy')!r}"
        )
    if blend_metrics.get("winning_weights") != expected_weights:
        raise ValueError(
            "Selected blend weights do not match the canonical LightGBM audit policy: "
            f"expected {expected_weights}, got {blend_metrics.get('winning_weights')}"
        )
    if blend_metrics.get("best_single_candidate_id") != lgb_meta["candidate_id"]:
        raise ValueError("Selected best single candidate is not the configured LightGBM candidate")
    blend_metrics_sha = compute_file_sha256(blend_metrics_path)

    train_sha = compute_file_sha256(train_path)
    print(f"✓ Train file SHA256: {train_sha}")

    dataset_sha = compute_dataset_sha256(train_path, test_path)
    expected_data_sha = lgb_meta["provenance"]["data_sha256"]
    if dataset_sha != expected_data_sha:
        raise ValueError(
            f"Combined dataset SHA256 mismatch! Actual={dataset_sha}, Expected={expected_data_sha}"
        )
    print(f"✓ Combined dataset SHA256 verified: {dataset_sha}")

    lgb_oof_path = PROJECT_ROOT / lgb_meta["artifact_paths"]["oof_predictions"]
    cb_oof_path = PROJECT_ROOT / cb_meta["artifact_paths"]["oof_predictions"]

    lgb_oof_sha = compute_file_sha256(lgb_oof_path)
    expected_lgb_sha = lgb_meta["artifact_checksums"]["oof_predictions"]
    if lgb_oof_sha != expected_lgb_sha:
        raise ValueError(
            f"LightGBM OOF SHA256 mismatch! Actual={lgb_oof_sha}, Expected={expected_lgb_sha}"
        )
    print(f"✓ LightGBM OOF SHA256 verified: {lgb_oof_sha}")

    cb_oof_sha = compute_file_sha256(cb_oof_path)
    expected_cb_sha = cb_meta["artifact_checksums"]["oof_predictions"]
    if cb_oof_sha != expected_cb_sha:
        raise ValueError(
            f"CatBoost OOF SHA256 mismatch! Actual={cb_oof_sha}, Expected={expected_cb_sha}"
        )
    print(f"✓ CatBoost OOF SHA256 verified: {cb_oof_sha}")

    # 2. Load DataFrames
    print("\nLoading training and out-of-fold predictions...")
    train_df = pd.read_csv(train_path)
    lgb_oof_df = pd.read_parquet(lgb_oof_path)
    cb_oof_df = pd.read_parquet(cb_oof_path)

    if args.smoke:
        print("Running SMOKE mode (first 2,000 rows)...")
        train_df = train_df.head(2000).copy()
        valid_indices = set(train_df["id"].values)
        lgb_oof_df = lgb_oof_df[lgb_oof_df["row_index"].isin(valid_indices)].copy()
        cb_oof_df = cb_oof_df[cb_oof_df["row_index"].isin(valid_indices)].copy()

    source_provenance = {
        "train_file_sha256": train_sha,
        "combined_train_test_data_sha256": dataset_sha,
        "canonical_policy": "LightGBM",
        "canonical_policy_manifest": str(blend_metrics_path.relative_to(PROJECT_ROOT)),
        "canonical_policy_manifest_sha256": blend_metrics_sha,
        "canonical_selected_policy": blend_metrics["selected_policy"],
        "canonical_lightgbm_candidate_id": lgb_meta["candidate_id"],
        "lightgbm_oof_sha256": lgb_oof_sha,
        "exploratory_policy": "0.70 * LightGBM + 0.30 * CatBoost",
        "exploratory_catboost_candidate_id": cb_meta["candidate_id"],
        "catboost_oof_sha256": cb_oof_sha,
    }

    # 3. Run audit
    result = run_severe_fn_audit(
        train_df=train_df,
        lgb_oof_df=lgb_oof_df,
        cb_oof_df=cb_oof_df,
        output_dir=output_dir,
        source_provenance=source_provenance,
    )

    print("\n" + "=" * 70)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 70)
    print(f"Canonical Baseline (LightGBM): {lgb_meta['candidate_id']}")
    print(
        "Exploratory 30/70 Sensitivity: "
        f"30% {cb_meta['candidate_id']} + 70% {lgb_meta['candidate_id']}"
    )
    print(f"Total Rows Analyzed:           {result.total_rows:,}")
    print(
        f"Severe False Negatives (p<=0.1): {result.severe_fn_count:,} ({result.severe_fn_rate:.4%})"
    )
    print(
        f"Top 1% Sample LogLoss Count:   {result.top_loss_count:,} "
        f"(Threshold: {result.top_loss_threshold:.4f})"
    )
    print(f"Margin Band [0.25, 0.75]:      {result.margin_band_25_75_count:,}")
    print(f"Margin Band [0.40, 0.60]:      {result.margin_band_40_60_count:,}")
    print("\nModel Consistency (on Severe FNs):")
    for k, v in result.model_consistency.items():
        print(f"  {k}: {v:.2%}")
    print("\nVariance Diagnostics (on Severe FNs):")
    for k, v in result.variance_summary.items():
        print(f"  {k}: {v:.6f}")
    print("\nNeighborhood Label Mixing (on Severe FNs):")
    for k, v in result.neighborhood_summary.items():
        print(f"  {k}: {v:.4f}")

    print("\nGenerated Artifacts:")
    for fname, sha in result.artifact_checksums.items():
        print(f"  - {fname}: {sha}")
    print("\nAudit successfully complete.")


if __name__ == "__main__":
    main()
