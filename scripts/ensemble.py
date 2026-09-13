"""CLI Script to Execute OOF-Safe Blending and Select Final Model Policy."""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.ensemble import (
    MIN_BLEND_LIFT_THRESHOLD,
    compute_dataset_sha256,
    evaluate_blend_grid,
    load_candidate_runs_predictions,
    materialize_hpo_finalist_predictions,
    promote_ensemble_run_transactionally,
    resolve_dry_run_output_dir,
    save_blended_oof_and_test_artifacts,
    save_ensemble_artifacts,
    update_final_config,
    validate_keyed_oof_alignment,
    validate_keyed_test_alignment,
    weighted_blend,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOF-Safe Blending and Ensembling Evaluation Engine"
    )
    parser.add_argument(
        "--tuning-studies",
        nargs="*",
        default=["outputs/tuning/catboost_hpo", "outputs/tuning/lightgbm_hpo"],
        help="Paths to full-protocol HPO tuning study directories.",
    )
    parser.add_argument(
        "--allow-baseline-runs",
        action="store_true",
        help="Explicitly permit using diagnostic baseline runs specified in --runs.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Explicit diagnostic baseline run directories (requires --allow-baseline-runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute fast validation without modifying canonical state.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save blend leaderboard and metrics artifacts.",
    )
    parser.add_argument(
        "--grid-steps",
        type=int,
        default=10,
        help="Granularity of weight grid search (default 10 -> step 0.1).",
    )
    parser.add_argument(
        "--config-out",
        default="configs/final.yaml",
        help="Path to write/update final locked YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()
    project_root = Path.cwd()

    if args.dry_run:
        target_output_dir = resolve_dry_run_output_dir(args.output_dir, project_root)
        staging_output_dir = target_output_dir
        canonical_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else project_root / "outputs" / "ensemble"
        )
    else:
        canonical_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else project_root / "outputs" / "ensemble"
        )
        staging_output_dir = (
            canonical_dir.parent / f".{canonical_dir.name}_staging_{uuid4().hex[:8]}"
        )

    oof_dict: dict[str, np.ndarray] = {}
    test_dict: dict[str, np.ndarray] = {}
    oof_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    components_info: dict[str, Any] = {}

    if args.runs and not args.allow_baseline_runs:
        raise ValueError(
            "Baseline diagnostic runs specified in --runs are not allowed unless "
            "--allow-baseline-runs is explicitly provided"
        )

    if args.allow_baseline_runs and args.runs:
        logger.info("Operating in DIAGNOSTIC BASELINE mode with runs: %s", args.runs)
        oof_dict, test_dict, y_true, run_payloads = load_candidate_runs_predictions(
            args.runs, project_root=project_root
        )
        for payload in run_payloads:
            cid = payload["run_id"]
            components_info[cid] = {
                "model_family": payload.get("model", "dummy"),
                "tuning_study": "baseline_run",
                "trial_number": 0,
                "params": {},
                "dry_run": False,
            }
    else:
        logger.info(
            "Materializing HPO finalists from tuning studies: %s (dry_run=%s)",
            args.tuning_studies,
            args.dry_run,
        )

        base_cfg = load_config(project_root / "configs" / "base.yaml", project_root=project_root)
        train_df, test_df = load_frames(base_cfg.data.train, base_cfg.data.test)
        contract = DatasetContract(
            id_column=base_cfg.data.id_column, target_column=base_cfg.data.target_column
        )
        validate_frames(train_df, test_df, contract)
        y_true = encode_target(train_df[base_cfg.data.target_column], contract)

        for study_path in args.tuning_studies:
            cand_output_dir = staging_output_dir / "candidates"
            res = materialize_hpo_finalist_predictions(
                tuning_study_dir=study_path,
                rank=1,
                project_root=project_root,
                output_dir=cand_output_dir,
                dry_run=args.dry_run,
            )
            cid = res["candidate_id"]
            oof_dfs[cid] = res["oof_df"]
            test_dfs[cid] = res["test_df"]
            components_info[cid] = res

        # Validate keyed alignment across all materialized finalists
        validate_keyed_oof_alignment(oof_dfs)
        validate_keyed_test_alignment(test_dfs, sample_submission_df=test_df)

        for cid in oof_dfs.keys():
            oof_p = oof_dfs[cid].groupby("row_index")["probability"].mean().sort_index().to_numpy()
            oof_dict[cid] = oof_p
            test_dict[cid] = test_dfs[cid][base_cfg.data.target_column].to_numpy()

    # Evaluate weight grid
    leaderboard_df = evaluate_blend_grid(oof_dict, y_true)
    best_blend_row = leaderboard_df.iloc[0]

    winning_weights_raw = json.loads(best_blend_row["weights_json"])
    winning_roc_auc = float(best_blend_row["roc_auc"])
    lift = float(best_blend_row["lift_over_best_single"])
    is_single_model_winner = bool(best_blend_row["is_single_model"])

    # Find best single candidate
    single_model_rows = leaderboard_df[leaderboard_df["is_single_model"]]
    best_single_row = single_model_rows.iloc[0]
    best_single_weights = json.loads(best_single_row["weights_json"])
    best_single_id = max(best_single_weights, key=best_single_weights.get)
    best_single_roc_auc = float(best_single_row["roc_auc"])

    # Enforce named MIN_BLEND_LIFT_THRESHOLD check
    if lift >= MIN_BLEND_LIFT_THRESHOLD and not is_single_model_winner:
        selected_policy = "blend"
        winning_weights = winning_weights_raw
        logger.info(
            "SELECTED BLEND POLICY: ROC-AUC = %.6f (lift = +%.6f >= threshold %.6f over %s = %.6f)",
            winning_roc_auc,
            lift,
            MIN_BLEND_LIFT_THRESHOLD,
            best_single_id,
            best_single_roc_auc,
        )
    else:
        selected_policy = "single_model"
        winning_weights = {cid: (1.0 if cid == best_single_id else 0.0) for cid in oof_dict.keys()}
        winning_roc_auc = best_single_roc_auc
        lift = 0.0
        logger.info(
            "NO SIGNIFICANT LIFT (%.6f < threshold %.6f). Selecting single best model %s = %.6f",
            lift,
            MIN_BLEND_LIFT_THRESHOLD,
            best_single_id,
            best_single_roc_auc,
        )

    # Compute blended OOF and test predictions
    if oof_dfs:
        first_cid = list(oof_dfs.keys())[0]
        blended_oof_df = oof_dfs[first_cid].copy()

        blended_probs = np.zeros(len(blended_oof_df), dtype=float)
        for cid, w in winning_weights.items():
            if w > 0:
                blended_probs += w * oof_dfs[cid]["probability"].to_numpy()
        blended_oof_df["probability"] = blended_probs

        first_test_cid = list(test_dfs.keys())[0]
        blended_test_df = test_dfs[first_test_cid].copy()
        blended_test_p = weighted_blend(test_dict, winning_weights)
        blended_test_df[base_cfg.data.target_column] = blended_test_p

        save_blended_oof_and_test_artifacts(
            blended_oof_df=blended_oof_df,
            blended_test_df=blended_test_df,
            n_rows=len(y_true),
            n_repeats=base_cfg.validation.n_repeats,
            y_true=y_true,
            output_dir=staging_output_dir,
        )

    # Save leaderboard and metrics artifacts
    metrics_payload = {
        "status": "completed",
        "selected_policy": selected_policy,
        "lift_threshold": MIN_BLEND_LIFT_THRESHOLD,
        "winning_weights": winning_weights,
        "best_single_candidate_id": best_single_id,
        "best_single_roc_auc": round(best_single_roc_auc, 6),
        "winning_roc_auc": round(winning_roc_auc, 6),
        "lift_over_best_single": round(lift, 8),
        "candidate_ids": list(components_info.keys()),
        "runtime_seconds": round(time.time() - start_time, 4),
        "dry_run": args.dry_run,
    }

    save_ensemble_artifacts(
        leaderboard_df,
        metrics_payload,
        output_dir=staging_output_dir,
        target_canonical_dir=target_output_dir if args.dry_run else canonical_dir,
        project_root=project_root,
    )

    if args.dry_run:
        logger.info(
            "DRY-RUN mode complete. Artifacts saved under %s. Canonical state untouched.",
            target_output_dir,
        )
    else:
        manifest_p = project_root / "outputs" / "splits" / "repeated_stratified_5x3.parquet"
        split_sha256 = hashlib.sha256(manifest_p.read_bytes()).hexdigest()

        base_cfg_path = project_root / "configs" / "base.yaml"
        base_cfg = load_config(base_cfg_path, project_root=project_root)
        data_sha256 = compute_dataset_sha256(base_cfg.data.train, base_cfg.data.test)

        staged_config_path = staging_output_dir.parent / f".final.yaml_staging_{uuid4().hex[:8]}"
        update_final_config(
            selected_policy=selected_policy,
            winning_weights=winning_weights,
            components_info=components_info,
            best_single_id=best_single_id,
            winning_roc_auc=winning_roc_auc,
            lift_over_best_single=lift,
            split_sha256=split_sha256,
            data_sha256=data_sha256,
            config_out_path=staged_config_path,
            project_root=project_root,
        )

        promote_ensemble_run_transactionally(
            staging_dir=staging_output_dir,
            output_root=canonical_dir,
            staged_config_path=staged_config_path,
            target_config_path=project_root / args.config_out,
        )

        logger.info("Ensemble evaluation complete. Locked %s", args.config_out)


if __name__ == "__main__":
    main()
