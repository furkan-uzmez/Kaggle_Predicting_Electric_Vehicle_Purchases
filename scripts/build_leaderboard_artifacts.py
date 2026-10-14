import argparse
import hashlib
import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ev_purchase.leaderboard import (
    CANONICAL_LEADERBOARD_MODELS,
    EXCLUDED_LEADERBOARD_MODELS,
    blend_oof,
    compute_all_pairwise_cis,
    compute_dual_spearman_correlations,
    compute_fold_scores,
    format_pairwise_ci_markdown,
    format_spearman_correlation_markdown,
    load_and_validate_recovered_models,
    paired_fold_comparison,
    parse_public_leaderboard_markdown,
    update_leaderboard_markdown,
    update_managed_section,
)

OOF_SOURCES = {
    "dummy": "outputs/runs/dummy_1018318c/oof_predictions.parquet",
    "logistic": "outputs/runs/logistic_865249de/oof_predictions.parquet",
    "lightgbm_baseline": "outputs/runs/lightgbm_11a90eac/oof_predictions.parquet",
    "xgboost_baseline": "outputs/runs/xgboost_70d0e5fd/oof_predictions.parquet",
    "catboost_baseline": "outputs/runs/catboost_0efc363f/oof_predictions.parquet",
    "catboost_hpo_trial22": (
        "outputs/ensemble/candidates/catboost_hpo_trial22/oof_predictions.parquet"
    ),
    "lightgbm_hpo_trial35": (
        "outputs/ensemble/candidates/lightgbm_hpo_trial35/oof_predictions.parquet"
    ),
}

ENSEMBLE_NAME = "ensemble_catboost30_lightgbm70"
BLEND_WEIGHTS = {
    "catboost_hpo_trial22": 0.3,
    "lightgbm_hpo_trial35": 0.7,
}


def _file_sha256(path: Path) -> str:
    """Compute sha256 hex string of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_leaderboard_artifacts(
    project_root: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Build deterministic leaderboard artifacts and manifest.

    Args:
        project_root: Path to repository root.
        output_dir: Path to directory where artifacts will be written.
        dry_run: If True, prints planned outputs and returns without writing.
    """
    planned_artifacts = [
        output_dir / "fold_scores.parquet",
        output_dir / "model_summary.parquet",
        output_dir / "paired_comparisons.parquet",
        output_dir / "public_leaderboard.parquet",
        output_dir / "pairwise_confidence_intervals.parquet",
        output_dir / "pairwise_confidence_intervals.csv",
        output_dir / "pairwise_fold_differences.parquet",
        output_dir / "pairwise_fold_differences.csv",
        output_dir / "spearman_correlations.json",
        output_dir / "leaderboard_manifest.json",
    ]

    if dry_run:
        print("Dry run requested. Planned output artifacts:")
        for path in planned_artifacts:
            print(f"  - {path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth targets
    train_csv_path = project_root / "data" / "playground-series-s6e9" / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_csv_path}")

    train_df = pd.read_csv(train_csv_path, usecols=["Will_Buy_EV"])
    y_true = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()

    # 2. Load OOF frames & calculate sha256s
    oof_frames: dict[str, pd.DataFrame] = {}
    input_sha256: dict[str, str] = {}
    sources_rel: dict[str, str] = {}

    for name, rel_path_str in OOF_SOURCES.items():
        abs_path = project_root / rel_path_str
        if not abs_path.exists():
            raise FileNotFoundError(f"OOF source file missing for model '{name}': {abs_path}")
        oof_frames[name] = pd.read_parquet(abs_path)
        input_sha256[name] = _file_sha256(abs_path)
        sources_rel[name] = rel_path_str

    # 3. Derive ensemble candidate OOF
    ensemble_oof = blend_oof(
        frames={
            "catboost_hpo_trial22": oof_frames["catboost_hpo_trial22"],
            "lightgbm_hpo_trial35": oof_frames["lightgbm_hpo_trial35"],
        },
        weights=BLEND_WEIGHTS,
    )
    oof_frames[ENSEMBLE_NAME] = ensemble_oof
    sources_rel[ENSEMBLE_NAME] = "derived_via_blend_oof"
    ens_bytes = ensemble_oof["probability"].to_numpy().tobytes()
    input_sha256[ENSEMBLE_NAME] = hashlib.sha256(ens_bytes).hexdigest()

    # 4. Compute fold scores for each model
    fold_score_list: list[pd.DataFrame] = []
    per_model_fold_df: dict[str, pd.DataFrame] = {}

    for model_name, oof_df in oof_frames.items():
        fs_df = compute_fold_scores(oof_df, y_true)
        fs_df_with_model = fs_df.copy()
        fs_df_with_model.insert(0, "model", model_name)
        fold_score_list.append(fs_df_with_model)
        per_model_fold_df[model_name] = fs_df

    long_fold_scores = pd.concat(fold_score_list, ignore_index=True)
    long_fold_scores = long_fold_scores.sort_values(by=["model", "repeat", "fold"]).reset_index(
        drop=True
    )

    # 5. Compute model summary
    summary_records = []
    for model_name, fs_df in per_model_fold_df.items():
        scores = fs_df["roc_auc"].to_numpy(dtype=float)
        summary_records.append(
            {
                "model": model_name,
                # Mean of the 15 per-fold ROC-AUC values. This is a different
                # estimator from the pooled out-of-fold ROC-AUC recorded in
                # leaderboard.md; small disagreements between them are expected.
                "mean_fold_roc_auc": float(np.mean(scores)),
                "fold_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "n_folds": int(len(scores)),
            }
        )
    model_summary = pd.DataFrame(summary_records)
    model_summary = model_summary.sort_values(by="mean_fold_roc_auc", ascending=False).reset_index(
        drop=True
    )

    # 6. Compute paired comparisons for all unique pairs
    model_names = list(oof_frames.keys())
    paired_records = []
    model_mean_map = dict(
        zip(model_summary["model"], model_summary["mean_fold_roc_auc"], strict=True)
    )

    for m1, m2 in combinations(model_names, 2):
        mean1 = model_mean_map[m1]
        mean2 = model_mean_map[m2]
        if mean1 >= mean2:
            model_a, model_b = m1, m2
        else:
            model_a, model_b = m2, m1

        comp = paired_fold_comparison(
            fold_scores_a=per_model_fold_df[model_a],
            fold_scores_b=per_model_fold_df[model_b],
            label_a=model_a,
            label_b=model_b,
        )
        paired_records.append(comp)

    paired_df = pd.DataFrame(paired_records)
    paired_df = paired_df.sort_values(
        by=["mean_diff", "label_a", "label_b"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # 7. Load policy block from configs/final.yaml
    cfg_path = project_root / "configs" / "final.yaml"
    split_sha256 = "unknown"
    data_sha256 = "unknown"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            policy = cfg.get("policy", {})
            split_sha256 = policy.get("split_sha256", "unknown")
            data_sha256 = policy.get("data_sha256", "unknown")

    # 8. Compute pairwise repeated-CV confidence intervals (Nadeau-Bengio)
    pairwise_ci_df, pairwise_diff_df = compute_all_pairwise_cis(
        per_model_fold_df=per_model_fold_df,
        models=CANONICAL_LEADERBOARD_MODELS,
        test_train_ratio=0.25,
        split_sha256=split_sha256,
    )

    # 9. Parse and update public leaderboard markdown with Spearman and pairwise CI sections
    leaderboard_md_path = project_root / "leaderboard.md"
    if not leaderboard_md_path.exists():
        raise FileNotFoundError(f"leaderboard.md not found at {leaderboard_md_path}")

    leaderboard_md_content = leaderboard_md_path.read_text(encoding="utf-8")
    public_leaderboard_df = parse_public_leaderboard_markdown(leaderboard_md_content)

    recovered_models = load_and_validate_recovered_models(project_root=project_root)
    for m_name, m_rec in recovered_models.items():
        sources_rel[m_name] = m_rec["source_path"]
        input_sha256[m_name] = m_rec["source_sha256"]

    spearman_data = compute_dual_spearman_correlations(
        public_leaderboard_df=public_leaderboard_df,
        recovered_models=recovered_models,
    )
    spearman_section_md = format_spearman_correlation_markdown(spearman_data)

    # Insert / update Spearman section before pairwise CI section
    updated_with_spearman_md = update_managed_section(
        original_md=leaderboard_md_content,
        start_marker="<!-- SPEARMAN_CORRELATION_START -->",
        end_marker="<!-- SPEARMAN_CORRELATION_END -->",
        new_content=spearman_section_md,
        insert_before="<!-- PAIRWISE_CI_START -->",
    )

    pairwise_section_md = format_pairwise_ci_markdown(
        pairwise_ci_df=pairwise_ci_df,
        exclusions=EXCLUDED_LEADERBOARD_MODELS,
    )
    updated_leaderboard_md = update_leaderboard_markdown(
        updated_with_spearman_md, pairwise_section_md
    )
    leaderboard_md_path.write_text(updated_leaderboard_md, encoding="utf-8")
    leaderboard_md_sha256 = _file_sha256(leaderboard_md_path)

    # Save Parquet, CSV & JSON artifacts
    fold_scores_path = output_dir / "fold_scores.parquet"
    model_summary_path = output_dir / "model_summary.parquet"
    paired_comparisons_path = output_dir / "paired_comparisons.parquet"
    public_leaderboard_path = output_dir / "public_leaderboard.parquet"
    pairwise_ci_parquet_path = output_dir / "pairwise_confidence_intervals.parquet"
    pairwise_ci_csv_path = output_dir / "pairwise_confidence_intervals.csv"
    pairwise_diff_parquet_path = output_dir / "pairwise_fold_differences.parquet"
    pairwise_diff_csv_path = output_dir / "pairwise_fold_differences.csv"
    spearman_json_path = output_dir / "spearman_correlations.json"
    manifest_path = output_dir / "leaderboard_manifest.json"

    long_fold_scores.to_parquet(fold_scores_path, index=False)
    model_summary.to_parquet(model_summary_path, index=False)
    paired_df.to_parquet(paired_comparisons_path, index=False)
    public_leaderboard_df.to_parquet(public_leaderboard_path, index=False)
    pairwise_ci_df.to_parquet(pairwise_ci_parquet_path, index=False)
    pairwise_ci_df.to_csv(pairwise_ci_csv_path, index=False)
    pairwise_diff_df.to_parquet(pairwise_diff_parquet_path, index=False)
    pairwise_diff_df.to_csv(pairwise_diff_csv_path, index=False)
    spearman_json_path.write_text(json.dumps(spearman_data, indent=2), encoding="utf-8")

    row_counts = {
        "fold_scores.parquet": int(len(long_fold_scores)),
        "model_summary.parquet": int(len(model_summary)),
        "paired_comparisons.parquet": int(len(paired_df)),
        "public_leaderboard.parquet": int(len(public_leaderboard_df)),
        "pairwise_confidence_intervals.parquet": int(len(pairwise_ci_df)),
        "pairwise_confidence_intervals.csv": int(len(pairwise_ci_df)),
        "pairwise_fold_differences.parquet": int(len(pairwise_diff_df)),
        "pairwise_fold_differences.csv": int(len(pairwise_diff_df)),
        "spearman_correlations.json": 2,
    }

    artifact_sha256 = {
        "fold_scores.parquet": _file_sha256(fold_scores_path),
        "model_summary.parquet": _file_sha256(model_summary_path),
        "paired_comparisons.parquet": _file_sha256(paired_comparisons_path),
        "public_leaderboard.parquet": _file_sha256(public_leaderboard_path),
        "pairwise_confidence_intervals.parquet": _file_sha256(pairwise_ci_parquet_path),
        "pairwise_confidence_intervals.csv": _file_sha256(pairwise_ci_csv_path),
        "pairwise_fold_differences.parquet": _file_sha256(pairwise_diff_parquet_path),
        "pairwise_fold_differences.csv": _file_sha256(pairwise_diff_csv_path),
        "spearman_correlations.json": _file_sha256(spearman_json_path),
    }

    manifest_payload = {
        "sources": sources_rel,
        "input_sha256": input_sha256,
        "leaderboard_md_sha256": leaderboard_md_sha256,
        "row_counts": row_counts,
        "artifact_sha256": artifact_sha256,
        "blend_weights": BLEND_WEIGHTS,
        "split_sha256": split_sha256,
        "data_sha256": data_sha256,
    }

    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"Leaderboard artifacts successfully written to {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build leaderboard artifacts.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/leaderboard",
        help="Path to output directory for leaderboard artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs and exit without creating files.",
    )

    args = parser.parse_args(argv)
    project_root = Path.cwd()
    output_dir = Path(args.output_dir)

    build_leaderboard_artifacts(
        project_root=project_root,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
