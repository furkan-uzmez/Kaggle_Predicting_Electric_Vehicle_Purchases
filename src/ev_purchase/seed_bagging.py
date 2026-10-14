"""Reproducible multi-seed LightGBM bagging experiment harness."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from importlib import metadata as importlib_metadata
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ev_purchase.data import DatasetContract, encode_target
from ev_purchase.models import build_model
from ev_purchase.targeted_feature_experiment import (
    ExperimentDecision,
    calculate_experiment_metrics,
    compute_combined_sha256,
    compute_file_sha256,
    compute_params_hash,
    evaluate_frozen_slices,
    find_material_slice_regressions,
    verify_frozen_slices,
    verify_provenance_and_baseline,
)
from ev_purchase.targeted_feature_experiment import (
    make_gate_decision as _make_targeted_gate_decision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SEEDS: tuple[int, ...] = (42, 137, 2026)
FROZEN_SLICES_SHA256 = "3fbbf9e2a5050176466e8b7545069e6c4eccd88bb6c5570d8f1765f0ff50ebde"
EXPECTED_FROZEN_SLICE_COUNT = 22
SEED_ALIASES: tuple[str, ...] = (
    "random_state",
    "bagging_seed",
    "feature_fraction_seed",
    "data_random_seed",
    "extra_seed",
)
THREAT_MODEL_STATEMENT = (
    "Threat Model: SHA-256 checksums, atomic per-file writes, and exception-safe staged directory "
    "replacement provide local self-consistency evidence against accidental corruption, handled "
    "publication failures, stale artifacts, and configuration or seed drift. A process or power "
    "loss during the two-rename replacement window may require restoring the hidden backup. The "
    "replaceable local manifests do not provide cryptographic authenticity against a coordinated "
    "malicious actor who can rewrite artifacts and manifests together; that requires externally "
    "anchored digital signatures or an append-only trusted store."
)


def configure_seed_aliases(model_params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy of LightGBM parameters with every stochastic alias fixed."""
    configured = dict(model_params)
    configured.update(dict.fromkeys(SEED_ALIASES, int(seed)))
    return configured


def _atomic_write_text(path: Path, content: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.tmp_", delete=False
    )
    temporary = Path(handle.name)
    try:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".tmp"
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.tmp_", suffix=suffix, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported tabular artifact suffix: {path.suffix}")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _artifact_record(
    path: Path, record_count: int, published_path: Path | None = None
) -> dict[str, Any]:
    recorded_path = published_path or path
    return {
        "path": _relative_path(recorded_path),
        "file_name": recorded_path.name,
        "sha256": compute_file_sha256(path),
        "row_count": int(record_count),
        "record_count": int(record_count),
    }


def _validate_model_seeds(model_seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in model_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("model_seeds must contain at least two unique seeds")
    return seeds


def _validate_manifest(
    manifest: pd.DataFrame,
    n_rows: int,
    folds: list[tuple[int, int]] | None,
) -> list[tuple[tuple[int, int], pd.DataFrame]]:
    required = {"row_index", "repeat", "fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    if manifest[list(required)].isna().any().any():
        raise ValueError("manifest keys must not contain missing values")

    keys = manifest[["row_index", "repeat", "fold"]].astype(int)
    if not np.array_equal(keys.to_numpy(), manifest[["row_index", "repeat", "fold"]].to_numpy()):
        raise ValueError("manifest keys must be integer-valued")
    if ((keys["row_index"] < 0) | (keys["row_index"] >= n_rows)).any():
        raise ValueError("manifest row_index is outside the training frame")
    if keys.duplicated(["repeat", "row_index"]).any():
        raise ValueError("manifest assigns a row more than once within a repeat")

    groups = list(keys.groupby(["repeat", "fold"], sort=True))
    available = {tuple(map(int, group_key)) for group_key, _ in groups}
    if folds is None:
        expected = {(repeat, fold) for repeat in range(3) for fold in range(5)}
        complete_rows = n_rows * 3
        complete_coverage = all(
            set(keys.loc[keys["repeat"] == repeat, "row_index"]) == set(range(n_rows))
            for repeat in range(3)
        )
        if available != expected or len(keys) != complete_rows or not complete_coverage:
            raise ValueError("manifest does not satisfy the canonical 5x3 full-row protocol")
        return groups

    selected = {tuple(map(int, fold)) for fold in folds}
    unknown = selected - available
    if unknown:
        raise ValueError(f"Requested manifest folds do not exist: {sorted(unknown)}")
    return [group for group in groups if tuple(map(int, group[0])) in selected]


def _authenticate_canonical_component(
    component: dict[str, Any], metadata_path: Path
) -> tuple[dict[str, Any], str]:
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "candidate_id": "lightgbm_hpo_trial35",
        "model_family": "lightgbm",
        "trial_number": 35,
        "feature_policy": "base",
        "params": metadata.get("params"),
    }
    for field, expected_value in expected.items():
        if component.get(field) != expected_value:
            raise ValueError(f"Canonical {field} does not match candidate metadata")
    if metadata.get("run_id") != "lightgbm_hpo_trial35":
        raise ValueError("Canonical run_id does not match candidate metadata")
    if metadata.get("status") != "completed" or metadata.get("protocol") != "full":
        raise ValueError("Canonical candidate metadata is not a completed full run")
    if metadata.get("effective_n_splits") != 5 or metadata.get("effective_n_repeats") != 3:
        raise ValueError("Canonical candidate metadata is not a 5x3 evaluation")
    return metadata, compute_file_sha256(metadata_path)


def _verify_frozen_slices_contract(path: Path) -> tuple[dict[str, Any], list[str]]:
    metadata = verify_frozen_slices(path)
    if metadata["sha256"] != FROZEN_SLICES_SHA256:
        raise ValueError(
            f"Frozen slices hash mismatch: expected {FROZEN_SLICES_SHA256}, "
            f"got {metadata['sha256']}"
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    names = [
        *payload["predeclared_slices"],
        *(f"cohort__{name}" for name in payload["cohorts"]),
    ]
    if len(names) != EXPECTED_FROZEN_SLICE_COUNT or len(set(names)) != len(names):
        raise ValueError("Frozen slices contract does not define exactly 22 unique outputs")
    return metadata, names


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("lightgbm", "numpy", "pandas", "scikit-learn", "scipy", "PyYAML"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _publish_staged_run(staging_dir: Path, output_dir: Path) -> None:
    """Atomically replace a complete run directory, restoring the previous run on failure."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup_{time.time_ns()}")
    had_previous = output_dir.exists()
    try:
        if had_previous:
            os.replace(output_dir, backup_dir)
        os.replace(staging_dir, output_dir)
    except Exception:
        if had_previous and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _validate_complete_outputs(
    candidate_oof: pd.DataFrame,
    per_seed_oof: pd.DataFrame,
    baseline_oof: pd.DataFrame,
    metrics: dict[str, Any],
    slices: pd.DataFrame,
    expected_slice_names: Sequence[str],
    n_rows: int,
) -> None:
    keys = ["repeat", "fold", "row_index"]
    expected_rows = n_rows * 3
    if len(candidate_oof) != expected_rows or len(per_seed_oof) != expected_rows:
        raise ValueError(f"Expected {expected_rows} candidate and per-seed OOF rows")
    if len(baseline_oof) != expected_rows:
        raise ValueError(f"Expected {expected_rows} canonical baseline OOF rows")
    for name, frame in (
        ("candidate", candidate_oof),
        ("per-seed", per_seed_oof),
        ("baseline", baseline_oof),
    ):
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} OOF contains duplicate keys")
    candidate_keys = candidate_oof[keys].sort_values(keys).reset_index(drop=True)
    if not candidate_keys.equals(per_seed_oof[keys].sort_values(keys).reset_index(drop=True)):
        raise ValueError("Candidate and per-seed OOF keys are not aligned")
    if not candidate_keys.equals(baseline_oof[keys].sort_values(keys).reset_index(drop=True)):
        raise ValueError("Candidate and canonical baseline OOF keys are not aligned")

    seed_columns = [f"prob_seed_{seed}" for seed in MODEL_SEEDS]
    probabilities = per_seed_oof[seed_columns].to_numpy(dtype=float)
    bagged = candidate_oof["probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Per-seed OOF probabilities are invalid")
    if not np.array_equal(bagged, probabilities.mean(axis=1)):
        raise ValueError("Bagged OOF is not the exact row-wise mean of seed predictions")
    if len(metrics["fold_metrics"]) != 15 or len(metrics["repeat_metrics"]) != 3:
        raise ValueError("Experiment metrics do not cover exactly 15 folds and 3 repeats")
    if list(slices["slice_name"]) != list(expected_slice_names):
        raise ValueError("Frozen slice/cohort outputs do not match the exact canonical contract")


def train_seed_bagged_cv(
    train_df: pd.DataFrame,
    manifest: pd.DataFrame,
    contract: DatasetContract,
    model_params: dict[str, Any],
    model_seeds: Sequence[int] = MODEL_SEEDS,
    folds: list[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one LightGBM per seed and fold, returning bagged and per-seed OOF rows."""
    seeds = _validate_model_seeds(model_seeds)

    raw_target = train_df[contract.target_column]
    if set(raw_target.dropna().unique()).issubset({0, 1}):
        y_true = raw_target.to_numpy(dtype=int)
    else:
        y_true = encode_target(raw_target, contract)
    drop_columns = [contract.id_column, contract.target_column]
    predictors = train_df.drop(columns=[col for col in drop_columns if col in train_df.columns])
    all_indices = np.arange(len(train_df))
    groups = _validate_manifest(manifest, len(train_df), folds)

    bagged_records: list[pd.DataFrame] = []
    seed_records: list[pd.DataFrame] = []
    for (repeat, fold), fold_frame in groups:
        val_idx = fold_frame["row_index"].to_numpy(dtype=int)
        if len(val_idx) == 0:
            raise ValueError(f"Fold ({repeat}, {fold}) has no validation rows")
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        x_train = predictors.iloc[train_idx]
        x_valid = predictors.iloc[val_idx]
        seed_probabilities: dict[str, np.ndarray] = {}

        for seed in seeds:
            configured = configure_seed_aliases(model_params, seed)
            model = build_model(
                model_name="lightgbm",
                seed=seed,
                sample_frame=x_train,
                params=configured,
            )
            model.fit(x_train, y_true[train_idx])
            probabilities = np.asarray(model.predict_proba(x_valid)[:, 1], dtype=float)
            if not np.isfinite(probabilities).all() or np.any(
                (probabilities < 0.0) | (probabilities > 1.0)
            ):
                raise ValueError(f"Seed {seed} produced invalid probabilities")
            seed_probabilities[f"prob_seed_{seed}"] = probabilities

        per_seed_fold = pd.DataFrame(
            {
                "row_index": val_idx,
                "repeat": int(repeat),
                "fold": int(fold),
                **seed_probabilities,
            }
        )
        bagged_fold = per_seed_fold[["row_index", "repeat", "fold"]].copy()
        bagged_fold["probability"] = per_seed_fold[list(seed_probabilities)].mean(axis=1)
        seed_records.append(per_seed_fold)
        bagged_records.append(bagged_fold)

    candidate = pd.concat(bagged_records, ignore_index=True)
    per_seed = pd.concat(seed_records, ignore_index=True)
    order = ["repeat", "fold", "row_index"]
    return (
        candidate.sort_values(order, kind="stable").reset_index(drop=True),
        per_seed.sort_values(order, kind="stable").reset_index(drop=True),
    )


def calculate_bagging_diversity_metrics(
    y_true: np.ndarray,
    per_seed_df: pd.DataFrame,
    model_seeds: Sequence[int] = MODEL_SEEDS,
) -> dict[str, Any]:
    """Calculate per-seed quality and pairwise prediction diversity diagnostics."""
    seeds = _validate_model_seeds(model_seeds)
    columns = [f"prob_seed_{seed}" for seed in seeds]
    missing = set(columns) - set(per_seed_df.columns)
    if missing:
        raise ValueError(f"per_seed_df missing prediction columns: {sorted(missing)}")
    if "row_index" in per_seed_df:
        labels = np.asarray(y_true)[per_seed_df["row_index"].to_numpy(dtype=int)]
    else:
        labels = np.asarray(y_true)
    if len(labels) != len(per_seed_df):
        raise ValueError("y_true and per_seed_df are not aligned")

    predictions = per_seed_df[columns].to_numpy(dtype=float)
    pairwise_pearson: dict[str, float] = {}
    pairwise_spearman: dict[str, float] = {}
    pairwise_mae: dict[str, float] = {}
    for left, right in combinations(range(len(seeds)), 2):
        key = f"seed_{seeds[left]}__seed_{seeds[right]}"
        left_probs = predictions[:, left]
        right_probs = predictions[:, right]
        pairwise_pearson[key] = float(np.corrcoef(left_probs, right_probs)[0, 1])
        pairwise_spearman[key] = float(spearmanr(left_probs, right_probs).statistic)
        pairwise_mae[key] = float(np.mean(np.abs(left_probs - right_probs)))

    seed_metrics: dict[str, dict[str, float]] = {}
    for index, seed in enumerate(seeds):
        probabilities = predictions[:, index]
        seed_metrics[f"seed_{seed}"] = {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        }

    return {
        "pairwise_pearson": pairwise_pearson,
        "pairwise_spearman": pairwise_spearman,
        "pairwise_mae": pairwise_mae,
        "seed_metrics": seed_metrics,
        "mean_pairwise_pearson": float(np.mean(list(pairwise_pearson.values()))),
        "mean_pairwise_spearman": float(np.mean(list(pairwise_spearman.values()))),
        "mean_pairwise_mae": float(np.mean(list(pairwise_mae.values()))),
        "mean_prediction_std": float(np.mean(np.std(predictions, axis=1, ddof=0))),
        "max_prediction_std": float(np.max(np.std(predictions, axis=1, ddof=0))),
    }


def calculate_bagging_experiment_metrics(
    y_true: np.ndarray,
    oof_baseline: pd.DataFrame,
    oof_candidate: pd.DataFrame,
    per_seed_oof: pd.DataFrame,
    n_rows: int,
    n_repeats: int = 3,
    n_splits: int = 5,
    model_seeds: Sequence[int] = MODEL_SEEDS,
) -> dict[str, Any]:
    """Calculate aligned candidate metrics, corrected inference, and seed diversity."""
    metrics = calculate_experiment_metrics(
        y_true=y_true,
        oof_baseline=oof_baseline,
        oof_candidate=oof_candidate,
        n_rows=n_rows,
        n_repeats=n_repeats,
        n_splits=n_splits,
    )
    seed_columns = [f"prob_seed_{int(seed)}" for seed in model_seeds]
    per_seed_global = (
        per_seed_oof.groupby("row_index", sort=True)[seed_columns].mean().reset_index()
    )
    metrics["diversity"] = calculate_bagging_diversity_metrics(
        y_true=y_true,
        per_seed_df=per_seed_global,
        model_seeds=model_seeds,
    )
    return metrics


def make_gate_decision(
    global_auc_delta: float,
    repeat_auc_deltas: list[float],
    nb_ci_lower: float,
    global_log_loss_delta: float,
    slice_regressions: list[str],
) -> ExperimentDecision:
    """Apply the preregistered targeted-experiment gate unchanged."""
    return _make_targeted_gate_decision(
        global_auc_delta=global_auc_delta,
        repeat_auc_deltas=repeat_auc_deltas,
        nb_ci_lower=nb_ci_lower,
        global_log_loss_delta=global_log_loss_delta,
        slice_regressions=slice_regressions,
    )


def build_bagging_training_provenance(
    model_params: dict[str, Any],
    model_seeds: Sequence[int],
    data_hash: str,
    train_file_sha256: str,
    split_hash: str,
    config_hash: str,
    candidate_oof_path: Path,
    candidate_oof_sha: str,
    per_seed_oof_path: Path | None = None,
    per_seed_oof_sha: str | None = None,
    candidate_row_count: int | None = None,
    per_seed_row_count: int | None = None,
    training_source_hash: str | None = None,
) -> dict[str, Any]:
    """Build the immutable training provenance sidecar for a bagging run."""
    seeds = [int(seed) for seed in model_seeds]
    return {
        "provenance_version": "1.0",
        "experiment_name": "lightgbm_3seed",
        "baseline_candidate_id": "lightgbm_hpo_trial35",
        "feature_policy": "base",
        "model_family": "lightgbm",
        "model_params": model_params,
        "model_params_hash": compute_params_hash(model_params),
        "model_seeds": seeds,
        "seed_aliases": list(SEED_ALIASES),
        "seed_policy": "same three explicit model seeds on every persisted CV fold",
        "data_hash": data_hash,
        "train_file_sha256": train_file_sha256,
        "split_hash": split_hash,
        "config_hash": config_hash,
        "training_source_hash": training_source_hash,
        "candidate_oof": {
            "path": _relative_path(candidate_oof_path),
            "sha256": candidate_oof_sha,
            "row_count": candidate_row_count,
        },
        "per_seed_oof": {
            "path": _relative_path(per_seed_oof_path) if per_seed_oof_path else None,
            "sha256": per_seed_oof_sha,
            "row_count": per_seed_row_count,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "threat_model": THREAT_MODEL_STATEMENT,
    }


def generate_bagging_report_markdown(
    global_metrics: dict[str, Any],
    fold_metrics: list[dict[str, Any]],
    repeat_metrics: list[dict[str, Any]],
    nadeau_bengio: dict[str, Any],
    diversity: dict[str, Any],
    slice_df: pd.DataFrame,
    decision: ExperimentDecision,
) -> str:
    """Render a deterministic human-readable report from experiment outputs."""
    lines = [
        "# LightGBM Three-Seed Bagging Experiment",
        "",
        "## Decision",
        f"- **Gate:** `{decision.decision}`",
        f"- **Reason:** {decision.reason}",
        "- **Baseline:** `lightgbm_hpo_trial35`",
        f"- **Model seeds:** `{list(MODEL_SEEDS)}`",
        "- **Feature policy:** unchanged canonical `base` features",
        "",
        "## Global Metrics",
        "| Metric | Baseline | Bagged | Delta |",
        "| :--- | ---: | ---: | ---: |",
        f"| ROC-AUC | {global_metrics['baseline_roc_auc']:.9f} | "
        f"{global_metrics['candidate_roc_auc']:.9f} | {global_metrics['delta_roc_auc']:+.9f} |",
        f"| PR-AUC | {global_metrics['baseline_pr_auc']:.9f} | "
        f"{global_metrics['candidate_pr_auc']:.9f} | {global_metrics['delta_pr_auc']:+.9f} |",
        f"| LogLoss | {global_metrics['baseline_log_loss']:.9f} | "
        f"{global_metrics['candidate_log_loss']:.9f} | "
        f"{global_metrics['delta_log_loss']:+.9f} |",
        "",
        "## Repeat Deltas",
        "| Repeat | Delta ROC-AUC | Delta PR-AUC | Delta LogLoss |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in repeat_metrics:
        lines.append(
            f"| {row['repeat']} | {row['delta_roc_auc']:+.9f} | "
            f"{row['delta_pr_auc']:+.9f} | {row['delta_log_loss']:+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Corrected Statistical Comparison",
            f"- Mean fold AUC delta: {nadeau_bengio['mean_difference']:+.9f}",
            f"- Nadeau-Bengio 95% CI: [{nadeau_bengio['ci_nb_lower']:+.9f}, "
            f"{nadeau_bengio['ci_nb_upper']:+.9f}]",
            f"- Corrected p-value: {nadeau_bengio['p_value_nb_raw']:.8g}",
            "",
            "## Seed Diversity",
            f"- Mean pairwise Pearson correlation: {diversity['mean_pairwise_pearson']:.9f}",
            f"- Mean pairwise Spearman correlation: {diversity['mean_pairwise_spearman']:.9f}",
            f"- Mean pairwise probability MAE: {diversity['mean_pairwise_mae']:.9f}",
            f"- Mean row-wise seed standard deviation: {diversity['mean_prediction_std']:.9f}",
            "",
            "## Fold Metrics",
            "| Repeat | Fold | Baseline AUC | Bagged AUC | Delta AUC |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_metrics:
        lines.append(
            f"| {row['repeat']} | {row['fold']} | {row['baseline_roc_auc']:.9f} | "
            f"{row['candidate_roc_auc']:.9f} | {row['delta_roc_auc']:+.9f} |"
        )
    lines.extend(
        [
            "",
            f"## Frozen Slices and Cohorts ({len(slice_df)})",
            "| Slice | n | Positives | Delta AUC | Delta LogLoss |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in slice_df.iterrows():
        delta_auc = "N/A" if pd.isna(row["delta_auc"]) else f"{row['delta_auc']:+.9f}"
        lines.append(
            f"| `{row['slice_name']}` | {int(row['n'])} | {int(row['positives'])} | "
            f"{delta_auc} | {row['delta_log_loss']:+.9f} |"
        )
    lines.extend(["", "## Integrity Threat Model", THREAT_MODEL_STATEMENT, ""])
    return "\n".join(lines)


def _load_canonical_contract(final_config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with final_config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    component = config["policy"]["components"]["lightgbm_hpo_trial35"]
    if component.get("feature_policy") != "base":
        raise ValueError("Canonical LightGBM feature policy is not 'base'")
    return config, component


def _smoke_summary(
    train_df: pd.DataFrame,
    baseline_oof: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    contract: DatasetContract,
) -> dict[str, Any]:
    y_true = encode_target(train_df[contract.target_column], contract)
    keys = ["repeat", "fold", "row_index"]
    baseline_fold = baseline_oof.loc[
        (baseline_oof["repeat"] == 0) & (baseline_oof["fold"] == 0)
    ].copy()
    if baseline_fold.duplicated(keys).any() or candidate_oof.duplicated(keys).any():
        raise ValueError("Smoke OOF inputs contain duplicate alignment keys")
    baseline_keys = baseline_fold[keys].sort_values(keys).reset_index(drop=True)
    candidate_keys = candidate_oof[keys].sort_values(keys).reset_index(drop=True)
    if not baseline_keys.equals(candidate_keys):
        raise ValueError("Smoke OOF inputs do not have exact one-to-one row alignment")
    aligned = baseline_fold.merge(
        candidate_oof,
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    labels = y_true[aligned["row_index"].to_numpy(dtype=int)]
    return {
        "smoke": True,
        "fold": [0, 0],
        "val_rows": len(aligned),
        "baseline_auc": float(roc_auc_score(labels, aligned["probability_baseline"])),
        "candidate_auc": float(roc_auc_score(labels, aligned["probability_candidate"])),
    }


def run_seed_bagging_experiment(
    train_path: Path | str,
    test_path: Path | str,
    manifest_path: Path | str,
    baseline_oof_path: Path | str,
    final_config_path: Path | str,
    frozen_slices_path: Path | str,
    output_dir: Path | str | None = None,
    smoke: bool = False,
    generation_command: str | None = None,
) -> dict[str, Any]:
    """Run the canonical 5x3 three-seed bagging comparison and persist its evidence."""
    started = time.perf_counter()
    train_path = Path(train_path).resolve()
    test_path = Path(test_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    baseline_oof_path = Path(baseline_oof_path).resolve()
    final_config_path = Path(final_config_path).resolve()
    frozen_slices_path = Path(frozen_slices_path).resolve()
    out_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else PROJECT_ROOT / "outputs" / "experiments" / "seed_bagging" / "lightgbm_3seed"
    )

    _, component = _load_canonical_contract(final_config_path)
    metadata_path = PROJECT_ROOT / component["candidate_metadata_path"]
    baseline_metadata, metadata_hash = _authenticate_canonical_component(component, metadata_path)
    expected_oof_hash = baseline_metadata["artifact_checksums"]["oof_predictions"]
    data_hash = compute_combined_sha256([train_path, test_path])
    train_hash = compute_file_sha256(train_path)
    test_hash = compute_file_sha256(test_path)
    split_hash = compute_file_sha256(manifest_path)
    config_hash = compute_file_sha256(final_config_path)
    baseline_hash = compute_file_sha256(baseline_oof_path)
    verify_provenance_and_baseline(
        component["provenance"]["data_sha256"],
        data_hash,
        component["provenance"]["split_sha256"],
        split_hash,
        expected_oof_hash,
        baseline_hash,
    )
    frozen_meta, expected_slice_names = _verify_frozen_slices_contract(frozen_slices_path)

    train_df = pd.read_csv(train_path)
    manifest = pd.read_parquet(manifest_path)
    baseline_oof = pd.read_parquet(baseline_oof_path)
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    folds = [(0, 0)] if smoke else None
    candidate_oof, per_seed_oof = train_seed_bagged_cv(
        train_df=train_df,
        manifest=manifest,
        contract=contract,
        model_params=component["params"],
        model_seeds=MODEL_SEEDS,
        folds=folds,
    )
    if smoke:
        result = _smoke_summary(train_df, baseline_oof, candidate_oof, contract)
        result["candidate_auc_delta"] = result["candidate_auc"] - result["baseline_auc"]
        result["runtime_seconds"] = time.perf_counter() - started
        return result

    y_true = encode_target(train_df[contract.target_column], contract)
    metrics = calculate_bagging_experiment_metrics(
        y_true=y_true,
        oof_baseline=baseline_oof,
        oof_candidate=candidate_oof,
        per_seed_oof=per_seed_oof,
        n_rows=len(train_df),
    )
    slices = evaluate_frozen_slices(
        train_df=train_df,
        y_true=y_true,
        p_baseline=metrics["p_baseline_global"],
        p_candidate=metrics["p_candidate_global"],
        frozen_slices_path=frozen_slices_path,
    )
    regressions = find_material_slice_regressions(slices)
    decision = make_gate_decision(
        global_auc_delta=metrics["global_metrics"]["delta_roc_auc"],
        repeat_auc_deltas=metrics["global_metrics"]["repeat_auc_deltas"],
        nb_ci_lower=metrics["nadeau_bengio"]["ci_nb_lower"],
        global_log_loss_delta=metrics["global_metrics"]["delta_log_loss"],
        slice_regressions=regressions,
    )
    report = generate_bagging_report_markdown(
        global_metrics=metrics["global_metrics"],
        fold_metrics=metrics["fold_metrics"],
        repeat_metrics=metrics["repeat_metrics"],
        nadeau_bengio=metrics["nadeau_bengio"],
        diversity=metrics["diversity"],
        slice_df=slices,
        decision=decision,
    )
    _validate_complete_outputs(
        candidate_oof=candidate_oof,
        per_seed_oof=per_seed_oof,
        baseline_oof=baseline_oof,
        metrics=metrics,
        slices=slices,
        expected_slice_names=expected_slice_names,
        n_rows=len(train_df),
    )

    runtime_seconds = time.perf_counter() - started
    observed_folds = sorted(
        [list(map(int, values)) for values in candidate_oof.groupby(["repeat", "fold"]).groups]
    )
    fold_count = len(observed_folds)
    model_count = fold_count * len(MODEL_SEEDS)
    source_path = Path(__file__).resolve()
    source_hash = compute_file_sha256(source_path)
    runner_path = PROJECT_ROOT / "scripts" / "run_seed_bagging_experiment.py"
    models_path = PROJECT_ROOT / "src" / "ev_purchase" / "models.py"
    evaluation_path = PROJECT_ROOT / "src" / "ev_purchase" / "targeted_feature_experiment.py"

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging_", dir=out_dir.parent)
    ).resolve()
    try:
        staged = {
            name: staging_dir / name
            for name in (
                "candidate_oof.parquet",
                "per_seed_oof.parquet",
                "training_provenance.json",
                "training_manifest.json",
                "fold_metrics.csv",
                "repeat_metrics.csv",
                "global_metrics.json",
                "diversity_metrics.json",
                "slice_metrics.csv",
                "report.md",
            )
        }
        published = {name: out_dir / name for name in staged}
        _atomic_write_frame(staged["candidate_oof.parquet"], candidate_oof)
        _atomic_write_frame(staged["per_seed_oof.parquet"], per_seed_oof)

        training_provenance = build_bagging_training_provenance(
            model_params=component["params"],
            model_seeds=MODEL_SEEDS,
            data_hash=data_hash,
            train_file_sha256=train_hash,
            split_hash=split_hash,
            config_hash=config_hash,
            candidate_oof_path=published["candidate_oof.parquet"],
            candidate_oof_sha=compute_file_sha256(staged["candidate_oof.parquet"]),
            per_seed_oof_path=published["per_seed_oof.parquet"],
            per_seed_oof_sha=compute_file_sha256(staged["per_seed_oof.parquet"]),
            candidate_row_count=len(candidate_oof),
            per_seed_row_count=len(per_seed_oof),
            training_source_hash=source_hash,
        )
        training_provenance.update(
            {
                "test_file_sha256": test_hash,
                "candidate_metadata_sha256": metadata_hash,
                "candidate_metadata_path": _relative_path(metadata_path),
                "observed_folds": observed_folds,
                "package_versions": _package_versions(),
                "git": _git_state(),
                "source_hashes": {
                    "seed_bagging": source_hash,
                    "models": compute_file_sha256(models_path),
                    "targeted_evaluation": compute_file_sha256(evaluation_path),
                    "runner": compute_file_sha256(runner_path),
                },
            }
        )
        _atomic_write_json(staged["training_provenance.json"], training_provenance)
        training_artifacts = {
            "candidate_oof": _artifact_record(
                staged["candidate_oof.parquet"],
                len(candidate_oof),
                published["candidate_oof.parquet"],
            ),
            "per_seed_oof": _artifact_record(
                staged["per_seed_oof.parquet"], len(per_seed_oof), published["per_seed_oof.parquet"]
            ),
            "training_provenance": _artifact_record(
                staged["training_provenance.json"],
                len(training_provenance),
                published["training_provenance.json"],
            ),
        }
        training_manifest = {
            "operation": "training",
            "status": "completed",
            "experiment_name": "lightgbm_3seed",
            "baseline_candidate_id": "lightgbm_hpo_trial35",
            "feature_policy": "base",
            "model_seeds": list(MODEL_SEEDS),
            "seed_aliases": list(SEED_ALIASES),
            "model_count": model_count,
            "fold_count": fold_count,
            "provenance": training_provenance,
            "artifacts": training_artifacts,
            "threat_model": THREAT_MODEL_STATEMENT,
        }
        _atomic_write_json(staged["training_manifest.json"], training_manifest)
        _atomic_write_frame(staged["fold_metrics.csv"], pd.DataFrame(metrics["fold_metrics"]))
        _atomic_write_frame(staged["repeat_metrics.csv"], pd.DataFrame(metrics["repeat_metrics"]))
        _atomic_write_json(
            staged["global_metrics.json"],
            {
                "global_metrics": metrics["global_metrics"],
                "nadeau_bengio": metrics["nadeau_bengio"],
            },
        )
        _atomic_write_json(staged["diversity_metrics.json"], metrics["diversity"])
        _atomic_write_frame(staged["slice_metrics.csv"], slices)
        _atomic_write_text(staged["report.md"], report)

        artifacts = {
            **training_artifacts,
            "training_manifest": _artifact_record(
                staged["training_manifest.json"],
                len(training_manifest),
                published["training_manifest.json"],
            ),
            "fold_metrics": _artifact_record(
                staged["fold_metrics.csv"],
                len(metrics["fold_metrics"]),
                published["fold_metrics.csv"],
            ),
            "repeat_metrics": _artifact_record(
                staged["repeat_metrics.csv"],
                len(metrics["repeat_metrics"]),
                published["repeat_metrics.csv"],
            ),
            "global_metrics": _artifact_record(
                staged["global_metrics.json"], 2, published["global_metrics.json"]
            ),
            "diversity_metrics": _artifact_record(
                staged["diversity_metrics.json"],
                len(metrics["diversity"]),
                published["diversity_metrics.json"],
            ),
            "slice_metrics": _artifact_record(
                staged["slice_metrics.csv"], len(slices), published["slice_metrics.csv"]
            ),
            "report": _artifact_record(
                staged["report.md"], len(report.splitlines()), published["report.md"]
            ),
        }
        evaluation_manifest = {
            "operation": "evaluation",
            "status": "completed",
            "publication": "exception_safe_staged_directory_replace",
            "experiment_name": "lightgbm_3seed",
            "baseline_candidate_id": "lightgbm_hpo_trial35",
            "feature_policy": "base",
            "model_seeds": list(MODEL_SEEDS),
            "seed_aliases": list(SEED_ALIASES),
            "model_count": model_count,
            "fold_count": fold_count,
            "runtime_seconds": runtime_seconds,
            "frozen_slices": {**frozen_meta, "evaluated_count": len(slices)},
            "gate_decision": {
                "decision": decision.decision,
                "reason": decision.reason,
                "details": decision.details,
            },
            "summary_metrics": {
                **metrics["global_metrics"],
                "nadeau_bengio": metrics["nadeau_bengio"],
                "diversity": metrics["diversity"],
            },
            "provenance": {
                "generation_command": generation_command or " ".join(sys.argv),
                "generation_argv": list(sys.argv),
                "data_sha256": data_hash,
                "train_file_sha256": train_hash,
                "test_file_sha256": test_hash,
                "split_manifest_sha256": split_hash,
                "baseline_oof_sha256": baseline_hash,
                "candidate_metadata_sha256": metadata_hash,
                "config_sha256": config_hash,
                "frozen_slices_sha256": frozen_meta["sha256"],
                "source_hashes": training_provenance["source_hashes"],
                "environment": training_provenance["environment"],
                "package_versions": training_provenance["package_versions"],
                "git": training_provenance["git"],
            },
            "artifacts": artifacts,
            "threat_model": THREAT_MODEL_STATEMENT,
        }
        _atomic_write_json(staging_dir / "evaluation_manifest.json", evaluation_manifest)
        for record in artifacts.values():
            staged_path = staging_dir / record["file_name"]
            if compute_file_sha256(staged_path) != record["sha256"]:
                raise ValueError(f"Staged artifact hash mismatch: {record['file_name']}")
        _publish_staged_run(staging_dir, out_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return {
        "decision": decision,
        "metrics": metrics,
        "slice_df": slices,
        "manifest": evaluation_manifest,
        "runtime_seconds": runtime_seconds,
        "output_dir": out_dir,
    }


__all__ = [
    "MODEL_SEEDS",
    "SEED_ALIASES",
    "THREAT_MODEL_STATEMENT",
    "build_bagging_training_provenance",
    "calculate_bagging_diversity_metrics",
    "calculate_bagging_experiment_metrics",
    "configure_seed_aliases",
    "generate_bagging_report_markdown",
    "make_gate_decision",
    "run_seed_bagging_experiment",
    "train_seed_bagged_cv",
]
