"""Fail-closed evaluation of FT-Transformer against the canonical LightGBM model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ev_purchase.data import DatasetContract, encode_target, validate_frames
from ev_purchase.targeted_feature_experiment import (
    calculate_experiment_metrics,
    evaluate_frozen_slices,
    find_material_slice_regressions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FT_ARTIFACT_DIR = PROJECT_ROOT / (
    "outputs/experiments/ft_transformer/ft_transformer_5x3/results/ft_transformer_5x3"
)
CANONICAL_DIR = PROJECT_ROOT / "outputs/ensemble/candidates/lightgbm_hpo_trial35"
TRAIN_PATH = PROJECT_ROOT / "data/playground-series-s6e9/train.csv"
TEST_PATH = PROJECT_ROOT / "data/playground-series-s6e9/test.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "data/playground-series-s6e9/sample_submission.csv"
SPLIT_PATH = PROJECT_ROOT / "outputs/splits/repeated_stratified_5x3.parquet"
FROZEN_SLICES_PATH = PROJECT_ROOT / "outputs/diagnostics/severe_fn_audit/frozen_slices.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/ft_transformer/evaluation"

EXPECTED_HASHES = {
    "train": "eae9eaa4e6378df405e755f853771d7e26d212bd93258349fc797b771021946a",
    "test": "539263f6caabc40afd5e2f0bc0ab16b10a2d1177c565fc71b866f0181d836b34",
    "sample_submission": "a9747a8b947e4e35505e3da4535a5a494978b012a7adb50973f13e598849dda5",
    "split_file": "b554570edc0bbcd20436e37d6aa4c9cad0dfc8242d3beef7815db32a93a8adee",
    "split_content": "5a6a1965ff57c5214035bc4e01ae51e306419b9c05aa04cae8faad9d53fd2206",
    "canonical_oof": "ef173826f69ccfc98ac2efb71e4bd0b7b0c84268be69932d1934f1a76552f8ef",
    "canonical_test": "580e61fc06663ff58e362c850001e0564114278a373b37cfb2048a8d0ab4bb0b",
    "frozen_slices": "3fbbf9e2a5050176466e8b7545069e6c4eccd88bb6c5570d8f1765f0ff50ebde",
    "ft_inventory": "34e5a36fbc56f131454547cc147cd80f83c7ac0fe4a7ffe2dad1bb5d5b5f6c1f",
}
EXPECTED_FT_CONFIG_SHA256 = "28fa358d8486b4f25e689921aff5af1bb301006513187f65f7a52e5c87a27f20"
EXPECTED_FT_SOURCE_SHA256 = "226274c391e867106ade6371268e50cccfbff6009a445738ba1a2c03ae0f6a75"
EXPECTED_FT_FINGERPRINT_SHA256 = "c23d12c3ff33d450375e1257819765901375583cb2670a2b1f0137458430ae51"
EXPECTED_COMBINED_DATA_SHA256 = "169e0aee03673611f12f2f24afba67399e0038279c049a15a5db67e952afee46"
EXPECTED_FT_SUPPLEMENTAL_FILES = {
    "kaggle-ev-purchase-ft-transformer.ipynb": (
        "2e727d1ea830b559b83da859a5baddf810735281d504eee5e6c4d3d75f258b9c"
    )
}
BLEND_WEIGHTS: tuple[float, ...] = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    1.0,
)
KEY_COLUMNS = ["repeat", "fold", "row_index"]


def _expected_ft_artifact_paths() -> set[str]:
    paths = {
        "fold_metrics.csv",
        "global_metrics.json",
        "oof_predictions.csv",
        "repeat_metrics.csv",
        "run_state.json",
        "submission.csv",
        "test_predictions.csv",
        "training_manifest.json",
    }
    for repeat in range(3):
        for fold in range(5):
            paths.update(
                {
                    f"checkpoints/fold_r{repeat}_f{fold}.pt",
                    f"checkpoints/preprocess_r{repeat}_f{fold}.pkl",
                    f"fold_metric_r{repeat}_f{fold}.json",
                    f"fold_oof_r{repeat}_f{fold}.csv",
                    f"fold_test_r{repeat}_f{fold}.csv",
                }
            )
    return paths


EXPECTED_FT_ARTIFACT_PATHS = _expected_ft_artifact_paths()


@dataclass(frozen=True)
class BlendGridResult:
    """Metrics and aligned row-level predictions for a fixed blend grid."""

    metrics: pd.DataFrame
    predictions: dict[float, np.ndarray]
    best_weight: float


def file_sha256(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def split_content_sha256(manifest: pd.DataFrame) -> str:
    """Hash canonical split assignments independently of parquet metadata."""
    columns = ["row_index", "repeat", "fold"]
    if list(manifest.columns) != columns:
        missing = set(columns) - set(manifest.columns)
        if missing:
            raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    array = manifest[columns].to_numpy(dtype="<i8", copy=True)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _count_records(path: Path) -> int | None:
    if path.suffix == ".csv":
        return sum(len(chunk) for chunk in pd.read_csv(path, chunksize=250_000))
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return len(payload)
    return None


def verify_artifact_inventory(
    artifact_dir: Path | str,
    expected_count: int = 83,
    expected_paths: set[str] | None = None,
    expected_inventory_sha256: str | None = None,
    approved_supplemental_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify every declared FT artifact without deserializing checkpoints or pickle files."""
    root = Path(artifact_dir).resolve()
    inventory_path = root / "artifact_inventory.json"
    if not inventory_path.is_file():
        raise ValueError(f"Missing FT artifact inventory: {inventory_path}")
    observed_inventory_sha256 = file_sha256(inventory_path)
    if (
        expected_inventory_sha256 is not None
        and observed_inventory_sha256 != expected_inventory_sha256
    ):
        raise ValueError(
            f"FT artifact inventory SHA-256 mismatch: expected {expected_inventory_sha256}, "
            f"got {observed_inventory_sha256}"
        )
    with inventory_path.open(encoding="utf-8") as handle:
        inventory = json.load(handle)
    artifacts = inventory.get("artifacts")
    if inventory.get("status") != "completed" or not isinstance(artifacts, list):
        raise ValueError("FT artifact inventory is not completed or has no artifact list")
    if inventory.get("artifact_count") != len(artifacts) or len(artifacts) != expected_count:
        raise ValueError("FT artifact inventory count mismatch")

    seen: set[str] = set()
    total_bytes = 0
    for record in artifacts:
        relative = record.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError(f"Invalid or duplicate artifact path: {relative!r}")
        lowered_parts = {part.lower() for part in Path(relative).parts}
        lowered_name = Path(relative).name.lower()
        if lowered_parts & {"runs", "development", "dev", "kaggle_oof"} or (
            "kaggle" in lowered_name and "oof" in lowered_name
        ):
            raise ValueError(f"FT inventory contains prohibited artifact path: {relative}")
        seen.add(relative)
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Artifact path escapes artifact root: {relative}")
        if not path.is_file():
            raise ValueError(f"Missing FT artifact: {relative}")
        if (root / relative).is_symlink():
            raise ValueError(f"FT artifact must not be a symlink: {relative}")
        byte_count = path.stat().st_size
        if byte_count != record.get("byte_count"):
            raise ValueError(f"Byte-count mismatch for FT artifact: {relative}")
        observed_hash = file_sha256(path)
        if observed_hash != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for FT artifact: {relative}")
        expected_rows = record.get("row_count")
        if expected_rows is not None and _count_records(path) != expected_rows:
            raise ValueError(f"Row-count mismatch for FT artifact: {relative}")
        total_bytes += byte_count
    if expected_paths is not None and seen != expected_paths:
        missing = sorted(expected_paths - seen)
        unexpected = sorted(seen - expected_paths)
        raise ValueError(
            f"FT artifact path set mismatch; missing={missing}, unexpected={unexpected}"
        )
    supplemental_files = approved_supplemental_files or {}
    for relative, expected_hash in supplemental_files.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or invalid approved FT supplemental file: {relative}")
        if file_sha256(path) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for approved FT supplemental file: {relative}")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    undeclared = sorted(actual_files - seen - set(supplemental_files) - {"artifact_inventory.json"})
    if undeclared:
        raise ValueError(
            f"FT artifact directory contains undeclared physical artifacts: {undeclared}"
        )
    return {
        "status": "verified",
        "artifact_count": len(artifacts),
        "total_bytes": total_bytes,
        "inventory_sha256": observed_inventory_sha256,
    }


def verify_run_fingerprint(
    training_manifest: dict[str, Any],
    run_state: dict[str, Any],
    expected_fingerprint_sha256: str | None = None,
) -> str:
    """Authenticate equal run fingerprints and their canonical JSON digest."""
    left = training_manifest.get("run_fingerprint")
    right = run_state.get("run_fingerprint")
    if not isinstance(left, dict) or left != right:
        raise ValueError("Training manifest and run state fingerprints do not match")
    expected_digest = _json_sha256(left)
    for name, payload in (("training manifest", training_manifest), ("run state", run_state)):
        if payload.get("run_fingerprint_sha256") != expected_digest:
            raise ValueError(f"{name} fingerprint digest mismatch")
    if expected_fingerprint_sha256 is not None and expected_digest != expected_fingerprint_sha256:
        raise ValueError(
            f"Run fingerprint digest mismatch: expected {expected_fingerprint_sha256}, "
            f"got {expected_digest}"
        )
    return expected_digest


def verify_ft_provenance(
    training_manifest: dict[str, Any],
    run_state: dict[str, Any],
    expected_config_sha256: str = EXPECTED_FT_CONFIG_SHA256,
    expected_source_sha256: str = EXPECTED_FT_SOURCE_SHA256,
    expected_fingerprint_sha256: str = EXPECTED_FT_FINGERPRINT_SHA256,
    expected_split_content_sha256: str = EXPECTED_HASHES["split_content"],
) -> str:
    """Authenticate exact FT config, source, and fingerprint provenance individually."""
    manifest_config = training_manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise ValueError("Training manifest missing config mapping")
    if training_manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            f"Training manifest config_sha256 mismatch: expected {expected_config_sha256}, "
            f"got {training_manifest.get('config_sha256')}"
        )
    observed_config_sha = _json_sha256(manifest_config)
    if observed_config_sha != expected_config_sha256:
        raise ValueError(
            "Training manifest config canonical digest mismatch: "
            f"expected {expected_config_sha256}, got {observed_config_sha}"
        )

    if training_manifest.get("source_sha256") != expected_source_sha256:
        raise ValueError(
            f"Training manifest source_sha256 mismatch: expected {expected_source_sha256}, "
            f"got {training_manifest.get('source_sha256')}"
        )

    if training_manifest.get("split_content_sha256") != expected_split_content_sha256:
        raise ValueError(
            "Training manifest split_content_sha256 mismatch: "
            f"expected {expected_split_content_sha256}, "
            f"got {training_manifest.get('split_content_sha256')}"
        )

    if training_manifest.get("run_fingerprint_sha256") != expected_fingerprint_sha256:
        raise ValueError(
            "Training manifest run_fingerprint_sha256 mismatch: "
            f"expected {expected_fingerprint_sha256}, "
            f"got {training_manifest.get('run_fingerprint_sha256')}"
        )
    if run_state.get("run_fingerprint_sha256") != expected_fingerprint_sha256:
        raise ValueError(
            "Run state run_fingerprint_sha256 mismatch: "
            f"expected {expected_fingerprint_sha256}, "
            f"got {run_state.get('run_fingerprint_sha256')}"
        )

    left_fp = training_manifest.get("run_fingerprint")
    right_fp = run_state.get("run_fingerprint")
    if not isinstance(left_fp, dict):
        raise ValueError("Training manifest missing run_fingerprint mapping")
    if not isinstance(right_fp, dict):
        raise ValueError("Run state missing run_fingerprint mapping")

    if left_fp.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            "Training manifest fingerprint config_sha256 mismatch: "
            f"expected {expected_config_sha256}, got {left_fp.get('config_sha256')}"
        )
    if right_fp.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            "Run state fingerprint config_sha256 mismatch: "
            f"expected {expected_config_sha256}, got {right_fp.get('config_sha256')}"
        )

    if left_fp.get("full_notebook_source_sha256") != expected_source_sha256:
        raise ValueError(
            f"Training manifest fingerprint source mismatch: expected {expected_source_sha256}, "
            f"got {left_fp.get('full_notebook_source_sha256')}"
        )
    if right_fp.get("full_notebook_source_sha256") != expected_source_sha256:
        raise ValueError(
            f"Run state fingerprint source mismatch: expected {expected_source_sha256}, "
            f"got {right_fp.get('full_notebook_source_sha256')}"
        )

    if left_fp.get("split_content_sha256") != expected_split_content_sha256:
        raise ValueError(
            "Training manifest fingerprint split_content_sha256 mismatch: "
            f"expected {expected_split_content_sha256}, got {left_fp.get('split_content_sha256')}"
        )
    if right_fp.get("split_content_sha256") != expected_split_content_sha256:
        raise ValueError(
            "Run state fingerprint split_content_sha256 mismatch: "
            f"expected {expected_split_content_sha256}, got {right_fp.get('split_content_sha256')}"
        )

    return verify_run_fingerprint(
        training_manifest, run_state, expected_fingerprint_sha256=expected_fingerprint_sha256
    )


def _validate_probabilities(frame: pd.DataFrame, name: str) -> None:
    probability = frame["probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"{name} contains invalid probabilities")


def validate_oof_alignment(
    ft_oof: pd.DataFrame,
    baseline_oof: pd.DataFrame,
    split_manifest: pd.DataFrame,
    y_true: np.ndarray,
    n_repeats: int = 3,
    n_splits: int = 5,
) -> None:
    """Fail if FT, canonical, split, or target rows differ in any key or value."""
    ft_required = {*KEY_COLUMNS, "probability", "target"}
    base_required = {*KEY_COLUMNS, "probability"}
    if missing := ft_required - set(ft_oof.columns):
        raise ValueError(f"FT OOF missing columns: {sorted(missing)}")
    if missing := base_required - set(baseline_oof.columns):
        raise ValueError(f"Canonical OOF missing columns: {sorted(missing)}")
    if missing := set(KEY_COLUMNS) - set(split_manifest.columns):
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    expected_rows = len(y_true) * n_repeats
    if any(len(frame) != expected_rows for frame in (ft_oof, baseline_oof, split_manifest)):
        raise ValueError(f"OOF row count must equal {expected_rows}")

    sorted_frames = []
    for name, frame in (
        ("FT", ft_oof),
        ("canonical", baseline_oof),
        ("split", split_manifest),
    ):
        keys = frame[KEY_COLUMNS]
        if keys.isna().any().any() or keys.duplicated().any():
            raise ValueError(f"{name} OOF keys contain nulls or duplicates")
        sorted_frames.append(frame.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True))
    ft_sorted, base_sorted, split_sorted = sorted_frames
    canonical_keys = base_sorted[KEY_COLUMNS]
    if not canonical_keys.equals(ft_sorted[KEY_COLUMNS]) or not canonical_keys.equals(
        split_sorted[KEY_COLUMNS]
    ):
        raise ValueError("FT, canonical, and split OOF keys do not align")
    if sorted(ft_sorted["repeat"].unique()) != list(range(n_repeats)):
        raise ValueError("FT repeats do not match the canonical protocol")
    if sorted(ft_sorted["fold"].unique()) != list(range(n_splits)):
        raise ValueError("FT folds do not match the canonical protocol")
    expected_row_index = np.arange(len(y_true))
    for repeat, repeat_rows in ft_sorted.groupby("repeat", sort=True):
        observed = np.sort(repeat_rows["row_index"].to_numpy(dtype=int))
        if not np.array_equal(observed, expected_row_index):
            raise ValueError(f"FT repeat {repeat} must contain each physical row exactly once")
    row_index = ft_sorted["row_index"].to_numpy(dtype=int)
    if np.any((row_index < 0) | (row_index >= len(y_true))):
        raise ValueError("FT row_index is outside the training frame")
    if not np.array_equal(ft_sorted["target"].to_numpy(dtype=int), y_true[row_index]):
        raise ValueError("FT OOF target does not match encoded training target")
    _validate_probabilities(ft_sorted, "FT OOF")
    _validate_probabilities(base_sorted, "Canonical OOF")


def _global_probabilities(oof: pd.DataFrame) -> np.ndarray:
    return oof.groupby("row_index", sort=True)["probability"].mean().to_numpy(dtype=float)


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def validate_reported_ft_metrics(
    y_true: np.ndarray,
    ft_oof: pd.DataFrame,
    reported_metrics: dict[str, Any],
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Recreate completed FT metrics from repeat-averaged OOF probabilities."""
    recreated = _binary_metrics(y_true, _global_probabilities(ft_oof))
    for metric, observed in recreated.items():
        reported = reported_metrics.get(metric)
        if reported is None or not math.isclose(
            observed, float(reported), abs_tol=tolerance, rel_tol=0
        ):
            raise ValueError(
                f"FT reported {metric} does not match recreated value: {reported!r} != {observed}"
            )
    return recreated


def _overlap(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    denominator = min(left_count, right_count)
    return {
        "left_count": left_count,
        "right_count": right_count,
        "intersection_count": intersection,
        "union_count": union,
        "jaccard": intersection / union if union else 1.0,
        "overlap_coefficient": intersection / denominator if denominator else 1.0,
    }


def calculate_diversity_metrics(
    y_true: np.ndarray, baseline_probability: np.ndarray, ft_probability: np.ndarray
) -> dict[str, Any]:
    """Calculate prediction, residual, and deterministic hard-error diversity."""
    y = np.asarray(y_true, dtype=int)
    baseline = np.asarray(baseline_probability, dtype=float)
    ft = np.asarray(ft_probability, dtype=float)
    if not (len(y) == len(baseline) == len(ft)):
        raise ValueError("Diversity inputs are not aligned")
    baseline_residual = y - baseline
    ft_residual = y - ft
    baseline_error = (baseline >= 0.5).astype(int) != y
    ft_error = (ft >= 0.5).astype(int) != y

    if np.std(baseline_error) == 0 or np.std(ft_error) == 0:
        hard_error_correlation = 0.0
    else:
        hard_error_correlation = float(np.corrcoef(baseline_error, ft_error)[0, 1])

    baseline_loss = -(y * np.log(np.clip(baseline, 1e-15, 1 - 1e-15))) - (
        (1 - y) * np.log(np.clip(1 - baseline, 1e-15, 1 - 1e-15))
    )
    ft_loss = -(y * np.log(np.clip(ft, 1e-15, 1 - 1e-15))) - (
        (1 - y) * np.log(np.clip(1 - ft, 1e-15, 1 - 1e-15))
    )
    n_top = max(1, math.ceil(0.01 * len(y)))
    row_index = np.arange(len(y))
    baseline_top_idx = np.lexsort((row_index, -baseline_loss))[:n_top]
    ft_top_idx = np.lexsort((row_index, -ft_loss))[:n_top]
    baseline_top = np.zeros(len(y), dtype=bool)
    ft_top = np.zeros(len(y), dtype=bool)
    baseline_top[baseline_top_idx] = True
    ft_top[ft_top_idx] = True

    return {
        "prediction_pearson": float(np.corrcoef(baseline, ft)[0, 1]),
        "prediction_spearman": float(spearmanr(baseline, ft).statistic),
        "residual_pearson": float(np.corrcoef(baseline_residual, ft_residual)[0, 1]),
        "residual_spearman": float(spearmanr(baseline_residual, ft_residual).statistic),
        "hard_error_indicator_correlation": hard_error_correlation,
        "mean_absolute_prediction_difference": float(np.mean(np.abs(baseline - ft))),
        "root_mean_squared_prediction_difference": float(
            np.sqrt(np.mean(np.square(baseline - ft)))
        ),
        "threshold_0_5_hard_error_overlap": _overlap(baseline_error, ft_error),
        "top_1pct_loss_overlap": _overlap(baseline_top, ft_top),
    }


def evaluate_blend_grid(
    y_true: np.ndarray,
    baseline_oof: pd.DataFrame,
    ft_oof: pd.DataFrame,
    weights: tuple[float, ...] = BLEND_WEIGHTS,
) -> BlendGridResult:
    """Evaluate the fixed grid using repeat-averaged global metrics."""
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("Blend weights must be non-empty and unique")
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("Blend weights must be within [0, 1]")
    base = baseline_oof.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    ft = ft_oof.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    if not base[KEY_COLUMNS].equals(ft[KEY_COLUMNS]):
        raise ValueError("Baseline and FT OOF keys do not align for blending")
    baseline_probability = base["probability"].to_numpy(dtype=float)
    ft_probability = ft["probability"].to_numpy(dtype=float)
    row_index = base["row_index"].to_numpy(dtype=int)
    records: list[dict[str, float]] = []
    predictions: dict[float, np.ndarray] = {}
    for raw_weight in weights:
        weight = float(raw_weight)
        if not 0 <= weight <= 1:
            raise ValueError(f"Blend weight outside [0, 1]: {weight}")
        blended = (1 - weight) * baseline_probability + weight * ft_probability
        predictions[weight] = blended
        global_probability = (
            pd.DataFrame({"row_index": row_index, "probability": blended})
            .groupby("row_index", sort=True)["probability"]
            .mean()
            .to_numpy()
        )
        records.append({"ft_weight": weight, **_binary_metrics(y_true, global_probability)})
    metrics = pd.DataFrame(records)
    best_row = metrics.sort_values(
        ["roc_auc", "pr_auc", "log_loss", "ft_weight"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    return BlendGridResult(
        metrics=metrics, predictions=predictions, best_weight=float(best_row.ft_weight)
    )


def make_promotion_decision(
    global_auc_delta: float,
    repeat_auc_deltas: list[float],
    nb_ci_lower: float,
    global_log_loss_delta: float,
    slice_regressions: list[str],
) -> dict[str, Any]:
    """Apply the preregistered binary promotion gate at exact boundaries."""
    scalar_values = (global_auc_delta, nb_ci_lower, global_log_loss_delta)
    if len(repeat_auc_deltas) != 3:
        raise ValueError("Promotion gate requires exactly three repeat AUC deltas")
    if not np.isfinite(np.asarray((*scalar_values, *repeat_auc_deltas), dtype=float)).all():
        raise ValueError("Promotion gate inputs must all be finite")
    criteria = {
        "global_auc_delta_at_least_0_00005": global_auc_delta >= 0.00005,
        "all_repeat_auc_deltas_nonnegative": all(delta >= 0 for delta in repeat_auc_deltas),
        "nadeau_bengio_ci_lower_positive": nb_ci_lower > 0,
        "log_loss_degradation_at_most_0_00020": global_log_loss_delta <= 0.00020,
        "no_material_frozen_slice_regression": not slice_regressions,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "decision": "GO" if not failed else "NO-GO",
        "promoted": not failed,
        "criteria": criteria,
        "failed_criteria": failed,
        "observed": {
            "global_auc_delta": float(global_auc_delta),
            "repeat_auc_deltas": [float(value) for value in repeat_auc_deltas],
            "nadeau_bengio_ci_lower": float(nb_ci_lower),
            "global_log_loss_delta": float(global_log_loss_delta),
            "slice_regressions": list(slice_regressions),
        },
    }


def build_selected_policy(analyzed_best_ft_weight: float, promoted: bool) -> dict[str, Any]:
    """Describe the OOF-selected policy without producing test predictions."""
    deployed_ft_weight = float(analyzed_best_ft_weight) if promoted else 0.0
    return {
        "scope": "oof_only",
        "policy_type": "blend" if promoted else "single_model",
        "candidate_id": "lightgbm_hpo_trial35" if not promoted else "lightgbm_ft_transformer",
        "ft_weight": deployed_ft_weight,
        "lightgbm_weight": 1.0 - deployed_ft_weight,
        "analyzed_best_ft_weight": float(analyzed_best_ft_weight),
        "promoted": bool(promoted),
        "test_blend_produced": False,
    }


def validate_canonical_metadata(metadata: dict[str, Any]) -> None:
    """Verify the canonical candidate is the pinned complete 5x3 LightGBM run."""
    identity = (
        metadata.get("candidate_id"),
        metadata.get("run_id"),
        metadata.get("model_family"),
        metadata.get("trial_number"),
        metadata.get("feature_policy"),
        metadata.get("status"),
        metadata.get("protocol"),
    )
    if identity != (
        "lightgbm_hpo_trial35",
        "lightgbm_hpo_trial35",
        "lightgbm",
        35,
        "base",
        "completed",
        "full",
    ):
        raise ValueError("Canonical LightGBM metadata identity mismatch")

    protocol = (
        metadata.get("effective_n_splits"),
        metadata.get("effective_n_repeats"),
        metadata.get("effective_n_rows"),
        metadata.get("evaluated_fold_ids"),
        metadata.get("dry_run"),
    )
    if protocol != (5, 3, 668665, list(range(15)), False):
        raise ValueError("Canonical LightGBM metadata protocol mismatch")

    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("split_sha256"),
        provenance.get("data_sha256"),
        provenance.get("model_name"),
        provenance.get("smoke"),
    ) != (
        EXPECTED_HASHES["split_file"],
        EXPECTED_COMBINED_DATA_SHA256,
        "lightgbm",
        False,
    ):
        raise ValueError("Canonical LightGBM metadata provenance mismatch")

    if metadata.get("artifact_checksums") != {
        "oof_predictions": EXPECTED_HASHES["canonical_oof"],
        "test_predictions": EXPECTED_HASHES["canonical_test"],
    }:
        raise ValueError("Canonical LightGBM metadata checksums mismatch")


def build_final_test_predictions(
    canonical_test: pd.DataFrame,
    ft_test: pd.DataFrame,
    weight: float,
    promoted: bool,
) -> pd.DataFrame:
    """Return canonical values unchanged unless the selected blend passes every gate."""
    expected_columns = ["id", "Will_Buy_EV"]
    if (
        list(canonical_test.columns) != expected_columns
        or list(ft_test.columns) != expected_columns
    ):
        raise ValueError(f"Test predictions must use exact schema {expected_columns}")
    if not np.array_equal(canonical_test["id"].to_numpy(), ft_test["id"].to_numpy()):
        raise ValueError("Canonical and FT test IDs do not align")
    if not promoted:
        return canonical_test.copy(deep=True)
    result = canonical_test.copy(deep=True)
    result["Will_Buy_EV"] = (1 - weight) * canonical_test["Will_Buy_EV"].to_numpy() + weight * (
        ft_test["Will_Buy_EV"].to_numpy()
    )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_hash(path: Path, expected: str, name: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {observed}")
    return observed


def validate_output_directory(output_dir: Path | str) -> Path:
    """Restrict publication to the single approved evaluation directory."""
    resolved = Path(output_dir).resolve()
    approved = DEFAULT_OUTPUT_DIR.resolve()
    if resolved != approved:
        raise ValueError(f"Output directory must be exactly the approved path: {approved}")
    return resolved


def _verify_source_contracts(
    ft_dir: Path,
    canonical_dir: Path,
    split_path: Path,
    frozen_slices_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inventory = verify_artifact_inventory(
        ft_dir,
        expected_count=len(EXPECTED_FT_ARTIFACT_PATHS),
        expected_paths=EXPECTED_FT_ARTIFACT_PATHS,
        expected_inventory_sha256=EXPECTED_HASHES["ft_inventory"],
        approved_supplemental_files=EXPECTED_FT_SUPPLEMENTAL_FILES,
    )
    integrity = {"ft_inventory": inventory}
    training_manifest = _read_json(ft_dir / "training_manifest.json")
    run_state = _read_json(ft_dir / "run_state.json")
    integrity["run_fingerprint_sha256"] = verify_ft_provenance(training_manifest, run_state)
    canonical_folds = [[repeat, fold] for repeat in range(3) for fold in range(5)]
    if training_manifest.get("status") != "completed" or run_state.get("status") != "completed":
        raise ValueError("FT run is not completed")
    if training_manifest.get("completed_folds") != canonical_folds:
        raise ValueError("FT training manifest does not contain all canonical folds")
    if run_state.get("completed_folds") != canonical_folds:
        raise ValueError("FT run state does not contain all canonical folds")
    fingerprint = training_manifest["run_fingerprint"]

    for path, expected, name in (
        (
            ft_dir / "artifact_inventory.json",
            EXPECTED_HASHES["ft_inventory"],
            "FT artifact inventory",
        ),
        (TRAIN_PATH, EXPECTED_HASHES["train"], "train CSV"),
        (TEST_PATH, EXPECTED_HASHES["test"], "test CSV"),
        (SAMPLE_SUBMISSION_PATH, EXPECTED_HASHES["sample_submission"], "sample submission"),
        (split_path, EXPECTED_HASHES["split_file"], "split parquet"),
        (frozen_slices_path, EXPECTED_HASHES["frozen_slices"], "frozen slices"),
        (
            canonical_dir / "oof_predictions.parquet",
            EXPECTED_HASHES["canonical_oof"],
            "canonical OOF",
        ),
        (
            canonical_dir / "test_predictions.parquet",
            EXPECTED_HASHES["canonical_test"],
            "canonical test",
        ),
    ):
        _require_hash(path, expected, name)
    if fingerprint.get("data_hashes") != training_manifest.get("data_hashes"):
        raise ValueError("FT fingerprint data hashes do not match its training manifest")
    expected_data_hashes = {
        "train_csv_sha256": EXPECTED_HASHES["train"],
        "test_csv_sha256": EXPECTED_HASHES["test"],
        "sample_submission_csv_sha256": EXPECTED_HASHES["sample_submission"],
    }
    if training_manifest.get("data_hashes") != expected_data_hashes:
        raise ValueError("FT training manifest data hashes are not canonical")

    split = pd.read_parquet(split_path)
    observed_split_content = split_content_sha256(split)
    if observed_split_content != EXPECTED_HASHES["split_content"]:
        raise ValueError("Canonical split content digest mismatch")
    integrity["split_content_sha256"] = observed_split_content

    canonical_metadata = _read_json(canonical_dir / "candidate_metadata.json")
    validate_canonical_metadata(canonical_metadata)

    frozen = _read_json(frozen_slices_path)
    frozen_names = [*frozen.get("predeclared_slices", {}), *frozen.get("cohorts", {})]
    if len(frozen_names) != 22 or len(set(frozen_names)) != 22:
        raise ValueError("Frozen-slice contract must contain exactly 22 unique definitions")
    integrity["verified_hashes"] = dict(EXPECTED_HASHES)
    return integrity, split


def _validate_ft_aggregate_artifacts(
    ft_dir: Path, ft_oof: pd.DataFrame, ft_test: pd.DataFrame
) -> None:
    fold_oof = pd.concat(
        [
            pd.read_csv(ft_dir / f"fold_oof_r{repeat}_f{fold}.csv")
            for repeat in range(3)
            for fold in range(5)
        ],
        ignore_index=True,
    )
    sort_columns = ["repeat", "fold", "row_index"]
    pd.testing.assert_frame_equal(
        fold_oof.sort_values(sort_columns, kind="stable").reset_index(drop=True),
        ft_oof.sort_values(sort_columns, kind="stable").reset_index(drop=True),
        check_exact=True,
    )
    test_parts = [
        pd.read_csv(ft_dir / f"fold_test_r{repeat}_f{fold}.csv")
        .sort_values("row_index", kind="stable")["probability"]
        .to_numpy(dtype=float)
        for repeat in range(3)
        for fold in range(5)
    ]
    expected_test = np.mean(np.vstack(test_parts), axis=0)
    actual_test = ft_test.sort_values("row_index", kind="stable")["probability"].to_numpy(
        dtype=float
    )
    if not np.allclose(actual_test, expected_test, atol=1e-15, rtol=0):
        raise ValueError("FT aggregate test predictions are not the mean of 15 fold predictions")


def _atomic_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _publish_staged_directory(staging_dir: Path, output_dir: Path) -> None:
    output_dir = validate_output_directory(output_dir)
    backup = output_dir.with_name(f".{output_dir.name}.backup")
    if backup.exists():
        if not output_dir.exists():
            raise RuntimeError(f"Stale evaluation backup requires manual review: {backup}")
        shutil.rmtree(backup)
    had_previous = output_dir.exists()
    try:
        if had_previous:
            os.replace(output_dir, backup)
        os.replace(staging_dir, output_dir)
    except Exception:
        if had_previous and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            warnings.warn(
                f"Evaluation published successfully, but backup cleanup failed for {backup}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _report_markdown(
    standalone: dict[str, dict[str, float]],
    selected_weight: float,
    selected_metrics: dict[str, Any],
    diversity: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    lgb = standalone["lightgbm"]
    ft = standalone["ft_transformer"]
    global_metrics = selected_metrics["global_metrics"]
    ci_lower = selected_metrics["nadeau_bengio"]["ci_nb_lower"]
    mean_abs_difference = diversity["mean_absolute_prediction_difference"]
    return "\n".join(
        [
            "# FT-Transformer Evaluation",
            "",
            "## Decision",
            f"- Gate: `{decision['decision']}`",
            f"- Selected FT weight: `{selected_weight:g}`",
            "- Final policy: `selected blend`"
            if decision["promoted"]
            else "- Final policy: `canonical LightGBM`",
            f"- Failed criteria: `{decision['failed_criteria']}`",
            "",
            "## Standalone Models",
            "| Model | ROC-AUC | PR-AUC | LogLoss |",
            "| :--- | ---: | ---: | ---: |",
            f"| LightGBM | {lgb['roc_auc']:.9f} | {lgb['pr_auc']:.9f} | {lgb['log_loss']:.9f} |",
            f"| FT-Transformer | {ft['roc_auc']:.9f} | {ft['pr_auc']:.9f} | {ft['log_loss']:.9f} |",
            "",
            "## Selected Blend",
            f"- ROC-AUC delta: {global_metrics['delta_roc_auc']:+.9f}",
            f"- PR-AUC delta: {global_metrics['delta_pr_auc']:+.9f}",
            f"- LogLoss delta: {global_metrics['delta_log_loss']:+.9f}",
            f"- Nadeau-Bengio 95% CI lower: {ci_lower:+.9f}",
            "",
            "## Diversity",
            f"- Prediction Pearson: {diversity['prediction_pearson']:.9f}",
            f"- Residual Pearson: {diversity['residual_pearson']:.9f}",
            f"- Mean absolute probability difference: {mean_abs_difference:.9f}",
            "",
        ]
    )


def run_evaluation(
    ft_dir: Path = FT_ARTIFACT_DIR,
    canonical_dir: Path = CANONICAL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split_path: Path = SPLIT_PATH,
    frozen_slices_path: Path = FROZEN_SLICES_PATH,
) -> dict[str, Any]:
    """Verify all inputs, evaluate fixed blends, apply gates, and publish atomically."""
    started_at = time.perf_counter()
    target_output_dir = validate_output_directory(output_dir)
    integrity, split = _verify_source_contracts(
        ft_dir, canonical_dir, split_path, frozen_slices_path
    )
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    validate_frames(train, test, contract)
    y_true = encode_target(train[contract.target_column], contract)

    ft_oof = pd.read_csv(ft_dir / "oof_predictions.csv")
    baseline_oof = pd.read_parquet(canonical_dir / "oof_predictions.parquet")
    validate_oof_alignment(ft_oof, baseline_oof, split, y_true)
    ft_test_rows = pd.read_csv(ft_dir / "test_predictions.csv")
    if list(ft_test_rows.columns) != ["row_index", "probability"]:
        raise ValueError("FT test prediction schema mismatch")
    if not np.array_equal(ft_test_rows["row_index"].to_numpy(), np.arange(len(test))):
        raise ValueError("FT test row_index does not match physical test-row positions")
    _validate_probabilities(ft_test_rows, "FT test")
    _validate_ft_aggregate_artifacts(ft_dir, ft_oof, ft_test_rows)

    ft_submission = pd.read_csv(ft_dir / "submission.csv")
    if list(ft_submission.columns) != [contract.id_column, contract.target_column]:
        raise ValueError("FT submission schema mismatch")
    if not np.array_equal(
        ft_submission[contract.id_column].to_numpy(), test[contract.id_column].to_numpy()
    ):
        raise ValueError("FT submission IDs do not align with test data")
    if not np.array_equal(
        ft_submission[contract.target_column].to_numpy(), ft_test_rows["probability"].to_numpy()
    ):
        raise ValueError("FT submission probabilities do not match aggregate FT test predictions")

    ft_reported = _read_json(ft_dir / "global_metrics.json")
    if ft_reported.get("status") != "completed" or ft_reported.get("completed_folds") != 15:
        raise ValueError("FT global metrics do not describe a completed 15-fold run")
    ft_metrics = validate_reported_ft_metrics(y_true, ft_oof, ft_reported)
    baseline_global = _global_probabilities(baseline_oof)
    ft_global = _global_probabilities(ft_oof)
    standalone = {
        "lightgbm": _binary_metrics(y_true, baseline_global),
        "ft_transformer": ft_metrics,
    }
    base_sorted = baseline_oof.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    ft_sorted = ft_oof.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    ft_comparison = calculate_experiment_metrics(
        y_true=y_true,
        oof_baseline=base_sorted,
        oof_candidate=ft_sorted,
        n_rows=len(train),
    )
    diversity = calculate_diversity_metrics(y_true, baseline_global, ft_global)
    blend_grid = evaluate_blend_grid(y_true, baseline_oof, ft_oof)
    selected_weight = blend_grid.best_weight

    selected_oof = base_sorted.copy()
    selected_oof["probability"] = blend_grid.predictions[selected_weight]
    selected_metrics = calculate_experiment_metrics(
        y_true=y_true,
        oof_baseline=base_sorted,
        oof_candidate=selected_oof,
        n_rows=len(train),
    )
    slice_metrics = evaluate_frozen_slices(
        train_df=train,
        y_true=y_true,
        p_baseline=selected_metrics["p_baseline_global"],
        p_candidate=selected_metrics["p_candidate_global"],
        frozen_slices_path=frozen_slices_path,
    )
    if len(slice_metrics) != 22 or slice_metrics["slice_name"].nunique() != 22:
        raise ValueError("Frozen-slice evaluation must produce exactly 22 unique rows")
    slice_regressions = find_material_slice_regressions(slice_metrics)
    global_metrics = selected_metrics["global_metrics"]
    decision = make_promotion_decision(
        global_auc_delta=global_metrics["delta_roc_auc"],
        repeat_auc_deltas=global_metrics["repeat_auc_deltas"],
        nb_ci_lower=selected_metrics["nadeau_bengio"]["ci_nb_lower"],
        global_log_loss_delta=global_metrics["delta_log_loss"],
        slice_regressions=slice_regressions,
    )

    canonical_test = pd.read_parquet(canonical_dir / "test_predictions.parquet")
    expected_test_columns = [contract.id_column, contract.target_column]
    if list(canonical_test.columns) != expected_test_columns:
        raise ValueError("Canonical test prediction schema mismatch")
    if not np.array_equal(
        canonical_test[contract.id_column].to_numpy(), test[contract.id_column].to_numpy()
    ):
        raise ValueError("Canonical test IDs do not align with physical test rows")
    canonical_test_probability = canonical_test[contract.target_column].to_numpy(dtype=float)
    if not np.isfinite(canonical_test_probability).all() or np.any(
        (canonical_test_probability < 0) | (canonical_test_probability > 1)
    ):
        raise ValueError("Canonical test predictions contain invalid probabilities")

    selected_policy = build_selected_policy(selected_weight, decision["promoted"])

    aligned_oof = pd.DataFrame(
        {
            "row_index": np.arange(len(train)),
            "target": y_true,
            "lightgbm_probability": selected_metrics["p_baseline_global"],
            "ft_transformer_probability": ft_global,
            "selected_probability": selected_metrics["p_candidate_global"],
        }
    )

    target_output_dir = validate_output_directory(target_output_dir)
    target_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{target_output_dir.name}.stage_", dir=target_output_dir.parent)
    )
    try:
        blend_grid.metrics.to_csv(staging_dir / "blend_grid.csv", index=False)
        pd.DataFrame(ft_comparison["fold_metrics"]).to_csv(
            staging_dir / "fold_metrics.csv", index=False
        )
        pd.DataFrame(ft_comparison["repeat_metrics"]).to_csv(
            staging_dir / "repeat_metrics.csv", index=False
        )
        slice_metrics.to_csv(staging_dir / "slice_metrics.csv", index=False)
        aligned_oof.to_parquet(staging_dir / "aligned_oof.parquet", index=False)
        _atomic_json(staging_dir / "integrity_report.json", integrity)
        metric_payload = {
            "scope": "oof_only",
            "standalone": standalone,
            "ft_vs_lightgbm": {
                "global_metrics": ft_comparison["global_metrics"],
                "nadeau_bengio": ft_comparison["nadeau_bengio"],
            },
            "analyzed_best_ft_weight": selected_weight,
            "selected_blend_vs_lightgbm": {
                "global_metrics": global_metrics,
                "nadeau_bengio": selected_metrics["nadeau_bengio"],
            },
            "promotion_decision": decision,
        }
        _atomic_json(staging_dir / "metrics.json", metric_payload)
        _atomic_json(staging_dir / "diversity.json", diversity)
        _atomic_json(staging_dir / "selected_policy.json", selected_policy)
        (staging_dir / "report.md").write_text(
            _report_markdown(standalone, selected_weight, selected_metrics, diversity, decision),
            encoding="utf-8",
        )
        artifacts = []
        for path in sorted(staging_dir.iterdir()):
            if path.name == "evaluation_manifest.json":
                continue
            record: dict[str, Any] = {
                "path": path.name,
                "byte_count": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            row_count = _count_records(path)
            if row_count is not None:
                record["row_count"] = row_count
            artifacts.append(record)
        manifest = {
            "status": "completed",
            "scope": "oof_only",
            "baseline": "lightgbm_hpo_trial35",
            "candidate": "ft_transformer_5x3",
            "analyzed_best_ft_weight": selected_weight,
            "decision": decision["decision"],
            "final_policy": selected_policy,
            "test_blend_produced": False,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
        _atomic_json(staging_dir / "evaluation_manifest.json", manifest)
        _publish_staged_directory(staging_dir, target_output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    runtime_seconds = time.perf_counter() - started_at
    return {
        "output_dir": str(target_output_dir),
        "selected_ft_weight": selected_weight,
        "decision": decision,
        "selected_policy": selected_policy,
        "standalone_metrics": standalone,
        "global_metrics": global_metrics,
        "nadeau_bengio": selected_metrics["nadeau_bengio"],
        "ft_vs_lightgbm": {
            "global_metrics": ft_comparison["global_metrics"],
            "nadeau_bengio": ft_comparison["nadeau_bengio"],
        },
        "repeat_auc_deltas": global_metrics["repeat_auc_deltas"],
        "diversity": diversity,
        "slice_regressions": slice_regressions,
        "slice_count": len(slice_metrics),
        "runtime_seconds": runtime_seconds,
    }
