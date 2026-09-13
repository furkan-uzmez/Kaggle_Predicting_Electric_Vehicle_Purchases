"""Repeated cross-validation training execution and aggregation engine."""

import hashlib
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from ev_purchase.artifacts import save_run_metrics
from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.features import apply_feature_policy
from ev_purchase.metrics import classification_metrics, validate_probabilities
from ev_purchase.models import build_model
from ev_purchase.splits import build_split_manifest, validate_split_manifest


def aggregate_repeated_oof(
    row_indices: np.ndarray,
    predictions: np.ndarray,
    n_rows: int,
    n_repeats: int,
) -> np.ndarray:
    """Aggregate repeated out-of-fold predictions by averaging over repeats.

    Args:
        row_indices: 1D array of row index integers.
        predictions: 1D array of fold validation probability predictions.
        n_rows: Total number of rows in the dataset.
        n_repeats: Expected number of prediction repeats per row.

    Returns:
        1D numpy array of length n_rows containing mean OOF probabilities.

    Raises:
        ValueError: If any row does not receive exactly n_repeats predictions.
    """
    sums = np.zeros(n_rows, dtype=float)
    counts = np.zeros(n_rows, dtype=int)
    np.add.at(sums, row_indices, predictions)
    np.add.at(counts, row_indices, 1)
    if not np.all(counts == n_repeats):
        raise ValueError("every row must have one OOF prediction per repeat")
    return sums / counts


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_combined_sha256(paths: list[Path]) -> str:
    """Compute combined SHA-256 hash of multiple files."""
    hasher = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def _validate_completed_run_payload(status: str, fold_metrics: Any, artifact_paths: Any) -> None:
    """Validate completed run payload fields before saving."""
    if status == "completed":
        if not fold_metrics:
            raise ValueError("fold_metrics cannot be empty for a completed run")
        if not artifact_paths:
            raise ValueError("artifact_paths cannot be empty for a completed run")


def run_cross_validation(
    config_path: Path | str,
    project_root: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute leakage-safe repeated cross-validation training run.

    Args:
        config_path: Path to YAML experiment configuration file.
        project_root: Repository root path (defaults to current working directory).
        split_manifest_path: Explicit path to split manifest parquet file (optional).
        smoke: If True, run fast smoke validation on deterministic sample subset.

    Returns:
        Dictionary containing run summary, aggregate metrics, OOF predictions, and artifact paths.
    """
    start_time = time.time()
    cfg_p = Path(config_path)
    root_p = Path(project_root) if project_root is not None else Path.cwd()

    if not cfg_p.is_absolute():
        cfg_p = (root_p / cfg_p).resolve()

    config = load_config(cfg_p, project_root=root_p)

    with open(cfg_p, encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f)

    model_config = raw_yaml.get("model", {})
    if isinstance(model_config, str):
        model_name = model_config
        model_params = {}
    elif isinstance(model_config, dict):
        model_name = model_config.get("name", "dummy")
        model_params = model_config.get("params", {})
    else:
        model_name = "dummy"
        model_params = {}

    train_df, test_df = load_frames(config.data.train, config.data.test)
    contract = DatasetContract(
        id_column=config.data.id_column, target_column=config.data.target_column
    )
    validate_frames(train_df, test_df, contract)
    y_encoded = encode_target(train_df[config.data.target_column], contract)

    if split_manifest_path is not None:
        manifest_p = Path(split_manifest_path)
        if not manifest_p.is_absolute():
            manifest_p = (root_p / manifest_p).resolve()
    else:
        manifest_p = (config.outputs.root / "splits" / "repeated_stratified_5x3.parquet").resolve()

    manifest = pd.read_parquet(manifest_p)
    validate_split_manifest(
        manifest=manifest,
        n_rows=len(train_df),
        n_splits=config.validation.n_splits,
        n_repeats=config.validation.n_repeats,
        y=y_encoded,
    )

    if smoke:
        smoke_n_rows = min(5000, len(train_df))
        train_df_use = train_df.iloc[:smoke_n_rows].reset_index(drop=True)
        y_encoded_use = y_encoded[:smoke_n_rows]
        test_df_use = test_df.iloc[: min(1000, len(test_df))].reset_index(drop=True)

        smoke_manifest = build_split_manifest(
            y=y_encoded_use,
            n_splits=config.validation.n_splits,
            n_repeats=1,
            random_state=config.validation.random_state,
        )
        validate_split_manifest(
            manifest=smoke_manifest,
            n_rows=smoke_n_rows,
            n_splits=config.validation.n_splits,
            n_repeats=1,
            y=y_encoded_use,
        )
        manifest_use = smoke_manifest
        n_repeats_use = 1
        status = "smoke_completed"
    else:
        train_df_use = train_df
        y_encoded_use = y_encoded
        test_df_use = test_df
        manifest_use = manifest
        n_repeats_use = config.validation.n_repeats
        status = "completed"

    X_train_raw = train_df_use.drop(columns=[config.data.id_column, config.data.target_column])
    X_test_raw = test_df_use.drop(columns=[config.data.id_column])

    X_train = apply_feature_policy("base", X_train_raw)
    X_test = apply_feature_policy("base", X_test_raw)

    oof_row_indices: list[int] = []
    oof_repeats: list[int] = []
    oof_folds: list[int] = []
    oof_probs: list[float] = []

    fold_groups = list(manifest_use.groupby(["repeat", "fold"], sort=True))
    n_folds_total = len(fold_groups)
    test_preds_matrix = np.zeros((len(test_df_use), n_folds_total), dtype=float)

    fold_metrics: list[dict[str, float | int]] = []
    all_indices = np.arange(len(train_df_use))

    for fold_idx, ((repeat, fold), fold_df) in enumerate(fold_groups):
        val_idx = fold_df["row_index"].to_numpy()
        train_idx = np.setdiff1d(all_indices, val_idx)

        X_tr = X_train.iloc[train_idx]
        y_tr = y_encoded_use[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_encoded_use[val_idx]

        fold_seed = config.project.seed + int(repeat) * 100 + int(fold)
        model = build_model(
            model_name=model_name,
            seed=fold_seed,
            sample_frame=X_tr,
            params=model_params,
        )

        model.fit(X_tr, y_tr)

        val_p = model.predict_proba(X_va)[:, 1]
        validate_probabilities(val_p)

        test_p = model.predict_proba(X_test)[:, 1]
        validate_probabilities(test_p)

        f_metrics = classification_metrics(y_va, val_p)
        fold_record: dict[str, float | int] = {
            "repeat": int(repeat),
            "fold": int(fold),
            **f_metrics,
        }
        fold_metrics.append(fold_record)

        oof_row_indices.extend(val_idx.tolist())
        oof_repeats.extend([int(repeat)] * len(val_idx))
        oof_folds.extend([int(fold)] * len(val_idx))
        oof_probs.extend(val_p.tolist())

        test_preds_matrix[:, fold_idx] = test_p

    oof_probs_arr = np.array(oof_probs)
    oof_row_indices_arr = np.array(oof_row_indices)
    oof_mean_probs = aggregate_repeated_oof(
        row_indices=oof_row_indices_arr,
        predictions=oof_probs_arr,
        n_rows=len(train_df_use),
        n_repeats=n_repeats_use,
    )
    validate_probabilities(oof_mean_probs)

    aggregate_metrics = classification_metrics(y_encoded_use, oof_mean_probs)
    test_mean_probs = test_preds_matrix.mean(axis=1)
    validate_probabilities(test_mean_probs)

    config_sha256 = _compute_file_sha256(cfg_p)
    data_sha256 = _compute_combined_sha256([config.data.train, config.data.test])
    if smoke:
        split_sha256 = hashlib.sha256(smoke_manifest.to_parquet()).hexdigest()
    else:
        split_sha256 = _compute_file_sha256(manifest_p)

    run_hash = uuid4().hex[:8]
    run_id = f"{model_name}_{'smoke_' if smoke else ''}{run_hash}"

    runs_dir = config.outputs.root / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    rel_metrics = f"outputs/runs/{run_id}/metrics.json"
    rel_oof = f"outputs/runs/{run_id}/oof_predictions.parquet"
    rel_test = f"outputs/runs/{run_id}/test_predictions.parquet"

    full_metrics = root_p / rel_metrics
    full_oof = root_p / rel_oof
    full_test = root_p / rel_test

    oof_df = pd.DataFrame(
        {
            "row_index": oof_row_indices,
            "repeat": oof_repeats,
            "fold": oof_folds,
            "probability": oof_probs,
        }
    )
    oof_df.to_parquet(full_oof)

    test_pred_df = pd.DataFrame(
        {
            config.data.id_column: test_df_use[config.data.id_column],
            config.data.target_column: test_mean_probs,
        }
    )
    test_pred_df.to_parquet(full_test)

    runtime_sec = time.time() - start_time

    payload = {
        "run_id": run_id,
        "status": status,
        "smoke": smoke,
        "effective_n_splits": config.validation.n_splits,
        "effective_n_repeats": n_repeats_use,
        "effective_n_rows": len(train_df_use),
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "split_sha256": split_sha256,
        "model": model_name,
        "seed": config.project.seed,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "runtime_seconds": round(runtime_sec, 4),
        "artifact_paths": {
            "metrics": rel_metrics,
            "oof_predictions": rel_oof,
            "test_predictions": rel_test,
        },
    }

    _validate_completed_run_payload(status, fold_metrics, payload["artifact_paths"])
    save_run_metrics(payload, full_metrics)

    return {
        "run_id": run_id,
        "status": status,
        "aggregate_metrics": aggregate_metrics,
        "fold_metrics": fold_metrics,
        "oof_probs": oof_mean_probs,
        "test_probs": test_mean_probs,
        "oof_df": oof_df,
        "metrics_path": full_metrics,
    }
