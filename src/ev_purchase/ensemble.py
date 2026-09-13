"""OOF-Safe Blending Module and Ensembling Engine."""

import hashlib
import json
import os
import re
import shutil
from collections.abc import Sequence
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier

from ev_purchase.analysis import assert_comparable_runs
from ev_purchase.artifacts import load_run_metrics
from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.features import apply_feature_policy
from ev_purchase.metrics import classification_metrics, validate_probabilities
from ev_purchase.models import build_model
from ev_purchase.splits import validate_split_manifest
from ev_purchase.training import aggregate_repeated_oof

MIN_BLEND_LIFT_THRESHOLD = 0.00005  # 5e-5 minimum ROC-AUC lift to justify blend over single model
CANDIDATE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


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


def compute_dataset_sha256(train_path: Path, test_path: Path) -> str:
    """Compute canonical combined dataset SHA-256 hash used across tuning and ensembling."""
    return _compute_combined_sha256([train_path, test_path])


def validate_candidate_id(cand_id: str) -> str:
    """Validate candidate ID string against path traversal and unsafe characters."""
    if not isinstance(cand_id, str) or not cand_id.strip():
        raise ValueError("Candidate ID must be a non-empty string")

    cand_id = cand_id.strip()
    if (
        ".." in cand_id
        or "/" in cand_id
        or "\\" in cand_id
        or not CANDIDATE_ID_REGEX.match(cand_id)
    ):
        raise ValueError(f"Invalid candidate ID format: {cand_id!r}")
    return cand_id


def resolve_dry_run_output_dir(
    output_dir_arg: Path | str | None, project_root: Path | str | None = None
) -> Path:
    """Resolve dry-run output directory, forcing output to outputs/ensemble_smoke."""
    root_p = Path(project_root) if project_root is not None else Path.cwd()
    canonical_ensemble = (root_p / "outputs" / "ensemble").resolve()
    smoke_ensemble = (root_p / "outputs" / "ensemble_smoke").resolve()

    if output_dir_arg is None:
        return smoke_ensemble

    target_p = Path(output_dir_arg)
    if not target_p.is_absolute():
        target_p = (root_p / target_p).resolve()

    try:
        rel = target_p.relative_to(canonical_ensemble)
        if str(rel) == "." or not str(rel).startswith("smoke"):
            return smoke_ensemble
    except ValueError:
        pass

    if target_p == smoke_ensemble or target_p == canonical_ensemble:
        return smoke_ensemble

    return target_p


def weighted_blend(
    predictions: dict[str, np.ndarray], weights: dict[str, float | int]
) -> np.ndarray:
    """Compute normalized weighted blend of prediction probability vectors."""
    if not isinstance(predictions, dict) or len(predictions) == 0:
        raise ValueError("predictions must be a non-empty dictionary")

    if not isinstance(weights, dict):
        raise TypeError("weights must be a dictionary")

    missing_keys = set(predictions.keys()) - set(weights.keys())
    if missing_keys:
        raise ValueError(f"Missing weights for model keys: {sorted(missing_keys)}")

    lengths: list[int] = []
    validated_preds: dict[str, np.ndarray] = {}

    for name, p_vec in predictions.items():
        if isinstance(p_vec, bool):
            raise TypeError(f"Prediction vector for '{name}' cannot be boolean")
        arr = np.asarray(p_vec)
        validate_probabilities(arr)
        lengths.append(len(arr))
        validated_preds[name] = arr

    if len(set(lengths)) > 1:
        raise ValueError("predictions vectors must be aligned and have identical length")

    n_rows = lengths[0]
    total_weight = 0.0
    normalized_weights: dict[str, float] = {}

    for name in predictions.keys():
        w = weights[name]
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise TypeError(f"Weight for '{name}' must be numeric (int or float), got {type(w)}")
        f_w = float(w)
        if not np.isfinite(f_w):
            raise ValueError(f"Weight for '{name}' must be finite, got {f_w}")
        if f_w < 0:
            raise ValueError(f"Weight for '{name}' must be non-negative, got {f_w}")
        total_weight += f_w

    if total_weight <= 0:
        raise ValueError("Total sum of weights across models must be positive")

    for name in predictions.keys():
        normalized_weights[name] = float(weights[name]) / total_weight

    blended = np.zeros(n_rows, dtype=float)
    for name, arr in validated_preds.items():
        blended += normalized_weights[name] * arr

    validate_probabilities(blended)
    return blended


def validate_keyed_oof_alignment(oof_dfs: dict[str, pd.DataFrame]) -> None:
    """Validate candidate OOF DataFrames for identical key columns and ordering."""
    if not oof_dfs:
        raise ValueError("oof_dfs cannot be empty")

    required_cols = ["row_index", "repeat", "fold", "probability"]
    cand_names = sorted(list(oof_dfs.keys()))
    ref_name = cand_names[0]
    ref_df = oof_dfs[ref_name]

    for col in required_cols:
        if col not in ref_df.columns:
            raise ValueError(f"Candidate '{ref_name}' OOF DataFrame missing column '{col}'")

    ref_keys = ref_df[["row_index", "repeat", "fold"]].to_numpy()

    for cand_name in cand_names[1:]:
        df = oof_dfs[cand_name]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Candidate '{cand_name}' OOF DataFrame missing column '{col}'")

        if len(df) != len(ref_df):
            raise ValueError(
                f"Misaligned OOF prediction keys: candidate '{cand_name}' row count ({len(df)}) "
                f"does not match reference '{ref_name}' ({len(ref_df)})"
            )

        cand_keys = df[["row_index", "repeat", "fold"]].to_numpy()
        if not np.array_equal(cand_keys, ref_keys):
            raise ValueError(
                f"Misaligned OOF prediction keys: candidate '{cand_name}' key ordering "
                f"does not match reference '{ref_name}'"
            )


def validate_keyed_test_alignment(
    test_dfs: dict[str, pd.DataFrame],
    sample_submission_df: pd.DataFrame | None = None,
    id_column: str = "id",
    target_column: str = "Will_Buy_EV",
) -> None:
    """Validate candidate test DataFrames for identical ID order matching sample submission."""
    if not test_dfs:
        raise ValueError("test_dfs cannot be empty")

    cand_names = sorted(list(test_dfs.keys()))
    ref_name = cand_names[0]
    ref_df = test_dfs[ref_name]

    if id_column not in ref_df.columns or target_column not in ref_df.columns:
        raise ValueError(f"Candidate '{ref_name}' test DataFrame missing columns")

    ref_ids = ref_df[id_column].to_numpy()

    if sample_submission_df is not None:
        if id_column not in sample_submission_df.columns:
            raise ValueError(f"Sample submission missing ID column '{id_column}'")
        expected_ids = sample_submission_df[id_column].to_numpy()
        if not np.array_equal(ref_ids, expected_ids):
            raise ValueError(
                "Misaligned test IDs: candidate test IDs do not match sample submission"
            )

    for cand_name in cand_names[1:]:
        df = test_dfs[cand_name]
        if id_column not in df.columns or target_column not in df.columns:
            raise ValueError(f"Candidate '{cand_name}' test DataFrame missing columns")

        cand_ids = df[id_column].to_numpy()
        if not np.array_equal(cand_ids, ref_ids):
            msg = f"Misaligned test IDs: candidate '{cand_name}' IDs do not match reference"
            raise ValueError(msg)


def validate_candidate_provenance(
    cand_prov: dict[str, Any],
    expected_data_sha: str,
    expected_split_sha: str,
    expected_model_name: str | None = None,
) -> None:
    """Validate tuning provenance matching current dataset and split SHA256 hashes."""
    if not isinstance(cand_prov, dict):
        raise TypeError("Candidate provenance must be a dictionary")

    if cand_prov.get("smoke", False):
        raise ValueError(
            "Candidate provenance indicates smoke mode; cannot use for final ensembling"
        )

    data_sha = cand_prov.get("data_sha256")
    if data_sha != expected_data_sha:
        raise ValueError(
            f"data_sha256 mismatch: candidate {data_sha!r} != expected {expected_data_sha!r}"
        )

    split_sha = cand_prov.get("split_sha256")
    if split_sha != expected_split_sha:
        raise ValueError(
            f"split_sha256 mismatch: candidate {split_sha!r} != expected {expected_split_sha!r}"
        )

    if expected_model_name is not None:
        model_name = cand_prov.get("model_name")
        if model_name != expected_model_name:
            raise ValueError(
                f"model_name mismatch: candidate {model_name!r} != expected {expected_model_name!r}"
            )


def validate_finalist_fold_metrics(
    fold_metrics_df: pd.DataFrame,
    trial_number: int,
    expected_fold_ids: list[int] | None = None,
) -> None:
    """Validate finalist fold metrics for exact fold identity without missing or duplicate folds."""
    if expected_fold_ids is None:
        expected_fold_ids = list(range(15))

    t_metrics = fold_metrics_df[fold_metrics_df["trial_number"] == trial_number]
    if len(t_metrics) == 0:
        raise ValueError(f"Trial number {trial_number} not found in fold metrics")

    if "fold_id" not in t_metrics.columns:
        raise ValueError("fold_metrics DataFrame missing 'fold_id' column")

    actual_fold_ids = sorted(t_metrics["fold_id"].tolist())
    if actual_fold_ids != sorted(expected_fold_ids):
        msg = f"fold_id mismatch for trial {trial_number}: actual != expected"
        raise ValueError(msg)

    if len(set(t_metrics["fold_id"])) != len(t_metrics):
        raise ValueError(
            f"Duplicate fold_id entries found in fold metrics for trial {trial_number}"
        )


def generate_weight_grid(
    model_names: Sequence[str], grid_steps: int = 10
) -> list[dict[str, float]]:
    """Generate deterministic finite weight grid for model candidates."""
    names = sorted(list(set(model_names)))
    n_models = len(names)

    if n_models == 0:
        return []

    if n_models == 1:
        return [{names[0]: 1.0}]

    grid: list[dict[str, float]] = []
    seen_tuples: set[tuple[float, ...]] = set()

    for name in names:
        w_dict = {m: 1.0 if m == name else 0.0 for m in names}
        tup = tuple(w_dict[m] for m in names)
        seen_tuples.add(tup)
        grid.append(w_dict)

    eq_val = round(1.0 / n_models, 6)
    eq_dict = {m: eq_val for m in names}
    eq_tup = tuple(eq_dict[m] for m in names)
    if eq_tup not in seen_tuples:
        seen_tuples.add(eq_tup)
        grid.append(eq_dict)

    for combo in combinations_with_replacement(range(n_models), grid_steps):
        counts = [0] * n_models
        for idx in combo:
            counts[idx] += 1

        total = sum(counts)
        if total == 0:
            continue

        norm = tuple(round(c / total, 6) for c in counts)
        if norm not in seen_tuples:
            seen_tuples.add(norm)
            w_dict = {names[i]: norm[i] for i in range(n_models)}
            grid.append(w_dict)

    return grid


def evaluate_blend_grid(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    weight_grid: list[dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate grid of candidate weight combinations on OOF predictions."""
    if not isinstance(oof_predictions, dict) or len(oof_predictions) == 0:
        raise ValueError("oof_predictions must be a non-empty dictionary")

    y_arr = np.asarray(y_true)
    if y_arr.ndim != 1:
        raise ValueError("y_true must be a 1D vector")

    n_rows = len(y_arr)
    model_names = sorted(list(oof_predictions.keys()))

    for name in model_names:
        p_vec = oof_predictions[name]
        validate_probabilities(p_vec)
        if len(p_vec) != n_rows:
            raise ValueError(
                f"OOF prediction vector for '{name}' length ({len(p_vec)}) "
                f"does not match y_true length ({n_rows})"
            )

    if weight_grid is None:
        weight_grid = generate_weight_grid(model_names)

    single_model_rocs: dict[str, float] = {}
    for name in model_names:
        single_metrics = classification_metrics(y_arr, oof_predictions[name])
        single_model_rocs[name] = single_metrics["roc_auc"]

    best_single_model = max(single_model_rocs, key=single_model_rocs.get)
    best_single_roc_auc = single_model_rocs[best_single_model]

    records: list[dict[str, Any]] = []

    for idx, weights in enumerate(weight_grid):
        blended_oof = weighted_blend(oof_predictions, weights)
        metrics = classification_metrics(y_arr, blended_oof)

        total_w = sum(weights.values())
        norm_w = {m: round(weights[m] / total_w, 6) for m in model_names}

        non_zero_models = [m for m, w in norm_w.items() if w > 0]
        is_single = len(non_zero_models) == 1
        is_equal = len(set(norm_w.values())) == 1

        lift = metrics["roc_auc"] - best_single_roc_auc

        rec = {
            "blend_id": f"blend_{idx:04d}",
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "log_loss": metrics["log_loss"],
            "lift_over_best_single": round(lift, 8),
            "is_single_model": is_single,
            "is_equal_weights": is_equal,
            "n_active_models": len(non_zero_models),
            "weights_json": json.dumps(norm_w, sort_keys=True),
        }
        for m in model_names:
            rec[f"weight_{m}"] = norm_w[m]

        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values(
        by=["roc_auc", "log_loss", "average_precision"],
        ascending=[False, True, False],
        kind="stable",
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def materialize_hpo_finalist_predictions(
    tuning_study_dir: Path | str,
    rank: int = 1,
    project_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    target_canonical_dir: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Materialize full-protocol OOF and test predictions for a rank-1 HPO finalist."""
    root_p = Path(project_root) if project_root is not None else Path.cwd()
    study_p = Path(tuning_study_dir)
    if not study_p.is_absolute():
        study_p = (root_p / study_p).resolve()

    meta_file = study_p / "metadata.json"
    finalists_file = study_p / "finalists.parquet"
    fold_metrics_file = study_p / "finalist_fold_metrics.parquet"

    if not meta_file.exists() or not finalists_file.exists() or not fold_metrics_file.exists():
        raise FileNotFoundError(f"Incomplete tuning study artifacts in {study_p}")

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("smoke", False):
        msg = f"HPO study '{meta.get('study_name')}' in smoke mode; cannot use for final ensembling"
        raise ValueError(msg)

    study_name = validate_candidate_id(meta["study_name"])
    model_family = meta["model_name"]

    base_cfg_path = root_p / "configs" / "base.yaml"
    cfg = load_config(base_cfg_path, project_root=root_p)
    expected_data_sha = compute_dataset_sha256(cfg.data.train, cfg.data.test)
    manifest_p = (cfg.outputs.root / "splits" / "repeated_stratified_5x3.parquet").resolve()
    expected_split_sha = _compute_file_sha256(manifest_p)

    validate_candidate_provenance(
        meta.get("provenance", {}),
        expected_data_sha=expected_data_sha,
        expected_split_sha=expected_split_sha,
        expected_model_name=model_family,
    )

    finalists_df = pd.read_parquet(finalists_file)
    finalist_rows = finalists_df[finalists_df["finalist_rank"] == rank]
    if len(finalist_rows) == 0:
        raise ValueError(f"Finalist rank {rank} not found in {finalists_file}")

    fin_rec = finalist_rows.iloc[0].to_dict()
    trial_num = int(fin_rec["trial_number"])

    candidate_id = validate_candidate_id(f"{study_name}_trial{trial_num}")

    expected_total_folds = cfg.validation.n_splits * cfg.validation.n_repeats
    expected_fold_ids = list(range(expected_total_folds))

    fold_metrics_df = pd.read_parquet(fold_metrics_file)
    validate_finalist_fold_metrics(
        fold_metrics_df, trial_number=trial_num, expected_fold_ids=expected_fold_ids
    )

    params: dict[str, Any] = {}
    for k, v in fin_rec.items():
        if k.startswith("param_"):
            p_name = k[6:]
            if pd.notna(v):
                if isinstance(v, (np.integer, int)):
                    params[p_name] = int(v)
                elif isinstance(v, (np.floating, float)):
                    params[p_name] = float(v)
                else:
                    params[p_name] = v

    if not params and model_family != "dummy":
        raise ValueError(f"No tuned parameters found for finalist trial {trial_num}")

    train_df, test_df = load_frames(cfg.data.train, cfg.data.test)
    contract = DatasetContract(id_column=cfg.data.id_column, target_column=cfg.data.target_column)
    validate_frames(train_df, test_df, contract)
    y_encoded = encode_target(train_df[cfg.data.target_column], contract)

    manifest = pd.read_parquet(manifest_p)
    validate_split_manifest(
        manifest=manifest,
        n_rows=len(train_df),
        n_splits=cfg.validation.n_splits,
        n_repeats=cfg.validation.n_repeats,
        y=y_encoded,
    )

    X_train_raw = train_df.drop(columns=[cfg.data.id_column, cfg.data.target_column])
    X_test_raw = test_df.drop(columns=[cfg.data.id_column])

    X_train = apply_feature_policy("base", X_train_raw)
    X_test = apply_feature_policy("base", X_test_raw)

    oof_row_indices: list[int] = []
    oof_repeats: list[int] = []
    oof_folds: list[int] = []
    oof_probs: list[float] = []

    fold_groups = list(manifest.groupby(["repeat", "fold"], sort=True))
    n_folds_total = len(fold_groups)
    test_preds_matrix = np.zeros((len(test_df), n_folds_total), dtype=float)
    all_indices = np.arange(len(train_df))

    for fold_idx, ((repeat, fold), fold_df_group) in enumerate(fold_groups):
        val_idx = fold_df_group["row_index"].to_numpy()
        train_idx = np.setdiff1d(all_indices, val_idx)

        X_tr = X_train.iloc[train_idx]
        y_tr = y_encoded[train_idx]
        X_va = X_train.iloc[val_idx]

        if dry_run:
            model: Any = DummyClassifier(strategy="prior")
            model.fit(X_tr, y_tr)
        else:
            fold_seed = cfg.project.seed + int(repeat) * 100 + int(fold)
            model = build_model(
                model_name=model_family,
                seed=fold_seed,
                sample_frame=X_tr,
                params=params,
            )
            model.fit(X_tr, y_tr)

        val_p = model.predict_proba(X_va)[:, 1]
        validate_probabilities(val_p)

        test_p = model.predict_proba(X_test)[:, 1]
        validate_probabilities(test_p)

        oof_row_indices.extend(val_idx.tolist())
        oof_repeats.extend([int(repeat)] * len(val_idx))
        oof_folds.extend([int(fold)] * len(val_idx))
        oof_probs.extend(val_p.tolist())

        test_preds_matrix[:, fold_idx] = test_p

    oof_df = pd.DataFrame(
        {
            "row_index": oof_row_indices,
            "repeat": oof_repeats,
            "fold": oof_folds,
            "probability": oof_probs,
        }
    )
    oof_df = oof_df.sort_values(["repeat", "fold", "row_index"], kind="stable").reset_index(
        drop=True
    )

    test_pred_df = pd.DataFrame(
        {
            cfg.data.id_column: test_df[cfg.data.id_column],
            cfg.data.target_column: test_preds_matrix.mean(axis=1),
        }
    )

    if output_dir is not None:
        target_root = Path(output_dir).resolve()
        if str(target_root).endswith(candidate_id):
            cand_dir = target_root
            containment_base = target_root.parent
        else:
            cand_dir = target_root / candidate_id
            containment_base = target_root
    else:
        dest_root = "ensemble_smoke" if dry_run else "ensemble"
        containment_base = (root_p / "outputs" / dest_root / "candidates").resolve()
        cand_dir = containment_base / candidate_id

    try:
        cand_dir.resolve().relative_to(containment_base)
    except ValueError as exc:
        raise ValueError(
            f"Candidate output path ({cand_dir}) escapes target root ({containment_base})"
        ) from exc

    cand_dir.mkdir(parents=True, exist_ok=True)

    oof_path = cand_dir / "oof_predictions.parquet"
    test_path = cand_dir / "test_predictions.parquet"
    meta_path = cand_dir / "candidate_metadata.json"

    oof_df.to_parquet(oof_path)
    test_pred_df.to_parquet(test_path)

    if target_canonical_dir is not None:
        canon_cand_p = Path(target_canonical_dir)
        canon_cand_dir = (
            canon_cand_p.resolve()
            if canon_cand_p.is_absolute()
            else (root_p / canon_cand_p).resolve()
        )
        canon_oof = canon_cand_dir / "oof_predictions.parquet"
        canon_test = canon_cand_dir / "test_predictions.parquet"
    else:
        # Always record the final canonical location, never the staging dir.
        # Canonical path: outputs/ensemble/candidates/<candidate_id>/
        canonical_cand_dir = (
            root_p / "outputs" / "ensemble" / "candidates" / candidate_id
        ).resolve()
        canon_oof = canonical_cand_dir / "oof_predictions.parquet"
        canon_test = canonical_cand_dir / "test_predictions.parquet"

    try:
        rel_oof = canon_oof.relative_to(root_p.resolve()).as_posix()
        rel_test = canon_test.relative_to(root_p.resolve()).as_posix()
    except ValueError:
        rel_oof = os.path.relpath(canon_oof, root_p).replace("\\", "/")
        rel_test = os.path.relpath(canon_test, root_p).replace("\\", "/")

    oof_checksum = _compute_file_sha256(oof_path)
    test_checksum = _compute_file_sha256(test_path)

    cand_meta = {
        "run_id": candidate_id,
        "candidate_id": candidate_id,
        "status": "smoke_completed" if dry_run else "completed",
        "protocol": "smoke" if dry_run else "full",
        "effective_n_splits": cfg.validation.n_splits,
        "effective_n_repeats": cfg.validation.n_repeats,
        "effective_n_rows": len(train_df),
        "evaluated_fold_ids": expected_fold_ids,
        "model_family": model_family,
        "tuning_study": study_name,
        "trial_number": trial_num,
        "finalist_rank": rank,
        "params": params,
        "seed_policy": {
            "base_seed": cfg.project.seed,
            "formula": "base_seed + repeat * 100 + fold",
        },
        "feature_policy": "base",
        "dry_run": dry_run,
        "provenance": meta.get("provenance", {}),
        "artifact_paths": {
            "oof_predictions": rel_oof,
            "test_predictions": rel_test,
        },
        "artifact_checksums": {
            "oof_predictions": oof_checksum,
            "test_predictions": test_checksum,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cand_meta, f, indent=2)

    return {
        "candidate_id": candidate_id,
        "model_family": model_family,
        "tuning_study": study_name,
        "trial_number": trial_num,
        "params": params,
        "dry_run": dry_run,
        "oof_df": oof_df,
        "test_df": test_pred_df,
        "metadata": cand_meta,
    }


def load_candidate_runs_predictions(
    run_dirs: list[Path | str], project_root: Path | str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    """Load and validate OOF and test predictions from diagnostic run directories."""
    root_p = Path(project_root) if project_root is not None else Path.cwd()

    if not run_dirs:
        raise ValueError("run_dirs list cannot be empty")

    run_payloads: list[dict[str, Any]] = []
    oof_dict: dict[str, np.ndarray] = {}
    test_dict: dict[str, np.ndarray] = {}
    oof_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}

    for r_dir in run_dirs:
        p = Path(r_dir)
        if not p.is_absolute():
            p = (root_p / p).resolve()

        metrics_file = p / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Run metrics file not found: {metrics_file}")

        payload = load_run_metrics(metrics_file)
        run_payloads.append(payload)

    assert_comparable_runs(run_payloads)

    base_cfg_path = root_p / "configs" / "base.yaml"
    cfg = load_config(base_cfg_path, project_root=root_p)
    train_df, test_df = load_frames(cfg.data.train, cfg.data.test)
    contract = DatasetContract(id_column=cfg.data.id_column, target_column=cfg.data.target_column)
    validate_frames(train_df, test_df, contract)
    y_true = encode_target(train_df[cfg.data.target_column], contract)

    for p_idx, payload in enumerate(run_payloads):
        r_dir = Path(run_dirs[p_idx])
        if not r_dir.is_absolute():
            r_dir = (root_p / r_dir).resolve()

        run_id = payload["run_id"]
        oof_path = r_dir / "oof_predictions.parquet"
        test_path = r_dir / "test_predictions.parquet"

        oof_df = pd.read_parquet(oof_path)
        test_df_pred = pd.read_parquet(test_path)

        oof_dfs[run_id] = oof_df
        test_dfs[run_id] = test_df_pred

        oof_mean = aggregate_repeated_oof(
            row_indices=oof_df["row_index"].to_numpy(),
            predictions=oof_df["probability"].to_numpy(),
            n_rows=len(train_df),
            n_repeats=cfg.validation.n_repeats,
        )
        validate_probabilities(oof_mean)
        oof_dict[run_id] = oof_mean

        test_p = test_df_pred[cfg.data.target_column].to_numpy()
        validate_probabilities(test_p)
        test_dict[run_id] = test_p

    validate_keyed_oof_alignment(oof_dfs)
    validate_keyed_test_alignment(test_dfs, sample_submission_df=test_df)

    return oof_dict, test_dict, y_true, run_payloads


def save_blended_oof_and_test_artifacts(
    blended_oof_df: pd.DataFrame,
    blended_test_df: pd.DataFrame,
    n_rows: int,
    n_repeats: int,
    y_true: np.ndarray | None = None,
    output_dir: Path | str = "outputs/ensemble",
) -> dict[str, Path]:
    """Save aligned blended OOF and test prediction artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    expected_oof_len = n_rows * n_repeats
    if len(blended_oof_df) != expected_oof_len:
        msg = f"Blended OOF row count ({len(blended_oof_df)}) != expected ({expected_oof_len})"
        raise ValueError(msg)

    if y_true is not None:
        y_arr = np.asarray(y_true)
        row_indices = blended_oof_df["row_index"].to_numpy()
        aligned_target = y_arr[row_indices]
        if "target" in blended_oof_df.columns:
            if not np.array_equal(blended_oof_df["target"].to_numpy(), aligned_target):
                raise ValueError("Target alignment mismatch in blended OOF DataFrame")
        else:
            blended_oof_df["target"] = aligned_target

    counts = blended_oof_df.groupby("row_index").size()
    if not (counts == n_repeats).all():
        raise ValueError(f"Every training row must have exactly {n_repeats} OOF predictions")

    oof_out = out_path / "blended_oof_predictions.parquet"
    test_out = out_path / "blended_test_predictions.parquet"

    blended_oof_df.to_parquet(oof_out)
    blended_test_df.to_parquet(test_out)

    return {
        "blended_oof_predictions": oof_out,
        "blended_test_predictions": test_out,
    }


def save_ensemble_artifacts(
    leaderboard_df: pd.DataFrame,
    metrics_payload: dict[str, Any],
    output_dir: Path | str = "outputs/ensemble",
    target_canonical_dir: Path | str | None = None,
    project_root: Path | str | None = None,
) -> dict[str, Path]:
    """Atomically save blend leaderboard parquet and metrics summary JSON artifacts."""
    root_p = Path(project_root) if project_root is not None else Path.cwd()
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = (root_p / out_path).resolve()

    out_path.mkdir(parents=True, exist_ok=True)

    lb_path = out_path / "blend_leaderboard.parquet"
    metrics_path = out_path / "blend_metrics.json"

    leaderboard_df.to_parquet(lb_path)

    if target_canonical_dir is not None:
        canon_p = Path(target_canonical_dir)
        canon_dir = canon_p.resolve() if canon_p.is_absolute() else (root_p / canon_p).resolve()
    else:
        canon_dir = out_path
    canon_lb = canon_dir / "blend_leaderboard.parquet"
    canon_metrics = canon_dir / "blend_metrics.json"

    try:
        rel_lb = canon_lb.relative_to(root_p.resolve()).as_posix()
        rel_metrics = canon_metrics.relative_to(root_p.resolve()).as_posix()
    except ValueError:
        rel_lb = os.path.relpath(canon_lb, root_p).replace("\\", "/")
        rel_metrics = os.path.relpath(canon_metrics, root_p).replace("\\", "/")

    metrics_payload["artifact_paths"] = {
        "blend_leaderboard": rel_lb,
        "blend_metrics": rel_metrics,
    }

    temp_path = metrics_path.with_name(f".{metrics_path.name}.tmp_{uuid4().hex}")
    formatted_json = json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(formatted_json)
        os.replace(temp_path, metrics_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return {
        "blend_leaderboard": lb_path,
        "blend_metrics": metrics_path,
    }


def promote_ensemble_run_transactionally(
    staging_dir: Path | str,
    output_root: Path | str,
    staged_config_path: Path | str | None = None,
    target_config_path: Path | str | None = None,
    post_promote_callback: Any | None = None,
    project_root: Path | str | None = None,
) -> None:
    """Transactionally promote staged ensemble run artifacts using whole-directory swap."""
    stage_p = Path(staging_dir).resolve()
    target_root = Path(output_root).resolve()

    if not stage_p.exists():
        raise FileNotFoundError(f"Staging directory not found: {stage_p}")

    target_existed_before = target_root.exists()
    backup_dir = target_root.parent / f".{target_root.name}_backup_{uuid4().hex[:8]}"

    staged_cfg_p = Path(staged_config_path).resolve() if staged_config_path else None
    target_cfg_p = Path(target_config_path).resolve() if target_config_path else None
    config_existed_before = target_cfg_p is not None and target_cfg_p.exists()
    backup_cfg = (
        target_cfg_p.parent / f".{target_cfg_p.name}_backup_{uuid4().hex[:8]}"
        if target_cfg_p
        else None
    )

    # 1. Prepare backups
    if target_existed_before:
        shutil.copytree(target_root, backup_dir)

    if config_existed_before and target_cfg_p and backup_cfg:
        shutil.copy2(target_cfg_p, backup_cfg)

    try:
        # 2. Whole-directory swap
        if target_root.exists():
            shutil.rmtree(target_root)

        shutil.move(str(stage_p), str(target_root))

        # 3. Config file move/promotion
        if staged_cfg_p and staged_cfg_p.exists() and target_cfg_p:
            if target_cfg_p.exists():
                target_cfg_p.unlink()
            target_cfg_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_cfg_p), str(target_cfg_p))

        # 4. Post-promotion validation & callback
        root_p = Path(project_root) if project_root is not None else Path.cwd()
        metrics_file = target_root / "blend_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    art_paths = meta.get("artifact_paths", {})
                    for _k, p_str in art_paths.items():
                        p_check = Path(p_str)
                        if not p_check.is_absolute():
                            p_check = (root_p / p_check).resolve()
                        if not p_check.exists():
                            msg = f"Post-promotion validation failed: artifact '{p_str}' missing"
                            raise FileNotFoundError(msg)
            except json.JSONDecodeError:
                pass

        cand_root = target_root / "candidates"
        if cand_root.exists():
            for c_dir in cand_root.iterdir():
                c_meta_file = c_dir / "candidate_metadata.json"
                if c_meta_file.exists():
                    try:
                        with open(c_meta_file, encoding="utf-8") as f:
                            c_meta = json.load(f)
                        if isinstance(c_meta, dict):
                            c_art_paths = c_meta.get("artifact_paths", {})
                            c_checksums = c_meta.get("artifact_checksums", {})
                            root_resolved = root_p.resolve()
                            for k, p_str in c_art_paths.items():
                                p_raw = Path(p_str)
                                # Fail closed: path must be relative (not absolute)
                                if p_raw.is_absolute():
                                    msg = (
                                        f"Post-promotion validation failed: candidate artifact "
                                        f"path '{p_str}' must be relative, not absolute"
                                    )
                                    raise ValueError(msg)
                                p_check = (root_resolved / p_raw).resolve()
                                # Fail closed: path must be contained under project root
                                try:
                                    p_check.relative_to(root_resolved)
                                except ValueError:
                                    msg = (
                                        f"Post-promotion validation failed: candidate artifact "
                                        f"'{p_str}' escapes project root"
                                    )
                                    raise ValueError(msg) from None
                                # Fail closed: path must not be a symlink
                                if p_check.is_symlink():
                                    msg = (
                                        f"Post-promotion validation failed: candidate artifact "
                                        f"'{p_str}' is a symlink"
                                    )
                                    raise ValueError(msg)
                                # Fail closed: file must exist
                                if not p_check.exists():
                                    msg = (
                                        f"Post-promotion validation failed: "
                                        f"candidate artifact '{p_str}' missing"
                                    )
                                    raise FileNotFoundError(msg)
                                # Fail closed: checksum must match
                                if k in c_checksums:
                                    expected_sha = c_checksums[k]
                                    actual_sha = _compute_file_sha256(p_check)
                                    if actual_sha != expected_sha:
                                        msg = f"Post-promotion checksum mismatch for '{p_str}'"
                                        raise ValueError(msg)
                    except json.JSONDecodeError:
                        pass

        if post_promote_callback is not None:
            post_promote_callback()

    except Exception:
        # ROLLBACK
        if target_root.exists():
            shutil.rmtree(target_root)
        if target_existed_before:
            if backup_dir.exists():
                shutil.copytree(backup_dir, target_root)
            else:
                target_root.mkdir(parents=True, exist_ok=True)

        if target_cfg_p and target_cfg_p.exists():
            target_cfg_p.unlink()
        if config_existed_before and backup_cfg and backup_cfg.exists():
            shutil.copy2(backup_cfg, target_cfg_p)

        raise
    finally:
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)
        if backup_cfg and backup_cfg.exists():
            backup_cfg.unlink()
        if staged_cfg_p and staged_cfg_p.exists():
            staged_cfg_p.unlink()
        if stage_p and stage_p.exists():
            shutil.rmtree(stage_p)


def update_final_config(
    selected_policy: str,
    winning_weights: dict[str, float],
    components_info: dict[str, Any],
    best_single_id: str,
    winning_roc_auc: float,
    lift_over_best_single: float,
    split_sha256: str,
    data_sha256: str,
    config_out_path: Path,
    project_root: Path,
) -> None:
    """Lock complete winning model/blend policy details into configs/final.yaml."""
    for cand_id, info in components_info.items():
        if (
            info.get("dry_run", False)
            or info.get("status") in ("smoke_completed", "incomplete", "failed")
            or info.get("protocol") == "smoke"
        ):
            raise ValueError(
                f"Cannot write configs/final.yaml from an incomplete/smoke run ({cand_id})"
            )

        cand_prov = info.get("provenance", {})
        if cand_prov.get("data_sha256") and cand_prov.get("data_sha256") != data_sha256:
            msg = f"data_sha256 mismatch: component {cand_id} != top-level"
            raise ValueError(msg)
        if cand_prov.get("split_sha256") and cand_prov.get("split_sha256") != split_sha256:
            msg = f"split_sha256 mismatch: component {cand_id} != top-level"
            raise ValueError(msg)

    base_cfg_path = project_root / "configs" / "base.yaml"
    with open(base_cfg_path, encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)

    source_run_ids = [
        info.get("run_id", cand_id) for cand_id, weight in winning_weights.items() if weight > 0
    ]
    if not source_run_ids:
        source_run_ids = [components_info.get(best_single_id, {}).get("run_id", best_single_id)]

    formatted_components: dict[str, Any] = {}
    for cand_id, info in components_info.items():
        weight = winning_weights.get(cand_id, 0.0)
        formatted_components[cand_id] = {
            "run_id": info.get("run_id", cand_id),
            "candidate_id": cand_id,
            "candidate_metadata_path": (
                f"outputs/ensemble/candidates/{cand_id}/candidate_metadata.json"
            ),
            "model_family": info.get("model_family", "catboost"),
            "tuning_study": info.get("tuning_study", ""),
            "trial_number": info.get("trial_number", 0),
            "params": info.get("params", {}),
            "seed_policy": {
                "base_seed": base_raw["project"]["seed"],
                "formula": "base_seed + repeat * 100 + fold",
            },
            "feature_policy": "base",
            "provenance": {
                "data_sha256": data_sha256,
                "split_sha256": split_sha256,
            },
            "weight": round(float(weight), 6),
        }

    final_yaml_content: dict[str, Any] = {
        "project": {
            "name": "kaggle_ev_purchase_final",
            "seed": base_raw["project"]["seed"],
        },
        "data": base_raw["data"],
        "validation": base_raw["validation"],
        "metrics": base_raw["metrics"],
        "outputs": base_raw["outputs"],
        "feature_policy": "base",
        "policy": {
            "type": selected_policy,
            "status": "real_materialization_completed",
            "lift_threshold": MIN_BLEND_LIFT_THRESHOLD,
            "selected_roc_auc": round(float(winning_roc_auc), 6),
            "lift_over_best_single": round(float(lift_over_best_single), 8),
            "best_single_candidate_id": best_single_id,
            "source_run_ids": source_run_ids,
            "analyzed_candidate_ids": list(components_info.keys()),
            "winning_weights": {k: round(v, 6) for k, v in winning_weights.items()},
            "components": formatted_components,
            "split_sha256": split_sha256,
            "data_sha256": data_sha256,
        },
    }

    config_out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_cfg = config_out_path.with_name(f".{config_out_path.name}.tmp_{uuid4().hex}")
    with open(temp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(final_yaml_content, f, sort_keys=False)

    os.replace(temp_cfg, config_out_path)
