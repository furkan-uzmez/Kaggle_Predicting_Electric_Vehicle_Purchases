"""Leakage-safe adversarial validation module for domain shift detection."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from ev_purchase.metrics import classification_metrics
from ev_purchase.models import GBDTCategoricalPrep


@dataclass
class AdversarialValidationResult:
    """Container for adversarial validation outputs and audit trail."""

    oof_domain_preds: np.ndarray
    y_domain: np.ndarray
    metrics: dict[str, float]
    fold_metrics: list[dict[str, Any]]
    feature_importance: pd.DataFrame
    interpretation: dict[str, str]
    used_columns: list[str]
    excluded_columns: list[str]
    row_counts: dict[str, int]
    target_prevalence: dict[str, float]
    model_params: dict[str, Any]
    cv_config: dict[str, Any]
    data_sha256: dict[str, str] | None = None
    artifact_paths: dict[str, str] | None = None


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_adversarial_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str = "id",
    target_column: str = "Will_Buy_EV",
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Combine train and test datasets with domain labels, strictly excluding id and target.

    Domain origin labels:
        train = 0
        test = 1

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        id_column: Name of identifier column to exclude.
        target_column: Name of true target column to exclude.

    Returns:
        X: Combined predictor DataFrame without id and target.
        y_domain: 1D numpy array of domain origin labels (0=train, 1=test).
        used_columns: List of predictor column names.
        excluded_columns: List of excluded column names.

    Raises:
        TypeError: If train_df or test_df are not DataFrames.
        ValueError: If feature columns mismatch between train and test.
    """
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError(f"train_df must be a pd.DataFrame, got {type(train_df).__name__}")
    if not isinstance(test_df, pd.DataFrame):
        raise TypeError(f"test_df must be a pd.DataFrame, got {type(test_df).__name__}")

    excluded_set = {id_column, target_column}

    # Extract predictors
    train_cols = [c for c in train_df.columns if c not in excluded_set]
    test_cols = [c for c in test_df.columns if c not in excluded_set]

    if set(train_cols) != set(test_cols):
        missing_in_test = sorted(set(train_cols) - set(test_cols))
        missing_in_train = sorted(set(test_cols) - set(train_cols))
        raise ValueError(
            f"Feature column mismatch between train and test. "
            f"Missing in test: {missing_in_test}; Missing in train: {missing_in_train}"
        )

    used_columns = sorted(train_cols)
    excluded_columns = sorted(
        (set(train_df.columns) | set(test_df.columns)) & excluded_set
    )

    X_train = train_df[used_columns].copy()
    X_test = test_df[used_columns].copy()

    # Assert contract: id and target MUST NOT be in predictors
    assert id_column not in X_train.columns, (
        f"Leakage violation: {id_column} in train predictors"
    )
    assert id_column not in X_test.columns, (
        f"Leakage violation: {id_column} in test predictors"
    )
    assert target_column not in X_train.columns, (
        f"Leakage violation: {target_column} in train predictors"
    )
    assert target_column not in X_test.columns, (
        f"Leakage violation: {target_column} in test predictors"
    )

    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_domain = np.concatenate(
        [
            np.zeros(len(train_df), dtype=int),
            np.ones(len(test_df), dtype=int),
        ]
    )

    return X, y_domain, used_columns, excluded_columns


def get_interpretation_bands(auc: float) -> dict[str, str]:
    """Provide structured diagnostic interpretation bands for adversarial ROC-AUC.

    Interpretation bands:
        ~0.50: little detectable shift (<= 0.55)
        0.55-0.60: mild shift
        >0.60: material diagnostic shift
    """
    if auc <= 0.55:
        shift_band = "little detectable shift"
        summary = (
            f"Adversarial ROC-AUC is {auc:.4f} (<= 0.55), indicating little detectable "
            f"covariate shift between train and test under this classifier and feature set."
        )
        action = (
            "Trust local cross-validation for model selection without domain adaptation; "
            "no adversarial sample reweighting or feature pruning is required."
        )
    elif auc <= 0.60:
        shift_band = "mild shift"
        summary = (
            f"Adversarial ROC-AUC is {auc:.4f} (0.55-0.60), indicating mild shift in "
            f"predictor distributions between train and test."
        )
        action = (
            "Monitor local CV against leaderboard feedback closely; inspect top shift "
            "features and test sensitivity to feature dropping."
        )
    else:
        shift_band = "material diagnostic shift"
        summary = (
            f"Adversarial ROC-AUC is {auc:.4f} (> 0.60), indicating material diagnostic "
            f"covariate shift between train and test."
        )
        action = (
            "Investigate domain shift before trusting CV; consider adversarial sample "
            "reweighting or pruning top domain-distinguishing features."
        )

    caveats = (
        "Feature importances indicate statistical association with domain origin "
        "(shift detection), NOT causality. Interpretation bands (~0.50 little shift, "
        "0.55-0.60 mild, >0.60 material) are diagnostic heuristics, not universal thresholds."
    )

    return {
        "shift_band": shift_band,
        "summary": summary,
        "action": action,
        "caveats": caveats,
    }


def run_adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str = "id",
    target_column: str = "Will_Buy_EV",
    n_splits: int = 5,
    random_state: int = 42,
    lgb_params: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = None,
    artifacts_dir: Path | str | None = None,
    data_sha256: dict[str, str] | None = None,
    source_generator_paths: list[str] | None = None,
) -> AdversarialValidationResult:
    """Run leakage-safe 5-fold adversarial validation classifying train (0) vs test (1).

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        id_column: Column name for ID (strictly excluded).
        target_column: Column name for true target (strictly excluded).
        n_splits: Number of cross-validation splits (default 5).
        random_state: Seed for StratifiedKFold and LightGBM.
        lgb_params: Hyperparameters for LGBMClassifier.
        early_stopping_rounds: Optional early stopping rounds.
        artifacts_dir: Directory to persist audit artifacts if specified.
        data_sha256: Optional precomputed data SHA-256 dictionary.
        source_generator_paths: Optional list of generator paths for audit trail.

    Returns:
        AdversarialValidationResult containing complete OOF probabilities, metrics,
        feature importance, and interpretation metadata.
    """
    X, y_domain, used_columns, excluded_columns = prepare_adversarial_data(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_domain_preds = np.full(len(X), fill_value=np.nan, dtype=float)
    assignment_counts = np.zeros(len(X), dtype=int)

    # Conservative, deterministic CPU settings
    base_lgb_params: dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if lgb_params:
        base_lgb_params.update(lgb_params)

    fold_metrics: list[dict[str, Any]] = []
    fold_importances: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_domain)):
        X_tr_raw = X.iloc[train_idx]
        y_tr = y_domain[train_idx]
        X_va_raw = X.iloc[val_idx]
        y_va = y_domain[val_idx]

        # Fold-local categorical preprocessing: fit ONLY on train split
        prep = GBDTCategoricalPrep()
        prep.fit(X_tr_raw)
        X_tr = prep.transform(X_tr_raw)
        X_va = prep.transform(X_va_raw)

        clf = LGBMClassifier(**base_lgb_params)

        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            import lightgbm as lgb_mod

            cb = [lgb_mod.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=cb,
            )
        else:
            clf.fit(X_tr, y_tr)

        val_probs = clf.predict_proba(X_va)[:, 1]
        oof_domain_preds[val_idx] = val_probs
        assignment_counts[val_idx] += 1

        metrics_fold = classification_metrics(y_va, val_probs)
        metrics_fold["fold"] = fold_idx
        metrics_fold["train_size"] = int(len(train_idx))
        metrics_fold["val_size"] = int(len(val_idx))
        fold_metrics.append(metrics_fold)

        fold_importances.append(clf.feature_importances_)

    # Assert complete, uncorrupted OOF coverage: exactly 1 validation assignment per row, no NaNs
    assert not np.isnan(oof_domain_preds).any(), (
        "Incomplete OOF predictions: unassigned NaNs detected"
    )
    assert (assignment_counts == 1).all(), (
        f"Validation assignment violation: "
        f"min={assignment_counts.min()}, max={assignment_counts.max()}"
    )

    # Overall OOF classification metrics
    overall_metrics = classification_metrics(y_domain, oof_domain_preds)
    fold_aucs = [float(f["roc_auc"]) for f in fold_metrics]
    overall_metrics["fold_roc_auc_mean"] = float(np.mean(fold_aucs))
    overall_metrics["fold_roc_auc_std"] = float(np.std(fold_aucs))
    overall_metrics["baseline_prevalence"] = float(np.mean(y_domain))

    # Feature importance aggregation
    importances_matrix = np.array(fold_importances)  # shape: (n_splits, n_features)
    imp_mean = np.mean(importances_matrix, axis=0)
    imp_std = np.std(importances_matrix, axis=0)

    feature_importance_df = pd.DataFrame(
        {
            "feature": used_columns,
            "importance_mean": imp_mean,
            "importance_std": imp_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    interpretation = get_interpretation_bands(overall_metrics["roc_auc"])

    # Target prevalence in train set if target column is present
    train_target_prevalence: float | None = None
    if target_column in train_df.columns:
        target_series = train_df[target_column]
        if set(target_series.dropna().unique()).issubset({"No", "Yes"}):
            train_target_prevalence = float((target_series == "Yes").mean())
        elif pd.api.types.is_numeric_dtype(target_series):
            train_target_prevalence = float(target_series.mean())

    target_prevalence_dict = {
        "domain_test_prevalence": float(np.mean(y_domain)),
    }
    if train_target_prevalence is not None:
        target_prevalence_dict["train_target_positive_rate"] = train_target_prevalence

    row_counts = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_rows": int(len(X)),
    }

    cv_config = {
        "strategy": "StratifiedKFold",
        "n_splits": int(n_splits),
        "shuffle": True,
        "random_state": int(random_state),
    }

    artifact_paths_dict: dict[str, str] | None = None

    if artifacts_dir is not None:
        art_dir = Path(artifacts_dir)
        art_dir.mkdir(parents=True, exist_ok=True)

        json_path = art_dir / "adversarial_validation.json"
        fold_csv_path = art_dir / "adversarial_fold_metrics.csv"
        feat_csv_path = art_dir / "adversarial_feature_importance.csv"

        # Format fold metrics DataFrame
        df_fold_metrics = pd.DataFrame(fold_metrics)[
            ["fold", "roc_auc", "average_precision", "log_loss", "train_size", "val_size"]
        ]
        df_fold_metrics.to_csv(fold_csv_path, index=False)

        # Feature importance CSV
        feature_importance_df.to_csv(feat_csv_path, index=False)

        # Prepare JSON payload
        source_paths = source_generator_paths or [
            "src/ev_purchase/adversarial_validation.py",
            "scripts/run_adversarial_validation.py",
            "scripts/build_notebooks.py",
        ]

        json_payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "metrics": overall_metrics,
            "fold_metrics": fold_metrics,
            "row_counts": row_counts,
            "target_prevalence": target_prevalence_dict,
            "used_columns": used_columns,
            "excluded_columns": excluded_columns,
            "model_params": {k: v for k, v in base_lgb_params.items() if not callable(v)},
            "cv_config": cv_config,
            "data_sha256": data_sha256 or {},
            "source_generator_paths": source_paths,
            "interpretation": interpretation,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, sort_keys=True)
            f.write("\n")

        artifact_paths_dict = {
            "adversarial_validation_json": str(json_path),
            "adversarial_fold_metrics_csv": str(fold_csv_path),
            "adversarial_feature_importance_csv": str(feat_csv_path),
        }

    return AdversarialValidationResult(
        oof_domain_preds=oof_domain_preds,
        y_domain=y_domain,
        metrics=overall_metrics,
        fold_metrics=fold_metrics,
        feature_importance=feature_importance_df,
        interpretation=interpretation,
        used_columns=used_columns,
        excluded_columns=excluded_columns,
        row_counts=row_counts,
        target_prevalence=target_prevalence_dict,
        model_params=base_lgb_params,
        cv_config=cv_config,
        data_sha256=data_sha256,
        artifact_paths=artifact_paths_dict,
    )
