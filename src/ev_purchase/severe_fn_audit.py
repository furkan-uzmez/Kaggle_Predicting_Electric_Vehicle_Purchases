"""Severe false-negative and label-noise learnability audit module.

Analyzes out-of-fold ensemble predictions to characterize severe false negatives,
top-loss examples, local label mixing, and coarse-cell enrichment without
asserting unverified causal label noise claims.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ev_purchase.adversarial_validation import compute_file_sha256

WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class SevereFnAuditResult:
    """Structured container for audit summary and artifact metadata."""

    total_rows: int
    severe_fn_count: int
    severe_fn_rate: float
    top_loss_count: int
    top_loss_threshold: float
    margin_band_25_75_count: int
    margin_band_40_60_count: int
    model_consistency: dict[str, float]
    variance_summary: dict[str, float]
    neighborhood_summary: dict[str, float]
    artifact_paths: dict[str, str]
    artifact_checksums: dict[str, str]
    source_provenance: dict[str, str]


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute standard Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Total number of trials.
        confidence: Confidence level (default 0.95).

    Returns:
        tuple (lower, upper) bounded in [0.0, 1.0].
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must be between 0 and n ({n}), got {k}")
    if n == 0:
        return (0.0, 0.0)

    # Use standard normal quantile approximation for confidence levels
    if math.isclose(confidence, 0.95):
        z = WILSON_Z_95
    else:
        # Approximate normal quantile
        # Fallback if other confidence level is requested
        from scipy.stats import norm

        z = float(norm.ppf((1.0 + confidence) / 2.0))

    p = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))

    lower = 0.0 if k == 0 else max(0.0, center - margin)
    upper = 1.0 if k == n else min(1.0, center + margin)
    return (lower, upper)


def compute_sample_log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Compute per-row binary log-loss with clipping.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        p: Predicted probabilities.
        eps: Numerical clipping bound.

    Returns:
        1D numpy array of sample log loss values.
    """
    p_clipped = np.clip(p, eps, 1.0 - eps)
    y_arr = np.asarray(y_true, dtype=float)
    return -(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log(1.0 - p_clipped))


def verify_and_align_oof_candidates(
    lgb_oof: pd.DataFrame,
    cb_oof: pd.DataFrame,
    expected_repeats: int = 3,
    expected_folds: int = 5,
    expected_n_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify OOF prediction artifacts, ensure 3-repeat coverage, and align row keys.

    Args:
        lgb_oof: LightGBM out-of-fold predictions.
        cb_oof: CatBoost out-of-fold predictions.
        expected_repeats: Expected number of repeats (default 3).
        expected_folds: Expected folds per repeat (default 5).

    Returns:
        Tuple of sorted, perfectly aligned DataFrames (lgb_sorted, cb_sorted).

    Raises:
        ValueError: If formats, repeats, row counts, or key alignments fail.
    """
    req_cols = {"row_index", "repeat", "fold", "probability"}
    if not req_cols.issubset(lgb_oof.columns):
        raise ValueError(f"lgb_oof missing required columns: {req_cols - set(lgb_oof.columns)}")
    if not req_cols.issubset(cb_oof.columns):
        raise ValueError(f"cb_oof missing required columns: {req_cols - set(cb_oof.columns)}")

    if len(lgb_oof) != len(cb_oof):
        raise ValueError(f"Mismatched OOF row count: lgb={len(lgb_oof)}, cb={len(cb_oof)}")

    expected_repeat_ids = list(range(expected_repeats))
    expected_fold_ids = list(range(expected_folds))

    def validate_frame(frame: pd.DataFrame, name: str) -> None:
        probabilities = frame["probability"]
        if not pd.api.types.is_numeric_dtype(probabilities):
            raise ValueError(f"Probabilities must be numeric in {name}")
        probability_values = probabilities.to_numpy(dtype=float)
        if not np.isfinite(probability_values).all():
            raise ValueError(f"Probabilities must be finite in {name}")
        if ((probability_values < 0.0) | (probability_values > 1.0)).any():
            raise ValueError(f"Probabilities must be in [0, 1] in {name}")

        for column in ("row_index", "repeat", "fold"):
            values = frame[column]
            if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
                raise ValueError(f"OOF key '{column}' must contain finite integers in {name}")
            numeric_values = values.to_numpy(dtype=float)
            if not np.isfinite(numeric_values).all() or not np.equal(
                numeric_values, np.floor(numeric_values)
            ).all():
                raise ValueError(f"OOF key '{column}' must contain finite integers in {name}")

        repeat_ids = sorted(frame["repeat"].astype(int).unique().tolist())
        if repeat_ids != expected_repeat_ids:
            raise ValueError(
                f"Repeat IDs mismatch in {name}: expected {expected_repeat_ids}, got {repeat_ids}"
            )
        fold_ids = sorted(frame["fold"].astype(int).unique().tolist())
        if fold_ids != expected_fold_ids:
            raise ValueError(
                f"Fold IDs mismatch in {name}: expected {expected_fold_ids}, got {fold_ids}"
            )

        keys = frame[["row_index", "repeat"]]
        if keys.duplicated().any():
            raise ValueError(f"Duplicate row keys (row_index, repeat) found in {name}")

        row_ids = set(frame["row_index"].astype(int).unique().tolist())
        if expected_n_rows is not None and row_ids != set(range(expected_n_rows)):
            raise ValueError(
                f"Row coverage incomplete in {name}: expected row_index 0..{expected_n_rows - 1}"
            )
        for repeat_id in expected_repeat_ids:
            repeat_rows = set(
                frame.loc[frame["repeat"] == repeat_id, "row_index"].astype(int).tolist()
            )
            if repeat_rows != row_ids:
                raise ValueError(
                    f"Row key mismatch: Row coverage incomplete in {name} for repeat {repeat_id}"
                )

    validate_frame(lgb_oof, "lgb_oof")
    validate_frame(cb_oof, "cb_oof")

    # Sort deterministically
    sort_cols = ["repeat", "fold", "row_index"]
    lgb_sorted = lgb_oof.sort_values(sort_cols).reset_index(drop=True)
    cb_sorted = cb_oof.sort_values(sort_cols).reset_index(drop=True)

    # Verify identical keys in exact alignment
    if not (
        lgb_sorted["row_index"].equals(cb_sorted["row_index"])
        and lgb_sorted["repeat"].equals(cb_sorted["repeat"])
        and lgb_sorted["fold"].equals(cb_sorted["fold"])
    ):
        raise ValueError("Row key mismatch between LightGBM and CatBoost OOF frames")

    return lgb_sorted, cb_sorted


def reconstruct_ensemble_oof(
    lgb_oof: pd.DataFrame,
    cb_oof: pd.DataFrame,
    lgb_weight: float = 0.70,
    cb_weight: float = 0.30,
    expected_repeats: int = 3,
    expected_folds: int = 5,
) -> pd.DataFrame:
    """Reconstruct repeat-level and row-mean ensemble predictions.

    Args:
        lgb_oof: LightGBM OOF predictions.
        cb_oof: CatBoost OOF predictions.
        lgb_weight: Ensemble weight for LightGBM (default 0.70).
        cb_weight: Ensemble weight for CatBoost (default 0.30).
        expected_repeats: Expected CV repeats (default 3).
        expected_folds: Expected CV folds (default 5).

    Returns:
        DataFrame aggregated by row_index with means, repeat std/min/max, and consistency flags.
    """
    lgb_sorted, cb_sorted = verify_and_align_oof_candidates(
        lgb_oof, cb_oof, expected_repeats=expected_repeats, expected_folds=expected_folds
    )

    merged = pd.DataFrame(
        {
            "row_index": lgb_sorted["row_index"],
            "repeat": lgb_sorted["repeat"],
            "fold": lgb_sorted["fold"],
            "p_lgb": lgb_sorted["probability"],
            "p_cb": cb_sorted["probability"],
        }
    )
    merged["p_canonical"] = merged["p_lgb"]
    merged["p_exploratory_30_70"] = lgb_weight * merged["p_lgb"] + cb_weight * merged["p_cb"]

    # Aggregate by row_index
    grouped = merged.groupby("row_index", as_index=False)
    agg_df = grouped.agg(
        p_lgb_mean=("p_lgb", "mean"),
        p_cb_mean=("p_cb", "mean"),
        p_canonical_mean=("p_canonical", "mean"),
        p_exploratory_30_70_mean=("p_exploratory_30_70", "mean"),
        p_lgb_std=("p_lgb", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        p_cb_std=("p_cb", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        p_canonical_std=(
            "p_canonical", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
        ),
        p_exploratory_30_70_std=(
            "p_exploratory_30_70", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
        ),
        p_lgb_min=("p_lgb", "min"),
        p_lgb_max=("p_lgb", "max"),
        p_cb_min=("p_cb", "min"),
        p_cb_max=("p_cb", "max"),
        p_canonical_min=("p_canonical", "min"),
        p_canonical_max=("p_canonical", "max"),
        p_exploratory_30_70_min=("p_exploratory_30_70", "min"),
        p_exploratory_30_70_max=("p_exploratory_30_70", "max"),
    )

    agg_df["cb_lgb_abs_diff"] = (agg_df["p_cb_mean"] - agg_df["p_lgb_mean"]).abs()
    agg_df["lgb_all_repeats_low_10"] = agg_df["p_lgb_max"] <= 0.10
    agg_df["cb_all_repeats_low_10"] = agg_df["p_cb_max"] <= 0.10
    agg_df["both_all_repeats_low_10"] = (
        agg_df["lgb_all_repeats_low_10"] & agg_df["cb_all_repeats_low_10"]
    )
    agg_df["canonical_all_repeats_low_10"] = agg_df["p_canonical_max"] <= 0.10
    agg_df["exploratory_30_70_all_repeats_low_10"] = (
        agg_df["p_exploratory_30_70_max"] <= 0.10
    )

    return agg_df.sort_values("row_index").reset_index(drop=True)


def select_deterministic_top_loss(
    df: pd.DataFrame,
    quantile: float = 0.99,
    n_top: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Select top sample log-loss rows using deterministic tie-breaking policy.

    Sorting is strictly descending by sample_log_loss, with secondary sort
    ascending by row_index to break any boundary ties deterministically.

    Args:
        df: DataFrame containing 'sample_log_loss' and 'row_index'.
        quantile: Quantile threshold for top loss (default 0.99 for top 1%).
        n_top: Optional exact count of top rows to select.

    Returns:
        tuple (top_rows_df, quantile_value).
    """
    losses = df["sample_log_loss"].to_numpy()
    q_val = float(np.quantile(losses, quantile))

    total_n = len(df)
    if n_top is None:
        n_top = int(math.ceil((1.0 - quantile) * total_n))

    # Deterministic full ranking: descending log loss, ascending row_index
    sorted_df = df.sort_values(by=["sample_log_loss", "row_index"], ascending=[False, True]).copy()

    sorted_df["loss_rank"] = np.arange(1, total_n + 1, dtype=int)
    sorted_df["loss_percentile"] = 1.0 - (sorted_df["loss_rank"] - 1.0) / total_n

    top_df = sorted_df.iloc[:n_top].copy()
    return top_df, q_val


def prepare_neighborhood_features(
    train_df: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "Will_Buy_EV",
) -> tuple[np.ndarray, list[str]]:
    """Prepare standardized and one-hot encoded features strictly excluding target/ids.

    Args:
        train_df: Full training DataFrame.
        id_col: ID column name to exclude.
        target_col: Target column name to exclude.

    Returns:
        tuple (feature_matrix as float32 np.ndarray, used_feature_column_names).
    """
    excluded_set = {
        id_col,
        target_col,
        "row_index",
        "sample_log_loss",
        "loss_rank",
        "loss_percentile",
        "y_true",
        "is_severe_fn",
    }
    audit_prefixes = (
        "p_",
        "probability",
        "prob_",
        "neighbor_",
        "lgb_all_repeats_",
        "cb_all_repeats_",
        "both_all_repeats_",
        "canonical_all_repeats_",
        "exploratory_30_70_all_repeats_",
    )
    feature_cols = [
        c
        for c in train_df.columns
        if c not in excluded_set and not str(c).lower().startswith(audit_prefixes)
    ]

    num_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c])
        and not isinstance(train_df[c].dtype, pd.CategoricalDtype)
    ]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
                cat_cols,
            ),
        ]
    )

    X_mat = preprocessor.fit_transform(train_df[feature_cols]).astype(np.float32)
    return X_mat, feature_cols


def compute_neighborhood_mixing(
    X_all: np.ndarray,
    y_true: np.ndarray,
    query_indices: np.ndarray,
    n_neighbors: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute nearest-neighbor label mixing for query rows.

    Leakage-safe: excludes the actual query row by index rather than assuming
    the first returned neighbor is self (ties from duplicate feature vectors
    can change nearest-neighbor ordering).

    Args:
        X_all: Feature matrix for all training points (N x D).
        y_true: 1D array of binary ground truth labels for all training points.
        query_indices: Indices in X_all of rows to query.
        n_neighbors: Number of distinct neighbors to inspect (default 10).

    Returns:
        tuple (neighbor_pos_frac, neighbor_mean_dist, neighbor_disagreement).
    """
    if len(query_indices) == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    # Fit KDTree on all points
    # Query n_neighbors + 1 to account for self-match
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="kd_tree", n_jobs=-1)
    nn.fit(X_all)

    distances, indices = nn.kneighbors(X_all[query_indices])

    neighbor_indices = np.empty((len(query_indices), n_neighbors), dtype=int)
    neighbor_distances = np.empty((len(query_indices), n_neighbors), dtype=float)
    for output_row, query_index in enumerate(query_indices):
        keep = indices[output_row] != query_index
        kept_indices = indices[output_row][keep][:n_neighbors]
        kept_distances = distances[output_row][keep][:n_neighbors]
        if len(kept_indices) != n_neighbors:
            raise ValueError(
                "Not enough non-self neighbors; reduce n_neighbors or provide more rows"
            )
        neighbor_indices[output_row] = kept_indices
        neighbor_distances[output_row] = kept_distances

    neighbor_y = y_true[neighbor_indices]
    neighbor_pos_frac = np.mean(neighbor_y, axis=1)
    neighbor_mean_dist = np.mean(neighbor_distances, axis=1)

    query_y = y_true[query_indices]
    neighbor_disagreement = np.abs(query_y - neighbor_pos_frac)

    return neighbor_pos_frac, neighbor_mean_dist, neighbor_disagreement


def evaluate_predeclared_slices(
    train_df: pd.DataFrame, *, return_predicates: bool = False
) -> dict[str, pd.Series] | tuple[dict[str, pd.Series], dict[str, dict[str, Any]]]:
    """Compute boolean masks for all predeclared slices and Concern x Subsidy cells.

    Args:
        train_df: Training DataFrame with actual column names.

    Returns:
        dict of slice_name -> pd.Series of boolean masks.
    """
    slices: dict[str, pd.Series] = {}
    predicates: dict[str, dict[str, Any]] = {}

    # 1. Concern x Subsidy cells
    concerns = [1.0, 2.0, 3.0, 4.0, 5.0]
    subsidies = ["No", "Yes"]
    for c in concerns:
        for s in subsidies:
            name = f"concern_x_subsidy__concern_{c}_subsidy_{s}"
            mask = (train_df["Environmental_Concern_Level"] == c) & (
                train_df["Subsidy_Available"] == s
            )
            slices[name] = mask
            predicates[name] = {
                "op": "all",
                "predicates": [
                    {"op": "eq", "column": "Environmental_Concern_Level", "value": c},
                    {"op": "eq", "column": "Subsidy_Available", "value": s},
                ],
            }

    # 2. Slices discovered in the prior OOF exploration and frozen now for
    # prospective comparisons. They were not preregistered before that analysis.
    slices["rural"] = train_df["City_Type"] == "Rural"
    predicates["rural"] = {"op": "eq", "column": "City_Type", "value": "Rural"}
    slices["workplace_charging_0"] = train_df["Charging_Stations_Near_Work"] == 0
    predicates["workplace_charging_0"] = {
        "op": "eq",
        "column": "Charging_Stations_Near_Work",
        "value": 0,
    }
    slices["home_charging_yes_stations_1_2"] = (train_df["Home_Charging_Possible"] == "Yes") & (
        train_df["Charging_Stations_Near_Home"].between(1, 2)
    )
    predicates["home_charging_yes_stations_1_2"] = {
        "op": "all",
        "predicates": [
            {"op": "eq", "column": "Home_Charging_Possible", "value": "Yes"},
            {
                "op": "between",
                "column": "Charging_Stations_Near_Home",
                "bounds": [1, 2],
            },
        ],
    }
    slices["age_34_43_income_78915_91619"] = (train_df["Age"].between(34, 43)) & (
        train_df["Annual_Income_USD"].between(78915, 91619)
    )
    predicates["age_34_43_income_78915_91619"] = {
        "op": "all",
        "predicates": [
            {"op": "between", "column": "Age", "bounds": [34, 43]},
            {
                "op": "between",
                "column": "Annual_Income_USD",
                "bounds": [78915, 91619],
            },
        ],
    }
    slices["commute_39_49_8_medium_anxiety"] = (
        train_df["Daily_Commute_km"].between(39.0, 49.8)
    ) & (train_df["Range_Anxiety_Level"] == "Medium")
    predicates["commute_39_49_8_medium_anxiety"] = {
        "op": "all",
        "predicates": [
            {"op": "between", "column": "Daily_Commute_km", "bounds": [39.0, 49.8]},
            {"op": "eq", "column": "Range_Anxiety_Level", "value": "Medium"},
        ],
    }
    slices["suv_2_cars"] = (train_df["Current_Car_Type"] == "SUV") & (
        train_df["Number_of_Cars_Owned"] == 2
    )
    predicates["suv_2_cars"] = {
        "op": "all",
        "predicates": [
            {"op": "eq", "column": "Current_Car_Type", "value": "SUV"},
            {"op": "eq", "column": "Number_of_Cars_Owned", "value": 2},
        ],
    }
    slices["gender_other"] = train_df["Gender"] == "Other"
    predicates["gender_other"] = {"op": "eq", "column": "Gender", "value": "Other"}

    # Overall baseline
    slices["overall"] = pd.Series(True, index=train_df.index)
    predicates["overall"] = {"op": "always_true"}

    if return_predicates:
        return slices, predicates
    return slices


def reconstruct_mask_from_predicate(
    df: pd.DataFrame, predicate: dict[str, Any]
) -> pd.Series:
    """Reconstruct a boolean Series mask from a serialized slice predicate dict.

    Supports atomic operators:
      - 'eq' (or '=='): equality check {"op": "eq", "column": "col", "value": val}
      - 'ne' (or '!='): inequality check {"op": "ne", "column": "col", "value": val}
      - 'between': inclusive range check
                   {"op": "between", "column": "col", "lower": low, "upper": high}
                   or {"op": "between", "column": "col", "bounds": [low, high]}
      - 'in' (or 'isin'): membership check {"op": "in", "column": "col", "values": [v1, v2]}
      - 'gt', 'gte' (or 'ge'), 'lt', 'lte' (or 'le')
      - 'always_true' (or empty dict {}): all True mask
      - 'always_false': all False mask

    Supports recursive logical operators:
      - 'all' (or 'and'): logical AND across child predicates {"op": "all", "predicates": [...]}
      - 'any' (or 'or'): logical OR across child predicates {"op": "any", "predicates": [...]}
      - Top-level {"all": [...]} or {"any": [...]}

    Args:
        df: pd.DataFrame on which to evaluate the predicate.
        predicate: Predicate specification dict.

    Returns:
        pd.Series of booleans strictly aligned to df.index.

    Raises:
        TypeError: If df is not a DataFrame, predicate is not a dict,
                   or child predicates list is invalid.
        ValueError: If column is missing, operator is unknown/missing,
                    or bounds are malformed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be pd.DataFrame, got {type(df).__name__}")
    if not isinstance(predicate, dict):
        raise TypeError(f"Expected predicate to be dict, got {type(predicate).__name__}")

    # Empty predicate matches all rows
    if not predicate:
        return pd.Series(True, index=df.index, dtype=bool)

    # Determine logical operator and children if specified top-level
    if "all" in predicate and "op" not in predicate and "operator" not in predicate:
        op = "all"
        children = predicate["all"]
    elif "and" in predicate and "op" not in predicate and "operator" not in predicate:
        op = "all"
        children = predicate["and"]
    elif "any" in predicate and "op" not in predicate and "operator" not in predicate:
        op = "any"
        children = predicate["any"]
    elif "or" in predicate and "op" not in predicate and "operator" not in predicate:
        op = "any"
        children = predicate["or"]
    else:
        raw_op = predicate.get("op") if "op" in predicate else predicate.get("operator")
        if raw_op is None:
            raise ValueError("Predicate must contain an 'op' or 'operator' field")
        op = str(raw_op).strip().lower()
        children = None

    # Handle universal truth/falsehood
    if op in ("always_true", "true", "overall"):
        return pd.Series(True, index=df.index, dtype=bool)
    if op in ("always_false", "false"):
        return pd.Series(False, index=df.index, dtype=bool)

    # Handle recursive logical operators
    if op in ("all", "and"):
        if children is None:
            for key in ("predicates", "conditions", "rules", "all", "and"):
                if key in predicate:
                    children = predicate[key]
                    break
        if children is None:
            raise ValueError(f"Logical '{op}' predicate requires 'predicates' list")
        if not isinstance(children, list):
            raise TypeError(f"'predicates' must be a list, got {type(children).__name__}")
        if len(children) == 0:
            return pd.Series(True, index=df.index, dtype=bool)

        mask = pd.Series(True, index=df.index, dtype=bool)
        for child in children:
            child_mask = reconstruct_mask_from_predicate(df, child)
            mask = mask & child_mask
        return mask.astype(bool)

    if op in ("any", "or"):
        if children is None:
            for key in ("predicates", "conditions", "rules", "any", "or"):
                if key in predicate:
                    children = predicate[key]
                    break
        if children is None:
            raise ValueError(f"Logical '{op}' predicate requires 'predicates' list")
        if not isinstance(children, list):
            raise TypeError(f"'predicates' must be a list, got {type(children).__name__}")
        if len(children) == 0:
            return pd.Series(False, index=df.index, dtype=bool)

        mask = pd.Series(False, index=df.index, dtype=bool)
        for child in children:
            child_mask = reconstruct_mask_from_predicate(df, child)
            mask = mask | child_mask
        return mask.astype(bool)

    # Atomic operators require a column
    col = predicate.get("column") if "column" in predicate else predicate.get("col")
    if not col:
        raise ValueError(f"Predicate with operator '{op}' requires 'column' field")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' referenced in predicate not found in DataFrame")

    s = df[col]

    if op in ("eq", "=="):
        if "value" not in predicate and "val" not in predicate:
            raise ValueError("Equality predicate requires 'value' field")
        val = predicate.get("value") if "value" in predicate else predicate.get("val")
        return (s == val).astype(bool)

    if op in ("ne", "!=", "<>"):
        if "value" not in predicate and "val" not in predicate:
            raise ValueError("Inequality predicate requires 'value' field")
        val = predicate.get("value") if "value" in predicate else predicate.get("val")
        return (s != val).astype(bool)

    if op == "between":
        lower = None
        upper = None
        if "bounds" in predicate:
            bounds = predicate["bounds"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    f"Malformed 'between' bounds: expected 2-element list/tuple, got {bounds}"
                )
            lower, upper = bounds[0], bounds[1]
        elif "lower" in predicate and "upper" in predicate:
            lower, upper = predicate["lower"], predicate["upper"]
        elif "low" in predicate and "high" in predicate:
            lower, upper = predicate["low"], predicate["high"]
        elif "min" in predicate and "max" in predicate:
            lower, upper = predicate["min"], predicate["max"]
        elif (
            "value" in predicate
            and isinstance(predicate["value"], (list, tuple))
            and len(predicate["value"]) == 2
        ):
            lower, upper = predicate["value"][0], predicate["value"][1]
        else:
            raise ValueError("Malformed 'between' predicate: missing lower/upper bounds")

        if lower is None or upper is None:
            raise ValueError("Malformed 'between' predicate: bounds cannot be None")

        try:
            if lower > upper:
                raise ValueError(
                    f"Malformed bounds in 'between' predicate: lower ({lower}) > upper ({upper})"
                )
        except TypeError as e:
            raise ValueError(
                f"Bounds in 'between' predicate are not comparable: {lower}, {upper}"
            ) from e

        return s.between(lower, upper, inclusive="both").astype(bool)

    if op in ("in", "isin"):
        vals = predicate.get("values") if "values" in predicate else predicate.get("value")
        if vals is None:
            raise ValueError("Membership 'in' predicate requires 'values' field")
        if isinstance(vals, str) or not hasattr(vals, "__iter__"):
            val_type = type(vals).__name__
            raise ValueError(
                f"Membership 'in' predicate 'values' must be an iterable, got {val_type}"
            )
        return s.isin(vals).astype(bool)

    if op in ("gt", ">"):
        if "value" not in predicate and "val" not in predicate:
            raise ValueError("Greater-than predicate requires 'value' field")
        val = predicate.get("value") if "value" in predicate else predicate.get("val")
        return (s > val).astype(bool)

    if op in ("gte", "ge", ">="):
        if "value" not in predicate and "val" not in predicate:
            raise ValueError("Greater-or-equal predicate requires 'value' field")
        val = predicate.get("value") if "value" in predicate else predicate.get("val")
        return (s >= val).astype(bool)

    if op in ("lt", "<"):
        if "value" not in predicate and "val" not in predicate:
            raise ValueError("Less-than predicate requires 'value' field")
        val = predicate.get("value") if "value" in predicate else predicate.get("val")
        return (s < val).astype(bool)

    if op in ("lte", "le", "<="):
        if "value" not in predicate and "val" not in predicate:
            raise ValueError("Less-or-equal predicate requires 'value' field")
        val = predicate.get("value") if "value" in predicate else predicate.get("val")
        return (s <= val).astype(bool)

    raise ValueError(f"Unsupported predicate operator: '{op}'")


def compute_coarse_cell_stats(
    train_df: pd.DataFrame,
    y_true: np.ndarray,
    p_canonical_mean: np.ndarray,
    is_severe_fn: np.ndarray,
    sample_log_loss: np.ndarray,
    slices: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Compute coarse-cell statistics, Wilson 95% CIs, and severe-FN enrichment.

    Args:
        train_df: Training DataFrame.
        y_true: Binary ground truth labels.
        p_canonical_mean: Canonical selected-policy mean predicted probabilities.
        is_severe_fn: Boolean array indicating severe FN status.
        sample_log_loss: Sample log loss array.
        slices: Optional precomputed slice dictionary.

    Returns:
        pd.DataFrame containing cell metrics.
    """
    if slices is None:
        slices = evaluate_predeclared_slices(train_df)

    total_n = len(train_df)
    total_severe_fn = int(np.sum(is_severe_fn))
    global_severe_fn_rate = float(total_severe_fn) / float(total_n) if total_n > 0 else 0.0

    records = []
    for cell_name, mask_series in slices.items():
        mask = mask_series.to_numpy()
        cell_n = int(np.sum(mask))
        if cell_n == 0:
            continue

        cell_y = y_true[mask]
        pos_count = int(np.sum(cell_y))
        pos_rate = float(pos_count) / float(cell_n)
        ci_low, ci_high = wilson_score_interval(pos_count, cell_n, confidence=0.95)

        cell_severe_fn = int(np.sum(is_severe_fn[mask]))
        severe_fn_prevalence_total = float(cell_severe_fn) / float(cell_n)
        severe_fn_rate_among_pos = (
            float(cell_severe_fn) / float(pos_count) if pos_count > 0 else 0.0
        )
        enrichment = (
            severe_fn_prevalence_total / global_severe_fn_rate
            if global_severe_fn_rate > 0
            else 0.0
        )

        cell_p = p_canonical_mean[mask]
        cell_loss = sample_log_loss[mask]

        cell_group = (
            "overall"
            if cell_name == "overall"
            else (
                "concern_x_subsidy"
                if cell_name.startswith("concern_x_subsidy")
                else "predeclared_slice"
            )
        )

        records.append(
            {
                "cell_group": cell_group,
                "cell_name": cell_name,
                "n": cell_n,
                "positive_count": pos_count,
                "positive_rate": float(pos_rate),
                "wilson_ci_lower": float(ci_low),
                "wilson_ci_upper": float(ci_high),
                "severe_fn_count": cell_severe_fn,
                "severe_fn_prevalence_total": float(severe_fn_prevalence_total),
                "severe_fn_rate_among_positives": float(severe_fn_rate_among_pos),
                "severe_fn_enrichment": float(enrichment),
                "mean_p_canonical": float(np.mean(cell_p)),
                "mean_sample_log_loss": float(np.mean(cell_loss)),
            }
        )

    return pd.DataFrame(records)


def run_severe_fn_audit(
    train_df: pd.DataFrame,
    lgb_oof_df: pd.DataFrame,
    cb_oof_df: pd.DataFrame,
    output_dir: Path | str,
    source_provenance: dict[str, str],
    id_col: str = "id",
    target_col: str = "Will_Buy_EV",
    severe_fn_prob_threshold: float = 0.10,
    top_loss_quantile: float = 0.99,
    n_neighbors: int = 10,
) -> SevereFnAuditResult:
    """Execute end-to-end severe false negative and learnability audit.

    Generates all deterministic artifacts, frozen slices JSON, and summary manifest.

    Args:
        train_df: Training DataFrame.
        lgb_oof_df: LightGBM OOF predictions.
        cb_oof_df: CatBoost OOF predictions.
        output_dir: Directory to store output artifacts.
        source_provenance: Dictionary of source checksums (data_sha256, oof_sha256s).
        id_col: Identifier column name.
        target_col: Target column name.
        severe_fn_prob_threshold: Probability threshold for severe FN (default 0.10).
        top_loss_quantile: Quantile for top sample loss (default 0.99).
        n_neighbors: Nearest neighbors count for label mixing.

    Returns:
        SevereFnAuditResult instance.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Align and reconstruct ensemble predictions
    ens_df = reconstruct_ensemble_oof(lgb_oof_df, cb_oof_df)
    total_rows = len(train_df)
    if len(ens_df) != total_rows:
        raise ValueError(
            f"Row count mismatch between train_df ({total_rows}) and ens_df ({len(ens_df)})"
        )

    # Target encoding
    y_raw = train_df[target_col].to_numpy()
    if pd.api.types.is_numeric_dtype(y_raw):
        y_true = y_raw.astype(int)
    else:
        y_true = (train_df[target_col] == "Yes").astype(int).to_numpy()

    p_canonical = ens_df["p_canonical_mean"].to_numpy()
    sample_loss = compute_sample_log_loss(y_true, p_canonical)

    # 2. Add target, loss, and ranks to master frame
    master_df = ens_df.copy()
    master_df["id"] = train_df[id_col].values
    master_df["y_true"] = y_true
    master_df["sample_log_loss"] = sample_loss

    # Full deterministic ranking
    master_df = master_df.sort_values(
        by=["sample_log_loss", "row_index"], ascending=[False, True]
    ).reset_index(drop=True)
    master_df["loss_rank"] = np.arange(1, total_rows + 1, dtype=int)
    master_df["loss_percentile"] = 1.0 - (master_df["loss_rank"] - 1.0) / total_rows

    # Re-sort by row_index for natural row alignment
    master_df = master_df.sort_values("row_index").reset_index(drop=True)

    # 3. Identify cohorts
    is_severe_fn = (master_df["y_true"] == 1) & (
        master_df["p_canonical_mean"] <= severe_fn_prob_threshold
    )
    severe_fn_indices = np.where(is_severe_fn.to_numpy())[0]
    severe_fn_count = len(severe_fn_indices)
    severe_fn_rate = float(severe_fn_count) / float(total_rows)

    # Top 1% sample loss
    top_loss_df, q_val = select_deterministic_top_loss(master_df, quantile=top_loss_quantile)
    top_loss_indices = top_loss_df["row_index"].to_numpy()
    top_loss_count = len(top_loss_df)

    # Margin bands
    margin_25_75_mask = (master_df["p_canonical_mean"] >= 0.25) & (
        master_df["p_canonical_mean"] <= 0.75
    )
    margin_40_60_mask = (master_df["p_canonical_mean"] >= 0.40) & (
        master_df["p_canonical_mean"] <= 0.60
    )
    margin_band_25_75_count = int(margin_25_75_mask.sum())
    margin_band_40_60_count = int(margin_40_60_mask.sum())

    # Predeclared slices
    slices, slice_predicates = evaluate_predeclared_slices(train_df, return_predicates=True)

    # 4. Prepare leakage-safe neighborhood features & compute mixing
    X_mat, feature_names = prepare_neighborhood_features(
        train_df, id_col=id_col, target_col=target_col
    )

    # Compute neighborhood mixing for severe FN rows
    fn_pos_frac, fn_mean_dist, fn_disagree = compute_neighborhood_mixing(
        X_mat, y_true, severe_fn_indices, n_neighbors=n_neighbors
    )

    # Compute neighborhood mixing for top loss rows
    top_pos_frac, top_mean_dist, top_disagree = compute_neighborhood_mixing(
        X_mat, y_true, top_loss_indices, n_neighbors=n_neighbors
    )

    # 5. Build severe_false_negatives.csv
    raw_feature_cols = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]

    fn_df = master_df.iloc[severe_fn_indices].copy().reset_index(drop=True)
    fn_df["neighbor_pos_frac"] = fn_pos_frac
    fn_df["neighbor_mean_dist"] = fn_mean_dist
    fn_df["neighbor_disagreement"] = fn_disagree

    # Attach raw feature columns
    for col in raw_feature_cols:
        if col in train_df.columns:
            fn_df[col] = train_df.loc[fn_df["row_index"], col].values

    # Build top_loss_examples.csv
    top_df_out = top_loss_df.copy().reset_index(drop=True)
    top_df_out["neighbor_pos_frac"] = top_pos_frac
    top_df_out["neighbor_mean_dist"] = top_mean_dist
    top_df_out["neighbor_disagreement"] = top_disagree
    for col in raw_feature_cols:
        if col in train_df.columns:
            top_df_out[col] = train_df.loc[top_df_out["row_index"], col].values

    # 6. Coarse cell stats
    stats_df = compute_coarse_cell_stats(
        train_df=train_df,
        y_true=y_true,
        p_canonical_mean=p_canonical,
        is_severe_fn=is_severe_fn.to_numpy(),
        sample_log_loss=sample_loss,
        slices=slices,
    )

    # 7. Model consistency metrics on severe FNs
    if severe_fn_count > 0:
        model_consistency = {
            "lgb_all_repeats_low_10_fraction": float(fn_df["lgb_all_repeats_low_10"].mean()),
            "cb_all_repeats_low_10_fraction": float(fn_df["cb_all_repeats_low_10"].mean()),
            "both_all_repeats_low_10_fraction": float(fn_df["both_all_repeats_low_10"].mean()),
            "canonical_all_repeats_low_10_fraction": float(
                fn_df["canonical_all_repeats_low_10"].mean()
            ),
            "exploratory_30_70_all_repeats_low_10_fraction": float(
                fn_df["exploratory_30_70_all_repeats_low_10"].mean()
            ),
        }
        variance_summary = {
            "severe_fn_mean_canonical_repeat_std": float(fn_df["p_canonical_std"].mean()),
            "severe_fn_max_canonical_repeat_std": float(fn_df["p_canonical_std"].max()),
            "severe_fn_mean_cb_lgb_disagreement": float(fn_df["cb_lgb_abs_diff"].mean()),
            "severe_fn_max_cb_lgb_disagreement": float(fn_df["cb_lgb_abs_diff"].max()),
        }
        neighborhood_summary = {
            "severe_fn_mean_neighbor_pos_frac": float(np.mean(fn_pos_frac)),
            "severe_fn_median_neighbor_pos_frac": float(np.median(fn_pos_frac)),
            "severe_fn_zero_pos_neighbors_fraction": float(np.mean(fn_pos_frac == 0.0)),
            "severe_fn_lte_10pct_pos_neighbors_fraction": float(np.mean(fn_pos_frac <= 0.10)),
        }
    else:
        model_consistency = {}
        variance_summary = {}
        neighborhood_summary = {}

    # 8. Write CSV artifacts deterministically
    fn_path = out_path / "severe_false_negatives.csv"
    top_path = out_path / "top_loss_examples.csv"
    stats_path = out_path / "coarse_cell_stats.csv"

    fn_df.to_csv(fn_path, index=False)
    top_df_out.to_csv(top_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    # 9. Build frozen_slices.json
    frozen_slices_payload: dict[str, Any] = {
        "version": "1.0",
        "description": (
            "Canonical LightGBM cohorts and frozen slice definitions for downstream experiments"
        ),
        "slice_provenance": {
            "status": "post_hoc_discovered_then_prospectively_frozen",
            "discovery_context": "Prior canonical OOF error exploration",
            "warning": (
                "These slices were not preregistered before the exploratory analysis; "
                "future experiments must reuse these predicates unchanged."
            ),
        },
        "source_provenance": source_provenance,
        "cohorts": {
            "severe_fn": {
                "definition": (
                    f"y_true == 1 and p_canonical_mean <= {severe_fn_prob_threshold}"
                ),
                "count": severe_fn_count,
                "rate": severe_fn_rate,
            },
            "top_1pct_loss": {
                "definition": (
                    f"Deterministic sample log-loss rank <= {top_loss_count} (q >= {q_val:.6f})"
                ),
                "count": top_loss_count,
                "threshold": q_val,
            },
            "margin_band_25_75": {
                "definition": "0.25 <= p_canonical_mean <= 0.75",
                "count": margin_band_25_75_count,
                "rate": float(margin_band_25_75_count) / float(total_rows),
            },
            "margin_band_40_60": {
                "definition": "0.40 <= p_canonical_mean <= 0.60",
                "count": margin_band_40_60_count,
                "rate": float(margin_band_40_60_count) / float(total_rows),
            },
        },
        "predeclared_slices": {},
    }

    for _, row in stats_df.iterrows():
        frozen_slices_payload["predeclared_slices"][row["cell_name"]] = {
            "cell_group": row["cell_group"],
            "provenance_status": "post_hoc_discovered_then_prospectively_frozen",
            "predicate": slice_predicates[row["cell_name"]],
            "n": int(row["n"]),
            "positive_count": int(row["positive_count"]),
            "positive_rate": float(row["positive_rate"]),
            "wilson_ci": [float(row["wilson_ci_lower"]), float(row["wilson_ci_upper"])],
            "severe_fn_count": int(row["severe_fn_count"]),
            "severe_fn_prevalence_total": float(row["severe_fn_prevalence_total"]),
            "severe_fn_rate_among_positives": float(row["severe_fn_rate_among_positives"]),
            "severe_fn_enrichment": float(row["severe_fn_enrichment"]),
        }

    frozen_path = out_path / "frozen_slices.json"
    with open(frozen_path, "w") as f:
        json.dump(frozen_slices_payload, f, indent=2)

    # 10. Write concise report.md
    report_path = out_path / "report.md"
    report_md = _generate_report_markdown(
        total_rows=total_rows,
        severe_fn_count=severe_fn_count,
        severe_fn_rate=severe_fn_rate,
        top_loss_count=top_loss_count,
        top_loss_threshold=q_val,
        margin_band_25_75_count=margin_band_25_75_count,
        margin_band_40_60_count=margin_band_40_60_count,
        model_consistency=model_consistency,
        variance_summary=variance_summary,
        neighborhood_summary=neighborhood_summary,
        stats_df=stats_df,
        source_provenance=source_provenance,
    )
    with open(report_path, "w") as f:
        f.write(report_md)

    # 11. Compute artifact checksums
    artifact_paths = {
        "severe_false_negatives": str(fn_path.name),
        "top_loss_examples": str(top_path.name),
        "coarse_cell_stats": str(stats_path.name),
        "frozen_slices": str(frozen_path.name),
        "report": str(report_path.name),
    }

    artifact_checksums = {
        fn_path.name: compute_file_sha256(fn_path),
        top_path.name: compute_file_sha256(top_path),
        stats_path.name: compute_file_sha256(stats_path),
        frozen_path.name: compute_file_sha256(frozen_path),
        report_path.name: compute_file_sha256(report_path),
    }

    # 12. Build manifest.json
    manifest_payload = {
        "version": "1.0",
        "description": "Severe false-negative and learnability audit manifest",
        "source_provenance": source_provenance,
        "summary": {
            "total_rows": total_rows,
            "severe_fn_count": severe_fn_count,
            "severe_fn_rate": severe_fn_rate,
            "top_loss_count": top_loss_count,
            "top_loss_threshold": q_val,
            "margin_band_25_75_count": margin_band_25_75_count,
            "margin_band_40_60_count": margin_band_40_60_count,
            "model_consistency": model_consistency,
            "variance_summary": variance_summary,
            "neighborhood_summary": neighborhood_summary,
        },
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums,
        "caveats": [
            (
                "No assertion that positive labels are incorrect: observed severe FNs "
                "represent learnability limits or missing latent variables under available "
                "tabular features."
            ),
            (
                "Nearest-neighbor calculations use standardized Euclidean distance on "
                "numerical and one-hot categorical features; distance concentration "
                "in tabular spaces applies."
            ),
            (
                "The canonical audit policy is the selected LightGBM OOF predictor. "
                "The 70% LightGBM / 30% CatBoost blend is retained only as an "
                "exploratory sensitivity policy."
            ),
        ],
    }

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    return SevereFnAuditResult(
        total_rows=total_rows,
        severe_fn_count=severe_fn_count,
        severe_fn_rate=severe_fn_rate,
        top_loss_count=top_loss_count,
        top_loss_threshold=q_val,
        margin_band_25_75_count=margin_band_25_75_count,
        margin_band_40_60_count=margin_band_40_60_count,
        model_consistency=model_consistency,
        variance_summary=variance_summary,
        neighborhood_summary=neighborhood_summary,
        artifact_paths=artifact_paths,
        artifact_checksums=artifact_checksums,
        source_provenance=source_provenance,
    )


def _generate_report_markdown(
    total_rows: int,
    severe_fn_count: int,
    severe_fn_rate: float,
    top_loss_count: int,
    top_loss_threshold: float,
    margin_band_25_75_count: int,
    margin_band_40_60_count: int,
    model_consistency: dict[str, float],
    variance_summary: dict[str, float],
    neighborhood_summary: dict[str, float],
    stats_df: pd.DataFrame,
    source_provenance: dict[str, str],
) -> str:
    """Generate concise audit report markdown with findings and recommendations."""
    lines = [
        "# Severe False-Negative & Label Learnability Audit Report",
        "",
        "## Executive Summary",
        f"- **Dataset size:** {total_rows:,} rows",
        (
            f"- **Severe False Negatives ($y=1, \\bar{{p}} \\le 0.10$):** "
            f"{severe_fn_count:,} rows ({severe_fn_rate:.4%})"
        ),
        (
            f"- **Top 1% Sample LogLoss Cohort:** {top_loss_count:,} rows "
            f"(threshold logloss $\\ge {top_loss_threshold:.4f}$)"
        ),
        (
            f"- **Margin Bands:** [0.25, 0.75] = {margin_band_25_75_count:,} rows; "
            f"[0.40, 0.60] = {margin_band_40_60_count:,} rows"
        ),
        "",
        "## Model Consistency & Variance Diagnostics",
    ]

    if model_consistency:
        lgb_frac = model_consistency.get("lgb_all_repeats_low_10_fraction", 0.0)
        cb_frac = model_consistency.get("cb_all_repeats_low_10_fraction", 0.0)
        both_frac = model_consistency.get("both_all_repeats_low_10_fraction", 0.0)
        canonical_frac = model_consistency.get("canonical_all_repeats_low_10_fraction", 0.0)
        exploratory_frac = model_consistency.get(
            "exploratory_30_70_all_repeats_low_10_fraction", 0.0
        )
        rep_std = variance_summary.get("severe_fn_mean_canonical_repeat_std", 0.0)
        disagree = variance_summary.get("severe_fn_mean_cb_lgb_disagreement", 0.0)

        lines.extend(
            [
                f"- **LightGBM all 3 repeats $\\le 0.10$:** {lgb_frac:.2%}",
                f"- **CatBoost all 3 repeats $\\le 0.10$:** {cb_frac:.2%}",
                f"- **Both model families all repeats $\\le 0.10$:** {both_frac:.2%}",
                f"- **Canonical LightGBM all repeats $\\le 0.10$:** {canonical_frac:.2%}",
                (
                    "- **Exploratory 70% LightGBM / 30% CatBoost all repeats "
                    f"$\\le 0.10$:** {exploratory_frac:.2%}"
                ),
                f"- **Mean canonical repeat std across severe FNs:** {rep_std:.6f}",
                f"- **Mean CB-LGB absolute disagreement:** {disagree:.6f}",
                "",
                (
                    "> **Diagnostic summary:** Among canonical severe false negatives, "
                    f"{canonical_frac:.2%} remain at or below 0.10 in every LightGBM repeat; "
                    f"the mean repeat standard deviation is {rep_std:.6f}, and mean "
                    f"CatBoost-LightGBM disagreement is {disagree:.6f}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Neighborhood Sensitivity Analysis (Non-decision-critical)",
        ]
    )

    if neighborhood_summary:
        mean_pos = neighborhood_summary.get("severe_fn_mean_neighbor_pos_frac", 0.0)
        med_pos = neighborhood_summary.get("severe_fn_median_neighbor_pos_frac", 0.0)
        zero_pos = neighborhood_summary.get("severe_fn_zero_pos_neighbors_fraction", 0.0)
        lte_10 = neighborhood_summary.get("severe_fn_lte_10pct_pos_neighbors_fraction", 0.0)

        lines.extend(
            [
                f"- **Mean neighbor positive fraction:** {mean_pos:.4f}",
                f"- **Median neighbor positive fraction:** {med_pos:.4f}",
                (
                    f"- **Fraction of severe FNs with 0 positive neighbors (out of 10):** "
                    f"{zero_pos:.2%}"
                ),
                f"- **Fraction with $\\le 10\\%$ positive neighbors:** {lte_10:.2%}",
                "",
                (
                    "> **Sensitivity-only neighborhood mix:** "
                    f"{zero_pos:.2%} have no positive neighbors and {lte_10:.2%} have at most "
                    "10% positive neighbors under the declared neighborhood metric."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Strongest Coarse-Cell Findings",
            (
                "| Cell / Slice | n | Positive Count | Positive Rate (Wilson 95% CI) | "
                "Severe FN Count | All-row Prevalence | FN Rate Among Positives | "
                "Severe FN Enrichment |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for _, r in stats_df.iterrows():
        ci_str = f"[{r['wilson_ci_lower']:.4f}, {r['wilson_ci_upper']:.4f}]"
        lines.append(
            f"| `{r['cell_name']}` | {r['n']:,} | {r['positive_count']:,} | "
            f"{r['positive_rate']:.4f} {ci_str} | {r['severe_fn_count']:,} | "
            f"{r['severe_fn_prevalence_total']:.4f} | "
            f"{r['severe_fn_rate_among_positives']:.4f} | "
            f"{r['severe_fn_enrichment']:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Methodological Caveats",
            (
                "1. **No Causal Label-Error Claims:** These diagnostics measure empirical "
                "feature-target consistency and model confidence, not ground truth veracity. "
                "Rows where $y=1$ but $p \\le 0.10$ may represent rare real-world adoption "
                "under uncaptured incentives, unobserved preferences, or irreducible "
                "stochastic choice."
            ),
            (
                "2. **Neighborhood Distance Limitations:** Neighborhoods are constructed using "
                "standardized Euclidean distance over continuous variables and one-hot categorical "
                "variables with balanced feature-block scaling. This geometry is one arbitrary "
                "representation of mixed tabular distance, so these results are sensitivity "
                "evidence only and are not used as a model-selection or label-quality decision."
            ),
            (
                "3. **Policy Separation:** Cohorts and loss metrics use the canonical "
                "LightGBM OOF policy. The 70% LightGBM / 30% CatBoost blend is reported "
                "separately as exploratory sensitivity analysis."
            ),
            "",
            "## Decision and Next Steps",
            (
                "- **Finding:** Review `coarse_cell_stats.csv` for observed slice prevalence, "
                "conditional false-negative rates, and enrichment. These descriptive results "
                "do not establish a causal explanation or label error."
            ),
            (
                "- **Decision:** **Proceed to Feature Interaction Tests (Concern x Subsidy)** "
                "while maintaining frozen slice tracking. Retain all rows; do not filter or "
                "downweight labels based solely on OOF loss."
            ),
        ]
    )

    lines.extend(["", "## Source Provenance"])
    for key, value in sorted(source_provenance.items()):
        lines.append(f"- **{key}:** `{value}`")

    return "\n".join(lines) + "\n"
