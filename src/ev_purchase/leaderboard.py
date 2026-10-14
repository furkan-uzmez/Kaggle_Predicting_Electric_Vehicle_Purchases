import hashlib
import json
import re
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from ev_purchase.metrics import classification_metrics


def compute_fold_scores(
    oof_df: pd.DataFrame,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], Any] | None = None,
) -> pd.DataFrame:
    """Compute per-fold classification metrics for out-of-fold predictions.

    Args:
        oof_df: DataFrame containing columns 'row_index', 'repeat', 'fold', 'probability'.
        y_true: 1D numpy array of ground truth targets indexed by row_index position.
        metric_fn: Optional custom metric function taking (y_true, y_pred). If None,
            classification_metrics is used and 'roc_auc' is extracted.

    Returns:
        DataFrame with columns 'repeat', 'fold', 'roc_auc', sorted by (repeat, fold).
    """
    required_cols = {"row_index", "repeat", "fold", "probability"}
    if not required_cols.issubset(oof_df.columns):
        missing = required_cols - set(oof_df.columns)
        raise ValueError(f"oof_df is missing required columns: {missing}")

    if metric_fn is None:
        metric_fn = classification_metrics

    records = []
    for (repeat, fold), group in oof_df.groupby(["repeat", "fold"], sort=False):
        row_indices = group["row_index"].to_numpy()
        probs = group["probability"].to_numpy()
        sub_y = y_true[row_indices]

        res = metric_fn(sub_y, probs)
        if isinstance(res, dict):
            score = float(res["roc_auc"])
        else:
            score = float(res)

        records.append({"repeat": int(repeat), "fold": int(fold), "roc_auc": score})

    res_df = pd.DataFrame(records)
    res_df = res_df.sort_values(by=["repeat", "fold"]).reset_index(drop=True)
    return res_df


def compute_repeat_averaged_roc_auc(oof_df: pd.DataFrame, y_true: np.ndarray) -> float:
    """Compute ROC-AUC after averaging each row's probability across CV repeats.

    Under RepeatedStratifiedKFold every row receives one out-of-fold prediction
    per repeat. Averaging those predictions before scoring is a bagging step: it
    cancels per-repeat noise and therefore yields a slightly higher, lower-variance
    score than pooling all repeat-level predictions into a single vector.

    This is the estimator recorded in ``leaderboard.md`` and in
    ``outputs/ensemble/blend_metrics.json``.

    Args:
        oof_df: DataFrame with columns 'row_index', 'repeat', 'fold', 'probability'.
        y_true: 1D numpy array of ground truth targets indexed by row_index position.

    Returns:
        ROC-AUC computed over per-row averaged probabilities.
    """
    required_cols = {"row_index", "repeat", "probability"}
    if not required_cols.issubset(oof_df.columns):
        missing = required_cols - set(oof_df.columns)
        raise ValueError(f"oof_df is missing required columns: {missing}")

    averaged = oof_df.groupby("row_index", sort=True)["probability"].mean()
    row_indices = averaged.index.to_numpy()
    return float(classification_metrics(y_true[row_indices], averaged.to_numpy())["roc_auc"])


def paired_fold_comparison(
    fold_scores_a: pd.DataFrame,
    fold_scores_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Perform per-fold paired difference comparison between two models.

    Args:
        fold_scores_a: DataFrame with columns 'repeat', 'fold', 'roc_auc'.
        fold_scores_b: DataFrame with columns 'repeat', 'fold', 'roc_auc'.
        label_a: Name label for model A.
        label_b: Name label for model B.

    Returns:
        Dict containing paired comparison statistics: label_a, label_b, mean_a,
        mean_b, mean_diff, paired_std, fold_std_a, fold_std_b, abs_ratio, n_folds.
    """
    required = {"repeat", "fold", "roc_auc"}
    if not required.issubset(fold_scores_a.columns) or not required.issubset(fold_scores_b.columns):
        msg = "fold_scores DataFrames must contain 'repeat', 'fold', and 'roc_auc' columns."
        raise ValueError(msg)

    df_a = fold_scores_a.sort_values(by=["repeat", "fold"]).reset_index(drop=True)
    df_b = fold_scores_b.sort_values(by=["repeat", "fold"]).reset_index(drop=True)

    if len(df_a) != len(df_b) or not df_a[["repeat", "fold"]].equals(df_b[["repeat", "fold"]]):
        raise ValueError("Fold keys differ between model A and model B.")

    a = df_a["roc_auc"].to_numpy(dtype=float)
    b = df_b["roc_auc"].to_numpy(dtype=float)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    diffs = a - b
    mean_diff = float(np.mean(diffs))

    paired_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if np.isclose(paired_std, 0.0, atol=1e-12):
        paired_std = 0.0

    fold_std_a = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    if np.isclose(fold_std_a, 0.0, atol=1e-12):
        fold_std_a = 0.0

    fold_std_b = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
    if np.isclose(fold_std_b, 0.0, atol=1e-12):
        fold_std_b = 0.0

    if paired_std == 0.0:
        abs_ratio = float("inf")
    else:
        abs_ratio = float(abs(mean_diff) / paired_std)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "paired_std": paired_std,
        "fold_std_a": fold_std_a,
        "fold_std_b": fold_std_b,
        "abs_ratio": abs_ratio,
        "n_folds": int(len(a)),
    }


def blend_oof(
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Blend out-of-fold probability predictions using weighted sum.

    Args:
        frames: Dict mapping model label to OOF DataFrame (row_index, repeat, fold, probability).
        weights: Dict mapping model label to weight. Must sum to 1.0 within 1e-9.

    Returns:
        Blended DataFrame with columns 'row_index', 'repeat', 'fold', 'probability'.
    """
    if set(weights.keys()) != set(frames.keys()):
        raise ValueError("Weights keys do not match frames keys.")

    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights contain negative values.")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1.0 within 1e-9 (sum = {weight_sum}).")

    required_cols = {"row_index", "repeat", "fold", "probability"}
    for name, frame in frames.items():
        if not required_cols.issubset(frame.columns):
            raise ValueError(f"Frame for model '{name}' missing required columns.")

    first_key = list(frames.keys())[0]
    base = frames[first_key][["row_index", "repeat", "fold"]].copy()

    total_prob = np.zeros(len(base), dtype=float)

    for name, frame in frames.items():
        w = weights[name]
        merged = pd.merge(
            base,
            frame,
            on=["row_index", "repeat", "fold"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(base):
            raise ValueError(f"OOF frame '{name}' could not be matched one-to-one with base frame.")
        total_prob += w * merged["probability"].to_numpy(dtype=float)

    result = base.copy()
    result["probability"] = total_prob
    return result


def _is_header_row(cells: list[str]) -> bool:
    """Return True when a markdown table row looks like a header rather than data.

    A header row is identified structurally: its numeric-metric columns must not
    parse as numbers. Relying only on keywords such as "Model" would misclassify
    data rows whose label contains those words.
    """
    if len(cells) < 6:
        return False

    for cell in (cells[3], cells[4], cells[5]):
        candidate = cell.replace("**", "").replace("+", "").replace("−", "-").strip()
        try:
            float(candidate)
        except ValueError:
            continue
        else:
            return False
    return True


def parse_public_leaderboard_markdown(text: str) -> pd.DataFrame:
    """Parse Kaggle public leaderboard results table from markdown text.

    Args:
        text: Markdown content containing the public leaderboard table.

    Returns:
        DataFrame with columns: submission_file, model_label, aggregate_oof_roc_auc,
        public_score, public_minus_cv.

    Note:
        ``aggregate_oof_roc_auc`` is the pooled out-of-fold ROC-AUC recorded in
        ``leaderboard.md``. It is a different estimator from the mean of per-fold
        ROC-AUC reported in ``model_summary.parquet`` and the two are not
        interchangeable.
    """
    if "<!-- SPEARMAN_CORRELATION_START -->" in text:
        text = text.split("<!-- SPEARMAN_CORRELATION_START -->")[0]
    elif "<!-- PAIRWISE_CI_START -->" in text:
        text = text.split("<!-- PAIRWISE_CI_START -->")[0]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]

    if not table_lines:
        raise ValueError("No markdown table found in input text.")

    rows = []
    header_found = False

    for line in table_lines:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if not cells:
            continue

        # Identify the header only once, and only before any data row has been
        # seen. Matching on word content alone would silently drop legitimate
        # data rows whose model label happens to contain "Model" or "Submission".
        if not header_found and not rows and _is_header_row(cells):
            header_found = True
            continue

        if all(re.match(r"^:?-+:?$", cell) for cell in cells):
            continue

        if header_found and len(cells) >= 6:
            sub_file = cells[1].strip("`").strip()
            model_label = cells[2].strip()

            def _parse_float(val: str) -> float:
                cleaned = val.replace("**", "").replace("+", "").replace("−", "-").strip()
                if cleaned.upper() in {"PENDING", "N/A", "NONE", "NAN"}:
                    return np.nan
                return float(cleaned)

            aggregate_oof = _parse_float(cells[3])
            public_score = _parse_float(cells[4])
            public_minus_cv = _parse_float(cells[5])

            rows.append(
                {
                    "submission_file": sub_file,
                    "model_label": model_label,
                    "aggregate_oof_roc_auc": aggregate_oof,
                    "public_score": public_score,
                    "public_minus_cv": public_minus_cv,
                }
            )

    if not rows:
        raise ValueError("Failed to parse any leaderboard rows from markdown table.")

    return pd.DataFrame(rows)


# =============================================================================
# Paired Repeated-CV Confidence Intervals (Nadeau-Bengio)
# =============================================================================

CANONICAL_LEADERBOARD_MODELS: list[str] = [
    "ensemble_catboost30_lightgbm70",
    "catboost_hpo_trial22",
    "lightgbm_hpo_trial35",
    "catboost_baseline",
    "xgboost_baseline",
    "logistic",
    "dummy",
]

EXCLUDED_LEADERBOARD_MODELS: dict[str, str] = {
    "submission_advanced_engineered.csv": (
        "Missing compatible 15-fold CV predictions under standard protocol "
        "(different feature pipeline)."
    ),
    "super_ensemble_pipeline - Version 7": (
        "External Kaggle notebook submission without local fold-level predictions."
    ),
    "lightgbm - Version 1": (
        "External Kaggle notebook submission without local fold-level predictions."
    ),
    "ft-transformer - Version 2": (
        "External Kaggle notebook submission without local fold-level predictions."
    ),
    "lightgbm_baseline": (
        "Internal baseline excluded from public leaderboard comparison "
        "(superseded by final tuned LightGBM / lightgbm_hpo_trial35)."
    ),
}

RECOVERED_LEADERBOARD_MODELS: dict[str, dict[str, Any]] = {
    "super_ensemble_pipeline - Version 7": {
        "submission_file": "super_ensemble_pipeline - Version 7",
        "model_label": "Super Ensemble Pipeline",
        "run_identifier": "super-ensemble-pipeline",
        "source_path": "outputs/runs/kaggle/super-ensemble-pipeline/super-ensemble-pipeline.ipynb",
        "source_sha256": "a0afc5780955eae731de3cf6667a58f0cdcddc526ad60d45c033a66392d35b2d",
        "local_cv_roc_auc": 0.942104,
        "recovered_local_cv_roc_auc": 0.942104,
        "public_score": 0.94161,
        "extraction_method": (
            "Extracted from executed notebook Cell 9 printed SLSQP ensemble output "
            "(🚀 Super Ensemble OOF ROC-AUC: 0.942104)."
        ),
        "precision_note": "6-decimal notebook output",
        "caveat": (
            "5x5 repeated CV / 25 folds differs from standard 5x3 / 15 folds, "
            "plus preprocessing leakage concern across folds."
        ),
    },
    "lightgbm - Version 1": {
        "submission_file": "lightgbm - Version 1",
        "model_label": "LightGBM Pipeline",
        "run_identifier": "lightgbm",
        "source_path": "outputs/runs/kaggle/lightgbm/results/experiment_manifest.json",
        "source_sha256": "ac29ff865f0d7ad7a04d24ab019aa25a76137a4aac99117c8907346ad74600c5",
        "local_cv_roc_auc": 0.9415344361329914,
        "recovered_local_cv_roc_auc": 0.9415344361329914,
        "public_score": 0.94157,
        "extraction_method": (
            "Extracted from experiment_manifest.json development_oof_metrics.roc_auc."
        ),
        "precision_note": "full float64 precision from experiment manifest",
        "caveat": "development OOF/protocol sample differs (80% development split sample).",
    },
    "ft-transformer - Version 2": {
        "submission_file": "ft-transformer - Version 2",
        "model_label": "FT-Transformer Pipeline",
        "run_identifier": "ft-transformer",
        "source_path": "outputs/runs/kaggle/ft-transformer/results/oof_predictions.csv",
        "source_sha256": "ccbdb4eb72ade9ea92e6a77c53b46e47a41deabdb26aec6e2e3bebdc18cdd50c",
        "local_cv_roc_auc": 0.9387246162841222,
        "recovered_local_cv_roc_auc": 0.9387246162841222,
        "public_score": 0.93807,
        "extraction_method": (
            "Recomputed from 668,665 persisted out-of-fold predictions in "
            "results/oof_predictions.csv via classification_metrics(target, oof_ft_transformer)."
        ),
        "precision_note": "full float64 precision recomputed from complete OOF vector",
        "caveat": "full-data 5-fold not 5x3 repeated CV.",
    },
}


def load_and_validate_recovered_models(
    project_root: Path,
    specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load, extract, and validate recovered model scores from persisted source artifacts on disk.

    Args:
        project_root: Root directory of the repository.
        specs: Specification dictionary. Defaults to RECOVERED_LEADERBOARD_MODELS.

    Returns:
        Dict mapping model name to fully resolved metadata, including fresh source_sha256
        and verified extracted scores.

    Raises:
        FileNotFoundError: If any source artifact is missing.
        ValueError: If extracted score differs from expected or structure is invalid.
    """
    if specs is None:
        specs = RECOVERED_LEADERBOARD_MODELS

    resolved: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        entry = dict(spec)
        rel_path = spec["source_path"]
        abs_path = project_root / rel_path

        if not abs_path.exists():
            raise FileNotFoundError(
                f"Source artifact missing for recovered model '{name}': {abs_path}"
            )

        content_bytes = abs_path.read_bytes()
        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        entry["source_sha256"] = sha256_hash

        run_id = spec.get("run_identifier", "")
        if run_id == "super-ensemble-pipeline":
            nb_data = json.loads(content_bytes.decode("utf-8"))
            extracted = None
            for cell in nb_data.get("cells", []):
                for out in cell.get("outputs", []):
                    for text_line in out.get("text", []):
                        match = re.search(r"Super Ensemble OOF ROC-AUC:\s*([0-9.]+)", text_line)
                        if match:
                            extracted = float(match.group(1))
                            break
                    if extracted is not None:
                        break
                if extracted is not None:
                    break

            if extracted is None:
                raise ValueError(f"Could not extract Super Ensemble OOF ROC-AUC from {abs_path}")
            if not np.isclose(extracted, spec["local_cv_roc_auc"], atol=1e-6):
                raise ValueError(
                    f"Extracted score {extracted} does not match expected "
                    f"{spec['local_cv_roc_auc']} in {abs_path}"
                )
            entry["extracted_score"] = extracted

        elif run_id == "lightgbm":
            manifest = json.loads(content_bytes.decode("utf-8"))
            dev_metrics = manifest.get("development_oof_metrics", {})
            if "roc_auc" not in dev_metrics:
                raise ValueError(f"Missing development_oof_metrics.roc_auc in {abs_path}")
            extracted = float(dev_metrics["roc_auc"])
            if not np.isclose(extracted, spec["local_cv_roc_auc"], atol=1e-14):
                raise ValueError(
                    f"Extracted score {extracted} does not match expected "
                    f"{spec['local_cv_roc_auc']} in {abs_path}"
                )
            entry["extracted_score"] = extracted

        elif run_id == "ft-transformer":
            oof_df = pd.read_csv(abs_path)
            if not {"target", "oof_ft_transformer"}.issubset(oof_df.columns):
                msg = (
                    f"Missing expected columns in {abs_path}: "
                    "required 'target', 'oof_ft_transformer'"
                )
                raise ValueError(msg)
            metrics = classification_metrics(
                oof_df["target"].to_numpy(),
                oof_df["oof_ft_transformer"].to_numpy(),
            )
            extracted = float(metrics["roc_auc"])
            if not np.isclose(extracted, spec["local_cv_roc_auc"], atol=1e-14):
                raise ValueError(
                    f"Computed ROC-AUC {extracted} does not match expected "
                    f"{spec['local_cv_roc_auc']} in {abs_path}"
                )
            entry["extracted_score"] = extracted
        else:
            entry["extracted_score"] = spec["local_cv_roc_auc"]

        resolved[name] = entry

    return resolved


def compute_paired_ci(
    scores_a: Sequence[float] | np.ndarray | pd.Series,
    scores_b: Sequence[float] | np.ndarray | pd.Series,
    test_train_ratio: float = 0.25,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute naive and Nadeau-Bengio corrected repeated-CV paired confidence intervals.

    Args:
        scores_a: Matched fold metric scores for Model A.
        scores_b: Matched fold metric scores for Model B.
        test_train_ratio: Ratio of test set size to train set size per fold.
            For 5-fold CV, test/train ratio = (1/5) / (4/5) = 0.25.
        confidence_level: Two-sided confidence level (default 0.95).

    Returns:
        Dict containing paired difference statistics, naive and Nadeau-Bengio
        standard errors, test statistics, p-values, 95% CIs, and conclusion.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Mismatched score array lengths: len(a)={len(a)}, len(b)={len(b)}")

    n = len(a)
    if n < 2:
        raise ValueError(f"At least 2 paired observations required, got n={n}")

    df = n - 1
    diffs = a - b
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_diff = float(np.mean(diffs))
    s = float(np.std(diffs, ddof=1))

    # Robust zero-variance handling without epsilon
    if s == 0.0:
        if mean_diff == 0.0:
            ci_lower = 0.0
            ci_upper = 0.0
            p_val = 1.0
            t_stat = 0.0
            conclusion = "inconclusive"
            note = "Zero variance with zero mean difference: degenerate CI [0, 0], p=1.0."
        else:
            ci_lower = mean_diff
            ci_upper = mean_diff
            p_val = 0.0
            t_stat = float("inf") if mean_diff > 0 else float("-inf")
            conclusion = "A higher" if mean_diff > 0 else "B higher"
            note = (
                "Zero variance with non-zero mean difference: degenerate CI [mean, mean], "
                "p=0.0 by computational convention."
            )

        return {
            "n": n,
            "df": df,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_difference": mean_diff,
            "sd": 0.0,
            "sd_difference": 0.0,
            "correction_ratio": test_train_ratio,
            "test_train_ratio": test_train_ratio,
            "se_naive": 0.0,
            "ci_naive_lower": ci_lower,
            "ci_naive_upper": ci_upper,
            "t_stat_naive": t_stat,
            "p_value_naive_raw": p_val,
            "se_nb": 0.0,
            "ci_nb_lower": ci_lower,
            "ci_nb_upper": ci_upper,
            "t_stat_nb": t_stat,
            "p_value_nb_raw": p_val,
            "zero_variance": True,
            "conclusion": conclusion,
            "note": note,
        }

    alpha = 1.0 - confidence_level
    t_crit = float(scipy.stats.t.ppf(1.0 - alpha / 2.0, df=df))

    # Naive paired t-interval
    se_naive = float(s / np.sqrt(n))
    t_stat_naive = float(mean_diff / se_naive)
    p_val_naive_raw = float(2.0 * scipy.stats.t.sf(abs(t_stat_naive), df=df))
    ci_naive_lower = float(mean_diff - t_crit * se_naive)
    ci_naive_upper = float(mean_diff + t_crit * se_naive)

    # Nadeau-Bengio corrected repeated-CV interval
    se_nb = float(s * np.sqrt((1.0 / n) + test_train_ratio))
    t_stat_nb = float(mean_diff / se_nb)
    p_val_nb_raw = float(2.0 * scipy.stats.t.sf(abs(t_stat_nb), df=df))
    ci_nb_lower = float(mean_diff - t_crit * se_nb)
    ci_nb_upper = float(mean_diff + t_crit * se_nb)

    # Conclusion based on NB CI: strictly positive -> A higher; strictly negative -> B higher.
    # Never call zero-containing CI equivalent.
    if ci_nb_lower > 0.0:
        conclusion = "A higher"
    elif ci_nb_upper < 0.0:
        conclusion = "B higher"
    else:
        conclusion = "inconclusive"

    return {
        "n": n,
        "df": df,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_difference": mean_diff,
        "sd": s,
        "sd_difference": s,
        "correction_ratio": test_train_ratio,
        "test_train_ratio": test_train_ratio,
        "se_naive": se_naive,
        "ci_naive_lower": ci_naive_lower,
        "ci_naive_upper": ci_naive_upper,
        "t_stat_naive": t_stat_naive,
        "p_value_naive_raw": p_val_naive_raw,
        "se_nb": se_nb,
        "ci_nb_lower": ci_nb_lower,
        "ci_nb_upper": ci_nb_upper,
        "t_stat_nb": t_stat_nb,
        "p_value_nb_raw": p_val_nb_raw,
        "zero_variance": False,
        "conclusion": conclusion,
        "note": "",
    }


# Alias for hypothesis test using Nadeau-Bengio corrected resampled t-test
compute_corrected_resampled_ttest = compute_paired_ci


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Compute family-wise error rate adjusted p-values using the Holm-Bonferroni method.

    Enforces monotonicity so that adjusted p-values remain non-decreasing with
    respect to the raw p-values.

    Args:
        p_values: Sequence of raw p-values.

    Returns:
        1D numpy array of adjusted p-values in the original input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    sorted_p = p[order]

    multipliers = n - np.arange(n)
    adjusted_sorted = sorted_p * multipliers

    # Cumulative maximum preserves step-down monotonicity
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    # Invert sorting to restore input ordering
    inv_order = np.empty(n, dtype=int)
    inv_order[order] = np.arange(n)
    return adjusted_sorted[inv_order]


def compute_all_pairwise_cis(
    per_model_fold_df: dict[str, pd.DataFrame],
    models: Sequence[str] | None = None,
    test_train_ratio: float = 0.25,
    split_sha256: str = "unknown",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise CI analysis and fold-level differences across all model pairs.

    Args:
        per_model_fold_df: Mapping of model names to fold score DataFrames
            (must contain 'repeat', 'fold', 'roc_auc').
        models: Canonical ordered sequence of model names. Defaults to
            CANONICAL_LEADERBOARD_MODELS.
        test_train_ratio: Ratio for Nadeau-Bengio correction (default 0.25).
        split_sha256: Policy split hash for tracking.

    Returns:
        Tuple of (pairwise_ci_df, pairwise_fold_differences_df).
    """
    if models is None:
        models = CANONICAL_LEADERBOARD_MODELS

    for m in models:
        if m not in per_model_fold_df:
            raise KeyError(f"Fold score data missing for eligible model '{m}'")

    pairwise_records: list[dict[str, Any]] = []
    fold_diff_records: list[dict[str, Any]] = []

    for m_a, m_b in combinations(models, 2):
        df_a = per_model_fold_df[m_a].sort_values(by=["repeat", "fold"]).reset_index(drop=True)
        df_b = per_model_fold_df[m_b].sort_values(by=["repeat", "fold"]).reset_index(drop=True)

        if len(df_a) != len(df_b) or not df_a[["repeat", "fold"]].equals(df_b[["repeat", "fold"]]):
            raise ValueError(f"Mismatched repeat/fold structure between '{m_a}' and '{m_b}'")

        scores_a = df_a["roc_auc"].to_numpy(dtype=float)
        scores_b = df_b["roc_auc"].to_numpy(dtype=float)
        diffs = scores_a - scores_b

        for r, f, sa, sb, d in zip(
            df_a["repeat"], df_a["fold"], scores_a, scores_b, diffs, strict=True
        ):
            fold_diff_records.append(
                {
                    "model_a": m_a,
                    "model_b": m_b,
                    "repeat": int(r),
                    "fold": int(f),
                    "score_a": float(sa),
                    "score_b": float(sb),
                    "difference": float(d),
                }
            )

        ci_stat = compute_paired_ci(
            scores_a,
            scores_b,
            test_train_ratio=test_train_ratio,
            confidence_level=0.95,
        )
        ci_stat["model_a"] = m_a
        ci_stat["model_b"] = m_b
        ci_stat["metric"] = "roc_auc"
        ci_stat["split_sha256"] = split_sha256
        ci_stat["protocol_split_hash"] = split_sha256

        pairwise_records.append(ci_stat)

    pairwise_ci_df = pd.DataFrame(pairwise_records)
    fold_diffs_df = pd.DataFrame(fold_diff_records)

    if not pairwise_ci_df.empty:
        pairwise_ci_df["p_value_nb_holm"] = holm_adjust(pairwise_ci_df["p_value_nb_raw"].to_numpy())
        pairwise_ci_df["p_value_naive_holm"] = holm_adjust(
            pairwise_ci_df["p_value_naive_raw"].to_numpy()
        )

    return pairwise_ci_df, fold_diffs_df


def format_pairwise_ci_markdown(
    pairwise_ci_df: pd.DataFrame,
    exclusions: dict[str, str] | None = None,
) -> str:
    """Format the pairwise confidence intervals section in markdown.

    Includes markers, methodology, explicit exclusions, caveats, and
    the 21-row summary table.
    """
    if exclusions is None:
        exclusions = EXCLUDED_LEADERBOARD_MODELS

    def _fmt_p(p: float) -> str:
        if p == 0.0:
            return "0.0000"
        if p < 0.0001:
            return f"{p:.2e}"
        return f"{p:.4f}"

    lines = [
        "<!-- PAIRWISE_CI_START -->",
        "## Pairwise Repeated-CV Confidence Intervals (Nadeau-Bengio)",
        "",
        "### Methodology & Statistical Protocol",
        (
            "- **Evaluation protocol**: 3 repeats of 5-fold stratified cross-validation "
            "($n = 15$ folds, $df = 14$)."
        ),
        (
            "- **Metric**: ROC-AUC. Per-fold difference defined as "
            "$\\Delta = \\text{score}_A - \\text{score}_B$, aligned by `(repeat, fold)`."
        ),
        (
            "- **Canonical ordering**: Model A is higher-ranked on the public leaderboard than "
            "Model B, so positive $\\Delta$ indicates Model A achieved higher ROC-AUC."
        ),
        (
            "- **Variance correction**: Standard paired t-intervals severely underestimate "
            "variance because repeated CV folds share training samples. We apply the "
            "Nadeau-Bengio correction (2003) with test/train ratio "
            "$r = (1/5) / (4/5) = 0.25$:"
        ),
        "  $$\\text{SE}_{\\text{NB}} = s \\sqrt{\\frac{1}{n} + r}$$",
        (
            "  For $n = 15$ and $r = 0.25$, $\\text{SE}_{\\text{NB}}$ is inflated by "
            "$\\sqrt{1 + 15 \\times 0.25} = \\sqrt{4.75} \\approx 2.18\\times$ "
            "relative to naive paired SE."
        ),
        (
            "- **Multiple testing**: Raw p-values are adjusted using the Holm-Bonferroni "
            "step-down procedure across the family of 21 comparisons."
        ),
        "- **Conclusions** (based on 95% Nadeau-Bengio CI):",
        "  - **A higher**: 95% NB CI is strictly positive ($>0$).",
        "  - **B higher**: 95% NB CI is strictly negative ($<0$).",
        (
            "  - **inconclusive**: 95% NB CI covers 0. A zero-containing interval is "
            "never claimed as equivalent."
        ),
        "",
        "### Explicit Exclusions",
        (
            "The following models/submissions are excluded from paired CV analysis due to "
            "lacking compatible fold-level predictions under the standard evaluation protocol:"
        ),
    ]

    for model, reason in exclusions.items():
        lines.append(f"- `{model}`: {reason}")

    lines.extend(
        [
            "",
            "### Statistical Caveats",
            (
                "1. **Fold overlap and correlation**: Repeated CV folds share training data, "
                "inducing positive correlation across test errors. The Nadeau-Bengio correction "
                "analytically accounts for this correlation but remains an approximation."
            ),
            (
                "2. **HPO selection bias**: Final hyperparameters were selected across multiple "
                "candidate evaluations on this validation split, which introduces selection bias "
                "(optimism) into observed CV differences."
            ),
            (
                "3. **Practical importance**: Statistical significance does not necessarily "
                "equate to practical importance or real-world impact."
            ),
            (
                "4. **Kaggle test distribution**: Local CV estimates performance on the training "
                "distribution; Kaggle public and private test sets are evaluated separately and "
                "may reflect distribution shifts."
            ),
            "",
            "### Pairwise Comparisons (21 Canonical Pairs)",
            "",
            "| Model A | Model B | Δ A−B | NB 95% CI | Raw p | Holm p | Conclusion |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in pairwise_ci_df.iterrows():
        diff_str = f"{row['mean_difference']:+.5f}"
        ci_str = f"[{row['ci_nb_lower']:+.5f}, {row['ci_nb_upper']:+.5f}]"
        raw_p_str = _fmt_p(row["p_value_nb_raw"])
        holm_p_str = _fmt_p(row["p_value_nb_holm"])
        ma = row["model_a"]
        mb = row["model_b"]
        conc = row["conclusion"]
        lines.append(
            f"| `{ma}` | `{mb}` | {diff_str} | {ci_str} | {raw_p_str} | {holm_p_str} | {conc} |"
        )

    lines.extend(["", "<!-- PAIRWISE_CI_END -->"])
    return "\n".join(lines)


def update_managed_section(
    original_md: str,
    start_marker: str,
    end_marker: str,
    new_content: str,
    insert_before: str | None = None,
) -> str:
    """Update a managed section in markdown between start and end markers idempotently.

    If markers exist, replaces the content between them. Otherwise, inserts the section
    before insert_before if present, or appends it.
    """
    pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
    if re.search(pattern, original_md, flags=re.DOTALL):
        return re.sub(pattern, lambda _: new_content, original_md, flags=re.DOTALL)

    if insert_before and insert_before in original_md:
        prefix, suffix = original_md.split(insert_before, 1)
        prefix_stripped = prefix.rstrip()
        return f"{prefix_stripped}\n\n{new_content}\n\n{insert_before}{suffix}"

    stripped = original_md.rstrip()
    return f"{stripped}\n\n{new_content}\n"


def update_leaderboard_markdown(original_md: str, pairwise_section: str) -> str:
    """Update leaderboard markdown with the generated pairwise CI section idempotently.

    If markers exist, replaces the content between them. Otherwise, appends the section.
    """
    return update_managed_section(
        original_md=original_md,
        start_marker="<!-- PAIRWISE_CI_START -->",
        end_marker="<!-- PAIRWISE_CI_END -->",
        new_content=pairwise_section,
    )


# =============================================================================
# Spearman Rank Correlation Analyses (Primary & Exploratory)
# =============================================================================


def compute_spearman_correlation(
    df: pd.DataFrame,
    x_col: str = "aggregate_oof_roc_auc",
    y_col: str = "public_score",
) -> dict[str, Any]:
    """Compute Spearman rank correlation on complete pairs with average ranks for ties.

    Args:
        df: DataFrame containing the columns specified by x_col and y_col.
        x_col: Column name for local CV scores (default 'aggregate_oof_roc_auc').
        y_col: Column name for Kaggle public scores (default 'public_score').

    Returns:
        Dict with correlation statistics, p-values, tie handling details, and row details.
    """
    n_total = len(df)
    valid_mask = df[x_col].notna() & df[y_col].notna()
    complete_df = df[valid_mask].copy()
    n_complete = len(complete_df)
    n_excluded = n_total - n_complete

    excluded_submissions: list[str] = []
    if "submission_file" in df.columns:
        excluded_submissions = df.loc[~valid_mask, "submission_file"].tolist()

    if n_complete < 2:
        raise ValueError(f"At least 2 complete observations required, got n={n_complete}")

    x = complete_df[x_col].to_numpy(dtype=float)
    y = complete_df[y_col].to_numpy(dtype=float)

    # scipy.stats.spearmanr assigns average ranks to ties by default
    res = scipy.stats.spearmanr(x, y)
    rho = float(res.statistic)
    p_val = float(res.pvalue)

    interpretation = (
        "Spearman rank correlation measures monotonic rank association between local CV "
        "and Kaggle public leaderboard scores; it does not imply identical ordering, causality, "
        "model superiority, or private-LB assurance."
    )

    included_rows = []
    for _, row in complete_df.iterrows():
        entry = {
            "submission_file": str(row.get("submission_file", "")),
            "model_label": str(row.get("model_label", "")),
            "local_score": float(row[x_col]),
            "public_score": float(row[y_col]),
        }
        included_rows.append(entry)

    return {
        "n_total": n_total,
        "n_complete": n_complete,
        "n": n_complete,
        "n_excluded": n_excluded,
        "rho": rho,
        "p_value": p_val,
        "tie_method": "average ranks",
        "alternative": "two-sided",
        "p_value_method": "asymptotic (scipy.stats.spearmanr)",
        "excluded_submissions": excluded_submissions,
        "included_rows": included_rows,
        "interpretation": interpretation,
    }


def compute_dual_spearman_correlations(
    public_leaderboard_df: pd.DataFrame,
    recovered_models: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute primary (comparable protocol) and exploratory (mixed protocol) Spearman analyses.

    Args:
        public_leaderboard_df: DataFrame parsed from public leaderboard table.
        recovered_models: Dict of recovered models with protocol deviations and caveats.
            Defaults to RECOVERED_LEADERBOARD_MODELS.

    Returns:
        Dict containing primary analysis, exploratory analysis, caveats, and metadata.
    """
    if recovered_models is None:
        recovered_models = RECOVERED_LEADERBOARD_MODELS

    # 1. Primary analysis: complete cases from public_leaderboard_df (n=8)
    primary_res = compute_spearman_correlation(
        public_leaderboard_df,
        x_col="aggregate_oof_roc_auc",
        y_col="public_score",
    )
    primary_payload = {
        "name": "primary_comparable",
        "label": "Primary Analysis (Comparable Protocol)",
        "protocol_description": (
            "Current 8 public leaderboard rows with standard repeated CV local scores"
        ),
        "n": primary_res["n_complete"],
        "rho": primary_res["rho"],
        "p_value": primary_res["p_value"],
        "tie_method": primary_res["tie_method"],
        "alternative": primary_res["alternative"],
        "p_value_method": primary_res["p_value_method"],
        "included_rows": primary_res["included_rows"],
        "excluded_submissions": primary_res["excluded_submissions"],
    }

    # 2. Exploratory analysis: primary complete rows plus recovered models (n=11)
    exploratory_rows = list(primary_res["included_rows"])
    recovered_submissions_list = []

    for name, rec in recovered_models.items():
        sub_file = rec["submission_file"]
        model_label = rec.get("model_label", name)
        local_cv = rec.get("local_cv_roc_auc", rec.get("recovered_local_cv_roc_auc"))
        pub_score = rec["public_score"]
        caveat = rec["caveat"]

        rec_entry = {
            "submission_file": sub_file,
            "model_label": model_label,
            "run_identifier": rec.get("run_identifier", ""),
            "source_path": rec.get("source_path", ""),
            "source_sha256": rec.get("source_sha256", ""),
            "local_cv_roc_auc": float(local_cv),
            "recovered_local_cv_roc_auc": float(local_cv),
            "public_score": float(pub_score),
            "extraction_method": rec.get("extraction_method", ""),
            "precision_note": rec.get("precision_note", ""),
            "caveat": caveat,
        }
        recovered_submissions_list.append(rec_entry)

        exploratory_rows.append(
            {
                "submission_file": sub_file,
                "model_label": model_label,
                "local_score": float(local_cv),
                "public_score": float(pub_score),
            }
        )

    exp_df = pd.DataFrame(exploratory_rows).rename(columns={"local_score": "aggregate_oof_roc_auc"})
    exp_res = compute_spearman_correlation(
        exp_df,
        x_col="aggregate_oof_roc_auc",
        y_col="public_score",
    )

    exploratory_payload = {
        "name": "exploratory_mixed_protocol",
        "label": "Exploratory Analysis (Mixed Protocol with Recovered Values)",
        "protocol_description": (
            "Primary 8 rows plus 3 recovered notebook submissions with protocol caveats"
        ),
        "n": exp_res["n_complete"],
        "rho": exp_res["rho"],
        "p_value": exp_res["p_value"],
        "tie_method": exp_res["tie_method"],
        "alternative": exp_res["alternative"],
        "p_value_method": exp_res["p_value_method"],
        "included_rows": exploratory_rows,
        "recovered_submissions": recovered_submissions_list,
    }

    caveats = {
        "asymptotic_p": (
            "SciPy two-sided p-values rely on asymptotic large-sample approximations; "
            "with small sample sizes (n=8 and n=11), p-values should be interpreted with caution."
        ),
        "ties_handling": "Average ranks are assigned to tied values.",
        "interpretation": (
            "Spearman rank correlation measures monotonic rank association between local CV "
            "and Kaggle public leaderboard scores; it does not imply identical ordering, "
            "causality, model superiority, or private-LB assurance."
        ),
    }

    metadata = {
        "source_table": "leaderboard.md",
        "primary_n": primary_payload["n"],
        "exploratory_n": exploratory_payload["n"],
        "num_recovered": len(recovered_submissions_list),
        "analyses_count": 2,
    }

    return {
        "analyses": {
            "primary": primary_payload,
            "exploratory": exploratory_payload,
        },
        "primary": primary_payload,
        "exploratory": exploratory_payload,
        "caveats": caveats,
        "metadata": metadata,
    }


def format_spearman_correlation_markdown(spearman_data: dict[str, Any]) -> str:
    """Format dual Spearman rank correlation analyses into markdown with managed markers."""
    primary = spearman_data.get("primary", {})
    exploratory = spearman_data.get("exploratory", {})

    def _fmt_p(p: float) -> str:
        if p == 0.0:
            return "0.0000"
        if p < 0.0001:
            return f"{p:.4e}"
        return f"{p:.4f}"

    lines = [
        "<!-- SPEARMAN_CORRELATION_START -->",
        "## Rank Correlation Analysis (Local CV vs. Public Leaderboard)",
        "",
        "### Primary Analysis (Comparable Protocol)",
        (
            f"- **Protocol**: Standard repeated cross-validation "
            f"($n = {primary.get('n', 8)}$ complete pairs)."
        ),
        (
            f"- **Spearman $\\rho$**: **{primary.get('rho', 0.0):.4f}** "
            f"($p = {_fmt_p(primary.get('p_value', 0.0))}$, two-sided asymptotic test)."
        ),
        "- **Tie handling**: Average ranks assigned to tied values.",
    ]

    included = primary.get("included_rows", [])
    if included:
        lines.append(f"- **Included submissions ($n = {len(included)}$)**:")
        for r in included:
            sub = r.get("submission_file", "")
            ls = r.get("local_score", 0.0)
            ps = r.get("public_score", 0.0)
            lines.append(f"  - `{sub}` (Local CV: {ls:.5f}, Public: {ps:.5f})")

    excluded = primary.get("excluded_submissions", [])
    if excluded:
        n_exc = len(excluded)
        lines.append(
            f"- **Excluded submissions ($n = {n_exc}$, lacking standard local CV in main table)**:"
        )
        for sub in excluded:
            lines.append(f"  - `{sub}`")

    recovered = exploratory.get("recovered_submissions", [])
    lines.extend(
        [
            "",
            "### Exploratory Analysis (Mixed Protocol with Recovered Values)",
            (
                f"- **Protocol**: Primary $n = {primary.get('n', 8)}$ submissions plus "
                f"{len(recovered)} recovered notebook "
                f"submissions ($n = {exploratory.get('n', 11)}$ total)."
            ),
            (
                f"- **Spearman $\\rho$**: **{exploratory.get('rho', 0.0):.4f}** "
                f"($p = {_fmt_p(exploratory.get('p_value', 0.0))}$, two-sided asymptotic test)."
            ),
            "- **Recovered submissions & protocol caveats**:",
        ]
    )

    for rec in recovered:
        sub = rec.get("submission_file", "")
        loc = rec.get("local_cv_roc_auc", rec.get("recovered_local_cv_roc_auc", 0.0))
        pub = rec.get("public_score", 0.0)
        source_p = rec.get("source_path", "")
        prec = rec.get("precision_note", "")
        cav = rec.get("caveat", "")

        source_desc = f"Source: `{source_p}`" if source_p else ""
        if prec and source_desc:
            source_desc += f" [{prec}]"

        detail_parts = []
        if source_desc:
            detail_parts.append(source_desc)
        if cav:
            detail_parts.append(f"Caveat: {cav}")

        details = f" ({'; '.join(detail_parts)})" if detail_parts else ""
        lines.append(f"  - `{sub}`: Recovered Local CV = {loc}, Public = {pub:.5f}{details}")

    lines.extend(
        [
            "",
            "### Statistical Caveats & Interpretation",
            (
                "1. **Asymptotic p-values**: SciPy two-sided p-values rely on asymptotic "
                "large-sample approximations; with small sample sizes ($n = 8$ and $n = 11$), "
                "p-values should be interpreted with caution."
            ),
            (
                "2. **Rank association vs. identical ordering**: A strong rank correlation "
                "indicates monotonic association across the metric range, but does not guarantee "
                "identical rank ordering among closely clustered top models."
            ),
            (
                "3. **No causality or model superiority**: Correlation reflects validation "
                "alignment, not causality or definitive proof of model superiority."
            ),
            (
                "4. **No private leaderboard assurance**: Public leaderboard scores represent "
                "a small test subset; correlation with public LB does not provide assurance "
                "against private leaderboard shakeup."
            ),
            "<!-- SPEARMAN_CORRELATION_END -->",
        ]
    )

    return "\n".join(lines)
