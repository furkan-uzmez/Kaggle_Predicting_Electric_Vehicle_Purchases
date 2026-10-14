import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ev_purchase.leaderboard import (  # noqa: E402
    CANONICAL_LEADERBOARD_MODELS,
    EXCLUDED_LEADERBOARD_MODELS,
    RECOVERED_LEADERBOARD_MODELS,
    blend_oof,
    compute_all_pairwise_cis,
    compute_dual_spearman_correlations,
    compute_fold_scores,
    compute_paired_ci,
    compute_spearman_correlation,
    format_pairwise_ci_markdown,
    format_spearman_correlation_markdown,
    holm_adjust,
    load_and_validate_recovered_models,
    paired_fold_comparison,
    parse_public_leaderboard_markdown,
    update_leaderboard_markdown,
    update_managed_section,
)
from scripts.build_leaderboard_artifacts import main  # noqa: E402


def test_compute_fold_scores_happy_path() -> None:
    y_true = np.array([0, 1, 0, 1])
    oof_df = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 0, 1, 2, 3],
            "repeat": [0, 0, 0, 0, 1, 1, 1, 1],
            "fold": [0, 0, 1, 1, 0, 0, 1, 1],
            "probability": [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75],
        }
    )

    res = compute_fold_scores(oof_df, y_true)
    assert list(res.columns) == ["repeat", "fold", "roc_auc"]
    assert len(res) == 4
    assert (res["roc_auc"] == 1.0).all()
    assert res["repeat"].tolist() == [0, 0, 1, 1]
    assert res["fold"].tolist() == [0, 1, 0, 1]


def test_compute_fold_scores_missing_columns_raises() -> None:
    y_true = np.array([0, 1])
    bad_df = pd.DataFrame({"row_index": [0, 1], "repeat": [0, 0], "fold": [0, 0]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_fold_scores(bad_df, y_true)


def test_paired_fold_comparison_happy_path() -> None:
    scores_a = pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "roc_auc": [0.95, 0.93]})
    scores_b = pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "roc_auc": [0.90, 0.89]})

    res = paired_fold_comparison(scores_a, scores_b, label_a="ModelA", label_b="ModelB")

    assert res["label_a"] == "ModelA"
    assert res["label_b"] == "ModelB"
    assert pytest.approx(res["mean_a"]) == 0.94
    assert pytest.approx(res["mean_b"]) == 0.895
    assert pytest.approx(res["mean_diff"]) == 0.045
    diffs = np.array([0.05, 0.04])
    assert pytest.approx(res["paired_std"]) == float(np.std(diffs, ddof=1))
    assert res["n_folds"] == 2
    assert res["abs_ratio"] == pytest.approx(abs(res["mean_diff"]) / res["paired_std"])


def test_paired_fold_comparison_contains_no_verdict_key() -> None:
    scores_a = pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "roc_auc": [0.95, 0.93]})
    scores_b = pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "roc_auc": [0.90, 0.89]})

    res = paired_fold_comparison(scores_a, scores_b, label_a="ModelA", label_b="ModelB")

    forbidden_substrings = ["verdict", "significance", "superior", "p_value", "pvalue", "winner"]
    for key in res.keys():
        for forbidden in forbidden_substrings:
            assert forbidden not in key.lower(), (
                f"Forbidden key component '{forbidden}' found in {key}"
            )


def test_paired_fold_comparison_mismatched_folds_raises() -> None:
    scores_a = pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "roc_auc": [0.95, 0.93]})
    scores_b = pd.DataFrame({"repeat": [0, 0], "fold": [0, 2], "roc_auc": [0.90, 0.89]})

    with pytest.raises(ValueError, match="Fold keys differ"):
        paired_fold_comparison(scores_a, scores_b, label_a="ModelA", label_b="ModelB")


def test_paired_fold_comparison_zero_paired_std_returns_inf() -> None:
    scores_a = pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "roc_auc": [0.95, 0.93]})
    scores_b = pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "roc_auc": [0.90, 0.88]})
    res = paired_fold_comparison(scores_a, scores_b, label_a="ModelA", label_b="ModelB")
    assert res["paired_std"] == 0.0
    assert res["abs_ratio"] == float("inf")


def test_blend_oof_happy_path() -> None:
    f1 = pd.DataFrame(
        {"row_index": [0, 1], "repeat": [0, 0], "fold": [0, 0], "probability": [0.2, 0.8]}
    )
    f2 = pd.DataFrame(
        {"row_index": [0, 1], "repeat": [0, 0], "fold": [0, 0], "probability": [0.4, 0.6]}
    )

    frames = {"m1": f1, "m2": f2}
    weights = {"m1": 0.3, "m2": 0.7}

    res = blend_oof(frames, weights)
    assert list(res.columns) == ["row_index", "repeat", "fold", "probability"]
    assert pytest.approx(res["probability"].iloc[0]) == 0.3 * 0.2 + 0.7 * 0.4
    assert pytest.approx(res["probability"].iloc[1]) == 0.3 * 0.8 + 0.7 * 0.6


def test_blend_oof_value_errors() -> None:
    f1 = pd.DataFrame({"row_index": [0], "repeat": [0], "fold": [0], "probability": [0.5]})
    f2 = pd.DataFrame({"row_index": [0], "repeat": [0], "fold": [0], "probability": [0.5]})

    with pytest.raises(ValueError, match="Weights keys do not match frames keys"):
        blend_oof({"m1": f1, "m2": f2}, {"m1": 1.0})

    with pytest.raises(ValueError, match="negative"):
        blend_oof({"m1": f1, "m2": f2}, {"m1": -0.1, "m2": 1.1})

    with pytest.raises(ValueError, match="sum to 1.0"):
        blend_oof({"m1": f1, "m2": f2}, {"m1": 0.5, "m2": 0.6})


def test_parse_public_leaderboard_markdown() -> None:
    markdown_text = """
# Kaggle Public Leaderboard Results

Recorded on 2026-09-16.

| Rank | Submission | Model | Local CV ROC-AUC | Public Score | Public − CV |
|---:|---|---|---:|---:|---:|
| 1 | `submission_ensemble_catboost30_lightgbm70.csv` | 30% CatBoost HPO + 70% LightGBM HPO | **0.94228** | **0.94208** | -0.00020 |
| 2 | `submission_advanced_engineered.csv` | Advanced Engineered 21 Features | **0.94228** | Pending | N/A |
| 3 | `submission_catboost_hpo_trial22.csv` | CatBoost HPO trial 22 | 0.94195 | 0.94195 | 0.00000 |
| 4 | `submission.csv` | Final tuned LightGBM | 0.94224 | 0.94180 | -0.00044 |
| 5 | `submission_dummy.csv` | Dummy baseline | 0.49999 | 0.50000 | +0.00001 |
"""  # noqa: E501
    df = parse_public_leaderboard_markdown(markdown_text)
    expected_cols = [
        "submission_file",
        "model_label",
        "aggregate_oof_roc_auc",
        "public_score",
        "public_minus_cv",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 5
    assert df["submission_file"].iloc[0] == "submission_ensemble_catboost30_lightgbm70.csv"
    assert df["model_label"].iloc[0] == "30% CatBoost HPO + 70% LightGBM HPO"
    assert pytest.approx(df["aggregate_oof_roc_auc"].iloc[0]) == 0.94228
    assert pytest.approx(df["public_score"].iloc[0]) == 0.94208
    assert np.isnan(df["public_score"].iloc[1])
    assert np.isnan(df["public_minus_cv"].iloc[1])
    assert pytest.approx(df["public_minus_cv"].iloc[0]) == -0.00020
    assert pytest.approx(df["public_minus_cv"].iloc[4]) == 0.00001


def test_script_main_dry_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "leaderboard"
    exit_code = main(["--output-dir", str(output_dir), "--dry-run"])
    assert exit_code == 0
    assert not output_dir.exists() or len(list(output_dir.iterdir())) == 0


def test_parse_public_leaderboard_markdown_rejects_local_cv_column_name() -> None:
    """The ambiguous 'local_cv_roc_auc' name must not reappear (metric collision guard)."""
    markdown_text = """
| Rank | Submission | Model | Local CV ROC-AUC | Public Score | Public − CV |
|---:|---|---|---:|---:|---:|
| 1 | `submission.csv` | Final tuned LightGBM | 0.94224 | 0.94180 | -0.00044 |
"""
    df = parse_public_leaderboard_markdown(markdown_text)
    assert "local_cv_roc_auc" not in df.columns
    assert "aggregate_oof_roc_auc" in df.columns


def test_compute_repeat_averaged_roc_auc_differs_from_pooled() -> None:
    """Repeat-averaging reduces variance, so it is a distinct estimator from pooling."""
    from ev_purchase.leaderboard import compute_repeat_averaged_roc_auc

    oof = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 0, 1, 2, 3],
            "repeat": [0, 0, 0, 0, 1, 1, 1, 1],
            "fold": [0, 0, 1, 1, 0, 0, 1, 1],
            "probability": [0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8],
        }
    )
    y_true = np.array([1, 0, 1, 0])
    score = compute_repeat_averaged_roc_auc(oof, y_true)
    # Averaging 0.9 and 0.1 per row collapses every row to 0.5 -> AUC 0.5.
    assert pytest.approx(score) == 0.5


def test_compute_repeat_averaged_roc_auc_matches_recorded_leaderboard_value() -> None:
    """The value recorded in leaderboard.md must be reproducible as repeat-averaged AUC."""
    from ev_purchase.leaderboard import compute_repeat_averaged_roc_auc

    oof_path = Path("outputs/ensemble/candidates/lightgbm_hpo_trial35/oof_predictions.parquet")
    train_path = Path("data/playground-series-s6e9/train.csv")
    if not oof_path.exists() or not train_path.exists():
        pytest.skip("source artifacts not available")

    y_true = (
        pd.read_csv(train_path, usecols=["Will_Buy_EV"])["Will_Buy_EV"]
        .map({"No": 0, "Yes": 1})
        .to_numpy()
    )
    score = compute_repeat_averaged_roc_auc(pd.read_parquet(oof_path), y_true)
    assert round(score, 5) == 0.94224


def test_parse_public_leaderboard_markdown_keeps_rows_containing_header_words() -> None:
    """A data row whose label contains 'Model'/'Submission' must not be silently dropped."""
    markdown_text = """
| Rank | Submission | Model | Local CV ROC-AUC | Public Score | Public − CV |
|---:|---|---|---:|---:|---:|
| 1 | `a.csv` | Model Soup blend | 0.90000 | 0.80000 | -0.10000 |
| 2 | `b.csv` | Plain LightGBM | 0.70000 | 0.60000 | -0.10000 |
"""
    df = parse_public_leaderboard_markdown(markdown_text)
    assert len(df) == 2, f"Expected 2 data rows, parser returned {len(df)}"
    assert list(df["submission_file"]) == ["a.csv", "b.csv"]
    assert df["model_label"].iloc[0] == "Model Soup blend"


def test_parse_public_leaderboard_markdown_rejects_unparseable_row() -> None:
    """Malformed numeric cells must fail loudly rather than be skipped."""
    markdown_text = """
| Rank | Submission | Model | Local CV ROC-AUC | Public Score | Public − CV |
|---:|---|---|---:|---:|---:|
| 1 | `a.csv` | Some model | not-a-number | 0.80000 | -0.10000 |
"""
    with pytest.raises(ValueError):
        parse_public_leaderboard_markdown(markdown_text)


def test_ensemble_input_sha256_is_content_derived() -> None:
    """Manifest checksum for the derived ensemble must hash content, not just weights."""
    import hashlib

    path = Path("outputs/leaderboard/leaderboard_manifest.json")
    if not path.exists():
        pytest.skip("leaderboard artifacts not generated")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored = manifest["input_sha256"]["ensemble_catboost30_lightgbm70"]
    weights = manifest["blend_weights"]
    weights_only = hashlib.sha256(f"blend:{weights}".encode()).hexdigest()
    assert stored != weights_only, (
        "Ensemble checksum only hashes the weight dict, so it cannot detect changed "
        "predictions. It must be derived from the blended prediction content."
    )


def test_persisted_model_summary_uses_mean_fold_roc_auc() -> None:
    """model_summary.parquet must name its metric unambiguously as a per-fold mean."""
    import pandas as pd

    path = Path("outputs/leaderboard/model_summary.parquet")
    if not path.exists():
        pytest.skip("leaderboard artifacts not generated")
    df = pd.read_parquet(path)
    assert "mean_fold_roc_auc" in df.columns
    assert "mean_roc_auc" not in df.columns


def test_persisted_public_leaderboard_uses_aggregate_oof_roc_auc() -> None:
    """public_leaderboard.parquet must name its metric unambiguously as pooled OOF."""
    import pandas as pd

    path = Path("outputs/leaderboard/public_leaderboard.parquet")
    if not path.exists():
        pytest.skip("leaderboard artifacts not generated")
    df = pd.read_parquet(path)
    assert "aggregate_oof_roc_auc" in df.columns
    assert "local_cv_roc_auc" not in df.columns


# =============================================================================
# Paired CI Analysis & Nadeau-Bengio Statistical Tests
# =============================================================================


def test_compute_paired_ci_known_example() -> None:
    """Validate paired CI calculations against scipy reference values with ddof=1."""
    import scipy.stats

    scores_a = np.array([0.95, 0.94, 0.96, 0.93, 0.92])
    scores_b = np.array([0.90, 0.91, 0.89, 0.88, 0.91])
    diffs = scores_a - scores_b
    n = len(diffs)
    df = n - 1

    res = compute_paired_ci(scores_a, scores_b, test_train_ratio=0.25, confidence_level=0.95)

    assert res["n"] == 5
    assert res["df"] == 4
    assert pytest.approx(res["mean_a"]) == float(np.mean(scores_a))
    assert pytest.approx(res["mean_b"]) == float(np.mean(scores_b))
    mean_diff = float(np.mean(diffs))
    assert pytest.approx(res["mean_difference"]) == mean_diff
    s = float(np.std(diffs, ddof=1))
    assert pytest.approx(res["sd"]) == s

    # Reference naive calculations
    se_naive = s / np.sqrt(n)
    t_crit = float(scipy.stats.t.ppf(0.975, df=df))
    t_stat_naive = mean_diff / se_naive
    p_val_naive = float(2.0 * scipy.stats.t.sf(abs(t_stat_naive), df=df))

    assert pytest.approx(res["se_naive"]) == se_naive
    assert pytest.approx(res["t_stat_naive"]) == t_stat_naive
    assert pytest.approx(res["p_value_naive_raw"]) == p_val_naive
    assert pytest.approx(res["ci_naive_lower"]) == mean_diff - t_crit * se_naive
    assert pytest.approx(res["ci_naive_upper"]) == mean_diff + t_crit * se_naive

    # Reference Nadeau-Bengio calculations (ratio = 0.25)
    se_nb = s * np.sqrt(1.0 / n + 0.25)
    t_stat_nb = mean_diff / se_nb
    p_val_nb = float(2.0 * scipy.stats.t.sf(abs(t_stat_nb), df=df))

    assert pytest.approx(res["se_nb"]) == se_nb
    assert pytest.approx(res["t_stat_nb"]) == t_stat_nb
    assert pytest.approx(res["p_value_nb_raw"]) == p_val_nb
    assert pytest.approx(res["ci_nb_lower"]) == mean_diff - t_crit * se_nb
    assert pytest.approx(res["ci_nb_upper"]) == mean_diff + t_crit * se_nb
    # Naive CI is strictly positive, but NB SE inflation causes the interval to cross zero
    assert res["ci_naive_lower"] > 0
    assert res["ci_nb_lower"] < 0.0 < res["ci_nb_upper"]
    assert res["conclusion"] == "inconclusive"


def test_compute_paired_ci_nb_se_inflation_factor() -> None:
    """Test NB SE inflation sqrt(4.75) relative to naive for n=15 and ratio=0.25."""
    # Create 15 synthetic paired differences
    np.random.seed(42)
    scores_a = 0.94 + np.random.normal(0, 0.01, size=15)
    scores_b = 0.93 + np.random.normal(0, 0.01, size=15)

    res = compute_paired_ci(scores_a, scores_b, test_train_ratio=0.25)

    expected_inflation = np.sqrt(1.0 + 15.0 * 0.25)  # sqrt(4.75)
    assert pytest.approx(res["se_nb"] / res["se_naive"], rel=1e-7) == expected_inflation
    assert pytest.approx(expected_inflation, rel=1e-7) == np.sqrt(4.75)


def test_holm_adjust_monotonicity_and_values() -> None:
    """Test Holm-Bonferroni correction maintains monotonicity and clips at 1.0."""
    # Test known small example
    raw_p = [0.01, 0.04, 0.03]
    # Sorted: p(1)=0.01 (*3 = 0.03), p(2)=0.03 (*2 = 0.06), p(3)=0.04 (*1 = 0.04)
    # Monotonicity cummax: p(1)->0.03, p(2)->0.06, p(3)->max(0.06, 0.04)=0.06
    # Original order restored: [0.03, 0.06, 0.06]
    adj_p = holm_adjust(raw_p)
    assert np.allclose(adj_p, [0.03, 0.06, 0.06])

    # Monotonicity check on arbitrary p-values
    p_seq = [0.001, 0.02, 0.019, 0.05, 0.045, 0.4, 0.35]
    adj_seq = holm_adjust(p_seq)
    # Sorted order of original p-values must have non-decreasing adjusted p-values
    sort_idx = np.argsort(p_seq)
    sorted_adj = adj_seq[sort_idx]
    assert np.all(np.diff(sorted_adj) >= 0.0), "Holm adjusted p-values must be non-decreasing"

    # Clipping to 1.0
    p_large = [0.6, 0.8]
    adj_large = holm_adjust(p_large)
    assert np.all(adj_large <= 1.0)
    assert adj_large.tolist() == [1.0, 1.0]

    # Empty and single
    assert len(holm_adjust([])) == 0
    assert np.allclose(holm_adjust([0.05]), [0.05])


def test_compute_paired_ci_zero_variance_branches() -> None:
    """Test zero variance robustly: mean==0 -> CI [0,0], p=1; nonzero -> degenerate CI, p=0."""
    # Branch 1: Zero mean, zero variance
    scores_a = np.array([0.90, 0.90, 0.90])
    scores_b = np.array([0.90, 0.90, 0.90])
    res_zero = compute_paired_ci(scores_a, scores_b)

    assert res_zero["zero_variance"] is True
    assert res_zero["sd"] == 0.0
    assert res_zero["ci_nb_lower"] == 0.0
    assert res_zero["ci_nb_upper"] == 0.0
    assert res_zero["ci_naive_lower"] == 0.0
    assert res_zero["ci_naive_upper"] == 0.0
    assert res_zero["p_value_nb_raw"] == 1.0
    assert res_zero["p_value_naive_raw"] == 1.0
    assert res_zero["conclusion"] == "inconclusive"

    # Branch 2: Positive mean, zero variance
    scores_pos_a = np.array([0.95, 0.95, 0.95])
    scores_pos_b = np.array([0.90, 0.90, 0.90])
    res_pos = compute_paired_ci(scores_pos_a, scores_pos_b)

    assert res_pos["zero_variance"] is True
    assert res_pos["sd"] == 0.0
    assert pytest.approx(res_pos["ci_nb_lower"]) == 0.05
    assert pytest.approx(res_pos["ci_nb_upper"]) == 0.05
    assert res_pos["p_value_nb_raw"] == 0.0
    assert res_pos["p_value_naive_raw"] == 0.0
    assert res_pos["conclusion"] == "A higher"
    assert "degenerate" in res_pos["note"].lower()

    # Branch 3: Negative mean, zero variance
    scores_neg_a = np.array([0.90, 0.90, 0.90])
    scores_neg_b = np.array([0.95, 0.95, 0.95])
    res_neg = compute_paired_ci(scores_neg_a, scores_neg_b)

    assert res_neg["zero_variance"] is True
    assert pytest.approx(res_neg["ci_nb_lower"]) == -0.05
    assert pytest.approx(res_neg["ci_nb_upper"]) == -0.05
    assert res_neg["p_value_nb_raw"] == 0.0
    assert res_neg["conclusion"] == "B higher"


def test_compute_paired_ci_conclusion_never_equivalent() -> None:
    """Never call zero-containing CI equivalent; must be 'inconclusive'."""
    # Construct differences where 0 is inside the CI
    scores_a = np.array([0.91, 0.89, 0.92, 0.88, 0.90])
    scores_b = np.array([0.90, 0.90, 0.90, 0.90, 0.90])
    res = compute_paired_ci(scores_a, scores_b)

    assert res["ci_nb_lower"] < 0.0 < res["ci_nb_upper"]
    assert res["conclusion"] == "inconclusive"
    assert res["conclusion"] != "equivalent"


def test_compute_all_pairwise_cis_contract() -> None:
    """Verify 21 canonical pairs, 315 fold diffs, ordering, signs, and exclusions."""
    # Build synthetic 15-fold scores for 7 eligible models + 1 internal ineligible model
    repeats = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    folds = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    per_model_df: dict[str, pd.DataFrame] = {}
    for idx, model_name in enumerate(CANONICAL_LEADERBOARD_MODELS):
        base_score = 0.94 - idx * 0.05
        per_model_df[model_name] = pd.DataFrame(
            {
                "repeat": repeats,
                "fold": folds,
                "roc_auc": base_score + np.array([0.001 * (f - 2) for f in folds]),
            }
        )
    # Add an ineligible model
    per_model_df["lightgbm_baseline"] = pd.DataFrame(
        {"repeat": repeats, "fold": folds, "roc_auc": [0.93] * 15}
    )

    pairwise_ci_df, fold_diffs_df = compute_all_pairwise_cis(
        per_model_fold_df=per_model_df,
        models=CANONICAL_LEADERBOARD_MODELS,
        test_train_ratio=0.25,
        split_sha256="test_hash_123",
    )

    # 1. Exactly 21 pairs
    assert len(pairwise_ci_df) == 21
    # 2. Exactly 315 fold differences (21 * 15)
    assert len(fold_diffs_df) == 315

    # 3. No ineligible models present
    models_in_ci = set(pairwise_ci_df["model_a"]).union(set(pairwise_ci_df["model_b"]))
    assert "lightgbm_baseline" not in models_in_ci
    assert "advanced_engineered" not in models_in_ci
    assert models_in_ci == set(CANONICAL_LEADERBOARD_MODELS)

    # 4. Canonical rank ordering: model_a must appear before model_b in CANONICAL_LEADERBOARD_MODELS
    rank_map = {m: i for i, m in enumerate(CANONICAL_LEADERBOARD_MODELS)}
    for _, row in pairwise_ci_df.iterrows():
        assert rank_map[row["model_a"]] < rank_map[row["model_b"]], (
            f"Canonical order violated: {row['model_a']} vs {row['model_b']}"
        )

    # 5. Signs consistent
    for _, row in pairwise_ci_df.iterrows():
        assert pytest.approx(row["mean_difference"]) == row["mean_a"] - row["mean_b"]

    for _, row in fold_diffs_df.iterrows():
        assert pytest.approx(row["difference"]) == row["score_a"] - row["score_b"]

    # 6. Complete matched folds: each pair must have 15 rows with matching repeats and folds
    for (_ma, _mb), grp in fold_diffs_df.groupby(["model_a", "model_b"]):
        assert len(grp) == 15
        assert grp["repeat"].tolist() == repeats
        assert grp["fold"].tolist() == folds


def test_format_pairwise_ci_markdown_and_managed_update() -> None:
    """Markdown format must have markers, 21 rows, caveats, exclusions, and be idempotent."""
    # Synthetic 21-row table
    records = []
    for i in range(21):
        records.append(
            {
                "model_a": f"model_{i}_a",
                "model_b": f"model_{i}_b",
                "mean_a": 0.94,
                "mean_b": 0.93,
                "mean_difference": 0.01,
                "sd": 0.005,
                "se_nb": 0.002,
                "ci_nb_lower": 0.005,
                "ci_nb_upper": 0.015,
                "p_value_nb_raw": 0.001,
                "p_value_nb_holm": 0.021,
                "conclusion": "A higher",
            }
        )
    ci_df = pd.DataFrame(records)
    section_md = format_pairwise_ci_markdown(ci_df, exclusions=EXCLUDED_LEADERBOARD_MODELS)

    assert "<!-- PAIRWISE_CI_START -->" in section_md
    assert "<!-- PAIRWISE_CI_END -->" in section_md
    assert "submission_advanced_engineered.csv" in section_md
    assert "lightgbm_baseline" in section_md
    assert "selection bias" in section_md.lower()
    assert "practical importance" in section_md.lower()

    # Test managed replacement idempotency
    original = "# Leaderboard\n\nSome table here\n"
    updated_once = update_leaderboard_markdown(original, section_md)
    assert "<!-- PAIRWISE_CI_START -->" in updated_once
    assert "Some table here" in updated_once

    # Running update again must produce identical string without duplication
    updated_twice = update_leaderboard_markdown(updated_once, section_md)
    assert updated_twice == updated_once
    assert updated_twice.count("<!-- PAIRWISE_CI_START -->") == 1
    assert updated_twice.count("<!-- PAIRWISE_CI_END -->") == 1


def test_persisted_pairwise_artifacts_integration() -> None:
    """Integration test checking disk artifacts, manifest counts, and fresh hashes."""
    ci_path = Path("outputs/leaderboard/pairwise_confidence_intervals.parquet")
    ci_csv_path = Path("outputs/leaderboard/pairwise_confidence_intervals.csv")
    diff_path = Path("outputs/leaderboard/pairwise_fold_differences.parquet")
    diff_csv_path = Path("outputs/leaderboard/pairwise_fold_differences.csv")
    manifest_path = Path("outputs/leaderboard/leaderboard_manifest.json")
    lb_md_path = Path("leaderboard.md")

    if not ci_path.exists() or not manifest_path.exists():
        pytest.skip("pairwise artifacts not generated yet")

    ci_df = pd.read_parquet(ci_path)
    ci_csv = pd.read_csv(ci_csv_path)
    diff_df = pd.read_parquet(diff_path)
    diff_csv = pd.read_csv(diff_csv_path)

    # 21 pairs, 315 fold differences
    assert len(ci_df) == 21
    assert len(ci_csv) == 21
    assert len(diff_df) == 315
    assert len(diff_csv) == 315

    # Rich schema validation
    expected_cols = {
        "model_a",
        "model_b",
        "mean_a",
        "mean_b",
        "mean_difference",
        "sd",
        "n",
        "df",
        "test_train_ratio",
        "se_naive",
        "ci_naive_lower",
        "ci_naive_upper",
        "t_stat_naive",
        "p_value_naive_raw",
        "p_value_naive_holm",
        "se_nb",
        "ci_nb_lower",
        "ci_nb_upper",
        "t_stat_nb",
        "p_value_nb_raw",
        "p_value_nb_holm",
        "zero_variance",
        "conclusion",
        "metric",
    }
    assert expected_cols.issubset(set(ci_df.columns)), (
        f"Missing columns: {expected_cols - set(ci_df.columns)}"
    )

    # Manifest row counts and hash freshness
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_counts"]["pairwise_confidence_intervals.parquet"] == 21
    assert manifest["row_counts"]["pairwise_confidence_intervals.csv"] == 21
    assert manifest["row_counts"]["pairwise_fold_differences.parquet"] == 315
    assert manifest["row_counts"]["pairwise_fold_differences.csv"] == 315

    # Recomputed leaderboard_md_sha256 must match current file on disk
    import hashlib

    current_lb_sha = hashlib.sha256(lb_md_path.read_bytes()).hexdigest()
    assert manifest["leaderboard_md_sha256"] == current_lb_sha


# =============================================================================
# Spearman Rank Correlation Tests
# =============================================================================


def test_compute_spearman_correlation_known_data_and_ties() -> None:
    """Validate Spearman rank correlation calculation with tied ranks and exclusions."""
    import scipy.stats

    df = pd.DataFrame(
        {
            "submission_file": ["s1", "s2", "s3", "s4", "s_nan_cv", "s_nan_pub"],
            "aggregate_oof_roc_auc": [0.90, 0.90, 0.85, 0.80, np.nan, 0.75],
            "public_score": [0.92, 0.91, 0.84, 0.81, 0.70, np.nan],
        }
    )

    res = compute_spearman_correlation(df)
    assert res["n_total"] == 6
    assert res["n_complete"] == 4
    assert res["excluded_submissions"] == ["s_nan_cv", "s_nan_pub"]

    # Compare with direct scipy.stats.spearmanr on complete cases
    ref = scipy.stats.spearmanr([0.90, 0.90, 0.85, 0.80], [0.92, 0.91, 0.84, 0.81])
    assert pytest.approx(res["rho"]) == float(ref.statistic)
    assert pytest.approx(res["p_value"]) == float(ref.pvalue)
    assert res["tie_method"] == "average ranks"
    assert res["alternative"] == "two-sided"
    assert "asymptotic" in res["p_value_method"].lower()


def test_compute_dual_spearman_correlations() -> None:
    """Verify dual Spearman analyses: primary (n=8) and exploratory (n=11)."""
    assert "super_ensemble_pipeline - Version 7" in RECOVERED_LEADERBOARD_MODELS
    assert "lightgbm - Version 1" in RECOVERED_LEADERBOARD_MODELS
    assert "ft-transformer - Version 2" in RECOVERED_LEADERBOARD_MODELS

    cv = [0.94228, 0.94228, 0.94195, 0.94224, 0.94192, 0.94179, 0.93810, 0.49999]
    pub = [0.94208, 0.94191, 0.94195, 0.94180, 0.94177, 0.94163, 0.93738, 0.50000]
    df = pd.DataFrame(
        {
            "submission_file": [
                "s_ens",
                "s_adv",
                "s_cb_hpo",
                "s_lgb_final",
                "s_cb_base",
                "s_xgb_base",
                "super_ensemble_pipeline - Version 7",
                "lightgbm - Version 1",
                "ft-transformer - Version 2",
                "s_logistic",
                "s_dummy",
            ],
            "model_label": [
                "Ensemble",
                "Adv Eng",
                "CB HPO",
                "LGB Final",
                "CB Base",
                "XGB Base",
                "Super Ensemble Pipeline",
                "LightGBM Pipeline",
                "FT-Transformer Pipeline",
                "Logistic",
                "Dummy",
            ],
            "aggregate_oof_roc_auc": [
                cv[0],
                cv[1],
                cv[2],
                cv[3],
                cv[4],
                cv[5],
                np.nan,
                np.nan,
                np.nan,
                cv[6],
                cv[7],
            ],
            "public_score": [
                pub[0],
                pub[1],
                pub[2],
                pub[3],
                pub[4],
                pub[5],
                0.94161,
                0.94157,
                0.93807,
                pub[6],
                pub[7],
            ],
            "public_minus_cv": [
                -0.00020,
                -0.00037,
                0.0,
                -0.00044,
                -0.00015,
                -0.00016,
                np.nan,
                np.nan,
                np.nan,
                -0.00072,
                0.00001,
            ],
        }
    )

    res = compute_dual_spearman_correlations(df)

    # Primary analysis checks
    prim = res["primary"]
    assert prim["n"] == 8
    assert pytest.approx(prim["rho"], rel=1e-10) == 0.9101959590540767
    assert pytest.approx(prim["p_value"], rel=1e-10) == 0.0016908608862384285
    assert len(prim["excluded_submissions"]) == 3

    # Exploratory analysis checks
    exp = res["exploratory"]
    assert exp["n"] == 11
    assert pytest.approx(exp["rho"], rel=1e-10) == 0.8929408132532823
    assert pytest.approx(exp["p_value"], rel=1e-10) == 0.00021517143044230897
    assert len(exp["recovered_submissions"]) == 3

    # Check recovered values and caveats
    recovered_map = {r["submission_file"]: r for r in exp["recovered_submissions"]}
    assert "super_ensemble_pipeline - Version 7" in recovered_map
    sep_rec = recovered_map["super_ensemble_pipeline - Version 7"]
    assert pytest.approx(sep_rec["local_cv_roc_auc"]) == 0.942104
    assert pytest.approx(sep_rec["public_score"]) == 0.94161
    assert "preprocessing leakage" in sep_rec["caveat"].lower()

    assert "lightgbm - Version 1" in recovered_map
    lgb_rec = recovered_map["lightgbm - Version 1"]
    assert pytest.approx(lgb_rec["local_cv_roc_auc"]) == 0.9415344361329914
    assert pytest.approx(lgb_rec["public_score"]) == 0.94157
    assert "protocol sample differs" in lgb_rec["caveat"].lower()

    assert "ft-transformer - Version 2" in recovered_map
    ft_rec = recovered_map["ft-transformer - Version 2"]
    assert pytest.approx(ft_rec["local_cv_roc_auc"]) == 0.9387246162841222
    assert pytest.approx(ft_rec["public_score"]) == 0.93807
    assert "5-fold not 5x3" in ft_rec["caveat"].lower()

    # Caveats & metadata
    caveats = res["caveats"]
    assert "asymptotic" in caveats["asymptotic_p"].lower()
    assert "average" in caveats["ties_handling"].lower()
    assert "rank association" in caveats["interpretation"].lower()
    assert "identical ordering" in caveats["interpretation"].lower()
    assert "causality" in caveats["interpretation"].lower()
    assert "superiority" in caveats["interpretation"].lower()
    assert "private" in caveats["interpretation"].lower()


def test_format_spearman_correlation_markdown_and_idempotence() -> None:
    """Verify markdown formatting with markers and idempotent managed replacement."""
    spearman_data = {
        "primary": {
            "n": 8,
            "rho": 0.9101959590540767,
            "p_value": 0.0016908608862384285,
            "included_rows": [
                {"submission_file": "sub1.csv", "local_score": 0.94228, "public_score": 0.94208},
            ],
            "excluded_submissions": ["super_ensemble_pipeline - Version 7"],
        },
        "exploratory": {
            "n": 11,
            "rho": 0.8929408132532823,
            "p_value": 0.00021517143044230897,
            "recovered_submissions": [
                {
                    "submission_file": "super_ensemble_pipeline - Version 7",
                    "local_cv_roc_auc": 0.942104,
                    "public_score": 0.94161,
                    "caveat": "preprocessing leakage concern.",
                },
                {
                    "submission_file": "lightgbm - Version 1",
                    "local_cv_roc_auc": 0.9415344361329914,
                    "public_score": 0.94157,
                    "caveat": "development OOF/protocol sample differs.",
                },
                {
                    "submission_file": "ft-transformer - Version 2",
                    "local_cv_roc_auc": 0.9387246162841222,
                    "public_score": 0.93807,
                    "caveat": "full-data 5-fold not 5x3 repeated CV.",
                },
            ],
        },
        "caveats": {
            "asymptotic_p": "SciPy two-sided p-values are asymptotic approximations.",
            "ties_handling": "Average ranks are used for ties.",
            "interpretation": (
                "Correlation indicates monotonic rank association, not identical ordering, "
                "causality, model superiority, or private-LB assurance."
            ),
        },
    }
    md_section = format_spearman_correlation_markdown(spearman_data)

    assert "<!-- SPEARMAN_CORRELATION_START -->" in md_section
    assert "<!-- SPEARMAN_CORRELATION_END -->" in md_section
    assert "Primary Analysis" in md_section
    assert "Exploratory Analysis" in md_section
    assert "0.9102" in md_section
    assert "0.0017" in md_section
    assert "0.8929" in md_section
    assert "0.0002" in md_section
    assert "super_ensemble_pipeline - Version 7" in md_section
    assert "preprocessing leakage" in md_section.lower()
    assert "asymptotic" in md_section.lower()
    assert "identical ordering" in md_section.lower()
    assert "causality" in md_section.lower()
    assert "superiority" in md_section.lower()
    assert "private" in md_section.lower()

    # Test insertion before pairwise CI section
    base_text = (
        "# Header\n\nTable\n\n"
        "<!-- PAIRWISE_CI_START -->\nPairwise content\n<!-- PAIRWISE_CI_END -->\n"
    )
    updated_once = update_managed_section(
        base_text,
        start_marker="<!-- SPEARMAN_CORRELATION_START -->",
        end_marker="<!-- SPEARMAN_CORRELATION_END -->",
        new_content=md_section,
        insert_before="<!-- PAIRWISE_CI_START -->",
    )
    assert "<!-- SPEARMAN_CORRELATION_START -->" in updated_once
    # Section should be placed before pairwise section
    pos_spearman = updated_once.index("<!-- SPEARMAN_CORRELATION_START -->")
    pos_pairwise = updated_once.index("<!-- PAIRWISE_CI_START -->")
    assert pos_spearman < pos_pairwise

    # Second update must be idempotent (identical string, no duplicated markers)
    updated_twice = update_managed_section(
        updated_once,
        start_marker="<!-- SPEARMAN_CORRELATION_START -->",
        end_marker="<!-- SPEARMAN_CORRELATION_END -->",
        new_content=md_section,
        insert_before="<!-- PAIRWISE_CI_START -->",
    )
    assert updated_twice == updated_once
    assert updated_twice.count("<!-- SPEARMAN_CORRELATION_START -->") == 1
    assert updated_twice.count("<!-- SPEARMAN_CORRELATION_END -->") == 1


def test_recovered_models_source_auditing_and_extraction() -> None:
    """Validate recovered models have persisted source artifacts, hashes, and matching scores."""
    import hashlib

    resolved = load_and_validate_recovered_models(project_root=PROJECT_ROOT)
    assert len(resolved) == 3

    # Super Ensemble
    assert "super_ensemble_pipeline - Version 7" in resolved
    sep = resolved["super_ensemble_pipeline - Version 7"]
    sep_path = PROJECT_ROOT / sep["source_path"]
    assert sep_path.exists()
    assert sep["source_sha256"] == hashlib.sha256(sep_path.read_bytes()).hexdigest()
    assert pytest.approx(sep["extracted_score"]) == 0.942104
    assert pytest.approx(sep["local_cv_roc_auc"]) == 0.942104
    assert pytest.approx(sep["public_score"]) == 0.94161
    assert "5x5 repeated cv" in sep["caveat"].lower()
    assert "25 folds" in sep["caveat"].lower()
    assert "5x3" in sep["caveat"].lower()
    assert "15 folds" in sep["caveat"].lower()
    assert "preprocessing leakage" in sep["caveat"].lower()

    # LightGBM
    assert "lightgbm - Version 1" in resolved
    lgb = resolved["lightgbm - Version 1"]
    lgb_path = PROJECT_ROOT / lgb["source_path"]
    assert lgb_path.exists()
    assert lgb["source_sha256"] == hashlib.sha256(lgb_path.read_bytes()).hexdigest()
    assert pytest.approx(lgb["extracted_score"]) == 0.9415344361329914
    assert pytest.approx(lgb["local_cv_roc_auc"]) == 0.9415344361329914
    assert pytest.approx(lgb["public_score"]) == 0.94157
    assert "development oof" in lgb["caveat"].lower()

    # FT-Transformer
    assert "ft-transformer - Version 2" in resolved
    ft = resolved["ft-transformer - Version 2"]
    ft_path = PROJECT_ROOT / ft["source_path"]
    assert ft_path.exists()
    assert ft["source_sha256"] == hashlib.sha256(ft_path.read_bytes()).hexdigest()
    assert pytest.approx(ft["extracted_score"]) == 0.9387246162841222
    assert pytest.approx(ft["local_cv_roc_auc"]) == 0.9387246162841222
    assert pytest.approx(ft["public_score"]) == 0.93807
    assert "5-fold not 5x3" in ft["caveat"].lower()


def test_persisted_dual_spearman_artifact_integration() -> None:
    """Integration test checking spearman_correlations.json, manifest, and fresh hash."""
    spearman_path = Path("outputs/leaderboard/spearman_correlations.json")
    manifest_path = Path("outputs/leaderboard/leaderboard_manifest.json")
    lb_md_path = Path("leaderboard.md")

    if not spearman_path.exists() or not manifest_path.exists():
        pytest.skip("spearman artifacts not generated yet")

    data = json.loads(spearman_path.read_text(encoding="utf-8"))
    assert "primary" in data
    assert "exploratory" in data
    assert data["primary"]["n"] == 8
    assert data["exploratory"]["n"] == 11
    assert pytest.approx(data["primary"]["rho"], rel=1e-7) == 0.9101959590540767
    assert pytest.approx(data["primary"]["p_value"], rel=1e-7) == 0.0016908608862384285
    assert pytest.approx(data["exploratory"]["rho"], rel=1e-7) == 0.8929408132532823
    assert pytest.approx(data["exploratory"]["p_value"], rel=1e-7) == 0.00021517143044230897

    # Check manifest row count is 2 (dual analyses)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_counts"]["spearman_correlations.json"] == 2
    import hashlib

    expected_sha = hashlib.sha256(spearman_path.read_bytes()).hexdigest()
    assert manifest["artifact_sha256"]["spearman_correlations.json"] == expected_sha

    # Check that manifest has sources and input_sha256 for recovered models
    for m in [
        "super_ensemble_pipeline - Version 7",
        "lightgbm - Version 1",
        "ft-transformer - Version 2",
    ]:
        assert m in manifest["sources"]
        assert m in manifest["input_sha256"]
        m_path = PROJECT_ROOT / manifest["sources"][m]
        assert manifest["input_sha256"][m] == hashlib.sha256(m_path.read_bytes()).hexdigest()

    # Check that recovered submissions in spearman_correlations.json contain audit metadata
    exp_rec = {r["submission_file"]: r for r in data["exploratory"]["recovered_submissions"]}
    for m in [
        "super_ensemble_pipeline - Version 7",
        "lightgbm - Version 1",
        "ft-transformer - Version 2",
    ]:
        assert m in exp_rec
        assert exp_rec[m]["source_path"].startswith("outputs/runs/kaggle/")
        assert len(exp_rec[m]["source_sha256"]) == 64
        assert exp_rec[m]["run_identifier"]

    # Check leaderboard.md sha256 in manifest matches disk
    current_lb_sha = hashlib.sha256(lb_md_path.read_bytes()).hexdigest()
    assert manifest["leaderboard_md_sha256"] == current_lb_sha

    # Check that leaderboard.md original table still contains N/A for rows 7, 8, 9
    lb_text = lb_md_path.read_text(encoding="utf-8")
    row_7 = (
        "| 7 | `super_ensemble_pipeline - Version 7` | "
        "Super Ensemble Pipeline | N/A | 0.94161 | N/A |"
    )
    assert row_7 in lb_text
    assert "| 8 | `lightgbm - Version 1` | LightGBM Pipeline | N/A | 0.94157 | N/A |" in lb_text
    assert (
        "| 9 | `ft-transformer - Version 2` | FT-Transformer Pipeline | N/A | 0.93807 | N/A |"
        in lb_text
    )

    # Check human-readable source path in leaderboard.md
    assert "outputs/runs/kaggle/super-ensemble-pipeline/super-ensemble-pipeline.ipynb" in lb_text
    assert "outputs/runs/kaggle/lightgbm/results/experiment_manifest.json" in lb_text
    assert "outputs/runs/kaggle/ft-transformer/results/oof_predictions.csv" in lb_text


def test_builder_byte_idempotency(tmp_path: Path) -> None:
    """Building leaderboard artifacts multiple times produces byte-identical outputs."""
    import hashlib

    from scripts.build_leaderboard_artifacts import build_leaderboard_artifacts

    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    build_leaderboard_artifacts(project_root=PROJECT_ROOT, output_dir=out_dir_1)
    build_leaderboard_artifacts(project_root=PROJECT_ROOT, output_dir=out_dir_2)

    files_1 = sorted([p.name for p in out_dir_1.iterdir()])
    files_2 = sorted([p.name for p in out_dir_2.iterdir()])
    assert files_1 == files_2

    for fname in files_1:
        bytes_1 = (out_dir_1 / fname).read_bytes()
        bytes_2 = (out_dir_2 / fname).read_bytes()
        sha1 = hashlib.sha256(bytes_1).hexdigest()
        sha2 = hashlib.sha256(bytes_2).hexdigest()
        assert sha1 == sha2, f"Artifact {fname} is not byte-identical between builds."
