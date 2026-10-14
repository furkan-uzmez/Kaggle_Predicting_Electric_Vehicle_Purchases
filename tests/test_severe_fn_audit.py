"""Unit and contract tests for severe false-negative and learnability audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ev_purchase.adversarial_validation import compute_file_sha256
from ev_purchase.severe_fn_audit import (
    compute_coarse_cell_stats,
    compute_neighborhood_mixing,
    compute_sample_log_loss,
    evaluate_predeclared_slices,
    prepare_neighborhood_features,
    reconstruct_ensemble_oof,
    reconstruct_mask_from_predicate,
    run_severe_fn_audit,
    select_deterministic_top_loss,
    verify_and_align_oof_candidates,
    wilson_score_interval,
)


def _make_dummy_oof_frame(
    n_rows: int = 100,
    n_repeats: int = 3,
    n_splits: int = 5,
    base_prob: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(n_repeats):
        for fold in range(n_splits):
            fold_rows = [i for i in range(n_rows) if i % n_splits == fold]
            for r in fold_rows:
                records.append(
                    {
                        "row_index": r,
                        "repeat": rep,
                        "fold": fold,
                        "probability": float(
                            np.clip(base_prob + rng.normal(0, 0.05), 0.001, 0.999)
                        ),
                    }
                )
    return pd.DataFrame(records).sort_values(["repeat", "fold", "row_index"]).reset_index(drop=True)


def _make_dummy_train_df(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "Age": rng.integers(18, 80, size=n_rows),
            "Annual_Income_USD": rng.uniform(20000, 150000, size=n_rows),
            "Daily_Commute_km": rng.uniform(5, 80, size=n_rows),
            "Number_of_Cars_Owned": rng.choice([1, 2, 3, 4], size=n_rows),
            "Charging_Stations_Near_Home": rng.integers(0, 10, size=n_rows),
            "Charging_Stations_Near_Work": rng.integers(0, 10, size=n_rows),
            "Environmental_Concern_Level": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_rows),
            "Gender": rng.choice(["Male", "Female", "Other"], size=n_rows),
            "City_Type": rng.choice(["Urban", "Suburban", "Rural"], size=n_rows),
            "Current_Car_Type": rng.choice(["Sedan", "SUV", "Hatchback", "Truck"], size=n_rows),
            "Home_Charging_Possible": rng.choice(["Yes", "No"], size=n_rows),
            "Subsidy_Available": rng.choice(["Yes", "No"], size=n_rows),
            "Range_Anxiety_Level": rng.choice(["Low", "Medium", "High"], size=n_rows),
            "Will_Buy_EV": rng.choice(["No", "Yes"], p=[0.8, 0.2], size=n_rows),
        }
    )


def test_wilson_score_interval_sanity() -> None:
    # Standard cases
    low, up = wilson_score_interval(50, 100, confidence=0.95)
    assert 0.39 < low < 0.42
    assert 0.58 < up < 0.61
    assert low < 0.5 < up

    # Zero successes
    low_0, up_0 = wilson_score_interval(0, 100, confidence=0.95)
    assert low_0 == 0.0
    assert 0.0 < up_0 < 0.05

    # All successes
    low_100, up_100 = wilson_score_interval(100, 100, confidence=0.95)
    assert 0.95 < low_100 < 1.0
    assert up_100 == 1.0

    # Edge cases
    assert wilson_score_interval(0, 0) == (0.0, 0.0)

    with pytest.raises(ValueError, match="k must be between 0 and n"):
        wilson_score_interval(105, 100)

    with pytest.raises(ValueError, match="k must be between 0 and n"):
        wilson_score_interval(-1, 100)


def test_compute_sample_log_loss() -> None:
    y_true = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.1, 0.9])
    losses = compute_sample_log_loss(y_true, p)
    assert len(losses) == 4
    # Confident correct: -log(0.9) ~ 0.105
    assert np.isclose(losses[0], -np.log(0.9), atol=1e-5)
    assert np.isclose(losses[1], -np.log(0.9), atol=1e-5)
    # Confident wrong: -log(0.1) ~ 2.302
    assert np.isclose(losses[2], -np.log(0.1), atol=1e-5)
    assert np.isclose(losses[3], -np.log(0.1), atol=1e-5)


def test_exact_row_key_alignment_rejection() -> None:
    lgb_df = _make_dummy_oof_frame(n_rows=50)
    cb_df = _make_dummy_oof_frame(n_rows=50)

    # Valid alignment passes
    merged = reconstruct_ensemble_oof(lgb_df, cb_df)
    assert len(merged) == 50

    # Mismatched row count
    cb_bad_count = cb_df.iloc[:-5].copy()
    with pytest.raises(ValueError, match="Mismatched OOF row count"):
        verify_and_align_oof_candidates(lgb_df, cb_bad_count)

    # Key mismatch (different row index)
    cb_bad_keys = cb_df.copy()
    cb_bad_keys.loc[0, "row_index"] = 999
    with pytest.raises(ValueError, match="Row key mismatch"):
        verify_and_align_oof_candidates(lgb_df, cb_bad_keys)

    # Duplicate keys (same length, but duplicate key)
    cb_dup = cb_df.copy()
    cb_dup.iloc[-1] = cb_dup.iloc[0]
    with pytest.raises(ValueError, match="Duplicate row keys"):
        verify_and_align_oof_candidates(lgb_df, cb_dup)


def test_oof_validation_all_failure_modes() -> None:
    base_lgb = _make_dummy_oof_frame(n_rows=50, n_repeats=3, n_splits=5)
    base_cb = _make_dummy_oof_frame(n_rows=50, n_repeats=3, n_splits=5)

    # 1. Non-finite probabilities
    bad_prob = base_lgb.copy()
    bad_prob.loc[0, "probability"] = np.nan
    with pytest.raises(ValueError, match="Probabilities must be finite"):
        verify_and_align_oof_candidates(bad_prob, base_cb)

    # 2. Probability < 0.0
    bad_prob_low = base_lgb.copy()
    bad_prob_low.loc[0, "probability"] = -0.01
    with pytest.raises(ValueError, match="Probabilities must be in"):
        verify_and_align_oof_candidates(bad_prob_low, base_cb)

    # 3. Probability > 1.0
    bad_prob_high = base_lgb.copy()
    bad_prob_high.loc[0, "probability"] = 1.05
    with pytest.raises(ValueError, match="Probabilities must be in"):
        verify_and_align_oof_candidates(bad_prob_high, base_cb)

    # 4. Invalid repeat IDs (e.g. repeats are 1, 2, 3 instead of 0, 1, 2)
    bad_rep_ids = base_lgb.copy()
    bad_rep_ids["repeat"] = bad_rep_ids["repeat"] + 1
    with pytest.raises(ValueError, match="Repeat IDs mismatch"):
        verify_and_align_oof_candidates(bad_rep_ids, base_cb, expected_repeats=3)

    # 5. Invalid fold IDs (e.g. fold 99 present)
    bad_fold = base_lgb.copy()
    bad_fold.loc[0, "fold"] = 99
    with pytest.raises(ValueError, match="Fold IDs mismatch"):
        verify_and_align_oof_candidates(bad_fold, base_cb, expected_folds=5)

    # 6. Missing row in a repeat (incomplete row coverage)
    bad_coverage = base_lgb.copy()
    # Drop one row and replace with duplicate from another row
    bad_coverage = bad_coverage[~((bad_coverage["row_index"] == 5) & (bad_coverage["repeat"] == 0))]
    dup_row = bad_coverage[(bad_coverage["row_index"] == 6) & (bad_coverage["repeat"] == 0)].copy()
    bad_coverage = pd.concat([bad_coverage, dup_row], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate row keys|Row coverage incomplete"):
        verify_and_align_oof_candidates(bad_coverage, base_cb, expected_n_rows=50)


def test_oof_3_repeat_coverage() -> None:
    lgb_2rep = _make_dummy_oof_frame(n_rows=50, n_repeats=2)
    cb_2rep = _make_dummy_oof_frame(n_rows=50, n_repeats=2)

    with pytest.raises(ValueError, match="Repeat IDs mismatch|Expected 3 repeats"):
        verify_and_align_oof_candidates(lgb_2rep, cb_2rep, expected_repeats=3)


def test_deterministic_top_1pct_ties() -> None:
    n_rows = 1000
    y_true = np.zeros(n_rows, dtype=int)
    y_true[:100] = 1

    # Create artificial ties at boundary
    p = np.full(n_rows, 0.5)
    p[:5] = 0.01  # very high loss for y=1: -log(0.01) ~ 4.6
    p[5:25] = 0.05  # identical loss: -log(0.05) ~ 2.995

    losses = compute_sample_log_loss(y_true, p)
    df = pd.DataFrame({"row_index": np.arange(n_rows), "sample_log_loss": losses})

    # Run selection
    top_df_1, q_val_1 = select_deterministic_top_loss(df, quantile=0.99, n_top=10)

    # Permute original order and run again
    df_perm = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    top_df_2, q_val_2 = select_deterministic_top_loss(df_perm, quantile=0.99, n_top=10)

    assert q_val_1 == q_val_2
    assert len(top_df_1) == 10
    assert len(top_df_2) == 10
    # Strict determinism: identical rows in identical order broken by row_index
    pd.testing.assert_frame_equal(
        top_df_1.reset_index(drop=True), top_df_2.reset_index(drop=True)
    )
    # Check that rank is strictly consecutive 1..10
    assert top_df_1["loss_rank"].tolist() == list(range(1, 11))


def test_no_target_in_neighborhood_features() -> None:
    train_df = _make_dummy_train_df(n_rows=50)
    train_df["p_canonical_mean"] = 0.2
    train_df["sample_log_loss"] = 0.5

    X_mat, used_cols = prepare_neighborhood_features(
        train_df, id_col="id", target_col="Will_Buy_EV"
    )

    assert "Will_Buy_EV" not in used_cols
    assert "id" not in used_cols
    assert "p_canonical_mean" not in used_cols
    assert "sample_log_loss" not in used_cols
    assert X_mat.shape[0] == 50
    assert X_mat.shape[1] > 0
    assert np.all(np.isfinite(X_mat))


def test_neighborhood_robust_self_exclusion_with_duplicates() -> None:
    # Row 0 and Row 1 have EXACT duplicate features
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],  # duplicate of row 0
            [1.05, 2.05],
            [10.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    y = np.array([1, 0, 0, 1, 1], dtype=int)

    # Query row 0: its nearest neighbor with distance 0 is row 1 (y=0)
    # Self-exclusion must exclude row 0, but KEEP row 1!
    pos_frac, mean_dist, disagreement = compute_neighborhood_mixing(
        X_all=X, y_true=y, query_indices=np.array([0]), n_neighbors=2
    )

    assert len(pos_frac) == 1
    # Neighbor 1 is row 1 (y=0, dist=0.0), neighbor 2 is row 2 (y=0, dist~0.07)
    assert pos_frac[0] == 0.0
    assert disagreement[0] == 1.0  # |1 - 0| = 1


def test_synthetic_known_cohorts() -> None:
    train_df = _make_dummy_train_df(n_rows=100)
    train_df["Will_Buy_EV"] = "No"

    # Set 5 known severe FNs: y=1 and mean p <= 0.10
    severe_fn_indices = [5, 12, 27, 42, 88]
    train_df.loc[severe_fn_indices, "Will_Buy_EV"] = "Yes"

    lgb_df = _make_dummy_oof_frame(n_rows=100, base_prob=0.4)
    cb_df = _make_dummy_oof_frame(n_rows=100, base_prob=0.4)

    # Force low predictions on severe FN rows
    lgb_df.loc[lgb_df["row_index"].isin(severe_fn_indices), "probability"] = 0.05
    cb_df.loc[cb_df["row_index"].isin(severe_fn_indices), "probability"] = 0.06

    ens_df = reconstruct_ensemble_oof(lgb_df, cb_df)
    y_true = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()

    # Canonical baseline is LightGBM (single model winning policy)
    severe_fn_mask = (y_true == 1) & (ens_df["p_canonical_mean"] <= 0.10)
    assert severe_fn_mask.sum() == 5
    assert sorted(np.where(severe_fn_mask)[0].tolist()) == severe_fn_indices

    # Margin bands
    ens_df.loc[0:9, "p_canonical_mean"] = 0.50
    m_25_75 = (ens_df["p_canonical_mean"] >= 0.25) & (ens_df["p_canonical_mean"] <= 0.75)
    m_40_60 = (ens_df["p_canonical_mean"] >= 0.40) & (ens_df["p_canonical_mean"] <= 0.60)
    assert m_25_75.sum() >= 10
    assert m_40_60.sum() >= 10


def test_reconstruct_ensemble_oof_canonical_and_exploratory() -> None:
    lgb_df = _make_dummy_oof_frame(n_rows=20, base_prob=0.8)
    cb_df = _make_dummy_oof_frame(n_rows=20, base_prob=0.2)

    ens_df = reconstruct_ensemble_oof(lgb_df, cb_df)
    assert len(ens_df) == 20

    # Canonical is single LightGBM
    np.testing.assert_allclose(ens_df["p_canonical_mean"], ens_df["p_lgb_mean"])

    # Exploratory 30/70 sensitivity
    expected_30_70 = 0.70 * ens_df["p_lgb_mean"] + 0.30 * ens_df["p_cb_mean"]
    np.testing.assert_allclose(ens_df["p_exploratory_30_70_mean"], expected_30_70)

    # CB-LGB abs disagreement
    expected_diff = np.abs(ens_df["p_cb_mean"] - ens_df["p_lgb_mean"])
    np.testing.assert_allclose(ens_df["cb_lgb_abs_diff"], expected_diff)


def test_frozen_slices_reconstructable_predicates() -> None:
    train_df = _make_dummy_train_df(n_rows=200)
    slices, predicates = evaluate_predeclared_slices(train_df, return_predicates=True)

    assert len(slices) == len(predicates)
    # Test every slice can be reconstructed from its predicate
    for slice_name, mask in slices.items():
        pred = predicates[slice_name]
        reconstructed = reconstruct_mask_from_predicate(train_df, pred)
        pd.testing.assert_series_equal(mask, reconstructed, check_names=False)


def test_coarse_cell_stats_prevalence_and_conditional_fn_rates() -> None:
    train_df = _make_dummy_train_df(n_rows=200)
    y_true = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    p_canonical = np.full(200, 0.2)
    # 5 positives have p <= 0.10
    pos_idx = np.where(y_true == 1)[0]
    p_canonical[pos_idx[:5]] = 0.05
    is_severe_fn = (y_true == 1) & (p_canonical <= 0.10)
    sample_loss = compute_sample_log_loss(y_true, p_canonical)

    slices = evaluate_predeclared_slices(train_df)
    stats_df = compute_coarse_cell_stats(
        train_df=train_df,
        y_true=y_true,
        p_canonical_mean=p_canonical,
        is_severe_fn=is_severe_fn,
        sample_log_loss=sample_loss,
        slices=slices,
    )

    assert "cell_name" in stats_df.columns
    assert "positive_rate" in stats_df.columns
    assert "wilson_ci_lower" in stats_df.columns
    assert "wilson_ci_upper" in stats_df.columns
    # Both denominators reported separately:
    assert "severe_fn_prevalence_total" in stats_df.columns
    assert "severe_fn_rate_among_positives" in stats_df.columns
    assert "severe_fn_enrichment" in stats_df.columns

    for _, row in stats_df.iterrows():
        assert 0.0 <= row["wilson_ci_lower"] <= row["wilson_ci_upper"] <= 1.0


def test_report_data_consistency_across_scenarios(tmp_path: Path) -> None:
    # Scenario A: Low severe FNs (0 rows)
    train_a = _make_dummy_train_df(n_rows=50)
    train_a["Will_Buy_EV"] = "No"
    lgb_a = _make_dummy_oof_frame(n_rows=50, base_prob=0.8)
    cb_a = _make_dummy_oof_frame(n_rows=50, base_prob=0.8)
    out_a = tmp_path / "scenario_a"

    res_a = run_severe_fn_audit(
        train_df=train_a,
        lgb_oof_df=lgb_a,
        cb_oof_df=cb_a,
        output_dir=out_a,
        source_provenance={"test": "a"},
    )
    assert res_a.severe_fn_count == 0
    report_a = (out_a / "report.md").read_text()
    assert "0 rows (0.0000%)" in report_a
    assert "No severe false negatives observed" in report_a or "0 rows" in report_a
    # No false claims of 90% or 59% when there are 0 FNs
    assert "Over 90%" not in report_a
    assert "Nearly 59%" not in report_a

    # Scenario B: Known severe FNs
    train_b = _make_dummy_train_df(n_rows=50)
    train_b["Will_Buy_EV"] = "No"
    train_b.loc[:9, "Will_Buy_EV"] = "Yes"
    lgb_b = _make_dummy_oof_frame(n_rows=50, base_prob=0.8)
    cb_b = _make_dummy_oof_frame(n_rows=50, base_prob=0.8)
    lgb_b.loc[lgb_b["row_index"] < 10, "probability"] = 0.02
    cb_b.loc[cb_b["row_index"] < 10, "probability"] = 0.02
    out_b = tmp_path / "scenario_b"

    res_b = run_severe_fn_audit(
        train_df=train_b,
        lgb_oof_df=lgb_b,
        cb_oof_df=cb_b,
        output_dir=out_b,
        source_provenance={"test": "b"},
    )
    report_b = (out_b / "report.md").read_text()
    assert f"{res_b.severe_fn_count:,} rows" in report_b
    assert f"{res_b.severe_fn_rate:.4%}" in report_b


def test_artifact_determinism_and_hash_freshness(tmp_path: Path) -> None:
    n_rows = 150
    train_df = _make_dummy_train_df(n_rows=n_rows)
    lgb_df = _make_dummy_oof_frame(n_rows=n_rows, base_prob=0.3)
    cb_df = _make_dummy_oof_frame(n_rows=n_rows, base_prob=0.3)

    out_dir_1 = tmp_path / "audit_1"
    out_dir_2 = tmp_path / "audit_2"

    _ = run_severe_fn_audit(
        train_df=train_df,
        lgb_oof_df=lgb_df,
        cb_oof_df=cb_df,
        output_dir=out_dir_1,
        source_provenance={
            "train_file_sha256": "dummy_train_sha",
            "combined_train_test_data_sha256": "dummy_combined_sha",
            "lightgbm_oof_sha256": "dummy_lgb_sha",
            "catboost_oof_sha256": "dummy_cb_sha",
        },
    )

    _ = run_severe_fn_audit(
        train_df=train_df,
        lgb_oof_df=lgb_df,
        cb_oof_df=cb_df,
        output_dir=out_dir_2,
        source_provenance={
            "train_file_sha256": "dummy_train_sha",
            "combined_train_test_data_sha256": "dummy_combined_sha",
            "lightgbm_oof_sha256": "dummy_lgb_sha",
            "catboost_oof_sha256": "dummy_cb_sha",
        },
    )

    expected_files = [
        "manifest.json",
        "severe_false_negatives.csv",
        "top_loss_examples.csv",
        "coarse_cell_stats.csv",
        "frozen_slices.json",
        "report.md",
    ]
    for fname in expected_files:
        assert (out_dir_1 / fname).is_file()
        assert (out_dir_2 / fname).is_file()

        sha1 = compute_file_sha256(out_dir_1 / fname)
        sha2 = compute_file_sha256(out_dir_2 / fname)
        assert sha1 == sha2, f"Determinism mismatch on {fname}"

    with open(out_dir_1 / "manifest.json") as f:
        manifest = json.load(f)
    for fname, recorded_sha in manifest["artifact_checksums"].items():
        actual_sha = compute_file_sha256(out_dir_1 / fname)
        assert recorded_sha == actual_sha

    # Verify frozen_slices.json predicates can reconstruct exact counts
    with open(out_dir_1 / "frozen_slices.json") as f:
        frozen_data = json.load(f)
    for _s_name, s_info in frozen_data["predeclared_slices"].items():
        reconstructed = reconstruct_mask_from_predicate(train_df, s_info["predicate"])
        assert int(reconstructed.sum()) == s_info["n"]


def test_runner_smoke_execution(tmp_path: Path) -> None:
    import subprocess
    import sys

    out_dir = tmp_path / "smoke_audit"
    cmd = [
        sys.executable,
        "scripts/run_severe_fn_audit.py",
        "--smoke",
        "--output-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "AUDIT RESULTS SUMMARY" in proc.stdout
    assert "Canonical Baseline (LightGBM):" in proc.stdout
    assert "Exploratory 30/70 Sensitivity:" in proc.stdout
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "severe_false_negatives.csv").is_file()
    assert (out_dir / "top_loss_examples.csv").is_file()
    assert (out_dir / "coarse_cell_stats.csv").is_file()
    assert (out_dir / "frozen_slices.json").is_file()
    assert (out_dir / "report.md").is_file()


def test_reconstruct_mask_from_predicate_eq_and_ne() -> None:
    df = pd.DataFrame(
        {
            "City_Type": ["Rural", "Urban", "Suburban"],
            "Charging_Stations": [0, 2, 5],
        },
        index=[10, 20, 30],
    )

    # String eq
    pred_eq_str = {"op": "eq", "column": "City_Type", "value": "Rural"}
    mask = reconstruct_mask_from_predicate(df, pred_eq_str)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert mask.index.tolist() == [10, 20, 30]
    assert mask.tolist() == [True, False, False]

    # Numeric eq (using '==' as operator alias)
    pred_eq_num = {"op": "==", "column": "Charging_Stations", "value": 2}
    mask_num = reconstruct_mask_from_predicate(df, pred_eq_num)
    assert mask_num.tolist() == [False, True, False]

    # ne
    pred_ne = {"op": "ne", "column": "City_Type", "value": "Rural"}
    mask_ne = reconstruct_mask_from_predicate(df, pred_ne)
    assert mask_ne.tolist() == [False, True, True]


def test_reconstruct_mask_from_predicate_between_inclusive() -> None:
    df = pd.DataFrame(
        {
            "Age": [30, 34, 40, 43, 50],
            "Income": [70000.0, 78915.0, 85000.0, 91619.0, 100000.0],
        },
        index=[101, 102, 103, 104, 105],
    )

    # Inclusive between using 'bounds' list: [34, 43]
    pred_bounds = {"op": "between", "column": "Age", "bounds": [34, 43]}
    mask = reconstruct_mask_from_predicate(df, pred_bounds)
    assert mask.index.tolist() == [101, 102, 103, 104, 105]
    # 34 and 43 are inclusive endpoints
    assert mask.tolist() == [False, True, True, True, False]

    # Inclusive between using 'lower' and 'upper'
    pred_low_up = {
        "op": "between",
        "column": "Income",
        "lower": 78915.0,
        "upper": 91619.0,
    }
    mask_inc = reconstruct_mask_from_predicate(df, pred_low_up)
    assert mask_inc.tolist() == [False, True, True, True, False]


def test_reconstruct_mask_from_predicate_in_membership() -> None:
    df = pd.DataFrame(
        {"Current_Car_Type": ["Sedan", "SUV", "Truck", "Hatchback"]},
        index=[1, 2, 3, 4],
    )

    pred_in = {"op": "in", "column": "Current_Car_Type", "values": ["SUV", "Truck"]}
    mask = reconstruct_mask_from_predicate(df, pred_in)
    assert mask.tolist() == [False, True, True, False]

    # Alias 'isin'
    pred_isin = {"op": "isin", "column": "Current_Car_Type", "values": ["Sedan"]}
    mask_isin = reconstruct_mask_from_predicate(df, pred_isin)
    assert mask_isin.tolist() == [True, False, False, False]


def test_reconstruct_mask_from_predicate_comparisons() -> None:
    df = pd.DataFrame({"Score": [10, 20, 30, 40]})

    gt_pred = {"op": "gt", "column": "Score", "value": 20}
    assert reconstruct_mask_from_predicate(df, gt_pred).tolist() == [False, False, True, True]

    gte_pred = {"op": "gte", "column": "Score", "value": 20}
    assert reconstruct_mask_from_predicate(df, gte_pred).tolist() == [False, True, True, True]

    lt_pred = {"op": "lt", "column": "Score", "value": 30}
    assert reconstruct_mask_from_predicate(df, lt_pred).tolist() == [True, True, False, False]

    lte_pred = {"op": "lte", "column": "Score", "value": 30}
    assert reconstruct_mask_from_predicate(df, lte_pred).tolist() == [True, True, True, False]


def test_reconstruct_mask_from_predicate_recursive_all_and_any() -> None:
    df = pd.DataFrame(
        {
            "Age": [25, 35, 45, 55],
            "City_Type": ["Rural", "Rural", "Urban", "Urban"],
            "Home_Charging": ["No", "Yes", "Yes", "No"],
        }
    )

    # Conjunction: Age between [30, 50] AND City_Type == Rural
    pred_all = {
        "op": "all",
        "predicates": [
            {"op": "between", "column": "Age", "bounds": [30, 50]},
            {"op": "eq", "column": "City_Type", "value": "Rural"},
        ],
    }
    mask_all = reconstruct_mask_from_predicate(df, pred_all)
    assert mask_all.tolist() == [False, True, False, False]

    # Disjunction: City_Type == Urban OR Home_Charging == Yes
    pred_any = {
        "op": "any",
        "predicates": [
            {"op": "eq", "column": "City_Type", "value": "Urban"},
            {"op": "eq", "column": "Home_Charging", "value": "Yes"},
        ],
    }
    mask_any = reconstruct_mask_from_predicate(df, pred_any)
    assert mask_any.tolist() == [False, True, True, True]

    # Nested logical: (Age < 30) OR (City_Type == Rural AND Home_Charging == Yes)
    pred_nested = {
        "op": "any",
        "predicates": [
            {"op": "lt", "column": "Age", "value": 30},
            {
                "op": "all",
                "predicates": [
                    {"op": "eq", "column": "City_Type", "value": "Rural"},
                    {"op": "eq", "column": "Home_Charging", "value": "Yes"},
                ],
            },
        ],
    }
    mask_nested = reconstruct_mask_from_predicate(df, pred_nested)
    assert mask_nested.tolist() == [True, True, False, False]

    # Top-level shorthand {"all": [...]}
    pred_shorthand = {
        "all": [
            {"op": "eq", "column": "City_Type", "value": "Rural"},
            {"op": "eq", "column": "Home_Charging", "value": "No"},
        ]
    }
    assert reconstruct_mask_from_predicate(df, pred_shorthand).tolist() == [
        True,
        False,
        False,
        False,
    ]

    # Empty all -> all True; empty any -> all False
    assert reconstruct_mask_from_predicate(df, {"op": "all", "predicates": []}).all()
    assert not reconstruct_mask_from_predicate(df, {"op": "any", "predicates": []}).any()


def test_reconstruct_mask_from_predicate_universal_and_empty() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
    assert reconstruct_mask_from_predicate(df, {}).all()
    assert reconstruct_mask_from_predicate(df, {"op": "always_true"}).all()
    assert not reconstruct_mask_from_predicate(df, {"op": "always_false"}).any()
    assert reconstruct_mask_from_predicate(df, {}).index.tolist() == [10, 20, 30]


def test_reconstruct_mask_from_predicate_validation_errors() -> None:
    df = pd.DataFrame({"Age": [20, 30], "City": ["Urban", "Rural"]})

    # Non-DataFrame input
    with pytest.raises(TypeError, match="Expected df to be pd.DataFrame"):
        reconstruct_mask_from_predicate([1, 2, 3], {"op": "always_true"})  # type: ignore[arg-type]

    # Non-dict predicate
    with pytest.raises(TypeError, match="Expected predicate to be dict"):
        reconstruct_mask_from_predicate(df, "not_a_dict")  # type: ignore[arg-type]

    # Missing operator
    with pytest.raises(ValueError, match="Predicate must contain an 'op' or 'operator' field"):
        reconstruct_mask_from_predicate(df, {"column": "Age", "value": 20})

    # Unsupported operator
    with pytest.raises(ValueError, match="Unsupported predicate operator"):
        reconstruct_mask_from_predicate(df, {"op": "regex_match", "column": "City", "value": ".*"})

    # Missing column in atomic predicate
    with pytest.raises(ValueError, match="requires 'column' field"):
        reconstruct_mask_from_predicate(df, {"op": "eq", "value": 20})

    # Column not in DataFrame
    with pytest.raises(ValueError, match="not found in DataFrame"):
        reconstruct_mask_from_predicate(df, {"op": "eq", "column": "NonExistent", "value": 20})

    # Missing value for equality
    with pytest.raises(ValueError, match="Equality predicate requires 'value' field"):
        reconstruct_mask_from_predicate(df, {"op": "eq", "column": "Age"})

    # Malformed between: lower > upper
    with pytest.raises(ValueError, match="Malformed bounds in 'between' predicate"):
        reconstruct_mask_from_predicate(
            df, {"op": "between", "column": "Age", "bounds": [50, 30]}
        )

    # Malformed between: missing bounds
    with pytest.raises(ValueError, match="Malformed 'between' predicate"):
        reconstruct_mask_from_predicate(df, {"op": "between", "column": "Age"})

    # Malformed between: invalid bounds length
    with pytest.raises(ValueError, match="expected 2-element list/tuple"):
        reconstruct_mask_from_predicate(
            df, {"op": "between", "column": "Age", "bounds": [10, 20, 30]}
        )

    # Malformed in: non-iterable / string
    with pytest.raises(ValueError, match="must be an iterable"):
        reconstruct_mask_from_predicate(df, {"op": "in", "column": "City", "values": "Urban"})

    # Malformed all/any: missing predicates
    with pytest.raises(ValueError, match="requires 'predicates' list"):
        reconstruct_mask_from_predicate(df, {"op": "all"})

    # Malformed all/any: predicates not a list
    with pytest.raises(TypeError, match="'predicates' must be a list"):
        reconstruct_mask_from_predicate(df, {"op": "all", "predicates": "not_a_list"})
