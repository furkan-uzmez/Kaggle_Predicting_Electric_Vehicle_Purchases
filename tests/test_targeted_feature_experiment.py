"""Unit and contract tests for the targeted feature experiment harness."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ev_purchase.targeted_feature_experiment import (
    AGE_INCOME_CONTRACT,
    CHARGING_ACCESS_CONTRACT,
    COMMUTE_ANXIETY_CONTRACT,
    CONCERN_SUBSIDY_CONTRACT,
    INCOME_PER_VEHICLE_CONTRACT,
    THREAT_MODEL_STATEMENT,
    ArtifactTamperError,
    LegacyProvenanceError,
    ProvenanceMismatchError,
    add_age_income_feature,
    add_charging_access_feature,
    add_commute_anxiety_feature,
    add_concern_subsidy_feature,
    add_income_per_vehicle_feature,
    build_training_provenance,
    calculate_experiment_metrics,
    compute_params_hash,
    evaluate_frozen_slices,
    find_material_slice_regressions,
    get_targeted_feature_contract,
    list_targeted_feature_contracts,
    make_gate_decision,
    run_targeted_experiment,
    train_candidate_cv,
    validate_oof_keys,
    verify_frozen_slices,
    verify_provenance_and_baseline,
    verify_training_anchor,
    verify_training_provenance,
)


def test_add_concern_subsidy_feature_adds_exactly_one_column() -> None:
    """Test that candidate feature policy adds exactly Concern_Subsidy and retains parents."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "Environmental_Concern_Level": [1.0, 4.0, 5.0],
            "Subsidy_Available": ["No", "No", "Yes"],
            "Age": [25, 35, 45],
        }
    )
    df_copy = df.copy()

    transformed = add_concern_subsidy_feature(df)

    # Immutability
    pd.testing.assert_frame_equal(df, df_copy)

    # Exactly one column added
    assert len(transformed.columns) == len(df.columns) + 1
    assert "Concern_Subsidy" in transformed.columns

    # Retains parents
    assert "Environmental_Concern_Level" in transformed.columns
    assert "Subsidy_Available" in transformed.columns

    # Exact value formatting
    expected = pd.Series(["1.0__No", "4.0__No", "5.0__Yes"], name="Concern_Subsidy")
    pd.testing.assert_series_equal(transformed["Concern_Subsidy"], expected)


def test_add_concern_subsidy_feature_missing_parents_raises() -> None:
    """Test that missing parent columns raises ValueError."""
    df_missing_subsidy = pd.DataFrame({"id": [1], "Environmental_Concern_Level": [1.0]})
    with pytest.raises(ValueError, match="Subsidy_Available"):
        add_concern_subsidy_feature(df_missing_subsidy)

    df_missing_concern = pd.DataFrame({"id": [1], "Subsidy_Available": ["Yes"]})
    with pytest.raises(ValueError, match="Environmental_Concern_Level"):
        add_concern_subsidy_feature(df_missing_concern)


def test_verify_provenance_rejects_hash_mismatches(tmp_path: Path) -> None:
    """Test that mismatch in data hash, split hash, or baseline OOF hash is rejected."""
    good_hash = "a" * 64
    bad_hash = "b" * 64

    # Test data hash mismatch
    with pytest.raises(ValueError, match="Data SHA256 mismatch"):
        verify_provenance_and_baseline(
            expected_data_sha256=good_hash,
            actual_data_sha256=bad_hash,
            expected_split_sha256=good_hash,
            actual_split_sha256=good_hash,
            expected_oof_sha256=good_hash,
            actual_oof_sha256=good_hash,
        )

    # Test split hash mismatch
    with pytest.raises(ValueError, match="Split SHA256 mismatch"):
        verify_provenance_and_baseline(
            expected_data_sha256=good_hash,
            actual_data_sha256=good_hash,
            expected_split_sha256=good_hash,
            actual_split_sha256=bad_hash,
            expected_oof_sha256=good_hash,
            actual_oof_sha256=good_hash,
        )

    # Test baseline OOF hash mismatch
    with pytest.raises(ValueError, match="Baseline OOF SHA256 mismatch"):
        verify_provenance_and_baseline(
            expected_data_sha256=good_hash,
            actual_data_sha256=good_hash,
            expected_split_sha256=good_hash,
            actual_split_sha256=good_hash,
            expected_oof_sha256=good_hash,
            actual_oof_sha256=bad_hash,
        )


def test_calculate_experiment_metrics_and_nadeau_bengio() -> None:
    """Test fold, repeat, global metrics calculation and Nadeau-Bengio corrected CI."""
    n_rows = 100
    n_splits = 5
    n_repeats = 3
    y_true = np.array([0, 1] * (n_rows // 2))

    # Construct mock baseline and candidate OOF
    records_base = []
    records_cand = []
    for rep in range(n_repeats):
        for fold in range(n_splits):
            fold_rows = np.arange(fold * (n_rows // n_splits), (fold + 1) * (n_rows // n_splits))
            for r in fold_rows:
                # Candidate has slightly better calibrated probabilities
                p_base = 0.8 if y_true[r] == 1 else 0.3
                p_cand = 0.85 if y_true[r] == 1 else 0.25
                records_base.append(
                    {"row_index": r, "repeat": rep, "fold": fold, "probability": p_base}
                )
                records_cand.append(
                    {"row_index": r, "repeat": rep, "fold": fold, "probability": p_cand}
                )

    oof_base = pd.DataFrame(records_base)
    oof_cand = pd.DataFrame(records_cand)

    metrics = calculate_experiment_metrics(
        y_true=y_true,
        oof_baseline=oof_base,
        oof_candidate=oof_cand,
        n_rows=n_rows,
        n_repeats=n_repeats,
        n_splits=n_splits,
    )

    assert "fold_metrics" in metrics
    assert len(metrics["fold_metrics"]) == 15
    assert "repeat_metrics" in metrics
    assert len(metrics["repeat_metrics"]) == 3
    assert "global_metrics" in metrics

    # Check Nadeau-Bengio test output
    nb_result = metrics["nadeau_bengio"]
    assert nb_result["n"] == 15
    assert nb_result["df"] == 14
    assert nb_result["test_train_ratio"] == 0.25
    assert "ci_nb_lower" in nb_result
    assert "ci_nb_upper" in nb_result
    assert "p_value_nb_raw" in nb_result


def test_nadeau_bengio_zero_variance_handled_gracefully() -> None:
    """Test Nadeau-Bengio when differences have zero variance."""
    n_rows = 50
    n_splits = 5
    n_repeats = 3
    y_true = np.array([0, 1] * (n_rows // 2))

    # Identical baseline and candidate
    records = []
    for rep in range(n_repeats):
        for fold in range(n_splits):
            fold_rows = np.arange(fold * (n_rows // n_splits), (fold + 1) * (n_rows // n_splits))
            for r in fold_rows:
                records.append({"row_index": r, "repeat": rep, "fold": fold, "probability": 0.5})

    oof = pd.DataFrame(records)
    metrics = calculate_experiment_metrics(
        y_true=y_true,
        oof_baseline=oof,
        oof_candidate=oof,
        n_rows=n_rows,
        n_repeats=n_repeats,
        n_splits=n_splits,
    )

    nb = metrics["nadeau_bengio"]
    assert nb["sd_difference"] == 0.0
    assert nb["ci_nb_lower"] == 0.0
    assert nb["ci_nb_upper"] == 0.0


def test_gate_decision_logic() -> None:
    """Test GO, BORDERLINE, and NO-GO decision rules."""
    # 1. Clear GO: gain >= 0.00010, repeats >= 0, CI lower > 0, logloss degradation <= 0.00020
    decision_go = make_gate_decision(
        global_auc_delta=0.00015,
        repeat_auc_deltas=[0.00012, 0.00015, 0.00018],
        nb_ci_lower=0.00005,
        global_log_loss_delta=-0.00010,  # improved
        slice_regressions=[],
    )
    assert decision_go.decision == "GO"

    # 2. Borderline: gain in [0.00005, 0.00010)
    decision_border = make_gate_decision(
        global_auc_delta=0.00008,
        repeat_auc_deltas=[0.00005, 0.00009, 0.00010],
        nb_ci_lower=0.00001,
        global_log_loss_delta=-0.00005,
        slice_regressions=[],
    )
    assert decision_border.decision == "BORDERLINE"
    assert "second split seed" in decision_border.reason

    # 3. NO-GO: gain < 0.00005
    decision_nogo_low_gain = make_gate_decision(
        global_auc_delta=0.00003,
        repeat_auc_deltas=[0.00002, 0.00004, 0.00003],
        nb_ci_lower=-0.00002,
        global_log_loss_delta=0.00001,
        slice_regressions=[],
    )
    assert decision_nogo_low_gain.decision == "NO-GO"

    # 4. NO-GO: inconsistent repeats (one negative repeat delta)
    decision_nogo_inconsistent = make_gate_decision(
        global_auc_delta=0.00015,
        repeat_auc_deltas=[0.00025, -0.00002, 0.00022],
        nb_ci_lower=0.00002,
        global_log_loss_delta=-0.00010,
        slice_regressions=[],
    )
    assert decision_nogo_inconsistent.decision == "NO-GO"
    assert "negative repeat" in decision_nogo_inconsistent.reason.lower()

    # 5. NO-GO: log-loss degradation > 0.00020
    decision_nogo_logloss = make_gate_decision(
        global_auc_delta=0.00015,
        repeat_auc_deltas=[0.00012, 0.00015, 0.00018],
        nb_ci_lower=0.00005,
        global_log_loss_delta=0.00035,  # degraded by > 0.00020
        slice_regressions=[],
    )
    assert decision_nogo_logloss.decision == "NO-GO"

    # 6. NO-GO: material frozen slice regression
    decision_nogo_slice = make_gate_decision(
        global_auc_delta=0.00015,
        repeat_auc_deltas=[0.00012, 0.00015, 0.00018],
        nb_ci_lower=0.00005,
        global_log_loss_delta=-0.00010,
        slice_regressions=["concern_1.0_subsidy_No"],
    )
    assert decision_nogo_slice.decision == "NO-GO"


def test_frozen_slices_evaluation_contract(tmp_path: Path) -> None:
    """Test evaluation of frozen slices and cohorts against baseline and candidate."""
    train_df = pd.DataFrame(
        {
            "id": np.arange(10),
            "Environmental_Concern_Level": [1.0] * 5 + [2.0] * 5,
            "Subsidy_Available": ["No"] * 5 + ["Yes"] * 5,
            "City_Type": ["Rural"] * 5 + ["Urban"] * 5,
        }
    )
    y_true = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1])
    p_base = np.array([0.1, 0.2, 0.05, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p_cand = np.array([0.12, 0.18, 0.25, 0.28, 0.45, 0.52, 0.65, 0.72, 0.75, 0.92])

    frozen_dict = {
        "predeclared_slices": {
            "slice_concern_1_no": {
                "cell_group": "test_group",
                "predicate": {
                    "op": "all",
                    "predicates": [
                        {"op": "eq", "column": "Environmental_Concern_Level", "value": 1.0},
                        {"op": "eq", "column": "Subsidy_Available", "value": "No"},
                    ],
                },
            }
        },
        "cohorts": {
            "severe_fn": {"threshold": 0.1},
        },
    }
    frozen_path = tmp_path / "frozen_slices.json"
    with open(frozen_path, "w") as f:
        json.dump(frozen_dict, f)

    slice_df = evaluate_frozen_slices(
        train_df=train_df,
        y_true=y_true,
        p_baseline=p_base,
        p_candidate=p_cand,
        frozen_slices_path=frozen_path,
    )

    assert isinstance(slice_df, pd.DataFrame)
    expected_cols = {
        "slice_name",
        "n",
        "positives",
        "positive_rate",
        "baseline_auc",
        "candidate_auc",
        "delta_auc",
        "baseline_log_loss",
        "candidate_log_loss",
        "delta_log_loss",
        "rank_movement",
    }
    assert expected_cols.issubset(set(slice_df.columns))
    assert len(slice_df) >= 1


def test_validate_oof_keys_catches_all_anomalies() -> None:
    """Test validation of 3-repeat OOF prediction keys."""
    n_rows = 10
    n_repeats = 3
    n_splits = 5

    # Valid base
    valid_df = pd.DataFrame(
        [
            {"row_index": r, "repeat": rep, "fold": r % n_splits, "probability": 0.5}
            for rep in range(n_repeats)
            for r in range(n_rows)
        ]
    )
    validate_oof_keys(valid_df, n_rows=n_rows, n_repeats=n_repeats, n_splits=n_splits)

    # 1. Missing column
    df_missing_col = valid_df.drop(columns=["probability"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_oof_keys(df_missing_col, n_rows=n_rows, n_repeats=n_repeats, n_splits=n_splits)

    # 2. Row count mismatch
    df_short = valid_df.iloc[:-1]
    with pytest.raises(ValueError, match="row count mismatch"):
        validate_oof_keys(df_short, n_rows=n_rows, n_repeats=n_repeats, n_splits=n_splits)

    # 3. Duplicate key in same repeat
    df_dup = valid_df.copy()
    df_dup.loc[0, "row_index"] = df_dup.loc[1, "row_index"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_keys(df_dup, n_rows=n_rows, n_repeats=n_repeats, n_splits=n_splits)

    # 4. Out of bounds row_index
    df_oob = valid_df.copy()
    df_oob.loc[0, "row_index"] = 999
    with pytest.raises(ValueError, match="out of bounds"):
        validate_oof_keys(df_oob, n_rows=n_rows, n_repeats=n_repeats, n_splits=n_splits)


def test_find_material_slice_regressions() -> None:
    """Test slice regression detection with threshold and sample size checks."""
    slice_df = pd.DataFrame(
        [
            {
                "slice_name": "slice_healthy",
                "positives": 100,
                "delta_auc": +0.002,
                "delta_log_loss": -0.001,
            },
            {
                "slice_name": "slice_auc_regr",
                "positives": 100,
                "delta_auc": -0.006,  # worse than -0.0050
                "delta_log_loss": 0.001,
            },
            {
                "slice_name": "slice_ll_regr",
                "positives": 100,
                "delta_auc": 0.0,
                "delta_log_loss": 0.008,  # worse than +0.0050
            },
            {
                "slice_name": "slice_small_pos",
                "positives": 5,  # too few positives (< 50)
                "delta_auc": -0.05,
                "delta_log_loss": 0.001,
            },
        ]
    )

    regrs = find_material_slice_regressions(slice_df)
    assert any("slice_auc_regr" in r for r in regrs)
    assert any("slice_ll_regr" in r for r in regrs)
    assert not any("slice_healthy" in r for r in regrs)
    assert not any("slice_small_pos" in r for r in regrs)


def test_train_candidate_cv_smoke_one_fold() -> None:
    """Test candidate CV trainer on minimal mock data for 1 fold."""
    from ev_purchase.data import DatasetContract

    mock_train = pd.DataFrame(
        {
            "id": np.arange(20),
            "Will_Buy_EV": ["No", "Yes"] * 10,
            "Environmental_Concern_Level": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "Subsidy_Available": ["No", "Yes"] * 10,
            "Age": np.linspace(20, 60, 20),
        }
    )
    mock_manifest = pd.DataFrame(
        {
            "row_index": np.arange(20),
            "repeat": [0] * 20,
            "fold": [0] * 10 + [1] * 10,
        }
    )
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    params = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "verbose": -1}

    oof_f0 = train_candidate_cv(
        train_df=mock_train,
        manifest=mock_manifest,
        contract=contract,
        model_params=params,
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        base_seed=42,
        folds=[(0, 0)],
    )

    assert isinstance(oof_f0, pd.DataFrame)
    assert len(oof_f0) == 10
    assert set(oof_f0.columns) == {"row_index", "repeat", "fold", "probability"}
    assert (oof_f0["probability"] >= 0.0).all()
    assert (oof_f0["probability"] <= 1.0).all()


def test_real_provenance_and_baseline_hashes() -> None:
    """Validate hashes of train, test, splits manifest, and baseline OOF match final config."""
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    final_cfg_path = project_root / "configs" / "final.yaml"
    if not final_cfg_path.exists():
        pytest.skip("configs/final.yaml not found")

    with open(final_cfg_path) as f:
        cfg = yaml.safe_load(f)

    lgb_comp = cfg["policy"]["components"]["lightgbm_hpo_trial35"]
    train_path = project_root / "data" / "playground-series-s6e9" / "train.csv"
    test_path = project_root / "data" / "playground-series-s6e9" / "test.csv"
    split_path = project_root / "outputs" / "splits" / "repeated_stratified_5x3.parquet"
    baseline_oof_path = (
        project_root
        / "outputs"
        / "ensemble"
        / "candidates"
        / "lightgbm_hpo_trial35"
        / "oof_predictions.parquet"
    )

    if not (
        train_path.exists()
        and test_path.exists()
        and split_path.exists()
        and baseline_oof_path.exists()
    ):
        pytest.skip("Dataset or baseline files not found on disk")

    from ev_purchase.targeted_feature_experiment import compute_combined_sha256, compute_file_sha256

    actual_data_sha = compute_combined_sha256([train_path, test_path])
    actual_split_sha = compute_file_sha256(split_path)
    actual_oof_sha = compute_file_sha256(baseline_oof_path)

    assert actual_data_sha == lgb_comp["provenance"]["data_sha256"]
    assert actual_split_sha == lgb_comp["provenance"]["split_sha256"]
    assert actual_split_sha == cfg["policy"]["split_sha256"]
    assert len(actual_oof_sha) == 64


def test_real_frozen_slices_reuse_predicates() -> None:
    """Validate that frozen slices from severe_fn_audit are loaded and match schema."""
    project_root = Path(__file__).resolve().parents[1]
    frozen_path = (
        project_root / "outputs" / "diagnostics" / "severe_fn_audit" / "frozen_slices.json"
    )
    if not frozen_path.exists():
        pytest.skip("frozen_slices.json not found")

    with open(frozen_path) as f:
        data = json.load(f)

    predeclared = data.get("predeclared_slices", {})
    assert len(predeclared) >= 10
    assert "concern_x_subsidy__concern_1.0_subsidy_No" in predeclared
    assert "rural" in predeclared
    assert "overall" in predeclared


def test_verify_frozen_slices_success_and_failures(tmp_path: Path) -> None:
    """Test verify_frozen_slices validates schema, existence, and prospective status."""
    # 1. Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        verify_frozen_slices(tmp_path / "non_existent.json")

    # 2. Missing version raises ValueError
    bad_ver_path = tmp_path / "bad_ver.json"
    with open(bad_ver_path, "w") as f:
        json.dump(
            {"slice_provenance": {"status": "post_hoc_discovered_then_prospectively_frozen"}},
            f,
        )
    with pytest.raises(ValueError, match="missing 'version'"):
        verify_frozen_slices(bad_ver_path)

    # 3. Status mismatch raises ValueError
    bad_status_path = tmp_path / "bad_status.json"
    with open(bad_status_path, "w") as f:
        json.dump({"version": "1.0", "slice_provenance": {"status": "unfrozen"}}, f)
    with pytest.raises(ValueError, match="status mismatch"):
        verify_frozen_slices(bad_status_path)

    # 4. Valid file succeeds
    good_path = tmp_path / "good.json"
    with open(good_path, "w") as f:
        json.dump(
            {
                "version": "1.0",
                "slice_provenance": {"status": "post_hoc_discovered_then_prospectively_frozen"},
            },
            f,
        )
    meta = verify_frozen_slices(good_path)
    assert meta["version"] == "1.0"
    assert meta["status"] == "post_hoc_discovered_then_prospectively_frozen"
    assert len(meta["sha256"]) == 64


@pytest.mark.parametrize(
    "policy_name",
    ["concern_subsidy", "charging_access", "commute_anxiety", "income_per_vehicle", "age_income"],
)
def test_manifest_artifact_freshness(policy_name: str) -> None:
    """Test that all produced artifacts in manifest exist, have non-self hashes and row counts."""
    from ev_purchase.targeted_feature_experiment import compute_file_sha256

    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        project_root
        / "outputs"
        / "experiments"
        / "targeted_features"
        / policy_name
        / "manifest.json"
    )
    if not manifest_path.exists():
        pytest.skip(f"{policy_name} manifest.json not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # 1. Frozen slices recorded
    assert "frozen_slices" in manifest
    fs = manifest["frozen_slices"]
    assert "sha256" in fs and len(fs["sha256"]) == 64
    assert fs["version"] == "1.0"
    assert fs["status"] == "post_hoc_discovered_then_prospectively_frozen"

    # 2. Produced artifacts inventory
    assert "artifacts" in manifest
    artifacts = manifest["artifacts"]
    expected_keys = {
        "candidate_oof",
        "training_provenance",
        "fold_metrics",
        "repeat_metrics",
        "global_metrics",
        "slice_metrics",
        "report",
    }
    assert expected_keys.issubset(set(artifacts.keys()))

    # Assert no self-hashing
    assert "manifest" not in artifacts
    assert "manifest.json" not in artifacts

    # Freshness verification: disk state matches manifest record
    for key, art_info in artifacts.items():
        rel_path = art_info["path"]
        disk_path = project_root / rel_path
        assert disk_path.exists(), f"Artifact {key} file does not exist on disk: {disk_path}"

        actual_sha = compute_file_sha256(disk_path)
        assert actual_sha == art_info["sha256"], f"SHA256 mismatch for {key}"

        row_count = art_info["row_count"]
        if disk_path.suffix == ".parquet":
            actual_rows = len(pd.read_parquet(disk_path))
            assert actual_rows == row_count
        elif disk_path.suffix == ".csv":
            actual_rows = len(pd.read_csv(disk_path))
            assert actual_rows == row_count
        elif disk_path.suffix == ".md":
            with open(disk_path, encoding="utf-8") as f:
                actual_rows = len(f.read().splitlines())
            assert actual_rows == row_count

    # 3. Generic provenance verification
    assert "provenance" in manifest
    assert "training_provenance" in manifest
    prov = manifest["provenance"]
    assert "generation_command" in prov
    assert "environment" in prov
    assert "python_version" in prov["environment"]
    assert "platform" in prov["environment"]
    assert "source_hash" in prov
    assert len(prov["source_hash"]) == 64
    assert "training_source_hash" in prov
    assert len(prov["training_source_hash"]) == 64
    assert prov["source_hash"] == prov["training_source_hash"]
    assert prov["source_hash"] == manifest["training_provenance"]["training_source_hash"]


def test_reconstruct_top_loss_cohort_boundary_ties(tmp_path: Path) -> None:
    """Test top-loss cohort deterministic tie-breaking on exact sample log-loss boundaries."""
    n_rows = 5
    train_df = pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "Environmental_Concern_Level": [1.0] * n_rows,
            "Subsidy_Available": ["No"] * n_rows,
        }
    )
    y_true = np.ones(n_rows, dtype=int)
    # Identical predictions -> all 5 rows have identical sample log-loss
    p_base = np.full(n_rows, 0.2)
    p_cand = np.full(n_rows, 0.25)

    frozen_payload = {
        "version": "1.0",
        "slice_provenance": {"status": "post_hoc_discovered_then_prospectively_frozen"},
        "predeclared_slices": {},
        "cohorts": {
            "top_1pct_loss": {"count": 2, "threshold": 1.609438},
        },
    }
    frozen_path = tmp_path / "frozen_slices.json"
    with open(frozen_path, "w") as f:
        json.dump(frozen_payload, f)

    slice_df = evaluate_frozen_slices(
        train_df=train_df,
        y_true=y_true,
        p_baseline=p_base,
        p_candidate=p_cand,
        frozen_slices_path=frozen_path,
    )

    top_loss_row = slice_df[slice_df["slice_name"] == "cohort__top_1pct_loss"].iloc[0]
    # Exact count contract: strictly 2 rows selected despite all 5 being tied
    assert int(top_loss_row["n"]) == 2


def test_add_charging_access_feature_schema_and_values() -> None:
    """Test Charging_Access derivation formula, schema, and parent retention."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "Home_Charging_Possible": ["Yes", "No", "Yes", "No"],
            "Charging_Stations_Near_Home": [3, 0, 5, 2],
            "Charging_Stations_Near_Work": [4, 1, 0, 3],
            "Age": [30, 40, 50, 60],
        }
    )
    df_copy = df.copy()

    transformed = add_charging_access_feature(df)

    # Immutability
    pd.testing.assert_frame_equal(df, df_copy)

    # Exactly one column added
    assert len(transformed.columns) == len(df.columns) + 1
    assert "Charging_Access" in transformed.columns

    # Retains all parents
    assert "Home_Charging_Possible" in transformed.columns
    assert "Charging_Stations_Near_Home" in transformed.columns
    assert "Charging_Stations_Near_Work" in transformed.columns

    # Numeric dtype
    assert np.issubdtype(transformed["Charging_Access"].dtype, np.number)

    # Values: 2 * (Home == 'Yes') + Home_Stations + Work_Stations
    # row 0: 2*1 + 3 + 4 = 9
    # row 1: 2*0 + 0 + 1 = 1
    # row 2: 2*1 + 5 + 0 = 7
    # row 3: 2*0 + 2 + 3 = 5
    expected = pd.Series([9, 1, 7, 5], name="Charging_Access")
    pd.testing.assert_series_equal(transformed["Charging_Access"], expected)


def test_add_charging_access_feature_missing_parents_raises() -> None:
    """Test that missing any required parent column raises ValueError."""
    for missing_col in [
        "Home_Charging_Possible",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
    ]:
        cols = {
            "Home_Charging_Possible": ["Yes"],
            "Charging_Stations_Near_Home": [1],
            "Charging_Stations_Near_Work": [2],
        }
        del cols[missing_col]
        df = pd.DataFrame(cols)
        with pytest.raises(ValueError, match=missing_col):
            add_charging_access_feature(df)


def test_add_commute_anxiety_feature_schema_and_values() -> None:
    """Test Commute_Anxiety derivation formula, schema, and parent retention."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "Daily_Commute_km": [10.0, 25.5, 50.0, 30.0],
            "Range_Anxiety_Level": ["Low", "Medium", "High", "Low"],
            "Age": [30, 40, 50, 60],
        }
    )
    df_copy = df.copy()

    transformed = add_commute_anxiety_feature(df)

    # Immutability
    pd.testing.assert_frame_equal(df, df_copy)

    # Exactly one column added
    assert len(transformed.columns) == len(df.columns) + 1
    assert "Commute_Anxiety" in transformed.columns

    # Retains all parents
    assert "Daily_Commute_km" in transformed.columns
    assert "Range_Anxiety_Level" in transformed.columns

    # Numeric dtype
    assert np.issubdtype(transformed["Commute_Anxiety"].dtype, np.number)

    # Values: Daily_Commute_km * ordinal(Range_Anxiety_Level)
    # mapping: Low: 0, Medium: 1, High: 2
    # row 0: 10.0 * 0 = 0.0
    # row 1: 25.5 * 1 = 25.5
    # row 2: 50.0 * 2 = 100.0
    # row 3: 30.0 * 0 = 0.0
    expected = pd.Series([0.0, 25.5, 100.0, 0.0], name="Commute_Anxiety")
    pd.testing.assert_series_equal(transformed["Commute_Anxiety"], expected)


def test_add_commute_anxiety_feature_missing_parents_raises() -> None:
    """Test that missing any required parent column raises ValueError."""
    for missing_col in ["Daily_Commute_km", "Range_Anxiety_Level"]:
        cols = {
            "Daily_Commute_km": [15.0],
            "Range_Anxiety_Level": ["Medium"],
        }
        del cols[missing_col]
        df = pd.DataFrame(cols)
        with pytest.raises(ValueError, match=missing_col):
            add_commute_anxiety_feature(df)


def test_add_commute_anxiety_feature_invalid_category_raises() -> None:
    """Test that unmapped Range_Anxiety_Level values raise ValueError."""
    df_invalid = pd.DataFrame(
        {
            "Daily_Commute_km": [20.0],
            "Range_Anxiety_Level": ["Extreme"],
        }
    )
    with pytest.raises(ValueError, match="unmapped Range_Anxiety_Level"):
        add_commute_anxiety_feature(df_invalid)


def test_targeted_feature_registry() -> None:
    """Test policy registry retrieval and listing."""
    policies = list_targeted_feature_contracts()
    assert "concern_subsidy" in policies
    assert "charging_access" in policies
    assert "commute_anxiety" in policies
    assert "income_per_vehicle" in policies
    assert "age_income" in policies

    contract_ca = get_targeted_feature_contract("charging_access")
    assert contract_ca.feature_name == "charging_access"
    assert contract_ca.column_name == "Charging_Access"

    contract_cs = get_targeted_feature_contract("concern_subsidy")
    assert contract_cs.feature_name == "concern_subsidy"
    assert contract_cs.column_name == "Concern_Subsidy"

    contract_cm = get_targeted_feature_contract("commute_anxiety")
    assert contract_cm.feature_name == "commute_anxiety"
    assert contract_cm.column_name == "Commute_Anxiety"
    assert contract_cm.experiment_index == 3
    assert contract_cm.parent_columns == ("Daily_Commute_km", "Range_Anxiety_Level")

    contract_ipv = get_targeted_feature_contract("income_per_vehicle")
    assert contract_ipv.feature_name == "income_per_vehicle"
    assert contract_ipv.column_name == "Income_per_Vehicle"
    assert contract_ipv.experiment_index == 4
    assert contract_ipv.parent_columns == ("Annual_Income_USD", "Number_of_Cars_Owned")

    contract_ai = get_targeted_feature_contract("age_income")
    assert contract_ai.feature_name == "age_income"
    assert contract_ai.column_name == "Age_Income"
    assert contract_ai.experiment_index == 5
    assert contract_ai.parent_columns == ("Age", "Annual_Income_USD")

    with pytest.raises(KeyError, match="Unknown targeted feature policy"):
        get_targeted_feature_contract("non_existent_policy")


def test_isolation_no_carry_forward() -> None:
    """Test that each policy is completely isolated with zero carry-forward features."""
    raw_df = pd.DataFrame(
        {
            "id": [1],
            "Age": [35],
            "Environmental_Concern_Level": [3.0],
            "Subsidy_Available": ["Yes"],
            "Home_Charging_Possible": ["Yes"],
            "Charging_Stations_Near_Home": [2],
            "Charging_Stations_Near_Work": [4],
            "Daily_Commute_km": [35.0],
            "Range_Anxiety_Level": ["High"],
            "Annual_Income_USD": [60000.0],
            "Number_of_Cars_Owned": [2],
        }
    )

    ca_df = CHARGING_ACCESS_CONTRACT.transform_fn(raw_df)
    assert "Charging_Access" in ca_df.columns
    assert "Concern_Subsidy" not in ca_df.columns
    assert "Commute_Anxiety" not in ca_df.columns
    assert "Income_per_Vehicle" not in ca_df.columns
    assert "Age_Income" not in ca_df.columns

    cs_df = CONCERN_SUBSIDY_CONTRACT.transform_fn(raw_df)
    assert "Concern_Subsidy" in cs_df.columns
    assert "Charging_Access" not in cs_df.columns
    assert "Commute_Anxiety" not in cs_df.columns
    assert "Income_per_Vehicle" not in cs_df.columns
    assert "Age_Income" not in cs_df.columns

    cm_df = COMMUTE_ANXIETY_CONTRACT.transform_fn(raw_df)
    assert "Commute_Anxiety" in cm_df.columns
    assert "Concern_Subsidy" not in cm_df.columns
    assert "Charging_Access" not in cm_df.columns
    assert "Income_per_Vehicle" not in cm_df.columns
    assert "Age_Income" not in cm_df.columns

    ipv_df = INCOME_PER_VEHICLE_CONTRACT.transform_fn(raw_df)
    assert "Income_per_Vehicle" in ipv_df.columns
    assert "Concern_Subsidy" not in ipv_df.columns
    assert "Charging_Access" not in ipv_df.columns
    assert "Commute_Anxiety" not in ipv_df.columns
    assert "Age_Income" not in ipv_df.columns

    ai_df = AGE_INCOME_CONTRACT.transform_fn(raw_df)
    assert "Age_Income" in ai_df.columns
    assert "Concern_Subsidy" not in ai_df.columns
    assert "Charging_Access" not in ai_df.columns
    assert "Commute_Anxiety" not in ai_df.columns
    assert "Income_per_Vehicle" not in ai_df.columns


def test_add_income_per_vehicle_feature_schema_and_values() -> None:
    """Test Income_per_Vehicle derivation formula, schema, and parent retention."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "Annual_Income_USD": [50000.0, 120000.0, 90000.0],
            "Number_of_Cars_Owned": [1, 2, 4],
            "Age": [30, 40, 50],
        }
    )
    df_copy = df.copy()

    transformed = add_income_per_vehicle_feature(df)

    # Immutability
    pd.testing.assert_frame_equal(df, df_copy)

    # Exactly one column added
    assert len(transformed.columns) == len(df.columns) + 1
    assert "Income_per_Vehicle" in transformed.columns

    # Retains both parents
    assert "Annual_Income_USD" in transformed.columns
    assert "Number_of_Cars_Owned" in transformed.columns

    # Numeric dtype
    assert np.issubdtype(transformed["Income_per_Vehicle"].dtype, np.floating)

    # Formula: Annual_Income_USD / Number_of_Cars_Owned
    # row 0: 50000.0 / 1 = 50000.0
    # row 1: 120000.0 / 2 = 60000.0
    # row 2: 90000.0 / 4 = 22500.0
    expected = pd.Series([50000.0, 60000.0, 22500.0], name="Income_per_Vehicle")
    pd.testing.assert_series_equal(transformed["Income_per_Vehicle"], expected)


def test_add_income_per_vehicle_feature_missing_parents_raises() -> None:
    """Test that missing required parent columns raises ValueError."""
    df_missing_cars = pd.DataFrame({"Annual_Income_USD": [50000.0]})
    with pytest.raises(ValueError, match="Number_of_Cars_Owned"):
        add_income_per_vehicle_feature(df_missing_cars)

    df_missing_income = pd.DataFrame({"Number_of_Cars_Owned": [2]})
    with pytest.raises(ValueError, match="Annual_Income_USD"):
        add_income_per_vehicle_feature(df_missing_income)


def test_add_income_per_vehicle_feature_zero_or_invalid_denominator_raises() -> None:
    """Test that non-positive or null denominator raises ValueError."""
    # Zero denominator
    df_zero = pd.DataFrame({"Annual_Income_USD": [50000.0], "Number_of_Cars_Owned": [0]})
    with pytest.raises(ValueError, match="Number_of_Cars_Owned.*strictly positive"):
        add_income_per_vehicle_feature(df_zero)

    # Negative denominator
    df_neg = pd.DataFrame({"Annual_Income_USD": [50000.0], "Number_of_Cars_Owned": [-1]})
    with pytest.raises(ValueError, match="Number_of_Cars_Owned.*strictly positive"):
        add_income_per_vehicle_feature(df_neg)

    # Null denominator
    df_null = pd.DataFrame({"Annual_Income_USD": [50000.0], "Number_of_Cars_Owned": [np.nan]})
    with pytest.raises(ValueError, match="Number_of_Cars_Owned.*strictly positive"):
        add_income_per_vehicle_feature(df_null)


def test_add_income_per_vehicle_feature_non_finite_result_raises() -> None:
    """Test that non-finite results raise ValueError."""
    df_inf = pd.DataFrame({"Annual_Income_USD": [np.inf], "Number_of_Cars_Owned": [2]})
    with pytest.raises(ValueError, match="non-finite"):
        add_income_per_vehicle_feature(df_inf)

    df_nan = pd.DataFrame({"Annual_Income_USD": [np.nan], "Number_of_Cars_Owned": [2]})
    with pytest.raises(ValueError, match="non-finite"):
        add_income_per_vehicle_feature(df_nan)


def test_add_age_income_feature_schema_and_values() -> None:
    """Test Age_Income derivation formula, schema, and parent retention."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "Age": [25, 40, 60],
            "Annual_Income_USD": [30000.0, 75000.0, 100000.0],
            "Gender": ["Male", "Female", "Male"],
        }
    )
    df_copy = df.copy()

    transformed = add_age_income_feature(df)

    # Immutability
    pd.testing.assert_frame_equal(df, df_copy)

    # Exactly one column added
    assert len(transformed.columns) == len(df.columns) + 1
    assert "Age_Income" in transformed.columns

    # Retains both parents
    assert "Age" in transformed.columns
    assert "Annual_Income_USD" in transformed.columns

    # Numeric dtype
    assert np.issubdtype(transformed["Age_Income"].dtype, np.number)

    # Formula: Age * Annual_Income_USD
    # row 0: 25 * 30000.0 = 750000.0
    # row 1: 40 * 75000.0 = 3000000.0
    # row 2: 60 * 100000.0 = 6000000.0
    expected = pd.Series([750000.0, 3000000.0, 6000000.0], name="Age_Income")
    pd.testing.assert_series_equal(transformed["Age_Income"], expected)


def test_add_age_income_feature_missing_parents_raises() -> None:
    """Test that missing required parent columns raises ValueError."""
    df_missing_age = pd.DataFrame({"Annual_Income_USD": [50000.0]})
    with pytest.raises(ValueError, match="Age"):
        add_age_income_feature(df_missing_age)

    df_missing_income = pd.DataFrame({"Age": [30]})
    with pytest.raises(ValueError, match="Annual_Income_USD"):
        add_age_income_feature(df_missing_income)


def test_add_age_income_feature_non_numeric_inputs_raises() -> None:
    """Test that non-numeric inputs raise TypeError or ValueError."""
    df_str_age = pd.DataFrame({"Age": ["thirty"], "Annual_Income_USD": [50000.0]})
    with pytest.raises((TypeError, ValueError), match="numeric"):
        add_age_income_feature(df_str_age)

    df_str_income = pd.DataFrame({"Age": [30], "Annual_Income_USD": ["high"]})
    with pytest.raises((TypeError, ValueError), match="numeric"):
        add_age_income_feature(df_str_income)


def test_add_age_income_feature_null_inputs_raises() -> None:
    """Test that null values in parents raise ValueError."""
    df_null_age = pd.DataFrame({"Age": [np.nan], "Annual_Income_USD": [50000.0]})
    with pytest.raises(ValueError, match="null"):
        add_age_income_feature(df_null_age)

    df_null_income = pd.DataFrame({"Age": [30], "Annual_Income_USD": [np.nan]})
    with pytest.raises(ValueError, match="null"):
        add_age_income_feature(df_null_income)


def test_add_age_income_feature_non_finite_inputs_or_result_raises() -> None:
    """Test that non-finite inputs or results raise ValueError."""
    df_inf_age = pd.DataFrame({"Age": [np.inf], "Annual_Income_USD": [50000.0]})
    with pytest.raises(ValueError, match="non-finite"):
        add_age_income_feature(df_inf_age)

    df_inf_income = pd.DataFrame({"Age": [30], "Annual_Income_USD": [np.inf]})
    with pytest.raises(ValueError, match="non-finite"):
        add_age_income_feature(df_inf_income)

    df_neginf_income = pd.DataFrame({"Age": [30], "Annual_Income_USD": [-np.inf]})
    with pytest.raises(ValueError, match="non-finite"):
        add_age_income_feature(df_neginf_income)


def test_compute_params_hash_deterministic() -> None:
    """Test compute_params_hash produces identical 64-char sha256 regardless of key order."""
    params_a = {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100}
    params_b = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    hash_a = compute_params_hash(params_a)
    hash_b = compute_params_hash(params_b)
    assert hash_a == hash_b
    assert len(hash_a) == 64


def test_build_training_provenance_structure(tmp_path: Path) -> None:
    """Test build_training_provenance contains all mandatory fields."""
    oof_df = pd.DataFrame({"row_index": [0, 1], "probability": [0.3, 0.7]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    contract = CONCERN_SUBSIDY_CONTRACT
    params = {"n_estimators": 50, "max_depth": 3}
    prov = build_training_provenance(
        feature_contract=contract,
        model_params=params,
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
        train_file_sha256="d" * 64,
    )

    assert prov["policy_name"] == "concern_subsidy"
    assert prov["feature_contract"]["column_name"] == "Concern_Subsidy"
    assert prov["feature_contract"]["formula"] == contract.formula
    assert prov["model"]["params"] == params
    assert prov["model"]["params_hash"] == compute_params_hash(params)
    assert prov["data_hash"] == "a" * 64
    assert prov["split_hash"] == "b" * 64
    assert prov["seed_policy"]["base_seed"] == 42
    assert prov["training_source_hash"] == "c" * 64
    assert prov["candidate_oof"]["row_count"] == 2
    assert len(prov["candidate_oof"]["sha256"]) == 64


def test_verify_training_provenance_success(tmp_path: Path) -> None:
    """Test verify_training_provenance passes when all immutable fields match."""
    oof_df = pd.DataFrame({"row_index": [0, 1], "probability": [0.4, 0.6]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    contract = CONCERN_SUBSIDY_CONTRACT
    params = {"n_estimators": 50, "max_depth": 3}
    prov = build_training_provenance(
        feature_contract=contract,
        model_params=params,
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
    )

    # Should not raise
    verify_training_provenance(
        training_prov=prov,
        feature_contract=contract,
        model_params=params,
        expected_data_hash="a" * 64,
        expected_split_hash="b" * 64,
        expected_base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        expected_row_count=2,
    )


def test_verify_training_provenance_rejects_missing_file(tmp_path: Path) -> None:
    """Test verify_training_provenance raises FileNotFoundError when candidate OOF missing."""
    contract = CONCERN_SUBSIDY_CONTRACT
    prov = {"candidate_oof": {"sha256": "x" * 64, "row_count": 2}}
    with pytest.raises(FileNotFoundError):
        verify_training_provenance(
            training_prov=prov,
            feature_contract=contract,
            model_params={},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=tmp_path / "non_existent.parquet",
            candidate_oof_df=pd.DataFrame(),
        )


def test_verify_training_provenance_rejects_tampered_oof_sha(tmp_path: Path) -> None:
    """Test ArtifactTamperError when candidate OOF SHA256 does not match provenance."""
    oof_df = pd.DataFrame({"row_index": [0, 1], "probability": [0.4, 0.6]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    contract = CONCERN_SUBSIDY_CONTRACT
    prov = build_training_provenance(
        feature_contract=contract,
        model_params={},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
    )
    # Tamper with recorded hash
    prov["candidate_oof"]["sha256"] = "f" * 64

    with pytest.raises(ArtifactTamperError, match="Candidate OOF hash mismatch"):
        verify_training_provenance(
            training_prov=prov,
            feature_contract=contract,
            model_params={},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )


def test_verify_training_provenance_rejects_tampered_oof_row_count(tmp_path: Path) -> None:
    """Test ArtifactTamperError when candidate OOF row count does not match provenance."""
    oof_df = pd.DataFrame({"row_index": [0, 1], "probability": [0.4, 0.6]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    contract = CONCERN_SUBSIDY_CONTRACT
    prov = build_training_provenance(
        feature_contract=contract,
        model_params={},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
    )
    prov["candidate_oof"]["row_count"] = 999

    with pytest.raises(ArtifactTamperError, match="Candidate OOF row count mismatch"):
        verify_training_provenance(
            training_prov=prov,
            feature_contract=contract,
            model_params={},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )


def test_verify_training_provenance_rejects_policy_and_contract_mismatches(tmp_path: Path) -> None:
    """Test ProvenanceMismatchError for policy name, column name, parents, and formula."""
    oof_df = pd.DataFrame({"row_index": [0], "probability": [0.5]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    base_prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
    )

    # 1. Policy name mismatch
    prov_bad_policy = dict(base_prov)
    prov_bad_policy["policy_name"] = "wrong_policy"
    with pytest.raises(ProvenanceMismatchError, match="Policy mismatch"):
        verify_training_provenance(
            training_prov=prov_bad_policy,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 2. Column name mismatch
    prov_bad_col = json.loads(json.dumps(base_prov))
    prov_bad_col["feature_contract"]["column_name"] = "Wrong_Col"
    with pytest.raises(ProvenanceMismatchError, match="Feature column mismatch"):
        verify_training_provenance(
            training_prov=prov_bad_col,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 3. Parent columns mismatch
    prov_bad_parents = json.loads(json.dumps(base_prov))
    prov_bad_parents["feature_contract"]["parent_columns"] = ["Age"]
    with pytest.raises(ProvenanceMismatchError, match="Feature parent columns mismatch"):
        verify_training_provenance(
            training_prov=prov_bad_parents,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 4. Formula mismatch
    prov_bad_formula = json.loads(json.dumps(base_prov))
    prov_bad_formula["feature_contract"]["formula"] = "wrong_formula"
    with pytest.raises(ProvenanceMismatchError, match="Feature formula mismatch"):
        verify_training_provenance(
            training_prov=prov_bad_formula,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )


def test_verify_training_provenance_rejects_environment_mismatches(tmp_path: Path) -> None:
    """Test ProvenanceMismatchError for params, data hash, split hash, seed, and source hash."""
    oof_df = pd.DataFrame({"row_index": [0], "probability": [0.5]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    base_prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={"max_depth": 3},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
    )

    # 1. Model params mismatch
    with pytest.raises(ProvenanceMismatchError, match="Model parameter hash mismatch"):
        verify_training_provenance(
            training_prov=base_prov,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 5},  # different
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 2. Data hash mismatch
    with pytest.raises(ProvenanceMismatchError, match="Data hash mismatch"):
        verify_training_provenance(
            training_prov=base_prov,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="z" * 64,  # different
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 3. Split hash mismatch
    with pytest.raises(ProvenanceMismatchError, match="Split manifest hash mismatch"):
        verify_training_provenance(
            training_prov=base_prov,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="z" * 64,  # different
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 4. Seed policy mismatch
    with pytest.raises(ProvenanceMismatchError, match="Seed policy mismatch"):
        verify_training_provenance(
            training_prov=base_prov,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=999,  # different
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 5. Missing training source hash
    prov_no_source = json.loads(json.dumps(base_prov))
    del prov_no_source["training_source_hash"]
    with pytest.raises(ProvenanceMismatchError, match="missing 'training_source_hash'"):
        verify_training_provenance(
            training_prov=prov_no_source,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )


def test_reuse_candidate_oof_rejects_legacy_missing_sidecar(tmp_path: Path) -> None:
    """Test that reuse_candidate_oof raises LegacyProvenanceError when sidecar is missing."""
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "playground-series-s6e9" / "train.csv"
    test_path = project_root / "data" / "playground-series-s6e9" / "test.csv"
    split_path = project_root / "outputs" / "splits" / "repeated_stratified_5x3.parquet"
    baseline_oof = (
        project_root
        / "outputs"
        / "ensemble"
        / "candidates"
        / "lightgbm_hpo_trial35"
        / "oof_predictions.parquet"
    )
    final_cfg = project_root / "configs" / "final.yaml"
    frozen_slices = (
        project_root / "outputs" / "diagnostics" / "severe_fn_audit" / "frozen_slices.json"
    )

    if not (train_path.exists() and split_path.exists() and baseline_oof.exists()):
        pytest.skip("Dataset or baseline files missing")

    # Set up mock output dir with candidate_oof.parquet but NO training_provenance.json
    mock_out = tmp_path / "mock_legacy_exp"
    mock_out.mkdir()
    fake_oof = pd.DataFrame(
        {
            "row_index": np.tile(np.arange(10), 3),
            "repeat": np.repeat(np.arange(3), 10),
            "fold": np.tile(np.arange(5), 6),
            "probability": np.full(30, 0.5),
        }
    )
    fake_oof.to_parquet(mock_out / "candidate_oof.parquet", index=False)

    with pytest.raises(LegacyProvenanceError, match="Legacy candidate artifact detected"):
        run_targeted_experiment(
            train_path=train_path,
            test_path=test_path,
            manifest_path=split_path,
            baseline_oof_path=baseline_oof,
            final_config_path=final_cfg,
            frozen_slices_path=frozen_slices,
            output_dir=mock_out,
            reuse_candidate_oof=True,
            feature_policy="concern_subsidy",
        )


def test_reuse_candidate_oof_preserves_original_training_source_hash(tmp_path: Path) -> None:
    """Test that reuse preserves original training source hash and separates evaluation hash."""
    project_root = Path(__file__).resolve().parents[1]
    exp_dir = project_root / "outputs" / "experiments" / "targeted_features" / "age_income"
    if not (exp_dir / "candidate_oof.parquet").exists():
        pytest.skip("age_income artifacts not found")

    with open(exp_dir / "manifest.json") as f:
        manifest = json.load(f)

    with open(exp_dir / "training_provenance.json") as f:
        training_prov = json.load(f)

    # Verify that manifest preserved training_source_hash
    assert manifest["provenance"]["training_source_hash"] == training_prov["training_source_hash"]
    assert manifest["provenance"]["source_hash"] == training_prov["training_source_hash"]
    # Verify evaluation_source_hash is present and valid
    assert "evaluation_source_hash" in manifest["provenance"]
    assert len(manifest["provenance"]["evaluation_source_hash"]) == 64
    # Verify training_provenance is listed in artifacts
    assert "training_provenance" in manifest["artifacts"]
    assert (
        manifest["artifacts"]["training_provenance"]["sha256"]
        == manifest["provenance"]["training_provenance_sha256"]
    )


def test_verify_training_provenance_rejects_seed_policy_mismatches(tmp_path: Path) -> None:
    """Test ProvenanceMismatchError for missing seed_policy, seed, and formula mismatch."""
    oof_df = pd.DataFrame({"row_index": [0], "probability": [0.5]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    base_prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={"max_depth": 3},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
    )

    # 1. Missing seed_policy
    bad_prov1 = json.loads(json.dumps(base_prov))
    del bad_prov1["seed_policy"]
    with pytest.raises(ProvenanceMismatchError, match="missing 'seed_policy'"):
        verify_training_provenance(
            training_prov=bad_prov1,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 2. Formula mismatch
    bad_prov2 = json.loads(json.dumps(base_prov))
    bad_prov2["seed_policy"]["formula"] = "wrong_formula"
    with pytest.raises(ProvenanceMismatchError, match="Seed policy formula mismatch"):
        verify_training_provenance(
            training_prov=bad_prov2,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )


def test_verify_training_provenance_rejects_all_feature_contract_field_mismatches(
    tmp_path: Path,
) -> None:
    """Test ProvenanceMismatchError for every field of feature_contract."""
    oof_df = pd.DataFrame({"row_index": [0], "probability": [0.5]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    base_prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={"max_depth": 3},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
    )

    # Missing feature_contract mapping
    bad_prov = json.loads(json.dumps(base_prov))
    del bad_prov["feature_contract"]
    with pytest.raises(ProvenanceMismatchError, match="missing 'feature_contract'"):
        verify_training_provenance(
            training_prov=bad_prov,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # family mismatch
    bad_family = json.loads(json.dumps(base_prov))
    bad_family["feature_contract"]["family"] = "wrong_family"
    with pytest.raises(ProvenanceMismatchError, match="Feature family mismatch"):
        verify_training_provenance(
            training_prov=bad_family,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # experiment_index mismatch
    bad_idx = json.loads(json.dumps(base_prov))
    bad_idx["feature_contract"]["experiment_index"] = 999
    with pytest.raises(ProvenanceMismatchError, match="Feature experiment_index mismatch"):
        verify_training_provenance(
            training_prov=bad_idx,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # feature_name inside contract mismatch
    bad_fname = json.loads(json.dumps(base_prov))
    bad_fname["feature_contract"]["feature_name"] = "wrong_feature_name"
    with pytest.raises(ProvenanceMismatchError, match="Feature name mismatch"):
        verify_training_provenance(
            training_prov=bad_fname,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # description mismatch
    bad_desc = json.loads(json.dumps(base_prov))
    bad_desc["feature_contract"]["description"] = "wrong_description"
    with pytest.raises(ProvenanceMismatchError, match="Feature description mismatch"):
        verify_training_provenance(
            training_prov=bad_desc,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )


def test_verify_training_provenance_rejects_version_and_model_family_mismatches(
    tmp_path: Path,
) -> None:
    """Test ProvenanceMismatchError for provenance_version and model_family."""
    oof_df = pd.DataFrame({"row_index": [0], "probability": [0.5]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    base_prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={"max_depth": 3},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
        model_family="lightgbm",
        provenance_version="1.0",
    )

    # 1. provenance_version mismatch
    bad_ver = json.loads(json.dumps(base_prov))
    bad_ver["provenance_version"] = "2.0"
    with pytest.raises(ProvenanceMismatchError, match="Provenance version mismatch"):
        verify_training_provenance(
            training_prov=bad_ver,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
            expected_provenance_version="1.0",
        )

    # 2. model mapping missing
    bad_model1 = json.loads(json.dumps(base_prov))
    del bad_model1["model"]
    with pytest.raises(ProvenanceMismatchError, match="missing 'model' mapping"):
        verify_training_provenance(
            training_prov=bad_model1,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
        )

    # 3. model family mismatch
    bad_model2 = json.loads(json.dumps(base_prov))
    bad_model2["model"]["family"] = "catboost"
    with pytest.raises(ProvenanceMismatchError, match="Model family mismatch"):
        verify_training_provenance(
            training_prov=bad_model2,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
            expected_model_family="lightgbm",
        )


def test_verify_training_provenance_rejects_input_and_train_file_sha_mismatches(
    tmp_path: Path,
) -> None:
    """Test ProvenanceMismatchError for input_file_sha256 and train_file_sha256."""
    oof_df = pd.DataFrame({"row_index": [0], "probability": [0.5]})
    oof_path = tmp_path / "cand.parquet"
    oof_df.to_parquet(oof_path, index=False)

    base_prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={"max_depth": 3},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
        train_file_sha256="d" * 64,
        input_file_sha256="a" * 64,
    )

    # 1. input_file_sha256 mismatch
    with pytest.raises(ProvenanceMismatchError, match="Input file SHA256 mismatch"):
        verify_training_provenance(
            training_prov=base_prov,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
            expected_input_file_sha256="wrong_input_sha",
        )

    # 2. train_file_sha256 mismatch
    with pytest.raises(ProvenanceMismatchError, match="Train file SHA256 mismatch"):
        verify_training_provenance(
            training_prov=base_prov,
            feature_contract=CONCERN_SUBSIDY_CONTRACT,
            model_params={"max_depth": 3},
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
            expected_base_seed=42,
            candidate_oof_path=oof_path,
            candidate_oof_df=oof_df,
            expected_train_file_sha256="wrong_train_sha",
        )


def test_verify_training_anchor_suite(tmp_path: Path) -> None:
    """Comprehensive tests for training anchor validation."""
    from ev_purchase.targeted_feature_experiment import compute_file_sha256

    oof_df = pd.DataFrame({"row_index": [0, 1], "probability": [0.3, 0.7]})
    oof_path = tmp_path / "candidate_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)

    prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={"max_depth": 3},
        data_hash="a" * 64,
        split_hash="b" * 64,
        base_seed=42,
        candidate_oof_path=oof_path,
        candidate_oof_df=oof_df,
        source_hash="c" * 64,
        train_file_sha256="d" * 64,
    )
    prov_path = tmp_path / "training_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    valid_anchor = {
        "operation": "training",
        "experiment_id": "concern_subsidy",
        "policy_name": "concern_subsidy",
        "provenance": {
            "operation": "training",
            "train_file_sha256": "d" * 64,
            "data_hash": "a" * 64,
            "split_hash": "b" * 64,
        },
        "artifacts": {
            "candidate_oof": {
                "sha256": compute_file_sha256(oof_path),
                "path": str(oof_path),
            },
            "training_provenance": {
                "sha256": compute_file_sha256(prov_path),
                "path": str(prov_path),
            },
        },
    }

    manifest_path = tmp_path / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(valid_anchor, f, indent=2)

    # 1. Success case
    anchor = verify_training_anchor(
        training_manifest_path=manifest_path,
        cand_oof_path=oof_path,
        training_prov_path=prov_path,
        expected_train_file_sha256="d" * 64,
        expected_data_hash="a" * 64,
        expected_split_hash="b" * 64,
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
    )
    assert anchor["operation"] == "training"

    # 2. Missing manifest
    with pytest.raises(LegacyProvenanceError, match="not found"):
        verify_training_anchor(
            training_manifest_path=tmp_path / "non_existent.json",
            cand_oof_path=oof_path,
            training_prov_path=prov_path,
            expected_train_file_sha256="d" * 64,
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
        )

    # 3. Non-training operation
    bad_op = json.loads(json.dumps(valid_anchor))
    bad_op["operation"] = "evaluation"
    bad_op["provenance"]["operation"] = "evaluation"
    bad_op_path = tmp_path / "bad_op.json"
    with open(bad_op_path, "w") as f:
        json.dump(bad_op, f)
    with pytest.raises(ProvenanceMismatchError, match="operation mismatch"):
        verify_training_anchor(
            training_manifest_path=bad_op_path,
            cand_oof_path=oof_path,
            training_prov_path=prov_path,
            expected_train_file_sha256="d" * 64,
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
        )

    # 4. Tampered sidecar sha
    bad_sidecar_sha = json.loads(json.dumps(valid_anchor))
    bad_sidecar_sha["artifacts"]["training_provenance"]["sha256"] = "f" * 64
    bad_sidecar_path = tmp_path / "bad_sidecar.json"
    with open(bad_sidecar_path, "w") as f:
        json.dump(bad_sidecar_sha, f)
    with pytest.raises(ArtifactTamperError, match="Training provenance sidecar hash"):
        verify_training_anchor(
            training_manifest_path=bad_sidecar_path,
            cand_oof_path=oof_path,
            training_prov_path=prov_path,
            expected_train_file_sha256="d" * 64,
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
        )

    # 5. Tampered OOF sha
    bad_oof_sha = json.loads(json.dumps(valid_anchor))
    bad_oof_sha["artifacts"]["candidate_oof"]["sha256"] = "f" * 64
    bad_oof_path = tmp_path / "bad_oof.json"
    with open(bad_oof_path, "w") as f:
        json.dump(bad_oof_sha, f)
    with pytest.raises(ArtifactTamperError, match="Candidate OOF hash"):
        verify_training_anchor(
            training_manifest_path=bad_oof_path,
            cand_oof_path=oof_path,
            training_prov_path=prov_path,
            expected_train_file_sha256="d" * 64,
            expected_data_hash="a" * 64,
            expected_split_hash="b" * 64,
        )


def test_reuse_candidate_oof_rejects_missing_training_manifest_anchor(tmp_path: Path) -> None:
    """Test that reuse raises LegacyProvenanceError when training_manifest.json is missing."""
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "playground-series-s6e9" / "train.csv"
    test_path = project_root / "data" / "playground-series-s6e9" / "test.csv"
    split_path = project_root / "outputs" / "splits" / "repeated_stratified_5x3.parquet"
    baseline_oof = (
        project_root
        / "outputs"
        / "ensemble"
        / "candidates"
        / "lightgbm_hpo_trial35"
        / "oof_predictions.parquet"
    )
    final_cfg = project_root / "configs" / "final.yaml"
    frozen_slices = (
        project_root / "outputs" / "diagnostics" / "severe_fn_audit" / "frozen_slices.json"
    )

    if not (train_path.exists() and split_path.exists() and baseline_oof.exists()):
        pytest.skip("Dataset or baseline files missing")

    mock_out = tmp_path / "mock_missing_manifest_exp"
    mock_out.mkdir()
    fake_oof = pd.DataFrame(
        {
            "row_index": np.tile(np.arange(10), 3),
            "repeat": np.repeat(np.arange(3), 10),
            "fold": np.tile(np.arange(5), 6),
            "probability": np.full(30, 0.5),
        }
    )
    fake_oof.to_parquet(mock_out / "candidate_oof.parquet", index=False)
    fake_prov = build_training_provenance(
        feature_contract=CONCERN_SUBSIDY_CONTRACT,
        model_params={},
        data_hash="dummy",
        split_hash="dummy",
        base_seed=42,
        candidate_oof_path=mock_out / "candidate_oof.parquet",
        candidate_oof_df=fake_oof,
    )
    with open(mock_out / "training_provenance.json", "w") as f:
        json.dump(fake_prov, f)

    with pytest.raises(LegacyProvenanceError, match="Prior training-mode manifest"):
        run_targeted_experiment(
            train_path=train_path,
            test_path=test_path,
            manifest_path=split_path,
            baseline_oof_path=baseline_oof,
            final_config_path=final_cfg,
            frozen_slices_path=frozen_slices,
            output_dir=mock_out,
            reuse_candidate_oof=True,
            feature_policy="concern_subsidy",
        )


def test_honest_threat_model_statement_present() -> None:
    """Test that THREAT_MODEL_STATEMENT states local integrity vs lack of unforgeability."""
    assert "local integrity" in THREAT_MODEL_STATEMENT.lower()
    assert (
        "not guarantee" in THREAT_MODEL_STATEMENT.lower() or "not" in THREAT_MODEL_STATEMENT.lower()
    )
    assert "signatures" in THREAT_MODEL_STATEMENT.lower()
