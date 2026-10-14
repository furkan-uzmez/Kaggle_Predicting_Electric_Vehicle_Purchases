"""Unit and contract tests for leakage-safe adversarial validation."""

import json
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import pytest

from ev_purchase.adversarial_validation import (
    AdversarialValidationResult,
    compute_file_sha256,
    prepare_adversarial_data,
    run_adversarial_validation,
)


def _make_dummy_data(
    n_train: int = 200,
    n_test: int = 100,
    shift: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_df = pd.DataFrame(
        {
            "id": np.arange(n_train),
            "Will_Buy_EV": rng.choice(["No", "Yes"], size=n_train),
            "num_feat1": rng.normal(0.0, 1.0, size=n_train),
            "num_feat2": rng.normal(5.0, 2.0, size=n_train),
            "cat_feat": rng.choice(["A", "B", "C"], size=n_train),
        }
    )
    test_df = pd.DataFrame(
        {
            "id": np.arange(n_train, n_train + n_test),
            "num_feat1": rng.normal(0.0 + shift, 1.0, size=n_test),
            "num_feat2": rng.normal(5.0, 2.0, size=n_test),
            "cat_feat": rng.choice(["A", "B", "C"], size=n_test),
        }
    )
    return train_df, test_df


def test_prepare_adversarial_data_excludes_id_and_target() -> None:
    train_df, test_df = _make_dummy_data()
    X, y_domain, used_cols, excluded_cols = prepare_adversarial_data(
        train_df, test_df, id_column="id", target_column="Will_Buy_EV"
    )

    assert "id" not in X.columns
    assert "Will_Buy_EV" not in X.columns
    assert set(excluded_cols) == {"id", "Will_Buy_EV"}
    assert set(used_cols) == {"num_feat1", "num_feat2", "cat_feat"}
    assert len(X) == len(train_df) + len(test_df)
    assert len(y_domain) == len(X)
    assert (y_domain[: len(train_df)] == 0).all()
    assert (y_domain[len(train_df) :] == 1).all()


def test_prepare_adversarial_data_rejects_missing_columns() -> None:
    train_df, test_df = _make_dummy_data()
    bad_train = train_df.drop(columns=["num_feat1"])
    with pytest.raises(ValueError, match="Feature column mismatch"):
        prepare_adversarial_data(bad_train, test_df, id_column="id", target_column="Will_Buy_EV")


def test_no_id_target_leakage_in_predictors() -> None:
    train_df, test_df = _make_dummy_data()
    test_df_with_target = test_df.copy()
    test_df_with_target["Will_Buy_EV"] = "No"
    # Even if target is present in test, it must be excluded
    X, y_domain, used_cols, excluded_cols = prepare_adversarial_data(
        train_df, test_df_with_target, id_column="id", target_column="Will_Buy_EV"
    )
    assert "Will_Buy_EV" not in X.columns
    assert "id" not in X.columns
    assert "Will_Buy_EV" in excluded_cols
    assert "id" in excluded_cols


def test_adversarial_validation_synthetic_same_distribution() -> None:
    train_df, test_df = _make_dummy_data(n_train=400, n_test=200, shift=0.0, seed=42)
    result = run_adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        id_column="id",
        target_column="Will_Buy_EV",
        n_splits=5,
        random_state=42,
        lgb_params={"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1},
    )

    assert isinstance(result, AdversarialValidationResult)
    assert len(result.oof_domain_preds) == len(train_df) + len(test_df)
    assert np.isfinite(result.oof_domain_preds).all()
    assert (result.oof_domain_preds >= 0.0).all() and (result.oof_domain_preds <= 1.0).all()

    # Same distribution: ROC-AUC should be near chance ~0.50 [0.40, 0.60]
    auc = result.metrics["roc_auc"]
    assert 0.40 <= auc <= 0.60, f"Expected AUC near chance ~0.50, got {auc:.4f}"
    assert result.interpretation["shift_band"] in ["little detectable shift", "mild shift"]
    assert "association" in result.interpretation["caveats"].lower()

    # Dynamic action recommendation for low shift: no domain adaptation needed
    action_text = result.interpretation["action"].lower()
    assert (
        "trust local cross-validation" in action_text
        or "no adversarial sample reweighting" in action_text
        or "without domain adaptation" in action_text
    )


def test_adversarial_validation_deliberate_shift_high_auc() -> None:
    train_df, test_df = _make_dummy_data(n_train=400, n_test=200, shift=3.0, seed=42)
    result = run_adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        id_column="id",
        target_column="Will_Buy_EV",
        n_splits=5,
        random_state=42,
        lgb_params={"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1},
    )

    # Shifted distribution: ROC-AUC should be substantially higher (>0.75)
    auc = result.metrics["roc_auc"]
    assert auc > 0.75, f"Expected high AUC on shifted data, got {auc:.4f}"
    assert result.interpretation["shift_band"] == "material diagnostic shift"

    # num_feat1 is the shifted feature, it should have top importance
    top_feature = result.feature_importance.iloc[0]["feature"]
    assert top_feature == "num_feat1"

    # Dynamic action recommendation for material shift: investigate shift, reweighting/pruning
    action_text = result.interpretation["action"].lower()
    assert (
        "investigate domain shift" in action_text
        or "reweighting" in action_text
        or "pruning" in action_text
    )


def test_adversarial_validation_action_branching_behavioral() -> None:
    """Verify interpretation and action branching dynamically across all shift levels."""
    from ev_purchase.adversarial_validation import get_interpretation_bands

    # Low shift band
    low = get_interpretation_bands(0.50)
    assert low["shift_band"] == "little detectable shift"
    assert "trust local cross-validation" in low["action"].lower()

    # Mild shift band
    mild = get_interpretation_bands(0.58)
    assert mild["shift_band"] == "mild shift"
    assert "monitor" in mild["action"].lower() or "inspect" in mild["action"].lower()

    # Material shift band
    mat = get_interpretation_bands(0.72)
    assert mat["shift_band"] == "material diagnostic shift"
    assert "investigate" in mat["action"].lower() or "reweighting" in mat["action"].lower()


def test_adversarial_validation_complete_oof_coverage() -> None:
    train_df, test_df = _make_dummy_data(n_train=250, n_test=125, seed=42)
    result = run_adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        id_column="id",
        target_column="Will_Buy_EV",
        n_splits=5,
        random_state=42,
        lgb_params={"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1},
    )

    total_len = len(train_df) + len(test_df)
    assert len(result.oof_domain_preds) == total_len
    # Verify no unassigned or NaN predictions exist
    assert not np.isnan(result.oof_domain_preds).any()
    assert np.isfinite(result.oof_domain_preds).all()
    assert (result.oof_domain_preds >= 0.0).all()
    assert (result.oof_domain_preds <= 1.0).all()

    # Check fold validation sizes partition total rows exactly
    val_sizes_sum = sum(f["val_size"] for f in result.fold_metrics)
    assert val_sizes_sum == total_len
    # Every fold has non-empty validation set
    for f in result.fold_metrics:
        assert f["val_size"] > 0
        assert f["train_size"] > 0
        assert f["val_size"] + f["train_size"] == total_len


def test_adversarial_artifacts_persistence_and_determinism(tmp_path: Path) -> None:
    train_df, test_df = _make_dummy_data(n_train=200, n_test=100, seed=42)
    res1 = run_adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        id_column="id",
        target_column="Will_Buy_EV",
        n_splits=5,
        random_state=42,
        artifacts_dir=tmp_path,
        lgb_params={"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1},
    )

    json_path = tmp_path / "adversarial_validation.json"
    fold_csv = tmp_path / "adversarial_fold_metrics.csv"
    feat_csv = tmp_path / "adversarial_feature_importance.csv"

    assert json_path.exists()
    assert fold_csv.exists()
    assert feat_csv.exists()

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    # Check schema contract
    assert payload["schema_version"] == "1.0.0"
    assert "metrics" in payload
    assert "roc_auc" in payload["metrics"]
    assert "average_precision" in payload["metrics"]
    assert "log_loss" in payload["metrics"]
    assert "fold_roc_auc_mean" in payload["metrics"]
    assert "fold_roc_auc_std" in payload["metrics"]
    assert "baseline_prevalence" in payload["metrics"]
    assert payload["row_counts"]["train_rows"] == 200
    assert payload["row_counts"]["test_rows"] == 100
    assert payload["row_counts"]["total_rows"] == 300
    assert payload["cv_config"]["strategy"] == "StratifiedKFold"
    assert payload["cv_config"]["n_splits"] == 5
    assert payload["cv_config"]["random_state"] == 42
    assert "caveats" in payload["interpretation"]

    df_fold = pd.read_csv(fold_csv)
    assert len(df_fold) == 5
    assert "fold" in df_fold.columns
    assert "roc_auc" in df_fold.columns

    df_feat = pd.read_csv(feat_csv)
    assert "feature" in df_feat.columns
    assert "importance_mean" in df_feat.columns
    assert "importance_std" in df_feat.columns
    # Must be sorted descending by importance_mean
    assert df_feat["importance_mean"].is_monotonic_decreasing

    # Determinism: run a second time with same seed and check exact equality
    res2 = run_adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        id_column="id",
        target_column="Will_Buy_EV",
        n_splits=5,
        random_state=42,
        lgb_params={"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1},
    )
    np.testing.assert_allclose(res1.oof_domain_preds, res2.oof_domain_preds)
    assert res1.metrics["roc_auc"] == res2.metrics["roc_auc"]


def test_compute_file_sha256(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("adversarial validation sha test", encoding="utf-8")
    h = compute_file_sha256(test_file)
    assert isinstance(h, str)
    assert len(h) == 64

    missing_file = tmp_path / "non_existent.txt"
    with pytest.raises(FileNotFoundError, match="File not found"):
        compute_file_sha256(missing_file)


def test_adversarial_artifacts_sha256_hashes_validated(tmp_path: Path) -> None:
    train_df, test_df = _make_dummy_data(n_train=100, n_test=50, seed=42)
    train_path = tmp_path / "dummy_train.csv"
    test_path = tmp_path / "dummy_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_sha = compute_file_sha256(train_path)
    test_sha = compute_file_sha256(test_path)
    assert len(train_sha) == 64
    assert len(test_sha) == 64

    data_sha256 = {
        "train_sha256": train_sha,
        "test_sha256": test_sha,
    }

    result = run_adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        id_column="id",
        target_column="Will_Buy_EV",
        n_splits=5,
        random_state=42,
        artifacts_dir=tmp_path / "artifacts",
        data_sha256=data_sha256,
        lgb_params={"n_estimators": 10, "learning_rate": 0.1, "num_leaves": 10, "verbose": -1},
    )

    json_path = tmp_path / "artifacts" / "adversarial_validation.json"
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["data_sha256"]["train_sha256"] == train_sha
    assert payload["data_sha256"]["test_sha256"] == test_sha
    assert payload["data_sha256"] == result.data_sha256


def test_notebook01_generator_source_sync() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB01, _notebook

    nb_gen = _notebook(NB01)
    nb_file = nbformat.read("notebooks/01_data_understanding.ipynb", as_version=4)
    assert len(nb_gen.cells) == len(nb_file.cells), (
        f"Cell count mismatch: gen {len(nb_gen.cells)} vs file {len(nb_file.cells)}"
    )
    for idx in range(len(nb_gen.cells)):
        assert nb_gen.cells[idx].source.strip() == nb_file.cells[idx].source.strip(), (
            f"Source mismatch at cell {idx}"
        )


def test_notebook01_consecutive_cells_run_in_order(tmp_path: Path) -> None:
    """Execute consecutive code cells of Notebook 01 in order with sampled data."""
    nb_file = nbformat.read("notebooks/01_data_understanding.ipynb", as_version=4)
    code_cells = [cell.source for cell in nb_file.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 9

    # Isolated execution scope mimicking top-level notebook execution
    scope: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(Path.cwd() / "notebooks" / "01_data_understanding.ipynb"),
    }

    # Execute Cell 0 (setup)
    exec(code_cells[0], scope)
    assert "PROJECT_ROOT" in scope
    assert "config" in scope

    # Execute Cell 1 (load data and contract validation)
    exec(code_cells[1], scope)
    assert "train_df" in scope
    assert "test_df" in scope

    # Subsample frames and redirect artifact path to tmp_path to keep test run fast (<2s)
    scope["train_df"] = scope["train_df"].head(200).copy()
    scope["test_df"] = scope["test_df"].head(100).copy()
    temp_artifacts = tmp_path / "diag_out"
    temp_artifacts.mkdir(parents=True, exist_ok=True)

    # Execute Cell 2 through Cell 7 (predictors, data quality, diagnostics, shift)
    for idx in range(2, 8):
        exec(code_cells[idx], scope)

    # Prepare cell 8 (adversarial validation) to write to temporary artifacts directory
    adv_code = code_cells[8].replace(
        'artifacts_dir = PROJECT_ROOT / "outputs" / "diagnostics"',
        f'artifacts_dir = Path(r"{temp_artifacts}")',
    )
    exec(adv_code, scope)

    assert "adversarial_res" in scope
    res: AdversarialValidationResult = scope["adversarial_res"]
    assert len(res.oof_domain_preds) == 300
    assert not np.isnan(res.oof_domain_preds).any()
    assert 0.0 <= res.metrics["roc_auc"] <= 1.0
    assert (temp_artifacts / "adversarial_validation.json").exists()
    assert (temp_artifacts / "adversarial_fold_metrics.csv").exists()
    assert (temp_artifacts / "adversarial_feature_importance.csv").exists()

    # Continue executing remaining consecutive code cells through end of notebook
    for idx in range(9, len(code_cells)):
        exec(code_cells[idx], scope)

    assert "decisions" in scope
    assert "deferred" in scope


def test_notebook01_source_and_compile_contract() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB01, _notebook

    nb = _notebook(NB01)
    nbformat.validate(nb)

    # Check that all code cells in NB01 compile without syntax errors
    compile_errors: list[str] = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            try:
                compile(cell.source, f"<nb01_cell_{idx}>", "exec")
            except SyntaxError as exc:
                compile_errors.append(f"Cell {idx}: {exc}")

    assert not compile_errors, f"Code cell compile errors: {compile_errors}"


def test_notebook01_markdown_neutrality_and_no_premature_claims() -> None:
    """Ensure entire NB01 markdown rejects premature conclusions and keeps neutral screening."""
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB01, _notebook

    nb = _notebook(NB01)
    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown").lower()

    forbidden_phrases = [
        "same generating process",
        "track the leaderboard",
        "trust cross-validation",
        "trust cv",
        "0.4988",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in md_text, f"Premature/unconditional claim in NB01 markdown: {phrase}"

    assert "screening only" in md_text
    assert "adversarial validation below determines" in md_text
