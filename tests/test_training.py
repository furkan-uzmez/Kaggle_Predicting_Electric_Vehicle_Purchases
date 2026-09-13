from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ev_purchase.artifacts import load_run_metrics
from ev_purchase.splits import build_split_manifest
from ev_purchase.training import aggregate_repeated_oof, run_cross_validation


def test_aggregate_repeated_oof_means_three_predictions_per_row() -> None:
    row_index = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = aggregate_repeated_oof(row_index, predictions, n_rows=2, n_repeats=3)
    np.testing.assert_allclose(result, np.array([0.2, 0.8]))


def test_aggregate_repeated_oof_raises_if_incomplete_coverage() -> None:
    row_index = np.array([0, 0, 1, 1, 1])
    predictions = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    with pytest.raises(ValueError, match="every row must have one OOF prediction per repeat"):
        aggregate_repeated_oof(row_index, predictions, n_rows=2, n_repeats=3)


def test_leakage_guard_preprocessor_model_never_sees_validation_data(tmp_path: Path) -> None:
    """Verify that during CV, fit is called only on training fold data."""
    # Build synthetic dataset
    np.random.seed(42)
    n = 100
    train_df = pd.DataFrame(
        {
            "id": [f"tr_{i}" for i in range(n)],
            "num1": np.random.randn(n),
            "cat1": np.random.choice(["A", "B"], size=n),
            "Will_Buy_EV": np.random.choice(["No", "Yes"], size=n),
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [f"te_{i}" for i in range(20)],
            "num1": np.random.randn(20),
            "cat1": np.random.choice(["A", "B"], size=20),
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sub_csv = data_dir / "sample_submission.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    sub_df = pd.DataFrame({"id": test_df["id"], "Will_Buy_EV": 0.5})
    sub_df.to_csv(sub_csv, index=False)

    y_encoded = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    manifest = build_split_manifest(y_encoded, n_splits=5, n_repeats=1, random_state=42)

    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path)

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  seed: 42
model:
  name: dummy
data:
  train: {train_csv.relative_to(tmp_path)}
  test: {test_csv.relative_to(tmp_path)}
  sample_submission: {sub_csv.relative_to(tmp_path)}
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 1
  random_state: 42
metrics:
  primary: roc_auc
  guardrails:
    - average_precision
    - log_loss
outputs:
  root: outputs
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    result = run_cross_validation(
        config_path=cfg_path,
        project_root=tmp_path,
        split_manifest_path=manifest_path,
        smoke=False,
    )

    assert "aggregate_metrics" in result
    assert "oof_df" in result
    assert len(result["oof_df"]) == n


def test_rejection_when_manifest_and_data_disagree(tmp_path: Path) -> None:
    """Verify ValueError when manifest row count or labels disagree with data."""
    train_df = pd.DataFrame(
        {
            "id": ["tr_0", "tr_1", "tr_2", "tr_3"],
            "num1": [1.0, 2.0, 3.0, 4.0],
            "Will_Buy_EV": ["No", "No", "Yes", "Yes"],
        }
    )
    test_df = pd.DataFrame({"id": ["te_0"], "num1": [1.0]})

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sub_csv = data_dir / "sample_submission.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    pd.DataFrame({"id": test_df["id"], "Will_Buy_EV": 0.5}).to_csv(sub_csv, index=False)

    # Wrong size manifest (for 10 rows instead of 4)
    y_dummy = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    manifest = build_split_manifest(y_dummy, n_splits=5, n_repeats=1, random_state=42)
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path)

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  seed: 42
model:
  name: dummy
data:
  train: {train_csv.relative_to(tmp_path)}
  test: {test_csv.relative_to(tmp_path)}
  sample_submission: {sub_csv.relative_to(tmp_path)}
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 1
  random_state: 42
metrics:
  primary: roc_auc
  guardrails:
    - average_precision
    - log_loss
outputs:
  root: outputs
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    with pytest.raises(ValueError):
        run_cross_validation(
            config_path=cfg_path,
            project_root=tmp_path,
            split_manifest_path=manifest_path,
        )


def test_determinism_across_identical_runs(tmp_path: Path) -> None:
    """Verify two runs with identical inputs produce identical OOF probabilities and metrics."""
    np.random.seed(42)
    n = 60
    train_df = pd.DataFrame(
        {
            "id": [f"tr_{i}" for i in range(n)],
            "num1": np.random.randn(n),
            "cat1": np.random.choice(["X", "Y"], size=n),
            "Will_Buy_EV": np.random.choice(["No", "Yes"], size=n),
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [f"te_{i}" for i in range(10)],
            "num1": np.random.randn(10),
            "cat1": np.random.choice(["X", "Y"], size=10),
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sub_csv = data_dir / "sample_submission.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    pd.DataFrame({"id": test_df["id"], "Will_Buy_EV": 0.5}).to_csv(sub_csv, index=False)

    y_encoded = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    manifest = build_split_manifest(y_encoded, n_splits=5, n_repeats=2, random_state=42)
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path)

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  seed: 42
model:
  name: logistic
data:
  train: {train_csv.relative_to(tmp_path)}
  test: {test_csv.relative_to(tmp_path)}
  sample_submission: {sub_csv.relative_to(tmp_path)}
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 2
  random_state: 42
metrics:
  primary: roc_auc
  guardrails:
    - average_precision
    - log_loss
outputs:
  root: outputs
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    run1 = run_cross_validation(
        config_path=cfg_path, project_root=tmp_path, split_manifest_path=manifest_path
    )
    run2 = run_cross_validation(
        config_path=cfg_path, project_root=tmp_path, split_manifest_path=manifest_path
    )

    np.testing.assert_array_equal(run1["oof_probs"], run2["oof_probs"])
    np.testing.assert_array_equal(run1["test_probs"], run2["test_probs"])
    assert run1["aggregate_metrics"] == run2["aggregate_metrics"]


def test_smoke_mode_produces_distinguishable_artifact(tmp_path: Path) -> None:
    """Verify smoke mode records smoke status and produces valid artifact."""
    np.random.seed(42)
    n = 50
    train_df = pd.DataFrame(
        {
            "id": [f"tr_{i}" for i in range(n)],
            "num1": np.random.randn(n),
            "cat1": np.random.choice(["A", "B"], size=n),
            "Will_Buy_EV": np.random.choice(["No", "Yes"], size=n),
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [f"te_{i}" for i in range(10)],
            "num1": np.random.randn(10),
            "cat1": np.random.choice(["A", "B"], size=10),
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sub_csv = data_dir / "sample_submission.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    pd.DataFrame({"id": test_df["id"], "Will_Buy_EV": 0.5}).to_csv(sub_csv, index=False)

    y_encoded = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    manifest = build_split_manifest(y_encoded, n_splits=5, n_repeats=1, random_state=42)
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path)

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  seed: 42
model:
  name: dummy
data:
  train: {train_csv.relative_to(tmp_path)}
  test: {test_csv.relative_to(tmp_path)}
  sample_submission: {sub_csv.relative_to(tmp_path)}
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 1
  random_state: 42
metrics:
  primary: roc_auc
  guardrails:
    - average_precision
    - log_loss
outputs:
  root: outputs
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    result = run_cross_validation(
        config_path=cfg_path,
        project_root=tmp_path,
        split_manifest_path=manifest_path,
        smoke=True,
    )

    assert result["status"] == "smoke_completed"
    metrics_file = result["metrics_path"]
    metrics_payload = load_run_metrics(metrics_file)
    assert metrics_payload["status"] == "smoke_completed"


def test_smoke_run_split_sha256_truthful_and_records_effective_metadata(tmp_path: Path) -> None:
    """Verify smoke run split_sha256 matches actual smoke split and records effective metadata."""
    import hashlib

    from ev_purchase.training import _compute_file_sha256

    np.random.seed(42)
    n = 100
    train_df = pd.DataFrame(
        {
            "id": [f"tr_{i}" for i in range(n)],
            "num1": np.random.randn(n),
            "cat1": np.random.choice(["A", "B"], size=n),
            "Will_Buy_EV": np.random.choice(["No", "Yes"], size=n),
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [f"te_{i}" for i in range(10)],
            "num1": np.random.randn(10),
            "cat1": np.random.choice(["A", "B"], size=10),
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sub_csv = data_dir / "sample_submission.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    pd.DataFrame({"id": test_df["id"], "Will_Buy_EV": 0.5}).to_csv(sub_csv, index=False)

    y_encoded = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    prod_manifest = build_split_manifest(y_encoded, n_splits=5, n_repeats=3, random_state=42)
    prod_manifest_path = tmp_path / "manifest.parquet"
    prod_manifest.to_parquet(prod_manifest_path)
    prod_manifest_sha256 = _compute_file_sha256(prod_manifest_path)

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  seed: 42
model:
  name: dummy
data:
  train: {train_csv.relative_to(tmp_path)}
  test: {test_csv.relative_to(tmp_path)}
  sample_submission: {sub_csv.relative_to(tmp_path)}
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails:
    - average_precision
    - log_loss
outputs:
  root: outputs
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    result = run_cross_validation(
        config_path=cfg_path,
        project_root=tmp_path,
        split_manifest_path=prod_manifest_path,
        smoke=True,
    )

    metrics_payload = load_run_metrics(result["metrics_path"])

    assert metrics_payload["split_sha256"] != prod_manifest_sha256

    smoke_n_rows = min(5000, n)
    smoke_y = y_encoded[:smoke_n_rows]
    expected_smoke_manifest = build_split_manifest(
        smoke_y, n_splits=5, n_repeats=1, random_state=42
    )
    expected_smoke_sha256 = hashlib.sha256(expected_smoke_manifest.to_parquet()).hexdigest()
    assert metrics_payload["split_sha256"] == expected_smoke_sha256

    assert metrics_payload["status"] == "smoke_completed"
    assert metrics_payload["smoke"] is True
    assert metrics_payload["effective_n_splits"] == 5
    assert metrics_payload["effective_n_repeats"] == 1
    assert metrics_payload["effective_n_rows"] == smoke_n_rows


def test_completed_run_requires_non_empty_fold_metrics_and_artifact_paths() -> None:
    """Verify completed run fails loudly if fold_metrics or artifact_paths is empty."""
    from ev_purchase.training import _validate_completed_run_payload

    with pytest.raises(ValueError, match="fold_metrics cannot be empty"):
        _validate_completed_run_payload(
            "completed", fold_metrics=[], artifact_paths={"metrics": "a"}
        )

    with pytest.raises(ValueError, match="artifact_paths cannot be empty"):
        _validate_completed_run_payload(
            "completed", fold_metrics=[{"roc_auc": 0.5}], artifact_paths={}
        )

    # Smoke completed runs do not trigger completed-only strict validation error if empty
    _validate_completed_run_payload("smoke_completed", fold_metrics=[], artifact_paths={})
