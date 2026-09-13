import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from ev_purchase.splits import build_split_manifest, validate_split_manifest  # noqa: E402
from scripts.prepare_data import write_split_artifacts_atomically  # noqa: E402


def test_repeated_manifest_has_one_validation_assignment_per_repeat() -> None:
    y = np.array([0, 1] * 50)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=3, random_state=42)
    assert len(manifest) == len(y) * 3
    assert manifest.groupby(["repeat", "row_index"]).size().eq(1).all()
    assert sorted(manifest["repeat"].unique()) == [0, 1, 2]
    assert sorted(manifest["fold"].unique()) == [0, 1, 2, 3, 4]


def test_repeated_manifest_is_deterministic() -> None:
    y = np.array([0, 1] * 50)
    first = build_split_manifest(y, n_splits=5, n_repeats=3, random_state=42)
    second = build_split_manifest(y, n_splits=5, n_repeats=3, random_state=42)
    assert first.equals(second)


def test_manifest_schema_dtypes_and_sorting() -> None:
    y = np.array([0, 1] * 30)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    assert list(manifest.columns) == ["row_index", "repeat", "fold"]
    assert manifest["row_index"].dtype == np.dtype("int64")
    assert manifest["repeat"].dtype == np.dtype("int64")
    assert manifest["fold"].dtype == np.dtype("int64")

    # Verify stable sorting
    sorted_df = manifest.sort_values(["repeat", "fold", "row_index"], kind="stable").reset_index(
        drop=True
    )
    assert manifest.equals(sorted_df)


def test_build_split_manifest_validates_y_input() -> None:
    # 2D array rejection
    with pytest.raises(ValueError, match="1D"):
        build_split_manifest(np.array([[0, 1], [1, 0]]), 5, 3, 42)

    # Empty array rejection
    with pytest.raises(ValueError, match="non-empty"):
        build_split_manifest(np.array([]), 5, 3, 42)

    # Missing / non-finite rejection
    with pytest.raises(ValueError, match="nonfinite|missing"):
        build_split_manifest(np.array([0, 1, np.nan]), 5, 3, 42)

    # Non-binary target rejection (3 classes)
    with pytest.raises(ValueError, match="binary"):
        build_split_manifest(np.array([0, 1, 2, 0, 1, 2]), 5, 3, 42)

    # Non-binary target rejection (single class)
    with pytest.raises(ValueError, match="binary"):
        build_split_manifest(np.array([1, 1, 1, 1, 1]), 5, 3, 42)

    # Insufficient rows per class rejection
    with pytest.raises(ValueError, match="at least"):
        build_split_manifest(np.array([0] * 3 + [1] * 10), n_splits=5, n_repeats=3, random_state=42)


def test_build_split_manifest_validates_parameters() -> None:
    y = np.array([0, 1] * 20)

    # n_splits < 2 or non-int
    with pytest.raises(ValueError, match="n_splits"):
        build_split_manifest(y, n_splits=1, n_repeats=3, random_state=42)
    with pytest.raises(ValueError, match="n_splits"):
        build_split_manifest(y, n_splits=5.0, n_repeats=3, random_state=42)  # type: ignore
    with pytest.raises(ValueError, match="n_splits"):
        build_split_manifest(y, n_splits=True, n_repeats=3, random_state=42)  # type: ignore

    # n_repeats < 1 or non-int
    with pytest.raises(ValueError, match="n_repeats"):
        build_split_manifest(y, n_splits=5, n_repeats=0, random_state=42)
    with pytest.raises(ValueError, match="n_repeats"):
        build_split_manifest(y, n_splits=5, n_repeats=3.5, random_state=42)  # type: ignore
    with pytest.raises(ValueError, match="n_repeats"):
        build_split_manifest(y, n_splits=5, n_repeats=False, random_state=42)  # type: ignore

    # random_state non-int
    with pytest.raises(ValueError, match="random_state"):
        build_split_manifest(y, n_splits=5, n_repeats=3, random_state=42.0)  # type: ignore
    with pytest.raises(ValueError, match="random_state"):
        build_split_manifest(y, n_splits=5, n_repeats=3, random_state=True)  # type: ignore


def test_validate_split_manifest_passes_valid_manifest() -> None:
    y = np.array([0, 1] * 50)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=3, random_state=42)
    validate_split_manifest(manifest, n_rows=100, n_splits=5, n_repeats=3, y=y)


def test_validate_split_manifest_rejects_invalid_manifests() -> None:
    y = np.array([0, 1] * 20)
    valid_manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)

    # Missing columns
    with pytest.raises(ValueError, match="columns"):
        bad_manifest = valid_manifest.drop(columns=["repeat"])
        validate_split_manifest(bad_manifest, n_rows=40, n_splits=5, n_repeats=2)

    # Duplicate assignment
    bad_duplicate = valid_manifest.copy()
    row_idx_col = bad_duplicate.columns.get_loc("row_index")
    bad_duplicate.iloc[0, row_idx_col] = bad_duplicate.iloc[1]["row_index"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_split_manifest(bad_duplicate, n_rows=40, n_splits=5, n_repeats=2)

    # Out of bounds / non-contiguous row indices
    bad_indices = valid_manifest.copy()
    bad_indices.loc[bad_indices["row_index"] == 0, "row_index"] = 999
    bad_indices = bad_indices.sort_values(
        ["repeat", "fold", "row_index"], kind="stable"
    ).reset_index(drop=True)
    with pytest.raises(ValueError, match="row index"):
        validate_split_manifest(bad_indices, n_rows=40, n_splits=5, n_repeats=2)

    # Mismatched row count
    with pytest.raises(ValueError, match="row count"):
        validate_split_manifest(valid_manifest, n_rows=50, n_splits=5, n_repeats=2)


def test_validate_split_manifest_rejects_int32_dtypes() -> None:
    y = np.array([0, 1] * 20)
    valid = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    int32_manifest = valid.astype("int32")
    with pytest.raises(ValueError, match="must have int64 dtype"):
        validate_split_manifest(int32_manifest, n_rows=40, n_splits=5, n_repeats=2)


def test_validate_split_manifest_malformed_params_and_y() -> None:
    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)

    # Invalid n_rows
    with pytest.raises(ValueError, match="n_rows"):
        validate_split_manifest(manifest, n_rows=0, n_splits=5, n_repeats=2)
    with pytest.raises(ValueError, match="n_rows"):
        validate_split_manifest(manifest, n_rows=True, n_splits=5, n_repeats=2)  # type: ignore
    with pytest.raises(ValueError, match="n_rows"):
        validate_split_manifest(manifest, n_rows=40.0, n_splits=5, n_repeats=2)  # type: ignore

    # Invalid n_splits
    with pytest.raises(ValueError, match="n_splits"):
        validate_split_manifest(manifest, n_rows=40, n_splits=1, n_repeats=2)
    with pytest.raises(ValueError, match="n_splits"):
        validate_split_manifest(manifest, n_rows=40, n_splits=True, n_repeats=2)  # type: ignore

    # Invalid n_repeats
    with pytest.raises(ValueError, match="n_repeats"):
        validate_split_manifest(manifest, n_rows=40, n_splits=5, n_repeats=0)
    with pytest.raises(ValueError, match="n_repeats"):
        validate_split_manifest(manifest, n_rows=40, n_splits=5, n_repeats=False)  # type: ignore

    # Mismatched y length vs n_rows
    with pytest.raises(ValueError, match="y length"):
        validate_split_manifest(
            manifest, n_rows=40, n_splits=5, n_repeats=2, y=np.array([0, 1] * 10)
        )


def test_validate_split_manifest_unsorted_and_extra_columns() -> None:
    y = np.array([0, 1] * 20)
    valid = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)

    # Extra column
    extra_col = valid.copy()
    extra_col["extra"] = 1
    with pytest.raises(ValueError, match="columns"):
        validate_split_manifest(extra_col, n_rows=40, n_splits=5, n_repeats=2)

    # Wrong column order
    wrong_order = valid[["repeat", "fold", "row_index"]]
    with pytest.raises(ValueError, match="columns"):
        validate_split_manifest(wrong_order, n_rows=40, n_splits=5, n_repeats=2)

    # Unsorted rows
    unsorted_df = valid.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="canonical sorted order"):
        validate_split_manifest(unsorted_df, n_rows=40, n_splits=5, n_repeats=2)


def test_validate_split_manifest_rejects_missing_or_swapped_repeat_fold_pair() -> None:
    y = np.array([0, 1] * 20)
    valid = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)

    # Replace all fold 0 rows in repeat 0 with fold 1
    bad_pairs = valid.copy()
    mask = (bad_pairs["repeat"] == 0) & (bad_pairs["fold"] == 0)
    bad_pairs.loc[mask, "fold"] = 1

    with pytest.raises(ValueError, match="manifest \\(repeat, fold\\) pairs mismatch"):
        validate_split_manifest(bad_pairs, n_rows=40, n_splits=5, n_repeats=2, y=y)


def test_validate_split_manifest_rejects_imbalanced_class_stratification() -> None:
    y = np.array([0, 1] * 20)
    valid = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)

    # Reassign some class 1 samples from fold 0 to fold 1 in repeat 0
    imbalanced = valid.copy()
    # Find two class-1 rows in repeat 0, fold 0
    r0_f0 = imbalanced[(imbalanced["repeat"] == 0) & (imbalanced["fold"] == 0)]
    class1_indices = [idx for idx in r0_f0["row_index"] if y[idx] == 1]

    # Change fold for two class-1 rows from 0 to 1
    imbalanced.loc[
        (imbalanced["repeat"] == 0) & (imbalanced["row_index"].isin(class1_indices[:2])), "fold"
    ] = 1
    imbalanced = imbalanced.sort_values(["repeat", "fold", "row_index"], kind="stable").reset_index(
        drop=True
    )

    with pytest.raises(ValueError, match="not stratified"):
        validate_split_manifest(imbalanced, n_rows=40, n_splits=5, n_repeats=2, y=y)


def test_atomic_write_failure_cleanup_and_preservation(tmp_path: Path) -> None:
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    parquet_path = output_dir / "test.parquet"
    json_path = output_dir / "test.json"

    # Pre-populate original files
    parquet_path.write_bytes(b"original parquet")
    json_path.write_bytes(b"original json")

    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    meta = {"test": 123}

    # Simulate exception during os.replace on the first replace
    with patch("os.replace", side_effect=RuntimeError("Disk write error")):
        with pytest.raises(RuntimeError, match="Disk write error"):
            write_split_artifacts_atomically(parquet_path, json_path, manifest, meta)

    # Verify temp files are cleaned up and original content preserved byte-for-byte
    assert not (output_dir / "test.parquet.tmp").exists()
    assert not (output_dir / "test.json.tmp").exists()
    assert not (output_dir / "test.parquet.bak").exists()
    assert not (output_dir / "test.json.bak").exists()
    assert parquet_path.read_bytes() == b"original parquet"
    assert json_path.read_bytes() == b"original json"


def test_atomic_write_second_replacement_failure_restores_originals(tmp_path: Path) -> None:
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    parquet_path = output_dir / "test.parquet"
    json_path = output_dir / "test.json"

    parquet_path.write_bytes(b"ORIGINAL PARQUET BYTES")
    json_path.write_bytes(b"ORIGINAL JSON BYTES")

    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    meta = {"test": 456}

    orig_replace = os.replace

    def mock_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src).name == "test.json.tmp":
            raise RuntimeError("Second replacement injected failure")
        orig_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError, match="Second replacement injected failure"):
            write_split_artifacts_atomically(parquet_path, json_path, manifest, meta)

    # Verify original files restored byte-for-byte
    assert parquet_path.read_bytes() == b"ORIGINAL PARQUET BYTES"
    assert json_path.read_bytes() == b"ORIGINAL JSON BYTES"
    assert not (output_dir / "test.parquet.tmp").exists()
    assert not (output_dir / "test.json.tmp").exists()
    assert not (output_dir / "test.parquet.bak").exists()
    assert not (output_dir / "test.json.bak").exists()


def test_atomic_write_rollback_parquet_restore_failure_preserves_parquet_bak(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    parquet_path = output_dir / "test.parquet"
    json_path = output_dir / "test.json"

    parquet_path.write_bytes(b"ORIGINAL PARQUET BYTES")
    json_path.write_bytes(b"ORIGINAL JSON BYTES")

    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    meta = {"test": 111}

    orig_replace = os.replace

    def mock_replace(src: str | Path, dst: str | Path) -> None:
        # Trigger error on second replace to enter rollback
        if Path(src).name == "test.json.tmp":
            raise RuntimeError("Second replace failed")
        # In rollback, trigger error when restoring parquet_bak to parquet_path
        if Path(src).name == "test.parquet.bak":
            raise RuntimeError("Parquet rollback restore failed")
        orig_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError, match="Atomic write failed"):
            write_split_artifacts_atomically(parquet_path, json_path, manifest, meta)

    # json_path was restored cleanly
    assert json_path.read_bytes() == b"ORIGINAL JSON BYTES"
    # parquet.bak survived because its restoration failed!
    assert (output_dir / "test.parquet.bak").exists()
    assert (output_dir / "test.parquet.bak").read_bytes() == b"ORIGINAL PARQUET BYTES"
    assert not (output_dir / "test.parquet.tmp").exists()
    assert not (output_dir / "test.json.tmp").exists()


def test_atomic_write_rollback_json_restore_failure_preserves_json_bak(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    parquet_path = output_dir / "test.parquet"
    json_path = output_dir / "test.json"

    parquet_path.write_bytes(b"ORIGINAL PARQUET BYTES")
    json_path.write_bytes(b"ORIGINAL JSON BYTES")

    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    meta = {"test": 222}

    orig_replace = os.replace

    def mock_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src).name == "test.json.tmp":
            raise RuntimeError("Second replace failed")
        if Path(src).name == "test.json.bak":
            raise RuntimeError("JSON rollback restore failed")
        orig_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError, match="Atomic write failed"):
            write_split_artifacts_atomically(parquet_path, json_path, manifest, meta)

    # parquet_path was restored cleanly
    assert parquet_path.read_bytes() == b"ORIGINAL PARQUET BYTES"
    # json.bak survived because its restoration failed!
    assert (output_dir / "test.json.bak").exists()
    assert (output_dir / "test.json.bak").read_bytes() == b"ORIGINAL JSON BYTES"
    assert not (output_dir / "test.parquet.tmp").exists()
    assert not (output_dir / "test.json.tmp").exists()


def test_atomic_write_rollback_both_restore_failures_preserves_both_baks(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    parquet_path = output_dir / "test.parquet"
    json_path = output_dir / "test.json"

    parquet_path.write_bytes(b"ORIGINAL PARQUET BYTES")
    json_path.write_bytes(b"ORIGINAL JSON BYTES")

    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    meta = {"test": 333}

    orig_replace = os.replace

    def mock_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src).name == "test.json.tmp":
            raise RuntimeError("Second replace failed")
        if ".bak" in Path(src).name:
            raise RuntimeError("Rollback restore failed for bak file")
        orig_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError, match="Atomic write failed"):
            write_split_artifacts_atomically(parquet_path, json_path, manifest, meta)

    # Both bak files survived!
    assert (output_dir / "test.parquet.bak").exists()
    assert (output_dir / "test.json.bak").exists()
    assert (output_dir / "test.parquet.bak").read_bytes() == b"ORIGINAL PARQUET BYTES"
    assert (output_dir / "test.json.bak").read_bytes() == b"ORIGINAL JSON BYTES"


def test_atomic_write_new_destination_second_replacement_failure_cleans_first(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    parquet_path = output_dir / "fresh.parquet"
    json_path = output_dir / "fresh.json"

    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    meta = {"test": 789}

    orig_replace = os.replace

    def mock_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src).name == "fresh.json.tmp":
            raise RuntimeError("Fresh second replacement failure")
        orig_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError, match="Fresh second replacement failure"):
            write_split_artifacts_atomically(parquet_path, json_path, manifest, meta)

    # Verify neither destination file remains (no half-pair left behind!)
    assert not parquet_path.exists()
    assert not json_path.exists()
    assert not (output_dir / "fresh.parquet.tmp").exists()
    assert not (output_dir / "fresh.json.tmp").exists()


def test_atomic_write_new_destination_first_replacement_failure_cleans_temps(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    parquet_path = output_dir / "fresh_first.parquet"
    json_path = output_dir / "fresh_first.json"

    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y, n_splits=5, n_repeats=2, random_state=42)
    meta = {"test": 101}

    orig_replace = os.replace

    def mock_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src).name == "fresh_first.parquet.tmp":
            raise RuntimeError("Fresh first replacement failure")
        orig_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError, match="Fresh first replacement failure"):
            write_split_artifacts_atomically(parquet_path, json_path, manifest, meta)

    # Verify neither destination file remains and all temps are cleaned
    assert not parquet_path.exists()
    assert not json_path.exists()
    assert not (output_dir / "fresh_first.parquet.tmp").exists()
    assert not (output_dir / "fresh_first.json.tmp").exists()
    assert not (output_dir / "fresh_first.parquet.bak").exists()
    assert not (output_dir / "fresh_first.json.bak").exists()


def _setup_test_env(tmp_path: Path, y_vals: list) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sample_csv = data_dir / "sample.csv"

    n_samples = len(y_vals)
    train_df = pd.DataFrame(
        {
            "id": range(n_samples),
            "feature": np.random.randn(n_samples),
            "Will_Buy_EV": y_vals,
        }
    )
    test_df = pd.DataFrame({"id": range(n_samples, n_samples + 10), "feature": np.random.randn(10)})
    sample_df = pd.DataFrame({"id": range(n_samples, n_samples + 10), "Will_Buy_EV": [0.5] * 10})

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    sample_df.to_csv(sample_csv, index=False)

    config_dict = {
        "project": {"seed": 42},
        "data": {
            "train": "data/train.csv",
            "test": "data/test.csv",
            "sample_submission": "data/sample.csv",
            "id_column": "id",
            "target_column": "Will_Buy_EV",
        },
        "validation": {
            "strategy": "repeated_stratified_kfold",
            "n_splits": 5,
            "n_repeats": 3,
            "random_state": 42,
        },
        "metrics": {"primary": "roc_auc", "guardrails": ["log_loss"]},
        "outputs": {"root": "outputs"},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config_dict, f)

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "prepare_data.py"
    return config_path, script_path, tmp_path / "outputs"


def test_numeric_target_cli_rejection(tmp_path: Path) -> None:
    # Setup environment with numeric targets (0, 1) - must be rejected
    config_path, script_path, _ = _setup_test_env(tmp_path, y_vals=[0, 1] * 25)

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--project-root",
        str(tmp_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    err_text = res.stderr + res.stdout
    assert "target labels must be exactly" in err_text


def test_prepare_data_script_cli_smoke(tmp_path: Path) -> None:
    config_path, script_path, outputs_dir = _setup_test_env(tmp_path, y_vals=["No", "Yes"] * 25)

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--project-root",
        str(tmp_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = res.stdout

    assert "outputs/splits/repeated_stratified_5x3.parquet" in out
    assert "15-fold" in out or "15 folds" in out
    assert "3 validation assignments" in out or "3 assignments" in out

    parquet_path = outputs_dir / "splits" / "repeated_stratified_5x3.parquet"
    json_path = outputs_dir / "splits" / "repeated_stratified_5x3.json"

    assert parquet_path.exists()
    assert json_path.exists()

    with json_path.open("r") as f:
        meta = json.load(f)
    assert meta["target_mapping"] == {"No": 0, "Yes": 1}
    assert meta["protocol_version"] == 1

    manifest = pd.read_parquet(parquet_path)
    assert len(manifest) == 50 * 3
    assert list(manifest.columns) == ["row_index", "repeat", "fold"]


def test_valid_existing_manifest_reuse(tmp_path: Path) -> None:
    config_path, script_path, outputs_dir = _setup_test_env(tmp_path, y_vals=["No", "Yes"] * 25)

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--project-root",
        str(tmp_path),
    ]

    # First run generates manifest
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    parquet_path = outputs_dir / "splits" / "repeated_stratified_5x3.parquet"
    assert parquet_path.exists()
    mtime_1 = parquet_path.stat().st_mtime_ns

    # Small pause to guarantee timestamp change if file were rewritten
    time.sleep(0.01)

    # Second run should reuse manifest without rewriting
    res2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out2 = res2.stdout

    assert "Reused existing valid split manifest" in out2
    mtime_2 = parquet_path.stat().st_mtime_ns
    assert mtime_1 == mtime_2


def test_invalid_or_stale_existing_artifact_failure(tmp_path: Path) -> None:
    config_path, script_path, outputs_dir = _setup_test_env(tmp_path, y_vals=["No", "Yes"] * 25)

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--project-root",
        str(tmp_path),
    ]

    # Generate initial valid files
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    parquet_path = outputs_dir / "splits" / "repeated_stratified_5x3.parquet"
    json_path = outputs_dir / "splits" / "repeated_stratified_5x3.json"

    # Case A: JSON missing
    json_path.unlink()
    res_a = subprocess.run(cmd, capture_output=True, text=True)
    assert res_a.returncode != 0
    assert "missing" in (res_a.stderr + res_a.stdout).lower()

    # Clean up parquet before regenerating
    parquet_path.unlink()
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Case B: Stale metadata (n_rows mismatch)
    with json_path.open("r") as f:
        meta = json.load(f)
    meta["n_rows"] = 9999
    with json_path.open("w") as f:
        json.dump(meta, f)

    res_b = subprocess.run(cmd, capture_output=True, text=True)
    assert res_b.returncode != 0
    err_b = (res_b.stderr + res_b.stdout).lower()
    assert "incompatible" in err_b or "mismatch" in err_b
