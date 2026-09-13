import json
from pathlib import Path
from typing import Any

import pytest

from ev_purchase.analysis import assert_comparable_runs, load_all_runs
from ev_purchase.artifacts import load_run_metrics, save_run_metrics, validate_run_metrics


def _valid_payload() -> dict[str, Any]:
    return {
        "run_id": "exp_001",
        "status": "completed",
        "config_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "model": "catboost",
        "seed": 42,
        "fold_metrics": [
            {"roc_auc": 0.85, "average_precision": 0.80, "log_loss": 0.35},
            {"roc_auc": 0.87, "average_precision": 0.82, "log_loss": 0.33},
        ],
        "aggregate_metrics": {"roc_auc": 0.86, "average_precision": 0.81, "log_loss": 0.34},
        "runtime_seconds": 12.5,
        "artifact_paths": {
            "oof_predictions": "outputs/oof/exp_001.parquet",
            "test_predictions": "outputs/test/exp_001.parquet",
        },
    }


def _strict_parse_constant(constant: str) -> None:
    raise ValueError(f"Strict JSON parsing rejected constant: {constant}")


def test_validate_run_metrics_valid_payload() -> None:
    payload = _valid_payload()
    validate_run_metrics(payload)


def test_save_run_metrics_success(tmp_path: Path) -> None:
    payload = _valid_payload()
    output_file = tmp_path / "metrics.json"

    saved_path = save_run_metrics(payload, output_file)
    assert saved_path.exists()

    loaded = load_run_metrics(output_file)
    assert loaded == payload


def test_save_run_metrics_rejects_missing_keys(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["data_sha256"]
    output_file = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="data_sha256"):
        save_run_metrics(payload, output_file)

    assert not output_file.exists()


def test_save_run_metrics_rejects_invalid_sha256(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["config_sha256"] = "invalid_hash"
    output_file = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="SHA-256|sha256"):
        save_run_metrics(payload, output_file)

    assert not output_file.exists()


def test_save_run_metrics_rejects_invalid_numeric_values(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["runtime_seconds"] = -1.0
    output_file = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="runtime_seconds"):
        save_run_metrics(payload, output_file)

    assert not output_file.exists()


def test_save_run_metrics_rejects_non_integer_seed(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["seed"] = True
    output_file = tmp_path / "metrics.json"

    with pytest.raises((ValueError, TypeError), match="seed"):
        save_run_metrics(payload, output_file)

    assert not output_file.exists()


def test_save_run_metrics_rejects_unsafe_paths(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["artifact_paths"]["oof_predictions"] = "/abs/path/oof.parquet"
    output_file = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="relative|absolute|traversal"):
        save_run_metrics(payload, output_file)

    payload_traversal = _valid_payload()
    payload_traversal["artifact_paths"]["oof_predictions"] = "../outside/oof.parquet"
    with pytest.raises(ValueError, match="relative|absolute|traversal"):
        save_run_metrics(payload_traversal, output_file)

    assert not output_file.exists()


def test_save_run_metrics_rejects_unserializable_value(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["aggregate_metrics"]["custom_obj"] = object()  # type: ignore[assignment]
    output_file = tmp_path / "metrics.json"

    with pytest.raises((ValueError, TypeError)):
        save_run_metrics(payload, output_file)

    assert not output_file.exists()


def test_dict_form_fold_metrics_valid(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["fold_metrics"] = {
        "fold0": {"roc_auc": 0.85, "average_precision": 0.80, "log_loss": 0.35},
        "fold1": {"roc_auc": 0.87, "average_precision": 0.82, "log_loss": 0.33},
    }
    output_file = tmp_path / "metrics.json"

    saved_path = save_run_metrics(payload, output_file)
    assert saved_path.exists()

    loaded = load_run_metrics(output_file)
    assert loaded == payload


def test_dict_form_fold_metrics_rejects_nan_and_inf(tmp_path: Path) -> None:
    payload_nan = _valid_payload()
    payload_nan["fold_metrics"] = {"fold0": {"roc_auc": float("nan")}}
    output_file_nan = tmp_path / "metrics_nan.json"

    with pytest.raises((ValueError, TypeError), match="fold0|finite|nan|JSON"):
        save_run_metrics(payload_nan, output_file_nan)
    assert not output_file_nan.exists()

    payload_inf = _valid_payload()
    payload_inf["fold_metrics"] = {"fold0": {"roc_auc": float("inf")}}
    output_file_inf = tmp_path / "metrics_inf.json"

    with pytest.raises((ValueError, TypeError), match="fold0|finite|inf|JSON"):
        save_run_metrics(payload_inf, output_file_inf)
    assert not output_file_inf.exists()


def test_dict_form_fold_metrics_rejects_non_dict_value(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["fold_metrics"] = {"fold0": "not-a-dict"}
    output_file = tmp_path / "metrics.json"

    with pytest.raises(TypeError, match="fold0"):
        save_run_metrics(payload, output_file)
    assert not output_file.exists()


def test_saved_json_strict_parsing_no_nan_or_infinity(tmp_path: Path) -> None:
    payload = _valid_payload()
    output_file = tmp_path / "metrics.json"
    save_run_metrics(payload, output_file)

    content = output_file.read_text(encoding="utf-8")
    assert "NaN" not in content
    assert "Infinity" not in content

    parsed = json.loads(content, parse_constant=_strict_parse_constant)
    assert parsed == payload


def test_atomic_failure_preserves_byte_identical_destination(tmp_path: Path) -> None:
    output_file = tmp_path / "metrics.json"
    valid_p = _valid_payload()

    save_run_metrics(valid_p, output_file)
    original_bytes = output_file.read_bytes()

    # Try saving dict-form fold metrics with NaN
    invalid_p = _valid_payload()
    invalid_p["fold_metrics"] = {"fold0": {"roc_auc": float("nan")}}

    with pytest.raises((ValueError, TypeError)):
        save_run_metrics(invalid_p, output_file)

    assert output_file.read_bytes() == original_bytes


def test_comparison_rejects_different_split_hashes() -> None:
    runs = [{"split_sha256": "a", "data_sha256": "x"}, {"split_sha256": "b", "data_sha256": "x"}]
    with pytest.raises(ValueError, match="split"):
        assert_comparable_runs(runs)


def test_comparison_rejects_differing_data_hashes() -> None:
    runs = [
        {"split_sha256": "s1", "data_sha256": "d1"},
        {"split_sha256": "s1", "data_sha256": "d2"},
    ]
    with pytest.raises(ValueError, match="data_sha256"):
        assert_comparable_runs(runs)


def test_comparison_rejects_differing_fold_counts() -> None:
    runs = [
        {"split_sha256": "s1", "data_sha256": "d1", "effective_n_splits": 5},
        {"split_sha256": "s1", "data_sha256": "d1", "effective_n_splits": 10},
    ]
    with pytest.raises(ValueError, match="fold count|effective_n_splits"):
        assert_comparable_runs(runs)


def test_comparison_rejects_differing_repeat_counts() -> None:
    runs = [
        {"split_sha256": "s1", "data_sha256": "d1", "effective_n_repeats": 3},
        {"split_sha256": "s1", "data_sha256": "d1", "effective_n_repeats": 1},
    ]
    with pytest.raises(ValueError, match="repeat count|effective_n_repeats"):
        assert_comparable_runs(runs)


def test_comparison_rejects_smoke_vs_full_mixing() -> None:
    runs = [
        {"split_sha256": "s1", "data_sha256": "d1", "smoke": False},
        {"split_sha256": "s1", "data_sha256": "d1", "smoke": True},
    ]
    with pytest.raises(ValueError, match="smoke"):
        assert_comparable_runs(runs)


def test_comparison_rejects_differing_targets() -> None:
    runs = [
        {"split_sha256": "s1", "data_sha256": "d1", "target": "y_binary"},
        {"split_sha256": "s1", "data_sha256": "d1", "target": "y_multi"},
    ]
    with pytest.raises(ValueError, match="target"):
        assert_comparable_runs(runs)


def test_comparison_rejects_differing_metrics_version_or_keys() -> None:
    runs_ver = [
        {"split_sha256": "s1", "data_sha256": "d1", "metrics_version": "v1"},
        {"split_sha256": "s1", "data_sha256": "d1", "metrics_version": "v2"},
    ]
    with pytest.raises(ValueError, match="metrics_version"):
        assert_comparable_runs(runs_ver)

    runs_keys = [
        {"split_sha256": "s1", "data_sha256": "d1", "aggregate_metrics": {"roc_auc": 0.9}},
        {"split_sha256": "s1", "data_sha256": "d1", "aggregate_metrics": {"log_loss": 0.3}},
    ]
    with pytest.raises(ValueError, match="aggregate_metrics"):
        assert_comparable_runs(runs_keys)


def test_comparison_accepts_identical_compatible_runs() -> None:
    run1 = _valid_payload()
    run2 = _valid_payload()
    run2["run_id"] = "exp_002"
    assert_comparable_runs([run1, run2])


def test_comparison_empty_and_single_run_handling() -> None:
    with pytest.raises(ValueError, match="empty"):
        assert_comparable_runs([])

    assert_comparable_runs([_valid_payload()])


def test_load_all_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    assert load_all_runs(runs_dir) == []

    # Create two run subdirectories with valid metrics.json
    run1_dir = runs_dir / "run1"
    run2_dir = runs_dir / "run2"

    p1 = _valid_payload()
    p1["run_id"] = "run1"
    save_run_metrics(p1, run1_dir / "metrics.json")

    p2 = _valid_payload()
    p2["run_id"] = "run2"
    save_run_metrics(p2, run2_dir / "metrics.json")

    loaded = load_all_runs(runs_dir)
    assert len(loaded) == 2
    assert loaded[0]["run_id"] == "run1"
    assert loaded[1]["run_id"] == "run2"
