"""Focused contracts for fail-closed FT-Transformer evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ev_purchase.ft_transformer_evaluation import (
    BLEND_WEIGHTS,
    CANONICAL_DIR,
    DEFAULT_OUTPUT_DIR,
    EXPECTED_FT_ARTIFACT_PATHS,
    EXPECTED_FT_CONFIG_SHA256,
    EXPECTED_FT_FINGERPRINT_SHA256,
    EXPECTED_FT_SOURCE_SHA256,
    EXPECTED_FT_SUPPLEMENTAL_FILES,
    EXPECTED_HASHES,
    FROZEN_SLICES_PATH,
    FT_ARTIFACT_DIR,
    SPLIT_PATH,
    _publish_staged_directory,
    _read_json,
    _verify_source_contracts,
    build_final_test_predictions,
    build_selected_policy,
    calculate_diversity_metrics,
    evaluate_blend_grid,
    make_promotion_decision,
    run_evaluation,
    split_content_sha256,
    validate_canonical_metadata,
    validate_oof_alignment,
    validate_output_directory,
    validate_reported_ft_metrics,
    verify_artifact_inventory,
    verify_ft_provenance,
    verify_run_fingerprint,
)


def _oof(probability: list[float], target: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "repeat": [0, 0, 0, 0],
            "fold": [0, 0, 1, 1],
            "probability": probability,
        }
    )
    if target is not None:
        frame["target"] = target
    return frame


def test_split_content_hash_uses_canonical_little_endian_int64_bytes() -> None:
    manifest = pd.DataFrame(
        {"row_index": [2, 0, 1], "repeat": [0, 0, 0], "fold": [1, 0, 1]}
    ).sort_values(["repeat", "fold", "row_index"], kind="stable")
    expected = hashlib.sha256(
        manifest[["row_index", "repeat", "fold"]]
        .to_numpy(dtype="<i8", copy=True)
        .tobytes(order="C")
    ).hexdigest()

    assert split_content_sha256(manifest) == expected


def test_artifact_inventory_rejects_tampered_file(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("probability\n0.2\n", encoding="utf-8")
    inventory = {
        "status": "completed",
        "artifact_count": 1,
        "artifacts": [
            {
                "path": artifact.name,
                "byte_count": artifact.stat().st_size,
                "row_count": 1,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    inventory_path = tmp_path / "artifact_inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    verify_artifact_inventory(tmp_path, expected_count=1)

    artifact.write_text("probability\n0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifact_inventory(tmp_path, expected_count=1)


def test_run_fingerprint_rejects_mismatch_and_bad_digest() -> None:
    fingerprint = {"experiment_name": "ft", "split_content_sha256": "abc"}
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {"run_fingerprint": fingerprint, "run_fingerprint_sha256": digest}
    state = dict(manifest)
    assert verify_run_fingerprint(manifest, state) == digest

    state["run_fingerprint"] = {**fingerprint, "experiment_name": "other"}
    with pytest.raises(ValueError, match="do not match"):
        verify_run_fingerprint(manifest, state)

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_run_fingerprint({**manifest, "run_fingerprint_sha256": "0" * 64}, manifest)


def test_oof_alignment_rejects_key_and_target_mismatches() -> None:
    target = np.array([0, 1, 0, 1])
    ft = _oof([0.1, 0.8, 0.2, 0.7], target.tolist())
    baseline = _oof([0.2, 0.7, 0.3, 0.6])
    split = baseline[["row_index", "repeat", "fold"]].copy()
    validate_oof_alignment(ft, baseline, split, target, n_repeats=1, n_splits=2)

    bad_keys = baseline.copy()
    bad_keys.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="keys do not align"):
        validate_oof_alignment(ft, bad_keys, split, target, n_repeats=1, n_splits=2)

    bad_target = ft.copy()
    bad_target.loc[0, "target"] = 1
    with pytest.raises(ValueError, match="target does not match"):
        validate_oof_alignment(bad_target, baseline, split, target, n_repeats=1, n_splits=2)


def test_reported_metrics_are_recreated_from_repeat_averaged_oof() -> None:
    target = np.array([0, 1, 0, 1])
    ft = _oof([0.1, 0.8, 0.2, 0.7], target.tolist())
    recreated = validate_reported_ft_metrics(
        target,
        ft,
        {"roc_auc": 1.0, "pr_auc": 1.0, "log_loss": 0.22708064055624455},
    )
    assert recreated["roc_auc"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="reported roc_auc"):
        validate_reported_ft_metrics(
            target,
            ft,
            {"roc_auc": 0.5, "pr_auc": 1.0, "log_loss": 0.22708064055624455},
        )


def test_diversity_metrics_include_prediction_residual_and_hard_error_overlap() -> None:
    target = np.array([0, 1, 0, 1])
    baseline = np.array([0.1, 0.4, 0.8, 0.9])
    ft = np.array([0.2, 0.7, 0.6, 0.3])

    metrics = calculate_diversity_metrics(target, baseline, ft)

    assert set(metrics) >= {
        "prediction_pearson",
        "prediction_spearman",
        "residual_pearson",
        "residual_spearman",
        "hard_error_indicator_correlation",
        "mean_absolute_prediction_difference",
        "threshold_0_5_hard_error_overlap",
        "top_1pct_loss_overlap",
    }
    assert metrics["threshold_0_5_hard_error_overlap"]["intersection_count"] == 1
    assert np.isfinite(metrics["hard_error_indicator_correlation"])


def test_blend_grid_contains_exact_endpoints_and_arithmetic_mean() -> None:
    target = np.array([0, 1, 0, 1])
    baseline = _oof([0.1, 0.8, 0.7, 0.6])
    ft = _oof([0.3, 0.6, 0.1, 0.9], target.tolist())

    grid = evaluate_blend_grid(target, baseline, ft, weights=(0.0, 0.5, 1.0))

    assert tuple(BLEND_WEIGHTS) == (
        0.0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        1.0,
    )
    np.testing.assert_array_equal(grid.predictions[0.0], baseline["probability"])
    np.testing.assert_array_equal(grid.predictions[1.0], ft["probability"])
    np.testing.assert_allclose(
        grid.predictions[0.5],
        (baseline["probability"].to_numpy() + ft["probability"].to_numpy()) / 2,
    )
    assert grid.best_weight in {0.0, 0.5, 1.0}


def test_blend_grid_ties_prefer_smallest_ft_weight() -> None:
    target = np.array([0, 1, 0, 1])
    baseline = _oof([0.1, 0.8, 0.2, 0.7])
    ft = _oof([0.1, 0.8, 0.2, 0.7], target.tolist())

    grid = evaluate_blend_grid(target, baseline, ft, weights=(1.0, 0.5, 0.0))

    assert grid.best_weight == 0.0


def test_oof_alignment_requires_each_physical_row_once_per_repeat() -> None:
    target = np.array([0, 1, 0, 1])
    repeated_keys = pd.DataFrame(
        {
            "row_index": [0, 0, 2, 3],
            "repeat": [0, 0, 0, 0],
            "fold": [0, 1, 0, 1],
            "probability": [0.1, 0.2, 0.3, 0.8],
        }
    )
    ft = repeated_keys.assign(target=[0, 0, 0, 1])
    split = repeated_keys[["row_index", "repeat", "fold"]].copy()

    with pytest.raises(ValueError, match="physical row exactly once"):
        validate_oof_alignment(
            ft,
            repeated_keys,
            split,
            target,
            n_repeats=1,
            n_splits=2,
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "global_auc_delta": 0.00005,
                "repeat_auc_deltas": [0.0, 0.00001, 0.00002],
                "nb_ci_lower": 1e-12,
                "global_log_loss_delta": 0.00020,
                "slice_regressions": [],
            },
            "GO",
        ),
        (
            {
                "global_auc_delta": 0.000049999,
                "repeat_auc_deltas": [0.1, 0.1, 0.1],
                "nb_ci_lower": 0.1,
                "global_log_loss_delta": 0.0,
                "slice_regressions": [],
            },
            "NO-GO",
        ),
        (
            {
                "global_auc_delta": 0.1,
                "repeat_auc_deltas": [0.1, -1e-12, 0.1],
                "nb_ci_lower": 0.1,
                "global_log_loss_delta": 0.0,
                "slice_regressions": [],
            },
            "NO-GO",
        ),
        (
            {
                "global_auc_delta": 0.1,
                "repeat_auc_deltas": [0.1, 0.1, 0.1],
                "nb_ci_lower": 0.0,
                "global_log_loss_delta": 0.0,
                "slice_regressions": [],
            },
            "NO-GO",
        ),
        (
            {
                "global_auc_delta": 0.1,
                "repeat_auc_deltas": [0.1, 0.1, 0.1],
                "nb_ci_lower": 0.1,
                "global_log_loss_delta": 0.000200001,
                "slice_regressions": [],
            },
            "NO-GO",
        ),
    ],
)
def test_promotion_gate_boundaries(kwargs: dict, expected: str) -> None:
    assert make_promotion_decision(**kwargs)["decision"] == expected


def test_final_policy_keeps_canonical_values_on_failed_gate() -> None:
    canonical = pd.DataFrame({"id": [10, 11], "Will_Buy_EV": [0.123, 0.987]})
    ft = pd.DataFrame({"id": [10, 11], "Will_Buy_EV": [0.9, 0.1]})

    final = build_final_test_predictions(canonical, ft, weight=0.5, promoted=False)
    pd.testing.assert_frame_equal(final, canonical, check_exact=True)

    blended = build_final_test_predictions(canonical, ft, weight=0.5, promoted=True)
    np.testing.assert_allclose(blended["Will_Buy_EV"], [0.5115, 0.5435])


def test_failed_gate_selects_canonical_policy_without_test_blend() -> None:
    policy = build_selected_policy(analyzed_best_ft_weight=0.1, promoted=False)

    assert policy == {
        "scope": "oof_only",
        "policy_type": "single_model",
        "candidate_id": "lightgbm_hpo_trial35",
        "ft_weight": 0.0,
        "lightgbm_weight": 1.0,
        "analyzed_best_ft_weight": 0.1,
        "promoted": False,
        "test_blend_produced": False,
    }


def test_canonical_metadata_requires_exact_full_5x3_contract() -> None:
    metadata = _read_json(CANONICAL_DIR / "candidate_metadata.json")
    validate_canonical_metadata(metadata)

    tampered = copy.deepcopy(metadata)
    tampered["effective_n_repeats"] = 1
    with pytest.raises(ValueError, match="metadata protocol mismatch"):
        validate_canonical_metadata(tampered)


def test_artifact_inventory_rejects_wrong_inventory_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("probability\n0.2\n", encoding="utf-8")
    inventory = {
        "status": "completed",
        "artifact_count": 1,
        "artifacts": [
            {
                "path": artifact.name,
                "byte_count": artifact.stat().st_size,
                "row_count": 1,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    inventory_path = tmp_path / "artifact_inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="FT artifact inventory SHA-256 mismatch"):
        verify_artifact_inventory(
            tmp_path,
            expected_count=1,
            expected_inventory_sha256="0" * 64,
        )


def test_artifact_inventory_verifies_custom_path_set_and_rejects_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("probability\n0.2\n", encoding="utf-8")
    inventory = {
        "status": "completed",
        "artifact_count": 1,
        "artifacts": [
            {
                "path": artifact.name,
                "byte_count": artifact.stat().st_size,
                "row_count": 1,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    inventory_path = tmp_path / "artifact_inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    result = verify_artifact_inventory(
        tmp_path,
        expected_count=1,
        expected_paths={"predictions.csv"},
    )
    assert result["status"] == "verified"

    with pytest.raises(ValueError, match="FT artifact path set mismatch"):
        verify_artifact_inventory(
            tmp_path,
            expected_count=1,
            expected_paths={"other_predictions.csv"},
        )


def test_artifact_inventory_rejects_prohibited_development_path(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    artifact = runs_dir / "kaggle_oof.csv"
    artifact.write_text("probability\n0.2\n", encoding="utf-8")
    inventory = {
        "status": "completed",
        "artifact_count": 1,
        "artifacts": [
            {
                "path": "runs/kaggle_oof.csv",
                "byte_count": artifact.stat().st_size,
                "row_count": 1,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "artifact_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="prohibited artifact path"):
        verify_artifact_inventory(tmp_path, expected_count=1)


def test_artifact_inventory_rejects_undeclared_physical_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("probability\n0.2\n", encoding="utf-8")
    inventory = {
        "status": "completed",
        "artifact_count": 1,
        "artifacts": [
            {
                "path": artifact.name,
                "byte_count": artifact.stat().st_size,
                "row_count": 1,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "artifact_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    undeclared = tmp_path / "development" / "kaggle_oof.csv"
    undeclared.parent.mkdir()
    undeclared.write_text("probability\n0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="undeclared physical artifacts"):
        verify_artifact_inventory(tmp_path, expected_count=1)


def test_publish_cleanup_failure_preserves_commit_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "value.txt").write_text("old", encoding="utf-8")
    staging_dir = tmp_path / "staging-first"
    staging_dir.mkdir()
    (staging_dir / "value.txt").write_text("first", encoding="utf-8")

    original_rmtree = __import__("shutil").rmtree
    failed_once = False

    def fail_first_backup_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal failed_once
        if Path(path).name == ".evaluation.backup" and not failed_once:
            failed_once = True
            raise OSError("injected backup cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("ev_purchase.ft_transformer_evaluation.DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        "ev_purchase.ft_transformer_evaluation.shutil.rmtree", fail_first_backup_cleanup
    )

    with pytest.warns(RuntimeWarning, match="published successfully"):
        _publish_staged_directory(staging_dir, output_dir)
    assert (output_dir / "value.txt").read_text(encoding="utf-8") == "first"

    second_staging_dir = tmp_path / "staging-second"
    second_staging_dir.mkdir()
    (second_staging_dir / "value.txt").write_text("second", encoding="utf-8")
    _publish_staged_directory(second_staging_dir, output_dir)

    assert (output_dir / "value.txt").read_text(encoding="utf-8") == "second"
    assert not (tmp_path / ".evaluation.backup").exists()


def test_verify_ft_provenance_authenticates_production_artifacts() -> None:
    manifest = _read_json(FT_ARTIFACT_DIR / "training_manifest.json")
    state = _read_json(FT_ARTIFACT_DIR / "run_state.json")
    digest = verify_ft_provenance(
        manifest,
        state,
        expected_config_sha256=EXPECTED_FT_CONFIG_SHA256,
        expected_source_sha256=EXPECTED_FT_SOURCE_SHA256,
        expected_fingerprint_sha256=EXPECTED_FT_FINGERPRINT_SHA256,
    )
    assert digest == EXPECTED_FT_FINGERPRINT_SHA256


@pytest.mark.parametrize(
    ("mutator", "match_message"),
    [
        (
            lambda m, s: m.__setitem__("config_sha256", "0" * 64),
            "Training manifest config_sha256 mismatch",
        ),
        (
            lambda m, s: m["config"].__setitem__("epochs", 999),
            "Training manifest config canonical digest mismatch",
        ),
        (
            lambda m, s: m.__setitem__("source_sha256", "0" * 64),
            "Training manifest source_sha256 mismatch",
        ),
        (
            lambda m, s: m["run_fingerprint"].__setitem__("config_sha256", "0" * 64),
            "Training manifest fingerprint config_sha256 mismatch",
        ),
        (
            lambda m, s: m["run_fingerprint"].__setitem__("full_notebook_source_sha256", "0" * 64),
            "Training manifest fingerprint source mismatch",
        ),
        (
            lambda m, s: s["run_fingerprint"].__setitem__("full_notebook_source_sha256", "0" * 64),
            "Run state fingerprint source mismatch",
        ),
        (
            lambda m, s: m["run_fingerprint"].__setitem__("experiment_name", "other_ft"),
            "Training manifest and run state fingerprints do not match",
        ),
        (
            lambda m, s: m.__setitem__("run_fingerprint_sha256", "0" * 64),
            "Training manifest run_fingerprint_sha256 mismatch",
        ),
        (
            lambda m, s: s.__setitem__("run_fingerprint_sha256", "0" * 64),
            "Run state run_fingerprint_sha256 mismatch",
        ),
        (
            lambda m, s: m.__setitem__("split_content_sha256", "0" * 64),
            "Training manifest split_content_sha256 mismatch",
        ),
    ],
)
def test_verify_ft_provenance_rejects_individual_field_mismatches(
    mutator: Any, match_message: str
) -> None:
    manifest = _read_json(FT_ARTIFACT_DIR / "training_manifest.json")
    state = _read_json(FT_ARTIFACT_DIR / "run_state.json")
    mutator(manifest, state)
    with pytest.raises(ValueError, match=match_message):
        verify_ft_provenance(manifest, state)


def test_verify_ft_provenance_rejects_mutable_self_consistency_bypass() -> None:
    manifest = copy.deepcopy(_read_json(FT_ARTIFACT_DIR / "training_manifest.json"))
    state = copy.deepcopy(_read_json(FT_ARTIFACT_DIR / "run_state.json"))

    manifest["config"]["epochs"] = 999
    altered_digest = hashlib.sha256(
        json.dumps(manifest["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["config_sha256"] = altered_digest

    with pytest.raises(ValueError, match="Training manifest config_sha256 mismatch"):
        verify_ft_provenance(manifest, state)


def test_validate_output_directory_enforces_approved_path(tmp_path: Path) -> None:
    assert validate_output_directory(DEFAULT_OUTPUT_DIR) == DEFAULT_OUTPUT_DIR.resolve()

    with pytest.raises(ValueError, match="approved path"):
        validate_output_directory(tmp_path)

    with pytest.raises(ValueError, match="approved path"):
        run_evaluation(output_dir=tmp_path)

    with pytest.raises(ValueError, match="approved path"):
        _publish_staged_directory(tmp_path, tmp_path)


def test_production_inventory_verifies_with_pinned_paths_and_digest() -> None:
    res = verify_artifact_inventory(
        FT_ARTIFACT_DIR,
        expected_count=len(EXPECTED_FT_ARTIFACT_PATHS),
        expected_paths=EXPECTED_FT_ARTIFACT_PATHS,
        expected_inventory_sha256=EXPECTED_HASHES["ft_inventory"],
        approved_supplemental_files=EXPECTED_FT_SUPPLEMENTAL_FILES,
    )
    assert res["status"] == "verified"
    assert res["inventory_sha256"] == EXPECTED_HASHES["ft_inventory"]

    with pytest.raises(ValueError, match="FT artifact inventory SHA-256 mismatch"):
        verify_artifact_inventory(
            FT_ARTIFACT_DIR,
            expected_count=len(EXPECTED_FT_ARTIFACT_PATHS),
            expected_paths=EXPECTED_FT_ARTIFACT_PATHS,
            expected_inventory_sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="FT artifact path set mismatch"):
        verify_artifact_inventory(
            FT_ARTIFACT_DIR,
            expected_count=len(EXPECTED_FT_ARTIFACT_PATHS),
            expected_paths={"bogus_artifact.csv"},
        )


def test_production_source_contracts_verify_cleanly() -> None:
    integrity, split = _verify_source_contracts(
        ft_dir=FT_ARTIFACT_DIR,
        canonical_dir=CANONICAL_DIR,
        split_path=SPLIT_PATH,
        frozen_slices_path=FROZEN_SLICES_PATH,
    )
    assert integrity["ft_inventory"]["status"] == "verified"
    assert integrity["ft_inventory"]["artifact_count"] == len(EXPECTED_FT_ARTIFACT_PATHS)
    assert integrity["run_fingerprint_sha256"] == EXPECTED_FT_FINGERPRINT_SHA256
    assert not split.empty
