"""Hardening tests for the canonical seed-bagging experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

import ev_purchase.seed_bagging as seed_bagging_module
from ev_purchase.data import DatasetContract
from ev_purchase.seed_bagging import (
    FROZEN_SLICES_SHA256,
    THREAT_MODEL_STATEMENT,
    _authenticate_canonical_component,
    _publish_staged_run,
    _verify_frozen_slices_contract,
    train_seed_bagged_cv,
)


def _minimal_training_inputs() -> tuple[pd.DataFrame, pd.DataFrame, DatasetContract]:
    train_df = pd.DataFrame({"id": [0, 1], "feature": [0.0, 1.0], "Will_Buy_EV": [0, 1]})
    manifest = pd.DataFrame({"row_index": [0, 1], "repeat": [0, 0], "fold": [0, 1]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    return train_df, manifest, contract


@pytest.mark.parametrize("seeds", [(42,), (42, 42, 137)])
def test_training_rejects_non_unique_or_single_seed_sets(seeds: tuple[int, ...]) -> None:
    train_df, manifest, contract = _minimal_training_inputs()

    with pytest.raises(ValueError, match="at least two unique seeds"):
        train_seed_bagged_cv(
            train_df=train_df,
            manifest=manifest,
            contract=contract,
            model_params={},
            model_seeds=seeds,
            folds=[(0, 0)],
        )


def test_full_training_rejects_noncanonical_manifest() -> None:
    train_df, manifest, contract = _minimal_training_inputs()

    with pytest.raises(ValueError, match="canonical 5x3"):
        train_seed_bagged_cv(
            train_df=train_df,
            manifest=manifest,
            contract=contract,
            model_params={},
            model_seeds=(42, 137),
        )


def test_canonical_component_must_match_candidate_metadata(tmp_path: Path) -> None:
    metadata = {
        "candidate_id": "lightgbm_hpo_trial35",
        "run_id": "lightgbm_hpo_trial35",
        "status": "completed",
        "protocol": "full",
        "model_family": "lightgbm",
        "trial_number": 35,
        "feature_policy": "base",
        "effective_n_splits": 5,
        "effective_n_repeats": 3,
        "params": {"n_estimators": 450},
    }
    metadata_path = tmp_path / "candidate_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    component = {
        "candidate_id": "lightgbm_hpo_trial35",
        "model_family": "lightgbm",
        "trial_number": 35,
        "feature_policy": "base",
        "params": {"n_estimators": 450},
    }

    loaded, digest = _authenticate_canonical_component(component, metadata_path)
    assert loaded == metadata
    assert len(digest) == 64

    component["params"] = {"n_estimators": 200}
    with pytest.raises(ValueError, match="params"):
        _authenticate_canonical_component(component, metadata_path)


def test_frozen_slice_contract_pins_file_hash(tmp_path: Path) -> None:
    bad_path = tmp_path / "frozen_slices.json"
    bad_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "slice_provenance": {"status": "post_hoc_discovered_then_prospectively_frozen"},
            }
        ),
        encoding="utf-8",
    )

    assert len(FROZEN_SLICES_SHA256) == 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_frozen_slices_contract(bad_path)


def test_publish_staged_run_replaces_complete_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale", encoding="utf-8")
    staging_dir = tmp_path / ".experiment.staging"
    staging_dir.mkdir()
    (staging_dir / "manifest.json").write_text("complete", encoding="utf-8")

    _publish_staged_run(staging_dir, output_dir)

    assert (output_dir / "manifest.json").read_text(encoding="utf-8") == "complete"
    assert not (output_dir / "stale.txt").exists()
    assert not staging_dir.exists()


def test_publish_staged_run_restores_previous_run_on_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("previous", encoding="utf-8")
    staging_dir = tmp_path / ".experiment.staging"
    staging_dir.mkdir()
    (staging_dir / "manifest.json").write_text("new", encoding="utf-8")
    real_replace = os.replace
    call_count = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(seed_bagging_module.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="simulated publication failure"):
        _publish_staged_run(staging_dir, output_dir)

    assert (output_dir / "sentinel.txt").read_text(encoding="utf-8") == "previous"
    assert not staging_dir.exists()


def test_threat_model_states_self_consistency_boundary() -> None:
    assert "self-consistency" in THREAT_MODEL_STATEMENT.lower()
    assert "coordinated malicious actor" in THREAT_MODEL_STATEMENT.lower()
    assert "power loss" in THREAT_MODEL_STATEMENT.lower()
