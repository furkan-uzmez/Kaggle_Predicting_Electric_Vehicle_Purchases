import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from ev_purchase.splits import build_split_manifest
from ev_purchase.submission import (
    build_submission,
    generate_final_model_and_submission,
    load_final_model,
    predict_final_model,
    validate_final_config_and_provenance,
)


def test_submission_preserves_sample_id_order() -> None:
    sample = pd.DataFrame({"id": [9, 4], "Will_Buy_EV": [0.0, 0.0]})
    result = build_submission(sample, np.array([0.8, 0.2]), "id", "Will_Buy_EV")
    assert result["id"].tolist() == [9, 4]
    assert result.columns.tolist() == ["id", "Will_Buy_EV"]


def test_submission_rejects_reversed_column_order() -> None:
    sample = pd.DataFrame({"Will_Buy_EV": [0.0, 0.0], "id": [9, 4]})
    with pytest.raises(ValueError, match="order"):
        build_submission(sample, np.array([0.8, 0.2]), "id", "Will_Buy_EV")


def test_submission_rejects_non_finite_probabilities() -> None:
    sample = pd.DataFrame({"id": [1], "Will_Buy_EV": [0.0]})
    with pytest.raises(ValueError, match="finite"):
        build_submission(sample, np.array([np.nan]), "id", "Will_Buy_EV")

    with pytest.raises(ValueError, match="finite"):
        build_submission(sample, np.array([np.inf]), "id", "Will_Buy_EV")

    with pytest.raises(ValueError, match="finite"):
        build_submission(sample, np.array([-np.inf]), "id", "Will_Buy_EV")


def test_submission_rejects_non_1d_probabilities() -> None:
    sample = pd.DataFrame({"id": [1], "Will_Buy_EV": [0.0]})
    with pytest.raises(ValueError, match="1D"):
        build_submission(sample, np.array([[0.5]]), "id", "Will_Buy_EV")


def test_submission_rejects_length_mismatch() -> None:
    sample = pd.DataFrame({"id": [1, 2], "Will_Buy_EV": [0.0, 0.0]})
    with pytest.raises(ValueError, match="length"):
        build_submission(sample, np.array([0.5]), "id", "Will_Buy_EV")


def test_submission_rejects_out_of_range_probabilities() -> None:
    sample = pd.DataFrame({"id": [1], "Will_Buy_EV": [0.0]})
    with pytest.raises(ValueError, match="range"):
        build_submission(sample, np.array([1.5]), "id", "Will_Buy_EV")

    with pytest.raises(ValueError, match="range"):
        build_submission(sample, np.array([-0.1]), "id", "Will_Buy_EV")


def test_submission_rejects_duplicate_ids() -> None:
    sample = pd.DataFrame({"id": [1, 1], "Will_Buy_EV": [0.0, 0.0]})
    with pytest.raises(ValueError, match="duplicate"):
        build_submission(sample, np.array([0.5, 0.6]), "id", "Will_Buy_EV")


def test_submission_rejects_null_ids() -> None:
    sample = pd.DataFrame({"id": [1, None], "Will_Buy_EV": [0.0, 0.0]})
    with pytest.raises(ValueError, match="null|missing"):
        build_submission(sample, np.array([0.5, 0.6]), "id", "Will_Buy_EV")


def test_submission_rejects_unexpected_columns() -> None:
    sample = pd.DataFrame({"id": [1], "Will_Buy_EV": [0.0], "extra": [123]})
    with pytest.raises(ValueError, match="column"):
        build_submission(sample, np.array([0.5]), "id", "Will_Buy_EV")


def _setup_mini_project(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sample_csv = data_dir / "sample_submission.csv"

    np.random.seed(42)
    n_train = 40
    n_test = 15

    train_ids = list(range(1, 1 + n_train))
    test_ids = list(range(100, 100 + n_test))

    train_df = pd.DataFrame(
        {
            "id": train_ids,
            "num1": np.random.randn(n_train),
            "cat1": np.random.choice(["X", "Y"], size=n_train),
            "Will_Buy_EV": np.random.choice(["No", "Yes"], size=n_train),
        }
    )

    test_df = pd.DataFrame(
        {
            "id": test_ids,
            "num1": np.random.randn(n_test),
            "cat1": np.random.choice(["X", "Y"], size=n_test),
        }
    )

    sample_df = pd.DataFrame(
        {
            "id": test_ids,
            "Will_Buy_EV": [0.5] * n_test,
        }
    )

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    sample_df.to_csv(sample_csv, index=False)

    hasher = hashlib.sha256()
    for p in sorted([train_csv, test_csv]):
        with open(p, "rb") as f:
            hasher.update(f.read())
    data_sha256 = hasher.hexdigest()

    y_encoded = (train_df["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    split_manifest = build_split_manifest(y=y_encoded, n_splits=2, n_repeats=1, random_state=42)

    splits_dir = tmp_path / "outputs" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_parquet = splits_dir / "repeated_stratified_5x3.parquet"
    split_manifest.to_parquet(manifest_parquet)

    split_hasher = hashlib.sha256()
    with open(manifest_parquet, "rb") as f:
        split_hasher.update(f.read())
    split_sha256 = split_hasher.hexdigest()

    # Candidate metadata setup with artifacts
    meta_dir = tmp_path / "outputs" / "ensemble" / "candidates" / "lightgbm_hpo_trial35"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cand_meta_path = meta_dir / "candidate_metadata.json"

    artifact_dir = tmp_path / "outputs" / "ensemble" / "artifacts" / "lightgbm_hpo_trial35"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    oof_p = artifact_dir / "oof_predictions.parquet"
    test_p = artifact_dir / "test_predictions.parquet"

    pd.DataFrame({"row_index": range(n_train), "probability": [0.5] * n_train}).to_parquet(oof_p)
    pd.DataFrame({"id": test_ids, "Will_Buy_EV": [0.5] * n_test}).to_parquet(test_p)

    oof_sha = hashlib.sha256(oof_p.read_bytes()).hexdigest()
    test_sha = hashlib.sha256(test_p.read_bytes()).hexdigest()

    rel_oof = str(oof_p.relative_to(tmp_path))
    rel_test = str(test_p.relative_to(tmp_path))
    rel_cand_meta = str(cand_meta_path.relative_to(tmp_path))

    cand_meta_dict = {
        "run_id": "lightgbm_hpo_trial35",
        "candidate_id": "lightgbm_hpo_trial35",
        "status": "completed",
        "protocol": "full",
        "effective_n_splits": 2,
        "effective_n_repeats": 1,
        "effective_n_rows": n_train,
        "evaluated_fold_ids": [0, 1],
        "model_family": "lightgbm",
        "tuning_study": "lightgbm_hpo",
        "trial_number": 35,
        "params": {"n_estimators": 5, "num_leaves": 4},
        "seed_policy": {"base_seed": 42, "formula": "base_seed + repeat * 100 + fold"},
        "feature_policy": "base",
        "dry_run": False,
        "provenance": {"data_sha256": data_sha256, "split_sha256": split_sha256},
        "artifact_paths": {
            "oof_predictions": rel_oof,
            "test_predictions": rel_test,
        },
        "artifact_checksums": {
            "oof_predictions": oof_sha,
            "test_predictions": test_sha,
        },
    }
    with open(cand_meta_path, "w") as f:
        json.dump(cand_meta_dict, f, indent=2)

    config_dict = {
        "project": {"name": "test_proj", "seed": 42},
        "data": {
            "train": str(train_csv.relative_to(tmp_path)),
            "test": str(test_csv.relative_to(tmp_path)),
            "sample_submission": str(sample_csv.relative_to(tmp_path)),
            "id_column": "id",
            "target_column": "Will_Buy_EV",
        },
        "validation": {
            "strategy": "repeated_stratified_kfold",
            "n_splits": 2,
            "n_repeats": 1,
            "random_state": 42,
        },
        "metrics": {"primary": "roc_auc", "guardrails": ["log_loss"]},
        "outputs": {"root": "outputs"},
        "feature_policy": "base",
        "policy": {
            "type": "single_model",
            "status": "real_materialization_completed",
            "best_single_candidate_id": "lightgbm_hpo_trial35",
            "source_run_ids": ["lightgbm_hpo_trial35"],
            "analyzed_candidate_ids": ["lightgbm_hpo_trial35"],
            "winning_weights": {"lightgbm_hpo_trial35": 1.0},
            "data_sha256": data_sha256,
            "split_sha256": split_sha256,
            "components": {
                "lightgbm_hpo_trial35": {
                    "run_id": "lightgbm_hpo_trial35",
                    "candidate_id": "lightgbm_hpo_trial35",
                    "candidate_metadata_path": rel_cand_meta,
                    "model_family": "lightgbm",
                    "tuning_study": "lightgbm_hpo",
                    "trial_number": 35,
                    "params": {"n_estimators": 5, "num_leaves": 4},
                    "seed_policy": {"base_seed": 42, "formula": "base_seed + repeat * 100 + fold"},
                    "feature_policy": "base",
                    "weight": 1.0,
                    "provenance": {"data_sha256": data_sha256, "split_sha256": split_sha256},
                }
            },
        },
    }

    # Add pyproject.toml as project marker
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
    cfg_file = tmp_path / "configs" / "final.yaml"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)

    return cfg_file, train_df, test_df, sample_df


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _write_yaml(path, payload) -> None:
    with open(path, "w") as f:
        yaml.dump(payload, f)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _mini_candidate_metadata_path(tmp_path):
    return (
        tmp_path
        / "outputs"
        / "ensemble"
        / "candidates"
        / "lightgbm_hpo_trial35"
        / "candidate_metadata.json"
    )


def test_actual_final_yaml_fails_closed_on_stale_declared_artifact_path(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)
    cand_meta["artifact_paths"]["oof_predictions"] = (
        "outputs/ensemble/staging/nonexistent_oof.parquet"
    )
    _write_json(meta_p, cand_meta)

    with pytest.raises(FileNotFoundError, match="Candidate artifact file not found"):
        validate_final_config_and_provenance(cfg_file, project_root=tmp_path)


def test_native_model_reload_parity_and_integrity(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)

    res = generate_final_model_and_submission(
        config_path=cfg_file,
        project_root=tmp_path,
    )

    model_dir = res["final_model_dir"]
    sub_path = res["submission_path"]

    assert (model_dir / "model.txt").exists()
    assert (model_dir / "prep_state.json").exists()
    assert (model_dir / "manifest.json").exists()
    assert not (model_dir / "model.joblib").exists()

    # Load reconstructed model
    model_data = load_final_model(model_dir)
    assert model_data["booster"] is not None

    X_test_raw = test_df.drop(columns=["id"])
    reloaded_preds = predict_final_model(model_dir, X_test_raw)

    sub_df = pd.read_csv(sub_path)
    in_mem_preds = sub_df["Will_Buy_EV"].to_numpy()
    np.testing.assert_allclose(reloaded_preds, in_mem_preds, atol=1e-5)

    # Loader Integrity Tests:
    # 1. Tampered artifact checksum
    model_txt_p = model_dir / "model.txt"
    original_text = model_txt_p.read_text()
    model_txt_p.write_text(original_text + "\n# tampered")
    with pytest.raises(ValueError, match="SHA256 checksum mismatch"):
        load_final_model(model_dir)
    model_txt_p.write_text(original_text)

    # 2. Exact Dtype mismatch: float64 vs int64
    X_int_dtype = X_test_raw.copy()
    X_int_dtype["num1"] = X_int_dtype["num1"].astype("int64")
    with pytest.raises(ValueError, match="Feature dtype mismatch"):
        predict_final_model(model_dir, X_int_dtype)

    # 3. Column reordering
    X_reordered = X_test_raw[list(reversed(X_test_raw.columns))]
    with pytest.raises(ValueError, match="Column mismatch"):
        predict_final_model(model_dir, X_reordered)

    # 4. Missing column
    X_missing = X_test_raw.drop(columns=[X_test_raw.columns[0]])
    with pytest.raises(ValueError, match="Column mismatch"):
        predict_final_model(model_dir, X_missing)

    # 5. Extra column
    X_extra = X_test_raw.copy()
    X_extra["extra_col"] = 1.0
    with pytest.raises(ValueError, match="Column mismatch"):
        predict_final_model(model_dir, X_extra)


def test_strict_manifest_schema(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    res = generate_final_model_and_submission(config_path=cfg_file, project_root=tmp_path)
    model_dir = res["final_model_dir"]
    manifest_p = model_dir / "manifest.json"

    with open(manifest_p) as f:
        manifest = json.load(f)

    # Test missing required manifest key
    m_bad = manifest.copy()
    del m_bad["feature_dtypes"]
    with open(manifest_p, "w") as f:
        json.dump(m_bad, f)
    with pytest.raises(ValueError, match="Manifest missing required keys"):
        load_final_model(model_dir)

    # Test non-hex artifact hash
    m_bad_hash = manifest.copy()
    m_bad_hash["artifact_hashes"]["model.txt"] = "NOT_HEX"
    with open(manifest_p, "w") as f:
        json.dump(m_bad_hash, f)
    with pytest.raises(ValueError, match="lowercase hex"):
        load_final_model(model_dir)

    # Restore manifest
    with open(manifest_p, "w") as f:
        json.dump(manifest, f)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda m: m.update({"feature_columns": ["num1", "num1"]}), "feature_columns"),
        (lambda m: m.update({"feature_columns": ["num1", ""]}), "feature_columns"),
        (lambda m: m.update({"feature_dtypes": {"num1": "float64"}}), "feature_dtypes keys"),
        (
            lambda m: m.update({"feature_dtypes": {**m["feature_dtypes"], "cat1": "not_a_dtype"}}),
            "valid non-empty pandas dtype",
        ),
        (lambda m: m.update({"policy": []}), "Manifest policy"),
        (lambda m: m.update({"model": {**m["model"], "family": "catboost"}}), "model.family"),
        (
            lambda m: m.update({"provenance": {**m["provenance"], "data_sha256": "bad"}}),
            "provenance.data_sha256|provenance data_sha256",
        ),
    ],
)
def test_strict_manifest_rejects_invalid_native_manifest_blocks(tmp_path, mutator, match) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    res = generate_final_model_and_submission(config_path=cfg_file, project_root=tmp_path)
    model_dir = res["final_model_dir"]
    manifest_p = model_dir / "manifest.json"
    manifest = _load_json(manifest_p)

    mutator(manifest)
    _write_json(manifest_p, manifest)

    with pytest.raises(ValueError, match=match):
        load_final_model(model_dir)


@pytest.mark.parametrize(
    ("prep_mutator", "match"),
    [
        (lambda p: p.update({"num_cols": ["num1", "num1"]}), "num_cols"),
        (lambda p: p.update({"cat_cols": ["cat1", "num1"]}), "disjoint"),
        (lambda p: p.update({"num_cols": []}), "partition"),
        (lambda p: p.update({"categories": {}}), "categories keys"),
        (lambda p: p.update({"categories": {"cat1": ["X", "X"]}}), "duplicates"),
        (lambda p: p.update({"categories": {"cat1": [["X"]]}}), "scalar"),
    ],
)
def test_strict_manifest_rejects_invalid_native_prep_state(tmp_path, prep_mutator, match) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    res = generate_final_model_and_submission(config_path=cfg_file, project_root=tmp_path)
    model_dir = res["final_model_dir"]
    prep_p = model_dir / "prep_state.json"
    manifest_p = model_dir / "manifest.json"
    prep_state = _load_json(prep_p)
    manifest = _load_json(manifest_p)

    prep_mutator(prep_state)
    _write_json(prep_p, prep_state)
    manifest["artifact_hashes"]["prep_state.json"] = hashlib.sha256(prep_p.read_bytes()).hexdigest()
    _write_json(manifest_p, manifest)

    with pytest.raises(ValueError, match=match):
        load_final_model(model_dir)


def test_transactional_rollback_on_second_promotion_failure(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)

    out_dir = tmp_path / "outputs"
    prior_sub_dir = out_dir / "submissions"
    prior_model_dir = out_dir / "final_model"
    prior_sub_dir.mkdir(parents=True)
    prior_model_dir.mkdir(parents=True)

    sentinel_sub = prior_sub_dir / "sentinel.txt"
    sentinel_model = prior_model_dir / "sentinel.txt"
    sentinel_sub.write_text("prior_sub")
    sentinel_model.write_text("prior_model")

    # Hook to trigger failure specifically between first promotion and second promotion
    def failing_hook():
        raise RuntimeError("Injected promotion failure")

    with pytest.raises(RuntimeError, match="Injected promotion failure"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
            _test_promotion_failure_hook=failing_hook,
        )

    # Verify prior state was restored exactly for BOTH directories
    assert sentinel_sub.exists()
    assert sentinel_sub.read_text() == "prior_sub"
    assert sentinel_model.exists()
    assert sentinel_model.read_text() == "prior_model"


def test_backup_cleanup_failure_does_not_rollback_commit(tmp_path, monkeypatch) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)

    out_dir = tmp_path / "outputs"
    prior_sub_dir = out_dir / "submissions"
    prior_model_dir = out_dir / "final_model"
    prior_sub_dir.mkdir(parents=True)
    prior_model_dir.mkdir(parents=True)
    (prior_sub_dir / "old.txt").write_text("old")
    (prior_model_dir / "old.txt").write_text("old")

    original_rmtree = shutil.rmtree

    def failing_rmtree(path, *args, **kwargs):
        if ".backup_" in str(path):
            raise OSError("Injected rmtree backup cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", failing_rmtree)

    # Must complete and return success even if backup cleanup fails post-commit!
    res = generate_final_model_and_submission(
        config_path=cfg_file,
        project_root=tmp_path,
    )

    assert res["submission_path"].exists()
    assert res["final_model_dir"].exists()


def test_symlink_defense_rejects_symlinked_targets(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)

    out_dir = tmp_path / "outputs"
    symlink_sub = out_dir / "submissions"
    target_dir = tmp_path / "target_dir"
    target_dir.mkdir(parents=True)
    symlink_sub.symlink_to(target_dir)

    with pytest.raises(ValueError, match="symlink detected"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_candidate_metadata_required_fail_closed(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)

    # Delete candidate metadata file
    cand_path = tmp_path / "outputs" / "ensemble" / "candidates"
    meta_p = cand_path / "lightgbm_hpo_trial35" / "candidate_metadata.json"
    meta_p.unlink()

    with pytest.raises(FileNotFoundError, match="Candidate metadata"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_candidate_artifact_declared_path_required_no_metadata_dir_fallback(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)

    fallback_p = meta_p.parent / "oof_predictions.parquet"
    fallback_p.write_bytes(b"stale artifact that old fallback would have accepted by path")
    cand_meta["artifact_paths"]["oof_predictions"] = (
        "outputs/ensemble/artifacts/missing/oof.parquet"
    )
    cand_meta["artifact_checksums"]["oof_predictions"] = hashlib.sha256(
        fallback_p.read_bytes()
    ).hexdigest()
    _write_json(meta_p, cand_meta)

    with pytest.raises(FileNotFoundError, match="Candidate artifact file not found"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_candidate_artifact_checksums_fail_closed(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)

    cand_path = tmp_path / "outputs" / "ensemble" / "artifacts" / "lightgbm_hpo_trial35"
    oof_p = cand_path / "oof_predictions.parquet"
    oof_p.write_bytes(b"tampered_oof")

    with pytest.raises(ValueError, match="Candidate artifact checksum mismatch"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_source_run_ids_bound_to_component_run_ids_not_weight_keys(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    cfg = _load_yaml(cfg_file)
    component = cfg["policy"]["components"]["lightgbm_hpo_trial35"]
    component["run_id"] = "actual_component_run_001"
    _write_yaml(cfg_file, cfg)

    with pytest.raises(ValueError, match="source_run_ids .* active component run_ids"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_candidate_metadata_run_id_required_and_must_equal_component_run_id(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    cfg = _load_yaml(cfg_file)
    cfg["policy"]["source_run_ids"] = ["actual_component_run_001"]
    cfg["policy"]["components"]["lightgbm_hpo_trial35"]["run_id"] = "actual_component_run_001"
    _write_yaml(cfg_file, cfg)

    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)
    cand_meta["run_id"] = "different_metadata_run"
    _write_json(meta_p, cand_meta)

    with pytest.raises(ValueError, match="Candidate metadata run_id mismatch"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_candidate_metadata_provenance_required_and_matches_all_hashes(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)
    del cand_meta["provenance"]
    _write_json(meta_p, cand_meta)

    with pytest.raises(ValueError, match="Candidate metadata provenance"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_candidate_metadata_provenance_checked_against_actual_hashes(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    cfg = _load_yaml(cfg_file)
    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)

    bogus_sha = "0" * 64
    cfg["policy"]["data_sha256"] = bogus_sha
    cfg["policy"]["components"]["lightgbm_hpo_trial35"]["provenance"]["data_sha256"] = bogus_sha
    cand_meta["provenance"]["data_sha256"] = bogus_sha
    _write_yaml(cfg_file, cfg)
    _write_json(meta_p, cand_meta)

    with pytest.raises(ValueError, match="Data SHA256 mismatch"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_unsafe_output_root_policy_rejection(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)

    # Attempt output_root outside project root
    with pytest.raises(ValueError, match="escapes project root"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
            output_root="/tmp",
        )


def test_candidate_artifact_absolute_declared_path_raises(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)
    cand_meta["artifact_paths"]["oof_predictions"] = "/tmp/abs_oof.parquet"
    _write_json(meta_p, cand_meta)

    with pytest.raises(ValueError, match="relative"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_candidate_artifact_extra_declared_path_unmatched_checksum_raises(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)
    cand_meta["artifact_paths"]["extra_art"] = "outputs/extra.parquet"
    _write_json(meta_p, cand_meta)

    with pytest.raises(ValueError, match="artifact_checksums"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_source_run_ids_exact_set_equality_allows_different_order(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    cfg = _load_yaml(cfg_file)
    cfg["policy"]["source_run_ids"] = ["lightgbm_hpo_trial35"]
    _write_yaml(cfg_file, cfg)

    res = generate_final_model_and_submission(
        config_path=cfg_file,
        project_root=tmp_path,
    )
    assert res["submission_path"].exists()


def test_candidate_metadata_provenance_split_sha256_mismatch_raises(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    meta_p = _mini_candidate_metadata_path(tmp_path)
    cand_meta = _load_json(meta_p)
    cand_meta["provenance"]["split_sha256"] = "a" * 64
    _write_json(meta_p, cand_meta)

    with pytest.raises(ValueError, match="split_sha256|provenance"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
        )


def test_strict_manifest_rejects_feature_dtypes_key_order_mismatch(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    res = generate_final_model_and_submission(config_path=cfg_file, project_root=tmp_path)
    model_dir = res["final_model_dir"]
    manifest_p = model_dir / "manifest.json"
    manifest = _load_json(manifest_p)

    # Reverse order of keys in feature_dtypes
    orig_keys = list(manifest["feature_dtypes"].keys())
    rev_dtypes = {k: manifest["feature_dtypes"][k] for k in reversed(orig_keys)}
    manifest["feature_dtypes"] = rev_dtypes
    _write_json(manifest_p, manifest)

    with pytest.raises(ValueError, match="feature_dtypes keys"):
        load_final_model(model_dir)


def test_generate_final_model_rejects_symlinked_custom_output_root(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    real_out = tmp_path / "outputs_custom"
    real_out.mkdir(parents=True, exist_ok=True)
    symlink_out = tmp_path / "outputs_symlink"
    symlink_out.symlink_to(real_out)

    with pytest.raises(ValueError, match="symlink detected"):
        generate_final_model_and_submission(
            config_path=cfg_file,
            project_root=tmp_path,
            output_root=symlink_out,
        )


def test_validate_final_config_rejects_symlinked_splits_dir(tmp_path) -> None:
    cfg_file, train_df, test_df, sample_df = _setup_mini_project(tmp_path)
    splits_dir = tmp_path / "outputs" / "splits"
    target_splits = tmp_path / "other_splits"
    target_splits.mkdir(parents=True, exist_ok=True)
    # Move manifest file to target_splits and replace splits_dir with symlink
    (target_splits / "repeated_stratified_5x3.parquet").write_bytes(
        (splits_dir / "repeated_stratified_5x3.parquet").read_bytes()
    )
    shutil.rmtree(splits_dir)
    splits_dir.symlink_to(target_splits)

    with pytest.raises(ValueError, match="symlink detected"):
        validate_final_config_and_provenance(
            config_path=cfg_file,
            project_root=tmp_path,
        )


