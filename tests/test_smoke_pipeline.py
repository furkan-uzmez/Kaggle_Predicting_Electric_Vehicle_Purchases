import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ev_purchase.splits import build_split_manifest, validate_split_manifest
from ev_purchase.submission import generate_final_model_and_submission
from ev_purchase.training import run_cross_validation


def _snapshot_production_files() -> dict[str, tuple[str, Any]]:
    proj_root = Path.cwd()
    targets = [
        proj_root / "configs" / "final.yaml",
    ]
    for d in [proj_root / "data", proj_root / "outputs"]:
        if d.exists():
            for p in d.rglob("*"):
                targets.append(p)

    snapshot = {}
    for p in targets:
        if p.exists():
            rel = str(p.relative_to(proj_root))
            if p.is_symlink():
                snapshot[rel] = ("symlink", os.readlink(p))
            elif p.is_dir():
                snapshot[rel] = ("dir", p.stat().st_mtime)
            elif p.is_file():
                mtime = p.stat().st_mtime
                hasher = hashlib.sha256()
                with open(p, "rb") as f:
                    hasher.update(f.read())
                snapshot[rel] = ("file", (mtime, hasher.hexdigest()))
    return snapshot


def _verify_production_snapshot(before_snapshot: dict[str, tuple[str, Any]]) -> None:
    after_snapshot = _snapshot_production_files()
    assert set(before_snapshot.keys()) == set(
        after_snapshot.keys()
    ), "Production files inventory changed during smoke test!"
    for rel_path, (b_type, b_data) in before_snapshot.items():
        a_type, a_data = after_snapshot[rel_path]
        assert a_type == b_type, f"Production entity type changed for {rel_path}"
        assert a_data == b_data, f"Production file modified: {rel_path}"


def test_smoke_pipeline_dummy_and_logistic_isolation(tmp_path) -> None:
    snapshot_before = _snapshot_production_files()

    np.random.seed(42)
    n_train = 30
    n_test = 10

    train_ids = list(range(100, 100 + n_train))
    test_ids = list(range(200, 200 + n_test))

    train_df = pd.DataFrame(
        {
            "id": train_ids,
            "num_feat": np.random.randn(n_train),
            "cat_feat": np.random.choice(["A", "B"], size=n_train),
            "Will_Buy_EV": np.random.choice(["No", "Yes"], size=n_train),
        }
    )

    test_df = pd.DataFrame(
        {
            "id": test_ids,
            "num_feat": np.random.randn(n_test),
            "cat_feat": np.random.choice(["A", "B"], size=n_test),
        }
    )

    sample_df = pd.DataFrame(
        {
            "id": test_ids,
            "Will_Buy_EV": [0.5] * n_test,
        }
    )

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'smoke'")

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sample_csv = data_dir / "sample_submission.csv"

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
    validate_split_manifest(
        manifest=split_manifest,
        n_rows=n_train,
        n_splits=2,
        n_repeats=1,
        y=y_encoded,
    )

    splits_dir = tmp_path / "outputs" / "splits"
    splits_dir.mkdir(parents=True)
    manifest_parquet = splits_dir / "repeated_stratified_5x3.parquet"
    split_manifest.to_parquet(manifest_parquet)

    split_hasher = hashlib.sha256()
    with open(manifest_parquet, "rb") as f:
        split_hasher.update(f.read())
    split_sha256 = split_hasher.hexdigest()

    # 1. Dummy model CV run
    dummy_config_dict = {
        "project": {"name": "smoke_test", "seed": 42},
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
        "model": {"name": "dummy"},
    }

    dummy_cfg_path = tmp_path / "configs" / "cv_dummy.yaml"
    dummy_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dummy_cfg_path, "w") as f:
        yaml.dump(dummy_config_dict, f)

    dummy_res = run_cross_validation(
        config_path=dummy_cfg_path,
        project_root=tmp_path,
        split_manifest_path=manifest_parquet,
        smoke=True,
    )
    assert dummy_res["status"] == "smoke_completed"
    assert dummy_res["metrics_path"].exists()

    # 2. Logistic model CV run
    logistic_config_dict = {
        "project": {"name": "smoke_test", "seed": 42},
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
        "model": {"name": "logistic"},
    }

    logistic_cfg_path = tmp_path / "configs" / "cv_logistic.yaml"
    with open(logistic_cfg_path, "w") as f:
        yaml.dump(logistic_config_dict, f)

    logistic_res = run_cross_validation(
        config_path=logistic_cfg_path,
        project_root=tmp_path,
        split_manifest_path=manifest_parquet,
        smoke=True,
    )
    assert logistic_res["status"] == "smoke_completed"
    assert logistic_res["metrics_path"].exists()

    # 3. Final model setup and submission generation
    meta_dir = tmp_path / "outputs" / "ensemble" / "candidates" / "lightgbm_hpo_trial35"
    meta_dir.mkdir(parents=True, exist_ok=True)

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
    cand_meta_path = meta_dir / "candidate_metadata.json"
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

    final_config_dict = {
        "project": {"name": "smoke_test", "seed": 42},
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
            "winning_weights": {
                "lightgbm_hpo_trial35": 1.0,
            },
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
                    "provenance": {
                        "data_sha256": data_sha256,
                        "split_sha256": split_sha256,
                    },
                }
            },
            "split_sha256": split_sha256,
            "data_sha256": data_sha256,
        },
    }

    config_path = tmp_path / "configs" / "final.yaml"
    with open(config_path, "w") as f:
        yaml.dump(final_config_dict, f)

    generate_final_model_and_submission(
        config_path=config_path,
        project_root=tmp_path,
    )

    out_root = tmp_path / "outputs"
    sub_path = out_root / "submissions" / "submission.csv"
    report_path = out_root / "submissions" / "validation_report.json"
    model_dir = out_root / "final_model"
    manifest_path = model_dir / "manifest.json"

    assert sub_path.exists()
    assert report_path.exists()
    assert manifest_path.exists()

    sub_df = pd.read_csv(sub_path)
    assert sub_df.columns.tolist() == ["id", "Will_Buy_EV"]
    assert sub_df["id"].tolist() == test_ids
    assert len(sub_df) == n_test

    with open(report_path) as f:
        report = json.load(f)
    assert report["row_count"] == n_test
    assert report["columns"] == ["id", "Will_Buy_EV"]
    assert report["id_order_match"] is True
    assert report["uniqueness_check"] is True
    assert report["finite_check"] is True
    assert report["range_check"] is True

    _verify_production_snapshot(snapshot_before)


def test_predict_cli_smoke(tmp_path) -> None:
    snapshot_before = _snapshot_production_files()

    np.random.seed(42)
    n_train = 30
    n_test = 10

    train_ids = list(range(1, 1 + n_train))
    test_ids = list(range(100, 100 + n_test))

    train_df = pd.DataFrame(
        {
            "id": train_ids,
            "f1": np.random.randn(n_train),
            "Will_Buy_EV": np.random.choice(["No", "Yes"], size=n_train),
        }
    )

    test_df = pd.DataFrame(
        {
            "id": test_ids,
            "f1": np.random.randn(n_test),
        }
    )

    sample_df = pd.DataFrame(
        {
            "id": test_ids,
            "Will_Buy_EV": [0.5] * n_test,
        }
    )

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'cli_smoke'")

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sample_csv = data_dir / "sample_submission.csv"

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
    splits_dir.mkdir(parents=True)
    manifest_parquet = splits_dir / "repeated_stratified_5x3.parquet"
    split_manifest.to_parquet(manifest_parquet)

    split_hasher = hashlib.sha256()
    with open(manifest_parquet, "rb") as f:
        split_hasher.update(f.read())
    split_sha256 = split_hasher.hexdigest()

    meta_dir = tmp_path / "outputs" / "ensemble" / "candidates" / "lightgbm_hpo_trial35"
    meta_dir.mkdir(parents=True, exist_ok=True)
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
    cand_meta_path = meta_dir / "candidate_metadata.json"
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
        "project": {"name": "cli_test", "seed": 42},
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
            "source_candidate_ids": ["lightgbm_hpo_trial35"],
            "source_run_ids": ["lightgbm_hpo_trial35"],
            "source_hpo_trials": {"lightgbm_hpo": 35},
            "winning_weights": {
                "lightgbm_hpo_trial35": 1.0,
            },
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
                    "provenance": {
                        "data_sha256": data_sha256,
                        "split_sha256": split_sha256,
                    },
                }
            },
            "split_sha256": split_sha256,
            "data_sha256": data_sha256,
        },
    }

    config_path = tmp_path / "configs" / "final.yaml"
    config_path.parent.mkdir(parents=True)
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    cmd = [
        sys.executable,
        str(Path.cwd() / "scripts" / "predict.py"),
        "--config",
        str(config_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
    err_msg = f"CLI failed with error:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert proc.returncode == 0, err_msg

    sub_csv = tmp_path / "outputs" / "submissions" / "submission.csv"
    assert sub_csv.exists()

    _verify_production_snapshot(snapshot_before)
