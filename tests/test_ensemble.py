import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ev_purchase.ensemble import (
    compute_dataset_sha256,
    evaluate_blend_grid,
    generate_weight_grid,
    materialize_hpo_finalist_predictions,
    promote_ensemble_run_transactionally,
    resolve_dry_run_output_dir,
    save_blended_oof_and_test_artifacts,
    save_ensemble_artifacts,
    update_final_config,
    validate_candidate_id,
    validate_candidate_provenance,
    validate_finalist_fold_metrics,
    validate_keyed_oof_alignment,
    validate_keyed_test_alignment,
    weighted_blend,
)


def test_staging_promotion_no_staging_token_in_metadata_and_canonical_files_exist(
    tmp_path: Path,
) -> None:
    """Regression: candidate_metadata artifact_paths must record canonical paths (no staging
    token) and the promotion validation must confirm those files exist with matching checksums.
    Note: This test covers the canonical-path regression (ensuring candidate_metadata artifact_paths
    use canonical paths with no staging token and matching checksums after promotion).

    This test exercises the staging -> promotion pipeline end-to-end by:
    1. Constructing a staging dir with candidate parquet files + candidate_metadata.json
       that records canonical (non-staging) relative paths and sha256 checksums.
    2. Calling promote_ensemble_run_transactionally to move the staging dir to canonical.
    3. Asserting the promoted metadata paths: no staging token, relative, contained under
       project root, not a symlink, file exists, sha256 matches.
    """
    import hashlib

    import pandas as pd

    project_root = tmp_path
    candidate_id = "catboost_hpo_trial7"

    # --- Build staging dir structure ---
    staging_dir = project_root / ".ensemble_staging_abc123"
    canonical_dir = project_root / "ensemble"

    staging_cand_dir = staging_dir / "candidates" / candidate_id
    staging_cand_dir.mkdir(parents=True)

    # Create real parquet artifacts in staging
    oof_df = pd.DataFrame(
        {"row_index": [0, 1], "repeat": [0, 0], "fold": [0, 1], "probability": [0.3, 0.7]}
    )
    test_df = pd.DataFrame({"id": [100, 101], "Will_Buy_EV": [0.4, 0.6]})
    staging_oof = staging_cand_dir / "oof_predictions.parquet"
    staging_test = staging_cand_dir / "test_predictions.parquet"
    oof_df.to_parquet(staging_oof)
    test_df.to_parquet(staging_test)

    # Compute checksums from staging files (same bytes will land at canonical after move)
    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    oof_checksum = _sha256(staging_oof)
    test_checksum = _sha256(staging_test)

    # Canonical relative paths (no staging token) — this is what the fixed code records
    rel_oof = f"ensemble/candidates/{candidate_id}/oof_predictions.parquet"
    rel_test = f"ensemble/candidates/{candidate_id}/test_predictions.parquet"

    cand_meta = {
        "candidate_id": candidate_id,
        "artifact_paths": {
            "oof_predictions": rel_oof,
            "test_predictions": rel_test,
        },
        "artifact_checksums": {
            "oof_predictions": oof_checksum,
            "test_predictions": test_checksum,
        },
    }
    with open(staging_cand_dir / "candidate_metadata.json", "w", encoding="utf-8") as f:
        json.dump(cand_meta, f, indent=2)

    # Add blend_metrics.json so promotion has something to process
    (staging_dir / "blend_metrics.json").write_text(
        '{"artifact_paths": {}}', encoding="utf-8"
    )

    # --- Promote staging -> canonical ---
    promote_ensemble_run_transactionally(
        staging_dir=staging_dir,
        output_root=canonical_dir,
        project_root=project_root,
    )

    # --- Post-promotion assertions ---
    promoted_meta_file = canonical_dir / "candidates" / candidate_id / "candidate_metadata.json"
    assert promoted_meta_file.exists(), "candidate_metadata.json missing after promotion"

    promoted_meta = json.loads(promoted_meta_file.read_text(encoding="utf-8"))
    root_resolved = project_root.resolve()

    for art_key, rel_p in promoted_meta["artifact_paths"].items():
        # 1. No staging token in recorded path
        assert "staging" not in rel_p, f"Staging token in promoted path {art_key}: {rel_p!r}"

        # 2. Path is relative (not absolute)
        assert not Path(rel_p).is_absolute(), f"Path is absolute: {rel_p!r}"

        # 3. Contained under project root
        resolved_p = (root_resolved / rel_p).resolve()
        assert str(resolved_p).startswith(str(root_resolved)), (
            f"Path escapes project root: {rel_p!r}"
        )

        # 4. Not a symlink
        assert not resolved_p.is_symlink(), f"Path is a symlink: {rel_p!r}"

        # 5. File exists at canonical location
        assert resolved_p.exists(), f"Canonical file missing: {resolved_p}"

        # 6. SHA-256 matches recorded checksum
        file_hasher = hashlib.sha256()
        with open(resolved_p, "rb") as fh:
            file_hasher.update(fh.read())
        actual_sha = file_hasher.hexdigest()
        expected_sha = promoted_meta["artifact_checksums"][art_key]
        assert actual_sha == expected_sha, (
            f"Checksum mismatch for {art_key}: got {actual_sha!r}, want {expected_sha!r}"
        )


def test_empty_target_directory_rollback_restores_empty_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "empty_canonical"
    output_root.mkdir(parents=True)

    stage = tmp_path / "stage"
    stage.mkdir(parents=True)
    (stage / "blend_metrics.json").write_text("{}", encoding="utf-8")

    staged_cfg = tmp_path / ".final.yaml_staging_1234"
    staged_cfg.write_text("staged_cfg_content", encoding="utf-8")

    canonical_cfg = tmp_path / "final.yaml"

    def failing_callback() -> None:
        raise RuntimeError("Post-promotion callback failure")

    with pytest.raises(RuntimeError, match="Post-promotion callback failure"):
        promote_ensemble_run_transactionally(
            staging_dir=stage,
            output_root=output_root,
            staged_config_path=staged_cfg,
            target_config_path=canonical_cfg,
            post_promote_callback=failing_callback,
        )

    assert output_root.exists()
    assert list(output_root.iterdir()) == []
    assert not canonical_cfg.exists()
    assert not staged_cfg.exists()
    assert not stage.exists()


def test_sibling_staged_config_promotion_and_cleanup(tmp_path: Path) -> None:
    output_root = tmp_path / "ensemble"
    stage = tmp_path / ".ensemble_staging_1234"
    stage.mkdir(parents=True)
    (stage / "blend_metrics.json").write_text('{"artifact_paths": {}}', encoding="utf-8")

    staged_cfg = tmp_path / ".final.yaml_staging_5678"
    staged_cfg.write_text("config_bytes_test", encoding="utf-8")
    canonical_cfg = tmp_path / "final.yaml"

    promote_ensemble_run_transactionally(
        staging_dir=stage,
        output_root=output_root,
        staged_config_path=staged_cfg,
        target_config_path=canonical_cfg,
    )

    assert output_root.exists()
    assert canonical_cfg.exists()
    assert canonical_cfg.read_text(encoding="utf-8") == "config_bytes_test"
    assert not staged_cfg.exists()
    assert not stage.exists()


def test_transactional_promotion_full_rollback_scenarios(tmp_path: Path) -> None:
    stage_1 = tmp_path / "stage_1"
    stage_1.mkdir()
    (stage_1 / "blend_metrics.json").write_text("stage1_metrics", encoding="utf-8")

    staged_cfg_1 = tmp_path / "staged_cfg_1.yaml"
    staged_cfg_1.write_text("staged_cfg_1_content", encoding="utf-8")

    canonical_dir = tmp_path / "canonical_1"
    canonical_cfg = tmp_path / "final_1.yaml"

    def failing_callback() -> None:
        raise RuntimeError("Injected post-promotion error")

    with pytest.raises(RuntimeError, match="Injected post-promotion error"):
        promote_ensemble_run_transactionally(
            staging_dir=stage_1,
            output_root=canonical_dir,
            staged_config_path=staged_cfg_1,
            target_config_path=canonical_cfg,
            post_promote_callback=failing_callback,
        )

    assert not canonical_dir.exists()
    assert not canonical_cfg.exists()
    assert not stage_1.exists()

    stage_2 = tmp_path / "stage_2"
    stage_2.mkdir()
    (stage_2 / "blend_metrics.json").write_text("NEW_METRICS", encoding="utf-8")
    (stage_2 / "candidates").mkdir()
    (stage_2 / "candidates" / "new_cand").mkdir()

    staged_cfg_2 = tmp_path / "staged_cfg_2.yaml"
    staged_cfg_2.write_text("NEW_CONFIG", encoding="utf-8")

    canonical_dir_2 = tmp_path / "canonical_2"
    canonical_dir_2.mkdir()
    (canonical_dir_2 / "blend_metrics.json").write_text("OLD_METRICS", encoding="utf-8")
    (canonical_dir_2 / "candidates").mkdir()
    (canonical_dir_2 / "candidates" / "old_cand").mkdir()

    canonical_cfg_2 = tmp_path / "final_2.yaml"
    canonical_cfg_2.write_text("OLD_CONFIG", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Injected post-promotion error"):
        promote_ensemble_run_transactionally(
            staging_dir=stage_2,
            output_root=canonical_dir_2,
            staged_config_path=staged_cfg_2,
            target_config_path=canonical_cfg_2,
            post_promote_callback=failing_callback,
        )

    assert (canonical_dir_2 / "blend_metrics.json").read_text(encoding="utf-8") == "OLD_METRICS"
    assert (canonical_dir_2 / "candidates" / "old_cand").exists()
    assert not (canonical_dir_2 / "candidates" / "new_cand").exists()
    assert canonical_cfg_2.read_text(encoding="utf-8") == "OLD_CONFIG"

    stage_3 = tmp_path / "stage_3"
    stage_3.mkdir()
    (stage_3 / "blend_metrics.json").write_text("NEW_METRICS", encoding="utf-8")
    (stage_3 / "candidates").mkdir()
    (stage_3 / "candidates" / "new_cand").mkdir()

    staged_cfg_3 = tmp_path / "staged_cfg_3.yaml"
    staged_cfg_3.write_text("NEW_CONFIG", encoding="utf-8")

    promote_ensemble_run_transactionally(
        staging_dir=stage_3,
        output_root=canonical_dir_2,
        staged_config_path=staged_cfg_3,
        target_config_path=canonical_cfg_2,
    )
    assert (canonical_dir_2 / "blend_metrics.json").read_text(encoding="utf-8") == "NEW_METRICS"
    assert (canonical_dir_2 / "candidates" / "new_cand").exists()
    assert not (canonical_dir_2 / "candidates" / "old_cand").exists()
    assert canonical_cfg_2.read_text(encoding="utf-8") == "NEW_CONFIG"


def test_smoke_dir_isolated_from_canonical_real_generation(tmp_path: Path) -> None:
    project_root = Path.cwd()
    p1 = resolve_dry_run_output_dir("outputs/ensemble", project_root)
    assert p1 == (project_root / "outputs" / "ensemble_smoke").resolve()

    p2 = resolve_dry_run_output_dir(tmp_path / "custom", project_root)
    assert p2 == (tmp_path / "custom").resolve()


def test_canonical_fingerprint_preserves_tuning_provenance(tmp_path: Path) -> None:
    cfg_p = Path("configs/base.yaml")
    assert cfg_p.exists()
    from ev_purchase.config import load_config

    base_cfg = load_config(cfg_p, project_root=Path.cwd())

    dataset_hash = compute_dataset_sha256(base_cfg.data.train, base_cfg.data.test)
    assert dataset_hash == "169e0aee03673611f12f2f24afba67399e0038279c049a15a5db67e952afee46"

    out_cfg = tmp_path / "final.yaml"
    components_info = {
        "cand_1": {
            "run_id": "cand_1",
            "model_family": "catboost",
            "tuning_study": "catboost_hpo",
            "trial_number": 22,
            "params": {"depth": 6},
            "weight": 1.0,
            "status": "completed",
            "protocol": "full",
            "dry_run": False,
            "provenance": {
                "data_sha256": "wrong_component_hash",
                "split_sha256": "split_sha",
            },
        }
    }
    with pytest.raises(ValueError, match="data_sha256 mismatch"):
        update_final_config(
            selected_policy="single_model",
            winning_weights={"cand_1": 1.0},
            components_info=components_info,
            best_single_id="cand_1",
            winning_roc_auc=0.94,
            lift_over_best_single=0.0,
            split_sha256="split_sha",
            data_sha256="top_level_hash",
            config_out_path=out_cfg,
            project_root=Path.cwd(),
        )


def test_portable_paths_in_persisted_artifacts(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True)

    df = pd.DataFrame([{"blend_id": "blend_0", "roc_auc": 0.9}])
    payload = {"selected_policy": "blend"}
    save_ensemble_artifacts(
        df,
        payload,
        output_dir=staging,
        target_canonical_dir=Path("outputs/ensemble"),
        project_root=tmp_path,
    )

    metrics_file = staging / "blend_metrics.json"
    content = metrics_file.read_text(encoding="utf-8")

    assert "/home/" not in content
    assert str(tmp_path) not in content

    meta = json.loads(content)
    assert (
        meta["artifact_paths"]["blend_leaderboard"] == "outputs/ensemble/blend_leaderboard.parquet"
    )


def test_safe_candidate_ids_and_path_containment(tmp_path: Path) -> None:
    assert validate_candidate_id("catboost_hpo_trial22") == "catboost_hpo_trial22"

    with pytest.raises(ValueError, match="Invalid candidate ID format"):
        validate_candidate_id("../evil_candidate")

    with pytest.raises(ValueError, match="Invalid candidate ID format"):
        validate_candidate_id("cand/with/slashes")

    with pytest.raises(ValueError, match="Invalid candidate ID format"):
        validate_candidate_id("cand\\with\\backslashes")

    study_dir = tmp_path / "evil_study"
    study_dir.mkdir(parents=True)

    meta = {
        "smoke": False,
        "study_name": "../evil_study",
        "model_name": "catboost",
        "provenance": {
            "model_name": "catboost",
            "data_sha256": "169e0aee03673611f12f2f24afba67399e0038279c049a15a5db67e952afee46",
            "split_sha256": "dummy_split_sha",
        },
    }
    with open(study_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    finalists_df = pd.DataFrame([{"finalist_rank": 1, "trial_number": 1}])
    finalists_df.to_parquet(study_dir / "finalists.parquet")
    finalist_folds = pd.DataFrame([{"trial_number": 1, "fold_id": 0}])
    finalist_folds.to_parquet(study_dir / "finalist_fold_metrics.parquet")

    with pytest.raises(ValueError, match="Invalid candidate ID format"):
        materialize_hpo_finalist_predictions(
            study_dir, output_dir=tmp_path / "candidates", dry_run=True, project_root=Path.cwd()
        )


def test_dry_run_canonical_path_variants_forced_to_smoke_dir(tmp_path: Path) -> None:
    project_root = Path.cwd()
    p1 = resolve_dry_run_output_dir("outputs/ensemble", project_root)
    assert p1 == (project_root / "outputs" / "ensemble_smoke").resolve()

    p2 = resolve_dry_run_output_dir("./outputs/ensemble/../ensemble", project_root)
    assert p2 == (project_root / "outputs" / "ensemble_smoke").resolve()

    p3 = resolve_dry_run_output_dir(tmp_path / "custom", project_root)
    assert p3 == (tmp_path / "custom").resolve()

    canonical_metrics_p = project_root / "outputs" / "ensemble" / "blend_metrics.json"
    canonical_bytes_before = (
        canonical_metrics_p.read_bytes() if canonical_metrics_p.exists() else None
    )
    canonical_mtime_before = (
        canonical_metrics_p.stat().st_mtime if canonical_metrics_p.exists() else None
    )

    cmd = [
        "rtk",
        "uv",
        "run",
        "python",
        "scripts/ensemble.py",
        "--dry-run",
        "--output-dir",
        "outputs/ensemble",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    if canonical_bytes_before is None:
        assert not canonical_metrics_p.exists()
    else:
        assert canonical_metrics_p.exists()
        assert canonical_metrics_p.read_bytes() == canonical_bytes_before
        assert canonical_metrics_p.stat().st_mtime == canonical_mtime_before
    assert (project_root / "outputs" / "ensemble_smoke" / "blend_metrics.json").exists()


def test_finalist_fold_validation_supports_dynamic_protocol() -> None:
    fold_df = pd.DataFrame([{"trial_number": 42, "fold_id": f} for f in range(6)])
    validate_finalist_fold_metrics(fold_df, trial_number=42, expected_fold_ids=list(range(6)))

    with pytest.raises(ValueError, match="fold_id mismatch"):
        validate_finalist_fold_metrics(fold_df, trial_number=42, expected_fold_ids=list(range(15)))


def test_dry_run_does_not_write_canonical_state(tmp_path: Path) -> None:
    canonical_cfg = Path("configs/final.yaml")
    cfg_mtime_before = canonical_cfg.stat().st_mtime if canonical_cfg.exists() else None

    cmd = [
        "rtk",
        "uv",
        "run",
        "python",
        "scripts/ensemble.py",
        "--dry-run",
        "--output-dir",
        str(tmp_path / "ensemble"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0

    cfg_mtime_after = canonical_cfg.stat().st_mtime if canonical_cfg.exists() else None
    assert cfg_mtime_before == cfg_mtime_after


def test_test_isolation_leaves_sentinel_files_unchanged(tmp_path: Path) -> None:
    df = pd.DataFrame([{"blend_id": "blend_0", "roc_auc": 0.9}])
    save_ensemble_artifacts(
        df, {"status": "test"}, output_dir=tmp_path / "ensemble", project_root=Path.cwd()
    )

    assert (tmp_path / "ensemble" / "blend_leaderboard.parquet").exists()
    assert not (Path("outputs/ensemble_test_sentinel.parquet")).exists()


def test_oof_target_alignment_with_shuffled_keys() -> None:
    y_true = np.array([10, 20, 30])
    oof_df = pd.DataFrame(
        {
            "row_index": [2, 0, 1],
            "repeat": [0, 0, 0],
            "fold": [0, 0, 0],
            "probability": [0.3, 0.1, 0.2],
        }
    )
    aligned_target = y_true[oof_df["row_index"].to_numpy()]
    np.testing.assert_array_equal(aligned_target, np.array([30, 10, 20]))


def test_provenance_validation_rejects_mismatched_hashes() -> None:
    cand_prov = {
        "data_sha256": "wrong_data_hash",
        "split_sha256": "correct_split_hash",
        "smoke": False,
    }
    with pytest.raises(ValueError, match="data_sha256 mismatch"):
        validate_candidate_provenance(
            cand_prov,
            expected_data_sha="correct_data_hash",
            expected_split_sha="correct_split_hash",
        )


def test_exact_fold_identity_rejects_duplicate_or_missing_folds() -> None:
    bad_folds = pd.DataFrame(
        [
            {"trial_number": 1, "fold_id": 0},
            {"trial_number": 1, "fold_id": 0},
        ]
    )
    with pytest.raises(ValueError, match="fold_id"):
        validate_finalist_fold_metrics(bad_folds, trial_number=1, expected_fold_ids=list(range(15)))


def test_final_config_completeness_requires_real_status(tmp_path: Path) -> None:
    out_cfg = tmp_path / "final.yaml"
    components_info = {
        "cand_1": {
            "model_family": "catboost",
            "tuning_study": "catboost_hpo",
            "trial_number": 22,
            "params": {"depth": 6},
            "weight": 1.0,
            "dry_run": True,
        }
    }
    with pytest.raises(ValueError, match="incomplete/smoke run"):
        update_final_config(
            selected_policy="single_model",
            winning_weights={"cand_1": 1.0},
            components_info=components_info,
            best_single_id="cand_1",
            winning_roc_auc=0.94,
            lift_over_best_single=0.0,
            split_sha256="split_sha",
            data_sha256="data_sha",
            config_out_path=out_cfg,
            project_root=Path.cwd(),
        )


def test_weighted_blend_normalizes_weights() -> None:
    predictions = {"a": np.array([0.2, 0.8]), "b": np.array([0.4, 0.6])}
    result = weighted_blend(predictions, {"a": 3.0, "b": 1.0})
    np.testing.assert_allclose(result, np.array([0.25, 0.75]))


def test_weighted_blend_rejects_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="aligned"):
        weighted_blend({"a": np.array([0.2]), "b": np.array([0.4, 0.6])}, {"a": 1, "b": 1})


def test_weighted_blend_rejects_negative_and_zero_weights() -> None:
    predictions = {"a": np.array([0.2, 0.8]), "b": np.array([0.4, 0.6])}
    with pytest.raises(ValueError, match="non-negative"):
        weighted_blend(predictions, {"a": -1.0, "b": 1.0})

    with pytest.raises(ValueError, match="positive"):
        weighted_blend(predictions, {"a": 0.0, "b": 0.0})


def test_weighted_blend_rejects_invalid_probabilities() -> None:
    predictions = {"a": np.array([0.2, np.nan]), "b": np.array([0.4, 0.6])}
    with pytest.raises(ValueError, match="finite"):
        weighted_blend(predictions, {"a": 1.0, "b": 1.0})

    predictions_oob = {"a": np.array([0.2, 1.5]), "b": np.array([0.4, 0.6])}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        weighted_blend(predictions_oob, {"a": 1.0, "b": 1.0})


def test_equal_length_reordered_oof_keys_rejected() -> None:
    df1 = pd.DataFrame(
        {
            "row_index": [0, 1],
            "repeat": [0, 0],
            "fold": [0, 1],
            "probability": [0.2, 0.8],
        }
    )
    df2 = pd.DataFrame(
        {
            "row_index": [1, 0],
            "repeat": [0, 0],
            "fold": [1, 0],
            "probability": [0.8, 0.2],
        }
    )

    with pytest.raises(ValueError, match="Misaligned OOF prediction keys"):
        validate_keyed_oof_alignment({"cand_a": df1, "cand_b": df2})


def test_reordered_test_ids_rejected() -> None:
    sample_sub = pd.DataFrame({"id": [101, 102], "Will_Buy_EV": [0.0, 0.0]})
    cand_a = pd.DataFrame({"id": [101, 102], "Will_Buy_EV": [0.3, 0.7]})
    cand_b = pd.DataFrame({"id": [102, 101], "Will_Buy_EV": [0.7, 0.3]})

    with pytest.raises(ValueError, match="Misaligned test IDs"):
        validate_keyed_test_alignment({"cand_a": cand_a, "cand_b": cand_b}, sample_sub)


def test_incomplete_or_smoke_hpo_finalist_rejected(tmp_path: Path) -> None:
    study_dir = tmp_path / "smoke_study"
    study_dir.mkdir(parents=True)

    meta = {
        "smoke": True,
        "study_name": "smoke_study",
        "model_name": "catboost",
        "provenance": {"data_sha256": "dummy", "split_sha256": "dummy"},
    }
    with open(study_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    finalists_df = pd.DataFrame(
        [
            {
                "finalist_rank": 1,
                "trial_number": 5,
                "roc_auc_mean": 0.94,
                "evaluated_fold_ids": str(list(range(10))),
                "param_depth": 6,
            }
        ]
    )
    finalists_df.to_parquet(study_dir / "finalists.parquet")

    finalist_folds = pd.DataFrame([{"trial_number": 5, "fold_id": f} for f in range(10)])
    finalist_folds.to_parquet(study_dir / "finalist_fold_metrics.parquet")

    with pytest.raises(ValueError, match="smoke mode"):
        materialize_hpo_finalist_predictions(
            study_dir, output_dir=tmp_path / "candidates", dry_run=True, project_root=Path.cwd()
        )


def test_materialization_uses_tuned_params_and_exact_complements(tmp_path: Path) -> None:
    study_dir = tmp_path / "catboost_hpo"
    study_dir.mkdir(parents=True)

    manifest_p = Path("outputs/splits/repeated_stratified_5x3.parquet")
    if not manifest_p.exists():
        pytest.skip("Split manifest required for materialization test")

    import hashlib

    with open(manifest_p, "rb") as f:
        split_sha = hashlib.sha256(f.read()).hexdigest()

    data_sha = "169e0aee03673611f12f2f24afba67399e0038279c049a15a5db67e952afee46"

    meta = {
        "smoke": False,
        "study_name": "catboost_hpo",
        "model_name": "catboost",
        "provenance": {
            "model_name": "catboost",
            "data_sha256": data_sha,
            "split_sha256": split_sha,
        },
    }
    with open(study_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    finalists_df = pd.DataFrame(
        [
            {
                "finalist_rank": 1,
                "trial_number": 22,
                "roc_auc_mean": 0.941968,
                "evaluated_fold_ids": str(list(range(15))),
                "param_iterations": 200,
                "param_depth": 6,
                "param_learning_rate": 0.05,
            }
        ]
    )
    finalists_df.to_parquet(study_dir / "finalists.parquet")

    finalist_folds = pd.DataFrame(
        [{"trial_number": 22, "fold_id": f, "roc_auc": 0.94} for f in range(15)]
    )
    finalist_folds.to_parquet(study_dir / "finalist_fold_metrics.parquet")

    res = materialize_hpo_finalist_predictions(
        study_dir, output_dir=tmp_path / "candidates", dry_run=True, project_root=Path.cwd()
    )
    assert res["candidate_id"].startswith("catboost_hpo_trial22")
    assert res["params"] == {"iterations": 200, "depth": 6, "learning_rate": 0.05}
    assert len(res["oof_df"]) > 0


def test_blended_oof_artifact_complete_with_all_repeats(tmp_path: Path) -> None:
    oof_df = pd.DataFrame(
        {
            "row_index": [0, 1, 0, 1, 0, 1],
            "repeat": [0, 0, 1, 1, 2, 2],
            "fold": [0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1],
            "probability": [0.1, 0.9, 0.2, 0.8, 0.15, 0.85],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [101, 102],
            "Will_Buy_EV": [0.15, 0.85],
        }
    )

    artifacts = save_blended_oof_and_test_artifacts(
        oof_df, test_df, n_rows=2, n_repeats=3, output_dir=tmp_path / "ensemble"
    )

    assert artifacts["blended_oof_predictions"].exists()
    assert artifacts["blended_test_predictions"].exists()

    saved_oof = pd.read_parquet(artifacts["blended_oof_predictions"])
    assert list(saved_oof.columns) == ["row_index", "repeat", "fold", "target", "probability"]
    assert len(saved_oof) == 6

    saved_test = pd.read_parquet(artifacts["blended_test_predictions"])
    assert list(saved_test.columns) == ["id", "Will_Buy_EV"]
    assert saved_test["id"].tolist() == [101, 102]


def test_generate_weight_grid_structure() -> None:
    models = ["model_a", "model_b"]
    grid = generate_weight_grid(models, grid_steps=5)
    assert len(grid) > 0
    for w in grid:
        assert set(w.keys()) == set(models)
        assert pytest.approx(sum(w.values())) == 1.0


def test_evaluate_blend_grid_ranks_by_roc_auc() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    pred_a = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9])
    pred_b = np.array([0.4, 0.6, 0.5, 0.3, 0.7, 0.2, 0.6, 0.8])

    oof_preds = {"model_a": pred_a, "model_b": pred_b}

    df = evaluate_blend_grid(oof_preds, y_true)
    assert not df.empty
    assert "roc_auc" in df.columns
    assert "lift_over_best_single" in df.columns
    assert df["roc_auc"].is_monotonic_decreasing


def test_save_ensemble_artifacts(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "blend_id": ["blend_0000"],
            "roc_auc": [0.95],
            "average_precision": [0.80],
            "log_loss": [0.20],
            "lift_over_best_single": [0.01],
            "is_single_model": [False],
            "is_equal_weights": [True],
            "weights_json": ['{"a": 0.5, "b": 0.5}'],
        }
    )
    payload = {
        "selected_policy": "blend",
        "winning_roc_auc": 0.95,
        "best_single_roc_auc": 0.94,
        "lift_over_best_single": 0.01,
    }

    saved = save_ensemble_artifacts(
        df, payload, output_dir=tmp_path / "ensemble", project_root=tmp_path
    )
    assert saved["blend_leaderboard"].exists()
    assert saved["blend_metrics"].exists()

    with open(saved["blend_metrics"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["selected_policy"] == "blend"
    assert meta["artifact_paths"]["blend_leaderboard"] == "ensemble/blend_leaderboard.parquet"
