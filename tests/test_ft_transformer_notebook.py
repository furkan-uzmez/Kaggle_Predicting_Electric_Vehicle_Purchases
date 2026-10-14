"""Static and pure-Python contracts for the Kaggle FT-Transformer notebook."""

import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

BUILDER_PATH = Path("scripts/build_ft_transformer_notebook.py")
NOTEBOOK_PATH = Path("notebooks/kaggle_ev_purchase_ft_transformer.ipynb")
EXPECTED_SPLIT_CONTENT_SHA256 = "5a6a1965ff57c5214035bc4e01ae51e306419b9c05aa04cae8faad9d53fd2206"


def _load_builder():
    spec = importlib.util.spec_from_file_location("ft_transformer_notebook_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonicalize(notebook: nbformat.NotebookNode) -> dict:
    payload = json.loads(nbformat.writes(notebook))
    for cell in payload["cells"]:
        cell.pop("id", None)
    return payload


def _sources(notebook: nbformat.NotebookNode) -> tuple[str, str]:
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    return markdown, code


def _resume_discovery_functions(builder):
    namespace = {"Path": Path, "json": json, "hashlib": hashlib}
    exec(builder.RESUME_DISCOVERY_CODE, namespace)
    return namespace


def _write_resume_state(root: Path, fingerprint: dict, fingerprint_sha256: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "run_fingerprint": fingerprint,
                "run_fingerprint_sha256": fingerprint_sha256,
                "completed_folds": [],
            }
        ),
        encoding="utf-8",
    )
    return state_path


def _fingerprint_sha256(fingerprint: dict) -> str:
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_builder_is_source_of_truth_and_all_code_compiles() -> None:
    builder = _load_builder()
    generated = builder.build_notebook()
    checked_in = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(generated)
    assert _canonicalize(checked_in) == _canonicalize(generated)
    for index, cell in enumerate(generated.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"<ft_transformer_cell_{index}>", "exec")


def test_builder_output_is_byte_stable_including_cell_ids() -> None:
    builder = _load_builder()
    first = nbformat.writes(builder.build_notebook())
    second = nbformat.writes(builder.build_notebook())
    assert first == second


def test_notebook_is_self_contained_and_uses_exact_raw_predictors() -> None:
    builder = _load_builder()
    notebook = builder.build_notebook()
    _markdown, code = _sources(notebook)
    forbidden = [
        "from ev_purchase",
        "import ev_purchase",
        "from src",
        "sys.path.append",
        "/home/",
        "/Users/",
    ]
    assert not [token for token in forbidden if token in code]
    expected_predictors = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]
    assert builder.RAW_PREDICTORS == expected_predictors
    assert 'TARGET_MAPPING = {"No": 0, "Yes": 1}' in code
    assert "assert list(X_train_raw.columns) == RAW_PREDICTORS" in code


def test_exact_5x3_split_and_portable_content_hash_contract() -> None:
    builder = _load_builder()
    notebook = builder.build_notebook()
    _markdown, code = _sources(notebook)
    assert builder.EXPECTED_SPLIT_CONTENT_SHA256 == EXPECTED_SPLIT_CONTENT_SHA256
    assert "RepeatedStratifiedKFold" in code
    assert "N_SPLITS = 5" in code
    assert "N_REPEATS = 3" in code
    assert "SPLIT_RANDOM_STATE = 42" in code
    assert '["repeat", "fold", "row_index"], kind="stable"' in code
    assert "EXPECTED_SPLIT_CONTENT_SHA256" in code
    manifest = pd.DataFrame(
        {
            "row_index": np.array([0, 1, 2, 3], dtype=np.int64),
            "repeat": np.array([0, 0, 1, 1], dtype=np.int64),
            "fold": np.array([0, 1, 0, 1], dtype=np.int64),
        }
    )
    expected = builder.hashlib.sha256(
        manifest[["row_index", "repeat", "fold"]]
        .to_numpy(dtype="<i8", copy=True)
        .tobytes(order="C")
    ).hexdigest()
    assert builder.split_content_sha256(manifest) == expected


def test_cuda_guard_amp_gradient_and_best_state_contracts() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    required = [
        "if not torch.cuda.is_available()",
        "CUDA is required",
        "torch.amp.autocast",
        "torch.amp.GradScaler",
        "clip_grad_norm_",
        "torch.isfinite(loss)",
        "torch.isfinite(param.grad)",
        "best_model_state",
        "model.load_state_dict(best_model_state)",
    ]
    for token in required:
        assert token in code, f"Missing training safety contract: {token}"


def test_preflight_proves_smoke_overfit_and_first_optimizer_step() -> None:
    notebook = _load_builder().build_notebook()
    markdown, code = _sources(notebook)
    required = [
        "run_one_batch_smoke_test",
        "run_tiny_overfit_test",
        "first_optimizer_step_verified",
        "scaler.step(optimizer)",
        "checkpoint",
        "resume",
    ]
    for token in required:
        assert token in code or token in markdown, f"Missing preflight contract: {token}"


def test_15_fold_resume_and_time_budget_contract() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    required = [
        "TOTAL_FOLDS = N_SPLITS * N_REPEATS",
        "MAX_RUNTIME_SECONDS = 8.5 * 60 * 60",
        "TIME_BUFFER_SECONDS",
        "completed_folds",
        "fold_checkpoint",
        "run_state.json",
        "time_budget",
        "graceful",
        "base_seed + repeat * 100 + fold",
    ]
    for token in required:
        assert token in code.lower() or token in code, f"Missing resume/time contract: {token}"


def test_oof_schema_counts_and_completeness_guards() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    assert '["row_index", "repeat", "fold", "probability", "target"]' in code
    assert "len(oof_df) == len(train_df) * N_REPEATS" in code
    assert 'groupby(["repeat", "row_index"]).size().eq(1).all()' in code
    assert "len(completed_folds) == TOTAL_FOLDS" in code
    assert "only fully completed folds" in code.lower()
    assert "oof_predictions.csv" in code


def test_metrics_provenance_and_artifact_inventory_contracts() -> None:
    notebook = _load_builder().build_notebook()
    markdown, code = _sources(notebook)
    for metric in ["roc_auc_score", "average_precision_score", "log_loss"]:
        assert metric in code
    for artifact in [
        "fold_metrics.csv",
        "repeat_metrics.csv",
        "global_metrics.json",
        "training_manifest.json",
        "artifact_inventory.json",
    ]:
        assert artifact in code
    for field in [
        "data_hashes",
        "split_content_sha256",
        "config_sha256",
        "source_sha256",
        "environment",
        "device",
        "sha256",
        "row_count",
    ]:
        assert field in code
    assert "/kaggle/working/ft_transformer_5x3" in code
    assert "attach" in markdown.lower() and "download" in markdown.lower()


def test_submission_only_after_all_folds_and_best_checkpoint_inference() -> None:
    notebook = _load_builder().build_notebook()
    markdown, code = _sources(notebook)
    assert "if run_complete:" in code
    assert "best_model_state" in code
    assert "submission.csv" in code
    assert "sample_submission" in code
    assert "incomplete" in markdown.lower()
    assert "do not" in markdown.lower() and "submission" in markdown.lower()


def test_resume_is_authenticated_and_cross_session_bootstrappable() -> None:
    notebook = _load_builder().build_notebook()
    markdown, code = _sources(notebook)
    required = [
        "RUN_FINGERPRINT",
        "run_fingerprint_sha256",
        "validate_run_fingerprint",
        "split_content_sha256",
        "config_sha256",
        "source_sha256",
        "restore_attached_resume_artifacts",
        "/kaggle/input",
        "shutil.copytree",
    ]
    for token in required:
        assert token in code, f"Missing authenticated resume contract: {token}"
    assert "attach" in markdown.lower()


def test_resume_discovery_supports_arbitrary_kaggle_slug_root(tmp_path: Path) -> None:
    builder = _load_builder()
    functions = _resume_discovery_functions(builder)
    fingerprint = {"experiment_name": "expected"}
    fingerprint_sha256 = _fingerprint_sha256(fingerprint)
    artifact_root = tmp_path / "arbitrary-upload-slug"
    _write_resume_state(artifact_root, fingerprint, fingerprint_sha256)

    selected = functions["select_resume_artifact_dir"](
        tmp_path, None, fingerprint, fingerprint_sha256
    )

    assert selected == artifact_root.resolve()


def test_resume_discovery_supports_nested_artifact_folder(tmp_path: Path) -> None:
    builder = _load_builder()
    functions = _resume_discovery_functions(builder)
    fingerprint = {"experiment_name": "expected"}
    fingerprint_sha256 = _fingerprint_sha256(fingerprint)
    artifact_root = tmp_path / "dataset-slug" / "exports" / "ft-run"
    _write_resume_state(artifact_root, fingerprint, fingerprint_sha256)

    selected = functions["select_resume_artifact_dir"](
        tmp_path, None, fingerprint, fingerprint_sha256
    )

    assert selected == artifact_root.resolve()


def test_automatic_resume_ignores_mismatch_but_explicit_override_rejects_it(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    functions = _resume_discovery_functions(builder)
    fingerprint = {"experiment_name": "expected"}
    fingerprint_sha256 = _fingerprint_sha256(fingerprint)
    mismatch = {"experiment_name": "other"}
    mismatch_sha256 = _fingerprint_sha256(mismatch)
    mismatched_root = tmp_path / "old-run"
    compatible_root = tmp_path / "current-run"
    _write_resume_state(mismatched_root, mismatch, mismatch_sha256)
    _write_resume_state(compatible_root, fingerprint, fingerprint_sha256)

    selected = functions["select_resume_artifact_dir"](
        tmp_path, None, fingerprint, fingerprint_sha256
    )
    assert selected == compatible_root.resolve()

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        functions["select_resume_artifact_dir"](
            tmp_path, mismatched_root, fingerprint, fingerprint_sha256
        )


def test_automatic_resume_rejects_multiple_compatible_candidates(tmp_path: Path) -> None:
    builder = _load_builder()
    functions = _resume_discovery_functions(builder)
    fingerprint = {"experiment_name": "expected"}
    fingerprint_sha256 = _fingerprint_sha256(fingerprint)
    _write_resume_state(tmp_path / "first-slug", fingerprint, fingerprint_sha256)
    _write_resume_state(tmp_path / "second-slug" / "nested", fingerprint, fingerprint_sha256)

    with pytest.raises(RuntimeError, match="Multiple fingerprint-compatible"):
        functions["select_resume_artifact_dir"](tmp_path, None, fingerprint, fingerprint_sha256)


def test_resume_handoff_documents_auto_discovery_and_override() -> None:
    notebook = _load_builder().build_notebook()
    markdown, _code = _sources(notebook)
    assert "automatically searches" in markdown.lower()
    assert "arbitrary dataset slug" in markdown.lower()
    assert "RESUME_INPUT_DIR" in markdown
    assert "manual copying" not in markdown.lower()


def test_resume_restores_rng_and_uses_epoch_deterministic_shuffle() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        "random.getstate()",
        "np.random.get_state()",
        "torch.get_rng_state()",
        "torch.cuda.get_rng_state_all()",
        "random.setstate",
        "np.random.set_state",
        "torch.set_rng_state",
        "torch.cuda.set_rng_state_all",
        "fold_seed + epoch",
        "dataloader_generator_state",
        "generator.set_state",
        "train_generator.get_state()",
        "def clone_checkpoint_value(value):",
        '"model_state": clone_checkpoint_value(model.state_dict())',
        '"optimizer_state": clone_checkpoint_value(optimizer.state_dict())',
    ]:
        assert token in code, f"Missing deterministic resume token: {token}"


def test_partial_global_metrics_empty_safety_and_full_manifest_settings() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        "available_prediction_metrics",
        "available_unique_rows",
        "FOLD_METRIC_COLUMNS",
        "num_workers",
        "pin_memory",
        "amp_dtype",
        "CosineAnnealingLR",
        "best_validation_loss",
        "StandardScaler",
        "OrdinalEncoder",
    ]:
        assert token in code, f"Missing partial-metric/manifest contract: {token}"


def test_atomic_prediction_publication_and_stale_output_quarantine() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        "atomic_csv_dump(oof_fold",
        "atomic_csv_dump(test_fold",
        "atomic_csv_dump(oof_df",
        "atomic_csv_dump(fold_metrics_df",
        "atomic_csv_dump(repeat_metrics_df",
        "atomic_csv_dump(submission",
        'for stale_name in ("submission.csv", "test_predictions.csv")',
        "quarantine_artifact(OUTPUT_DIR / stale_name",
        "quarantine_artifact(stale_manifest_path",
    ]:
        assert token in code, f"Missing atomic/stale-output contract: {token}"


def test_completed_fold_predictions_have_authenticated_metadata() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        '"fold_identity"',
        '"oof_prediction"',
        '"test_prediction"',
        'metadata.get("sha256") != sha256_file(prediction_path)',
        'metadata.get("row_count") != len(frame)',
        'metadata.get("schema") != list(frame.columns)',
    ]:
        assert token in code, f"Missing authenticated fold metadata contract: {token}"


def test_completed_checkpoint_can_regenerate_missing_fold_outputs() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        "recoverable_checkpoint",
        'recoverable_checkpoint.get("status") == "completed"',
        "completed_folds.remove(fold_key)",
        "Regenerating missing fold outputs from authenticated completed checkpoint",
    ]:
        assert token in code, f"Missing completed-checkpoint recovery contract: {token}"


def test_stale_complete_outputs_are_quarantined_before_training_resumes() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    assert "def quarantine_complete_only_outputs():" in code
    orchestration_start = code.index("started_at = time.time()")
    training_loop_start = code.index("for repeat in range(N_REPEATS):")
    quarantine_call = code.index("\nquarantine_complete_only_outputs()", orchestration_start)
    assert orchestration_start < quarantine_call < training_loop_start
    prevalidation_start = code.index("for fold_key in list(completed_folds):")
    assert quarantine_call < prevalidation_start


def test_checkpoints_are_bound_to_fold_identity() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    assert "def validate_fold_checkpoint(payload, source, repeat, fold):" in code
    assert 'payload.get("repeat") != repeat or payload.get("fold") != fold' in code
    assert code.count("validate_fold_checkpoint(") >= 4


def test_attached_resume_staging_verifies_copied_file_hashes() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        "def directory_file_hashes(root):",
        "source_hashes = directory_file_hashes(source_dir)",
        "staged_hashes = directory_file_hashes(stage_dir)",
        "if staged_hashes != source_hashes:",
    ]:
        assert token in code, f"Missing staged-copy verification contract: {token}"


@pytest.mark.parametrize(
    "items",
    [
        [[0, 0], [0, 0]],
        [[-1, 0]],
        [[0, 5]],
        [[0.0, 0]],
        [[0]],
        ["0,0"],
    ],
)
def test_completed_fold_state_validation_rejects_malformed_items(items: list[object]) -> None:
    builder = _load_builder()
    with pytest.raises(ValueError):
        builder.validate_completed_fold_items(items)


def test_completed_fold_state_validation_accepts_canonical_subset() -> None:
    builder = _load_builder()
    assert builder.validate_completed_fold_items([[0, 0], [2, 4]]) == {(0, 0), (2, 4)}
    _markdown, code = _sources(builder.build_notebook())
    assert 'completed_folds = validate_completed_fold_items(run_state["completed_folds"])' in code


def test_fingerprint_binds_runtime_and_exhaustive_config() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")',
        "torch.use_deterministic_algorithms(True)",
        '"runtime_compatibility": RUNTIME_COMPATIBILITY',
        '"split_random_state": SPLIT_RANDOM_STATE',
        '"target_mapping": TARGET_MAPPING',
        '"optimizer": "AdamW"',
        '"max_runtime_seconds": MAX_RUNTIME_SECONDS',
    ]:
        assert token in code, f"Missing runtime/config fingerprint contract: {token}"


def test_resumed_fold_semantics_and_metrics_are_revalidated() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    for token in [
        "def require_integer_series(series, name):",
        'require_integer_series(resumed_oof["row_index"]',
        'require_integer_series(resumed_oof["target"]',
        '"n_validation": len(expected_index)',
        '"roc_auc": float(roc_auc_score(',
        "validate_persisted_fold_metric(metric_row, expected_metric_row",
    ]:
        assert token in code, f"Missing strict resumed-fold validation: {token}"


def test_tiny_overfit_uses_balanced_float32_memorization_check() -> None:
    notebook = _load_builder().build_notebook()
    _markdown, code = _sources(notebook)
    start = code.index("def run_tiny_overfit_test")
    end = code.index("@torch.inference_mode()", start)
    function_source = code[start:end]
    for token in ["n_per_class", "loss.backward()", "minimum_loss", "accuracy"]:
        assert token in function_source
    assert "amp_autocast" not in function_source
    assert "GradScaler" not in function_source
