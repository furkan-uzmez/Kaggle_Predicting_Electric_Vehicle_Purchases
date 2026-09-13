"""Submission generation, final model retraining, reloadable artifacts, and validation utilities."""

import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.features import apply_feature_policy
from ev_purchase.metrics import validate_probabilities
from ev_purchase.models import GBDTCategoricalPrep, build_model
from ev_purchase.splits import validate_split_manifest

logger = logging.getLogger(__name__)
HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_hex_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX_SHA256_PATTERN.match(value):
        raise ValueError(f"{name} must be a 64-character lowercase hex SHA256 string")
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_unique_string_list(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of unique non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be a non-empty list of unique non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain only non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must contain unique strings")
    return value


def _require_pandas_dtype_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a valid non-empty pandas dtype string")
    try:
        pd.api.types.pandas_dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid non-empty pandas dtype string") from exc
    return value


def _is_valid_json_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool | str | int):
        return True
    if isinstance(value, float):
        return np.isfinite(value)
    return False


def _coerce_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def build_submission(
    sample: pd.DataFrame,
    probabilities: np.ndarray | list[float] | pd.Series,
    id_column: str = "id",
    target_column: str = "Will_Buy_EV",
) -> pd.DataFrame:
    """Build a validated submission DataFrame matching exact sample format and ID order.

    Args:
        sample: Sample submission DataFrame (must contain exact [id_column, target_column] order).
        probabilities: 1D array of predicted probabilities.
        id_column: Name of the ID column.
        target_column: Name of the target prediction column.

    Returns:
        DataFrame with exact columns [id_column, target_column] and exact sample ID ordering.

    Raises:
        TypeError: If inputs are invalid types.
        ValueError: If schema, length, values, or IDs fail validation checks.
    """
    if not isinstance(sample, pd.DataFrame):
        raise TypeError(f"sample must be a pandas DataFrame, got {type(sample).__name__}")

    if not isinstance(id_column, str) or not id_column.strip():
        raise ValueError("id_column must be a non-empty string")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("target_column must be a non-empty string")

    expected_cols = [id_column, target_column]
    if list(sample.columns) != expected_cols:
        raise ValueError(
            f"sample columns must exactly equal {expected_cols} in order, "
            f"got {list(sample.columns)}"
        )

    if sample[id_column].isna().any():
        raise ValueError(f"sample '{id_column}' contains null/missing values")
    if not sample[id_column].is_unique:
        raise ValueError(f"sample '{id_column}' contains duplicate values")

    probs_arr = np.asarray(probabilities, dtype=float)
    if probs_arr.ndim != 1:
        raise ValueError(f"probabilities must be a 1D array, got shape {probs_arr.shape}")

    if len(probs_arr) != len(sample):
        raise ValueError(
            f"probabilities length mismatch: got {len(probs_arr)}, expected {len(sample)}"
        )

    if not np.all(np.isfinite(probs_arr)):
        raise ValueError("probabilities contain non-finite values (NaN or Infinity)")

    if np.any(probs_arr < 0.0) or np.any(probs_arr > 1.0):
        raise ValueError("probabilities contain values outside range [0.0, 1.0]")

    submission = pd.DataFrame(
        {
            id_column: sample[id_column].values,
            target_column: probs_arr,
        }
    )
    return submission


def _compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_combined_sha256(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def _get_package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
    }
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    return versions


def _check_symlinks(path: Path) -> None:
    """Inspect unresolved path and every existing parent/component for symlinks."""
    curr = path
    while True:
        if curr.is_symlink() or (os.path.exists(curr) and os.path.islink(curr)):
            raise ValueError(f"Safe path policy violation: symlink detected at {curr}")
        if curr == curr.parent:
            break
        curr = curr.parent


def _resolve_project_and_output_roots(
    config_path: Path,
    project_root: str | Path | None,
    output_root: str | Path | None,
    config_outputs_root: Path,
) -> tuple[Path, Path]:
    """Strictly resolve project root and output root without any safety bypass."""
    _check_symlinks(config_path)

    if project_root is not None:
        raw_root = Path(project_root)
        _check_symlinks(raw_root)
        root_p = raw_root.resolve()
    else:
        cfg_p = config_path.resolve()
        curr = cfg_p.parent
        found_root = None
        while curr != curr.parent:
            if (curr / "pyproject.toml").exists() or (curr / "configs" / "final.yaml").exists():
                found_root = curr
                break
            curr = curr.parent
        root_p = found_root if found_root is not None else cfg_p.parent.parent

    _check_symlinks(root_p)

    if output_root is not None:
        raw_out = Path(output_root)
        _check_symlinks(raw_out)
        out_base = raw_out.resolve()
    else:
        raw_out = root_p / config_outputs_root
        _check_symlinks(raw_out)
        out_base = raw_out.resolve()

    if out_base.exists() and not out_base.is_dir():
        raise ValueError(f"Safe output-root policy violation: not a directory: {out_base}")

    if out_base == Path("/").resolve():
        raise ValueError("Safe output-root policy violation: filesystem root disallowed")
    if out_base == Path.home().resolve():
        raise ValueError("Safe output-root policy violation: user home directory disallowed")
    if out_base == root_p:
        raise ValueError("Safe output-root policy violation: project root disallowed")

    try:
        out_base.relative_to(root_p)
    except ValueError:
        raise ValueError(
            f"Safe output-root policy violation: output path {out_base} "
            f"escapes project root {root_p}"
        ) from None

    return root_p, out_base


def load_final_model(model_dir: str | Path) -> dict[str, Any]:
    """Load and reconstruct native LightGBM model and preprocessing transformer from directory.

    Args:
        model_dir: Directory containing manifest.json, model.txt, and prep_state.json.

    Returns:
        Dict containing loaded 'booster', fitted 'prep' transformer, and 'manifest'.
    """
    dir_p = Path(model_dir)
    _check_symlinks(dir_p)

    manifest_p = dir_p / "manifest.json"
    model_txt_p = dir_p / "model.txt"
    prep_json_p = dir_p / "prep_state.json"

    _check_symlinks(manifest_p)
    _check_symlinks(model_txt_p)
    _check_symlinks(prep_json_p)

    if not manifest_p.exists() or not model_txt_p.exists() or not prep_json_p.exists():
        raise FileNotFoundError(f"Missing required model artifacts in directory: {dir_p}")

    try:
        manifest_p.resolve().relative_to(dir_p.resolve())
        model_txt_p.resolve().relative_to(dir_p.resolve())
        prep_json_p.resolve().relative_to(dir_p.resolve())
    except ValueError:
        raise ValueError(f"Artifact path escapes model_dir {dir_p}") from None

    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)

    required_manifest_keys = {
        "schema_version",
        "created_at",
        "policy",
        "model",
        "feature_policy",
        "feature_columns",
        "feature_dtypes",
        "trusted_boundary_statement",
        "loader_instructions",
        "provenance",
        "package_versions",
        "artifact_hashes",
    }
    missing_keys = required_manifest_keys - set(manifest.keys())
    if missing_keys:
        raise ValueError(f"Manifest missing required keys: {sorted(missing_keys)}")

    if manifest.get("schema_version") != "1.0":
        raise ValueError(
            f"Unsupported manifest schema_version: {manifest.get('schema_version')}"
        )

    policy_block = _require_mapping(manifest.get("policy"), "Manifest policy")
    if policy_block.get("type") != "single_model":
        raise ValueError("Manifest policy.type must be 'single_model'")
    if policy_block.get("status") != "real_materialization_completed":
        raise ValueError("Manifest policy.status must be 'real_materialization_completed'")
    _require_non_empty_string(
        policy_block.get("best_single_candidate_id"),
        "Manifest policy.best_single_candidate_id",
    )

    model_block = _require_mapping(manifest.get("model"), "Manifest model")
    if model_block.get("family") != "lightgbm":
        raise ValueError("Manifest model.family must be 'lightgbm'")
    _require_mapping(model_block.get("params"), "Manifest model.params")
    if "seed" not in model_block or not isinstance(model_block.get("seed"), int):
        raise ValueError("Manifest model.seed must be an integer")

    provenance_block = _require_mapping(manifest.get("provenance"), "Manifest provenance")
    for prov_key in ("config_sha256", "data_sha256", "split_sha256"):
        _require_hex_sha256(provenance_block.get(prov_key), f"Manifest provenance.{prov_key}")

    feature_columns = _require_unique_string_list(
        manifest.get("feature_columns"),
        "feature_columns",
    )

    feature_dtypes = _require_mapping(manifest.get("feature_dtypes"), "feature_dtypes")
    if list(feature_dtypes.keys()) != feature_columns:
        raise ValueError("feature_dtypes keys must exactly equal feature_columns")
    for col in feature_columns:
        _require_pandas_dtype_string(feature_dtypes[col], f"feature_dtypes['{col}']")

    art_hashes = manifest.get("artifact_hashes", {})
    missing_art_keys = {"model.txt", "prep_state.json"} - set(art_hashes.keys())
    if not isinstance(art_hashes, dict) or missing_art_keys:
        raise ValueError("Manifest artifact_hashes missing mandatory model.txt or prep_state.json")

    for k, h_val in art_hashes.items():
        if not isinstance(h_val, str) or not HEX_SHA256_PATTERN.match(h_val):
            raise ValueError(f"Artifact hash for '{k}' must be 64-character lowercase hex string")

    actual_model_sha = _compute_file_sha256(model_txt_p)
    if actual_model_sha != art_hashes["model.txt"]:
        raise ValueError(
            f"Artifact SHA256 checksum mismatch for model.txt: "
            f"expected {art_hashes['model.txt']}, got {actual_model_sha}"
        )

    actual_prep_sha = _compute_file_sha256(prep_json_p)
    if actual_prep_sha != art_hashes["prep_state.json"]:
        raise ValueError(
            f"Artifact SHA256 checksum mismatch for prep_state.json: "
            f"expected {art_hashes['prep_state.json']}, got {actual_prep_sha}"
        )

    with open(prep_json_p, encoding="utf-8") as f:
        prep_state = json.load(f)

    required_prep_keys = {"num_cols", "cat_cols", "categories"}
    if not required_prep_keys.issubset(prep_state.keys()):
        missing = sorted(required_prep_keys - set(prep_state.keys()))
        raise ValueError(f"prep_state.json missing required keys: {missing}")

    num_cols = _require_unique_string_list(
        prep_state["num_cols"],
        "prep_state num_cols",
        allow_empty=True,
    )
    cat_cols = _require_unique_string_list(
        prep_state["cat_cols"],
        "prep_state cat_cols",
        allow_empty=True,
    )
    if not set(num_cols).isdisjoint(set(cat_cols)):
        raise ValueError("prep_state num_cols and cat_cols must be disjoint")
    if set(num_cols) | set(cat_cols) != set(feature_columns):
        raise ValueError("prep_state num_cols and cat_cols must partition feature_columns")

    categories = prep_state["categories"]
    if not isinstance(categories, dict) or list(categories.keys()) != cat_cols:
        raise ValueError("prep_state categories keys must match cat_cols")

    for c_name, c_list in categories.items():
        if not isinstance(c_list, list):
            raise ValueError(f"prep_state categories for '{c_name}' must be a list")
        if any(not _is_valid_json_scalar(item) for item in c_list):
            raise ValueError(
                f"prep_state categories for '{c_name}' must contain valid scalar values"
            )
        normalized = [json.dumps(item, sort_keys=True) for item in c_list]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"prep_state categories for '{c_name}' must not contain duplicates")

    prep = GBDTCategoricalPrep()
    prep.num_cols_ = num_cols
    prep.cat_cols_ = cat_cols
    prep.categories_ = categories

    booster = lgb.Booster(model_file=str(model_txt_p))

    return {
        "booster": booster,
        "prep": prep,
        "manifest": manifest,
        "feature_columns": feature_columns,
        "feature_dtypes": feature_dtypes,
    }


def predict_final_model(model_dir: str | Path, X: pd.DataFrame) -> np.ndarray:
    """Generate 1D probability predictions using loaded native model and preprocessing.

    Args:
        model_dir: Directory containing native model artifacts.
        X: Feature DataFrame matching training feature columns.

    Returns:
        1D numpy array of predicted probabilities.
    """
    model_data = load_final_model(model_dir)
    prep: GBDTCategoricalPrep = model_data["prep"]
    booster: lgb.Booster = model_data["booster"]
    expected_cols = model_data["feature_columns"]
    expected_dtypes = model_data["feature_dtypes"]

    if list(X.columns) != expected_cols:
        raise ValueError(
            f"Column mismatch: expected exact column name & order {expected_cols}, "
            f"got {list(X.columns)}"
        )

    for col, expected_dtype_str in expected_dtypes.items():
        actual_dtype_str = str(X[col].dtype)
        if actual_dtype_str != expected_dtype_str:
            raise ValueError(
                f"Feature dtype mismatch for column '{col}': "
                f"expected exact dtype '{expected_dtype_str}', got '{actual_dtype_str}'"
            )

    X_prep = prep.transform(X)
    probs = booster.predict(X_prep)
    probs_arr = np.asarray(probs, dtype=float)

    if probs_arr.ndim != 1 or len(probs_arr) != len(X):
        raise ValueError("Loaded model produced invalid probability output shape")
    if not np.all(np.isfinite(probs_arr)) or np.any(probs_arr < 0.0) or np.any(probs_arr > 1.0):
        raise ValueError("Loaded model produced out-of-range or non-finite probabilities")

    return probs_arr


def validate_final_config_and_provenance(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate locked final configuration, candidate metadata, split manifest, and data provenance.

    Args:
        config_path: Path to locked final.yaml configuration file.
        project_root: Repository root path (optional).

    Returns:
        Dict containing validated configuration details, policy, and candidate info.

    Raises:
        ValueError / FileNotFoundError: If any validation or hash check fails.
    """
    raw_cfg_path = Path(config_path)
    _check_symlinks(raw_cfg_path)
    cfg_p = raw_cfg_path.resolve()

    with open(cfg_p, encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f)

    out_cfg_str = raw_yaml.get("outputs", {}).get("root", "outputs")
    root_p, out_base = _resolve_project_and_output_roots(
        config_path=cfg_p,
        project_root=project_root,
        output_root=None,
        config_outputs_root=Path(out_cfg_str),
    )

    config = load_config(cfg_p, project_root=root_p)

    policy = _require_mapping(raw_yaml.get("policy"), "Policy")
    status = policy.get("status")
    if status != "real_materialization_completed":
        raise ValueError(
            f"Locked policy status must be 'real_materialization_completed', got '{status}'"
        )

    policy_type = policy.get("type")
    if policy_type != "single_model":
        raise ValueError(f"Only 'single_model' policy type is supported, got '{policy_type}'")

    candidate_id = policy.get("best_single_candidate_id")
    _require_non_empty_string(candidate_id, "Policy best_single_candidate_id")

    source_run_ids = _require_unique_string_list(
        policy.get("source_run_ids"),
        "Policy source_run_ids",
    )
    winning_weights = _require_mapping(policy.get("winning_weights"), "Policy winning_weights")
    if not winning_weights or candidate_id not in winning_weights:
        raise ValueError("Policy winning_weights missing selected candidate")

    selected_weight = _coerce_float(
        winning_weights[candidate_id],
        f"Policy winning_weights[{candidate_id}]",
    )
    if selected_weight != 1.0:
        raise ValueError(
            f"Selected candidate weight must be 1.0, got {winning_weights[candidate_id]}"
        )

    for k, w in winning_weights.items():
        _require_non_empty_string(k, "Policy winning_weights key")
        weight = _coerce_float(w, f"Policy winning_weights[{k}]")
        if k != candidate_id and weight != 0.0:
            raise ValueError(f"Non-selected candidate '{k}' weight must be 0.0, got {w}")

    components = _require_mapping(policy.get("components"), "Policy components")
    if set(winning_weights.keys()) - set(components.keys()):
        raise ValueError("Policy winning_weights keys must exist in components")
    cand_info = _require_mapping(components.get(candidate_id), f"Policy component {candidate_id}")
    if not cand_info:
        raise ValueError(f"Candidate '{candidate_id}' not found in policy components")

    run_id = _require_non_empty_string(cand_info.get("run_id"), "Component run_id")
    if cand_info.get("candidate_id") != candidate_id:
        raise ValueError("Component candidate_id mismatch")

    active_component_run_ids: list[str] = []
    for component_id, weight_value in winning_weights.items():
        if _coerce_float(weight_value, f"Policy winning_weights[{component_id}]") > 0.0:
            component = _require_mapping(
                components.get(component_id),
                f"Policy component {component_id}",
            )
            active_component_run_ids.append(
                _require_non_empty_string(
                    component.get("run_id"),
                    f"Component {component_id} run_id",
                )
            )
    if set(source_run_ids) != set(active_component_run_ids) or len(source_run_ids) != len(
        set(active_component_run_ids)
    ):
        msg = (
            f"source_run_ids {source_run_ids} mismatch active component run_ids "
            f"{active_component_run_ids}"
        )
        raise ValueError(msg)

    model_family = cand_info.get("model_family")
    if model_family != "lightgbm":
        raise ValueError(f"Final model family must be 'lightgbm', got '{model_family}'")

    model_params = cand_info.get("params", {})
    if not isinstance(model_params, dict):
        raise ValueError("Candidate parameters must be a mapping")
    if not model_params:
        raise ValueError("Candidate parameters cannot be empty")

    tuning_study = cand_info.get("tuning_study")
    trial_number = cand_info.get("trial_number")
    seed_policy = cand_info.get("seed_policy")
    if not tuning_study or trial_number is None or not seed_policy:
        raise ValueError("Candidate missing tuning_study, trial_number, or seed_policy")
    _require_mapping(seed_policy, "Candidate seed_policy")

    comp_cand_meta_path = cand_info.get("candidate_metadata_path")
    if (
        not isinstance(comp_cand_meta_path, str)
        or not comp_cand_meta_path
        or ".." in comp_cand_meta_path
    ):
        raise ValueError("Component missing or invalid candidate_metadata_path")

    raw_meta_p = root_p / comp_cand_meta_path
    _check_symlinks(raw_meta_p)
    try:
        cand_meta_p = raw_meta_p.resolve()
        cand_meta_p.relative_to(root_p)
    except ValueError:
        msg = f"Candidate metadata path escapes project root: {comp_cand_meta_path}"
        raise ValueError(msg) from None

    if not cand_meta_p.exists():
        raise FileNotFoundError(f"Candidate metadata file not found: {cand_meta_p}")

    with open(cand_meta_p, encoding="utf-8") as f:
        cand_meta = json.load(f)

    cand_meta = _require_mapping(cand_meta, "Candidate metadata")
    if cand_meta.get("candidate_id") != candidate_id:
        raise ValueError("Candidate metadata candidate_id mismatch")
    if cand_meta.get("run_id") != run_id:
        raise ValueError("Candidate metadata run_id mismatch")
    if cand_meta.get("status") != "completed":
        raise ValueError("Candidate metadata status is not completed")
    if cand_meta.get("protocol") != "full":
        raise ValueError("Candidate metadata protocol is not full")
    if cand_meta.get("dry_run") is not False:
        raise ValueError("Candidate metadata indicates dry_run is not False")
    if cand_meta.get("model_family") != model_family:
        raise ValueError("Candidate metadata model_family mismatch")
    if cand_meta.get("params") != model_params:
        raise ValueError("Candidate metadata params mismatch")
    if (
        cand_meta.get("tuning_study") != tuning_study
        or cand_meta.get("trial_number") != trial_number
    ):
        raise ValueError("Candidate metadata tuning_study or trial_number mismatch")
    if cand_meta.get("seed_policy") != seed_policy:
        raise ValueError("Candidate metadata seed_policy mismatch")

    base_seed = config.project.seed
    cfg_feature_policy = raw_yaml.get("feature_policy", "base")
    cand_feature_policy = cand_info.get("feature_policy", "base")
    if (
        cfg_feature_policy != cand_feature_policy
        or cand_meta.get("feature_policy") != cfg_feature_policy
    ):
        raise ValueError("Feature policy mismatch")

    # Fail-closed hash and split manifest validation
    top_data_sha = _require_hex_sha256(policy.get("data_sha256"), "Policy data_sha256")
    top_split_sha = _require_hex_sha256(policy.get("split_sha256"), "Policy split_sha256")

    cand_prov = _require_mapping(cand_info.get("provenance"), "Component provenance")
    cand_data_sha = _require_hex_sha256(
        cand_prov.get("data_sha256"),
        "Component provenance data_sha256",
    )
    cand_split_sha = _require_hex_sha256(
        cand_prov.get("split_sha256"),
        "Component provenance split_sha256",
    )
    if cand_data_sha != top_data_sha or cand_split_sha != top_split_sha:
        raise ValueError("Component provenance hashes do not match top-level policy hashes")

    cand_meta_prov = _require_mapping(cand_meta.get("provenance"), "Candidate metadata provenance")
    cand_meta_data_sha = _require_hex_sha256(
        cand_meta_prov.get("data_sha256"),
        "Candidate metadata provenance data_sha256",
    )
    cand_meta_split_sha = _require_hex_sha256(
        cand_meta_prov.get("split_sha256"),
        "Candidate metadata provenance split_sha256",
    )
    if cand_meta_data_sha != top_data_sha or cand_meta_split_sha != top_split_sha:
        raise ValueError(
            "Candidate metadata provenance hashes do not match policy/component hashes"
        )

    data_sha256_actual = _compute_combined_sha256([config.data.train, config.data.test])
    if data_sha256_actual != top_data_sha:
        raise ValueError(
            f"Data SHA256 mismatch! Policy: {top_data_sha}, Actual data: {data_sha256_actual}"
        )

    raw_split_manifest_p = out_base / "splits" / "repeated_stratified_5x3.parquet"
    _check_symlinks(raw_split_manifest_p)
    try:
        split_manifest_p = raw_split_manifest_p.resolve()
        split_manifest_p.relative_to(out_base)
    except ValueError:
        raise ValueError(
            f"Split manifest path escapes output root {out_base}: {raw_split_manifest_p}"
        ) from None

    if not split_manifest_p.exists():
        raise FileNotFoundError(f"Split manifest file not found: {split_manifest_p}")

    split_sha256_actual = _compute_file_sha256(split_manifest_p)
    if split_sha256_actual != top_split_sha:
        raise ValueError(
            f"Split SHA256 mismatch! Policy: {top_split_sha}, Actual split: {split_sha256_actual}"
        )

    # Load datasets
    train_df, test_df = load_frames(config.data.train, config.data.test)
    sample_df = pd.read_csv(config.data.sample_submission)
    contract = DatasetContract(
        id_column=config.data.id_column, target_column=config.data.target_column
    )
    validate_frames(train_df, test_df, contract)
    y_encoded = encode_target(train_df[config.data.target_column], contract)

    # Validate candidate fold / row counts
    if cand_meta.get("effective_n_rows") != len(train_df):
        raise ValueError("Candidate metadata effective_n_rows mismatch with train_df length")
    if cand_meta.get("effective_n_splits") != config.validation.n_splits:
        raise ValueError("Candidate metadata effective_n_splits mismatch with validation config")
    if cand_meta.get("effective_n_repeats") != config.validation.n_repeats:
        raise ValueError("Candidate metadata effective_n_repeats mismatch with validation config")

    expected_eval_folds = list(range(config.validation.n_splits * config.validation.n_repeats))
    if cand_meta.get("evaluated_fold_ids") != expected_eval_folds:
        raise ValueError("Candidate metadata evaluated_fold_ids mismatch with expected fold IDs")

    # Validate candidate artifact paths and checksums
    cand_art_paths = _require_mapping(cand_meta.get("artifact_paths"), "Candidate artifact_paths")
    cand_art_checksums = _require_mapping(
        cand_meta.get("artifact_checksums"),
        "Candidate artifact_checksums",
    )
    mandatory_art_keys = {"oof_predictions", "test_predictions"}
    if not mandatory_art_keys.issubset(cand_art_paths.keys()) or not mandatory_art_keys.issubset(
        cand_art_checksums.keys()
    ):
        msg = "Candidate metadata missing mandatory artifact_paths or artifact_checksums"
        raise ValueError(msg)

    if set(cand_art_paths.keys()) != set(cand_art_checksums.keys()):
        raise ValueError(
            "Candidate metadata artifact_paths keys must match artifact_checksums keys"
        )

    for art_key, art_rel_str in cand_art_paths.items():
        if (
            not isinstance(art_rel_str, str)
            or not art_rel_str.strip()
            or Path(art_rel_str).is_absolute()
            or ".." in art_rel_str
        ):
            raise ValueError(f"Invalid artifact path for {art_key}: {art_rel_str} must be relative")

        raw_art_p = root_p / art_rel_str
        _check_symlinks(raw_art_p)
        try:
            resolved_art_p = raw_art_p.resolve()
            resolved_art_p.relative_to(root_p)
        except ValueError:
            msg = f"Candidate artifact path escapes project root: {art_rel_str}"
            raise ValueError(msg) from None

        if not resolved_art_p.exists():
            raise FileNotFoundError(f"Candidate artifact file not found: {resolved_art_p}")

        expected_hash = cand_art_checksums[art_key]
        if not isinstance(expected_hash, str) or not HEX_SHA256_PATTERN.match(expected_hash):
            msg = f"Candidate artifact checksum for {art_key} must be 64-char lowercase hex"
            raise ValueError(msg)

        actual_hash = _compute_file_sha256(resolved_art_p)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Candidate artifact checksum mismatch for {art_key}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    # Validate split manifest with target labels
    manifest = pd.read_parquet(split_manifest_p)
    validate_split_manifest(
        manifest=manifest,
        n_rows=len(train_df),
        n_splits=config.validation.n_splits,
        n_repeats=config.validation.n_repeats,
        y=y_encoded,
    )

    # ID sequence invariant check
    id_col = config.data.id_column
    if not test_df[id_col].equals(sample_df[id_col]):
        raise ValueError(
            "test_df ID sequence must exactly equal sample_submission ID sequence "
            "in order and values"
        )

    return {
        "valid": True,
        "policy_type": policy_type,
        "best_single_candidate_id": candidate_id,
        "model_family": model_family,
        "params": model_params,
        "seed": base_seed,
        "feature_policy": cfg_feature_policy,
        "config": config,
        "train_df": train_df,
        "test_df": test_df,
        "sample_df": sample_df,
        "y_encoded": y_encoded,
        "split_manifest_path": split_manifest_p,
        "root_path": root_p,
        "output_base": out_base,
    }


def generate_final_model_and_submission(
    config_path: str | Path,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    _test_promotion_failure_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute final model retraining, generate submission, and write validated artifacts.

    Args:
        config_path: Path to locked final.yaml config.
        project_root: Repository root directory.
        output_root: Directory to write output subdirectories.
        _test_promotion_failure_hook: Internal test hook for testing transactional rollback.

    Returns:
        Summary dict containing submission metadata, metrics, and artifact paths.
    """
    start_time = time.time()
    val_res = validate_final_config_and_provenance(config_path, project_root=project_root)

    config = val_res["config"]
    root_p, out_base = _resolve_project_and_output_roots(
        config_path=Path(config_path),
        project_root=project_root if project_root is not None else val_res["root_path"],
        output_root=output_root,
        config_outputs_root=Path(config.outputs.root),
    )

    train_df = val_res["train_df"]
    test_df = val_res["test_df"]
    sample_df = val_res["sample_df"]
    y_encoded = val_res["y_encoded"]
    model_family = val_res["model_family"]
    base_seed = val_res["seed"]
    model_params = val_res["params"]
    cfg_feature_policy = val_res["feature_policy"]
    candidate_id = val_res["best_single_candidate_id"]
    policy_type = val_res["policy_type"]
    config_sha256 = _compute_file_sha256(Path(config_path).resolve())
    top_data_sha = _compute_combined_sha256([config.data.train, config.data.test])
    split_manifest_p = val_res["split_manifest_path"]
    top_split_sha = _compute_file_sha256(split_manifest_p)

    X_train_raw = train_df.drop(columns=[config.data.id_column, config.data.target_column])
    X_test_raw = test_df.drop(columns=[config.data.id_column])

    X_train = apply_feature_policy(cfg_feature_policy, X_train_raw)
    X_test = apply_feature_policy(cfg_feature_policy, X_test_raw)

    # Retrain on full train dataset
    full_model = build_model(
        model_name=model_family,
        seed=base_seed,
        sample_frame=X_train,
        params=model_params,
    )
    full_model.fit(X_train, y_encoded)

    test_probs = full_model.predict_proba(X_test)[:, 1]
    validate_probabilities(test_probs)

    submission_df = build_submission(
        sample=sample_df,
        probabilities=test_probs,
        id_column=config.data.id_column,
        target_column=config.data.target_column,
    )

    out_base.mkdir(parents=True, exist_ok=True)
    target_sub_dir = out_base / "submissions"
    target_model_dir = out_base / "final_model"

    _check_symlinks(target_sub_dir)
    _check_symlinks(target_model_dir)

    token = uuid4().hex[:8]
    staging_sub_dir = out_base / f".staging_{token}_submissions"
    staging_model_dir = out_base / f".staging_{token}_final_model"
    backup_sub_dir = out_base / f".backup_{token}_submissions"
    backup_model_dir = out_base / f".backup_{token}_final_model"

    staging_sub_dir.mkdir(parents=True, exist_ok=True)
    staging_model_dir.mkdir(parents=True, exist_ok=True)

    sub_backed_up = False
    model_backed_up = False
    sub_promoted = False
    model_promoted = False

    try:
        # Save native LightGBM booster + prep state (NO joblib/pickle)
        clf_step = full_model.named_steps["clf"]
        prep_step = full_model.named_steps["prep"]

        model_txt_path = staging_model_dir / "model.txt"
        clf_step.booster_.save_model(str(model_txt_path))
        txt_sha = _compute_file_sha256(model_txt_path)

        prep_json_path = staging_model_dir / "prep_state.json"
        prep_data = {
            "num_cols": prep_step.num_cols_,
            "cat_cols": prep_step.cat_cols_,
            "categories": prep_step.categories_,
        }
        with open(prep_json_path, "w", encoding="utf-8") as f:
            json.dump(prep_data, f, indent=2)
        prep_sha = _compute_file_sha256(prep_json_path)

        artifact_hashes = {
            "model.txt": txt_sha,
            "prep_state.json": prep_sha,
        }

        manifest_data = {
            "schema_version": "1.0",
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "policy": {
                "type": policy_type,
                "status": "real_materialization_completed",
                "best_single_candidate_id": candidate_id,
            },
            "model": {
                "family": model_family,
                "params": model_params,
                "seed": base_seed,
            },
            "feature_policy": cfg_feature_policy,
            "feature_columns": list(X_train.columns),
            "feature_dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()},
            "trusted_boundary_statement": (
                "Local project runtime boundary. Model saved natively via LightGBM booster text "
                "format and deterministic JSON preprocessing state without arbitrary Python pickle."
            ),
            "loader_instructions": (
                "Load via ev_purchase.submission.load_final_model(model_dir) and "
                "predict via ev_purchase.submission.predict_final_model(model_dir, X)."
            ),
            "provenance": {
                "config_sha256": config_sha256,
                "data_sha256": top_data_sha,
                "split_sha256": top_split_sha,
            },
            "package_versions": _get_package_versions(),
            "artifact_hashes": artifact_hashes,
        }

        manifest_path = staging_model_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        artifact_hashes["manifest.json"] = _compute_file_sha256(manifest_path)

        # Write submission CSV
        sub_file_path = staging_sub_dir / "submission.csv"
        submission_df.to_csv(sub_file_path, index=False)
        sub_bytes = sub_file_path.stat().st_size
        sub_sha256 = _compute_file_sha256(sub_file_path)

        runtime_sec = round(time.time() - start_time, 4)

        id_col = config.data.id_column
        # Pre-promotion verification from staging file
        saved_sub = pd.read_csv(sub_file_path)
        if list(saved_sub.columns) != [config.data.id_column, config.data.target_column]:
            raise RuntimeError("Saved submission columns mismatch expected order")
        if len(saved_sub) != len(sample_df):
            raise RuntimeError("Saved submission row count mismatch")

        id_seq_match_sample = bool(saved_sub[id_col].equals(sample_df[id_col]))
        id_seq_match_test = bool(saved_sub[id_col].equals(test_df[id_col]))

        if not id_seq_match_sample or not id_seq_match_test:
            raise RuntimeError("Saved submission ID sequence does not match test and sample")

        saved_probs = saved_sub[config.data.target_column].to_numpy()
        finite_check = bool(np.all(np.isfinite(saved_probs)))
        range_check = bool(np.all(saved_probs >= 0.0) and np.all(saved_probs <= 1.0))
        unique_check = bool(saved_sub[id_col].is_unique)
        null_check = bool(not saved_sub.isna().any().any())

        if not finite_check or not range_check or not unique_check or not null_check:
            raise RuntimeError("Saved submission data invariants check failed")

        if not np.allclose(saved_probs, test_probs, atol=1e-5):
            raise RuntimeError("Saved probabilities float precision mismatch")

        report_data = {
            "row_count": len(saved_sub),
            "columns": list(saved_sub.columns),
            "id_column": config.data.id_column,
            "target_column": config.data.target_column,
            "id_sequence_match_submission_vs_test": id_seq_match_test,
            "id_sequence_match_submission_vs_sample": id_seq_match_sample,
            "id_order_match": id_seq_match_sample,
            "uniqueness_check": unique_check,
            "finite_check": finite_check,
            "range_check": range_check,
            "null_check": null_check,
            "target_stats": {
                "min": float(np.min(saved_probs)),
                "max": float(np.max(saved_probs)),
                "mean": float(np.mean(saved_probs)),
                "std": float(np.std(saved_probs)),
            },
            "sha256": sub_sha256,
            "bytes": sub_bytes,
            "provenance": {
                "config_sha256": config_sha256,
                "data_sha256": top_data_sha,
                "split_sha256": top_split_sha,
                "candidate_id": candidate_id,
                "policy_type": policy_type,
                "model_artifact_sha256": artifact_hashes,
            },
            "package_versions": _get_package_versions(),
            "runtime_seconds": runtime_sec,
        }

        report_path = staging_sub_dir / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 1. Backup Phase for BOTH target directories
        if target_sub_dir.exists():
            target_sub_dir.rename(backup_sub_dir)
            sub_backed_up = True

        if target_model_dir.exists():
            target_model_dir.rename(backup_model_dir)
            model_backed_up = True

        # 2. Promotion Phase for BOTH target directories
        staging_sub_dir.rename(target_sub_dir)
        sub_promoted = True

        if _test_promotion_failure_hook is not None:
            _test_promotion_failure_hook()

        staging_model_dir.rename(target_model_dir)
        model_promoted = True

        # 3. Post-promotion verification
        promoted_csv = target_sub_dir / "submission.csv"
        if not promoted_csv.exists() or _compute_file_sha256(promoted_csv) != sub_sha256:
            raise RuntimeError("Post-promotion verification failed for submission.csv")

        promoted_manifest = target_model_dir / "manifest.json"
        if not promoted_manifest.exists():
            raise RuntimeError("Post-promotion verification failed for manifest.json")

        # 4. Irrevocable Success Cleanup (Cleanup errors MUST NOT enter rollback!)
        if sub_backed_up and backup_sub_dir.exists():
            try:
                shutil.rmtree(backup_sub_dir)
            except Exception as exc:
                logger.warning("Failed to clean up backup directory %s: %s", backup_sub_dir, exc)

        if model_backed_up and backup_model_dir.exists():
            try:
                shutil.rmtree(backup_model_dir)
            except Exception as exc:
                logger.warning("Failed to clean up backup directory %s: %s", backup_model_dir, exc)

    except Exception:
        # Atomic Rollback of BOTH directories to exact prior state ONLY BEFORE COMMIT
        if sub_promoted and target_sub_dir.exists():
            shutil.rmtree(target_sub_dir, ignore_errors=True)
        if sub_backed_up and backup_sub_dir.exists():
            if target_sub_dir.exists():
                shutil.rmtree(target_sub_dir, ignore_errors=True)
            backup_sub_dir.rename(target_sub_dir)

        if model_promoted and target_model_dir.exists():
            shutil.rmtree(target_model_dir, ignore_errors=True)
        if model_backed_up and backup_model_dir.exists():
            if target_model_dir.exists():
                shutil.rmtree(target_model_dir, ignore_errors=True)
            backup_model_dir.rename(target_model_dir)

        if staging_sub_dir.exists():
            shutil.rmtree(staging_sub_dir, ignore_errors=True)
        if staging_model_dir.exists():
            shutil.rmtree(staging_model_dir, ignore_errors=True)
        raise

    return {
        "submission_path": target_sub_dir / "submission.csv",
        "validation_report_path": target_sub_dir / "validation_report.json",
        "final_model_dir": target_model_dir,
        "row_count": len(submission_df),
        "target_stats": report_data["target_stats"],
        "sha256": sub_sha256,
        "runtime_seconds": runtime_sec,
    }
