"""Targeted-feature experiment harness and evaluation module.

Provides a reusable framework for single-feature policies, starting with
the sequential Concern×Subsidy LightGBM experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ev_purchase.data import DatasetContract, encode_target
from ev_purchase.leaderboard import compute_corrected_resampled_ttest
from ev_purchase.models import build_model
from ev_purchase.severe_fn_audit import (
    compute_sample_log_loss,
    reconstruct_mask_from_predicate,
    select_deterministic_top_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

THREAT_MODEL_STATEMENT = (
    "Threat Model: Local integrity verification protects against accidental data corruption, "
    "independent file tampering, stale artifacts, and configuration drift (such as modified "
    "hyperparameters, seed divergence, or changed split/dataset files). It does NOT guarantee "
    "cryptographic authenticity against coordinated malicious adversaries capable of rewriting "
    "both the candidate artifacts and provenance records simultaneously without digital signatures."
)


def _atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Write text atomically to destination path using a temp file and os.replace."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=p.parent,
        prefix=f".{p.name}.tmp_",
        delete=False,
        encoding=encoding,
    )
    temp_path = Path(temp.name)
    try:
        temp.write(text)
        temp.flush()
        os.fsync(temp.fileno())
        temp.close()
        os.replace(temp_path, p)
    except Exception:
        temp.close()
        if temp_path.exists():
            temp_path.unlink()
        raise


def _atomic_write_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Write JSON data atomically to destination path."""
    serialized = json.dumps(data, indent=indent)
    _atomic_write_text(path, serialized)


def _atomic_write_parquet(path: Path | str, df: pd.DataFrame, **kwargs: Any) -> None:
    """Write pandas DataFrame to Parquet atomically."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        dir=p.parent,
        prefix=f".{p.name}.tmp_",
        delete=False,
        suffix=".parquet",
    )
    temp_path = Path(temp.name)
    temp.close()
    try:
        df.to_parquet(temp_path, **kwargs)
        os.replace(temp_path, p)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _atomic_write_csv(path: Path | str, df: pd.DataFrame, **kwargs: Any) -> None:
    """Write pandas DataFrame to CSV atomically."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        dir=p.parent,
        prefix=f".{p.name}.tmp_",
        delete=False,
        suffix=".csv",
    )
    temp_path = Path(temp.name)
    temp.close()
    try:
        df.to_csv(temp_path, **kwargs)
        os.replace(temp_path, p)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _rel_path_or_str(path: Path | str) -> str:
    """Format path relative to project root if within workspace, else string."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class TargetedFeatureContract:
    """Specification contract for a targeted single-feature policy."""

    feature_name: str
    column_name: str
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame]
    description: str
    family: str = "targeted_feature_policy"
    experiment_index: int = 1
    parent_columns: tuple[str, ...] = ("Environmental_Concern_Level", "Subsidy_Available")
    formula: str = ""


class TargetedFeatureProvenanceError(ValueError):
    """Base error for targeted feature experiment provenance violations."""


class LegacyProvenanceError(TargetedFeatureProvenanceError):
    """Raised when an unverified legacy candidate artifact lacks a training provenance record."""


class ProvenanceMismatchError(TargetedFeatureProvenanceError):
    """Raised when candidate training provenance fails contract verification."""


class ArtifactTamperError(TargetedFeatureProvenanceError):
    """Raised when an artifact checksum or record count differs from its provenance record."""


@dataclass(frozen=True)
class ExperimentDecision:
    """Container for GO/BORDERLINE/NO-GO gate decision."""

    decision: str  # 'GO', 'BORDERLINE', 'NO-GO'
    reason: str
    details: dict[str, Any]


def add_concern_subsidy_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exactly one categorical feature: Concern_Subsidy.

    Concern_Subsidy = Environmental_Concern_Level.astype(str) + '__' + Subsidy_Available
    Retains all parent columns and original rows without mutating input.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    required_parents = ["Environmental_Concern_Level", "Subsidy_Available"]
    for col in required_parents:
        if col not in frame.columns:
            cols = list(frame.columns)
            raise ValueError(f"Required parent column '{col}' is missing from columns: {cols}")

    df = frame.copy()
    df["Concern_Subsidy"] = (
        df["Environmental_Concern_Level"].astype(str) + "__" + df["Subsidy_Available"].astype(str)
    )
    return df


CONCERN_SUBSIDY_CONTRACT = TargetedFeatureContract(
    feature_name="concern_subsidy",
    column_name="Concern_Subsidy",
    transform_fn=add_concern_subsidy_feature,
    description="Cross-feature between Environmental_Concern_Level and Subsidy_Available",
    family="targeted_feature_policy",
    experiment_index=1,
    parent_columns=("Environmental_Concern_Level", "Subsidy_Available"),
    formula="Environmental_Concern_Level.astype(str) + '__' + Subsidy_Available",
)


def add_charging_access_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exactly one numeric feature: Charging_Access.

    Formula: 2 * (Home_Charging_Possible == 'Yes') + Near_Home + Near_Work.
    Retains all parent columns and original rows without mutating input.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    required_parents = [
        "Home_Charging_Possible",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
    ]
    for col in required_parents:
        if col not in frame.columns:
            cols = list(frame.columns)
            raise ValueError(f"Required parent column '{col}' is missing from columns: {cols}")

    df = frame.copy()
    home_ind = (df["Home_Charging_Possible"] == "Yes").astype(int)
    df["Charging_Access"] = (
        2 * home_ind + df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    )
    return df


CHARGING_ACCESS_CONTRACT = TargetedFeatureContract(
    feature_name="charging_access",
    column_name="Charging_Access",
    transform_fn=add_charging_access_feature,
    description="Charging access index combining home charging indicator and station density",
    family="targeted_feature_policy",
    experiment_index=2,
    parent_columns=(
        "Home_Charging_Possible",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
    ),
    formula=(
        "2 * (Home_Charging_Possible == 'Yes') "
        "+ Charging_Stations_Near_Home + Charging_Stations_Near_Work"
    ),
)


def add_commute_anxiety_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exactly one numeric feature: Commute_Anxiety.

    Formula: Daily_Commute_km * ordinal(Range_Anxiety_Level)
    where ordinal mapping is {'Low': 0, 'Medium': 1, 'High': 2}.
    Retains all parent columns and original rows without mutating input.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    required_parents = [
        "Daily_Commute_km",
        "Range_Anxiety_Level",
    ]
    for col in required_parents:
        if col not in frame.columns:
            cols = list(frame.columns)
            raise ValueError(f"Required parent column '{col}' is missing from columns: {cols}")

    df = frame.copy()
    anxiety_mapping = {"Low": 0, "Medium": 1, "High": 2}
    ordinal_anxiety = df["Range_Anxiety_Level"].map(anxiety_mapping)
    if ordinal_anxiety.isna().any():
        unmapped = df.loc[ordinal_anxiety.isna(), "Range_Anxiety_Level"].unique().tolist()
        raise ValueError(f"Encountered unmapped Range_Anxiety_Level values: {unmapped}")
    df["Commute_Anxiety"] = df["Daily_Commute_km"] * ordinal_anxiety
    return df


COMMUTE_ANXIETY_CONTRACT = TargetedFeatureContract(
    feature_name="commute_anxiety",
    column_name="Commute_Anxiety",
    transform_fn=add_commute_anxiety_feature,
    description="Interaction of daily commute distance and ordinal range anxiety level",
    family="targeted_feature_policy",
    experiment_index=3,
    parent_columns=(
        "Daily_Commute_km",
        "Range_Anxiety_Level",
    ),
    formula="Daily_Commute_km * ordinal(Range_Anxiety_Level)",
)


def add_income_per_vehicle_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exactly one numeric feature: Income_per_Vehicle.

    Formula: Annual_Income_USD / Number_of_Cars_Owned.
    Validates that Number_of_Cars_Owned is strictly positive (> 0) and non-null,
    and that the resulting quotient is finite. Retains both parent columns
    without mutating the input DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    required_parents = ["Annual_Income_USD", "Number_of_Cars_Owned"]
    for col in required_parents:
        if col not in frame.columns:
            cols = list(frame.columns)
            raise ValueError(f"Required parent column '{col}' is missing from columns: {cols}")

    df = frame.copy()
    cars = df["Number_of_Cars_Owned"]
    if cars.isna().any() or (cars <= 0).any():
        raise ValueError(
            "Denominator 'Number_of_Cars_Owned' must be strictly positive (> 0) and non-null."
        )

    income_per_veh = df["Annual_Income_USD"] / cars
    if not np.isfinite(income_per_veh).all():
        raise ValueError("Income_per_Vehicle calculation resulted in non-finite values.")

    df["Income_per_Vehicle"] = income_per_veh
    return df


INCOME_PER_VEHICLE_CONTRACT = TargetedFeatureContract(
    feature_name="income_per_vehicle",
    column_name="Income_per_Vehicle",
    transform_fn=add_income_per_vehicle_feature,
    description="Ratio of annual income to number of cars owned",
    family="targeted_feature_policy",
    experiment_index=4,
    parent_columns=(
        "Annual_Income_USD",
        "Number_of_Cars_Owned",
    ),
    formula="Annual_Income_USD / Number_of_Cars_Owned",
)


def add_age_income_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exactly one numeric feature: Age_Income.

    Formula: Age * Annual_Income_USD.
    Validates that parent columns 'Age' and 'Annual_Income_USD' are numeric,
    non-null, and finite, and that the resulting product is finite.
    Retains both parent columns without mutating the input DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    required_parents = ["Age", "Annual_Income_USD"]
    for col in required_parents:
        if col not in frame.columns:
            cols = list(frame.columns)
            raise ValueError(f"Required parent column '{col}' is missing from columns: {cols}")

    age_col = frame["Age"]
    income_col = frame["Annual_Income_USD"]

    if not pd.api.types.is_numeric_dtype(age_col):
        raise TypeError("Parent column 'Age' must have numeric dtype.")
    if not pd.api.types.is_numeric_dtype(income_col):
        raise TypeError("Parent column 'Annual_Income_USD' must have numeric dtype.")

    if age_col.isna().any():
        raise ValueError("Column 'Age' contains null values.")
    if income_col.isna().any():
        raise ValueError("Column 'Annual_Income_USD' contains null values.")

    if not np.isfinite(age_col).all():
        raise ValueError("Column 'Age' contains non-finite values.")
    if not np.isfinite(income_col).all():
        raise ValueError("Column 'Annual_Income_USD' contains non-finite values.")

    result = age_col * income_col
    if not np.isfinite(result).all():
        raise ValueError("Age_Income calculation resulted in non-finite values.")

    df = frame.copy()
    df["Age_Income"] = result
    return df


AGE_INCOME_CONTRACT = TargetedFeatureContract(
    feature_name="age_income",
    column_name="Age_Income",
    transform_fn=add_age_income_feature,
    description="Interaction of age and annual income (Age * Annual_Income_USD)",
    family="targeted_feature_policy",
    experiment_index=5,
    parent_columns=(
        "Age",
        "Annual_Income_USD",
    ),
    formula="Age * Annual_Income_USD",
)

TARGETED_FEATURE_REGISTRY: dict[str, TargetedFeatureContract] = {
    "concern_subsidy": CONCERN_SUBSIDY_CONTRACT,
    "charging_access": CHARGING_ACCESS_CONTRACT,
    "commute_anxiety": COMMUTE_ANXIETY_CONTRACT,
    "income_per_vehicle": INCOME_PER_VEHICLE_CONTRACT,
    "age_income": AGE_INCOME_CONTRACT,
}


def get_targeted_feature_contract(name: str) -> TargetedFeatureContract:
    """Retrieve targeted feature contract by policy name."""
    clean_name = name.strip().lower()
    if clean_name not in TARGETED_FEATURE_REGISTRY:
        supported = sorted(TARGETED_FEATURE_REGISTRY.keys())
        raise KeyError(f"Unknown targeted feature policy '{name}'. Supported policies: {supported}")
    return TARGETED_FEATURE_REGISTRY[clean_name]


def list_targeted_feature_contracts() -> list[str]:
    """List available targeted feature policy names."""
    return sorted(TARGETED_FEATURE_REGISTRY.keys())


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_combined_sha256(paths: list[Path | str]) -> str:
    """Compute combined SHA-256 hash of multiple files."""
    hasher = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def verify_provenance_and_baseline(
    expected_data_sha256: str,
    actual_data_sha256: str,
    expected_split_sha256: str,
    actual_split_sha256: str,
    expected_oof_sha256: str,
    actual_oof_sha256: str,
) -> None:
    """Verify data, split, and baseline OOF hashes match canonical specification.

    Raises:
        ValueError: If any SHA-256 hash does not match expected value.
    """
    if actual_data_sha256 != expected_data_sha256:
        raise ValueError(
            f"Data SHA256 mismatch: expected {expected_data_sha256}, got {actual_data_sha256}"
        )
    if actual_split_sha256 != expected_split_sha256:
        raise ValueError(
            f"Split SHA256 mismatch: expected {expected_split_sha256}, got {actual_split_sha256}"
        )
    if actual_oof_sha256 != expected_oof_sha256:
        raise ValueError(
            f"Baseline OOF SHA256 mismatch: expected {expected_oof_sha256}, got {actual_oof_sha256}"
        )


def verify_frozen_slices(
    frozen_slices_path: Path | str,
    expected_status: str = "post_hoc_discovered_then_prospectively_frozen",
) -> dict[str, Any]:
    """Verify frozen slices file exists, is valid, and matches prospective status.

    Raises:
        FileNotFoundError: If the frozen slices file does not exist.
        ValueError: If version is missing or status does not match expected_status.
    """
    path = Path(frozen_slices_path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen slices file not found: {path}")

    sha256 = compute_file_sha256(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("version")
    if not version:
        raise ValueError(f"Frozen slices file {path} missing 'version'")

    status = data.get("slice_provenance", {}).get("status")
    if status != expected_status:
        msg = f"Frozen slices status mismatch: expected {expected_status!r}, got {status!r}"
        raise ValueError(msg)

    return {
        "path": str(path),
        "sha256": sha256,
        "version": str(version),
        "status": str(status),
    }


def compute_params_hash(params: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of model parameters."""
    serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_training_provenance(
    feature_contract: TargetedFeatureContract,
    model_params: dict[str, Any],
    data_hash: str,
    split_hash: str,
    base_seed: int,
    candidate_oof_path: Path | str,
    candidate_oof_df: pd.DataFrame,
    source_hash: str | None = None,
    train_file_sha256: str | None = None,
    input_file_sha256: str | None = None,
    model_family: str = "lightgbm",
    provenance_version: str = "1.0",
    seed_formula: str = "base_seed + repeat * 100 + fold",
) -> dict[str, Any]:
    """Construct an immutable training provenance record for a candidate model."""
    cand_path = Path(candidate_oof_path)
    cand_sha = compute_file_sha256(cand_path)
    current_src_sha = (
        source_hash if source_hash is not None else compute_file_sha256(Path(__file__).resolve())
    )
    input_sha = input_file_sha256 if input_file_sha256 is not None else data_hash

    return {
        "provenance_version": provenance_version,
        "policy_name": feature_contract.feature_name,
        "feature_contract": {
            "family": feature_contract.family,
            "experiment_index": feature_contract.experiment_index,
            "feature_name": feature_contract.feature_name,
            "column_name": feature_contract.column_name,
            "parent_columns": list(feature_contract.parent_columns),
            "formula": feature_contract.formula,
            "description": feature_contract.description,
        },
        "model": {
            "family": model_family,
            "params": model_params,
            "params_hash": compute_params_hash(model_params),
        },
        "data_hash": data_hash,
        "input_file_sha256": input_sha,
        "train_file_sha256": train_file_sha256,
        "split_hash": split_hash,
        "seed_policy": {
            "base_seed": base_seed,
            "formula": seed_formula,
        },
        "training_source_hash": current_src_sha,
        "threat_model": THREAT_MODEL_STATEMENT,
        "candidate_oof": {
            "path": _rel_path_or_str(cand_path),
            "file_name": cand_path.name,
            "sha256": cand_sha,
            "row_count": len(candidate_oof_df),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        },
    }


def verify_training_provenance(
    training_prov: dict[str, Any],
    feature_contract: TargetedFeatureContract,
    model_params: dict[str, Any],
    expected_data_hash: str,
    expected_split_hash: str,
    expected_base_seed: int,
    candidate_oof_path: Path | str,
    candidate_oof_df: pd.DataFrame,
    expected_row_count: int | None = None,
    expected_train_file_sha256: str | None = None,
    expected_input_file_sha256: str | None = None,
    expected_model_family: str = "lightgbm",
    expected_provenance_version: str = "1.0",
    expected_seed_formula: str = "base_seed + repeat * 100 + fold",
) -> None:
    """Verify all immutable fields of candidate training provenance before reuse.

    Raises:
        ArtifactTamperError: If candidate OOF hash or row count on disk does not match record.
        ProvenanceMismatchError: If policy, contract, model params, data, split, or seed mismatch.
    """
    cand_path = Path(candidate_oof_path)
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate OOF file not found at {cand_path}")

    # 1. Candidate OOF integrity
    prov_oof = training_prov.get("candidate_oof")
    if not isinstance(prov_oof, dict):
        raise ProvenanceMismatchError("Training provenance missing 'candidate_oof' mapping")

    expected_oof_sha = prov_oof.get("sha256")
    if not expected_oof_sha:
        raise ProvenanceMismatchError("Training provenance missing candidate_oof.sha256")

    actual_oof_sha = compute_file_sha256(cand_path)
    if actual_oof_sha != expected_oof_sha:
        raise ArtifactTamperError(
            f"Candidate OOF hash mismatch: expected {expected_oof_sha} from training provenance, "
            f"got {actual_oof_sha} on disk."
        )

    prov_oof_rows = prov_oof.get("row_count")
    actual_oof_rows = len(candidate_oof_df)
    if prov_oof_rows is not None and actual_oof_rows != prov_oof_rows:
        raise ArtifactTamperError(
            f"Candidate OOF row count mismatch: expected {prov_oof_rows} from training provenance, "
            f"got {actual_oof_rows} in dataframe."
        )

    if expected_row_count is not None and actual_oof_rows != expected_row_count:
        raise ArtifactTamperError(
            f"Candidate OOF row count mismatch with dataset: expected {expected_row_count}, "
            f"got {actual_oof_rows}."
        )

    # 2. Provenance Version
    prov_ver = training_prov.get("provenance_version")
    if prov_ver != expected_provenance_version:
        raise ProvenanceMismatchError(
            f"Provenance version mismatch: expected {expected_provenance_version!r}, "
            f"got {prov_ver!r}."
        )

    # 3. Policy name
    prov_policy = training_prov.get("policy_name")
    if prov_policy != feature_contract.feature_name:
        raise ProvenanceMismatchError(
            f"Policy mismatch: training provenance specifies policy '{prov_policy}', "
            f"but current evaluation requested '{feature_contract.feature_name}'."
        )

    # 4. Feature contract
    fc_info = training_prov.get("feature_contract")
    if not isinstance(fc_info, dict):
        raise ProvenanceMismatchError("Training provenance missing 'feature_contract' mapping.")

    if fc_info.get("family") != feature_contract.family:
        raise ProvenanceMismatchError(
            f"Feature family mismatch: expected {feature_contract.family!r}, "
            f"got {fc_info.get('family')!r}."
        )

    if fc_info.get("experiment_index") != feature_contract.experiment_index:
        raise ProvenanceMismatchError(
            f"Feature experiment_index mismatch: expected {feature_contract.experiment_index!r}, "
            f"got {fc_info.get('experiment_index')!r}."
        )

    if fc_info.get("feature_name") != feature_contract.feature_name:
        raise ProvenanceMismatchError(
            f"Feature name mismatch: expected {feature_contract.feature_name!r}, "
            f"got {fc_info.get('feature_name')!r}."
        )

    prov_col = fc_info.get("column_name")
    if prov_col != feature_contract.column_name:
        raise ProvenanceMismatchError(
            f"Feature column mismatch: training provenance specifies '{prov_col}', "
            f"expected '{feature_contract.column_name}'."
        )

    if list(fc_info.get("parent_columns", [])) != list(feature_contract.parent_columns):
        prov_parents = fc_info.get("parent_columns")
        exp_parents = list(feature_contract.parent_columns)
        raise ProvenanceMismatchError(
            f"Feature parent columns mismatch: training provenance specifies "
            f"{prov_parents}, expected {exp_parents}."
        )

    if fc_info.get("formula") != feature_contract.formula:
        raise ProvenanceMismatchError(
            f"Feature formula mismatch: training provenance specifies '{fc_info.get('formula')}', "
            f"expected '{feature_contract.formula}'."
        )

    if fc_info.get("description") != feature_contract.description:
        raise ProvenanceMismatchError(
            f"Feature description mismatch: training provenance specifies "
            f"'{fc_info.get('description')}', expected '{feature_contract.description}'."
        )

    # 5. Model
    prov_model = training_prov.get("model")
    if not isinstance(prov_model, dict):
        raise ProvenanceMismatchError("Training provenance missing 'model' mapping.")

    prov_model_family = prov_model.get("family")
    if prov_model_family != expected_model_family:
        raise ProvenanceMismatchError(
            f"Model family mismatch: expected {expected_model_family!r}, got {prov_model_family!r}."
        )

    expected_params_hash = compute_params_hash(model_params)
    prov_params_hash = prov_model.get("params_hash")
    if prov_params_hash != expected_params_hash:
        raise ProvenanceMismatchError(
            f"Model parameter hash mismatch: training provenance has {prov_params_hash}, "
            f"current config requires {expected_params_hash}."
        )
    if prov_model.get("params") != model_params:
        raise ProvenanceMismatchError(
            "Model parameters in training provenance do not match current config parameters."
        )

    # 6. Data and input hashes
    prov_data_hash = training_prov.get("data_hash")
    if prov_data_hash != expected_data_hash:
        raise ProvenanceMismatchError(
            f"Data hash mismatch: training provenance was generated on data {prov_data_hash}, "
            f"current data hash is {expected_data_hash}."
        )

    target_input_sha = (
        expected_input_file_sha256 if expected_input_file_sha256 is not None else expected_data_hash
    )
    prov_input_sha = training_prov.get("input_file_sha256") or prov_data_hash
    if prov_input_sha != target_input_sha:
        raise ProvenanceMismatchError(
            f"Input file SHA256 mismatch: expected {target_input_sha}, got {prov_input_sha}."
        )

    if expected_train_file_sha256 is not None:
        prov_train_sha = training_prov.get("train_file_sha256")
        if not prov_train_sha or prov_train_sha != expected_train_file_sha256:
            raise ProvenanceMismatchError(
                f"Train file SHA256 mismatch: expected {expected_train_file_sha256}, "
                f"got {prov_train_sha}."
            )

    # 7. Split hash
    prov_split_hash = training_prov.get("split_hash")
    if prov_split_hash != expected_split_hash:
        raise ProvenanceMismatchError(
            f"Split manifest hash mismatch: training provenance used split {prov_split_hash}, "
            f"current split hash is {expected_split_hash}."
        )

    # 8. Seed policy
    prov_seed_policy = training_prov.get("seed_policy")
    if not isinstance(prov_seed_policy, dict):
        raise ProvenanceMismatchError("Training provenance missing 'seed_policy' mapping.")

    prov_seed = prov_seed_policy.get("base_seed")
    if prov_seed != expected_base_seed:
        raise ProvenanceMismatchError(
            f"Seed policy mismatch: training provenance used base seed {prov_seed}, "
            f"expected {expected_base_seed}."
        )

    prov_seed_formula = prov_seed_policy.get("formula")
    if prov_seed_formula != expected_seed_formula:
        raise ProvenanceMismatchError(
            f"Seed policy formula mismatch: expected {expected_seed_formula!r}, "
            f"got {prov_seed_formula!r}."
        )

    # 9. Training source hash presence
    if not training_prov.get("training_source_hash"):
        raise ProvenanceMismatchError(
            "Training provenance record is missing 'training_source_hash'."
        )


def verify_training_anchor(
    training_manifest_path: Path | str,
    cand_oof_path: Path | str,
    training_prov_path: Path | str,
    expected_train_file_sha256: str,
    expected_data_hash: str,
    expected_split_hash: str,
    feature_contract: TargetedFeatureContract | None = None,
) -> dict[str, Any]:
    """Verify that prior training-mode manifest acts as a valid training anchor.

    Threat Model:
        Local integrity against accidental/independent tamper and configuration drift.
        This does NOT guarantee authenticity against coordinated malicious rewriting
        without cryptographic digital signatures.

    Raises:
        LegacyProvenanceError: If training manifest does not exist.
        ProvenanceMismatchError: If operation is not 'training' or input data hashes mismatch.
        ArtifactTamperError: If artifact checksums in the training manifest disagree with disk.
    """
    manifest_p = Path(training_manifest_path)
    if not manifest_p.exists():
        raise LegacyProvenanceError(
            f"Prior training-mode manifest '{manifest_p.name}' not found at {manifest_p}. "
            "Reuse requires a prior training-mode manifest whose provenance operation "
            "is 'training'. Candidate must be retrained freshly in training mode."
        )

    with open(manifest_p, encoding="utf-8") as f:
        anchor = json.load(f)

    op = anchor.get("operation") or anchor.get("provenance", {}).get("operation")
    if op != "training":
        raise ProvenanceMismatchError(
            f"Training anchor operation mismatch: expected 'training', got {op!r}."
        )

    artifacts = anchor.get("artifacts", {})
    if "training_provenance" not in artifacts:
        raise ProvenanceMismatchError(
            "Training manifest artifact inventory is missing 'training_provenance' entry."
        )
    expected_prov_sha = artifacts["training_provenance"].get("sha256")
    actual_prov_sha = compute_file_sha256(training_prov_path)
    if actual_prov_sha != expected_prov_sha:
        raise ArtifactTamperError(
            f"Training provenance sidecar hash {actual_prov_sha} "
            f"does not match training anchor record {expected_prov_sha}."
        )

    if "candidate_oof" not in artifacts:
        raise ProvenanceMismatchError(
            "Training manifest artifact inventory is missing 'candidate_oof' entry."
        )
    expected_oof_sha = artifacts["candidate_oof"].get("sha256")
    actual_oof_sha = compute_file_sha256(cand_oof_path)
    if actual_oof_sha != expected_oof_sha:
        raise ArtifactTamperError(
            f"Candidate OOF hash {actual_oof_sha} does not match training anchor "
            f"record {expected_oof_sha}."
        )

    prov = anchor.get("provenance", {})
    if (
        prov.get("train_file_sha256")
        and prov.get("train_file_sha256") != expected_train_file_sha256
    ):
        raise ProvenanceMismatchError(
            f"Training anchor train_file_sha256 mismatch: expected {expected_train_file_sha256}, "
            f"got {prov.get('train_file_sha256')}."
        )
    anchor_data_hash = (
        prov.get("data_hash")
        or prov.get("input_file_sha256")
        or prov.get("combined_train_test_sha256")
    )
    if anchor_data_hash and anchor_data_hash != expected_data_hash:
        raise ProvenanceMismatchError(
            f"Training anchor data_hash mismatch: expected {expected_data_hash}, "
            f"got {anchor_data_hash}."
        )
    if prov.get("split_hash") and prov.get("split_hash") != expected_split_hash:
        raise ProvenanceMismatchError(
            f"Training anchor split_hash mismatch: expected {expected_split_hash}, "
            f"got {prov.get('split_hash')}."
        )

    if feature_contract is not None:
        policy = anchor.get("policy_name") or anchor.get("experiment_id")
        if policy and policy != feature_contract.feature_name:
            raise ProvenanceMismatchError(
                f"Training anchor policy mismatch: expected {feature_contract.feature_name!r}, "
                f"got {policy!r}."
            )

    return anchor


def validate_oof_keys(
    oof_df: pd.DataFrame,
    n_rows: int,
    n_repeats: int = 3,
    n_splits: int = 5,
    name: str = "oof_predictions",
) -> None:
    """Validate completeness and uniqueness of 3-repeat OOF prediction keys.

    Raises:
        ValueError: If columns are missing, rows are missing/duplicated, or bounds invalid.
    """
    expected_cols = {"row_index", "repeat", "fold", "probability"}
    missing = expected_cols - set(oof_df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    expected_len = n_rows * n_repeats
    if len(oof_df) != expected_len:
        msg = f"{name} row count mismatch: expected {expected_len}, got {len(oof_df)}"
        raise ValueError(msg)

    # Check unique keys per repeat
    repeats = sorted(int(r) for r in oof_df["repeat"].unique())
    if repeats != list(range(n_repeats)):
        exp_reps = list(range(n_repeats))
        raise ValueError(f"{name} repeats mismatch: expected {exp_reps}, got {repeats}")

    folds = sorted(int(f) for f in oof_df["fold"].unique())
    if folds != list(range(n_splits)):
        exp_folds = list(range(n_splits))
        raise ValueError(f"{name} folds mismatch: expected {exp_folds}, got {folds}")

    # Check uniqueness of (repeat, row_index)
    dups = oof_df.duplicated(subset=["repeat", "row_index"])
    if dups.any():
        num_dups = int(dups.sum())
        raise ValueError(f"{name} contains {num_dups} duplicate (repeat, row_index) entries")

    # Check row_index bounds
    r_min = int(oof_df["row_index"].min())
    r_max = int(oof_df["row_index"].max())
    if r_min < 0 or r_max >= n_rows:
        raise ValueError(
            f"{name} row_index out of bounds [0, {n_rows - 1}]: min={r_min}, max={r_max}"
        )


def calculate_experiment_metrics(
    y_true: np.ndarray,
    oof_baseline: pd.DataFrame,
    oof_candidate: pd.DataFrame,
    n_rows: int,
    n_repeats: int = 3,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Calculate fold, repeat, global metrics and Nadeau-Bengio corrected resampled t-test."""
    validate_oof_keys(
        oof_baseline,
        n_rows=n_rows,
        n_repeats=n_repeats,
        n_splits=n_splits,
        name="oof_baseline",
    )
    validate_oof_keys(
        oof_candidate,
        n_rows=n_rows,
        n_repeats=n_repeats,
        n_splits=n_splits,
        name="oof_candidate",
    )

    # Sort canonically
    base_sorted = oof_baseline.sort_values(
        ["repeat", "fold", "row_index"], kind="stable"
    ).reset_index(drop=True)
    cand_sorted = oof_candidate.sort_values(
        ["repeat", "fold", "row_index"], kind="stable"
    ).reset_index(drop=True)

    # Verify keys match exactly
    if not (base_sorted["repeat"].to_numpy() == cand_sorted["repeat"].to_numpy()).all():
        raise ValueError("Baseline and candidate OOF repeats do not align")
    if not (base_sorted["fold"].to_numpy() == cand_sorted["fold"].to_numpy()).all():
        raise ValueError("Baseline and candidate OOF folds do not align")
    if not (base_sorted["row_index"].to_numpy() == cand_sorted["row_index"].to_numpy()).all():
        raise ValueError("Baseline and candidate OOF row indices do not align")

    fold_metrics: list[dict[str, Any]] = []
    base_fold_aucs: list[float] = []
    cand_fold_aucs: list[float] = []

    # 1. Per-fold metrics
    for (rep, fold), g_base in base_sorted.groupby(["repeat", "fold"], sort=True):
        g_cand = cand_sorted[(cand_sorted["repeat"] == rep) & (cand_sorted["fold"] == fold)]
        val_idx = g_base["row_index"].to_numpy()
        y_fold = y_true[val_idx]
        p_base = g_base["probability"].to_numpy()
        p_cand = g_cand["probability"].to_numpy()

        auc_b = float(roc_auc_score(y_fold, p_base))
        auc_c = float(roc_auc_score(y_fold, p_cand))
        pr_b = float(average_precision_score(y_fold, p_base))
        pr_c = float(average_precision_score(y_fold, p_cand))
        ll_b = float(log_loss(y_fold, p_base, labels=[0, 1]))
        ll_c = float(log_loss(y_fold, p_cand, labels=[0, 1]))

        base_fold_aucs.append(auc_b)
        cand_fold_aucs.append(auc_c)

        fold_metrics.append(
            {
                "repeat": int(rep),
                "fold": int(fold),
                "baseline_roc_auc": auc_b,
                "candidate_roc_auc": auc_c,
                "delta_roc_auc": auc_c - auc_b,
                "baseline_pr_auc": pr_b,
                "candidate_pr_auc": pr_c,
                "delta_pr_auc": pr_c - pr_b,
                "baseline_log_loss": ll_b,
                "candidate_log_loss": ll_c,
                "delta_log_loss": ll_c - ll_b,
            }
        )

    # 2. Per-repeat pooled metrics
    repeat_metrics: list[dict[str, Any]] = []
    repeat_auc_deltas: list[float] = []
    for rep in range(n_repeats):
        rep_base = base_sorted[base_sorted["repeat"] == rep].sort_values("row_index")
        rep_cand = cand_sorted[cand_sorted["repeat"] == rep].sort_values("row_index")
        p_base_rep = rep_base["probability"].to_numpy()
        p_cand_rep = rep_cand["probability"].to_numpy()

        auc_b_rep = float(roc_auc_score(y_true, p_base_rep))
        auc_c_rep = float(roc_auc_score(y_true, p_cand_rep))
        pr_b_rep = float(average_precision_score(y_true, p_base_rep))
        pr_c_rep = float(average_precision_score(y_true, p_cand_rep))
        ll_b_rep = float(log_loss(y_true, p_base_rep, labels=[0, 1]))
        ll_c_rep = float(log_loss(y_true, p_cand_rep, labels=[0, 1]))

        d_auc_rep = auc_c_rep - auc_b_rep
        repeat_auc_deltas.append(d_auc_rep)

        repeat_metrics.append(
            {
                "repeat": int(rep),
                "baseline_roc_auc": auc_b_rep,
                "candidate_roc_auc": auc_c_rep,
                "delta_roc_auc": d_auc_rep,
                "baseline_pr_auc": pr_b_rep,
                "candidate_pr_auc": pr_c_rep,
                "delta_pr_auc": pr_c_rep - pr_b_rep,
                "baseline_log_loss": ll_b_rep,
                "candidate_log_loss": ll_c_rep,
                "delta_log_loss": ll_c_rep - ll_b_rep,
            }
        )

    # 3. Global repeat-averaged OOF metrics
    p_base_global = base_sorted.groupby("row_index")["probability"].mean().sort_index().to_numpy()
    p_cand_global = cand_sorted.groupby("row_index")["probability"].mean().sort_index().to_numpy()

    g_auc_b = float(roc_auc_score(y_true, p_base_global))
    g_auc_c = float(roc_auc_score(y_true, p_cand_global))
    g_pr_b = float(average_precision_score(y_true, p_base_global))
    g_pr_c = float(average_precision_score(y_true, p_cand_global))
    g_ll_b = float(log_loss(y_true, p_base_global, labels=[0, 1]))
    g_ll_c = float(log_loss(y_true, p_cand_global, labels=[0, 1]))

    global_metrics = {
        "baseline_roc_auc": g_auc_b,
        "candidate_roc_auc": g_auc_c,
        "delta_roc_auc": g_auc_c - g_auc_b,
        "baseline_pr_auc": g_pr_b,
        "candidate_pr_auc": g_pr_c,
        "delta_pr_auc": g_pr_c - g_pr_b,
        "baseline_log_loss": g_ll_b,
        "candidate_log_loss": g_ll_c,
        "delta_log_loss": g_ll_c - g_ll_b,
        "repeat_auc_deltas": repeat_auc_deltas,
    }

    # 4. Nadeau-Bengio test (Candidate - Baseline)
    test_train_ratio = 1.0 / (n_splits - 1.0)  # 0.25 for 5-fold CV
    nb_result = compute_corrected_resampled_ttest(
        scores_a=cand_fold_aucs,
        scores_b=base_fold_aucs,
        test_train_ratio=test_train_ratio,
        confidence_level=0.95,
    )

    return {
        "fold_metrics": fold_metrics,
        "repeat_metrics": repeat_metrics,
        "global_metrics": global_metrics,
        "nadeau_bengio": nb_result,
        "p_baseline_global": p_base_global,
        "p_candidate_global": p_cand_global,
    }


def evaluate_frozen_slices(
    train_df: pd.DataFrame,
    y_true: np.ndarray,
    p_baseline: np.ndarray,
    p_candidate: np.ndarray,
    frozen_slices_path: Path | str,
) -> pd.DataFrame:
    """Evaluate candidate vs baseline across frozen slices and cohorts.

    Uses exact predicates loaded from frozen_slices.json.
    """
    p_frozen = Path(frozen_slices_path)
    with open(p_frozen, encoding="utf-8") as f:
        frozen_data = json.load(f)

    predeclared = frozen_data.get("predeclared_slices", {})
    slice_records: list[dict[str, Any]] = []

    # Compute global ranks for rank movement
    rank_base = pd.Series(p_baseline).rank(ascending=False).to_numpy()
    rank_cand = pd.Series(p_candidate).rank(ascending=False).to_numpy()

    # 1. Evaluate predeclared slices
    for cell_name, cell_info in predeclared.items():
        predicate = cell_info["predicate"]
        mask = reconstruct_mask_from_predicate(train_df, predicate).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue

        y_s = y_true[mask]
        pos = int(y_s.sum())
        pos_rate = pos / n

        p_b_s = p_baseline[mask]
        p_c_s = p_candidate[mask]

        if 0 < pos < n:
            auc_b = float(roc_auc_score(y_s, p_b_s))
            auc_c = float(roc_auc_score(y_s, p_c_s))
            d_auc = auc_c - auc_b
        else:
            auc_b = np.nan
            auc_c = np.nan
            d_auc = np.nan

        if pos > 0:
            pr_b = float(average_precision_score(y_s, p_b_s))
            pr_c = float(average_precision_score(y_s, p_c_s))
            d_pr = pr_c - pr_b
        else:
            pr_b = np.nan
            pr_c = np.nan
            d_pr = np.nan

        ll_b = float(log_loss(y_s, p_b_s, labels=[0, 1]))
        ll_c = float(log_loss(y_s, p_c_s, labels=[0, 1]))
        d_ll = ll_c - ll_b

        # Rank movement: base rank - cand rank (positive = ranked higher/better probability)
        rank_mov = float(np.mean(rank_base[mask] - rank_cand[mask]))
        prob_delta = float(np.mean(p_c_s - p_b_s))

        slice_records.append(
            {
                "slice_name": cell_name,
                "cell_group": cell_info.get("cell_group", "predeclared_slice"),
                "n": n,
                "positives": pos,
                "positive_rate": pos_rate,
                "baseline_auc": auc_b,
                "candidate_auc": auc_c,
                "delta_auc": d_auc,
                "baseline_pr_auc": pr_b,
                "candidate_pr_auc": pr_c,
                "delta_pr_auc": d_pr,
                "baseline_log_loss": ll_b,
                "candidate_log_loss": ll_c,
                "delta_log_loss": d_ll,
                "rank_movement": rank_mov,
                "mean_prob_delta": prob_delta,
            }
        )

    # 2. Evaluate cohorts (severe_fn, top_1pct_loss, margin bands)
    cohort_masks: dict[str, np.ndarray] = {}

    # Severe false negatives: y_true == 1 and p_baseline <= 0.1
    cohort_masks["cohort__severe_fn"] = (y_true == 1) & (p_baseline <= 0.1)

    # Top-loss cohort: deterministic frozen rank/count contract matching severe_fn_audit
    top_loss_info = frozen_data.get("cohorts", {}).get("top_1pct_loss", {})
    n_top = top_loss_info.get("count")
    if n_top is None:
        n_top = int(math.ceil(0.01 * len(train_df)))

    base_sample_loss = compute_sample_log_loss(y_true, p_baseline)
    loss_df = pd.DataFrame(
        {
            "sample_log_loss": base_sample_loss,
            "row_index": np.arange(len(train_df)),
        }
    )
    top_df, _ = select_deterministic_top_loss(loss_df, n_top=n_top)
    top_loss_mask = np.zeros(len(train_df), dtype=bool)
    top_loss_mask[top_df["row_index"].to_numpy()] = True
    cohort_masks["cohort__top_1pct_loss"] = top_loss_mask

    # Margin bands
    cohort_masks["cohort__margin_band_25_75"] = (p_baseline >= 0.25) & (p_baseline <= 0.75)
    cohort_masks["cohort__margin_band_40_60"] = (p_baseline >= 0.40) & (p_baseline <= 0.60)

    for cohort_name, mask in cohort_masks.items():
        n = int(mask.sum())
        if n == 0:
            continue

        y_s = y_true[mask]
        pos = int(y_s.sum())
        pos_rate = pos / n

        p_b_s = p_baseline[mask]
        p_c_s = p_candidate[mask]

        if 0 < pos < n:
            auc_b = float(roc_auc_score(y_s, p_b_s))
            auc_c = float(roc_auc_score(y_s, p_c_s))
            d_auc = auc_c - auc_b
        else:
            auc_b = np.nan
            auc_c = np.nan
            d_auc = np.nan

        if pos > 0:
            pr_b = float(average_precision_score(y_s, p_b_s))
            pr_c = float(average_precision_score(y_s, p_c_s))
            d_pr = pr_c - pr_b
        else:
            pr_b = np.nan
            pr_c = np.nan
            d_pr = np.nan

        ll_b = float(log_loss(y_s, p_b_s, labels=[0, 1]))
        ll_c = float(log_loss(y_s, p_c_s, labels=[0, 1]))
        d_ll = ll_c - ll_b

        rank_mov = float(np.mean(rank_base[mask] - rank_cand[mask]))
        prob_delta = float(np.mean(p_c_s - p_b_s))

        slice_records.append(
            {
                "slice_name": cohort_name,
                "cell_group": "cohort",
                "n": n,
                "positives": pos,
                "positive_rate": pos_rate,
                "baseline_auc": auc_b,
                "candidate_auc": auc_c,
                "delta_auc": d_auc,
                "baseline_pr_auc": pr_b,
                "candidate_pr_auc": pr_c,
                "delta_pr_auc": d_pr,
                "baseline_log_loss": ll_b,
                "candidate_log_loss": ll_c,
                "delta_log_loss": d_ll,
                "rank_movement": rank_mov,
                "mean_prob_delta": prob_delta,
            }
        )

    return pd.DataFrame(slice_records)


def make_gate_decision(
    global_auc_delta: float,
    repeat_auc_deltas: list[float],
    nb_ci_lower: float,
    global_log_loss_delta: float,
    slice_regressions: list[str],
) -> ExperimentDecision:
    """Evaluate GO, BORDERLINE, and NO-GO gate criteria dynamically from metrics.

    Criteria:
    - GO gate:
      * global AUC gain >= 0.00010
      * all three repeat deltas >= 0
      * Nadeau-Bengio CI lower > 0
      * LogLoss degradation <= 0.00020
      * no material frozen-slice regression
    - Borderline:
      * global AUC gain between 0.00005 and 0.00010, all repeats >= 0
      * requires second split seed
    - NO-GO:
      * global AUC gain < 0.00005, or inconsistent repeats (any repeat delta < 0),
        or log-loss degradation > 0.00020, or slice regressions.
    """
    details = {
        "global_auc_delta": global_auc_delta,
        "repeat_auc_deltas": repeat_auc_deltas,
        "nb_ci_lower": nb_ci_lower,
        "global_log_loss_delta": global_log_loss_delta,
        "slice_regressions": slice_regressions,
    }

    # Failure conditions
    reasons: list[str] = []
    if any(d < 0.0 for d in repeat_auc_deltas):
        reasons.append(
            f"Inconsistent repeats: observed negative repeat delta(s) {repeat_auc_deltas}"
        )

    if global_log_loss_delta > 0.00020:
        reasons.append(
            f"LogLoss degradation exceeded threshold: +{global_log_loss_delta:.6f} > 0.00020"
        )

    if slice_regressions:
        reasons.append(f"Material frozen-slice regression detected in: {slice_regressions}")

    if reasons:
        return ExperimentDecision(
            decision="NO-GO",
            reason="; ".join(reasons),
            details=details,
        )

    # Check for GO gate
    if global_auc_delta >= 0.00010:
        if nb_ci_lower > 0.0:
            return ExperimentDecision(
                decision="GO",
                reason=(
                    f"Passed all criteria: Global AUC gain {global_auc_delta:+.6f} >= 0.00010, "
                    f"all repeats positive, NB 95% CI lower {nb_ci_lower:+.6f} > 0, "
                    f"LogLoss delta {global_log_loss_delta:+.6f} <= 0.00020, no slice regression."
                ),
                details=details,
            )
        else:
            return ExperimentDecision(
                decision="NO-GO",
                reason=(
                    f"Global AUC gain {global_auc_delta:+.6f} >= 0.00010 but Nadeau-Bengio 95% CI "
                    f"lower bound {nb_ci_lower:+.6f} <= 0 (statistically inconclusive)."
                ),
                details=details,
            )

    # Check for Borderline gate
    if 0.00005 <= global_auc_delta < 0.00010:
        return ExperimentDecision(
            decision="BORDERLINE",
            reason=(
                f"Borderline AUC gain ({global_auc_delta:+.6f} in [0.00005, 0.00010)) "
                "requires second split seed validation before deployment."
            ),
            details=details,
        )

    # Otherwise NO-GO due to insufficient gain
    return ExperimentDecision(
        decision="NO-GO",
        reason=f"Insufficient AUC gain: {global_auc_delta:+.6f} < 0.00005 minimum threshold.",
        details=details,
    )


def find_material_slice_regressions(
    slice_df: pd.DataFrame,
    auc_drop_threshold: float = 0.0050,
    log_loss_increase_threshold: float = 0.0050,
    min_positives: int = 50,
) -> list[str]:
    """Identify slices with material performance regression.

    A slice is considered to have a material regression if:
    - delta_auc <= -auc_drop_threshold (for slices with >= min_positives where AUC is valid), OR
    - delta_log_loss >= log_loss_increase_threshold
    """
    regressions: list[str] = []
    for _, row in slice_df.iterrows():
        slice_name = str(row["slice_name"])
        pos = row.get("positives", 0)
        d_auc = row.get("delta_auc", np.nan)
        d_ll = row.get("delta_log_loss", np.nan)

        if pos >= min_positives and pd.notna(d_auc) and d_auc <= -auc_drop_threshold:
            regressions.append(f"{slice_name} (AUC drop: {d_auc:+.4f})")
        elif pd.notna(d_ll) and d_ll >= log_loss_increase_threshold:
            regressions.append(f"{slice_name} (LogLoss increase: {d_ll:+.4f})")
    return regressions


def train_candidate_cv(
    train_df: pd.DataFrame,
    manifest: pd.DataFrame,
    contract: DatasetContract,
    model_params: dict[str, Any],
    feature_contract: TargetedFeatureContract = CONCERN_SUBSIDY_CONTRACT,
    base_seed: int = 42,
    model_family: str = "lightgbm",
    folds: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Train candidate model across specified folds and generate OOF predictions.

    Applies the targeted feature transform strictly to predictors, retaining parent columns.
    Uses build_model() with deterministic seeds: base_seed + repeat * 100 + fold.
    """
    y_true = encode_target(train_df[contract.target_column], contract)
    transformed_df = feature_contract.transform_fn(train_df)

    drop_cols = [contract.id_column, contract.target_column]
    X_cand = transformed_df.drop(columns=[c for c in drop_cols if c in transformed_df.columns])
    all_indices = np.arange(len(train_df))

    fold_groups = list(manifest.groupby(["repeat", "fold"], sort=True))
    if folds is not None:
        folds_set = set(folds)
        fold_groups = [g for g in fold_groups if g[0] in folds_set]

    oof_row_indices: list[int] = []
    oof_repeats: list[int] = []
    oof_folds: list[int] = []
    oof_probs: list[float] = []

    for (repeat, fold), group in fold_groups:
        val_idx = group["row_index"].to_numpy()
        train_idx = np.setdiff1d(all_indices, val_idx)

        X_tr = X_cand.iloc[train_idx]
        y_tr = y_true[train_idx]
        X_va = X_cand.iloc[val_idx]

        fold_seed = base_seed + int(repeat) * 100 + int(fold)
        model = build_model(
            model_name=model_family,
            seed=fold_seed,
            sample_frame=X_tr,
            params=model_params,
        )
        model.fit(X_tr, y_tr)
        val_p = model.predict_proba(X_va)[:, 1]

        oof_row_indices.extend(val_idx.tolist())
        oof_repeats.extend([int(repeat)] * len(val_idx))
        oof_folds.extend([int(fold)] * len(val_idx))
        oof_probs.extend(val_p.tolist())

    return pd.DataFrame(
        {
            "row_index": oof_row_indices,
            "repeat": oof_repeats,
            "fold": oof_folds,
            "probability": oof_probs,
        }
    )


def generate_experiment_report_markdown(
    global_metrics: dict[str, Any],
    fold_metrics: list[dict[str, Any]],
    repeat_metrics: list[dict[str, Any]],
    nadeau_bengio: dict[str, Any],
    slice_df: pd.DataFrame,
    decision: ExperimentDecision,
    candidate_meta: dict[str, Any],
) -> str:
    """Generate dynamic Markdown report for the targeted feature experiment."""
    b_auc = global_metrics["baseline_roc_auc"]
    c_auc = global_metrics["candidate_roc_auc"]
    d_auc = global_metrics["delta_roc_auc"]
    b_pr = global_metrics["baseline_pr_auc"]
    c_pr = global_metrics["candidate_pr_auc"]
    d_pr = global_metrics["delta_pr_auc"]
    b_ll = global_metrics["baseline_log_loss"]
    c_ll = global_metrics["candidate_log_loss"]
    d_ll = global_metrics["delta_log_loss"]

    base_cand_id = candidate_meta.get("baseline_candidate_id", "lightgbm_hpo_trial35")
    cand_feat = candidate_meta.get("candidate_feature", "Concern_Subsidy")
    exp_name = candidate_meta.get("experiment_id", "targeted_feature")
    parent_cols = candidate_meta.get(
        "parent_columns", ["Environmental_Concern_Level", "Subsidy_Available"]
    )
    parent_cols_str = ", ".join(f"`{c}`" for c in parent_cols)

    lines: list[str] = [
        f"# Targeted Feature Experiment: {cand_feat} ({exp_name})",
        "",
        "## Executive Summary",
        f"- **Experiment Gate Decision:** `{decision.decision}`",
        f"- **Decision Rationale:** {decision.reason}",
        f"- **Baseline Model:** `{base_cand_id}`",
        f"- **Targeted Feature Added:** `{cand_feat}`",
        f"- **Parent Columns Retained:** {parent_cols_str}",
        "",
        "## Overall Metric Comparison (Pooled Repeat-Averaged)",
        "| Metric | Baseline | Candidate | Delta | Evaluation Gate Rule |",
        "| :--- | :--- | :--- | :--- | :--- |",
        (
            f"| **ROC-AUC** | {b_auc:.6f} | {c_auc:.6f} | {d_auc:+.6f} | "
            "Gain $\\ge +0.00010$ (GO), $[+0.00005, +0.00010)$ (Borderline) |"
        ),
        f"| **PR-AUC** | {b_pr:.6f} | {c_pr:.6f} | {d_pr:+.6f} | Guardrail tracking |",
        f"| **LogLoss** | {b_ll:.6f} | {c_ll:.6f} | {d_ll:+.6f} | Degradation $\\le +0.00020$ |",
        "",
        "## Repeat-Level Stability",
        (
            "| Repeat | Baseline ROC-AUC | Candidate ROC-AUC | Delta ROC-AUC | Delta PR-AUC | "
            "Delta LogLoss | Status |"
        ),
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for rm in repeat_metrics:
        rep_id = rm["repeat"]
        r_d_auc = rm["delta_roc_auc"]
        status = "PASSED (>=0)" if r_d_auc >= 0 else "FAILED (<0)"
        lines.append(
            f"| Repeat {rep_id} | {rm['baseline_roc_auc']:.6f} | {rm['candidate_roc_auc']:.6f} | "
            f"{r_d_auc:+.6f} | {rm['delta_pr_auc']:+.6f} | {rm['delta_log_loss']:+.6f} | {status} |"
        )

    nb_ratio = nadeau_bengio.get("test_train_ratio", 0.25)
    nb_df = nadeau_bengio.get("df", 14)
    nb_mean = nadeau_bengio.get("mean_difference", 0.0)
    nb_sd = nadeau_bengio.get("sd_difference", 0.0)
    nb_se_naive = nadeau_bengio.get("se_naive", 0.0)
    nb_se = nadeau_bengio.get("se_nb", 0.0)
    ci_l = nadeau_bengio.get("ci_nb_lower", 0.0)
    ci_u = nadeau_bengio.get("ci_nb_upper", 0.0)
    p_val = nadeau_bengio.get("p_value_nb_raw", 1.0)
    nb_conc = nadeau_bengio.get("conclusion", "inconclusive")

    lines.extend(
        [
            "",
            "## Nadeau-Bengio Corrected Resampled Hypothesis Test",
            "- **Test Formulation:** Candidate ROC-AUC − Baseline ROC-AUC over 15 CV folds",
            f"- **Test/Train Split Ratio:** {nb_ratio:.2f} ($1/4$ for 5-fold CV)",
            f"- **Degrees of Freedom:** {nb_df}",
            f"- **Mean Difference:** {nb_mean:+.6f}",
            f"- **Sample SD Difference:** {nb_sd:.6f}",
            f"- **Naive SE:** {nb_se_naive:.6f}",
            f"- **Nadeau-Bengio Corrected SE:** {nb_se:.6f}",
            f"- **Nadeau-Bengio 95% CI:** [{ci_l:+.6f}, {ci_u:+.6f}]",
            f"- **p-value (NB Corrected):** {p_val:.4e}",
            f"- **Statistical Conclusion:** `{nb_conc}`",
            "",
            "## 15-Fold Detailed Performance Deltas",
            (
                "| Repeat | Fold | Baseline AUC | Candidate AUC | Delta AUC | Delta PR-AUC | "
                "Delta LogLoss |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for fm in fold_metrics:
        lines.append(
            f"| {fm['repeat']} | {fm['fold']} | {fm['baseline_roc_auc']:.6f} | "
            f"{fm['candidate_roc_auc']:.6f} | {fm['delta_roc_auc']:+.6f} | "
            f"{fm['delta_pr_auc']:+.6f} | {fm['delta_log_loss']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Frozen Slices & Diagnostic Cohorts Analysis",
            (
                "| Slice / Cohort | n | Positives | Base AUC | Cand AUC | Delta AUC | "
                "Base LogLoss | Cand LogLoss | Delta LogLoss | Rank Movement |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for _, row in slice_df.iterrows():
        b_auc_str = f"{row['baseline_auc']:.4f}" if pd.notna(row["baseline_auc"]) else "N/A"
        c_auc_str = f"{row['candidate_auc']:.4f}" if pd.notna(row["candidate_auc"]) else "N/A"
        d_auc_str = f"{row['delta_auc']:+.4f}" if pd.notna(row["delta_auc"]) else "N/A"
        lines.append(
            f"| `{row['slice_name']}` | {int(row['n']):,} | {int(row['positives']):,} | "
            f"{b_auc_str} | {c_auc_str} | {d_auc_str} | {row['baseline_log_loss']:.4f} | "
            f"{row['candidate_log_loss']:.4f} | {row['delta_log_loss']:+.4f} | "
            f"{row['rank_movement']:+.1f} |"
        )

    pass_auc = "PASS" if d_auc >= 0.00010 else "FAIL"
    rep_deltas = [round(d, 6) for d in global_metrics.get("repeat_auc_deltas", [])]
    pass_reps = (
        "PASS" if all(d >= 0 for d in global_metrics.get("repeat_auc_deltas", [])) else "FAIL"
    )
    pass_ci = "PASS" if ci_l > 0 else "FAIL"
    pass_ll = "PASS" if d_ll <= 0.00020 else "FAIL"
    has_regr = bool(decision.details.get("slice_regressions"))
    pass_slice = "FAIL" if has_regr else "PASS"

    lines.extend(
        [
            "",
            "## Threat Model & Provenance Anchor",
            (
                "- **Integrity Scope:** Local integrity verification against accidental file "
                "modification, independent tampering, and configuration drift."
            ),
            (
                "- **Authenticity Limits:** Does not guarantee authenticity against coordinated "
                "malicious rewriting without cryptographic digital signatures."
            ),
            (
                "- **Training Anchor:** Requires a prior training-mode receipt "
                "(`training_manifest.json`) verifying sidecar checksums before artifact reuse."
            ),
            "",
            "## Gate Decision Audit Checklist",
            f"1. **AUC Gain $\\ge +0.00010$:** {pass_auc} (Observed: {d_auc:+.6f})",
            f"2. **All Repeat Deltas $\\ge 0.0$:** {pass_reps} (Observed: {rep_deltas})",
            f"3. **NB 95% CI Lower Bound > 0.0:** {pass_ci} (Observed: {ci_l:+.6f})",
            f"4. **LogLoss Degradation $\\le +0.00020$:** {pass_ll} (Observed: {d_ll:+.6f})",
            f"5. **No Material Frozen-Slice Regression:** {pass_slice}",
            "",
            f"**Final Status:** `{decision.decision}`",
            "",
        ]
    )

    return "\n".join(lines)


def run_targeted_experiment(
    train_path: Path | str,
    test_path: Path | str,
    manifest_path: Path | str,
    baseline_oof_path: Path | str,
    final_config_path: Path | str,
    frozen_slices_path: Path | str,
    output_dir: Path | str | None = None,
    smoke: bool = False,
    reuse_candidate_oof: bool = False,
    feature_policy: str | TargetedFeatureContract = "concern_subsidy",
    generation_command: str | None = None,
) -> dict[str, Any]:
    """Execute complete targeted-feature experiment with strict provenance and reporting."""
    train_path = Path(train_path)
    test_path = Path(test_path)
    manifest_path = Path(manifest_path)
    baseline_oof_path = Path(baseline_oof_path)
    final_config_path = Path(final_config_path)
    frozen_slices_path = Path(frozen_slices_path)

    generation_cmd = generation_command if generation_command is not None else " ".join(sys.argv)

    if isinstance(feature_policy, str):
        feature_contract = get_targeted_feature_contract(feature_policy)
    else:
        feature_contract = feature_policy

    if output_dir is None:
        out_dir = (
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "targeted_features"
            / feature_contract.feature_name
        )
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load config and verify provenance
    import yaml

    with open(final_config_path) as f:
        final_cfg = yaml.safe_load(f)

    lgb_component = final_cfg["policy"]["components"]["lightgbm_hpo_trial35"]
    expected_data_sha = lgb_component["provenance"]["data_sha256"]
    expected_split_sha = lgb_component["provenance"]["split_sha256"]

    # Baseline metadata for OOF checksum
    meta_rel = Path(lgb_component["candidate_metadata_path"])
    meta_path = final_config_path.parent.parent / meta_rel
    if meta_path.exists():
        with open(meta_path) as f:
            meta_json = json.load(f)
        expected_oof_sha = meta_json["artifact_checksums"]["oof_predictions"]
    else:
        expected_oof_sha = compute_file_sha256(baseline_oof_path)

    actual_data_sha = compute_combined_sha256([train_path, test_path])
    actual_split_sha = compute_file_sha256(manifest_path)
    actual_oof_sha = compute_file_sha256(baseline_oof_path)
    actual_train_sha = compute_file_sha256(train_path)

    verify_provenance_and_baseline(
        expected_data_sha256=expected_data_sha,
        actual_data_sha256=actual_data_sha,
        expected_split_sha256=expected_split_sha,
        actual_split_sha256=actual_split_sha,
        expected_oof_sha256=expected_oof_sha,
        actual_oof_sha256=actual_oof_sha,
    )

    # Verify frozen slices before evaluation
    frozen_slices_meta = verify_frozen_slices(frozen_slices_path)

    # 2. Load data
    train_df = pd.read_csv(train_path)
    manifest = pd.read_parquet(manifest_path)
    baseline_oof = pd.read_parquet(baseline_oof_path)
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    y_true = encode_target(train_df["Will_Buy_EV"], contract)

    model_params = lgb_component["params"]
    base_seed = lgb_component["seed_policy"]["base_seed"]

    # Smoke mode: sample one fold or full 15 folds
    folds_to_run = [(0, 0)] if smoke else None

    # 3. Train or reuse candidate cross-validation
    cand_oof_path = out_dir / "candidate_oof.parquet"
    training_prov_path = out_dir / "training_provenance.json"
    training_manifest_path = out_dir / "training_manifest.json"
    current_source_hash = compute_file_sha256(Path(__file__).resolve())

    if not smoke and reuse_candidate_oof:
        if not cand_oof_path.exists():
            raise FileNotFoundError(
                f"Candidate OOF file requested for reuse does not exist: {cand_oof_path}"
            )
        if not training_prov_path.exists():
            sidecar_name = training_prov_path.name
            raise LegacyProvenanceError(
                f"Legacy candidate artifact detected at {cand_oof_path}. "
                f"Reuse is forbidden because training provenance sidecar '{sidecar_name}' "
                "is missing. Candidate must be retrained freshly with training provenance."
            )

        # Verify prior training anchor before reuse
        verify_training_anchor(
            training_manifest_path=training_manifest_path,
            cand_oof_path=cand_oof_path,
            training_prov_path=training_prov_path,
            expected_train_file_sha256=actual_train_sha,
            expected_data_hash=actual_data_sha,
            expected_split_hash=actual_split_sha,
            feature_contract=feature_contract,
        )

        with open(training_prov_path, encoding="utf-8") as f:
            training_prov = json.load(f)

        print(f"Reusing existing candidate OOF from {cand_oof_path} with verified provenance...")
        candidate_oof = pd.read_parquet(cand_oof_path)

        verify_training_provenance(
            training_prov=training_prov,
            feature_contract=feature_contract,
            model_params=model_params,
            expected_data_hash=actual_data_sha,
            expected_split_hash=actual_split_sha,
            expected_base_seed=base_seed,
            candidate_oof_path=cand_oof_path,
            candidate_oof_df=candidate_oof,
            expected_row_count=len(train_df) * 3,
            expected_train_file_sha256=actual_train_sha,
            expected_input_file_sha256=actual_data_sha,
            expected_model_family="lightgbm",
            expected_provenance_version="1.0",
            expected_seed_formula="base_seed + repeat * 100 + fold",
        )
        training_source_hash = training_prov["training_source_hash"]
    else:
        print(
            f"Training candidate model (policy={feature_contract.feature_name}, smoke={smoke})..."
        )
        candidate_oof = train_candidate_cv(
            train_df=train_df,
            manifest=manifest,
            contract=contract,
            model_params=model_params,
            feature_contract=feature_contract,
            base_seed=base_seed,
            model_family="lightgbm",
            folds=folds_to_run,
        )
        if not smoke:
            # 1. Atomically write candidate OOF
            _atomic_write_parquet(cand_oof_path, candidate_oof, index=False)
            cand_oof_sha = compute_file_sha256(cand_oof_path)

            # 2. Build and atomically write training provenance sidecar
            training_prov = build_training_provenance(
                feature_contract=feature_contract,
                model_params=model_params,
                data_hash=actual_data_sha,
                split_hash=actual_split_sha,
                base_seed=base_seed,
                candidate_oof_path=cand_oof_path,
                candidate_oof_df=candidate_oof,
                source_hash=current_source_hash,
                train_file_sha256=actual_train_sha,
                input_file_sha256=actual_data_sha,
                model_family="lightgbm",
                provenance_version="1.0",
                seed_formula="base_seed + repeat * 100 + fold",
            )
            _atomic_write_json(training_prov_path, training_prov)
            training_prov_sha = compute_file_sha256(training_prov_path)
            training_source_hash = current_source_hash

            # 3. Build and atomically write training-mode receipt/anchor
            training_manifest = {
                "operation": "training",
                "experiment_id": feature_contract.feature_name,
                "status": "completed",
                "policy_name": feature_contract.feature_name,
                "feature_contract": {
                    "family": feature_contract.family,
                    "experiment_index": feature_contract.experiment_index,
                    "feature_name": feature_contract.feature_name,
                    "column_name": feature_contract.column_name,
                    "parent_columns": list(feature_contract.parent_columns),
                    "formula": feature_contract.formula,
                    "description": feature_contract.description,
                },
                "model": {
                    "family": "lightgbm",
                    "params": model_params,
                    "params_hash": compute_params_hash(model_params),
                },
                "seed_policy": {
                    "base_seed": base_seed,
                    "formula": "base_seed + repeat * 100 + fold",
                },
                "provenance_version": "1.0",
                "provenance": {
                    "operation": "training",
                    "train_file_sha256": actual_train_sha,
                    "input_file_sha256": actual_data_sha,
                    "data_hash": actual_data_sha,
                    "split_manifest_sha256": actual_split_sha,
                    "split_hash": actual_split_sha,
                    "candidate_oof_sha256": cand_oof_sha,
                    "training_provenance_sha256": training_prov_sha,
                    "training_source_hash": training_source_hash,
                    "threat_model": THREAT_MODEL_STATEMENT,
                    "environment": {
                        "python_version": sys.version.split()[0],
                        "platform": sys.platform,
                    },
                },
                "artifacts": {
                    "candidate_oof": {
                        "path": _rel_path_or_str(cand_oof_path),
                        "file_name": cand_oof_path.name,
                        "sha256": cand_oof_sha,
                        "row_count": len(candidate_oof),
                        "record_count": len(candidate_oof),
                    },
                    "training_provenance": {
                        "path": _rel_path_or_str(training_prov_path),
                        "file_name": training_prov_path.name,
                        "sha256": training_prov_sha,
                        "row_count": len(training_prov),
                        "record_count": len(training_prov),
                    },
                },
            }
            _atomic_write_json(training_manifest_path, training_manifest)

    # If smoke, return early diagnostics
    if smoke:
        val_idx = manifest[(manifest["repeat"] == 0) & (manifest["fold"] == 0)][
            "row_index"
        ].to_numpy()
        b_smoke = baseline_oof[(baseline_oof["repeat"] == 0) & (baseline_oof["fold"] == 0)].copy()
        c_smoke = candidate_oof.copy()
        b_auc = float(roc_auc_score(y_true[val_idx], b_smoke["probability"]))
        c_auc = float(roc_auc_score(y_true[val_idx], c_smoke["probability"]))
        return {
            "smoke": True,
            "baseline_auc": b_auc,
            "candidate_auc": c_auc,
            "delta_auc": c_auc - b_auc,
            "val_rows": len(val_idx),
        }

    # 4. Validate OOF keys and calculate metrics
    n_rows = len(train_df)
    validate_oof_keys(candidate_oof, n_rows=n_rows, n_repeats=3, n_splits=5, name="candidate_oof")

    metrics_res = calculate_experiment_metrics(
        y_true=y_true,
        oof_baseline=baseline_oof,
        oof_candidate=candidate_oof,
        n_rows=n_rows,
        n_repeats=3,
        n_splits=5,
    )

    # 5. Evaluate frozen slices
    slice_df = evaluate_frozen_slices(
        train_df=train_df,
        y_true=y_true,
        p_baseline=metrics_res["p_baseline_global"],
        p_candidate=metrics_res["p_candidate_global"],
        frozen_slices_path=frozen_slices_path,
    )

    # 6. Check for slice regressions and make gate decision
    slice_regressions = find_material_slice_regressions(slice_df)
    decision = make_gate_decision(
        global_auc_delta=metrics_res["global_metrics"]["delta_roc_auc"],
        repeat_auc_deltas=metrics_res["global_metrics"]["repeat_auc_deltas"],
        nb_ci_lower=metrics_res["nadeau_bengio"]["ci_nb_lower"],
        global_log_loss_delta=metrics_res["global_metrics"]["delta_log_loss"],
        slice_regressions=slice_regressions,
    )

    # 7. Save candidate artifacts atomically
    cand_oof_sha = compute_file_sha256(cand_oof_path)
    training_prov_sha = compute_file_sha256(training_prov_path)

    fold_metrics_df = pd.DataFrame(metrics_res["fold_metrics"])
    fold_metrics_path = out_dir / "fold_metrics.csv"
    _atomic_write_csv(fold_metrics_path, fold_metrics_df, index=False)
    fold_metrics_sha = compute_file_sha256(fold_metrics_path)

    repeat_metrics_df = pd.DataFrame(metrics_res["repeat_metrics"])
    repeat_metrics_path = out_dir / "repeat_metrics.csv"
    _atomic_write_csv(repeat_metrics_path, repeat_metrics_df, index=False)
    repeat_metrics_sha = compute_file_sha256(repeat_metrics_path)

    global_metrics_payload = {
        "global_metrics": metrics_res["global_metrics"],
        "nadeau_bengio": metrics_res["nadeau_bengio"],
    }
    global_metrics_path = out_dir / "global_metrics.json"
    _atomic_write_json(global_metrics_path, global_metrics_payload, indent=2)
    global_metrics_sha = compute_file_sha256(global_metrics_path)

    slice_metrics_path = out_dir / "slice_metrics.csv"
    _atomic_write_csv(slice_metrics_path, slice_df, index=False)
    slice_metrics_sha = compute_file_sha256(slice_metrics_path)

    # 8. Dynamic Report
    report_md = generate_experiment_report_markdown(
        global_metrics=metrics_res["global_metrics"],
        fold_metrics=metrics_res["fold_metrics"],
        repeat_metrics=metrics_res["repeat_metrics"],
        nadeau_bengio=metrics_res["nadeau_bengio"],
        slice_df=slice_df,
        decision=decision,
        candidate_meta={
            "experiment_id": feature_contract.feature_name,
            "baseline_candidate_id": "lightgbm_hpo_trial35",
            "candidate_feature": feature_contract.column_name,
            "parent_columns": list(feature_contract.parent_columns),
        },
    )
    report_path = out_dir / "report.md"
    _atomic_write_text(report_path, report_md)
    report_sha = compute_file_sha256(report_path)

    # Produced artifacts inventory (strictly non-self-hashing)
    produced_artifacts = {
        "candidate_oof": {
            "path": _rel_path_or_str(cand_oof_path),
            "file_name": cand_oof_path.name,
            "sha256": cand_oof_sha,
            "row_count": len(candidate_oof),
            "record_count": len(candidate_oof),
        },
        "training_provenance": {
            "path": _rel_path_or_str(training_prov_path),
            "file_name": training_prov_path.name,
            "sha256": training_prov_sha,
            "row_count": len(training_prov),
            "record_count": len(training_prov),
        },
        "fold_metrics": {
            "path": _rel_path_or_str(fold_metrics_path),
            "file_name": fold_metrics_path.name,
            "sha256": fold_metrics_sha,
            "row_count": len(fold_metrics_df),
            "record_count": len(fold_metrics_df),
        },
        "repeat_metrics": {
            "path": _rel_path_or_str(repeat_metrics_path),
            "file_name": repeat_metrics_path.name,
            "sha256": repeat_metrics_sha,
            "row_count": len(repeat_metrics_df),
            "record_count": len(repeat_metrics_df),
        },
        "global_metrics": {
            "path": _rel_path_or_str(global_metrics_path),
            "file_name": global_metrics_path.name,
            "sha256": global_metrics_sha,
            "row_count": len(global_metrics_payload),
            "record_count": len(global_metrics_payload),
        },
        "slice_metrics": {
            "path": _rel_path_or_str(slice_metrics_path),
            "file_name": slice_metrics_path.name,
            "sha256": slice_metrics_sha,
            "row_count": len(slice_df),
            "record_count": len(slice_df),
        },
        "report": {
            "path": _rel_path_or_str(report_path),
            "file_name": report_path.name,
            "sha256": report_sha,
            "row_count": len(report_md.splitlines()),
            "record_count": len(report_md.splitlines()),
        },
    }
    if training_manifest_path.exists():
        produced_artifacts["training_manifest"] = {
            "path": _rel_path_or_str(training_manifest_path),
            "file_name": training_manifest_path.name,
            "sha256": compute_file_sha256(training_manifest_path),
            "row_count": len(json.loads(training_manifest_path.read_text(encoding="utf-8"))),
            "record_count": len(json.loads(training_manifest_path.read_text(encoding="utf-8"))),
        }

    # 9. Evaluation Manifest (atomic writes, never overwriting training_manifest.json)
    manifest_payload = {
        "experiment_id": feature_contract.feature_name,
        "status": "completed",
        "operation": "evaluation",
        "baseline_candidate_id": "lightgbm_hpo_trial35",
        "candidate_feature": feature_contract.column_name,
        "parent_columns": list(feature_contract.parent_columns),
        "params": model_params,
        "seed_policy": {
            "base_seed": base_seed,
            "formula": "base_seed + repeat * 100 + fold",
        },
        "frozen_slices": {
            "path": _rel_path_or_str(frozen_slices_path),
            "sha256": frozen_slices_meta["sha256"],
            "version": frozen_slices_meta["version"],
            "schema_version": frozen_slices_meta["version"],
            "schema": frozen_slices_meta["version"],
            "status": frozen_slices_meta["status"],
            "provenance_status": frozen_slices_meta["status"],
        },
        "training_provenance": training_prov,
        "training_anchor": {
            "path": _rel_path_or_str(training_manifest_path),
            "sha256": compute_file_sha256(training_manifest_path),
            "operation": "training",
        },
        "threat_model": THREAT_MODEL_STATEMENT,
        "provenance": {
            "operation": "evaluation",
            "train_file_sha256": actual_train_sha,
            "input_file_sha256": actual_data_sha,
            "combined_train_test_sha256": actual_data_sha,
            "split_manifest_sha256": actual_split_sha,
            "baseline_oof_sha256": actual_oof_sha,
            "candidate_oof_sha256": cand_oof_sha,
            "frozen_slices_sha256": frozen_slices_meta["sha256"],
            "frozen_slices_path": _rel_path_or_str(frozen_slices_path),
            "frozen_slices_version": frozen_slices_meta["version"],
            "frozen_slices_schema": frozen_slices_meta["version"],
            "frozen_slices_provenance_status": frozen_slices_meta["status"],
            "generation_command": generation_cmd,
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
            },
            "training_source_hash": training_source_hash,
            "evaluation_source_hash": current_source_hash,
            "source_hash": training_source_hash,
            "training_provenance_path": _rel_path_or_str(training_prov_path),
            "training_provenance_sha256": training_prov_sha,
            "training_manifest_path": _rel_path_or_str(training_manifest_path),
            "training_manifest_sha256": compute_file_sha256(training_manifest_path),
            "threat_model": THREAT_MODEL_STATEMENT,
        },
        "artifacts": produced_artifacts,
        "produced_artifacts": produced_artifacts,
        "gate_decision": {
            "decision": decision.decision,
            "reason": decision.reason,
            "details": decision.details,
        },
        "summary_metrics": {
            "global_roc_auc_baseline": metrics_res["global_metrics"]["baseline_roc_auc"],
            "global_roc_auc_candidate": metrics_res["global_metrics"]["candidate_roc_auc"],
            "global_roc_auc_delta": metrics_res["global_metrics"]["delta_roc_auc"],
            "global_pr_auc_delta": metrics_res["global_metrics"]["delta_pr_auc"],
            "global_log_loss_delta": metrics_res["global_metrics"]["delta_log_loss"],
            "nadeau_bengio_ci": [
                metrics_res["nadeau_bengio"]["ci_nb_lower"],
                metrics_res["nadeau_bengio"]["ci_nb_upper"],
            ],
            "repeat_auc_deltas": metrics_res["global_metrics"]["repeat_auc_deltas"],
        },
    }

    manifest_path = out_dir / "manifest.json"
    _atomic_write_json(manifest_path, manifest_payload, indent=2)
    evaluation_manifest_path = out_dir / "evaluation_manifest.json"
    _atomic_write_json(evaluation_manifest_path, manifest_payload, indent=2)

    return {
        "decision": decision,
        "metrics": metrics_res,
        "slice_df": slice_df,
        "manifest": manifest_payload,
    }
