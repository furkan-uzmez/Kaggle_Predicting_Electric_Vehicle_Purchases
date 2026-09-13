from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetContract:
    id_column: str
    target_column: str
    negative_label: str = "No"
    positive_label: str = "Yes"


def load_frames(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test dataframes from CSV files."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def validate_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    contract: DatasetContract,
) -> None:
    """Validate schema and data invariants for train and test dataframes."""
    if train.columns.has_duplicates or test.columns.has_duplicates:
        raise ValueError("duplicate column names detected in dataframes")

    if contract.target_column not in train:
        raise ValueError("target missing from train")
    if contract.target_column in test:
        raise ValueError("target must not appear in test")

    if contract.id_column not in train or contract.id_column not in test:
        raise ValueError("ID column missing")

    if train[contract.id_column].isna().any() or test[contract.id_column].isna().any():
        raise ValueError("IDs contain missing values")

    expected_cols = train.drop(columns=[contract.target_column]).columns.tolist()
    if test.columns.tolist() != expected_cols:
        raise ValueError("train/test feature schema mismatch")

    if not train[contract.id_column].is_unique or not test[contract.id_column].is_unique:
        raise ValueError("IDs must be unique")

    if train[contract.id_column].isin(test[contract.id_column]).any():
        raise ValueError("train/test ID overlap")

    if train[contract.target_column].isna().any():
        raise ValueError("target has missing labels")

    unique_targets = set(train[contract.target_column].dropna().unique())
    expected_labels = {contract.negative_label, contract.positive_label}
    if unique_targets != expected_labels:
        raise ValueError(f"target labels must be exactly {expected_labels}, got {unique_targets}")


def encode_target(target: Any, contract: DatasetContract) -> np.ndarray:
    """Encode raw target series into integer binary labels 0 and 1 without mutating input."""
    target_series = pd.Series(target)
    if target_series.isna().any():
        raise ValueError("target contains missing values")

    unique_vals = set(target_series.unique())
    expected_labels = {contract.negative_label, contract.positive_label}
    if not unique_vals.issubset(expected_labels):
        raise ValueError(f"unknown label found in target: {unique_vals - expected_labels}")

    mapping = {contract.negative_label: 0, contract.positive_label: 1}
    return target_series.map(mapping).astype("int64").to_numpy()
