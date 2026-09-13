import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def _validate_types_and_values(
    n_splits: int,
    n_repeats: int,
    random_state: int = 42,
) -> None:
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be integer >= 2")
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise ValueError("n_repeats must be integer >= 1")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be integer")


def _validate_n_rows(n_rows: Any) -> int:
    if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows < 1:
        raise ValueError("n_rows must be integer >= 1")
    return n_rows


def _validate_target(y: Any, n_splits: int) -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D vector")
    if y_arr.size == 0:
        raise ValueError("y must be non-empty")
    if not np.issubdtype(y_arr.dtype, np.number) and not np.issubdtype(y_arr.dtype, np.bool_):
        raise ValueError("y contains non-numeric values")
    if not np.isfinite(y_arr).all():
        raise ValueError("y contains missing or non-finite values")

    unique_vals = set(np.unique(y_arr))
    if unique_vals != {0, 1}:
        raise ValueError("y must be binary containing exactly {0, 1}")

    count_0 = int((y_arr == 0).sum())
    count_1 = int((y_arr == 1).sum())
    if count_0 < n_splits or count_1 < n_splits:
        raise ValueError(f"each class must have at least n_splits ({n_splits}) samples")

    return y_arr


def build_split_manifest(
    y: Any,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a deterministic RepeatedStratifiedKFold manifest using preallocated int64 arrays."""
    _validate_types_and_values(n_splits, n_repeats, random_state)
    y_arr = _validate_target(y, n_splits)

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    n_rows = len(y_arr)
    total_assignments = n_rows * n_repeats

    row_indices = np.empty(total_assignments, dtype=np.int64)
    repeats = np.empty(total_assignments, dtype=np.int64)
    folds = np.empty(total_assignments, dtype=np.int64)

    dummy = np.zeros((n_rows, 1))
    offset = 0
    for split_index, (_, validation_indices) in enumerate(splitter.split(dummy, y_arr)):
        repeat, fold = divmod(split_index, n_splits)
        n_val = len(validation_indices)
        row_indices[offset : offset + n_val] = validation_indices
        repeats[offset : offset + n_val] = repeat
        folds[offset : offset + n_val] = fold
        offset += n_val

    manifest = pd.DataFrame({"row_index": row_indices, "repeat": repeats, "fold": folds})
    return manifest.sort_values(["repeat", "fold", "row_index"], kind="stable").reset_index(
        drop=True
    )


def validate_split_manifest(
    manifest: pd.DataFrame,
    n_rows: int,
    n_splits: int = 5,
    n_repeats: int = 3,
    y: Any | None = None,
) -> None:
    """Validate completeness, coverage, contiguousness, and stratification of split manifest."""
    _validate_n_rows(n_rows)
    _validate_types_and_values(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    if not isinstance(manifest, pd.DataFrame):
        raise ValueError("manifest must be a pandas DataFrame")

    expected_cols = ["row_index", "repeat", "fold"]
    if list(manifest.columns) != expected_cols:
        raise ValueError(f"manifest columns must be exactly {expected_cols}")

    for col in expected_cols:
        if manifest[col].dtype != np.dtype("int64"):
            msg = f"manifest column '{col}' must have int64 dtype, got {manifest[col].dtype}"
            raise ValueError(msg)

    expected_total_rows = n_rows * n_repeats
    if len(manifest) != expected_total_rows:
        raise ValueError(
            f"manifest row count mismatch: expected {expected_total_rows}, got {len(manifest)}"
        )

    # Duplicate check per repeat must precede sorting and domain checks
    duplicates = manifest.groupby(["repeat", "row_index"]).size()
    if (duplicates > 1).any():
        raise ValueError("duplicate (repeat, row_index) assignments found in split manifest")

    # Cartesian product of (repeat, fold) pairs check
    expected_pairs = set(itertools.product(range(n_repeats), range(n_splits)))
    actual_pairs = set(zip(manifest["repeat"], manifest["fold"], strict=True))
    if actual_pairs != expected_pairs:
        missing = expected_pairs - actual_pairs
        extra = actual_pairs - expected_pairs
        msg = f"manifest (repeat, fold) pairs mismatch: missing {missing}, extra {extra}"
        raise ValueError(msg)

    actual_repeats = set(manifest["repeat"].unique())
    expected_repeats = set(range(n_repeats))
    if actual_repeats != expected_repeats:
        msg = f"manifest repeats mismatch: expected {expected_repeats}, got {actual_repeats}"
        raise ValueError(msg)

    actual_folds = set(manifest["fold"].unique())
    expected_folds = set(range(n_splits))
    if actual_folds != expected_folds:
        raise ValueError(f"manifest folds mismatch: expected {expected_folds}, got {actual_folds}")

    actual_row_indices = set(manifest["row_index"].unique())
    expected_row_indices = set(range(n_rows))
    if actual_row_indices != expected_row_indices:
        raise ValueError("row index values must be contiguous from 0 to N-1")

    sorted_df = manifest.sort_values(["repeat", "fold", "row_index"], kind="stable").reset_index(
        drop=True
    )
    if not manifest.reset_index(drop=True).equals(sorted_df):
        raise ValueError("manifest is not in canonical sorted order")

    if y is not None:
        y_arr = _validate_target(y, n_splits)
        if len(y_arr) != n_rows:
            raise ValueError(f"y length ({len(y_arr)}) does not match n_rows ({n_rows})")

        # Mathematical fold stratification validation per repeat
        for r in range(n_repeats):
            repeat_df = manifest[manifest["repeat"] == r]

            counts_cls0: list[int] = []
            counts_cls1: list[int] = []

            for f in range(n_splits):
                fold_df = repeat_df[repeat_df["fold"] == f]
                val_indices = fold_df["row_index"].to_numpy()
                y_fold = y_arr[val_indices]
                unique_classes = set(np.unique(y_fold))
                if unique_classes != {0, 1}:
                    msg = (
                        f"validation fold (repeat={r}, fold={f}) "
                        f"is not stratified (classes={unique_classes})"
                    )
                    raise ValueError(msg)

                counts_cls0.append(int((y_fold == 0).sum()))
                counts_cls1.append(int((y_fold == 1).sum()))

            if max(counts_cls0) - min(counts_cls0) > 1:
                raise ValueError(
                    f"repeat {r} class 0 counts across folds are not stratified: {counts_cls0}"
                )
            if max(counts_cls1) - min(counts_cls1) > 1:
                raise ValueError(
                    f"repeat {r} class 1 counts across folds are not stratified: {counts_cls1}"
                )
