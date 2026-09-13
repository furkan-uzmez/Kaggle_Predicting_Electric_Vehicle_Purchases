"""Fold-local preprocessing pipeline factories for EV purchase prediction."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_linear_preprocessor(
    frame: pd.DataFrame,
    id_column: str = "id",
    target_column: str = "Will_Buy_EV",
) -> ColumnTransformer:
    """Create an unfitted ColumnTransformer for linear/logistic models.

    Note on Fold-Local Output Width:
        Output width is fitted per fold based on fold training data (e.g., SimpleImputer with
        add_indicator=True only emits indicator columns when missing values exist in that fold).
        Callers must treat each fold's fitted matrix independently and not assume identical column
        counts across different training folds.

    Args:
        frame: Feature DataFrame (must not contain id or target columns).
        id_column: Column name for ID to defensively check.
        target_column: Column name for Target to defensively check.

    Returns:
        An unfitted ColumnTransformer instance.

    Raises:
        TypeError: If frame is not a pandas DataFrame.
        ValueError: If frame is empty (0 columns) or contains id/target columns.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    if len(frame.columns) == 0:
        raise ValueError("frame must contain at least one feature column")

    if id_column in frame.columns:
        raise ValueError(f"id column '{id_column}' must be excluded before preprocessing")
    if target_column in frame.columns:
        raise ValueError(f"target column '{target_column}' must be excluded before preprocessing")

    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]

    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    return ColumnTransformer(transformers=transformers)
