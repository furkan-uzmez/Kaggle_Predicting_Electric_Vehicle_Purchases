"""Model factory mapping configuration model names to scikit-learn compatible estimators."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ev_purchase.preprocessing import make_linear_preprocessor


class GBDTCategoricalPrep(BaseEstimator, TransformerMixin):
    """Transformer for LightGBM and XGBoost to handle categorical features as native categories."""

    def fit(self, X: Any, y: Any = None) -> "GBDTCategoricalPrep":
        df = pd.DataFrame(X)
        self.num_cols_ = df.select_dtypes(include="number").columns.tolist()
        self.cat_cols_ = [c for c in df.columns if c not in self.num_cols_]
        self.categories_: dict[str, list[str]] = {}
        for c in self.cat_cols_:
            unique_cats = df[c].dropna().astype(str).unique().tolist()
            self.categories_[c] = sorted(unique_cats)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df_out = pd.DataFrame(X).copy()
        for c in self.cat_cols_:
            vals = df_out[c].astype(str)
            valid_mask = vals.isin(self.categories_[c]) & df_out[c].notna()
            clean_vals = np.where(valid_mask, vals, np.nan)
            cat_dtype = pd.CategoricalDtype(categories=self.categories_[c], ordered=False)
            df_out[c] = pd.Categorical(clean_vals, dtype=cat_dtype)
        return df_out


class CatBoostPrep(BaseEstimator, TransformerMixin):
    """Transformer for CatBoost categorical features."""

    def fit(self, X: Any, y: Any = None) -> "CatBoostPrep":
        df = pd.DataFrame(X)
        self.num_cols_ = df.select_dtypes(include="number").columns.tolist()
        self.cat_cols_ = [c for c in df.columns if c not in self.num_cols_]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df_out = pd.DataFrame(X).copy()
        for c in self.cat_cols_:
            df_out[c] = df_out[c].fillna("missing").astype(str)
        return df_out


SUPPORTED_MODELS = {"dummy", "logistic", "catboost", "lightgbm", "xgboost"}


def build_model(
    model_name: str,
    seed: int = 42,
    sample_frame: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Build and return an unfitted estimator or pipeline for the specified model_name.

    Args:
        model_name: Name of model ('dummy', 'logistic', 'catboost', 'lightgbm', 'xgboost').
        seed: Random seed for deterministic initialization.
        sample_frame: Sample feature DataFrame to discover schema/columns for preprocessing.
        params: Additional model-specific hyperparameters.

    Returns:
        Unfitted scikit-learn estimator or Pipeline.

    Raises:
        ValueError: If model_name is not supported.
        ImportError: If required GBDT package is missing.
    """
    clean_name = model_name.strip().lower()
    if clean_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported models: {sorted(SUPPORTED_MODELS)}"
        )

    extra_params = params or {}

    if clean_name == "dummy":
        clf = DummyClassifier(strategy="prior")
        if sample_frame is not None:
            prep = make_linear_preprocessor(sample_frame)
            return Pipeline([("preprocessor", prep), ("classifier", clf)])
        return clf

    if clean_name == "logistic":
        lr_kwargs: dict[str, Any] = {"max_iter": 1000, "random_state": seed, "solver": "lbfgs"}
        lr_kwargs.update(extra_params)
        clf = LogisticRegression(**lr_kwargs)
        if sample_frame is not None:
            prep = make_linear_preprocessor(sample_frame)
            return Pipeline([("preprocessor", prep), ("classifier", clf)])
        return clf

    if clean_name == "catboost":
        try:
            import catboost as cb
        except ImportError as exc:
            raise ImportError(
                "CatBoost is not installed. Install catboost to use the 'catboost' model."
            ) from exc

        prep = CatBoostPrep()
        cb_kwargs: dict[str, Any] = {"verbose": 0, "random_seed": seed}
        cb_kwargs.update(extra_params)

        if sample_frame is not None:
            prep.fit(sample_frame)
            cb_kwargs["cat_features"] = prep.cat_cols_

        clf = cb.CatBoostClassifier(**cb_kwargs)
        return Pipeline([("prep", prep), ("clf", clf)])

    if clean_name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBM is not installed. Install lightgbm to use the 'lightgbm' model."
            ) from exc

        lgb_kwargs: dict[str, Any] = {"random_state": seed, "verbose": -1}
        lgb_kwargs.update(extra_params)

        prep = GBDTCategoricalPrep()
        clf = lgb.LGBMClassifier(**lgb_kwargs)
        return Pipeline([("prep", prep), ("clf", clf)])

    if clean_name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install xgboost to use the 'xgboost' model."
            ) from exc

        xgb_kwargs: dict[str, Any] = {
            "random_state": seed,
            "enable_categorical": True,
            "tree_method": "hist",
        }
        xgb_kwargs.update(extra_params)

        prep = GBDTCategoricalPrep()
        clf = xgb.XGBClassifier(**xgb_kwargs)
        return Pipeline([("prep", prep), ("clf", clf)])

    raise ValueError(f"Unhandled model name '{clean_name}'")
