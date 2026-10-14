"""Fold-local stateful feature encoders for leakage-safe value-level encoding.

Every encoder in this module is fitted on a single outer-training partition and
applied unchanged to that fold's validation, test, and inference rows. Statistics
are never learned from validation data, and no encoder reads validation labels.

Each encoder exposes ``get_artifact()`` returning a JSON-serializable dict so a
training run can persist per-fold provenance alongside its metrics.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_VALUE_COLUMNS = ("Annual_Income_USD", "Daily_Commute_km")


def _require_dataframe(frame: Any, name: str = "frame") -> None:
    """Raise TypeError when the supplied object is not a pandas DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(frame).__name__}")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ValueError when any required column is absent from the frame."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")


def _require_fitted(fitted: bool, cls_name: str) -> None:
    """Raise RuntimeError when transform is called before fit."""
    if not fitted:
        raise RuntimeError(f"{cls_name} must be fitted with fit(...) before transform(...)")


def _normalize_columns(columns: Sequence[str] | None) -> list[str]:
    """Return a validated, ordered list of target column names."""
    resolved = list(DEFAULT_VALUE_COLUMNS) if columns is None else list(columns)
    if not resolved:
        raise ValueError("columns must contain at least one column name")
    return resolved


def _jsonable_mapping(mapping: dict[Any, Any]) -> dict[str, Any]:
    """Convert a value-keyed mapping into a string-keyed JSON-safe dict."""
    converted: dict[str, Any] = {}
    for key, value in mapping.items():
        converted[str(key)] = value.item() if hasattr(value, "item") else value
    return converted


class ValueFrequencyEncoder:
    """Encode how often each value of a high-cardinality column occurs.

    Counts are learned from the training fold only. A value never seen during
    ``fit`` receives ``unseen_count`` (default ``0``), which is a documented,
    deterministic fallback rather than a missing value.

    This targets the synthetic mass points in the EV dataset: a small number of
    values (for example ``Annual_Income_USD == 30000``) carry a large share of
    the rows, and their occurrence count is informative on its own.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        unseen_count: int = 0,
        suffix: str = "_freq",
    ) -> None:
        self.columns = _normalize_columns(columns)
        self.unseen_count = unseen_count
        self.suffix = suffix
        self.mappings_: dict[str, dict[Any, int]] = {}
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> "ValueFrequencyEncoder":
        """Learn per-value counts from the training-fold frame."""
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        self.mappings_ = {
            column: {
                key: int(count) for key, count in frame[column].value_counts(dropna=False).items()
            }
            for column in self.columns
        }
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map learned counts onto ``frame`` without mutating it."""
        _require_fitted(self._fitted, type(self).__name__)
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        encoded = {
            f"{column}{self.suffix}": frame[column]
            .map(self.mappings_[column])
            .fillna(self.unseen_count)
            .astype("int64")
            .to_numpy()
            for column in self.columns
        }
        return pd.DataFrame(encoded, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``frame`` and return its encoded columns."""
        return self.fit(frame).transform(frame)

    def get_artifact(self) -> dict[str, Any]:
        """Return the fitted state as a JSON-serializable dict."""
        _require_fitted(self._fitted, type(self).__name__)
        return {
            "encoder": type(self).__name__,
            "columns": list(self.columns),
            "unseen_count": self.unseen_count,
            "mappings": {
                column: _jsonable_mapping(mapping) for column, mapping in self.mappings_.items()
            },
        }


class FloorFlagEncoder:
    """Flag training-fold mass-point values with binary indicator columns.

    Mass points are either supplied explicitly through ``values`` or detected
    from the training fold using exactly one threshold: ``min_count`` (absolute
    row count) or ``min_share`` (fraction of training-fold rows). Detection never
    inspects validation or test data.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        min_count: int | None = None,
        min_share: float | None = None,
        values: dict[str, Sequence[Any]] | None = None,
    ) -> None:
        self.columns = _normalize_columns(columns)
        if values is None:
            supplied = [threshold is not None for threshold in (min_count, min_share)]
            if sum(supplied) != 1:
                raise ValueError("provide exactly one of min_count or min_share")
        if min_count is not None and (not isinstance(min_count, int) or min_count < 1):
            raise ValueError("min_count must be an integer >= 1")
        if min_share is not None and not 0.0 < float(min_share) <= 1.0:
            raise ValueError("min_share must fall in (0, 1]")
        self.min_count = min_count
        self.min_share = min_share
        self.values = values
        self.mass_points_: dict[str, list[Any]] = {}
        self._fitted = False

    def _detect(self, series: pd.Series) -> list[Any]:
        """Return training-fold values that clear the configured threshold."""
        counts = series.value_counts(dropna=False)
        threshold = (
            self.min_count
            if self.min_count is not None
            else max(1, int(np.ceil(float(self.min_share) * len(series))))
        )
        return sorted(counts[counts >= threshold].index.tolist())

    def fit(self, frame: pd.DataFrame) -> "FloorFlagEncoder":
        """Identify mass points from the training-fold frame."""
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        if self.values is not None:
            self.mass_points_ = {
                column: sorted(self.values.get(column, [])) for column in self.columns
            }
        else:
            self.mass_points_ = {column: self._detect(frame[column]) for column in self.columns}
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Emit one binary indicator column per learned mass point."""
        _require_fitted(self._fitted, type(self).__name__)
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        encoded: dict[str, np.ndarray] = {}
        for column in self.columns:
            for value in self.mass_points_[column]:
                encoded[f"{column}_is_{value}"] = (
                    frame[column].to_numpy() == value
                ).astype("int64")
        return pd.DataFrame(encoded, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``frame`` and return its indicator columns."""
        return self.fit(frame).transform(frame)

    def get_artifact(self) -> dict[str, Any]:
        """Return the fitted state as a JSON-serializable dict."""
        _require_fitted(self._fitted, type(self).__name__)
        return {
            "encoder": type(self).__name__,
            "columns": list(self.columns),
            "min_count": self.min_count,
            "min_share": self.min_share,
            "mass_points": {
                column: [str(value) for value in points]
                for column, points in self.mass_points_.items()
            },
        }


class SmoothedTargetEncoder:
    """Encode per-value smoothed target rate with inner cross-fitting.

    Two distinct mappings are produced from one ``fit`` call:

    - ``mappings_`` uses the full training fold and is applied to validation,
      test, and inference rows through ``transform``.
    - Training rows returned by ``fit_transform`` are encoded from inner folds
      that exclude each row's own target, so a row never sees its own label.

    Smoothing is additive toward the training-fold global mean::

        encoded(value) = (positives + m * global_mean) / (count + m)

    ``transform`` deliberately accepts no target argument, making it impossible
    to pass validation labels into the encoder.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        m: float = 20.0,
        n_inner_folds: int = 5,
        random_state: int = 42,
        suffix: str = "_te",
    ) -> None:
        self.columns = _normalize_columns(columns)
        if float(m) < 0:
            raise ValueError("m must be >= 0")
        if not isinstance(n_inner_folds, int) or n_inner_folds < 2:
            raise ValueError("n_inner_folds must be an integer >= 2")
        self.m = float(m)
        self.n_inner_folds = n_inner_folds
        self.random_state = random_state
        self.suffix = suffix
        self.mappings_: dict[str, dict[Any, float]] = {}
        self.global_mean_: float = float("nan")
        self._fitted = False

    def _validate_target(self, frame: pd.DataFrame, y: Any) -> np.ndarray:
        """Return a validated binary target array aligned with ``frame``."""
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D vector")
        if y_arr.shape[0] != len(frame):
            raise ValueError(
                f"y length ({y_arr.shape[0]}) must match frame length ({len(frame)})"
            )
        unique = set(np.unique(y_arr).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("y must be binary containing only {0, 1}")
        return y_arr.astype("float64")

    def _smoothed_map(self, values: pd.Series, y: np.ndarray, prior: float) -> dict[Any, float]:
        """Compute the smoothed target rate for each value in ``values``."""
        stats = pd.DataFrame({"value": values.to_numpy(), "y": y}).groupby("value")["y"]
        counts = stats.count()
        positives = stats.sum()
        smoothed = (positives + self.m * prior) / (counts + self.m)
        return {key: float(rate) for key, rate in smoothed.items()}

    def fit(self, frame: pd.DataFrame, y: Any) -> "SmoothedTargetEncoder":
        """Learn full training-fold statistics for validation/test mapping."""
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        y_arr = self._validate_target(frame, y)
        self.global_mean_ = float(y_arr.mean())
        self.mappings_ = {
            column: self._smoothed_map(frame[column], y_arr, self.global_mean_)
            for column in self.columns
        }
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map full training-fold statistics onto ``frame``.

        Takes no target argument so validation labels cannot leak in.
        """
        _require_fitted(self._fitted, type(self).__name__)
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        encoded = {
            f"{column}{self.suffix}": frame[column]
            .map(self.mappings_[column])
            .fillna(self.global_mean_)
            .astype("float64")
            .to_numpy()
            for column in self.columns
        }
        return pd.DataFrame(encoded, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame, y: Any) -> pd.DataFrame:
        """Fit, then encode training rows using inner cross-fitting.

        Each training row is encoded from a mapping built on the inner-training
        portion that excludes that row, so its own label never contributes to
        its own encoded value.
        """
        self.fit(frame, y)
        y_arr = self._validate_target(frame, y)
        n_rows = len(frame)
        effective_folds = min(self.n_inner_folds, n_rows)
        encoded = {
            f"{column}{self.suffix}": np.full(n_rows, self.global_mean_, dtype="float64")
            for column in self.columns
        }

        if effective_folds < 2:
            return pd.DataFrame(encoded, index=frame.index)

        positions = np.arange(n_rows)
        if len(set(y_arr.tolist())) < 2 or np.bincount(y_arr.astype(int)).min() < effective_folds:
            rng = np.random.default_rng(self.random_state)
            shuffled = rng.permutation(positions)
            folds = [(np.setdiff1d(positions, part), part) for part in np.array_split(shuffled, effective_folds)]
        else:
            splitter = StratifiedKFold(
                n_splits=effective_folds, shuffle=True, random_state=self.random_state
            )
            folds = list(splitter.split(positions.reshape(-1, 1), y_arr))

        for inner_train, inner_valid in folds:
            if len(inner_train) == 0 or len(inner_valid) == 0:
                continue
            inner_prior = float(y_arr[inner_train].mean())
            for column in self.columns:
                inner_map = self._smoothed_map(
                    frame[column].iloc[inner_train], y_arr[inner_train], inner_prior
                )
                mapped = (
                    frame[column]
                    .iloc[inner_valid]
                    .map(inner_map)
                    .fillna(inner_prior)
                    .astype("float64")
                    .to_numpy()
                )
                encoded[f"{column}{self.suffix}"][inner_valid] = mapped

        return pd.DataFrame(encoded, index=frame.index)

    def get_artifact(self) -> dict[str, Any]:
        """Return the fitted state as a JSON-serializable dict."""
        _require_fitted(self._fitted, type(self).__name__)
        return {
            "encoder": type(self).__name__,
            "columns": list(self.columns),
            "m": self.m,
            "n_inner_folds": self.n_inner_folds,
            "random_state": self.random_state,
            "global_mean": self.global_mean_,
            "mappings": {
                column: _jsonable_mapping(mapping) for column, mapping in self.mappings_.items()
            },
        }


class UnsupervisedCombinedFrequencyEncoder:
    """Count value occurrences across labeled and unlabeled frames, using no labels.

    When ``use_unlabeled`` is enabled, counts combine the training fold with an
    additional unlabeled frame (in practice the competition test predictors).

    This is **transductive**. Reading test predictors is legitimate for a
    competition with a fixed, fully available test set, and it uses no labels
    whatsoever. It nonetheless violates the production rule that test predictors
    must never contribute to fitting a transformer, because a deployed model
    cannot see future inputs. The flag therefore defaults to disabled and must be
    switched on deliberately, with the choice recorded in the run artifact.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        use_unlabeled: bool = False,
        unseen_count: int = 0,
        suffix: str = "_allfreq",
    ) -> None:
        self.columns = _normalize_columns(columns)
        self.use_unlabeled = bool(use_unlabeled)
        self.unseen_count = unseen_count
        self.suffix = suffix
        self.mappings_: dict[str, dict[Any, int]] = {}
        self._fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        unlabeled_frame: pd.DataFrame | None = None,
    ) -> "UnsupervisedCombinedFrequencyEncoder":
        """Learn per-value counts, optionally including unlabeled rows.

        ``unlabeled_frame`` is keyword-only and no target parameter exists, so a
        caller cannot accidentally supply labels.
        """
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        sources = [frame[self.columns]]
        if self.use_unlabeled and unlabeled_frame is not None:
            _require_dataframe(unlabeled_frame, "unlabeled_frame")
            _require_columns(unlabeled_frame, self.columns)
            sources.append(unlabeled_frame[self.columns])
        combined = pd.concat(sources, axis=0, ignore_index=True)
        self.mappings_ = {
            column: {
                key: int(count)
                for key, count in combined[column].value_counts(dropna=False).items()
            }
            for column in self.columns
        }
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map learned combined counts onto ``frame``."""
        _require_fitted(self._fitted, type(self).__name__)
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        encoded = {
            f"{column}{self.suffix}": frame[column]
            .map(self.mappings_[column])
            .fillna(self.unseen_count)
            .astype("int64")
            .to_numpy()
            for column in self.columns
        }
        return pd.DataFrame(encoded, index=frame.index)

    def get_artifact(self) -> dict[str, Any]:
        """Return the fitted state as a JSON-serializable dict."""
        _require_fitted(self._fitted, type(self).__name__)
        return {
            "encoder": type(self).__name__,
            "columns": list(self.columns),
            "use_unlabeled": self.use_unlabeled,
            "unseen_count": self.unseen_count,
            "mappings": {
                column: _jsonable_mapping(mapping) for column, mapping in self.mappings_.items()
            },
        }


class DigitExtractor:
    """Extract individual decimal digits from numeric columns.

    Digit ``k`` of value ``x`` is ``floor(|x| / 10**k) % 10``, covering integer
    places for ``k >= 0`` and fractional places for ``k < 0``. Extraction is
    stateless but still implements ``fit`` so it composes with the other
    fold-local encoders.

    Digits are taken from the absolute value, so sign is preserved by the raw
    column rather than duplicated here. Fractional digits are computed with
    decimal rounding before the modulo so that values such as ``0.7``, which are
    not exactly representable in binary floating point, yield ``7`` rather than
    ``6``.

    Only justify this family when the numeric column has documented quantization
    or source-generation structure. In the EV dataset the income column is
    integer-valued with heavy mass points, which is such a structure. Never apply
    it to opaque row or entity identifiers.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        k_range: tuple[int, int] = (-4, 3),
        suffix: str = "_digit_",
    ) -> None:
        self.columns = _normalize_columns(columns)
        self.k_range = k_range
        self.suffix = suffix
        self._fitted = False

    def _validate_k_range(self) -> list[int]:
        """Return the ordered list of digit positions to emit."""
        if (
            not isinstance(self.k_range, tuple)
            or len(self.k_range) != 2
            or not all(isinstance(bound, int) for bound in self.k_range)
        ):
            raise ValueError("k_range must be a tuple of two integers (low, high)")
        low, high = self.k_range
        if low > high:
            raise ValueError(f"k_range low ({low}) must not exceed high ({high})")
        return list(range(low, high + 1))

    def fit(self, frame: pd.DataFrame) -> "DigitExtractor":
        """Validate column presence, dtype, and digit range."""
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        self._validate_k_range()
        for column in self.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError(f"column '{column}' must be numeric for digit extraction")
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Emit one digit column per (column, position) pair."""
        _require_fitted(self._fitted, type(self).__name__)
        _require_dataframe(frame)
        _require_columns(frame, self.columns)
        positions = self._validate_k_range()
        encoded: dict[str, np.ndarray] = {}
        for column in sorted(self.columns):
            magnitude = np.abs(frame[column].to_numpy(dtype="float64"))
            for k in positions:
                scaled = magnitude / (10.0**k)
                # Round away binary representation error before flooring.
                digits = np.floor(np.round(scaled, 6)) % 10
                encoded[f"{column}{self.suffix}{k}"] = np.nan_to_num(digits, nan=-1.0).astype(
                    "int64"
                )
        return pd.DataFrame(encoded, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``frame`` and return its digit columns."""
        return self.fit(frame).transform(frame)

    def get_artifact(self) -> dict[str, Any]:
        """Return the fitted state as a JSON-serializable dict."""
        _require_fitted(self._fitted, type(self).__name__)
        return {
            "encoder": type(self).__name__,
            "columns": list(self.columns),
            "k_range": list(self.k_range),
        }
