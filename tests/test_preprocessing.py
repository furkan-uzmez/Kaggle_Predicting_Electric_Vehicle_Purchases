"""Tests for preprocessing and feature policy modules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from ev_purchase.features import (
    apply_feature_policy,
    base_features,
    get_feature_policy,
    list_feature_policies,
    register_feature_policy,
    reset_feature_policies,
    unregister_feature_policy,
)
from ev_purchase.preprocessing import make_linear_preprocessor


def test_linear_preprocessor_returns_unfitted_transformer() -> None:
    """Verify that make_linear_preprocessor returns an unfitted ColumnTransformer."""
    train = pd.DataFrame({"age": [20.0, 30.0], "region": ["a", "b"]})
    preprocessor = make_linear_preprocessor(train)

    with pytest.raises(NotFittedError):
        preprocessor.transform(train)


def test_linear_preprocessor_handles_unseen_categories() -> None:
    """Verify unseen categories at transform time produce identical column counts."""
    train = pd.DataFrame({"age": [20.0, None, 40.0], "region": ["a", "b", "a"]})
    validation = pd.DataFrame({"age": [None], "region": ["never_seen"]})
    preprocessor = make_linear_preprocessor(train)

    transformed_train = preprocessor.fit_transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert transformed_train.shape[1] == transformed_validation.shape[1]


def test_preprocessor_does_not_include_id_or_target() -> None:
    """Verify preprocessor works on model frame with id and target dropped."""
    frame = pd.DataFrame({"id": [1], "x": [2.0], "Will_Buy_EV": [0]})
    model_frame = frame.drop(columns=["id", "Will_Buy_EV"])
    preprocessor = make_linear_preprocessor(model_frame)
    assert preprocessor.fit_transform(model_frame).shape[1] == 1


def test_preprocessor_defensively_rejects_id_or_target() -> None:
    """Verify preprocessor defensively rejects frames containing id or target columns."""
    frame_with_id = pd.DataFrame({"id": [1, 2], "age": [20.0, 30.0]})
    with pytest.raises(ValueError, match="id column 'id'"):
        make_linear_preprocessor(frame_with_id)

    frame_with_target = pd.DataFrame({"Will_Buy_EV": [0, 1], "age": [20.0, 30.0]})
    with pytest.raises(ValueError, match="target column 'Will_Buy_EV'"):
        make_linear_preprocessor(frame_with_target)

    # Custom column names
    frame_custom = pd.DataFrame({"user_id": [1, 2], "target": [0, 1], "age": [20.0, 30.0]})
    with pytest.raises(ValueError, match="user_id"):
        make_linear_preprocessor(frame_custom, id_column="user_id", target_column="target")


def test_preprocessor_input_validation() -> None:
    """Verify input validation for make_linear_preprocessor."""
    with pytest.raises(TypeError, match="frame must be a pandas DataFrame"):
        make_linear_preprocessor([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one feature column"):
        make_linear_preprocessor(pd.DataFrame())


def test_preprocessor_handles_missing_values() -> None:
    """Verify missing value imputation for numeric and categorical columns."""
    df = pd.DataFrame(
        {
            "age": [25.0, np.nan, 35.0, 45.0],
            "city": ["NY", np.nan, "LA", "NY"],
        }
    )
    preprocessor = make_linear_preprocessor(df)
    transformed = preprocessor.fit_transform(df)

    # SimpleImputer with add_indicator=True adds a missing indicator column for numeric
    # No NaNs in transformed output
    assert not np.isnan(transformed).any()


def test_preprocessor_determinism() -> None:
    """Verify deterministic construction and column ordering."""
    df = pd.DataFrame({"b": [1.0, 2.0], "a": ["x", "y"], "c": [10, 20]})
    prep1 = make_linear_preprocessor(df)
    prep2 = make_linear_preprocessor(df)

    t1 = prep1.fit_transform(df)
    t2 = prep2.fit_transform(df)

    np.testing.assert_array_equal(t1, t2)


def test_base_feature_policy_non_mutation() -> None:
    """Verify feature policy returns copy and does not mutate input."""
    original = pd.DataFrame({"id": [1, 2], "age": [30.0, 40.0], "city": ["A", "B"]})
    copy_before = original.copy(deep=True)

    res_base = base_features(original)
    pd.testing.assert_frame_equal(original, copy_before)
    pd.testing.assert_frame_equal(res_base, original)
    assert res_base is not original

    res_applied = apply_feature_policy("base", original)
    pd.testing.assert_frame_equal(original, copy_before)
    pd.testing.assert_frame_equal(res_applied, original)


def test_feature_policy_lookup_and_unknown_policy() -> None:
    """Verify policy lookup and error handling for unknown policies."""
    assert "base" in list_feature_policies()
    assert get_feature_policy("base") is base_features

    with pytest.raises(ValueError, match="Unknown feature policy 'nonexistent'"):
        get_feature_policy("nonexistent")

    with pytest.raises(TypeError, match="frame must be a pandas DataFrame"):
        apply_feature_policy("base", "not_a_df")  # type: ignore[arg-type]


def test_register_feature_policy() -> None:
    """Verify registering a custom feature policy."""
    reset_feature_policies()

    def custom_policy(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "age" in df_copy:
            df_copy["age_squared"] = df_copy["age"] ** 2
        return df_copy

    register_feature_policy("custom_test", custom_policy)
    assert "custom_test" in list_feature_policies()

    df = pd.DataFrame({"age": [2.0, 3.0]})
    df_copy = df.copy()
    res = apply_feature_policy("custom_test", df)

    assert "age_squared" in res.columns
    assert res["age_squared"].tolist() == [4.0, 9.0]
    pd.testing.assert_frame_equal(df, df_copy)
    reset_feature_policies()


def test_linear_preprocessor_fold_dependent_output_width() -> None:
    """Verify that fit_transform width can differ across folds depending on presence of NaNs."""
    fold_no_nan = pd.DataFrame({"age": [20.0, 30.0, 40.0]})
    fold_with_nan = pd.DataFrame({"age": [20.0, np.nan, 40.0]})

    prep1 = make_linear_preprocessor(fold_no_nan)
    prep2 = make_linear_preprocessor(fold_with_nan)

    out1 = prep1.fit_transform(fold_no_nan)
    out2 = prep2.fit_transform(fold_with_nan)

    assert out1.shape[1] == 1
    assert out2.shape[1] == 2
    assert out1.shape[1] != out2.shape[1]


def test_linear_preprocessor_fitted_width_invariance_on_transform() -> None:
    """Verify transform output width matches fitted width regardless of NaNs/unseen categories."""
    train_fold = pd.DataFrame({"age": [20.0, 30.0, 40.0], "region": ["a", "b", "a"]})
    val_fold = pd.DataFrame({"age": [np.nan, 25.0], "region": ["unseen_category", "a"]})

    prep = make_linear_preprocessor(train_fold)
    fitted_out = prep.fit_transform(train_fold)
    transform_out = prep.transform(val_fold)

    assert transform_out.shape[1] == fitted_out.shape[1]


def test_register_feature_policy_rejects_duplicate_without_overwrite() -> None:
    """Verify registering a duplicate policy name raises ValueError unless overwrite=True."""
    reset_feature_policies()

    def policy1(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def policy2(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    register_feature_policy("dup_policy", policy1)

    with pytest.raises(ValueError, match="already registered"):
        register_feature_policy("dup_policy", policy2)

    register_feature_policy("dup_policy", policy2, overwrite=True)
    assert get_feature_policy("dup_policy") is policy2

    reset_feature_policies()


def test_unregister_and_reset_feature_policies() -> None:
    """Verify unregistering a policy and resetting registry to built-ins."""
    reset_feature_policies()

    def policy1(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    register_feature_policy("temp_p", policy1)
    assert "temp_p" in list_feature_policies()

    unregister_feature_policy("temp_p")
    assert "temp_p" not in list_feature_policies()

    register_feature_policy("temp_p2", policy1)
    reset_feature_policies()
    assert list_feature_policies() == ["base"]


def test_cannot_unregister_or_clobber_built_in_base_policy() -> None:
    """Verify built-in 'base' policy cannot be unregistered or silently overwritten."""
    reset_feature_policies()

    with pytest.raises(ValueError, match="built-in"):
        unregister_feature_policy("base")

    def fake_base(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    with pytest.raises(ValueError, match="already registered"):
        register_feature_policy("base", fake_base)

    reset_feature_policies()


def test_apply_feature_policy_raises_on_mutation() -> None:
    """Verify apply_feature_policy raises RuntimeError when a policy mutates input."""
    reset_feature_policies()

    def mutating_policy(df: pd.DataFrame) -> pd.DataFrame:
        df["mutated_col"] = 999
        return df

    register_feature_policy("mutating", mutating_policy)
    test_df = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(RuntimeError, match="mutated its input"):
        apply_feature_policy("mutating", test_df)

    reset_feature_policies()
