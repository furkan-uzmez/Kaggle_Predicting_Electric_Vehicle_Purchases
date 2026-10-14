"""Tests for fold-local stateful feature encoders."""

import json

import numpy as np
import pandas as pd
import pytest

from ev_purchase.fold_encoders import (
    DigitExtractor,
    FloorFlagEncoder,
    SmoothedTargetEncoder,
    UnsupervisedCombinedFrequencyEncoder,
    ValueFrequencyEncoder,
)


@pytest.fixture
def train_frame() -> pd.DataFrame:
    """Small deterministic training frame with repeated mass-point values."""
    return pd.DataFrame(
        {
            "Annual_Income_USD": [30000, 30000, 30000, 45000, 45000, 71234],
            "Daily_Commute_km": [5.0, 5.0, 12.5, 12.5, 33.0, 5.0],
        }
    )


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    """Validation frame containing one value unseen in training."""
    return pd.DataFrame(
        {
            "Annual_Income_USD": [30000, 45000, 99999],
            "Daily_Commute_km": [5.0, 33.0, 77.5],
        }
    )


class TestValueFrequencyEncoder:
    def test_learns_training_fold_counts(self, train_frame):
        enc = ValueFrequencyEncoder(columns=["Annual_Income_USD", "Daily_Commute_km"])
        out = enc.fit(train_frame).transform(train_frame)
        assert out["Annual_Income_USD_freq"].tolist() == [3, 3, 3, 2, 2, 1]
        assert out["Daily_Commute_km_freq"].tolist() == [3, 3, 2, 2, 1, 3]

    def test_unseen_value_gets_zero_fallback(self, train_frame, valid_frame):
        enc = ValueFrequencyEncoder(columns=["Annual_Income_USD"])
        out = enc.fit(train_frame).transform(valid_frame)
        # 99999 never appears in the training fold.
        assert out["Annual_Income_USD_freq"].tolist() == [3, 2, 0]

    def test_validation_values_do_not_alter_learned_mapping(self, train_frame, valid_frame):
        enc = ValueFrequencyEncoder(columns=["Annual_Income_USD"])
        enc.fit(train_frame)
        before = dict(enc.mappings_["Annual_Income_USD"])
        enc.transform(valid_frame)
        assert enc.mappings_["Annual_Income_USD"] == before
        assert 99999 not in enc.mappings_["Annual_Income_USD"]

    def test_does_not_mutate_input(self, train_frame):
        enc = ValueFrequencyEncoder(columns=["Annual_Income_USD"])
        original = train_frame.copy(deep=True)
        enc.fit(train_frame).transform(train_frame)
        pd.testing.assert_frame_equal(train_frame, original)

    def test_deterministic_across_calls(self, train_frame):
        enc = ValueFrequencyEncoder(columns=["Annual_Income_USD"]).fit(train_frame)
        pd.testing.assert_frame_equal(enc.transform(train_frame), enc.transform(train_frame))

    def test_transform_before_fit_raises(self, train_frame):
        with pytest.raises(RuntimeError, match="fit"):
            ValueFrequencyEncoder(columns=["Annual_Income_USD"]).transform(train_frame)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            ValueFrequencyEncoder(columns=["Annual_Income_USD"]).fit([1, 2, 3])

    def test_missing_column_raises(self, train_frame):
        with pytest.raises(ValueError, match="missing"):
            ValueFrequencyEncoder(columns=["Nope"]).fit(train_frame)

    def test_artifact_is_json_serializable(self, train_frame):
        enc = ValueFrequencyEncoder(columns=["Annual_Income_USD"]).fit(train_frame)
        payload = json.dumps(enc.get_artifact())
        assert "Annual_Income_USD" in payload


class TestFloorFlagEncoder:
    def test_detects_mass_points_by_min_count(self, train_frame):
        enc = FloorFlagEncoder(columns=["Annual_Income_USD", "Daily_Commute_km"], min_count=3)
        out = enc.fit(train_frame).transform(train_frame)
        assert out["Annual_Income_USD_is_30000"].tolist() == [1, 1, 1, 0, 0, 0]
        assert out["Daily_Commute_km_is_5.0"].tolist() == [1, 1, 0, 0, 0, 1]

    def test_min_share_threshold(self, train_frame):
        enc = FloorFlagEncoder(columns=["Annual_Income_USD"], min_share=0.4)
        enc.fit(train_frame)
        # Only 30000 reaches 3/6 = 0.5 of the training fold.
        assert enc.mass_points_["Annual_Income_USD"] == [30000]

    def test_explicit_value_list_skips_detection(self, train_frame):
        enc = FloorFlagEncoder(columns=["Annual_Income_USD"], values={"Annual_Income_USD": [45000]})
        out = enc.fit(train_frame).transform(train_frame)
        assert out["Annual_Income_USD_is_45000"].tolist() == [0, 0, 0, 1, 1, 0]

    def test_mass_points_not_derived_from_validation(self, train_frame, valid_frame):
        enc = FloorFlagEncoder(columns=["Annual_Income_USD"], min_count=3).fit(train_frame)
        out = enc.transform(valid_frame)
        assert enc.mass_points_["Annual_Income_USD"] == [30000]
        assert out["Annual_Income_USD_is_30000"].tolist() == [1, 0, 0]

    def test_does_not_mutate_input(self, train_frame):
        enc = FloorFlagEncoder(columns=["Annual_Income_USD"], min_count=3)
        original = train_frame.copy(deep=True)
        enc.fit(train_frame).transform(train_frame)
        pd.testing.assert_frame_equal(train_frame, original)

    def test_transform_before_fit_raises(self, train_frame):
        with pytest.raises(RuntimeError, match="fit"):
            FloorFlagEncoder(columns=["Annual_Income_USD"], min_count=3).transform(train_frame)

    def test_requires_one_threshold(self, train_frame):
        with pytest.raises(ValueError, match="exactly one"):
            FloorFlagEncoder(columns=["Annual_Income_USD"], min_count=3, min_share=0.5)

    def test_artifact_is_json_serializable(self, train_frame):
        enc = FloorFlagEncoder(columns=["Annual_Income_USD"], min_count=3).fit(train_frame)
        assert "30000" in json.dumps(enc.get_artifact())


class TestSmoothedTargetEncoder:
    def test_smoothing_matches_hand_computation(self):
        # Value A: 4 rows, 3 positives. Global mean = 4/8 = 0.5, m = 2.
        frame = pd.DataFrame({"v": ["A", "A", "A", "A", "B", "B", "B", "B"]})
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        enc = SmoothedTargetEncoder(columns=["v"], m=2, n_inner_folds=2, random_state=0)
        enc.fit(frame, y)
        # (3 + 2*0.5) / (4 + 2) = 4/6
        assert enc.mappings_["v"]["A"] == pytest.approx(4 / 6)
        # (1 + 2*0.5) / (4 + 2) = 2/6
        assert enc.mappings_["v"]["B"] == pytest.approx(2 / 6)
        assert enc.global_mean_ == pytest.approx(0.5)

    def test_inner_cross_fitting_differs_from_naive_encoding(self):
        """Training rows must not receive an encoding built from their own label."""
        frame = pd.DataFrame({"v": ["A"] * 6 + ["B"] * 6})
        y = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        enc = SmoothedTargetEncoder(columns=["v"], m=0, n_inner_folds=3, random_state=0)
        train_encoded = enc.fit_transform(frame, y)["v_te"].to_numpy()
        naive = enc.transform(frame)["v_te"].to_numpy()
        # Naive mapping is constant per value; cross-fitted must vary within a value.
        assert len(set(np.round(naive, 10))) == 2
        assert len(set(np.round(train_encoded[:6], 10))) > 1
        assert not np.allclose(train_encoded, naive)

    def test_positive_row_gets_lower_encoding_than_naive(self):
        """A row's own positive label must not inflate its own encoding."""
        frame = pd.DataFrame({"v": ["A"] * 4})
        y = np.array([1, 0, 0, 0])
        enc = SmoothedTargetEncoder(columns=["v"], m=0, n_inner_folds=4, random_state=0)
        encoded = enc.fit_transform(frame, y)["v_te"].to_numpy()
        # Leave-one-out style: the positive row sees 0/3, negatives see 1/3.
        assert encoded[0] == pytest.approx(0.0)
        assert encoded[1] == pytest.approx(1 / 3)

    def test_validation_uses_full_training_statistics(self):
        frame = pd.DataFrame({"v": ["A", "A", "B", "B"]})
        y = np.array([1, 0, 1, 1])
        enc = SmoothedTargetEncoder(columns=["v"], m=0, n_inner_folds=2, random_state=0)
        enc.fit(frame, y)
        out = enc.transform(pd.DataFrame({"v": ["A", "B"]}))
        assert out["v_te"].tolist() == pytest.approx([0.5, 1.0])

    def test_unseen_value_falls_back_to_global_mean(self):
        frame = pd.DataFrame({"v": ["A", "A", "B", "B"]})
        y = np.array([1, 0, 1, 0])
        enc = SmoothedTargetEncoder(columns=["v"], m=0, n_inner_folds=2, random_state=0).fit(
            frame, y
        )
        out = enc.transform(pd.DataFrame({"v": ["ZZZ"]}))
        assert out["v_te"].tolist() == pytest.approx([0.5])

    def test_never_sees_validation_labels(self):
        frame = pd.DataFrame({"v": ["A", "A", "B", "B"]})
        y = np.array([1, 0, 1, 0])
        enc = SmoothedTargetEncoder(columns=["v"], m=0, n_inner_folds=2, random_state=0).fit(
            frame, y
        )
        # transform takes no y argument at all.
        with pytest.raises(TypeError):
            enc.transform(frame, y)

    def test_deterministic_given_seed(self):
        frame = pd.DataFrame({"v": ["A"] * 6 + ["B"] * 6})
        y = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        a = SmoothedTargetEncoder(columns=["v"], m=1, n_inner_folds=3, random_state=7).fit_transform(
            frame, y
        )
        b = SmoothedTargetEncoder(columns=["v"], m=1, n_inner_folds=3, random_state=7).fit_transform(
            frame, y
        )
        pd.testing.assert_frame_equal(a, b)

    def test_does_not_mutate_input(self):
        frame = pd.DataFrame({"v": ["A", "A", "B", "B"]})
        y = np.array([1, 0, 1, 0])
        original = frame.copy(deep=True)
        SmoothedTargetEncoder(columns=["v"], n_inner_folds=2, random_state=0).fit_transform(frame, y)
        pd.testing.assert_frame_equal(frame, original)

    def test_length_mismatch_raises(self):
        frame = pd.DataFrame({"v": ["A", "B"]})
        with pytest.raises(ValueError, match="length"):
            SmoothedTargetEncoder(columns=["v"], n_inner_folds=2).fit(frame, np.array([1]))

    def test_non_binary_target_raises(self):
        frame = pd.DataFrame({"v": ["A", "B"]})
        with pytest.raises(ValueError, match="binary"):
            SmoothedTargetEncoder(columns=["v"], n_inner_folds=2).fit(frame, np.array([1, 2]))

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            SmoothedTargetEncoder(columns=["v"], n_inner_folds=2).transform(
                pd.DataFrame({"v": ["A"]})
            )

    def test_artifact_is_json_serializable(self):
        frame = pd.DataFrame({"v": ["A", "A", "B", "B"]})
        y = np.array([1, 0, 1, 0])
        enc = SmoothedTargetEncoder(columns=["v"], n_inner_folds=2, random_state=0).fit(frame, y)
        payload = json.loads(json.dumps(enc.get_artifact()))
        assert payload["global_mean"] == pytest.approx(0.5)


class TestUnsupervisedCombinedFrequencyEncoder:
    def test_disabled_by_default_uses_labeled_frame_only(self, train_frame, valid_frame):
        enc = UnsupervisedCombinedFrequencyEncoder(columns=["Annual_Income_USD"])
        assert enc.use_unlabeled is False
        enc.fit(train_frame, unlabeled_frame=valid_frame)
        assert enc.mappings_["Annual_Income_USD"][30000] == 3

    def test_enabled_combines_unlabeled_counts(self, train_frame, valid_frame):
        enc = UnsupervisedCombinedFrequencyEncoder(
            columns=["Annual_Income_USD"], use_unlabeled=True
        )
        enc.fit(train_frame, unlabeled_frame=valid_frame)
        # 3 in train + 1 in the unlabeled frame.
        assert enc.mappings_["Annual_Income_USD"][30000] == 4
        assert enc.mappings_["Annual_Income_USD"][99999] == 1

    def test_uses_no_labels(self, train_frame):
        enc = UnsupervisedCombinedFrequencyEncoder(columns=["Annual_Income_USD"])
        # fit accepts no target parameter whatsoever.
        with pytest.raises(TypeError):
            enc.fit(train_frame, np.array([1, 0, 1, 0, 1, 0]))

    def test_docstring_declares_transductive_risk(self):
        doc = UnsupervisedCombinedFrequencyEncoder.__doc__ or ""
        assert "transductive" in doc.lower()

    def test_does_not_mutate_inputs(self, train_frame, valid_frame):
        t_orig, v_orig = train_frame.copy(deep=True), valid_frame.copy(deep=True)
        enc = UnsupervisedCombinedFrequencyEncoder(
            columns=["Annual_Income_USD"], use_unlabeled=True
        )
        enc.fit(train_frame, unlabeled_frame=valid_frame).transform(train_frame)
        pd.testing.assert_frame_equal(train_frame, t_orig)
        pd.testing.assert_frame_equal(valid_frame, v_orig)

    def test_transform_before_fit_raises(self, train_frame):
        with pytest.raises(RuntimeError, match="fit"):
            UnsupervisedCombinedFrequencyEncoder(columns=["Annual_Income_USD"]).transform(
                train_frame
            )

    def test_artifact_records_flag(self, train_frame):
        enc = UnsupervisedCombinedFrequencyEncoder(
            columns=["Annual_Income_USD"], use_unlabeled=True
        ).fit(train_frame)
        payload = json.loads(json.dumps(enc.get_artifact()))
        assert payload["use_unlabeled"] is True


class TestDigitExtractor:
    def test_extracts_positive_integer_digits(self):
        frame = pd.DataFrame({"x": [12345]})
        out = DigitExtractor(columns=["x"], k_range=(0, 4)).fit(frame).transform(frame)
        assert out["x_digit_0"].tolist() == [5]
        assert out["x_digit_1"].tolist() == [4]
        assert out["x_digit_2"].tolist() == [3]
        assert out["x_digit_3"].tolist() == [2]
        assert out["x_digit_4"].tolist() == [1]

    def test_extracts_fractional_digits(self):
        frame = pd.DataFrame({"x": [12.34]})
        out = DigitExtractor(columns=["x"], k_range=(-2, 0)).fit(frame).transform(frame)
        assert out["x_digit_-1"].tolist() == [3]
        assert out["x_digit_-2"].tolist() == [4]

    def test_handles_negative_values_by_magnitude(self):
        frame = pd.DataFrame({"x": [-45]})
        out = DigitExtractor(columns=["x"], k_range=(0, 1)).fit(frame).transform(frame)
        assert out["x_digit_0"].tolist() == [5]
        assert out["x_digit_1"].tolist() == [4]

    def test_handles_zero(self):
        frame = pd.DataFrame({"x": [0]})
        out = DigitExtractor(columns=["x"], k_range=(0, 2)).fit(frame).transform(frame)
        assert out["x_digit_0"].tolist() == [0]
        assert out["x_digit_2"].tolist() == [0]

    def test_floating_point_representation_is_safe(self):
        # 0.7 is not exactly representable; naive int(x*10)%10 yields 6.
        frame = pd.DataFrame({"x": [0.7, 2.9, 8.3]})
        out = DigitExtractor(columns=["x"], k_range=(-1, -1)).fit(frame).transform(frame)
        assert out["x_digit_-1"].tolist() == [7, 9, 3]

    def test_column_names_are_ordered_deterministically(self):
        frame = pd.DataFrame({"x": [11], "y": [22]})
        out = DigitExtractor(columns=["y", "x"], k_range=(0, 1)).fit(frame).transform(frame)
        assert list(out.columns) == ["x_digit_0", "x_digit_1", "y_digit_0", "y_digit_1"]

    def test_deterministic_across_calls(self):
        frame = pd.DataFrame({"x": [12345.678]})
        enc = DigitExtractor(columns=["x"], k_range=(-3, 4)).fit(frame)
        pd.testing.assert_frame_equal(enc.transform(frame), enc.transform(frame))

    def test_does_not_mutate_input(self):
        frame = pd.DataFrame({"x": [123]})
        original = frame.copy(deep=True)
        DigitExtractor(columns=["x"], k_range=(0, 2)).fit(frame).transform(frame)
        pd.testing.assert_frame_equal(frame, original)

    def test_non_numeric_column_raises(self):
        frame = pd.DataFrame({"x": ["abc"]})
        with pytest.raises(ValueError, match="numeric"):
            DigitExtractor(columns=["x"], k_range=(0, 1)).fit(frame)

    def test_invalid_k_range_raises(self):
        frame = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="k_range"):
            DigitExtractor(columns=["x"], k_range=(3, 1)).fit(frame)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            DigitExtractor(columns=["x"], k_range=(0, 1)).transform(pd.DataFrame({"x": [1]}))

    def test_artifact_is_json_serializable(self):
        frame = pd.DataFrame({"x": [1]})
        enc = DigitExtractor(columns=["x"], k_range=(0, 2)).fit(frame)
        payload = json.loads(json.dumps(enc.get_artifact()))
        assert payload["k_range"] == [0, 2]
