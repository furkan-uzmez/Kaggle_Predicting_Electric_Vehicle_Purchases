import numpy as np
import pytest

from ev_purchase.metrics import classification_metrics, validate_probabilities


def test_metrics_return_required_keys_and_float_types() -> None:
    y_true = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    result = classification_metrics(y_true, probs)
    assert set(result.keys()) == {"roc_auc", "average_precision", "log_loss"}
    for value in result.values():
        assert isinstance(value, float)
        assert not isinstance(value, bool)


def test_probability_validation_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_probabilities(np.array([0.2, 1.1]))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_probabilities(np.array([-0.1, 0.5]))


def test_probability_validation_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_probabilities(np.array([0.2, np.nan]))

    with pytest.raises(ValueError, match="finite"):
        validate_probabilities(np.array([0.2, np.inf]))


def test_probability_validation_rejects_invalid_ndim_and_empty() -> None:
    with pytest.raises(ValueError, match="one-dimensional|1D"):
        validate_probabilities(np.array([[0.2, 0.5]]))

    with pytest.raises(ValueError, match="empty|1D"):
        validate_probabilities(np.array([]))


def test_probability_validation_rejects_booleans() -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_probabilities(np.array([True, False]))  # type: ignore[arg-type]


def test_classification_metrics_rejects_one_class_y() -> None:
    y_true = np.array([0, 0, 0, 0])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="both binary classes"):
        classification_metrics(y_true, probs)


def test_classification_metrics_rejects_mismatched_lengths() -> None:
    y_true = np.array([0, 1, 0])
    probs = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="length"):
        classification_metrics(y_true, probs)


def test_classification_metrics_rejects_non_binary_y() -> None:
    y_true = np.array([0, 1, 2])
    probs = np.array([0.1, 0.9, 0.5])
    with pytest.raises(ValueError, match="binary"):
        classification_metrics(y_true, probs)
