import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def validate_probabilities(probabilities: np.ndarray) -> None:
    """Validate that probabilities is a finite 1D vector with values in [0, 1]."""
    if isinstance(probabilities, bool):
        raise TypeError("probabilities must be a numpy array, not bool")

    if not isinstance(probabilities, np.ndarray):
        probabilities_arr = np.asarray(probabilities)
    else:
        probabilities_arr = probabilities

    if np.issubdtype(probabilities_arr.dtype, np.bool_):
        raise TypeError("probabilities vector cannot be boolean")

    if not np.issubdtype(probabilities_arr.dtype, np.number):
        raise ValueError("probabilities must contain numeric values")

    if probabilities_arr.ndim != 1 or probabilities_arr.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional vector")

    if not np.isfinite(probabilities_arr).all():
        raise ValueError("probabilities must be a finite one-dimensional vector")

    if ((probabilities_arr < 0) | (probabilities_arr > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate ROC-AUC, PR-AUC, and LogLoss for binary classification probabilities."""
    validate_probabilities(probabilities)

    if isinstance(y_true, bool):
        raise TypeError("y_true must be a numpy array, not bool")

    y_arr = np.asarray(y_true)

    if y_arr.ndim != 1:
        raise ValueError("y_true must be a 1D vector")

    if len(y_arr) != len(probabilities):
        raise ValueError(
            f"y_true length ({len(y_arr)}) must match probabilities length ({len(probabilities)})"
        )

    if not np.issubdtype(y_arr.dtype, np.number) and not np.issubdtype(y_arr.dtype, np.bool_):
        raise ValueError("y_true contains non-numeric values")

    if not np.isfinite(y_arr).all():
        raise ValueError("y_true contains missing or non-finite values")

    unique_vals = set(np.unique(y_arr))
    if not unique_vals.issubset({0, 1}):
        raise ValueError("y_true must be binary containing only 0 and 1 values")

    if unique_vals != {0, 1}:
        raise ValueError("y_true must contain both binary classes {0, 1}")

    return {
        "roc_auc": float(roc_auc_score(y_arr, probabilities)),
        "average_precision": float(average_precision_score(y_arr, probabilities)),
        "log_loss": float(log_loss(y_arr, probabilities, labels=[0, 1])),
    }
