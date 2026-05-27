import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> dict[str, float]:
    if mask is not None:
        valid = mask.astype(bool)
        if valid.sum() == 0:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
