"""
Evaluation metrics for CPT profile reconstruction and Robertson SBT zone
classification, matching the definitions in paper_draft.txt Section 3.4.

Regression metrics (r2_score, mae, rmse, mape) operate on flat arrays of
depth-position values in physical units (MPa for qc, kPa for fs).

Classification metrics (confusion_matrix, per_class_accuracy, multiclass_mcc)
operate on integer zone labels 1-9 and are used to compare the Robertson
SBT zone computed from a reconstructed profile against the zone computed
from the measured profile at the same depth-positions.
"""

import numpy as np

N_ZONES = 9   # Robertson (1990) SBTn zones, labelled 1-9


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0 if ss_res > 0.0 else 1.0
    return float(1.0 - ss_res / ss_tot)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    """Mean absolute percentage error, in percent.

    Only meaningful for channels bounded away from zero — qc has a plausibility
    floor of 0.01 MPa throughout this project (cptfm/sources/usgs.py, bro.py),
    so eps=1e-3 only guards against exact-zero division without materially
    biasing the result. Not used for fs, whose measured values approach zero
    in soft near-surface clay.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_ZONES) -> np.ndarray:
    """Rows are true zone (1-indexed), columns are predicted zone. Shape (n_classes, n_classes)."""
    y_true = np.asarray(y_true, dtype=np.int64) - 1
    y_pred = np.asarray(y_pred, dtype=np.int64) - 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """One-vs-rest accuracy per class: (TP + TN) / total, for each class in turn."""
    total = cm.sum()
    n = cm.shape[0]
    acc = np.zeros(n, dtype=np.float64)
    for k in range(n):
        tp = cm[k, k]
        tn = total - cm[k, :].sum() - cm[:, k].sum() + tp
        acc[k] = (tp + tn) / total if total > 0 else np.nan
    return acc


def overall_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else float("nan")


def multiclass_mcc(cm: np.ndarray) -> float:
    """Generalized (Gorodkin 2004) Matthews correlation coefficient over a
    multi-class confusion matrix, reducing to the standard binary MCC when
    n_classes == 2. Equivalent to sklearn.metrics.matthews_corrcoef.
    """
    cm = cm.astype(np.float64)
    s = cm.sum()
    if s == 0:
        return float("nan")
    t_k = cm.sum(axis=1)   # true count per class
    p_k = cm.sum(axis=0)   # predicted count per class
    c = np.trace(cm)

    numerator = c * s - np.dot(t_k, p_k)
    denom = np.sqrt((s ** 2 - np.dot(p_k, p_k)) * (s ** 2 - np.dot(t_k, t_k)))
    if denom == 0.0:
        return 0.0
    return float(numerator / denom)


def macro_ovr_mcc(cm: np.ndarray) -> float:
    """Macro-average of the one-vs-rest binary MCC computed separately per class."""
    cm = cm.astype(np.float64)   # avoid int64 overflow in the four-way product below
    n = cm.shape[0]
    total = cm.sum()
    vals = []
    for k in range(n):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        vals.append(0.0 if denom == 0 else (tp * tn - fp * fn) / denom)
    return float(np.mean(vals))
