"""Evaluation kept separate so ground truth cannot leak into detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .data import validate_gt


def score_metrics(gt: np.ndarray, score: np.ndarray) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64)
    gt = validate_gt(gt, score.shape)
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    labels = gt.astype(np.uint8).ravel()
    values = score.ravel()
    fpr, tpr, _ = roc_curve(labels, values)
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr_at_95 = float(fpr[eligible[0]]) if eligible.size else 1.0
    k = max(1, int(np.ceil(0.01 * values.size)))
    top = np.argpartition(values, -k)[-k:]
    return {
        "AP": float(average_precision_score(labels, values)),
        "ROC_AUC": float(roc_auc_score(labels, values)),
        "FPR_at_95_TPR": fpr_at_95,
        "precision_at_top_1pct": float(labels[top].mean()),
    }


def compare_scores(
    gt: np.ndarray,
    original: np.ndarray,
    calibrated: np.ndarray,
) -> dict[str, dict[str, float]]:
    base = score_metrics(gt, original)
    magcal = score_metrics(gt, calibrated)
    delta = {key: magcal[key] - base[key] for key in base}
    return {"original": base, "calibrated": magcal, "delta": delta}


def percentile_rank(score: np.ndarray) -> np.ndarray:
    """Return [0,1] ranks where larger values are better detections."""

    flat = np.asarray(score, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, flat.size)
    return ranks.reshape(score.shape)
