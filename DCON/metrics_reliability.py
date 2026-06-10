"""Metrics for test-time segmentation reliability estimation."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-8


def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def safe_minmax(x, eps: float = EPS):
    arr = _as_numpy(x).astype(np.float64)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def dice_per_class(pred, gt, num_classes: int, include_background: bool = False, eps: float = EPS):
    pred = _as_numpy(pred).astype(np.int64)
    gt = _as_numpy(gt).astype(np.int64)
    start = 0 if include_background else 1
    dices = []
    for cls in range(start, int(num_classes)):
        p = pred == cls
        g = gt == cls
        denom = p.sum() + g.sum()
        if denom == 0:
            dices.append(1.0)
        else:
            dices.append(float((2.0 * np.logical_and(p, g).sum() + eps) / (denom + eps)))
    return np.asarray(dices, dtype=np.float64)


def mean_foreground_dice(pred, gt, num_classes: int):
    dices = dice_per_class(pred, gt, num_classes=num_classes, include_background=False)
    return float(np.mean(dices)) if dices.size else 1.0


def selective_dice(pred, gt, keep_mask, num_classes: int, eps: float = EPS):
    pred = _as_numpy(pred).astype(np.int64)
    gt = _as_numpy(gt).astype(np.int64)
    keep = _as_numpy(keep_mask).astype(bool)
    if keep.sum() == 0:
        return float("nan")
    return mean_foreground_dice(pred[keep], gt[keep], num_classes=num_classes)


def foreground_selective_dice(pred, gt, keep_mask, num_classes: int, eps: float = EPS):
    pred = _as_numpy(pred).astype(np.int64)
    gt = _as_numpy(gt).astype(np.int64)
    keep = _as_numpy(keep_mask).astype(bool)
    fg = np.logical_or(pred > 0, gt > 0)
    mask = np.logical_and(keep, fg)
    if mask.sum() == 0:
        return float("nan")
    return mean_foreground_dice(pred[mask], gt[mask], num_classes=num_classes)


def fpr_at_tpr(y_true, scores, target_tpr: float = 0.95):
    y_true = _as_numpy(y_true).astype(np.uint8).ravel()
    scores = _as_numpy(scores).astype(np.float64).ravel()
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    tpr = tp / max(float(pos.sum()), 1.0)
    fpr = fp / max(float(neg.sum()), 1.0)
    idx = np.searchsorted(tpr, target_tpr, side="left")
    if idx >= len(fpr):
        return 1.0
    return float(fpr[idx])


def binary_detection_metrics(y_true, scores):
    y_true = _as_numpy(y_true).astype(np.uint8).ravel()
    scores = _as_numpy(scores).astype(np.float64).ravel()
    valid = np.isfinite(scores)
    y_true = y_true[valid]
    scores = scores[valid]
    out = {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan"), "n": int(y_true.size)}
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return out
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["aupr"] = float(average_precision_score(y_true, scores))
    except Exception:
        out["auroc"] = _rank_auc(y_true, scores)
        out["aupr"] = _average_precision(y_true, scores)
    out["fpr95"] = fpr_at_tpr(y_true, scores, 0.95)
    return out


def _rank_auc(y_true, scores):
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = np.empty_like(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks[order] = np.arange(1, scores.size + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def _average_precision(y_true, scores):
    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / np.arange(1, sorted_true.size + 1)
    positives = max(float((y_true == 1).sum()), 1.0)
    return float((precision * (sorted_true == 1)).sum() / positives)


def correlation_metrics(reliability: Sequence[float], dice: Sequence[float]):
    rel = np.asarray(reliability, dtype=np.float64)
    dsc = np.asarray(dice, dtype=np.float64)
    valid = np.isfinite(rel) & np.isfinite(dsc)
    rel = rel[valid]
    dsc = dsc[valid]
    out = {"spearman": float("nan"), "pearson": float("nan"), "mae": float("nan"), "n": int(rel.size)}
    if rel.size == 0:
        return out
    out["mae"] = float(np.mean(np.abs(rel - dsc)))
    if rel.size < 2 or np.std(rel) < EPS or np.std(dsc) < EPS:
        return out
    try:
        from scipy.stats import pearsonr, spearmanr

        out["spearman"] = float(spearmanr(rel, dsc).correlation)
        out["pearson"] = float(pearsonr(rel, dsc).statistic)
    except Exception:
        out["spearman"] = float(np.corrcoef(_rankdata(rel), _rankdata(dsc))[0, 1])
        out["pearson"] = float(np.corrcoef(rel, dsc)[0, 1])
    return out


def _rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1)
    return ranks


def case_risk_coverage(reliability: Sequence[float], dice: Sequence[float]):
    rel = np.asarray(reliability, dtype=np.float64)
    dsc = np.asarray(dice, dtype=np.float64)
    valid = np.isfinite(rel) & np.isfinite(dsc)
    rel = rel[valid]
    dsc = dsc[valid]
    if rel.size == 0:
        return [], float("nan"), float("nan")
    order = np.argsort(-rel, kind="mergesort")
    sorted_dice = dsc[order]
    coverages = np.arange(1, sorted_dice.size + 1, dtype=np.float64) / float(sorted_dice.size)
    risks = 1.0 - np.cumsum(sorted_dice) / np.arange(1, sorted_dice.size + 1)
    aurc = float(np.trapz(risks, coverages)) if risks.size > 1 else float(risks[0])

    oracle = np.sort(dsc)[::-1]
    oracle_risks = 1.0 - np.cumsum(oracle) / np.arange(1, oracle.size + 1)
    oracle_aurc = float(np.trapz(oracle_risks, coverages)) if oracle_risks.size > 1 else float(oracle_risks[0])
    rows = [
        {"coverage": float(cov), "risk": float(risk)}
        for cov, risk in zip(coverages, risks)
    ]
    return rows, aurc, aurc - oracle_aurc


def pixel_selective_curve(pred, gt, unreliability, num_classes: int, coverages: Iterable[float]):
    pred = _as_numpy(pred).astype(np.int64)
    gt = _as_numpy(gt).astype(np.int64)
    score = _as_numpy(unreliability).astype(np.float64)
    flat_score = score.ravel()
    rows = []
    for coverage in coverages:
        coverage = float(coverage)
        if not 0.0 < coverage <= 1.0:
            continue
        keep_count = max(1, int(math.ceil(flat_score.size * coverage)))
        threshold = np.partition(flat_score, flat_score.size - keep_count)[flat_score.size - keep_count]
        keep = score >= threshold
        dsc = selective_dice(pred, gt, keep, num_classes=num_classes)
        fg_dsc = foreground_selective_dice(pred, gt, keep, num_classes=num_classes)
        rows.append(
            {
                "coverage": coverage,
                "risk": float(1.0 - dsc) if np.isfinite(dsc) else float("nan"),
                "dice": float(dsc),
                "foreground_dice": float(fg_dsc),
            }
        )
    return rows
