"""Evaluation utilities for reliability-gated TTA experiments.

Ground truth is consumed only by this module after prediction is complete.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def dice_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int, eps: float = 1e-7) -> np.ndarray:
    """Compute dense-mask Dice per class for one volume."""

    scores = np.zeros((num_classes,), dtype=np.float64)
    for cls_idx in range(num_classes):
        pred_c = pred == cls_idx
        gt_c = gt == cls_idx
        denom = pred_c.sum() + gt_c.sum()
        scores[cls_idx] = (2.0 * np.logical_and(pred_c, gt_c).sum() + eps) / (denom + eps)
    return scores


def _connected_components(mask: np.ndarray) -> int:
    """Count connected components in a 3D binary mask using scipy when present."""

    if mask.sum() == 0:
        return 0
    try:
        from scipy import ndimage

        _, count = ndimage.label(mask)
        return int(count)
    except Exception:
        # Lightweight fallback: enough to flag severe fragmentation without
        # adding a hard dependency.
        visited = np.zeros(mask.shape, dtype=bool)
        coords = np.argwhere(mask)
        count = 0
        neighbors = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        for start in coords:
            z, y, x = map(int, start)
            if visited[z, y, x]:
                continue
            count += 1
            stack = [(z, y, x)]
            visited[z, y, x] = True
            while stack:
                cz, cy, cx = stack.pop()
                for dz, dy, dx in neighbors:
                    nz, ny, nx = cz + dz, cy + dy, cx + dx
                    if (
                        0 <= nz < mask.shape[0]
                        and 0 <= ny < mask.shape[1]
                        and 0 <= nx < mask.shape[2]
                        and mask[nz, ny, nx]
                        and not visited[nz, ny, nx]
                    ):
                        visited[nz, ny, nx] = True
                        stack.append((nz, ny, nx))
        return count


def evaluate_volume_failures(
    scan_id: str,
    pred: np.ndarray,
    gt: np.ndarray,
    label_names: Sequence[str],
    r_class: np.ndarray | None = None,
    r_region: np.ndarray | None = None,
    min_area: int = 20,
    low_dice_threshold: float = 0.5,
    overseg_ratio: float = 1.5,
    fragmentation_components: int = 3,
) -> Dict[str, float | int | str]:
    """Evaluate Dice and failure modes for one completed target volume."""

    pred = np.asarray(pred).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)
    num_classes = len(label_names)
    dice = dice_per_class(pred, gt, num_classes)

    fg_dice = dice[1:] if num_classes > 1 else dice
    row: Dict[str, float | int | str] = {
        "scan_id": scan_id,
        "mean_dice": float(np.mean(fg_dice)),
        "low_dice_case": int(float(np.mean(fg_dice)) < low_dice_threshold),
        "FP": int(np.logical_and(pred > 0, gt == 0).sum()),
        "FN": int(np.logical_and(pred == 0, gt > 0).sum()),
    }
    for cls_idx, name in enumerate(label_names):
        row[f"dice_{name}"] = float(dice[cls_idx])

    absent = missed = overseg = fragmented = 0
    absent_opportunities = present_opportunities = 0
    for cls_idx in range(1, num_classes):
        pred_area = int((pred == cls_idx).sum())
        gt_area = int((gt == cls_idx).sum())
        pred_present = pred_area > min_area
        gt_present = gt_area > min_area
        if not gt_present:
            absent_opportunities += 1
            absent += int(pred_present)
        else:
            present_opportunities += 1
            missed += int(not pred_present)
            overseg += int(pred_area > overseg_ratio * max(gt_area, 1))
        fragmented += int(_connected_components(pred == cls_idx) > fragmentation_components)

    row.update(
        {
            "absent_hallucination_count": absent,
            "absent_hallucination_rate": float(absent / max(absent_opportunities, 1)),
            "missed_organ_count": missed,
            "missed_organ_rate": float(missed / max(present_opportunities, 1)),
            "over_segmentation_count": overseg,
            "fragmentation_count": fragmented,
        }
    )

    if r_class is not None:
        r_class = np.asarray(r_class, dtype=np.float64)
        row["mean_R_class_fg"] = float(np.mean(r_class[1:])) if r_class.size > 1 else float(np.mean(r_class))
        row["min_R_class_fg"] = float(np.min(r_class[1:])) if r_class.size > 1 else float(np.min(r_class))
        for cls_idx, name in enumerate(label_names):
            if cls_idx < r_class.size:
                row[f"R_class_{name}"] = float(r_class[cls_idx])
    if r_region is not None:
        r_region = np.asarray(r_region, dtype=np.float64)
        row["mean_R_region"] = float(np.mean(r_region))
        row["p10_R_region"] = float(np.percentile(r_region, 10))
    return row


def summarize_rows(rows: Sequence[Dict[str, float | int | str]], label_names: Sequence[str]) -> Dict[str, float]:
    """Aggregate case-level metrics into one method-level summary."""

    if not rows:
        return {}
    out: Dict[str, float] = {"num_cases": float(len(rows))}
    numeric_keys = [
        "mean_dice",
        "low_dice_case",
        "absent_hallucination_count",
        "absent_hallucination_rate",
        "missed_organ_count",
        "missed_organ_rate",
        "over_segmentation_count",
        "fragmentation_count",
        "FP",
        "FN",
        "mean_R_class_fg",
        "min_R_class_fg",
        "mean_R_region",
        "p10_R_region",
    ]
    for name in label_names:
        numeric_keys.append(f"dice_{name}")

    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and row[key] != ""]
        if values:
            if key.endswith("_count") or key in {"FP", "FN"}:
                out[key] = float(np.sum(values))
            elif key == "low_dice_case":
                out["low_dice_rate"] = float(np.mean(values))
            else:
                out[key] = float(np.mean(values))
    return out


def write_csv(path: str | os.PathLike[str], rows: Sequence[Dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV with a stable union schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_debug_arrays(
    out_dir: str | os.PathLike[str],
    scan_id: str,
    pred: np.ndarray,
    r_region: np.ndarray | None,
    r_class: np.ndarray | None,
) -> None:
    """Persist prediction and reliability arrays for post-hoc debugging."""

    debug_dir = Path(out_dir) / "debug_arrays"
    debug_dir.mkdir(parents=True, exist_ok=True)
    np.save(debug_dir / f"{scan_id}_pred.npy", pred.astype(np.uint8))
    if r_region is not None:
        np.save(debug_dir / f"{scan_id}_R_region.npy", np.asarray(r_region, dtype=np.float32))
    if r_class is not None:
        np.save(debug_dir / f"{scan_id}_R_class.npy", np.asarray(r_class, dtype=np.float32))


def write_top_failure_cases(
    out_dir: str | os.PathLike[str],
    rows: Sequence[Dict[str, object]],
    top_k: int = 10,
) -> None:
    """Write the worst cases by hallucination count and Dice into a debug folder."""

    out_dir = Path(out_dir)
    top_dir = out_dir / "top_failure_cases"
    top_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("absent_hallucination_count", 0)),
            float(row.get("mean_dice", 1.0)),
        ),
    )[:top_k]
    write_csv(top_dir / "top_failure_cases.csv", sorted_rows)

    debug_dir = out_dir / "debug_arrays"
    for row in sorted_rows:
        scan_id = str(row["scan_id"])
        for suffix in ("pred", "R_region", "R_class"):
            src = debug_dir / f"{scan_id}_{suffix}.npy"
            if src.exists():
                shutil.copy2(src, top_dir / src.name)
