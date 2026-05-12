#!/usr/bin/env python3
"""Shared utilities for DCON segmentation failure analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io as skio


IMAGE_EXTENSIONS = (
    ".nii.gz",
    ".nii",
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)

FAILURE_TYPES = [
    "absent_hallucination",
    "missed_organ",
    "over_segmentation",
    "under_segmentation",
    "fragmentation",
    "topology_suspicious",
    "small_organ_failure",
    "normal",
]


@dataclass
class FailureThresholds:
    """Configurable thresholds used to assign failure labels."""

    min_pred_area: float = 10.0
    min_gt_area: float = 10.0
    low_dice: float = 0.30
    over_area_ratio: float = 1.50
    under_area_ratio: float = 0.50
    max_components: int = 3
    min_largest_component_ratio: float = 0.75
    small_organ_dice: float = 0.60
    topology_max_components: int = 3
    topology_min_largest_component_ratio: float = 0.70
    foreground_max_components: int = 10
    fp_fn_dominance_ratio: float = 1.25


def parse_int_set(values: Optional[Sequence[str] | str]) -> set[int]:
    """Parse CLI-provided integer ids from comma-separated strings or lists."""

    if values is None:
        return set()
    if isinstance(values, str):
        raw_items = values.replace(",", " ").split()
    else:
        raw_items: List[str] = []
        for value in values:
            raw_items.extend(str(value).replace(",", " ").split())
    return {int(item) for item in raw_items if item.strip()}


def load_class_map(
    path: Path,
    small_organ_ids: Optional[Iterable[int]] = None,
    cardiac_class_ids: Optional[Iterable[int]] = None,
) -> Tuple[Dict[int, str], set[int], set[int]]:
    """Load class id/name mapping and optional semantic groups from JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    class_names: Dict[int, str] = {}
    small_ids: set[int] = set()
    cardiac_ids: set[int] = set()

    if isinstance(data, dict) and "classes" in data:
        for item in data["classes"]:
            cid = int(item["id"])
            class_names[cid] = str(item.get("name", cid))
            if item.get("small", False):
                small_ids.add(cid)
            if item.get("cardiac", False) or item.get("topology", False):
                cardiac_ids.add(cid)
    elif isinstance(data, dict) and "class_map" in data:
        for key, value in data["class_map"].items():
            class_names[int(key)] = str(value)
    elif isinstance(data, dict):
        for key, value in data.items():
            if str(key).isdigit():
                class_names[int(key)] = str(value)
    else:
        raise ValueError(f"Unsupported class map format: {path}")

    for key in ("small_organs", "small_organ_ids"):
        if isinstance(data, dict) and key in data:
            small_ids.update(int(item) for item in data[key])
    for key in ("cardiac_classes", "cardiac_class_ids", "topology_classes"):
        if isinstance(data, dict) and key in data:
            cardiac_ids.update(int(item) for item in data[key])

    if small_organ_ids:
        small_ids.update(int(item) for item in small_organ_ids)
    if cardiac_class_ids:
        cardiac_ids.update(int(item) for item in cardiac_class_ids)

    return dict(sorted(class_names.items())), small_ids, cardiac_ids


def read_array(path: Path) -> np.ndarray:
    """Read a 2D/3D image or mask from NIfTI, NPY/NPZ, or common image formats."""

    name = path.name.lower()
    if name.endswith(".npy"):
        return np.load(path)
    if name.endswith(".npz"):
        data = np.load(path)
        for key in ("mask", "pred", "gt", "image", "entropy", "arr_0"):
            if key in data:
                return data[key]
        first_key = sorted(data.files)[0]
        return data[first_key]
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        try:
            import SimpleITK as sitk

            return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        except Exception:
            try:
                import nibabel as nib

                return np.asarray(nib.load(str(path)).get_fdata())
            except Exception as exc:
                raise ImportError(
                    "Reading NIfTI requires SimpleITK or nibabel to be installed."
                ) from exc
    arr = skio.imread(path)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., 0]
    return np.asarray(arr)


def normalize_volume_shape(arr: np.ndarray) -> np.ndarray:
    """Return an array as Z x H x W, preserving 2D data as one slice."""

    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def cast_mask(arr: np.ndarray) -> np.ndarray:
    """Convert mask-like arrays to integer labels."""

    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return arr.astype(np.int32)


def discover_files(directory: Path) -> Dict[str, Path]:
    """Index supported files by case id with common suffixes removed."""

    files: Dict[str, Path] = {}
    if not directory:
        return files
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue
        stem = strip_known_suffix(path.name)
        files[stem] = path
    return files


def strip_known_suffix(name: str) -> str:
    """Remove extension and common role suffix from a file name."""

    lowered = name.lower()
    for ext in IMAGE_EXTENSIONS:
        if lowered.endswith(ext):
            name = name[: -len(ext)]
            break
    for suffix in ("_pred", "-pred", "_prediction", "_gt", "-gt", "_label", "_mask", "_image", "-image"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def paired_cases(pred_dir: Path, gt_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Return cases present in both prediction and ground-truth directories."""

    pred_files = discover_files(pred_dir)
    gt_files = discover_files(gt_dir)
    shared = sorted(set(pred_files) & set(gt_files))
    if not shared:
        raise FileNotFoundError(
            f"No paired cases found between pred_dir={pred_dir} and gt_dir={gt_dir}."
        )
    return [(case_id, pred_files[case_id], gt_files[case_id]) for case_id in shared]


def discover_volume_triplets(volume_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Discover <case>_image/_gt/_pred triplets from one visualization volume directory."""

    cases: Dict[str, Dict[str, Path]] = {}
    for path in sorted(volume_dir.iterdir()):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if not any(lowered.endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue
        case_id = strip_known_suffix(path.name)
        if "_pred." in lowered or lowered.endswith("_pred.nii.gz") or lowered.endswith("_prediction.nii.gz"):
            role = "pred"
        elif "_gt." in lowered or lowered.endswith("_gt.nii.gz"):
            role = "gt"
        elif "_label." in lowered or lowered.endswith("_label.nii.gz"):
            role = "gt"
        elif "_image." in lowered or lowered.endswith("_image.nii.gz"):
            role = "image"
        elif "_entropy." in lowered or lowered.endswith("_entropy.nii.gz"):
            role = "entropy"
        else:
            continue
        cases.setdefault(case_id, {})[role] = path

    complete = {
        case_id: parts
        for case_id, parts in cases.items()
        if "pred" in parts and "gt" in parts
    }
    if not complete:
        raise FileNotFoundError(f"No _pred/_gt volume pairs found in {volume_dir}")
    return dict(sorted(complete.items()))


def optional_case_path(directory: Optional[Path], case_id: str) -> Optional[Path]:
    """Find an optional image or entropy file matching a case id."""

    if directory is None:
        return None
    files = discover_files(directory)
    return files.get(case_id)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""

    if den == 0:
        return default
    return float(num) / float(den)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute binary segmentation overlap and error counts."""

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    pred_area = int(pred_b.sum())
    gt_area = int(gt_b.sum())
    dice = 1.0 if pred_area + gt_area == 0 else safe_div(2 * tp, pred_area + gt_area)
    iou = 1.0 if tp + fp + fn == 0 else safe_div(tp, tp + fp + fn)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_ratio": float(safe_div(fp, pred_area, 0.0)),
        "fn_ratio": float(safe_div(fn, gt_area, 0.0)),
        "pred_area": pred_area,
        "gt_area": gt_area,
        "area_error_ratio": float(safe_div(pred_area - gt_area, gt_area, math.inf if pred_area > 0 else 0.0)),
    }


def component_metrics(mask: np.ndarray) -> Dict[str, float]:
    """Compute connected-component count, largest component, centroid, and bbox."""

    mask_b = mask.astype(bool)
    area = int(mask_b.sum())
    if area == 0:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "bbox_x1": np.nan,
            "bbox_y1": np.nan,
            "bbox_x2": np.nan,
            "bbox_y2": np.nan,
        }

    labeled, count = ndi.label(mask_b)
    component_sizes = np.bincount(labeled.ravel())
    if len(component_sizes) > 1:
        largest_area = int(component_sizes[1:].max())
    else:
        largest_area = 0
    ys, xs = np.where(mask_b)
    return {
        "component_count": int(count),
        "largest_component_area": largest_area,
        "largest_component_ratio": float(safe_div(largest_area, area, 0.0)),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "bbox_x1": int(xs.min()),
        "bbox_y1": int(ys.min()),
        "bbox_x2": int(xs.max() + 1),
        "bbox_y2": int(ys.max() + 1),
    }


def foreground_component_count(mask: np.ndarray) -> int:
    """Count connected components in any non-background prediction."""

    _, count = ndi.label(np.asarray(mask) > 0)
    return int(count)


def entropy_metrics(
    entropy_slice: Optional[np.ndarray],
    pred_mask: np.ndarray,
) -> Tuple[float, float]:
    """Compute entropy means for the whole slice and predicted foreground."""

    if entropy_slice is None:
        return np.nan, np.nan
    entropy = np.asarray(entropy_slice, dtype=np.float32)
    mean_all = float(np.nanmean(entropy))
    fg = pred_mask.astype(bool)
    mean_fg = float(np.nanmean(entropy[fg])) if np.any(fg) else np.nan
    return mean_all, mean_fg


def assign_failure_type(
    metrics: Dict[str, float],
    class_id: int,
    small_organ_ids: set[int],
    cardiac_class_ids: set[int],
    foreground_components: int,
    thresholds: FailureThresholds,
) -> str:
    """Assign the dominant failure type using transparent threshold rules."""

    gt_area = float(metrics["gt_area"])
    pred_area = float(metrics["pred_area"])
    dice = float(metrics["dice"])
    component_count = int(metrics["component_count"])
    largest_component_ratio = float(metrics["largest_component_ratio"])

    if gt_area == 0 and pred_area > thresholds.min_pred_area:
        return "absent_hallucination"
    if gt_area > thresholds.min_gt_area and (pred_area == 0 or dice < thresholds.low_dice):
        return "missed_organ"
    if gt_area > 0 and safe_div(pred_area, gt_area, math.inf) > thresholds.over_area_ratio:
        return "over_segmentation"
    if gt_area > 0 and safe_div(pred_area, gt_area, 0.0) < thresholds.under_area_ratio:
        return "under_segmentation"
    if (
        component_count > thresholds.max_components
        and largest_component_ratio < thresholds.min_largest_component_ratio
    ):
        return "fragmentation"
    if pred_area > thresholds.min_pred_area and class_id in cardiac_class_ids and (
        component_count > thresholds.topology_max_components
        or largest_component_ratio < thresholds.topology_min_largest_component_ratio
        or foreground_components > thresholds.foreground_max_components
    ):
        return "topology_suspicious"
    if class_id in small_organ_ids and dice < thresholds.small_organ_dice:
        return "small_organ_failure"
    return "normal"


def failure_priority_score(row: pd.Series) -> float:
    """Score a row so that low-Dice, high-error, large-area failures rank first."""

    gt_area = float(row.get("gt_area", 0.0))
    pred_area = float(row.get("pred_area", 0.0))
    fp = float(row.get("fp", 0.0))
    fn = float(row.get("fn", 0.0))
    dice = float(row.get("dice", 1.0))
    failure_weight = 1.0 if row.get("failure_type") != "normal" else 0.25
    return float((1.0 - dice) * math.log1p(max(gt_area, pred_area)) + failure_weight * math.log1p(fp + fn))


def low_dice_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Return rows considered low-Dice while excluding both-empty normal rows."""

    active = (df["gt_area"] > 0) | (df["pred_area"] > 0)
    return active & (df["dice"] < threshold)


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: Optional[int] = None) -> str:
    """Render a compact Markdown table."""

    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    view = df.loc[:, list(columns)].copy()
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if np.isfinite(value) else str(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ensure_parent(path: Path) -> None:
    """Create the parent directory for an output path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    """Write a JSON file with stable formatting."""

    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
