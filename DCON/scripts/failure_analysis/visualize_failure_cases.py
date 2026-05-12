#!/usr/bin/env python3
"""Visualize top DCON segmentation failure cases as PNG panels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from failure_common import (
    cast_mask,
    ensure_parent,
    failure_priority_score,
    normalize_volume_shape,
    optional_case_path,
    read_array,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Render top failure rows into PNG panels.")
    parser.add_argument("--failure_csv", required=True, type=Path, help="CSV from failure_analyzer.py.")
    parser.add_argument("--out_dir", default="outputs/failure_cases", type=Path)
    parser.add_argument("--pred_dir", default=None, type=Path, help="Optional prediction directory override.")
    parser.add_argument("--gt_dir", default=None, type=Path, help="Optional GT directory override.")
    parser.add_argument("--image_dir", default=None, type=Path, help="Optional image directory.")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--only_failures", action="store_true", help="Exclude rows with failure_type=normal.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize one image slice to [0, 1] for display."""

    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def safe_name(value: str) -> str:
    """Create a filesystem-safe name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def top_failure_rows(df: pd.DataFrame, top_k: int, only_failures: bool) -> pd.DataFrame:
    """Rank rows by failure severity."""

    active = df[(df["gt_area"] > 0) | (df["pred_area"] > 0)].copy()
    if only_failures:
        active = active[active["failure_type"] != "normal"]
    active["severity_score"] = active.apply(failure_priority_score, axis=1)
    return active.sort_values("severity_score", ascending=False).head(top_k)


def load_case_volume(
    row: pd.Series,
    role: str,
    override_dir: Optional[Path],
    cache: Dict[Tuple[str, str], np.ndarray],
) -> np.ndarray:
    """Load pred or GT volume for a row, with caching."""

    case_id = str(row["case_id"])
    key = (role, case_id)
    if key in cache:
        return cache[key]

    if override_dir is not None:
        path = optional_case_path(override_dir, case_id)
        if path is None:
            raise FileNotFoundError(f"No {role} file found for case {case_id} in {override_dir}")
    else:
        column = f"{role}_path"
        if column not in row or pd.isna(row[column]):
            raise ValueError(f"CSV row lacks {column}; provide --{role}_dir.")
        path = Path(str(row[column]))
    volume = cast_mask(normalize_volume_shape(read_array(path)))
    cache[key] = volume
    return volume


def load_image_volume(
    case_id: str,
    image_dir: Optional[Path],
    image_path: Optional[str],
    fallback_shape: Tuple[int, int, int],
    cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Load optional image volume or return zeros if unavailable."""

    if case_id in cache:
        return cache[case_id]
    if image_dir is None and image_path:
        path = Path(image_path)
        volume = (
            normalize_volume_shape(read_array(path)).astype(np.float32)
            if path.exists()
            else np.zeros(fallback_shape, dtype=np.float32)
        )
    elif image_dir is None:
        volume = np.zeros(fallback_shape, dtype=np.float32)
    else:
        path = optional_case_path(image_dir, case_id)
        if path is None:
            volume = np.zeros(fallback_shape, dtype=np.float32)
        else:
            volume = normalize_volume_shape(read_array(path)).astype(np.float32)
            if volume.shape != fallback_shape:
                volume = np.zeros(fallback_shape, dtype=np.float32)
    if volume.shape != fallback_shape:
        volume = np.zeros(fallback_shape, dtype=np.float32)
    cache[case_id] = volume
    return volume


def error_rgb(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create an RGB TP/FP/FN error map."""

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    rgb = np.zeros((*pred.shape, 3), dtype=np.float32)
    rgb[np.logical_and(pred, gt)] = (0.20, 0.75, 0.25)
    rgb[np.logical_and(pred, ~gt)] = (0.90, 0.15, 0.15)
    rgb[np.logical_and(~pred, gt)] = (0.15, 0.35, 0.95)
    return rgb


def render_case(
    row: pd.Series,
    pred_volume: np.ndarray,
    gt_volume: np.ndarray,
    image_volume: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    """Render one failure row as a five-panel figure."""

    slice_id = int(row["slice_id"])
    class_id = int(row["class_id"])
    pred_mask = pred_volume[slice_id] == class_id
    gt_mask = gt_volume[slice_id] == class_id
    image = normalize_image(image_volume[slice_id])
    components, _ = ndi.label(pred_mask)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), constrained_layout=True)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("image")
    axes[1].imshow(gt_mask, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("gt")
    axes[2].imshow(pred_mask, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("pred")
    axes[3].imshow(error_rgb(pred_mask, gt_mask))
    axes[3].set_title("error: TP green, FP red, FN blue")
    axes[4].imshow(components, cmap="nipy_spectral")
    axes[4].set_title("component map")
    for ax in axes:
        ax.axis("off")

    title = (
        f"{row['case_id']} | z={slice_id} | {row['class_name']} | {row['failure_type']}\n"
        f"Dice={float(row['dice']):.3f}, FP={int(row['fp'])}, FN={int(row['fn'])}, "
        f"pred_area={int(row['pred_area'])}, gt_area={int(row['gt_area'])}"
    )
    fig.suptitle(title, fontsize=10)
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    """Entry point."""

    args = parse_args()
    df = pd.read_csv(args.failure_csv)
    rows = top_failure_rows(df, args.top_k, args.only_failures)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    volume_cache: Dict[Tuple[str, str], np.ndarray] = {}
    image_cache: Dict[str, np.ndarray] = {}
    manifest = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        pred_volume = load_case_volume(row, "pred", args.pred_dir, volume_cache)
        gt_volume = load_case_volume(row, "gt", args.gt_dir, volume_cache)
        if pred_volume.shape != gt_volume.shape:
            raise ValueError(f"Shape mismatch for {row['case_id']}: pred={pred_volume.shape}, gt={gt_volume.shape}")
        image_path = None
        if "image_path" in row and not pd.isna(row["image_path"]) and str(row["image_path"]):
            image_path = str(row["image_path"])
        image_volume = load_image_volume(str(row["case_id"]), args.image_dir, image_path, pred_volume.shape, image_cache)
        filename = (
            f"{rank:03d}_{safe_name(row['direction'])}_{safe_name(row['case_id'])}_"
            f"z{int(row['slice_id']):03d}_c{int(row['class_id'])}_{safe_name(row['failure_type'])}.png"
        )
        out_path = args.out_dir / filename
        render_case(row, pred_volume, gt_volume, image_volume, out_path, args.dpi)
        manifest.append({"rank": rank, "path": str(out_path), **row.to_dict()})

    pd.DataFrame(manifest).to_csv(args.out_dir / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} failure visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
