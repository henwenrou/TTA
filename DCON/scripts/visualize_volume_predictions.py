#!/usr/bin/env python3
"""Build slice-wise GT/pred overlay boards and metrics from NIfTI volumes.

Expected input layout:

    <volume_dir>/
      <scan_id>_image.nii.gz
      <scan_id>_gt.nii.gz
      <scan_id>_pred.nii.gz

The script writes:

    <output_dir>/
      case_metrics.csv
      slice_metrics.csv
      cases/<scan_id>/slices/*.png
      cases/<scan_id>/<scan_id>_overview.png
      global_worst_slices.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import SimpleITK as sitk


PALETTE = [
    (230, 57, 70),
    (29, 145, 192),
    (255, 183, 3),
    (61, 153, 112),
    (168, 85, 247),
    (244, 114, 182),
    (249, 115, 22),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize segmentation GT/pred volumes as slice-wise overlay boards."
    )
    parser.add_argument(
        "--volume-dir",
        required=True,
        help="Directory containing <scan_id>_image.nii.gz, _gt.nii.gz, and _pred.nii.gz.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <volume_dir>/visualization.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Resized tile size used inside overview boards.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for segmentation masks.",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=24,
        help="Number of worst slices to include in the global error board.",
    )
    parser.add_argument(
        "--global-sort",
        choices=("worst", "scan_asc", "scan_desc", "z_asc", "z_desc"),
        default="worst",
        help=(
            "Ordering of rows in global_worst_slices.png. "
            "'worst' keeps the previous behavior; other modes sort by scan/z order."
        ),
    )
    parser.add_argument(
        "--scan-id",
        nargs="+",
        default=None,
        help=(
            "Only visualize the specified scan ids, for example: "
            "--scan-id CHAOST2_6 CHAOST2_13"
        ),
    )
    parser.add_argument(
        "--skip-all-empty-in-global",
        action="store_true",
        help="Exclude slices with empty GT and empty prediction from the global worst-slice board.",
    )
    return parser.parse_args()


def load_nii(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def discover_scans(volume_dir: Path) -> Dict[str, Dict[str, Path]]:
    scans: Dict[str, Dict[str, Path]] = {}
    for image_path in sorted(volume_dir.glob("*_image.nii.gz")):
        scan_id = image_path.name[: -len("_image.nii.gz")]
        scans.setdefault(scan_id, {})["image"] = image_path
    for gt_path in sorted(volume_dir.glob("*_gt.nii.gz")):
        scan_id = gt_path.name[: -len("_gt.nii.gz")]
        scans.setdefault(scan_id, {})["gt"] = gt_path
    for pred_path in sorted(volume_dir.glob("*_pred.nii.gz")):
        scan_id = pred_path.name[: -len("_pred.nii.gz")]
        scans.setdefault(scan_id, {})["pred"] = pred_path

    missing = {
        scan_id: parts for scan_id, parts in scans.items()
        if set(parts.keys()) != {"image", "gt", "pred"}
    }
    if missing:
        details = ", ".join(
            f"{scan_id}:{sorted(parts.keys())}" for scan_id, parts in sorted(missing.items())
        )
        raise FileNotFoundError(f"Incomplete scan triplets in {volume_dir}: {details}")
    return scans


def parse_scan_key(scan_id: str) -> Tuple[str, int | str]:
    if "_" not in scan_id:
        return scan_id, scan_id
    prefix, suffix = scan_id.rsplit("_", 1)
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, suffix


def normalize_image_slice(image_slice: np.ndarray) -> np.ndarray:
    image_slice = image_slice.astype(np.float32)
    low = float(np.percentile(image_slice, 1))
    high = float(np.percentile(image_slice, 99))
    if high <= low:
        low = float(image_slice.min())
        high = float(image_slice.max())
    if high <= low:
        return np.zeros_like(image_slice, dtype=np.uint8)
    scaled = np.clip((image_slice - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def class_color(class_id: int) -> Tuple[int, int, int]:
    return PALETTE[(class_id - 1) % len(PALETTE)]


def extract_contour(mask: np.ndarray) -> np.ndarray:
    contour = np.zeros_like(mask, dtype=bool)
    contour[:-1, :] |= mask[:-1, :] != mask[1:, :]
    contour[1:, :] |= mask[1:, :] != mask[:-1, :]
    contour[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    contour[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return contour & mask


def overlay_mask_on_image(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    alpha: float,
) -> Image.Image:
    gray = normalize_image_slice(image_slice)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    out = rgb.copy()
    max_label = int(label_slice.max())

    for class_id in range(1, max_label + 1):
        class_mask = label_slice == class_id
        if not np.any(class_mask):
            continue
        color = np.asarray(class_color(class_id), dtype=np.float32)
        out[class_mask] = out[class_mask] * (1.0 - alpha) + color * alpha
        contour = extract_contour(class_mask)
        out[contour] = color

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def resize_for_board(image: Image.Image, tile_size: int) -> Image.Image:
    return image.resize((tile_size, tile_size), resample=Image.Resampling.BILINEAR)


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter) / denom


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return inter / union


def classwise_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
) -> Tuple[float, Dict[int, float]]:
    scores: Dict[int, float] = {}
    active_scores: List[float] = []
    for class_id in range(1, num_classes):
        pred_c = pred == class_id
        gt_c = gt == class_id
        denom = int(pred_c.sum() + gt_c.sum())
        if denom == 0:
            score = 1.0
        else:
            score = (2.0 * np.logical_and(pred_c, gt_c).sum()) / denom
            active_scores.append(float(score))
        scores[class_id] = float(score)
    if not active_scores:
        return 1.0, scores
    return float(np.mean(active_scores)), scores


def compute_slice_metrics(
    pred_slice: np.ndarray,
    gt_slice: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    pred_fg = pred_slice > 0
    gt_fg = gt_slice > 0
    fp = int(np.logical_and(pred_fg, ~gt_fg).sum())
    fn = int(np.logical_and(~pred_fg, gt_fg).sum())
    mean_class_dice, class_scores = classwise_dice(pred_slice, gt_slice, num_classes)

    metrics: Dict[str, float] = {
        "fg_dice": binary_dice(pred_fg, gt_fg),
        "fg_iou": binary_iou(pred_fg, gt_fg),
        "mean_class_dice": mean_class_dice,
        "gt_pixels": int(gt_fg.sum()),
        "pred_pixels": int(pred_fg.sum()),
        "fp_pixels": fp,
        "fn_pixels": fn,
        "all_empty": int(gt_fg.sum() == 0 and pred_fg.sum() == 0),
    }
    for class_id, score in class_scores.items():
        metrics[f"class_{class_id}_dice"] = score
    return metrics


def compute_case_metrics(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    case_scores = compute_slice_metrics(pred_vol, gt_vol, num_classes)
    case_scores["num_slices"] = int(gt_vol.shape[0])
    return case_scores


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(20, 20, 20))


def build_case_overview(
    scan_id: str,
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    tile_size: int,
) -> None:
    pad = 16
    header_h = 44
    row_title_h = 34
    width = pad * 3 + tile_size * 2
    height = pad + header_h + len(rows) * (row_title_h + tile_size + pad)
    board = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(board)

    draw_text(draw, (pad, pad), f"{scan_id} | slices={len(rows)} | left=GT overlay | right=Pred overlay")

    y = pad + header_h
    for row in rows:
        metrics = row["metrics"]
        draw_text(
            draw,
            (pad, y),
            (
                f"z={row['z_id']:03d} "
                f"meanDice={metrics['mean_class_dice']:.4f} "
                f"fgDice={metrics['fg_dice']:.4f} "
                f"IoU={metrics['fg_iou']:.4f} "
                f"FP={int(metrics['fp_pixels'])} "
                f"FN={int(metrics['fn_pixels'])}"
            ),
        )
        y += row_title_h
        board.paste(row["gt_tile"], (pad, y))
        board.paste(row["pred_tile"], (pad * 2 + tile_size, y))
        y += tile_size + pad

    board.save(output_path)


def build_global_overview(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    tile_size: int,
    title: str,
) -> None:
    if not rows:
        return

    pad = 16
    header_h = 44
    row_title_h = 34
    width = pad * 3 + tile_size * 2
    height = pad + header_h + len(rows) * (row_title_h + tile_size + pad)
    board = Image.new("RGB", (width, height), color=(252, 248, 246))
    draw = ImageDraw.Draw(board)
    draw_text(draw, (pad, pad), title)

    y = pad + header_h
    for row in rows:
        metrics = row["metrics"]
        draw_text(
            draw,
            (pad, y),
            (
                f"{row['scan_id']} z={row['z_id']:03d} "
                f"meanDice={metrics['mean_class_dice']:.4f} "
                f"fgDice={metrics['fg_dice']:.4f} "
                f"FP={int(metrics['fp_pixels'])} "
                f"FN={int(metrics['fn_pixels'])}"
            ),
        )
        y += row_title_h
        board.paste(row["gt_tile"], (pad, y))
        board.paste(row["pred_tile"], (pad * 2 + tile_size, y))
        y += tile_size + pad

    board.save(output_path)


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    volume_dir = Path(args.volume_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else volume_dir / "visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scans = discover_scans(volume_dir)
    if args.scan_id:
        selected = set(args.scan_id)
        scans = {scan_id: parts for scan_id, parts in scans.items() if scan_id in selected}
        missing = sorted(selected - set(scans.keys()))
        if missing:
            raise ValueError(f"Requested scan ids not found in {volume_dir}: {missing}")
        if not scans:
            raise ValueError("No scans left after applying --scan-id filter.")

    case_csv_rows: List[Dict[str, object]] = []
    slice_csv_rows: List[Dict[str, object]] = []
    global_rows: List[Dict[str, object]] = []

    for scan_id, parts in sorted(scans.items()):
        image_vol = load_nii(parts["image"])
        gt_vol = load_nii(parts["gt"]).astype(np.int32)
        pred_vol = load_nii(parts["pred"]).astype(np.int32)

        if image_vol.shape != gt_vol.shape or gt_vol.shape != pred_vol.shape:
            raise ValueError(
                f"Shape mismatch for {scan_id}: "
                f"image={image_vol.shape}, gt={gt_vol.shape}, pred={pred_vol.shape}"
            )

        num_classes = int(max(gt_vol.max(), pred_vol.max()) + 1)
        case_metrics = compute_case_metrics(pred_vol, gt_vol, num_classes)
        case_row = {"scan_id": scan_id, **case_metrics}
        case_csv_rows.append(case_row)

        case_dir = output_dir / "cases" / scan_id
        slice_dir = case_dir / "slices"
        slice_dir.mkdir(parents=True, exist_ok=True)

        overview_rows: List[Dict[str, object]] = []

        for z_id in range(image_vol.shape[0]):
            image_slice = image_vol[z_id]
            gt_slice = gt_vol[z_id]
            pred_slice = pred_vol[z_id]
            metrics = compute_slice_metrics(pred_slice, gt_slice, num_classes)

            gt_overlay = overlay_mask_on_image(image_slice, gt_slice, args.alpha)
            pred_overlay = overlay_mask_on_image(image_slice, pred_slice, args.alpha)
            gt_overlay.save(slice_dir / f"z{z_id:03d}_gt_overlay.png")
            pred_overlay.save(slice_dir / f"z{z_id:03d}_pred_overlay.png")

            slice_row = {"scan_id": scan_id, "z_id": z_id, **metrics}
            slice_csv_rows.append(slice_row)

            board_row = {
                "scan_id": scan_id,
                "z_id": z_id,
                "metrics": metrics,
                "gt_tile": resize_for_board(gt_overlay, args.tile_size),
                "pred_tile": resize_for_board(pred_overlay, args.tile_size),
            }
            overview_rows.append(board_row)
            if not (args.skip_all_empty_in_global and metrics["all_empty"]):
                global_rows.append(board_row)

        build_case_overview(
            scan_id=scan_id,
            rows=overview_rows,
            output_path=case_dir / f"{scan_id}_overview.png",
            tile_size=args.tile_size,
        )

    save_csv(output_dir / "case_metrics.csv", case_csv_rows)
    save_csv(output_dir / "slice_metrics.csv", slice_csv_rows)

    if args.global_sort == "worst":
        global_rows.sort(
            key=lambda row: (
                float(row["metrics"]["mean_class_dice"]),
                float(row["metrics"]["fg_dice"]),
                -float(row["metrics"]["fp_pixels"] + row["metrics"]["fn_pixels"]),
            )
        )
        global_title = "Worst slices by mean_class_dice | left=GT overlay | right=Pred overlay"
        global_rows_to_render = global_rows[: args.worst_k]
    elif args.global_sort == "scan_asc":
        global_rows.sort(key=lambda row: (parse_scan_key(str(row["scan_id"])), int(row["z_id"])))
        global_title = "Slices sorted by scan asc, z asc | left=GT overlay | right=Pred overlay"
        global_rows_to_render = global_rows
    elif args.global_sort == "scan_desc":
        global_rows.sort(key=lambda row: (parse_scan_key(str(row["scan_id"])), int(row["z_id"])), reverse=True)
        global_title = "Slices sorted by scan desc, z desc | left=GT overlay | right=Pred overlay"
        global_rows_to_render = global_rows
    elif args.global_sort == "z_asc":
        global_rows.sort(key=lambda row: (int(row["z_id"]), parse_scan_key(str(row["scan_id"]))))
        global_title = "Slices sorted by z asc, scan asc | left=GT overlay | right=Pred overlay"
        global_rows_to_render = global_rows
    else:
        global_rows.sort(key=lambda row: (int(row["z_id"]), parse_scan_key(str(row["scan_id"]))), reverse=True)
        global_title = "Slices sorted by z desc, scan desc | left=GT overlay | right=Pred overlay"
        global_rows_to_render = global_rows

    build_global_overview(
        rows=global_rows_to_render,
        output_path=output_dir / "global_worst_slices.png",
        tile_size=args.tile_size,
        title=global_title,
    )

    print(f"Visualization completed. Output written to: {output_dir}")


if __name__ == "__main__":
    main()
