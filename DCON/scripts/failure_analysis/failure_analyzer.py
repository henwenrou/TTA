#!/usr/bin/env python3
"""Slice/case/class failure analyzer for DCON medical segmentation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from failure_common import (
    FailureThresholds,
    assign_failure_type,
    binary_metrics,
    cast_mask,
    component_metrics,
    ensure_parent,
    entropy_metrics,
    foreground_component_count,
    load_class_map,
    normalize_volume_shape,
    optional_case_path,
    paired_cases,
    parse_int_set,
    read_array,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute slice-wise, case-wise, and class-wise failure metrics."
    )
    parser.add_argument("--pred_dir", required=True, type=Path, help="Directory with prediction masks.")
    parser.add_argument("--gt_dir", required=True, type=Path, help="Directory with ground-truth masks.")
    parser.add_argument("--image_dir", default=None, type=Path, help="Optional directory with images.")
    parser.add_argument("--entropy_dir", default=None, type=Path, help="Optional directory with entropy maps.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. abdominal or cardiac.")
    parser.add_argument("--direction", required=True, help="Transfer direction name, e.g. bSSFP_to_LGE.")
    parser.add_argument("--class_map", required=True, type=Path, help="JSON class mapping.")
    parser.add_argument("--out_csv", required=True, type=Path, help="Output slice/class CSV path.")
    parser.add_argument("--out_json", default=None, type=Path, help="Optional output summary JSON path.")
    parser.add_argument(
        "--small_organ_ids",
        nargs="*",
        default=None,
        help="Optional ids treated as small organs, comma-separated or space-separated.",
    )
    parser.add_argument(
        "--cardiac_class_ids",
        nargs="*",
        default=None,
        help="Optional ids for topology checks, comma-separated or space-separated.",
    )
    parser.add_argument("--min_pred_area", type=float, default=10.0)
    parser.add_argument("--min_gt_area", type=float, default=10.0)
    parser.add_argument("--low_dice", type=float, default=0.30)
    parser.add_argument("--over_area_ratio", type=float, default=1.50)
    parser.add_argument("--under_area_ratio", type=float, default=0.50)
    parser.add_argument("--max_components", type=int, default=3)
    parser.add_argument("--min_largest_component_ratio", type=float, default=0.75)
    parser.add_argument("--small_organ_dice", type=float, default=0.60)
    parser.add_argument("--topology_max_components", type=int, default=3)
    parser.add_argument("--topology_min_largest_component_ratio", type=float, default=0.70)
    parser.add_argument("--foreground_max_components", type=int, default=10)
    parser.add_argument("--fp_fn_dominance_ratio", type=float, default=1.25)
    return parser.parse_args()


def thresholds_from_args(args: argparse.Namespace) -> FailureThresholds:
    """Build a threshold dataclass from parsed CLI args."""

    return FailureThresholds(
        min_pred_area=args.min_pred_area,
        min_gt_area=args.min_gt_area,
        low_dice=args.low_dice,
        over_area_ratio=args.over_area_ratio,
        under_area_ratio=args.under_area_ratio,
        max_components=args.max_components,
        min_largest_component_ratio=args.min_largest_component_ratio,
        small_organ_dice=args.small_organ_dice,
        topology_max_components=args.topology_max_components,
        topology_min_largest_component_ratio=args.topology_min_largest_component_ratio,
        foreground_max_components=args.foreground_max_components,
        fp_fn_dominance_ratio=args.fp_fn_dominance_ratio,
    )


def load_optional_volume(directory: Optional[Path], case_id: str) -> Optional[np.ndarray]:
    """Load an optional volume for a case when present."""

    path = optional_case_path(directory, case_id)
    if path is None:
        return None
    return normalize_volume_shape(read_array(path))


def validate_shapes(case_id: str, pred: np.ndarray, gt: np.ndarray, entropy: Optional[np.ndarray]) -> None:
    """Raise a clear error if required volumes do not align."""

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {case_id}: pred={pred.shape}, gt={gt.shape}")
    if entropy is not None and entropy.shape != pred.shape:
        raise ValueError(
            f"Entropy shape mismatch for {case_id}: entropy={entropy.shape}, pred={pred.shape}"
        )


def analyze_case(
    case_id: str,
    pred_path: Path,
    gt_path: Path,
    image_dir: Optional[Path],
    entropy_dir: Optional[Path],
    dataset: str,
    direction: str,
    class_names: Dict[int, str],
    small_organ_ids: set[int],
    cardiac_class_ids: set[int],
    thresholds: FailureThresholds,
) -> List[Dict[str, object]]:
    """Analyze one case and return per-slice/per-class rows."""

    pred = cast_mask(normalize_volume_shape(read_array(pred_path)))
    gt = cast_mask(normalize_volume_shape(read_array(gt_path)))
    image_path = optional_case_path(image_dir, case_id)
    entropy = load_optional_volume(entropy_dir, case_id)
    validate_shapes(case_id, pred, gt, entropy)

    rows: List[Dict[str, object]] = []
    for slice_id in range(pred.shape[0]):
        pred_slice = pred[slice_id]
        gt_slice = gt[slice_id]
        entropy_slice = None if entropy is None else entropy[slice_id]
        fg_components = foreground_component_count(pred_slice)

        for class_id, class_name in class_names.items():
            pred_mask = pred_slice == class_id
            gt_mask = gt_slice == class_id
            metrics = binary_metrics(pred_mask, gt_mask)
            metrics.update(component_metrics(pred_mask))
            ent_mean, ent_fg_mean = entropy_metrics(entropy_slice, pred_mask)
            metrics["entropy_mean"] = ent_mean
            metrics["entropy_foreground_mean"] = ent_fg_mean
            failure_type = assign_failure_type(
                metrics=metrics,
                class_id=class_id,
                small_organ_ids=small_organ_ids,
                cardiac_class_ids=cardiac_class_ids,
                foreground_components=fg_components,
                thresholds=thresholds,
            )

            rows.append(
                {
                    "dataset": dataset,
                    "direction": direction,
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "gt_present": bool(metrics["gt_area"] > 0),
                    "pred_present": bool(metrics["pred_area"] > 0),
                    **metrics,
                    "foreground_component_count": fg_components,
                    "failure_type": failure_type,
                    "pred_path": str(pred_path),
                    "gt_path": str(gt_path),
                    "image_path": "" if image_path is None else str(image_path),
                }
            )
    return rows


def build_summary(df: pd.DataFrame, thresholds: FailureThresholds) -> Dict[str, object]:
    """Build a compact machine-readable summary for quick inspection."""

    active = df[(df["gt_area"] > 0) | (df["pred_area"] > 0)]
    low = active[active["dice"] < thresholds.low_dice]
    return {
        "rows": int(len(df)),
        "active_rows": int(len(active)),
        "mean_dice_active": float(active["dice"].mean()) if len(active) else 1.0,
        "low_dice_rows": int(len(low)),
        "failure_type_counts": {
            str(k): int(v) for k, v in df["failure_type"].value_counts().to_dict().items()
        },
        "class_mean_dice": {
            str(k): float(v) for k, v in active.groupby("class_name")["dice"].mean().to_dict().items()
        },
    }


def main() -> None:
    """Run failure analysis and write CSV/JSON outputs."""

    args = parse_args()
    thresholds = thresholds_from_args(args)
    class_names, small_ids, cardiac_ids = load_class_map(
        args.class_map,
        small_organ_ids=parse_int_set(args.small_organ_ids),
        cardiac_class_ids=parse_int_set(args.cardiac_class_ids),
    )

    rows: List[Dict[str, object]] = []
    for case_id, pred_path, gt_path in paired_cases(args.pred_dir, args.gt_dir):
        rows.extend(
            analyze_case(
                case_id=case_id,
                pred_path=pred_path,
                gt_path=gt_path,
                image_dir=args.image_dir,
                entropy_dir=args.entropy_dir,
                dataset=args.dataset,
                direction=args.direction,
                class_names=class_names,
                small_organ_ids=small_ids,
                cardiac_class_ids=cardiac_ids,
                thresholds=thresholds,
            )
        )

    df = pd.DataFrame(rows)
    ensure_parent(args.out_csv)
    df.to_csv(args.out_csv, index=False)

    out_json = args.out_json or args.out_csv.with_suffix(".json")
    write_json(
        out_json,
        {
            "dataset": args.dataset,
            "direction": args.direction,
            "class_map": {str(k): v for k, v in class_names.items()},
            "small_organ_ids": sorted(small_ids),
            "cardiac_class_ids": sorted(cardiac_ids),
            "thresholds": thresholds.__dict__,
            "summary": build_summary(df, thresholds),
        },
    )
    print(json.dumps({"out_csv": str(args.out_csv), "out_json": str(out_json), "rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
