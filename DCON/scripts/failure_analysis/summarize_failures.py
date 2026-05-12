#!/usr/bin/env python3
"""Summarize DCON failure-analysis CSVs into a Markdown report."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from failure_common import (
    FAILURE_TYPES,
    ensure_parent,
    failure_priority_score,
    low_dice_mask,
    markdown_table,
    safe_div,
)


KEY_FAILURES = [
    "absent_hallucination",
    "fragmentation",
    "over_segmentation",
    "missed_organ",
]


RECOMMENDATION_LIBRARY: Dict[str, Dict[str, List[str]]] = {
    "absent_hallucination": {
        "gates": [
            "class-presence gate",
            "absent-class suppression",
            "source-trained presence prediction head",
        ],
        "training": ["presence-aware auxiliary head", "presence-aware loss"],
        "story": "Use source GT to teach slice-level organ presence, then block class-wise TTA updates and logits when the class is unlikely to exist.",
    },
    "fragmentation": {
        "gates": ["connected component reliability gate", "largest component filtering baseline"],
        "training": ["topology-aware regularization", "clDice / soft skeleton loss"],
        "story": "Make adaptation topology-aware instead of only entropy-aware, so fragmented predictions are treated as unreliable pseudo-labels.",
    },
    "topology_suspicious": {
        "gates": ["foreground component-count gate", "cardiac topology reliability gate"],
        "training": ["topology-aware training loss", "skeleton consistency / connectedness proxy"],
        "story": "For cardiac transfer, constrain LV/Myo/RV predictions to remain anatomically coherent under domain shift.",
    },
    "over_segmentation": {
        "gates": ["area-ratio reliability gate", "source-calibrated class area prior"],
        "training": ["area distribution regularization"],
        "story": "Calibrate target predictions against source-trained anatomical size priors without using target labels.",
    },
    "small_organ_failure": {
        "gates": ["small-organ reliability weighting"],
        "training": [
            "class-balanced Dice / focal loss",
            "prototype-based small-organ enhancement",
            "foreground contrastive loss",
        ],
        "story": "Allocate training and TTA reliability to unstable small structures instead of letting large organs dominate the update signal.",
    },
    "missed_organ": {
        "gates": ["foreground recall reliability gate", "prototype-guided recovery"],
        "training": ["foreground recall enhancement", "low-confidence foreground mining"],
        "story": "Recover missing target-domain foreground by mining low-confidence but anatomy-compatible foreground candidates.",
    },
    "under_segmentation": {
        "gates": ["foreground recall reliability gate"],
        "training": ["foreground recall enhancement", "prototype-guided recovery"],
        "story": "Prevent conservative adaptation from erasing valid target-domain foreground.",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Create failure_report.md from analyzer CSVs.")
    parser.add_argument("--csv_files", nargs="+", required=True, help="CSV files or glob patterns.")
    parser.add_argument("--out_md", default="outputs/failure_report.md", type=Path)
    parser.add_argument("--out_csv", default=None, type=Path, help="Optional ranked-row CSV output.")
    parser.add_argument("--low_dice_threshold", type=float, default=0.50)
    parser.add_argument("--fp_fn_dominance_ratio", type=float, default=1.25)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def expand_csv_files(patterns: Sequence[str]) -> List[Path]:
    """Expand shell or quoted glob patterns into existing CSV paths."""

    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No CSV files found from patterns: {patterns}")
    return existing


def read_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    """Read and concatenate analyzer CSVs."""

    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    required = {"dataset", "direction", "case_id", "class_name", "dice", "fp", "fn", "failure_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def active_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where GT or prediction contains the class."""

    return df[(df["gt_area"] > 0) | (df["pred_area"] > 0)].copy()


def direction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute direction-level mean Dice and error load."""

    active = active_df(df)
    grouped = active.groupby(["dataset", "direction"], dropna=False)
    out = grouped.agg(
        mean_dice=("dice", "mean"),
        rows=("dice", "size"),
        cases=("case_id", "nunique"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
    ).reset_index()
    out["mean_dice"] = out["mean_dice"].round(4)
    return out.sort_values(["dataset", "direction"])


def class_summary(df: pd.DataFrame, dominance_ratio: float) -> pd.DataFrame:
    """Compute class-level stability and FP/FN dominance."""

    active = active_df(df)
    out = active.groupby(["dataset", "direction", "class_id", "class_name"], dropna=False).agg(
        mean_dice=("dice", "mean"),
        std_dice=("dice", "std"),
        low_dice_rate=("dice", lambda x: float((x < 0.5).mean())),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
        rows=("dice", "size"),
    ).reset_index()
    out["fp_fn_ratio"] = out.apply(lambda r: safe_div(r["fp"], r["fn"], np.inf if r["fp"] > 0 else 0.0), axis=1)
    out["dominance"] = out.apply(lambda r: fp_fn_label(r["fp"], r["fn"], dominance_ratio), axis=1)
    for col in ("mean_dice", "std_dice", "low_dice_rate", "fp_fn_ratio"):
        out[col] = out[col].astype(float).round(4)
    return out.sort_values(["mean_dice", "low_dice_rate"], ascending=[True, False])


def fp_fn_label(fp: float, fn: float, dominance_ratio: float) -> str:
    """Classify whether FP or FN dominates."""

    if fp > fn * dominance_ratio:
        return "FP_dominant"
    if fn > fp * dominance_ratio:
        return "FN_dominant"
    return "balanced"


def failure_summary(df: pd.DataFrame, low_threshold: float) -> pd.DataFrame:
    """Rank failure types by low-Dice loss contribution."""

    active = active_df(df)
    low = active[low_dice_mask(active, low_threshold)].copy()
    total_active = max(len(active), 1)
    total_low = max(len(low), 1)
    low["dice_loss"] = 1.0 - low["dice"].clip(0.0, 1.0)
    active_loss = active.copy()
    active_loss["dice_loss"] = 1.0 - active_loss["dice"].clip(0.0, 1.0)

    rows = []
    for failure_type in FAILURE_TYPES:
        active_ft = active_loss[active_loss["failure_type"] == failure_type]
        low_ft = low[low["failure_type"] == failure_type]
        rows.append(
            {
                "failure_type": failure_type,
                "count": int(len(active_ft)),
                "count_pct": safe_div(len(active_ft), total_active),
                "low_dice_count": int(len(low_ft)),
                "low_dice_pct": safe_div(len(low_ft), total_low),
                "dice_loss_sum": float(active_ft["dice_loss"].sum()),
                "low_dice_loss_sum": float(low_ft["dice_loss"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    denom = max(float(out["low_dice_loss_sum"].sum()), 1e-12)
    out["low_dice_loss_share"] = out["low_dice_loss_sum"] / denom
    for col in ("count_pct", "low_dice_pct", "dice_loss_sum", "low_dice_loss_sum", "low_dice_loss_share"):
        out[col] = out[col].astype(float).round(4)
    return out.sort_values(["low_dice_loss_sum", "low_dice_count"], ascending=False)


def severe_rows(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Return top severe row-level failures."""

    active = active_df(df).copy()
    active["severity_score"] = active.apply(failure_priority_score, axis=1)
    columns = [
        "severity_score",
        "dataset",
        "direction",
        "case_id",
        "slice_id",
        "class_name",
        "failure_type",
        "dice",
        "fp",
        "fn",
        "pred_area",
        "gt_area",
        "component_count",
        "largest_component_ratio",
    ]
    out = active.sort_values("severity_score", ascending=False).head(top_k)
    out["severity_score"] = out["severity_score"].round(4)
    out["dice"] = out["dice"].round(4)
    out["largest_component_ratio"] = out["largest_component_ratio"].round(4)
    return out[columns]


def key_failure_breakdown(df: pd.DataFrame, low_threshold: float) -> pd.DataFrame:
    """Calculate key failure percentages among low-Dice rows."""

    active = active_df(df)
    low = active[low_dice_mask(active, low_threshold)]
    total = max(len(low), 1)
    return pd.DataFrame(
        [
            {
                "failure_type": failure_type,
                "low_dice_count": int((low["failure_type"] == failure_type).sum()),
                "low_dice_pct": round(safe_div((low["failure_type"] == failure_type).sum(), total), 4),
            }
            for failure_type in KEY_FAILURES
        ]
    ).sort_values("low_dice_pct", ascending=False)


def top_problems(failure_rank: pd.DataFrame) -> List[str]:
    """Select the top three actionable failure types."""

    candidates = failure_rank[
        (failure_rank["failure_type"] != "normal")
        & ((failure_rank["low_dice_count"] > 0) | (failure_rank["low_dice_loss_sum"] > 0))
    ]
    return candidates["failure_type"].head(3).tolist()


def global_fp_fn_diagnosis(class_rank: pd.DataFrame, dominance_ratio: float) -> str:
    """Return global FP/FN diagnosis text."""

    fp = float(class_rank["fp"].sum())
    fn = float(class_rank["fn"].sum())
    if fp > fn * dominance_ratio:
        return f"FP dominates globally (FP={int(fp)}, FN={int(fn)}). Prioritize conservative adaptation and unreliable pseudo-label suppression."
    if fn > fp * dominance_ratio:
        return f"FN dominates globally (FP={int(fp)}, FN={int(fn)}). Prioritize foreground recall enhancement and prototype-guided recovery."
    return f"FP and FN are balanced globally (FP={int(fp)}, FN={int(fn)}). Prioritize class-specific gates rather than one global bias."


def recommendation_text(problem: str) -> str:
    """Format gate/training recommendations for a failure type."""

    rec = RECOMMENDATION_LIBRARY.get(problem, {})
    gates = ", ".join(rec.get("gates", ["reliability-gated TTA loss"]))
    training = ", ".join(rec.get("training", ["anatomy reliability regularization"]))
    story = rec.get("story", "Use reliability to prevent unreliable target predictions from driving adaptation.")
    return f"**{problem}**: gates = {gates}; training = {training}. Narrative: {story}"


def overall_conclusion(
    failure_rank: pd.DataFrame,
    class_rank: pd.DataFrame,
    direction_rank: pd.DataFrame,
    dominance_ratio: float,
) -> str:
    """Build a direct conclusion paragraph."""

    top_failure = failure_rank[failure_rank["failure_type"] != "normal"].iloc[0]
    worst_class = class_rank.iloc[0]
    worst_direction = direction_rank.sort_values("mean_dice", ascending=True).iloc[0]
    fpfn = global_fp_fn_diagnosis(class_rank, dominance_ratio)
    module = module_for_failure(str(top_failure["failure_type"]))
    return (
        f"The largest measurable Dice drop source is **{top_failure['failure_type']}**, "
        f"which contributes {top_failure['low_dice_loss_share']:.1%} of low-Dice loss. "
        f"The weakest class is **{worst_class['class_name']}** in {worst_class['direction']} "
        f"(mean Dice {worst_class['mean_dice']:.4f}), and the weakest direction is "
        f"**{worst_direction['direction']}** (mean Dice {worst_direction['mean_dice']:.4f}). "
        f"{fpfn} If only one module is allowed, implement **{module}** first."
    )


def module_for_failure(failure_type: str) -> str:
    """Choose the single best module for a dominant failure mode."""

    if failure_type == "absent_hallucination":
        return "presence-aware auxiliary head with class-presence gate"
    if failure_type in {"fragmentation", "topology_suspicious"}:
        return "anatomy reliability regularization with component/topology-gated TTA"
    if failure_type == "over_segmentation":
        return "source-calibrated area prior with area-ratio reliability gate"
    if failure_type in {"missed_organ", "under_segmentation"}:
        return "prototype-guided foreground recovery with reliability-gated TTA"
    if failure_type == "small_organ_failure":
        return "small-organ reliability weighting with class-balanced source training"
    return "reliability-gated TTA loss"


def build_report(df: pd.DataFrame, args: argparse.Namespace) -> str:
    """Build the complete Markdown report."""

    dsum = direction_summary(df)
    csum = class_summary(df, args.fp_fn_dominance_ratio)
    fsum = failure_summary(df, args.low_dice_threshold)
    ksum = key_failure_breakdown(df, args.low_dice_threshold)
    top = severe_rows(df, args.top_k)
    problems = top_problems(fsum)
    priority = ["P0", "P1", "P2"]
    first_module = module_for_failure(problems[0]) if problems else "reliability-gated TTA loss"

    rec_lines = [recommendation_text(problem) for problem in problems]
    if not rec_lines:
        rec_lines = ["No dominant non-normal failure type found; inspect low-Dice rows manually."]

    p_lines = []
    for idx, problem in enumerate(problems[:3]):
        p_lines.append(f"- **{priority[idx]}**: solve **{problem}**. {recommendation_text(problem)}")

    key_pct = ", ".join(
        f"{row.failure_type}={row.low_dice_pct:.1%}" for row in ksum.itertuples(index=False)
    )
    decision = (
        f"Prioritize **testing-time gates first** for immediate risk control, then add the matching "
        f"training-stage constraint for the paper story. The one-module recommendation is "
        f"**{first_module}**. Its narrative: source-domain anatomy supervision learns reliability, "
        f"and TTA updates are applied only where class/region predictions are anatomically credible."
    )

    return "\n\n".join(
        [
            "# 1. Overall conclusion\n"
            + overall_conclusion(fsum, csum, dsum, args.fp_fn_dominance_ratio)
            + "\n\n"
            + decision,
            "# 2. Direction-level summary\n" + markdown_table(dsum, dsum.columns),
            "# 3. Class-level summary\n"
            + markdown_table(
                csum,
                [
                    "dataset",
                    "direction",
                    "class_name",
                    "mean_dice",
                    "std_dice",
                    "low_dice_rate",
                    "fp",
                    "fn",
                    "dominance",
                ],
                max_rows=50,
            ),
            "# 4. Failure-type ranking\n"
            + markdown_table(
                fsum,
                [
                    "failure_type",
                    "count",
                    "count_pct",
                    "low_dice_count",
                    "low_dice_pct",
                    "low_dice_loss_share",
                ],
            )
            + f"\n\nLow-Dice key failure composition: {key_pct}.",
            "# 5. FP vs FN diagnosis\n"
            + global_fp_fn_diagnosis(csum, args.fp_fn_dominance_ratio)
            + "\n\n"
            + markdown_table(
                csum.sort_values(["dominance", "mean_dice"]),
                ["dataset", "direction", "class_name", "fp", "fn", "fp_fn_ratio", "dominance"],
                max_rows=50,
            ),
            "# 6. What should be solved first\n" + "\n".join(p_lines),
            "# 7. Recommended gates\n"
            + "\n".join(f"- {line}" for line in rec_lines)
            + "\n- **Reliability-gated TTA loss**: use `loss = R_region * L_entropy + R_class * L_prototype`; do not entropy-minimize unreliable pixels/classes.",
            "# 8. Recommended training-stage constraints\n"
            + "Target GT is used only for this offline failure analysis. The deployable method should use source labels, source-calibrated statistics, target images, and model predictions only.\n"
            + "- **Presence-aware auxiliary head**: train slice-level class presence labels from source GT and use presence probability as class-wise reliability at test time.\n"
            + "- **Anatomy reliability regularization**: enforce weak/strong augmentation consistency on reliable source regions, then reuse multi-view consistency as a test-time reliability score.\n"
            + "- **Source-calibrated area prior**: estimate class area distributions on source predictions/labels and reject target predictions with abnormal area ratios without using target GT.\n"
            + "- **Topology-aware training loss**: add clDice, soft skeleton consistency, or connectedness proxy for cardiac classes; start with an interface so it can be ablated cleanly.\n"
            + "- **Reliability-gated TTA loss**: update with `R_region * L_entropy + R_class * L_prototype`; low-reliability regions should not drive adaptation.",
            "# Top 20 severe failure rows\n" + markdown_table(top, top.columns, max_rows=args.top_k),
        ]
    )


def main() -> None:
    """Entry point."""

    args = parse_args()
    df = read_csvs(expand_csv_files(args.csv_files))
    report = build_report(df, args)
    ensure_parent(args.out_md)
    args.out_md.write_text(report + "\n", encoding="utf-8")

    if args.out_csv:
        ensure_parent(args.out_csv)
        severe_rows(df, args.top_k).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
