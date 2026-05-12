#!/usr/bin/env python3
"""Generate gate and training-constraint recommendations from failure statistics."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from failure_common import ensure_parent, low_dice_mask, safe_div, write_json


RECOMMENDATIONS: Dict[str, Dict[str, List[str] | str]] = {
    "absent_hallucination": {
        "test_time_gates": [
            "class-presence gate",
            "absent-class suppression",
            "source-trained presence prediction head",
        ],
        "training_constraints": ["presence-aware auxiliary head", "presence-aware loss"],
        "paper_story": "Source GT provides slice-level class-presence supervision; target TTA then suppresses classes whose presence probability is low.",
    },
    "fragmentation": {
        "test_time_gates": [
            "connected component reliability gate",
            "largest component filtering baseline",
        ],
        "training_constraints": [
            "topology-aware regularization",
            "clDice / soft skeleton loss",
        ],
        "paper_story": "Reliability should include anatomical connectedness, not only pixel confidence.",
    },
    "topology_suspicious": {
        "test_time_gates": [
            "cardiac topology reliability gate",
            "foreground component-count gate",
        ],
        "training_constraints": [
            "topology-aware training loss",
            "skeleton consistency",
            "connectedness proxy",
        ],
        "paper_story": "Cardiac classes need topology-preserving adaptation to avoid anatomically invalid LV/Myo/RV predictions.",
    },
    "over_segmentation": {
        "test_time_gates": [
            "area-ratio reliability gate",
            "source-calibrated class area prior",
        ],
        "training_constraints": ["area distribution regularization"],
        "paper_story": "A source-calibrated anatomy prior detects abnormal target area expansion without target labels.",
    },
    "small_organ_failure": {
        "test_time_gates": ["small-organ reliability weighting"],
        "training_constraints": [
            "class-balanced Dice / focal loss",
            "prototype-based small-organ enhancement",
            "foreground contrastive loss",
        ],
        "paper_story": "Small structures need explicit reliability and contrastive/prototype support because global TTA losses are dominated by large organs.",
    },
    "fp_dominant": {
        "test_time_gates": [
            "conservative adaptation",
            "unreliable pseudo-label suppression",
            "entropy loss only on reliable foreground/background",
        ],
        "training_constraints": ["presence-aware loss", "source-calibrated area prior"],
        "paper_story": "When FP dominates, adaptation should be conservative and avoid reinforcing hallucinated foreground.",
    },
    "fn_dominant": {
        "test_time_gates": [
            "foreground recall enhancement",
            "low-confidence foreground mining",
            "prototype-guided recovery",
        ],
        "training_constraints": ["foreground contrastive loss", "prototype-based recovery"],
        "paper_story": "When FN dominates, reliability should recover plausible foreground instead of simply suppressing uncertain predictions.",
    },
}


TRAINING_MODULES = [
    {
        "name": "Presence-aware auxiliary head",
        "where": "source training",
        "implementation": "Generate per-slice class-presence labels from GT masks and train a multi-label presence head with BCE.",
        "test_usage": "Use class presence probability as `R_class` to gate logits, pseudo-labels, entropy loss, and prototype loss.",
        "failure_modes": ["absent_hallucination", "FP_dominant"],
    },
    {
        "name": "Anatomy reliability regularization",
        "where": "source training",
        "implementation": "Apply weak/strong augmentation to the same image and enforce prediction consistency only on reliable regions.",
        "test_usage": "Use multi-view consistency as `R_region` during TTA.",
        "failure_modes": ["fragmentation", "topology_suspicious", "small_organ_failure"],
    },
    {
        "name": "Source-calibrated area prior",
        "where": "post-source-training calibration",
        "implementation": "Estimate per-class area distributions from source labels or source predictions and save robust percentiles.",
        "test_usage": "Downweight classes/slices whose predicted area falls outside calibrated source ranges.",
        "failure_modes": ["over_segmentation", "under_segmentation", "FP_dominant"],
    },
    {
        "name": "Topology-aware training loss",
        "where": "source training for cardiac classes",
        "implementation": "Add clDice, soft-skeleton consistency, or a connectedness proxy behind a loss interface for clean ablation.",
        "test_usage": "Combine with component-count and largest-component reliability checks.",
        "failure_modes": ["fragmentation", "topology_suspicious"],
    },
    {
        "name": "Reliability-gated TTA loss",
        "where": "test-time adaptation",
        "implementation": "`loss = R_region * L_entropy + R_class * L_prototype`; detach reliability scores unless explicitly optimizing them.",
        "test_usage": "Low-reliability pixels/classes do not update the model.",
        "failure_modes": ["absent_hallucination", "fragmentation", "over_segmentation", "small_organ_failure"],
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Recommend gates and training constraints.")
    parser.add_argument("--csv_files", nargs="+", required=True, help="Failure CSVs or glob patterns.")
    parser.add_argument("--out_md", default="outputs/failure_recommendations.md", type=Path)
    parser.add_argument("--out_json", default="outputs/failure_recommendations.json", type=Path)
    parser.add_argument("--low_dice_threshold", type=float, default=0.50)
    parser.add_argument("--high_failure_share", type=float, default=0.15)
    parser.add_argument("--fp_fn_dominance_ratio", type=float, default=1.25)
    return parser.parse_args()


def expand_csv_files(patterns: Sequence[str]) -> List[Path]:
    """Expand glob patterns into CSV paths."""

    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
        if not matches and Path(pattern).exists():
            paths.append(Path(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files found for {patterns}")
    return paths


def read_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    """Read analyzer CSV files."""

    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def failure_shares(df: pd.DataFrame, low_dice_threshold: float) -> pd.DataFrame:
    """Compute low-Dice shares by failure type."""

    active = df[(df["gt_area"] > 0) | (df["pred_area"] > 0)]
    low = active[low_dice_mask(active, low_dice_threshold)]
    total = max(len(low), 1)
    rows = []
    for failure_type, part in low.groupby("failure_type"):
        rows.append(
            {
                "failure_type": failure_type,
                "low_dice_count": int(len(part)),
                "low_dice_share": float(safe_div(len(part), total)),
                "mean_dice": float(part["dice"].mean()) if len(part) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["failure_type", "low_dice_count", "low_dice_share", "mean_dice"])
    return pd.DataFrame(rows).sort_values("low_dice_share", ascending=False)


def fp_fn_mode(df: pd.DataFrame, dominance_ratio: float) -> str:
    """Determine whether FP or FN dominates globally."""

    fp = float(df["fp"].sum())
    fn = float(df["fn"].sum())
    if fp > fn * dominance_ratio:
        return "fp_dominant"
    if fn > fp * dominance_ratio:
        return "fn_dominant"
    return "balanced"


def selected_recommendations(
    shares: pd.DataFrame,
    mode: str,
    high_failure_share: float,
) -> List[Dict[str, object]]:
    """Select recommendations triggered by observed failure shares."""

    selected: List[Dict[str, object]] = []
    for row in shares.itertuples(index=False):
        if row.failure_type == "normal" or row.low_dice_share < high_failure_share:
            continue
        rec = RECOMMENDATIONS.get(row.failure_type)
        if rec:
            selected.append({"trigger": row.failure_type, "share": row.low_dice_share, **rec})
    if mode in {"fp_dominant", "fn_dominant"}:
        selected.append({"trigger": mode, "share": None, **RECOMMENDATIONS[mode]})
    if not selected and len(shares):
        top = shares.iloc[0]["failure_type"]
        rec = RECOMMENDATIONS.get(str(top))
        if rec:
            selected.append({"trigger": str(top), "share": float(shares.iloc[0]["low_dice_share"]), **rec})
    return selected


def choose_single_module(selected: List[Dict[str, object]]) -> str:
    """Choose one module if implementation bandwidth is limited."""

    triggers = [str(item["trigger"]) for item in selected]
    if "absent_hallucination" in triggers or "fp_dominant" in triggers:
        return "Presence-aware auxiliary head + class-presence gate"
    if "fragmentation" in triggers or "topology_suspicious" in triggers:
        return "Anatomy reliability regularization + topology/component reliability gate"
    if "over_segmentation" in triggers:
        return "Source-calibrated area prior + area-ratio reliability gate"
    if "small_organ_failure" in triggers:
        return "Small-organ reliability weighting + prototype-based enhancement"
    if "fn_dominant" in triggers:
        return "Prototype-guided foreground recovery"
    return "Reliability-gated TTA loss"


def render_markdown(
    shares: pd.DataFrame,
    selected: List[Dict[str, object]],
    mode: str,
    single_module: str,
) -> str:
    """Render recommendations as Markdown."""

    lines = [
        "# Failure-Driven Recommendations",
        "",
        f"Global FP/FN mode: **{mode}**.",
        f"If only one module can be implemented, choose **{single_module}**.",
        "",
        "## Triggered Recommendations",
    ]
    if not selected:
        lines.append("_No strong trigger found. Use reliability-gated TTA as the default safe module._")
    for item in selected:
        share = item.get("share")
        share_text = "global error balance" if share is None else f"{float(share):.1%} of low-Dice rows"
        lines.extend(
            [
                f"### {item['trigger']} ({share_text})",
                "- Test-time gates: " + ", ".join(item["test_time_gates"]),
                "- Training-stage constraints: " + ", ".join(item["training_constraints"]),
                "- Paper story: " + str(item["paper_story"]),
                "",
            ]
        )

    lines.extend(["## Training Modules To Write Into The Method"])
    for module in TRAINING_MODULES:
        lines.extend(
            [
                f"### {module['name']}",
                f"- Stage: {module['where']}",
                f"- Implementation: {module['implementation']}",
                f"- Test-time use: {module['test_usage']}",
                f"- Targets: {', '.join(module['failure_modes'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Method Pseudocode Interfaces",
            "",
            "Target GT is used only by `failure_analyzer.py` for offline diagnosis. The following modules use source labels, source-calibrated statistics, target images, and model predictions only.",
            "",
            "```python",
            "presence_label[c] = 1 if (source_gt == c).sum() > min_gt_area else 0",
            "logits, presence_prob = model(source_image)",
            "loss = seg_loss(logits, source_gt) + lambda_presence * BCE(presence_prob, presence_label)",
            "# test time",
            "R_class[c] = stop_grad(presence_prob[c])",
            "```",
            "",
            "```python",
            "pred_weak = model(weak_aug(source_image))",
            "pred_strong = invert_aug(model(strong_aug(source_image)))",
            "R_region = confidence(pred_weak).detach() > tau",
            "loss_cons = mean(R_region * KL(pred_weak.detach(), pred_strong))",
            "```",
            "",
            "```python",
            "area_prior[c] = robust_percentiles(area(source_gt == c), q=(5, 95))",
            "# test time, no target GT",
            "area_ratio_ok = p05[c] <= area(pred == c) <= p95[c]",
            "R_class[c] *= float(area_ratio_ok)",
            "```",
            "",
            "```python",
            "topology_loss = cldice_loss(prob[:, cardiac_ids], one_hot_gt[:, cardiac_ids])",
            "# fallback proxy if clDice is unavailable: penalize fragmented soft masks",
            "loss = seg_loss + lambda_topology * topology_loss",
            "```",
            "",
            "```python",
            "loss_tta = mean(R_region * entropy_loss(probs)) + lambda_proto * mean(R_class * prototype_loss(features, probs))",
            "loss_tta.backward()",
            "```",
            "",
        ]
    )

    if not shares.empty:
        lines.extend(["## Low-Dice Failure Shares", ""])
        lines.append("| failure_type | low_dice_count | low_dice_share | mean_dice |")
        lines.append("| --- | --- | --- | --- |")
        for row in shares.itertuples(index=False):
            lines.append(
                f"| {row.failure_type} | {row.low_dice_count} | {row.low_dice_share:.4f} | {row.mean_dice:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Entry point."""

    args = parse_args()
    df = read_csvs(expand_csv_files(args.csv_files))
    shares = failure_shares(df, args.low_dice_threshold)
    mode = fp_fn_mode(df, args.fp_fn_dominance_ratio)
    selected = selected_recommendations(shares, mode, args.high_failure_share)
    single_module = choose_single_module(selected)

    ensure_parent(args.out_md)
    args.out_md.write_text(render_markdown(shares, selected, mode, single_module), encoding="utf-8")
    write_json(
        args.out_json,
        {
            "fp_fn_mode": mode,
            "single_module": single_module,
            "low_dice_failure_shares": shares.to_dict(orient="records"),
            "recommendations": selected,
            "training_modules": TRAINING_MODULES,
        },
    )
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
