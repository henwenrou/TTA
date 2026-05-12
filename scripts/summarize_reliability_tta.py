#!/usr/bin/env python3
"""Summarize reliability-gated multi-view TTA experiment outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

DEFAULT_METHODS = [
    "source_only",
    "entropy_tta",
    "mv_consistency_tta",
    "reliability_gated_mv_tta",
]

ABLATIONS = [
    "views=4 vs views=8",
    "steps=1 vs steps=3",
    "with / without R_class",
    "with / without R_region",
    "with / without logit_damping",
]


def parse_args() -> argparse.Namespace:
    """Parse summary arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="outputs/reliability_tta")
    parser.add_argument("--direction", default=None)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    return parser.parse_args()


def read_csv_dict(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into dictionaries; return an empty list if absent."""

    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    """Safely parse a CSV field as float."""

    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except Exception:
        return default


def rank_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute binary AUROC from scores without sklearn."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def method_summary(method_dir: Path) -> Dict[str, float]:
    """Load aggregate summary for one method."""

    rows = read_csv_dict(method_dir / "failure_summary.csv")
    return {} if not rows else {key: as_float(rows[0], key) for key in rows[0].keys()}


def method_metrics(method_dir: Path) -> List[Dict[str, str]]:
    """Load case-level metrics for one method."""

    return read_csv_dict(method_dir / "metrics.csv")


def fmt(value: float, digits: int = 4) -> str:
    """Format finite numbers for Markdown tables."""

    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def reliability_diagnosis(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    """Compute AUROC diagnostics for reliability scores."""

    absent_labels = [int(as_float(row, "absent_hallucination_count", 0.0) > 0) for row in rows]
    absent_scores = [1.0 - as_float(row, "min_R_class_fg") for row in rows]
    low_dice_labels = [int(as_float(row, "low_dice_case", 0.0) > 0) for row in rows]
    low_region_scores = [1.0 - as_float(row, "mean_R_region") for row in rows]
    return {
        "auroc_r_class_absent": rank_auc(absent_scores, absent_labels),
        "auroc_r_region_low_dice": rank_auc(low_region_scores, low_dice_labels),
    }


def conclusion(rows: Dict[str, Dict[str, float]]) -> str:
    """Generate a conservative overall conclusion from available summaries."""

    gated = rows.get("reliability_gated_mv_tta", {})
    entropy = rows.get("entropy_tta", {})
    mv = rows.get("mv_consistency_tta", {})
    if not gated:
        return "Reliability-gated multi-view TTA has not been run yet, so effectiveness cannot be judged."

    baselines = [row for row in (entropy, mv) if row]
    if not baselines:
        return "Only partial results are available; run entropy_tta and mv_consistency_tta before judging the method."

    best_baseline_dice = max(row.get("mean_dice", float("-inf")) for row in baselines)
    best_baseline_absent = min(row.get("absent_hallucination_count", float("inf")) for row in baselines)
    best_baseline_low = min(row.get("low_dice_rate", float("inf")) for row in baselines)
    improved = (
        gated.get("mean_dice", float("-inf")) >= best_baseline_dice
        and gated.get("absent_hallucination_count", float("inf")) <= best_baseline_absent
        and gated.get("low_dice_rate", float("inf")) <= best_baseline_low
    )
    if improved:
        return (
            "Reliability-gated multi-view TTA is effective in this run: it matches or improves Dice "
            "while reducing hallucination and low-Dice failure signals versus the TTA baselines."
        )
    return (
        "Reliability-gated multi-view TTA is not clearly effective in this run. "
        "If this holds after ablations, the next direction should be a source-trained evidence head "
        "or presence-aware training rather than relying on multi-view reliability alone."
    )


def available_direction_dirs(out_dir: Path, requested: str | None) -> List[Path]:
    """Resolve which direction folders should be summarized."""

    if requested:
        return [out_dir / requested]
    return [path for path in sorted(out_dir.iterdir()) if path.is_dir()] if out_dir.exists() else []


def build_summary_for_direction(direction_dir: Path, methods: Sequence[str]) -> str:
    """Build the Markdown summary for one direction."""

    summaries = {method: method_summary(direction_dir / method) for method in methods}
    metrics = {method: method_metrics(direction_dir / method) for method in methods}
    gated_diag = reliability_diagnosis(metrics.get("reliability_gated_mv_tta", []))

    lines = [
        "# Overall conclusion",
        conclusion(summaries),
        "",
        "# Method comparison",
        "| method | mean Dice | low dice rate | absent hallucinations | absent rate | missed organs | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = summaries.get(method, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt(row.get("mean_dice", float("nan"))),
                    fmt(row.get("low_dice_rate", float("nan"))),
                    fmt(row.get("absent_hallucination_count", float("nan")), 0),
                    fmt(row.get("absent_hallucination_rate", float("nan"))),
                    fmt(row.get("missed_organ_count", float("nan")), 0),
                    fmt(row.get("FP", float("nan")), 0),
                    fmt(row.get("FN", float("nan")), 0),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "# Failure reduction",
            "Primary check: absent_hallucination should drop without a compensating rise in missed organs or FP.",
        ]
    )
    source_absent = summaries.get("source_only", {}).get("absent_hallucination_count", float("nan"))
    gated_absent = summaries.get("reliability_gated_mv_tta", {}).get("absent_hallucination_count", float("nan"))
    lines.append(f"source_only absent_hallucination_count: {fmt(source_absent, 0)}")
    lines.append(f"reliability_gated_mv_tta absent_hallucination_count: {fmt(gated_absent, 0)}")

    lines.extend(
        [
            "",
            "# Reliability diagnosis",
            f"AUROC(R_class low -> absent_hallucination): {fmt(gated_diag['auroc_r_class_absent'])}",
            f"AUROC(R_region low -> low_dice): {fmt(gated_diag['auroc_r_region_low_dice'])}",
            "",
            "# Ablation",
        ]
    )
    for item in ABLATIONS:
        lines.append(f"- {item}: run corresponding settings and place outputs under this direction to compare.")

    lines.extend(
        [
            "",
            "# Final recommendation",
            (
                "Continue toward a formal method only if reliability_gated_mv_tta improves Dice, "
                "reduces absent hallucination and low-Dice cases, and does not increase missed organs. "
                "Otherwise prioritize a source-trained evidence head or presence-aware training."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    direction_dirs = available_direction_dirs(out_dir, args.direction)
    if not direction_dirs:
        raise FileNotFoundError(f"No reliability TTA outputs found under {out_dir}")

    combined = []
    for direction_dir in direction_dirs:
        text = build_summary_for_direction(direction_dir, methods)
        (direction_dir / "summary.md").write_text(text, encoding="utf-8")
        combined.append(f"## {direction_dir.name}\n\n{text}")

    (out_dir / "summary.md").write_text("\n\n".join(combined), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
