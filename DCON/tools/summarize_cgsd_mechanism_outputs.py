#!/usr/bin/env python3
"""Summarize CGSD mechanism metrics CSV and optional training logs."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


METRIC_KEYS = [
    "mean_dstab_core",
    "mean_dstab_boundary",
    "mean_dstab_background",
    "topk_core_ratio",
    "topk_boundary_ratio",
    "topk_background_ratio",
    "d_str",
    "d_sty",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize CGSD-SAAM mechanism outputs.")
    parser.add_argument("--metrics_csv", required=True, type=Path)
    parser.add_argument("--full_log", type=Path, default=None)
    parser.add_argument("--wo_log", type=Path, default=None)
    parser.add_argument("--out_dir", required=True, type=Path)
    return parser.parse_args()


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def nanmean(values):
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def summarize_metrics(metrics_csv):
    grouped = defaultdict(lambda: defaultdict(list))
    with metrics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = str(row.get("epoch", ""))
            method = row.get("method", "")
            for key in METRIC_KEYS:
                grouped[(epoch, method)][key].append(safe_float(row.get(key)))

    rows = []
    for (epoch, method), metrics in sorted(grouped.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        out = {"epoch": epoch, "method": method}
        for key in METRIC_KEYS:
            out[key] = nanmean(metrics[key])
        out["d_sty_minus_d_str"] = (
            out["d_sty"] - out["d_str"]
            if not math.isnan(out["d_sty"]) and not math.isnan(out["d_str"])
            else math.nan
        )
        rows.append(out)
    return rows


MECH_RE = re.compile(
    r"Tr-Epoch:(?P<epoch>\d+),Iter:(?P<iter>\d+).*?"
    r"mech_core:(?P<core>[-+0-9.eE]+).*?"
    r"mech_boundary:(?P<boundary>[-+0-9.eE]+).*?"
    r"mech_background:(?P<background>[-+0-9.eE]+).*?"
    r"mech_topk_core:(?P<topk_core>[-+0-9.eE]+).*?"
    r"mech_topk_boundary:(?P<topk_boundary>[-+0-9.eE]+)"
    r"(?:.*?mech_d_str:(?P<d_str>[-+0-9.eE]+).*?mech_d_sty:(?P<d_sty>[-+0-9.eE]+))?"
)


def parse_log(path, method):
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = MECH_RE.search(line)
        if not match:
            continue
        item = {
            "method": method,
            "epoch": int(match.group("epoch")),
            "iter": int(match.group("iter")),
            "mean_dstab_core": safe_float(match.group("core")),
            "mean_dstab_boundary": safe_float(match.group("boundary")),
            "mean_dstab_background": safe_float(match.group("background")),
            "topk_core_ratio": safe_float(match.group("topk_core")),
            "topk_boundary_ratio": safe_float(match.group("topk_boundary")),
            "d_str": safe_float(match.group("d_str")),
            "d_sty": safe_float(match.group("d_sty")),
        }
        item["d_sty_minus_d_str"] = (
            item["d_sty"] - item["d_str"]
            if not math.isnan(item["d_sty"]) and not math.isnan(item["d_str"])
            else math.nan
        )
        rows.append(item)
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "nan" if math.isnan(value) else f"{value:.6f}"


def build_report(metric_rows, full_log_rows, wo_log_rows):
    lines = []
    lines.append("CGSD-SAAM mechanism summary")
    lines.append("")
    lines.append("Checkpoint metrics by epoch:")
    lines.append("epoch | method | core | boundary | topk_core | topk_boundary | d_str | d_sty | d_sty-d_str")
    for row in metric_rows:
        lines.append(
            f"{row['epoch']} | {row['method']} | {fmt(row['mean_dstab_core'])} | "
            f"{fmt(row['mean_dstab_boundary'])} | {fmt(row['topk_core_ratio'])} | "
            f"{fmt(row['topk_boundary_ratio'])} | {fmt(row['d_str'])} | "
            f"{fmt(row['d_sty'])} | {fmt(row['d_sty_minus_d_str'])}"
        )

    full_metric = [r for r in metric_rows if r["method"] == "full_saa"]
    if full_metric:
        d_gap = [r["d_sty_minus_d_str"] for r in full_metric if not math.isnan(r["d_sty_minus_d_str"])]
        lines.append("")
        lines.append(
            "Checkpoint d_sty>d_str count: "
            f"{sum(1 for v in d_gap if v > 0)}/{len(d_gap)}"
        )

    lines.append("")
    lines.append(f"Full training log mechanism samples: {len(full_log_rows)}")
    lines.append(f"w/o CGSD training log mechanism samples: {len(wo_log_rows)}")
    if wo_log_rows and all(math.isnan(r["d_str"]) for r in wo_log_rows):
        lines.append("w/o CGSD log has no d_str/d_sty by design, because the style branch is disabled.")
    if not full_log_rows:
        lines.append("No full SAA training log was provided, so training-time style-branch collapse cannot be verified from log.txt.")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = summarize_metrics(args.metrics_csv)
    write_csv(
        args.out_dir / "checkpoint_epoch_summary.csv",
        metric_rows,
        ["epoch", "method", *METRIC_KEYS, "d_sty_minus_d_str"],
    )

    full_log_rows = parse_log(args.full_log, "full_saa")
    wo_log_rows = parse_log(args.wo_log, "wo_cgsd")
    write_csv(
        args.out_dir / "training_log_samples.csv",
        full_log_rows + wo_log_rows,
        [
            "method", "epoch", "iter", "mean_dstab_core", "mean_dstab_boundary",
            "mean_dstab_background", "topk_core_ratio", "topk_boundary_ratio",
            "d_str", "d_sty", "d_sty_minus_d_str",
        ],
    )

    report = build_report(metric_rows, full_log_rows, wo_log_rows)
    (args.out_dir / "mechanism_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
