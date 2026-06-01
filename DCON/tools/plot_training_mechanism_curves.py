#!/usr/bin/env python3
"""Plot CGSD mechanism curves recorded during training."""

import argparse
import ast
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training-time d_str/d_sty and SAAM mechanism scalar curves."
    )
    parser.add_argument("--exp_dir", required=True, type=Path,
                        help="Experiment directory, e.g. ckpts/bSSFP/exp_name")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--smooth", type=int, default=5,
                        help="Moving-average window over logged sample points.")
    parser.add_argument("--collapse_threshold", type=float, default=0.02,
                        help="Style branch is flagged as collapsed if late d_sty is below this.")
    return parser.parse_args()


def read_metric_csv(path):
    if not path.exists():
        return np.array([]), np.array([])

    steps = []
    values = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed = None
        try:
            parsed = ast.literal_eval(line)
        except Exception:
            parsed = None

        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            step, value = parsed[0], parsed[1]
        else:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            step, value = parts[0], parts[1]
        try:
            steps.append(float(step))
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    return np.asarray(steps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def smooth_values(values, window):
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def plot_two_curves(x, y1, y2, label1, label2, title, ylabel, out_path, smooth=1):
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    if y1.size:
        ax.plot(x[: y1.size], y1, alpha=0.35, linewidth=1.0, color="#1f77b4")
        ax.plot(x[: y1.size], smooth_values(y1, smooth), label=label1, linewidth=1.8, color="#1f77b4")
    if y2.size:
        ax.plot(x[: y2.size], y2, alpha=0.35, linewidth=1.0, color="#d62728")
        ax.plot(x[: y2.size], smooth_values(y2, smooth), label=label2, linewidth=1.8, color="#d62728")
    ax.set_title(title)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_region_curves(metric_dir, out_dir, smooth):
    curves = {}
    for name in (
        "mechanism_mean_dstab_core",
        "mechanism_mean_dstab_boundary",
        "mechanism_mean_dstab_background",
    ):
        steps, values = read_metric_csv(metric_dir / f"{name}.csv")
        if values.size:
            curves[name] = (steps, values)
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    for name, (steps, values) in curves.items():
        label = name.replace("mechanism_mean_dstab_", "")
        ax.plot(steps, smooth_values(values, smooth), label=label, linewidth=1.8)
    ax.set_title("Training-time d_stab by Region")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("d_stab")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "training_dstab_region_curves.png", dpi=180)
    plt.close(fig)


def summarize(d_str, d_sty, collapse_threshold):
    tail = max(1, int(math.ceil(0.2 * max(d_sty.size, 1))))
    late_d_str = float(np.nanmean(d_str[-tail:])) if d_str.size else math.nan
    late_d_sty = float(np.nanmean(d_sty[-tail:])) if d_sty.size else math.nan
    ratio = late_d_sty / max(late_d_str, 1e-8) if not math.isnan(late_d_str) and not math.isnan(late_d_sty) else math.nan
    return {
        "late_d_str": late_d_str,
        "late_d_sty": late_d_sty,
        "late_d_sty_over_d_str": ratio,
        "style_collapsed": bool(not math.isnan(late_d_sty) and late_d_sty < collapse_threshold),
        "d_sty_greater_than_d_str": bool(not math.isnan(ratio) and ratio > 1.0),
    }


def main():
    args = parse_args()
    metric_dir = args.exp_dir / "log" / "train"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    steps_str, d_str = read_metric_csv(metric_dir / "mechanism_d_str.csv")
    steps_sty, d_sty = read_metric_csv(metric_dir / "mechanism_d_sty.csv")
    if d_str.size == 0 and d_sty.size == 0:
        raise FileNotFoundError(
            f"No mechanism_d_str.csv or mechanism_d_sty.csv found in {metric_dir}. "
            "Train with --mechanism_log_interval > 0 and --use_cgsd 1."
        )

    x = steps_sty if steps_sty.size else steps_str
    plot_two_curves(
        x=x,
        y1=d_str,
        y2=d_sty,
        label1="d_str",
        label2="d_sty",
        title="Style Branch Distance During Training",
        ylabel="Cosine distance",
        out_path=args.out_dir / "training_d_str_d_sty_curve.png",
        smooth=args.smooth,
    )
    plot_region_curves(metric_dir, args.out_dir, args.smooth)

    summary = summarize(d_str, d_sty, args.collapse_threshold)
    with (args.out_dir / "training_mechanism_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    text = [
        "Training-time CGSD mechanism summary",
        f"late d_str: {summary['late_d_str']:.6f}",
        f"late d_sty: {summary['late_d_sty']:.6f}",
        f"late d_sty / d_str: {summary['late_d_sty_over_d_str']:.6f}",
        f"style_collapsed: {summary['style_collapsed']}",
        f"d_sty_greater_than_d_str: {summary['d_sty_greater_than_d_str']}",
    ]
    (args.out_dir / "training_mechanism_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))
    print(f"Figures: {args.out_dir}")


if __name__ == "__main__":
    main()
