#!/usr/bin/env python3
"""Redraw d_stab distribution figures from cached values without loading models."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Redraw d_stab figures from dstab_values.npz.")
    parser.add_argument("--result_dir", type=Path, required=True,
                        help="Directory containing dstab_values.npz and dstab_summary.csv.")
    parser.add_argument("--clip_percentile", type=float, default=95.0,
                        help="x-axis upper limit percentile over combined d_stab values.")
    parser.add_argument("--show_fliers", action="store_true",
                        help="Show outliers. Default hides them for a cleaner paper figure.")
    parser.add_argument("--flier_size", type=float, default=0.5)
    parser.add_argument("--flier_alpha", type=float, default=0.15)
    parser.add_argument("--unstable_ylim", type=float, default=0.4)
    parser.add_argument("--suffix", default="redraw",
                        help="Suffix for output filenames.")
    return parser.parse_args()


def read_summary(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ordered_values(npz, summary_rows):
    values = []
    labels = []
    for row in summary_rows:
        label = row["variant"]
        if label not in npz:
            raise KeyError(f"{label!r} not found in dstab_values.npz. Available: {list(npz.keys())}")
        labels.append(label)
        values.append(np.asarray(npz[label], dtype=np.float64))
    return labels, values


def plot_distribution(result_dir, labels, values, summary_rows, args):
    combined = np.concatenate(values)
    x_upper = float(np.percentile(combined, args.clip_percentile))
    tau = float(summary_rows[0]["tau"])

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    colors = ["#8da0cb", "#fc8d62"]
    box = ax.boxplot(
        values,
        vert=False,
        labels=labels,
        widths=0.62,
        showfliers=args.show_fliers,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"edgecolor": "#4d4d4d", "linewidth": 1.8},
        whiskerprops={"color": "#666666", "linewidth": 1.6},
        capprops={"color": "#666666", "linewidth": 1.6},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#4d4d4d",
            "markeredgecolor": "none",
            "markersize": args.flier_size,
            "alpha": args.flier_alpha,
        },
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    ax.axvline(tau, color="#cc0000", linestyle="--", linewidth=1.4, label=f"tau={tau:.4f}")
    ax.set_xlim(0.0, x_upper)
    ax.set_xlabel("d_stab")
    ax.set_ylabel("Variant")
    ax.set_title("SAAM d_stab Distribution at Final Checkpoint")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    ax.text(
        0.01,
        0.02,
        f"x-axis clipped at P{args.clip_percentile:g} for visualization",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    text_x = x_upper * 0.97
    for ypos, row in enumerate(summary_rows, start=1):
        ax.text(
            text_x,
            ypos + 0.31,
            f"mean={float(row['mean_d_stab']):.4f}\nunstable={float(row['unstable_ratio']):.3f}",
            ha="right",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )

    png = result_dir / f"dstab_distribution_{args.suffix}.png"
    pdf = result_dir / f"dstab_distribution_{args.suffix}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_unstable_bar(result_dir, summary_rows, args):
    labels = [row["variant"] for row in summary_rows]
    values = [float(row["unstable_ratio"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#8da0cb", "#fc8d62"], edgecolor="#222222", width=0.58)
    if args.unstable_ylim and args.unstable_ylim > 0:
        ax.set_ylim(0.0, args.unstable_ylim)
    else:
        ax.set_ylim(0.0, max(values) * 1.15)
    ax.set_ylabel("unstable ratio")
    ax.set_title("Unstable Ratio at Final Checkpoint")
    ax.grid(axis="y", alpha=0.25)
    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center")

    png = result_dir / f"unstable_ratio_bar_{args.suffix}.png"
    pdf = result_dir / f"unstable_ratio_bar_{args.suffix}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    args = parse_args()
    values_path = args.result_dir / "dstab_values.npz"
    summary_path = args.result_dir / "dstab_summary.csv"
    if not values_path.exists():
        raise FileNotFoundError(
            f"{values_path} not found. Older runs did not save raw d_stab values; "
            "rerun plot_final_dstab_distribution.py once to create this cache."
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found.")
    if not (0.0 < args.clip_percentile <= 100.0):
        raise ValueError("--clip_percentile must be in (0, 100].")

    summary_rows = read_summary(summary_path)
    npz = np.load(values_path)
    labels, values = ordered_values(npz, summary_rows)
    dist_png, dist_pdf = plot_distribution(args.result_dir, labels, values, summary_rows, args)
    bar_png, bar_pdf = plot_unstable_bar(args.result_dir, summary_rows, args)
    print(f"[OK] Saved distribution: {dist_png}")
    print(f"[OK] Saved distribution: {dist_pdf}")
    print(f"[OK] Saved unstable bar: {bar_png}")
    print(f"[OK] Saved unstable bar: {bar_pdf}")


if __name__ == "__main__":
    main()
