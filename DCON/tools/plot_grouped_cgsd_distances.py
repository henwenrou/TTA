#!/usr/bin/env python3
"""Plot grouped running means for CGSD D_struct and D_style logs."""

import argparse
import ast
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group mechanism_d_str/mechanism_d_sty values and plot two curves."
    )
    parser.add_argument("--d_str_csv", type=Path, default=Path("mechanism_d_str.csv"))
    parser.add_argument("--d_sty_csv", type=Path, default=Path("mechanism_d_sty_w.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results_cgsd_grouped_curves"))
    parser.add_argument("--group_size", type=int, default=100,
                        help="Number of logged samples per averaged point.")
    parser.add_argument("--title", default="Training CGSD Distance Running Mean")
    parser.add_argument("--plot_mode", default="dual_axis", choices=["dual_axis", "same_axis"],
                        help="dual_axis makes small D_struct changes easier to see.")
    parser.add_argument("--struct_axis_headroom", type=float, default=3.0,
                        help="Right-axis max is max(D_struct) * this value in dual-axis mode.")
    parser.add_argument("--align", default="iteration", choices=["iteration", "position"],
                        help="Align D_struct/D_style by logged iteration or by row position.")
    return parser.parse_args()


def read_metric_csv(path):
    steps = []
    values = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
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

    if not values:
        raise ValueError(f"No numeric values parsed from {path}")
    return np.asarray(steps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def group_mean(steps, values, group_size):
    if group_size <= 0:
        raise ValueError("--group_size must be > 0")
    grouped_steps = []
    grouped_values = []
    for start in range(0, values.size, group_size):
        end = min(start + group_size, values.size)
        grouped_steps.append(float(np.mean(steps[start:end])))
        grouped_values.append(float(np.mean(values[start:end])))
    return np.asarray(grouped_steps), np.asarray(grouped_values)


def align_series(str_steps, d_str, sty_steps, d_sty, mode):
    if mode == "position":
        n = min(d_str.size, d_sty.size)
        if d_str.size != d_sty.size:
            print(f"[WARN] Length mismatch: d_str={d_str.size}, d_sty={d_sty.size}; using first {n} rows.")
        return str_steps[:n], d_str[:n], d_sty[:n]

    str_map = {int(step): float(value) for step, value in zip(str_steps, d_str)}
    sty_map = {int(step): float(value) for step, value in zip(sty_steps, d_sty)}
    common_steps = sorted(set(str_map) & set(sty_map))
    if not common_steps:
        raise ValueError("No common logged iterations found between D_struct and D_style CSVs.")
    dropped_str = len(str_map) - len(common_steps)
    dropped_sty = len(sty_map) - len(common_steps)
    if dropped_str or dropped_sty:
        print(
            f"[INFO] Iteration alignment: common={len(common_steps)} "
            f"dropped_d_str={dropped_str} dropped_d_sty={dropped_sty}"
        )
    steps = np.asarray(common_steps, dtype=np.float64)
    return (
        steps,
        np.asarray([str_map[int(step)] for step in common_steps], dtype=np.float64),
        np.asarray([sty_map[int(step)] for step in common_steps], dtype=np.float64),
    )


def write_grouped_csv(path, steps, d_str, d_sty):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group_index", "mean_iter", "D_struct", "D_style", "Separation_Ratio"],
        )
        writer.writeheader()
        for idx, (step, str_val, sty_val) in enumerate(zip(steps, d_str, d_sty), start=1):
            writer.writerow({
                "group_index": idx,
                "mean_iter": step,
                "D_struct": str_val,
                "D_style": sty_val,
                "Separation_Ratio": sty_val / (str_val + 1e-8),
            })


def plot_curves_matplotlib(path_png, path_pdf, steps, d_str, d_sty, title, plot_mode, struct_axis_headroom):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel("Training iteration")
    ax.grid(alpha=0.25)

    if plot_mode == "dual_axis":
        ax.plot(steps, d_sty, label="D_style", color="#d62728", linewidth=2.0)
        ax.set_ylabel("D_style (projector cosine distance)", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#d62728")
        ax.set_ylim(0.0, max(float(np.max(d_sty)) * 1.08, 1e-6))

        ax2 = ax.twinx()
        ax2.plot(steps, d_str, label="D_struct", color="#1f77b4", linewidth=2.0)
        ax2.set_ylabel("D_struct (projector cosine distance)", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.set_ylim(0.0, max(float(np.max(d_str)) * struct_axis_headroom, 1e-6))

        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    else:
        ax.plot(steps, d_str, label="D_struct", color="#1f77b4", linewidth=1.9)
        ax.plot(steps, d_sty, label="D_style", color="#d62728", linewidth=1.9)
        ax.set_ylabel("Projector cosine distance")
        ax.legend()

    fig.savefig(path_png, dpi=200)
    fig.savefig(path_pdf)
    plt.close(fig)


def polyline(points, x_min, x_max, y_min, y_max, left, top, width, height):
    coords = []
    x_den = max(x_max - x_min, 1e-8)
    y_den = max(y_max - y_min, 1e-8)
    for x, y in points:
        px = left + (x - x_min) / x_den * width
        py = top + (1.0 - (y - y_min) / y_den) * height
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def plot_curves_svg(path_svg, steps, d_str, d_sty, title, plot_mode, struct_axis_headroom):
    canvas_w, canvas_h = 980, 560
    left, top, width, height = 92, 58, 820, 390
    x_min, x_max = float(np.min(steps)), float(np.max(steps))
    y_min = struct_y_min = 0.0
    style_y_max = float(np.max(d_sty)) * 1.08 if np.max(d_sty) > 0 else 1.0
    if plot_mode == "dual_axis":
        struct_y_max = float(np.max(d_str)) * struct_axis_headroom if np.max(d_str) > 0 else 1.0
    else:
        struct_y_max = style_y_max = float(max(np.max(d_str), np.max(d_sty))) * 1.08
    str_points = polyline(zip(steps, d_str), x_min, x_max, struct_y_min, struct_y_max, left, top, width, height)
    sty_points = polyline(zip(steps, d_sty), x_min, x_max, y_min, style_y_max, left, top, width, height)

    grid = []
    for i in range(6):
        y = top + i * height / 5
        val = style_y_max - i * (style_y_max - y_min) / 5
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>')
        grid.append(f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="13" fill="#d62728">{val:.3f}</text>')
        if plot_mode == "dual_axis":
            rval = struct_y_max - i * (struct_y_max - struct_y_min) / 5
            grid.append(f'<text x="{left + width + 10}" y="{y + 5:.2f}" text-anchor="start" font-size="13" fill="#1f77b4">{rval:.4f}</text>')
    for i in range(6):
        x = left + i * width / 5
        val = x_min + i * (x_max - x_min) / 5
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" stroke="#eeeeee" stroke-width="1"/>')
        grid.append(f'<text x="{x:.2f}" y="{top + height + 26}" text-anchor="middle" font-size="13" fill="#333">{val:.0f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{canvas_w / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial, sans-serif" fill="#111">{title}</text>
  {''.join(grid)}
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#222" stroke-width="1.2"/>
  <polyline points="{sty_points}" fill="none" stroke="#d62728" stroke-width="3"/>
  <polyline points="{str_points}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  <text x="{left + width / 2}" y="{canvas_h - 28}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#111">Training iteration</text>
  <text x="25" y="{top + height / 2}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#d62728" transform="rotate(-90 25 {top + height / 2})">D_style</text>
  <text x="{canvas_w - 24}" y="{top + height / 2}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#1f77b4" transform="rotate(90 {canvas_w - 24} {top + height / 2})">D_struct</text>
  <line x1="{left + width - 210}" y1="{top + 26}" x2="{left + width - 165}" y2="{top + 26}" stroke="#d62728" stroke-width="3"/>
  <text x="{left + width - 155}" y="{top + 31}" font-size="15" font-family="Arial, sans-serif" fill="#111">D_style</text>
  <line x1="{left + width - 210}" y1="{top + 52}" x2="{left + width - 165}" y2="{top + 52}" stroke="#1f77b4" stroke-width="3"/>
  <text x="{left + width - 155}" y="{top + 57}" font-size="15" font-family="Arial, sans-serif" fill="#111">D_struct</text>
</svg>
'''
    path_svg.write_text(svg, encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    str_steps, d_str = read_metric_csv(args.d_str_csv)
    sty_steps, d_sty = read_metric_csv(args.d_sty_csv)
    steps, d_str, d_sty = align_series(str_steps, d_str, sty_steps, d_sty, args.align)
    grouped_steps, grouped_str = group_mean(steps, d_str, args.group_size)
    _, grouped_sty = group_mean(steps, d_sty, args.group_size)

    suffix = f"g{args.group_size}_{args.plot_mode}_{args.align}"
    csv_path = args.out_dir / f"grouped_cgsd_distances_{suffix}.csv"
    png_path = args.out_dir / f"grouped_cgsd_distances_{suffix}.png"
    pdf_path = args.out_dir / f"grouped_cgsd_distances_{suffix}.pdf"
    svg_path = args.out_dir / f"grouped_cgsd_distances_{suffix}.svg"

    write_grouped_csv(csv_path, grouped_steps, grouped_str, grouped_sty)
    if HAS_MATPLOTLIB:
        plot_curves_matplotlib(
            png_path, pdf_path, grouped_steps, grouped_str, grouped_sty,
            args.title, args.plot_mode, args.struct_axis_headroom,
        )
    else:
        plot_curves_svg(
            svg_path, grouped_steps, grouped_str, grouped_sty,
            args.title, args.plot_mode, args.struct_axis_headroom,
        )

    print(f"[OK] Saved grouped CSV: {csv_path}")
    if HAS_MATPLOTLIB:
        print(f"[OK] Saved plot PNG: {png_path}")
        print(f"[OK] Saved plot PDF: {pdf_path}")
    else:
        print(f"[OK] Saved plot SVG: {svg_path}")


if __name__ == "__main__":
    main()
