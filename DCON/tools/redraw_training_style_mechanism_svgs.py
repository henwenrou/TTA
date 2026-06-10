#!/usr/bin/env python3
"""Redraw mechanism SVGs in the training-analysis plotting style.

The script reads existing CSV files only. It does not load models, replay
checkpoints, or change any training outputs.
"""

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"
GRID = "#d9d9d9"
AXIS = "#111111"
TEXT = "#111111"


def parse_args():
    parser = argparse.ArgumentParser(description="Redraw editable training-style mechanism SVGs from CSVs.")
    parser.add_argument("--dstab_csv", type=Path, default=Path("dstabvsepoch.csv"))
    parser.add_argument("--erank_csv", type=Path, default=Path("cgsd_style_effective_rank.csv"))
    parser.add_argument("--grouped_csv", type=Path,
                        default=Path("results_cgsd_grouped_curves/grouped_cgsd_distances_g100_dual_axis_iteration.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results_mechanism_epoch_svg"))
    parser.add_argument("--grouped_out_dir", type=Path, default=Path("results_cgsd_grouped_curves"))
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--combined_width", type=int, default=1700)
    parser.add_argument("--combined_height", type=int, default=560)
    return parser.parse_args()


def as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_variant_metric(path, metric):
    rows = read_csv(path)
    series = {}
    for row in rows:
        variant = row.get("variant", "")
        epoch = as_float(row.get("epoch"))
        value = as_float(row.get(metric))
        if math.isnan(epoch) or math.isnan(value):
            continue
        series.setdefault(variant, []).append((epoch, value))
    for variant in series:
        series[variant].sort(key=lambda x: x[0])
    return series


def read_grouped(path):
    rows = read_csv(path)
    x, d_struct, d_style = [], [], []
    for row in rows:
        step = as_float(row.get("mean_iter"))
        struct = as_float(row.get("D_struct"))
        style = as_float(row.get("D_style"))
        if math.isnan(step) or math.isnan(struct) or math.isnan(style):
            continue
        x.append(step)
        d_struct.append(struct)
        d_style.append(style)
    if not x:
        raise ValueError(f"No grouped CGSD values parsed from {path}")
    return x, d_struct, d_style


def fmt_tick(value):
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def nice_ticks(vmin, vmax, count=5):
    if vmax <= vmin:
        return [vmin]
    raw = (vmax - vmin) / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = raw / power
    if step <= 1:
        step = 1
    elif step <= 2:
        step = 2
    elif step <= 5:
        step = 5
    else:
        step = 10
    step *= power
    start = math.floor(vmin / step) * step
    ticks = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.25 and value <= vmax + step * 0.05:
            ticks.append(value)
        value += step
    return ticks[:7]


def padded_limits(values, pad_frac=0.08, zero=False):
    vmin, vmax = min(values), max(values)
    if zero:
        vmin = 0.0
    pad = (vmax - vmin) * pad_frac if vmax > vmin else max(abs(vmax), 1.0) * pad_frac
    return (0.0 if zero else vmin - pad), vmax + pad


def xy(points, x_min, x_max, y_min, y_max, left, top, width, height):
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    return [
        (
            left + (x - x_min) / x_span * width,
            top + (1.0 - (y - y_min) / y_span) * height,
        )
        for x, y in points
    ]


def line_svg(coords, color, dashed=False, marker=True, width=3.4):
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    out = [
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
    ]
    if marker:
        for x, y in coords:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="{color}" stroke="white" stroke-width="1.4"/>')
    return "\n".join(out)


def base_plot_elements(width, height, title, xlabel, ylabel, x_ticks, y_ticks,
                       x_min, x_max, y_min, y_max, left, top, plot_w, plot_h):
    elements = [
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="26" fill="{TEXT}">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="{AXIS}" stroke-width="1.8"/>',
    ]
    for tick in y_ticks:
        _, py = xy([(x_min, tick)], x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)[0]
        elements.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="{GRID}" stroke-width="1" opacity="0.75"/>')
        elements.append(f'<text x="{left - 12}" y="{py + 6:.2f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="17" fill="{TEXT}">{fmt_tick(tick)}</text>')
    for tick in x_ticks:
        px, _ = xy([(tick, y_min)], x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)[0]
        elements.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="1" opacity="0.75"/>')
        elements.append(f'<text x="{px:.2f}" y="{top + plot_h + 34}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="18" fill="{TEXT}">{fmt_tick(tick)}</text>')
    elements.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="22" fill="{TEXT}">{escape(xlabel)}</text>')
    elements.append(f'<text x="27" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="22" fill="{TEXT}" transform="rotate(-90 27 {top + plot_h / 2:.1f})">{escape(ylabel)}</text>')
    return elements


def plot_variant_svg(path, series, title, ylabel, width, height, baseline=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = [v for v in ("w_CGSD", "w/o_CGSD") if v in series]
    all_points = [p for v in variants for p in series[v]]
    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points]
    if baseline is not None:
        y_values.append(float(baseline))
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = padded_limits(y_values, 0.08)
    left, right, top, bottom = 92, 34, 58, 84
    plot_w, plot_h = width - left - right, height - top - bottom
    x_ticks = sorted(set(x_values))
    if len(x_ticks) > 7:
        x_ticks = [x for x in x_ticks if int(x) % 50 == 0]
    y_ticks = nice_ticks(y_min, y_max, 5)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    elements.extend(base_plot_elements(width, height, title, "Epoch", ylabel, x_ticks, y_ticks,
                                       x_min, x_max, y_min, y_max, left, top, plot_w, plot_h))
    if baseline is not None:
        _, baseline_y = xy([(x_min, float(baseline))], x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)[0]
        elements.append(
            f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_w}" y2="{baseline_y:.2f}" '
            f'stroke="#555555" stroke-width="1.8" stroke-dasharray="8 6" opacity="0.85"/>'
        )
        elements.append(
            f'<text x="{left + plot_w - 8}" y="{baseline_y - 8:.2f}" text-anchor="end" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#555555">1.0</text>'
        )
    styles = {
        "w_CGSD": (BLUE, False, "w/ CGSD"),
        "w/o_CGSD": (ORANGE, True, "w/o CGSD"),
    }
    for variant in variants:
        color, dashed, _label = styles[variant]
        elements.append(line_svg(xy(series[variant], x_min, x_max, y_min, y_max, left, top, plot_w, plot_h), color, dashed))

    legend_x, legend_y = left + plot_w - 190, top + 34
    elements.append(f'<rect x="{legend_x - 14}" y="{legend_y - 24}" width="178" height="{32 * len(variants) + 18}" fill="white" opacity="0.82" stroke="#cccccc" stroke-width="1"/>')
    for idx, variant in enumerate(variants):
        color, dashed, label = styles[variant]
        y = legend_y + idx * 34
        dash = ' stroke-dasharray="10 7"' if dashed else ""
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 44}" y2="{y}" stroke="{color}" stroke-width="3.4" stroke-linecap="round"{dash}/>')
        elements.append(f'<circle cx="{legend_x + 22}" cy="{y}" r="6.5" fill="{color}" stroke="white" stroke-width="1.2"/>')
        elements.append(f'<text x="{legend_x + 58}" y="{y + 7}" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="{TEXT}">{escape(label)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def combine_two_svg(path, left_svg, right_svg, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = width / 2.0
    inners = []
    for svg_path in (left_svg, right_svg):
        text = svg_path.read_text(encoding="utf-8")
        inners.append(text.split(">", 1)[1].rsplit("</svg>", 1)[0])
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<g transform="scale({panel_w / 900:.6f} {height / 560:.6f})">{inners[0]}</g>',
        f'<g transform="translate({panel_w:.2f},0) scale({panel_w / 900:.6f} {height / 560:.6f})">{inners[1]}</g>',
        "</svg>",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def plot_grouped_dual_axis(path, steps, d_struct, d_style, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max = min(steps), max(steps)
    # Use zero-based dual axes like the training-analysis reference figure.
    # This keeps D_style visually above D_struct while preserving both scales.
    y1_min, y1_max = 0.0, max(d_style) * 1.08
    y2_min, y2_max = 0.0, max(d_struct) * 3.0
    left, right, top, bottom = 94, 94, 58, 84
    plot_w, plot_h = width - left - right, height - top - bottom
    x_ticks = [steps[0], *[s for s in steps if int(s) % 1000 in (98, 99, 0)], steps[-1]]
    seen = set()
    x_ticks = [x for x in x_ticks if not (round(x, 2) in seen or seen.add(round(x, 2)))]
    if len(x_ticks) > 8:
        stride = math.ceil(len(x_ticks) / 7)
        x_ticks = x_ticks[::stride]
        if steps[-1] not in x_ticks:
            x_ticks.append(steps[-1])
    y1_ticks = nice_ticks(y1_min, y1_max, 5)
    y2_ticks = nice_ticks(y2_min, y2_max, 5)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    elements.extend(base_plot_elements(width, height, "Training CGSD Distance Running Mean",
                                       "Training iteration", "D_style", x_ticks, y1_ticks,
                                       x_min, x_max, y1_min, y1_max, left, top, plot_w, plot_h))
    # Match the reference: left axis is D_style/red, right axis is D_struct/blue.
    for idx, element in enumerate(elements):
        if f'x="{left - 12}"' in element and 'text-anchor="end"' in element:
            elements[idx] = element.replace(f'fill="{TEXT}"', f'fill="{RED}"')
        if 'rotate(-90' in element and '>D_style<' in element:
            elements[idx] = element.replace(f'fill="{TEXT}"', f'fill="{RED}"').replace('font-size="22"', 'font-size="22" font-weight="700"')
    for tick in y2_ticks:
        _, py = xy([(x_min, tick)], x_min, x_max, y2_min, y2_max, left, top, plot_w, plot_h)[0]
        elements.append(f'<text x="{left + plot_w + 12}" y="{py + 6:.2f}" text-anchor="start" font-family="Arial, Helvetica, sans-serif" font-size="17" fill="{BLUE}">{fmt_tick(tick)}</text>')
    elements.append(f'<text x="{width - 26}" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="{BLUE}" transform="rotate(90 {width - 26} {top + plot_h / 2:.1f})">D_struct</text>')

    style_coords = xy(list(zip(steps, d_style)), x_min, x_max, y1_min, y1_max, left, top, plot_w, plot_h)
    struct_coords = xy(list(zip(steps, d_struct)), x_min, x_max, y2_min, y2_max, left, top, plot_w, plot_h)
    elements.append(line_svg(style_coords, RED, dashed=False, marker=False, width=3.8))
    elements.append(line_svg(struct_coords, BLUE, dashed=False, marker=False, width=3.8))

    legend_x, legend_y = left + plot_w - 220, top + 34
    elements.append(f'<rect x="{legend_x - 14}" y="{legend_y - 24}" width="204" height="86" fill="white" opacity="0.82" stroke="#cccccc" stroke-width="1"/>')
    for idx, (color, label) in enumerate(((RED, "D_style"), (BLUE, "D_struct"))):
        y = legend_y + idx * 34
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 44}" y2="{y}" stroke="{color}" stroke-width="3.4" stroke-linecap="round"/>')
        elements.append(f'<text x="{legend_x + 58}" y="{y + 7}" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="{TEXT}">{label}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def main():
    args = parse_args()
    dstab = collect_variant_metric(args.dstab_csv, "d_stab")
    dstyle = collect_variant_metric(args.erank_csv, "D_style")
    erank = collect_variant_metric(args.erank_csv, "Effective_Rank")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dstab_svg = args.out_dir / "d_stab_vs_epoch_training_style.svg"
    dstyle_svg = args.out_dir / "D_style_vs_epoch_training_style.svg"
    erank_svg = args.out_dir / "effective_rank_vs_epoch_training_style.svg"
    plot_variant_svg(dstab_svg, dstab, "d_stab vs Epoch", "d_stab", args.width, args.height)
    plot_variant_svg(dstyle_svg, dstyle, "D_style vs Epoch", "D_style", args.width, args.height)
    plot_variant_svg(erank_svg, erank, "Effective Rank vs Epoch", "Effective Rank", args.width, args.height, baseline=1.0)
    combine_two_svg(args.out_dir / "dstab_erank_training_style.svg", dstab_svg, erank_svg,
                    args.combined_width, args.combined_height)
    combine_two_svg(args.out_dir / "cgsd_style_effective_rank_training_style.svg", dstyle_svg, erank_svg,
                    args.combined_width, args.combined_height)

    steps, d_struct, d_style = read_grouped(args.grouped_csv)
    args.grouped_out_dir.mkdir(parents=True, exist_ok=True)
    grouped_svg = args.grouped_out_dir / "grouped_cgsd_distances_training_style.svg"
    plot_grouped_dual_axis(grouped_svg, steps, d_struct, d_style, args.width, args.height)
    print(f"[OK] Saved d_stab SVG: {dstab_svg}")
    print(f"[OK] Saved D_style SVG: {dstyle_svg}")
    print(f"[OK] Saved effective-rank SVG: {erank_svg}")
    print(f"[OK] Saved combined SVG: {args.out_dir / 'dstab_erank_training_style.svg'}")
    print(f"[OK] Saved style/effective-rank SVG: {args.out_dir / 'cgsd_style_effective_rank_training_style.svg'}")
    print(f"[OK] Saved grouped CGSD SVG: {grouped_svg}")


if __name__ == "__main__":
    main()
