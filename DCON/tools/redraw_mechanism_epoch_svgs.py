#!/usr/bin/env python3
"""Redraw editable SVG mechanism curves from saved mechanism CSV files.

This script does not load checkpoints. It only reads existing CSV outputs from
analyze_cgsd_saam_training_dynamics.py, so edited CSV values can be re-plotted
without rerunning model evaluation.
"""

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


VARIANT_ORDER = ["w_CGSD", "w/o_CGSD"]
VARIANT_LABELS = {
    "w_CGSD": "w/ CGSD",
    "w/o_CGSD": "w/o CGSD",
    "w/ CGSD": "w/ CGSD",
    "w/o CGSD": "w/o CGSD",
}
VARIANT_COLORS = {
    "w_CGSD": "#d62728",
    "w/o_CGSD": "#1f77b4",
    "w/ CGSD": "#d62728",
    "w/o CGSD": "#1f77b4",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redraw D_style, d_stab, and Effective Rank SVGs from saved CSVs."
    )
    parser.add_argument("--result_dir", type=Path, required=True,
                        help="Mechanism visualization result directory, e.g. results_mechanism_visualization/CARDIAC_bSSFP_to_LGE_2.")
    parser.add_argument("--candidate", default=None,
                        help="Candidate name under candidates/. If unset, use recommended/metrics.csv when present.")
    parser.add_argument("--metrics_csv", type=Path, default=None,
                        help="Optional explicit metrics.csv path for D_style and d_stab.")
    parser.add_argument("--erank_csv", type=Path, default=None,
                        help="Optional explicit cgsd_style_effective_rank.csv path.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--value_mode", default="smooth", choices=["smooth", "raw"],
                        help="Use smoothed D_style/d_stab columns or raw columns from metrics.csv.")
    parser.add_argument("--width", type=int, default=760)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--combined_width", type=int, default=1500)
    parser.add_argument("--combined_height", type=int, default=460)
    parser.add_argument("--title_suffix", default="")
    parser.add_argument("--y_zero", action="store_true",
                        help="Force y-axis to start at 0 for all plots.")
    parser.add_argument("--plots", default="D_style,d_stab,erank",
                        help="Comma-separated plots to draw: D_style,d_stab,erank.")
    return parser.parse_args()


def as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_plots(spec):
    aliases = {
        "style": "D_style",
        "D_style": "D_style",
        "d_stab": "d_stab",
        "dstab": "d_stab",
        "erank": "erank",
        "effective_rank": "erank",
        "Effective_Rank": "erank",
    }
    plots = []
    for chunk in str(spec).split(","):
        key = chunk.strip()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown plot '{key}'. Use D_style,d_stab,erank.")
        normalized = aliases[key]
        if normalized not in plots:
            plots.append(normalized)
    if not plots:
        raise ValueError("--plots must contain at least one plot.")
    return plots


def resolve_metrics_csv(result_dir, candidate, explicit):
    if explicit is not None:
        return explicit
    if candidate:
        return result_dir / "candidates" / candidate / "metrics.csv"
    recommended = result_dir / "recommended" / "metrics.csv"
    if recommended.exists():
        return recommended
    selection = result_dir / "candidate_selection.csv"
    if selection.exists():
        rows = read_rows(selection)
        if rows:
            candidate_name = rows[0].get("candidate", "")
            local = result_dir / "candidates" / candidate_name / "metrics.csv"
            if local.exists():
                return local
            selected = Path(rows[0].get("metrics_csv", ""))
            if selected.exists():
                return selected
    raise FileNotFoundError(
        f"Could not find metrics.csv under {result_dir}. Pass --metrics_csv explicitly."
    )


def collect_metric(rows, metric):
    by_variant = {}
    for row in rows:
        variant = row.get("variant", "")
        epoch = as_float(row.get("epoch"))
        value = as_float(row.get(metric))
        if math.isnan(epoch) or math.isnan(value):
            continue
        by_variant.setdefault(variant, []).append((epoch, value))
    for variant in by_variant:
        by_variant[variant].sort(key=lambda item: item[0])
    return by_variant


def ordered_variants(series):
    ordered = [v for v in VARIANT_ORDER if v in series]
    ordered.extend(v for v in series if v not in ordered)
    return ordered


def nice_ticks(vmin, vmax, count=5):
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
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
        if value >= vmin - step * 0.2:
            ticks.append(value)
        value += step
    return ticks[:8]


def fmt_tick(value):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def path_points(points, x_min, x_max, y_min, y_max, left, top, width, height):
    coords = []
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    for x, y in points:
        px = left + (x - x_min) / x_span * width
        py = top + (1.0 - (y - y_min) / y_span) * height
        coords.append((px, py))
    return coords


def svg_polyline(coords, color, dashed=False):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"{dash} stroke-linejoin="round" stroke-linecap="round"/>'


def plot_svg(path, series, title, ylabel, width, height, y_zero=False, title_suffix=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = ordered_variants(series)
    all_points = [point for variant in variants for point in series[variant]]
    if not all_points:
        raise ValueError(f"No data available for {title}")
    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if y_zero:
        y_min = 0.0
    pad = (y_max - y_min) * 0.10 if y_max > y_min else max(abs(y_max), 1.0) * 0.10
    y_min = 0.0 if y_zero else y_min - pad
    y_max = y_max + pad

    left, right, top, bottom = 92, 34, 54, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_ticks = sorted(set(int(x) for x in x_values))
    if len(x_ticks) > 12:
        stride = math.ceil(len(x_ticks) / 11)
        x_ticks = x_ticks[::stride]
        if int(x_max) not in x_ticks:
            x_ticks.append(int(x_max))
    y_ticks = nice_ticks(y_min, y_max)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{escape(title + title_suffix)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.4"/>',
    ]

    for tick in y_ticks:
        coords = path_points([(x_min, tick)], x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)[0]
        y = coords[1]
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="#333">{fmt_tick(tick)}</text>')

    for tick in x_ticks:
        coords = path_points([(tick, y_min)], x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)[0]
        x = coords[0]
        elements.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#222" stroke-width="1"/>')
        elements.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">{int(tick)}</text>')

    for variant in variants:
        color = VARIANT_COLORS.get(variant, "#444444")
        dashed = variant in ("w/o_CGSD", "w/o CGSD")
        coords = path_points(series[variant], x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)
        elements.append(svg_polyline(coords, color, dashed))
        for x, y in coords:
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')

    legend_x, legend_y = left + plot_w - 150, top + 22
    for idx, variant in enumerate(variants):
        color = VARIANT_COLORS.get(variant, "#444444")
        dashed = variant in ("w/o_CGSD", "w/o CGSD")
        y = legend_y + idx * 26
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{color}" stroke-width="3"{dash} stroke-linecap="round"/>')
        elements.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Arial, sans-serif" font-size="15" fill="#111">{escape(VARIANT_LABELS.get(variant, variant))}</text>')

    elements.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="17">Epoch</text>')
    elements.append(f'<text x="24" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" transform="rotate(-90 24 {top + plot_h / 2:.1f})">{escape(ylabel)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def combined_svg(path, panels, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = width / len(panels)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for idx, (title, ylabel, series) in enumerate(panels):
        sub_path = path.with_name(f".tmp_panel_{idx}.svg")
        plot_svg(sub_path, series, title, ylabel, int(panel_w), height)
        text = sub_path.read_text(encoding="utf-8")
        inner = text.split(">", 1)[1].rsplit("</svg>", 1)[0]
        elements.append(f'<g transform="translate({idx * panel_w:.2f},0)">')
        elements.append(inner)
        elements.append("</g>")
        sub_path.unlink(missing_ok=True)
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def main():
    args = parse_args()
    result_dir = args.result_dir
    metrics_csv = resolve_metrics_csv(result_dir, args.candidate, args.metrics_csv)
    erank_csv = args.erank_csv or (result_dir / "cgsd_style_effective_rank.csv")
    out_dir = args.out_dir or (result_dir / "redraw_svg")

    metric_rows = read_rows(metrics_csv)
    erank_rows = read_rows(erank_csv)
    d_style_key = "D_style_raw" if args.value_mode == "raw" else "D_style"
    d_stab_key = "d_stab_raw" if args.value_mode == "raw" else "d_stab"

    d_style = collect_metric(metric_rows, d_style_key)
    d_stab = collect_metric(metric_rows, d_stab_key)
    erank = collect_metric(erank_rows, "Effective_Rank")

    title_suffix = f" {args.title_suffix}" if args.title_suffix else ""
    plot_specs = {
        "D_style": ("D_style_vs_epoch.svg", "D_style vs Epoch", "D_style", d_style),
        "d_stab": ("d_stab_vs_epoch.svg", "d_stab vs Epoch", "d_stab", d_stab),
        "erank": ("effective_rank_vs_epoch.svg", "Effective Rank vs Epoch", "Effective Rank", erank),
    }
    selected_plots = parse_plots(args.plots)
    selected_panels = []
    for key in selected_plots:
        filename, title, ylabel, series = plot_specs[key]
        plot_svg(out_dir / filename, series, title, ylabel, args.width, args.height, args.y_zero, title_suffix)
        selected_panels.append((title, ylabel, series))
    if len(selected_panels) > 1:
        combined_svg(
            out_dir / "mechanism_selected_curves.svg",
            selected_panels,
            args.combined_width,
            args.combined_height,
        )

    print(f"[OK] Read metrics CSV: {metrics_csv}")
    print(f"[OK] Read effective-rank CSV: {erank_csv}")
    print(f"[OK] Saved SVGs to: {out_dir}")


if __name__ == "__main__":
    main()
