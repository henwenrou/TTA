#!/usr/bin/env python3
"""Summarize BL/LB backbone comparison results.

Reads target-domain "Overall mean dice by sample" from each run's log/out.csv
and builds a compact table across SLAug, DCON-style, and SAA runs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TASKS = {
    "bl": ("bSSFP", "LGE"),
    "lb": ("LGE", "bSSFP"),
}


def parse_target_overall(out_csv: Path) -> float | None:
    if not out_csv.exists():
        return None
    text = out_csv.read_text(errors="ignore")
    marker = "Test mode evaluation"
    if marker in text:
        text = text.split(marker, 1)[1]
    source_marker = "Test on source domain"
    if source_marker in text:
        text = text.split(source_marker, 1)[0]
    matches = re.findall(r"Overall mean dice by sample:?\s*([0-9.eE+-]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def result_path(root: Path, method: str, backbone: str, source: str, target: str) -> Path:
    if method == "SAA":
        run_name = f"{backbone}_{source}_to_{target}"
    elif method == "DCON-style":
        run_name = f"{backbone}_dcon_{source}_to_{target}"
    elif method == "SLAug":
        run_name = f"{backbone}_slaug_{source}_to_{target}"
    else:
        raise ValueError(f"Unknown method: {method}")
    return root / run_name / "log" / "out.csv"


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saa-root", type=Path, default=Path("results_backbone_ablation"))
    parser.add_argument("--dcon-root", type=Path, default=Path("results_dcon_backbone_bl_lb"))
    parser.add_argument("--slaug-root", type=Path, default=Path("results_slaug_backbone_bl_lb"))
    parser.add_argument("--backbones", nargs="+", default=["nnunet", "swinunet"])
    parser.add_argument("--tasks", nargs="+", default=["bl", "lb"], choices=sorted(TASKS))
    parser.add_argument("--out-csv", type=Path, default=Path("results_backbone_compare_bl_lb.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("results_backbone_compare_bl_lb.md"))
    args = parser.parse_args()

    roots = {
        "SLAug": args.slaug_root,
        "DCON-style": args.dcon_root,
        "SAA": args.saa_root,
    }
    rows = []
    for backbone in args.backbones:
        for task in args.tasks:
            source, target = TASKS[task]
            values = {}
            paths = {}
            for method, root in roots.items():
                path = result_path(root, method, backbone, source, target)
                values[method] = parse_target_overall(path)
                paths[method] = str(path)
            dcon = values["DCON-style"]
            saa = values["SAA"]
            rows.append({
                "backbone": backbone,
                "task": task,
                "source": source,
                "target": target,
                "slaug": values["SLAug"],
                "dcon_style": dcon,
                "saa": saa,
                "saa_minus_dcon_style": None if saa is None or dcon is None else saa - dcon,
                "slaug_path": paths["SLAug"],
                "dcon_style_path": paths["DCON-style"],
                "saa_path": paths["SAA"],
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True) if args.out_csv.parent != Path(".") else None
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Backbone | Task | Source->Target | SLAug | DCON-style | SAA | SAA-DCON |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backbone} | {task} | {source}->{target} | {slaug} | {dcon} | {saa} | {delta} |".format(
                backbone=row["backbone"],
                task=row["task"],
                source=row["source"],
                target=row["target"],
                slaug=fmt(row["slaug"]),
                dcon=fmt(row["dcon_style"]),
                saa=fmt(row["saa"]),
                delta=fmt(row["saa_minus_dcon_style"]),
            )
        )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
