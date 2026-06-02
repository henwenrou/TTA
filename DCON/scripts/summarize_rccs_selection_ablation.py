#!/usr/bin/env python3
"""Summarize RCCS selection ablation out.csv files into the reviewer table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROWS = [
    ("none", "w/o RCCS", "no random-conv candidate selection"),
    ("random", "RandomConv random pick", "random candidate from K candidates"),
    ("min", "RCCS min-distance", "argmin cosine distance to anchor"),
    ("max", "RCCS max-distance", "argmax cosine distance to anchor"),
]

TASKS = {
    "cardiac_bl": ("bSSFP", "LGE"),
    "cardiac_lb": ("LGE", "bSSFP"),
    "abdominal_sc": ("SABSCT", "CHAOST2"),
    "abdominal_cs": ("CHAOST2", "SABSCT"),
}


def target_overall(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    target_block = text.split("test for source domain", 1)[0]
    matches = re.findall(r"Overall mean dice by sample:?\s*([0-9]*\.?[0-9]+)", target_block)
    return float(matches[-1]) if matches else None


def selected_tasks(name: str) -> list[tuple[str, str]]:
    if name == "all":
        return list(TASKS.values())
    if name not in TASKS:
        raise ValueError(f"Unknown task set: {name}")
    return [TASKS[name]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="./ckpts", type=Path)
    parser.add_argument(
        "--tasks",
        default="cardiac_bl",
        choices=["cardiac_bl", "cardiac_lb", "abdominal_sc", "abdominal_cs", "all"],
    )
    args = parser.parse_args()

    tasks = selected_tasks(args.tasks)

    print("| Strong view strategy | Selection rule | Overall Dice |")
    print("| -------------------- | -------------- | ------------ |")
    for select, strategy, rule in ROWS:
        values = []
        for source, target in tasks:
            expname = f"saa_rccs_select_{select}_{source}_to_{target}"
            path = args.ckpt_dir / source / expname / "log" / "out.csv"
            value = target_overall(path)
            if value is not None:
                values.append(value)
        dice = "" if not values else f"{sum(values) / len(values):.4f}"
        print(f"| {strategy} | {rule} | {dice} |")

    print("\nLog paths:")
    for select, _, _ in ROWS:
        for source, target in tasks:
            expname = f"saa_rccs_select_{select}_{source}_to_{target}"
            exp_dir = args.ckpt_dir / source / expname
            print(f"- {source}->{target} {select}: {exp_dir / 'log' / 'out.csv'}")
            print(f"- {source}->{target} {select}: {exp_dir / 'log.txt'}")


if __name__ == "__main__":
    main()
