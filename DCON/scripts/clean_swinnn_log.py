#!/usr/bin/env python3
"""Clean verbose test logs and keep checkpoint/model metrics.

By default this reads swinnn.txt and writes:
  - swinnn_cleaned.txt: compact readable blocks
  - swinnn_cleaned.csv: one row per checkpoint/run

The parser is intentionally line-based so it works with copied terminal logs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRIC_KEYS = {
    "bg": ("target_bg_mean", "target_bg_std"),
    "LV": ("target_lv_mean", "target_lv_std"),
    "Myo": ("target_myo_mean", "target_myo_std"),
    "RV": ("target_rv_mean", "target_rv_std"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("swinnn.txt"),
        help="Raw log file to clean. Default: swinnn.txt",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("swinnn_cleaned.txt"),
        help="Cleaned text output. Default: swinnn_cleaned.txt",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("swinnn_cleaned.csv"),
        help="CSV summary output. Default: swinnn_cleaned.csv",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Also keep source-domain evaluation blocks.",
    )
    return parser.parse_args()


def checkpoint_name(path: str) -> str:
    if not path:
        return ""
    return Path(path).name


def expname_from_checkpoint(path: str) -> str:
    name = checkpoint_name(path)
    return name[:-4] if name.endswith(".pth") else name


def extract_namespace_value(line: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=('[^']*'|[^,\)]+)", line)
    if match is None:
        return ""
    return match.group(1).strip().strip("'")


def ensure_run(runs: list[dict], current: dict | None) -> dict:
    if current is None:
        current = new_run()
        runs.append(current)
    return current


def new_run() -> dict:
    return {
        "task": "",
        "source": "",
        "target": "",
        "backbone": "",
        "checkpoint": "",
        "checkpoint_file": "",
        "expname": "",
        "checkpoint_dir": "",
        "experiment_dir": "",
        "target_block": [],
        "source_block": [],
    }


def has_run_content(run: dict) -> bool:
    return any(
        run.get(key)
        for key in (
            "task",
            "backbone",
            "checkpoint",
            "checkpoint_dir",
            "experiment_dir",
            "target_block",
            "source_block",
        )
    )


def parse_log(text: str) -> list[dict]:
    lines = text.splitlines()
    runs: list[dict] = []
    current: dict | None = None
    capture: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        header = re.search(r"Testing\s+\w+:\s+([^,]+),\s+backbone=([^\s]+)", line)
        if header:
            if current is None or has_run_content(current):
                current = new_run()
                runs.append(current)
            current["task"] = header.group(1).strip()
            current["backbone"] = header.group(2).strip()
            parts = current["task"].split("->", 1)
            if len(parts) == 2:
                current["source"], current["target"] = parts[0].strip(), parts[1].strip()
            capture = None
            i += 1
            continue

        if "config:Namespace(" in line:
            current = ensure_run(runs, current)
            for key in ("backbone", "expname", "tr_domain", "target_domain", "resume_path"):
                value = extract_namespace_value(line, key)
                if not value:
                    continue
                if key == "tr_domain":
                    current["source"] = value
                elif key == "target_domain":
                    current["target"] = value
                elif key == "resume_path":
                    current["checkpoint"] = value
                    current["checkpoint_file"] = checkpoint_name(value)
                    current["expname"] = current["expname"] or expname_from_checkpoint(value)
                else:
                    current[key] = value
            if current["source"] and current["target"] and not current["task"]:
                current["task"] = f"{current['source']}->{current['target']}"
            capture = None
            i += 1
            continue

        if stripped.startswith("Checkpoint:"):
            current = ensure_run(runs, current)
            value = stripped.split(":", 1)[1].strip()
            current["checkpoint"] = value
            current["checkpoint_file"] = checkpoint_name(value)
            current["expname"] = current["expname"] or expname_from_checkpoint(value)
            capture = None
            i += 1
            continue

        if stripped.startswith("Loading checkpoint:"):
            current = ensure_run(runs, current)
            value = stripped.split(":", 1)[1].strip()
            current["checkpoint"] = value
            current["checkpoint_file"] = checkpoint_name(value)
            current["expname"] = current["expname"] or expname_from_checkpoint(value)
            capture = None
            i += 1
            continue

        if stripped.startswith("Checkpoint directory:"):
            current = ensure_run(runs, current)
            current["checkpoint_dir"] = stripped.split(":", 1)[1].strip()
            capture = None
            i += 1
            continue

        if stripped.startswith("Experiment directory:"):
            current = ensure_run(runs, current)
            current["experiment_dir"] = stripped.split(":", 1)[1].strip()
            value = current["experiment_dir"].rstrip("/").split("/")[-1]
            current["expname"] = current["expname"] or value
            capture = None
            i += 1
            continue

        if re.search(r"\bbackbone:\s*", stripped, flags=re.IGNORECASE):
            current = ensure_run(runs, current)
            current["backbone"] = stripped.split(":", 1)[1].strip()
            i += 1
            continue

        if re.search(r"\bname:\s*", stripped, flags=re.IGNORECASE):
            current = ensure_run(runs, current)
            current["expname"] = stripped.split(":", 1)[1].strip()
            i += 1
            continue

        if stripped == "Testing on target domain...":
            current = ensure_run(runs, current)
            current["target_block"].clear()
            if i > 0 and set(lines[i - 1].strip()) == {"="}:
                current["target_block"].append(lines[i - 1])
            current["target_block"].append(line)
            if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"="}:
                current["target_block"].append(lines[i + 1])
                i += 1
            capture = "target_block"
            i += 1
            continue

        if stripped == "Testing on source domain...":
            current = ensure_run(runs, current)
            current["source_block"].clear()
            if i > 0 and set(lines[i - 1].strip()) == {"="}:
                current["source_block"].append(lines[i - 1])
            current["source_block"].append(line)
            if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"="}:
                current["source_block"].append(lines[i + 1])
                i += 1
            capture = "source_block"
            i += 1
            continue

        if capture:
            if (
                stripped.startswith("Test completed!")
                or stripped.startswith("Testing ")
                or stripped.startswith("Checkpoint directory:")
                or stripped.startswith("[")
            ):
                capture = None
                continue
            current[capture].append(line)

        i += 1

    return [run for run in runs if has_run_content(run)]


def clean_block(block: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in block:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if "evaluation:" in stripped and "%" in stripped:
            cleaned.append(re.sub(r"\s*\[[0-9:.<>, ?it/s]+\]\s*$", "", line))
            continue
        cleaned.append(line)
    while cleaned and (cleaned[-1] == "" or set(cleaned[-1].strip()) == {"="}):
        cleaned.pop()
    return cleaned


def format_run(run: dict, include_source: bool) -> str:
    lines = [
        "=" * 80,
        f"Task: {run.get('task') or '-'}",
        f"Backbone: {run.get('backbone') or '-'}",
        f"Checkpoint: {run.get('checkpoint') or '-'}",
        f"Checkpoint file: {run.get('checkpoint_file') or '-'}",
        f"Experiment: {run.get('expname') or '-'}",
    ]
    if run.get("checkpoint_dir"):
        lines.append(f"Checkpoint directory: {run['checkpoint_dir']}")
    if run.get("experiment_dir"):
        lines.append(f"Experiment directory: {run['experiment_dir']}")
    lines.append("=" * 80)
    lines.append("")
    lines.extend(clean_block(run.get("target_block", [])))
    if include_source and run.get("source_block"):
        lines.append("")
        lines.extend(clean_block(run["source_block"]))
    return "\n".join(lines).rstrip()


def block_metric(block: list[str], organ: str) -> tuple[str, str]:
    text = "\n".join(block)
    pattern = (
        rf"Organ\s+{re.escape(organ)}\s+with dice:\s+mean:\s*([0-9.eE+-]+)"
        rf"\s*,\s*std:\s*([0-9.eE+-]+)"
    )
    match = re.search(pattern, text)
    if match is None:
        return "", ""
    return match.group(1), match.group(2)


def block_overall(block: list[str], name: str) -> str:
    match = re.search(rf"{re.escape(name)}\s+([0-9.eE+-]+)", "\n".join(block))
    return "" if match is None else match.group(1)


def write_csv(path: Path, runs: list[dict], include_source: bool) -> None:
    fields = [
        "task",
        "source",
        "target",
        "backbone",
        "checkpoint_file",
        "checkpoint",
        "expname",
        "target_bg_mean",
        "target_bg_std",
        "target_lv_mean",
        "target_lv_std",
        "target_myo_mean",
        "target_myo_std",
        "target_rv_mean",
        "target_rv_std",
        "target_overall_by_sample",
        "target_overall_by_domain",
    ]
    if include_source:
        fields.extend(["source_overall_by_sample", "source_overall_by_domain"])

    rows = []
    for run in runs:
        row = {field: run.get(field, "") for field in fields}
        for organ, keys in METRIC_KEYS.items():
            row[keys[0]], row[keys[1]] = block_metric(run.get("target_block", []), organ)
        row["target_overall_by_sample"] = block_overall(
            run.get("target_block", []), "Overall mean dice by sample"
        )
        row["target_overall_by_domain"] = block_overall(
            run.get("target_block", []), "Overall mean dice by domain"
        )
        if include_source:
            row["source_overall_by_sample"] = block_overall(
                run.get("source_block", []), "Overall mean dice by sample"
            )
            row["source_overall_by_domain"] = block_overall(
                run.get("source_block", []), "Overall mean dice by domain"
            )
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"Input log not found: {args.input}\n"
            "Put swinnn.txt in the current directory, or pass it with -i /path/to/log.txt"
        )
    text = args.input.read_text(encoding="utf-8", errors="ignore")
    runs = parse_log(text)
    if not runs:
        raise SystemExit(f"No test result blocks found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = "\n\n".join(format_run(run, args.include_source) for run in runs)
    args.output.write_text(cleaned + "\n", encoding="utf-8")
    write_csv(args.csv, runs, args.include_source)
    print(f"Parsed {len(runs)} run(s)")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
