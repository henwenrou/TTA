#!/usr/bin/env python3
"""Profile approximate incremental training cost of SAA modules.

This script runs a sequence of short training-only jobs with matched settings:
ERM, two-view+SGF, +CGSD, +SAAM, and +RCCS. It reports each variant's
time/epoch and peak memory, plus the adjacent time delta as an approximate
module overhead.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SAA module-level training cost.")
    parser.add_argument("--source", default="CHAOST2", choices=["CHAOST2", "SABSCT", "bSSFP", "LGE"])
    parser.add_argument("--dataset", default="ABDOMINAL", choices=["ABDOMINAL", "CARDIAC"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--local-aug-type", default="clp", choices=["clp", "lla", "none"],
                        help="clp=class affine CLP, lla=Bezier location-scale.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stream-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dataset_cfg(dataset: str, source: str) -> dict[str, int]:
    if dataset == "ABDOMINAL":
        return {"nclass": 5, "sgf_grid_size": 3}
    if dataset == "CARDIAC":
        return {"nclass": 4, "sgf_grid_size": 18}
    raise ValueError(dataset)


def build_cmd(root: Path, args: argparse.Namespace, variant: dict[str, str | int]) -> list[str]:
    cfg = dataset_cfg(args.dataset, args.source)
    expname = str(variant["expname"])
    cmd = [
        sys.executable, "train.py",
        "--profile_cost", "1",
        "--profile_method", str(variant["label"]),
        "--num_workers", str(args.num_workers),
        "--expname", expname,
        "--phase", "train",
        "--ckpt_dir", str(root / "ckpts"),
        "--gpu_ids", args.gpu,
        "--f_seed", "42",
        "--lr", "0.0005",
        "--model", "unet",
        "--batchSize", str(args.batch_size),
        "--all_epoch", str(args.epochs),
        "--validation_freq", "999999",
        "--display_freq", "999999",
        "--save_freq", "999999",
        "--data_name", args.dataset,
        "--nclass", str(cfg["nclass"]),
        "--tr_domain", args.source,
        "--save_prediction", "False",
        "--w_ce", "1.0",
        "--w_dice", "1.0",
        "--w_seg", "1.0",
        "--local_aug_type", args.local_aug_type,
        "--use_sgf", str(variant["use_sgf"]),
        "--sgf_grid_size", str(cfg["sgf_grid_size"]),
        "--use_cgsd", str(variant["use_cgsd"]),
        "--use_projector", "1" if int(variant["use_cgsd"]) else "0",
        "--use_separate_cgsd_optimizer", "1",
        "--lambda_str", "0.3",
        "--lambda_sty", "0.3",
        "--use_saam", str(variant["use_saam"]),
        "--saam_tau", "0.5",
        "--saam_topk", "0.3",
        "--saam_stability_mode", "mean",
        "--lambda_01", "1.0",
        "--lambda_02", "1.0",
        "--saam_warmup_epochs", "50",
        "--saam_rampup_epochs", "100",
        "--anchor_seg_alpha", "0.0",
        "--strong_seg_alpha", "1.0",
        "--use_rccs", str(variant["use_rccs"]),
        "--p_rccs", "0.3",
        "--rccs_candidates", "4",
        "--rccs_metric", "cos",
        "--rccs_embed_dim", "128",
    ]
    if int(variant.get("erm_only", 0)):
        cmd += ["--erm_only", "1", "--use_sgf", "0", "--use_cgsd", "0", "--use_saam", "0", "--use_rccs", "0"]
    return cmd


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, stream_log: bool, dry_run: bool) -> None:
    print("\n[run]", " ".join(cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        if stream_log:
            proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            code = proc.wait()
        else:
            code = subprocess.call(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
    if code != 0:
        raise RuntimeError(f"Command failed: {log_path}")


def summarize_profile(path: Path, warmup_epochs: int) -> tuple[float, float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if int(r["epoch"]) > warmup_epochs]
    if not kept:
        kept = rows
    mean_time = sum(float(r["train_time_sec"]) for r in kept) / len(kept)
    peak_mem = max(float(r["peak_mem_gb"]) for r in kept)
    return mean_time, peak_mem


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_name = args.run_name or f"saa_module_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "cost_profiles" / run_name

    variants = [
        {"key": "erm", "label": "ERM", "erm_only": 1, "use_sgf": 0, "use_cgsd": 0, "use_saam": 0, "use_rccs": 0},
        {"key": "twoview_sgf", "label": "+two-view+SGF", "use_sgf": 1, "use_cgsd": 0, "use_saam": 0, "use_rccs": 0},
        {"key": "cgsd", "label": "+CGSD", "use_sgf": 1, "use_cgsd": 1, "use_saam": 0, "use_rccs": 0},
        {"key": "saam", "label": "+SAAM", "use_sgf": 1, "use_cgsd": 1, "use_saam": 1, "use_rccs": 0},
        {"key": "full", "label": "+RCCS/full SAA", "use_sgf": 1, "use_cgsd": 1, "use_saam": 1, "use_rccs": 1},
    ]
    for variant in variants:
        variant["expname"] = f"{run_name}_{variant['key']}_{args.source}"
        cmd = build_cmd(root, args, variant)
        run_cmd(cmd, root, out_dir / f"{variant['key']}.log", args.stream_log, args.dry_run)

    if args.dry_run:
        return

    rows = []
    previous_time = None
    for variant in variants:
        profile_path = root / "ckpts" / args.source / str(variant["expname"]) / "log" / "cost_profile.csv"
        mean_time, peak_mem = summarize_profile(profile_path, args.warmup_epochs)
        delta = "" if previous_time is None else f"{mean_time - previous_time:.3f}"
        rows.append([variant["label"], f"{mean_time:.3f}", delta, f"{peak_mem:.3f}", "1.0x", str(profile_path)])
        previous_time = mean_time

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "saa_module_cost_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Train time / epoch (s)", "Delta vs previous (s)", "Peak GPU memory (GB)", "Inference cost", "Profile CSV"])
        writer.writerows(rows)

    md_path = out_dir / "saa_module_cost_summary.md"
    with md_path.open("w") as f:
        f.write(f"Local augmentation: {args.local_aug_type}\n\n")
        f.write("| Variant | Train time / epoch | Delta vs previous | Peak GPU memory | Inference cost |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row[0]} | {row[1]}s | {row[2] or '-'} | {row[3]} GB | {row[4]} |\n")

    print(f"\nWrote:\n  {csv_path}\n  {md_path}")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
