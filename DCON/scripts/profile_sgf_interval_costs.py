#!/usr/bin/env python3
"""Profile SGF interval cost for view construction.

Runs matched two-view training variants:
  - no_sgf: use local strong view directly
  - sgf_every_1: compute SGF saliency every iteration
  - sgf_every_2: compute SGF saliency every 2 iterations
  - sgf_every_4: compute SGF saliency every 4 iterations

CGSD, SAAM, and RCCS are disabled to isolate view-construction cost.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SGF interval training overhead.")
    parser.add_argument("--source", default="CHAOST2", choices=["CHAOST2", "SABSCT", "bSSFP", "LGE"])
    parser.add_argument("--dataset", default="ABDOMINAL", choices=["ABDOMINAL", "CARDIAC"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--local-aug-type", default="clp", choices=["clp", "lla", "none"])
    parser.add_argument("--data-root", default=None, help="Dataset root. Defaults to <project>/data.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stream-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dataset_cfg(dataset: str) -> dict[str, int]:
    if dataset == "ABDOMINAL":
        return {"nclass": 5, "sgf_grid_size": 3}
    if dataset == "CARDIAC":
        return {"nclass": 4, "sgf_grid_size": 18}
    raise ValueError(dataset)


def build_cmd(root: Path, args: argparse.Namespace, expname: str, label: str, use_sgf: int, sgf_interval: int) -> list[str]:
    cfg = dataset_cfg(args.dataset)
    return [
        sys.executable, "train.py",
        "--profile_cost", "1",
        "--profile_method", label,
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
        "--use_sgf", str(use_sgf),
        "--sgf_grid_size", str(cfg["sgf_grid_size"]),
        "--sgf_interval", str(sgf_interval),
        "--use_cgsd", "0",
        "--use_projector", "0",
        "--use_saam", "0",
        "--use_rccs", "0",
    ]


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, stream_log: bool, dry_run: bool, data_root: Path) -> None:
    print("\n[run]", " ".join(cmd))
    print(f"SAA_DATA_ROOT={data_root}")
    if dry_run:
        return
    env = os.environ.copy()
    env["SAA_DATA_ROOT"] = str(data_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        if stream_log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            code = proc.wait()
        else:
            code = subprocess.call(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    if code != 0:
        raise RuntimeError(f"Command failed. See log: {log_path}")


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
    data_root = Path(args.data_root).resolve() if args.data_root else root / "data"
    if not args.dry_run and not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    run_name = args.run_name or f"sgf_interval_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "cost_profiles" / run_name

    variants = [
        ("no_sgf", "no SGF", 0, 1, 0.0),
        ("sgf_every_1", "SGF every 1", 1, 1, 1.0),
        ("sgf_every_2", "SGF every 2", 1, 2, 0.5),
        ("sgf_every_4", "SGF every 4", 1, 4, 0.25),
    ]

    for key, label, use_sgf, sgf_interval, _active_fraction in variants:
        expname = f"{run_name}_{key}_{args.source}"
        cmd = build_cmd(root, args, expname, label, use_sgf, sgf_interval)
        run_cmd(cmd, root, out_dir / f"{key}.log", args.stream_log, args.dry_run, data_root)

    if args.dry_run:
        return

    results = []
    no_sgf_time = None
    for key, label, _use_sgf, _sgf_interval, active_fraction in variants:
        expname = f"{run_name}_{key}_{args.source}"
        profile_path = root / "ckpts" / args.source / expname / "log" / "cost_profile.csv"
        mean_time, peak_mem = summarize_profile(profile_path, args.warmup_epochs)
        if no_sgf_time is None:
            no_sgf_time = mean_time
        overhead = mean_time - no_sgf_time
        per_active_overhead = overhead / active_fraction if active_fraction > 0 else 0.0
        results.append([
            label,
            active_fraction,
            mean_time,
            overhead,
            per_active_overhead,
            peak_mem,
            str(profile_path),
        ])

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sgf_interval_cost_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Variant",
            "SGF active fraction",
            "Train time / epoch (s)",
            "SGF overhead vs no SGF (s)",
            "Overhead normalized by active fraction (s)",
            "Peak GPU memory (GB)",
            "Profile CSV",
        ])
        for row in results:
            writer.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.6f}", row[6]])

    md_path = out_dir / "sgf_interval_cost_summary.md"
    with md_path.open("w") as f:
        f.write(f"Local augmentation: {args.local_aug_type}\n\n")
        f.write("| Variant | SGF active fraction | Train time / epoch | SGF overhead vs no SGF | Normalized overhead | Peak GPU memory |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in results:
            f.write(f"| {row[0]} | {row[1]:.2f} | {row[2]:.3f}s | {row[3]:.3f}s | {row[4]:.3f}s | {row[5]:.3f} GB |\n")

    print(f"\nWrote:\n  {csv_path}\n  {md_path}")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
