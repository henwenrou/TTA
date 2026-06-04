#!/usr/bin/env python3
"""Profile the original small DCON baseline as a standalone run.

The original DCON code does not have built-in profiling flags. This wrapper:
1. launches DCON/DCON/train.py,
2. streams the training log to the terminal and a file,
3. samples peak GPU memory with nvidia-smi,
4. parses "End of epoch ... Time Taken: ... sec",
5. writes cost_profile.csv and cost_summary.md.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile original small DCON training cost.")
    parser.add_argument("--source", default="CHAOST2", choices=["CHAOST2", "SABSCT"],
                        help="Source domain for ABDOMINAL.")
    parser.add_argument("--gpu", default="0", help="GPU id passed to DCON train.py.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of profiling epochs.")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Drop this many initial epochs when summarizing.")
    parser.add_argument("--batch-size", type=int, default=20, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--data-root", default=None,
                        help="Data root containing abdominal/CHAOST2 and abdominal/SABSCT. Defaults to DCON/data.")
    parser.add_argument("--run-name", default=None,
                        help="Optional run name. Defaults to small_dcon_cost_<timestamp>.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for log and summary files.")
    parser.add_argument("--no-root-symlinks", action="store_true",
                        help="Do not create /CHAOST2 and /SABSCT symlinks for the original DCON loader.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    return parser.parse_args()


def query_gpu_memory_gb(gpu_id: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                gpu_id,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip().splitlines()[0].strip()) / 1024.0
    except (ValueError, IndexError):
        return None


def ensure_small_dcon_data_links(data_root: Path, small_dcon_root: Path) -> None:
    abdominal_root = data_root / "abdominal"
    for domain in ["CHAOST2", "SABSCT"]:
        src = abdominal_root / domain
        if not src.is_dir():
            raise FileNotFoundError(f"Missing data directory: {src}")
        for dst in [Path("/") / domain, small_dcon_root / domain]:
            if dst.exists():
                continue
            try:
                dst.symlink_to(src, target_is_directory=True)
                print(f"Created symlink: {dst} -> {src}")
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot create {dst}. Original DCON expects both /{domain}/processed "
                    f"and {small_dcon_root / domain}/processed to resolve. "
                    f"Run as root or create the symlink manually."
                ) from exc


def build_command(args: argparse.Namespace, expname: str) -> list[str]:
    return [
        sys.executable, "train.py",
        "--expname", expname,
        "--phase", "train",
        "--gpu_ids", args.gpu,
        "--f_seed", str(args.seed),
        "--f_determin", "1",
        "--lr", "0.0005",
        "--model", "unet",
        "--batchSize", str(args.batch_size),
        "--all_epoch", str(args.epochs),
        "--validation_freq", "999999",
        "--testfreq", "999999",
        "--display_freq", "999999",
        "--data_name", "ABDOMINAL",
        "--nclass", "5",
        "--tr_domain", args.source,
        "--consist_f", "1",
        "--contrast_f", "1",
        "--fmethod", "asymr",
        "--save_prediction", "False",
        "--w_ce", "1.0",
        "--w_dice", "1.0",
        "--w_seg", "1.0",
        "--w_consist", "1.0",
        "--w_contrast", "1.0",
        "--num_augs1", "6",
        "--augflag1", "False",
        "--f_dropout1", "0",
        "--dropout_rate1", "0.0",
        "--f_dropout2", "1",
        "--dropout_rate2", "0.5",
        "--gls_nlayer", "4",
        "--gls_interm", "2",
        "--gls_outnorm", "frob",
        "--glsmix_f", "1",
        "--mixalpha", "0.2",
        "--temperature", "0.05",
        "--n_view", "10",
    ]


def start_gpu_monitor(gpu_id: str, stop_event: threading.Event, peak_holder: dict[str, float | None]) -> threading.Thread:
    def monitor() -> None:
        while not stop_event.is_set():
            current = query_gpu_memory_gb(gpu_id)
            if current is not None:
                peak = peak_holder.get("peak")
                peak_holder["peak"] = current if peak is None else max(peak, current)
            time.sleep(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread


def run_and_capture(cmd: list[str], cwd: Path, log_path: Path, gpu_id: str, dry_run: bool) -> float | None:
    print(f"cwd={cwd}")
    print("Command:", " ".join(cmd))
    if dry_run:
        return None

    stop_event = threading.Event()
    peak_holder: dict[str, float | None] = {"peak": query_gpu_memory_gb(gpu_id)}
    monitor_thread = start_gpu_monitor(gpu_id, stop_event, peak_holder)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = proc.wait()

    stop_event.set()
    monitor_thread.join(timeout=2.0)
    current = query_gpu_memory_gb(gpu_id)
    if current is not None:
        peak = peak_holder.get("peak")
        peak_holder["peak"] = current if peak is None else max(peak, current)

    if return_code != 0:
        raise RuntimeError(f"Small DCON failed with exit code {return_code}. See log: {log_path}")
    return peak_holder.get("peak")


def parse_epoch_times(log_path: Path) -> list[tuple[int, float]]:
    pattern = re.compile(r"End of epoch\s+(\d+)\s*/\s*\d+\s+Time Taken:\s*([0-9.]+)\s+sec")
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows.append((int(match.group(1)), float(match.group(2))))
    if not rows:
        raise RuntimeError(f"Could not parse epoch times from {log_path}")
    return rows


def write_outputs(
    rows: list[tuple[int, float]],
    peak_mem_gb: float | None,
    warmup_epochs: int,
    out_dir: Path,
    log_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_csv = out_dir / "cost_profile.csv"
    with profile_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "epoch", "train_time_sec", "peak_mem_gb"])
        for epoch, train_time_sec in rows:
            writer.writerow(["DCON", epoch, f"{train_time_sec:.6f}", f"{(peak_mem_gb or 0.0):.6f}"])

    kept = [(epoch, sec) for epoch, sec in rows if epoch > warmup_epochs]
    if not kept:
        kept = rows
    mean_time = sum(sec for _, sec in kept) / len(kept)
    peak_mem = peak_mem_gb or 0.0

    summary_csv = out_dir / "cost_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Train time / epoch (s)", "GPU memory (GB)", "Inference cost", "Profile CSV", "Log"])
        writer.writerow(["DCON", f"{mean_time:.6f}", f"{peak_mem:.6f}", "1.0x", profile_csv, log_path])

    summary_md = out_dir / "cost_summary.md"
    summary_md.write_text(
        "| Method | Train time / epoch | GPU memory | Inference cost |\n"
        "|---|---:|---:|---:|\n"
        f"| DCON | {mean_time:.2f}s | {peak_mem:.2f} GB | 1.0x |\n"
    )

    print(f"\nWrote:\n  {profile_csv}\n  {summary_csv}\n  {summary_md}")
    print(summary_md.read_text())


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    small_dcon_root = project_root / "DCON"
    run_name = args.run_name or f"small_dcon_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else project_root / "cost_profiles" / run_name
    data_root = Path(args.data_root).resolve() if args.data_root else project_root / "data"

    if not args.no_root_symlinks:
        ensure_small_dcon_data_links(data_root, small_dcon_root)

    log_path = out_dir / "dcon.log"
    cmd = build_command(args, run_name)
    peak_mem_gb = run_and_capture(cmd, small_dcon_root, log_path, args.gpu, args.dry_run)
    if args.dry_run:
        return

    rows = parse_epoch_times(log_path)
    write_outputs(rows, peak_mem_gb, args.warmup_epochs, out_dir, log_path)


if __name__ == "__main__":
    main()
