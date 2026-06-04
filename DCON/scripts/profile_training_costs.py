#!/usr/bin/env python3
"""Run one-shot ERM/DCON/SAA training-cost profiling and summarize the table.

The script launches short training-only profiling runs. Each training entry
records per-epoch wall time and peak CUDA memory into cost_profile.csv.
Validation, checkpoint saving, and final testing are disabled by --profile_cost.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile ERM, DCON, and SAA training cost.")
    parser.add_argument("--dataset", choices=["ABDOMINAL"], default="ABDOMINAL",
                        help="Dataset to profile. The bundled original DCON entry supports ABDOMINAL.")
    parser.add_argument("--source", default="CHAOST2", help="Source training domain.")
    parser.add_argument("--gpu", default="0", help="GPU id passed to the training scripts.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of profiling epochs per method.")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Drop this many initial epochs when averaging time.")
    parser.add_argument("--batch-size", type=int, default=20, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="SAA/ERM DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--methods", nargs="+", default=["ERM", "DCON", "SAA"],
                        choices=["ERM", "DCON", "SAA"], help="Methods to run.")
    parser.add_argument("--run-name", default=None,
                        help="Optional run name. Defaults to a timestamped cost_profile_* name.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for summary files and subprocess logs.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse an existing cost_profile.csv when present.")
    parser.add_argument("--gpu-poll-interval", type=float, default=0.5,
                        help="Seconds between nvidia-smi GPU-memory samples.")
    parser.add_argument("--stream-log", action="store_true",
                        help="Stream each training subprocess log to the terminal while also saving it.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def dataset_config(dataset: str) -> dict[str, str | int]:
    if dataset != "ABDOMINAL":
        raise ValueError("Only ABDOMINAL is supported because the bundled original DCON script imports AbdominalDataset.")
    return {"nclass": 5, "sgf_grid_size": 3}


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
        first_value = result.stdout.strip().splitlines()[0].strip()
        return float(first_value) / 1024.0
    except (ValueError, IndexError):
        return None


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
    gpu_id: str,
    poll_interval: float,
    stream_log: bool,
) -> float | None:
    printable = " ".join(cmd)
    print(f"\n[run] cwd={cwd}")
    print(printable)
    if dry_run:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    peak_holder: dict[str, float | None] = {"peak": query_gpu_memory_gb(gpu_id)}

    def monitor_gpu() -> None:
        while not stop_event.is_set():
            current_mem_gb = query_gpu_memory_gb(gpu_id)
            if current_mem_gb is not None:
                peak = peak_holder["peak"]
                peak_holder["peak"] = current_mem_gb if peak is None else max(peak, current_mem_gb)
            time.sleep(max(poll_interval, 0.1))

    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    with log_path.open("w") as log_file:
        if stream_log:
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
            proc.wait()
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc.wait()

    stop_event.set()
    monitor_thread.join(timeout=2.0)
    current_mem_gb = query_gpu_memory_gb(gpu_id)
    if current_mem_gb is not None:
        peak = peak_holder["peak"]
        peak_holder["peak"] = current_mem_gb if peak is None else max(peak, current_mem_gb)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. See log: {log_path}")
    return peak_holder["peak"]


def profile_csv_path(root: Path, source: str, expname: str) -> Path:
    return root / "ckpts" / source / expname / "log" / "cost_profile.csv"


def build_erm_command(root: Path, args: argparse.Namespace, expname: str, cfg: dict[str, str | int]) -> list[str]:
    return [
        sys.executable, "train.py",
        "--profile_cost", "1",
        "--profile_method", "ERM",
        "--erm_only", "1",
        "--use_sgf", "0",
        "--use_cgsd", "0",
        "--use_projector", "0",
        "--use_saam", "0",
        "--use_rccs", "0",
        "--num_workers", str(args.num_workers),
        "--expname", expname,
        "--phase", "train",
        "--ckpt_dir", str(root / "ckpts"),
        "--gpu_ids", args.gpu,
        "--f_seed", str(args.seed),
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
    ]


def build_saa_command(root: Path, args: argparse.Namespace, expname: str, cfg: dict[str, str | int]) -> list[str]:
    return [
        sys.executable, "train.py",
        "--profile_cost", "1",
        "--profile_method", "SAA",
        "--use_sgf", "1",
        "--sgf_grid_size", str(cfg["sgf_grid_size"]),
        "--num_workers", str(args.num_workers),
        "--expname", expname,
        "--phase", "train",
        "--ckpt_dir", str(root / "ckpts"),
        "--gpu_ids", args.gpu,
        "--f_seed", str(args.seed),
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
        "--use_cgsd", "1",
        "--cgsd_layer", "1",
        "--use_projector", "1",
        "--use_separate_cgsd_optimizer", "1",
        "--lambda_str", "0.3",
        "--lambda_sty", "0.3",
        "--use_saam", "1",
        "--saam_tau", "0.5",
        "--saam_topk", "0.3",
        "--saam_stability_mode", "mean",
        "--lambda_01", "1.0",
        "--lambda_02", "1.0",
        "--saam_warmup_epochs", "50",
        "--saam_rampup_epochs", "100",
        "--anchor_seg_alpha", "0.0",
        "--strong_seg_alpha", "1.0",
        "--use_rccs", "1",
        "--p_rccs", "0.3",
        "--rccs_candidates", "4",
        "--rccs_metric", "cos",
        "--rccs_embed_dim", "128",
    ]


def build_dcon_command(args: argparse.Namespace, expname: str, cfg: dict[str, str | int]) -> list[str]:
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
        "--data_name", args.dataset,
        "--nclass", str(cfg["nclass"]),
        "--tr_domain", args.source,
        "--consist_f", "1",
        "--contrast_f", "1",
        "--fmethod", "asymr",
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


def write_dcon_profile_from_log(log_path: Path, profile_path: Path, peak_mem_gb: float | None) -> None:
    pattern = re.compile(r"End of epoch\s+(\d+)\s*/\s*\d+\s+Time Taken:\s*([0-9.]+)\s+sec")
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({
                "method": "DCON",
                "epoch": int(match.group(1)),
                "train_time_sec": float(match.group(2)),
                "peak_mem_gb": peak_mem_gb if peak_mem_gb is not None else 0.0,
            })
    if not rows:
        raise RuntimeError(f"Could not parse DCON epoch timing from log: {log_path}")

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "epoch", "train_time_sec", "peak_mem_gb"])
        writer.writeheader()
        writer.writerows(rows)


def summarize_profile(
    path: Path,
    method: str,
    warmup_epochs: int,
    external_peak_mem_gb: float | None = None,
) -> dict[str, str | float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing profile CSV for {method}: {path}")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty profile CSV for {method}: {path}")

    last_epoch = max(int(row["epoch"]) for row in rows)
    kept = [
        row for row in rows
        if int(row["epoch"]) > warmup_epochs and (len(rows) <= warmup_epochs + 1 or int(row["epoch"]) < last_epoch)
    ]
    if not kept:
        kept = rows

    times = [float(row["train_time_sec"]) for row in kept]
    mems = [float(row["peak_mem_gb"]) for row in kept]
    return {
        "Method": method,
        "Train time / epoch (s)": sum(times) / len(times),
        "GPU memory (GB)": external_peak_mem_gb if external_peak_mem_gb is not None else max(mems),
        "Inference cost": "1.0x",
        "Profile CSV": str(path),
    }


def write_summary(rows: list[dict[str, str | float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cost_summary.csv"
    md_path = out_dir / "cost_summary.md"

    with csv_path.open("w", newline="") as f:
        fieldnames = ["Method", "Train time / epoch (s)", "GPU memory (GB)", "Inference cost", "Profile CSV"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with md_path.open("w") as f:
        f.write("| Method | Train time / epoch | GPU memory | Inference cost |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['Method']} | {row['Train time / epoch (s)']:.2f}s | "
                f"{row['GPU memory (GB)']:.2f} GB | {row['Inference cost']} |\n"
            )

    print(f"\nSummary written to:\n  {csv_path}\n  {md_path}")
    print(md_path.read_text())


def main() -> None:
    args = parse_args()
    cfg = dataset_config(args.dataset)

    root = Path(__file__).resolve().parents[1]
    original_dcon_root = root / "DCON"
    run_name = args.run_name or f"cost_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "cost_profiles" / run_name

    method_specs = {
        "ERM": {
            "cwd": root,
            "expname": f"{run_name}_erm_{args.source}",
            "command": lambda exp: build_erm_command(root, args, exp, cfg),
        },
        "DCON": {
            "cwd": original_dcon_root,
            "expname": f"{run_name}_dcon_{args.source}",
            "command": lambda exp: build_dcon_command(args, exp, cfg),
        },
        "SAA": {
            "cwd": root,
            "expname": f"{run_name}_saa_{args.source}",
            "command": lambda exp: build_saa_command(root, args, exp, cfg),
        },
    }

    summaries = []
    monitor_peaks = {}
    for method in args.methods:
        spec = method_specs[method]
        expname = spec["expname"]
        profile_path = profile_csv_path(root, args.source, expname)
        if args.skip_existing and profile_path.exists():
            print(f"\n[skip] Reusing existing {method} profile: {profile_path}")
        else:
            cmd = spec["command"](expname)
            log_path = out_dir / f"{method.lower()}.log"
            monitor_peaks[method] = run_command(
                cmd,
                Path(spec["cwd"]),
                log_path,
                args.dry_run,
                args.gpu,
                args.gpu_poll_interval,
                args.stream_log,
            )
            if method == "DCON" and not args.dry_run:
                write_dcon_profile_from_log(log_path, profile_path, monitor_peaks[method])
        if not args.dry_run:
            summaries.append(summarize_profile(
                profile_path,
                method,
                args.warmup_epochs,
                external_peak_mem_gb=monitor_peaks.get(method),
            ))

    if not args.dry_run:
        write_summary(summaries, out_dir)


if __name__ == "__main__":
    main()
