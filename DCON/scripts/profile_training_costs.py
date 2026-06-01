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
import subprocess
import sys
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
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def dataset_config(dataset: str) -> dict[str, str | int]:
    if dataset != "ABDOMINAL":
        raise ValueError("Only ABDOMINAL is supported because the bundled original DCON script imports AbdominalDataset.")
    return {"nclass": 5, "sgf_grid_size": 3}


def run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n[run] cwd={cwd}")
    print(printable)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. See log: {log_path}")


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
        "--profile_cost", "1",
        "--profile_method", "DCON",
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


def summarize_profile(path: Path, method: str, warmup_epochs: int) -> dict[str, str | float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing profile CSV for {method}: {path}")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty profile CSV for {method}: {path}")

    kept = [row for row in rows if int(row["epoch"]) > warmup_epochs]
    if not kept:
        kept = rows

    times = [float(row["train_time_sec"]) for row in kept]
    mems = [float(row["peak_mem_gb"]) for row in kept]
    return {
        "Method": method,
        "Train time / epoch (s)": sum(times) / len(times),
        "GPU memory (GB)": max(mems),
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
    for method in args.methods:
        spec = method_specs[method]
        expname = spec["expname"]
        profile_path = profile_csv_path(root, args.source, expname)
        if args.skip_existing and profile_path.exists():
            print(f"\n[skip] Reusing existing {method} profile: {profile_path}")
        else:
            cmd = spec["command"](expname)
            run_command(cmd, Path(spec["cwd"]), out_dir / f"{method.lower()}.log", args.dry_run)
        if not args.dry_run:
            summaries.append(summarize_profile(profile_path, method, args.warmup_epochs))

    if not args.dry_run:
        write_summary(summaries, out_dir)


if __name__ == "__main__":
    main()
