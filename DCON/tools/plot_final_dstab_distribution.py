#!/usr/bin/env python3
"""Final-checkpoint d_stab distribution visualization for w/ vs w/o CGSD."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models.saam import StabilityAwareAlignmentModule
from tools.analyze_dstab import (
    CheckpointSpec,
    DATASET_NCLASS,
    build_dataset,
    build_views_from_record,
    forward_triplet,
    load_model,
    mean_foreground_dice,
    safe_nanmean,
    set_seed,
    target_for_dataset,
)


VARIANTS = [
    ("w/o CGSD", False),
    ("w/ CGSD", True),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot final-checkpoint d_stab distribution and unstable ratio."
    )
    parser.add_argument("--full_ckpt", type=Path, default=None,
                        help="Final w/ CGSD checkpoint path.")
    parser.add_argument("--wo_cgsd_ckpt", type=Path, default=None,
                        help="Final w/o CGSD checkpoint path.")
    parser.add_argument("--full_ckpt_template", type=str, default=None,
                        help="w/ CGSD template containing {epoch}.")
    parser.add_argument("--wo_cgsd_ckpt_template", type=str, default=None,
                        help="w/o CGSD template containing {epoch}.")
    parser.add_argument("--epoch", type=int, default=300,
                        help="Final epoch used when checkpoint templates are passed.")
    parser.add_argument("--data_name", required=True, choices=["CARDIAC", "ABDOMINAL"])
    parser.add_argument("--tr_domain", "--source", dest="tr_domain", required=True)
    parser.add_argument("--target_domain", "--target", dest="target_domain", default=None)
    parser.add_argument("--split", default="target_test",
                        choices=["target_test", "target_trtest", "target_trval",
                                 "source_trtest", "source_trval", "source_train"])
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results_dstab_distribution")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--max_slices", type=int, default=0,
                        help="0 means analyze the whole selected split.")
    parser.add_argument("--max_plot_values", type=int, default=200000,
                        help="Per-variant random subsample size for violin plotting; summary uses all values.")
    parser.add_argument("--boxplot_clip_percentile", type=float, default=99.0,
                        help="x-axis upper limit percentile computed from combined d_stab values.")
    parser.add_argument("--flier_size", type=float, default=0.5)
    parser.add_argument("--flier_alpha", type=float, default=0.15)
    parser.add_argument("--hide_fliers", action="store_true",
                        help="Hide boxplot outliers; max remains in dstab_summary.csv.")
    parser.add_argument("--unstable_ylim", type=float, default=0.4,
                        help="Y-axis upper limit for unstable_ratio_bar. Use <=0 for auto.")
    parser.add_argument("--save_values", action="store_true", default=True,
                        help="Save full d_stab arrays for no-model redraws.")
    parser.add_argument("--dstab_tau", type=float, default=None,
                        help="Fixed tau for unstable ratio.")
    parser.add_argument("--dstab_tau_percentile", type=float, default=75.0,
                        help="Combined percentile used when --dstab_tau is unset.")
    parser.add_argument("--nclass", type=int, default=None)
    parser.add_argument("--tile_z_dim", type=int, default=3)
    parser.add_argument("--cgsd_layer", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--use_temperature", type=int, default=0)
    parser.add_argument("--gate_tau", type=float, default=0.1)
    parser.add_argument("--saam_tau", type=float, default=0.5)
    parser.add_argument("--saam_topk", type=float, default=0.3)
    parser.add_argument("--saam_stability_mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an existing output directory.")
    return parser.parse_args()


def unique_dir(path):
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def resolve_ckpts(args):
    if args.full_ckpt_template:
        full = Path(args.full_ckpt_template.format(epoch=args.epoch))
    elif args.full_ckpt:
        full = args.full_ckpt
    else:
        raise ValueError("Pass --full_ckpt or --full_ckpt_template.")

    if args.wo_cgsd_ckpt_template:
        wo = Path(args.wo_cgsd_ckpt_template.format(epoch=args.epoch))
    elif args.wo_cgsd_ckpt:
        wo = args.wo_cgsd_ckpt
    else:
        raise ValueError("Pass --wo_cgsd_ckpt or --wo_cgsd_ckpt_template.")

    if not full.exists():
        raise FileNotFoundError(f"w/ CGSD checkpoint not found: {full}")
    if not wo.exists():
        raise FileNotFoundError(f"w/o CGSD checkpoint not found: {wo}")
    return full, wo


def flatten_valid(tensor):
    values = tensor.detach().reshape(-1).float().cpu()
    return values[torch.isfinite(values)]


def collect_variant_values(variant_name, use_cgsd, ckpt_path, dataset, saam, args, device):
    spec = CheckpointSpec(
        path=Path(ckpt_path),
        variant=variant_name,
        use_cgsd=use_cgsd,
        epoch=str(args.epoch),
    )
    model = load_model(spec, args, device)
    max_slices = args.max_slices if args.max_slices > 0 else len(dataset)
    max_slices = min(max_slices, len(dataset))

    d_stab_chunks = []
    dice_values = []
    num_samples = 0

    with torch.no_grad():
        for index in tqdm(range(max_slices), desc=f"d_stab distribution {variant_name}"):
            counted_sample = False
            for view_id in range(args.num_views):
                anchor, base, strong, label, _record = build_views_from_record(
                    dataset, index, view_id, args
                )
                anchor = anchor.to(device).float()
                base = base.to(device).float()
                strong = strong.to(device).float()
                pred0, encs, _f_str, _f_sty = forward_triplet(
                    model, anchor, base, strong, use_cgsd
                )

                # SAAM stability distance: d_stab is the first tensor returned by compute_stability().
                d_stab, _d_01, _d_02, _d_12 = saam.compute_stability(*encs)
                valid = flatten_valid(d_stab)
                if valid.numel() > 0:
                    d_stab_chunks.append(valid)

                if view_id == 0:
                    label_np = label[0, 0].numpy().astype(np.int64)
                    pred_np = torch.argmax(pred0, dim=1)[0].detach().cpu().numpy()
                    dice_values.append(mean_foreground_dice(pred_np, label_np, args.nclass))
                    counted_sample = True
            if counted_sample:
                num_samples += 1

    if not d_stab_chunks:
        raise RuntimeError(f"No d_stab values collected for {variant_name}.")
    values = torch.cat(d_stab_chunks)
    return {
        "variant": variant_name,
        "checkpoint": str(ckpt_path),
        "values": values,
        "num_samples": num_samples,
        "dice": safe_nanmean(dice_values),
    }


def sample_for_plot(values, max_values, seed):
    if max_values <= 0 or values.numel() <= max_values:
        return values.numpy()
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(values.numel(), generator=generator)[:max_values]
    return values[idx].numpy()


def summarize(values, tau):
    return {
        "num_values": int(values.numel()),
        "mean_d_stab": float(values.mean().item()),
        "median_d_stab": float(values.median().item()),
        "std_d_stab": float(values.std(unbiased=False).item()),
        "q25_d_stab": float(torch.quantile(values, 0.25).item()),
        "q75_d_stab": float(torch.quantile(values, 0.75).item()),
        "max_d_stab": float(values.max().item()),
        "tau": float(tau),
        "unstable_ratio": float((values > tau).float().mean().item()),
    }


def write_summary(path, results, tau):
    fieldnames = [
        "variant", "num_values", "mean_d_stab", "median_d_stab", "std_d_stab",
        "q25_d_stab", "q75_d_stab", "max_d_stab", "tau", "unstable_ratio", "dice",
    ]
    rows = []
    for result in results:
        stats = summarize(result["values"], tau)
        rows.append({
            "variant": result["variant"],
            **stats,
            "dice": result["dice"],
        })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_distribution(path_png, path_pdf, plot_values, summary_rows, tau, x_upper, args):
    labels = [item["variant"] for item in plot_values]
    values = [item["values"] for item in plot_values]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    colors = ["#8da0cb", "#fc8d62"]

    box = ax.boxplot(
        values,
        vert=False,
        labels=labels,
        widths=0.62,
        showfliers=not args.hide_fliers,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"edgecolor": "#4d4d4d", "linewidth": 1.8},
        whiskerprops={"color": "#666666", "linewidth": 1.6},
        capprops={"color": "#666666", "linewidth": 1.6},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#4d4d4d",
            "markeredgecolor": "none",
            "markersize": args.flier_size,
            "alpha": args.flier_alpha,
        },
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    ax.axvline(tau, color="#cc0000", linestyle="--", linewidth=1.4, label=f"tau={tau:.4f}")
    ax.set_xlim(0.0, x_upper)
    ax.set_xlabel("d_stab")
    ax.set_ylabel("Variant")
    ax.set_title("SAAM d_stab Distribution at Final Checkpoint")
    ax.text(
        0.01,
        0.02,
        f"x-axis clipped at P{args.boxplot_clip_percentile:g} for visualization",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    text_x = x_upper * 0.97
    for ypos, row in enumerate(summary_rows, start=1):
        ax.text(
            text_x,
            ypos + 0.31,
            f"mean={float(row['mean_d_stab']):.4f}\nunstable={float(row['unstable_ratio']):.3f}",
            ha="right",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )

    fig.savefig(path_png, dpi=200)
    fig.savefig(path_pdf)
    plt.close(fig)


def plot_unstable_bar(path_png, path_pdf, summary_rows, unstable_ylim):
    labels = [row["variant"] for row in summary_rows]
    values = [float(row["unstable_ratio"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#8da0cb", "#fc8d62"], edgecolor="#222222")
    if unstable_ylim and unstable_ylim > 0:
        ax.set_ylim(0.0, unstable_ylim)
    else:
        ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    ax.set_ylabel("unstable ratio")
    ax.set_title("Unstable Ratio at Final Checkpoint")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center")
    fig.savefig(path_png, dpi=200)
    fig.savefig(path_pdf)
    plt.close(fig)


def main():
    args = parse_args()
    args.nclass = args.nclass or DATASET_NCLASS[args.data_name]
    args.target_domain = target_for_dataset(args.data_name, args.tr_domain, args.target_domain)
    if args.num_views < 1:
        raise ValueError("--num_views must be >= 1")
    if args.dstab_tau is not None and not math.isfinite(args.dstab_tau):
        raise ValueError("--dstab_tau must be finite.")
    if args.dstab_tau is None and not (0.0 < args.dstab_tau_percentile < 100.0):
        raise ValueError("--dstab_tau_percentile must be in (0, 100).")
    if not (0.0 < args.boxplot_clip_percentile <= 100.0):
        raise ValueError("--boxplot_clip_percentile must be in (0, 100].")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    set_seed(args.seed)
    full_ckpt, wo_ckpt = resolve_ckpts(args)

    task = f"{args.data_name}_{args.tr_domain}_to_{args.target_domain}"
    out_dir = args.out_dir / task
    if not args.overwrite:
        out_dir = unique_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    saam = StabilityAwareAlignmentModule(
        tau=args.saam_tau,
        topk_ratio=args.saam_topk,
        stability_mode=args.saam_stability_mode,
    ).to(device).eval()

    ckpts = {
        "w/o CGSD": wo_ckpt,
        "w/ CGSD": full_ckpt,
    }
    results = [
        collect_variant_values(name, use_cgsd, ckpts[name], dataset, saam, args, device)
        for name, use_cgsd in VARIANTS
    ]
    combined = torch.cat([result["values"] for result in results])
    x_upper = float(torch.quantile(combined, args.boxplot_clip_percentile / 100.0).item())
    if not math.isfinite(x_upper) or x_upper <= 0:
        x_upper = float(combined.max().item())
    if args.dstab_tau is not None:
        tau = float(args.dstab_tau)
        tau_mode = "fixed"
    else:
        tau = float(torch.quantile(combined, args.dstab_tau_percentile / 100.0).item())
        tau_mode = f"combined_p{args.dstab_tau_percentile:g}"

    summary_csv = out_dir / "dstab_summary.csv"
    summary_rows = write_summary(summary_csv, results, tau)
    if args.save_values:
        np.savez_compressed(
            out_dir / "dstab_values.npz",
            **{result["variant"]: result["values"].numpy() for result in results},
        )

    plot_values = [
        {
            "variant": result["variant"],
            "values": sample_for_plot(result["values"], args.max_plot_values, args.seed + idx),
        }
        for idx, result in enumerate(results)
    ]
    plot_distribution(
        out_dir / "dstab_distribution.png",
        out_dir / "dstab_distribution.pdf",
        plot_values,
        summary_rows,
        tau,
        x_upper,
        args,
    )
    plot_unstable_bar(
        out_dir / "unstable_ratio_bar.png",
        out_dir / "unstable_ratio_bar.pdf",
        summary_rows,
        args.unstable_ylim,
    )

    log_text = "\n".join([
        f"task: {task}",
        f"split: {args.split}",
        f"epoch: {args.epoch}",
        f"num_views: {args.num_views}",
        f"tau_mode: {tau_mode}",
        f"tau: {tau:.8f}",
        "d_stab source: StabilityAwareAlignmentModule.compute_stability(...)[0]",
        f"summary_csv: {summary_csv}",
    ])
    (out_dir / "dstab_analysis.log").write_text(log_text + "\n", encoding="utf-8")

    print(f"[OK] Saved d_stab distribution to: {out_dir / 'dstab_distribution.png'}")
    print(f"[OK] Saved unstable ratio bar to: {out_dir / 'unstable_ratio_bar.png'}")
    print(f"[OK] Saved d_stab summary to: {summary_csv}")
    print(f"[INFO] tau used for unstable_ratio: {tau:.8f} ({tau_mode})")


if __name__ == "__main__":
    main()
