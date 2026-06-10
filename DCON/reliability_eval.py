#!/usr/bin/env python
"""Test-time reliability estimation for DCON medical segmentation."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import dataloaders.AbdominalDataset as ABD
import dataloaders.CardiacDataset as CARD
from metrics_reliability import (
    binary_detection_metrics,
    case_risk_coverage,
    correlation_metrics,
    foreground_selective_dice,
    mean_foreground_dice,
    pixel_selective_curve,
    safe_minmax,
    selective_dice,
)
from models.exp_trainer import Train_process
from models.saam import StabilityAwareAlignmentModule
from tent_baseline import EpisodicTent


EPS = 1e-8


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_opt(args, tta="none"):
    """Minimal option namespace accepted by DCON Train_process and datasets."""
    return SimpleNamespace(
        expname=args.expname,
        phase="test",
        ckpt_dir=os.path.join(THIS_DIR, "ckpts"),
        resume_path=args.resume_path,
        gpu_ids=args.gpu_ids,
        f_seed=args.seed,
        lr=args.lr,
        model="unet",
        backbone=args.backbone,
        batchSize=args.batch_size,
        data_name=args.data_name,
        nclass=args.nclass,
        tr_domain=args.tr_domain,
        target_domain=args.target_domain,
        save_prediction=False,
        eval_source_domain=False,
        tta=tta,
        source_access=False,
        lambda_source=0.0,
        tent_lr=args.tent_lr,
        tent_steps=args.tent_steps,
        use_cgsd=args.use_cgsd,
        use_projector=args.use_projector,
        cgsd_layer=args.cgsd_layer,
        proj_dim=args.proj_dim,
        proj_hidden_channels=args.proj_hidden_channels,
        use_separate_cgsd_optimizer=0,
        use_temperature=args.use_temperature,
        gate_tau=args.gate_tau,
        use_saam=args.use_saam,
        saam_tau=args.saam_tau,
        saam_topk=args.saam_topk,
        saam_stability_mode=args.saam_stability_mode,
        saam_weight_type="stability",
        align_weight_type="stability",
        saam_mask_ablation="w_times_m",
        uncertainty_tau=0.5,
        uncertainty_view_mode="anchor_only",
        use_rccs=0,
        rccs_select="none",
        use_sgf=0,
        sgf_view2_only=0,
        local_aug_type="none",
        clp_alpha_min=0.75,
        clp_alpha_max=1.25,
        clp_beta_min=-0.15,
        clp_beta_max=0.15,
        clp_perturb_background=1,
        clp_seed=None,
        w_dice=1.0,
        w_ce=1.0,
        w_seg=1.0,
        quiet_console=True,
    )


def infer_target(data_name, tr_domain, explicit=None):
    if explicit is not None:
        return explicit
    if data_name == "CARDIAC":
        return "bSSFP" if tr_domain == "LGE" else "LGE"
    if data_name == "ABDOMINAL":
        return "CHAOST2" if tr_domain == "SABSCT" else "SABSCT"
    raise ValueError(f"Unsupported data_name={data_name}")


def build_dataset(args, opt):
    target = infer_target(args.data_name, args.tr_domain, args.target_domain)
    if args.data_name == "CARDIAC":
        label_names = CARD.LABEL_NAME
        dataset = CARD.get_test(modality=[target], opt=opt)
    elif args.data_name == "ABDOMINAL":
        label_names = ABD.LABEL_NAME
        dataset = ABD.get_test(modality=[target], norm_func=None, opt=opt)
    else:
        raise ValueError(f"Unsupported data_name={args.data_name}")
    return dataset, label_names, target


def forward_logits_and_feature(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        logits = output[0]
        feature = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else None
        return logits, feature
    return output, None


def entropy_unreliability(prob):
    class_count = max(int(prob.shape[1]), 2)
    return (-(prob.clamp_min(EPS) * prob.clamp_min(EPS).log()).sum(1) / np.log(class_count)).clamp(0.0, 1.0)


def probability_scores(logits):
    prob = F.softmax(logits, dim=1)
    entropy = entropy_unreliability(prob)
    confidence = prob.max(dim=1)[0]
    top2 = torch.topk(prob, k=min(2, prob.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    return {
        "entropy": entropy,
        "confidence": (1.0 - confidence).clamp(0.0, 1.0),
        "margin": (1.0 - margin).clamp(0.0, 1.0),
    }


def augmented_logits(model, images, num_views=5, noise_std=0.01, intensity_jitter=0.05):
    views = []
    logits0, _ = forward_logits_and_feature(model, images)
    views.append(logits0)
    if len(views) < num_views:
        logits, _ = forward_logits_and_feature(model, images.flip(dims=(3,)))
        views.append(logits.flip(dims=(3,)))
    if len(views) < num_views:
        logits, _ = forward_logits_and_feature(model, images.flip(dims=(2,)))
        views.append(logits.flip(dims=(2,)))
    signs = [1.0, -1.0]
    idx = 0
    while len(views) < num_views:
        if idx % 2 == 0:
            view = images + float(noise_std) * torch.randn_like(images)
        else:
            view = images * (1.0 + signs[idx % len(signs)] * float(intensity_jitter))
        logits, _ = forward_logits_and_feature(model, view)
        views.append(logits)
        idx += 1
    return torch.stack(views, dim=0)


def tta_uncertainty_scores(model, images, num_views=5, noise_std=0.01, intensity_jitter=0.05):
    logits_stack = augmented_logits(model, images, num_views=num_views, noise_std=noise_std, intensity_jitter=intensity_jitter)
    prob_stack = F.softmax(logits_stack, dim=2)
    mean_prob = prob_stack.mean(dim=0)
    variance = prob_stack.var(dim=0, unbiased=False).mean(dim=1)
    pred_stack = prob_stack.argmax(dim=2)
    mode = torch.mode(pred_stack, dim=0).values
    disagreement = (pred_stack != mode.unsqueeze(0)).float().mean(dim=0)
    mean_entropy = entropy_unreliability(mean_prob)
    return {
        "tta_variance": safe_tensor_minmax(variance),
        "tta_disagreement": disagreement.clamp(0.0, 1.0),
        "tta_entropy": mean_entropy,
    }


def safe_tensor_minmax(x):
    flat = x.detach()
    lo = flat.amin(dim=(-2, -1), keepdim=True)
    hi = flat.amax(dim=(-2, -1), keepdim=True)
    return ((flat - lo) / (hi - lo).clamp_min(EPS)).clamp(0.0, 1.0)


def ours_instability_score(model, images, args):
    module = StabilityAwareAlignmentModule(
        tau=args.saam_tau,
        topk_ratio=args.saam_topk,
        stability_mode=args.saam_stability_mode,
    ).cuda()
    _, f0 = forward_logits_and_feature(model, images)
    _, f1 = forward_logits_and_feature(model, images.flip(dims=(3,)))
    _, f2 = forward_logits_and_feature(model, images + float(args.tta_noise_std) * torch.randn_like(images))
    if f0 is None or f1 is None or f2 is None:
        return None
    f1 = f1.flip(dims=(3,))
    d_stab, _, _, _ = module.compute_stability(f0, f1, f2)
    d_up = F.interpolate(d_stab.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    return safe_tensor_minmax(d_up)


def tensor_to_numpy_2d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def normalize_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < EPS:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def label_uint8(arr, num_classes):
    arr = np.asarray(arr, dtype=np.int64)
    if num_classes <= 1:
        return normalize_uint8(arr)
    return np.clip(arr * int(255 / max(num_classes - 1, 1)), 0, 255).astype(np.uint8)


def save_png(path, arr, labels=False, num_classes=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = label_uint8(arr, num_classes) if labels else normalize_uint8(arr)
    Image.fromarray(img, mode="L").save(path)


def save_visuals(out_dir, scan_id, z_id, image, gt, pred, maps, num_classes):
    prefix = os.path.join(out_dir, "visualizations", str(scan_id), f"z{int(z_id):04d}")
    save_png(prefix + "_image.png", image)
    save_png(prefix + "_gt.png", gt, labels=True, num_classes=num_classes)
    save_png(prefix + "_pred.png", pred, labels=True, num_classes=num_classes)
    save_png(prefix + "_error.png", (pred != gt).astype(np.float32))
    key_to_name = {
        "entropy": "entropy",
        "confidence": "confidence",
        "tta_variance": "tta_variance",
        "ours_instability": "ours_instability",
    }
    for key, name in key_to_name.items():
        if key in maps:
            arr = 1.0 - maps[key] if key == "confidence" else maps[key]
            save_png(prefix + f"_{name}.png", arr)


class PixelSampler:
    def __init__(self, max_samples, seed):
        self.max_samples = int(max_samples)
        self.rng = np.random.default_rng(seed)
        self.labels = defaultdict(list)
        self.scores = defaultdict(list)

    def add(self, method, labels, scores):
        labels = np.asarray(labels, dtype=np.uint8).ravel()
        scores = np.asarray(scores, dtype=np.float32).ravel()
        if labels.size == 0:
            return
        if self.max_samples > 0 and labels.size > self.max_samples:
            idx = self.rng.choice(labels.size, size=self.max_samples, replace=False)
            labels = labels[idx]
            scores = scores[idx]
        self.labels[method].append(labels)
        self.scores[method].append(scores)

    def metrics(self):
        out = {}
        for method in self.scores:
            labels = np.concatenate(self.labels[method], axis=0)
            scores = np.concatenate(self.scores[method], axis=0)
            if self.max_samples > 0 and labels.size > self.max_samples:
                idx = self.rng.choice(labels.size, size=self.max_samples, replace=False)
                labels = labels[idx]
                scores = scores[idx]
            out[method] = binary_detection_metrics(labels, scores)
        return out


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def finalize_scan(scan, args, state):
    scan_id = scan["scan_id"]
    gt = np.stack(scan["gt"], axis=0).astype(np.int64)
    pred = np.stack(scan["pred"], axis=0).astype(np.int64)
    image = np.stack(scan["image"], axis=0).astype(np.float32)
    dice = mean_foreground_dice(pred, gt, args.nclass)
    error = (pred != gt).astype(np.uint8)
    state["case_dice"][scan_id] = dice

    if scan.get("tent_pred"):
        tent_pred = np.stack(scan["tent_pred"], axis=0).astype(np.int64)
        tent_dice = mean_foreground_dice(tent_pred, gt, args.nclass)
        state["tent_rows"].append(
            {
                "task": state["task"],
                "case_id": scan_id,
                "before_dice": dice,
                "after_tent_dice": tent_dice,
                "delta_dice": tent_dice - dice,
                "tent_hurt": int(tent_dice < dice),
                "ours_instability_mean": float(np.mean(np.stack(scan["maps"].get("ours_instability", [np.zeros_like(gt[0])]), axis=0))),
            }
        )

    for method, maps in scan["maps"].items():
        score = np.stack(maps, axis=0).astype(np.float32)
        score = np.clip(score, 0.0, 1.0)
        reliability = float(1.0 - np.mean(score))
        state["case_rows"].append(
            {
                "row_type": "case",
                "task": state["task"],
                "case_id": scan_id,
                "method": method,
                "dice": dice,
                "reliability": reliability,
                "unreliability": float(np.mean(score)),
                "spearman": "",
                "pearson": "",
                "mae": "",
            }
        )
        state["case_method_values"][method].append((reliability, dice))
        state["pixel_sampler"].add(method, error, score)

        for drop in args.selective_drop:
            keep = score <= np.quantile(score, 1.0 - float(drop))
            sel_dice = selective_dice(pred, gt, keep, num_classes=args.nclass)
            fg_dice = foreground_selective_dice(pred, gt, keep, num_classes=args.nclass)
            state["risk_rows"].append(
                {
                    "task": state["task"],
                    "method": method,
                    "scope": "pixel_selective",
                    "coverage": 1.0 - float(drop),
                    "risk": 1.0 - sel_dice if np.isfinite(sel_dice) else "",
                    "dice": sel_dice,
                    "foreground_dice": fg_dice,
                    "aurc": "",
                    "eaurc": "",
                    "drop_fraction": float(drop),
                }
            )

        reliability_map = 1.0 - score
        for row in pixel_selective_curve(pred, gt, reliability_map, args.nclass, args.pixel_coverages):
            state["risk_rows"].append(
                {
                    "task": state["task"],
                    "method": method,
                    "scope": "pixel_risk_coverage",
                    "coverage": row["coverage"],
                    "risk": row["risk"],
                    "dice": row["dice"],
                    "foreground_dice": row["foreground_dice"],
                    "aurc": "",
                    "eaurc": "",
                    "drop_fraction": "",
                }
            )

    if state["viz_saved"] < args.max_viz or args.max_viz < 0:
        candidates = np.where(gt.reshape(gt.shape[0], -1).sum(axis=1) > 0)[0]
        z = int(candidates[len(candidates) // 2]) if candidates.size else gt.shape[0] // 2
        viz_maps = {name: np.stack(maps, axis=0)[z] for name, maps in scan["maps"].items()}
        save_visuals(args.output_dir, scan_id, z, image[z], gt[z], pred[z], viz_maps, args.nclass)
        state["viz_saved"] += 1


def summarize_outputs(args, state):
    pixel_rows = []
    for method, metrics in state["pixel_sampler"].metrics().items():
        pixel_rows.append(
            {
                "task": state["task"],
                "method": method,
                "auroc": metrics["auroc"],
                "aupr": metrics["aupr"],
                "fpr95": metrics["fpr95"],
                "n_pixels": metrics["n"],
            }
        )

    for method, values in state["case_method_values"].items():
        reliability = [v[0] for v in values]
        dice = [v[1] for v in values]
        corr = correlation_metrics(reliability, dice)
        state["case_rows"].append(
            {
                "row_type": "summary",
                "task": state["task"],
                "case_id": "__summary__",
                "method": method,
                "dice": float(np.mean(dice)) if dice else "",
                "reliability": float(np.mean(reliability)) if reliability else "",
                "unreliability": float(1.0 - np.mean(reliability)) if reliability else "",
                "spearman": corr["spearman"],
                "pearson": corr["pearson"],
                "mae": corr["mae"],
            }
        )
        curve, aurc, eaurc = case_risk_coverage(reliability, dice)
        for row in curve:
            state["risk_rows"].append(
                {
                    "task": state["task"],
                    "method": method,
                    "scope": "case_risk_coverage",
                    "coverage": row["coverage"],
                    "risk": row["risk"],
                    "dice": 1.0 - row["risk"],
                    "foreground_dice": "",
                    "aurc": aurc,
                    "eaurc": eaurc,
                    "drop_fraction": "",
                }
            )

    add_pixel_curve_summaries(state)
    save_risk_coverage_plots(args.output_dir, state["risk_rows"])

    if state["tent_rows"]:
        before = np.asarray([r["before_dice"] for r in state["tent_rows"]], dtype=np.float64)
        after = np.asarray([r["after_tent_dice"] for r in state["tent_rows"]], dtype=np.float64)
        instab = np.asarray([r["ours_instability_mean"] for r in state["tent_rows"]], dtype=np.float64)
        hurt = (after < before).astype(np.float64)
        corr = correlation_metrics(instab, hurt)
        state["tent_rows"].append(
            {
                "task": state["task"],
                "case_id": "__summary__",
                "before_dice": float(np.mean(before)),
                "after_tent_dice": float(np.mean(after)),
                "delta_dice": float(np.mean(after - before)),
                "tent_hurt": int(np.sum(after < before)),
                "ours_instability_mean": float(np.mean(instab)),
                "hurt_instability_spearman": corr["spearman"],
                "hurt_instability_pearson": corr["pearson"],
            }
        )

    write_csv(
        os.path.join(args.output_dir, "per_pixel_metrics.csv"),
        pixel_rows,
        ["task", "method", "auroc", "aupr", "fpr95", "n_pixels"],
    )
    write_csv(
        os.path.join(args.output_dir, "per_case_quality.csv"),
        state["case_rows"],
        ["row_type", "task", "case_id", "method", "dice", "reliability", "unreliability", "spearman", "pearson", "mae"],
    )
    write_csv(
        os.path.join(args.output_dir, "risk_coverage.csv"),
        state["risk_rows"],
        ["task", "method", "scope", "coverage", "risk", "dice", "foreground_dice", "aurc", "eaurc", "drop_fraction"],
    )
    write_csv(
        os.path.join(args.output_dir, "tent_before_after.csv"),
        state["tent_rows"],
        [
            "task",
            "case_id",
            "before_dice",
            "after_tent_dice",
            "delta_dice",
            "tent_hurt",
            "ours_instability_mean",
            "hurt_instability_spearman",
            "hurt_instability_pearson",
        ],
    )


def add_pixel_curve_summaries(state):
    grouped = defaultdict(list)
    for row in state["risk_rows"]:
        if row.get("scope") != "pixel_risk_coverage":
            continue
        try:
            key = (row["method"], float(row["coverage"]))
            grouped[key].append(row)
        except Exception:
            continue

    by_method = defaultdict(list)
    for (method, coverage), rows in grouped.items():
        risks = np.asarray([float(r["risk"]) for r in rows if r.get("risk") != ""], dtype=np.float64)
        dices = np.asarray([float(r["dice"]) for r in rows if r.get("dice") != ""], dtype=np.float64)
        fg = np.asarray([float(r["foreground_dice"]) for r in rows if r.get("foreground_dice") != ""], dtype=np.float64)
        if risks.size == 0:
            continue
        by_method[method].append(
            {
                "coverage": coverage,
                "risk": float(np.nanmean(risks)),
                "dice": float(np.nanmean(dices)) if dices.size else "",
                "foreground_dice": float(np.nanmean(fg)) if fg.size else "",
            }
        )

    for method, rows in by_method.items():
        rows = sorted(rows, key=lambda r: r["coverage"])
        coverages = np.asarray([r["coverage"] for r in rows], dtype=np.float64)
        risks = np.asarray([r["risk"] for r in rows], dtype=np.float64)
        aurc = float(np.trapz(risks, coverages)) if risks.size > 1 else float(risks[0])
        for row in rows:
            state["risk_rows"].append(
                {
                    "task": state["task"],
                    "method": method,
                    "scope": "pixel_risk_coverage_summary",
                    "coverage": row["coverage"],
                    "risk": row["risk"],
                    "dice": row["dice"],
                    "foreground_dice": row["foreground_dice"],
                    "aurc": aurc,
                    "eaurc": "",
                    "drop_fraction": "",
                }
            )


def save_risk_coverage_plots(out_dir, risk_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    for scope in ("case_risk_coverage", "pixel_risk_coverage_summary"):
        grouped = defaultdict(list)
        for row in risk_rows:
            if row.get("scope") != scope:
                continue
            try:
                grouped[row["method"]].append((float(row["coverage"]), float(row["risk"])))
            except Exception:
                continue
        if not grouped:
            continue
        plt.figure(figsize=(7, 5))
        for method, points in sorted(grouped.items()):
            points = sorted(points)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=method)
        plt.xlabel("Coverage")
        plt.ylabel("Risk (1 - Dice)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", f"{scope}.png"), dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str, default="reliability_eval")
    parser.add_argument("--data_name", type=str, choices=["CARDIAC", "ABDOMINAL"], required=True)
    parser.add_argument("--nclass", type=int, required=True)
    parser.add_argument("--tr_domain", type=str, required=True)
    parser.add_argument("--target_domain", type=str, default=None)
    parser.add_argument("--resume_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="unet", choices=["unet", "nnunet", "swinunet"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use_cgsd", type=int, default=0)
    parser.add_argument("--use_projector", type=int, default=0)
    parser.add_argument("--cgsd_layer", type=int, default=1)
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--proj_hidden_channels", type=int, default=8)
    parser.add_argument("--use_temperature", type=int, default=0)
    parser.add_argument("--gate_tau", type=float, default=0.1)
    parser.add_argument("--use_saam", type=int, default=1)
    parser.add_argument("--saam_tau", type=float, default=0.5)
    parser.add_argument("--saam_topk", type=float, default=0.3)
    parser.add_argument("--saam_stability_mode", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--tta_views", type=int, default=5)
    parser.add_argument("--tta_noise_std", type=float, default=0.01)
    parser.add_argument("--tta_intensity_jitter", type=float, default=0.05)
    parser.add_argument("--run_tent", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--tent_lr", type=float, default=1e-4)
    parser.add_argument("--tent_steps", type=int, default=1)
    parser.add_argument("--tent_reset_each_batch", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_viz", type=int, default=40)
    parser.add_argument("--max_pixel_samples", type=int, default=2000000)
    parser.add_argument("--selective_drop", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--pixel_coverages", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if not torch.cuda.is_available():
        raise RuntimeError("This DCON code path calls .cuda(); CUDA is required for reliability_eval.py.")

    args.resume_path = os.path.abspath(args.resume_path)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    opt = build_opt(args, tta="none")
    dataset, label_names, target = build_dataset(args, opt)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    task = f"{args.tr_domain}_to_{target}"

    source_runner = Train_process(opt, reloaddir=args.resume_path, istest=1)
    source_runner.netseg.eval()
    tent_adapter = None
    if args.run_tent:
        tent_opt = build_opt(args, tta="none")
        tent_runner = Train_process(tent_opt, reloaddir=args.resume_path, istest=1)
        tent_adapter = EpisodicTent(
            tent_runner.netseg,
            lr=args.tent_lr,
            steps=args.tent_steps,
            reset_each_batch=args.tent_reset_each_batch,
        )

    state = {
        "task": task,
        "case_dice": {},
        "case_rows": [],
        "case_method_values": defaultdict(list),
        "risk_rows": [],
        "tent_rows": [],
        "pixel_sampler": PixelSampler(args.max_pixel_samples, args.seed),
        "viz_saved": 0,
    }
    scan = None
    completed_cases = 0

    for batch in tqdm(loader, total=len(loader), desc=f"reliability {task}"):
        if scan is None or bool(batch["is_start"]):
            scan = {
                "scan_id": str(batch["scan_id"][0]),
                "gt": [],
                "pred": [],
                "image": [],
                "maps": defaultdict(list),
                "tent_pred": [],
            }

        images = batch["base_view"].float().cuda(non_blocking=True)
        gt = batch["label"].long().cuda(non_blocking=True)
        image_np = tensor_to_numpy_2d(batch["base_view"][0, 1])

        with torch.no_grad():
            logits, _ = forward_logits_and_feature(source_runner.netseg, images)
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)
            score_maps = probability_scores(logits)
            tta_maps = tta_uncertainty_scores(
                source_runner.netseg,
                images,
                num_views=args.tta_views,
                noise_std=args.tta_noise_std,
                intensity_jitter=args.tta_intensity_jitter,
            )
            score_maps.update(tta_maps)
            ours = ours_instability_score(source_runner.netseg, images, args)
            if ours is not None:
                score_maps["ours_instability"] = ours

        if tent_adapter is not None:
            tent_logits = tent_adapter.forward(images)
            if tent_logits.shape[-2:] != gt.shape[-2:]:
                tent_logits = F.interpolate(tent_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            tent_pred = tent_logits.argmax(dim=1)
            tent_scores = probability_scores(tent_logits)
            for name, value in tent_scores.items():
                score_maps[f"tent_{name}"] = value
            scan["tent_pred"].append(tensor_to_numpy_2d(tent_pred))

        gt_np = tensor_to_numpy_2d(gt)
        pred_np = tensor_to_numpy_2d(pred)
        scan["gt"].append(gt_np)
        scan["pred"].append(pred_np)
        scan["image"].append(image_np)
        for name, score in score_maps.items():
            scan["maps"][name].append(tensor_to_numpy_2d(score))

        if bool(batch["is_end"]):
            finalize_scan(scan, args, state)
            completed_cases += 1
            scan = None
            if args.max_cases > 0 and completed_cases >= args.max_cases:
                break

    if scan is not None:
        finalize_scan(scan, args, state)
    summarize_outputs(args, state)
    print(f"Saved reliability outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
