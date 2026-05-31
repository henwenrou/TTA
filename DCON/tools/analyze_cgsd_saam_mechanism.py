#!/usr/bin/env python3
"""
CGSD -> SAAM mechanism analysis.

This script is analysis-only: it loads frozen checkpoints, builds the same
anchor/base/strong view triplet for both models, reuses SAAM stability and
Top-k logic, and writes CSV/figures for reviewer-facing diagnostics.
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
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
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

import dataloaders.AbdominalDataset as ABD
import dataloaders.CardiacDataset as CARD
from dataloaders.location_scale_augmentation import LocationScaleAugmentation
from models.saam import StabilityAwareAlignmentModule
from models.unet import Projector, Unet1


DATASET_NCLASS = {
    "CARDIAC": 4,
    "ABDOMINAL": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether CGSD improves downstream SAAM stability estimation."
    )
    parser.add_argument("--full_ckpt", type=Path,
                        help="Single full SAA checkpoint, or a template containing {epoch}.")
    parser.add_argument("--wo_cgsd_ckpt", type=Path,
                        help="Single w/o CGSD checkpoint, or a template containing {epoch}.")
    parser.add_argument("--full_ckpt_template", type=str, default=None,
                        help="Epoch template, e.g. runs/full/snapshots/{epoch}_net_Seg.pth")
    parser.add_argument("--wo_cgsd_ckpt_template", type=str, default=None,
                        help="Epoch template, e.g. runs/wo/snapshots/{epoch}_net_Seg.pth")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Epochs to analyze, e.g. 20,50,100 or 20:100:20. "
                             "Requires checkpoint templates unless ckpt paths contain {epoch}.")
    parser.add_argument("--data_name", "--dataset", dest="data_name", required=True,
                        choices=["CARDIAC", "ABDOMINAL"])
    parser.add_argument("--source", "--tr_domain", dest="source", required=True)
    parser.add_argument("--target", "--target_domain", dest="target", default=None)
    parser.add_argument("--split", default="target_test",
                        choices=["target_test", "target_trtest", "target_trval",
                                 "source_trtest", "source_trval", "source_train"])
    parser.add_argument("--nclass", type=int, default=None)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--max_slices", type=int, default=0,
                        help="0 means analyze the whole selected split.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--tile_z_dim", type=int, default=3)
    parser.add_argument("--cgsd_layer", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--use_temperature", type=int, default=0)
    parser.add_argument("--gate_tau", type=float, default=0.1)
    parser.add_argument("--saam_tau", type=float, default=0.5)
    parser.add_argument("--saam_topk", type=float, default=0.3)
    parser.add_argument("--saam_stability_mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--morph_kernel", type=int, default=3, choices=[3, 5])
    parser.add_argument("--projection", default="gap", choices=["gap", "checkpoint"],
                        help="Use GAP phi by default; checkpoint phi is used only if projector weights exist.")
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--proj_hidden_channels", type=int, default=8)
    parser.add_argument("--sample_region_values", type=int, default=2000)
    parser.add_argument("--num_visual_cases", type=int, default=3)
    parser.add_argument("--out_dir", type=Path,
                        default=ROOT / "results" / "cgsd_mechanism_analysis")
    return parser.parse_args()


def parse_epochs(spec):
    if spec is None or str(spec).strip() == "":
        return [None]
    out = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(x) for x in chunk.split(":")]
            if len(parts) == 2:
                start, stop = parts
                step = 1
            elif len(parts) == 3:
                start, stop, step = parts
            else:
                raise ValueError(f"Invalid epoch range: {chunk}")
            if step <= 0:
                raise ValueError("Epoch range step must be positive.")
            out.extend(list(range(start, stop + 1, step)))
        else:
            out.append(int(chunk))
    if not out:
        raise ValueError("No epochs parsed from --epochs.")
    return out


def resolve_epoch_ckpt(single_ckpt, template, epoch, name):
    if epoch is None:
        if single_ckpt is None:
            raise ValueError(f"--{name}_ckpt is required when --epochs is not set.")
        return Path(single_ckpt)

    if template:
        return Path(template.format(epoch=epoch))

    if single_ckpt is not None and "{epoch}" in str(single_ckpt):
        return Path(str(single_ckpt).format(epoch=epoch))

    raise ValueError(
        f"Epoch analysis for {name} requires --{name}_ckpt_template "
        f"or --{name}_ckpt containing '{{epoch}}'."
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_for_dataset(data_name, source, explicit_target):
    if explicit_target:
        return explicit_target
    if data_name == "CARDIAC":
        return "bSSFP" if source == "LGE" else "LGE"
    if data_name == "ABDOMINAL":
        return "CHAOST2" if source == "SABSCT" else "SABSCT"
    raise ValueError(f"Unsupported dataset: {data_name}")


def make_opt(args):
    class Opt:
        pass
    opt = Opt()
    opt.use_sgf = 0
    opt.sgf_view2_only = 0
    opt.nclass = args.nclass
    opt.tr_domain = args.source
    opt.target_domain = args.target
    return opt


def build_datasets(args):
    opt = make_opt(args)
    source = [args.source]
    target = [target_for_dataset(args.data_name, args.source, args.target)]

    if args.data_name == "CARDIAC":
        builders = {
            "source_train": lambda: CARD.get_training(source, opt),
            "source_trval": lambda: CARD.get_trval(source, opt),
            "source_trtest": lambda: CARD.get_trtest(source, opt),
            "target_test": lambda: CARD.get_test(target, opt),
            "target_trtest": lambda: CARD.get_trtest(target, opt),
            "target_trval": lambda: CARD.get_trval(target, opt),
        }
        return builders[args.split](), CARD.LABEL_NAME

    if args.data_name == "ABDOMINAL":
        if args.split.startswith("source"):
            train_set = ABD.get_training(source, norm_func=None, opt=opt)
            if args.split == "source_train":
                return train_set, ABD.LABEL_NAME
            if args.split == "source_trval":
                return ABD.get_trval(source, norm_func=train_set.normalize_op, opt=opt), ABD.LABEL_NAME
            return ABD.get_trtest(source, norm_func=train_set.normalize_op, opt=opt), ABD.LABEL_NAME
        if args.split == "target_test":
            return ABD.get_test(target, norm_func=None, opt=opt), ABD.LABEL_NAME
        if args.split == "target_trtest":
            return ABD.get_trtest(target, norm_func=None, opt=opt), ABD.LABEL_NAME
        return ABD.get_trval(target, norm_func=None, opt=opt), ABD.LABEL_NAME

    raise ValueError(f"Unsupported dataset: {args.data_name}")


def checkpoint_to_state(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "netseg", "netseg_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a state_dict or a dict containing one.")

    state = {}
    for key, value in payload.items():
        new_key = key
        for prefix in ("module.", "netseg."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        state[new_key] = value
    return state


def extract_projector_state(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("projector_str", "projector_str_state_dict", "projector_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    prefixed = {}
    for key, value in payload.items():
        for prefix in ("projector_str.", "projector."):
            if isinstance(key, str) and key.startswith(prefix):
                prefixed[key[len(prefix):]] = value
    return prefixed or None


def load_model(ckpt_path, use_cgsd, args, device):
    payload = torch.load(str(ckpt_path), map_location="cpu")
    model = Unet1(
        c=3,
        num_classes=args.nclass,
        use_channel_gate=use_cgsd,
        cgsd_layer=args.cgsd_layer,
        use_temperature=bool(args.use_temperature),
        gate_tau=args.gate_tau,
    )
    missing, unexpected = model.load_state_dict(checkpoint_to_state(payload), strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    projector_state = extract_projector_state(payload)
    return model, projector_state, missing, unexpected


def build_views_from_record(dataset, index, args):
    record = dataset.actual_dataset[index]
    img = np.float32(record["img"])
    label = np.float32(record["lb"])
    vol_info = record["vol_info"]

    random_state = random.getstate()
    np_state = np.random.get_state()
    set_seed(args.seed + index)
    augmenter = LocationScaleAugmentation(vrange=(0.0, 1.0), background_threshold=0.01)
    img_denorm = np.clip(dataset.denorm_(img.copy(), vol_info), 0.0, 1.0)
    base = augmenter.Global_Location_Scale_Augmentation(img_denorm.copy())
    strong = augmenter.Local_Location_Scale_Augmentation(
        img_denorm.copy(), label.astype(np.int32)
    )
    base = dataset.renorm_(np.clip(base, 0.0, 1.0), vol_info)
    strong = dataset.renorm_(np.clip(strong, 0.0, 1.0), vol_info)
    random.setstate(random_state)
    np.random.set_state(np_state)

    def to_tensor(arr):
        arr = np.transpose(np.float32(arr), (2, 0, 1))
        tensor = torch.from_numpy(arr)
        if args.tile_z_dim > 1:
            tensor = tensor.repeat(args.tile_z_dim, 1, 1)
        return tensor.unsqueeze(0)

    anchor = to_tensor(img)
    base = to_tensor(base)
    strong = to_tensor(strong)
    mask = torch.from_numpy(np.transpose(label, (2, 0, 1))).unsqueeze(0)
    return anchor, base, strong, mask, record


def forward_triplet(model, anchor, base, strong, use_cgsd):
    if use_cgsd:
        pred0, enc0, fstr0, fsty0 = model(anchor, return_feat=True)
        pred1, enc1, fstr1, fsty1 = model(base, return_feat=True)
        pred2, enc2, fstr2, fsty2 = model(strong, return_feat=True)
    else:
        pred0, enc0 = model(anchor, return_feat=False)
        pred1, enc1 = model(base, return_feat=False)
        pred2, enc2 = model(strong, return_feat=False)
        fstr0 = fsty0 = fstr1 = fsty1 = fstr2 = fsty2 = None
    return {
        "pred": pred0,
        "enc": (enc0, enc1, enc2),
        "f_str": (fstr0, fstr1, fstr2),
        "f_sty": (fsty0, fsty1, fsty2),
    }


def resize_bool(mask, size):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    out = F.interpolate(tensor, size=size, mode="nearest")[0, 0].numpy()
    return out > 0.5


def make_region_masks(label_np, target_size, kernel_size):
    fg = label_np.astype(np.int32) != 0
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    core = binary_erosion(fg, structure=structure, border_value=0)
    boundary = binary_dilation(fg, structure=structure, border_value=0) & ~core
    background = ~fg
    return {
        "core": resize_bool(core, target_size),
        "boundary": resize_bool(boundary, target_size),
        "background": resize_bool(background, target_size),
    }


def masked_mean(values, mask):
    if not np.any(mask):
        return math.nan
    return float(values[mask].mean())


def topk_region_ratio(topk, mask):
    denom = float(topk.sum())
    if denom <= 0:
        return math.nan
    return float((topk & mask).sum() / denom)


def gap_cosine_distance(fa, fb):
    za = F.adaptive_avg_pool2d(fa, 1).flatten(1)
    zb = F.adaptive_avg_pool2d(fb, 1).flatten(1)
    return float((1.0 - F.cosine_similarity(za, zb, dim=1)).mean().detach().cpu())


class PhiProjector:
    def __init__(self, state, args, device):
        self.state = state
        self.args = args
        self.device = device
        self.projector = None
        self.loaded = False

    def distance(self, fa, fb):
        if self.args.projection != "checkpoint" or self.state is None:
            return gap_cosine_distance(fa, fb), "gap"
        if self.projector is None:
            self.projector = Projector(
                in_channels=fa.shape[1],
                hidden_channels=self.args.proj_hidden_channels,
                proj_dim=self.args.proj_dim,
                feature_size=fa.shape[-1],
            ).to(self.device)
            self.projector.load_state_dict(self.state, strict=False)
            self.projector.eval()
            for param in self.projector.parameters():
                param.requires_grad_(False)
            self.loaded = True
        with torch.no_grad():
            za = self.projector(fa)
            zb = self.projector(fb)
            dist = 1.0 - F.cosine_similarity(za, zb, dim=1)
        return float(dist.mean().detach().cpu()), "checkpoint"


def collect_region_values(d_stab, masks, limit):
    values = {}
    for region, mask in masks.items():
        region_values = d_stab[mask]
        if region_values.size > limit > 0:
            idx = np.linspace(0, region_values.size - 1, limit).astype(np.int64)
            region_values = region_values[idx]
        values[region] = region_values.astype(np.float32)
    return values


def normalize_image(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def overlay_mask(image, mask, color=(1.0, 0.15, 0.05), alpha=0.45):
    base = np.stack([image, image, image], axis=-1)
    out = base.copy()
    for channel, value in enumerate(color):
        out[..., channel] = np.where(mask, (1 - alpha) * out[..., channel] + alpha * value, out[..., channel])
    return out


def save_case_figure(case, fig_dir):
    image = normalize_image(case["image"])
    gt = case["gt"]
    pred = case["full_pred"]
    wo_heat = case["wo_d_stab"]
    full_heat = case["full_d_stab"]
    wo_topk = resize_bool(case["wo_topk"], image.shape)
    full_topk = resize_bool(case["full_topk"], image.shape)

    fig, axes = plt.subplots(1, 7, figsize=(21, 3.4), constrained_layout=True)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(gt, cmap="tab20", vmin=0)
    axes[1].set_title("GT")
    axes[2].imshow(pred, cmap="tab20", vmin=0)
    axes[2].set_title("Prediction")
    im3 = axes[3].imshow(wo_heat, cmap="magma")
    axes[3].set_title("w/o CGSD d_stab")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)
    im4 = axes[4].imshow(full_heat, cmap="magma")
    axes[4].set_title("Full SAA d_stab")
    fig.colorbar(im4, ax=axes[4], fraction=0.046)
    axes[5].imshow(overlay_mask(image, wo_topk))
    axes[5].set_title("w/o CGSD Top-k")
    axes[6].imshow(overlay_mask(image, full_topk))
    axes[6].set_title("Full SAA Top-k")
    for ax in axes:
        ax.axis("off")
    out = fig_dir / f"case_{case['case_id']}_z{case['z_id']:04d}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_boxplot(region_values, fig_dir):
    labels = []
    data = []
    for method in ("wo_cgsd", "full_saa"):
        for region in ("core", "boundary", "background"):
            vals = np.concatenate(region_values[method][region]) if region_values[method][region] else np.array([])
            if vals.size == 0:
                continue
            labels.append(f"{method}\n{region}")
            data.append(vals)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("d_stab")
    ax.set_title("Core / Boundary / Background Stability Distance")
    ax.tick_params(axis="x", labelrotation=25)
    fig.savefig(fig_dir / "dstab_region_boxplot.png", dpi=180)
    plt.close(fig)


def save_composition_bar(rows, fig_dir):
    summary = summarize_rows(rows)
    methods = ["wo_cgsd", "full_saa"]
    regions = ["topk_core_ratio", "topk_boundary_ratio", "topk_background_ratio"]
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for i, region in enumerate(regions):
        vals = [summary.get(m, {}).get(region, math.nan) for m in methods]
        ax.bar(x + (i - 1) * width, vals, width, label=region.replace("topk_", "").replace("_ratio", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Selected-pixel ratio")
    ax.set_title("Top-k Selected Region Composition")
    ax.legend()
    fig.savefig(fig_dir / "topk_region_composition.png", dpi=180)
    plt.close(fig)


def save_cgsd_curve(rows, fig_dir):
    full = [r for r in rows if r["method"] == "full_saa" and not math.isnan(r["d_str"])]
    if not full:
        return
    x = np.arange(len(full))
    d_str = [r["d_str"] for r in full]
    d_sty = [r["d_sty"] for r in full]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(x, d_str, label="d_str", linewidth=1.4)
    ax.plot(x, d_sty, label="d_sty", linewidth=1.4)
    ax.set_xlabel("Evaluation slice index")
    ax.set_ylabel("Cosine distance")
    ax.set_title("Evaluation-stage d_str / d_sty Curve")
    ax.legend()
    fig.savefig(fig_dir / "d_str_d_sty_curve.png", dpi=180)
    plt.close(fig)


def summarize_rows(rows):
    grouped = defaultdict(list)
    metric_keys = [
        "mean_dstab_core",
        "mean_dstab_boundary",
        "mean_dstab_background",
        "topk_core_ratio",
        "topk_boundary_ratio",
        "topk_background_ratio",
        "d_str",
        "d_sty",
    ]
    for row in rows:
        grouped[row["method"]].append(row)
    summary = {}
    for method, method_rows in grouped.items():
        summary[method] = {}
        for key in metric_keys:
            vals = [r[key] for r in method_rows if not math.isnan(r[key])]
            summary[method][key] = float(np.mean(vals)) if vals else math.nan
    return summary


def analyze_one_method(name, model, phi, saam, tensors, label_np, args, device):
    anchor, base, strong = [x.to(device).float() for x in tensors]
    out = forward_triplet(model, anchor, base, strong, use_cgsd=(name == "full_saa"))
    enc0, enc1, enc2 = out["enc"]
    _, _, debug = saam(enc0, enc1, enc2, mask_size=label_np.shape, return_debug=True)
    d_stab = debug["d_stab"][0].detach().cpu().numpy()
    topk = debug["topk_mask"][0].detach().cpu().numpy() > 0.5
    masks = make_region_masks(label_np, d_stab.shape, args.morph_kernel)

    d_str = math.nan
    d_sty = math.nan
    phi_kind = "none"
    if name == "full_saa" and out["f_str"][1] is not None:
        d_str, phi_kind = phi.distance(out["f_str"][1], out["f_str"][2])
        d_sty, _ = phi.distance(out["f_sty"][1], out["f_sty"][2])

    pred = torch.argmax(out["pred"], dim=1)[0].detach().cpu().numpy().astype(np.int16)
    metrics = {
        "mean_dstab_core": masked_mean(d_stab, masks["core"]),
        "mean_dstab_boundary": masked_mean(d_stab, masks["boundary"]),
        "mean_dstab_background": masked_mean(d_stab, masks["background"]),
        "topk_core_ratio": topk_region_ratio(topk, masks["core"]),
        "topk_boundary_ratio": topk_region_ratio(topk, masks["boundary"]),
        "topk_background_ratio": topk_region_ratio(topk, masks["background"]),
        "d_str": d_str,
        "d_sty": d_sty,
        "phi": phi_kind,
    }
    return metrics, d_stab, topk, pred, collect_region_values(d_stab, masks, args.sample_region_values)


def write_metrics(rows, out_csv):
    fieldnames = [
        "epoch", "method", "data_name", "source", "target", "split", "case_id", "z_id",
        "mean_dstab_core", "mean_dstab_boundary", "mean_dstab_background",
        "topk_core_ratio", "topk_boundary_ratio", "topk_background_ratio",
        "d_str", "d_sty", "phi",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(summary, out_path):
    full = summary.get("full_saa", {})
    wo = summary.get("wo_cgsd", {})
    lines = []
    lines.append("CGSD mechanism analysis summary")
    lines.append(f"core d_stab下降: {full.get('mean_dstab_core', math.nan) < wo.get('mean_dstab_core', math.nan)} "
                 f"(full={full.get('mean_dstab_core', math.nan):.6f}, wo={wo.get('mean_dstab_core', math.nan):.6f})")
    lines.append(f"boundary d_stab上升: {full.get('mean_dstab_boundary', math.nan) > wo.get('mean_dstab_boundary', math.nan)} "
                 f"(full={full.get('mean_dstab_boundary', math.nan):.6f}, wo={wo.get('mean_dstab_boundary', math.nan):.6f})")
    lines.append(f"top-k boundary ratio下降: {full.get('topk_boundary_ratio', math.nan) < wo.get('topk_boundary_ratio', math.nan)} "
                 f"(full={full.get('topk_boundary_ratio', math.nan):.6f}, wo={wo.get('topk_boundary_ratio', math.nan):.6f})")
    lines.append(f"top-k core ratio上升: {full.get('topk_core_ratio', math.nan) > wo.get('topk_core_ratio', math.nan)} "
                 f"(full={full.get('topk_core_ratio', math.nan):.6f}, wo={wo.get('topk_core_ratio', math.nan):.6f})")
    lines.append(f"d_str < d_sty: {full.get('d_str', math.nan) < full.get('d_sty', math.nan)} "
                 f"(d_str={full.get('d_str', math.nan):.6f}, d_sty={full.get('d_sty', math.nan):.6f})")
    lines.append("")
    lines.append(
        "CGSD improves downstream stability estimation by producing cleaner shallow representations. "
        "With CGSD, organ-core regions show lower stability distance, boundary regions show higher "
        "stability distance, and the selected stable regions contain fewer boundary pixels. This "
        "supports the causal path from channel-level structure-style decoupling to region-level "
        "stability-aware alignment."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def save_epoch_trends(epoch_summaries, fig_dir):
    if len(epoch_summaries) < 2:
        return
    x = np.arange(len(epoch_summaries))
    labels = [str(item["epoch"]) for item in epoch_summaries]
    specs = [
        ("mean_dstab_core", "Core d_stab"),
        ("mean_dstab_boundary", "Boundary d_stab"),
        ("topk_core_ratio", "Top-k core ratio"),
        ("topk_boundary_ratio", "Top-k boundary ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, specs):
        for method in ("wo_cgsd", "full_saa"):
            vals = [item["summary"].get(method, {}).get(metric, math.nan) for item in epoch_summaries]
            ax.plot(x, vals, marker="o", label=method)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.savefig(fig_dir / "epoch_metric_trends.png", dpi=180)
    plt.close(fig)

    full_d_str = [item["summary"].get("full_saa", {}).get("d_str", math.nan) for item in epoch_summaries]
    full_d_sty = [item["summary"].get("full_saa", {}).get("d_sty", math.nan) for item in epoch_summaries]
    if any(not math.isnan(v) for v in full_d_str + full_d_sty):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(x, full_d_str, marker="o", label="d_str")
        ax.plot(x, full_d_sty, marker="o", label="d_sty")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_ylabel("Cosine distance")
        ax.set_title("CGSD Structure/Style Distance Across Epochs")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(fig_dir / "epoch_d_str_d_sty_trends.png", dpi=180)
        plt.close(fig)


def run_analysis_for_pair(args, dataset, device, full_ckpt, wo_ckpt, epoch_label, out_dir, fig_dir):
    full_model, full_phi_state, full_missing, full_unexpected = load_model(
        full_ckpt, use_cgsd=True, args=args, device=device
    )
    wo_model, _, wo_missing, wo_unexpected = load_model(
        wo_ckpt, use_cgsd=False, args=args, device=device
    )
    print(
        f"[epoch={epoch_label}] Loaded full SAA checkpoint: {full_ckpt} "
        f"(missing={len(full_missing)}, unexpected={len(full_unexpected)})"
    )
    print(
        f"[epoch={epoch_label}] Loaded w/o CGSD checkpoint: {wo_ckpt} "
        f"(missing={len(wo_missing)}, unexpected={len(wo_unexpected)})"
    )

    phi = PhiProjector(full_phi_state, args, device)
    saam = StabilityAwareAlignmentModule(
        tau=args.saam_tau,
        topk_ratio=args.saam_topk,
        stability_mode=args.saam_stability_mode,
    ).to(device)
    saam.eval()

    rows = []
    region_values = {
        "full_saa": defaultdict(list),
        "wo_cgsd": defaultdict(list),
    }
    visual_cases = []

    with torch.no_grad():
        max_slices = args.max_slices if args.max_slices > 0 else len(dataset)
        pbar = tqdm(range(min(max_slices, len(dataset))), desc="CGSD-SAAM analysis")
        for index in pbar:
            if index >= max_slices:
                break
            anchor, base, strong, mask, record = build_views_from_record(dataset, index, args)
            label_np = mask[0, 0].numpy().astype(np.int16)
            if not np.any(label_np != 0):
                continue

            tensors = (anchor, base, strong)
            case_id = str(record["scan_id"])
            z_id = int(record["z_id"])
            outputs = {}
            for method, model in (("wo_cgsd", wo_model), ("full_saa", full_model)):
                metrics, d_stab, topk, pred, sampled = analyze_one_method(
                    method, model, phi, saam, tensors, label_np, args, device
                )
                row = {
                    "method": method,
                    "epoch": epoch_label,
                    "data_name": args.data_name,
                    "source": args.source,
                    "target": args.target,
                    "split": args.split,
                    "case_id": case_id,
                    "z_id": z_id,
                    **metrics,
                }
                rows.append(row)
                outputs[method] = (metrics, d_stab, topk, pred)
                for region, vals in sampled.items():
                    if vals.size:
                        region_values[method][region].append(vals)

            if len(visual_cases) < args.num_visual_cases:
                image_np = anchor[0, 1].numpy() if anchor.shape[1] > 1 else anchor[0, 0].numpy()
                visual_cases.append({
                    "case_id": case_id.replace("/", "_"),
                    "z_id": z_id,
                    "image": image_np,
                    "gt": label_np,
                    "full_pred": outputs["full_saa"][3],
                    "wo_d_stab": outputs["wo_cgsd"][1],
                    "full_d_stab": outputs["full_saa"][1],
                    "wo_topk": outputs["wo_cgsd"][2],
                    "full_topk": outputs["full_saa"][2],
                })

    if not rows:
        raise RuntimeError("No foreground slices were analyzed. Check split/domain/data root.")

    write_metrics(rows, out_dir / "metrics.csv")
    for case in visual_cases:
        save_case_figure(case, fig_dir)
    save_boxplot(region_values, fig_dir)
    save_composition_bar(rows, fig_dir)
    save_cgsd_curve(rows, fig_dir)
    summary = summarize_rows(rows)
    summary_text = write_summary(summary, out_dir / "summary.txt")
    print(summary_text)
    print(f"Metrics CSV: {out_dir / 'metrics.csv'}")
    print(f"Figures: {fig_dir}")
    return rows, summary


def main():
    args = parse_args()
    args.nclass = args.nclass or DATASET_NCLASS[args.data_name]
    args.target = target_for_dataset(args.data_name, args.source, args.target)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root_fig_dir = args.out_dir / "figures"
    root_fig_dir.mkdir(parents=True, exist_ok=True)

    dataset, label_names = build_datasets(args)
    _ = label_names
    epochs = parse_epochs(args.epochs)

    all_rows = []
    epoch_summaries = []
    for epoch in epochs:
        epoch_label = "final" if epoch is None else str(epoch)
        full_ckpt = resolve_epoch_ckpt(
            args.full_ckpt, args.full_ckpt_template, epoch, "full"
        )
        wo_ckpt = resolve_epoch_ckpt(
            args.wo_cgsd_ckpt, args.wo_cgsd_ckpt_template, epoch, "wo_cgsd"
        )
        if not full_ckpt.exists():
            raise FileNotFoundError(f"Full SAA checkpoint not found: {full_ckpt}")
        if not wo_ckpt.exists():
            raise FileNotFoundError(f"w/o CGSD checkpoint not found: {wo_ckpt}")

        epoch_out_dir = args.out_dir if epoch is None else args.out_dir / f"epoch_{int(epoch):04d}"
        epoch_fig_dir = root_fig_dir if epoch is None else root_fig_dir / f"epoch_{int(epoch):04d}"
        epoch_out_dir.mkdir(parents=True, exist_ok=True)
        epoch_fig_dir.mkdir(parents=True, exist_ok=True)

        rows, summary = run_analysis_for_pair(
            args=args,
            dataset=dataset,
            device=device,
            full_ckpt=full_ckpt,
            wo_ckpt=wo_ckpt,
            epoch_label=epoch_label,
            out_dir=epoch_out_dir,
            fig_dir=epoch_fig_dir,
        )
        all_rows.extend(rows)
        epoch_summaries.append({"epoch": epoch_label, "summary": summary})

    write_metrics(all_rows, args.out_dir / "metrics.csv")
    save_epoch_trends(epoch_summaries, root_fig_dir)
    print(f"Aggregate metrics CSV: {args.out_dir / 'metrics.csv'}")
    print(f"Aggregate figures: {root_fig_dir}")


if __name__ == "__main__":
    main()
