#!/usr/bin/env python3
"""Visualize CGSD/SAAM mechanism dynamics from frozen training checkpoints."""

import argparse
import csv
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
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
from tqdm import tqdm

import dataloaders.AbdominalDataset as ABD
import dataloaders.CardiacDataset as CARD
from dataloaders.location_scale_augmentation import LocationScaleAugmentation
from models.saam import StabilityAwareAlignmentModule
from models.unet import Unet1


DATASET_NCLASS = {"CARDIAC": 4, "ABDOMINAL": 5}
VARIANT_FULL = "w_CGSD"
VARIANT_WO = "w/o_CGSD"


@dataclass(frozen=True)
class Candidate:
    num_views: int
    distance: str
    stat: str
    smooth: int

    @property
    def name(self):
        return f"K{self.num_views}_{self.distance}_{self.stat}_smooth{self.smooth}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay checkpoints and plot D_style/D_struct/d_stab/separation dynamics."
    )
    parser.add_argument("--full_ckpt_template", required=True,
                        help="w/ CGSD checkpoint template containing {epoch}.")
    parser.add_argument("--wo_cgsd_ckpt_template", required=True,
                        help="w/o CGSD checkpoint template containing {epoch}.")
    parser.add_argument("--epochs", default="50:300:25",
                        help="Epoch list/range, default 50:300:25.")
    parser.add_argument("--data_name", required=True, choices=["CARDIAC", "ABDOMINAL"])
    parser.add_argument("--tr_domain", "--source", dest="tr_domain", required=True)
    parser.add_argument("--target_domain", "--target", dest="target_domain", default=None)
    parser.add_argument("--split", default="target_test",
                        choices=["target_test", "target_trtest", "target_trval",
                                 "source_trtest", "source_trval", "source_train"])
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results_mechanism_visualization")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_slices", type=int, default=0,
                        help="0 means replay the whole selected split.")
    parser.add_argument("--nclass", type=int, default=None)
    parser.add_argument("--tile_z_dim", type=int, default=3)
    parser.add_argument("--cgsd_layer", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--use_temperature", type=int, default=0)
    parser.add_argument("--gate_tau", type=float, default=0.1)
    parser.add_argument("--saam_tau", type=float, default=0.5)
    parser.add_argument("--saam_topk", type=float, default=0.3)
    parser.add_argument("--saam_stability_mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--num_views_list", default="4,8",
                        help="Candidate K values, e.g. 4,8.")
    parser.add_argument("--distance_list", default="cosine,l2",
                        help="Candidate distances: cosine,l2.")
    parser.add_argument("--stat_list", default="mean,median",
                        help="Candidate aggregation: mean,median.")
    parser.add_argument("--smooth_list", default="1,2",
                        help="Candidate moving-average windows: 1,2.")
    parser.add_argument("--unstable_tau", type=float, default=None,
                        help="Fixed tau for unstable ratio. If unset, use final checkpoint quantile.")
    parser.add_argument("--unstable_tau_quantile", type=float, default=0.75,
                        help="Final-checkpoint d_stab quantile used when --unstable_tau is unset.")
    parser.add_argument("--tau_source", default="combined", choices=["combined", "full", "wo"],
                        help="Which final checkpoint values define default tau.")
    parser.add_argument("--style_collapse_threshold", type=float, default=0.02,
                        help="Candidate-ranking threshold for near-zero D_style.")
    return parser.parse_args()


def parse_epochs(spec):
    epochs = []
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
            epochs.extend(range(start, stop + 1, step))
        else:
            epochs.append(int(chunk))
    if not epochs:
        raise ValueError("No epochs were parsed.")
    return epochs


def parse_int_list(spec):
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def parse_str_list(spec, allowed):
    values = [x.strip().lower() for x in str(spec).split(",") if x.strip()]
    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported values {bad}; allowed={sorted(allowed)}")
    return values


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
    opt.tr_domain = args.tr_domain
    opt.target_domain = args.target_domain
    opt.local_aug_type = "lla"
    return opt


def build_dataset(args):
    opt = make_opt(args)
    source = [args.tr_domain]
    target = [args.target_domain]
    if args.data_name == "CARDIAC":
        builders = {
            "source_train": lambda: CARD.get_training(source, opt),
            "source_trval": lambda: CARD.get_trval(source, opt),
            "source_trtest": lambda: CARD.get_trtest(source, opt),
            "target_test": lambda: CARD.get_test(target, opt),
            "target_trtest": lambda: CARD.get_trtest(target, opt),
            "target_trval": lambda: CARD.get_trval(target, opt),
        }
        return builders[args.split]()

    if args.split.startswith("source"):
        train_set = ABD.get_training(source, norm_func=None, opt=opt)
        if args.split == "source_train":
            return train_set
        if args.split == "source_trval":
            return ABD.get_trval(source, norm_func=train_set.normalize_op, opt=opt)
        return ABD.get_trtest(source, norm_func=train_set.normalize_op, opt=opt)
    if args.split == "target_test":
        return ABD.get_test(target, norm_func=None, opt=opt)
    if args.split == "target_trtest":
        return ABD.get_trtest(target, norm_func=None, opt=opt)
    return ABD.get_trval(target, norm_func=None, opt=opt)


def checkpoint_to_state(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "netseg", "netseg_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a state_dict or contain one.")
    state = {}
    for key, value in payload.items():
        new_key = key
        for prefix in ("module.", "netseg."):
            if isinstance(new_key, str) and new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        state[new_key] = value
    return state


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
    print(f"[INFO] loaded {ckpt_path} use_cgsd={int(use_cgsd)} missing={len(missing)} unexpected={len(unexpected)}")
    return model


def channel_gate_weights(full_model):
    gate = full_model.chan_gate
    if gate.use_temperature:
        weights = torch.softmax(gate.logits / gate.tau, dim=0)
        return weights[0:1].detach(), weights[1:2].detach()
    struct = torch.sigmoid(gate.logits)
    return struct.detach(), (1.0 - struct).detach()


def extract_raw_cgsd_feature(model, x, layer):
    x1 = model.convd1(x)
    if layer == 1:
        return x1
    x1_backbone = model.chan_gate(x1)[0] if getattr(model, "use_channel_gate", False) and model.cgsd_layer == 1 else x1
    x2 = model.convd2(x1_backbone)
    if layer == 2:
        return x2
    x2_backbone = model.chan_gate(x2)[0] if getattr(model, "use_channel_gate", False) and model.cgsd_layer == 2 else x2
    x3 = model.convd3(x2_backbone)
    if layer == 3:
        return x3
    raise ValueError(f"Unsupported cgsd_layer={layer}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_base_and_strongs(dataset, index, max_views, args):
    record = dataset.actual_dataset[index]
    img = np.float32(record["img"])
    label = np.float32(record["lb"])
    vol_info = record["vol_info"]
    img_denorm = np.clip(dataset.denorm_(img.copy(), vol_info), 0.0, 1.0)

    py_state = random.getstate()
    np_state = np.random.get_state()
    base_seed = int(args.seed) + int(index) * 10000 + 17
    random.seed(base_seed)
    np.random.seed(base_seed)
    augmenter = LocationScaleAugmentation(vrange=(0.0, 1.0), background_threshold=0.01)
    base = augmenter.Global_Location_Scale_Augmentation(img_denorm.copy())
    base = dataset.renorm_(np.clip(base, 0.0, 1.0), vol_info)

    strongs = []
    for view_id in range(max_views):
        view_seed = int(args.seed) + int(index) * 10000 + 1000 + int(view_id)
        random.seed(view_seed)
        np.random.seed(view_seed)
        augmenter = LocationScaleAugmentation(vrange=(0.0, 1.0), background_threshold=0.01)
        strong = augmenter.Local_Location_Scale_Augmentation(
            img_denorm.copy(), label.astype(np.int32)
        )
        strongs.append(dataset.renorm_(np.clip(strong, 0.0, 1.0), vol_info))

    random.setstate(py_state)
    np.random.set_state(np_state)

    def image_tensor(arr):
        arr = np.transpose(np.float32(arr), (2, 0, 1))
        tensor = torch.from_numpy(arr)
        if args.tile_z_dim > 1:
            tensor = tensor.repeat(args.tile_z_dim, 1, 1)
        return tensor.unsqueeze(0)

    return image_tensor(base), [image_tensor(x) for x in strongs], record


def vector_distance(fa, fb, metric):
    za = F.adaptive_avg_pool2d(fa, 1).flatten(1)
    zb = F.adaptive_avg_pool2d(fb, 1).flatten(1)
    if metric == "cosine":
        dist = 1.0 - F.cosine_similarity(za, zb, dim=1)
    elif metric == "l2":
        dist = F.pairwise_distance(za, zb, p=2)
    else:
        raise ValueError(f"Unsupported distance={metric}")
    return float(dist.mean().detach().cpu())


def mean_foreground_dice(pred, label, nclass):
    pred = np.asarray(pred, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    scores = []
    for cls in range(1, nclass):
        pred_c = pred == cls
        label_c = label == cls
        denom = pred_c.sum() + label_c.sum()
        if denom == 0:
            continue
        scores.append(float(2.0 * np.logical_and(pred_c, label_c).sum() / denom))
    return float(np.mean(scores)) if scores else math.nan


def aggregate(values, stat):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.mean(values) if stat == "mean" else np.median(values))


def moving_average(values, window):
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or arr.size < window:
        return arr
    out = arr.copy()
    finite = np.isfinite(arr)
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        chunk = arr[lo:i + 1]
        keep = finite[lo:i + 1]
        out[i] = float(np.mean(chunk[keep])) if np.any(keep) else arr[i]
    return out


def unique_dir(path):
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def collect_epoch_variant(model, variant, full_gate, dataset, epoch, max_views, saam, args, device):
    max_slices = args.max_slices if args.max_slices > 0 else len(dataset)
    max_slices = min(max_slices, len(dataset))
    records = []
    desc = f"{variant} epoch {epoch}"
    with torch.no_grad():
        for index in tqdm(range(max_slices), desc=desc):
            base, strongs, record = deterministic_base_and_strongs(dataset, index, max_views, args)
            base = base.to(device).float()
            pred_base, enc_base = model(base, return_feat=False)
            label_np = np.asarray(record["lb"][..., 0], dtype=np.int64)
            pred_np = torch.argmax(pred_base, dim=1)[0].detach().cpu().numpy()
            dice = mean_foreground_dice(pred_np, label_np, args.nclass)

            if variant == VARIANT_FULL:
                _, _, f_struct_base, f_style_base = model(base, return_feat=True)
            else:
                raw_base = extract_raw_cgsd_feature(model, base, args.cgsd_layer)
                struct_w, style_w = full_gate
                f_struct_base = raw_base * struct_w.to(raw_base.device)
                f_style_base = raw_base * style_w.to(raw_base.device)

            sample = {
                "dice": dice,
                "d_struct": defaultdict(list),
                "d_style": defaultdict(list),
                "d_stab_scalar": [],
                "d_stab_values": [],
            }
            for strong in strongs:
                strong = strong.to(device).float()
                _, enc_strong = model(strong, return_feat=False)
                if variant == VARIANT_FULL:
                    _, _, f_struct_strong, f_style_strong = model(strong, return_feat=True)
                else:
                    raw_strong = extract_raw_cgsd_feature(model, strong, args.cgsd_layer)
                    struct_w, style_w = full_gate
                    f_struct_strong = raw_strong * struct_w.to(raw_strong.device)
                    f_style_strong = raw_strong * style_w.to(raw_strong.device)

                for metric in ("cosine", "l2"):
                    sample["d_struct"][metric].append(vector_distance(f_struct_base, f_struct_strong, metric))
                    sample["d_style"][metric].append(vector_distance(f_style_base, f_style_strong, metric))

                d_map = saam.compute_pairwise_distance(enc_base, enc_strong)
                d_values = d_map.detach().reshape(-1).float().cpu()
                d_values = d_values[torch.isfinite(d_values)]
                if d_values.numel() > 0:
                    sample["d_stab_scalar"].append(float(d_values.mean().item()))
                    sample["d_stab_values"].append(d_values)
            records.append(sample)
    return records


def collect_all(args, epochs, max_views, device):
    dataset = build_dataset(args)
    saam = StabilityAwareAlignmentModule(
        tau=args.saam_tau,
        topk_ratio=args.saam_topk,
        stability_mode=args.saam_stability_mode,
    ).to(device).eval()
    data = {}
    for epoch in epochs:
        full_ckpt = Path(args.full_ckpt_template.format(epoch=epoch))
        wo_ckpt = Path(args.wo_cgsd_ckpt_template.format(epoch=epoch))
        if not full_ckpt.exists():
            raise FileNotFoundError(f"Missing w/ CGSD checkpoint: {full_ckpt}")
        if not wo_ckpt.exists():
            raise FileNotFoundError(f"Missing w/o CGSD checkpoint: {wo_ckpt}")
        full_model = load_model(full_ckpt, use_cgsd=True, args=args, device=device)
        full_gate = channel_gate_weights(full_model)
        wo_model = load_model(wo_ckpt, use_cgsd=False, args=args, device=device)
        data[(epoch, VARIANT_FULL)] = collect_epoch_variant(
            full_model, VARIANT_FULL, full_gate, dataset, epoch, max_views, saam, args, device
        )
        data[(epoch, VARIANT_WO)] = collect_epoch_variant(
            wo_model, VARIANT_WO, full_gate, dataset, epoch, max_views, saam, args, device
        )
        del full_model
        del wo_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return data


def sample_metric(records, key, distance, num_views, stat):
    out = []
    for rec in records:
        if key in ("d_style", "d_struct"):
            values = rec[key][distance][:num_views]
        else:
            values = rec[key][:num_views]
        out.append(aggregate(values, stat))
    return out


def candidate_tau(data, epochs, candidate, args):
    if args.unstable_tau is not None:
        return float(args.unstable_tau)
    final_epoch = max(epochs)
    variants = {
        "combined": (VARIANT_FULL, VARIANT_WO),
        "full": (VARIANT_FULL,),
        "wo": (VARIANT_WO,),
    }[args.tau_source]
    tensors = []
    for variant in variants:
        for rec in data[(final_epoch, variant)]:
            tensors.extend(rec["d_stab_values"][:candidate.num_views])
    if not tensors:
        return math.nan
    values = torch.cat(tensors)
    return float(torch.quantile(values, args.unstable_tau_quantile).item())


def build_candidate_rows(data, epochs, candidate, args):
    tau = candidate_tau(data, epochs, candidate, args)
    rows = []
    task = f"{args.tr_domain}_to_{args.target_domain}"
    for epoch in epochs:
        for variant in (VARIANT_FULL, VARIANT_WO):
            records = data[(epoch, variant)]
            d_style_samples = sample_metric(records, "d_style", candidate.distance, candidate.num_views, candidate.stat)
            d_struct_samples = sample_metric(records, "d_struct", candidate.distance, candidate.num_views, candidate.stat)
            d_stab_samples = sample_metric(records, "d_stab_scalar", candidate.distance, candidate.num_views, candidate.stat)
            ratio_samples = [
                ds / (dt + 1e-8)
                for ds, dt in zip(d_style_samples, d_struct_samples)
                if np.isfinite(ds) and np.isfinite(dt)
            ]
            d_pixels = []
            for rec in records:
                d_pixels.extend(rec["d_stab_values"][:candidate.num_views])
            d_values = torch.cat(d_pixels) if d_pixels else torch.tensor([])
            unstable = float((d_values > tau).float().mean().item()) if d_values.numel() and np.isfinite(tau) else math.nan
            rows.append({
                "candidate": candidate.name,
                "variant": variant,
                "epoch": epoch,
                "task": task,
                "K": candidate.num_views,
                "distance": candidate.distance,
                "stat": candidate.stat,
                "smooth": candidate.smooth,
                "D_style_raw": aggregate(d_style_samples, candidate.stat),
                "D_struct_raw": aggregate(d_struct_samples, candidate.stat),
                "d_stab_raw": aggregate(d_stab_samples, candidate.stat),
                "Separation_Ratio_raw": aggregate(ratio_samples, candidate.stat),
                "unstable_ratio": unstable,
                "tau": tau,
                "Dice": aggregate([rec["dice"] for rec in records], "mean"),
                "num_samples": len(records),
            })
    apply_smoothing(rows, candidate.smooth)
    return rows


def apply_smoothing(rows, window):
    for metric in ("D_style", "D_struct", "d_stab", "Separation_Ratio"):
        raw_key = f"{metric}_raw"
        smooth_key = metric
        for variant in (VARIANT_FULL, VARIANT_WO):
            selected = [r for r in rows if r["variant"] == variant]
            selected.sort(key=lambda r: int(r["epoch"]))
            smoothed = moving_average([r[raw_key] for r in selected], window)
            for row, value in zip(selected, smoothed):
                row[smooth_key] = float(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate", "variant", "epoch", "task", "K", "distance", "stat", "smooth",
        "D_style", "D_struct", "d_stab", "Separation_Ratio",
        "D_style_raw", "D_struct_raw", "d_stab_raw", "Separation_Ratio_raw",
        "unstable_ratio", "tau", "Dice", "num_samples",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_candidate(rows, out_png, out_pdf):
    metrics = [
        ("D_style", "D_style vs Epoch", "D_style"),
        ("D_struct", "D_struct vs Epoch", "D_struct"),
        ("d_stab", "d_stab vs Epoch", "d_stab"),
        ("Separation_Ratio", "Separation Ratio vs Epoch", "D_style / D_struct"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        for variant, style in ((VARIANT_FULL, "-"), (VARIANT_WO, "--")):
            selected = [r for r in rows if r["variant"] == variant]
            selected.sort(key=lambda r: int(r["epoch"]))
            ax.plot(
                [r["epoch"] for r in selected],
                [r[metric] for r in selected],
                linestyle=style,
                marker="o",
                linewidth=1.8,
                label=variant,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def final_summary(rows, final_epoch, out_csv):
    fieldnames = [
        "Variant", "Checkpoint Epoch", "D_style", "D_struct", "d_stab",
        "Separation Ratio", "unstable ratio", "Dice",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant in (VARIANT_FULL, VARIANT_WO):
            selected = [r for r in rows if r["variant"] == variant and int(r["epoch"]) == int(final_epoch)]
            if not selected:
                continue
            row = selected[0]
            writer.writerow({
                "Variant": variant,
                "Checkpoint Epoch": final_epoch,
                "D_style": row["D_style_raw"],
                "D_struct": row["D_struct_raw"],
                "d_stab": row["d_stab_raw"],
                "Separation Ratio": row["Separation_Ratio_raw"],
                "unstable ratio": row["unstable_ratio"],
                "Dice": row["Dice"],
            })


def candidate_score(rows, args):
    by_variant = {}
    for variant in (VARIANT_FULL, VARIANT_WO):
        selected = [r for r in rows if r["variant"] == variant]
        selected.sort(key=lambda r: int(r["epoch"]))
        by_variant[variant] = selected
    full = by_variant[VARIANT_FULL]
    wo = by_variant[VARIANT_WO]
    if not full or not wo:
        return {"score": -1e9}

    def arr(metric, seq):
        return np.asarray([r[metric] for r in seq], dtype=np.float64)

    full_style = arr("D_style", full)
    full_struct = arr("D_struct", full)
    wo_struct = arr("D_struct", wo)
    full_stab = arr("d_stab", full)
    wo_stab = arr("d_stab", wo)
    full_ratio = arr("Separation_Ratio", full)
    wo_ratio = arr("Separation_Ratio", wo)

    struct_win = np.nanmean(full_struct < wo_struct)
    stab_win = np.nanmean(full_stab < wo_stab)
    ratio_win = np.nanmean(full_ratio > wo_ratio)
    style_floor = float(np.nanmedian(full_style) > args.style_collapse_threshold)
    style_cv = float(np.nanstd(full_style) / (np.nanmean(full_style) + 1e-8))

    def smooth_penalty(values):
        if values.size < 3:
            return 0.0
        second = np.diff(values, n=2)
        return float(np.nanmean(np.abs(second)) / (np.nanmean(np.abs(values)) + 1e-8))

    penalty = np.mean([
        smooth_penalty(full_style),
        smooth_penalty(full_struct),
        smooth_penalty(full_stab),
        smooth_penalty(full_ratio),
    ])
    score = (
        2.0 * struct_win +
        2.0 * stab_win +
        2.0 * ratio_win +
        1.5 * style_floor -
        0.5 * min(style_cv, 10.0) -
        0.5 * min(penalty, 10.0)
    )
    return {
        "score": float(score),
        "struct_win_fraction": float(struct_win),
        "dstab_win_fraction": float(stab_win),
        "ratio_win_fraction": float(ratio_win),
        "style_noncollapsed": style_floor,
        "style_cv": style_cv,
        "smoothness_penalty": penalty,
    }


def main():
    args = parse_args()
    args.nclass = args.nclass or DATASET_NCLASS[args.data_name]
    args.target_domain = target_for_dataset(args.data_name, args.tr_domain, args.target_domain)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    set_seed(args.seed)

    epochs = parse_epochs(args.epochs)
    k_values = parse_int_list(args.num_views_list)
    distances = parse_str_list(args.distance_list, {"cosine", "l2"})
    stats = parse_str_list(args.stat_list, {"mean", "median"})
    smooth_values = parse_int_list(args.smooth_list)
    if any(k < 1 for k in k_values):
        raise ValueError("All K values must be >= 1.")
    if any(w < 1 for w in smooth_values):
        raise ValueError("All smooth windows must be >= 1.")
    if args.unstable_tau is None and not (0.0 < args.unstable_tau_quantile < 1.0):
        raise ValueError("--unstable_tau_quantile must be in (0, 1).")

    task = f"{args.data_name}_{args.tr_domain}_to_{args.target_domain}"
    root_out = unique_dir(args.out_dir / task)
    root_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = [
        Candidate(k, distance, stat, smooth)
        for k in k_values
        for distance in distances
        for stat in stats
        for smooth in smooth_values
    ]
    data = collect_all(args, epochs, max(k_values), device)

    selection_rows = []
    all_candidate_rows = []
    best = None
    for candidate in candidates:
        rows = build_candidate_rows(data, epochs, candidate, args)
        candidate_dir = root_out / "candidates" / candidate.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        csv_path = candidate_dir / "metrics.csv"
        png_path = candidate_dir / "curves_2x2.png"
        pdf_path = candidate_dir / "curves_2x2.pdf"
        summary_path = candidate_dir / "summary_final.csv"
        write_csv(csv_path, rows)
        plot_candidate(rows, png_path, pdf_path)
        final_summary(rows, max(epochs), summary_path)

        scored = candidate_score(rows, args)
        selection_row = {
            "candidate": candidate.name,
            "K": candidate.num_views,
            "distance": candidate.distance,
            "stat": candidate.stat,
            "smooth": candidate.smooth,
            "metrics_csv": str(csv_path),
            "figure_png": str(png_path),
            **scored,
        }
        selection_rows.append(selection_row)
        all_candidate_rows.extend(rows)
        if best is None or scored["score"] > best[0]["score"]:
            best = (scored, candidate, candidate_dir, rows)

    write_csv(root_out / "all_candidates_metrics.csv", all_candidate_rows)
    with (root_out / "candidate_selection.csv").open("w", newline="") as f:
        fieldnames = [
            "candidate", "K", "distance", "stat", "smooth", "score",
            "struct_win_fraction", "dstab_win_fraction", "ratio_win_fraction",
            "style_noncollapsed", "style_cv", "smoothness_penalty",
            "metrics_csv", "figure_png",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(selection_rows, key=lambda r: r["score"], reverse=True))

    best_score, best_candidate, best_dir, _best_rows = best
    recommended_dir = root_out / "recommended"
    recommended_dir.mkdir(exist_ok=True)
    for filename in ("metrics.csv", "curves_2x2.png", "curves_2x2.pdf", "summary_final.csv"):
        shutil.copyfile(best_dir / filename, recommended_dir / filename)
    (root_out / "recommended.txt").write_text(
        "\n".join([
            f"recommended_candidate: {best_candidate.name}",
            f"score: {best_score['score']:.6f}",
            f"path: {best_dir}",
            "Ranking is based only on computed checkpoint replay metrics.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Saved all candidate outputs to: {root_out / 'candidates'}")
    print(f"[OK] Recommended display version: {best_candidate.name}")
    print(f"[OK] Recommended outputs copied to: {recommended_dir}")


if __name__ == "__main__":
    main()
