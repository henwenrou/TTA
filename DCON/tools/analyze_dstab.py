#!/usr/bin/env python3
"""Checkpoint replay for SAAM d_stab and unstable-ratio analysis."""

import argparse
import csv
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import dataloaders.AbdominalDataset as ABD
import dataloaders.CardiacDataset as CARD
from dataloaders.location_scale_augmentation import LocationScaleAugmentation
from models.saam import StabilityAwareAlignmentModule
from models.unet import Unet1


DATASET_NCLASS = {
    "CARDIAC": 4,
    "ABDOMINAL": 5,
}


@dataclass
class CheckpointSpec:
    path: Path
    variant: str
    use_cgsd: bool
    epoch: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze SAAM d_stab and unstable ratio for frozen checkpoints."
    )
    parser.add_argument("--ckpt_path", action="append", type=str, default=[],
                        help="Checkpoint path. Repeat for multiple variants; may contain {epoch}.")
    parser.add_argument("--ckpt_template", action="append", type=str, default=[],
                        help="Checkpoint template containing {epoch}. Repeat for multiple variants.")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Epoch list/range, e.g. 25,50,75 or 25:300:25.")
    parser.add_argument("--data_name", required=True, choices=["CARDIAC", "ABDOMINAL"])
    parser.add_argument("--tr_domain", "--source", dest="tr_domain", required=True)
    parser.add_argument("--target_domain", "--target", dest="target_domain", default=None)
    parser.add_argument("--variant_name", action="append", default=[],
                        help="Variant label. Repeat in the same order as checkpoints/templates.")
    parser.add_argument("--use_cgsd", action="append", type=int, default=[],
                        help="Whether each checkpoint uses CGSD. Repeat or omit to infer from variant.")
    parser.add_argument("--save_csv", type=Path, default=None,
                        help="Output CSV path. Defaults under results_dstab_analysis/.")
    parser.add_argument("--dstab_tau", type=float, default=None,
                        help="Fixed threshold for unstable_ratio.")
    parser.add_argument("--dstab_tau_quantile", type=float, default=None,
                        help="Quantile threshold computed over all compared d_stab values.")
    parser.add_argument("--num_views", type=int, default=4,
                        help="Number of deterministic strong perturbation views per sample.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", default="target_test",
                        choices=["target_test", "target_trtest", "target_trval",
                                 "source_trtest", "source_trval", "source_train"])
    parser.add_argument("--max_slices", type=int, default=0,
                        help="0 means analyze the whole selected split.")
    parser.add_argument("--nclass", type=int, default=None)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--tile_z_dim", type=int, default=3)
    parser.add_argument("--cgsd_layer", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--use_temperature", type=int, default=0)
    parser.add_argument("--gate_tau", type=float, default=0.1)
    parser.add_argument("--saam_tau", type=float, default=0.5)
    parser.add_argument("--saam_topk", type=float, default=0.3)
    parser.add_argument("--saam_stability_mode", default="mean", choices=["mean", "max"])
    return parser.parse_args()


def parse_epochs(spec):
    if spec is None or str(spec).strip() == "":
        return [None]
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
            if step <= 0:
                raise ValueError("Epoch range step must be positive.")
            epochs.extend(range(start, stop + 1, step))
        else:
            epochs.append(int(chunk))
    if not epochs:
        raise ValueError("No epochs parsed from --epochs.")
    return epochs


def target_for_dataset(data_name, source, explicit_target):
    if explicit_target:
        return explicit_target
    if data_name == "CARDIAC":
        return "bSSFP" if source == "LGE" else "LGE"
    if data_name == "ABDOMINAL":
        return "CHAOST2" if source == "SABSCT" else "SABSCT"
    raise ValueError(f"Unsupported dataset: {data_name}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_epoch(path):
    name = str(path)
    for pattern in (r"([0-9]+)_net_Seg", r"epoch[_-]?([0-9]+)", r"/([0-9]+)[^/]*$"):
        match = re.search(pattern, name)
        if match:
            return match.group(1)
    nums = re.findall(r"([0-9]+)", name)
    return nums[-1] if nums else "unknown"


def infer_use_cgsd(variant):
    lowered = str(variant).lower()
    disabled_tokens = ("w/o", "wo_", "without", "no_cgsd", "wocgsd", "wo-cgsd")
    return not any(token in lowered for token in disabled_tokens)


def expand_specs(args):
    inputs = [(Path(x), False) for x in args.ckpt_path]
    inputs.extend((Path(x), True) for x in args.ckpt_template)
    if not inputs:
        raise ValueError("At least one --ckpt_path or --ckpt_template is required.")

    variants = args.variant_name or [f"variant_{i}" for i in range(len(inputs))]
    if len(variants) == 1 and len(inputs) > 1:
        variants = [variants[0] for _ in inputs]
    if len(variants) != len(inputs):
        raise ValueError("--variant_name count must match checkpoint/template count, or be length 1.")

    use_cgsd_flags = args.use_cgsd
    if len(use_cgsd_flags) == 1 and len(inputs) > 1:
        use_cgsd_flags = [use_cgsd_flags[0] for _ in inputs]
    if use_cgsd_flags and len(use_cgsd_flags) != len(inputs):
        raise ValueError("--use_cgsd count must match checkpoint/template count, or be length 1.")

    specs = []
    for idx, ((path, is_template), variant) in enumerate(zip(inputs, variants)):
        use_cgsd = bool(use_cgsd_flags[idx]) if use_cgsd_flags else infer_use_cgsd(variant)
        if is_template or "{epoch}" in str(path):
            epochs = parse_epochs(args.epochs)
            if epochs == [None]:
                raise ValueError("Checkpoint templates require --epochs.")
            for epoch in epochs:
                ckpt = Path(str(path).format(epoch=epoch))
                specs.append(CheckpointSpec(ckpt, variant, use_cgsd, str(epoch)))
        else:
            specs.append(CheckpointSpec(path, variant, use_cgsd, infer_epoch(path)))
    return specs


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

    if args.data_name == "ABDOMINAL":
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

    raise ValueError(f"Unsupported dataset: {args.data_name}")


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


def load_model(spec, args, device):
    payload = torch.load(str(spec.path), map_location="cpu")
    model = Unet1(
        c=3,
        num_classes=args.nclass,
        use_channel_gate=spec.use_cgsd,
        cgsd_layer=args.cgsd_layer,
        use_temperature=bool(args.use_temperature),
        gate_tau=args.gate_tau,
    )
    missing, unexpected = model.load_state_dict(checkpoint_to_state(payload), strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(
        f"[INFO] Loaded {spec.variant} epoch={spec.epoch}: {spec.path} "
        f"(use_cgsd={int(spec.use_cgsd)}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return model


def build_views_from_record(dataset, index, view_id, args):
    record = dataset.actual_dataset[index]
    img = np.float32(record["img"])
    label = np.float32(record["lb"])
    vol_info = record["vol_info"]

    py_state = random.getstate()
    np_state = np.random.get_state()
    seed = int(args.seed) + int(index) * max(1, int(args.num_views)) + int(view_id)
    random.seed(seed)
    np.random.seed(seed)

    augmenter = LocationScaleAugmentation(vrange=(0.0, 1.0), background_threshold=0.01)
    img_denorm = np.clip(dataset.denorm_(img.copy(), vol_info), 0.0, 1.0)
    base = augmenter.Global_Location_Scale_Augmentation(img_denorm.copy())
    strong = augmenter.Local_Location_Scale_Augmentation(
        img_denorm.copy(), label.astype(np.int32)
    )
    base = dataset.renorm_(np.clip(base, 0.0, 1.0), vol_info)
    strong = dataset.renorm_(np.clip(strong, 0.0, 1.0), vol_info)

    random.setstate(py_state)
    np.random.set_state(np_state)

    def image_tensor(arr):
        arr = np.transpose(np.float32(arr), (2, 0, 1))
        tensor = torch.from_numpy(arr)
        if args.tile_z_dim > 1:
            tensor = tensor.repeat(args.tile_z_dim, 1, 1)
        return tensor.unsqueeze(0)

    anchor = image_tensor(img)
    base = image_tensor(base)
    strong = image_tensor(strong)
    label_tensor = torch.from_numpy(np.transpose(label, (2, 0, 1))).unsqueeze(0)
    return anchor, base, strong, label_tensor, record


def forward_triplet(model, anchor, base, strong, use_cgsd):
    if use_cgsd:
        pred0, enc0, fstr0, fsty0 = model(anchor, return_feat=True)
        pred1, enc1, fstr1, fsty1 = model(base, return_feat=True)
        pred2, enc2, fstr2, fsty2 = model(strong, return_feat=True)
    else:
        pred0, enc0 = model(anchor, return_feat=False)
        pred1, enc1 = model(base, return_feat=False)
        pred2, enc2 = model(strong, return_feat=False)
        fstr1 = fstr2 = fsty1 = fsty2 = None
    return pred0, (enc0, enc1, enc2), (fstr1, fstr2), (fsty1, fsty2)


def flatten_d_stab(d_stab):
    values = d_stab.detach().reshape(-1).float().cpu()
    return values[torch.isfinite(values)]


def gap_distance(fa, fb):
    za = F.adaptive_avg_pool2d(fa, 1).flatten(1)
    zb = F.adaptive_avg_pool2d(fb, 1).flatten(1)
    return float((1.0 - F.cosine_similarity(za, zb, dim=1)).mean().detach().cpu())


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


def safe_nanmean(values):
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def analyze_checkpoint(spec, dataset, saam, args, device):
    model = load_model(spec, args, device)
    max_slices = args.max_slices if args.max_slices > 0 else len(dataset)
    max_slices = min(max_slices, len(dataset))

    dstab_values = []
    dice_values = []
    d_struct_values = []
    d_style_values = []
    num_samples = 0

    desc = f"d_stab {spec.variant} e{spec.epoch}"
    with torch.no_grad():
        for index in tqdm(range(max_slices), desc=desc):
            sample_counted = False
            for view_id in range(args.num_views):
                anchor, base, strong, label, _record = build_views_from_record(
                    dataset, index, view_id, args
                )
                anchor = anchor.to(device).float()
                base = base.to(device).float()
                strong = strong.to(device).float()

                pred0, encs, f_str, f_sty = forward_triplet(
                    model, anchor, base, strong, spec.use_cgsd
                )
                d_stab, _, _, _ = saam.compute_stability(*encs)
                valid = flatten_d_stab(d_stab)
                if valid.numel() > 0:
                    dstab_values.append(valid)

                if spec.use_cgsd and f_str[0] is not None and f_str[1] is not None:
                    d_struct_values.append(gap_distance(f_str[0], f_str[1]))
                    d_style_values.append(gap_distance(f_sty[0], f_sty[1]))

                if view_id == 0:
                    label_np = label[0, 0].numpy().astype(np.int64)
                    pred_np = torch.argmax(pred0, dim=1)[0].detach().cpu().numpy()
                    dice_values.append(mean_foreground_dice(pred_np, label_np, args.nclass))
                    sample_counted = True
            if sample_counted:
                num_samples += 1

    if not dstab_values:
        raise RuntimeError(f"No valid d_stab values collected for {spec.path}.")

    return {
        "spec": spec,
        "values": torch.cat(dstab_values),
        "num_samples": num_samples,
        "dice": safe_nanmean(dice_values),
        "d_struct": safe_nanmean(d_struct_values),
        "d_style": safe_nanmean(d_style_values),
    }


def summarize_values(values, tau):
    return {
        "mean_d_stab": float(values.mean().item()),
        "median_d_stab": float(values.median().item()),
        "p75_d_stab": float(torch.quantile(values, 0.75).item()),
        "p90_d_stab": float(torch.quantile(values, 0.90).item()),
        "unstable_ratio": float((values > tau).float().mean().item()),
    }


def unique_path(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def write_outputs(results, tau, args, save_csv):
    task = f"{args.tr_domain}_to_{args.target_domain}"
    fieldnames = [
        "variant", "checkpoint", "epoch", "task",
        "mean_d_stab", "median_d_stab", "p75_d_stab", "p90_d_stab",
        "unstable_ratio", "tau", "num_samples", "num_views", "dice",
        "d_struct", "d_style",
    ]

    rows = []
    for result in results:
        spec = result["spec"]
        stats = summarize_values(result["values"], tau)
        rows.append({
            "variant": spec.variant,
            "checkpoint": str(spec.path),
            "epoch": spec.epoch,
            "task": task,
            **stats,
            "tau": tau,
            "num_samples": result["num_samples"],
            "num_views": args.num_views,
            "dice": result["dice"],
            "d_struct": result["d_struct"],
            "d_style": result["d_style"],
        })

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    with save_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = save_csv.with_name(f"{save_csv.stem}_summary{save_csv.suffix}")
    summary_csv = unique_path(summary_csv)
    summary_fields = ["variant", "task", "mean_d_stab", "unstable_ratio", "dice", "d_struct", "d_style"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for variant, variant_rows in grouped.items():
            writer.writerow({
                "variant": variant,
                "task": task,
                "mean_d_stab": float(np.mean([float(r["mean_d_stab"]) for r in variant_rows])),
                "unstable_ratio": float(np.mean([float(r["unstable_ratio"]) for r in variant_rows])),
                "dice": safe_nanmean([float(r["dice"]) for r in variant_rows]),
                "d_struct": safe_nanmean([float(r["d_struct"]) for r in variant_rows]),
                "d_style": safe_nanmean([float(r["d_style"]) for r in variant_rows]),
            })
    return summary_csv


def main():
    args = parse_args()
    args.nclass = args.nclass or DATASET_NCLASS[args.data_name]
    args.target_domain = target_for_dataset(args.data_name, args.tr_domain, args.target_domain)
    if args.num_views < 1:
        raise ValueError("--num_views must be >= 1")
    if args.dstab_tau is not None and args.dstab_tau_quantile is not None:
        raise ValueError("Use either --dstab_tau or --dstab_tau_quantile, not both.")
    if args.dstab_tau_quantile is not None and not (0.0 < args.dstab_tau_quantile < 1.0):
        raise ValueError("--dstab_tau_quantile must be in (0, 1).")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = expand_specs(args)
    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {spec.path}")

    dataset = build_dataset(args)
    saam = StabilityAwareAlignmentModule(
        tau=args.saam_tau,
        topk_ratio=args.saam_topk,
        stability_mode=args.saam_stability_mode,
    ).to(device)
    saam.eval()

    results = [analyze_checkpoint(spec, dataset, saam, args, device) for spec in specs]
    all_values = torch.cat([result["values"] for result in results])
    if args.dstab_tau is not None:
        tau = float(args.dstab_tau)
    else:
        quantile = 0.70 if args.dstab_tau_quantile is None else float(args.dstab_tau_quantile)
        tau = float(torch.quantile(all_values, quantile).item())

    default_csv = (
        ROOT / "results_dstab_analysis" /
        f"{args.data_name}_{args.tr_domain}_to_{args.target_domain}_dstab.csv"
    )
    save_csv = unique_path(args.save_csv or default_csv)
    summary_csv = write_outputs(results, tau, args, save_csv)

    print(f"[OK] Saved d_stab statistics to: {save_csv}")
    print(f"[OK] Saved d_stab summary to: {summary_csv}")
    print(f"[INFO] tau used for unstable_ratio: {tau:.6f}")


if __name__ == "__main__":
    main()
