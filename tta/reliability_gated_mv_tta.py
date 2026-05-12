#!/usr/bin/env python3
"""Run reliability-gated multi-view TTA on DCON U-Net segmentation checkpoints."""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DCON_ROOT = ROOT / "DCON"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DCON_ROOT) not in sys.path:
    sys.path.insert(0, str(DCON_ROOT))

from tta.eval_tta_failures import (  # noqa: E402
    evaluate_volume_failures,
    save_debug_arrays,
    summarize_rows,
    write_csv,
    write_top_failure_cases,
)
from tta.multiview_transforms import inverse_multiview_tensor, make_multiview_batch  # noqa: E402
from tta.tta_losses import (  # noqa: E402
    compute_reliability_scores,
    entropy_minimization_loss,
    multiview_consistency_loss,
    reliability_gated_mv_loss,
)

METHODS = (
    "source_only",
    "entropy_tta",
    "mv_consistency_tta",
    "reliability_gated_mv_tta",
)


def str2bool(value: str | bool) -> bool:
    """Parse shell-friendly boolean values."""

    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the minimal reliability TTA experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--data_name", dest="dataset", default="abdominal")
    parser.add_argument("--direction", required=True, help="Example: CHAOST2_to_SABSCT")
    parser.add_argument("--ckpt", "--restore_from", dest="ckpt", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--update", default="bn_affine", choices=["bn_affine"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--out_dir", default="outputs/reliability_tta")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--low_dice_threshold", type=float, default=0.5)
    parser.add_argument("--beta_region", type=float, default=10.0)
    parser.add_argument("--beta_area", type=float, default=4.0)
    parser.add_argument("--beta_presence", type=float, default=4.0)
    parser.add_argument("--tau_class", type=float, default=0.0)
    parser.add_argument("--lambda_ent", type=float, default=1.0)
    parser.add_argument("--lambda_cons", type=float, default=1.0)
    parser.add_argument("--use_r_class", type=str2bool, default=True)
    parser.add_argument("--use_r_region", type=str2bool, default=True)
    parser.add_argument("--logit_damping", type=str2bool, default=False)
    parser.add_argument("--damping_alpha", type=float, default=1.0)
    parser.add_argument("--top_k_failures", type=int, default=10)
    return parser.parse_args()


def split_direction(direction: str) -> Tuple[str, str]:
    """Split source/target direction names like ``CHAOST2_to_SABSCT``."""

    if "_to_" in direction:
        source, target = direction.split("_to_", 1)
    elif "->" in direction:
        source, target = direction.split("->", 1)
    else:
        raise ValueError(f"Direction must look like source_to_target, got {direction}")
    return source, target


def canonical_dataset_name(name: str) -> str:
    """Normalize user dataset names to DCON constants."""

    lowered = name.lower()
    if lowered in {"abdominal", "abdomen", "abd"}:
        return "ABDOMINAL"
    if lowered in {"cardiac", "heart"}:
        return "CARDIAC"
    raise ValueError(f"Unsupported dataset for this runner: {name}")


def make_dcon_opt(args: argparse.Namespace, data_name: str, source: str, target: str, nclass: int) -> SimpleNamespace:
    """Create the small option namespace required by DCON datasets."""

    return SimpleNamespace(
        data_name=data_name,
        tr_domain=source,
        target_domain=target,
        nclass=nclass,
        model="unet",
        use_sgf=0,
        sgf_view2_only=0,
        use_cgsd=0,
        cgsd_layer=1,
        use_temperature=0,
        gate_tau=0.1,
        use_projector=0,
        num_workers=args.num_workers,
    )


def build_dataset(args: argparse.Namespace, data_name: str, source: str, target: str):
    """Build the target-domain DCON test dataset and label names."""

    os.environ["SAA_DATA_ROOT"] = str(Path(args.data_root).resolve())
    if data_name == "ABDOMINAL":
        import dataloaders.AbdominalDataset as ABD

        opt = make_dcon_opt(args, data_name, source, target, nclass=len(ABD.LABEL_NAME))
        dataset = ABD.get_test(modality=[target], norm_func=None, opt=opt)
        return dataset, ABD.LABEL_NAME, opt
    if data_name == "CARDIAC":
        import dataloaders.CardiacDataset as cardiac_cls

        opt = make_dcon_opt(args, data_name, source, target, nclass=len(cardiac_cls.LABEL_NAME))
        dataset = cardiac_cls.get_test(modality=[target], opt=opt)
        return dataset, cardiac_cls.LABEL_NAME, opt
    raise ValueError(data_name)


def checkpoint_state_dict(path: str | os.PathLike[str]) -> Dict[str, torch.Tensor]:
    """Load a DCON checkpoint and return a model state dict."""

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "netseg", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload at {path}")
    state = {}
    for key, value in payload.items():
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        if clean_key.startswith("netseg."):
            clean_key = clean_key[len("netseg.") :]
        if torch.is_tensor(value):
            state[clean_key] = value
    return state


def infer_cgsd_config(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Infer optional ChannelGate settings from checkpoint keys."""

    logits = state.get("chan_gate.logits")
    if logits is None:
        return {"use_channel_gate": False}
    channels = int(logits.shape[1])
    layer_by_channels = {16: 1, 32: 2, 64: 3}
    return {
        "use_channel_gate": True,
        "cgsd_layer": layer_by_channels.get(channels, 1),
        "use_temperature": int(logits.shape[0]) == 2,
    }


def build_model(num_classes: int, state: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    """Instantiate DCON Unet1 and load the source checkpoint."""

    from models.unet import Unet1

    cgsd = infer_cgsd_config(state)
    model = Unet1(
        c=3,
        num_classes=num_classes,
        use_channel_gate=bool(cgsd.get("use_channel_gate", False)),
        cgsd_layer=int(cgsd.get("cgsd_layer", 1)),
        use_temperature=bool(cgsd.get("use_temperature", False)),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[checkpoint] missing keys ignored: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys ignored: {len(unexpected)}")
    return model.to(device)


def forward_logits(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Run a DCON model and normalize tuple outputs to logits."""

    output = model(image)
    return output[0] if isinstance(output, (tuple, list)) else output


def set_bn_tracking(model: nn.Module, enabled: bool) -> None:
    """Set BatchNorm tracking mode without deleting source running buffers."""

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = enabled


def configure_bn_affine_updates(model: nn.Module) -> List[nn.Parameter]:
    """Freeze the model except BatchNorm2d affine weight/bias parameters."""

    model.train()
    model.requires_grad_(False)
    params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.track_running_stats = False
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
    if not params:
        raise RuntimeError("No BatchNorm2d affine parameters found for TTA.")
    return params


def make_optimizer(params: Sequence[nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    """Create the requested small-parameter TTA optimizer."""

    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    return torch.optim.Adam(params, lr=args.lr)


def multiview_probs(
    model: nn.Module,
    image: torch.Tensor,
    args: argparse.Namespace,
    r_class_for_damping: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward all views and inverse-map probabilities to original coordinates."""

    batch, _, height, width = image.shape
    view_batch, views = make_multiview_batch(image, num_views=args.views)
    logits = forward_logits(model, view_batch)
    if r_class_for_damping is not None:
        penalty = float(args.damping_alpha) * (1.0 - r_class_for_damping.to(logits.device))
        penalty = penalty.repeat(len(views), 1)[:, :, None, None]
        logits = logits - penalty
    probs = torch.softmax(logits, dim=1)
    return inverse_multiview_tensor(probs, views, batch, (height, width))


def compute_slice_reliability(
    model: nn.Module,
    image: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    """Compute current multi-view reliability for a target slice."""

    probs = multiview_probs(model, image, args)
    return compute_reliability_scores(
        probs,
        min_area=args.min_area,
        beta_region=args.beta_region,
        beta_area=args.beta_area,
        beta_presence=args.beta_presence,
    )


def adapt_and_predict_slice(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    image: torch.Tensor,
    method: str,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run per-slice TTA updates and return prediction plus reliability outputs."""

    if method == "source_only":
        model.eval()
        with torch.no_grad():
            pred = torch.argmax(forward_logits(model, image), dim=1)
            rel = compute_slice_reliability(model, image, args)
        return (
            pred[0].detach().cpu().numpy().astype(np.uint8),
            rel["R_region"][0, 0].detach().cpu().numpy(),
            rel["R_class"][0].detach().cpu().numpy(),
        )

    if optimizer is None:
        raise RuntimeError(f"{method} requires an optimizer")

    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if method == "entropy_tta":
            logits = forward_logits(model, image)
            loss = entropy_minimization_loss(logits)
        elif method == "mv_consistency_tta":
            probs = multiview_probs(model, image, args)
            loss = multiview_consistency_loss(probs)
        elif method == "reliability_gated_mv_tta":
            probs = multiview_probs(model, image, args)
            rel = compute_reliability_scores(
                probs,
                min_area=args.min_area,
                beta_region=args.beta_region,
                beta_area=args.beta_area,
                beta_presence=args.beta_presence,
            )
            if args.logit_damping:
                probs = multiview_probs(model, image, args, r_class_for_damping=rel["R_class"])
                rel = compute_reliability_scores(
                    probs,
                    min_area=args.min_area,
                    beta_region=args.beta_region,
                    beta_area=args.beta_area,
                    beta_presence=args.beta_presence,
                )
            loss_dict = reliability_gated_mv_loss(
                probs,
                rel,
                lambda_ent=args.lambda_ent,
                lambda_cons=args.lambda_cons,
                use_region_gate=args.use_r_region,
                use_class_gate=args.use_r_class,
                tau_class=args.tau_class,
            )
            loss = loss_dict["loss"]
        else:
            raise ValueError(f"Unknown method: {method}")
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if method in {"mv_consistency_tta", "reliability_gated_mv_tta"}:
            rel = compute_slice_reliability(model, image, args)
            probs = rel["mean_prob"]
            if method == "reliability_gated_mv_tta" and args.logit_damping:
                penalty = float(args.damping_alpha) * (1.0 - rel["R_class"].to(probs.device))
                probs = torch.softmax(torch.log(probs.clamp_min(1e-6)) - penalty[:, :, None, None], dim=1)
            pred = probs.argmax(dim=1)
        else:
            logits = forward_logits(model, image)
            pred = torch.argmax(logits, dim=1)
            rel = compute_slice_reliability(model, image, args)

    return (
        pred[0].detach().cpu().numpy().astype(np.uint8),
        rel["R_region"][0, 0].detach().cpu().numpy(),
        rel["R_class"][0].detach().cpu().numpy(),
    )


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a model state dict for exact per-volume source resets."""

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def batch_bool(value) -> bool:
    """Convert DataLoader-collated bool fields to Python bool."""

    if torch.is_tensor(value):
        return bool(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def batch_int(value) -> int:
    """Convert DataLoader-collated integer fields to Python int."""

    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def run_method(
    method: str,
    args: argparse.Namespace,
    dataset,
    loader: DataLoader,
    label_names: Sequence[str],
    source_state: Dict[str, torch.Tensor],
    state_for_build: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Run one TTA method over the target test loader and write all outputs."""

    method_dir = Path(args.out_dir) / args.direction / method
    method_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(len(label_names), state_for_build, device)
    rows = []
    current = None
    pred_vol = gt_vol = r_region_vol = r_class_slices = None
    optimizer = None

    progress = tqdm(loader, desc=f"{method}", leave=True)
    for batch in progress:
        if batch_bool(batch["is_start"]):
            set_bn_tracking(model, True)
            model.load_state_dict(copy.deepcopy(source_state), strict=False)
            if method == "source_only":
                optimizer = None
                model.eval()
            else:
                params = configure_bn_affine_updates(model)
                optimizer = make_optimizer(params, args)

            current = str(batch["scan_id"][0])
            nframe = batch_int(batch["nframe"])
            _, _, height, width = batch["base_view"].shape
            pred_vol = np.zeros((nframe, height, width), dtype=np.uint8)
            gt_vol = np.zeros((nframe, height, width), dtype=np.uint8)
            r_region_vol = np.zeros((nframe, height, width), dtype=np.float32)
            r_class_slices = []

        if current is None or pred_vol is None or gt_vol is None or r_region_vol is None:
            raise RuntimeError("Loader yielded a slice before volume start.")

        z_id = batch_int(batch["z_id"])
        image = batch["base_view"].to(device=device, dtype=torch.float32)
        pred_slice, r_region, r_class = adapt_and_predict_slice(model, optimizer, image, method, args)
        pred_vol[z_id] = pred_slice
        gt_vol[z_id] = batch["label"][0, 0].numpy().astype(np.uint8)
        if r_region is not None:
            r_region_vol[z_id] = r_region.astype(np.float32)
        if r_class is not None:
            r_class_slices.append(r_class.astype(np.float32))

        if batch_bool(batch["is_end"]):
            r_class_vol = np.mean(np.stack(r_class_slices, axis=0), axis=0) if r_class_slices else None
            row = evaluate_volume_failures(
                current,
                pred_vol,
                gt_vol,
                label_names,
                r_class=r_class_vol,
                r_region=r_region_vol,
                min_area=args.min_area,
                low_dice_threshold=args.low_dice_threshold,
            )
            rows.append(row)
            save_debug_arrays(method_dir, current, pred_vol, r_region_vol, r_class_vol)
            progress.set_postfix(
                dice=f"{row['mean_dice']:.4f}",
                hallucinations=int(row["absent_hallucination_count"]),
            )
            current = None

    summary = summarize_rows(rows, label_names)
    write_csv(method_dir / "metrics.csv", rows)
    write_csv(method_dir / "failure_summary.csv", [summary])
    write_top_failure_cases(method_dir, rows, top_k=args.top_k_failures)


def seed_everything(seed: int) -> None:
    """Seed RNGs used by the experiment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    args.ckpt = str(Path(args.ckpt).expanduser())
    args.out_dir = str(Path(args.out_dir).expanduser())
    seed_everything(args.seed)
    source, target = split_direction(args.direction)
    data_name = canonical_dataset_name(args.dataset)
    dataset, label_names, _ = build_dataset(args, data_name, source, target)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    requested = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {METHODS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_state = checkpoint_state_dict(args.ckpt)
    base_model = build_model(len(label_names), raw_state, device=torch.device("cpu"))
    source_state = clone_state_dict(base_model)

    print(
        f"Running reliability TTA: dataset={data_name}, direction={source}->{target}, "
        f"cases={len(dataset.scan_ids[target]) if hasattr(dataset, 'scan_ids') and target in dataset.scan_ids else 'unknown'}, "
        f"device={device}, methods={requested}"
    )
    for method in requested:
        run_method(method, args, dataset, loader, label_names, source_state, raw_state, device)


if __name__ == "__main__":
    main()
