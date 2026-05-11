"""Prototype helpers for SAAM-SPMM.

Tensor conventions:
  feature:      [B, D, h, w]
  pseudo_label: [B, H, W]
  stable_mask:  [B, H, W]
  prototypes:   [K, D]
"""

import logging
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _forward_logits_features(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        logits = output[0]
        feature = output[1] if len(output) > 1 else None
    else:
        logits = output
        feature = None
    if feature is None:
        raise RuntimeError("SAAM-SPMM requires model(images) to return (logits, bottleneck_feature).")
    if feature.dim() != 4:
        raise ValueError(f"Expected bottleneck feature [B,D,h,w], got {tuple(feature.shape)}")
    return logits, feature


def _extract_image_label(batch):
    if isinstance(batch, dict):
        image = batch.get("base_view", batch.get("image", None))
        label = batch.get("label", None)
        if image is None or label is None:
            raise KeyError("source prototype export batch must contain base_view/image and label")
        return image, label
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type for prototype export: {type(batch)}")


@torch.no_grad()
def export_source_prototypes(model, loader, num_classes, device, save_path, data_name=None, tr_domain=None):
    """Export source class prototypes without storing raw source images."""
    model.eval()
    num_classes = int(num_classes)
    sums = None
    sq_sums = None
    counts = torch.zeros(num_classes, device=device, dtype=torch.float64)

    for batch in tqdm(loader, total=len(loader), desc="Export source prototypes"):
        images, labels = _extract_image_label(batch)
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        _, feature = _forward_logits_features(model, images)
        feature = feature.detach()
        labels_2d = labels.squeeze(1) if labels.dim() == 4 else labels
        labels_feat = F.interpolate(
            labels_2d.unsqueeze(1).float(),
            size=feature.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, feature.size(1)).double()
        label_flat = labels_feat.reshape(-1)
        if sums is None:
            sums = torch.zeros(num_classes, feature.size(1), device=device, dtype=torch.float64)
            sq_sums = torch.zeros_like(sums)
        for class_idx in range(num_classes):
            mask = label_flat == class_idx
            count = mask.sum()
            if count.item() == 0:
                continue
            class_feat = feature_flat[mask]
            sums[class_idx] += class_feat.sum(dim=0)
            sq_sums[class_idx] += class_feat.pow(2).sum(dim=0)
            counts[class_idx] += count.double()

    if sums is None:
        raise RuntimeError("No source batches were available for prototype export.")

    safe_counts = counts.clamp_min(1.0).unsqueeze(1)
    mean = sums / safe_counts
    var = (sq_sums / safe_counts) - mean.pow(2)
    var = var.clamp_min(0.0)
    prototype = mean.float()
    payload = {
        "prototype": prototype.cpu(),
        "mean": mean.float().cpu(),
        "var": var.float().cpu(),
        "count": counts.float().cpu(),
        "num_classes": num_classes,
        "feature_dim": int(prototype.size(1)),
        "data_name": data_name,
        "tr_domain": tr_domain,
    }
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(payload, save_path)
    logger.info(
        "Saved source prototypes to %s with prototype shape=%s count=%s",
        save_path,
        tuple(prototype.shape),
        counts.detach().cpu().tolist(),
    )
    return payload


def load_source_prototypes(path, device):
    if path is None:
        raise ValueError("source_prototype_path is required when use_source_anchor=1")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source prototype file not found: {path}")
    payload = torch.load(path, map_location=device)
    if "prototype" not in payload:
        raise KeyError(f"Source prototype file lacks 'prototype': {path}")
    payload["prototype"] = payload["prototype"].to(device).float()
    if "mean" in payload:
        payload["mean"] = payload["mean"].to(device).float()
    if "var" in payload:
        payload["var"] = payload["var"].to(device).float()
    if "count" in payload:
        payload["count"] = payload["count"].to(device).float()
    return payload


def compute_target_prototypes(feature, pseudo_label, stable_mask, num_classes):
    """Compute target class prototypes using stable pixels only."""
    if feature.dim() != 4:
        raise ValueError(f"feature expects [B,D,h,w], got {tuple(feature.shape)}")
    labels = pseudo_label.detach()
    if labels.dim() != 3:
        raise ValueError(f"pseudo_label expects [B,H,W], got {tuple(labels.shape)}")
    labels_feat = F.interpolate(
        labels.unsqueeze(1).float(),
        size=feature.shape[-2:],
        mode="nearest",
    ).squeeze(1).long()
    stable_feat = F.interpolate(
        stable_mask.unsqueeze(1).float(),
        size=feature.shape[-2:],
        mode="nearest",
    ).squeeze(1) > 0.5
    feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, feature.size(1))
    labels_flat = labels_feat.reshape(-1)
    stable_flat = stable_feat.reshape(-1)
    prototypes = feature.new_zeros(int(num_classes), feature.size(1))
    valid = torch.zeros(int(num_classes), device=feature.device, dtype=torch.bool)
    for class_idx in range(int(num_classes)):
        mask = stable_flat & (labels_flat == class_idx)
        if torch.any(mask):
            prototypes[class_idx] = feature_flat[mask].mean(dim=0)
            valid[class_idx] = True
    return prototypes, valid


def prototype_matching_loss(target_proto, source_proto, valid_mask, loss_type="cosine"):
    """Match target prototypes to source/memory prototypes."""
    if valid_mask is None:
        valid_mask = torch.ones(target_proto.size(0), device=target_proto.device, dtype=torch.bool)
    valid_mask = valid_mask.bool()
    if not torch.any(valid_mask):
        return target_proto.sum() * 0.0
    target = target_proto[valid_mask]
    source = source_proto[valid_mask].to(target)
    loss_type = str(loss_type).lower()
    if loss_type == "cosine":
        target = F.normalize(target, p=2, dim=1)
        source = F.normalize(source, p=2, dim=1)
        return (1.0 - (target * source).sum(dim=1)).mean()
    if loss_type == "mse":
        return F.mse_loss(target, source)
    raise ValueError(f"Unknown prototype loss type: {loss_type}")


class OnlinePrototypeMemory:
    """EMA target prototype memory updated from stable target pixels only."""

    def __init__(self, num_classes, feature_dim, momentum=0.9, device=None):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.prototypes = torch.zeros(self.num_classes, self.feature_dim, device=device)
        self.valid = torch.zeros(self.num_classes, device=device, dtype=torch.bool)

    @torch.no_grad()
    def update(self, target_proto, valid_mask):
        target_proto = target_proto.detach()
        valid_mask = valid_mask.detach().bool()
        updated = []
        for class_idx in torch.where(valid_mask)[0].tolist():
            if self.valid[class_idx]:
                self.prototypes[class_idx].mul_(self.momentum).add_(target_proto[class_idx], alpha=1.0 - self.momentum)
            else:
                self.prototypes[class_idx].copy_(target_proto[class_idx])
                self.valid[class_idx] = True
            updated.append(int(class_idx))
        return updated
