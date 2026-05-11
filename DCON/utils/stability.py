"""Stability estimation and weighted target losses for SAAM-SPMM.

Tensor conventions:
  images:     [B, C, H, W]
  views:      [V, B, C, H, W]
  logits:     [B, K, H, W]
  prob_stack: [V, B, K, H, W]
  weight:     [B, H, W]
"""

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel(device, dtype):
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    kernel = kernel / kernel.sum()
    return kernel


def _blur(images):
    b, c, _, _ = images.shape
    kernel = _gaussian_kernel(images.device, images.dtype).view(1, 1, 3, 3)
    kernel = kernel.repeat(c, 1, 1, 1)
    return F.conv2d(images, kernel, padding=1, groups=c)


def _dropout_image(images, drop_prob=0.05):
    keep = (torch.rand_like(images[:, :1]) > drop_prob).to(images.dtype)
    return images * keep


def _affine(images, max_translate=0.015, max_rotate_deg=2.0):
    b = images.shape[0]
    angle = (torch.rand(b, device=images.device, dtype=images.dtype) * 2.0 - 1.0)
    angle = angle * (max_rotate_deg * math.pi / 180.0)
    tx = (torch.rand(b, device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_translate
    ty = (torch.rand(b, device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_translate
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    theta = torch.zeros(b, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def build_weak_views(images, num_views=5):
    """Build weak target views.

    Args:
        images: input target image tensor [B, C, H, W].
        num_views: number of views V. View 0 is the original image.

    Returns:
        stacked weak views [V, B, C, H, W].
    """
    if images.dim() != 4:
        raise ValueError(f"build_weak_views expects [B,C,H,W], got {tuple(images.shape)}")
    num_views = max(1, int(num_views))
    views = [images]
    builders = (
        lambda x: torch.clamp(x * 0.95, min=float(x.detach().min()), max=float(x.detach().max())),
        lambda x: torch.clamp(x + 0.01 * torch.randn_like(x), min=float(x.detach().min()), max=float(x.detach().max())),
        _blur,
        _dropout_image,
        _affine,
        lambda x: torch.clamp(torch.sign(x) * torch.abs(x).clamp_min(1e-6).pow(1.05), min=float(x.detach().min()), max=float(x.detach().max())),
    )
    idx = 0
    while len(views) < num_views:
        views.append(builders[idx % len(builders)](images))
        idx += 1
    return torch.stack(views, dim=0)


def compute_stability_map(prob_stack, metric="variance"):
    """Compute a pixel-wise stability map [B, H, W], not class-wise stability."""
    if prob_stack.dim() != 5:
        raise ValueError(f"prob_stack expects [V,B,C,H,W], got {tuple(prob_stack.shape)}")
    metric = str(metric).lower()
    eps = 1e-6
    if metric == "variance":
        var_map = prob_stack.var(dim=0, unbiased=False).mean(dim=1)
        stability = 1.0 - var_map
    elif metric == "kl":
        mean_prob = prob_stack.mean(dim=0).clamp_min(eps)
        kl = (prob_stack.clamp_min(eps) * (prob_stack.clamp_min(eps).log() - mean_prob.log())).sum(dim=2)
        stability = torch.exp(-kl.mean(dim=0))
    elif metric == "entropy":
        entropy = -(prob_stack.clamp_min(eps) * prob_stack.clamp_min(eps).log()).sum(dim=2)
        ent_var = entropy.var(dim=0, unbiased=False)
        stability = torch.exp(-ent_var)
    else:
        raise ValueError(f"Unknown SAAM stability metric: {metric}")
    return stability.clamp(0.0, 1.0)


def build_stable_mask(stability, stable_threshold=None, stable_topk_percent=0.3):
    """Create a binary stable mask [B, H, W] from threshold or top-k stable pixels."""
    if stability.dim() != 3:
        raise ValueError(f"stability expects [B,H,W], got {tuple(stability.shape)}")
    if stable_threshold is not None:
        return (stability >= float(stable_threshold)).to(stability.dtype)

    topk = float(stable_topk_percent)
    if not 0.0 < topk <= 1.0:
        raise ValueError(f"stable_topk_percent must be in (0,1], got {topk}")
    b, h, w = stability.shape
    flat = stability.reshape(b, -1)
    k = max(1, int(flat.size(1) * topk))
    _, indices = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)
    return mask.view(b, h, w)


def _resize_weight(weight, size):
    if weight.dim() != 3:
        raise ValueError(f"weight expects [B,H,W], got {tuple(weight.shape)}")
    if weight.shape[-2:] == size:
        return weight
    return F.interpolate(weight.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)


def stability_weighted_entropy(logits, weight):
    """Weighted pixel entropy over logits [B,K,H,W] with weight [B,H,W]."""
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1)
    weight = _resize_weight(weight, entropy.shape[-2:]).clamp_min(0.0)
    return (entropy * weight).sum() / weight.sum().clamp_min(1e-6)


def stability_weighted_consistency(prob_stack, weight):
    """Weighted soft probability consistency across views."""
    if prob_stack.dim() != 5:
        raise ValueError(f"prob_stack expects [V,B,C,H,W], got {tuple(prob_stack.shape)}")
    mean_prob = prob_stack.mean(dim=0).detach()
    mse = (prob_stack - mean_prob.unsqueeze(0)).pow(2).mean(dim=2)
    weight = _resize_weight(weight, prob_stack.shape[-2:]).clamp_min(0.0)
    return (mse * weight.unsqueeze(0)).sum() / (weight.sum() * prob_stack.size(0)).clamp_min(1e-6)
