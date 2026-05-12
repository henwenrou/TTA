"""Multi-view tensor transforms for reliability-gated segmentation TTA.

Each view has an inverse operation for probability/logit tensors so that all
predictions are compared in the original image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ViewTransform:
    """Description of one deterministic TTA view and its inverse metadata."""

    name: str
    crop_box: Tuple[int, int, int, int] | None = None
    gamma: float | None = None
    noise_std: float | None = None


def _gamma_perturb(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply a per-sample gamma perturbation while preserving z-score scale."""

    reduce_dims = tuple(range(1, x.dim()))
    xmin = x.amin(dim=reduce_dims, keepdim=True)
    xmax = x.amax(dim=reduce_dims, keepdim=True)
    scaled = (x - xmin) / (xmax - xmin + 1e-6)
    scaled = torch.clamp(scaled, 0.0, 1.0).pow(gamma)
    return scaled * (xmax - xmin + 1e-6) + xmin


def _crop_resize(x: torch.Tensor, scale: float = 0.9) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Center-crop and resize a tensor back to its original spatial size."""

    _, _, height, width = x.shape
    crop_h = max(1, int(round(height * scale)))
    crop_w = max(1, int(round(width * scale)))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = x[:, :, top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
    return resized, (top, left, crop_h, crop_w)


def make_multiview_batch(
    image: torch.Tensor,
    num_views: int = 8,
    noise_std: float = 0.03,
    gamma: float = 1.15,
    crop_scale: float = 0.9,
) -> Tuple[torch.Tensor, List[ViewTransform]]:
    """Build a stacked batch of TTA views for a BCHW image tensor.

    Args:
        image: Target image tensor with shape ``[B, C, H, W]``.
        num_views: Number of views to return. The canonical order is identity,
            horizontal flip, vertical flip, rotate 90, rotate -90, gamma,
            gaussian noise, and small crop-resize.
        noise_std: Standard deviation for normalized-image Gaussian noise.
        gamma: Gamma exponent used for the intensity perturbation.
        crop_scale: Fraction of the spatial field retained by crop-resize.

    Returns:
        A tuple ``(view_batch, transforms)`` where ``view_batch`` has shape
        ``[V * B, C, H, W]``.
    """

    if image.dim() != 4:
        raise ValueError(f"Expected BCHW image, got shape {tuple(image.shape)}")
    if num_views < 1 or num_views > 8:
        raise ValueError(f"num_views must be in [1, 8], got {num_views}")

    view_specs: List[Tuple[torch.Tensor, ViewTransform]] = [
        (image, ViewTransform("identity")),
        (torch.flip(image, dims=(3,)), ViewTransform("hflip")),
        (torch.flip(image, dims=(2,)), ViewTransform("vflip")),
        (torch.rot90(image, k=1, dims=(2, 3)), ViewTransform("rot90")),
        (torch.rot90(image, k=-1, dims=(2, 3)), ViewTransform("rot-90")),
        (_gamma_perturb(image, gamma), ViewTransform("gamma", gamma=gamma)),
        (image + torch.randn_like(image) * noise_std, ViewTransform("gaussian_noise", noise_std=noise_std)),
    ]
    cropped, crop_box = _crop_resize(image, scale=crop_scale)
    view_specs.append((cropped, ViewTransform("crop_resize", crop_box=crop_box)))

    selected = view_specs[:num_views]
    return torch.cat([view for view, _ in selected], dim=0), [spec for _, spec in selected]


def inverse_view_tensor(
    tensor: torch.Tensor,
    view: ViewTransform,
    original_size: Sequence[int],
) -> torch.Tensor:
    """Map a view-space BCHW tensor back to original image coordinates."""

    height, width = int(original_size[0]), int(original_size[1])
    if view.name == "identity":
        return tensor
    if view.name == "hflip":
        return torch.flip(tensor, dims=(3,))
    if view.name == "vflip":
        return torch.flip(tensor, dims=(2,))
    if view.name == "rot90":
        return torch.rot90(tensor, k=-1, dims=(2, 3))
    if view.name == "rot-90":
        return torch.rot90(tensor, k=1, dims=(2, 3))
    if view.name in {"gamma", "gaussian_noise"}:
        return tensor
    if view.name == "crop_resize":
        if view.crop_box is None:
            raise ValueError("crop_resize inverse requires crop_box metadata")
        top, left, crop_h, crop_w = view.crop_box
        crop_probs = F.interpolate(tensor, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
        out = tensor.new_full(
            (tensor.shape[0], tensor.shape[1], height, width),
            1.0 / max(float(tensor.shape[1]), 1.0),
        )
        out[:, :, top : top + crop_h, left : left + crop_w] = crop_probs
        return out
    raise ValueError(f"Unknown view transform: {view.name}")


def inverse_multiview_tensor(
    tensor: torch.Tensor,
    views: Sequence[ViewTransform],
    batch_size: int,
    original_size: Sequence[int],
) -> torch.Tensor:
    """Inverse-map a stacked ``[V * B, C, H, W]`` tensor to ``[V, B, C, H, W]``."""

    if tensor.shape[0] != len(views) * batch_size:
        raise ValueError(
            f"Expected first dim {len(views) * batch_size}, got {tensor.shape[0]}"
        )
    restored = []
    for idx, view in enumerate(views):
        chunk = tensor[idx * batch_size : (idx + 1) * batch_size]
        restored.append(inverse_view_tensor(chunk, view, original_size))
    return torch.stack(restored, dim=0)
