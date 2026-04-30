import math
import random

import torch
import torch.nn.functional as F


def marginal_entropy(logits):
    """MEMO marginal entropy over augmented segmentation predictions."""
    log_probs = logits.log_softmax(dim=1)
    normalizer = math.log(logits.shape[0]) + math.log(logits.shape[2]) + math.log(logits.shape[3])
    avg_log_probs = torch.logsumexp(log_probs, dim=(0, 2, 3)) - normalizer
    avg_log_probs = torch.clamp(avg_log_probs, min=torch.finfo(avg_log_probs.dtype).min)
    return -(avg_log_probs.exp() * avg_log_probs).sum()


def build_memo_batch(
    image,
    n_augmentations,
    include_identity=True,
    hflip_p=0.0,
):
    """Create conservative medical-image MEMO views for one test slice.

    The input is already z-score normalized by the DCON dataset. Intensity
    transforms are applied in a per-slice 0-1 window and mapped back to the
    original normalized range so the network input scale stays close to test
    data.
    """
    if image.shape[0] != 1:
        raise ValueError(f"MEMO expects test batch size 1, got {image.shape[0]}")
    if n_augmentations < 1:
        raise ValueError(f"n_augmentations must be >= 1, got {n_augmentations}")

    base = image.detach()
    views = []
    if include_identity:
        views.append(base)

    while len(views) < n_augmentations:
        views.append(_medical_memo_aug(base, hflip_p=hflip_p))

    return torch.cat(views[:n_augmentations], dim=0)


def _medical_memo_aug(image, hflip_p=0.0):
    out = image

    if random.random() < 0.85:
        out = _random_affine(out)
    if hflip_p > 0.0 and random.random() < hflip_p:
        out = torch.flip(out, dims=(3,))
    if random.random() < 0.95:
        out = _random_intensity(out)

    return out.contiguous()


def _random_affine(
    image,
    max_rotate=10.0,
    max_shift=0.05,
    scale_range=(0.9, 1.1),
    max_shear=5.0,
):
    _, _, height, width = image.shape
    angle = math.radians(random.uniform(-max_rotate, max_rotate))
    shear = math.radians(random.uniform(-max_shear, max_shear))
    scale = random.uniform(scale_range[0], scale_range[1])
    inv_scale = 1.0 / max(scale, 1e-6)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    tan_s = math.tan(shear)

    # affine_grid expects output-to-input normalized coordinates.
    matrix = torch.tensor(
        [
            [inv_scale * cos_a, inv_scale * (-sin_a + tan_s * cos_a)],
            [inv_scale * sin_a, inv_scale * (cos_a + tan_s * sin_a)],
        ],
        dtype=image.dtype,
        device=image.device,
    )

    shift_x = random.uniform(-max_shift, max_shift)
    shift_y = random.uniform(-max_shift, max_shift)
    theta = torch.zeros((1, 2, 3), dtype=image.dtype, device=image.device)
    theta[0, :, :2] = matrix
    theta[0, 0, 2] = shift_x * 2.0 * width / max(width - 1, 1)
    theta[0, 1, 2] = shift_y * 2.0 * height / max(height - 1, 1)

    grid = F.affine_grid(theta, image.size(), align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _random_intensity(image):
    unit, low, high = _to_unit_range(image)
    original_unit = unit

    if random.random() < 0.75:
        unit = _random_monotone_curve(unit)
    if random.random() < 0.85:
        unit = _location_scale(unit)
    if random.random() < 0.85:
        gamma = random.uniform(0.7, 1.5)
        unit = torch.pow(torch.clamp(unit, min=1e-5, max=1.0), gamma)
    if random.random() < 0.75:
        contrast = random.uniform(0.8, 1.25)
        brightness = random.uniform(-0.05, 0.05)
        mean = unit.mean(dim=(-2, -1), keepdim=True)
        unit = (unit - mean) * contrast + mean + brightness
    if random.random() < 0.5:
        unit = unit + torch.randn_like(unit) * random.uniform(0.0, 0.03)

    unit = torch.clamp(unit, 0.0, 1.0)
    background = original_unit <= 0.01
    unit = torch.where(background, original_unit, unit)
    return unit * (high - low) + low


def _to_unit_range(image):
    low = image.amin(dim=(-2, -1), keepdim=True)
    high = image.amax(dim=(-2, -1), keepdim=True)
    scale = torch.clamp(high - low, min=1e-6)
    return torch.clamp((image - low) / scale, 0.0, 1.0), low, high


def _random_monotone_curve(unit):
    x_mid = sorted([random.uniform(0.0, 1.0) for _ in range(2)])
    y_mid = sorted([random.uniform(0.0, 1.0) for _ in range(2)])
    xp = torch.tensor([0.0] + x_mid + [1.0], dtype=unit.dtype, device=unit.device)
    yp = torch.tensor([0.0] + y_mid + [1.0], dtype=unit.dtype, device=unit.device)

    bucket = torch.bucketize(unit.contiguous(), xp[1:-1])
    x0 = xp[bucket]
    x1 = xp[bucket + 1]
    y0 = yp[bucket]
    y1 = yp[bucket + 1]
    weight = (unit - x0) / torch.clamp(x1 - x0, min=1e-6)
    return y0 + weight * (y1 - y0)


def _location_scale(unit):
    scale = min(max(random.gauss(1.0, 0.1), 0.9), 1.1)
    location = random.gauss(0.0, 0.5)
    flat = unit.detach().flatten()
    low_q = torch.quantile(flat, 0.20).item()
    high_q = torch.quantile(flat, 0.80).item()
    location = min(max(location, -low_q), 1.0 - high_q)
    return torch.clamp(unit * scale + location, 0.0, 1.0)
