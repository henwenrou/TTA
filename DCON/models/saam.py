"""
SAAM: Stability-Aware Alignment Module

This module:
- computes pairwise distances across three views and estimates spatial stability
- selects stable regions with a Top-k rule
- applies selective alignment only on stable anchor-base and anchor-strong regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StabilityAwareAlignmentModule(nn.Module):
    """
    Stability estimation and selective alignment module used by SAAM.

    Args:
        tau: soft stability temperature tau in exp(-d/tau)
        topk_ratio: ratio of stable pixels to select (e.g., 0.3 for top 30%)
        stability_mode: 'mean' or 'max' for aggregating three distances
    """

    def __init__(self, tau=0.3, topk_ratio=0.3, stability_mode='mean'):
        super().__init__()
        self.tau = tau
        self.topk_ratio = topk_ratio
        self.stability_mode = stability_mode

        assert stability_mode in ['mean', 'max'], \
            f"stability_mode must be 'mean' or 'max', got {stability_mode}"

    def compute_pairwise_distance(self, f1, f2):
        """
        Compute the per-pixel cosine distance between two feature maps.

        Args:
            f1, f2: feature maps of shape [B, C, h, w]

        Returns:
            distance: [B, h, w] cosine distance (1 - cos)
        """
        # Normalize features along channel dimension
        f1_norm = F.normalize(f1, dim=1, p=2)  # [B, C, h, w]
        f2_norm = F.normalize(f2, dim=1, p=2)  # [B, C, h, w]

        # Cosine similarity: sum over channel dimension
        cos_sim = (f1_norm * f2_norm).sum(dim=1)  # [B, h, w]

        # Cosine distance: 1 - cos
        distance = 1.0 - cos_sim  # [B, h, w]

        return distance

    def compute_stability(self, f_0, f_1, f_2):
        """
        Compute three-view stability from the three pairwise distances (01/02/12).

        Args:
            f_0: anchor features [B, C, h, w]
            f_1: base-view features [B, C, h, w]
            f_2: strong-view features [B, C, h, w]

        Returns:
            d_stab: stability distance [B, h, w], where smaller means more stable
            d_01, d_02, d_12: pairwise distances [B, h, w]
        """
        # Compute three pairwise distances
        d_01 = self.compute_pairwise_distance(f_0, f_1)  # [B, h, w]
        d_02 = self.compute_pairwise_distance(f_0, f_2)  # [B, h, w]
        d_12 = self.compute_pairwise_distance(f_1, f_2)  # [B, h, w]

        # Aggregate stability distance
        if self.stability_mode == 'mean':
            # More permissive aggregation.
            d_stab = (d_01 + d_02 + d_12) / 3.0
        else:  # 'max'
            # More conservative aggregation.
            d_stab = torch.max(torch.max(d_01, d_02), d_12)

        return d_stab, d_01, d_02, d_12

    def compute_topk_mask(self, d_stab, k_ratio):
        """
        Compute the binary Top-k stable-region mask.

        Args:
            d_stab: stability distance [B, h, w]
            k_ratio: Top-k ratio, e.g. 0.3 keeps the most stable 30%

        Returns:
            topk_mask: binary mask [B, h, w], where 1=stable and 0=unstable
        """
        B, h, w = d_stab.shape

        # Flatten spatial dimensions for Top-k selection
        d_stab_flat = d_stab.view(B, -1)  # [B, h*w]

        # Compute k (number of pixels to select)
        num_pixels = h * w
        k = max(1, int(num_pixels * k_ratio))

        # Get Top-k indices (smallest distances = most stable)
        # topk returns (values, indices)
        _, topk_indices = torch.topk(d_stab_flat, k, dim=1, largest=False, sorted=False)

        # Create binary mask
        topk_mask = torch.zeros_like(d_stab_flat)  # [B, h*w]
        topk_mask.scatter_(1, topk_indices, 1.0)  # Set Top-k positions to 1

        # Reshape back to spatial dimensions
        topk_mask = topk_mask.view(B, h, w)  # [B, h, w]

        return topk_mask

    def compute_alignment_weights(self, d_stab, topk_mask):
        """
        Compute SAAM weights as hard selection times soft reliability.

        Args:
            d_stab: stability distance [B, h, w]
            topk_mask: binary Top-k mask [B, h, w]

        Returns:
            W: SAAM weights [B, h, w]
        """
        # Soft reliability: R = exp(-d_stab / tau)
        R = torch.exp(-d_stab / self.tau)  # [B, h, w]

        # Final SAAM weight: W = S * R (hard × soft)
        W = topk_mask * R  # [B, h, w]

        return W, R

    def forward(self, f_0, f_1, f_2, mask_size):
        """
        Compute stability and return the upsampled SAAM weights.

        Args:
            f_0, f_1, f_2: three-view features [B, C, h, w]
            mask_size: target resolution `(H, W)` for the mask/output

        Returns:
            W_up: upsampled SAAM weights [B, H, W]
            stats: statistics dictionary for logging
        """
        # 1. Compute stability
        d_stab, d_01, d_02, d_12 = self.compute_stability(f_0, f_1, f_2)

        # 2. Compute Top-k mask
        topk_mask = self.compute_topk_mask(d_stab, self.topk_ratio)

        # 3. Compute SAAM weights
        W, R = self.compute_alignment_weights(d_stab, topk_mask)

        # 4. Upsample to mask resolution
        # Use bilinear interpolation to match the paper description.
        W_up = F.interpolate(
            W.unsqueeze(1),  # [B, 1, h, w]
            size=mask_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # [B, H, W]

        R_up = F.interpolate(
            R.unsqueeze(1),
            size=mask_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # [B, H, W]

        # 5. Compute statistics for logging
        with torch.no_grad():
            stats = {
                # Distance statistics
                'd_stab_mean': d_stab.mean().item(),
                'd_stab_p10': torch.quantile(d_stab, 0.1).item(),
                'd_stab_p50': torch.quantile(d_stab, 0.5).item(),
                'd_stab_p90': torch.quantile(d_stab, 0.9).item(),
                'd_01_mean': d_01.mean().item(),
                'd_02_mean': d_02.mean().item(),
                'd_12_mean': d_12.mean().item(),

                # Gate statistics
                'R_mean': R.mean().item(),
                'R_p10': torch.quantile(R, 0.1).item(),
                'R_p50': torch.quantile(R, 0.5).item(),
                'R_p90': torch.quantile(R, 0.9).item(),
                'W_mean': W.mean().item(),

                # Top-k statistics
                'topk_selected_ratio': topk_mask.mean().item(),  # Should be close to topk_ratio.
            }

        return W_up, stats


def compute_saam_loss(q_0, q_1, q_2, mask, W, lambda_01=1.0, lambda_02=1.0):
    """
    Compute the SAAM alignment loss `L_01 + L_02`.

    Alignment is enforced only inside stable regions weighted by `W`.

    Args:
        q_0, q_1, q_2: alignment features [B, C', H, W]
        mask: object mask [B, H, W] or [B, 1, H, W]
        W: SAAM weights [B, H, W]
        lambda_01: anchor-base alignment weight
        lambda_02: anchor-strong alignment weight

    Returns:
        L_align: total alignment loss
        L_01: anchor-base alignment loss
        L_02: anchor-strong alignment loss
        effective_pixels: effective pixel count, useful for checking over-strict gating
    """
    # Ensure mask is [B, H, W]
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    # Normalize alignment features
    q_0_norm = F.normalize(q_0, dim=1, p=2)  # [B, C', H, W]
    q_1_norm = F.normalize(q_1, dim=1, p=2)
    q_2_norm = F.normalize(q_2, dim=1, p=2)

    # Compute cosine similarity
    cos_01 = (q_0_norm * q_1_norm).sum(dim=1)  # [B, H, W]
    cos_02 = (q_0_norm * q_2_norm).sum(dim=1)  # [B, H, W]

    # Alignment distance: 1 - cos
    dist_01 = 1.0 - cos_01  # [B, H, W]
    dist_02 = 1.0 - cos_02  # [B, H, W]

    # Apply foreground filter and SAAM weights: M * W
    weight = mask * W  # [B, H, W]

    # Weighted alignment loss
    # L_01 = sum(M * W * dist_01) / (sum(M * W) + eps)
    eps = 1e-8

    numerator_01 = (weight * dist_01).sum()
    numerator_02 = (weight * dist_02).sum()
    denominator = weight.sum() + eps

    L_01 = numerator_01 / denominator
    L_02 = numerator_02 / denominator

    # Total alignment loss
    L_align = lambda_01 * L_01 + lambda_02 * L_02

    # Effective pixels, useful as a sanity check.
    effective_pixels = weight.sum().item()

    return L_align, L_01, L_02, effective_pixels


# ========== Helper for exp_trainer integration ==========

def compute_saam_weights_and_loss(
    f_0, f_1, f_2,  # encoder features [B, C, h, w]
    q_0, q_1, q_2,  # proj features for alignment [B, C', H, W]
    mask,           # object mask [B, H, W]
    tau=0.3,
    topk_ratio=0.3,
    lambda_01=1.0,
    lambda_02=1.0,
    stability_mode='mean',
    stability_module=None
):
    """
    Compute SAAM weights and the alignment loss end-to-end.

    This convenience wrapper is used by `exp_trainer` to:
    1. compute SAAM weights
    2. compute the alignment loss

    Args:
        f_0, f_1, f_2: encoder features used for stability estimation
        q_0, q_1, q_2: projected features used for alignment
        mask: object mask
        tau, topk_ratio, lambda_01, lambda_02: hyperparameters
        stability_mode: 'mean' or 'max'
        stability_module: optional reusable StabilityAwareAlignmentModule instance

    Returns:
        loss_dict: {
            'L_align': total alignment loss,
            'L_01': anchor-base loss,
            'L_02': anchor-strong loss
        }
        stats: statistics dictionary
    """
    # Get the mask resolution.
    if mask.dim() == 4:
        H, W = mask.shape[2], mask.shape[3]
    else:
        H, W = mask.shape[1], mask.shape[2]

    # Create or reuse the stability module.
    if stability_module is None:
        stability_module = StabilityAwareAlignmentModule(
            tau=tau,
            topk_ratio=topk_ratio,
            stability_mode=stability_mode
        ).cuda()

    # Compute SAAM weights.
    W_up, stats = stability_module(f_0, f_1, f_2, mask_size=(H, W))

    # Compute the alignment loss.
    L_align, L_01, L_02, effective_pixels = compute_saam_loss(
        q_0, q_1, q_2, mask, W_up, lambda_01, lambda_02
    )

    # Add effective_pixels to the returned statistics.
    stats['effective_pixels'] = effective_pixels

    # Build the loss dictionary.
    loss_dict = {
        'L_align': L_align,
        'L_01': L_01,
        'L_02': L_02
    }

    return loss_dict, stats
