"""Soft shape consistency losses for SAAM-SPMM."""

import torch
import torch.nn.functional as F


def _resize_weight(weight, size):
    if weight.shape[-2:] == size:
        return weight
    return F.interpolate(weight.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)


def soft_dice_consistency(prob_stack, weight):
    """Soft Dice consistency between each view and the mean probability.

    Args:
        prob_stack: [V, B, K, H, W].
        weight: [B, H, W].
    """
    eps = 1e-6
    mean_prob = prob_stack.mean(dim=0).detach()
    weight = _resize_weight(weight, prob_stack.shape[-2:]).clamp_min(0.0)
    w = weight.unsqueeze(1)
    losses = []
    for view_prob in prob_stack:
        intersection = (view_prob * mean_prob * w).sum(dim=(2, 3))
        denom = ((view_prob + mean_prob) * w).sum(dim=(2, 3)).clamp_min(eps)
        dice = (2.0 * intersection + eps) / (denom + eps)
        losses.append(1.0 - dice.mean())
    return torch.stack(losses).mean()


def soft_boundary_map(prob):
    """Compute a differentiable boundary magnitude map from soft probabilities."""
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx + dy


def soft_boundary_consistency(prob_stack, weight):
    """Soft boundary consistency across probability views."""
    boundary_stack = torch.stack([soft_boundary_map(prob) for prob in prob_stack], dim=0)
    mean_boundary = boundary_stack.mean(dim=0).detach()
    weight = _resize_weight(weight, prob_stack.shape[-2:]).clamp_min(0.0)
    loss_map = (boundary_stack - mean_boundary.unsqueeze(0)).abs().mean(dim=2)
    return (loss_map * weight.unsqueeze(0)).sum() / (weight.sum() * prob_stack.size(0)).clamp_min(1e-6)


def shape_consistency_loss(prob_stack, weight):
    """Combined soft Dice and boundary consistency."""
    return soft_dice_consistency(prob_stack, weight) + soft_boundary_consistency(prob_stack, weight)
