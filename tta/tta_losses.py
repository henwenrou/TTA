"""Losses and reliability scores for multi-view segmentation TTA."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def pixel_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return per-pixel entropy for a probability tensor ``[B, C, H, W]``."""

    return -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=1)


def entropy_minimization_loss(logits: torch.Tensor) -> torch.Tensor:
    """Standard target entropy minimization loss for source-free TTA."""

    return pixel_entropy(torch.softmax(logits, dim=1)).mean()


def compute_reliability_scores(
    probs_views: torch.Tensor,
    min_area: int = 20,
    beta_region: float = 10.0,
    beta_area: float = 4.0,
    beta_presence: float = 4.0,
) -> Dict[str, torch.Tensor]:
    """Compute region and class reliability from inverse-mapped probabilities.

    Args:
        probs_views: Probability maps in original coordinates with shape
            ``[V, B, C, H, W]``.
        min_area: Predicted-pixel threshold used to define class presence.
        beta_region: Exponential decay for region-level probability variance.
        beta_area: Exponential decay for normalized class-area variance.
        beta_presence: Exponential decay for class-presence disagreement.

    Returns:
        Dictionary containing ``mean_prob``, ``var_prob``, ``R_region`` with
        shape ``[B, 1, H, W]``, and ``R_class`` with shape ``[B, C]``.
    """

    if probs_views.dim() != 5:
        raise ValueError(f"Expected [V,B,C,H,W], got {tuple(probs_views.shape)}")
    views, batch, classes, height, width = probs_views.shape
    mean_prob = probs_views.mean(dim=0)
    var_prob = probs_views.var(dim=0, unbiased=False)
    var_mean = var_prob.mean(dim=1, keepdim=True)
    r_region = torch.exp(-float(beta_region) * var_mean).clamp(0.0, 1.0)

    hard = probs_views.argmax(dim=2)
    area_values = []
    presence_values = []
    norm = float(height * width)
    for cls_idx in range(classes):
        area = (hard == cls_idx).float().flatten(2).sum(dim=2)
        area_values.append(area)
        presence_values.append((area > float(min_area)).float())

    areas = torch.stack(area_values, dim=2)
    presences = torch.stack(presence_values, dim=2)
    area_var_norm = areas.var(dim=0, unbiased=False) / max(norm * norm, 1.0)
    presence_var = presences.var(dim=0, unbiased=False)
    r_class = torch.exp(
        -float(beta_area) * area_var_norm - float(beta_presence) * presence_var
    ).clamp(0.0, 1.0)

    return {
        "mean_prob": mean_prob,
        "var_prob": var_prob,
        "R_region": r_region.detach(),
        "R_class": r_class.detach(),
        "area_var_norm": area_var_norm.detach(),
        "presence_var": presence_var.detach(),
    }


def _class_weight_tensor(
    r_class: Optional[torch.Tensor],
    classes: int,
    use_class_gate: bool = True,
    tau_class: float = 0.0,
) -> torch.Tensor:
    """Convert class reliability to a broadcastable class-weight tensor."""

    if r_class is None or not use_class_gate:
        return torch.ones((1, classes, 1, 1), device=r_class.device if r_class is not None else None)
    weights = r_class
    if tau_class > 0.0:
        weights = torch.where(weights >= tau_class, weights, torch.zeros_like(weights))
    return weights[:, :, None, None]


def multiview_consistency_loss(
    probs_views: torch.Tensor,
    pseudo: Optional[torch.Tensor] = None,
    region_weight: Optional[torch.Tensor] = None,
    class_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """KL consistency between each inverse-mapped view and a mean pseudo-label."""

    if pseudo is None:
        pseudo = probs_views.mean(dim=0).detach()
    else:
        pseudo = pseudo.detach()

    total = probs_views.new_zeros(())
    for idx in range(probs_views.shape[0]):
        prob = probs_views[idx].clamp_min(eps)
        kl = prob * (prob.log() - pseudo.clamp_min(eps).log())
        if class_weight is not None:
            kl = kl * class_weight
        kl = kl.sum(dim=1, keepdim=True)
        if region_weight is not None:
            kl = kl * region_weight
        total = total + kl.mean()
    return total / float(probs_views.shape[0])


def reliability_gated_mv_loss(
    probs_views: torch.Tensor,
    reliability: Dict[str, torch.Tensor],
    lambda_ent: float = 1.0,
    lambda_cons: float = 1.0,
    use_region_gate: bool = True,
    use_class_gate: bool = True,
    tau_class: float = 0.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute reliability-gated entropy and multi-view consistency losses."""

    mean_prob = reliability["mean_prob"]
    classes = mean_prob.shape[1]
    region_weight = reliability["R_region"] if use_region_gate else None
    class_weight = _class_weight_tensor(
        reliability.get("R_class"),
        classes,
        use_class_gate=use_class_gate,
        tau_class=tau_class,
    ).to(mean_prob.device)

    ent_terms = -mean_prob.clamp_min(eps) * mean_prob.clamp_min(eps).log()
    ent_terms = ent_terms * class_weight
    ent_map = ent_terms.sum(dim=1, keepdim=True)
    if region_weight is not None:
        ent_map = ent_map * region_weight
    loss_ent = ent_map.mean()

    loss_cons = multiview_consistency_loss(
        probs_views,
        pseudo=mean_prob.detach(),
        region_weight=region_weight,
        class_weight=class_weight,
        eps=eps,
    )
    total = float(lambda_ent) * loss_ent + float(lambda_cons) * loss_cons
    return {"loss": total, "loss_ent": loss_ent.detach(), "loss_cons": loss_cons.detach()}
