"""SAAM-SPMM test-time adaptation for DCON medical segmentation.

SAAM-SPMM is independent from the existing SM-PPM adapter. It adapts online from
target images only and may use pre-exported source prototypes. No raw source
images are accessed at test time.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.prototype import (
    OnlinePrototypeMemory,
    compute_target_prototypes,
    load_source_prototypes,
    prototype_matching_loss,
)
from utils.shape_consistency import shape_consistency_loss
from utils.stability import (
    build_stable_mask,
    build_weak_views,
    compute_stability_map,
    stability_weighted_consistency,
    stability_weighted_entropy,
)


logger = logging.getLogger(__name__)


def forward_logits_features(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        logits = output[0]
        feature = output[1] if len(output) > 1 else None
    else:
        logits = output
        feature = None
    if feature is None:
        raise RuntimeError("SAAM-SPMM requires the model to return (logits, bottleneck_feature).")
    if feature.dim() != 4:
        raise ValueError(f"SAAM-SPMM expects bottleneck feature [B,D,h,w], got {tuple(feature.shape)}")
    return logits, feature


def configure_model_for_saam_spmm(model, update_scope="bn_affine"):
    """Configure trainable parameters for SAAM-SPMM."""
    update_scope = str(update_scope)
    names = []
    if update_scope == "bn_affine":
        model.train()
        model.requires_grad_(False)
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                module.requires_grad_(True)
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        names.append(f"{module_name}.{param_name}")
        params = [param for param in model.parameters() if param.requires_grad]
    elif update_scope == "all":
        model.train()
        model.requires_grad_(True)
        params = [param for param in model.parameters() if param.requires_grad]
        names = [name for name, param in model.named_parameters() if param.requires_grad]
    else:
        raise ValueError(f"Unknown SAAM-SPMM update scope: {update_scope}")
    return params, names


class SAAMSPMMAdapter:
    """Stability-aware source-prototype and target-memory matching adapter."""

    def __init__(
        self,
        model,
        optimizer,
        device,
        num_classes,
        steps=1,
        num_views=5,
        use_saam=True,
        use_stable_mask=True,
        use_source_anchor=True,
        use_shape_consistency=True,
        source_prototype_path=None,
        saam_metric="variance",
        stable_threshold=None,
        stable_topk_percent=0.3,
        unstable_weight=0.1,
        lambda_ent=1.0,
        lambda_proto=1.0,
        lambda_shape=0.1,
        lambda_cons=1.0,
        proto_momentum=0.9,
        proto_loss="cosine",
        log_interval=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.num_views = int(num_views)
        self.use_saam = bool(use_saam)
        self.use_stable_mask = bool(use_stable_mask)
        self.use_source_anchor = bool(use_source_anchor)
        self.use_shape_consistency = bool(use_shape_consistency)
        self.saam_metric = str(saam_metric)
        self.stable_threshold = stable_threshold
        self.stable_topk_percent = float(stable_topk_percent)
        self.unstable_weight = float(unstable_weight)
        self.lambda_ent = float(lambda_ent)
        self.lambda_proto = float(lambda_proto)
        self.lambda_shape = float(lambda_shape)
        self.lambda_cons = float(lambda_cons)
        self.proto_momentum = float(proto_momentum)
        self.proto_loss = str(proto_loss)
        self.log_interval = int(log_interval)
        self.num_forwards = 0
        self.adaptation_steps = 0
        self.memory = None
        self.last_losses = {}

        if self.steps < 1:
            raise ValueError("SAAM-SPMM steps must be >= 1")
        if self.num_views < 1:
            raise ValueError("SAAM-SPMM num_views must be >= 1")
        if not 0.0 <= self.unstable_weight <= 1.0:
            raise ValueError("unstable_weight must be in [0, 1]")
        if self.lambda_ent < 0 or self.lambda_proto < 0 or self.lambda_shape < 0 or self.lambda_cons < 0:
            raise ValueError("SAAM-SPMM loss weights must be non-negative")
        if not 0.0 <= self.proto_momentum <= 1.0:
            raise ValueError("proto_momentum must be in [0, 1]")

        self.source_payload = None
        self.source_proto = None
        if self.use_source_anchor:
            self.source_payload = load_source_prototypes(source_prototype_path, device)
            self.source_proto = self.source_payload["prototype"]
            if self.source_proto.size(0) != self.num_classes:
                raise ValueError(
                    f"Source prototype class count {self.source_proto.size(0)} "
                    f"does not match num_classes={self.num_classes}"
                )
            if "count" in self.source_payload:
                self.source_valid = self.source_payload["count"].to(device) > 0
            else:
                self.source_valid = torch.ones(self.num_classes, device=device, dtype=torch.bool)
        else:
            self.source_valid = torch.zeros(self.num_classes, device=device, dtype=torch.bool)

        self.updated_parameter_count = sum(
            p.numel() for group in self.optimizer.param_groups for p in group["params"]
        )
        logger.info(self.feature_summary())

    def feature_summary(self):
        source_shape = None if self.source_proto is None else tuple(self.source_proto.shape)
        return (
            "SAAM-SPMM enabled:\n"
            f"  steps={self.steps} num_views={self.num_views} saam_metric={self.saam_metric}\n"
            f"  use_saam={self.use_saam} use_stable_mask={self.use_stable_mask} "
            f"use_source_anchor={self.use_source_anchor} use_shape_consistency={self.use_shape_consistency}\n"
            f"  stable_threshold={self.stable_threshold} stable_topk_percent={self.stable_topk_percent} "
            f"unstable_weight={self.unstable_weight}\n"
            f"  lambda_ent={self.lambda_ent} lambda_proto={self.lambda_proto} "
            f"lambda_shape={self.lambda_shape} lambda_cons={self.lambda_cons}\n"
            f"  proto_loss={self.proto_loss} proto_momentum={self.proto_momentum} "
            f"source_proto_shape={source_shape}\n"
            f"  updated_parameter_count={self.updated_parameter_count}"
        )

    def _forward_views(self, images):
        """Forward V weak views.

        Args:
            images: target images [B, C, H, W].

        Returns:
            views: [V, B, C, H, W]
            logits_list: list of V logits [B, K, H, W]
            feature_list: list of V bottleneck features [B, D, h, w]
            prob_stack: [V, B, K, H, W]
        """
        views = build_weak_views(images, self.num_views)
        logits_list = []
        feature_list = []
        for view_idx in range(views.size(0)):
            logits, feature = forward_logits_features(self.model, views[view_idx])
            if logits.shape[-2:] != images.shape[-2:]:
                logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
            logits_list.append(logits)
            feature_list.append(feature)
        prob_stack = torch.stack([F.softmax(logits, dim=1) for logits in logits_list], dim=0)
        return views, logits_list, feature_list, prob_stack

    def _build_weight(self, prob_stack):
        if self.use_saam:
            stability = compute_stability_map(prob_stack, metric=self.saam_metric)
        else:
            stability = prob_stack.new_ones(prob_stack.size(1), prob_stack.size(3), prob_stack.size(4))

        if self.use_stable_mask:
            stable_mask = build_stable_mask(
                stability,
                stable_threshold=self.stable_threshold,
                stable_topk_percent=self.stable_topk_percent,
            )
            weight = stable_mask + (1.0 - stable_mask) * self.unstable_weight
            weight = weight * stability
        else:
            stable_mask = torch.ones_like(stability)
            weight = stability
        return stability, stable_mask, weight.clamp_min(0.0)

    def _ensure_memory(self, feature):
        if self.memory is None:
            self.memory = OnlinePrototypeMemory(
                num_classes=self.num_classes,
                feature_dim=feature.size(1),
                momentum=self.proto_momentum,
                device=feature.device,
            )
        if self.memory.feature_dim != feature.size(1):
            raise ValueError(
                f"SAAM-SPMM memory feature_dim={self.memory.feature_dim} but current feature_dim={feature.size(1)}"
            )

    def _proto_loss(self, target_proto, valid):
        zero = target_proto.sum() * 0.0
        losses = []
        if self.use_source_anchor:
            if self.source_proto.size(1) != target_proto.size(1):
                raise ValueError(
                    f"Source prototype feature_dim={self.source_proto.size(1)} "
                    f"does not match target feature_dim={target_proto.size(1)}"
                )
            losses.append(prototype_matching_loss(target_proto, self.source_proto, valid & self.source_valid, self.proto_loss))
        if self.memory is not None and torch.any(self.memory.valid & valid):
            mem_valid = self.memory.valid & valid
            losses.append(prototype_matching_loss(target_proto, self.memory.prototypes.detach(), mem_valid, self.proto_loss))
        if len(losses) == 0:
            return zero
        return torch.stack(losses).mean()

    @torch.enable_grad()
    def _adapt_once(self, images):
        self.optimizer.zero_grad(set_to_none=True)
        views, logits_list, feature_list, prob_stack = self._forward_views(images)
        main_logits = logits_list[0]
        main_feature = feature_list[0]
        main_prob = prob_stack[0]

        stability, stable_mask, weight = self._build_weight(prob_stack)
        pseudo_label = main_prob.detach().argmax(dim=1)
        target_proto, proto_valid = compute_target_prototypes(
            main_feature,
            pseudo_label,
            stable_mask,
            self.num_classes,
        )
        self._ensure_memory(main_feature)

        loss_entropy = stability_weighted_entropy(main_logits, weight)
        loss_cons = stability_weighted_consistency(prob_stack, weight)
        loss_proto = self._proto_loss(target_proto, proto_valid)
        if self.use_shape_consistency:
            loss_shape = shape_consistency_loss(prob_stack, weight)
        else:
            loss_shape = main_logits.sum() * 0.0

        loss_total = (
            self.lambda_ent * loss_entropy
            + self.lambda_proto * loss_proto
            + self.lambda_shape * loss_shape
            + self.lambda_cons * loss_cons
        )
        loss_total.backward()
        self.optimizer.step()
        self.adaptation_steps += 1
        memory_updated = self.memory.update(target_proto, proto_valid)

        with torch.no_grad():
            stats = {
                "loss_total": loss_total.detach(),
                "loss_entropy": loss_entropy.detach(),
                "loss_proto": loss_proto.detach(),
                "loss_shape": loss_shape.detach(),
                "loss_cons": loss_cons.detach(),
                "stability_mean": stability.detach().mean(),
                "stability_min": stability.detach().min(),
                "stability_max": stability.detach().max(),
                "stable_ratio": stable_mask.detach().mean(),
                "weight_mean": weight.detach().mean(),
                "proto_valid_count": proto_valid.detach().float().sum(),
                "memory_updated": memory_updated,
                "views_shape": tuple(views.shape),
                "logits_shape": tuple(main_logits.shape),
                "feature_shape": tuple(main_feature.shape),
                "prob_stack_shape": tuple(prob_stack.shape),
                "stability_shape": tuple(stability.shape),
                "stable_mask_shape": tuple(stable_mask.shape),
                "target_proto_shape": tuple(target_proto.shape),
            }
        return stats

    def _log_step(self, stats, step_idx):
        should_log = self.log_interval > 0 and ((self.num_forwards + 1) % self.log_interval == 0)
        if not should_log:
            return
        logger.info(
            "saam_spmm target_batch=%d step=%d/%d "
            "loss_total=%.6f loss_entropy=%.6f loss_proto=%.6f loss_shape=%.6f loss_cons=%.6f "
            "stable_ratio=%.6f stability_mean=%.6f stability_min=%.6f stability_max=%.6f "
            "weight_mean=%.6f proto_valid=%.1f memory_updated=%s "
            "views_shape=%s logits_shape=%s feature_shape=%s prob_stack_shape=%s "
            "stability_shape=%s stable_mask_shape=%s target_proto_shape=%s",
            self.num_forwards + 1,
            step_idx + 1,
            self.steps,
            float(stats["loss_total"].item()),
            float(stats["loss_entropy"].item()),
            float(stats["loss_proto"].item()),
            float(stats["loss_shape"].item()),
            float(stats["loss_cons"].item()),
            float(stats["stable_ratio"].item()),
            float(stats["stability_mean"].item()),
            float(stats["stability_min"].item()),
            float(stats["stability_max"].item()),
            float(stats["weight_mean"].item()),
            float(stats["proto_valid_count"].item()),
            stats["memory_updated"],
            stats["views_shape"],
            stats["logits_shape"],
            stats["feature_shape"],
            stats["prob_stack_shape"],
            stats["stability_shape"],
            stats["stable_mask_shape"],
            stats["target_proto_shape"],
        )

    @torch.enable_grad()
    def forward(self, target_img):
        self.model.train()
        images = target_img.to(self.device, non_blocking=True).float()
        stats = None
        for step_idx in range(self.steps):
            stats = self._adapt_once(images)
            self._log_step(stats, step_idx)

        self.model.eval()
        with torch.no_grad():
            logits, _ = forward_logits_features(self.model, images)
            if logits.shape[-2:] != images.shape[-2:]:
                logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        self.num_forwards += 1
        if stats is not None:
            self.last_losses = {
                "saam_spmm_loss_total": float(stats["loss_total"].item()),
                "saam_spmm_loss_entropy": float(stats["loss_entropy"].item()),
                "saam_spmm_loss_proto": float(stats["loss_proto"].item()),
                "saam_spmm_loss_shape": float(stats["loss_shape"].item()),
                "saam_spmm_loss_cons": float(stats["loss_cons"].item()),
                "saam_spmm_stable_ratio": float(stats["stable_ratio"].item()),
                "saam_spmm_stability_mean": float(stats["stability_mean"].item()),
                "saam_spmm_weight_mean": float(stats["weight_mean"].item()),
                "saam_spmm_proto_valid_count": float(stats["proto_valid_count"].item()),
                "saam_spmm_adaptation_steps": float(self.adaptation_steps),
                "saam_spmm_updated_params": float(self.updated_parameter_count),
            }
        return logits
