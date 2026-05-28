"""Source-prototype logit calibration for DCON medical segmentation.

This is a frozen-model post-processing baseline. It performs one target forward,
computes target-feature distances to offline source prototypes, and subtracts a
prototype-distance penalty from logits. It does not adapt model parameters.
"""

import logging

import torch
import torch.nn.functional as F

from utils.prototype import load_source_prototypes


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
        raise RuntimeError(
            "SourcePrototypeLogitCalibrationAdapter requires model(images) to "
            "return (logits, bottleneck_feature), matching SAAM-SPMM feature "
            "extraction. No feature hook fallback is configured."
        )
    if not torch.is_tensor(feature) or feature.dim() != 4:
        raise ValueError(
            f"Source prototype calibration expects bottleneck feature [B,D,h,w], "
            f"got {type(feature)} with shape {getattr(feature, 'shape', None)}"
        )
    if not torch.is_tensor(logits) or logits.dim() != 4:
        raise ValueError(
            f"Source prototype calibration expects logits [B,K,H,W], "
            f"got {type(logits)} with shape {getattr(logits, 'shape', None)}"
        )
    return logits, feature


class SourcePrototypeLogitCalibrationAdapter:
    """Frozen source-prototype distance calibration for segmentation logits."""

    def __init__(
        self,
        model,
        device,
        num_classes,
        source_prototype_path,
        lambda_proto_calib=0.1,
        min_count=10,
        use_var=True,
        norm="minmax",
        feature_norm=False,
        eps=1e-6,
        log_interval=1,
    ):
        self.model = model
        self.device = device
        self.num_classes = int(num_classes)
        self.lambda_proto_calib = float(lambda_proto_calib)
        self.min_count = float(min_count)
        self.use_var = bool(use_var)
        self.norm = str(norm).lower()
        self.feature_norm = bool(feature_norm)
        self.eps = float(eps)
        self.log_interval = int(log_interval)
        self.num_forwards = 0
        self.last_losses = {}

        if self.lambda_proto_calib < 0.0:
            raise ValueError(f"proto_calib_lambda must be >= 0, got {self.lambda_proto_calib}")
        if self.min_count < 0.0:
            raise ValueError(f"proto_calib_min_count must be >= 0, got {self.min_count}")
        if self.norm not in {"minmax", "zscore", "none"}:
            raise ValueError(f"proto_calib_norm must be one of minmax/zscore/none, got {self.norm}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

        self.source_payload = load_source_prototypes(source_prototype_path, device)
        self.source_proto = self.source_payload["prototype"].to(device).float()
        self.source_var = self.source_payload.get("var", None)
        if self.source_var is not None:
            self.source_var = self.source_var.to(device).float()
        self.source_count = self.source_payload.get("count", None)
        if self.source_count is None:
            self.source_count = torch.ones(self.source_proto.size(0), device=device)
        else:
            self.source_count = self.source_count.to(device).float()

        if self.source_proto.dim() != 2:
            raise ValueError(f"Source prototype must have shape [K,D], got {tuple(self.source_proto.shape)}")
        if self.source_proto.size(0) != self.num_classes:
            raise ValueError(
                f"Source prototype class count {self.source_proto.size(0)} does not match "
                f"num_classes={self.num_classes}"
            )
        if self.source_count.numel() != self.num_classes:
            raise ValueError(
                f"Source count length {self.source_count.numel()} does not match num_classes={self.num_classes}"
            )
        if self.source_var is not None and self.source_var.shape != self.source_proto.shape:
            raise ValueError(
                f"Source var shape {tuple(self.source_var.shape)} does not match prototype "
                f"shape {tuple(self.source_proto.shape)}"
            )

        self.reliability = self._compute_reliability()
        logger.info(self.feature_summary())

    def feature_summary(self):
        return (
            "Source-Prototype Logit Calibration enabled:\n"
            "  frozen_model=True optimizer=False backward=False source_loader=False\n"
            f"  lambda={self.lambda_proto_calib} min_count={self.min_count} "
            f"use_var={self.use_var} norm={self.norm} feature_norm={self.feature_norm}\n"
            f"  source_proto_shape={tuple(self.source_proto.shape)} "
            f"has_var={self.source_var is not None} valid_classes={int((self.reliability > 0).sum().item())}"
        )

    def _compute_reliability(self):
        count_w = torch.log1p(self.source_count.clamp_min(0.0))
        count_w = count_w / count_w.max().clamp_min(self.eps)

        if self.use_var and self.source_var is not None:
            var_mean = self.source_var.mean(dim=1).clamp_min(0.0)
            var_w = 1.0 / (var_mean + self.eps)
            var_w = var_w / var_w.max().clamp_min(self.eps)
        else:
            var_w = torch.ones_like(count_w)

        reliability = count_w * var_w
        reliability = torch.where(
            self.source_count > self.min_count,
            reliability,
            torch.zeros_like(reliability),
        )
        return reliability.clamp(0.0, 1.0)

    def _distance_map(self, feature):
        if feature.size(1) != self.source_proto.size(1):
            raise ValueError(
                f"Target bottleneck feature_dim={feature.size(1)} does not match source "
                f"prototype feature_dim={self.source_proto.size(1)}. Check that prototypes "
                "were exported from the same model/checkpoint architecture."
            )

        proto = self.source_proto
        feat = feature
        if self.feature_norm:
            feat = F.normalize(feat, p=2, dim=1)
            proto = F.normalize(proto, p=2, dim=1)

        diff2 = (feat.unsqueeze(1) - proto.view(1, proto.size(0), proto.size(1), 1, 1)).pow(2)
        if self.use_var and self.source_var is not None:
            denom = self.source_var.view(1, self.source_var.size(0), self.source_var.size(1), 1, 1)
            diff2 = diff2 / denom.clamp_min(self.eps)
        return diff2.mean(dim=2)

    def _normalize_distance(self, dist):
        if self.norm == "none":
            return dist

        flat = dist.reshape(dist.size(0), -1)
        if self.norm == "minmax":
            min_v = flat.min(dim=1)[0].view(-1, 1, 1, 1)
            max_v = flat.max(dim=1)[0].view(-1, 1, 1, 1)
            return (dist - min_v) / (max_v - min_v).clamp_min(self.eps)

        mean = flat.mean(dim=1).view(-1, 1, 1, 1)
        std = flat.std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        return (dist - mean) / std.clamp_min(self.eps)

    def _log_batch(self, stats):
        if self.log_interval <= 0 or self.num_forwards % self.log_interval != 0:
            return
        logger.info(
            "source_proto_calib batch=%d lambda=%.6f valid_classes=%d "
            "reliability=%s mean_distance=%s logits_before_mean=%.6f "
            "logits_before_std=%.6f logits_after_mean=%.6f logits_after_std=%.6f",
            self.num_forwards,
            self.lambda_proto_calib,
            int(stats["valid_classes"]),
            stats["reliability"],
            stats["mean_distance"],
            float(stats["logits_before_mean"]),
            float(stats["logits_before_std"]),
            float(stats["logits_after_mean"]),
            float(stats["logits_after_std"]),
        )

    @torch.no_grad()
    def forward(self, target_img):
        self.model.eval()
        images = target_img.to(self.device, non_blocking=True).float()
        logits, feature = forward_logits_features(self.model, images)

        if logits.size(1) != self.num_classes:
            raise ValueError(
                f"Logit class count {logits.size(1)} does not match num_classes={self.num_classes}"
            )

        dist = self._distance_map(feature)
        mean_distance = dist.mean(dim=(0, 2, 3))
        if dist.shape[-2:] != logits.shape[-2:]:
            dist = F.interpolate(dist, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        norm_dist = self._normalize_distance(dist)

        penalty = self.lambda_proto_calib * self.reliability.view(1, -1, 1, 1) * norm_dist
        calibrated_logits = logits - penalty.to(logits)

        self.num_forwards += 1
        stats = {
            "proto_calib_lambda": self.lambda_proto_calib,
            "valid_classes": int((self.reliability > 0).sum().item()),
            "reliability": [round(float(x), 6) for x in self.reliability.detach().cpu().tolist()],
            "mean_distance": [round(float(x), 6) for x in mean_distance.detach().cpu().tolist()],
            "logits_before_mean": float(logits.detach().mean().item()),
            "logits_before_std": float(logits.detach().std(unbiased=False).item()),
            "logits_after_mean": float(calibrated_logits.detach().mean().item()),
            "logits_after_std": float(calibrated_logits.detach().std(unbiased=False).item()),
        }
        self.last_losses = {
            "source_proto_calib_lambda": float(stats["proto_calib_lambda"]),
            "source_proto_calib_valid_classes": float(stats["valid_classes"]),
            "source_proto_calib_logits_before_mean": stats["logits_before_mean"],
            "source_proto_calib_logits_before_std": stats["logits_before_std"],
            "source_proto_calib_logits_after_mean": stats["logits_after_mean"],
            "source_proto_calib_logits_after_std": stats["logits_after_std"],
        }
        self._log_batch(stats)
        return calibrated_logits
