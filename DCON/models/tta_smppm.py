"""Medical SM-PPM test-time adaptation and ablations for DCON.

This file implements a lightweight tensor-space SM path for DCON: target
intensity statistics are transferred onto source images with AdaIN before the
source-supervised adaptation update. PPM uses target bottleneck prototypes to
build source-pixel weights.
"""

import logging
from copy import deepcopy

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


SMPPM_ABLATION_MODES = (
    "full",
    "source_ce_only",
    "sm_ce",
    "ppm_ce",
    "source_free_proto",
)
STYLE_MIXING_AVAILABLE = True


def _calc_mean_std(x, eps=1e-5):
    if x.dim() != 4:
        raise ValueError(f"SM style mixing expects [B, C, H, W], got {tuple(x.shape)}")
    var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return mean, (var + eps).sqrt()


def _match_batch(stat, batch_size):
    if stat.size(0) == batch_size:
        return stat
    if stat.size(0) == 1:
        return stat.expand(batch_size, -1, -1, -1)
    repeat = (batch_size + stat.size(0) - 1) // stat.size(0)
    return stat.repeat(repeat, 1, 1, 1)[:batch_size]


def medical_adain_style_mix(source_img, target_img, alpha=1.0, eps=1e-5):
    """Transfer target tensor statistics onto source images without moving labels."""
    target_img = target_img.detach()
    src_mean, src_std = _calc_mean_std(source_img, eps)
    tgt_mean, tgt_std = _calc_mean_std(target_img, eps)
    tgt_mean = _match_batch(tgt_mean, source_img.size(0))
    tgt_std = _match_batch(tgt_std, source_img.size(0))
    source_norm = (source_img - src_mean) / src_std
    stylized = source_norm * tgt_std + tgt_mean
    alpha = float(alpha)
    return alpha * stylized + (1.0 - alpha) * source_img


def forward_logits_features(model, images):
    """Return segmentation logits and the DCON bottleneck feature."""
    output = model(images)
    if isinstance(output, (tuple, list)):
        logits = output[0]
        feature = output[1] if len(output) > 1 else None
        return logits, feature
    return output, None


class SMPPMAdapter:
    """SM-PPM adapter for DCON medical segmentation and ablations.

    For each target batch in the original/full path:
      1. Extract target bottleneck features and pool them into patch prototypes.
      2. Fetch one labeled source batch.
      3. Update the segmentation model on source labels with SM-PPM pixel
         weights from source entropy and source-target feature similarity.
      4. Predict the target batch with the updated model.

    ``source_free_proto`` never reads a source batch. It adapts only from target
    pseudo-label confidence, target class prototypes, masked entropy, and
    prototype compactness.
    """

    def __init__(
        self,
        model,
        optimizer,
        source_loader,
        device,
        num_classes,
        steps=1,
        patch_size=8,
        feature_size=32,
        episodic=False,
        segmentation_criterion=None,
        ablation_mode="full",
        source_free_tau=0.7,
        source_free_entropy_threshold=None,
        source_free_lambda_proto=1.0,
        source_free_entropy_weight=1.0,
        style_alpha=1.0,
        log_interval=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.ablation_mode = str(ablation_mode)
        if self.ablation_mode not in SMPPM_ABLATION_MODES:
            raise ValueError(
                f"Unknown SM-PPM ablation mode: {self.ablation_mode}. "
                f"Expected one of {SMPPM_ABLATION_MODES}."
            )
        self.uses_source_loader = self.ablation_mode in {"full", "source_ce_only", "sm_ce", "ppm_ce"}
        self.uses_source_label = self.uses_source_loader
        self.uses_sm = self.ablation_mode in {"full", "sm_ce"}
        self.uses_ppm = self.ablation_mode in {"full", "ppm_ce"}
        self.uses_target_only_loss = self.ablation_mode == "source_free_proto"

        if self.uses_source_loader and source_loader is None:
            raise ValueError(
                f"SM-PPM ablation mode {self.ablation_mode} requires a labeled source_loader."
            )
        if self.uses_source_loader and segmentation_criterion is None:
            raise ValueError(
                f"SM-PPM ablation mode {self.ablation_mode} requires a weighted segmentation_criterion."
            )

        self.source_loader = source_loader
        self.source_iter = iter(source_loader) if self.uses_source_loader else None
        self.device = device
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.patch_size = int(patch_size)
        self.feature_size = int(feature_size)
        self.episodic = bool(episodic)
        self.segmentation_criterion = segmentation_criterion
        self.source_free_tau = float(source_free_tau)
        self.source_free_entropy_threshold = source_free_entropy_threshold
        if self.source_free_entropy_threshold is not None:
            self.source_free_entropy_threshold = float(self.source_free_entropy_threshold)
        self.source_free_lambda_proto = float(source_free_lambda_proto)
        self.source_free_entropy_weight = float(source_free_entropy_weight)
        self.style_alpha = float(style_alpha)
        self.log_interval = int(log_interval)
        self.model_state = deepcopy(model.state_dict()) if self.episodic else None
        self.optimizer_state = deepcopy(optimizer.state_dict()) if self.episodic else None
        self.last_losses = {}
        self.num_forwards = 0

        if self.steps < 1:
            raise ValueError(f"SM-PPM steps must be >= 1, got {self.steps}.")
        if self.patch_size < 1:
            raise ValueError(f"SM-PPM patch_size must be >= 1, got {self.patch_size}.")
        if self.feature_size < self.patch_size:
            raise ValueError(
                f"SM-PPM feature_size={self.feature_size} must be >= patch_size={self.patch_size}."
            )
        if self.feature_size % self.patch_size != 0:
            raise ValueError(
                "SM-PPM feature_size must be divisible by patch_size "
                f"(got {self.feature_size} and {self.patch_size})."
            )
        if not 0.0 <= self.source_free_tau <= 1.0:
            raise ValueError(f"source_free_tau must be in [0, 1], got {self.source_free_tau}.")
        if self.source_free_entropy_threshold is not None and self.source_free_entropy_threshold < 0.0:
            raise ValueError(
                "source_free_entropy_threshold must be non-negative when set, "
                f"got {self.source_free_entropy_threshold}."
            )
        if self.source_free_lambda_proto < 0.0:
            raise ValueError(
                f"source_free_lambda_proto must be >= 0, got {self.source_free_lambda_proto}."
            )
        if self.source_free_entropy_weight < 0.0:
            raise ValueError(
                f"source_free_entropy_weight must be >= 0, got {self.source_free_entropy_weight}."
            )
        if not 0.0 <= self.style_alpha <= 1.0:
            raise ValueError(f"style_alpha must be in [0, 1], got {self.style_alpha}.")
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}.")

        logger.info(
            "[SM-PPM Ablation Mode] %s", self.ablation_mode
        )
        logger.info(
            "SM-PPM ablation flags: use_source_loader=%s use_source_label=%s "
            "use_SM=%s use_PPM=%s use_target_only_loss=%s steps_per_target_batch=%d "
            "source_loss=%s style_alpha=%.4f",
            self.uses_source_loader,
            self.uses_source_label,
            self.uses_sm,
            self.uses_ppm,
            self.uses_target_only_loss,
            self.steps,
            "ce_only" if self.ablation_mode in {"source_ce_only", "sm_ce", "ppm_ce"} else (
                "target_entropy_plus_proto_compact" if self.uses_target_only_loss else "original_dcon_segmentation_criterion"
            ),
            self.style_alpha,
        )
        logger.info(
            "SM style mixing backend: tensor-space AdaIN from target image "
            "statistics onto source images; source labels remain aligned."
        )
        if self.ablation_mode == "source_free_proto":
            logger.info(
                "source_free_proto config: tau=%.4f entropy_threshold=%s "
                "entropy_weight=%.4f lambda_proto=%.4f source_loader_passed=%s",
                self.source_free_tau,
                str(self.source_free_entropy_threshold),
                self.source_free_entropy_weight,
                self.source_free_lambda_proto,
                self.source_loader is not None,
            )
        else:
            logger.info(
                "source-dependent SM-PPM ablation initialized: target labels are "
                "evaluation-only; source labels supervise adaptation."
            )

    def feature_summary(self):
        return (
            f"[SM-PPM Ablation Mode] {self.ablation_mode}\n"
            f"  use_source_loader={self.uses_source_loader}\n"
            f"  use_source_label={self.uses_source_label}\n"
            f"  use_SM={self.uses_sm}\n"
            f"  use_PPM={self.uses_ppm}\n"
            f"  use_target_only_loss={self.uses_target_only_loss}\n"
            f"  steps_per_target_batch={self.steps}\n"
            f"  source_loss="
            f"{'ce_only' if self.ablation_mode in {'source_ce_only', 'sm_ce', 'ppm_ce'} else ('target_entropy_plus_proto_compact' if self.uses_target_only_loss else 'original_dcon_segmentation_criterion')}\n"
            f"  style_backend=medical_adain\n"
            f"  style_alpha={self.style_alpha}\n"
            f"  log_interval={self.log_interval}\n"
            f"  sm_unavailable_reason="
            f"{'current DCON tta_smppm.py has no explicit SM style-mixing implementation' if not STYLE_MIXING_AVAILABLE else 'available'}"
        )

    def _should_log_batch(self, batch_index):
        return self.log_interval > 0 and batch_index % self.log_interval == 0

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            return
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

    def _next_source_batch(self):
        if not self.uses_source_loader:
            raise RuntimeError(
                f"SM-PPM mode {self.ablation_mode} must not request source batches."
            )
        try:
            return next(self.source_iter)
        except StopIteration:
            self.source_iter = iter(self.source_loader)
            try:
                return next(self.source_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "SM-PPM source_loader yielded no batches. Check the source "
                    "training split, smppm_src_batch_size, and drop_last setting."
                ) from exc

    def _extract_source(self, batch):
        if isinstance(batch, dict):
            image = batch.get("base_view", None)
            if image is None:
                image = batch.get("image", None)
            label = batch.get("label", None)
            if image is None or label is None:
                raise KeyError("SM-PPM source batch must contain an image/base_view and label.")
            return image, label

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(f"Unsupported SM-PPM source batch type: {type(batch)}")

    def _require_feature(self, feature):
        if feature is None:
            raise RuntimeError(
                "SM-PPM requires the segmentation model to return bottleneck features "
                "as (logits, feature)."
            )
        if not torch.is_tensor(feature) or feature.dim() != 4:
            raise ValueError(f"SM-PPM expects feature [B,C,H,W], got {type(feature)}.")
        return feature

    @torch.no_grad()
    def _target_prototypes(self, target_img):
        _, target_feature = forward_logits_features(self.model, target_img)
        target_feature = self._require_feature(target_feature)
        target_feature = F.interpolate(
            target_feature,
            size=(self.feature_size, self.feature_size),
            mode="bilinear",
            align_corners=False,
        )
        p = self.patch_size
        patches = target_feature.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().permute(2, 3, 0, 1, 4, 5)
        prototypes = patches.reshape(-1, target_feature.size(0), target_feature.size(1), p * p)
        prototypes = prototypes.mean(dim=(1, 3))
        return F.normalize(prototypes, p=2, dim=1)

    def _similarity_confidence(self, source_feature, prototypes, output_size):
        source_feature = self._require_feature(source_feature)
        source_norm = F.normalize(source_feature, p=2, dim=1)
        sim = torch.einsum("bchw,nc->bnhw", source_norm, prototypes.to(source_norm))
        conf = sim.max(dim=1, keepdim=True)[0]
        conf = ((conf + 1.0) * 0.5).clamp(0.0, 1.0)
        return F.interpolate(conf, size=output_size, mode="bilinear", align_corners=False)

    def _normalized_entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        normalizer = torch.log(torch.tensor(float(self.num_classes), device=logits.device, dtype=logits.dtype))
        return (entropy / normalizer.clamp_min(1e-6)).clamp(0.0, 1.0)

    def _weighted_cross_entropy(self, logits, labels, pixel_weight):
        labels_2d = labels.squeeze(1).long() if labels.dim() == 4 else labels.long()
        if pixel_weight.shape[2:] != labels_2d.shape[-2:]:
            pixel_weight = F.interpolate(
                pixel_weight,
                size=labels_2d.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        weights = pixel_weight.squeeze(1).clamp_min(0.0)
        ce_map = F.cross_entropy(logits, labels_2d, reduction="none")
        return (ce_map * weights).sum() / weights.sum().clamp_min(1e-6)

    @torch.enable_grad()
    def _adapt_one_source_batch(self, source_img, source_label, prototypes=None, target_img=None):
        self.optimizer.zero_grad()

        if self.uses_sm:
            if target_img is None:
                raise RuntimeError(f"SM-PPM mode {self.ablation_mode} requires target images for SM.")
            adapt_img = medical_adain_style_mix(
                source_img,
                target_img,
                alpha=self.style_alpha,
            )
        else:
            adapt_img = source_img

        logits, source_feature = forward_logits_features(self.model, adapt_img)
        if logits.shape[2:] != source_label.shape[-2:]:
            logits = F.interpolate(logits, size=source_label.shape[-2:], mode="bilinear", align_corners=False)

        entropy = self._normalized_entropy(logits).detach()
        if self.uses_ppm:
            if prototypes is None:
                raise RuntimeError(f"SM-PPM mode {self.ablation_mode} requires target prototypes.")
            source_feature = self._require_feature(source_feature)
            confidence = self._similarity_confidence(
                source_feature=source_feature,
                prototypes=prototypes.detach(),
                output_size=source_label.shape[-2:],
            ).detach()
            pixel_weight = (confidence * (1.0 - entropy)).clamp(0.0, 1.0)
        else:
            confidence = torch.ones_like(entropy)
            pixel_weight = torch.ones_like(entropy)

        if self.ablation_mode in {"source_ce_only", "sm_ce", "ppm_ce"}:
            loss = self._weighted_cross_entropy(logits, source_label, pixel_weight)
        else:
            loss = self.segmentation_criterion(logits, source_label.long(), pixel_weight)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss,
            "pixel_weight": pixel_weight,
            "confidence": confidence,
            "entropy": entropy,
        }

    def _source_free_losses(self, logits, feature):
        feature = self._require_feature(feature)
        if logits.shape[2:] != feature.shape[-2:]:
            feature_logits = F.interpolate(
                logits,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            feature_logits = logits

        probs = F.softmax(logits, dim=1)
        max_prob, pseudo_label = probs.detach().max(dim=1)
        norm_entropy = self._normalized_entropy(logits)

        reliable_mask = max_prob > self.source_free_tau
        if self.source_free_entropy_threshold is not None:
            reliable_mask = reliable_mask & (
                norm_entropy.detach().squeeze(1) < self.source_free_entropy_threshold
            )

        reliable_float = reliable_mask.float()
        reliable_count = reliable_float.sum()
        if reliable_count.item() > 0:
            masked_entropy = (norm_entropy.squeeze(1) * reliable_float).sum() / reliable_count.clamp_min(1.0)
        else:
            masked_entropy = norm_entropy.sum() * 0.0

        with torch.no_grad():
            feature_probs = F.softmax(feature_logits, dim=1)
            _, feature_pseudo = feature_probs.max(dim=1)
            feature_reliable = F.interpolate(
                reliable_float.unsqueeze(1),
                size=feature.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()

        feature_norm = F.normalize(feature, p=2, dim=1)
        feature_flat = feature_norm.permute(0, 2, 3, 1)

        proto_loss = feature_norm.sum() * 0.0
        valid_classes = 0
        reliable_feature_pixels = 0
        for class_idx in range(self.num_classes):
            class_mask = feature_reliable & (feature_pseudo == class_idx)
            if not torch.any(class_mask):
                continue
            class_features = feature_flat[class_mask]
            prototype = F.normalize(class_features.mean(dim=0, keepdim=True), p=2, dim=1)
            cosine = (class_features * prototype).sum(dim=1)
            proto_loss = proto_loss + (1.0 - cosine).mean()
            valid_classes += 1
            reliable_feature_pixels += int(class_mask.sum().detach().item())

        if valid_classes > 0:
            proto_loss = proto_loss / float(valid_classes)

        total_loss = (
            self.source_free_entropy_weight * masked_entropy
            + self.source_free_lambda_proto * proto_loss
        )
        return {
            "loss": total_loss,
            "entropy": masked_entropy,
            "proto": proto_loss,
            "reliable_fraction": reliable_float.mean().detach(),
            "reliable_pixels": reliable_count.detach(),
            "proto_classes": torch.tensor(float(valid_classes), device=logits.device),
            "proto_pixels": torch.tensor(float(reliable_feature_pixels), device=logits.device),
        }

    @torch.enable_grad()
    def _adapt_one_target_batch(self, target_img):
        self.optimizer.zero_grad()
        logits, feature = forward_logits_features(self.model, target_img)
        losses = self._source_free_losses(logits, feature)
        losses["loss"].backward()
        self.optimizer.step()
        return losses

    @torch.enable_grad()
    def _forward_source_dependent(self, target_img):
        if self.episodic:
            self.reset()

        self.model.train()
        target_img = target_img.to(self.device, non_blocking=True).float()
        prototypes = self._target_prototypes(target_img) if self.uses_ppm else None

        step_stats = None
        for step_idx in range(self.steps):
            batch = self._next_source_batch()
            source_img, source_label = self._extract_source(batch)
            source_img = source_img.to(self.device, non_blocking=True).float()
            source_label = source_label.to(self.device, non_blocking=True).long()
            step_stats = self._adapt_one_source_batch(
                source_img,
                source_label,
                prototypes,
                target_img=target_img if self.uses_sm else None,
            )
            if self._should_log_batch(self.num_forwards + 1):
                logger.info(
                    "smppm_mode=%s target_batch=%d step=%d/%d "
                    "loss_total=%.6f loss_source=%.6f loss_entropy=0.000000 "
                    "loss_proto=0.000000 weight_mean=%.6f weight_max=%.6f "
                    "source_loader=%s source_label=%s SM=%s PPM=%s target_only_loss=%s",
                    self.ablation_mode,
                    self.num_forwards + 1,
                    step_idx + 1,
                    self.steps,
                    float(step_stats["loss"].detach().item()),
                    float(step_stats["loss"].detach().item()),
                    float(step_stats["pixel_weight"].detach().mean().item()),
                    float(step_stats["pixel_weight"].detach().max().item()),
                    self.uses_source_loader,
                    self.uses_source_label,
                    self.uses_sm,
                    self.uses_ppm,
                    self.uses_target_only_loss,
                )

        self.model.eval()
        with torch.no_grad():
            logits, _ = forward_logits_features(self.model, target_img)

        self.num_forwards += 1
        if step_stats is not None:
            self.last_losses = {
                "smppm_mode": self.ablation_mode,
                "smppm_loss_total": float(step_stats["loss"].detach().item()),
                "smppm_loss_source": float(step_stats["loss"].detach().item()),
                "smppm_loss_entropy": 0.0,
                "smppm_loss_proto": 0.0,
                "smppm_weight_mean": float(step_stats["pixel_weight"].detach().mean().item()),
                "smppm_weight_max": float(step_stats["pixel_weight"].detach().max().item()),
                "smppm_use_source_loader": float(self.uses_source_loader),
                "smppm_use_source_label": float(self.uses_source_label),
                "smppm_use_sm": float(self.uses_sm),
                "smppm_use_ppm": float(self.uses_ppm),
                "smppm_use_target_only_loss": float(self.uses_target_only_loss),
            }
            if self._should_log_batch(self.num_forwards):
                logger.info(
                    "smppm_mode=%s smppm_loss_total=%.6f smppm_loss_source=%.6f "
                    "smppm_weight_mean=%.6f smppm_weight_max=%.6f",
                    self.ablation_mode,
                    self.last_losses["smppm_loss_total"],
                    self.last_losses["smppm_loss_source"],
                    self.last_losses["smppm_weight_mean"],
                    self.last_losses["smppm_weight_max"],
                )

        return logits

    @torch.enable_grad()
    def _forward_source_free(self, target_img):
        if self.episodic:
            self.reset()

        self.model.train()
        target_img = target_img.to(self.device, non_blocking=True).float()

        step_stats = None
        for step_idx in range(self.steps):
            step_stats = self._adapt_one_target_batch(target_img)
            if self._should_log_batch(self.num_forwards + 1):
                logger.info(
                    "smppm_mode=%s target_batch=%d step=%d/%d "
                    "loss_total=%.6f loss_source=0.000000 loss_entropy=%.6f "
                    "loss_proto=%.6f reliable_fraction=%.6f reliable_pixels=%.1f "
                    "proto_classes=%.1f proto_pixels=%.1f source_loader=%s "
                    "source_label=%s SM=%s PPM=%s target_only_loss=%s",
                    self.ablation_mode,
                    self.num_forwards + 1,
                    step_idx + 1,
                    self.steps,
                    float(step_stats["loss"].detach().item()),
                    float(step_stats["entropy"].detach().item()),
                    float(step_stats["proto"].detach().item()),
                    float(step_stats["reliable_fraction"].detach().item()),
                    float(step_stats["reliable_pixels"].detach().item()),
                    float(step_stats["proto_classes"].detach().item()),
                    float(step_stats["proto_pixels"].detach().item()),
                    self.uses_source_loader,
                    self.uses_source_label,
                    self.uses_sm,
                    self.uses_ppm,
                    self.uses_target_only_loss,
                )

        self.model.eval()
        with torch.no_grad():
            logits, _ = forward_logits_features(self.model, target_img)

        self.num_forwards += 1
        if step_stats is not None:
            self.last_losses = {
                "smppm_mode": self.ablation_mode,
                "smppm_loss_total": float(step_stats["loss"].detach().item()),
                "smppm_loss_source": 0.0,
                "smppm_loss_entropy": float(step_stats["entropy"].detach().item()),
                "smppm_loss_proto": float(step_stats["proto"].detach().item()),
                "smppm_reliable_fraction": float(step_stats["reliable_fraction"].detach().item()),
                "smppm_reliable_pixels": float(step_stats["reliable_pixels"].detach().item()),
                "smppm_proto_classes": float(step_stats["proto_classes"].detach().item()),
                "smppm_proto_pixels": float(step_stats["proto_pixels"].detach().item()),
                "smppm_weight_mean": 1.0,
                "smppm_weight_max": 1.0,
                "smppm_use_source_loader": float(self.uses_source_loader),
                "smppm_use_source_label": float(self.uses_source_label),
                "smppm_use_sm": float(self.uses_sm),
                "smppm_use_ppm": float(self.uses_ppm),
                "smppm_use_target_only_loss": float(self.uses_target_only_loss),
            }
            if self._should_log_batch(self.num_forwards):
                logger.info(
                    "smppm_mode=%s smppm_loss_total=%.6f smppm_loss_entropy=%.6f "
                    "smppm_loss_proto=%.6f smppm_reliable_fraction=%.6f",
                    self.ablation_mode,
                    self.last_losses["smppm_loss_total"],
                    self.last_losses["smppm_loss_entropy"],
                    self.last_losses["smppm_loss_proto"],
                    self.last_losses["smppm_reliable_fraction"],
                )

        return logits

    @torch.enable_grad()
    def forward(self, target_img):
        if self.ablation_mode == "source_free_proto":
            return self._forward_source_free(target_img)
        return self._forward_source_dependent(target_img)
