"""Medical SM-PPM test-time adaptation for DCON.

SM-PPM is source-dependent TTA: each target batch provides feature prototypes,
while labeled source batches provide the supervised adaptation loss. Target
labels are evaluation-only and must not be used by this adapter.
"""

import logging
from copy import deepcopy

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def forward_logits_features(model, images):
    """Return segmentation logits and the DCON bottleneck feature."""
    output = model(images)
    if isinstance(output, (tuple, list)):
        logits = output[0]
        feature = output[1] if len(output) > 1 else None
        return logits, feature
    return output, None


class SMPPMAdapter:
    """Source-dependent SM-PPM adapter for DCON medical segmentation.

    For each target batch:
      1. Extract target bottleneck features and pool them into patch prototypes.
      2. Fetch one labeled source batch.
      3. Update the segmentation model on source labels with SM-PPM pixel
         weights from source entropy and source-target feature similarity.
      4. Predict the target batch with the updated model.
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
    ):
        if source_loader is None:
            raise ValueError("SM-PPM requires a labeled source_loader; it is not source-free TTA.")
        if segmentation_criterion is None:
            raise ValueError("SM-PPM requires a weighted segmentation_criterion.")

        self.model = model
        self.optimizer = optimizer
        self.source_loader = source_loader
        self.source_iter = iter(source_loader)
        self.device = device
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.patch_size = int(patch_size)
        self.feature_size = int(feature_size)
        self.episodic = bool(episodic)
        self.segmentation_criterion = segmentation_criterion
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

        logger.info(
            "SM-PPM initialized as source-dependent supervised TTA: target "
            "images provide feature prototypes only; source labels provide the loss."
        )

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            return
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

    def _next_source_batch(self):
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

    @torch.enable_grad()
    def _adapt_one_source_batch(self, source_img, source_label, prototypes):
        self.optimizer.zero_grad()

        logits, source_feature = forward_logits_features(self.model, source_img)
        source_feature = self._require_feature(source_feature)
        if logits.shape[2:] != source_label.shape[-2:]:
            logits = F.interpolate(logits, size=source_label.shape[-2:], mode="bilinear", align_corners=False)

        entropy = self._normalized_entropy(logits).detach()
        confidence = self._similarity_confidence(
            source_feature=source_feature,
            prototypes=prototypes.detach(),
            output_size=source_label.shape[-2:],
        ).detach()
        pixel_weight = (confidence * (1.0 - entropy)).clamp(0.0, 1.0)

        loss = self.segmentation_criterion(logits, source_label.long(), pixel_weight)
        loss.backward()
        self.optimizer.step()
        return loss, pixel_weight

    @torch.enable_grad()
    def forward(self, target_img):
        if self.episodic:
            self.reset()

        self.model.train()
        target_img = target_img.to(self.device, non_blocking=True).float()
        prototypes = self._target_prototypes(target_img)

        loss = None
        pixel_weight = None
        for _ in range(self.steps):
            batch = self._next_source_batch()
            source_img, source_label = self._extract_source(batch)
            source_img = source_img.to(self.device, non_blocking=True).float()
            source_label = source_label.to(self.device, non_blocking=True).long()
            loss, pixel_weight = self._adapt_one_source_batch(source_img, source_label, prototypes)

        self.model.eval()
        with torch.no_grad():
            logits, _ = forward_logits_features(self.model, target_img)

        self.num_forwards += 1
        if loss is not None:
            self.last_losses = {
                "smppm_loss_source": float(loss.detach().item()),
                "smppm_weight_mean": float(pixel_weight.detach().mean().item()),
                "smppm_weight_max": float(pixel_weight.detach().max().item()),
            }
            logger.info(
                "smppm_loss_source=%.6f smppm_weight_mean=%.6f smppm_weight_max=%.6f",
                self.last_losses["smppm_loss_source"],
                self.last_losses["smppm_weight_mean"],
                self.last_losses["smppm_weight_max"],
            )

        return logits
