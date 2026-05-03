"""Medical ASM test-time adaptation for DCON.

This module intentionally keeps ASM separate from the common source-free TTA
baselines. ASM is source-dependent: each target batch supplies only image
style/statistics, while adaptation is supervised by a labeled source batch.
Target labels must never be used in the adaptation loss.
"""

import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def forward_logits_features(model, images):
    """Return segmentation logits and an optional feature tensor.

    DCON's U-Net returns ``(logits, bottleneck_feature)``. This helper also
    supports models that return logits only, and tuple/list outputs from other
    segmentation networks.
    """
    output = model(images)
    if isinstance(output, (tuple, list)):
        logits = output[0]
        feature = output[1] if len(output) > 1 else None
        return logits, feature
    return output, None


def _calc_mean_std(x, eps=1e-5):
    if x.dim() != 4:
        raise ValueError(f"ASM style transfer expects [B, C, H, W], got {tuple(x.shape)}")
    var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return mean, (var + eps).sqrt()


def _match_batch(stat, batch_size):
    """Repeat target statistics to match the source batch size."""
    if stat.size(0) == batch_size:
        return stat
    if stat.size(0) == 1:
        return stat.expand(batch_size, -1, -1, -1)
    repeat = (batch_size + stat.size(0) - 1) // stat.size(0)
    return stat.repeat(repeat, 1, 1, 1)[:batch_size]


class MedicalASMStyleTransfer(nn.Module):
    """AdaIN-like tensor-space style transfer for medical slices.

    The operation changes only per-sample/channel intensity statistics in the
    normalized tensor space already used by DCON. It performs no rotation, crop,
    affine, elastic, RGB/PIL conversion, or ImageNet normalization, so the source
    anatomy remains aligned with the source label.
    """

    def __init__(self, eps=1e-5, sampling_init=2.0):
        super().__init__()
        self.eps = eps
        self.sampling_init = sampling_init

    def _init_sampling(self, src):
        sampling = torch.full(
            (1, src.size(1), 1, 1),
            float(self.sampling_init),
            device=src.device,
            dtype=src.dtype,
        )
        return sampling.requires_grad_(True)

    def forward(self, src, tgt, sampling=None):
        # Target images provide only style statistics; no target labels or
        # pseudo-labels are involved in ASM adaptation.
        tgt = tgt.detach()

        src_mean, src_std = _calc_mean_std(src, self.eps)
        tgt_mean, tgt_std = _calc_mean_std(tgt, self.eps)
        tgt_mean = _match_batch(tgt_mean, src.size(0))
        tgt_std = _match_batch(tgt_std, src.size(0))

        if sampling is None or sampling.size(1) != src.size(1):
            sampling = self._init_sampling(src)
        else:
            sampling = sampling.detach().to(device=src.device, dtype=src.dtype).requires_grad_(True)

        src_norm = (src - src_mean) / src_std
        stylized_full = src_norm * tgt_std + tgt_mean
        alpha = torch.sigmoid(sampling).expand(src.size(0), -1, -1, -1)
        stylized = alpha * stylized_full + (1.0 - alpha) * src
        return stylized, sampling


class ASMAdapter:
    """Source-dependent ASM adapter for DCON medical segmentation.

    For each target batch:
      1. Fetch one labeled source batch.
      2. Transfer target intensity statistics onto the source image tensor.
      3. Train on ``[stylized_source, original_source]`` with duplicated source
         labels.
      4. Add feature mean-square regularization when the model exposes a
         bottleneck feature.
      5. Predict the target batch with the updated model.

    The target batch is never used for a supervised or pseudo-label loss.
    """

    def __init__(
        self,
        model,
        optimizer,
        source_loader,
        device,
        num_classes,
        steps=1,
        inner_steps=2,
        lambda_reg=2e-4,
        sampling_step=20.0,
        episodic=False,
        style_backend="medical_adain",
        segmentation_criterion=None,
    ):
        if source_loader is None:
            raise ValueError("ASM requires a labeled source_loader; it is not source-free TTA.")
        if style_backend != "medical_adain":
            raise ValueError(
                f"Unsupported ASM style backend '{style_backend}'. "
                "DCON currently implements 'medical_adain' only."
            )

        self.model = model
        self.optimizer = optimizer
        self.source_loader = source_loader
        self.source_iter = iter(source_loader)
        self.device = device
        self.num_classes = num_classes
        self.steps = int(steps)
        self.inner_steps = int(inner_steps)
        self.lambda_reg = float(lambda_reg)
        self.sampling_step = float(sampling_step)
        self.episodic = bool(episodic)
        self.style_backend = style_backend
        self.segmentation_criterion = segmentation_criterion
        self.style_transfer = MedicalASMStyleTransfer().to(device)
        self.model_state = deepcopy(model.state_dict()) if self.episodic else None
        self.optimizer_state = deepcopy(optimizer.state_dict()) if self.episodic else None
        self.last_losses = {}
        self.num_forwards = 0

        logger.info(
            "ASM initialized as source-dependent supervised TTA: target images "
            "provide style statistics only; source images and source labels "
            "provide the adaptation loss."
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
                    "ASM source_loader yielded no batches. Check the source "
                    "training split, asm_src_batch_size, and drop_last setting."
                ) from exc

    def _extract_source(self, batch):
        if isinstance(batch, dict):
            # base_view is the aligned source training view in DCON. anchor_view
            # is intentionally not used here because the training label follows
            # the shared training transform applied to base_view/strong_view.
            image = batch.get("base_view", None)
            if image is None:
                image = batch.get("image", None)
            label = batch.get("label", None)
            if image is None or label is None:
                raise KeyError("ASM source batch must contain an image/base_view and label.")
            return image, label

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(f"Unsupported ASM source batch type: {type(batch)}")

    def _segmentation_loss(self, logits, labels):
        if self.segmentation_criterion is not None:
            result = self.segmentation_criterion(logits, labels)
            if isinstance(result, tuple):
                return result[0]
            return result
        labels_2d = labels.squeeze(1) if labels.dim() == 4 else labels
        return F.cross_entropy(logits, labels_2d.long())

    def _feature_regularization(self, features, stylized_batch_size):
        if features is None:
            return None
        if not torch.is_tensor(features):
            return None
        if features.dim() > 0 and features.size(0) >= stylized_batch_size:
            features = features[:stylized_batch_size]
        return features.pow(2).mean()

    def _adapt_one_source_batch(self, source_img, source_label, target_img):
        sampling = None
        loss_seg = None
        loss_reg = None
        loss_total = None

        for _ in range(self.inner_steps):
            self.optimizer.zero_grad()

            stylized_src, sampling = self.style_transfer(source_img, target_img, sampling)
            mixed_img = torch.cat([stylized_src, source_img], dim=0)
            mixed_label = torch.cat([source_label, source_label], dim=0).long()

            logits, features = forward_logits_features(self.model, mixed_img)
            if logits.shape[2:] != mixed_label.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=mixed_label.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss_seg = self._segmentation_loss(logits, mixed_label)
            reg = self._feature_regularization(features, stylized_batch_size=source_img.size(0))
            if reg is None:
                loss_reg = torch.zeros((), device=logits.device, dtype=logits.dtype)
            else:
                loss_reg = reg
            loss_total = loss_seg + self.lambda_reg * loss_reg

            if sampling.requires_grad:
                sampling.retain_grad()
            loss_total.backward()

            if sampling.requires_grad and sampling.grad is not None:
                denom = max(float(loss_total.detach().item()), 1e-6)
                step = self.sampling_step / denom
                with torch.no_grad():
                    sampling = (sampling + step * sampling.grad).clamp(-10.0, 10.0).detach()

            self.optimizer.step()

        return loss_seg, loss_reg, loss_total

    @torch.enable_grad()
    def forward(self, target_img):
        if self.episodic:
            self.reset()

        self.model.train()
        target_img = target_img.to(self.device, non_blocking=True).float()

        loss_seg = loss_reg = loss_total = None
        for _ in range(self.steps):
            batch = self._next_source_batch()
            source_img, source_label = self._extract_source(batch)
            source_img = source_img.to(self.device, non_blocking=True).float()
            source_label = source_label.to(self.device, non_blocking=True).long()

            loss_seg, loss_reg, loss_total = self._adapt_one_source_batch(
                source_img=source_img,
                source_label=source_label,
                target_img=target_img,
            )

        self.model.eval()
        with torch.no_grad():
            logits, _ = forward_logits_features(self.model, target_img)

        self.num_forwards += 1
        if loss_total is not None:
            self.last_losses = {
                "asm_loss_seg": float(loss_seg.detach().item()),
                "asm_loss_reg": float(loss_reg.detach().item()),
                "asm_loss_total": float(loss_total.detach().item()),
            }
            logger.info(
                "asm_loss_seg=%.6f asm_loss_reg=%.6f asm_loss_total=%.6f",
                self.last_losses["asm_loss_seg"],
                self.last_losses["asm_loss_reg"],
                self.last_losses["asm_loss_total"],
            )

        return logits
