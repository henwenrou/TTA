"""Medical GTTA test-time adaptation for DCON.

This is a lightweight medical-image adaptation of GTTA. It keeps GTTA's
source-supervised update and filtered target pseudo-label self-training, but
replaces the original VGG/decoder AdaIN module with tensor-space class-aware
intensity moment matching. Target labels are evaluation-only.
"""

import logging
from copy import deepcopy

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class GTTAAdapter:
    """Source-dependent GTTA adapter for DCON medical segmentation."""

    def __init__(
        self,
        model,
        optimizer,
        source_loader,
        device,
        num_classes,
        steps=1,
        lambda_ce_trg=0.1,
        pseudo_momentum=0.9,
        style_alpha=1.0,
        include_original=True,
        episodic=False,
        ignore_label=255,
        min_class_pixels=8,
        segmentation_criterion=None,
    ):
        if source_loader is None:
            raise ValueError("GTTA requires a labeled source_loader; it is not source-free TTA.")
        if segmentation_criterion is None:
            raise ValueError("GTTA requires a source segmentation_criterion.")

        self.model = model
        self.optimizer = optimizer
        self.source_loader = source_loader
        self.source_iter = iter(source_loader)
        self.device = device
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.lambda_ce_trg = float(lambda_ce_trg)
        self.pseudo_momentum = float(pseudo_momentum)
        self.style_alpha = float(style_alpha)
        self.include_original = bool(include_original)
        self.episodic = bool(episodic)
        self.ignore_label = int(ignore_label)
        self.min_class_pixels = int(min_class_pixels)
        self.segmentation_criterion = segmentation_criterion
        self.avg_conf = torch.tensor(0.9, device=device)
        self.model_state = deepcopy(model.state_dict()) if self.episodic else None
        self.optimizer_state = deepcopy(optimizer.state_dict()) if self.episodic else None
        self.last_losses = {}
        self.num_forwards = 0

        if self.steps < 1:
            raise ValueError(f"GTTA steps must be >= 1, got {self.steps}.")
        if not 0.0 <= self.pseudo_momentum <= 1.0:
            raise ValueError(f"GTTA pseudo_momentum must be in [0, 1], got {self.pseudo_momentum}.")
        if not 0.0 <= self.style_alpha <= 1.0:
            raise ValueError(f"GTTA style_alpha must be in [0, 1], got {self.style_alpha}.")

        logger.info(
            "GTTA initialized as source-dependent supervised TTA: source labels "
            "supervise adaptation; target pseudo labels are model-generated."
        )

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            return
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.avg_conf = torch.tensor(0.9, device=self.device)

    def _next_source_batch(self):
        try:
            return next(self.source_iter)
        except StopIteration:
            self.source_iter = iter(self.source_loader)
            try:
                return next(self.source_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "GTTA source_loader yielded no batches. Check the source "
                    "training split, gtta_src_batch_size, and drop_last setting."
                ) from exc

    def _extract_source(self, batch):
        if isinstance(batch, dict):
            image = batch.get("base_view", None)
            if image is None:
                image = batch.get("image", None)
            label = batch.get("label", None)
            if image is None or label is None:
                raise KeyError("GTTA source batch must contain an image/base_view and label.")
            return image, label

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(f"Unsupported GTTA source batch type: {type(batch)}")

    @torch.no_grad()
    def _create_pseudo_labels(self, target_img):
        logits = forward_logits(self.model, target_img)
        logits = self._resize_logits(logits, target_img.shape[-2:])
        probs = F.softmax(logits, dim=1)
        confidences, pseudo = probs.max(dim=1)
        self.avg_conf = (
            self.pseudo_momentum * self.avg_conf.to(confidences.device)
            + (1.0 - self.pseudo_momentum) * confidences.mean()
        )
        threshold = torch.sqrt(self.avg_conf.clamp_min(1e-6))
        pseudo_thr = pseudo.clone()
        pseudo_thr[confidences < threshold] = self.ignore_label
        valid_ratio = (pseudo_thr != self.ignore_label).float().mean()
        return pseudo_thr, valid_ratio

    @torch.no_grad()
    def _target_class_moments(self, target_img, pseudo_labels):
        moments = {}
        labels = pseudo_labels.long()
        for class_nr in range(self.num_classes):
            mask = labels == class_nr
            if int(mask.sum().item()) < self.min_class_pixels:
                continue
            values = target_img.permute(0, 2, 3, 1)[mask]
            if values.numel() == 0:
                continue
            mean = values.mean(dim=0)
            std = values.std(dim=0, unbiased=False).clamp_min(1e-5)
            moments[class_nr] = (mean.detach(), std.detach())
        return moments

    @torch.no_grad()
    def _class_aware_adain(self, source_img, source_label, target_moments):
        if len(target_moments) == 0 or self.style_alpha <= 0.0:
            return source_img

        labels = source_label.squeeze(1).long() if source_label.dim() == 4 else source_label.long()
        stylized = source_img.clone()
        for batch_idx in range(source_img.size(0)):
            for class_nr, (target_mean, target_std) in target_moments.items():
                mask = labels[batch_idx] == class_nr
                if int(mask.sum().item()) < self.min_class_pixels:
                    continue

                values = source_img[batch_idx, :, mask]
                source_mean = values.mean(dim=1).view(-1, 1)
                source_std = values.std(dim=1, unbiased=False).clamp_min(1e-5).view(-1, 1)
                target_mean = target_mean.to(source_img.device, source_img.dtype).view(-1, 1)
                target_std = target_std.to(source_img.device, source_img.dtype).view(-1, 1)
                matched = (values - source_mean) / source_std * target_std + target_mean
                stylized[batch_idx, :, mask] = (
                    self.style_alpha * matched + (1.0 - self.style_alpha) * values
                )
        return stylized.contiguous()

    def _resize_logits(self, logits, target_hw):
        if logits.shape[2:] == target_hw:
            return logits
        return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)

    def _source_loss(self, logits, labels):
        result = self.segmentation_criterion(logits, labels.long())
        if isinstance(result, tuple):
            return result[0]
        return result

    def _target_loss(self, target_img, pseudo_labels):
        valid = pseudo_labels != self.ignore_label
        if not bool(valid.any()):
            return torch.zeros((), device=target_img.device, dtype=target_img.dtype)

        logits = forward_logits(self.model, target_img)
        logits = self._resize_logits(logits, pseudo_labels.shape[-2:])
        return F.cross_entropy(logits, pseudo_labels.long(), ignore_index=self.ignore_label)

    @torch.enable_grad()
    def _adapt_once(self, target_img, pseudo_labels, target_moments):
        batch = self._next_source_batch()
        source_img, source_label = self._extract_source(batch)
        source_img = source_img.to(self.device, non_blocking=True).float()
        source_label = source_label.to(self.device, non_blocking=True).long()

        with torch.no_grad():
            stylized_source = self._class_aware_adain(source_img, source_label, target_moments)

        if self.include_original:
            train_img = torch.cat([stylized_source, source_img], dim=0)
            train_label = torch.cat([source_label, source_label], dim=0)
        else:
            train_img = stylized_source
            train_label = source_label

        self.optimizer.zero_grad()
        source_logits = forward_logits(self.model, train_img)
        source_logits = self._resize_logits(source_logits, train_label.shape[-2:])
        loss_src = self._source_loss(source_logits, train_label)
        loss_src.backward()
        self.optimizer.step()

        loss_trg = torch.zeros((), device=self.device)
        if self.lambda_ce_trg > 0.0:
            self.optimizer.zero_grad()
            loss_trg = self._target_loss(target_img, pseudo_labels) * self.lambda_ce_trg
            if loss_trg.requires_grad:
                loss_trg.backward()
                self.optimizer.step()

        return loss_src, loss_trg, loss_src.detach() + loss_trg.detach()

    @torch.enable_grad()
    def forward(self, target_img):
        if self.episodic:
            self.reset()

        self.model.train()
        target_img = target_img.to(self.device, non_blocking=True).float()
        pseudo_labels, valid_ratio = self._create_pseudo_labels(target_img)
        target_moments = self._target_class_moments(target_img, pseudo_labels)

        loss_src = loss_trg = loss_total = None
        for _ in range(self.steps):
            loss_src, loss_trg, loss_total = self._adapt_once(target_img, pseudo_labels, target_moments)

        self.model.eval()
        with torch.no_grad():
            logits = forward_logits(self.model, target_img)

        self.num_forwards += 1
        if loss_total is not None:
            self.last_losses = {
                "gtta_loss_src": float(loss_src.detach().item()),
                "gtta_loss_trg": float(loss_trg.detach().item()),
                "gtta_loss_total": float(loss_total.detach().item()),
                "gtta_pseudo_valid_ratio": float(valid_ratio.detach().item()),
            }
            logger.info(
                "gtta_loss_src=%.6f gtta_loss_trg=%.6f gtta_loss_total=%.6f gtta_pseudo_valid_ratio=%.6f",
                self.last_losses["gtta_loss_src"],
                self.last_losses["gtta_loss_trg"],
                self.last_losses["gtta_loss_total"],
                self.last_losses["gtta_pseudo_valid_ratio"],
            )

        return logits
