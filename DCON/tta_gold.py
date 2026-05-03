import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@torch.no_grad()
def _update_ema_model(ema_model, model, momentum):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)


class GOLDAdapter:
    """GOLD test-time adapter for DCON U-Net segmentation.

    This follows the GOLD segmentation baseline: CoTTA-like student/EMA updates
    plus a low-rank pre-classifier feature adapter estimated from confident
    teacher pixels. The target label is never used.
    """
    def __init__(
        self,
        model,
        optimizer,
        num_classes,
        steps=1,
        rank=128,
        tau=0.95,
        alpha=0.02,
        t_eig=10,
        mt=0.999,
        s_lr=5e-3,
        s_init_scale=0.0,
        s_clip=0.5,
        adapter_scale=0.05,
        max_pixels_per_batch=512,
        min_pixels_per_batch=64,
        n_augmentations=6,
        rst=0.01,
        ap=0.9,
        episodic=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.rank = int(rank)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.t_eig = int(t_eig)
        self.mt = float(mt)
        self.s_lr = float(s_lr)
        self.s_init_scale = float(s_init_scale)
        self.s_clip = float(s_clip)
        self.adapter_scale = float(adapter_scale)
        self.max_pixels_per_batch = int(max_pixels_per_batch)
        self.min_pixels_per_batch = int(min_pixels_per_batch)
        self.n_augmentations = int(n_augmentations)
        self.rst = float(rst)
        self.ap = float(ap)
        self.episodic = bool(episodic)

        self.device = next(self.model.parameters()).device
        self.model_ema = deepcopy(self.model).to(self.device)
        self.model_anchor = deepcopy(self.model).to(self.device)
        for teacher in (self.model_ema, self.model_anchor):
            teacher.eval()
            for param in teacher.parameters():
                param.detach_()
                param.requires_grad_(False)

        self.source_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.model_state = deepcopy(self.model.state_dict())

        self.classifier_layer = self._find_classifier_layer()
        self.feature_dim = None
        self.M = None
        self.V = None
        self.S = None
        self.s_optimizer = None
        self.step_counter = 0
        self.last_losses = {}

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.model_ema.load_state_dict(self.model_state, strict=True)
        self.model_anchor.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.feature_dim = None
        self.M = None
        self.V = None
        self.S = None
        self.s_optimizer = None
        self.step_counter = 0

    def _find_classifier_layer(self):
        if hasattr(self.model, "seg1") and isinstance(self.model.seg1, nn.Conv2d):
            if self.model.seg1.out_channels == self.num_classes:
                return self.model.seg1

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == self.num_classes:
                return module
        raise RuntimeError(f"Cannot find Conv2d classifier with out_channels={self.num_classes}.")

    @torch.no_grad()
    def _classifier_weight_matrix(self, dtype):
        weight = self.classifier_layer.weight
        if weight.shape[2:] == (1, 1):
            W = weight.squeeze(-1).squeeze(-1)
        else:
            W = weight.mean(dim=(2, 3))
        return W.to(device=self.device, dtype=dtype).contiguous()

    def _forward_capture_precls(self, images):
        holder = {}

        def hook(_module, inputs, output):
            holder["features"] = inputs[0]
            holder["head_logits"] = output

        handle = self.classifier_layer.register_forward_hook(hook)
        logits = _forward_logits(self.model, images)
        handle.remove()

        if "features" not in holder:
            raise RuntimeError("GOLD hook failed to capture pre-classifier features.")
        return holder["features"], logits

    def _lazy_init(self, features):
        if features.ndim != 4:
            raise ValueError(f"Expected pre-classifier features [B,L,H,W], got {features.shape}.")

        feature_dim = int(features.shape[1])
        self.feature_dim = feature_dim
        W = self._classifier_weight_matrix(features.dtype)
        self.M = W.t().matmul(W).contiguous()
        self._update_subspace()

        rank = min(self.rank, feature_dim)
        if self.s_init_scale != 0.0:
            init = self.s_init_scale * torch.randn(rank, device=self.device, dtype=features.dtype)
        else:
            init = torch.zeros(rank, device=self.device, dtype=features.dtype)
        self.S = nn.Parameter(init)
        self.s_optimizer = torch.optim.SGD([self.S], lr=self.s_lr, momentum=0.9)

    @torch.no_grad()
    def _update_subspace(self):
        Msym = 0.5 * (self.M + self.M.t())
        eigenvalues, eigenvectors = torch.linalg.eigh(Msym)
        rank = min(self.rank, Msym.shape[0])
        idx = torch.argsort(eigenvalues, descending=True)[:rank]
        self.V = eigenvectors[:, idx].contiguous().to(self.device)

    def _logits_from_features(self, features):
        return self.classifier_layer(features)

    def _apply_adapter(self, features):
        V = self.V.to(features.device)
        S = self.S.to(features.device)
        projected = torch.einsum("blhw,lr->brhw", features, V)
        scaled = S.view(1, -1, 1, 1) * projected
        delta = torch.einsum("lr,brhw->blhw", V, scaled)
        return features + self.adapter_scale * delta

    def _sample_confident_indices(self, logits_teacher, feature_hw):
        probs = F.softmax(logits_teacher, dim=1)
        if probs.shape[2:] != feature_hw:
            probs = F.interpolate(probs, size=feature_hw, mode="bilinear", align_corners=False)
        conf, pred = probs.max(dim=1)
        mask_flat = (conf >= self.tau).flatten()
        idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        if idx.numel() < self.min_pixels_per_batch:
            return None
        if idx.numel() > self.max_pixels_per_batch:
            perm = torch.randperm(idx.numel(), device=idx.device)[:self.max_pixels_per_batch]
            idx = idx[perm]
        return idx, pred.flatten()[idx]

    def _compute_agop_batch(self, features, logits_teacher):
        batch, channels, height, width = features.shape
        sampled = self._sample_confident_indices(logits_teacher, (height, width))
        if sampled is None:
            return None
        idx, pred_flat = sampled
        b = idx // (height * width)
        rem = idx % (height * width)
        yy = rem // width
        xx = rem % width

        logits = self._logits_from_features(features)
        if logits.shape[2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)

        score_sum = torch.zeros((), device=features.device, dtype=logits.dtype)
        for i in range(int(idx.numel())):
            score_sum = score_sum + logits[b[i], pred_flat[i], yy[i], xx[i]]

        grads = torch.autograd.grad(
            score_sum,
            features,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        M_batch = features.new_zeros((channels, channels))
        for i in range(int(idx.numel())):
            g = grads[b[i], :, yy[i], xx[i]]
            M_batch += torch.outer(g, g)
        return (M_batch / float(max(idx.numel(), 1))).detach()

    @torch.no_grad()
    def _ensemble_prediction(self, images, ema_logits):
        inp_shape = images.shape[2:]
        ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75][:max(0, self.n_augmentations)]
        for ratio in ratios:
            aug_shape = (
                max(1, int(inp_shape[0] * ratio)),
                max(1, int(inp_shape[1] * ratio)),
            )
            flip = [random.random() <= 0.5 for _ in range(images.shape[0])]
            aug_images = torch.cat(
                [images[i:i + 1].flip(dims=(3,)) if fp else images[i:i + 1]
                 for i, fp in enumerate(flip)],
                dim=0,
            )
            aug_images = F.interpolate(aug_images, size=aug_shape, mode="bilinear", align_corners=False)
            aug_logits = _forward_logits(self.model_ema, aug_images)
            aug_logits = torch.cat(
                [aug_logits[i:i + 1].flip(dims=(3,)) if fp else aug_logits[i:i + 1]
                 for i, fp in enumerate(flip)],
                dim=0,
            )
            ema_logits = ema_logits + F.interpolate(aug_logits, size=inp_shape, mode="bilinear", align_corners=False)
        return ema_logits / float(len(ratios) + 1)

    @torch.no_grad()
    def _stochastic_restore(self):
        if self.rst <= 0.0:
            return
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not (name.endswith("weight") or name.endswith("bias")):
                continue
            if name not in self.source_state:
                continue
            mask = torch.rand_like(param, dtype=torch.float32) < self.rst
            source_param = self.source_state[name].to(param.device)
            param.data.copy_(torch.where(mask, source_param, param.data))

    @torch.enable_grad()
    def _adapt_once(self, images):
        self.model.train()
        outputs = _forward_logits(self.model, images)

        with torch.no_grad():
            anchor_logits = _forward_logits(self.model_anchor, images)
            anchor_conf = torch.softmax(anchor_logits, dim=1).max(dim=1)[0]
            outputs_ema = _forward_logits(self.model_ema, images)
            if anchor_conf.mean() < self.ap:
                outputs_ema = self._ensemble_prediction(images, outputs_ema)

        loss_student = (-(outputs_ema.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_student.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        _update_ema_model(self.model_ema, self.model, self.mt)
        self._stochastic_restore()

        features, _ = self._forward_capture_precls(images)
        if self.feature_dim is None:
            self._lazy_init(features.detach())

        features_for_agop = features.detach().requires_grad_(True)
        M_batch = self._compute_agop_batch(features_for_agop, outputs_ema.detach())
        if M_batch is not None:
            with torch.no_grad():
                self.M = (1.0 - self.alpha) * self.M + self.alpha * M_batch.to(self.device)

        self.step_counter += 1
        if self.step_counter % max(1, self.t_eig) == 0:
            self._update_subspace()

        adapted_features = self._apply_adapter(features_for_agop)
        logits_adapted = self._logits_from_features(adapted_features)
        probs_ema = F.softmax(outputs_ema.detach(), dim=1)
        probs_adapted = F.softmax(logits_adapted, dim=1)
        if probs_adapted.shape[2:] != probs_ema.shape[2:]:
            probs_adapted = F.interpolate(probs_adapted, size=probs_ema.shape[2:], mode="bilinear", align_corners=False)

        loss_agop = (-(probs_ema * torch.clamp(probs_adapted, min=1e-6).log()).sum(1)).mean()
        self.s_optimizer.zero_grad(set_to_none=True)
        loss_agop.backward()
        if self.S.grad is not None:
            torch.nn.utils.clip_grad_norm_([self.S], max_norm=1.0)
        self.s_optimizer.step()
        with torch.no_grad():
            self.S.clamp_(min=-self.s_clip, max=self.s_clip)

        self.last_losses = {
            "student": float(loss_student.detach().cpu()),
            "agop": float(loss_agop.detach().cpu()),
            "confident_pixels": 0 if M_batch is None else 1,
        }

    @torch.enable_grad()
    def forward(self, images):
        if self.episodic:
            self.reset()
        images = images.to(self.device)
        for _ in range(self.steps):
            self._adapt_once(images)
        with torch.no_grad():
            return _forward_logits(self.model_ema, images)
