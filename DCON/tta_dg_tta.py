from copy import deepcopy
import math
import random

import torch
import torch.nn.functional as F


def forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def softmax_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


def symmetric_kl(logits_a, logits_b):
    log_p = F.log_softmax(logits_a, dim=1)
    log_q = F.log_softmax(logits_b, dim=1)
    p = log_p.exp()
    q = log_q.exp()
    return 0.5 * (
        F.kl_div(log_p, q.detach(), reduction="batchmean")
        + F.kl_div(log_q, p.detach(), reduction="batchmean")
    )


def _theta_inverse(theta):
    batch = theta.size(0)
    full = torch.zeros(batch, 3, 3, dtype=theta.dtype, device=theta.device)
    full[:, :2, :] = theta
    full[:, 2, 2] = 1.0
    inv = torch.linalg.inv(full)
    return inv[:, :2, :]


def _random_affine_theta(batch, device, dtype, strength):
    max_rotate = 12.0 * strength
    max_shift = 0.06 * strength
    scale_delta = 0.12 * strength
    max_shear = 5.0 * strength

    theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
    for idx in range(batch):
        angle = math.radians(random.uniform(-max_rotate, max_rotate))
        shear = math.radians(random.uniform(-max_shear, max_shear))
        scale = random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
        inv_scale = 1.0 / max(scale, 1e-6)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tan_s = math.tan(shear)

        theta[idx, 0, 0] = inv_scale * cos_a
        theta[idx, 0, 1] = inv_scale * (-sin_a + tan_s * cos_a)
        theta[idx, 1, 0] = inv_scale * sin_a
        theta[idx, 1, 1] = inv_scale * (cos_a + tan_s * sin_a)
        theta[idx, 0, 2] = random.uniform(-max_shift, max_shift)
        theta[idx, 1, 2] = random.uniform(-max_shift, max_shift)
    return theta


def _intensity_jitter(images, strength):
    if strength <= 0:
        return images
    scale = 1.0 + (torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) - 0.5) * (0.30 * strength)
    shift = (torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) - 0.5) * (0.20 * strength)
    noise = torch.randn_like(images) * (0.015 * strength)
    return images * scale + shift + noise


def build_dgtta_augmented_view(images, strength=1.0):
    theta = _random_affine_theta(
        batch=images.size(0),
        device=images.device,
        dtype=images.dtype,
        strength=float(strength),
    )
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    aug = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    aug = _intensity_jitter(aug, float(strength))
    inv_theta = _theta_inverse(theta)
    return aug.contiguous(), inv_theta


def align_logits_to_original(logits, inv_theta):
    grid = F.affine_grid(inv_theta.to(dtype=logits.dtype), logits.size(), align_corners=False)
    return F.grid_sample(logits, grid, mode="bilinear", padding_mode="border", align_corners=False)


class DGTTAAdapter:
    """DCON adapter for MedSeg-TTA DG-TTA style output consistency.

    The legacy MedSeg-TTA DG-TTA entrypoint owns its own U-Net and image/mask
    directory loader. This adapter keeps only the portable method core:
    online BN-affine updates from spatial/intensity consistency on target
    images. Target labels are used only by the surrounding DCON evaluator.
    """

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        transform_strength=1.0,
        entropy_weight=0.05,
        bn_l2_reg=1e-4,
        episodic=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.steps = int(steps)
        self.transform_strength = float(transform_strength)
        self.entropy_weight = float(entropy_weight)
        self.bn_l2_reg = float(bn_l2_reg)
        self.episodic = bool(episodic)
        self.model_state = deepcopy(model.state_dict()) if self.episodic else None
        self.optimizer_state = deepcopy(optimizer.state_dict()) if self.episodic else None
        self.bn_snapshots = self._snapshot_bn_affine()
        self.last_losses = {}

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            return
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.bn_snapshots = self._snapshot_bn_affine()

    def _snapshot_bn_affine(self):
        snapshots = []
        for module in self.model.modules():
            if module.__class__.__name__ != "BatchNorm2d":
                continue
            weight = module.weight.detach().clone() if module.weight is not None else None
            bias = module.bias.detach().clone() if module.bias is not None else None
            snapshots.append((module, weight, bias))
        return snapshots

    def _bn_regularization(self):
        if self.bn_l2_reg <= 0 or not self.bn_snapshots:
            return None
        reg = None
        for module, weight0, bias0 in self.bn_snapshots:
            terms = []
            if module.weight is not None and weight0 is not None:
                terms.append((module.weight - weight0.to(module.weight.device)).pow(2).mean())
            if module.bias is not None and bias0 is not None:
                terms.append((module.bias - bias0.to(module.bias.device)).pow(2).mean())
            for term in terms:
                reg = term if reg is None else reg + term
        return reg

    @torch.enable_grad()
    def forward(self, images):
        if self.episodic:
            self.reset()

        self.model.train()
        images = images.float()
        loss = None
        consistency = None
        entropy = None
        reg = None

        for _ in range(self.steps):
            logits = forward_logits(self.model, images)
            aug_images, inv_theta = build_dgtta_augmented_view(images, self.transform_strength)
            aug_logits = forward_logits(self.model, aug_images)
            aug_logits = align_logits_to_original(aug_logits, inv_theta)

            consistency = symmetric_kl(logits, aug_logits)
            entropy = softmax_entropy(logits)
            loss = consistency + self.entropy_weight * entropy

            reg = self._bn_regularization()
            if reg is not None:
                loss = loss + self.bn_l2_reg * reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            final_logits = forward_logits(self.model, images)

        self.last_losses = {
            "dgtta_loss": float(loss.detach().item()) if loss is not None else 0.0,
            "dgtta_consistency": float(consistency.detach().item()) if consistency is not None else 0.0,
            "dgtta_entropy": float(entropy.detach().item()) if entropy is not None else 0.0,
            "dgtta_bn_reg": float(reg.detach().item()) if reg is not None else 0.0,
        }
        return final_logits
