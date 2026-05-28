from copy import deepcopy

import torch
import torch.nn.functional as F


def forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def softmax_entropy_seg(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


def _apply_weak_view(x, factor):
    if factor == 0:
        return x
    if factor == 1:
        return torch.rot90(x, 1, dims=(-2, -1))
    if factor == 2:
        return torch.rot90(x, 2, dims=(-2, -1))
    if factor == 3:
        return torch.rot90(x, 3, dims=(-2, -1))
    if factor == 4:
        return torch.flip(x, dims=(-1,))
    raise ValueError(f"Unsupported weak-view factor: {factor}")


def _invert_weak_view(x, factor):
    if factor == 0:
        return x
    if factor == 1:
        return torch.rot90(x, -1, dims=(-2, -1))
    if factor == 2:
        return torch.rot90(x, 2, dims=(-2, -1))
    if factor == 3:
        return torch.rot90(x, -3, dims=(-2, -1))
    if factor == 4:
        return torch.flip(x, dims=(-1,))
    raise ValueError(f"Unsupported weak-view factor: {factor}")


def _intensity_style_aug(images, strength=1.0):
    if strength <= 0:
        return images
    scale = 1.0 + (torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) - 0.5) * (0.35 * strength)
    shift = (torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) - 0.5) * (0.20 * strength)
    noise = torch.randn_like(images) * (0.015 * strength)
    return (images * scale + shift + noise).contiguous()


class GraTaAdapter:
    """Gradient-alignment TTA adapted to DCON's U-Net and NIfTI slice batches.

    This keeps the GraTa update rule but uses DCON-compatible losses:
    entropy as the auxiliary target loss and weak/strong prediction consistency
    as the pseudo loss. Target labels are never read by this adapter.
    """

    def __init__(
        self,
        model,
        params,
        base_optimizer,
        device,
        aux_loss="ent",
        pse_loss="consis",
        steps=1,
        weak_views=5,
        style_strength=1.0,
        perturb_eps=1e-12,
        episodic=False,
    ):
        self.model = model
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.device = device
        self.aux_loss = aux_loss
        self.pse_loss = pse_loss
        self.steps = int(steps)
        self.weak_views = int(weak_views)
        self.style_strength = float(style_strength)
        self.perturb_eps = float(perturb_eps)
        self.episodic = bool(episodic)
        self.init_lr = self.base_optimizer.param_groups[0]["lr"]
        self.source_state = deepcopy(self.model.state_dict())
        self._old_params = {}
        self._aux_grads = {}
        self.last_losses = {}

        supported = {"ent", "consis"}
        if self.aux_loss not in supported:
            raise ValueError(f"GraTa DCON adapter supports aux_loss in {sorted(supported)}, got {self.aux_loss}")
        if self.pse_loss not in supported:
            raise ValueError(f"GraTa DCON adapter supports pse_loss in {sorted(supported)}, got {self.pse_loss}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if not 1 <= self.weak_views <= 5:
            raise ValueError(f"weak_views must be in [1, 5], got {self.weak_views}")
        if len(self.params) == 0:
            raise ValueError("GraTa requires at least one trainable parameter.")

    def reset(self):
        self.model.load_state_dict(self.source_state, strict=True)
        self.base_optimizer.state.clear()
        self._old_params.clear()
        self._aux_grads.clear()

    def _loss(self, images, loss_name):
        if loss_name == "ent":
            return self._entropy_loss(images)
        if loss_name == "consis":
            return self._consistency_loss(images)
        raise ValueError(f"Unsupported GraTa loss: {loss_name}")

    def _entropy_loss(self, images):
        logits = forward_logits(self.model, images)
        return softmax_entropy_seg(logits)

    def _consistency_loss(self, images):
        with torch.no_grad():
            probs = []
            for factor in range(self.weak_views):
                weak_images = _apply_weak_view(images, factor)
                weak_logits = forward_logits(self.model, weak_images)
                weak_logits = _invert_weak_view(weak_logits, factor)
                probs.append(F.softmax(weak_logits, dim=1))
            target_probs = torch.stack(probs, dim=0).mean(dim=0).detach()

        strong_images = _intensity_style_aug(images, self.style_strength)
        strong_logits = forward_logits(self.model, strong_images)
        if strong_logits.shape[2:] != target_probs.shape[2:]:
            strong_logits = F.interpolate(
                strong_logits,
                size=target_probs.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return -(target_probs * F.log_softmax(strong_logits, dim=1)).sum(1).mean()

    @torch.no_grad()
    def _store_and_subtract_grad(self):
        self._old_params.clear()
        self._aux_grads.clear()
        for param in self.params:
            if param.grad is None:
                continue
            self._old_params[param] = param.data.clone()
            self._aux_grads[param] = param.grad.detach().clone()
            param.data.sub_(param.grad)

    @torch.no_grad()
    def _restore_params(self):
        for param, old_param in self._old_params.items():
            param.data.copy_(old_param)

    @torch.no_grad()
    def _grad_norm(self, key=None):
        vals = []
        for param in self.params:
            if key is None:
                grad = param.grad
            else:
                if key != "grata_aux_g":
                    raise ValueError(f"Unsupported GraTa gradient cache key: {key}")
                grad = self._aux_grads.get(param, None)
            if grad is not None:
                vals.append(grad.norm(p=2))
        if len(vals) == 0:
            return torch.zeros((), device=self.device)
        return torch.norm(torch.stack([v.to(self.device) for v in vals]), p=2)

    @torch.no_grad()
    def _cosine(self):
        inner = torch.zeros((), device=self.device)
        for param in self.params:
            aux_grad = self._aux_grads.get(param, None)
            if aux_grad is None or param.grad is None:
                continue
            inner = inner + torch.sum(aux_grad.to(param.grad.device) * param.grad.detach()).to(self.device)
        return inner / (self._grad_norm() * self._grad_norm("grata_aux_g") + self.perturb_eps)

    @staticmethod
    def _lr_scale(cosine):
        return 0.25 * (cosine + 1.0).pow(2).clamp_min(0.0)

    def _adapt_once(self, images):
        self.base_optimizer.zero_grad()
        aux = self._loss(images, self.aux_loss)
        aux.backward()

        self._store_and_subtract_grad()

        self.base_optimizer.zero_grad()
        pse = self._loss(images, self.pse_loss)
        pse.backward()

        cosine = self._cosine()
        self._restore_params()

        lr_scale = self._lr_scale(cosine.detach())
        self.base_optimizer.param_groups[0]["lr"] = self.init_lr * float(lr_scale.item())
        self.base_optimizer.step()

        return aux.detach(), pse.detach(), cosine.detach(), lr_scale.detach()

    def forward(self, images):
        if self.episodic:
            self.reset()

        images = images.to(self.device).float()
        self.model.train()

        aux = pse = cosine = lr_scale = None
        for _ in range(self.steps):
            aux, pse, cosine, lr_scale = self._adapt_once(images)

        with torch.no_grad():
            logits = forward_logits(self.model, images)

        self.last_losses = {
            "grata_loss_aux": float(aux.item()) if aux is not None else 0.0,
            "grata_loss_pse": float(pse.item()) if pse is not None else 0.0,
            "grata_grad_cosine": float(cosine.item()) if cosine is not None else 0.0,
            "grata_lr_scale": float(lr_scale.item()) if lr_scale is not None else 0.0,
            "grata_updated_params": float(sum(param.numel() for param in self.params)),
        }
        return logits
