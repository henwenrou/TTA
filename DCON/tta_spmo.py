from copy import deepcopy

import torch
import torch.nn.functional as F


def forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def class_entropy(probs, weights):
    log_probs = torch.log(probs.clamp_min(1e-8))
    return -(probs * log_probs * weights.view(1, -1, 1, 1)).sum(dim=1).mean()


def norm_soft_size(probs, power=1.0, eps=1e-8):
    max_prob = probs.max(dim=1, keepdim=True)[0].clamp_min(eps)
    response = torch.pow(probs / max_prob, float(power))
    sizes = response.sum(dim=(2, 3))
    return sizes / sizes.sum(dim=1, keepdim=True).clamp_min(eps)


def proportion_kl(probs, prior_props, power=1.0, eps=1e-8):
    props = norm_soft_size(probs, power=power, eps=eps).clamp_min(eps)
    props = props / props.sum(dim=1, keepdim=True).clamp_min(eps)
    prior_props = prior_props.clamp_min(eps)
    prior_props = prior_props / prior_props.sum(dim=1, keepdim=True).clamp_min(eps)
    return (props * (torch.log(props) - torch.log(prior_props))).sum(dim=1).mean()


def hard_one_hot(logits, num_classes):
    labels = logits.argmax(dim=1)
    return F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).to(logits.dtype)


def soft_centroid(mask, eps=1e-8):
    batch, channels, height, width = mask.shape
    y = torch.linspace(0.0, 1.0, height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
    mass = mask.sum(dim=(2, 3)).clamp_min(eps)
    cy = (mask * y).sum(dim=(2, 3)) / mass
    cx = (mask * x).sum(dim=(2, 3)) / mass
    return torch.stack([cy, cx], dim=2)


def soft_spread(mask, centroid=None, eps=1e-8):
    batch, channels, height, width = mask.shape
    if centroid is None:
        centroid = soft_centroid(mask, eps=eps)
    y = torch.linspace(0.0, 1.0, height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
    mass = mask.sum(dim=(2, 3)).clamp_min(eps)
    dy2 = (y - centroid[:, :, 0].view(batch, channels, 1, 1)).pow(2)
    dx2 = (x - centroid[:, :, 1].view(batch, channels, 1, 1)).pow(2)
    sy = ((mask * dy2).sum(dim=(2, 3)) / mass).clamp_min(eps).sqrt()
    sx = ((mask * dx2).sum(dim=(2, 3)) / mass).clamp_min(eps).sqrt()
    return torch.stack([sy, sx], dim=2)


def shape_moment_loss(probs, prior_mask, mode="all", min_pixels=10, eps=1e-8):
    if mode == "none":
        return probs.sum() * 0.0

    num_classes = probs.shape[1]
    if num_classes <= 1:
        return probs.sum() * 0.0

    fg = slice(1, None)
    valid = prior_mask[:, fg].sum(dim=(2, 3)) >= float(min_pixels)
    if not valid.any():
        return probs.sum() * 0.0

    losses = []
    if mode in ("centroid", "all"):
        pred_centroid = soft_centroid(probs, eps=eps)[:, fg]
        prior_centroid = soft_centroid(prior_mask, eps=eps)[:, fg].detach()
        centroid_diff = (pred_centroid - prior_centroid).pow(2).sum(dim=2)
        losses.append(centroid_diff[valid].mean())

    if mode in ("dist_centroid", "all"):
        pred_centroid = soft_centroid(probs, eps=eps)
        prior_centroid = soft_centroid(prior_mask, eps=eps)
        pred_spread = soft_spread(probs, centroid=pred_centroid, eps=eps)[:, fg]
        prior_spread = soft_spread(prior_mask, centroid=prior_centroid, eps=eps)[:, fg].detach()
        spread_diff = (pred_spread - prior_spread).pow(2).sum(dim=2)
        losses.append(spread_diff[valid].mean())

    if not losses:
        raise ValueError(f"Unknown SPMO moment mode: {mode}")
    return torch.stack([loss.reshape(()) for loss in losses]).sum()


class SPMOAdapter:
    """SPMO-TTA core adapted to DCON's model, loader, and evaluator.

    The original SPMO-TTA code reads per-slice source predictions from a
    size CSV and optimizes entropy plus size/moment constraints. This adapter
    builds the same kind of source-prediction prior on the fly with a frozen
    source model, avoiding a separate data conversion or CSV generation step.
    """

    def __init__(
        self,
        model,
        source_model,
        optimizer,
        num_classes,
        steps=1,
        entropy_weight=1.0,
        prior_weight=1.0,
        moment_weight=0.05,
        moment_mode="all",
        softmax_temp=1.0,
        size_power=1.0,
        bg_entropy_weight=0.1,
        prior_eps=1e-6,
        min_pixels=10,
        source_pseudo="hard",
        episodic=False,
    ):
        self.model = model
        self.source_model = source_model
        self.optimizer = optimizer
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.entropy_weight = float(entropy_weight)
        self.prior_weight = float(prior_weight)
        self.moment_weight = float(moment_weight)
        self.moment_mode = str(moment_mode)
        self.softmax_temp = float(softmax_temp)
        self.size_power = float(size_power)
        self.bg_entropy_weight = float(bg_entropy_weight)
        self.prior_eps = float(prior_eps)
        self.min_pixels = int(min_pixels)
        self.source_pseudo = str(source_pseudo)
        self.episodic = bool(episodic)
        self.model_state = deepcopy(model.state_dict()) if self.episodic else None
        self.optimizer_state = deepcopy(optimizer.state_dict()) if self.episodic else None
        self.last_losses = {}

        if self.steps < 1:
            raise ValueError(f"SPMO steps must be >= 1, got {self.steps}.")
        if self.moment_mode not in ("none", "centroid", "dist_centroid", "all"):
            raise ValueError(f"Unknown SPMO moment mode: {self.moment_mode}")
        if self.source_pseudo not in ("hard", "soft"):
            raise ValueError(f"Unknown SPMO source pseudo mode: {self.source_pseudo}")

        self.source_model.eval()
        self.source_model.requires_grad_(False)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            return
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @torch.no_grad()
    def source_prior(self, images):
        logits = forward_logits(self.source_model, images)
        source_probs = F.softmax(logits, dim=1)
        if self.source_pseudo == "hard":
            source_mask = hard_one_hot(logits, self.num_classes)
        else:
            source_mask = source_probs
        prior_props = source_mask.sum(dim=(2, 3))
        prior_props = prior_props / prior_props.sum(dim=1, keepdim=True).clamp_min(self.prior_eps)
        return source_mask.detach(), prior_props.detach()

    @torch.enable_grad()
    def forward(self, images):
        if self.episodic:
            self.reset()

        images = images.float()
        source_mask, prior_props = self.source_prior(images)
        weights = torch.ones(self.num_classes, device=images.device, dtype=images.dtype)
        weights[0] = self.bg_entropy_weight

        self.model.train()
        loss = None
        entropy = None
        size_prior = None
        moment = None

        for _ in range(self.steps):
            logits = forward_logits(self.model, images)
            probs = F.softmax(logits / self.softmax_temp, dim=1)
            entropy = class_entropy(probs, weights)
            size_prior = proportion_kl(
                probs,
                prior_props,
                power=self.size_power,
                eps=self.prior_eps,
            )
            moment = shape_moment_loss(
                probs,
                source_mask,
                mode=self.moment_mode,
                min_pixels=self.min_pixels,
                eps=self.prior_eps,
            )
            loss = (
                self.entropy_weight * entropy
                + self.prior_weight * size_prior
                + self.moment_weight * moment
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.train()
        with torch.no_grad():
            final_logits = forward_logits(self.model, images)

        self.last_losses = {
            "spmo_loss": float(loss.detach().cpu()) if loss is not None else 0.0,
            "spmo_entropy": float(entropy.detach().cpu()) if entropy is not None else 0.0,
            "spmo_size_prior": float(size_prior.detach().cpu()) if size_prior is not None else 0.0,
            "spmo_moment": float(moment.detach().cpu()) if moment is not None else 0.0,
        }
        return final_logits
