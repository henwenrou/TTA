from copy import deepcopy
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_logits_and_features(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        logits = output[0]
        features = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else None
    else:
        logits = output
        features = None
    return logits, features


def _softmax_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


def _soft_dice_loss(probs, targets, eps=1e-6):
    if probs.shape[2:] != targets.shape[2:]:
        targets = F.interpolate(targets, size=probs.shape[2:], mode="bilinear", align_corners=False)

    dims = (0, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = (probs + targets).sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _spatial_kl_per_channel(teacher, student, temp=2.0, eps=1e-12):
    if teacher is None or student is None:
        return None
    if teacher.shape[2:] != student.shape[2:]:
        student = F.interpolate(student, size=teacher.shape[2:], mode="bilinear", align_corners=False)
    if teacher.shape[1] != student.shape[1]:
        return student.new_zeros(())

    t = teacher.detach().flatten(2) / max(float(temp), eps)
    s = student.flatten(2) / max(float(temp), eps)

    t_logp = F.log_softmax(t, dim=2)
    s_logp = F.log_softmax(s, dim=2)
    t_prob = t_logp.exp()
    return (t_prob * (t_logp - s_logp)).sum(dim=2).mean()


def _safe_logit(values, eps=1e-4):
    values = torch.clamp(values, eps, 1.0 - eps)
    return torch.log(values / (1.0 - values))


class SAMTTABezierTransform(nn.Module):
    """SAM-TTA-style learnable cubic Bezier intensity transform for DCON inputs."""

    def __init__(self, num_control_points=4):
        super().__init__()
        if num_control_points != 4:
            raise ValueError("Only cubic Bezier transforms with 4 control points are supported.")

        base = torch.tensor([1e-4, 1.0 / 3.0, 2.0 / 3.0, 1.0 - 1e-4], dtype=torch.float32)
        control_points = torch.stack([base.clone(), base.clone(), base.clone()], dim=0)
        self.control_logits = nn.Parameter(_safe_logit(control_points))

    @staticmethod
    def _bezier_curve(control_points, x):
        p0, p1, p2, p3 = control_points
        t = x
        omt = 1.0 - t
        out = (
            comb(3, 0) * omt.pow(3) * p0
            + comb(3, 1) * t * omt.pow(2) * p1
            + comb(3, 2) * t.pow(2) * omt * p2
            + comb(3, 3) * t.pow(3) * p3
        )
        return out.clamp(0.0, 1.0)

    def forward(self, images):
        if images.dim() != 4:
            raise ValueError(f"Expected images [B,C,H,W], got {tuple(images.shape)}.")

        base = images.mean(dim=1, keepdim=True)
        vmin = base.amin(dim=(2, 3), keepdim=True)
        vmax = base.amax(dim=(2, 3), keepdim=True)
        scale = (vmax - vmin).clamp_min(1e-6)
        unit = ((base - vmin) / scale).clamp(0.0, 1.0)

        cps = torch.sigmoid(self.control_logits).to(device=images.device, dtype=images.dtype)
        channels = [self._bezier_curve(cps[idx], unit) for idx in range(3)]
        transformed_unit = torch.cat(channels, dim=1)
        return transformed_unit * scale + vmin


def configure_model_for_samtta(model, update_scope="bn_affine"):
    """Prepare the DCON U-Net trainable surface used by the SAM-TTA adapter."""

    model.train()
    model.requires_grad_(False)

    params = []
    names = []
    if update_scope == "transform_only":
        return params, names

    if update_scope == "bn_affine":
        for module_name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name in ("weight", "bias"):
                        params.append(param)
                        names.append(f"{module_name}.{param_name}")
        if len(params) == 0:
            raise RuntimeError("SAM-TTA bn_affine mode requires BatchNorm2d affine parameters.")
        return params, names

    if update_scope == "all":
        model.requires_grad_(True)
        for name, param in model.named_parameters():
            params.append(param)
            names.append(name)
        return params, names

    raise ValueError(f"Unknown SAM-TTA update scope: {update_scope}")


class SAMTTAAdapter:
    """Source-free SAM-TTA adaptation core ported to DCON U-Net segmentation.

    The original SAM-TTA updates a SAM LoRA/prompt student plus a learnable
    Bezier input transform, guided by an EMA teacher, IoU confidence, prediction
    consistency and feature consistency. DCON has no SAM IoU head, so this
    adapter uses teacher softmax confidence as the reliability weight and
    applies the same idea to U-Net logits and bottleneck features.
    """

    def __init__(
        self,
        model,
        transform,
        optimizer,
        steps=1,
        ema_momentum=0.95,
        dpc_weight=1.0,
        feature_weight=0.1,
        entropy_weight=0.05,
        transform_reg_weight=0.01,
        feature_temp=2.0,
        episodic=False,
    ):
        self.model = model
        self.transform = transform
        self.optimizer = optimizer
        self.steps = int(steps)
        self.ema_momentum = float(ema_momentum)
        self.dpc_weight = float(dpc_weight)
        self.feature_weight = float(feature_weight)
        self.entropy_weight = float(entropy_weight)
        self.transform_reg_weight = float(transform_reg_weight)
        self.feature_temp = float(feature_temp)
        self.episodic = bool(episodic)

        self.teacher = deepcopy(model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.detach_()
            param.requires_grad_(False)

        self.model_state = deepcopy(model.state_dict()) if self.episodic else None
        self.teacher_state = deepcopy(self.teacher.state_dict()) if self.episodic else None
        self.transform_state = deepcopy(transform.state_dict()) if self.episodic else None
        self.optimizer_state = deepcopy(optimizer.state_dict()) if self.episodic else None
        self.last_losses = {}

    def reset(self):
        if not self.episodic:
            return
        self.model.load_state_dict(self.model_state, strict=True)
        self.teacher.load_state_dict(self.teacher_state, strict=True)
        self.transform.load_state_dict(self.transform_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @torch.no_grad()
    def _update_teacher(self):
        for teacher_param, student_param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data.mul_(self.ema_momentum).add_(student_param.data, alpha=1.0 - self.ema_momentum)

    def _dual_scale_prediction_consistency(self, student_logits, teacher_logits, feature_hw=None):
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
        high_loss = _soft_dice_loss(student_probs, teacher_probs)

        if feature_hw is None:
            return high_loss, high_loss

        student_low = F.softmax(
            F.interpolate(student_logits, size=feature_hw, mode="bilinear", align_corners=False),
            dim=1,
        )
        teacher_low = F.softmax(
            F.interpolate(teacher_logits.detach(), size=feature_hw, mode="bilinear", align_corners=False),
            dim=1,
        )
        low_loss = _soft_dice_loss(student_low, teacher_low)
        return 0.5 * (high_loss + low_loss), low_loss

    @torch.enable_grad()
    def forward(self, images):
        if self.episodic:
            self.reset()

        images = images.float()
        self.model.train()
        self.teacher.eval()

        loss = None
        dpc_loss = None
        low_loss = None
        feat_loss = None
        entropy_loss = None
        transform_reg = None
        teacher_conf = None

        for _ in range(self.steps):
            transformed = self.transform(images)

            with torch.no_grad():
                teacher_logits, teacher_features = _forward_logits_and_features(self.teacher, transformed.detach())
                if teacher_logits.shape[2:] != images.shape[2:]:
                    teacher_logits = F.interpolate(
                        teacher_logits,
                        size=images.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                teacher_prob = F.softmax(teacher_logits, dim=1)
                teacher_conf = teacher_prob.max(dim=1)[0].mean().clamp(0.05, 1.0)

            student_logits, student_features = _forward_logits_and_features(self.model, transformed)
            if student_logits.shape[2:] != images.shape[2:]:
                student_logits = F.interpolate(
                    student_logits,
                    size=images.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            feature_hw = student_features.shape[2:] if student_features is not None else None
            dpc_loss, low_loss = self._dual_scale_prediction_consistency(
                student_logits,
                teacher_logits,
                feature_hw=feature_hw,
            )
            feat_loss = _spatial_kl_per_channel(
                teacher_features,
                student_features,
                temp=self.feature_temp,
            )
            if feat_loss is None:
                feat_loss = student_logits.new_zeros(())
            entropy_loss = _softmax_entropy(student_logits)
            transform_reg = F.mse_loss(transformed, images)

            loss = (
                self.dpc_weight * teacher_conf.detach() * dpc_loss
                + self.feature_weight * feat_loss
                + self.entropy_weight * entropy_loss
                + self.transform_reg_weight * transform_reg
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._update_teacher()

        self.model.eval()
        with torch.no_grad():
            final_images = self.transform(images)
            final_logits, _ = _forward_logits_and_features(self.model, final_images)
            if final_logits.shape[2:] != images.shape[2:]:
                final_logits = F.interpolate(
                    final_logits,
                    size=images.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

        self.last_losses = {
            "samtta_loss": float(loss.detach().item()) if loss is not None else 0.0,
            "samtta_dpc": float(dpc_loss.detach().item()) if dpc_loss is not None else 0.0,
            "samtta_low_dpc": float(low_loss.detach().item()) if low_loss is not None else 0.0,
            "samtta_feature": float(feat_loss.detach().item()) if feat_loss is not None else 0.0,
            "samtta_entropy": float(entropy_loss.detach().item()) if entropy_loss is not None else 0.0,
            "samtta_transform_reg": float(transform_reg.detach().item()) if transform_reg is not None else 0.0,
            "samtta_teacher_conf": float(teacher_conf.detach().item()) if teacher_conf is not None else 0.0,
        }
        return final_logits
