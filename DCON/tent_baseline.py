"""Episodic TENT baseline for DCON segmentation reliability evaluation."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


def softmax_entropy_map(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def entropy_loss(logits):
    return softmax_entropy_map(logits).mean()


def configure_tent_norm_affine(model):
    """Configure model for classic TENT: update normalization affine only."""
    model.train()
    model.requires_grad_(False)
    names = []
    params = []
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ("weight", "bias") and param is not None:
                    param.requires_grad_(True)
                    names.append(f"{module_name}.{param_name}")
                    params.append(param)
    if not params:
        raise RuntimeError("TENT found no normalization affine parameters to update.")
    return params, names


def _forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class EpisodicTent:
    """Reset-before-forward TENT adapter.

    The adapter never reads target labels. By default it restores the source
    checkpoint state before every target batch/case, which avoids target-domain
    accumulation across evaluation samples.
    """

    def __init__(self, model, lr=1e-4, steps=1, reset_each_batch=True):
        self.model = model
        self.lr = float(lr)
        self.steps = max(1, int(steps))
        self.reset_each_batch = bool(reset_each_batch)
        self.source_state = deepcopy(model.state_dict())
        self.params, self.param_names = configure_tent_norm_affine(self.model)
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.last_loss = None

    def reset(self):
        self.model.load_state_dict(self.source_state, strict=False)
        self.params, self.param_names = configure_tent_norm_affine(self.model)
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.last_loss = None

    @torch.enable_grad()
    def forward(self, images):
        if self.reset_each_batch:
            self.reset()
        logits = None
        self.model.train()
        for _ in range(self.steps):
            logits = _forward_logits(self.model, images)
            loss = entropy_loss(logits)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().item())
        with torch.no_grad():
            logits = _forward_logits(self.model, images)
        return logits.detach()
