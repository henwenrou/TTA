from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def softmax_entropy_seg(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


class PromptCrossAttention(nn.Module):
    """PASS-style bottleneck prompt matching for DCON U-Net features."""

    def __init__(self, channels, sparsity=0.1):
        super().__init__()
        self.channels = int(channels)
        self.sparsity = float(sparsity)
        self.query = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels)
        self.key = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels)
        self.value = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels)

    def forward(self, x_q, x_kv):
        query = self.query(x_q)
        key = self.key(x_kv)
        value = self.value(x_kv)

        batch_size, channels, height, width = query.shape
        query = query.view(batch_size, channels, -1)
        key = key.view(batch_size, channels, -1).permute(0, 2, 1)
        value = value.view(batch_size, channels, -1)

        attn_weights = torch.matmul(query, key) / (channels ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        keep = max(1, int(self.sparsity * attn_weights.size(1)))
        sorted_values, sorted_indices = torch.sort(attn_weights, descending=True, dim=1)
        del sorted_values
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(1, sorted_indices[:, :keep, :], 1.0)
        attn_weights = attn_weights * mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.view(batch_size, channels, height, width)
        return x_q + attn_output, attn_weights


class PASSPromptedUNet(nn.Module):
    """Wrap DCON Unet1 with PASS input and bottleneck prompts.

    DCON's source U-Net is kept as the backbone. The wrapper adds the two PASS
    adaptation surfaces: an image-space data adaptor and a shape prompt injected
    into the bottleneck feature before decoding.
    """

    def __init__(
        self,
        backbone,
        prompt_size=12,
        adaptor_hidden=64,
        perturb_scale=1.0,
        prompt_scale=1.0,
        prompt_sparsity=0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.perturb_scale = float(perturb_scale)
        self.prompt_scale = float(prompt_scale)

        in_channels = int(self.backbone.convd1.conv1.in_channels)
        bottleneck_channels = int(self.backbone.convd5.conv3.out_channels)
        prompt_size = max(1, int(prompt_size))

        self.data_prompt = nn.Parameter(
            torch.zeros(1, bottleneck_channels, prompt_size, prompt_size)
        )
        self.prompt2feature = PromptCrossAttention(
            bottleneck_channels,
            sparsity=prompt_sparsity,
        )
        self.data_adaptor = nn.Sequential(
            nn.Conv2d(in_channels, adaptor_hidden, kernel_size=1),
            nn.InstanceNorm2d(adaptor_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(adaptor_hidden, in_channels, kernel_size=1),
        )

    def _prompt_bottleneck(self, x5):
        prompt = self.data_prompt
        if prompt.shape[-2:] != x5.shape[-2:]:
            prompt = F.interpolate(prompt, size=x5.shape[-2:], mode="bilinear", align_corners=False)
        prompt = prompt.expand(x5.shape[0], -1, -1, -1)
        prompted, attn_weights = self.prompt2feature(x5, prompt)
        prompted = x5 + self.prompt_scale * (prompted - x5)
        return prompted, attn_weights

    def forward(self, x, training=False, return_feat=False):
        perturbation = self.data_adaptor(x)
        x = x + self.perturb_scale * perturbation

        fcgsd_str, fcgsd_sty = None, None
        backbone = self.backbone

        x1 = backbone.convd1(x)
        if backbone.use_channel_gate and backbone.cgsd_layer == 1:
            x1_str, x1_sty = backbone.chan_gate(x1)
            x1_backbone = x1_str
            if return_feat:
                fcgsd_str = x1_backbone
                fcgsd_sty = x1_sty
        else:
            x1_backbone = x1

        x2 = backbone.convd2(x1_backbone)
        if backbone.use_channel_gate and backbone.cgsd_layer == 2:
            x2_str, x2_sty = backbone.chan_gate(x2)
            x2_backbone = x2_str
            if return_feat:
                fcgsd_str = x2_backbone
                fcgsd_sty = x2_sty
        else:
            x2_backbone = x2

        x3 = backbone.convd3(x2_backbone)
        if backbone.use_channel_gate and backbone.cgsd_layer == 3:
            x3_str, x3_sty = backbone.chan_gate(x3)
            x3_backbone = x3_str
            if return_feat:
                fcgsd_str = x3_backbone
                fcgsd_sty = x3_sty
        else:
            x3_backbone = x3

        x4 = backbone.convd4(x3_backbone)
        x5 = backbone.convd5(x4)
        x5_prompted, attn_weights = self._prompt_bottleneck(x5)

        y4 = backbone.convu4(x5_prompted, x4)
        y3 = backbone.convu3(y4, x3_backbone)
        y2 = backbone.convu2(y3, x2_backbone)
        y1 = backbone.convu1(y2, x1_backbone)
        pred = backbone.seg1(y1)

        if training:
            return pred, x5_prompted, perturbation, attn_weights
        if return_feat and backbone.use_channel_gate:
            return pred, x5_prompted, fcgsd_str, fcgsd_sty
        return pred, x5_prompted


class BatchNormFeatureMonitor:
    """Capture BN inputs and compare them with source running statistics."""

    def __init__(self, model, alpha=0.01, max_layers=0):
        self.alpha = float(alpha)
        self.max_layers = int(max_layers)
        self.records = []
        self.handles = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if module.running_mean is None or module.running_var is None:
                continue
            if self.max_layers > 0 and len(self.records) >= self.max_layers:
                break

            record = {
                "name": name,
                "mean": module.running_mean.detach().clone(),
                "var": module.running_var.detach().clone(),
                "input": None,
            }
            self.records.append(record)
            self.handles.append(module.register_forward_pre_hook(self._make_hook(record)))

        if len(self.records) == 0:
            raise RuntimeError("PASS requires BatchNorm2d running statistics, but none were found.")

    @staticmethod
    def _make_hook(record):
        def hook(_module, inputs):
            record["input"] = inputs[0]

        return hook

    def clear(self):
        for record in self.records:
            record["input"] = None

    def loss(self):
        losses = []
        for record in self.records:
            x = record["input"]
            if x is None:
                continue
            if x.dim() != 4:
                continue
            cur_mean = x.mean(dim=(0, 2, 3))
            cur_var = x.var(dim=(0, 2, 3), unbiased=False)
            src_mean = record["mean"].to(device=x.device, dtype=x.dtype)
            src_var = record["var"].to(device=x.device, dtype=x.dtype)
            losses.append(
                (cur_mean - src_mean).abs().mean()
                + self.alpha * (cur_var - src_var).abs().mean()
            )

        if len(losses) == 0:
            raise RuntimeError("PASS BN monitor did not capture any BatchNorm inputs.")
        return torch.stack([loss.reshape(()) for loss in losses]).mean()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def configure_pass_model(model, train_bn_affine=True):
    model.train()
    model.requires_grad_(False)

    trainable_names = []
    if train_bn_affine:
        for module_name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name in ("weight", "bias"):
                        param.requires_grad_(True)
                        trainable_names.append(f"{module_name}.{param_name}")

    prompt_tokens = ("data_prompt", "prompt2feature", "data_adaptor")
    for name, param in model.named_parameters():
        if any(token in name for token in prompt_tokens):
            param.requires_grad_(True)
            trainable_names.append(name)

    return trainable_names


class PASSAdapter:
    """Source-free PASS adapter for DCON segmentation."""

    def __init__(
        self,
        model,
        source_model,
        optimizer,
        bn_monitor,
        steps=1,
        entropy_weight=0.0,
        ema_decay=0.94,
        min_momentum_constant=0.01,
        episodic=False,
        use_source_fallback=True,
    ):
        self.model = model
        self.source_model = source_model
        self.optimizer = optimizer
        self.bn_monitor = bn_monitor
        self.steps = int(steps)
        self.entropy_weight = float(entropy_weight)
        self.ema_decay = float(ema_decay)
        self.min_momentum_constant = float(min_momentum_constant)
        self.episodic = bool(episodic)
        self.use_source_fallback = bool(use_source_fallback)
        self.momentum_prev = 0.1
        self.num_forwards = 0
        self.last_losses = {}

        if self.steps < 1:
            raise ValueError(f"PASS steps must be >= 1, got {self.steps}.")

        self.device = next(self.model.parameters()).device
        self.target_model = deepcopy(self.model).to(self.device)
        self.source_model = self.source_model.to(self.device)
        self.source_model.eval()
        self.target_model.eval()

        for frozen_model in (self.source_model, self.target_model):
            for param in frozen_model.parameters():
                param.detach_()
                param.requires_grad_(False)

        self.initial_model_state = deepcopy(self.model.state_dict())
        self.initial_target_state = deepcopy(self.target_model.state_dict())
        self.initial_optimizer_state = deepcopy(self.optimizer.state_dict())

    def reset(self):
        self.model.load_state_dict(self.initial_model_state, strict=True)
        self.target_model.load_state_dict(self.initial_target_state, strict=True)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        self.momentum_prev = 0.1

    def reset_online_from_target(self):
        self.model.load_state_dict(self.target_model.state_dict(), strict=True)

    @torch.no_grad()
    def update_target_network(self):
        momentum_new = self.momentum_prev * self.ema_decay
        momentum = min(1.0, momentum_new + self.min_momentum_constant)
        self.momentum_prev = momentum_new

        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            if online_param.requires_grad:
                target_param.data.mul_(1.0 - momentum).add_(online_param.data, alpha=momentum)

    @torch.enable_grad()
    def forward(self, images, is_start=False):
        self.num_forwards += 1
        if self.episodic:
            self.reset()
        else:
            self.reset_online_from_target()

        loss = None
        bn_loss = None
        entropy_loss = None
        logits = None

        self.model.train()
        for _ in range(self.steps):
            self.bn_monitor.clear()
            logits = _forward_logits(self.model, images)
            bn_loss = self.bn_monitor.loss()
            if self.entropy_weight > 0.0:
                entropy_loss = softmax_entropy_seg(logits)
                loss = bn_loss + self.entropy_weight * entropy_loss
            else:
                entropy_loss = logits.new_zeros(())
                loss = bn_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_target_network()

        self.model.eval()
        with torch.no_grad():
            adapted_logits = _forward_logits(self.model, images)
            source_logits = _forward_logits(self.source_model, images)
            adapted_entropy = softmax_entropy_seg(adapted_logits)
            source_entropy = softmax_entropy_seg(source_logits)

            if self.use_source_fallback and adapted_entropy > source_entropy:
                final_logits = source_logits
                used_source = True
            else:
                final_logits = adapted_logits
                used_source = False

        self.last_losses = {
            "loss": float(loss.detach().cpu()) if loss is not None else 0.0,
            "bn_loss": float(bn_loss.detach().cpu()) if bn_loss is not None else 0.0,
            "entropy_loss": float(entropy_loss.detach().cpu()) if entropy_loss is not None else 0.0,
            "adapted_entropy": float(adapted_entropy.detach().cpu()),
            "source_entropy": float(source_entropy.detach().cpu()),
            "used_source_fallback": float(used_source),
        }
        return final_logits
