from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class VPTTABatchNorm2d(nn.BatchNorm2d):
    """BatchNorm layer used by VPTTA to expose the BN-stat matching loss."""

    def __init__(self, source_bn, warm_n=5):
        if not isinstance(source_bn, nn.BatchNorm2d):
            raise TypeError(f"Expected BatchNorm2d, got {type(source_bn)}")
        if not source_bn.track_running_stats:
            raise ValueError("VPTTA requires source BatchNorm running statistics.")

        super().__init__(
            source_bn.num_features,
            eps=source_bn.eps,
            momentum=source_bn.momentum,
            affine=source_bn.affine,
            track_running_stats=True,
        )
        device = source_bn.running_mean.device
        dtype = source_bn.running_mean.dtype
        self.to(device=device, dtype=dtype)
        self.load_state_dict(source_bn.state_dict(), strict=True)
        self.warm_n = int(warm_n)
        self.sample_num = 0
        self.new_sample = False
        self.bn_loss = torch.zeros(())

    def _target_statistics(self, x):
        if self.new_sample:
            self.sample_num += 1

        channels = x.shape[1]
        cur_mu = x.mean((0, 2, 3), keepdim=True).detach()
        cur_var = x.var((0, 2, 3), keepdim=True, unbiased=False).detach()

        src_mu = self.running_mean.view(1, channels, 1, 1)
        src_var = self.running_var.view(1, channels, 1, 1)

        warm_n = max(float(self.warm_n), 1.0)
        momentum = 1.0 / ((np.sqrt(max(self.sample_num, 1)) / warm_n) + 1.0)
        new_mu = momentum * cur_mu + (1.0 - momentum) * src_mu
        new_var = momentum * cur_var + (1.0 - momentum) * src_var
        return new_mu, new_var

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"VPTTABatchNorm2d expects a 4D tensor, got {x.shape}.")

        channels = x.shape[1]
        new_mu, new_var = self._target_statistics(x)

        cur_mu = x.mean((2, 3), keepdim=True)
        cur_std = x.std((2, 3), keepdim=True, unbiased=False)
        tgt_std = torch.sqrt(torch.clamp(new_var, min=self.eps))
        self.bn_loss = (new_mu - cur_mu).abs().mean() + (tgt_std - cur_std).abs().mean()

        weight = self.weight.view(1, channels, 1, 1) if self.affine else 1.0
        bias = self.bias.view(1, channels, 1, 1) if self.affine else 0.0
        return ((x - new_mu) / tgt_std) * weight + bias


def convert_batchnorm_to_vptta(module, warm_n=5):
    """Replace every BatchNorm2d in a module tree with VPTTA BatchNorm."""
    converted = 0
    for name, child in module.named_children():
        if isinstance(child, VPTTABatchNorm2d):
            continue
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, VPTTABatchNorm2d(child, warm_n=warm_n))
            converted += 1
        else:
            converted += convert_batchnorm_to_vptta(child, warm_n=warm_n)
    return converted


class FrequencyPrompt(nn.Module):
    """Low-frequency amplitude prompt from VPTTA, adapted to DCON inputs."""

    def __init__(self, in_channels=3, image_size=192, prompt_alpha=0.01, prompt_size=None):
        super().__init__()
        if prompt_size is None:
            prompt_size = int(image_size * prompt_alpha)
        prompt_size = max(int(prompt_size), 1)
        self.prompt_size = prompt_size
        self.data_prompt = nn.Parameter(torch.ones((1, int(in_channels), prompt_size, prompt_size)))

    def update(self, init_data):
        if init_data.dim() != 4:
            raise ValueError(f"Prompt init data must be 4D, got {init_data.shape}.")
        if init_data.shape[0] != 1:
            init_data = init_data.mean(dim=0, keepdim=True)
        if init_data.shape != self.data_prompt.shape:
            raise ValueError(
                f"Prompt init shape {tuple(init_data.shape)} does not match "
                f"{tuple(self.data_prompt.shape)}."
            )
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    @staticmethod
    def _ifft(amplitude, phase, height, width):
        real = torch.cos(phase) * amplitude
        imag = torch.sin(phase) * amplitude
        fft = torch.complex(real=real, imag=imag)
        return torch.fft.ifft2(fft, dim=(-2, -1), s=(height, width)).real

    def forward(self, x):
        batch, channels, height, width = x.shape
        prompt_size = self.prompt_size
        if channels != self.data_prompt.shape[1]:
            raise ValueError(
                f"Prompt channels={self.data_prompt.shape[1]} but input channels={channels}."
            )
        if prompt_size > min(height, width):
            raise ValueError(
                f"Prompt size {prompt_size} is larger than input spatial size {height}x{width}."
            )

        top = (height - prompt_size) // 2
        bottom = height - top - prompt_size
        left = (width - prompt_size) // 2
        right = width - left - prompt_size

        fft = torch.fft.fft2(x, dim=(-2, -1))
        amplitude = torch.fft.fftshift(torch.abs(fft), dim=(-2, -1))
        phase = torch.angle(fft)

        prompt = F.pad(self.data_prompt, [left, right, top, bottom], mode="constant", value=1.0)
        prompted_amplitude = torch.fft.ifftshift(amplitude * prompt, dim=(-2, -1))
        low_frequency = amplitude[:, :, top:top + prompt_size, left:left + prompt_size]
        prompted_x = self._ifft(prompted_amplitude, phase, height, width)
        return prompted_x, low_frequency


class PromptMemory:
    """Nearest-neighbor memory bank for VPTTA prompt initialization."""

    def __init__(self, size, dimension):
        self.size = int(size)
        self.dimension = int(dimension)
        self.memory = OrderedDict()

    def get_size(self):
        return len(self.memory)

    def push(self, keys, prompts):
        keys = np.asarray(keys, dtype=np.float32).reshape(len(keys), self.dimension)
        prompts = np.asarray(prompts, dtype=np.float32)
        if prompts.shape[0] == 1 and len(keys) > 1:
            prompts = np.repeat(prompts, len(keys), axis=0)

        for key, prompt in zip(keys, prompts):
            while len(self.memory) >= self.size > 0:
                self.memory.popitem(last=False)
            if self.size > 0:
                self.memory[key.tobytes()] = prompt.copy()

    def get_neighbours(self, keys, k):
        if len(self.memory) == 0:
            raise RuntimeError("Cannot query an empty VPTTA memory bank.")

        keys = np.asarray(keys, dtype=np.float32).reshape(len(keys), self.dimension)
        all_keys = np.stack(
            [np.frombuffer(key, dtype=np.float32) for key in self.memory.keys()],
            axis=0,
        )
        all_prompts = list(self.memory.values())
        k = min(int(k), len(all_prompts))
        samples = []

        all_norm = np.linalg.norm(all_keys, axis=1) + 1e-12
        for key in keys:
            key_norm = np.linalg.norm(key) + 1e-12
            similarity = np.dot(all_keys, key.T) / (all_norm * key_norm)
            neighbour_idx = np.argpartition(similarity, -k)[-k:]
            weights = similarity[neighbour_idx] / 0.2
            weights = weights - weights.max()
            weights = np.exp(weights)
            weights = weights / (weights.sum() + 1e-12)

            prompt = np.zeros_like(all_prompts[neighbour_idx[0]], dtype=np.float32)
            for idx, weight in zip(neighbour_idx, weights):
                prompt += all_prompts[idx] * weight
            samples.append(prompt)

        return torch.from_numpy(np.stack(samples, axis=0))


class VPTTAAdapter:
    """Source-free VPTTA adapter for DCON medical segmentation."""

    def __init__(
        self,
        model,
        prompt,
        optimizer,
        memory_bank,
        steps=1,
        neighbor=16,
    ):
        self.model = model
        self.prompt = prompt
        self.optimizer = optimizer
        self.memory_bank = memory_bank
        self.steps = int(steps)
        self.neighbor = int(neighbor)
        self.last_losses = {}
        self.num_forwards = 0

        if self.steps < 1:
            raise ValueError(f"VPTTA steps must be >= 1, got {self.steps}.")
        if self.neighbor < 1:
            raise ValueError(f"VPTTA neighbor must be >= 1, got {self.neighbor}.")

        self.model.eval()
        self.model.requires_grad_(False)

    def _vptta_bn_layers(self):
        return [module for module in self.model.modules() if isinstance(module, VPTTABatchNorm2d)]

    def _set_new_sample(self, new_sample):
        for module in self._vptta_bn_layers():
            module.new_sample = bool(new_sample)

    def _bn_loss(self):
        losses = [module.bn_loss for module in self._vptta_bn_layers()]
        if len(losses) == 0:
            raise RuntimeError("VPTTA requires converted VPTTABatchNorm2d layers.")
        return torch.stack([loss.reshape(()) for loss in losses]).mean()

    def _init_prompt(self, images):
        device = self.prompt.data_prompt.device
        dtype = self.prompt.data_prompt.dtype

        if self.memory_bank.get_size() >= self.neighbor:
            with torch.no_grad():
                _, low_frequency = self.prompt(images)
            init_data = self.memory_bank.get_neighbours(
                low_frequency.detach().cpu().numpy(),
                k=self.neighbor,
            ).to(device=device, dtype=dtype)
        else:
            init_data = torch.ones_like(self.prompt.data_prompt)

        self.prompt.update(init_data)

    @torch.enable_grad()
    def forward(self, images):
        self.num_forwards += 1
        self._init_prompt(images)

        self.model.eval()
        self.prompt.train()
        loss = None
        for step in range(self.steps):
            self._set_new_sample(step == 0)
            prompted_images, _ = self.prompt(images)
            _ = _forward_logits(self.model, prompted_images)
            loss = self._bn_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._set_new_sample(False)

        self.model.eval()
        self.prompt.eval()
        with torch.no_grad():
            prompted_images, low_frequency = self.prompt(images)
            logits = _forward_logits(self.model, prompted_images)

        self.memory_bank.push(
            low_frequency.detach().cpu().numpy(),
            self.prompt.data_prompt.detach().cpu().numpy(),
        )
        self.last_losses = {
            "bn_loss": float(loss.detach().cpu()) if loss is not None else 0.0,
            "memory_size": self.memory_bank.get_size(),
        }
        return logits
