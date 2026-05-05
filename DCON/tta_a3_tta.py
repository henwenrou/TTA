from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def configure_model_for_a3_tta(model):
    """A3-TTA updates the online segmentation model parameters."""
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    return model


@torch.no_grad()
def _update_ema_model(ema_model, model, momentum):
    momentum = float(max(0.0, min(0.9999, momentum)))
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)


def _cross_entropy_to_teacher(student_logits, teacher_logits):
    batch, classes, height, width = student_logits.shape
    normalizer = (
        batch
        * height
        * width
        * torch.log2(torch.tensor(float(classes), device=student_logits.device))
    )
    return -(teacher_logits.softmax(1) * student_logits.log_softmax(1)).sum() / normalizer


def _pixel_entropy(logits, prob=True):
    probs = F.softmax(logits, dim=1) if prob else logits
    return -(probs * torch.log(probs + 1e-5)).sum(1)


def _matrix_entropy(matrix, prob=False):
    probs = F.softmax(matrix, dim=1) if prob else matrix
    return -(probs * torch.log(probs + 1e-5)).mean()


class A3PrototypePool:
    """Memory pool used by A3-TTA for low-CCD bottleneck prototypes."""

    def __init__(self, max_length=40):
        self.max_length = int(max_length)
        self.memory = []

    def clear(self):
        self.memory = []

    @property
    def size(self):
        return len(self.memory)

    def update(self, features, scores):
        if len(features) != len(scores):
            raise ValueError(f"features/scores length mismatch: {len(features)} vs {len(scores)}")

        sorted_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]))
        for index in sorted_indices:
            feature = features[index].detach()
            score = float(scores[index])
            if len(self.memory) < self.max_length:
                self.memory.append((feature, score))
                continue

            worst_index = max(range(len(self.memory)), key=lambda i: self.memory[i][1])
            if score < self.memory[worst_index][1]:
                self.memory[worst_index] = (feature, score)

    def nearest(self, features, top_k=1):
        if len(self.memory) == 0:
            return features, features, features.new_zeros((features.shape[0],))

        memory_features = torch.stack([item[0] for item in self.memory]).to(
            device=features.device,
            dtype=features.dtype,
        )
        similarities = F.cosine_similarity(features.unsqueeze(1), memory_features.unsqueeze(0), dim=2)
        k = min(int(top_k), int(memory_features.shape[0]))
        indices = similarities.argsort(dim=1, descending=True)[:, :k]

        weights = []
        nearest_features = []
        for batch_index in range(features.shape[0]):
            sims = similarities[batch_index, indices[batch_index]]
            rate = sims.mean()
            weight = rate * torch.exp(sims) / torch.clamp(torch.exp(sims).sum(), min=1e-6)
            nearest_features.append(memory_features[indices[batch_index, 0]])
            weights.append(weight.mean())

        return torch.stack(nearest_features), features, torch.stack(weights)


class A3TTAAdapter:
    """A3-TTA adapted to DCON's Unet1 segmentation model.

    The official code expects get_feature/get_output hooks. DCON's U-Net exposes
    encoder/decoder blocks directly, so this adapter reconstructs the bottleneck
    feature path around convd1-5 and convu4-1.
    """

    def __init__(
        self,
        model,
        optimizer,
        num_classes,
        steps=1,
        mt_alpha=0.99,
        pool_size=40,
        top_k=1,
        feature_loss_weight=1.0,
        entropy_match_weight=5.0,
        ema_loss_weight=1.0,
        episodic=False,
        reset_on_scan_start=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.mt_alpha = float(mt_alpha)
        self.top_k = int(top_k)
        self.feature_loss_weight = float(feature_loss_weight)
        self.entropy_match_weight = float(entropy_match_weight)
        self.ema_loss_weight = float(ema_loss_weight)
        self.episodic = bool(episodic)
        self.reset_on_scan_start = bool(reset_on_scan_start)

        if self.steps < 1:
            raise ValueError(f"A3-TTA steps must be >= 1, got {self.steps}.")
        if self.top_k < 1:
            raise ValueError(f"A3-TTA top_k must be >= 1, got {self.top_k}.")

        self.device = next(self.model.parameters()).device
        self.model_anchor = deepcopy(self.model).to(self.device)
        self.model_ema = deepcopy(self.model).to(self.device)
        for teacher in (self.model_anchor, self.model_ema):
            teacher.eval()
            for param in teacher.parameters():
                param.detach_()
                param.requires_grad_(False)

        self.source_state = deepcopy(self.model.state_dict())
        self.ema_state = deepcopy(self.model_ema.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.pool = A3PrototypePool(max_length=pool_size)
        self.last_losses = {}

    def reset(self):
        self.model.load_state_dict(self.source_state, strict=True)
        self.model_ema.load_state_dict(self.ema_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.pool.clear()
        self.last_losses = {}

    def _encode(self, images, model):
        x1 = model.convd1(images)
        if getattr(model, "use_channel_gate", False) and getattr(model, "cgsd_layer", 1) == 1:
            x1, _ = model.chan_gate(x1)

        x2 = model.convd2(x1)
        if getattr(model, "use_channel_gate", False) and getattr(model, "cgsd_layer", 1) == 2:
            x2, _ = model.chan_gate(x2)

        x3 = model.convd3(x2)
        if getattr(model, "use_channel_gate", False) and getattr(model, "cgsd_layer", 1) == 3:
            x3, _ = model.chan_gate(x3)

        x4 = model.convd4(x3)
        x5 = model.convd5(x4)
        return x5, (x1, x2, x3, x4)

    def _decode(self, bottleneck, skips, model):
        x1, x2, x3, x4 = skips
        y4 = model.convu4(bottleneck, x4)
        y3 = model.convu3(y4, x3)
        y2 = model.convu2(y3, x2)
        y1 = model.convu1(y2, x1)
        return model.seg1(y1)

    @staticmethod
    def _flatten_bottleneck(features):
        batch, channels, height, width = features.shape
        return features.reshape(batch, channels * height * width), (channels, height, width)

    @staticmethod
    def _restore_bottleneck(flat_features, shape):
        channels, height, width = shape
        return flat_features.reshape(flat_features.shape[0], channels, height, width)

    @torch.no_grad()
    def _ccd_scores(self, images):
        scores = []
        self.model_anchor.eval()
        logits = _forward_logits(self.model_anchor, images)
        probs = logits.softmax(1)
        for index in range(probs.shape[0]):
            pred = probs[index:index + 1].permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            correlation = torch.matmul(pred.t(), pred) / float(max(pred.shape[0], 1))
            scores.append(float(_matrix_entropy(correlation, prob=False).detach().cpu()))
        return scores

    @torch.enable_grad()
    def _forward_and_adapt(self, images):
        self.model.train()
        ccd_scores = self._ccd_scores(images)

        bottleneck, skips = self._encode(images, self.model)
        logits = self._decode(bottleneck, skips, self.model)
        flat_feature, bottleneck_shape = self._flatten_bottleneck(bottleneck)

        self.pool.update(flat_feature, ccd_scores)
        pool_feature, current_feature, pool_weight = self.pool.nearest(flat_feature, top_k=self.top_k)

        mean = pool_feature.mean()
        std = torch.clamp(pool_feature.std(unbiased=False), min=1e-6)
        blended_feature = current_feature * (1.0 - pool_weight.view(-1, 1)) + pool_feature * pool_weight.view(-1, 1)
        aligned_feature = (blended_feature - mean) / std
        aligned_bottleneck = self._restore_bottleneck(aligned_feature, bottleneck_shape)
        aligned_logits = self._decode(aligned_bottleneck, skips, self.model)

        with torch.no_grad():
            ema_logits = _forward_logits(self.model_ema, images)

        feature_loss = _cross_entropy_to_teacher(aligned_logits, logits.detach())
        entropy_match_loss = F.l1_loss(_pixel_entropy(aligned_logits), _pixel_entropy(logits.detach()))
        ema_loss = _cross_entropy_to_teacher(logits, ema_logits)
        loss = (
            self.feature_loss_weight * feature_loss
            + self.entropy_match_weight * entropy_match_loss
            + self.ema_loss_weight * ema_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        dynamic_mt = 1.0 - float(ema_loss.detach().cpu())
        _update_ema_model(self.model_ema, self.model, dynamic_mt if self.mt_alpha < 0 else min(dynamic_mt, self.mt_alpha))

        self.last_losses = {
            "loss": float(loss.detach().cpu()),
            "feature_loss": float(feature_loss.detach().cpu()),
            "entropy_match_loss": float(entropy_match_loss.detach().cpu()),
            "ema_loss": float(ema_loss.detach().cpu()),
            "ema_momentum": float(max(0.0, min(0.9999, dynamic_mt if self.mt_alpha < 0 else min(dynamic_mt, self.mt_alpha)))),
            "pool_size": int(self.pool.size),
            "ccd_mean": float(sum(ccd_scores) / max(len(ccd_scores), 1)),
            "pool_weight_mean": float(pool_weight.detach().mean().cpu()),
        }

    @torch.enable_grad()
    def forward(self, images, names=None, is_start=False):
        del names
        if self.episodic or (self.reset_on_scan_start and is_start):
            self.reset()

        images = images.to(self.device)
        for _ in range(self.steps):
            self._forward_and_adapt(images)

        self.model.eval()
        with torch.no_grad():
            return _forward_logits(self.model, images)
