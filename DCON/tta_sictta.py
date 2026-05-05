from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_model_for_sictta(model):
    """Use target-batch BN statistics, matching the official SicTTA setup."""
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def _forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class SicTTAPrototypePool:
    def __init__(self, max_length):
        self.max_length = int(max_length)
        self.feature_bank = None
        self.image_bank = None
        self.mask_bank = None
        self.name_list = []

    def clear(self):
        self.feature_bank = None
        self.image_bank = None
        self.mask_bank = None
        self.name_list = []

    @property
    def size(self):
        return 0 if self.feature_bank is None else int(self.feature_bank.shape[0])

    def get_pool_feature(self, feature, top_k):
        if self.feature_bank is None or self.feature_bank.numel() == 0:
            return feature, None, None, None, 0

        similarities = F.cosine_similarity(feature.unsqueeze(1), self.feature_bank.unsqueeze(0), dim=2)
        k = min(int(top_k), int(self.feature_bank.shape[0]))
        neighbour_idx = similarities.argsort(dim=1, descending=True)[:, :k]

        rates = similarities[0, neighbour_idx[0]].mean()
        weights = rates * torch.exp(similarities[0, neighbour_idx[0]])
        weights = weights / torch.clamp(weights.sum(), min=1e-6)

        mixed = feature * (1.0 - rates)
        for i in range(k):
            mixed = mixed + self.feature_bank[neighbour_idx[:, i]] * weights[i]

        return (
            mixed,
            self.feature_bank[neighbour_idx],
            self.image_bank[neighbour_idx] if self.image_bank is not None else None,
            self.mask_bank[neighbour_idx] if self.mask_bank is not None else None,
            self.size,
        )

    def _append_limited(self, bank, value):
        value = value.detach()
        if bank is None:
            bank = value
        else:
            bank = torch.cat([bank, value], dim=0)
        if bank.shape[0] > self.max_length:
            bank = bank[-self.max_length:]
        return bank

    def update(self, feature, image, mask, name=None):
        self.feature_bank = self._append_limited(self.feature_bank, feature)
        self.image_bank = self._append_limited(self.image_bank, image)
        self.mask_bank = self._append_limited(self.mask_bank, mask)
        if name is not None:
            self.name_list.append(str(name))
            if len(self.name_list) > self.max_length:
                self.name_list = self.name_list[-self.max_length:]


class SicTTAAdapter:
    """SicTTA adapter for DCON's Unet1 segmentation model.

    The official SicTTA code expects a model with get_feature/get_output hooks.
    DCON's Unet1 exposes its encoder/decoder blocks directly, so this adapter
    reconstructs that bottleneck-feature path around convd1-5 and convu4-1.
    """
    def __init__(
        self,
        model,
        num_classes,
        max_lens=40,
        topk=5,
        threshold=0.9,
        select_points=200,
        episodic=False,
    ):
        self.model = model
        self.model_anchor = deepcopy(model)
        self.model_anchor.eval()
        for param in self.model_anchor.parameters():
            param.detach_()
            param.requires_grad_(False)

        self.num_classes = int(num_classes)
        self.max_lens = int(max_lens)
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.select_points = int(select_points)
        self.episodic = bool(episodic)
        self.entropy_history = []
        self.pool = SicTTAPrototypePool(max_length=self.max_lens)
        self.last_stats = {}

    def reset(self):
        self.entropy_history = []
        self.pool.clear()
        self.last_stats = {}

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

    @staticmethod
    def _entropy(prob_matrix):
        probs = F.softmax(prob_matrix, dim=1)
        return -(probs * torch.log(probs + 1e-5)).sum(1).mean()

    @torch.no_grad()
    def _ccd_accepts(self, image, threshold):
        logits = _forward_logits(self.model_anchor, image)
        probs = logits.softmax(1).permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        if probs.shape[0] == 0:
            return False, None

        n_select = min(self.select_points, probs.shape[0])
        perm = torch.randperm(probs.shape[0], device=probs.device)[:n_select]
        sampled = F.normalize(probs[perm], dim=1)
        entropy_value = float(self._entropy(torch.matmul(sampled.t(), sampled)).detach().cpu())

        self.entropy_history.append(entropy_value)
        if len(self.entropy_history) > self.max_lens:
            self.entropy_history = self.entropy_history[-self.max_lens:]

        sorted_history = sorted(self.entropy_history)
        cutoff_index = int(len(sorted_history) * (1.0 - threshold))
        if cutoff_index <= 0:
            return False, entropy_value

        cutoff = sorted_history[:cutoff_index][-1]
        return entropy_value <= cutoff, entropy_value

    @torch.no_grad()
    def forward(self, images, names=None):
        if images.shape[0] != 1:
            raise ValueError(f"SicTTA is a single-image adapter; got batch size {images.shape[0]}.")

        if self.episodic:
            self.reset()

        device = next(self.model.parameters()).device
        images = images.to(device)
        self.model.train()

        bottleneck, _ = self._encode(images, self.model)
        flat_feature, bottleneck_shape = self._flatten_bottleneck(bottleneck)
        mixed_feature, _, pool_images, _, pool_size = self.pool.get_pool_feature(flat_feature, self.topk)

        threshold = self.threshold if pool_size >= self.topk else pool_size / float(max(self.topk, 1))
        accepted, entropy_value = self._ccd_accepts(images, threshold=threshold)

        current_logits = _forward_logits(self.model, images)
        if accepted:
            name = None
            if names is not None:
                name = names[0] if isinstance(names, (list, tuple)) else names
            self.pool.update(flat_feature, images, current_logits.softmax(1), name=name)

        if pool_images is not None:
            reference_images = pool_images[0]
            joint_images = torch.cat((images, reference_images), dim=0)
            joint_bottleneck, joint_skips = self._encode(joint_images, self.model)
            joint_bottleneck[0:1] = self._restore_bottleneck(mixed_feature, bottleneck_shape)
            output = self._decode(joint_bottleneck, joint_skips, self.model)[0:1]
        else:
            output = _forward_logits(self.model_anchor, images)

        self.last_stats = {
            "accepted": bool(accepted),
            "pool_size": int(self.pool.size),
            "threshold": float(threshold),
            "ccd_entropy": entropy_value,
        }
        return output
