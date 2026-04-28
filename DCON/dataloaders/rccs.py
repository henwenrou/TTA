"""
RCCS: Random Convolution Candidate Selection

Select the semantically closest sample from multiple ProRandConv candidates.
This module implements the candidate-selection step used by RCCS.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def gaussian_random_field(amplitude, flag_normalize=True):
    noise = (
        torch.randn(*amplitude.shape, dtype=torch.complex64, device=amplitude.device)
        + 1j * torch.randn(*amplitude.shape, dtype=torch.complex64, device=amplitude.device)
    )

    gfield = torch.fft.ifft2(noise * amplitude).real

    if flag_normalize:
        gfield = gfield - torch.mean(gfield, dim=(-1, -2), keepdim=True)
        gfield = gfield / torch.std(gfield, dim=(-1, -2), keepdim=True)

    return gfield


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma_gamma, sigma_beta, epsilon):
        super().__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        self.k = kernel_size
        self.sigma_w = 1 / np.sqrt(self.k * self.k * self.c_in)
        self.sigma_gamma = sigma_gamma
        self.sigma_beta = sigma_beta
        self.epsilon = epsilon

        y, x = torch.meshgrid(
            torch.arange(-(self.k // 2), self.k // 2 + 1),
            torch.arange(-(self.k // 2), self.k // 2 + 1),
        )
        grid = y ** 2 + x ** 2
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0), persistent=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.k // 2)
        self.reset_parameters()

    def reset_parameters(self):
        exp_factor = torch.exp(-self.grid / (2 * self.sigma_gamma ** 2))
        with torch.no_grad():
            exp_factor = exp_factor.expand(self.c_in, self.c_out, self.k, self.k)
            self.conv.weight.copy_(
                exp_factor
                * torch.randn(
                    self.c_out,
                    self.c_in,
                    self.k,
                    self.k,
                    device=self.conv.weight.device,
                )
                * self.sigma_w
            )

    def forward(self, x):
        return self.conv(x)


class ProRandConvNet(nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.c_in = 3
        self.c_out = 3
        self.k = 3
        self.l_min = 1
        self.l_max = 10
        self.sigma_w = 1 / np.sqrt(self.k * self.k * self.c_in)
        y, x = torch.meshgrid(
            torch.arange(-(self.k // 2), self.k // 2 + 1),
            torch.arange(-(self.k // 2), self.k // 2 + 1),
        )
        grid = y ** 2 + x ** 2
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0), persistent=True)
        self.sigma_gamma = 0.5
        self.sigma_beta = 0.5
        self.epsilon = 1e-4
        self.b_delta = 0.2 if size < 128 else 0.5
        k_ind = torch.meshgrid(
            torch.arange(-int((size + 1) / 2), -int((size + 1) / 2) + size),
            torch.arange(-int((size + 1) / 2), -int((size + 1) / 2) + size),
        )
        k_ind = torch.fft.fftshift(torch.stack(k_ind))
        alpha = 10
        amplitude = torch.pow(k_ind[0] ** 2 + k_ind[1] ** 2 + 1e-10, -alpha / 4.0)
        amplitude[0, 0] = 0
        self.register_buffer("amplitude", amplitude.unsqueeze(0), persistent=True)

    def forward(self, x):
        sigma_g = torch.rand(1, device=x.device)[0]

        weight = (
            torch.exp(-self.grid / (2 * sigma_g * sigma_g))
            * torch.randn(self.c_out, self.c_in, self.k, self.k, device=x.device)
            * self.sigma_w
        )

        sigma_delta = torch.rand(1, device=x.device)[0] * self.b_delta + self.epsilon
        delta_p = (
            torch.randn(2 * self.k * self.k, 1, 1, device=x.device)
            * sigma_delta
            * gaussian_random_field(self.amplitude.repeat(2 * self.k * self.k, 1, 1))
        ).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        gamma = torch.randn(self.c_out, device=x.device) * self.sigma_gamma
        beta = torch.randn(self.c_out, device=x.device) * self.sigma_beta
        l_num = torch.randint(low=self.l_min, high=self.l_max + 1, size=(1,), device=x.device)[0]

        custom_conv = CustomConv2d(
            in_channels=self.c_in,
            out_channels=self.c_out,
            kernel_size=3,
            sigma_gamma=self.sigma_gamma,
            sigma_beta=self.sigma_beta,
            epsilon=self.epsilon,
        ).cuda()

        for _ in range(l_num):
            x = deform_conv2d(x, delta_p, weight, padding=1)
            x = F.instance_norm(x, weight=gamma, bias=beta)
            x = torch.tanh(x)
        return x


class RandomConvCandidateSelection(nn.Module):
    """
    RCCS wrapper.

    Samples multiple ProRandConv candidates from the same input, measures
    semantic distance with `cls_net`, and returns the candidate that stays
    closest to the original semantics.

    Args:
        base_aug: ProRandConvNet instance used as the candidate generator
        n_candidates: number of candidates (default: 4)
        sem_metric: semantic distance metric, "cos" or "l2" (default: "cos")
        prefer_change: whether to prefer candidates with stronger appearance changes
        lambda_change: weight for the change-preference term (default: 0.0)
        change_metric: appearance-change metric (default: "l1")
        return_stats: whether to return selection statistics (default: False)
    """

    def __init__(self,
                 base_aug,
                 n_candidates=4,
                 sem_metric="cos",
                 prefer_change=False,
                 lambda_change=0.0,
                 change_metric="l1",
                 return_stats=False):
        super().__init__()

        self.base_aug = base_aug
        self.n_candidates = n_candidates
        self.sem_metric = sem_metric
        self.prefer_change = prefer_change
        self.lambda_change = lambda_change
        self.change_metric = change_metric
        self.return_stats = return_stats

        # Validate configuration.
        assert sem_metric in ["cos", "l2"], f"sem_metric must be 'cos' or 'l2', got {sem_metric}"
        assert change_metric in ["l1", "l2"], f"change_metric must be 'l1' or 'l2', got {change_metric}"
        assert n_candidates >= 1, f"n_candidates must be >= 1, got {n_candidates}"


    def forward(self, x, cls_net):
        """
        Forward pass.

        Args:
            x: input image tensor [B, C, H, W]
            cls_net: frozen semantic feature extractor

        Returns:
            x_best: candidate with the smallest semantic distance [B, C, H, W]
            or `(x_best, stats)` when `return_stats=True`
        """
        B, C, H, W = x.shape
        device = x.device

        # Fast path when there is only one candidate.
        if self.n_candidates == 1:
            x_aug = self.base_aug(x)
            if self.return_stats:
                return x_aug, {}
            return x_aug

        # Generate candidates with a preallocated tensor to reduce allocations.
        # Pre-allocate stacked tensor to avoid creating N separate tensors
        candidates_stack = torch.empty(
            self.n_candidates, *x.shape,
            dtype=x.dtype, device=x.device
        )  # [N, B, C, H, W]

        for i in range(self.n_candidates):
            candidates_stack[i] = self.base_aug(x)

        # Extract semantic features from the original image.
        # IMPORTANT: We do NOT change cls_net's training mode here!
        # Rationale:
        # 1. cls_net.encoder is a reference to external self.netseg
        # 2. Changing to eval mode would affect BN behavior in the main training loop
        # 3. torch.no_grad() is sufficient to prevent gradient computation
        # 4. Keeping training mode ensures consistent feature representations
        with torch.no_grad():
            z_orig = cls_net(x)  # [B, D]

            # Extract semantic features for all candidates.
            # Reshape [N, B, C, H, W] to [N*B, C, H, W].
            candidates_flat = candidates_stack.view(self.n_candidates * B, C, H, W)
            z_candidates_flat = cls_net(candidates_flat)  # [N*B, D]

            # Reshape back to [N, B, D].
            D = z_candidates_flat.shape[1]
            z_candidates = z_candidates_flat.view(self.n_candidates, B, D)

        # Compute semantic distance in a vectorized form.
        # z_orig: [B, D], z_candidates: [N, B, D]
        if self.sem_metric == "cos":
            # Cosine distance: d = 1 - cos_sim
            # Expand z_orig for broadcasting: [1, B, D]
            z_orig_expanded = z_orig.unsqueeze(0)  # [1, B, D]

            # Compute cosine similarity for all candidates at once
            cos_sim = F.cosine_similarity(z_orig_expanded, z_candidates, dim=2)  # [N, B]
            d_sem_all = 1.0 - cos_sim  # [N, B]

        elif self.sem_metric == "l2":
            # L2 distance.
            diff = z_candidates - z_orig.unsqueeze(0)  # [N, B, D]
            d_sem_all = torch.norm(diff, p=2, dim=2)  # [N, B]

        # Compute appearance-change distance when requested.
        d_change_all = None
        if self.prefer_change and self.lambda_change > 0:
            d_change_all = torch.empty(
                self.n_candidates, B,
                dtype=x.dtype, device=x.device
            )  # [N, B]
            for i in range(self.n_candidates):
                x_i = candidates_stack[i]  # [B, C, H, W]

                if self.change_metric == "l1":
                    d_change = torch.abs(x - x_i).mean(dim=(1, 2, 3))  # [B]
                elif self.change_metric == "l2":
                    diff = (x - x_i).reshape(B, -1)
                    d_change = torch.norm(diff, p=2, dim=1)  # [B]

                d_change_all[i] = d_change

        # Select the best candidate per sample instead of sharing one index per batch.
        if d_change_all is not None:
            scores_all = d_sem_all - self.lambda_change * d_change_all  # [N, B]
        else:
            scores_all = d_sem_all

        # Guard numerical stability and fall back to candidate 0 when needed.
        valid_mask = torch.isfinite(scores_all).all(dim=0)  # [B]
        if not valid_mask.all():
            import warnings
            warnings.warn(
                "RCCS: Invalid candidate scores detected for part of the batch. "
                "Falling back to candidate 0 for those samples.",
                RuntimeWarning
            )
        safe_scores_all = scores_all.clone()
        safe_scores_all[:, ~valid_mask] = float("inf")
        best_idx = torch.argmin(safe_scores_all, dim=0)  # [B]
        best_idx = torch.where(valid_mask, best_idx, torch.zeros_like(best_idx))

        sample_idx = torch.arange(B, device=device)
        x_best = candidates_stack[best_idx, sample_idx]  # [B, C, H, W]

        # Free memory: Delete the full candidates stack after extraction
        # Keep only the selected candidate
        del candidates_stack

        # Build selection statistics.
        if self.return_stats:
            d_sem_best = d_sem_all[best_idx, sample_idx]  # [B]
            stats = {
                "d_sem_best": d_sem_best.mean().item(),
                "d_sem_min": d_sem_all.min().item(),
                "d_sem_mean": d_sem_all.mean().item(),
                "d_sem_max": d_sem_all.max().item(),
                "selected_idx": best_idx.float().mean().item(),
            }

            if d_change_all is not None:
                d_change_best = d_change_all[best_idx, sample_idx]  # [B]
                stats["d_change_best"] = d_change_best.mean().item()
                stats["d_change_mean"] = d_change_all.mean().item()

            return x_best, stats

        return x_best


class RCCSFeatureEncoder(nn.Module):
    """
    Feature encoder used for RCCS semantic matching.

    Extracts encoder features from the segmentation network and maps them
    to a semantic vector using GAP + Linear.

    Args:
        encoder: full segmentation network (the complete U-Net)
        feature_dim: feature dimension before global average pooling
        output_dim: semantic embedding dimension (default: 128)
    """

    def __init__(self, encoder, feature_dim, output_dim=128):
        super().__init__()
        self.encoder = encoder
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, output_dim)

        # Freeze the encoder branch used for matching.
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extract semantic features for distance computation.

        Args:
            x: input image tensor [B, C, H, W]

        Returns:
            z: semantic embedding [B, output_dim]

        Design Note:
            This wrapper uses the FULL U-Net (encoder + decoder) but only extracts
            encoder features (x5/bottleneck). This is inefficient (wastes computation
            on decoder) but simplifies integration without modifying U-Net architecture.

            Future optimization: Add `forward_encoder()` method to U-Net to avoid
            running the decoder.
        """
        # Extract features through the encoder.
        with torch.no_grad():
            # IMPORTANT: self.encoder is actually the full U-Net, not just encoder!
            # We run full forward pass (encoder + decoder + seg head) but only use encf (x5).
            # The decoder output (y1_pred) is computed but discarded.
            #
            # Why this design?
            # - Simplifies integration: no need to modify U-Net architecture
            # - Trade-off: Wastes ~30-40% computation on decoder
            # - Future: Refactor U-Net to expose encoder-only forward pass
            _, encf = self.encoder(x, return_feat=False)

            # encf = x5 (bottleneck features from convd5)
            # Shape: [B, 256, H/16, W/16] for default U-Net with n=16
            feat = encf

        # Global Average Pooling: [B, feature_dim, H', W'] -> [B, feature_dim]
        feat = self.gap(feat)  # [B, feature_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, feature_dim]

        # Linear projection to output dimension
        z = self.fc(feat)  # [B, output_dim]

        return z


def test_rccs():
    """
    Basic smoke test for RCCS.
    """
    print("Testing RandomConvCandidateSelection...")

    # Create the base augmenter.
    base_aug = ProRandConvNet(size=224).cuda()

    # Create the RCCS wrapper.
    rccs_aug = RandomConvCandidateSelection(
        base_aug=base_aug,
        n_candidates=4,
        sem_metric="cos",
        return_stats=True
    ).cuda()

    # Create a simple mock cls_net.
    class SimpleCls(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 128)

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    cls_net = SimpleCls().cuda()

    # Run a forward pass.
    x = torch.randn(2, 3, 224, 224).cuda()

    x_aug, stats = rccs_aug(x, cls_net)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_aug.shape}")
    print(f"Stats: {stats}")

    # Validate output shape.
    assert x_aug.shape == x.shape, "Output shape mismatch!"

    # Validate selection logic.
    assert stats['d_sem_best'] == stats['d_sem_min'], "Selection logic error!"

    print("All tests passed!")


if __name__ == "__main__":
    test_rccs()
