# Main training loop.

import torch
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import os
import random
import models.segloss as segloss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .unet import *
from .sgf import get_sgf_map
from .saam import StabilityAwareAlignmentModule, compute_saam_loss
import sys
sys.path.append('..')
from dataloaders.rccs import ProRandConvNet, RandomConvCandidateSelection, RCCSFeatureEncoder


def softmax_entropy_seg(logits):
    """Pixel-wise prediction entropy for segmentation logits."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


def configure_model_for_tent(model):
    """Official TENT setup: update only BatchNorm affine parameters."""
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def collect_bn_affine_params(model):
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ("weight", "bias"):
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


class AlphaBatchNorm2d(nn.Module):
    """Blend source BatchNorm statistics with current test-batch statistics."""
    def __init__(self, layer, alpha):
        super().__init__()
        if not isinstance(layer, nn.BatchNorm2d):
            raise TypeError(f"Expected BatchNorm2d, got {type(layer)}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.layer = layer
        self.alpha = float(alpha)

    def forward(self, x):
        dims = (0, 2, 3)
        batch_mean = x.mean(dim=dims)
        batch_var = x.var(dim=dims, unbiased=False)
        running_mean = (1.0 - self.alpha) * self.layer.running_mean + self.alpha * batch_mean
        running_var = (1.0 - self.alpha) * self.layer.running_var + self.alpha * batch_var
        return F.batch_norm(
            x,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            training=False,
            momentum=0.0,
            eps=self.layer.eps,
        )


def replace_bn_with_alpha_bn(module, alpha):
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, AlphaBatchNorm2d(child, alpha))
            replaced += 1
        else:
            replaced += replace_bn_with_alpha_bn(child, alpha)
    return replaced


def forward_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@torch.no_grad()
def update_ema_model(ema_model, model, momentum):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)
    return ema_model


class Train_process():
    def __init__(self, opt,reloaddir=None,istest=None):
        super(Train_process, self).__init__()
        self.opt = opt
        self.n_cls = opt.nclass
        self.epoch=0

        if opt.model=='unet':
            # Mainline training always uses c=3 with tile_z_dim=3.
            use_cgsd = bool(opt.use_cgsd)
            use_channel_gate = use_cgsd
            cgsd_layer = getattr(opt, 'cgsd_layer', 1)  # Default to layer 1

            # CCSDG-style temperature sharpening (optional)
            use_temperature = getattr(opt, 'use_temperature', 0) == 1
            gate_tau = getattr(opt, 'gate_tau', 0.1)

            self.netseg = Unet1(c=3, num_classes=self.n_cls,
                               use_channel_gate=use_channel_gate,
                               cgsd_layer=cgsd_layer,
                               use_temperature=use_temperature,
                               gate_tau=gate_tau)
            self.netseg = self.netseg.cuda()
            total_params = sum(p.numel() for p in self.netseg.parameters())
            print(f"Number of parameters (segmentation): {total_params}")
            if use_channel_gate:
                channel_sizes = {1: 16, 2: 32, 3: 64}
                gate_type = "softmax (CCSDG-style)" if use_temperature else "sigmoid"
                print(f"CGSD enabled at encoder layer {cgsd_layer} ({channel_sizes[cgsd_layer]} channels)")
                print(f"  - Gate type: {gate_type}")
                if use_temperature:
                    print(f"  - Temperature: {gate_tau}")
            else:
                print("CGSD disabled: model follows the old no-CGSD branch")

            # Initialize the shared projector for CGSD (if enabled)
            if use_channel_gate and getattr(opt, 'use_projector', 1) == 1:
                # Auto-detect CGSD layer channels
                channel_sizes = {1: 16, 2: 32, 3: 64}
                cgsd_channels = channel_sizes[cgsd_layer]

                # Get projector configuration
                proj_dim = getattr(opt, 'proj_dim', 1024)
                proj_hidden = getattr(opt, 'proj_hidden_channels', 8)

                # Auto-detect feature map size by running dummy forward pass
                # This ensures projector matches actual feature map dimensions
                with torch.no_grad():
                    # Create dummy input with shape [1, C, H, W]
                    # Use batch size 1 to minimize memory
                    dummy_input = torch.randn(1, 3, 192, 192).cuda()  # Typical CARDIAC image size
                    # Forward through encoder to get feature map at cgsd_layer
                    x1 = self.netseg.convd1(dummy_input)
                    if cgsd_layer >= 2:
                        x2 = self.netseg.convd2(x1)
                        if cgsd_layer >= 3:
                            x3 = self.netseg.convd3(x2)
                            feature_map = x3
                        else:
                            feature_map = x2
                    else:
                        feature_map = x1

                    actual_feature_size = feature_map.shape[2]  # H dimension (should equal W)
                    print(f"Auto-detected feature map size at Layer {cgsd_layer}: {actual_feature_size}x{actual_feature_size}")

                # Paper CGSD uses one shared projector phi(.) for both
                # structure and style features.
                from .unet import Projector
                self.projector_str = Projector(
                    in_channels=cgsd_channels,
                    hidden_channels=proj_hidden,
                    proj_dim=proj_dim,
                    feature_size=actual_feature_size
                ).cuda()

                # Strict paper mode uses the shared projector_str for both
                # structure and style branches.
                self.projector_sty = None

                proj_params = sum(p.numel() for p in self.projector_str.parameters())
                print("Shared CGSD projector initialized:")
                print(f"  - Input channels: {cgsd_channels} (Layer {cgsd_layer})")
                print(f"  - Projection dim: {proj_dim}")
                print(f"  - Hidden channels: {proj_hidden}")
                print(f"  - Number of parameters: {proj_params}")
            else:
                self.projector_str = None
                self.projector_sty = None
        else:
            print("no this model")
        
        if istest == 1:
            print("reloaddir:",reloaddir)
            # Load checkpoint with strict=False to handle CGSD config mismatch.
            # This allows loading checkpoints across use_cgsd on/off transitions.
            state_dict = torch.load(reloaddir)
            missing_keys, unexpected_keys = self.netseg.load_state_dict(state_dict, strict=False)

            if len(unexpected_keys) > 0:
                print(f"⚠️  Warning: Ignoring unexpected keys in checkpoint: {unexpected_keys}")
            if len(missing_keys) > 0:
                print(f"⚠️  Warning: Missing keys in checkpoint (will use random init): {missing_keys}")
           
                
    
        x=256
        projfunc= nn.Sequential(
                nn.Conv2d(x, x, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True),
                nn.Conv2d(x, x, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True)
        )
        projfunc=projfunc.cuda()
        self.projfunc=projfunc
        
      
       
        # SAAM module (if enabled)
        if hasattr(opt, 'use_saam') and opt.use_saam:
            self.saam_module = StabilityAwareAlignmentModule(
                tau=opt.saam_tau if hasattr(opt, 'saam_tau') else 0.3,
                topk_ratio=opt.saam_topk if hasattr(opt, 'saam_topk') else 0.3,
                stability_mode=opt.saam_stability_mode if hasattr(opt, 'saam_stability_mode') else 'mean'
            ).cuda()
            print(f"SAAM initialized: tau={opt.saam_tau}, topk={opt.saam_topk}, mode={opt.saam_stability_mode}")
        else:
            self.saam_module = None

        # RCCS module (if enabled)
        if hasattr(opt, 'use_rccs') and opt.use_rccs:
            # Get image size from dataset
            # All datasets in this codebase use 192x192 resolution
            # (PROSTATE, CARDIAC, ABDOMINAL all confirmed to be 192x192)
            img_size = 192

            # Create base augmenter (ProRandConvNet)
            self.prorandconv = ProRandConvNet(size=img_size).cuda()

            # Create RCCS wrapper
            n_candidates = getattr(opt, 'rccs_candidates', 4)
            sem_metric = getattr(opt, 'rccs_metric', 'cos')
            prefer_change = getattr(opt, 'prefer_change', False)
            lambda_change = getattr(opt, 'lambda_change', 0.0)
            change_metric = getattr(opt, 'change_metric', 'l1')

            self.rccs_aug = RandomConvCandidateSelection(
                base_aug=self.prorandconv,
                n_candidates=n_candidates,
                sem_metric=sem_metric,
                prefer_change=prefer_change,
                lambda_change=lambda_change,
                change_metric=change_metric,
                return_stats=True  # Always return stats for logging
            ).cuda()

            # Create RCCSFeatureEncoder for semantic distance computation
            # Use encoder from segmentation network
            # Auto-detect encoder feature dimension from U-Net
            # IMPORTANT: encf is x5 (from convd5), not x4!
            # See unet.py:235 -> return y1_pred, x5
            encoder_dim = self.netseg.convd5.conv2.out_channels  # x5 feature dim (16*n = 256)
            cls_output_dim = getattr(opt, 'rccs_embed_dim', 128)

            self.cls_net = RCCSFeatureEncoder(
                encoder=self.netseg,  # Use full network, will extract intermediate features
                feature_dim=encoder_dim,
                output_dim=cls_output_dim
            ).cuda()

            # Freeze cls_net
            for param in self.cls_net.parameters():
                param.requires_grad = False

            rccs_params = sum(p.numel() for p in self.prorandconv.parameters())
            print("RCCS initialized:")
            print(f"  - candidates: {n_candidates}")
            print(f"  - metric: {sem_metric}")
            print(f"  - prefer_change: {prefer_change}")
            print(f"  - lambda_change: {lambda_change}")
            print(f"  - Number of parameters (ProRandConv): {rccs_params}")
        else:
            self.rccs_aug = None
            self.cls_net = None

       
        self.criterionDice = segloss.SoftDiceLoss(self.n_cls).cuda()
        self.ScoreDiceEval = segloss.Efficient_DiceScore(self.n_cls, ignore_chan0 = False).cuda()
        self.criterionCE = segloss.My_CE(nclass = self.n_cls,batch_size = self.opt.batchSize, weight = torch.ones(self.n_cls,)).cuda()

        # Optimizer setup: separate optimizers for CGSD and backbone
        # Optimizer 1: optimizer_seg for the segmentation backbone.
        # Excludes ChannelGate parameters (trained separately in optimizer_cgsd)
        params_seg = list(self.netseg.parameters())

        # If using CGSD with separate optimizer, exclude gate parameters from optimizer_seg
        use_separate_cgsd_optimizer = (
            use_channel_gate and
            getattr(opt, 'use_projector', 1) == 1 and
            getattr(opt, 'use_separate_cgsd_optimizer', 1) == 1
        )

        if use_separate_cgsd_optimizer:
            # Filter out ChannelGate parameters (chan_gate.logits)
            params_seg_filtered = []
            for name, param in self.netseg.named_parameters():
                if 'chan_gate' not in name:
                    params_seg_filtered.append(param)
            self.optimizer_seg = torch.optim.Adam(params_seg_filtered, lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.00003)
            print(f"optimizer_seg initialized: {len(params_seg_filtered)} parameter groups (excludes ChannelGate)")
        else:
            self.optimizer_seg = torch.optim.Adam(params_seg, lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.00003)
            print(f"optimizer_seg initialized: {len(params_seg)} parameter groups (includes all)")

        # Optimizer 2: optimizer_cgsd for Projector + ChannelGate
        # Only created if CGSD with projector is enabled
        if use_separate_cgsd_optimizer:
            # Collect ChannelGate parameters
            params_cgsd = []
            for name, param in self.netseg.named_parameters():
                if 'chan_gate' in name:
                    params_cgsd.append(param)

            # Add shared CGSD projector parameters exactly once.
            if self.projector_str is not None:
                params_cgsd += list(self.projector_str.parameters())

            # Use SGD with momentum for the CGSD branch.
            # Handle None explicitly: getattr returns None if attribute exists but is None
            cgsd_lr = getattr(opt, 'cgsd_lr', opt.lr)
            if cgsd_lr is None:
                cgsd_lr = opt.lr
            cgsd_momentum = getattr(opt, 'cgsd_momentum', 0.99)
            self.optimizer_cgsd = torch.optim.SGD(params_cgsd, lr=cgsd_lr, momentum=cgsd_momentum, nesterov=True)
            print(f"optimizer_cgsd initialized: {len(params_cgsd)} parameter groups (ChannelGate + Projector)")
            print(f"  - Learning rate: {cgsd_lr}")
            print(f"  - Momentum: {cgsd_momentum}")
            print(f"  - Optimizer type: SGD with Nesterov momentum")
        else:
            self.optimizer_cgsd = None

        self.tta_mode = getattr(opt, 'tta', 'none')
        self.tent_optimizer = None
        self.tent_param_names = []
        self.cotta_optimizer = None
        self.cotta_model_ema = None
        self.cotta_model_anchor = None
        self.cotta_source_state = None
        if istest == 1:
            if self.tta_mode == 'tent':
                self.configure_tent()
            elif self.tta_mode == 'norm_test':
                self.configure_alpha_norm(alpha=1.0)
            elif self.tta_mode == 'norm_alpha':
                self.configure_alpha_norm(alpha=getattr(self.opt, 'bn_alpha', 0.1))
            elif self.tta_mode == 'norm_ema':
                self.configure_norm_ema()
            elif self.tta_mode == 'cotta':
                self.configure_cotta()

    def configure_alpha_norm(self, alpha):
        replaced = replace_bn_with_alpha_bn(self.netseg, alpha)
        if replaced == 0:
            raise RuntimeError(f"{self.tta_mode} requires BatchNorm2d layers, but none were found.")
        self.netseg.eval()
        print(f"{self.tta_mode} enabled: replaced {replaced} BatchNorm2d layers with alpha={alpha}")

    def configure_norm_ema(self):
        if not any(isinstance(m, nn.BatchNorm2d) for m in self.netseg.modules()):
            raise RuntimeError("norm_ema requires BatchNorm2d layers, but none were found.")
        self.netseg.train()
        print("norm_ema enabled: updating BatchNorm running statistics online")

    def configure_tent(self):
        configure_model_for_tent(self.netseg)
        params, names = collect_bn_affine_params(self.netseg)
        if len(params) == 0:
            raise RuntimeError("TENT requires BatchNorm2d affine parameters, but none were found.")

        self.tent_param_names = names
        self.tent_optimizer = torch.optim.Adam(
            params,
            lr=getattr(self.opt, 'tent_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        print(f"TENT enabled: updating {len(params)} BatchNorm affine tensors")
        for name in names:
            print(f"  - {name}")

    def configure_cotta(self):
        self.netseg.train()
        for param in self.netseg.parameters():
            param.requires_grad_(True)

        self.cotta_optimizer = torch.optim.Adam(
            [p for p in self.netseg.parameters() if p.requires_grad],
            lr=getattr(self.opt, 'cotta_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        self.cotta_model_ema = deepcopy(self.netseg)
        self.cotta_model_anchor = deepcopy(self.netseg)
        for model in (self.cotta_model_ema, self.cotta_model_anchor):
            model.eval()
            for param in model.parameters():
                param.detach_()
                param.requires_grad_(False)

        self.cotta_source_state = deepcopy(self.netseg.state_dict())
        print(
            "CoTTA enabled: "
            f"lr={getattr(self.opt, 'cotta_lr', 1e-4)}, "
            f"steps={getattr(self.opt, 'cotta_steps', 1)}, "
            f"mt={getattr(self.opt, 'cotta_mt', 0.999)}, "
            f"rst={getattr(self.opt, 'cotta_rst', 0.01)}, "
            f"ap={getattr(self.opt, 'cotta_ap', 0.9)}"
        )

    @torch.no_grad()
    def cotta_ensemble_prediction(self, images, ema_logits):
        inp_shape = images.shape[2:]
        ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        for ratio in ratios:
            aug_shape = (
                max(1, int(inp_shape[0] * ratio)),
                max(1, int(inp_shape[1] * ratio)),
            )
            flip = [random.random() <= 0.5 for _ in range(images.shape[0])]
            aug_images = torch.cat(
                [images[i:i + 1].flip(dims=(3,)) if fp else images[i:i + 1]
                 for i, fp in enumerate(flip)],
                dim=0,
            )
            aug_images = F.interpolate(aug_images, size=aug_shape, mode='bilinear', align_corners=False)
            aug_logits = forward_logits(self.cotta_model_ema, aug_images)
            aug_logits = torch.cat(
                [aug_logits[i:i + 1].flip(dims=(3,)) if fp else aug_logits[i:i + 1]
                 for i, fp in enumerate(flip)],
                dim=0,
            )
            ema_logits = ema_logits + F.interpolate(aug_logits, size=inp_shape, mode='bilinear', align_corners=False)
        return ema_logits / float(len(ratios) + 1)

    @torch.no_grad()
    def cotta_stochastic_restore(self):
        rst = getattr(self.opt, 'cotta_rst', 0.01)
        if rst <= 0.0:
            return
        for name, param in self.netseg.named_parameters():
            if not param.requires_grad or name not in self.cotta_source_state:
                continue
            if not (name.endswith('weight') or name.endswith('bias')):
                continue
            mask = torch.rand_like(param, dtype=torch.float32) < rst
            source_param = self.cotta_source_state[name].to(param.device)
            param.data.copy_(torch.where(mask, source_param, param.data))

    def te_func_norm(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.tta_mode == 'norm_ema':
            self.netseg.train()
            with torch.no_grad():
                _ = forward_logits(self.netseg, self.input_img_te)
            self.netseg.eval()
        else:
            self.netseg.eval()

        with torch.no_grad():
            logits = forward_logits(self.netseg, self.input_img_te)
            seg = torch.argmax(logits, 1)

        self.netseg.zero_grad()
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_tent(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.tent_optimizer is None:
            raise RuntimeError("TENT optimizer is not initialized. Call configure_tent() first.")

        logits = None
        loss = None
        self.netseg.train()
        for _ in range(getattr(self.opt, 'tent_steps', 1)):
            logits = forward_logits(self.netseg, self.input_img_te)
            loss = softmax_entropy_seg(logits)
            self.tent_optimizer.zero_grad()
            loss.backward()
            self.tent_optimizer.step()

        seg = torch.argmax(logits.detach(), 1)
        self.netseg.zero_grad()
        self.last_tent_loss = loss.detach() if loss is not None else None
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_cotta(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.cotta_optimizer is None:
            raise RuntimeError("CoTTA optimizer is not initialized. Call configure_cotta() first.")

        self.netseg.train()
        outputs_ema = None
        for _ in range(getattr(self.opt, 'cotta_steps', 1)):
            outputs = forward_logits(self.netseg, self.input_img_te)

            with torch.no_grad():
                anchor_logits = forward_logits(self.cotta_model_anchor, self.input_img_te)
                anchor_prob = torch.softmax(anchor_logits, dim=1).max(dim=1)[0]
                outputs_ema = forward_logits(self.cotta_model_ema, self.input_img_te)
                if anchor_prob.mean() < getattr(self.opt, 'cotta_ap', 0.9):
                    outputs_ema = self.cotta_ensemble_prediction(self.input_img_te, outputs_ema)

            loss = (-(outputs_ema.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()
            self.cotta_optimizer.zero_grad()
            loss.backward()
            self.cotta_optimizer.step()

            update_ema_model(self.cotta_model_ema, self.netseg, getattr(self.opt, 'cotta_mt', 0.999))
            self.cotta_stochastic_restore()

        with torch.no_grad():
            final_logits = forward_logits(self.cotta_model_ema, self.input_img_te)
            seg = torch.argmax(final_logits, 1)

        self.netseg.zero_grad()
        self.last_cotta_loss = loss.detach()
        return self.input_mask_te, seg

    def te_func(self,input):
        tta_mode = getattr(self.opt, 'tta', 'none')
        if tta_mode == 'tent':
            return self.te_func_tent(input)
        if tta_mode in ['norm_test', 'norm_alpha', 'norm_ema']:
            return self.te_func_norm(input)
        if tta_mode == 'cotta':
            return self.te_func_cotta(input)

        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        self.netseg.eval()

        with torch.no_grad():
            img,mask= self.input_img_te,self.input_mask_te
            seg = forward_logits(self.netseg, img)
            seg = torch.argmax(seg, 1)

        self.netseg.zero_grad()
        self.netseg.train()

        return mask, seg
    
    def forward_seg_train(self, input_img, return_feat=False):
        """
        Forward pass for segmentation training.

        Args:
            input_img: input image [B, C, H, W]
            return_feat: if True, return shallow layer features for L_str computation (CGSD)

        Returns:
            if return_feat=False: (pred, encf, loss_dice, loss_ce)
            if return_feat=True: (pred, encf, loss_dice, loss_ce, f1_str, f1_sty)
        """
        # Request structure/style features only when CGSD is enabled.
        gate_on = bool(self.opt.use_cgsd)
        if return_feat and gate_on:
            pred, encf, f1_str, f1_sty = self.netseg(input_img, return_feat=True)
        else:
            pred, encf = self.netseg(input_img, return_feat=False)
            f1_str, f1_sty = None, None

        loss_dice = self.criterionDice(input=pred, target=self.input_mask)
        loss_ce = self.criterionCE(inputs=pred, targets=self.input_mask.long())

        self.seg_tr = pred.detach()

        if return_feat:
            return pred, encf, loss_dice, loss_ce, f1_str, f1_sty
        else:
            return pred, encf, loss_dice, loss_ce

    def get_saam_lambdas(self, epoch):
        """
        Warmup + rampup schedule for SAAM:
        - 0 ~ E_warm: lambda = 0 (warmup, no alignment)
        - E_warm ~ E_warm + E_ramp: linear rampup from 0 to target
        - > E_warm + E_ramp: lambda = target (full alignment with stability gating)

        Returns:
            (lambda_01_cur, lambda_02_cur): current weights for Anchor↔Base and Anchor↔Strong
        """
        if not hasattr(self.opt, 'use_saam') or not self.opt.use_saam:
            return 0.0, 0.0

        base_01 = self.opt.lambda_01 if hasattr(self.opt, 'lambda_01') else 1.0
        base_02 = self.opt.lambda_02 if hasattr(self.opt, 'lambda_02') else 1.0
        E_warm = self.opt.saam_warmup_epochs if hasattr(self.opt, 'saam_warmup_epochs') else 50
        E_ramp = self.opt.saam_rampup_epochs if hasattr(self.opt, 'saam_rampup_epochs') else 100

        if epoch < E_warm:
            # Warmup: no SAAM alignment
            return 0.0, 0.0
        elif epoch < E_warm + E_ramp:
            # Rampup: linearly increase from 0 to base
            progress = (epoch - E_warm) / E_ramp
            return base_01 * progress, base_02 * progress
        else:
            # Full strength
            return base_01, base_02
    

    def compute_structure_consistency_loss(self, f1_str, f2_str):
        """
        Paper-style structure consistency term:
        d(phi(f_str^1), phi(f_str^2)), where d is cosine distance.

        Args:
            f1_str: structure features from view1 [B, C, H, W]
            f2_str: structure features from view2 [B, C, H, W]
        """
        use_projector = (
            hasattr(self, 'projector_str') and
            self.projector_str is not None and
            getattr(self.opt, 'use_projector', 1) == 1
        )
        if use_projector:
            z1 = self.projector_str(f1_str)
            z2 = self.projector_str(f2_str)
        else:
            z1 = F.adaptive_avg_pool2d(f1_str, 1).view(f1_str.size(0), -1)
            z2 = F.adaptive_avg_pool2d(f2_str, 1).view(f2_str.size(0), -1)

        cos_sim = F.cosine_similarity(z1, z2, dim=1)
        return (1 - cos_sim).mean()

    def compute_style_divergence_loss(self, f1_sty, f2_sty):
        """
        Paper-style style diversity term:
        -d(phi(f_sty^1), phi(f_sty^2)), where d is cosine distance.
        """
        use_projector = (
            hasattr(self, 'projector_str') and
            self.projector_str is not None and
            getattr(self.opt, 'use_projector', 1) == 1
        )

        if use_projector:
            z1 = self.projector_str(f1_sty)
            z2 = self.projector_str(f2_sty)
        else:
            z1 = F.adaptive_avg_pool2d(f1_sty, 1).view(f1_sty.size(0), -1)
            z2 = F.adaptive_avg_pool2d(f2_sty, 1).view(f2_sty.size(0), -1)

        cos_sim = F.cosine_similarity(z1, z2, dim=1)
        return -(1 - cos_sim).mean()

    def get_lambda_str_with_warmup(self, epoch):
        """
        Strict paper setting: CGSD uses a fixed weight from epoch 1.
        """
        return self.opt.lambda_str

    def forward_saam(self, encf_0, encf_1, encf_2, pred_0, pred_1, pred_2, epoch):
        """
        Selective alignment step for SAAM.

        Args:
            encf_0: Orig encoder features [B, C, h, w]
            encf_1: View-1/base encoder features [B, C, h, w]
            encf_2: View-2/strong encoder features [B, C, h, w]
            pred_0: Predictions from anchor view [B, nclass, H, W]
            pred_1: Predictions from base view [B, nclass, H, W]
            pred_2: Predictions from strong view [B, nclass, H, W]
            epoch: current epoch (for warmup/rampup schedule)

        Returns:
            Sets self.loss_saam, self.loss_saam_01, self.loss_saam_02
            Also sets self.saam_stats for logging
        """
        if not hasattr(self.opt, 'use_saam') or not self.opt.use_saam or self.saam_module is None:
            self.loss_saam = torch.zeros(1).cuda()
            self.loss_saam_01 = torch.zeros(1).cuda()
            self.loss_saam_02 = torch.zeros(1).cuda()
            self.saam_stats = {}
            return

        # Get current lambda weights (warmup/rampup)
        lambda_01_cur, lambda_02_cur = self.get_saam_lambdas(epoch)

        # If in warmup, skip computation (lambda=0)
        if lambda_01_cur == 0.0 and lambda_02_cur == 0.0:
            self.loss_saam = torch.zeros(1).cuda()
            self.loss_saam_01 = torch.zeros(1).cuda()
            self.loss_saam_02 = torch.zeros(1).cuda()
            self.saam_stats = {
                'lambda_01': 0.0,
                'lambda_02': 0.0,
                'status': 'warmup'
            }
            return

        # Get mask and resolution
        gt = self.input_mask
        H, W = gt.shape[2], gt.shape[3]

        # Project encoder features for alignment
        # Use projfunc to get alignment features (like OFC)
        encf_all = torch.cat([encf_0, encf_1, encf_2], dim=0)  # [3B, C, h, w]
        projf_all = self.projfunc(encf_all)  # [3B, C', h, w]

        # Split back to three views
        B = encf_0.shape[0]
        q_0 = projf_all[:B]      # [B, C', h, w]
        q_1 = projf_all[B:2*B]   # [B, C', h, w]
        q_2 = projf_all[2*B:]    # [B, C', h, w]

        # Upsample q features to mask resolution
        q_0_up = F.interpolate(q_0, size=[H, W], mode="bilinear", align_corners=False)
        q_1_up = F.interpolate(q_1, size=[H, W], mode="bilinear", align_corners=False)
        q_2_up = F.interpolate(q_2, size=[H, W], mode="bilinear", align_corners=False)

        # Compute stability gate weights
        W_up, stats = self.saam_module(encf_0, encf_1, encf_2, mask_size=(H, W))

        # Create binary object mask: (GT foreground) OR (Pred foreground)
        # This matches the original pairwise alignment masking logic.
        gt_2d = gt.squeeze(1) if gt.dim() == 4 else gt  # [B, H, W]

        # Get predicted labels from all three views
        pred_0_label = torch.argmax(pred_0.detach(), dim=1)  # [B, H, W]
        pred_1_label = torch.argmax(pred_1.detach(), dim=1)  # [B, H, W]
        pred_2_label = torch.argmax(pred_2.detach(), dim=1)  # [B, H, W]

        # Create binary mask: foreground = (GT != 0) OR (any pred != 0)
        # Union of GT and all three predictions for maximum coverage
        mask = torch.zeros_like(gt_2d, dtype=torch.float32)
        mask[(gt_2d != 0) | (pred_0_label != 0) | (pred_1_label != 0) | (pred_2_label != 0)] = 1.0
        # Coverage ratio of foreground pixels (for logging/debug)
        mask_coverage = (mask.sum() / mask.numel()).item()

        L_align, L_01, L_02, effective_pixels = compute_saam_loss(
            q_0_up, q_1_up, q_2_up, mask, W_up,
            lambda_01=lambda_01_cur, lambda_02=lambda_02_cur
        )

        # Store losses
        self.loss_saam = L_align
        self.loss_saam_01 = L_01
        self.loss_saam_02 = L_02

        # Store statistics for logging
        self.saam_stats = stats
        self.saam_stats['lambda_01'] = lambda_01_cur
        self.saam_stats['lambda_02'] = lambda_02_cur
        self.saam_stats['L_01'] = L_01.item()
        self.saam_stats['L_02'] = L_02.item()
        self.saam_stats['L_saam'] = L_align.item()
        self.saam_stats['mask_coverage'] = mask_coverage  # Track mask coverage
        self.saam_stats['effective_pixels'] = effective_pixels  # Sum of M*W (useful to debug gate too strict)


    # Read anchor/base/strong views from the batch and run one mainline training step.
    def tr_func(self, train_batch, epoch):
        self.epoch = epoch

        for param in self.netseg.parameters():
            param.requires_grad = True

        w_seg = self.opt.w_seg
        w_ce = self.opt.w_ce
        w_dice = self.opt.w_dice
        use_saam = bool(getattr(self.opt, 'use_saam', 0))
        gate_on = bool(self.opt.use_cgsd)
        return_feat = gate_on
        sgf_view2_only = bool(getattr(self.opt, 'sgf_view2_only', 0)) and bool(getattr(self.opt, 'use_sgf', 0))

        self.input_mask = train_batch['label'].float().cuda()
        self.input_anchor = train_batch['anchor_view'].float().cuda()
        zero = torch.zeros(1, device=self.input_mask.device)

        self.loss_str = zero.clone()
        self.loss_sty = zero.clone()
        self.loss_cgsd = zero.clone()
        self.loss_saam = zero.clone()
        self.loss_saam_01 = zero.clone()
        self.loss_saam_02 = zero.clone()
        self.saam_stats = {}
        self.rccs_applied = False
        self.rccs_stats = {}
        self.rccs_applied_base = False
        loss0 = zero.clone()
        loss1 = zero.clone()
        loss2 = zero.clone()
        loss_dice0 = zero.clone()
        loss_ce0 = zero.clone()
        loss_dice1 = zero.clone()
        loss_ce1 = zero.clone()
        loss_dice2 = zero.clone()
        loss_ce2 = zero.clone()

        pred_all0 = None
        pred_all1 = None
        pred_all2 = None
        encf0 = None
        encf1 = None
        encf2 = None
        f1_str_v1 = None
        f1_sty_v1 = None
        f1_str_v2 = None
        f1_sty_v2 = None

        self.input_base = train_batch['base_view'].float().cuda()

        if hasattr(self.opt, 'use_sgf') and self.opt.use_sgf:
            self.input_strong_seed = train_batch['strong_view'].float().cuda()
            self.input_base_for_loss = None if sgf_view2_only else self.input_base

            rccs_apply_to_base = bool(getattr(self.opt, 'rccs_apply_to_base', 0))
            rccs_apply_to_saam = bool(getattr(self.opt, 'rccs_apply_to_saam', 0))
            p_rccs = getattr(self.opt, 'p_rccs', 0.3)
            rccs_applied_strong = False

            if sgf_view2_only:
                rccs_apply_to_base = False
                rccs_apply_to_saam = False

            self.input_base_sup = None
            if not sgf_view2_only:
                self.input_base_sup = self.input_base
                if self.rccs_aug is not None and rccs_apply_to_base and np.random.rand() < p_rccs:
                    self.input_base_sup, self.rccs_stats = self.rccs_aug(self.input_base, self.cls_net)
                    self.rccs_applied = True
                    self.rccs_applied_base = True

                input_var1 = Variable(self.input_base_sup.detach().clone(), requires_grad=True)

                if self.rccs_applied_base and not rccs_apply_to_saam:
                    if return_feat:
                        pred_all1, encf1, _, _, f1_str_v1, f1_sty_v1 = self.forward_seg_train(
                            self.input_base, return_feat=True)
                    else:
                        pred_all1, encf1, _, _ = self.forward_seg_train(
                            self.input_base, return_feat=False)

                    _, _, loss_dice1, loss_ce1 = self.forward_seg_train(
                        input_var1, return_feat=False)
                else:
                    if return_feat:
                        pred_all1, encf1, loss_dice1, loss_ce1, f1_str_v1, f1_sty_v1 = self.forward_seg_train(
                            input_var1, return_feat=True)
                    else:
                        pred_all1, encf1, loss_dice1, loss_ce1 = self.forward_seg_train(
                            input_var1, return_feat=False)

                loss1 = loss_dice1 * w_dice + loss_ce1 * w_ce
                grad = torch.autograd.grad(loss1, input_var1, retain_graph=True, create_graph=False)[0]
                gradient = torch.sqrt(torch.mean(grad ** 2, dim=1, keepdim=True)).detach()
            else:
                base_var = self.input_base.detach().clone().requires_grad_(True)
                _, _, loss_dice_gip, loss_ce_gip = self.forward_seg_train(
                    base_var, return_feat=False)
                loss_gip = loss_dice_gip * w_dice + loss_ce_gip * w_ce
                grad = torch.autograd.grad(loss_gip, base_var, retain_graph=False, create_graph=False)[0]
                gradient = torch.sqrt(torch.mean(grad ** 2, dim=1, keepdim=True)).detach()

            saliency = get_sgf_map(gradient, self.opt.sgf_grid_size)
            # SGF corresponds to the paper's saliency-guided fusion of GIP and CLP.
            self.input_strong = self.input_base * saliency + self.input_strong_seed * (1 - saliency)
            self.input_strong_sup = self.input_strong

            if self.rccs_aug is not None and not rccs_apply_to_base and np.random.rand() < p_rccs:
                self.input_strong_sup, self.rccs_stats = self.rccs_aug(self.input_strong, self.cls_net)
                self.rccs_applied = True
                rccs_applied_strong = True

            if rccs_applied_strong and not rccs_apply_to_saam:
                if return_feat:
                    pred_all2, encf2, _, _, f1_str_v2, f1_sty_v2 = self.forward_seg_train(
                        self.input_strong, return_feat=True)
                else:
                    pred_all2, encf2, _, _ = self.forward_seg_train(
                        self.input_strong, return_feat=False)

                _, _, loss_dice2, loss_ce2 = self.forward_seg_train(
                    self.input_strong_sup, return_feat=False)
            else:
                if return_feat:
                    pred_all2, encf2, loss_dice2, loss_ce2, f1_str_v2, f1_sty_v2 = self.forward_seg_train(
                        self.input_strong_sup, return_feat=True)
                else:
                    pred_all2, encf2, loss_dice2, loss_ce2 = self.forward_seg_train(
                        self.input_strong_sup, return_feat=False)

            loss2 = loss_dice2 * w_dice + loss_ce2 * w_ce

            if use_saam and not sgf_view2_only:
                pred_all0, encf0, loss_dice0, loss_ce0 = self.forward_seg_train(
                    self.input_anchor, return_feat=False)
                loss0 = loss_dice0 * w_dice + loss_ce0 * w_ce
                self.forward_saam(encf0, encf1, encf2, pred_all0, pred_all1, pred_all2, epoch)
        else:
            self.input_base_for_loss = self.input_base
            self.input_strong = train_batch['strong_view'].float().cuda()

            if return_feat:
                pred_all1, encf1, loss_dice1, loss_ce1, f1_str_v1, f1_sty_v1 = self.forward_seg_train(
                    self.input_base, return_feat=True)
                pred_all2, encf2, loss_dice2, loss_ce2, f1_str_v2, f1_sty_v2 = self.forward_seg_train(
                    self.input_strong, return_feat=True)
            else:
                pred_all1, encf1, loss_dice1, loss_ce1 = self.forward_seg_train(
                    self.input_base, return_feat=False)
                pred_all2, encf2, loss_dice2, loss_ce2 = self.forward_seg_train(
                    self.input_strong, return_feat=False)

            loss1 = loss_dice1 * w_dice + loss_ce1 * w_ce
            loss2 = loss_dice2 * w_dice + loss_ce2 * w_ce

            if use_saam:
                pred_all0, encf0, loss_dice0, loss_ce0 = self.forward_seg_train(
                    self.input_anchor, return_feat=False)
                loss0 = loss_dice0 * w_dice + loss_ce0 * w_ce
                self.forward_saam(encf0, encf1, encf2, pred_all0, pred_all1, pred_all2, epoch)

        if gate_on and f1_str_v1 is not None and f1_str_v2 is not None and self.opt.lambda_str > 0:
            lambda_str_cur = self.get_lambda_str_with_warmup(epoch)
            if lambda_str_cur > 0:
                self.loss_str = lambda_str_cur * self.compute_structure_consistency_loss(
                    f1_str_v1, f1_str_v2)
                lambda_sty = getattr(self.opt, 'lambda_sty', 0.3)
                if lambda_sty > 0 and f1_sty_v1 is not None and f1_sty_v2 is not None:
                    self.loss_sty = lambda_sty * self.compute_style_divergence_loss(f1_sty_v1, f1_sty_v2)

                if self.optimizer_cgsd is not None:
                    self.optimizer_cgsd.zero_grad()
                    loss_cgsd = self.loss_str + self.loss_sty
                    loss_cgsd.backward(retain_graph=True)
                    self.optimizer_cgsd.step()
                    self.loss_cgsd = loss_cgsd.detach()

        if sgf_view2_only:
            self.loss_seg = loss2 * w_seg
            self.loss_dice = loss_dice2 * w_dice
            self.loss_ce = loss_ce2 * w_ce
            self.loss_seg1 = zero.clone()
            self.loss_seg2 = loss2
            self.loss_dice1 = zero.clone()
            self.loss_dice2 = loss_dice2
            self.loss_ce1 = zero.clone()
            self.loss_ce2 = loss_ce2
        elif use_saam:
            alpha_0 = getattr(self.opt, 'anchor_seg_alpha', 0.0)
            alpha_s = getattr(self.opt, 'strong_seg_alpha', 0.0)
            denom = 1.0 + alpha_0 + alpha_s

            self.loss_seg = (loss1 + alpha_0 * loss0 + alpha_s * loss2) / denom * w_seg
            self.loss_dice = (loss_dice1 + alpha_0 * loss_dice0 + alpha_s * loss_dice2) * w_dice / denom
            self.loss_ce = (loss_ce1 + alpha_0 * loss_ce0 + alpha_s * loss_ce2) * w_ce / denom
            self.loss_seg1 = loss1
            self.loss_seg0 = alpha_0 * loss0
            self.loss_seg2 = alpha_s * loss2
            self.loss_dice0 = loss_dice0
            self.loss_dice1 = loss_dice1
            self.loss_dice2 = loss_dice2
            self.loss_ce0 = loss_ce0
            self.loss_ce1 = loss_ce1
            self.loss_ce2 = loss_ce2
        else:
            alpha = getattr(self.opt, 'seg_alpha_view2', 1.0)
            denom = 1.0 + alpha

            self.loss_seg = (loss1 + alpha * loss2) / denom * w_seg
            self.loss_dice = (loss_dice1 + alpha * loss_dice2) * w_dice / denom
            self.loss_ce = (loss_ce1 + alpha * loss_ce2) * w_ce / denom
            self.loss_seg1 = loss1
            self.loss_seg2 = alpha * loss2
            self.loss_dice1 = loss_dice1
            self.loss_dice2 = loss_dice2
            self.loss_ce1 = loss_ce1
            self.loss_ce2 = loss_ce2

        if self.optimizer_cgsd is not None:
            self.loss_all = self.loss_seg + self.loss_saam
        else:
            self.loss_all = self.loss_seg + self.loss_saam + self.loss_str + self.loss_sty

        self.optimizer_seg.zero_grad()
        self.loss_all.backward()
        self.optimizer_seg.step()

        tr_log = [('dice', self.loss_dice), ('ce', self.loss_ce), ('seg', self.loss_seg),
                  ('lr', self.get_lr()), ('loss', self.loss_all)]

        if gate_on:
            tr_log.append(('str', self.loss_str))
            tr_log.append(('sty', self.loss_sty))
            tr_log.append(('cgsd', self.loss_cgsd))

        if use_saam:
            tr_log.append(('saam', self.loss_saam))
            tr_log.append(('saam_01', self.loss_saam_01))
            tr_log.append(('saam_02', self.loss_saam_02))
            if self.saam_stats:
                for key, val in self.saam_stats.items():
                    if isinstance(val, (int, float)):
                        tr_log.append((f'saam_{key}', torch.tensor(val, device=self.input_mask.device)))

        if self.rccs_aug is not None:
            tr_log.append(('rccs_applied', torch.tensor(1.0 if self.rccs_applied else 0.0, device=self.input_mask.device)))
            if self.rccs_applied and self.rccs_stats:
                for key, val in self.rccs_stats.items():
                    if isinstance(val, (int, float)):
                        tr_log.append((f'rccs_{key}', torch.tensor(val, device=self.input_mask.device)))
                tr_log.append(('rccs_on_view1', torch.tensor(1.0 if self.rccs_applied_base else 0.0, device=self.input_mask.device)))

        for param in self.netseg.parameters():
            param.requires_grad = False

        return OrderedDict(tr_log)
                

    def get_img_tr(self):
        img_base = t2n(self.input_base.detach())
        pred_tr = t2n(torch.argmax(self.seg_tr, dim =1, keepdim = True))
        gth_tr  = t2n(self.input_mask.detach())
        img_anchor = t2n(self.input_anchor.detach())
      
        ret_visuals = OrderedDict([
            ('img_anchor', img_anchor),
            ('img_base', img_base),
            ('seg_tr', pred_tr),
            ('gth_tr', gth_tr),
        ])
        if hasattr(self, 'input_strong') and self.input_strong is not None:
            ret_visuals['img_strong'] = t2n(self.input_strong.detach())

        return ret_visuals


    def save(self, snapshot_dir,label):
        save_filename = '%s_net_%s.pth' % (label, 'Seg')
        save_path = os.path.join(snapshot_dir, save_filename)
        print("save_path:",save_path)
        torch.save(self.netseg.state_dict(), save_path)


    def get_lr(self):
        lr = self.optimizer_seg.param_groups[0]['lr']
        x=[lr]
        x=torch.Tensor(x)
        return x

def t2n(x):
    if isinstance(x, np.ndarray):
        return x
    if x.is_cuda:
        x = x.data.cpu()
    else:
        x = x.data
    return np.float32(x.numpy())
