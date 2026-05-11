# Main training loop.

import torch
from collections import OrderedDict
from copy import deepcopy
import logging
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
from .tta_asm import ASMAdapter
from .tta_gtta import GTTAAdapter
from .tta_smppm import SMPPMAdapter
from .tta_saam_spmm import SAAMSPMMAdapter, configure_model_for_saam_spmm
from tta_dg_tta import DGTTAAdapter
from tta_memo import build_memo_batch, marginal_entropy
from tta_gold import GOLDAdapter
from tta_vptta import (
    FrequencyPrompt,
    PromptMemory,
    VPTTAAdapter,
    convert_batchnorm_to_vptta,
)
from tta_pass import (
    BatchNormFeatureMonitor,
    PASSAdapter,
    PASSPromptedUNet,
    configure_pass_model,
)
from tta_sictta import SicTTAAdapter, configure_model_for_sictta
from tta_samtta import SAMTTAAdapter, SAMTTABezierTransform, configure_model_for_samtta
from tta_a3_tta import A3TTAAdapter, configure_model_for_a3_tta
from tta_spmo import SPMOAdapter
import sys
sys.path.append('..')
from dataloaders.rccs import ProRandConvNet, RandomConvCandidateSelection, RCCSFeatureEncoder


logger = logging.getLogger(__name__)


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


def parameter_count(params):
    return sum(param.numel() for param in params)


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
    def __init__(self, opt,reloaddir=None,istest=None, source_loader=None):
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
        self.tent_source_loader = source_loader
        self.tent_source_iter = iter(source_loader) if source_loader is not None else None
        self.sar_optimizer = None
        self.sar_param_names = []
        self.sar_params = []
        self.sar_source_loader = source_loader
        self.sar_source_iter = iter(source_loader) if source_loader is not None else None
        self.dgtta_optimizer = None
        self.dgtta_adapter = None
        self.cotta_optimizer = None
        self.cotta_model_ema = None
        self.cotta_model_anchor = None
        self.cotta_source_state = None
        self.memo_optimizer = None
        self.asm_optimizer = None
        self.asm_adapter = None
        self.asm_source_loader = source_loader
        self.smppm_optimizer = None
        self.smppm_adapter = None
        self.smppm_source_loader = source_loader
        self.saam_spmm_optimizer = None
        self.saam_spmm_adapter = None
        self.gtta_optimizer = None
        self.gtta_adapter = None
        self.gtta_source_loader = source_loader
        self.gold_optimizer = None
        self.gold_adapter = None
        self.vptta_optimizer = None
        self.vptta_prompt = None
        self.vptta_memory = None
        self.vptta_adapter = None
        self.pass_optimizer = None
        self.pass_adapter = None
        self.pass_bn_monitor = None
        self.pass_source_model = None
        self.samtta_optimizer = None
        self.samtta_transform = None
        self.samtta_adapter = None
        self.spmo_optimizer = None
        self.spmo_adapter = None
        self.spmo_source_model = None
        self.sictta_adapter = None
        self.a3_optimizer = None
        self.a3_adapter = None
        if istest == 1:
            if self.tta_mode == 'tent':
                self.configure_tent()
            elif self.tta_mode == 'sar':
                self.configure_sar()
            elif self.tta_mode == 'dg_tta':
                self.configure_dg_tta()
            elif self.tta_mode == 'norm_test':
                self.configure_alpha_norm(alpha=1.0)
            elif self.tta_mode == 'norm_alpha':
                self.configure_alpha_norm(alpha=getattr(self.opt, 'bn_alpha', 0.1))
            elif self.tta_mode == 'norm_ema':
                self.configure_norm_ema()
            elif self.tta_mode == 'cotta':
                self.configure_cotta()
            elif self.tta_mode == 'memo':
                self.configure_memo()
            elif self.tta_mode == 'asm':
                if source_loader is None:
                    print("ASM requested: call configure_asm(source_loader) before te_func_asm.")
                else:
                    self.configure_asm(source_loader)
            elif self.tta_mode == 'sm_ppm':
                smppm_mode = getattr(self.opt, 'smppm_ablation_mode', 'full')
                if source_loader is None and smppm_mode != 'source_free_proto':
                    print("SM-PPM requested: call configure_smppm(source_loader) before te_func_smppm.")
                else:
                    self.configure_smppm(source_loader)
            elif self.tta_mode == 'source_ce_only':
                self.opt.smppm_ablation_mode = 'source_ce_only'
                if source_loader is None:
                    print("source_ce_only requested: call configure_smppm(source_loader) before te_func_smppm.")
                else:
                    self.configure_smppm(source_loader)
            elif self.tta_mode == 'saam_spmm':
                self.configure_saam_spmm()
            elif self.tta_mode == 'gtta':
                if source_loader is None:
                    print("GTTA requested: call configure_gtta(source_loader) before te_func_gtta.")
                else:
                    self.configure_gtta(source_loader)
            elif self.tta_mode == 'gold':
                self.configure_gold()
            elif self.tta_mode == 'vptta':
                self.configure_vptta()
            elif self.tta_mode == 'pass':
                self.configure_pass()
            elif self.tta_mode == 'samtta':
                self.configure_samtta()
            elif self.tta_mode == 'spmo':
                self.configure_spmo()
            elif self.tta_mode == 'sictta':
                self.configure_sictta()
            elif self.tta_mode == 'a3_tta':
                self.configure_a3_tta()

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
        msg = (
            f"TENT enabled: updating {len(params)} BatchNorm affine tensors "
            f"({parameter_count(params)} parameters), "
            f"source_access={getattr(self.opt, 'source_access', False)}, "
            f"lambda_source={getattr(self.opt, 'lambda_source', 0.0)}"
        )
        print(msg)
        logger.info(msg)
        for name in names:
            print(f"  - {name}")

    def configure_sar(self):
        configure_model_for_tent(self.netseg)
        params, names = collect_bn_affine_params(self.netseg)
        if len(params) == 0:
            raise RuntimeError("SAR requires BatchNorm2d affine parameters, but none were found.")

        self.sar_params = params
        self.sar_param_names = names
        self.sar_optimizer = torch.optim.Adam(
            params,
            lr=getattr(self.opt, 'sar_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        msg = (
            f"SAR enabled: lightweight BN-affine sharpness-aware entropy minimization; "
            f"updating {len(params)} BatchNorm affine tensors "
            f"({parameter_count(params)} parameters), "
            f"lr={getattr(self.opt, 'sar_lr', 1e-4)}, "
            f"steps={getattr(self.opt, 'sar_steps', 1)}, "
            f"rho={getattr(self.opt, 'sar_rho', 0.05)}, "
            f"source_access={getattr(self.opt, 'source_access', False)}, "
            f"lambda_source={getattr(self.opt, 'lambda_source', 0.0)}"
        )
        print(msg)
        logger.info(msg)
        for name in names:
            print(f"  - {name}")

    def configure_dg_tta(self):
        configure_model_for_tent(self.netseg)
        params, names = collect_bn_affine_params(self.netseg)
        if len(params) == 0:
            raise RuntimeError("DG-TTA requires BatchNorm2d affine parameters, but none were found.")

        self.dgtta_optimizer = torch.optim.Adam(
            params,
            lr=getattr(self.opt, 'dgtta_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        self.dgtta_adapter = DGTTAAdapter(
            model=self.netseg,
            optimizer=self.dgtta_optimizer,
            steps=getattr(self.opt, 'dgtta_steps', 1),
            transform_strength=getattr(self.opt, 'dgtta_transform_strength', 1.0),
            entropy_weight=getattr(self.opt, 'dgtta_entropy_weight', 0.05),
            bn_l2_reg=getattr(self.opt, 'dgtta_bn_l2_reg', 1e-4),
            episodic=getattr(self.opt, 'dgtta_episodic', False),
        )
        print(
            "DG-TTA enabled: MedSeg-TTA output-consistency adapter updating "
            f"{len(params)} BatchNorm affine tensors; "
            f"lr={getattr(self.opt, 'dgtta_lr', 1e-4)}, "
            f"steps={getattr(self.opt, 'dgtta_steps', 1)}, "
            f"strength={getattr(self.opt, 'dgtta_transform_strength', 1.0)}, "
            f"entropy_weight={getattr(self.opt, 'dgtta_entropy_weight', 0.05)}, "
            f"bn_l2_reg={getattr(self.opt, 'dgtta_bn_l2_reg', 1e-4)}, "
            f"episodic={getattr(self.opt, 'dgtta_episodic', False)}"
        )
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
            f"ap={getattr(self.opt, 'cotta_ap', 0.9)}, "
            f"source_access={getattr(self.opt, 'source_access', False)}, "
            f"lambda_source={getattr(self.opt, 'lambda_source', 0.0)}"
        )

    def configure_memo(self):
        update_scope = getattr(self.opt, 'memo_update_scope', 'all')
        self.netseg.train()
        self.netseg.requires_grad_(False)

        if update_scope == 'bn_affine':
            params, names = collect_bn_affine_params(self.netseg)
            if len(params) == 0:
                raise RuntimeError("MEMO bn_affine mode requires BatchNorm2d affine parameters, but none were found.")
            for param in params:
                param.requires_grad_(True)
            print(f"MEMO enabled: updating {len(params)} BatchNorm affine tensors")
            for name in names:
                print(f"  - {name}")
        elif update_scope == 'all':
            params = list(self.netseg.parameters())
            for param in params:
                param.requires_grad_(True)
            print(f"MEMO enabled: updating all segmentation parameters ({len(params)} tensors)")
        else:
            raise ValueError(f"Unknown memo_update_scope: {update_scope}")

        self.memo_optimizer = torch.optim.Adam(
            params,
            lr=getattr(self.opt, 'memo_lr', 1e-5),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        print(
            "MEMO config: "
            f"lr={getattr(self.opt, 'memo_lr', 1e-5)}, "
            f"steps={getattr(self.opt, 'memo_steps', 1)}, "
            f"n_aug={getattr(self.opt, 'memo_n_augmentations', 8)}, "
            f"include_identity={getattr(self.opt, 'memo_include_identity', 1)}, "
            f"hflip_p={getattr(self.opt, 'memo_hflip_p', 0.0)}, "
            f"scope={update_scope}"
        )

    def asm_segmentation_loss(self, logits, labels):
        """Reuse DCON's supervised Dice+CE segmentation loss for ASM source batches."""
        labels = labels.long()
        loss_dice = self.criterionDice(input=logits, target=labels)
        loss_ce = self.criterionCE(inputs=logits, targets=labels)
        loss = (loss_dice * self.opt.w_dice + loss_ce * self.opt.w_ce) * self.opt.w_seg
        return loss

    def _source_anchor_enabled(self):
        return bool(getattr(self.opt, 'source_access', False)) and getattr(self.opt, 'lambda_source', 0.0) > 0.0

    def _source_iter_attr(self):
        if self.tta_mode == 'sar':
            return 'sar_source_loader', 'sar_source_iter'
        return 'tent_source_loader', 'tent_source_iter'

    def _next_anchor_source_batch(self):
        loader_attr, iter_attr = self._source_iter_attr()
        source_loader = getattr(self, loader_attr, None)
        if source_loader is None:
            raise RuntimeError(
                f"{self.tta_mode} with --source_access true requires a labeled "
                "source-domain training loader."
            )
        source_iter = getattr(self, iter_attr, None)
        if source_iter is None:
            source_iter = iter(source_loader)
            setattr(self, iter_attr, source_iter)
        try:
            return next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            setattr(self, iter_attr, source_iter)
            try:
                return next(source_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{self.tta_mode} source loader yielded no batches. Check "
                    "the source training split, batch size, and drop_last setting."
                ) from exc

    def _extract_source_anchor(self, batch):
        if isinstance(batch, dict):
            image = batch.get("base_view", None)
            if image is None:
                image = batch.get("image", None)
            label = batch.get("label", None)
            if image is None or label is None:
                raise KeyError("Source anchor batch must contain an image/base_view and label.")
            return image, label
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(f"Unsupported source anchor batch type: {type(batch)}")

    def _source_anchor_loss(self, batch=None):
        if batch is None:
            batch = self._next_anchor_source_batch()
        source_img, source_label = self._extract_source_anchor(batch)
        source_img = source_img.cuda(non_blocking=True).float()
        source_label = source_label.cuda(non_blocking=True).long()
        source_logits = forward_logits(self.netseg, source_img)
        if source_logits.shape[2:] != source_label.shape[-2:]:
            source_logits = F.interpolate(
                source_logits,
                size=source_label.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        return self.asm_segmentation_loss(source_logits, source_label)

    def smppm_segmentation_loss(self, logits, labels, pixel_weight):
        """DCON Dice+CE source loss with SM-PPM pixel confidence weights."""
        labels_2d = labels.squeeze(1).long() if labels.dim() == 4 else labels.long()
        if pixel_weight.shape[2:] != labels_2d.shape[-2:]:
            pixel_weight = F.interpolate(
                pixel_weight,
                size=labels_2d.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        weights = pixel_weight.squeeze(1).clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-6)

        ce_map = F.cross_entropy(logits, labels_2d, reduction='none')
        loss_ce = (ce_map * weights).sum() / denom

        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(labels_2d, num_classes=self.n_cls).permute(0, 3, 1, 2).to(probs)
        weights_bc = weights.unsqueeze(1)
        smooth = 1e-5
        inter = (weights_bc * probs * target_onehot).sum(dim=(2, 3))
        union = (weights_bc * (probs + target_onehot)).sum(dim=(2, 3)) + smooth
        loss_dice = 1.0 - (2.0 * inter / union).mean()

        loss = (loss_dice * self.opt.w_dice + loss_ce * self.opt.w_ce) * self.opt.w_seg
        return loss

    def configure_asm(self, source_loader=None):
        if source_loader is None:
            source_loader = self.asm_source_loader
        if source_loader is None:
            raise RuntimeError(
                "ASM requires a labeled source-domain training loader. "
                "It is source-dependent TTA, not source-free TTA."
            )

        self.asm_source_loader = source_loader
        self.netseg.train()
        for param in self.netseg.parameters():
            param.requires_grad_(True)

        self.asm_optimizer = torch.optim.Adam(
            [p for p in self.netseg.parameters() if p.requires_grad],
            lr=getattr(self.opt, 'asm_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        self.asm_adapter = ASMAdapter(
            model=self.netseg,
            optimizer=self.asm_optimizer,
            source_loader=source_loader,
            device=next(self.netseg.parameters()).device,
            num_classes=self.n_cls,
            steps=getattr(self.opt, 'asm_steps', 1),
            inner_steps=getattr(self.opt, 'asm_inner_steps', 2),
            lambda_reg=getattr(self.opt, 'asm_lambda_reg', 2e-4),
            sampling_step=getattr(self.opt, 'asm_sampling_step', 20.0),
            episodic=getattr(self.opt, 'asm_episodic', False),
            style_backend=getattr(self.opt, 'asm_style_backend', 'medical_adain'),
            segmentation_criterion=self.asm_segmentation_loss,
        )

        msg = (
            "ASM enabled: source-dependent supervised TTA. "
            "Target images provide style/statistics only; target labels are used only for evaluation. "
            f"lr={getattr(self.opt, 'asm_lr', 1e-4)}, "
            f"steps={getattr(self.opt, 'asm_steps', 1)}, "
            f"inner_steps={getattr(self.opt, 'asm_inner_steps', 2)}, "
            f"lambda_reg={getattr(self.opt, 'asm_lambda_reg', 2e-4)}, "
            f"sampling_step={getattr(self.opt, 'asm_sampling_step', 20.0)}, "
            f"style_backend={getattr(self.opt, 'asm_style_backend', 'medical_adain')}, "
            f"episodic={getattr(self.opt, 'asm_episodic', False)}"
        )
        print(msg)
        logger.info(msg)

    def configure_smppm(self, source_loader=None):
        ablation_mode = getattr(self.opt, 'smppm_ablation_mode', 'full')
        if source_loader is None:
            source_loader = self.smppm_source_loader
        if source_loader is None and ablation_mode != 'source_free_proto':
            raise RuntimeError(
                f"SM-PPM ablation mode {ablation_mode} requires a labeled "
                "source-domain training loader."
            )

        self.smppm_source_loader = source_loader
        self.netseg.train()
        for param in self.netseg.parameters():
            param.requires_grad_(True)

        self.smppm_optimizer = torch.optim.SGD(
            [p for p in self.netseg.parameters() if p.requires_grad],
            lr=getattr(self.opt, 'smppm_lr', 2.5e-4),
            momentum=getattr(self.opt, 'smppm_momentum', 0.9),
            weight_decay=getattr(self.opt, 'smppm_wd', 5e-4),
            nesterov=True,
        )
        self.smppm_adapter = SMPPMAdapter(
            model=self.netseg,
            optimizer=self.smppm_optimizer,
            source_loader=source_loader,
            device=next(self.netseg.parameters()).device,
            num_classes=self.n_cls,
            steps=getattr(self.opt, 'smppm_steps', 1),
            patch_size=getattr(self.opt, 'smppm_patch_size', 8),
            feature_size=getattr(self.opt, 'smppm_feature_size', 32),
            episodic=getattr(self.opt, 'smppm_episodic', False),
            segmentation_criterion=self.smppm_segmentation_loss,
            ablation_mode=ablation_mode,
            source_free_tau=getattr(self.opt, 'smppm_source_free_tau', 0.7),
            source_free_entropy_threshold=getattr(self.opt, 'smppm_source_free_entropy_threshold', None),
            source_free_lambda_proto=getattr(self.opt, 'smppm_source_free_lambda_proto', 1.0),
            source_free_entropy_weight=getattr(self.opt, 'smppm_source_free_entropy_weight', 1.0),
            style_alpha=getattr(self.opt, 'smppm_style_alpha', 1.0),
            log_interval=getattr(self.opt, 'smppm_log_interval', 0),
        )

        msg = (
            "SM-PPM enabled. "
            f"ablation_mode={ablation_mode}, "
            "Target labels are used only for evaluation. "
            f"lr={getattr(self.opt, 'smppm_lr', 2.5e-4)}, "
            f"momentum={getattr(self.opt, 'smppm_momentum', 0.9)}, "
            f"wd={getattr(self.opt, 'smppm_wd', 5e-4)}, "
            f"steps={getattr(self.opt, 'smppm_steps', 1)}, "
            f"src_batch_size={getattr(self.opt, 'smppm_src_batch_size', 2)}, "
            f"patch_size={getattr(self.opt, 'smppm_patch_size', 8)}, "
            f"feature_size={getattr(self.opt, 'smppm_feature_size', 32)}, "
            f"episodic={getattr(self.opt, 'smppm_episodic', False)}, "
            f"source_free_tau={getattr(self.opt, 'smppm_source_free_tau', 0.7)}, "
            "source_free_entropy_threshold="
            f"{getattr(self.opt, 'smppm_source_free_entropy_threshold', None)}, "
            f"source_free_entropy_weight={getattr(self.opt, 'smppm_source_free_entropy_weight', 1.0)}, "
            f"source_free_lambda_proto={getattr(self.opt, 'smppm_source_free_lambda_proto', 1.0)}, "
            f"style_alpha={getattr(self.opt, 'smppm_style_alpha', 1.0)}, "
            f"log_interval={getattr(self.opt, 'smppm_log_interval', 0)}"
        )
        if not getattr(self.opt, 'quiet_console', False):
            print(msg)
            print(self.smppm_adapter.feature_summary())
        logger.info(msg)
        logger.info(self.smppm_adapter.feature_summary())

    def configure_saam_spmm(self):
        update_scope = getattr(self.opt, 'saam_spmm_update_scope', 'bn_affine')
        params, names = configure_model_for_saam_spmm(self.netseg, update_scope=update_scope)
        if len(params) == 0:
            raise RuntimeError("SAAM-SPMM found no trainable parameters.")

        self.saam_spmm_optimizer = torch.optim.Adam(
            params,
            lr=getattr(self.opt, 'saam_spmm_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=getattr(self.opt, 'saam_spmm_weight_decay', 0.0),
        )
        self.saam_spmm_adapter = SAAMSPMMAdapter(
            model=self.netseg,
            optimizer=self.saam_spmm_optimizer,
            device=next(self.netseg.parameters()).device,
            num_classes=self.n_cls,
            steps=getattr(self.opt, 'smppm_steps', 1),
            num_views=getattr(self.opt, 'num_views', 5),
            use_saam=bool(getattr(self.opt, 'use_saam', 1)),
            use_stable_mask=bool(getattr(self.opt, 'use_stable_mask', 1)),
            use_source_anchor=bool(getattr(self.opt, 'use_source_anchor', 1)),
            use_shape_consistency=bool(getattr(self.opt, 'use_shape_consistency', 1)),
            source_prototype_path=getattr(self.opt, 'source_prototype_path', None),
            saam_metric=getattr(self.opt, 'saam_metric', 'variance'),
            stable_threshold=getattr(self.opt, 'stable_threshold', None),
            stable_topk_percent=getattr(self.opt, 'stable_topk_percent', 0.3),
            unstable_weight=getattr(self.opt, 'unstable_weight', 0.1),
            lambda_ent=getattr(self.opt, 'lambda_ent', 1.0),
            lambda_proto=getattr(self.opt, 'lambda_proto', 1.0),
            lambda_shape=getattr(self.opt, 'lambda_shape', 0.1),
            lambda_cons=getattr(self.opt, 'lambda_cons', 1.0),
            proto_momentum=getattr(self.opt, 'proto_momentum', 0.9),
            proto_loss=getattr(self.opt, 'proto_loss', 'cosine'),
            log_interval=getattr(self.opt, 'smppm_log_interval', 1),
        )

        msg = (
            "SAAM-SPMM enabled: source-prototype-assisted target-only online TTA. "
            "No source images are read during test-time adaptation. "
            f"scope={update_scope}, lr={getattr(self.opt, 'saam_spmm_lr', 1e-4)}, "
            f"steps={getattr(self.opt, 'smppm_steps', 1)}, "
            f"num_views={getattr(self.opt, 'num_views', 5)}, "
            f"use_saam={getattr(self.opt, 'use_saam', 1)}, "
            f"use_stable_mask={getattr(self.opt, 'use_stable_mask', 1)}, "
            f"use_source_anchor={getattr(self.opt, 'use_source_anchor', 1)}, "
            f"use_shape_consistency={getattr(self.opt, 'use_shape_consistency', 1)}, "
            f"source_prototype_path={getattr(self.opt, 'source_prototype_path', None)}, "
            f"trainable_tensors={len(names)}"
        )
        if not getattr(self.opt, 'quiet_console', False):
            print(msg)
            print(self.saam_spmm_adapter.feature_summary())
        logger.info(msg)
        logger.info(self.saam_spmm_adapter.feature_summary())
        for name in names:
            logger.info(f"  SAAM-SPMM trainable: {name}")

    def configure_gtta(self, source_loader=None):
        if source_loader is None:
            source_loader = self.gtta_source_loader
        if source_loader is None:
            raise RuntimeError(
                "GTTA requires a labeled source-domain training loader. "
                "It is source-dependent TTA, not source-free TTA."
            )

        self.gtta_source_loader = source_loader
        self.netseg.train()
        for param in self.netseg.parameters():
            param.requires_grad_(True)

        self.gtta_optimizer = torch.optim.SGD(
            [p for p in self.netseg.parameters() if p.requires_grad],
            lr=getattr(self.opt, 'gtta_lr', 2.5e-4),
            momentum=getattr(self.opt, 'gtta_momentum', 0.9),
            weight_decay=getattr(self.opt, 'gtta_wd', 5e-4),
            nesterov=True,
        )
        self.gtta_adapter = GTTAAdapter(
            model=self.netseg,
            optimizer=self.gtta_optimizer,
            source_loader=source_loader,
            device=next(self.netseg.parameters()).device,
            num_classes=self.n_cls,
            steps=getattr(self.opt, 'gtta_steps', 1),
            lambda_ce_trg=getattr(self.opt, 'gtta_lambda_ce_trg', 0.1),
            pseudo_momentum=getattr(self.opt, 'gtta_pseudo_momentum', 0.9),
            style_alpha=getattr(self.opt, 'gtta_style_alpha', 1.0),
            include_original=bool(getattr(self.opt, 'gtta_include_original', 1)),
            episodic=getattr(self.opt, 'gtta_episodic', False),
            ignore_label=getattr(self.opt, 'gtta_ignore_label', 255),
            segmentation_criterion=self.asm_segmentation_loss,
        )

        msg = (
            "GTTA enabled: source-dependent supervised TTA with medical class-aware AdaIN. "
            "Target labels are used only for evaluation. "
            f"lr={getattr(self.opt, 'gtta_lr', 2.5e-4)}, "
            f"momentum={getattr(self.opt, 'gtta_momentum', 0.9)}, "
            f"wd={getattr(self.opt, 'gtta_wd', 5e-4)}, "
            f"steps={getattr(self.opt, 'gtta_steps', 1)}, "
            f"src_batch_size={getattr(self.opt, 'gtta_src_batch_size', 2)}, "
            f"lambda_ce_trg={getattr(self.opt, 'gtta_lambda_ce_trg', 0.1)}, "
            f"pseudo_momentum={getattr(self.opt, 'gtta_pseudo_momentum', 0.9)}, "
            f"style_alpha={getattr(self.opt, 'gtta_style_alpha', 1.0)}, "
            f"include_original={getattr(self.opt, 'gtta_include_original', 1)}, "
            f"episodic={getattr(self.opt, 'gtta_episodic', False)}"
        )
        print(msg)
        logger.info(msg)

    def configure_gold(self):
        self.netseg.train()
        for param in self.netseg.parameters():
            param.requires_grad_(True)

        self.gold_optimizer = torch.optim.SGD(
            [p for p in self.netseg.parameters() if p.requires_grad],
            lr=getattr(self.opt, 'gold_lr', 2.5e-4),
            momentum=getattr(self.opt, 'gold_momentum', 0.9),
            weight_decay=getattr(self.opt, 'gold_wd', 5e-4),
        )
        self.gold_adapter = GOLDAdapter(
            model=self.netseg,
            optimizer=self.gold_optimizer,
            num_classes=self.n_cls,
            steps=getattr(self.opt, 'gold_steps', 1),
            rank=getattr(self.opt, 'gold_rank', 128),
            tau=getattr(self.opt, 'gold_tau', 0.95),
            alpha=getattr(self.opt, 'gold_alpha', 0.02),
            t_eig=getattr(self.opt, 'gold_t_eig', 10),
            mt=getattr(self.opt, 'gold_mt', 0.999),
            s_lr=getattr(self.opt, 'gold_s_lr', 5e-3),
            s_init_scale=getattr(self.opt, 'gold_s_init_scale', 0.0),
            s_clip=getattr(self.opt, 'gold_s_clip', 0.5),
            adapter_scale=getattr(self.opt, 'gold_adapter_scale', 0.05),
            max_pixels_per_batch=getattr(self.opt, 'gold_max_pixels_per_batch', 512),
            min_pixels_per_batch=getattr(self.opt, 'gold_min_pixels_per_batch', 64),
            n_augmentations=getattr(self.opt, 'gold_n_augmentations', 6),
            rst=getattr(self.opt, 'gold_rst', 0.01),
            ap=getattr(self.opt, 'gold_ap', 0.9),
            episodic=getattr(self.opt, 'gold_episodic', False),
        )

        msg = (
            "GOLD enabled: source-free TTA with EMA teacher, stochastic restore, "
            "AGOP subspace, and low-rank pre-classifier feature adapter. "
            f"lr={getattr(self.opt, 'gold_lr', 2.5e-4)}, "
            f"steps={getattr(self.opt, 'gold_steps', 1)}, "
            f"rank={getattr(self.opt, 'gold_rank', 128)}, "
            f"tau={getattr(self.opt, 'gold_tau', 0.95)}, "
            f"alpha={getattr(self.opt, 'gold_alpha', 0.02)}, "
            f"t_eig={getattr(self.opt, 'gold_t_eig', 10)}, "
            f"mt={getattr(self.opt, 'gold_mt', 0.999)}, "
            f"s_lr={getattr(self.opt, 'gold_s_lr', 5e-3)}, "
            f"adapter_scale={getattr(self.opt, 'gold_adapter_scale', 0.05)}, "
            f"rst={getattr(self.opt, 'gold_rst', 0.01)}, "
            f"ap={getattr(self.opt, 'gold_ap', 0.9)}, "
            f"episodic={getattr(self.opt, 'gold_episodic', False)}"
        )
        print(msg)
        logger.info(msg)

    def configure_vptta(self):
        warm_n = getattr(self.opt, 'vptta_warm_n', 5)
        converted = convert_batchnorm_to_vptta(self.netseg, warm_n=warm_n)
        if converted == 0:
            raise RuntimeError("VPTTA requires BatchNorm2d layers, but none were found.")

        self.netseg.eval()
        self.netseg.requires_grad_(False)

        in_channels = self.netseg.convd1.conv1.in_channels if hasattr(self.netseg, 'convd1') else 3
        self.vptta_prompt = FrequencyPrompt(
            in_channels=in_channels,
            image_size=getattr(self.opt, 'vptta_image_size', 192),
            prompt_alpha=getattr(self.opt, 'vptta_prompt_alpha', 0.01),
            prompt_size=getattr(self.opt, 'vptta_prompt_size', None),
        ).cuda()

        optimizer_name = getattr(self.opt, 'vptta_optimizer', 'Adam')
        if optimizer_name == 'SGD':
            self.vptta_optimizer = torch.optim.SGD(
                self.vptta_prompt.parameters(),
                lr=getattr(self.opt, 'vptta_lr', 1e-2),
                momentum=getattr(self.opt, 'vptta_momentum', 0.99),
                nesterov=True,
                weight_decay=getattr(self.opt, 'vptta_weight_decay', 0.0),
            )
        elif optimizer_name == 'Adam':
            self.vptta_optimizer = torch.optim.Adam(
                self.vptta_prompt.parameters(),
                lr=getattr(self.opt, 'vptta_lr', 1e-2),
                betas=(
                    getattr(self.opt, 'vptta_beta1', 0.9),
                    getattr(self.opt, 'vptta_beta2', 0.99),
                ),
                weight_decay=getattr(self.opt, 'vptta_weight_decay', 0.0),
            )
        else:
            raise ValueError(f"Unknown VPTTA optimizer: {optimizer_name}")

        self.vptta_memory = PromptMemory(
            size=getattr(self.opt, 'vptta_memory_size', 40),
            dimension=self.vptta_prompt.data_prompt.numel(),
        )
        self.vptta_adapter = VPTTAAdapter(
            model=self.netseg,
            prompt=self.vptta_prompt,
            optimizer=self.vptta_optimizer,
            memory_bank=self.vptta_memory,
            steps=getattr(self.opt, 'vptta_steps', 1),
            neighbor=getattr(self.opt, 'vptta_neighbor', 16),
        )

        msg = (
            "VPTTA enabled: source-free frequency prompt TTA. "
            "Only the low-frequency prompt is updated; target labels are used only for evaluation. "
            f"converted_bn={converted}, "
            f"optimizer={optimizer_name}, "
            f"lr={getattr(self.opt, 'vptta_lr', 1e-2)}, "
            f"steps={getattr(self.opt, 'vptta_steps', 1)}, "
            f"prompt_size={self.vptta_prompt.prompt_size}, "
            f"prompt_alpha={getattr(self.opt, 'vptta_prompt_alpha', 0.01)}, "
            f"memory_size={getattr(self.opt, 'vptta_memory_size', 40)}, "
            f"neighbor={getattr(self.opt, 'vptta_neighbor', 16)}, "
            f"warm_n={warm_n}"
        )
        print(msg)
        logger.info(msg)

    def configure_pass(self):
        self.pass_source_model = deepcopy(self.netseg)
        self.pass_source_model.eval()
        for param in self.pass_source_model.parameters():
            param.detach_()
            param.requires_grad_(False)

        prompt_size = getattr(self.opt, 'pass_prompt_size', None)
        if prompt_size is None:
            image_size = getattr(self.opt, 'pass_image_size', 192)
            prompt_size = max(1, int(image_size) // 16)

        self.netseg = PASSPromptedUNet(
            self.netseg,
            prompt_size=prompt_size,
            adaptor_hidden=getattr(self.opt, 'pass_adaptor_hidden', 64),
            perturb_scale=getattr(self.opt, 'pass_perturb_scale', 1.0),
            prompt_scale=getattr(self.opt, 'pass_prompt_scale', 1.0),
            prompt_sparsity=getattr(self.opt, 'pass_prompt_sparsity', 0.1),
        ).cuda()

        trainable_names = configure_pass_model(
            self.netseg,
            train_bn_affine=getattr(self.opt, 'pass_update_bn_affine', True),
        )
        params = [param for param in self.netseg.parameters() if param.requires_grad]
        if len(params) == 0:
            raise RuntimeError("PASS found no trainable prompt/BN parameters.")

        optimizer_name = getattr(self.opt, 'pass_optimizer', 'Adam')
        if optimizer_name == 'SGD':
            self.pass_optimizer = torch.optim.SGD(
                params,
                lr=getattr(self.opt, 'pass_lr', 5e-3),
                momentum=getattr(self.opt, 'pass_momentum', 0.99),
                nesterov=True,
                weight_decay=getattr(self.opt, 'pass_weight_decay', 0.0),
            )
        elif optimizer_name == 'Adam':
            self.pass_optimizer = torch.optim.Adam(
                params,
                lr=getattr(self.opt, 'pass_lr', 5e-3),
                betas=(
                    getattr(self.opt, 'pass_beta1', 0.9),
                    getattr(self.opt, 'pass_beta2', 0.999),
                ),
                weight_decay=getattr(self.opt, 'pass_weight_decay', 0.0),
            )
        else:
            raise ValueError(f"Unknown PASS optimizer: {optimizer_name}")

        self.pass_bn_monitor = BatchNormFeatureMonitor(
            self.netseg,
            alpha=getattr(self.opt, 'pass_bn_alpha', 0.01),
            max_layers=getattr(self.opt, 'pass_bn_layers', 0),
        )
        self.pass_adapter = PASSAdapter(
            model=self.netseg,
            source_model=self.pass_source_model,
            optimizer=self.pass_optimizer,
            bn_monitor=self.pass_bn_monitor,
            steps=getattr(self.opt, 'pass_steps', 1),
            entropy_weight=getattr(self.opt, 'pass_entropy_weight', 0.0),
            ema_decay=getattr(self.opt, 'pass_ema_decay', 0.94),
            min_momentum_constant=getattr(self.opt, 'pass_min_momentum_constant', 0.01),
            episodic=getattr(self.opt, 'pass_episodic', False),
            use_source_fallback=getattr(self.opt, 'pass_use_source_fallback', True),
        )

        msg = (
            "PASS enabled: source-free style/shape prompt TTA adapted to DCON U-Net. "
            "Target labels are used only for evaluation. "
            f"optimizer={optimizer_name}, "
            f"lr={getattr(self.opt, 'pass_lr', 5e-3)}, "
            f"steps={getattr(self.opt, 'pass_steps', 1)}, "
            f"prompt_size={prompt_size}, "
            f"bn_layers={len(self.pass_bn_monitor.records)}, "
            f"bn_alpha={getattr(self.opt, 'pass_bn_alpha', 0.01)}, "
            f"entropy_weight={getattr(self.opt, 'pass_entropy_weight', 0.0)}, "
            f"ema_decay={getattr(self.opt, 'pass_ema_decay', 0.94)}, "
            f"source_fallback={getattr(self.opt, 'pass_use_source_fallback', True)}, "
            f"episodic={getattr(self.opt, 'pass_episodic', False)}, "
            f"trainable_tensors={len(trainable_names)}"
        )
        print(msg)
        logger.info(msg)
        for name in trainable_names:
            logger.info(f"  PASS trainable: {name}")

    def configure_samtta(self):
        update_scope = getattr(self.opt, 'samtta_update_scope', 'bn_affine')
        model_params, model_param_names = configure_model_for_samtta(
            self.netseg,
            update_scope=update_scope,
        )

        self.samtta_transform = SAMTTABezierTransform().cuda()
        param_groups = [
            {
                'params': list(self.samtta_transform.parameters()),
                'lr': getattr(self.opt, 'samtta_transform_lr', 1e-2),
            }
        ]
        if len(model_params) > 0:
            param_groups.append({
                'params': model_params,
                'lr': getattr(self.opt, 'samtta_lr', 1e-4),
            })

        self.samtta_optimizer = torch.optim.Adam(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=getattr(self.opt, 'samtta_weight_decay', 0.0),
        )
        self.samtta_adapter = SAMTTAAdapter(
            model=self.netseg,
            transform=self.samtta_transform,
            optimizer=self.samtta_optimizer,
            steps=getattr(self.opt, 'samtta_steps', 1),
            ema_momentum=getattr(self.opt, 'samtta_ema_momentum', 0.95),
            dpc_weight=getattr(self.opt, 'samtta_dpc_weight', 1.0),
            feature_weight=getattr(self.opt, 'samtta_feature_weight', 0.1),
            entropy_weight=getattr(self.opt, 'samtta_entropy_weight', 0.05),
            transform_reg_weight=getattr(self.opt, 'samtta_transform_reg_weight', 0.01),
            feature_temp=getattr(self.opt, 'samtta_feature_temp', 2.0),
            episodic=getattr(self.opt, 'samtta_episodic', False),
        )

        msg = (
            "SAM-TTA enabled: source-free Bezier input transform plus EMA teacher-student "
            "prediction/feature consistency adapted to DCON U-Net. "
            "Target labels are used only for evaluation. "
            f"scope={update_scope}, "
            f"lr={getattr(self.opt, 'samtta_lr', 1e-4)}, "
            f"transform_lr={getattr(self.opt, 'samtta_transform_lr', 1e-2)}, "
            f"steps={getattr(self.opt, 'samtta_steps', 1)}, "
            f"ema={getattr(self.opt, 'samtta_ema_momentum', 0.95)}, "
            f"dpc_w={getattr(self.opt, 'samtta_dpc_weight', 1.0)}, "
            f"feature_w={getattr(self.opt, 'samtta_feature_weight', 0.1)}, "
            f"entropy_w={getattr(self.opt, 'samtta_entropy_weight', 0.05)}, "
            f"transform_reg_w={getattr(self.opt, 'samtta_transform_reg_weight', 0.01)}, "
            f"episodic={getattr(self.opt, 'samtta_episodic', False)}, "
            f"trainable_model_tensors={len(model_param_names)}"
        )
        print(msg)
        logger.info(msg)
        for name in model_param_names:
            logger.info(f"  SAM-TTA trainable: {name}")

    def configure_spmo(self):
        self.spmo_source_model = deepcopy(self.netseg)
        self.spmo_source_model.eval()
        for param in self.spmo_source_model.parameters():
            param.detach_()
            param.requires_grad_(False)

        update_scope = getattr(self.opt, 'spmo_update_scope', 'bn_affine')
        if update_scope == 'bn_affine':
            configure_model_for_tent(self.netseg)
            params, names = collect_bn_affine_params(self.netseg)
            if len(params) == 0:
                raise RuntimeError("SPMO requires BatchNorm2d affine parameters, but none were found.")
            trainable_desc = f"{len(params)} BatchNorm affine tensors"
        elif update_scope == 'all':
            self.netseg.train()
            self.netseg.requires_grad_(True)
            params = [param for param in self.netseg.parameters() if param.requires_grad]
            names = [name for name, param in self.netseg.named_parameters() if param.requires_grad]
            trainable_desc = f"all segmentation parameters ({len(params)} tensors)"
        else:
            raise ValueError(f"Unknown spmo_update_scope: {update_scope}")

        self.spmo_optimizer = torch.optim.Adam(
            params,
            lr=getattr(self.opt, 'spmo_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=getattr(self.opt, 'spmo_weight_decay', 0.0),
        )
        self.spmo_adapter = SPMOAdapter(
            model=self.netseg,
            source_model=self.spmo_source_model,
            optimizer=self.spmo_optimizer,
            num_classes=self.n_cls,
            steps=getattr(self.opt, 'spmo_steps', 1),
            entropy_weight=getattr(self.opt, 'spmo_entropy_weight', 1.0),
            prior_weight=getattr(self.opt, 'spmo_prior_weight', 1.0),
            moment_weight=getattr(self.opt, 'spmo_moment_weight', 0.05),
            moment_mode=getattr(self.opt, 'spmo_moment_mode', 'all'),
            softmax_temp=getattr(self.opt, 'spmo_softmax_temp', 1.0),
            size_power=getattr(self.opt, 'spmo_size_power', 1.0),
            bg_entropy_weight=getattr(self.opt, 'spmo_bg_entropy_weight', 0.1),
            prior_eps=getattr(self.opt, 'spmo_prior_eps', 1e-6),
            min_pixels=getattr(self.opt, 'spmo_min_pixels', 10),
            source_pseudo=getattr(self.opt, 'spmo_source_pseudo', 'hard'),
            episodic=getattr(self.opt, 'spmo_episodic', False),
        )

        msg = (
            "SPMO enabled: source-free shape-moment TTA adapted to DCON. "
            "A frozen source model supplies per-slice size and moment priors; "
            "target labels are used only for evaluation. "
            f"scope={update_scope}, trainable={trainable_desc}, "
            f"lr={getattr(self.opt, 'spmo_lr', 1e-4)}, "
            f"steps={getattr(self.opt, 'spmo_steps', 1)}, "
            f"entropy_weight={getattr(self.opt, 'spmo_entropy_weight', 1.0)}, "
            f"prior_weight={getattr(self.opt, 'spmo_prior_weight', 1.0)}, "
            f"moment_weight={getattr(self.opt, 'spmo_moment_weight', 0.05)}, "
            f"moment_mode={getattr(self.opt, 'spmo_moment_mode', 'all')}, "
            f"source_pseudo={getattr(self.opt, 'spmo_source_pseudo', 'hard')}, "
            f"episodic={getattr(self.opt, 'spmo_episodic', False)}"
        )
        print(msg)
        logger.info(msg)
        for name in names:
            logger.info(f"  SPMO trainable: {name}")

    def configure_sictta(self):
        self.sictta_adapter = SicTTAAdapter(
            model=self.netseg,
            num_classes=self.n_cls,
            max_lens=getattr(self.opt, 'sictta_max_lens', 40),
            topk=getattr(self.opt, 'sictta_topk', 5),
            threshold=getattr(self.opt, 'sictta_threshold', 0.9),
            select_points=getattr(self.opt, 'sictta_select_points', 200),
            episodic=getattr(self.opt, 'sictta_episodic', False),
        )
        configure_model_for_sictta(self.netseg)

        msg = (
            "SicTTA enabled: source-free single-image continual TTA with "
            "source-anchor CCD filtering and bottleneck prototype feature mixing. "
            "Target labels are used only for evaluation. "
            f"max_lens={getattr(self.opt, 'sictta_max_lens', 40)}, "
            f"topk={getattr(self.opt, 'sictta_topk', 5)}, "
            f"threshold={getattr(self.opt, 'sictta_threshold', 0.9)}, "
            f"select_points={getattr(self.opt, 'sictta_select_points', 200)}, "
            f"episodic={getattr(self.opt, 'sictta_episodic', False)}"
        )
        print(msg)
        logger.info(msg)

    def configure_a3_tta(self):
        configure_model_for_a3_tta(self.netseg)
        params = [param for param in self.netseg.parameters() if param.requires_grad]
        if len(params) == 0:
            raise RuntimeError("A3-TTA found no trainable segmentation parameters.")

        self.a3_optimizer = torch.optim.Adam(
            params,
            lr=getattr(self.opt, 'a3_lr', 1e-4),
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )
        self.a3_adapter = A3TTAAdapter(
            model=self.netseg,
            optimizer=self.a3_optimizer,
            num_classes=self.n_cls,
            steps=getattr(self.opt, 'a3_steps', 1),
            mt_alpha=getattr(self.opt, 'a3_mt', 0.99),
            pool_size=getattr(self.opt, 'a3_pool_size', 40),
            top_k=getattr(self.opt, 'a3_top_k', 1),
            feature_loss_weight=getattr(self.opt, 'a3_feature_loss_weight', 1.0),
            entropy_match_weight=getattr(self.opt, 'a3_entropy_match_weight', 5.0),
            ema_loss_weight=getattr(self.opt, 'a3_ema_loss_weight', 1.0),
            episodic=getattr(self.opt, 'a3_episodic', False),
            reset_on_scan_start=getattr(self.opt, 'a3_reset_on_scan_start', False),
        )

        msg = (
            "A3-TTA enabled: source-free online anchor alignment adapted to DCON U-Net. "
            "Target labels are used only for evaluation. "
            f"lr={getattr(self.opt, 'a3_lr', 1e-4)}, "
            f"steps={getattr(self.opt, 'a3_steps', 1)}, "
            f"pool_size={getattr(self.opt, 'a3_pool_size', 40)}, "
            f"top_k={getattr(self.opt, 'a3_top_k', 1)}, "
            f"mt={getattr(self.opt, 'a3_mt', 0.99)}, "
            f"feature_w={getattr(self.opt, 'a3_feature_loss_weight', 1.0)}, "
            f"entropy_w={getattr(self.opt, 'a3_entropy_match_weight', 5.0)}, "
            f"ema_w={getattr(self.opt, 'a3_ema_loss_weight', 1.0)}, "
            f"episodic={getattr(self.opt, 'a3_episodic', False)}, "
            f"reset_on_scan_start={getattr(self.opt, 'a3_reset_on_scan_start', False)}, "
            f"trainable_tensors={len(params)}"
        )
        print(msg)
        logger.info(msg)

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
        loss_tta = None
        loss_source = None
        self.netseg.train()
        for _ in range(getattr(self.opt, 'tent_steps', 1)):
            logits = forward_logits(self.netseg, self.input_img_te)
            loss_tta = softmax_entropy_seg(logits)
            if self._source_anchor_enabled():
                loss_source = self._source_anchor_loss()
                loss = loss_tta + getattr(self.opt, 'lambda_source', 0.0) * loss_source
            else:
                loss_source = torch.zeros((), device=self.input_img_te.device)
                loss = loss_tta
            self.tent_optimizer.zero_grad()
            loss.backward()
            self.tent_optimizer.step()

        seg = torch.argmax(logits.detach(), 1)
        self.netseg.zero_grad()
        self.last_tent_loss = loss.detach() if loss is not None else None
        self.last_tent_losses = {
            "tent_loss_total": float(loss.detach().item()) if loss is not None else 0.0,
            "tent_loss_tta": float(loss_tta.detach().item()) if loss_tta is not None else 0.0,
            "tent_loss_source": float(loss_source.detach().item()) if loss_source is not None else 0.0,
            "tent_source_access": float(self._source_anchor_enabled()),
            "tent_updated_params": float(parameter_count(self.tent_optimizer.param_groups[0]["params"])),
        }
        return self.input_mask_te, seg

    def _sar_grad_norm(self):
        shared_device = self.sar_params[0].device
        grad_norms = [
            param.grad.detach().norm(p=2).to(shared_device)
            for param in self.sar_params
            if param.grad is not None
        ]
        if len(grad_norms) == 0:
            return torch.zeros((), device=shared_device)
        norm = torch.norm(torch.stack(grad_norms), p=2)
        return norm

    @torch.no_grad()
    def _sar_perturb(self, grad_norm):
        scale = getattr(self.opt, 'sar_rho', 0.05) / (grad_norm + 1e-12)
        perturbations = []
        for param in self.sar_params:
            if param.grad is None:
                perturbations.append(None)
                continue
            eps = param.grad.detach() * scale.to(param.device)
            param.add_(eps)
            perturbations.append(eps)
        return perturbations

    @torch.no_grad()
    def _sar_restore(self, perturbations):
        for param, eps in zip(self.sar_params, perturbations):
            if eps is not None:
                param.sub_(eps)

    def _sar_total_loss(self, source_batch=None):
        logits = forward_logits(self.netseg, self.input_img_te)
        loss_tta = softmax_entropy_seg(logits)
        if self._source_anchor_enabled():
            loss_source = self._source_anchor_loss(source_batch)
            loss = loss_tta + getattr(self.opt, 'lambda_source', 0.0) * loss_source
        else:
            loss_source = torch.zeros((), device=self.input_img_te.device)
            loss = loss_tta
        return logits, loss, loss_tta, loss_source

    @torch.enable_grad()
    def te_func_sar(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.sar_optimizer is None:
            raise RuntimeError("SAR optimizer is not initialized. Call configure_sar() first.")

        logits = None
        loss = None
        loss_tta = None
        loss_source = None
        self.netseg.train()
        for _ in range(getattr(self.opt, 'sar_steps', 1)):
            source_batch = self._next_anchor_source_batch() if self._source_anchor_enabled() else None
            self.sar_optimizer.zero_grad()
            _, first_loss, _, _ = self._sar_total_loss(source_batch)
            first_loss.backward()
            grad_norm = self._sar_grad_norm()
            perturbations = self._sar_perturb(grad_norm)

            self.sar_optimizer.zero_grad()
            logits, loss, loss_tta, loss_source = self._sar_total_loss(source_batch)
            loss.backward()
            self._sar_restore(perturbations)
            self.sar_optimizer.step()

        with torch.no_grad():
            final_logits = forward_logits(self.netseg, self.input_img_te)
        seg = torch.argmax(final_logits.detach(), 1)
        self.netseg.zero_grad()
        self.last_sar_losses = {
            "sar_loss_total": float(loss.detach().item()) if loss is not None else 0.0,
            "sar_loss_tta": float(loss_tta.detach().item()) if loss_tta is not None else 0.0,
            "sar_loss_source": float(loss_source.detach().item()) if loss_source is not None else 0.0,
            "sar_source_access": float(self._source_anchor_enabled()),
            "sar_updated_params": float(parameter_count(self.sar_params)),
        }
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_dg_tta(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.dgtta_adapter is None:
            raise RuntimeError("DG-TTA adapter is not initialized. Call configure_dg_tta() first.")

        logits = self.dgtta_adapter.forward(self.input_img_te)
        seg = torch.argmax(logits.detach(), 1)
        self.netseg.zero_grad()
        self.last_dgtta_losses = self.dgtta_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_cotta(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.cotta_optimizer is None:
            raise RuntimeError("CoTTA optimizer is not initialized. Call configure_cotta() first.")

        self.netseg.train()
        outputs_ema = None
        loss = None
        loss_tta = None
        loss_source = None
        for _ in range(getattr(self.opt, 'cotta_steps', 1)):
            outputs = forward_logits(self.netseg, self.input_img_te)

            with torch.no_grad():
                anchor_logits = forward_logits(self.cotta_model_anchor, self.input_img_te)
                anchor_prob = torch.softmax(anchor_logits, dim=1).max(dim=1)[0]
                outputs_ema = forward_logits(self.cotta_model_ema, self.input_img_te)
                if anchor_prob.mean() < getattr(self.opt, 'cotta_ap', 0.9):
                    outputs_ema = self.cotta_ensemble_prediction(self.input_img_te, outputs_ema)

            loss_tta = (-(outputs_ema.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()
            if self._source_anchor_enabled():
                loss_source = self._source_anchor_loss()
                loss = loss_tta + getattr(self.opt, 'lambda_source', 0.0) * loss_source
            else:
                loss_source = torch.zeros((), device=self.input_img_te.device)
                loss = loss_tta
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
        self.last_cotta_losses = {
            "cotta_loss_total": float(loss.detach().item()) if loss is not None else 0.0,
            "cotta_loss_tta": float(loss_tta.detach().item()) if loss_tta is not None else 0.0,
            "cotta_loss_source": float(loss_source.detach().item()) if loss_source is not None else 0.0,
            "cotta_source_access": float(self._source_anchor_enabled()),
            "cotta_updated_params": float(parameter_count(self.cotta_optimizer.param_groups[0]["params"])),
        }
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_memo(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.memo_optimizer is None:
            raise RuntimeError("MEMO optimizer is not initialized. Call configure_memo() first.")

        self.netseg.train()
        loss = None
        for _ in range(getattr(self.opt, 'memo_steps', 1)):
            memo_batch = build_memo_batch(
                self.input_img_te,
                n_augmentations=getattr(self.opt, 'memo_n_augmentations', 8),
                include_identity=bool(getattr(self.opt, 'memo_include_identity', 1)),
                hflip_p=getattr(self.opt, 'memo_hflip_p', 0.0),
            )
            outputs = forward_logits(self.netseg, memo_batch)
            loss = marginal_entropy(outputs)
            self.memo_optimizer.zero_grad()
            loss.backward()
            self.memo_optimizer.step()

        with torch.no_grad():
            final_logits = forward_logits(self.netseg, self.input_img_te)
            seg = torch.argmax(final_logits, 1)

        self.netseg.zero_grad()
        self.last_memo_loss = loss.detach() if loss is not None else None
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_asm(self, input):
        self.input_img_te = input['image'].float().cuda()
        # This label is returned for evaluation only. ASM adaptation below never
        # uses the target label or a target pseudo-label for its loss.
        self.input_mask_te = input['label'].float().cuda()

        if self.asm_adapter is None:
            raise RuntimeError("ASM adapter is not initialized. Call configure_asm(source_loader) first.")

        logits = self.asm_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_asm_losses = self.asm_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_smppm(self, input):
        self.input_img_te = input['image'].float().cuda()
        # This label is returned for evaluation only. SM-PPM adaptation below
        # uses source labels and target image features, never target labels.
        self.input_mask_te = input['label'].float().cuda()

        if self.smppm_adapter is None:
            raise RuntimeError("SM-PPM adapter is not initialized. Call configure_smppm(source_loader) first.")

        logits = self.smppm_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_smppm_losses = self.smppm_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_saam_spmm(self, input):
        self.input_img_te = input['image'].float().cuda()
        # Target labels are returned for evaluation only. SAAM-SPMM adaptation
        # uses target predictions/features and optional pre-exported source
        # prototypes, never raw source images and never target labels.
        self.input_mask_te = input['label'].float().cuda()

        if self.saam_spmm_adapter is None:
            raise RuntimeError("SAAM-SPMM adapter is not initialized. Call configure_saam_spmm() first.")

        logits = self.saam_spmm_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_saam_spmm_losses = self.saam_spmm_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_gtta(self, input):
        self.input_img_te = input['image'].float().cuda()
        # This label is returned for evaluation only. GTTA adaptation below uses
        # source labels and model-generated target pseudo-labels, never target labels.
        self.input_mask_te = input['label'].float().cuda()

        if self.gtta_adapter is None:
            raise RuntimeError("GTTA adapter is not initialized. Call configure_gtta(source_loader) first.")

        logits = self.gtta_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_gtta_losses = self.gtta_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_gold(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.gold_adapter is None:
            raise RuntimeError("GOLD adapter is not initialized. Call configure_gold() first.")

        logits = self.gold_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_gold_losses = self.gold_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_vptta(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.vptta_adapter is None:
            raise RuntimeError("VPTTA adapter is not initialized. Call configure_vptta() first.")

        logits = self.vptta_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_vptta_losses = self.vptta_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_pass(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.pass_adapter is None:
            raise RuntimeError("PASS adapter is not initialized. Call configure_pass() first.")

        logits = self.pass_adapter.forward(
            self.input_img_te,
            is_start=bool(input.get('is_start', False)),
        )
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_pass_losses = self.pass_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_samtta(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.samtta_adapter is None:
            raise RuntimeError("SAM-TTA adapter is not initialized. Call configure_samtta() first.")

        logits = self.samtta_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_samtta_losses = self.samtta_adapter.last_losses
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_spmo(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.spmo_adapter is None:
            raise RuntimeError("SPMO adapter is not initialized. Call configure_spmo() first.")

        logits = self.spmo_adapter.forward(self.input_img_te)
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_spmo_losses = self.spmo_adapter.last_losses
        return self.input_mask_te, seg

    @torch.no_grad()
    def te_func_sictta(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.sictta_adapter is None:
            raise RuntimeError("SicTTA adapter is not initialized. Call configure_sictta() first.")

        logits = self.sictta_adapter.forward(self.input_img_te, names=input.get('names', None))
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_sictta_stats = self.sictta_adapter.last_stats
        return self.input_mask_te, seg

    @torch.enable_grad()
    def te_func_a3_tta(self, input):
        self.input_img_te = input['image'].float().cuda()
        self.input_mask_te = input['label'].float().cuda()

        if self.a3_adapter is None:
            raise RuntimeError("A3-TTA adapter is not initialized. Call configure_a3_tta() first.")

        logits = self.a3_adapter.forward(
            self.input_img_te,
            names=input.get('names', None),
            is_start=bool(input.get('is_start', False)),
        )
        if logits.shape[2:] != self.input_mask_te.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=self.input_mask_te.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        seg = torch.argmax(logits.detach(), 1)

        self.netseg.zero_grad()
        self.last_a3_losses = self.a3_adapter.last_losses
        return self.input_mask_te, seg

    def te_func(self,input):
        tta_mode = getattr(self.opt, 'tta', 'none')
        if tta_mode == 'tent':
            return self.te_func_tent(input)
        if tta_mode == 'sar':
            return self.te_func_sar(input)
        if tta_mode == 'dg_tta':
            return self.te_func_dg_tta(input)
        if tta_mode in ['norm_test', 'norm_alpha', 'norm_ema']:
            return self.te_func_norm(input)
        if tta_mode == 'cotta':
            return self.te_func_cotta(input)
        if tta_mode == 'memo':
            return self.te_func_memo(input)
        if tta_mode == 'asm':
            return self.te_func_asm(input)
        if tta_mode in ['sm_ppm', 'source_ce_only']:
            return self.te_func_smppm(input)
        if tta_mode == 'saam_spmm':
            return self.te_func_saam_spmm(input)
        if tta_mode == 'gtta':
            return self.te_func_gtta(input)
        if tta_mode == 'gold':
            return self.te_func_gold(input)
        if tta_mode == 'vptta':
            return self.te_func_vptta(input)
        if tta_mode == 'pass':
            return self.te_func_pass(input)
        if tta_mode == 'samtta':
            return self.te_func_samtta(input)
        if tta_mode == 'spmo':
            return self.te_func_spmo(input)
        if tta_mode == 'sictta':
            return self.te_func_sictta(input)
        if tta_mode == 'a3_tta':
            return self.te_func_a3_tta(input)

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
