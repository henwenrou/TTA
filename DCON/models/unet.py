import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Stage 2: ChannelGate for shallow-layer channel decoupling
# ============================================================================
class ChannelGate(nn.Module):
    """
    Lightweight channel-wise gating mechanism for decoupling structure and style features.

    Args:
        num_channels: Number of channels in the first encoder layer
        use_temperature: Whether to use temperature sharpening (CCSDG-style)
        tau: Temperature parameter for sharpening (lower = more decisive)

    Usage:
        f_str, f_sty = channel_gate(f1)
        - f_str: structure features (used in backbone)
        - f_sty: style features (used only in loss computation)
    """
    def __init__(self, num_channels, use_temperature=False, tau=0.1):
        super(ChannelGate, self).__init__()
        self.use_temperature = use_temperature
        self.tau = tau

        if use_temperature:
            # CCSDG-style: 2 competing logits with softmax
            # [2, C, 1, 1]: first row for structure, second row for style
            self.logits = nn.Parameter(torch.randn(2, num_channels, 1, 1))
        else:
            # Our original style: single logit with sigmoid
            # [1, C, 1, 1]: sigmoid(logit) for structure, 1-sigmoid for style
            self.logits = nn.Parameter(torch.randn(1, num_channels, 1, 1) * 2.0)

    def forward(self, f):
        """
        Args:
            f: input feature [B, C, H, W]
        Returns:
            f_str: structure features [B, C, H, W]
            f_sty: style features [B, C, H, W]
        """
        if self.use_temperature:
            # CCSDG-style: softmax with temperature sharpening
            weights = torch.softmax(self.logits / self.tau, dim=0)  # [2, C, 1, 1]
            f_str = f * weights[0:1]  # [B, C, H, W] * [1, C, 1, 1]
            f_sty = f * weights[1:2]  # [B, C, H, W] * [1, C, 1, 1]
        else:
            # Our original: sigmoid
            m = torch.sigmoid(self.logits)  # [1, C, 1, 1]
            f_str = f * m
            f_sty = f * (1.0 - m)

        return f_str, f_sty


class Projector(nn.Module):
    """
    CCSDG-inspired projector for mapping channel-gated features to abstract semantic space.

    Transforms high-dimensional spatial features [B, C, H, W] to low-dimensional semantic
    vectors [B, proj_dim] through:
      1. Channel reduction (C → hidden_channels)
      2. Spatial downsampling (MaxPool)
      3. Learned projection (FC layer)
      4. L2 normalization (unit hypersphere)

    Args:
        in_channels: Input feature channels (default: 16 for Layer 1)
        hidden_channels: Intermediate channels after conv (default: 8)
        proj_dim: Output projection dimension (default: 1024)
        feature_size: Expected spatial size of input features (default: 256)

    Usage:
        projector = Projector(in_channels=16, proj_dim=1024)
        f_str = channel_gate(layer1_features)[0]  # [B, 16, 256, 256]
        z_str = projector(f_str)  # [B, 1024], L2-normalized
    """
    def __init__(self, in_channels=16, hidden_channels=8, proj_dim=1024, feature_size=256):
        super(Projector, self).__init__()

        # Channel reduction: C → hidden_channels
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)

        # Spatial downsampling: H×W → H/2×W/2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size after conv+pool
        # feature_size / 2 (due to pool)
        pooled_size = feature_size // 2
        fc_input_size = hidden_channels * pooled_size * pooled_size

        # Learned projection to abstract space
        self.fc = nn.Linear(fc_input_size, proj_dim)

        self.proj_dim = proj_dim
        self.fc_input_size = fc_input_size

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            z: L2-normalized projection [B, proj_dim]
        """
        # Conv + BN + ReLU + Pool
        h = self.conv(x)          # [B, C, H, W] → [B, hidden_channels, H, W]
        h = self.bn(h)
        h = F.relu(h)
        h = self.pool(h)          # [B, hidden_channels, H, W] → [B, hidden_channels, H/2, W/2]

        # Flatten
        batch_size = h.size(0)
        h = h.view(batch_size, -1)  # [B, hidden_channels * H/2 * W/2]

        # Check if flattened size matches expected
        if h.size(1) != self.fc_input_size:
            raise RuntimeError(
                f"Projector input size mismatch! Expected {self.fc_input_size}, got {h.size(1)}. "
                f"This usually means the feature_size passed to Projector.__init__ doesn't match "
                f"the actual spatial size of input features. Please initialize Projector with the "
                f"correct feature_size parameter."
            )

        # Project to abstract space
        z = self.fc(h)  # [B, proj_dim]

        # L2 normalize: project to unit hypersphere
        # This makes cosine similarity = dot product, and stabilizes training
        z = F.normalize(z, p=2, dim=1)  # ||z|| = 1

        return z


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m


#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        if activation == 'relu': 
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):


        if not self.first:
            x = self.maxpool2D(x)

        #layer 1 conv, bn
        x = self.conv1(x)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.activation(y)

        #layer 3 conv, bn
        z = self.conv3(y)
        z = self.bn3(z)
        z = self.activation(z)

        return z


class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, activation='relu'):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, prev):
        #layer 1 conv, bn, relu
        if not self.first:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.activation(y)

        return y

class Unet1(nn.Module):
    supports_cgsd = True

    def __init__(self, c=3, n=16, norm='bn', num_classes=2, activation='relu',
                 use_channel_gate=False, cgsd_layer=1, use_temperature=False, gate_tau=0.1):
        super(Unet1, self).__init__()

        self.use_channel_gate = use_channel_gate
        self.cgsd_layer = cgsd_layer  # 1, 2, or 3 - which encoder layer to apply CGSD

        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)

        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.seg1 = nn.Conv2d(2*n, num_classes, 3, padding=1)

        # Stage 2: Structure/Style feature decoupling via learnable channel gating
        if self.use_channel_gate:
            if cgsd_layer == 1:
                self.chan_gate = ChannelGate(num_channels=n, use_temperature=use_temperature, tau=gate_tau)      # 16 channels
            elif cgsd_layer == 2:
                self.chan_gate = ChannelGate(num_channels=2*n, use_temperature=use_temperature, tau=gate_tau)    # 32 channels
            elif cgsd_layer == 3:
                self.chan_gate = ChannelGate(num_channels=4*n, use_temperature=use_temperature, tau=gate_tau)    # 64 channels
            else:
                raise ValueError(f"cgsd_layer must be 1, 2, or 3, got {cgsd_layer}")
            print(f"[Unet1] Using ChannelGate at Layer {cgsd_layer}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        """
        Args:
            x: input image [B, C, H, W]
            return_feat: if True, return (pred, fcgsd_str, fcgsd_sty) for CGSD loss computation

        Returns:
            if return_feat=False: (y1_pred, x5)
            if return_feat=True: (y1_pred, x5, fcgsd_str, fcgsd_sty)
        """
        # Initialize feature outputs
        fcgsd_str, fcgsd_sty = None, None

        # Encoder Layer 1
        x1 = self.convd1(x)

        if self.use_channel_gate and self.cgsd_layer == 1:
            x1_str, x1_sty = self.chan_gate(x1)
            x1_backbone = x1_str
        else:
            x1_backbone = x1

        # Extract CGSD features at Layer 1
        if self.use_channel_gate and self.cgsd_layer == 1 and return_feat:
            fcgsd_str = x1_backbone
            fcgsd_sty = x1_sty

        # Encoder Layer 2
        x2 = self.convd2(x1_backbone)

        if self.use_channel_gate and self.cgsd_layer == 2:
            x2_str, x2_sty = self.chan_gate(x2)
            x2_backbone = x2_str
        else:
            x2_backbone = x2

        # Extract CGSD features at Layer 2
        if self.use_channel_gate and self.cgsd_layer == 2 and return_feat:
            fcgsd_str = x2_backbone
            fcgsd_sty = x2_sty

        # Encoder Layer 3
        x3 = self.convd3(x2_backbone)

        if self.use_channel_gate and self.cgsd_layer == 3:
            x3_str, x3_sty = self.chan_gate(x3)
            x3_backbone = x3_str
        else:
            x3_backbone = x3

        # Extract CGSD features at Layer 3
        if self.use_channel_gate and self.cgsd_layer == 3 and return_feat:
            fcgsd_str = x3_backbone
            fcgsd_sty = x3_sty

        # Encoder Layer 4 and 5 (no CGSD applied here)
        x4 = self.convd4(x3_backbone)
        x5 = self.convd5(x4)

        # Decoder (use backbone features for skip connections to match forward features)
        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3_backbone)  # Use backbone (structure) features
        y2 = self.convu2(y3, x2_backbone)  # Use backbone (structure) features
        y1 = self.convu1(y2, x1_backbone)  # Use backbone (structure) features
        y1_pred = self.seg1(y1)

        if return_feat and self.use_channel_gate:
            return y1_pred, x5, fcgsd_str, fcgsd_sty
        else:
            return y1_pred, x5


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm='in', activation='leakyrelu'):
        super().__init__()
        padding = kernel_size // 2
        if norm == 'in_affine':
            norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            norm_layer = normalization(out_channels, norm)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True),
            norm_layer,
            nn.LeakyReLU(0.01, inplace=True) if activation == 'leakyrelu' else nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NNUNetConvBlock(nn.Module):
    """Two-convolution block used by the nnU-Net-style 2D backbone."""

    def __init__(self, in_channels, out_channels, stride=1, norm='in'):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=stride, norm=norm, activation='leakyrelu'),
            ConvNormAct(out_channels, out_channels, norm=norm, activation='leakyrelu'),
        )

    def forward(self, x):
        return self.block(x)


class NNUNetStyleUNet(nn.Module):
    """
    Standalone nnU-Net-style 2D encoder-decoder backbone.

    This intentionally does not import or reproduce the nnU-Net training
    pipeline. It only follows the common nnU-Net architectural choices:
    strided-convolution downsampling, transposed-convolution upsampling, deep
    encoder-decoder skips, InstanceNorm, and LeakyReLU.
    """

    supports_cgsd = False

    def __init__(self, c=3, num_classes=2, base_channels=32, norm='in_affine'):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            min(base_channels * 16, 320),
        ]
        self.feature_channels = channels[-1]

        self.enc1 = NNUNetConvBlock(c, channels[0], stride=1, norm=norm)
        self.enc2 = NNUNetConvBlock(channels[0], channels[1], stride=2, norm=norm)
        self.enc3 = NNUNetConvBlock(channels[1], channels[2], stride=2, norm=norm)
        self.enc4 = NNUNetConvBlock(channels[2], channels[3], stride=2, norm=norm)
        self.enc5 = NNUNetConvBlock(channels[3], channels[4], stride=2, norm=norm)

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.dec4 = NNUNetConvBlock(channels[3] * 2, channels[3], norm=norm)
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = NNUNetConvBlock(channels[2] * 2, channels[2], norm=norm)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = NNUNetConvBlock(channels[1] * 2, channels[1], norm=norm)
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = NNUNetConvBlock(channels[0] * 2, channels[0], norm=norm)

        self.seg = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _match_size(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x, return_feat=False):
        input_size = x.shape[-2:]
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        feat = self.enc5(x4)

        y4 = self._match_size(self.up4(feat), x4)
        y4 = self.dec4(torch.cat([y4, x4], dim=1))
        y3 = self._match_size(self.up3(y4), x3)
        y3 = self.dec3(torch.cat([y3, x3], dim=1))
        y2 = self._match_size(self.up2(y3), x2)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self._match_size(self.up1(y2), x1)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))

        logits = self.seg(y1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits, feat


def _window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, c)


def _window_reverse(windows, window_size, h, w, b):
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(n, n, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        return self.proj(out)


class SwinTransformerBlock2D(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if self.shift_size >= self.window_size:
            raise ValueError("shift_size must be smaller than window_size")
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, self.window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x_norm.shape[1], x_norm.shape[2]

        if self.shift_size > 0:
            shifted = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x_norm

        windows = _window_partition(shifted, self.window_size)
        attn_windows = self.attn(windows)
        shifted = _window_reverse(attn_windows, self.window_size, hp, wp, b)

        if self.shift_size > 0:
            x_attn = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn = shifted
        x_attn = x_attn[:, :h, :w, :].contiguous()

        x = shortcut + x_attn.permute(0, 3, 1, 2).contiguous()
        x_mlp = self.norm2(x.permute(0, 2, 3, 1).contiguous())
        x_mlp = self.mlp(x_mlp).permute(0, 3, 1, 2).contiguous()
        return x + x_mlp


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7):
        super().__init__()
        blocks = []
        for idx in range(depth):
            shift = 0 if idx % 2 == 0 else window_size // 2
            blocks.append(SwinTransformerBlock2D(dim, num_heads, window_size, shift))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class SwinUNetWrapper(nn.Module):
    """
    Lightweight Swin-UNet-style segmentation backbone with the DCON interface.

    The model keeps DCON's 2D input/output contract while replacing encoder
    blocks with window self-attention stages and a U-Net decoder.
    """

    supports_cgsd = False

    def __init__(self, c=3, num_classes=2, base_channels=24, window_size=7):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        high_channels = max(base_channels // 2, 8)
        self.feature_channels = channels[-1]

        self.high_res_stem = ConvNormAct(c, high_channels, norm='bn', activation='relu')
        self.patch_embed = ConvNormAct(high_channels, channels[0], stride=2, norm='bn', activation='relu')
        self.stage1 = SwinStage(channels[0], depth=2, num_heads=2, window_size=window_size)
        self.down2 = ConvNormAct(channels[0], channels[1], stride=2, norm='bn', activation='relu')
        self.stage2 = SwinStage(channels[1], depth=2, num_heads=4, window_size=window_size)
        self.down3 = ConvNormAct(channels[1], channels[2], stride=2, norm='bn', activation='relu')
        self.stage3 = SwinStage(channels[2], depth=2, num_heads=4, window_size=window_size)
        self.down4 = ConvNormAct(channels[2], channels[3], stride=2, norm='bn', activation='relu')
        self.stage4 = SwinStage(channels[3], depth=2, num_heads=8, window_size=window_size)

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = NNUNetConvBlock(channels[2] * 2, channels[2], norm='bn')
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = NNUNetConvBlock(channels[1] * 2, channels[1], norm='bn')
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = NNUNetConvBlock(channels[0] * 2, channels[0], norm='bn')
        self.up0 = nn.ConvTranspose2d(channels[0], high_channels, kernel_size=2, stride=2)
        self.dec0 = NNUNetConvBlock(high_channels * 2, high_channels, norm='bn')
        self.seg = nn.Conv2d(high_channels, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _match_size(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x, return_feat=False):
        input_size = x.shape[-2:]
        x0 = self.high_res_stem(x)
        x1 = self.stage1(self.patch_embed(x0))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        feat = self.stage4(self.down4(x3))

        y3 = self._match_size(self.up3(feat), x3)
        y3 = self.dec3(torch.cat([y3, x3], dim=1))
        y2 = self._match_size(self.up2(y3), x2)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self._match_size(self.up1(y2), x1)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        y0 = self._match_size(self.up0(y1), x0)
        y0 = self.dec0(torch.cat([y0, x0], dim=1))

        logits = self.seg(y0)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits, feat
