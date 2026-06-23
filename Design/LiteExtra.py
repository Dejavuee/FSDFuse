import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def to_var(x, requires_grad=True):
    """Wrap a tensor in a Variable, moving it to CUDA if available."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def to_3d(x):
    """Reshape (B, C, H, W) → (B, H*W, C)."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Reshape (B, H*W, C) → (B, C, H, W)."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ──────────────────────────────────────────────────────────────────────────────
#  Meta-Learning Compatible Layers
# ──────────────────────────────────────────────────────────────────────────────

class MetaDepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise-separable convolution with optional meta-learning forward pass.

    When ``meta=True`` the stored leaf parameters (``weight_dw``, ``weight_pw``,
    and their biases) are used directly via ``F.conv2d``, allowing outer-loop
    gradient updates to flow through them.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.weight_dw = to_var(self.depthwise.weight.data, requires_grad=True)
        self.weight_pw = to_var(self.pointwise.weight.data, requires_grad=True)
        self.bias_dw   = to_var(self.depthwise.bias.data,  requires_grad=True) \
                         if self.depthwise.bias is not None else None
        self.bias_pw   = to_var(self.pointwise.bias.data,  requires_grad=True) \
                         if self.pointwise.bias is not None else None

    def named_leaves(self):
        return [
            ('weight_dw', self.weight_dw), ('bias_dw', self.bias_dw),
            ('weight_pw', self.weight_pw), ('bias_pw', self.bias_pw),
        ]

    def _to_device(self, x):
        for v in [self.weight_dw, self.bias_dw, self.weight_pw, self.bias_pw]:
            if v is not None and x.is_cuda and not v.is_cuda:
                v.data = v.data.cuda()

    def forward(self, x, meta=False):
        if meta:
            self._to_device(x)
            x = F.conv2d(x, self.weight_dw, self.bias_dw,
                         self.depthwise.stride, self.depthwise.padding,
                         self.depthwise.dilation, self.depthwise.groups)
            x = F.conv2d(x, self.weight_pw, self.bias_pw,
                         self.pointwise.stride, self.pointwise.padding,
                         self.pointwise.dilation, self.pointwise.groups)
        else:
            x = self.depthwise(x)
            x = self.pointwise(x)
        return x


class MetaReLU(nn.Module):
    """ReLU activation compatible with the meta-learning forward interface."""

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x, inplace=self.inplace)


# ──────────────────────────────────────────────────────────────────────────────
#  Layer Normalisation
# ──────────────────────────────────────────────────────────────────────────────

class BiasFree_LayerNorm(nn.Module):
    """Layer normalisation without a bias term."""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape    = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight         = nn.Parameter(torch.ones(normalized_shape))
        self.weight_meta    = to_var(self.weight.data, requires_grad=True)
        self.normalized_shape = normalized_shape

    def named_leaves(self):
        return [('weight', self.weight)]

    def _to_device(self, x):
        if x.is_cuda:
            self.weight_meta.cuda()

    def forward(self, x, meta=False):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        w = self.weight_meta if meta else self.weight
        if meta:
            self._to_device(x)
        return x / torch.sqrt(sigma + 1e-5) * w


class WithBias_LayerNorm(nn.Module):
    """Layer normalisation with a learnable bias term."""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape    = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight         = nn.Parameter(torch.ones(normalized_shape))
        self.bias           = nn.Parameter(torch.zeros(normalized_shape))
        self.weight_meta    = to_var(self.weight.data, requires_grad=True)
        self.bias_meta      = to_var(self.bias.data,   requires_grad=True)
        self.normalized_shape = normalized_shape

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def _to_device(self, x):
        for v in [self.weight_meta, self.bias_meta]:
            if x.is_cuda:
                v.cuda()

    def forward(self, x, meta=False):
        mu    = x.mean(-1, keepdim=True)
        sigma = x.var(-1,  keepdim=True, unbiased=False)
        if meta:
            self._to_device(x)
            return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight_meta + self.bias_meta
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Spatial layer normalisation — reshapes to/from (B, H*W, C) internally."""

    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = (BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree'
                     else WithBias_LayerNorm(dim))

    def forward(self, x, meta=False):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x), meta=meta), h, w)


# ──────────────────────────────────────────────────────────────────────────────
#  Lite Transformer Building Blocks
# ──────────────────────────────────────────────────────────────────────────────

class LiteAttention(nn.Module):
    """
    Channel-wise self-attention using depthwise-separable QKV projections.

    Computes attention in the channel dimension rather than the spatial
    dimension, keeping the complexity linear in sequence length.
    """

    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv  = MetaDepthwiseSeparableConv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.proj = MetaDepthwiseSeparableConv2d(dim, dim,     kernel_size=1, bias=qkv_bias)

    def forward(self, x, meta=False):
        b, c, h, w = x.shape
        q, k, v    = self.qkv(x, meta=meta).chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w',
                         head=self.num_heads, h=h, w=w)
        return self.proj(out, meta=meta)


class LiteMlp(nn.Module):
    """Gated MLP (GLU variant) using depthwise-separable convolutions."""

    def __init__(self, in_features, hidden_features=None, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden = int(in_features * ffn_expansion_factor)
        self.project_in  = MetaDepthwiseSeparableConv2d(in_features, hidden * 2,
                                                         kernel_size=1, bias=bias)
        self.dwconv      = MetaDepthwiseSeparableConv2d(hidden * 2, hidden * 2,
                                                         kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = MetaDepthwiseSeparableConv2d(hidden, in_features,
                                                         kernel_size=1, bias=bias)

    def forward(self, x, meta=False):
        x       = self.project_in(x, meta=meta)
        x1, x2  = self.dwconv(x, meta=meta).chunk(2, dim=1)
        x       = F.gelu(x1) * x2
        return self.project_out(x, meta=meta)


class LiteFeatureExtractionBlock(nn.Module):
    """Pre-norm Transformer block: LayerNorm → Attention + LayerNorm → MLP."""

    def __init__(self, dim, num_heads, ffn_expansion_factor=1, qkv_bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn  = LiteAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp   = LiteMlp(in_features=dim, ffn_expansion_factor=ffn_expansion_factor)
        self.act   = MetaReLU(inplace=True)

    def forward(self, x, meta=False):
        x = x + self.attn(self.norm1(x), meta=meta)
        x = x + self.mlp(self.norm2(x),  meta=meta)
        return self.act(x)


# ──────────────────────────────────────────────────────────────────────────────
#  FeatureExtractor
# ──────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Shared shallow encoder for SAR and visible-light inputs.

    A single 1×1 projection followed by stacked ``LiteFeatureExtractionBlock``
    modules processes both modalities with shared weights, producing a pair of
    feature maps suitable for downstream decoupling.

    Args:
        in_channels:          Number of input channels (1 for grayscale).
        out_channels:         Feature channel width throughout the encoder.
        num_blocks:           Number of stacked transformer blocks.
        num_heads:            Attention heads per block.
        ffn_expansion_factor: Hidden-dimension multiplier in the MLP.
    """

    def __init__(self, in_channels=1, out_channels=8,
                 num_blocks=3, num_heads=4, ffn_expansion_factor=2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.blocks  = nn.Sequential(*[
            LiteFeatureExtractionBlock(dim=out_channels, num_heads=num_heads,
                                       ffn_expansion_factor=ffn_expansion_factor)
            for _ in range(num_blocks)
        ])

    def forward(self, sar, vis, meta=False):
        sar = self.conv_in(sar)
        vis = self.conv_in(vis)
        for block in self.blocks:
            sar = block(sar, meta=meta)
            vis = block(vis, meta=meta)
        return sar, vis


# ──────────────────────────────────────────────────────────────────────────────
#  FeatureDecoder
# ──────────────────────────────────────────────────────────────────────────────

class FeatureDecoder(nn.Module):
    """
    Self-supervised reconstruction decoder for Stage 1 training.

    Accepts 2 or 4 decoupled feature tensors, optionally adapts the channel
    count, compresses them, refines with stacked transformer blocks, and
    reconstructs the source image.

    Args:
        in_channels:          Channel width of each input feature map.
        out_channels:         Number of output image channels.
        num_blocks:           Number of stacked transformer blocks.
        num_heads:            Attention heads per block.
        ffn_expansion_factor: Hidden-dimension multiplier in the MLP.
        use_checkpoint:       Use gradient checkpointing to reduce memory usage.
    """

    def __init__(self, in_channels=256, out_channels=3,
                 num_blocks=2, num_heads=4, ffn_expansion_factor=2,
                 use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels    = in_channels
        self.out_channels   = out_channels

        self.adapt_4c_to_2c  = nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1, bias=False)
        self.reduce_channel  = nn.Conv2d(in_channels * 2, in_channels,     kernel_size=1, bias=False)

        self.blocks = nn.ModuleList([
            LiteFeatureExtractionBlock(dim=in_channels, num_heads=num_heads,
                                       ffn_expansion_factor=ffn_expansion_factor)
            for _ in range(num_blocks)
        ])

        self.output = nn.Sequential(
            nn.Conv2d(in_channels,      in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels // 2, out_channels,     kernel_size=3, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def _checkpoint_wrapper(self, block, x, meta):
        return block(x, meta=meta)

    def forward(self, *inputs, Vis=None, meta=False):
        if len(inputs) == 2:
            x = torch.cat(inputs, dim=1)
        elif len(inputs) == 4:
            x = torch.cat(inputs, dim=1)
        else:
            raise ValueError("Expected 2 or 4 input feature tensors.")

        if x.shape[1] == self.in_channels * 4:
            x = self.adapt_4c_to_2c(x)

        x = self.reduce_channel(x)

        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(self._checkpoint_wrapper, block, x, meta,
                                use_reentrant=False)
            else:
                x = block(x, meta=meta)

        x = self.output(x)

        if Vis is not None:
            x = x + F.interpolate(Vis, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.sigmoid(x)


# ──────────────────────────────────────────────────────────────────────────────
#  FusionReconstructorV2
# ──────────────────────────────────────────────────────────────────────────────

class FusionReconstructorV2(nn.Module):
    """
    Dual-path fusion reconstructor — decodes fused features directly to an image.

    Two parallel convolutional paths extract complementary information from the
    low- and high-frequency fused features, then a single
    ``LiteFeatureExtractionBlock`` bridge captures cross-spatial dependencies
    before the output head produces the final grayscale image.

    Args:
        feat_dim:             Channel width of the input low/high feature maps.
        out_channels:         Number of output image channels (typically 1).
        num_heads:            Attention heads in the bridge block.
        ffn_expansion_factor: MLP expansion factor in the bridge block.
        use_checkpoint:       Use gradient checkpointing during training.
    """

    def __init__(self, feat_dim=128, out_channels=1,
                 num_heads=4, ffn_expansion_factor=2,
                 use_checkpoint=True):
        super().__init__()
        self.feat_dim        = feat_dim
        self.out_channels    = out_channels
        self.use_checkpoint  = use_checkpoint
        mid = feat_dim // 2  # 64

        # Low-frequency path: global brightness and large-scale structure
        self.low_path = nn.Sequential(
            nn.Conv2d(feat_dim, mid,      3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid // 2,      3, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )

        # High-frequency path: edges and texture details
        self.high_path = nn.Sequential(
            nn.Conv2d(feat_dim, mid,      3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid // 2,      5, padding=2, bias=False),
            nn.LeakyReLU(0.2),
        )

        # Bridge: one transformer block to capture cross-position dependencies
        self.bridge = LiteFeatureExtractionBlock(
            dim=mid, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(mid,      mid // 2,    3, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid // 2, out_channels, 3, padding=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for all convolutions; near-zero init for the output layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last_conv = self.output_head[-1]
        nn.init.normal_(last_conv.weight, std=0.01)
        nn.init.zeros_(last_conv.bias)

    def _bridge_wrapper(self, x, meta=False):
        return self.bridge(x, meta=meta)

    def forward(self, low_feat, high_feat, vis_orig=None, sar_orig=None):
        """
        Args:
            low_feat:  Low-frequency fused features  (B, feat_dim, H, W).
            high_feat: High-frequency fused features (B, feat_dim, H, W).
            vis_orig:  Unused — retained for API compatibility.
            sar_orig:  Unused — retained for API compatibility.

        Returns:
            Fused image tensor of shape (B, out_channels, H, W) in [0, 1].
        """
        low_out  = self.low_path(low_feat)               # (B, mid/2, H, W)
        high_out = self.high_path(high_feat)              # (B, mid/2, H, W)
        combined = torch.cat([low_out, high_out], dim=1)  # (B, mid,   H, W)

        if self.training and self.use_checkpoint:
            combined = checkpoint(self._bridge_wrapper, combined, False,
                                  use_reentrant=False)
        else:
            combined = self.bridge(combined)

        return torch.sigmoid(self.output_head(combined))