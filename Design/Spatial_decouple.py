import torch
from torch.nn import ModuleList
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from torch import Tensor

from .Mamba_integration import LayerNorm2d, VSSBlock, PatchMerging2D

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ParallelConvs(nn.Module):

    def __init__(self, convs):
        super().__init__()
        self.convs = convs

    def forward(self, x):

        return torch.cat([conv(x) for conv in self.convs], dim=1)


class ChannelAttention(nn.Module):

    def __init__(self, module_channels: int, modality: str):
        super().__init__()
        self.modality = modality
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        act_fn = nn.LeakyReLU(0.1, inplace=True) if modality == "sar" else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Conv2d(module_channels, module_channels // 4, kernel_size=1, padding=0, bias=False),
            act_fn,
            nn.Conv2d(module_channels // 4, module_channels, kernel_size=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0 if modality == "sar" else 0.0)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.avg_pool(x)
        attn_weight = self.mlp(avg_out)
        return x * self.sigmoid(attn_weight)


class FusionBlockWithAttention(nn.Module):

    def __init__(self, channels, modality="vis"):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.channel_att = ChannelAttention(module_channels=channels, modality=modality)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            ChannelAttention(module_channels=channels, modality=modality)
        )

    def forward(self, x, skip):

        x = self.upsample(x)

        skip_att = self.channel_att(skip)

        fused_feat = self.fusion(x + skip_att)

        return fused_feat


class MultiScaleFusionFPN(nn.Module):


    def __init__(self, in_channels_list=[128, 64, 32, 16, 8], out_channels=128, modality="vis"):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.modality = modality


        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            )


        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.fusion_blocks.append(
                FusionBlockWithAttention(channels=out_channels, modality=modality)
            )

        self.final_adjust = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, features):

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        fused = laterals[0]
        for i in range(len(self.fusion_blocks)):
            skip_feat = laterals[i + 1]
            fused = self.fusion_blocks[i](fused, skip_feat)

        out = self.final_adjust(fused)

        return out


class SpatialHighLowDecouplerWithVis(nn.Module):

    def __init__(
            self,
            in_channels: int = 8,
            out_channels: int = 128,
            num_scales: int = 4,
            drop_path_rate: float = 0.1,
            modality: str = "SAR",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.modality = modality.lower()

        self._device = DEVICE

        assert self.modality in ["sar", "vis"], f"Modality must be 'SAR' or 'Vis', got {modality}"

        self.dim_up = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self._init_high_freq_branch()

        self._init_low_freq_branch(drop_path_rate)

        self.vis_cache: Dict[str, Any] = {}

        self.high_freq_fuse = MultiScaleFusionFPN(
            in_channels_list=[128, 64, 32, 16, 8],
            out_channels=out_channels,
            modality=self.modality
        )
        self.low_freq_fuse = MultiScaleFusionFPN(
            in_channels_list=[128, 64, 32, 16, 8],
            out_channels=out_channels,
            modality=self.modality
        )

        self.low_freq_norm = LayerNorm2d(self.out_channels)
        self.vis_cache: Dict[str, Any] = {}
        self.to(self._device)

    def _init_high_freq_branch(self) -> None:

        self.high_freq_scales = nn.ModuleList()
        act_layer = nn.LeakyReLU(0.1, inplace=True) if self.modality == "sar" else nn.GELU()

        current_in_channels = self.in_channels

        for i in range(self.num_scales):
            scale_out_channels = current_in_channels * 2

            conv3 = nn.Sequential(
                nn.Conv2d(current_in_channels, scale_out_channels // 2, 3, padding=1),
                nn.BatchNorm2d(scale_out_channels // 2) if self.modality != "sar" else nn.Identity(),
            )
            conv5 = nn.Sequential(
                nn.Conv2d(current_in_channels, scale_out_channels // 2, 5, padding=2),
                nn.BatchNorm2d(scale_out_channels // 2) if self.modality != "sar" else nn.Identity(),
            )

            layers = [
                ParallelConvs(ModuleList([conv3, conv5])),
                nn.Conv2d(scale_out_channels, scale_out_channels, 1, padding=0),
                act_layer,
                nn.Conv2d(scale_out_channels, scale_out_channels, 3, padding=1),
                nn.BatchNorm2d(scale_out_channels) if self.modality != "sar" else nn.Identity(),
                act_layer
            ]
            self.high_freq_scales.append(nn.Sequential(*layers))
            current_in_channels = scale_out_channels

    def _init_low_freq_branch(self, drop_path_rate: float) -> None:

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_scales)]

        initial_dim = self.in_channels * 2
        self.patch_embed = nn.Sequential(
            nn.Conv2d(self.in_channels, initial_dim, kernel_size=2, stride=2),
            LayerNorm2d(initial_dim)
        )

        self.low_freq_blocks = nn.ModuleList()
        self.low_freq_downsamples = nn.ModuleList()

        curr_channels = initial_dim

        for i in range(self.num_scales):

            block = VSSBlock(
                hidden_dim=curr_channels,
                drop_path=dpr[i],
                norm_layer=LayerNorm2d,
                channel_first=True,
                ssm_d_state=16,
                ssm_ratio=1.0,
                ssm_dt_rank="auto",
                ssm_act_layer=nn.SiLU,
                ssm_conv=3,
                use_checkpoint=False,
            )
            self.low_freq_blocks.append(block)

            if i < self.num_scales - 1:
                downsample = PatchMerging2D(
                    dim=curr_channels,
                    out_dim=curr_channels * 2,
                    norm_layer=LayerNorm2d,
                    channel_first=True
                )
                self.low_freq_downsamples.append(downsample)
                curr_channels *= 2

    def forward_low_freq_with_gate(self, x_up: Tensor, high_scale_feats: List[Tensor]) -> Tensor:
        low_scale_feats = []

        low_feat = self.patch_embed(x_up)

        for i in range(self.num_scales):
            high_feat_i = high_scale_feats[i]

            low_feat = self.low_freq_blocks[i](low_feat, gate_input=high_feat_i)

            low_scale_feats.append(low_feat)

            if i < self.num_scales - 1:
                low_feat = self.low_freq_downsamples[i](low_feat)

        fuse_list = [x_up] + low_scale_feats
        fuse_list.reverse()

        out = self.low_freq_fuse(fuse_list)
        return self.low_freq_norm(out)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_up = self.dim_up(x)

        high_scale_feats = []
        current_input = x_up
        for extractor in self.high_freq_scales:
            x_down = F.avg_pool2d(current_input, kernel_size=2, stride=2)
            feat = extractor(x_down)
            high_scale_feats.append(feat)
            current_input = feat

        high_scale_feats_fuse = [x_up] + high_scale_feats
        high_scale_feats_fuse.reverse()
        high_feat = self.high_freq_fuse(high_scale_feats_fuse)

        low_feat = self.forward_low_freq_with_gate(x_up, high_scale_feats)

        return low_feat, high_feat


    def to(self, device: Optional[Union[torch.device, str, int]] = None, **kwargs) -> 'SpatialHighLowDecouplerWithVis':

        if device is None:
            device = self._device
        else:
            device = torch.device(device)
        self._device = device
        return super().to(device, **kwargs)

    def train(self, mode: bool = True) -> 'SpatialHighLowDecouplerWithVis':
        super().train(mode)
        if mode:
            self.vis_cache.clear()
        return self

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"num_scales={self.num_scales}, modality={self.modality}, "
                f"device={self._device}")

