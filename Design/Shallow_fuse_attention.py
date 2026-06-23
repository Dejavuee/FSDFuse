
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):


    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=[2, 3])                   # [B, C]
        max_pool = x.amax(dim=[2, 3])                   # [B, C]
        att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))  # [B, C]
        return x * att.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):


    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        max_pool = x.amax(dim=1, keepdim=True)    # [B, 1, H, W]
        att = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * att


class SpaFreqFuseBlockV2(nn.Module):

    def __init__(self, dim=128, reduction=8):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
        )
        self.channel_att = ChannelAttention(dim, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=7)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, spa_feat, freq_feat):
        identity = (spa_feat + freq_feat) * 0.5
        concat = torch.cat([spa_feat, freq_feat], dim=1)
        out = self.fuse(concat)
        out = self.channel_att(out)
        out = self.spatial_att(out)
        return identity + self.res_scale * out

class CrossModalGateLowFreqV2(nn.Module):


    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, dim * 2, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, 2, 1),
            nn.Softmax(dim=1)
        )

        self.residual_branch = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.Tanh()
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, vis_feat, sar_feat):
        B, C, H, W = vis_feat.shape
        concat = torch.cat([vis_feat, sar_feat], dim=1)

        ch_weight = self.channel_att(concat)
        vis_cal = vis_feat * ch_weight[:, :C]
        sar_cal = sar_feat * ch_weight[:, C:]
        concat_cal = torch.cat([vis_cal, sar_cal], dim=1)
        spatial_w = self.spatial_gate(concat_cal)
        w_vis, w_sar = spatial_w[:, 0:1], spatial_w[:, 1:2]
        soft_fused = w_vis * vis_cal + w_sar * sar_cal

        delta = self.residual_branch(concat)
        clamped_scale = self.residual_scale.clamp(-0.3, 0.3)
        fused = soft_fused + clamped_scale * delta

        return fused



class AdditiveHighFreqGate(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.vis_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Sigmoid()
        )
        self.sar_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Sigmoid()
        )
        self.recalibrate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, vis_feat, sar_feat):
        concat = torch.cat([vis_feat, sar_feat], dim=1)
        w_vis = self.vis_gate(concat)
        w_sar = self.sar_gate(concat)
        fused = w_vis * vis_feat + w_sar * sar_feat
        return self.recalibrate(fused)