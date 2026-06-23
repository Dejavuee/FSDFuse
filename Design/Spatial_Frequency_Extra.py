import torch
import torch.nn as nn
import pynvml
import time
from tensorboardX import SummaryWriter
from .FrequencyDecouple import DualModalFrequencyDecoupler
from .Spatial_decouple import SpatialHighLowDecouplerWithVis as SpatialDecoupler
from .LiteExtra import FeatureExtractor


class Dual_Domain_Extra(nn.Module):
    def __init__(self):
        super().__init__()
        self.basicExtra = FeatureExtractor(in_channels=3,
                                           out_channels=8,
                                           num_blocks=4,
                                           num_heads=8,
                                           ffn_expansion_factor=2)
        self.Frequencydecouple = DualModalFrequencyDecoupler(in_channels=8,
                                                             out_channels=128,
                                                             groups=8,
                                                             sparsity_threshold=0.00001,
                                                             hidden_size_factor=1)
        self.SpatialdecoupleVis = SpatialDecoupler(in_channels=8,
                                                   out_channels=128,
                                                   num_scales=4,
                                                   drop_path_rate=0.1,
                                                   modality="Vis")
        self.SpatialdecoupleSAR = SpatialDecoupler(in_channels=8,
                                                   out_channels=128,
                                                   num_scales=4,
                                                   drop_path_rate=0.1,
                                                   modality="SAR")
        self.sar_adapter    = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.vis_adapter    = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.adapter_bn_sar = nn.BatchNorm2d(3)
        self.adapter_bn_vis = nn.BatchNorm2d(3)

    def forward(self, sar, vis):
        sar = self.adapter_bn_sar(self.sar_adapter(sar))
        vis = self.adapter_bn_vis(self.vis_adapter(vis))

        sar = self.basicExtra(sar)
        vis = self.basicExtra(vis)

        sar_spa_low, sar_spa_high = self.SpatialdecoupleSAR(sar)
        vis_spa_low, vis_spa_high = self.SpatialdecoupleVis(vis)


        (sar_freq_low, sar_freq_high, _sar_mask_l, _sar_mask_h,
         vis_freq_low, vis_freq_high, _vis_mask_l, _vis_mask_h) = \
            self.Frequencydecouple(sar, vis)

        return (sar_spa_low, sar_spa_high,
                vis_spa_low, vis_spa_high,
                sar_freq_low, sar_freq_high,
                vis_freq_low, vis_freq_high)


# ─────────────────────────────────────────────────────────────────────────────
def get_gpu_memory():
    pynvml.nvmlInit()
    handle   = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used     = mem_info.used  / 1024 / 1024
    total    = mem_info.total / 1024 / 1024
    pynvml.nvmlShutdown()
    return used, total

