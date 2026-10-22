"""FSDFuse network and modality-specific reconstruction decoders."""
from torch import nn
from .feature_modules import FeatureExtractor, FeatureDecoder, FusionReconstructor
from .frequency_decoupling import DualModalFrequencyDecoupler
from .spatial_decoupling import DualBranchSpatialDecoupler
from .fusion_modules import LowFrequencyFusionGate, HighFrequencyFusionGate, SpatialFrequencyBandCombiner

class FSDFuse(nn.Module):
    def __init__(self, sar_temperature=1 / 7.5, visible_temperature=1 / 7.5):
        super().__init__()
        self.model_config = {"sar_temperature": sar_temperature, "visible_temperature": visible_temperature}
        self.feature_extractor = FeatureExtractor()
        self.visible_spatial_encoder = DualBranchSpatialDecoupler(modality="Vis")
        self.sar_spatial_encoder = DualBranchSpatialDecoupler(modality="SAR")
        self.frequency_encoder = DualModalFrequencyDecoupler(
            sar_soomsigma=sar_temperature, vis_soomsigma=visible_temperature)
        self.spatial_low_fusion = LowFrequencyFusionGate(dim=128, reduction=16)
        self.frequency_low_fusion = LowFrequencyFusionGate(dim=128, reduction=16)
        self.spatial_high_fusion = HighFrequencyFusionGate(dim=128)
        self.frequency_high_fusion = HighFrequencyFusionGate(dim=128)
        self.low_band_combiner = SpatialFrequencyBandCombiner(dim=128)
        self.high_band_combiner = SpatialFrequencyBandCombiner(dim=128)
        self.reconstructor = FusionReconstructor(feat_dim=128, out_channels=1)

    def encode(self, visible, sar):
        if visible.shape != sar.shape or visible.ndim != 4 or visible.shape[1] != 1:
            raise ValueError("Expected matching Bx1xHxW visible Y and SAR tensors")
        if any(n % 16 for n in visible.shape[-2:]):
            raise ValueError("Height and width must be divisible by 16")
        sf, vf = self.feature_extractor(sar, visible)
        vl, vh = self.visible_spatial_encoder(vf)
        sl, sh = self.sar_spatial_encoder(sf)
        fls, fhs, _, _, flv, fhv, _, _ = self.frequency_encoder(sf, vf)
        return {"vis": (vl, vh, flv, fhv), "sar": (sl, sh, fls, fhs)}

    def fuse_features(self, features, visible, sar, detach_encoder=False):
        v, s = features["vis"], features["sar"]
        if detach_encoder:
            v, s = tuple(x.detach() for x in v), tuple(x.detach() for x in s)
        low = self.low_band_combiner(self.spatial_low_fusion(v[0], s[0]),
                                     self.frequency_low_fusion(v[2], s[2]))
        high = self.high_band_combiner(self.spatial_high_fusion(v[1], s[1]),
                                      self.frequency_high_fusion(v[3], s[3]))
        return self.reconstructor(low, high, visible, sar).clamp(0, 1)

    def forward(self, visible, sar):
        return self.fuse_features(self.encode(visible, sar), visible, sar)

class ReconstructionDecoders(nn.Module):
    """Two modality-specific decoders used in Stage I only."""
    def __init__(self):
        super().__init__()
        self.vis = FeatureDecoder(in_channels=128, out_channels=1)
        self.sar = FeatureDecoder(in_channels=128, out_channels=1)

    def forward(self, features):
        outputs = {}
        for key in ("vis", "sar"):
            low, high, flow, fhigh = features[key]
            outputs[key] = getattr(self, key)(high, low, fhigh, flow)
        return outputs
