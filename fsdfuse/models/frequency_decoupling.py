"""Mean-conditioned dual-Sigmoid ADFD with independent complex band filters.

``sigma`` denotes the sigmoid transition temperature. Each modality maps the
two predicted Sigmoid values into its own ordered cutoff intervals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyDecoupler(nn.Module):
    def __init__(self, in_channels: int, groups: int = 8, smooth_type: str = 'sigmoid',
                 sigma: float = 1 / 7.5, sparsity_threshold: float = 1e-5,
                 hidden_size_factor: int = 1, r_low_const: float = 0.15,
                 r_low_max: float = 0.23, r_high_const: float = 0.39,
                 r_high_max: float = 0.45):
        super().__init__()
        if groups < 1 or in_channels % groups or sigma <= 0 or smooth_type != "sigmoid":
            raise ValueError("ADFD requires divisible groups, positive temperature and sigmoid masks")
        self.groups = groups
        self.smooth_type = smooth_type
        self.sigma = sigma
        self.sparsity_threshold = sparsity_threshold
        self.block_size = in_channels // groups
        self.r_low_const, self.r_low_max = r_low_const, r_low_max
        self.r_high_const, self.r_high_max = r_high_const, r_high_max
        if not (0 <= r_low_const < r_low_const + r_low_max <= r_high_const
                < r_high_const + r_high_max <= 1):
            raise ValueError("Cutoff intervals must be ordered within [0,1]")

        self.radius_w1 = nn.Parameter(0.05 * torch.randn(groups, self.block_size, self.block_size * hidden_size_factor))
        self.radius_b1 = nn.Parameter(torch.zeros(groups, self.block_size * hidden_size_factor))
        self.radius_w2 = nn.Parameter(0.05 * torch.randn(groups, self.block_size * hidden_size_factor, 2))
        self.radius_b2 = nn.Parameter(torch.zeros(groups, 2))



        def init_afno_weights():
            w1 = nn.Parameter(0.02 * torch.randn(2, groups, self.block_size, self.block_size * hidden_size_factor))
            b1 = nn.Parameter(torch.zeros(2, groups, self.block_size * hidden_size_factor))
            w2 = nn.Parameter(0.02 * torch.randn(2, groups, self.block_size * hidden_size_factor, self.block_size))
            b2 = nn.Parameter(torch.zeros(2, groups, self.block_size))
            gain = nn.Parameter(torch.ones(1, groups, self.block_size, 1, 1) * 0.2)
            return nn.ParameterDict({'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'gain': gain})

        self.afno_params_low  = init_afno_weights()
        self.afno_params_high = init_afno_weights()

        self.channel_group_attn = nn.Sequential(
            nn.Conv2d(in_channels, groups * 2, 1), nn.ReLU(),
            nn.Conv2d(groups * 2, groups, 1), nn.Softmax(dim=1)
        )
        self.intra_group_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=groups), nn.Sigmoid()
        )

        self._diag: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    def _apply_afno(self, fft_filtered, params):
        B, C, H, W = fft_filtered.shape
        x = fft_filtered.reshape(B, self.groups, self.block_size, H, W)

        w1_real, w1_imag = params['w1'][0], params['w1'][1]
        b1_r = params['b1'][0, :, :, None, None]
        b1_i = params['b1'][1, :, :, None, None]

        o1_r = (torch.einsum('bgchw,gci->bgihw', x.real, w1_real)
                - torch.einsum('bgchw,gci->bgihw', x.imag, w1_imag) + b1_r)
        o1_i = (torch.einsum('bgchw,gci->bgihw', x.real, w1_imag)
                + torch.einsum('bgchw,gci->bgihw', x.imag, w1_real) + b1_i)

        o1_r = F.softshrink(o1_r, lambd=self.sparsity_threshold)
        o1_i = F.softshrink(o1_i, lambd=self.sparsity_threshold)
        x_mid = torch.view_as_complex(torch.stack([o1_r, o1_i], dim=-1))

        w2_real, w2_imag = params['w2'][0], params['w2'][1]
        b2_r = params['b2'][0, :, :, None, None]
        b2_i = params['b2'][1, :, :, None, None]

        o2_r = (torch.einsum('bgihw,gic->bgchw', x_mid.real, w2_real)
                - torch.einsum('bgihw,gic->bgchw', x_mid.imag, w2_imag) + b2_r)
        o2_i = (torch.einsum('bgihw,gic->bgchw', x_mid.real, w2_imag)
                + torch.einsum('bgihw,gic->bgchw', x_mid.imag, w2_real) + b2_i)

        o2_r = torch.tanh(o2_r) * 0.15
        o2_i = torch.tanh(o2_i) * 0.15
        res = torch.view_as_complex(torch.stack([o2_r, o2_i], dim=-1))

        branch_gate = torch.sigmoid(
            fft_filtered.reshape(B, self.groups, self.block_size, H, W).abs())
        safe_gain = torch.clamp(params['gain'], 0.0, 0.5)
        out = (res * branch_gate + x * 0.05) * safe_gain

        return out.reshape(B, C, H, W)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        device = x.device

        channel_feat = F.adaptive_avg_pool2d(x, 1).reshape(B, self.groups, self.block_size)

        self._channel_feat_for_loss = channel_feat.detach()

        radius_hid = F.relu(
            torch.einsum('bgc,gci->bgi', channel_feat, self.radius_w1) + self.radius_b1)

        pre_sigmoid = (
            torch.einsum('bgi,gio->bgo', radius_hid, self.radius_w2) + self.radius_b2)
        cutoffs = torch.sigmoid(pre_sigmoid)
        r_low = self.r_low_const + self.r_low_max * cutoffs[..., 0]
        r_high = self.r_high_const + self.r_high_max * cutoffs[..., 1]

        self._r_low  = r_low.detach()
        self._r_high = r_high.detach()
        self._diag = {
            'radius_logits': pre_sigmoid.detach(),                 # (B, G, 2)
            'channel_feat_mean':  channel_feat.mean(dim=-1).detach(),  # (B, G)
        }
        # ────────────────────────────────────────────────────────────────────

        # ── 2. Mask generation ───────────────────────────────────────────────
        fy = torch.linspace(-1, 1, H, device=device)
        fx = torch.linspace(-1, 1, W, device=device) * (W / H)
        y, x_c = torch.meshgrid(fy, fx, indexing='ij')
        dist = torch.sqrt(x_c ** 2 + y ** 2).unsqueeze(0).unsqueeze(0) / math.sqrt(1 + (W / H)**2)


        r_l_exp = (r_low.view(B, self.groups, 1, 1, 1)
                   .expand(B, self.groups, self.block_size, H, W)
                   .reshape(B, C, H, W))
        r_h_exp = (r_high.view(B, self.groups, 1, 1, 1)
                   .expand(B, self.groups, self.block_size, H, W)
                   .reshape(B, C, H, W))


        if self.smooth_type == 'sigmoid':
            mask_l = torch.sigmoid((r_l_exp - dist) / self.sigma)
            mask_h = torch.sigmoid((dist - r_h_exp) / self.sigma)

        # ── 3. Attention gating ──────────────────────────────────────────────
        group_attn     = self.channel_group_attn(F.adaptive_avg_pool2d(x, 1))
        group_attn_exp = (group_attn.unsqueeze(2)
                          .expand(B, self.groups, self.block_size, 1, 1)
                          .reshape(B, C, 1, 1))
        intra_attn  = self.intra_group_attn(F.adaptive_avg_pool2d(x, 1))
        # Bounded affine modulation: psi(a) = 0.5 + 0.5a, with a in [0, 1].
        final_attn = 0.5 + 0.5 * (group_attn_exp * intra_attn)

        mask_l = (mask_l * final_attn).clamp(min=1e-6)
        mask_h = (mask_h * final_attn).clamp(min=1e-6)

        # ── 4. Frequency filtering ───────────────────────────────────────────
        with torch.amp.autocast(x.device.type, enabled=False):
            x_fft  = torch.fft.fftshift(torch.fft.fft2(x.float(), norm='ortho'), dim=(-2, -1))
            low_f  = self._apply_afno(x_fft * mask_l.float(), self.afno_params_low)
            high_f = self._apply_afno(x_fft * mask_h.float(), self.afno_params_high)
            low_out  = torch.real(torch.fft.ifft2(
                torch.fft.ifftshift(low_f,  dim=(-2, -1)), s=(H, W), norm='ortho'))
            high_out = torch.real(torch.fft.ifft2(
                torch.fft.ifftshift(high_f, dim=(-2, -1)), s=(H, W), norm='ortho'))

        return low_out.to(x.dtype), high_out.to(x.dtype), mask_l, mask_h





# ─────────────────────────────────────────────────────────────────────────────
class DualModalFrequencyDecoupler(nn.Module):
    """Dual-modality frequency decoupling (ADFD module)."""

    def __init__(self,
                 in_channels: int = 8,
                 out_channels: int = 128,
                 sar_soomtype: str = 'sigmoid',
                 sar_soomsigma: float = 1 / 7.5,
                 vis_soomtype: str = 'sigmoid',
                 vis_soomsigma: float = 1 / 7.5,
                 groups: int = 8,
                 sparsity_threshold: float = 0.00001,
                 hidden_size_factor: int = 1):
        super().__init__()

        self.sar_decoupler = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            FrequencyDecoupler(
                in_channels=out_channels,
                smooth_type=sar_soomtype,
                sigma=sar_soomsigma,
                groups=groups,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
            )
        )

        self.vis_decoupler = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            FrequencyDecoupler(
                in_channels=out_channels,
                smooth_type=vis_soomtype,
                sigma=vis_soomsigma,
                r_low_const=0.20,
                r_low_max=0.20,
                r_high_const=0.40,
                r_high_max=0.40,
                groups=groups,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
            )
        )

    def forward(self, sar: torch.Tensor, visible: torch.Tensor):
        sar_low,  sar_high,  sar_mask_l,  sar_mask_h  = self.sar_decoupler(sar)
        vis_low,  vis_high,  vis_mask_l,  vis_mask_h  = self.vis_decoupler(visible)
        return (sar_low, sar_high, sar_mask_l, sar_mask_h,
                vis_low, vis_high, vis_mask_l, vis_mask_h)
