import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from typing import Optional, Any, Union, Literal, Callable
from functools import partial
from timm.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit




class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class MambaFineGrainedGate(nn.Module):

    def __init__(
            self,
            d_inner: int,
            k_group: int = 4,
            activation: str = "sigmoid",
            init_type: str = "uniform",
            channel_first: bool = True,
            debug: bool = False,
            **kwargs
    ):

        super().__init__()

        self.d_inner = d_inner
        self.k_group = k_group
        self.channel_first = channel_first
        self.debug = debug

        self.gate_proj = nn.Parameter(torch.randn(k_group, d_inner, d_inner))
        self.gate_bias = nn.Parameter(torch.zeros(k_group, d_inner))

        if activation == "sigmoid":
            self.gate_act = nn.Sigmoid()
        elif activation == "silu":
            self.gate_act = nn.SiLU()
        elif activation == "tanh":
            self.gate_act = nn.Tanh()
        else:
            raise ValueError(f"Unknow: {activation}")

        self._init_weights(init_type)

        if debug:
            self.register_buffer('gate_mean', torch.zeros(1))
            self.register_buffer('gate_std', torch.zeros(1))
            self.register_buffer('gate_sparsity', torch.zeros(1))

    def _init_weights(self, init_type: str):

        with torch.no_grad():
            if init_type == "uniform":
                nn.init.uniform_(self.gate_proj, -0.1, 0.1)
            elif init_type == "normal":
                nn.init.normal_(self.gate_proj, std=0.02)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.gate_proj)
            elif init_type == "zero":
                nn.init.zeros_(self.gate_proj)
            else:
                raise ValueError(f"Unknow: {init_type}")

            if self.gate_bias is not None:
                nn.init.constant_(self.gate_bias, 0.1)

    def forward(
            self,
            gate_input: torch.Tensor,
            scan_output: torch.Tensor,
            return_gate: bool = False
    ):
        B, D, H, W, K = self._get_dimensions(gate_input, scan_output)
        gate_input_expanded = gate_input.unsqueeze(1).repeat(1, K, 1, 1, 1)
        L = H * W
        gate_input_flat = gate_input_expanded.view(B, K, D, L).transpose(2, 3)
        gate_proj_used = self.gate_proj.to(gate_input_flat.device, gate_input_flat.dtype)
        gate_bias_used = self.gate_bias.to(gate_input_flat.device, gate_input_flat.dtype)

        score = torch.einsum("bklc, kcd -> bkld", gate_input_flat, gate_proj_used)

        if self.gate_bias is not None:
            score = score + gate_bias_used.view(1, K, 1, D)

        gate_scores = self.gate_act(score)

        gate_scores = gate_scores.transpose(2, 3).view(B, K, D, H, W)

        gated_output = scan_output * gate_scores

        if return_gate:
            return gated_output, gate_scores
        else:
            return gated_output, None

    def _get_dimensions(self, gate_input: torch.Tensor, scan_output: torch.Tensor):
        if self.channel_first:
            B, D, H, W = gate_input.shape
            K = scan_output.shape[1]
        else:
            B, H, W, D = gate_input.shape
            K = scan_output.shape[1]
        return B, D, H, W, K



class MambaSSMInit:

    @staticmethod
    def init_dt_proj(
            dt_rank: int,
            d_inner: int,
            dt_scale: float = 1.0,
            dt_init: str = "random",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> tuple:

        factory_kwargs = {"device": device, "dtype": dtype}
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unknown_dt_init: {dt_init}")

        u = torch.rand(d_inner, device=device, dtype=dtype if dtype != torch.float16 else torch.float32)
        dt = torch.exp(
            u * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_weight_decay = True

        return dt_proj.weight, dt_proj.bias

    @staticmethod
    def init_A_logs(
            d_state: int,
            d_inner: int,
            k_group: int = 4,
            device: Optional[torch.device] = None,
    ) -> nn.Parameter:
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A.view(1, -1).repeat(d_inner, 1).contiguous()
        A_logs = torch.log(A)
        if k_group > 1:
            A_logs = A_logs.unsqueeze(0).repeat(k_group, 1, 1).contiguous()
            A_logs = A_logs.flatten(0, 1)

        A_logs = nn.Parameter(A_logs)
        A_logs._no_weight_decay = True

        return A_logs

    @staticmethod
    def init_Ds(
            d_inner: int,
            k_group: int = 4,
            device: Optional[torch.device] = None,
    ) -> nn.Parameter:

        Ds = torch.ones(k_group * d_inner, device=device, dtype=torch.float32)
        Ds = nn.Parameter(Ds)
        Ds._no_weight_decay = True

        return Ds

    @classmethod
    def init_all_params(
            cls,
            d_state: int,
            dt_rank: int,
            d_inner: int,
            k_group: int = 4,
            dt_scale: float = 1.0,
            dt_init: str = "random",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> tuple:

        dt_weights = []
        dt_biases = []

        for _ in range(k_group):
            weight, bias = cls.init_dt_proj(
                dt_rank, d_inner, dt_scale, dt_init,
                dt_min, dt_max, dt_init_floor, device, dtype
            )
            dt_weights.append(weight)
            dt_biases.append(bias)

        dt_projs_weight = nn.Parameter(torch.stack(dt_weights, dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack(dt_biases, dim=0))  # (K, inner)

        A_logs = cls.init_A_logs(d_state, d_inner, k_group, device)
        Ds = cls.init_Ds(d_inner, k_group, device)

        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class UniversalSS2D(nn.Module):

    def __init__(
            self,

            d_model: int,
            d_model2: Optional[int] = None,

            mode: Literal["self", "gate", "cross"] = "self",

            d_state: int = 16,
            ssm_ratio: float = 1.0,
            dt_rank: Union[int, str] = "auto",
            dt_scale: float = 1.0,
            dt_init: str = "random",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,

            use_gate: bool = True,
            gate_type: str = "elementwise",

            k_group: int = 4,
            scan_mode: str = "cross2d",

            d_conv: int = 3,
            conv_bias: bool = True,

            dropout: float = 0.0,
            bias: bool = False,
            channel_first: bool = True,
            act_layer: nn.Module = nn.SiLU,

            fusion_type: Literal["add", "concat", "cross_attn"] = "concat",
            **kwargs,
    ):
        super().__init__()

        # 保存配置
        self.mode = mode
        self.use_gate = use_gate
        self.channel_first = channel_first
        self.k_group = k_group
        self.scan_mode = scan_mode
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        Linear = Linear2d if channel_first else nn.Linear
        self.disable_z = False
        self.disable_z_act = False
        self.disable_force32 = False
        self.oact = True

        self.d_model1 = d_model
        self.d_inner1 = int(ssm_ratio * d_model)

        self.is_cross = mode == "cross"
        if self.is_cross:
            self.d_model2 = d_model2 if d_model2 is not None else d_model
            self.d_inner2 = int(ssm_ratio * self.d_model2)
            scan_inner = self.d_inner2
            param_inner = self.d_inner1

        else:
            self.d_model2 = d_model
            self.d_inner2 = self.d_inner1
            scan_inner = self.d_inner1
            param_inner = self.d_inner1

        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        factory_kwargs = {
            "device": kwargs.get("device", None),
            "dtype": kwargs.get("dtype", None),
        }

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = \
            MambaSSMInit.init_all_params(
                d_state=d_state,
                dt_rank=self.dt_rank,
                d_inner=scan_inner,
                k_group=self.k_group,
                dt_scale=dt_scale,
                dt_init=dt_init,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                **factory_kwargs
            )

        self.act = act_layer()
        self.out_act = nn.GELU() if self.oact else nn.Identity()

        if self.is_cross:
            self.in_proj_x = Linear(self.d_model1, self.d_inner1, bias=bias, **factory_kwargs)
            self.in_proj_y = Linear(self.d_model2, self.d_inner2, bias=bias, **factory_kwargs)

        else:
            proj_out_dim = self.d_inner1 if self.disable_z else (self.d_inner1 * 2)
            self.in_proj = Linear(self.d_model1, proj_out_dim, bias=bias, **factory_kwargs)

        x_proj_inner = param_inner if self.is_cross else scan_inner
        self.x_proj = [
            nn.Linear(x_proj_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.with_dconv = d_conv > 0
        if self.with_dconv:
            if self.is_cross:
                self.conv2d_x = nn.Conv2d(
                    self.d_inner1, self.d_inner1,
                    kernel_size=d_conv, padding=(d_conv - 1) // 2,
                    groups=self.d_inner1, bias=conv_bias,
                    **factory_kwargs
                )
                self.conv2d_y = nn.Conv2d(
                    self.d_inner2, self.d_inner2,
                    kernel_size=d_conv, padding=(d_conv - 1) // 2,
                    groups=self.d_inner2, bias=conv_bias,
                    **factory_kwargs
                )

            else:
                self.conv2d = nn.Conv2d(
                    self.d_inner1, self.d_inner1,
                    kernel_size=d_conv, padding=(d_conv - 1) // 2,
                    groups=self.d_inner1, bias=conv_bias,
                    **factory_kwargs
                )

        self.forward_ss2d = partial(
            self.forward_corev2,
            force_fp32=(not self.disable_force32),
            selective_scan_backend="core"
        )

        if self.is_cross:
            self.out_proj = Linear(self.d_inner2, self.d_model2, bias=bias, **factory_kwargs)
        else:
            self.out_proj = Linear(self.d_inner1, self.d_model1, bias=bias, **factory_kwargs)

        if use_gate:
            gate_inner = param_inner if self.is_cross else scan_inner
            self.gate_module = MambaFineGrainedGate(
                d_inner=gate_inner,
                k_group=k_group,
                activation="sigmoid",
                channel_first=channel_first,
            )

        output_inner = self.d_inner2 if self.is_cross else self.d_inner1
        if channel_first:
            self.out_norm = LayerNorm2d(output_inner)
        else:
            self.out_norm = nn.LayerNorm(output_inner)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if mode == "gate":
            self.forward = self._forward_gate
        elif mode == "cross":
            self.forward = self._forward_cross
        else:
            self.forward = self._forward_self

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            gate_input: torch.Tensor = None,
            # ==============================
            force_fp32=False,
            # ==============================
            ssoflex=True,
            no_einsum=False,
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):

        if gate_input is None:
            gate_input = x
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D_inner, R = self.k_group, self.d_inner1, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                           force_torch=scan_force_torch)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                             bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)

        ys, gate_scores = self.gate_module(gate_input, ys)

        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                         force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forward_corefuse(self, y: torch.Tensor, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, 2, 3).contiguous().view(B, -1, L)], dim=1)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        y_hwwh = torch.stack([y.view(B, -1, L), torch.transpose(y, 2, 3).contiguous().view(B, -1, L)], dim=1)
        ys = torch.cat([y_hwwh, torch.flip(y_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.float(), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs_f = xs.float().view(B, -1, L)
        ys_f = ys.float().view(B, -1, L)
        dts_f = dts.contiguous().float().view(B, -1, L)
        Bs_f = Bs.contiguous().float().view(B, K, -1, L)
        Cs_f = Cs.contiguous().float().view(B, K, -1, L)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        Ds = self.Ds.float().view(-1)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(ys_f, dts_f, As, Bs_f, Cs_f, Ds, dt_bias, True).view(B, K, -1, L)

        y1 = out_y[:, 0].view(B, -1, H, W)

        y2 = out_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous()

        y3 = torch.flip(out_y[:, 2], dims=[-1]).view(B, -1, H, W)

        y4 = torch.flip(out_y[:, 3], dims=[-1]).view(B, -1, W, H).transpose(2, 3).contiguous()

        combined_y = torch.stack([y1, y2, y3, y4], dim=1)

        gated_y = combined_y

        return gated_y[:, 0], gated_y[:, 1], gated_y[:, 2], gated_y[:, 3]

    def _forward_self(self, x: torch.Tensor, **kwargs):

        return self._forward_base(x, None, is_cross=False, **kwargs)

    def _forward_gate(self, x: torch.Tensor, gate_input: Optional[torch.Tensor] = None, **kwargs):

        return self._forward_base(x, gate_input, is_cross=False, **kwargs)

    def _forward_cross(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        return self._forward_base(x, y, is_cross=True, **kwargs)

    def _forward_base(self, x: torch.Tensor, y_or_gate: Optional[torch.Tensor],
                      is_cross: bool, **kwargs):
        if is_cross:

            x_f = self.in_proj_x(x)
            y_f = self.in_proj_y(y_or_gate)

            if not self.channel_first:
                x_f = x_f.permute(0, 3, 1, 2).contiguous()
                y_f = y_f.permute(0, 3, 1, 2).contiguous()

            if hasattr(self, 'conv2d_x'):
                x_f = self.act(self.conv2d_x(x_f))
                y_f = self.act(self.conv2d_y(y_f))

            B, C, H, W = x.shape
            ya1, ya2, ya3, ya4 = self.forward_corefuse(y_f, x_f, **kwargs)
            ya = ya1 + ya2 + ya3 + ya4

            if not self.channel_first:
                ya = ya.permute(0, 2, 3, 1).contiguous()

            ya = self.out_norm(ya)
            ya = self.out_act(ya)
            return self.dropout(self.out_proj(ya))
        else:

            x_proj = self.in_proj(x)

            z = None
            if not self.disable_z:
                split_dim = 1 if self.channel_first else -1
                x_proj, z = x_proj.chunk(2, dim=split_dim)
                if not self.disable_z_act:
                    z = self.act(z)

            if not self.channel_first:
                x_proj = x_proj.permute(0, 3, 1, 2).contiguous()

            if self.with_dconv:
                x_proj = self.conv2d(x_proj)

            x_act = self.act(x_proj)

            scan_input = x_act
            param_source = x_act
            gate_input = y_or_gate

            force_fp32 = (not self.disable_force32)
            ssoflex = getattr(self, "ssoflex", True)
            no_einsum = getattr(self, "no_einsum", False)
            selective_scan_backend = getattr(self, "selective_scan_backend", "core")
            scan_mode = getattr(self, "scan_mode", "cross2d")
            scan_force_torch = getattr(self, "scan_force_torch", False)
            delta_softplus = True

            _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                           str) else scan_mode
            assert isinstance(_scan_mode, int)

            scan_output = self.forward_ss2d(
                x=scan_input,
                gate_input=gate_input,
                force_fp32=force_fp32,
                ssoflex=ssoflex,
                no_einsum=no_einsum,
                selective_scan_backend=selective_scan_backend,
                scan_mode=scan_mode,
                scan_force_torch=scan_force_torch,
                **kwargs
            )

            y_out = scan_output

            y_out = self.out_act(y_out)

            if not self.disable_z and z is not None:
                y_out = y_out * z

            out = self.out_proj(y_out)

            out = self.dropout(out)

            return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=1.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)

            self.op = UniversalSS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                mode="gate",
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                use_gate=True,
                dropout=ssm_drop_rate,
                channel_first=channel_first,
                **kwargs
            )


        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first
            )

    def _forward(self, input: torch.Tensor, gate_input: Optional[torch.Tensor] = None):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                # x = x + self.drop_path(self.norm(self.op(x, gate_input=gate_input if gate_input is not None else x)))
                x = x + self.drop_path(self.norm(self.op(x, gate_input)))
            else:
                x = x + self.drop_path(self.norm(self.op(x, gate_input)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor, gate_input: Optional[torch.Tensor] = None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, gate_input)
        else:
            return self._forward(input, gate_input)
