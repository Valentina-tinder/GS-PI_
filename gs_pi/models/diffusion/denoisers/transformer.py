import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim, device=device, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.zeros(ndim, device=device, dtype=dtype))
            if bias
            else None
        )

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-05)


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self, *, heads: int, dropout: float = 0.0, use_flash_attn: bool = True
    ):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        requested_flash_attn = bool(use_flash_attn)
        self.use_flash_attn = requested_flash_attn and xops is not None
        if requested_flash_attn and xops is None:
            warnings.warn(
                "xformers is not available; falling back to standard attention. Training will still run, but memory use and step time may increase.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, qkv):
        (bs, n_ctx, width) = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        (q, k, v) = torch.split(qkv, attn_ch, dim=-1)
        if self.use_flash_attn and q.is_cuda:
            out = xops.memory_efficient_attention(
                q, k, v, p=self.dropout if self.training else 0.0
            )
        else:
            scale = 1 / math.sqrt(math.sqrt(attn_ch))
            weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)
            weight = torch.softmax(weight, dim=-1)
            weight = self.attn_dropout(weight)
            out = torch.einsum("bhts,bshc->bthc", weight, v)
        out = out.reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        bias: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=bias)
        self.c_proj = nn.Linear(width, width, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.attention = QKVMultiheadAttention(
            heads=heads, dropout=dropout, use_flash_attn=use_flash_attn
        )
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        x = self.resid_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, bias=bias)
        self.c_proj = nn.Linear(width * 4, width, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        bias: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            width=width,
            heads=heads,
            dropout=dropout,
            init_scale=init_scale,
            bias=bias,
            use_flash_attn=use_flash_attn,
        )
        if bias:
            self.ln_1 = nn.LayerNorm(width)
            self.ln_2 = nn.LayerNorm(width)
        else:
            self.ln_1 = LayerNorm(width, bias=bias)
            self.ln_2 = LayerNorm(width, bias=bias)
        self.mlp = MLP(width=width, bias=bias, dropout=dropout, init_scale=init_scale)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        layers: int,
        heads: int,
        dropout: float = 0.0,
        init_scale: float = 0.25,
        bias: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width=width,
                    heads=heads,
                    dropout=dropout,
                    init_scale=init_scale,
                    bias=bias,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class RCWCrossAttention(nn.Module):
    """交叉注意力: Q 来自一个源, KV 来自另一个源, 支持 q_dim != kv_dim"""

    def __init__(self, q_dim: int, kv_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = q_dim // heads
        self.dropout = dropout
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        (B, Nq, _) = q_input.shape
        Nkv = kv_input.shape[1]
        q = self.q_proj(q_input).view(B, Nq, self.heads, self.head_dim)
        k = self.k_proj(kv_input).view(B, Nkv, self.heads, self.head_dim)
        v = self.v_proj(kv_input).view(B, Nkv, self.heads, self.head_dim)
        if xops is not None and q.is_cuda:
            out = xops.memory_efficient_attention(
                q, k, v, p=self.dropout if self.training else 0.0
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            weight = torch.einsum("bqhd,bkhd->bhqk", q * scale, k)
            weight = torch.softmax(weight, dim=-1)
            out = torch.einsum("bhqk,bkhd->bqhd", weight, v)
        out = out.reshape(B, Nq, -1)
        return self.out_proj(out)


class ComputeBlock(nn.Module):
    """Latent 自注意力块: ln → self-attn → ln → MLP (pre-norm residual)"""

    def __init__(self, z_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(z_dim)
        self.self_attn = RCWCrossAttention(
            q_dim=z_dim, kv_dim=z_dim, heads=heads, dropout=dropout
        )
        self.ln_2 = nn.LayerNorm(z_dim)
        self.mlp = MLP(width=z_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = self.ln_1(z)
        z = z + self.self_attn(z_norm, z_norm)
        z = z + self.mlp(self.ln_2(z))
        return z


class ReadBlock(nn.Module):
    """Latent 从 point tokens 读取信息: Q=z, KV=x"""

    def __init__(self, z_dim: int, x_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_z = nn.LayerNorm(z_dim)
        self.ln_x = nn.LayerNorm(x_dim)
        self.cross_attn = RCWCrossAttention(
            q_dim=z_dim, kv_dim=x_dim, heads=heads, dropout=dropout
        )
        self.ln_post = nn.LayerNorm(z_dim)
        self.mlp = MLP(width=z_dim, dropout=dropout)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = z + self.cross_attn(self.ln_z(z), self.ln_x(x))
        z = z + self.mlp(self.ln_post(z))
        return z


class WriteBlock(nn.Module):
    """Point tokens 从 latent 读取信息: Q=x, KV=z"""

    def __init__(self, x_dim: int, z_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_x = nn.LayerNorm(x_dim)
        self.ln_z = nn.LayerNorm(z_dim)
        self.cross_attn = RCWCrossAttention(
            q_dim=x_dim, kv_dim=z_dim, heads=heads, dropout=dropout
        )
        self.ln_post = nn.LayerNorm(x_dim)
        self.mlp = MLP(width=x_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.ln_x(x), self.ln_z(z))
        x = x + self.mlp(self.ln_post(x))
        return x


class RCWBlock(nn.Module):
    """一个完整的 Read → Compute(×num_compute_layers) → Write 循环"""

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        heads: int,
        num_compute_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.read = ReadBlock(z_dim=z_dim, x_dim=x_dim, heads=heads, dropout=dropout)
        self.compute = nn.ModuleList(
            [
                ComputeBlock(z_dim=z_dim, heads=heads, dropout=dropout)
                for _ in range(num_compute_layers)
            ]
        )
        self.write = WriteBlock(x_dim=x_dim, z_dim=z_dim, heads=heads, dropout=dropout)

    def forward(
        self, z: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.read(z, x)
        for compute_block in self.compute:
            z = compute_block(z)
        x = self.write(x, z)
        return (z, x)


class TwoStreamBackbone(nn.Module):
    """替代原 Transformer 的完整双流骨干"""

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        heads: int,
        num_latents: int = 256,
        num_blocks: int = 6,
        num_compute_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.num_latents = num_latents
        self.z_init = nn.Parameter(torch.randn(1, num_latents, z_dim) * 0.02)
        self.cond_proj = nn.Linear(x_dim, z_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(z_dim, z_dim * 4), nn.GELU(), nn.Linear(z_dim * 4, z_dim)
        )
        self.blocks = nn.ModuleList(
            [
                RCWBlock(
                    z_dim=z_dim,
                    x_dim=x_dim,
                    heads=heads,
                    num_compute_layers=num_compute_layers,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, N, x_dim] point tokens
        t_embed: [B, z_dim] timestep embedding
        cond_tokens: [B, L, x_dim] DINO condition tokens (可选)
        """
        B = x.shape[0]
        z = self.z_init.expand(B, -1, -1)
        t_token = self.time_proj(t_embed).unsqueeze(1)
        z_parts = [z, t_token]
        if cond_tokens is not None:
            cond_z = self.cond_proj(cond_tokens)
            z_parts.append(cond_z)
        z = torch.cat(z_parts, dim=1)
        for block in self.blocks:
            (z, x) = block(z, x)
        return x


class NPCDTransformer(nn.Module):
    def __init__(
        self,
        *,
        coords_dim: int,
        feats_dim: int,
        num_points: int = 512,
        cond_point_dim: int = 0,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        use_flash_attn: bool = True,
        use_two_stream: bool = False,
        backbone_type: Optional[str] = None,
        z_dim: int = 768,
        num_latents: int = 256,
        num_blocks: int = 6,
        num_compute_layers: int = 4,
        cond_point_top_k: int = 0,
        num_global_views: int = 0,
        view_id_embed_dim: int = 0,
        topk_view_id_conditioning: str = "embed",
        topk_include_confidence: bool = True,
        normal_feature_start: Optional[int] = None,
        normal_feature_stop: Optional[int] = None,
    ):
        super().__init__()
        self.coords_dim = coords_dim
        self.feats_dim = feats_dim
        self.num_points = int(num_points)
        self.cond_point_dim = cond_point_dim
        if backbone_type is None:
            resolved_backbone_type = "two_stream" if use_two_stream else "transformer"
        else:
            resolved_backbone_type = (
                str(backbone_type).strip().lower().replace("-", "_")
            )
            if resolved_backbone_type not in {"transformer", "two_stream"}:
                raise ValueError(
                    f"unsupported backbone_type={backbone_type!r}; expected one of 'transformer' or 'two_stream'"
                )
            if use_two_stream and resolved_backbone_type != "two_stream":
                raise ValueError(
                    f"use_two_stream=True conflicts with explicit backbone_type={backbone_type!r}"
                )
        self.backbone_type = resolved_backbone_type
        self.cond_point_top_k = int(cond_point_top_k)
        self.num_global_views = int(num_global_views)
        self.view_id_embed_dim = int(view_id_embed_dim)
        self.topk_view_id_conditioning = str(topk_view_id_conditioning).lower()
        self.topk_include_confidence = bool(topk_include_confidence)
        if (normal_feature_start is None) != (normal_feature_stop is None):
            raise ValueError(
                "normal_feature_start 与 normal_feature_stop 必须同时设置或同时为 None"
            )
        self.normal_feature_start = (
            None if normal_feature_start is None else int(normal_feature_start)
        )
        self.normal_feature_stop = (
            None if normal_feature_stop is None else int(normal_feature_stop)
        )
        if self.normal_feature_start is not None and (
            not 0 <= self.normal_feature_start < self.normal_feature_stop <= feats_dim
        ):
            raise ValueError(
                f"normal feature slice 超出 feats 范围：[{self.normal_feature_start}:{self.normal_feature_stop}] vs feats_dim={feats_dim}"
            )
        self._normal_finetune_hook_handles = []
        self._normal_finetune_head_parameters = ()
        self._normal_finetune_backbone_parameters = ()
        self.use_topk_cond_expand = (
            self.cond_point_top_k > 0
            and self.topk_view_id_conditioning in {"embed", "onehot", "none"}
        )
        if self.use_topk_cond_expand:
            expected_cond_dim = self.cond_point_top_k * 5
            if cond_point_dim != expected_cond_dim:
                raise ValueError(
                    f"cond_point_top_k={self.cond_point_top_k} requires cond_point_dim={expected_cond_dim} (=K*5), got {cond_point_dim}"
                )
            conf_dim = 1 if self.topk_include_confidence else 0
            if self.topk_view_id_conditioning == "embed":
                if self.num_global_views <= 0:
                    raise ValueError(
                        "topk_view_id_conditioning='embed' requires num_global_views > 0"
                    )
                if self.view_id_embed_dim <= 0:
                    raise ValueError(
                        "topk_view_id_conditioning='embed' requires view_id_embed_dim > 0"
                    )
                self.view_id_embed = nn.Embedding(
                    self.num_global_views, self.view_id_embed_dim
                )
                nn.init.normal_(self.view_id_embed.weight, mean=0.0, std=0.02)
                self.topk_view_id_expanded_dim = self.view_id_embed_dim
            elif self.topk_view_id_conditioning == "onehot":
                if self.num_global_views <= 0:
                    raise ValueError(
                        "topk_view_id_conditioning='onehot' requires num_global_views > 0"
                    )
                self.view_id_embed = None
                self.topk_view_id_expanded_dim = self.num_global_views
            elif self.topk_view_id_conditioning == "none":
                self.view_id_embed = None
                self.topk_view_id_expanded_dim = 0
            else:
                raise ValueError(
                    f"Unsupported topk_view_id_conditioning={self.topk_view_id_conditioning!r}; expected 'embed', 'onehot', or 'none'."
                )
            effective_cond_dim = self.cond_point_top_k * (
                3 + conf_dim + self.topk_view_id_expanded_dim
            )
        else:
            self.view_id_embed = None
            self.topk_view_id_expanded_dim = 0
            effective_cond_dim = cond_point_dim
        self.input_channels = coords_dim + feats_dim + effective_cond_dim
        self.output_channels = coords_dim + feats_dim
        self.use_two_stream = self.backbone_type == "two_stream"
        self.input_proj = nn.Linear(self.input_channels, width)
        self.ln_pre = nn.LayerNorm(width)
        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, self.output_channels)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()
        if self.backbone_type == "two_stream":
            self.time_embed = MLP(
                width=z_dim, init_scale=init_scale * math.sqrt(1.0 / z_dim)
            )
            self.backbone = TwoStreamBackbone(
                z_dim=z_dim,
                x_dim=width,
                heads=heads,
                num_latents=num_latents,
                num_blocks=num_blocks,
                num_compute_layers=num_compute_layers,
            )
        else:
            self.time_embed = MLP(
                width=width, init_scale=init_scale * math.sqrt(1.0 / width)
            )
            self.backbone = Transformer(
                width=width,
                layers=layers,
                heads=heads,
                use_flash_attn=use_flash_attn,
                init_scale=init_scale,
            )
        self.supports_variable_num_points = bool(
            getattr(
                self.backbone,
                "supports_variable_num_points",
                self.backbone_type in {"transformer", "two_stream"},
            )
        )
        self.supports_full_pointcloud_training = bool(
            getattr(self.backbone, "supports_full_pointcloud_training", False)
        )
        self.requires_full_pointcloud_inference = bool(
            getattr(self.backbone, "requires_full_pointcloud_inference", False)
        )

    def configure_normal_finetune(
        self, unfreeze_last_n_blocks: int = 4
    ) -> Dict[str, int]:
        """Expose only normal projection slices and the last backbone blocks.

        Projection tensors keep their historical checkpoint shapes. Gradient hooks
        mask the shared tensors so AdamW can update only the three normal input
        columns/output rows. The selected backbone blocks stay trainable from DDP's
        perspective; the trainer freezes them during bootstrap by assigning LR=0.
        """
        if self.normal_feature_start is None or self.normal_feature_stop is None:
            raise ValueError("normal finetune 需要包含 normal group 的 feature_schema")
        for handle in self._normal_finetune_hook_handles:
            handle.remove()
        self._normal_finetune_hook_handles = []
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        head_parameters = (
            self.input_proj.weight,
            self.output_proj.weight,
            self.output_proj.bias,
        )
        for parameter in head_parameters:
            parameter.requires_grad_(True)
        if self.backbone_type == "transformer":
            blocks = self.backbone.resblocks
        else:
            blocks = self.backbone.blocks
        num_blocks = len(blocks)
        requested_blocks = int(unfreeze_last_n_blocks)
        if requested_blocks < 0 or requested_blocks > num_blocks:
            raise ValueError(
                f"unfreeze_last_n_blocks 必须在 0..{num_blocks}，当前为 {requested_blocks}"
            )
        selected_blocks = (
            list(blocks[num_blocks - requested_blocks :]) if requested_blocks else []
        )
        backbone_parameters = tuple(
            (parameter for block in selected_blocks for parameter in block.parameters())
        )
        for parameter in backbone_parameters:
            parameter.requires_grad_(True)
        normal_input_start = self.coords_dim + self.normal_feature_start
        normal_input_stop = self.coords_dim + self.normal_feature_stop
        normal_output_start = self.coords_dim + self.normal_feature_start
        normal_output_stop = self.coords_dim + self.normal_feature_stop

        def mask_input_weight(gradient):
            masked = torch.zeros_like(gradient)
            masked[:, normal_input_start:normal_input_stop] = gradient[
                :, normal_input_start:normal_input_stop
            ]
            return masked

        def mask_output_weight(gradient):
            masked = torch.zeros_like(gradient)
            masked[normal_output_start:normal_output_stop] = gradient[
                normal_output_start:normal_output_stop
            ]
            return masked

        def mask_output_bias(gradient):
            masked = torch.zeros_like(gradient)
            masked[normal_output_start:normal_output_stop] = gradient[
                normal_output_start:normal_output_stop
            ]
            return masked

        self._normal_finetune_hook_handles = [
            self.input_proj.weight.register_hook(mask_input_weight),
            self.output_proj.weight.register_hook(mask_output_weight),
            self.output_proj.bias.register_hook(mask_output_bias),
        ]
        self._normal_finetune_head_parameters = head_parameters
        self._normal_finetune_backbone_parameters = backbone_parameters
        return {
            "normal_head_parameter_tensors": len(head_parameters),
            "normal_backbone_parameter_tensors": len(backbone_parameters),
            "unfrozen_backbone_blocks": requested_blocks,
        }

    def normal_finetune_parameter_groups(self) -> Dict[str, Tuple[nn.Parameter, ...]]:
        if not self._normal_finetune_head_parameters:
            raise RuntimeError(
                "normal finetune 尚未配置；请在 DDP 包装前调用 configure_normal_finetune()"
            )
        return {
            "head": tuple(self._normal_finetune_head_parameters),
            "backbone": tuple(self._normal_finetune_backbone_parameters),
        }

    def _expand_topk_cond_point_colors(
        self, cond_point_colors: torch.Tensor
    ) -> torch.Tensor:
        """Expand packed Top-K cond features by replacing view_id with learned features.

        Input layout (per dataset.pack_topk_cond_point_colors):
          [B, K*3 + K + K, N] = RGB | conf | view_id(float, 0..V-1)

        Output layout fed into the input projection:
          [B, K*3 + K(conf, optional) + K*D, N] where D is either
          ``view_id_embed_dim``, ``num_global_views`` (onehot), or 0 when
          ``topk_view_id_conditioning='none'``.
        """
        k = self.cond_point_top_k
        (b, c, n) = cond_point_colors.shape
        if c != k * 5:
            raise ValueError(
                f"cond_point_colors channel count {c} does not match K*5={k * 5}"
            )
        rgb = cond_point_colors[:, : 3 * k]
        conf = cond_point_colors[:, 3 * k : 4 * k]
        view_id = cond_point_colors[:, 4 * k : 5 * k]
        view_feat = None
        if self.topk_view_id_conditioning == "embed":
            view_id_long = view_id.round().clamp(0, self.num_global_views - 1).long()
            view_feat = self.view_id_embed(view_id_long)
            view_feat = (
                view_feat.permute(0, 1, 3, 2)
                .contiguous()
                .view(b, k * self.view_id_embed_dim, n)
            )
        elif self.topk_view_id_conditioning == "onehot":
            view_id_long = view_id.round().clamp(0, self.num_global_views - 1).long()
            view_feat = F.one_hot(view_id_long, num_classes=self.num_global_views).to(
                rgb.dtype
            )
            view_feat = (
                view_feat.permute(0, 1, 3, 2)
                .contiguous()
                .view(b, k * self.num_global_views, n)
            )
        elif self.topk_view_id_conditioning == "none":
            view_feat = None
        else:
            raise ValueError(
                f"Unsupported topk_view_id_conditioning={self.topk_view_id_conditioning!r}"
            )
        parts = [rgb]
        if self.topk_include_confidence:
            parts.append(conf)
        if view_feat is not None:
            parts.append(view_feat)
        return torch.cat(parts, dim=1)

    def forward(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor = None,
        cond_point_colors: torch.Tensor = None,
        view_source: torch.Tensor = None,
        light_probe: torch.Tensor = None,
        source_uv: torch.Tensor = None,
    ):
        """
        cond_tokens: [N, L, Width] 外部传入的条件特征 (例如图片编码后的特征)
        cond_point_colors: [N, C, num_points] 逐点多视角 RGB(+confidence+view_id) 条件
          - standard 模式: C = V*3 + V (RGB + one-hot) 或类似
          - topk 模式:     C = K*5 (RGB*K + conf*K + view_id*K)，模型内部展开
        """
        if cond_point_colors is not None and self.use_topk_cond_expand:
            cond_point_colors = self._expand_topk_cond_point_colors(cond_point_colors)
        parts = [coords, feats]
        if cond_point_colors is not None:
            parts.append(cond_point_colors)
        x = torch.cat(parts, dim=1)
        if self.backbone_type == "two_stream":
            return self._forward_two_stream(x, t, cond_tokens)
        return self._forward_original(x, t, cond_tokens)

    def _forward_original(
        self, x: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor = None
    ):
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        extra_tokens = [(t_embed, True)]
        if cond_tokens is not None:
            extra_tokens.append((cond_tokens, True))
            if not hasattr(self, "_debug_printed"):
                print(
                    f"\n[DEBUG-Transformer] cond_tokens 已传入! shape: {cond_tokens.shape}"
                )
                print(
                    f"[DEBUG-Transformer] 总 token 数: 1(time) + {cond_tokens.shape[1]}(cond) + {x.shape[2]}(points) = {1 + cond_tokens.shape[1] + x.shape[2]}\n"
                )
                self._debug_printed = True
        elif not hasattr(self, "_debug_printed"):
            print(
                f"\n[WARNING-Transformer] cond_tokens 为 None! 模型未接收到条件信息!\n"
            )
            self._debug_printed = True
        pred = self._forward_with_cond(x, extra_tokens)
        pred_coords = pred[:, : self.coords_dim]
        pred_feats = pred[:, self.coords_dim :]
        return (pred_coords, pred_feats)

    def _forward_two_stream(
        self, x: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor = None
    ):
        h = self.input_proj(x.permute(0, 2, 1))
        h = self.ln_pre(h)
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.z_dim))
        h = self.backbone(h, t_embed, cond_tokens)
        h = self.ln_post(h)
        h = self.output_proj(h)
        pred = h.permute(0, 2, 1)
        pred_coords = pred[:, : self.coords_dim]
        pred_feats = pred[:, self.coords_dim :]
        return (pred_coords, pred_feats)

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))
        extra_tokens = [
            emb[:, None] if len(emb.shape) == 2 else emb
            for (emb, as_token) in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)
        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum((h.shape[1] for h in extra_tokens)) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)
