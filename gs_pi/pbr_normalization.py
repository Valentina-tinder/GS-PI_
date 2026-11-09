"""Schema-aware torch normalization for PBR + Blender world normals."""

from __future__ import annotations

import torch
import torch.nn as nn

from gs_pi.pbr_schema import get_feature_schema
from gs_pi.utils import to_torch


class PBRNormalFeatureNormalization(nn.Module):
    """Keep legacy PBR statistics separate from signed world normals.

    PBR channels use the same statistics that the old five-channel
    ``MinusOneToOneNormalization(scale_per_axis=False)`` computes from material
    data.  World-normal channels use identity mapping, so adding normal cannot
    change the PBR scale.
    """

    def __init__(self, feature_schema="pbr_normal_world_v1", eps=1e-8):
        super().__init__()
        self.feature_schema = get_feature_schema(feature_schema)
        self.dim = self.feature_schema.feats_dim
        self.normal_group = self.feature_schema.group("normal")
        self.normal_slice = self.normal_group.channel_slice
        self.pbr_stop = self.normal_group.start
        self.eps = float(eps)
        self.register_buffer("min", torch.tensor([-1.0]))
        self.register_buffer("max", torch.tensor([1.0]))
        self.register_buffer("shift", torch.zeros(self.dim))
        self.register_buffer("scale", torch.ones(self.dim))

    @torch.no_grad()
    def set_from_all_data(self, data):
        data = to_torch(data, self.shift.device).detach()
        if data.ndim < 2 or data.shape[0] != self.dim:
            raise ValueError(
                f"schema {self.feature_schema.name!r} 需要 [{self.dim}, N] 特征，"
                f"当前 shape={tuple(data.shape)}"
            )
        data = data.reshape(self.dim, -1)
        pbr_data = data[: self.pbr_stop]
        if not torch.isfinite(pbr_data).all():
            raise ValueError("PBR normalization 统计中发现 NaN/Inf")

        pbr_min = pbr_data.min(dim=1).values
        pbr_max = pbr_data.max(dim=1).values
        pbr_shift = (pbr_min + pbr_max) / 2.0
        # Scalar scale matches the legacy five-channel implementation, but is
        # intentionally computed from PBR only.
        pbr_scale = ((pbr_max - pbr_min) / 2.0).max().clamp_min(self.eps)
        self.shift[: self.pbr_stop].copy_(pbr_shift)
        self.scale[: self.pbr_stop].fill_(pbr_scale)
        self.shift[self.normal_slice].zero_()
        self.scale[self.normal_slice].fill_(1.0)
        self.min.fill_(-1.0)
        self.max.fill_(1.0)

    def forward(self, x):
        return (x - self.shift[None, :, None]) / self.scale[None, :, None]

    def denormalize(self, x):
        return x * self.scale[None, :, None] + self.shift[None, :, None]
