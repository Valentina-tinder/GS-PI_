"""Shared feature schemas for legacy PBR and optional world-normal training.

The legacy five-channel path remains the default.  Normal-aware code must opt in
with ``pbr_normal_world_v1``; importing this module does not change existing
dataset or model behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FeatureGroup:
    """A named contiguous group in a ``[..., C, N]`` feature tensor."""

    name: str
    start: int
    stop: int
    value_range: tuple[float, float]
    normalization: str
    space: str | None = None

    @property
    def channel_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def channels(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FeatureSchema:
    """Declarative layout for the point features predicted by GS-PI."""

    name: str
    groups: tuple[FeatureGroup, ...]

    @property
    def feats_dim(self) -> int:
        return max(group.stop for group in self.groups)

    def group(self, name: str) -> FeatureGroup:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"feature group {name!r} is not present in schema {self.name!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feats_dim": self.feats_dim,
            "groups": [
                {
                    "name": group.name,
                    "start": group.start,
                    "stop": group.stop,
                    "value_range": list(group.value_range),
                    "normalization": group.normalization,
                    "space": group.space,
                }
                for group in self.groups
            ],
        }


PBR_V1 = FeatureSchema(
    name="pbr_v1",
    groups=(
        FeatureGroup("albedo", 0, 3, (0.0, 1.0), "zero_one_to_minus_one_one"),
        FeatureGroup("roughness", 3, 4, (0.0, 1.0), "zero_one_to_minus_one_one"),
        FeatureGroup("metallic", 4, 5, (0.0, 1.0), "zero_one_to_minus_one_one"),
    ),
)

PBR_NORMAL_WORLD_V1 = FeatureSchema(
    name="pbr_normal_world_v1",
    groups=(
        *PBR_V1.groups,
        FeatureGroup(
            "normal",
            5,
            8,
            (-1.0, 1.0),
            "unit_vector_identity",
            space="blender_world",
        ),
    ),
)

_SCHEMAS: Mapping[str, FeatureSchema] = {
    PBR_V1.name: PBR_V1,
    PBR_NORMAL_WORLD_V1.name: PBR_NORMAL_WORLD_V1,
}


def get_feature_schema(name: str | FeatureSchema | None = None) -> FeatureSchema:
    """Resolve a schema while keeping the historical five-channel default."""

    if isinstance(name, FeatureSchema):
        return name
    key = PBR_V1.name if name is None else str(name)
    try:
        return _SCHEMAS[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature_schema={key!r}; expected one of {sorted(_SCHEMAS)}"
        ) from exc


def validate_feature_tensor(
    features: Any,
    schema: str | FeatureSchema | None = None,
    *,
    channel_axis: int = -2,
) -> None:
    """Validate only the channel layout; works for NumPy and torch tensors."""

    resolved = get_feature_schema(schema)
    ndim = int(features.ndim)
    axis = channel_axis if channel_axis >= 0 else ndim + channel_axis
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"invalid channel_axis={channel_axis} for shape {features.shape}"
        )
    actual = int(features.shape[axis])
    if actual != resolved.feats_dim:
        raise ValueError(
            f"schema {resolved.name!r} expects {resolved.feats_dim} feature channels, "
            f"got {actual} for shape {tuple(features.shape)}"
        )


def split_feature_channels(
    features: Any,
    schema: str | FeatureSchema | None = None,
    *,
    channel_axis: int = -2,
) -> dict[str, Any]:
    """Return named channel views without copying the underlying tensor."""

    resolved = get_feature_schema(schema)
    validate_feature_tensor(features, resolved, channel_axis=channel_axis)
    ndim = int(features.ndim)
    axis = channel_axis if channel_axis >= 0 else ndim + channel_axis
    result: dict[str, Any] = {}
    for group in resolved.groups:
        index = [slice(None)] * ndim
        index[axis] = group.channel_slice
        result[group.name] = features[tuple(index)]
    return result


def normalize_world_normals_numpy(
    normals: np.ndarray,
    *,
    axis: int = -1,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize vectors without inventing a direction for near-zero inputs."""

    values = np.asarray(normals, dtype=np.float32)
    lengths = np.linalg.norm(values, axis=axis, keepdims=True)
    finite = np.isfinite(values).all(axis=axis)
    valid = finite & (np.squeeze(lengths, axis=axis) > float(eps))
    normalized = np.zeros_like(values, dtype=np.float32)
    np.divide(
        values,
        np.maximum(lengths, eps),
        out=normalized,
        where=np.expand_dims(valid, axis=axis),
    )
    return normalized, valid


def encode_normal_rgb_numpy(normals: np.ndarray) -> np.ndarray:
    """Encode signed linear normals to linear RGB in ``[0, 1]``."""

    return np.clip((np.asarray(normals, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)


def decode_normal_rgb_numpy(
    rgb: np.ndarray,
    *,
    center: float = 0.5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode Blender's remapped EXR direction and return raw vector lengths."""

    direction = np.asarray(rgb, dtype=np.float32) - float(center)
    raw_lengths = np.linalg.norm(direction, axis=-1)
    normalized, valid = normalize_world_normals_numpy(direction, axis=-1, eps=eps)
    return normalized, valid, raw_lengths
