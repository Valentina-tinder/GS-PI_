from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


def _compress_secondary_topk_confidence(conf: torch.Tensor) -> torch.Tensor:
    """Compress non-slot0 Top-K confidence into {-1} U [0, 0.5].

    Semantics:
      - invalid sentinel ``-1`` is preserved
      - active positive confidence ``(0, 1]`` becomes ``(0, 0.5]``
      - non-positive but non-sentinel confidence is clamped to ``0`` so it
        stays consistent with the already-zeroed RGB for inactive slots
    """
    invalid_mask = conf.le(-1.0 + 1e-6)
    scaled_conf = conf.clamp_min(0.0).mul(0.5)
    return torch.where(invalid_mask, conf, scaled_conf)


def build_cond_point_features_2d_projection(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_maps: torch.Tensor,
    rgb_maps: torch.Tensor,
    *,
    sigma_scale: float = 0.5,
    base_margin: float = 0.01,
    sample_mode: str = "nearest",
    align_corners: bool = True,
    rgb_soft_mask: bool = False,
    rgb_soft_mask_floor: float = 0.0,
    eps: float = 1e-6,
    return_debug: bool = False,
) -> TensorOrTuple:
    """Build per-point multi-view RGB + confidence features with full tensor ops.

    Args:
        points: [N, 3] world-space points.
        extrinsics: [V, 4, 4] world-to-camera matrices.
        intrinsics: [V, 3, 3] camera intrinsics.
        depth_maps: [V, 1, H, W] rendered depth maps.
        rgb_maps: [V, 3, H, W] rendered RGB maps in [-1, 1].

    Returns:
        features: [V * 4, N], laid out as [RGB(view_1..V), confidence(view_1..V)].
        If ``return_debug=True``, also returns a dict with intermediate tensors.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be [N, 3], got {tuple(points.shape)}")
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics must be [V, 4, 4], got {tuple(extrinsics.shape)}")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must be [V, 3, 3], got {tuple(intrinsics.shape)}")
    if depth_maps.ndim != 4 or depth_maps.shape[1] != 1:
        raise ValueError(
            f"depth_maps must be [V, 1, H, W], got {tuple(depth_maps.shape)}"
        )
    if rgb_maps.ndim != 4 or rgb_maps.shape[1] != 3:
        raise ValueError(f"rgb_maps must be [V, 3, H, W], got {tuple(rgb_maps.shape)}")

    num_views = extrinsics.shape[0]
    if (
        intrinsics.shape[0] != num_views
        or depth_maps.shape[0] != num_views
        or rgb_maps.shape[0] != num_views
    ):
        raise ValueError(
            "View dimension mismatch: "
            f"extrinsics={extrinsics.shape[0]}, intrinsics={intrinsics.shape[0]}, "
            f"depth_maps={depth_maps.shape[0]}, rgb_maps={rgb_maps.shape[0]}"
        )
    if depth_maps.shape[-2:] != rgb_maps.shape[-2:]:
        raise ValueError(
            f"depth_maps and rgb_maps must share H/W, got {tuple(depth_maps.shape[-2:])} "
            f"vs {tuple(rgb_maps.shape[-2:])}"
        )
    if sample_mode not in {"nearest", "bilinear"}:
        raise ValueError(
            f"sample_mode must be 'nearest' or 'bilinear', got {sample_mode!r}"
        )
    if not 0.0 <= rgb_soft_mask_floor <= 1.0:
        raise ValueError(
            f"rgb_soft_mask_floor must be within [0, 1], got {rgb_soft_mask_floor!r}"
        )

    device = points.device
    compute_dtype = points.dtype if points.is_floating_point() else torch.float32
    if device.type == "cpu" and compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    points = points.to(device=device, dtype=compute_dtype)
    extrinsics = extrinsics.to(device=device, dtype=compute_dtype)
    intrinsics = intrinsics.to(device=device, dtype=compute_dtype)
    depth_maps = depth_maps.to(device=device, dtype=compute_dtype)
    rgb_maps = rgb_maps.to(device=device, dtype=compute_dtype)

    num_points = points.shape[0]
    _, _, height, width = depth_maps.shape

    ones = torch.ones((num_points, 1), device=device, dtype=compute_dtype)
    points_h = torch.cat([points, ones], dim=-1)  # [N, 4]

    # [V, N, 3] camera-space coordinates, extrinsics are world-to-camera.
    points_cam = torch.einsum("vij,nj->vni", extrinsics[:, :3, :], points_h)
    z_proj = points_cam[..., 2]  # [V, N]
    safe_z = z_proj.clamp_min(eps)

    pixels_h = torch.einsum("vij,vnj->vni", intrinsics, points_cam)
    u = pixels_h[..., 0] / safe_z
    v = pixels_h[..., 1] / safe_z

    finite_mask = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z_proj)
    positive_depth = z_proj > 0
    in_bounds = (
        finite_mask & (u >= 0) & (u <= (width - 1)) & (v >= 0) & (v <= (height - 1))
    )
    proj_valid_mask = positive_depth & in_bounds

    if align_corners:
        if width > 1:
            grid_x = 2.0 * u / (width - 1) - 1.0
        else:
            grid_x = torch.zeros_like(u)
        if height > 1:
            grid_y = 2.0 * v / (height - 1) - 1.0
        else:
            grid_y = torch.zeros_like(v)
    else:
        grid_x = 2.0 * (u + 0.5) / width - 1.0
        grid_y = 2.0 * (v + 0.5) / height - 1.0

    grid = torch.stack([grid_x, grid_y], dim=-1)  # [V, N, 2]
    grid = torch.where(torch.isfinite(grid), grid, torch.full_like(grid, 2.0))
    grid = grid.unsqueeze(2)  # [V, N, 1, 2]

    z_render = (
        F.grid_sample(
            depth_maps,
            grid,
            mode=sample_mode,
            padding_mode="zeros",
            align_corners=align_corners,
        )
        .squeeze(-1)
        .squeeze(1)
    )  # [V, N]

    sampled_rgb_raw = F.grid_sample(
        rgb_maps,
        grid,
        mode=sample_mode,
        padding_mode="zeros",
        align_corners=align_corners,
    ).squeeze(-1)  # [V, 3, N]

    depth_supported = torch.isfinite(z_render) & (z_render > eps)
    # valid_mask only decides whether this view can produce a real sampled depth/color.
    valid_mask = proj_valid_mask & depth_supported

    delta_z = torch.abs(z_proj - z_render)  # [V, N]
    inf_delta = torch.full_like(delta_z, float("inf"))
    valid_delta = torch.where(valid_mask, delta_z, inf_delta)
    min_delta_z = valid_delta.min(dim=0).values  # [N]
    has_valid_view = valid_mask.any(dim=0)  # [N]

    min_delta_safe = torch.where(
        has_valid_view, min_delta_z, torch.zeros_like(min_delta_z)
    )
    sigma = min_delta_safe * sigma_scale + base_margin
    sigma_sq = sigma.square().clamp_min(eps)

    score = torch.exp(
        -((delta_z - min_delta_safe.unsqueeze(0)).square())
        / (2.0 * sigma_sq.unsqueeze(0))
    )
    confidence = score.mul(2.0).sub(1.0).clamp_(-1.0, 1.0)
    confidence = torch.where(valid_mask, confidence, torch.full_like(confidence, -1.0))

    # Hard rule: confidence <= 0 means this view contributes no RGB.
    inactive_mask = confidence.le(0.0)
    sampled_rgb = sampled_rgb_raw
    sampled_rgb = torch.where(
        inactive_mask.unsqueeze(1), torch.zeros_like(sampled_rgb), sampled_rgb
    )
    rgb_weight = torch.where(
        inactive_mask, torch.zeros_like(confidence), torch.ones_like(confidence)
    )
    if rgb_soft_mask:
        conf01 = confidence.add(1.0).mul(0.5).clamp(0.0, 1.0)
        rgb_weight = torch.where(
            ~inactive_mask,
            conf01.clamp_min(rgb_soft_mask_floor),
            rgb_weight,
        )
        sampled_rgb = sampled_rgb * rgb_weight.unsqueeze(1)
    features = torch.cat(
        [sampled_rgb.reshape(num_views * 3, num_points), confidence], dim=0
    )

    if not return_debug:
        return features

    debug = {
        "u": u,
        "v": v,
        "z_proj": z_proj,
        "z_render": z_render,
        "delta_z": delta_z,
        "min_delta_z": min_delta_safe,
        "sigma": sigma,
        "proj_valid_mask": proj_valid_mask,
        "valid_mask": valid_mask,
        "sampled_rgb_raw": sampled_rgb_raw,
        "sampled_rgb": sampled_rgb,
        "rgb_weight": rgb_weight,
        "confidence": confidence,
    }
    return features, debug


def build_cond_point_features_topk(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_maps: torch.Tensor,
    rgb_maps: torch.Tensor,
    *,
    top_k: int,
    primary_view_id: Optional[torch.Tensor] = None,
    primary_rgb: Optional[torch.Tensor] = None,
    sigma_scale: float = 0.5,
    base_margin: float = 0.05,
    sample_mode: str = "nearest",
    align_corners: bool = True,
    eps: float = 1e-6,
    return_debug: bool = False,
):
    """Per-point Top-K visible-view features (RGB + confidence + view_id).

    Selects the K views per point. If ``primary_view_id`` is provided, slot-0 is
    forced to that 0-based view id and gets confidence=1 by construction; if
    ``primary_rgb`` is also provided, slot-0 RGB is taken from it directly,
    otherwise slot-0 RGB is gathered from the corresponding sampled source view.
    The remaining K-1 slots are selected from the other views by confidence, and
    their stored confidence is compressed into ``{-1} U [0, 0.5]`` so slot-0
    remains the only full-strength confidence channel. Otherwise all K slots are
    selected purely by confidence.

    Returns a dict (always, regardless of ``return_debug``):
        topk_rgb:     [K, 3, N] in [-1, 1]
        topk_conf:    [K, N]    in [-1, 1]
        topk_view_id: [K, N] long, 0-based global camera id (0..V-1)
        num_views:    int (V)
        k:            int (K)

    If ``return_debug`` is True, also includes ``debug`` key with intermediate
    tensors from the underlying confidence computation.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Reuse the projection/confidence pipeline by calling the base builder with
    # return_debug=True. We discard ``features`` and rebuild Top-K directly from
    # ``debug``: this keeps the geometry semantics identical to the V-channel
    # path so downstream comparisons are apples-to-apples.
    _, debug = build_cond_point_features_2d_projection(
        points=points,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        depth_maps=depth_maps,
        rgb_maps=rgb_maps,
        sigma_scale=sigma_scale,
        base_margin=base_margin,
        sample_mode=sample_mode,
        align_corners=align_corners,
        rgb_soft_mask=False,
        rgb_soft_mask_floor=0.0,
        eps=eps,
        return_debug=True,
    )

    confidence = debug["confidence"]  # [V, N] in [-1, 1]
    sampled_rgb_raw = debug[
        "sampled_rgb_raw"
    ]  # [V, 3, N] in [-1, 1] (before hard gating)
    sampled_rgb = debug[
        "sampled_rgb"
    ]  # [V, 3, N] in [-1, 1] (already gated by hard rule)

    num_views, num_points = confidence.shape
    if top_k > num_views:
        raise ValueError(f"top_k={top_k} exceeds num_views={num_views}")

    if primary_view_id is None:
        topk_conf, topk_idx = torch.topk(confidence, k=top_k, dim=0)  # [K, N]
        rgb_gather_idx = topk_idx.unsqueeze(1).expand(-1, 3, -1)  # [K, 3, N]
        topk_rgb = torch.gather(sampled_rgb, 0, rgb_gather_idx)  # [K, 3, N]
        topk_view_id = topk_idx.to(dtype=torch.long)  # [K, N], 0-based
    else:
        primary_view_id = torch.as_tensor(
            primary_view_id,
            device=confidence.device,
            dtype=torch.long,
        ).contiguous()
        if primary_view_id.ndim != 1 or primary_view_id.shape[0] != num_points:
            raise ValueError(
                f"primary_view_id must be [N], got {tuple(primary_view_id.shape)} for N={num_points}"
            )
        if (
            int(primary_view_id.min().item()) < 0
            or int(primary_view_id.max().item()) >= num_views
        ):
            raise ValueError(
                f"primary_view_id must be within [0, {num_views - 1}], "
                f"got [{int(primary_view_id.min().item())}, {int(primary_view_id.max().item())}]"
            )

        slot0_view_id = primary_view_id.view(1, num_points)  # [1, N]
        if primary_rgb is not None:
            primary_rgb = torch.as_tensor(
                primary_rgb,
                device=confidence.device,
                dtype=sampled_rgb_raw.dtype,
            ).contiguous()
            if primary_rgb.shape != (3, num_points):
                raise ValueError(
                    f"primary_rgb must be [3, N], got {tuple(primary_rgb.shape)} for N={num_points}"
                )
            slot0_rgb = primary_rgb.unsqueeze(0)  # [1, 3, N]
        else:
            slot0_rgb_idx = slot0_view_id.unsqueeze(1).expand(-1, 3, -1)  # [1, 3, N]
            slot0_rgb = torch.gather(sampled_rgb_raw, 0, slot0_rgb_idx)  # [1, 3, N]
        slot0_conf = torch.ones(
            (1, num_points), device=confidence.device, dtype=confidence.dtype
        )

        if top_k == 1:
            topk_rgb = slot0_rgb
            topk_conf = slot0_conf
            topk_view_id = slot0_view_id
        else:
            remaining_confidence = confidence.clone()
            remaining_confidence.scatter_(
                0,
                slot0_view_id,
                torch.full_like(slot0_view_id, float("-inf"), dtype=confidence.dtype),
            )
            remaining_conf, remaining_idx = torch.topk(
                remaining_confidence, k=top_k - 1, dim=0
            )
            remaining_rgb_idx = remaining_idx.unsqueeze(1).expand(-1, 3, -1)
            remaining_rgb = torch.gather(sampled_rgb, 0, remaining_rgb_idx)
            remaining_conf = _compress_secondary_topk_confidence(remaining_conf)

            topk_rgb = torch.cat([slot0_rgb, remaining_rgb], dim=0)
            topk_conf = torch.cat([slot0_conf, remaining_conf], dim=0)
            topk_view_id = torch.cat(
                [slot0_view_id, remaining_idx.to(dtype=torch.long)], dim=0
            )

    # For points whose Top-K includes "no real visibility" slots (confidence==-1),
    # zero out the RGB so it does not pollute downstream consumers. The slot is
    # still occupied (model can read confidence and decide to ignore it) but the
    # RGB carries no spurious signal.
    invisible_slot = topk_conf.le(-1.0 + 1e-6)  # [K, N]
    topk_rgb = torch.where(
        invisible_slot.unsqueeze(1), torch.zeros_like(topk_rgb), topk_rgb
    )

    out = {
        "topk_rgb": topk_rgb,
        "topk_conf": topk_conf,
        "topk_view_id": topk_view_id,
        "num_views": int(num_views),
        "k": int(top_k),
    }
    if return_debug:
        out["debug"] = debug
    return out
