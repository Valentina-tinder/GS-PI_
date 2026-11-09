#!/usr/bin/env python3
"""Offline precompute Top-K per-point cond features for PBRConditionalDataset.

For each object:
  1. Load the merged point cloud (coords).
  2. Build per-view depth maps (rasterized from per-view PLYs or the merged PLY).
  3. Call ``build_cond_point_features_topk`` with the merged PLY ``view_source``
     as slot-0 semantics:
       - slot 0 RGB     = the source-view pixel value
       - slot 0 conf    = 1
       - slot 0 view_id = merged PLY view_source (converted to 0-based)
     The remaining K-1 slots are selected from the other views by confidence.
  4. Persist a ``.pt`` payload containing:
       - topk_rgb     [K, 3, N]   in [-1, 1]
       - topk_conf    [K, N]      in [-1, 1]
       - topk_view_id [K, N]      long, 0..V-1
       - num_views    int (V)
       - k            int (K)
       - object_id, point_cloud_path, num_points, ordered XYZ fingerprint, metadata

These caches are consumed by ``PBRConditionalDataset`` when
``cond_point_features_dir`` points here and the training config sets
``cond_point_mode: topk``.

Usage example (single-shard CPU):
    python scripts/tools/precompute_topk_cond_point_features.py \\
        --dataset-root /path/to/data \\
        --output-dir    /path/to/data \\
        --num-views 32 --top-k 6 --device cpu --num-workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gs_pi.data.cond_point_features import build_cond_point_features_topk
from gs_pi.data.srn import PBRConditionalDataset
from gs_pi.pointcloud_storage import ViewPointCloudReader

# Reuse helpers from the existing V-channel precompute script.
from scripts.tools.precompute_projected_cond_point_features import (  # noqa: E402
    apply_round_robin_shard,
    build_depth_maps,
    configure_parallel_cpu_environment,
    get_available_cpu_count,
    init_worker_process,
    list_object_ids,
    load_cameras_from_metadata,
    load_points_from_ply,
    load_rgb_maps,
    resolve_threads_per_worker,
    resolve_view_ply_preference,
    store_dtype_from_name,
)


ACTIVE_PROGRESS_BAR = None


def log(message: str) -> None:
    if ACTIVE_PROGRESS_BAR is not None:
        ACTIVE_PROGRESS_BAR.write(message)
    else:
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Top-K cond features to disk"
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--object-ids", type=str, nargs="*", default=None)
    parser.add_argument(
        "--object-ids-file",
        type=str,
        default=None,
        help="Optional newline-delimited object id list to avoid shell ARG_MAX limits.",
    )
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="CPU worker processes. Default: auto based on available CPUs.",
    )
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument(
        "--num-views",
        type=int,
        default=32,
        help="Number of camera views available per object (V).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="K: number of views kept per point (by confidence).",
    )
    parser.add_argument("--sensor-width", type=float, default=36.0)
    parser.add_argument("--focal-mm", type=float, default=35.0)
    parser.add_argument("--sigma-scale", type=float, default=0.5)
    parser.add_argument(
        "--base-margin",
        type=float,
        default=0.05,
        help="Lower bound on sigma in confidence; larger = smoother/fewer NaNs.",
    )
    parser.add_argument(
        "--sample-mode", type=str, default="nearest", choices=["nearest", "bilinear"]
    )
    parser.add_argument(
        "--point-cloud-prefer-extract-output",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Where to find merged_model_render_color.ply. "
        "'true' = strict extract_output, 'false' = strict object root.",
    )
    parser.add_argument(
        "--depth-source",
        type=str,
        default="view_ply",
        choices=["view_ply", "merged_ply"],
    )
    parser.add_argument(
        "--view-ply-prefer-extract-output",
        type=str,
        default="true",
        choices=["auto", "true", "false"],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--store-dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-name", type=str, default="summary.csv")
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def resolve_point_cloud_preference(mode: str) -> bool:
    return {"true": True, "false": False}[mode]


def resolve_point_cloud_path(
    dataset_root: Path,
    obj_id: str,
    *,
    prefer_extract_output: bool,
) -> Path:
    point_cloud_dir = Path(
        PBRConditionalDataset.resolve_point_cloud_dir(
            root_dir=str(dataset_root),
            obj_id=obj_id,
            prefer_extract_output=prefer_extract_output,
            fallback_to_other=False,
        )
    )
    return point_cloud_dir / "merged_model_render_color.ply"


def select_object_ids(
    dataset_root: Path,
    explicit_ids: List[str] | None,
    explicit_ids_file: Path | None,
    max_objects: int | None,
    *,
    shard_index: int,
    num_shards: int,
) -> List[str]:
    if explicit_ids and explicit_ids_file is not None:
        raise ValueError("Use only one of --object-ids or --object-ids-file")

    file_ids: List[str] | None = None
    if explicit_ids_file is not None:
        file_ids = [
            line.strip()
            for line in explicit_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if explicit_ids:
        available = set(list_object_ids(dataset_root))
        missing = [oid for oid in explicit_ids if oid not in available]
        if missing:
            raise FileNotFoundError(
                f"Object ids not found under {dataset_root}: {missing}"
            )
        object_ids = list(explicit_ids)
    elif file_ids:
        available = set(list_object_ids(dataset_root))
        missing = [oid for oid in file_ids if oid not in available]
        if missing:
            raise FileNotFoundError(
                f"Object ids not found under {dataset_root}: {missing[:32]}"
            )
        object_ids = list(file_ids)
    else:
        object_ids = list_object_ids(dataset_root)
    if max_objects is not None:
        object_ids = object_ids[:max_objects]
    return apply_round_robin_shard(
        object_ids, shard_index=shard_index, num_shards=num_shards
    )


def resolve_summary_name(
    summary_name: str, *, shard_index: int, num_shards: int
) -> str:
    if num_shards == 1 or summary_name != "summary.csv":
        return summary_name
    pad = max(2, len(str(num_shards - 1)), len(str(num_shards)))
    return f"summary.shard_{shard_index:0{pad}d}_of_{num_shards:0{pad}d}.csv"


def validate_metadata_file(metadata_path: Path, *, num_views: int) -> List[str]:
    if not metadata_path.exists():
        return [f"missing metadata: {metadata_path}"]

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return [f"invalid metadata: {metadata_path} ({exc})"]

    if not isinstance(metadata, list):
        return [f"invalid metadata format: {metadata_path} (expected list)"]

    view_ids = set()
    errors: List[str] = []
    for idx, item in enumerate(metadata):
        if not isinstance(item, dict):
            errors.append(f"invalid metadata entry[{idx}] in {metadata_path}")
            continue
        if "view_id" not in item:
            errors.append(f"missing view_id in metadata entry[{idx}] @ {metadata_path}")
            continue
        if "camera_position" not in item:
            errors.append(
                f"missing camera_position for view_id={item['view_id']} @ {metadata_path}"
            )
            continue
        try:
            view_ids.add(int(item["view_id"]))
        except Exception:  # noqa: BLE001
            errors.append(f"non-integer view_id={item['view_id']!r} @ {metadata_path}")

    missing_view_ids = [str(v) for v in range(1, num_views + 1) if v not in view_ids]
    if missing_view_ids:
        preview = ",".join(missing_view_ids[:8])
        suffix = "..." if len(missing_view_ids) > 8 else ""
        errors.append(
            f"metadata missing view_id(s): {preview}{suffix} @ {metadata_path}"
        )
    return errors


def precheck_object_inputs(
    obj_id: str,
    *,
    dataset_root: Path,
    num_views: int,
    point_cloud_prefer_extract_output: bool,
    depth_source: str,
    view_ply_prefer_extract_output: bool | None,
) -> List[str]:
    obj_dir = dataset_root / obj_id
    errors: List[str] = []

    point_cloud_path = resolve_point_cloud_path(
        dataset_root,
        obj_id,
        prefer_extract_output=point_cloud_prefer_extract_output,
    )
    if not point_cloud_path.exists():
        errors.append(f"missing merged point cloud: {point_cloud_path}")

    errors.extend(
        validate_metadata_file(obj_dir / "metadata.json", num_views=num_views)
    )

    for view_idx in range(1, num_views + 1):
        render_path = obj_dir / f"Render_{view_idx}.png"
        if not render_path.exists():
            errors.append(f"missing render: {render_path}")

    if depth_source == "view_ply":
        try:
            missing = ViewPointCloudReader(
                obj_dir, view_ply_prefer_extract_output
            ).missing_views(num_views)
            errors.extend(
                f"missing view ply/data: {obj_dir}, view={v}" for v in missing
            )
        except (OSError, ValueError) as exc:
            errors.append(f"invalid compact view data: {obj_dir}: {exc}")

    return errors


def build_precheck_exception(errors: List[str]) -> Exception:
    preview = errors[:8]
    remaining = len(errors) - len(preview)
    detail = "; ".join(preview)
    if remaining > 0:
        detail += f"; ... (+{remaining} more)"
    return FileNotFoundError(f"precheck failed: {detail}")


def load_view_ply_rgb01(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    return view_vertices_xyz_rgb01(ply["vertex"].data, str(ply_path))


def view_vertices_xyz_rgb01(vertex, source) -> tuple[np.ndarray, np.ndarray]:
    field_names = set(vertex.dtype.names or ())

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(field_names):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(
            np.float32
        )
    elif {"albedo_red", "albedo_green", "albedo_blue"}.issubset(field_names):
        rgb = np.stack(
            [vertex["albedo_red"], vertex["albedo_green"], vertex["albedo_blue"]],
            axis=1,
        ).astype(np.float32)
    else:
        raise KeyError(f"{source} missing red/green/blue or albedo fields")

    if rgb.size > 0 and float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    return xyz, np.clip(rgb, 0.0, 1.0)


def load_point_cloud_rgb01(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    field_names = set(vertex.data.dtype.names or ())

    if {"red", "green", "blue"}.issubset(field_names):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(
            np.float32
        )
    elif {"albedo_red", "albedo_green", "albedo_blue"}.issubset(field_names):
        rgb = np.stack(
            [vertex["albedo_red"], vertex["albedo_green"], vertex["albedo_blue"]],
            axis=1,
        ).astype(np.float32)
    else:
        raise KeyError(f"{ply_path} missing red/green/blue or albedo fields")

    if rgb.size > 0 and float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def build_primary_rgb_from_view_source(
    *,
    query_points: torch.Tensor,
    view_source_1based: np.ndarray,
    point_cloud_path: Path,
    view_ply_prefer_extract_output: bool | None,
    allow_merged_ply_rgb_fallback: bool = False,
) -> tuple[torch.Tensor, Dict[str, float]]:
    query_xyz = query_points.detach().cpu().numpy().astype(np.float32, copy=False)
    num_points = int(query_xyz.shape[0])
    primary_rgb = np.zeros((num_points, 3), dtype=np.float32)
    nn_dist = np.full((num_points,), np.inf, dtype=np.float32)
    merged_rgb: np.ndarray | None = None
    fallback_points = 0

    base_dir = str(point_cloud_path.parent)
    view_reader = ViewPointCloudReader(base_dir, view_ply_prefer_extract_output)
    for view_idx in sorted(np.unique(view_source_1based).tolist()):
        view_idx = int(view_idx)
        if view_idx <= 0:
            raise ValueError(
                f"view_source must be 1-based positive ids, got {view_idx}"
            )

        view_ply_path = Path(
            PBRConditionalDataset.resolve_view_ply_path(
                base_dir,
                view_idx,
                prefer_extract_output=view_ply_prefer_extract_output,
            )
        )
        try:
            vertex = view_reader.read(view_idx)
        except FileNotFoundError:
            if not allow_merged_ply_rgb_fallback:
                raise FileNotFoundError(
                    f"Missing source view ply for slot-0 RGB: {view_ply_path}"
                )
            if merged_rgb is None:
                merged_rgb = load_point_cloud_rgb01(point_cloud_path)
                if int(merged_rgb.shape[0]) != num_points:
                    raise ValueError(
                        f"merged PLY RGB fallback requires matching point order/count: "
                        f"rgb={int(merged_rgb.shape[0])} query={num_points} @ {point_cloud_path}"
                    )
            mask = view_source_1based == view_idx
            idx = np.flatnonzero(mask)
            nn_dist[idx] = 0.0
            primary_rgb[idx] = merged_rgb[idx]
            fallback_points += int(idx.shape[0])
            continue

        view_xyz, view_rgb = view_vertices_xyz_rgb01(vertex, str(view_ply_path))
        tree = cKDTree(view_xyz)
        mask = view_source_1based == view_idx
        idx = np.flatnonzero(mask)
        d, nn_idx = tree.query(query_xyz[idx], k=1)
        nn_dist[idx] = d.astype(np.float32)
        primary_rgb[idx] = view_rgb[nn_idx]

    if not np.isfinite(nn_dist).all():
        bad = int((~np.isfinite(nn_dist)).sum())
        raise ValueError(
            f"slot-0 RGB nearest-neighbour mapping failed for {bad} points"
        )

    stats = {
        "slot0_rgb_view_ply_nn_mean": float(nn_dist.mean()),
        "slot0_rgb_view_ply_nn_max": float(nn_dist.max()),
        "slot0_rgb_view_ply_nn_p95": float(np.quantile(nn_dist, 0.95)),
        "slot0_rgb_merged_ply_fallback_points": float(fallback_points),
    }
    if fallback_points:
        stats["slot0_rgb_source"] = "view_ply_with_merged_ply_fallback"
    else:
        stats["slot0_rgb_source"] = "view_ply"
    primary_rgb_signed = torch.from_numpy(primary_rgb.T * 2.0 - 1.0).to(
        device=query_points.device,
        dtype=torch.float32,
    )  # [3, N] in [-1, 1]
    return primary_rgb_signed, stats


@torch.no_grad()
def process_object(
    obj_id: str,
    *,
    dataset_root: Path,
    output_dir: Path,
    num_views: int,
    top_k: int,
    sensor_width: float,
    focal_mm: float,
    sigma_scale: float,
    base_margin: float,
    sample_mode: str,
    point_cloud_prefer_extract_output: bool,
    depth_source: str,
    view_ply_prefer_extract_output: bool | None,
    device: torch.device,
    store_dtype: torch.dtype,
) -> Dict[str, object]:
    obj_dir = dataset_root / obj_id
    point_cloud_path = resolve_point_cloud_path(
        dataset_root,
        obj_id,
        prefer_extract_output=point_cloud_prefer_extract_output,
    )
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Missing merged point cloud: {point_cloud_path}")

    points_cpu = load_points_from_ply(point_cloud_path)
    point_cloud_xyz_fingerprint = PBRConditionalDataset.compute_ordered_xyz_fingerprint(
        points_cpu,
        layout="points_first",
    )
    points = points_cpu.to(device=device, dtype=torch.float32)
    view_source = PBRConditionalDataset.load_view_source_data(str(point_cloud_path))
    if int(view_source.shape[0]) != int(points.shape[0]):
        raise ValueError(
            f"view_source length mismatch for {obj_id}: "
            f"points={int(points.shape[0])} vs view_source={int(view_source.shape[0])}"
        )
    primary_view_id = torch.from_numpy(view_source.astype("int64", copy=False) - 1).to(
        device=device,
        dtype=torch.long,
    )
    primary_rgb, primary_rgb_stats = build_primary_rgb_from_view_source(
        query_points=points,
        view_source_1based=view_source,
        point_cloud_path=point_cloud_path,
        view_ply_prefer_extract_output=view_ply_prefer_extract_output,
        allow_merged_ply_rgb_fallback=(depth_source == "merged_ply"),
    )

    intrinsics, extrinsics, height, width = load_cameras_from_metadata(
        obj_dir,
        num_views=num_views,
        sensor_width=sensor_width,
        focal_mm=focal_mm,
    )
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=device, dtype=torch.float32)
    rgb_maps = load_rgb_maps(obj_dir, num_views=num_views).to(
        device=device, dtype=torch.float32
    )
    depth_maps = build_depth_maps(
        obj_dir=obj_dir,
        query_points=points,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        num_views=num_views,
        height=height,
        width=width,
        depth_source=depth_source,
        view_ply_prefer_extract_output=view_ply_prefer_extract_output,
        device=device,
    )

    result = build_cond_point_features_topk(
        points=points,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        depth_maps=depth_maps,
        rgb_maps=rgb_maps,
        top_k=top_k,
        primary_view_id=primary_view_id,
        primary_rgb=primary_rgb,
        sigma_scale=sigma_scale,
        base_margin=base_margin,
        sample_mode=sample_mode,
        return_debug=False,
    )

    topk_rgb = result["topk_rgb"].detach().cpu().to(dtype=store_dtype)  # [K, 3, N]
    topk_conf = result["topk_conf"].detach().cpu().to(dtype=store_dtype)  # [K, N]
    topk_view_id = result["topk_view_id"].detach().cpu().to(dtype=torch.int16)  # [K, N]

    # Summary statistics (for sanity-checking outputs).
    best_conf = topk_conf[0].float()
    mean_best_conf = float(best_conf.mean().item())
    slot0_conf_all_one = bool(torch.allclose(best_conf, torch.ones_like(best_conf)))
    slot0_view_source_match_fraction = float(
        (topk_view_id[0].to(dtype=torch.long) == primary_view_id.detach().cpu())
        .float()
        .mean()
        .item()
    )
    valid_mask = topk_conf > -1.0 + 1e-3
    frac_points_all_slots_valid = float(valid_mask.all(dim=0).float().mean().item())
    mean_slots_valid_per_point = float(valid_mask.sum(dim=0).float().mean().item())

    # Cross-view RGB consistency on points whose all K slots are valid.
    # Confidence can look perfect (Δz < σ everywhere) while the sampled colours
    # actually disagree a lot — GS depth noise + hole-fill fake surfaces will
    # show up here. Range is rescaled to the [0, 1] image domain.
    cross_view_rgb_range_mean = float("nan")
    cross_view_rgb_range_p50 = float("nan")
    cross_view_rgb_range_p90 = float("nan")
    all_valid = valid_mask.all(dim=0)
    if int(all_valid.sum().item()) >= 10:
        rgb_sel = topk_rgb.float()[:, :, all_valid]  # [K, 3, M], values in [-1, 1]
        per_point_range = (rgb_sel.max(dim=0).values - rgb_sel.min(dim=0).values).mean(
            dim=0
        ) / 2.0
        cross_view_rgb_range_mean = float(per_point_range.mean().item())
        cross_view_rgb_range_p50 = float(per_point_range.quantile(0.50).item())
        cross_view_rgb_range_p90 = float(per_point_range.quantile(0.90).item())

    payload = {
        "format": "topk_v1",
        "k": int(top_k),
        "num_views": int(num_views),
        "topk_rgb": topk_rgb,
        "topk_conf": topk_conf,
        "topk_view_id": topk_view_id,
        "object_id": obj_id,
        "point_cloud_path": str(point_cloud_path),
        "num_points": int(points.shape[0]),
        "point_cloud_prefer_extract_output": bool(point_cloud_prefer_extract_output),
        "point_cloud_xyz_fingerprint": point_cloud_xyz_fingerprint,
        "depth_source": depth_source,
        "view_ply_prefer_extract_output": view_ply_prefer_extract_output,
        "sample_mode": sample_mode,
        "sigma_scale": float(sigma_scale),
        "base_margin": float(base_margin),
        "slot0_semantics": (
            "forced_view_source_with_merged_ply_rgb"
            if primary_rgb_stats.get("slot0_rgb_source")
            == "view_ply_with_merged_ply_fallback"
            else "forced_view_source_with_view_ply_rgb"
        ),
        "view_source_indexing_in_ply": "1-based",
        "mean_best_conf": mean_best_conf,
        "slot0_conf_all_one": slot0_conf_all_one,
        "slot0_view_source_match_fraction": slot0_view_source_match_fraction,
        **primary_rgb_stats,
        "frac_points_all_slots_valid": frac_points_all_slots_valid,
        "mean_slots_valid_per_point": mean_slots_valid_per_point,
        "cross_view_rgb_range_mean": cross_view_rgb_range_mean,
        "cross_view_rgb_range_p50": cross_view_rgb_range_p50,
        "cross_view_rgb_range_p90": cross_view_rgb_range_p90,
    }
    output_path = output_dir / f"{obj_id}.pt"
    torch.save(payload, output_path)

    return {
        "object_id": obj_id,
        "output_path": str(output_path),
        "num_points": int(points.shape[0]),
        "k": int(top_k),
        "num_views": int(num_views),
        "mean_best_conf": mean_best_conf,
        "frac_points_all_slots_valid": frac_points_all_slots_valid,
        "mean_slots_valid_per_point": mean_slots_valid_per_point,
        "cross_view_rgb_range_mean": cross_view_rgb_range_mean,
        "cross_view_rgb_range_p50": cross_view_rgb_range_p50,
        "cross_view_rgb_range_p90": cross_view_rgb_range_p90,
    }


def build_task_config(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    output_dir: Path,
    point_cloud_prefer_extract_output: bool,
    view_ply_prefer_extract_output: bool | None,
) -> Dict[str, object]:
    return {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "num_views": int(args.num_views),
        "top_k": int(args.top_k),
        "sensor_width": float(args.sensor_width),
        "focal_mm": float(args.focal_mm),
        "sigma_scale": float(args.sigma_scale),
        "base_margin": float(args.base_margin),
        "sample_mode": args.sample_mode,
        "point_cloud_prefer_extract_output": bool(point_cloud_prefer_extract_output),
        "depth_source": args.depth_source,
        "view_ply_prefer_extract_output": view_ply_prefer_extract_output,
        "device": args.device,
        "store_dtype": args.store_dtype,
    }


def run_object_task(obj_id: str, task_config: Dict[str, object]) -> Dict[str, object]:
    return process_object(
        obj_id,
        dataset_root=Path(str(task_config["dataset_root"])),
        output_dir=Path(str(task_config["output_dir"])),
        num_views=int(task_config["num_views"]),
        top_k=int(task_config["top_k"]),
        sensor_width=float(task_config["sensor_width"]),
        focal_mm=float(task_config["focal_mm"]),
        sigma_scale=float(task_config["sigma_scale"]),
        base_margin=float(task_config["base_margin"]),
        sample_mode=str(task_config["sample_mode"]),
        point_cloud_prefer_extract_output=bool(
            task_config["point_cloud_prefer_extract_output"]
        ),
        depth_source=str(task_config["depth_source"]),
        view_ply_prefer_extract_output=task_config["view_ply_prefer_extract_output"],
        device=torch.device(str(task_config["device"])),
        store_dtype=store_dtype_from_name(str(task_config["store_dtype"])),
    )


def build_skip_row(obj_id: str, output_path: Path) -> Dict[str, object]:
    return {
        "object_id": obj_id,
        "status": "skipped",
        "output_path": str(output_path),
        "num_points": "",
        "k": "",
        "num_views": "",
        "mean_best_conf": "",
        "frac_points_all_slots_valid": "",
        "mean_slots_valid_per_point": "",
        "cross_view_rgb_range_mean": "",
        "cross_view_rgb_range_p50": "",
        "cross_view_rgb_range_p90": "",
        "error": "",
    }


def build_ok_row(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "object_id": result["object_id"],
        "status": "ok",
        "output_path": result["output_path"],
        "num_points": result["num_points"],
        "k": result["k"],
        "num_views": result["num_views"],
        "mean_best_conf": result["mean_best_conf"],
        "frac_points_all_slots_valid": result["frac_points_all_slots_valid"],
        "mean_slots_valid_per_point": result["mean_slots_valid_per_point"],
        "cross_view_rgb_range_mean": result["cross_view_rgb_range_mean"],
        "cross_view_rgb_range_p50": result["cross_view_rgb_range_p50"],
        "cross_view_rgb_range_p90": result["cross_view_rgb_range_p90"],
        "error": "",
    }


def build_error_row(
    obj_id: str, output_path: Path, exc: Exception
) -> Dict[str, object]:
    return {
        "object_id": obj_id,
        "status": "error",
        "output_path": str(output_path),
        "num_points": "",
        "k": "",
        "num_views": "",
        "mean_best_conf": "",
        "frac_points_all_slots_valid": "",
        "mean_slots_valid_per_point": "",
        "cross_view_rgb_range_mean": "",
        "cross_view_rgb_range_p50": "",
        "cross_view_rgb_range_p90": "",
        "error": repr(exc),
    }


def collect_pending_objects(
    object_ids: List[str],
    *,
    output_dir: Path,
    overwrite: bool,
):
    pending_ids: List[str] = []
    rows_by_object: Dict[str, Dict[str, object]] = {}
    for obj_id in object_ids:
        output_path = output_dir / f"{obj_id}.pt"
        if output_path.exists() and not overwrite:
            rows_by_object[obj_id] = build_skip_row(obj_id, output_path)
        else:
            pending_ids.append(obj_id)
    return pending_ids, rows_by_object


def precheck_pending_objects(
    object_ids: List[str],
    *,
    dataset_root: Path,
    output_dir: Path,
    num_views: int,
    point_cloud_prefer_extract_output: bool,
    depth_source: str,
    view_ply_prefer_extract_output: bool | None,
) -> tuple[List[str], Dict[str, Dict[str, object]]]:
    runnable_ids: List[str] = []
    rows_by_object: Dict[str, Dict[str, object]] = {}
    for obj_id in object_ids:
        errors = precheck_object_inputs(
            obj_id,
            dataset_root=dataset_root,
            num_views=num_views,
            point_cloud_prefer_extract_output=point_cloud_prefer_extract_output,
            depth_source=depth_source,
            view_ply_prefer_extract_output=view_ply_prefer_extract_output,
        )
        if errors:
            rows_by_object[obj_id] = build_error_row(
                obj_id,
                output_dir / f"{obj_id}.pt",
                build_precheck_exception(errors),
            )
            continue
        runnable_ids.append(obj_id)
    return runnable_ids, rows_by_object


def resolve_num_workers(requested: int | None, *, num_pending: int):
    available = get_available_cpu_count()
    if num_pending <= 0:
        return 1, False, available
    if requested is not None:
        return max(1, requested), False, available
    auto = 1 if available <= 2 else min(16, available - 1)
    auto = min(auto, num_pending)
    return max(1, auto), True, available


def save_summary_csv(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "object_id",
        "status",
        "output_path",
        "num_points",
        "k",
        "num_views",
        "mean_best_conf",
        "frac_points_all_slots_valid",
        "mean_slots_valid_per_point",
        "cross_view_rgb_range_mean",
        "cross_view_rgb_range_p50",
        "cross_view_rgb_range_p90",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    global ACTIVE_PROGRESS_BAR

    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.num_workers is not None and args.num_workers <= 0:
        raise ValueError(f"num_workers must be >= 1, got {args.num_workers}")
    if args.num_workers is not None and args.num_workers > 1 and device.type != "cpu":
        raise ValueError("--num-workers > 1 only supports CPU today.")
    if args.top_k <= 0 or args.top_k > args.num_views:
        raise ValueError(
            f"top_k must satisfy 1 <= top_k <= num_views, got {args.top_k}/{args.num_views}"
        )

    view_ply_pref = resolve_view_ply_preference(args.view_ply_prefer_extract_output)
    pc_pref = resolve_point_cloud_preference(args.point_cloud_prefer_extract_output)

    object_ids = select_object_ids(
        dataset_root,
        args.object_ids,
        Path(args.object_ids_file) if args.object_ids_file else None,
        args.max_objects,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    summary_name = resolve_summary_name(
        args.summary_name, shard_index=args.shard_index, num_shards=args.num_shards
    )

    log(
        f"Top-K precompute: K={args.top_k}, V={args.num_views}, objects={len(object_ids)} "
        f"from {dataset_root} -> {output_dir}"
    )

    pending_ids, rows_by_object = collect_pending_objects(
        object_ids,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    runnable_ids, precheck_rows = precheck_pending_objects(
        pending_ids,
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_views=args.num_views,
        point_cloud_prefer_extract_output=pc_pref,
        depth_source=args.depth_source,
        view_ply_prefer_extract_output=view_ply_pref,
    )
    rows_by_object.update(precheck_rows)
    pending_ids = runnable_ids

    num_workers, auto_selected, available_cpus = resolve_num_workers(
        args.num_workers,
        num_pending=len(pending_ids),
    )
    if device.type != "cpu":
        num_workers = 1
        auto_selected = False
    threads_per_worker = resolve_threads_per_worker(args, num_workers=num_workers)
    if auto_selected and pending_ids:
        log(
            f"Auto-selected workers={num_workers} from available_cpus={available_cpus} "
            f"for pending_objects={len(pending_ids)}"
        )
    existing_skip_count = sum(
        row["status"] == "skipped" for row in rows_by_object.values()
    )
    precheck_error_count = sum(
        row["status"] == "error" for row in rows_by_object.values()
    )
    if existing_skip_count:
        log(f"Pre-scan: skipping {existing_skip_count} existing output file(s)")
    if precheck_error_count:
        log(
            f"Pre-scan: filtered {precheck_error_count} object(s) with missing/invalid inputs "
            f"before execution"
        )

    task_config = build_task_config(
        args,
        dataset_root=dataset_root,
        output_dir=output_dir,
        point_cloud_prefer_extract_output=pc_pref,
        view_ply_prefer_extract_output=view_ply_pref,
    )

    num_ok = 0
    num_error = precheck_error_count
    num_skipped = existing_skip_count
    progress_bar = None
    if pending_ids:
        progress_bar = tqdm(
            total=len(pending_ids),
            desc="Objects",
            unit="obj",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        ACTIVE_PROGRESS_BAR = progress_bar

    start_time = time.perf_counter()
    try:
        if num_workers == 1:
            for idx, obj_id in enumerate(pending_ids, start=1):
                output_path = output_dir / f"{obj_id}.pt"
                try:
                    result = run_object_task(obj_id, task_config)
                    row = build_ok_row(result)
                    num_ok += 1
                except Exception as exc:  # noqa: BLE001
                    row = build_error_row(obj_id, output_path, exc)
                    num_error += 1
                    print(f"[ERROR] {obj_id}: {exc}", file=sys.stderr, flush=True)
                rows_by_object[obj_id] = row
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(
                        f"ok={num_ok} skip={num_skipped} err={num_error}"
                    )
                if (
                    idx == 1
                    or idx % max(args.print_every, 1) == 0
                    or idx == len(pending_ids)
                ):
                    elapsed = time.perf_counter() - start_time
                    if row["status"] == "ok":
                        log(
                            f"[{idx}/{len(pending_ids)}] {obj_id} points={row['num_points']} "
                            f"mean_best_conf={float(row['mean_best_conf']):.4f} "
                            f"mean_slots_valid={float(row['mean_slots_valid_per_point']):.2f}/{args.top_k} "
                            f"elapsed={elapsed:.1f}s"
                        )
                    else:
                        log(
                            f"[{idx}/{len(pending_ids)}] {obj_id} status={row['status']} "
                            f"err={row['error']}"
                        )
        elif pending_ids:
            assert threads_per_worker is not None
            configure_parallel_cpu_environment(threads_per_worker)
            max_workers = min(num_workers, len(pending_ids))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=init_worker_process,
                initargs=(threads_per_worker,),
            ) as executor:
                future_to_obj = {
                    executor.submit(run_object_task, obj_id, task_config): obj_id
                    for obj_id in pending_ids
                }
                for completed, future in enumerate(
                    as_completed(future_to_obj), start=1
                ):
                    obj_id = future_to_obj[future]
                    output_path = output_dir / f"{obj_id}.pt"
                    try:
                        result = future.result()
                        row = build_ok_row(result)
                        num_ok += 1
                    except Exception as exc:  # noqa: BLE001
                        row = build_error_row(obj_id, output_path, exc)
                        num_error += 1
                        print(f"[ERROR] {obj_id}: {exc}", file=sys.stderr, flush=True)
                    rows_by_object[obj_id] = row
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(
                            f"ok={num_ok} skip={num_skipped} err={num_error}"
                        )
                    if (
                        completed == 1
                        or completed % max(args.print_every, 1) == 0
                        or completed == len(pending_ids)
                    ):
                        elapsed = time.perf_counter() - start_time
                        if row["status"] == "ok":
                            log(
                                f"[{completed}/{len(pending_ids)}] {obj_id} ok "
                                f"mean_best_conf={float(row['mean_best_conf']):.4f} "
                                f"elapsed={elapsed:.1f}s"
                            )
                        else:
                            log(
                                f"[{completed}/{len(pending_ids)}] {obj_id} "
                                f"status={row['status']} err={row['error']}"
                            )
    finally:
        if progress_bar is not None:
            progress_bar.close()
            ACTIVE_PROGRESS_BAR = None

    rows = [rows_by_object[oid] for oid in object_ids]
    summary_path = output_dir / summary_name
    save_summary_csv(summary_path, rows)
    num_ok = sum(r["status"] == "ok" for r in rows)
    num_skipped = sum(r["status"] == "skipped" for r in rows)
    num_error = sum(r["status"] == "error" for r in rows)
    elapsed = time.perf_counter() - start_time
    log(
        f"Finished. ok={num_ok} skipped={num_skipped} error={num_error} "
        f"| summary={summary_path} | elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
