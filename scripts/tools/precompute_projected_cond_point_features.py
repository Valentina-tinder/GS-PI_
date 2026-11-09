#!/usr/bin/env python3
"""Offline precompute projected cond_point_colors for PBRConditionalDataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gs_pi.data.cond_point_features import build_cond_point_features_2d_projection
from gs_pi.data.srn import PBRConditionalDataset
from gs_pi.pointcloud_storage import ViewPointCloudReader


CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

ACTIVE_PROGRESS_BAR = None


def log(message: str) -> None:
    if ACTIVE_PROGRESS_BAR is not None:
        ACTIVE_PROGRESS_BAR.write(message)
    else:
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute projected cond_point_colors to disk"
    )
    parser.add_argument(
        "--dataset-root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store <obj_id>.pt feature files",
    )
    parser.add_argument(
        "--object-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit object ids to process",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Optional max number of objects to process",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to process after object selection",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total number of round-robin shards"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for CPU parallelism; defaults to an automatic CPU-based value",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="CPU threads reserved per worker; defaults to 1 when --num-workers > 1",
    )
    parser.add_argument(
        "--num-views", type=int, default=6, help="Number of camera views per object"
    )
    parser.add_argument(
        "--sensor-width", type=float, default=36.0, help="Camera sensor width in mm"
    )
    parser.add_argument(
        "--focal-mm", type=float, default=35.0, help="Camera focal length in mm"
    )
    parser.add_argument(
        "--cond-point-dim",
        type=int,
        default=24,
        help="Expected cond_point_colors channel count",
    )
    parser.add_argument(
        "--point-cloud-prefer-extract-output",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Point source for merged_model_render_color.ply: auto=cond_point_dim==24, true=extract_output first, false=root only",
    )
    parser.add_argument(
        "--depth-source",
        type=str,
        default="view_ply",
        choices=["view_ply", "merged_ply"],
        help="Point source used to build depth maps",
    )
    parser.add_argument(
        "--view-ply-prefer-extract-output",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="For --depth-source view_ply: auto=prefer extract_output then root, true=extract_output only, false=root only",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear"],
        help="grid_sample mode for projected feature generation",
    )
    parser.add_argument(
        "--enable-rgb-soft-mask",
        dest="rgb_soft_mask",
        action="store_true",
        help="Enable optional RGB soft masking for views with confidence > 0",
    )
    parser.add_argument(
        "--disable-rgb-soft-mask",
        dest="rgb_soft_mask",
        action="store_false",
        help="Disable soft masking and use the hard rule: confidence <= 0 -> RGB = 0",
    )
    parser.add_argument(
        "--rgb-soft-mask-floor",
        type=float,
        default=0.0,
        help="Optional lower bound for RGB weight when soft mask is enabled",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--store-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Tensor dtype stored on disk",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <obj_id>.pt outputs",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary.csv",
        help="Summary CSV name under output dir",
    )
    parser.add_argument(
        "--print-every", type=int, default=20, help="Progress print interval"
    )
    parser.set_defaults(rgb_soft_mask=False)
    return parser.parse_args()


def list_object_ids(dataset_root: Path) -> List[str]:
    return sorted(p.name for p in dataset_root.iterdir() if p.is_dir())


def apply_round_robin_shard(
    object_ids: List[str], shard_index: int, num_shards: int
) -> List[str]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must satisfy 0 <= shard_index < num_shards, got {shard_index}/{num_shards}"
        )
    if num_shards == 1:
        return object_ids

    # Round-robin sharding keeps global sorted order while roughly balancing work.
    return [
        obj_id
        for idx, obj_id in enumerate(object_ids)
        if idx % num_shards == shard_index
    ]


def select_object_ids(
    dataset_root: Path,
    explicit_ids: List[str] | None,
    max_objects: int | None,
    *,
    shard_index: int,
    num_shards: int,
) -> List[str]:
    if explicit_ids:
        available = set(list_object_ids(dataset_root))
        missing = [obj_id for obj_id in explicit_ids if obj_id not in available]
        if missing:
            raise FileNotFoundError(
                f"Object ids not found under {dataset_root}: {missing}"
            )
        object_ids = list(explicit_ids)
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


def store_dtype_from_name(store_dtype: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[store_dtype]


def get_available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def resolve_num_workers(
    requested_num_workers: int | None,
    *,
    num_pending_objects: int,
) -> tuple[int, bool, int]:
    available_cpus = get_available_cpu_count()
    if num_pending_objects <= 0:
        return 1, requested_num_workers is None, available_cpus
    if requested_num_workers is not None:
        return requested_num_workers, False, available_cpus

    if available_cpus <= 2:
        auto_workers = 1
    else:
        # Conservative default: leave a little headroom and cap process fan-out.
        auto_workers = min(16, available_cpus - 1)
    auto_workers = min(auto_workers, num_pending_objects)
    return max(1, auto_workers), True, available_cpus


def resolve_threads_per_worker(
    args: argparse.Namespace, *, num_workers: int
) -> int | None:
    if args.threads_per_worker is not None:
        if args.threads_per_worker <= 0:
            raise ValueError(
                f"threads_per_worker must be >= 1, got {args.threads_per_worker}"
            )
        return args.threads_per_worker
    if num_workers > 1:
        return 1
    return None


def validate_runtime_args(args: argparse.Namespace, device: torch.device) -> None:
    if args.num_workers is not None and args.num_workers <= 0:
        raise ValueError(f"num_workers must be >= 1, got {args.num_workers}")
    if args.num_workers is not None and args.num_workers > 1 and device.type != "cpu":
        raise ValueError(
            f"--num-workers only supports CPU execution today; got device={args.device!r}. "
            "Use --device cpu or launch multiple shards manually for cuda."
        )
    if not 0.0 <= args.rgb_soft_mask_floor <= 1.0:
        raise ValueError(
            f"--rgb-soft-mask-floor must be within [0, 1], got {args.rgb_soft_mask_floor}"
        )


def resolve_view_ply_preference(mode: str) -> bool | None:
    if mode == "auto":
        return None
    if mode == "true":
        return True
    if mode == "false":
        return False
    raise ValueError(f"Unsupported view_ply preference mode: {mode!r}")


def resolve_point_cloud_preference(mode: str, *, cond_point_dim: int) -> bool:
    if mode == "auto":
        return cond_point_dim == 24
    if mode == "true":
        return True
    if mode == "false":
        return False
    raise ValueError(f"Unsupported point cloud preference mode: {mode!r}")


def configure_parallel_cpu_environment(threads_per_worker: int) -> None:
    value = str(max(1, threads_per_worker))
    for env_name in CPU_THREAD_ENV_VARS:
        os.environ[env_name] = value


def init_worker_process(threads_per_worker: int | None) -> None:
    if threads_per_worker is None:
        return

    threads = max(1, int(threads_per_worker))
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_points_from_ply(ply_path: Path) -> torch.Tensor:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float32
    )
    return torch.from_numpy(points)


def load_cameras_from_metadata(
    obj_dir: Path,
    *,
    num_views: int,
    sensor_width: float,
    focal_mm: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    with (obj_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    sample_image = Image.open(obj_dir / "Render_1.png").convert("RGB")
    width, height = sample_image.size
    fx = (focal_mm / sensor_width) * width
    fy = fx
    cx = width * 0.5
    cy = height * 0.5

    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(num_views, 1, 1)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    metadata_by_view = {int(item["view_id"]): item for item in metadata}
    extrinsics = []
    for view_idx in range(1, num_views + 1):
        if view_idx not in metadata_by_view:
            raise KeyError(f"Missing view_id={view_idx} in {obj_dir / 'metadata.json'}")
        position = torch.tensor(
            metadata_by_view[view_idx]["camera_position"], dtype=torch.float32
        )
        forward = -position
        forward = forward / forward.norm().clamp_min(1e-6)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        right = torch.cross(forward, up, dim=0)
        right = right / right.norm().clamp_min(1e-6)
        down = torch.cross(forward, right, dim=0)
        down = down / down.norm().clamp_min(1e-6)

        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = forward
        c2w[:3, 3] = position

        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = c2w[:3, :3].T
        w2c[:3, 3] = -w2c[:3, :3] @ c2w[:3, 3]
        extrinsics.append(w2c)

    return intrinsics, torch.stack(extrinsics, dim=0), height, width


def load_rgb_maps(obj_dir: Path, *, num_views: int) -> torch.Tensor:
    rgb_maps = []
    for view_idx in range(1, num_views + 1):
        image = Image.open(obj_dir / f"Render_{view_idx}.png").convert("RGB")
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        rgb_maps.append(rgb * 2.0 - 1.0)
    return torch.stack(rgb_maps, dim=0)


def rasterize_depth_map(
    points: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    *,
    height: int,
    width: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rasterize a single-view depth map with nearest-pixel z-buffer."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N, 3], got {tuple(points.shape)}")

    device = points.device
    dtype = points.dtype
    ones = torch.ones((points.shape[0], 1), device=device, dtype=dtype)
    points_h = torch.cat([points, ones], dim=-1)

    points_cam = torch.einsum("ij,nj->ni", extrinsic[:3, :], points_h)
    z_proj = points_cam[:, 2]
    safe_z = z_proj.clamp_min(eps)
    pixels_h = torch.einsum("ij,nj->ni", intrinsic, points_cam)
    u = pixels_h[:, 0] / safe_z
    v = pixels_h[:, 1] / safe_z

    x = torch.round(u).to(torch.long)
    y = torch.round(v).to(torch.long)
    valid = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(z_proj)
        & (z_proj > 0)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    depth_flat = torch.full((height * width,), float("inf"), device=device, dtype=dtype)
    if valid.any():
        linear_idx = y[valid] * width + x[valid]
        depth_flat.scatter_reduce_(
            0, linear_idx, z_proj[valid], reduce="amin", include_self=True
        )
    depth_flat[torch.isinf(depth_flat)] = 0.0
    return depth_flat.view(1, height, width)


def build_depth_maps(
    *,
    obj_dir: Path,
    query_points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    num_views: int,
    height: int,
    width: int,
    depth_source: str,
    view_ply_prefer_extract_output: bool | None,
    device: torch.device,
) -> torch.Tensor:
    depth_maps = []
    view_reader = ViewPointCloudReader(obj_dir, view_ply_prefer_extract_output)
    for view_idx in range(num_views):
        if depth_source == "view_ply":
            vertex = view_reader.read(view_idx + 1)
            depth_points = torch.from_numpy(
                np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
                    np.float32
                )
            )
        else:
            depth_points = query_points

        depth_map = rasterize_depth_map(
            depth_points.to(device=device, dtype=torch.float32),
            extrinsics[view_idx],
            intrinsics[view_idx],
            height=height,
            width=width,
        )
        depth_maps.append(depth_map)
    return torch.stack(depth_maps, dim=0)


def summarize_projected_features(debug: Dict[str, torch.Tensor]) -> Dict[str, float]:
    valid_counts = debug["valid_mask"].sum(dim=0).float()
    valid_conf = debug["confidence"][debug["valid_mask"]]
    return {
        "mean_valid_views": float(valid_counts.mean().item()),
        "min_valid_views": int(valid_counts.min().item()),
        "max_valid_views": int(valid_counts.max().item()),
        "mean_valid_conf": float(valid_conf.mean().item())
        if valid_conf.numel() > 0
        else float("nan"),
        "min_valid_conf": float(valid_conf.min().item())
        if valid_conf.numel() > 0
        else float("nan"),
        "max_valid_conf": float(valid_conf.max().item())
        if valid_conf.numel() > 0
        else float("nan"),
    }


@torch.no_grad()
def process_object(
    obj_id: str,
    *,
    dataset_root: Path,
    output_dir: Path,
    num_views: int,
    sensor_width: float,
    focal_mm: float,
    cond_point_dim: int,
    point_cloud_prefer_extract_output: bool,
    depth_source: str,
    view_ply_prefer_extract_output: bool | None,
    sample_mode: str,
    rgb_soft_mask: bool,
    rgb_soft_mask_floor: float,
    device: torch.device,
    store_dtype: torch.dtype,
) -> Dict[str, object]:
    obj_dir = dataset_root / obj_id
    point_cloud_path = Path(
        PBRConditionalDataset.resolve_point_cloud_path(
            root_dir=str(dataset_root),
            obj_id=obj_id,
            prefer_extract_output=point_cloud_prefer_extract_output,
        )
    )
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Missing merged point cloud: {point_cloud_path}")

    points = load_points_from_ply(point_cloud_path).to(
        device=device, dtype=torch.float32
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

    features, debug = build_cond_point_features_2d_projection(
        points,
        extrinsics,
        intrinsics,
        depth_maps,
        rgb_maps,
        sample_mode=sample_mode,
        rgb_soft_mask=rgb_soft_mask,
        rgb_soft_mask_floor=rgb_soft_mask_floor,
        return_debug=True,
    )
    summary = summarize_projected_features(debug)

    output_path = output_dir / f"{obj_id}.pt"
    payload = {
        "cond_point_colors": features.detach().cpu().to(dtype=store_dtype),
        "object_id": obj_id,
        "point_cloud_path": str(point_cloud_path),
        "num_points": int(points.shape[0]),
        "cond_point_dim": int(features.shape[0]),
        "point_cloud_prefer_extract_output": bool(point_cloud_prefer_extract_output),
        "depth_source": depth_source,
        "view_ply_prefer_extract_output": view_ply_prefer_extract_output,
        "sample_mode": sample_mode,
        "rgb_soft_mask": rgb_soft_mask,
        "rgb_soft_mask_floor": float(rgb_soft_mask_floor),
        **summary,
    }
    torch.save(payload, output_path)

    return {
        "object_id": obj_id,
        "output_path": str(output_path),
        "num_points": int(points.shape[0]),
        "feature_dim": int(features.shape[0]),
        **summary,
    }


def build_task_config(
    *,
    dataset_root: Path,
    output_dir: Path,
    num_views: int,
    sensor_width: float,
    focal_mm: float,
    cond_point_dim: int,
    point_cloud_prefer_extract_output: bool,
    depth_source: str,
    view_ply_prefer_extract_output: bool | None,
    sample_mode: str,
    rgb_soft_mask: bool,
    rgb_soft_mask_floor: float,
    device: str,
    store_dtype: str,
) -> Dict[str, object]:
    return {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "num_views": int(num_views),
        "sensor_width": float(sensor_width),
        "focal_mm": float(focal_mm),
        "cond_point_dim": int(cond_point_dim),
        "point_cloud_prefer_extract_output": bool(point_cloud_prefer_extract_output),
        "depth_source": depth_source,
        "view_ply_prefer_extract_output": view_ply_prefer_extract_output,
        "sample_mode": sample_mode,
        "rgb_soft_mask": bool(rgb_soft_mask),
        "rgb_soft_mask_floor": float(rgb_soft_mask_floor),
        "device": device,
        "store_dtype": store_dtype,
    }


def run_object_task(obj_id: str, task_config: Dict[str, object]) -> Dict[str, object]:
    return process_object(
        obj_id,
        dataset_root=Path(str(task_config["dataset_root"])),
        output_dir=Path(str(task_config["output_dir"])),
        num_views=int(task_config["num_views"]),
        sensor_width=float(task_config["sensor_width"]),
        focal_mm=float(task_config["focal_mm"]),
        cond_point_dim=int(task_config["cond_point_dim"]),
        point_cloud_prefer_extract_output=bool(
            task_config["point_cloud_prefer_extract_output"]
        ),
        depth_source=str(task_config["depth_source"]),
        view_ply_prefer_extract_output=task_config["view_ply_prefer_extract_output"],
        sample_mode=str(task_config["sample_mode"]),
        rgb_soft_mask=bool(task_config["rgb_soft_mask"]),
        rgb_soft_mask_floor=float(task_config["rgb_soft_mask_floor"]),
        device=torch.device(str(task_config["device"])),
        store_dtype=store_dtype_from_name(str(task_config["store_dtype"])),
    )


def build_skip_row(obj_id: str, output_path: Path) -> Dict[str, object]:
    return {
        "object_id": obj_id,
        "status": "skipped",
        "output_path": str(output_path),
        "num_points": "",
        "feature_dim": "",
        "mean_valid_views": "",
        "min_valid_views": "",
        "max_valid_views": "",
        "mean_valid_conf": "",
        "min_valid_conf": "",
        "max_valid_conf": "",
        "error": "",
    }


def build_ok_row(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "object_id": result["object_id"],
        "status": "ok",
        "output_path": result["output_path"],
        "num_points": result["num_points"],
        "feature_dim": result["feature_dim"],
        "mean_valid_views": result["mean_valid_views"],
        "min_valid_views": result["min_valid_views"],
        "max_valid_views": result["max_valid_views"],
        "mean_valid_conf": result["mean_valid_conf"],
        "min_valid_conf": result["min_valid_conf"],
        "max_valid_conf": result["max_valid_conf"],
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
        "feature_dim": "",
        "mean_valid_views": "",
        "min_valid_views": "",
        "max_valid_views": "",
        "mean_valid_conf": "",
        "min_valid_conf": "",
        "max_valid_conf": "",
        "error": repr(exc),
    }


def collect_pending_objects(
    object_ids: List[str],
    *,
    output_dir: Path,
    overwrite: bool,
) -> tuple[List[str], Dict[str, Dict[str, object]]]:
    pending_ids: List[str] = []
    rows_by_object: Dict[str, Dict[str, object]] = {}
    for obj_id in object_ids:
        output_path = output_dir / f"{obj_id}.pt"
        if output_path.exists() and not overwrite:
            rows_by_object[obj_id] = build_skip_row(obj_id, output_path)
        else:
            pending_ids.append(obj_id)
    return pending_ids, rows_by_object


def log_object_completion(
    row: Dict[str, object],
    *,
    completed: int,
    total: int,
    elapsed: float,
) -> None:
    status = row["status"]
    obj_id = str(row["object_id"])
    if status == "ok":
        log(
            f"[{completed}/{total}] {obj_id} -> {row['output_path']} "
            f"| points={row['num_points']} mean_conf={float(row['mean_valid_conf']):.4f} "
            f"| elapsed={elapsed:.1f}s"
        )
        return
    if status == "skipped":
        log(
            f"[{completed}/{total}] skipped existing {row['output_path']} | elapsed={elapsed:.1f}s"
        )
        return
    log(
        f"[{completed}/{total}] {obj_id} failed | error={row['error']} | elapsed={elapsed:.1f}s"
    )


def update_progress_bar(
    progress_bar: tqdm,
    *,
    num_ok: int,
    num_skipped: int,
    num_error: int,
) -> None:
    progress_bar.set_postfix_str(
        f"ok={num_ok} skip={num_skipped} err={num_error} remain={progress_bar.total - progress_bar.n}"
    )


def save_summary_csv(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "object_id",
        "status",
        "output_path",
        "num_points",
        "feature_dim",
        "mean_valid_views",
        "min_valid_views",
        "max_valid_views",
        "mean_valid_conf",
        "min_valid_conf",
        "max_valid_conf",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    global ACTIVE_PROGRESS_BAR

    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    validate_runtime_args(args, device)
    view_ply_prefer_extract_output = resolve_view_ply_preference(
        args.view_ply_prefer_extract_output
    )
    point_cloud_prefer_extract_output = resolve_point_cloud_preference(
        args.point_cloud_prefer_extract_output,
        cond_point_dim=args.cond_point_dim,
    )
    object_ids = select_object_ids(
        dataset_root,
        args.object_ids,
        args.max_objects,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    summary_name = resolve_summary_name(
        args.summary_name,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    log(
        f"Precomputing projected cond_point_colors for {len(object_ids)} object(s) "
        f"from {dataset_root} -> {output_dir}"
    )
    if args.num_shards > 1:
        preview = ", ".join(object_ids[:5]) if object_ids else "<empty>"
        suffix = " ..." if len(object_ids) > 5 else ""
        log(
            f"Shard assignment: shard={args.shard_index}/{args.num_shards} "
            f"| selected={len(object_ids)} | first_ids={preview}{suffix}"
        )

    pending_ids, rows_by_object = collect_pending_objects(
        object_ids,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    num_workers, auto_selected_workers, available_cpus = resolve_num_workers(
        args.num_workers,
        num_pending_objects=len(pending_ids),
    )
    if device.type != "cpu":
        num_workers = 1
        auto_selected_workers = False
    elif num_workers > 1:
        threads_per_worker = resolve_threads_per_worker(args, num_workers=num_workers)
        log(
            f"Parallel mode: workers={num_workers} "
            f"| threads_per_worker={threads_per_worker}"
        )
    else:
        threads_per_worker = resolve_threads_per_worker(args, num_workers=num_workers)
    if auto_selected_workers and pending_ids:
        log(
            f"Auto-selected workers={num_workers} from available_cpus={available_cpus} "
            f"for pending_objects={len(pending_ids)}"
        )
    if rows_by_object:
        log(f"Pre-scan: skipped {len(rows_by_object)} existing output file(s)")

    task_config = build_task_config(
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_views=args.num_views,
        sensor_width=args.sensor_width,
        focal_mm=args.focal_mm,
        cond_point_dim=args.cond_point_dim,
        point_cloud_prefer_extract_output=point_cloud_prefer_extract_output,
        depth_source=args.depth_source,
        view_ply_prefer_extract_output=view_ply_prefer_extract_output,
        sample_mode=args.sample_mode,
        rgb_soft_mask=args.rgb_soft_mask,
        rgb_soft_mask_floor=args.rgb_soft_mask_floor,
        device=args.device,
        store_dtype=args.store_dtype,
    )

    num_ok = 0
    num_error = 0
    num_skipped = len(rows_by_object)
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
        update_progress_bar(
            progress_bar,
            num_ok=num_ok,
            num_skipped=num_skipped,
            num_error=num_error,
        )

    start_time = time.perf_counter()
    progress_interval = max(args.print_every, 1)
    try:
        if num_workers == 1:
            for idx, obj_id in enumerate(pending_ids, start=1):
                if idx == 1 or (idx - 1) % progress_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    completed = len(rows_by_object) + 1
                    log(
                        f"[{completed}/{len(object_ids)}] processing {obj_id} | elapsed={elapsed:.1f}s"
                    )

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
                    update_progress_bar(
                        progress_bar,
                        num_ok=num_ok,
                        num_skipped=num_skipped,
                        num_error=num_error,
                    )
                if idx % progress_interval == 0 or idx == len(pending_ids):
                    elapsed = time.perf_counter() - start_time
                    log_object_completion(
                        row,
                        completed=len(rows_by_object),
                        total=len(object_ids),
                        elapsed=elapsed,
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
                for completed_pending, future in enumerate(
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
                        update_progress_bar(
                            progress_bar,
                            num_ok=num_ok,
                            num_skipped=num_skipped,
                            num_error=num_error,
                        )
                    if (
                        completed_pending == 1
                        or completed_pending % progress_interval == 0
                        or completed_pending == len(pending_ids)
                    ):
                        elapsed = time.perf_counter() - start_time
                        log_object_completion(
                            row,
                            completed=len(rows_by_object),
                            total=len(object_ids),
                            elapsed=elapsed,
                        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
            ACTIVE_PROGRESS_BAR = None

    missing_rows = [obj_id for obj_id in object_ids if obj_id not in rows_by_object]
    if missing_rows:
        raise RuntimeError(
            f"Missing summary rows for {len(missing_rows)} object(s): {missing_rows[:5]}"
        )

    rows = [rows_by_object[obj_id] for obj_id in object_ids]
    summary_path = output_dir / summary_name
    save_summary_csv(summary_path, rows)
    num_ok = sum(row["status"] == "ok" for row in rows)
    num_skipped = sum(row["status"] == "skipped" for row in rows)
    num_error = sum(row["status"] == "error" for row in rows)
    elapsed = time.perf_counter() - start_time
    log(
        f"Finished. ok={num_ok} skipped={num_skipped} error={num_error} "
        f"| summary={summary_path} | elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
