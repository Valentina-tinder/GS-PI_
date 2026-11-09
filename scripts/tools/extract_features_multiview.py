#!/usr/bin/env python3
"""DINOv2 多视角图像特征离线提取（per-view 4D 缓存格式）

与旧版 ``extract_features.py`` 的差别：
  - 支持任意视角数 V（默认 32），不再绑死 6 view。
  - 缓存格式从 [V*N, D] 的扁平张量改成 [V, H, W, D] 的 per-view patch grid，
    便于训练时按需做空间池化（10×10 → 5×5 / 4×4）和视角采样，避免
    cond_tokens 长度随 V 线性爆炸。
  - 数据完整性过滤复用 ``PBRConditionalDataset``（require_all_render_views 等）。

输出目录默认与旧版隔离（避免覆盖 6-view cache），新位置:
  /path/to/data

用法示例：
    python scripts/tools/extract_features_multiview.py \\
        --root-dir /path/to/data \\
        --save-dir /path/to/data \\
        --num-views 32 \\
        --image-size 140 \\
        --batch-size 4 \\
        --num-workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as torchvision_F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gs_pi.data.srn import PBRConditionalDataset  # noqa: E402
from gs_pi.models.diffusion.diffusion_model import DINOv2ImageEncoder  # noqa: E402


class PBRRenderImageDataset(Dataset):
    """Load multi-view Render_*.png tensors for a filtered list of object ids.

    ``PBRConditionalDataset`` is still used upstream for dataset validation /
    object filtering, but its current ``__getitem__`` no longer returns image
    tensors. This lightweight dataset restores the expected [V, 3, H, W]
    batches for offline DINO extraction.
    """

    def __init__(self, *, root_dir: str, obj_ids: list[str], num_views: int):
        self.root_dir = Path(root_dir)
        self.obj_ids = list(obj_ids)
        self.num_views = int(num_views)

    def __len__(self) -> int:
        return len(self.obj_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        obj_id = self.obj_ids[idx]
        obj_dir = self.root_dir / obj_id

        views = []
        for view_idx in range(1, self.num_views + 1):
            img_path = obj_dir / f"Render_{view_idx}.png"
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                views.append(torchvision_F.to_tensor(img))  # [3, H, W] in [0, 1]

        images = torch.stack(views, dim=0)  # [V, 3, H, W]
        return {"images": images, "name": obj_id}


def validate_explicit_render_object_ids(
    *,
    root_dir: Path,
    object_ids: list[str],
    num_views: int,
) -> list[str]:
    ready: list[str] = []
    missing: list[str] = []
    for obj_id in object_ids:
        obj_dir = root_dir / obj_id
        if not obj_dir.is_dir():
            missing.append(f"{obj_id}: missing dir")
            continue
        absent_views = [
            str(view_idx)
            for view_idx in range(1, int(num_views) + 1)
            if not (obj_dir / f"Render_{view_idx}.png").is_file()
        ]
        if absent_views:
            preview = ",".join(absent_views[:8])
            suffix = "..." if len(absent_views) > 8 else ""
            missing.append(f"{obj_id}: missing Render views {preview}{suffix}")
            continue
        ready.append(obj_id)
    if missing:
        raise FileNotFoundError(
            "Explicit object ids are not ready for DINO extraction under "
            f"{root_dir}: {missing}"
        )
    return ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--object-ids",
        nargs="*",
        default=None,
        help="Optional explicit object ids to process.",
    )
    parser.add_argument(
        "--object-ids-file",
        default=None,
        help="Optional newline-delimited object id list.",
    )
    parser.add_argument("--num-views", type=int, default=32)
    parser.add_argument(
        "--image-size",
        type=int,
        default=140,
        help="DINO 输入分辨率 (会被强制 resize 到这个尺寸)。"
        "vits14 patch=14, 140 -> 10x10 patches per view.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="单卡同时跑几个对象 (V*B 张图同时进 DINO)，按显存调",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--model-size", default="vits14", choices=["vits14", "vitb14", "vitl14"]
    )
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument(
        "--proj-seed",
        type=int,
        default=0,
        help="DINO patch token -> output_dim 线性投影的固定 seed。"
        "设为固定值可保证多次重提 cache 落在同一特征空间。",
    )
    parser.add_argument(
        "--store-dtype", default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prefer-extract-output",
        default="true",
        choices=["true", "false", "auto"],
        help="Point-cloud location used only for PBRConditionalDataset validation. "
        "true keeps the historical root/<obj>/extract_output behavior; "
        "false uses root/<obj>; auto delegates to the dataset default.",
    )
    return parser.parse_args()


def resolve_prefer_extract_output(value: str) -> bool | None:
    if value == "auto":
        return None
    return {"true": True, "false": False}[value]


@torch.no_grad()
def encode_batch_per_view(
    encoder: DINOv2ImageEncoder, images: torch.Tensor, *, image_size: int
) -> torch.Tensor:
    """images: [B, V, 3, H, W] -> [B, V, H_patch, W_patch, D] (per-view feature grid).

    Mirrors the internals of ``DINOv2ImageEncoder.forward`` but stops before the
    [B, V*N, D] flatten so we can persist patch grids per view.
    """
    b, v, c, _, _ = images.shape
    x_flat = images.view(b * v, c, *images.shape[-2:])
    x_flat = F.interpolate(
        x_flat, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    ret = encoder.dino.forward_features(x_flat)
    patch_tokens = ret["x_norm_patchtokens"]  # [B*V, N, dino_dim]
    out = encoder.proj(patch_tokens)  # [B*V, N, D]
    n, d = out.shape[1], out.shape[2]
    side = int(round(n**0.5))
    if side * side != n:
        raise RuntimeError(
            f"Patch token count {n} 不是完全平方数；当前不支持非方形 patch 网格"
        )
    out = out.view(b, v, side, side, d)
    return out


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    store_dtype = {"float16": torch.float16, "float32": torch.float32}[args.store_dtype]

    explicit_ids = [
        str(obj_id).strip() for obj_id in (args.object_ids or []) if str(obj_id).strip()
    ]
    explicit_ids_file = (
        Path(args.object_ids_file).expanduser().resolve()
        if args.object_ids_file
        else None
    )
    if explicit_ids and explicit_ids_file is not None:
        raise ValueError("Use only one of --object-ids or --object-ids-file")
    if explicit_ids_file is not None and not explicit_ids_file.is_file():
        raise FileNotFoundError(f"object id list does not exist: {explicit_ids_file}")

    if explicit_ids_file is not None:
        explicit_ids = [
            line.strip()
            for line in explicit_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if explicit_ids:
        filtered_obj_ids = list(dict.fromkeys(explicit_ids))
        obj_ids = validate_explicit_render_object_ids(
            root_dir=root_dir,
            object_ids=filtered_obj_ids,
            num_views=int(args.num_views),
        )
        print(f"[filter] explicit object ids bypass dataset filter: {obj_ids}")
    else:
        # 用 PBRConditionalDataset 自带的完整性过滤拿到只包含可处理对象的列表。
        # 这里走 standard 模式 + 不要求 cond_point cache（DINO 提取早于 cond cache）。
        dataset = PBRConditionalDataset(
            root_dir=str(root_dir),
            num_views=args.num_views,
            num_points=512,
            dino_features_dir=None,
            cond_point_features_dir=None,
            require_precomputed_cond_point_features=False,
            skip_missing_precomputed_cond_point_features=False,
            cond_point_mode="standard",
            cond_point_dim=24,
            prefer_extract_output=resolve_prefer_extract_output(
                args.prefer_extract_output
            ),
            category=None,  # 提全量
            require_metadata=True,
            require_all_render_views=True,  # 必须 32 张全
            require_all_view_plys=False,
        )
        obj_ids = list(dataset.obj_dirs)
    if args.num_shards > 1:
        obj_ids = [
            oid
            for i, oid in enumerate(obj_ids)
            if i % args.num_shards == args.shard_index
        ]
        print(
            f"[shard {args.shard_index}/{args.num_shards}] 处理 {len(obj_ids)} 个对象"
        )
    if args.max_objects is not None:
        obj_ids = obj_ids[: args.max_objects]

    if not args.overwrite:
        before = len(obj_ids)
        obj_ids = [oid for oid in obj_ids if not (save_dir / f"{oid}.pt").exists()]
        print(f"跳过已存在缓存 {before - len(obj_ids)} 个，剩余 {len(obj_ids)} 待处理")

    pending_dataset = PBRRenderImageDataset(
        root_dir=args.root_dir,
        obj_ids=obj_ids,
        num_views=args.num_views,
    )
    loader = DataLoader(
        pending_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    encoder = (
        DINOv2ImageEncoder(
            output_dim=args.output_dim,
            model_size=args.model_size,
            proj_seed=args.proj_seed,
        )
        .to(device)
        .eval()
    )

    print(
        f"开始提取: V={args.num_views}, image_size={args.image_size}, "
        f"output_dim={args.output_dim}, proj_seed={args.proj_seed}, save_dir={save_dir}"
    )
    pbar = tqdm(loader, total=len(loader), unit="batch")
    saved = 0
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)  # [B, V, 3, H, W]
        names = batch["name"]
        feats = encode_batch_per_view(
            encoder, images, image_size=args.image_size
        )  # [B, V, H, W, D]
        feats_cpu = feats.detach().to(dtype=store_dtype, device="cpu")
        for i, name in enumerate(names):
            payload = {
                "format": "per_view_v1",
                # A contiguous batch slice can still own the entire batch's
                # storage; clone so each object file stores only its own views.
                "features": feats_cpu[i].clone(
                    memory_format=torch.contiguous_format
                ),  # [V,H,W,D]
                "num_views": int(args.num_views),
                "patch_side": int(feats_cpu.shape[2]),
                "output_dim": int(args.output_dim),
                "image_size": int(args.image_size),
                "model_size": args.model_size,
                "dino_embed_dim": int(encoder.dino_dim),
                "proj_seed": None
                if encoder.proj_seed is None
                else int(encoder.proj_seed),
                "projection_type": "seeded_linear",
            }
            torch.save(payload, save_dir / f"{name}.pt")
            saved += 1
        pbar.set_postfix_str(f"saved={saved}")

    print(f"Done. Saved {saved} feature files to {save_dir}")


if __name__ == "__main__":
    main()
