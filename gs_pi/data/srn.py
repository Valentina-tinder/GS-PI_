import os
import os.path as osp
import hashlib
import random
from random import shuffle
import queue
import threading
from PIL import Image
import PIL
import numpy as np
import tqdm
import torchvision.transforms.functional as torchvision_F
import torch
from torch.utils.data import default_collate
import torch.nn.functional as F
from gs_pi.utils import chunks
from .cond_point_features import build_cond_point_features_2d_projection
from .dataset import Dataset, Sample
from .registry import register_dataset
from plyfile import PlyData
from scipy.spatial import cKDTree
import json
import pickle
import zlib
from gs_pi.pbr_schema import get_feature_schema, normalize_world_normals_numpy
from gs_pi.pointcloud_storage import ViewPointCloudReader


@register_dataset
class PBRConditionalDataset(Dataset):
    POINT_CLOUD_XYZ_FINGERPRINT_VERSION = "ordered_xyz_float32_le_v1"
    NORMAL_VIEW_SOURCE_KNN_CACHE_FORMAT = "normal_view_source_knn_v1"
    DEFAULT_FULL_POINTCLOUD_VIS_TRAIN_IDS = ()
    DEFAULT_FULL_POINTCLOUD_VIS_TEST_IDS = ()

    def __init__(
        self,
        root_dir,
        num_views=6,
        num_points=512,
        dino_features_dir=None,
        sensor_width=36.0,
        focal_mm=35.0,
        max_samples=None,
        sample_repeat_factor=1,
        cond_point_dim=18,
        category=None,
        category_glb_dir=None,
        cond_point_features_dir=None,
        cond_point_features_ext=".pt",
        cond_point_mode="standard",
        cond_point_top_k=0,
        prefer_extract_output=None,
        require_precomputed_cond_point_features=False,
        skip_missing_precomputed_cond_point_features=None,
        skip_missing_dino_features=False,
        require_all_render_views=True,
        require_metadata=True,
        require_all_view_plys=False,
        dino_target_side=None,
        dino_subsample_views=None,
        dino_features_in_memory=True,
        dino_prepool_in_memory=False,
        dino_mmap_path=None,
        ply_data_in_memory=False,
        cond_point_features_in_memory=False,
        quality_csv_path=None,
        quality_max_p50=None,
        quality_max_p90=None,
        full_pointcloud_vis_train_ids=None,
        full_pointcloud_vis_test_ids=None,
        excluded_train_obj_ids=None,
        feature_schema="pbr_v1",
        require_normal=None,
        return_normal_valid_mask=None,
        normal_field_names=("nx", "ny", "nz"),
        normal_valid_field="normal_valid",
        normal_eps=1e-06,
        normal_hard_sampling_ratio=0.0,
        normal_hard_pool_fraction=0.2,
        normal_hard_knn=8,
        normal_hard_max_candidates=20000,
        normal_view_source_sampling_enabled=False,
        normal_view_source_knn_cache_dir=None,
        normal_view_source_neighbor_k=4,
        normal_view_source_anchor_count=51,
        normal_view_source_max_distance_ratio=None,
        view_source_balanced_sampling_num_views=0,
        return_view_source=False,
        normal_view_source_relation_edges_enabled=False,
        normal_view_source_relation_anchors_per_view=12,
        full_pointcloud_training=False,
        light_probe_dir=None,
        light_probe_source_kind=None,
        light_probe_filename=None,
        light_probe_match_render_strength=False,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.light_probe_dir = light_probe_dir
        self.light_probe_source_kind = light_probe_source_kind
        self.light_probe_filename = light_probe_filename
        self.light_probe_match_render_strength = bool(light_probe_match_render_strength)
        self._light_probe_cache = {}
        self.num_views = num_views
        self.num_points = num_points
        self.full_pointcloud_training = bool(full_pointcloud_training)
        self.dino_features_dir = dino_features_dir
        self.skip_missing_dino_features = bool(skip_missing_dino_features)
        self.sensor_width = sensor_width
        self.focal_mm = focal_mm
        self.max_samples = max_samples
        self.sample_repeat_factor = int(sample_repeat_factor)
        if self.sample_repeat_factor < 1:
            raise ValueError(
                f"sample_repeat_factor 必须 >= 1，当前为 {self.sample_repeat_factor}"
            )
        self.cond_point_dim = cond_point_dim
        self.category = category
        if category and not category_glb_dir:
            raise ValueError("category filtering requires category_glb_dir")
        if light_probe_dir is not None:
            raise ValueError("Light conditioning is not included in this release")
        self.category_glb_dir = category_glb_dir
        self.cond_point_features_dir = cond_point_features_dir
        self.cond_point_features_ext = cond_point_features_ext
        if cond_point_mode not in ("standard", "topk"):
            raise ValueError(
                f"cond_point_mode 必须是 'standard' 或 'topk', 得到 {cond_point_mode!r}"
            )
        self.cond_point_mode = cond_point_mode
        self.cond_point_top_k = int(cond_point_top_k)
        if self.cond_point_mode == "topk":
            if self.cond_point_top_k <= 0:
                raise ValueError(
                    f"cond_point_mode='topk' 需要 cond_point_top_k >= 1, 得到 {self.cond_point_top_k}"
                )
            expected_dim = self.cond_point_top_k * 5
            if cond_point_dim != expected_dim:
                raise ValueError(
                    f"cond_point_mode='topk' 且 cond_point_top_k={self.cond_point_top_k} 时，cond_point_dim 必须等于 5*K={expected_dim}，当前 {cond_point_dim}"
                )
        self.prefer_extract_output = prefer_extract_output
        self.require_precomputed_cond_point_features = (
            require_precomputed_cond_point_features
        )
        if skip_missing_precomputed_cond_point_features is None:
            self.skip_missing_precomputed_cond_point_features = bool(
                require_precomputed_cond_point_features
            )
        else:
            self.skip_missing_precomputed_cond_point_features = bool(
                skip_missing_precomputed_cond_point_features
            )
        self.require_all_render_views = bool(require_all_render_views)
        self.require_metadata = bool(require_metadata)
        self.require_all_view_plys = bool(require_all_view_plys)
        self.dino_target_side = (
            None if dino_target_side is None else int(dino_target_side)
        )
        if self.dino_target_side is not None and self.dino_target_side <= 0:
            raise ValueError(
                f"dino_target_side 必须 >= 1 或为 None, 得到 {self.dino_target_side}"
            )
        self.dino_subsample_views = (
            None if dino_subsample_views is None else int(dino_subsample_views)
        )
        if self.dino_subsample_views is not None:
            if self.dino_subsample_views <= 0 or self.dino_subsample_views > num_views:
                raise ValueError(
                    f"dino_subsample_views 必须 1..num_views({num_views}) 或 None, 得到 {self.dino_subsample_views}"
                )
        self.quality_csv_path = quality_csv_path
        self.quality_max_p50 = (
            None if quality_max_p50 is None else float(quality_max_p50)
        )
        self.quality_max_p90 = (
            None if quality_max_p90 is None else float(quality_max_p90)
        )
        if full_pointcloud_vis_train_ids is None:
            full_pointcloud_vis_train_ids = self.DEFAULT_FULL_POINTCLOUD_VIS_TRAIN_IDS
        if full_pointcloud_vis_test_ids is None:
            full_pointcloud_vis_test_ids = self.DEFAULT_FULL_POINTCLOUD_VIS_TEST_IDS
        if excluded_train_obj_ids is None:
            excluded_train_obj_ids = full_pointcloud_vis_test_ids
        self.full_pointcloud_vis_train_ids = tuple(
            dict.fromkeys(full_pointcloud_vis_train_ids)
        )
        self.full_pointcloud_vis_test_ids = tuple(
            dict.fromkeys(full_pointcloud_vis_test_ids)
        )
        self.excluded_train_obj_ids = tuple(dict.fromkeys(excluded_train_obj_ids))
        self.feature_schema = get_feature_schema(feature_schema)
        self.normal_field_names = tuple((str(name) for name in normal_field_names))
        if len(self.normal_field_names) != 3:
            raise ValueError(
                f"normal_field_names 必须正好包含 3 个字段名，当前为 {self.normal_field_names!r}"
            )
        self.normal_valid_field = str(normal_valid_field)
        self.normal_eps = float(normal_eps)
        if self.normal_eps <= 0:
            raise ValueError(f"normal_eps 必须 > 0，当前为 {self.normal_eps}")
        has_normal_group = any(
            (group.name == "normal" for group in self.feature_schema.groups)
        )
        self.require_normal = (
            has_normal_group if require_normal is None else bool(require_normal)
        )
        self.return_normal_valid_mask = (
            has_normal_group
            if return_normal_valid_mask is None
            else bool(return_normal_valid_mask)
        )
        if has_normal_group and (not self.require_normal):
            raise ValueError(
                "feature_schema='pbr_normal_world_v1' 必须同时要求 Blender GT normal；不允许回退到几何法线或静默填充。"
            )
        if self.return_normal_valid_mask and (not has_normal_group):
            raise ValueError(
                "return_normal_valid_mask=true 仅适用于 feature_schema='pbr_normal_world_v1'"
            )
        self.normal_hard_sampling_ratio = float(normal_hard_sampling_ratio)
        self.normal_hard_pool_fraction = float(normal_hard_pool_fraction)
        self.normal_hard_knn = int(normal_hard_knn)
        self.normal_hard_max_candidates = int(normal_hard_max_candidates)
        if not 0.0 <= self.normal_hard_sampling_ratio <= 1.0:
            raise ValueError("normal_hard_sampling_ratio 必须在 [0,1]")
        if not 0.0 < self.normal_hard_pool_fraction <= 1.0:
            raise ValueError("normal_hard_pool_fraction 必须在 (0,1]")
        if self.normal_hard_knn < 1:
            raise ValueError("normal_hard_knn 必须 >= 1")
        if self.normal_hard_max_candidates < 2:
            raise ValueError("normal_hard_max_candidates 必须 >= 2")
        if self.normal_hard_sampling_ratio > 0.0 and (not has_normal_group):
            raise ValueError(
                "normal-aware hard sampling 需要 pbr_normal_world_v1 schema"
            )
        self.normal_view_source_sampling_enabled = bool(
            normal_view_source_sampling_enabled
        )
        self.normal_view_source_neighbor_k = int(normal_view_source_neighbor_k)
        self.normal_view_source_anchor_count = int(normal_view_source_anchor_count)
        self.normal_view_source_max_distance_ratio = (
            None
            if normal_view_source_max_distance_ratio is None
            else float(normal_view_source_max_distance_ratio)
        )
        self.normal_view_source_knn_cache_dir = (
            None
            if normal_view_source_knn_cache_dir is None
            else os.path.abspath(normal_view_source_knn_cache_dir)
        )
        self.view_source_balanced_sampling_num_views = int(
            view_source_balanced_sampling_num_views
        )
        self.normal_view_source_relation_edges_enabled = bool(
            normal_view_source_relation_edges_enabled
        )
        self.normal_view_source_relation_anchors_per_view = int(
            normal_view_source_relation_anchors_per_view
        )
        self.return_view_source = bool(
            return_view_source
            or self.view_source_balanced_sampling_num_views > 0
            or self.normal_view_source_relation_edges_enabled
        )
        if self.normal_view_source_sampling_enabled:
            if not has_normal_group:
                raise ValueError(
                    "normal_view_source_sampling_enabled=true 需要 pbr_normal_world_v1 schema"
                )
            if self.normal_hard_sampling_ratio > 0.0:
                raise ValueError(
                    "同 view_source 分组采样不能与 normal_hard_sampling_ratio>0 同时启用"
                )
            if self.normal_view_source_knn_cache_dir is None:
                raise ValueError(
                    "启用同 view_source 分组采样时必须设置 normal_view_source_knn_cache_dir"
                )
            if self.normal_view_source_neighbor_k < 1:
                raise ValueError("normal_view_source_neighbor_k 必须 >= 1")
            if self.normal_view_source_anchor_count < 1:
                raise ValueError("normal_view_source_anchor_count 必须 >= 1")
            grouped_points = self.normal_view_source_anchor_count * (
                self.normal_view_source_neighbor_k + 1
            )
            if grouped_points > self.num_points:
                raise ValueError(
                    f"同 view_source 分组点预算超过 num_points: anchors={self.normal_view_source_anchor_count}, K={self.normal_view_source_neighbor_k}, grouped_points={grouped_points}, num_points={self.num_points}"
                )
            if (
                self.normal_view_source_max_distance_ratio is not None
                and self.normal_view_source_max_distance_ratio <= 0.0
            ):
                raise ValueError(
                    "normal_view_source_max_distance_ratio 必须 > 0 或为 None"
                )
        if self.view_source_balanced_sampling_num_views < 0:
            raise ValueError("view_source_balanced_sampling_num_views 必须 >= 0")
        if self.view_source_balanced_sampling_num_views > 0:
            selected_views = self.view_source_balanced_sampling_num_views
            if selected_views > int(self.num_views):
                raise ValueError(
                    f"view_source_balanced_sampling_num_views 不能超过 num_views: {selected_views} vs {self.num_views}"
                )
            if self.num_points % selected_views != 0:
                raise ValueError(
                    f"view-balanced sampling 要求 num_points 可被视角数整除: num_points={self.num_points}, views={selected_views}"
                )
            if self.normal_hard_sampling_ratio > 0.0:
                raise ValueError(
                    "view-balanced sampling 不能与 normal_hard_sampling_ratio>0 同时启用"
                )
            if self.normal_view_source_sampling_enabled:
                raise ValueError(
                    "view-balanced sampling 不能与旧版 normal_view_source_sampling_enabled 同时启用"
                )
        if self.normal_view_source_relation_edges_enabled:
            if not has_normal_group:
                raise ValueError(
                    "normal_view_source_relation_edges_enabled=true 需要 pbr_normal_world_v1 schema"
                )
            if self.view_source_balanced_sampling_num_views <= 0:
                raise ValueError(
                    "sampled same-view relation edges 要求启用 view-balanced sampling"
                )
            if self.normal_view_source_neighbor_k < 1:
                raise ValueError("normal_view_source_neighbor_k 必须 >= 1")
            if self.normal_view_source_relation_anchors_per_view < 1:
                raise ValueError(
                    "normal_view_source_relation_anchors_per_view 必须 >= 1"
                )
            points_per_view = (
                self.num_points // self.view_source_balanced_sampling_num_views
            )
            if points_per_view <= self.normal_view_source_neighbor_k:
                raise ValueError(
                    f"每个 view 的点数必须大于 relation K: points_per_view={points_per_view}, K={self.normal_view_source_neighbor_k}"
                )
        if self.full_pointcloud_training:
            if self.normal_hard_sampling_ratio > 0.0:
                raise ValueError(
                    "full_pointcloud_training=true 已使用全部点，不能同时启用 normal_hard_sampling_ratio"
                )
            if self.normal_view_source_sampling_enabled:
                raise ValueError(
                    "full_pointcloud_training=true 不能使用面向固定采样预算的 normal_view_source_sampling_enabled"
                )
            if self.view_source_balanced_sampling_num_views > 0:
                raise ValueError(
                    "full_pointcloud_training=true 不能启用 view-balanced sampling"
                )
            if self.normal_view_source_relation_edges_enabled:
                raise ValueError(
                    "full_pointcloud_training=true 不能启用固定预算 relation edges"
                )
        self._normal_hard_pool_cache = {}
        self._normal_hard_pool_stats = {}
        self._normal_view_source_knn_cache = {}
        self._view_source_cache = {}
        self._view_source_group_cache = {}
        self.ply_data_in_memory = bool(ply_data_in_memory)
        self.cond_point_features_in_memory = bool(cond_point_features_in_memory)
        self.dino_prepool_in_memory = bool(dino_prepool_in_memory)
        if self.full_pointcloud_training and (
            self.ply_data_in_memory or self.cond_point_features_in_memory
        ):
            raise ValueError(
                "full_pointcloud_training=true 时禁止把全部 PLY/逐点条件预载到 RAM；请设置 ply_data_in_memory=false 且 cond_point_features_in_memory=false"
            )
        self._ply_data_cache = {}
        self._cond_point_features_cache = {}
        self.dino_mmap_path = dino_mmap_path
        self._dino_mmap_features = None
        self._dino_mmap_index = None
        self._dino_mmap_meta = None
        if self.dino_features_dir is not None:
            self.dino_features_dir = os.path.abspath(self.dino_features_dir)
            if self.skip_missing_dino_features and (
                not os.path.isdir(self.dino_features_dir)
            ):
                raise FileNotFoundError(
                    f"DINO 特征目录不存在: {self.dino_features_dir}"
                )
        if self.cond_point_features_dir is not None:
            self.cond_point_features_dir = os.path.abspath(self.cond_point_features_dir)
            if self.skip_missing_precomputed_cond_point_features and (
                not os.path.isdir(self.cond_point_features_dir)
            ):
                raise FileNotFoundError(
                    f"预计算 cond_point_colors 目录不存在: {self.cond_point_features_dir}"
                )
        super().__init__(
            root=root_dir, num_views=num_views, num_points=num_points, **kwargs
        )
        self._dino_cache = {}
        self.dino_features_in_memory = bool(dino_features_in_memory)
        if self.dino_prepool_in_memory and (not self.dino_features_in_memory):
            raise ValueError(
                "dino_prepool_in_memory=true requires dino_features_in_memory=true"
            )
        if self.dino_mmap_path is not None:
            self._try_load_dino_mmap()
            self.dino_features_in_memory = False
        if (
            self.dino_features_in_memory
            and self.dino_features_dir is not None
            and getattr(self, "obj_dirs", None)
        ):
            self._preload_dino_features_into_memory()
        if self.ply_data_in_memory or self.cond_point_features_in_memory:
            self._preload_training_arrays_into_memory()

    @staticmethod
    def resolve_point_cloud_dir(
        root_dir, obj_id, prefer_extract_output=False, fallback_to_other=True
    ):
        """解析 merged_model_render_color.ply 所在目录。

        Args:
            prefer_extract_output:
                True  -> 优先/使用 ``extract_output``。
                False -> 优先/使用物体根目录。
            fallback_to_other:
                为 True 时，优先目录不存在则回退到另一目录；
                为 False 时，不回退，严格按 prefer_extract_output 选择目录。
        """
        obj_dir = os.path.join(root_dir, obj_id)
        extract_dir = os.path.join(obj_dir, "extract_output")
        extract_ply = os.path.join(extract_dir, "merged_model_render_color.ply")
        root_ply = os.path.join(obj_dir, "merged_model_render_color.ply")
        if prefer_extract_output:
            if os.path.exists(extract_ply):
                return extract_dir
            if fallback_to_other and os.path.exists(root_ply):
                return obj_dir
            return extract_dir
        if os.path.exists(root_ply):
            return obj_dir
        if fallback_to_other and os.path.exists(extract_ply):
            return extract_dir
        return obj_dir

    @staticmethod
    def resolve_point_cloud_path(root_dir, obj_id, prefer_extract_output=False):
        ply_dir = PBRConditionalDataset.resolve_point_cloud_dir(
            root_dir=root_dir,
            obj_id=obj_id,
            prefer_extract_output=prefer_extract_output,
        )
        return os.path.join(ply_dir, "merged_model_render_color.ply")

    @staticmethod
    def resolve_view_ply_path(base_dir, view_idx, prefer_extract_output=None):
        """解析单视角 view_XX.ply。

        Args:
            base_dir:
                物体根目录或其 ``extract_output`` 子目录。
            view_idx:
                1-based view index.
            prefer_extract_output:
                ``True``  -> 严格返回 ``extract_output/view_XX.ply``。
                ``False`` -> 严格返回根目录下 ``view_XX.ply``。
                ``None``  -> 先查 ``extract_output``，不存在则回退根目录。
        """
        base_dir = os.path.abspath(base_dir)
        if os.path.basename(base_dir) == "extract_output":
            extract_dir = base_dir
            obj_dir = os.path.dirname(base_dir)
        else:
            obj_dir = base_dir
            extract_dir = os.path.join(obj_dir, "extract_output")
        filename = f"view_{view_idx:02d}.ply"
        extract_path = os.path.join(extract_dir, filename)
        root_path = os.path.join(obj_dir, filename)
        if prefer_extract_output is True:
            return extract_path
        if prefer_extract_output is False:
            return root_path
        if os.path.exists(extract_path):
            return extract_path
        if os.path.exists(root_path):
            return root_path
        return extract_path

    def _get_ply_dir(self, obj_id):
        """根据当前训练设置解析 PLY 文件所在目录。"""
        if self.prefer_extract_output is None:
            prefer_extract_output = self.cond_point_dim == 24
            fallback_to_other = True
        else:
            prefer_extract_output = bool(self.prefer_extract_output)
            fallback_to_other = False
        return self.resolve_point_cloud_dir(
            root_dir=self.root_dir,
            obj_id=obj_id,
            prefer_extract_output=prefer_extract_output,
            fallback_to_other=fallback_to_other,
        )

    def _get_cond_point_features_path(self, obj_id):
        if self.cond_point_features_dir is None:
            return None
        return os.path.join(
            self.cond_point_features_dir, f"{obj_id}{self.cond_point_features_ext}"
        )

    def _get_dino_feature_path(self, obj_id):
        if self.dino_features_dir is None:
            return None
        return os.path.join(self.dino_features_dir, f"{obj_id}.pt")

    def _load_quality_blacklist(self):
        """读取 Top-K 质量扫描 CSV，返回需剔除的 object_id 集合。"""
        if self.quality_csv_path is None:
            return None
        if self.quality_max_p50 is None and self.quality_max_p90 is None:
            return None
        csv_path = os.path.abspath(self.quality_csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"quality_csv_path 不存在: {csv_path}")
        import csv as _csv

        def _tofloat(s):
            if s is None or s == "":
                return None
            try:
                return float(s)
            except ValueError:
                return None

        bad = set()
        n_total = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                oid = row.get("object_id") or row.get("obj_id")
                if not oid:
                    continue
                n_total += 1
                p50 = _tofloat(row.get("albedo_cross_view_range_p50"))
                p90 = _tofloat(row.get("albedo_cross_view_range_p90"))
                if (
                    self.quality_max_p50 is not None
                    and p50 is not None
                    and (p50 > self.quality_max_p50)
                ):
                    bad.add(oid)
                    continue
                if (
                    self.quality_max_p90 is not None
                    and p90 is not None
                    and (p90 > self.quality_max_p90)
                ):
                    bad.add(oid)
                    continue
        print(
            f"质量黑名单: {len(bad)} / {n_total} (p50>{self.quality_max_p50}, p90>{self.quality_max_p90}) from {csv_path}"
        )
        return bad

    def _preload_dino_features_into_memory(self):
        """主进程预加载所有 obj 的 DINO 特征到 self._dino_cache。

        - 仅遍历 self.obj_dirs（即过滤后真正会被训练的对象）
        - per_view_v1 cache 存为 fp16，flat tensor 也存为 fp16，省一半内存
        - Linux fork 出的 DataLoader workers 通过 COW 共享，不需要额外复制
        - 任意对象加载失败仅打印警告，不中断初始化（运行时回退到磁盘读）
        """
        if self.dino_features_dir is None:
            return
        loaded = 0
        skipped = 0
        bytes_total = 0
        import time as _t

        t0 = _t.time()
        for obj_id in self.obj_dirs:
            fp = (
                self._get_dino_feature_path(obj_id)
                if hasattr(self, "_get_dino_feature_path")
                else os.path.join(self.dino_features_dir, f"{obj_id}.pt")
            )
            if fp is None or not os.path.exists(fp):
                skipped += 1
                continue
            try:
                payload = torch.load(fp, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"[DINO-cache] 跳过 {fp}: {e}")
                skipped += 1
                continue
            if isinstance(payload, dict) and payload.get("format") == "per_view_v1":
                feats = (
                    torch.as_tensor(payload["features"]).to(torch.float16).contiguous()
                )
                if self.dino_prepool_in_memory and self.dino_target_side is not None:
                    if feats.ndim != 4:
                        raise ValueError(
                            f"per_view_v1 cache features must be [V,H,W,D], got {tuple(feats.shape)} @ {fp}"
                        )
                    (v, h, w, d) = feats.shape
                    if h != self.dino_target_side or w != self.dino_target_side:
                        pooled = F.adaptive_avg_pool2d(
                            feats.float().permute(0, 3, 1, 2),
                            (self.dino_target_side, self.dino_target_side),
                        )
                        feats = pooled.permute(0, 2, 3, 1).contiguous()
                payload = dict(payload)
                payload["features"] = feats
                if self.dino_prepool_in_memory:
                    payload["prepooled_target_side"] = self.dino_target_side
                bytes_total += feats.numel() * feats.element_size()
            else:
                t = torch.as_tensor(payload).to(torch.float16).contiguous()
                payload = t
                bytes_total += t.numel() * t.element_size()
            self._dino_cache[fp] = payload
            loaded += 1
        elapsed = _t.time() - t0
        gb = bytes_total / 1024**3
        print(
            f"[DINO-cache] 预加载完成: {loaded} 个对象 ({gb:.2f} GB), 跳过 {skipped} 个, prepool={self.dino_prepool_in_memory}, 用时 {elapsed:.1f}s"
        )

    def _preload_training_arrays_into_memory(self):
        """Preload parsed PLY arrays and/or packed conditioning features.

        Parsing a full PLY and loading its Top-K ``.pt`` file for every sample
        dominates short GPU steps.  The 95-object training split is small enough
        to retain these immutable arrays in host memory and share them with
        persistent DataLoader workers through copy-on-write.
        """
        import time as _t

        t0 = _t.time()
        ply_bytes = 0
        cond_bytes = 0
        for obj_id in self.obj_dirs:
            ply_dir = self._get_ply_dir(obj_id)
            ply_path = os.path.join(ply_dir, "merged_model_render_color.ply")
            ply_key = os.path.realpath(ply_path)
            full_point_count = None
            if self.ply_data_in_memory:
                need_view_source = bool(getattr(self, "return_view_source", False))
                loaded = self.load_full_ply_data(
                    ply_path,
                    cond_point_dim=self.cond_point_dim,
                    feature_schema=self.feature_schema,
                    normal_field_names=self.normal_field_names,
                    normal_valid_field=self.normal_valid_field,
                    normal_eps=self.normal_eps,
                    return_normal_valid_mask=self.return_normal_valid_mask,
                    return_view_source=need_view_source,
                )
                if self.return_normal_valid_mask and need_view_source:
                    (coords, feats, normal_valid_mask, view_source) = loaded
                elif self.return_normal_valid_mask:
                    (coords, feats, normal_valid_mask) = loaded
                    view_source = None
                elif need_view_source:
                    (coords, feats, view_source) = loaded
                    normal_valid_mask = None
                else:
                    (coords, feats) = loaded
                    view_source = None
                    normal_valid_mask = None
                if self.return_normal_valid_mask:
                    normal_valid_mask = np.ascontiguousarray(normal_valid_mask)
                    ply_bytes += normal_valid_mask.nbytes
                coords = np.ascontiguousarray(coords)
                feats = np.ascontiguousarray(feats)
                full_point_count = int(coords.shape[1])
                ply_bytes += coords.nbytes + feats.nbytes
                self._ply_data_cache[ply_key] = (coords, feats, normal_valid_mask)
                if view_source is not None:
                    source_dtype = (
                        np.int8
                        if int(view_source.min()) >= -128
                        and int(view_source.max()) <= 127
                        else np.int32
                    )
                    view_source = np.ascontiguousarray(view_source, dtype=source_dtype)
                    self._view_source_cache[ply_key] = view_source
                    self._view_source_group_cache[ply_key] = (
                        self._build_view_source_group_descriptors(view_source)
                    )
                    ply_bytes += view_source.nbytes
            if not self.cond_point_features_in_memory or self.cond_point_dim <= 0:
                continue
            feature_path = self._get_cond_point_features_path(obj_id)
            if feature_path is None or not os.path.exists(feature_path):
                if self.require_precomputed_cond_point_features:
                    raise FileNotFoundError(
                        f"missing required conditioning cache: {feature_path}"
                    )
                continue
            if self.cond_point_mode == "topk":
                payload = self.load_precomputed_topk_cond_features(feature_path)
                if payload["k"] != self.cond_point_top_k:
                    raise ValueError(
                        f"Top-K cache k={payload['k']} does not match cond_point_top_k={self.cond_point_top_k} @ {feature_path}"
                    )
                if payload["num_views"] != self.num_views:
                    raise ValueError(
                        f"Top-K cache num_views={payload['num_views']} does not match num_views={self.num_views} @ {feature_path}"
                    )
                features = self.pack_topk_cond_point_colors(
                    payload["topk_rgb"], payload["topk_conf"], payload["topk_view_id"]
                )
                record = {
                    "features": features,
                    "k": payload["k"],
                    "num_views": payload["num_views"],
                }
            else:
                features = self.load_precomputed_cond_point_features(
                    feature_path, expected_dim=self.cond_point_dim
                )
                record = {"features": features}
            if full_point_count is not None and features.shape[1] != full_point_count:
                raise ValueError(
                    f"conditioning cache points={features.shape[1]} do not match PLY points={full_point_count} @ {feature_path}"
                )
            cond_bytes += features.numel() * features.element_size()
            self._cond_point_features_cache[feature_path] = record
        elapsed = _t.time() - t0
        print(
            f"[training-cache] ready: ply_objects={len(self._ply_data_cache)} ({ply_bytes / 1024**3:.2f} GB), cond_objects={len(self._cond_point_features_cache)} ({cond_bytes / 1024**3:.2f} GB), elapsed={elapsed:.1f}s"
        )

    def _try_load_dino_mmap(self):
        """Open the packed mmap DINO cache (shared across all DDP ranks).

        Expects ``dino_mmap_path`` to contain ``dino_cache.bin`` +
        ``dino_cache.idx.json`` produced by
        ``scripts/tools/pack_dino_cache_mmap.py``.
        """
        import json as _json
        import numpy as _np

        base = os.path.abspath(self.dino_mmap_path)
        bin_path = os.path.join(base, "dino_cache.bin")
        idx_path = os.path.join(base, "dino_cache.idx.json")
        if not (os.path.exists(bin_path) and os.path.exists(idx_path)):
            raise FileNotFoundError(
                f"dino_mmap_path 缺少 dino_cache.bin / dino_cache.idx.json: {base}"
            )
        with open(idx_path, "r") as f:
            idx = _json.load(f)
        meta = idx["meta"]
        if meta.get("format") != "dino_mmap_v1":
            raise ValueError(f"unsupported mmap format: {meta.get('format')}")
        (V, H, W, D) = meta["shape_per_obj"]
        if V != self.num_views:
            raise ValueError(
                f"dino mmap num_views={V} 与 dataset num_views={self.num_views} 不一致"
            )
        dtype = _np.dtype(meta["dtype"])
        n_obj = int(meta.get("num_objects", len(idx["obj_id_to_index"])))
        arr = _np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_obj, V, H, W, D))
        self._dino_mmap_features = torch.from_numpy(arr)
        self._dino_mmap_index = idx["obj_id_to_index"]
        self._dino_mmap_meta = meta
        gb = n_obj * V * H * W * D * dtype.itemsize / 1024**3
        print(
            f"[DINO-mmap] 已映射 {n_obj} 个对象 ({gb:.2f} GB, {dtype}) shape_per_obj={[V, H, W, D]} path={bin_path}"
        )

    def _load_and_process_dino_features(
        self, feature_path, *, apply_view_subsample=True
    ):
        """Load DINO cache and optionally avg-pool per view / subsample views.

        Supported cache formats:
          1) 旧版扁平: ``torch.Tensor`` of shape ``[V*N, D]`` (e.g. 6 view × 100 patch).
             保持原样返回，不做池化/采样（因为没有 per-view 结构信息）。
          2) 新版 per-view: ``dict`` with ``format='per_view_v1'`` containing
             ``features: [V, H, W, D]``. 可按 ``dino_target_side`` 做 AvgPool2d，
             并在 ``apply_view_subsample=True`` 时按 ``dino_subsample_views``
             随机挑 M 个视角。
        """
        cache = getattr(self, "_dino_cache", None)
        payload = None
        if self._dino_mmap_features is not None:
            obj_id = os.path.splitext(os.path.basename(feature_path))[0]
            idx = self._dino_mmap_index.get(obj_id)
            if idx is not None:
                feats_fp16 = self._dino_mmap_features[idx]
                payload = {"format": "per_view_v1", "features": feats_fp16}
        if payload is None and cache is not None and (feature_path in cache):
            payload = cache[feature_path]
        if payload is None:
            payload = self._torch_load_with_fallback(feature_path)
        if isinstance(payload, dict) and payload.get("format") == "per_view_v1":
            features = payload["features"]
            features = torch.as_tensor(features).float()
            if features.ndim != 4:
                raise ValueError(
                    f"per_view_v1 cache features 必须是 [V, H, W, D]，得到 {tuple(features.shape)} @ {feature_path}"
                )
            (v, h, w, d) = features.shape
            if v != self.num_views:
                raise ValueError(
                    f"DINO 缓存 V={v} 与 dataset num_views={self.num_views} 不一致 @ {feature_path}"
                )
            if self.dino_target_side is not None and self.dino_target_side != min(h, w):
                x = features.permute(0, 3, 1, 2)
                x = F.adaptive_avg_pool2d(
                    x, (self.dino_target_side, self.dino_target_side)
                )
                features = x.permute(0, 2, 3, 1).contiguous()
                h = w = self.dino_target_side
            if (
                apply_view_subsample
                and self.dino_subsample_views is not None
                and (self.dino_subsample_views < v)
            ):
                perm = torch.randperm(v)[: self.dino_subsample_views]
                features = features.index_select(0, perm)
                v = self.dino_subsample_views
            return features.reshape(v * h * w, d).contiguous()
        tensor = torch.as_tensor(payload)
        if tensor.ndim != 2:
            raise ValueError(
                f"旧版 DINO 缓存须为 [L, D], 得到 {tuple(tensor.shape)} @ {feature_path}"
            )
        return tensor.float().contiguous()

    def _validate_full_dino_condition(self, cond_tokens, *, feature_path):
        """Require the complete known DINO token grid for full/full training.

        Legacy flat caches do not encode their view/grid decomposition.  When
        ``dino_target_side`` is known, the full/full route can nevertheless
        prove completeness from the flattened shape and must not silently
        accept a 16-view ``L=400`` cache in place of the configured 32-view
        ``L=800`` condition.  This helper is called only by the opt-in full
        pointcloud path; historical 512-point training keeps its old behavior.
        """
        target_side = getattr(self, "dino_target_side", None)
        if (
            not getattr(self, "full_pointcloud_training", False)
            or self.dino_features_dir is None
            or target_side is None
        ):
            return
        expected_shape = (
            int(self.num_views) * int(target_side) * int(target_side),
            512,
        )
        actual_shape = (
            tuple(cond_tokens.shape) if torch.is_tensor(cond_tokens) else None
        )
        if actual_shape != expected_shape:
            raise ValueError(
                f"full/full DINO 条件 shape 必须严格为 {list(expected_shape)} = [num_views({self.num_views}) * dino_target_side({target_side})^2, 512]，得到 {actual_shape!r} @ {feature_path}"
            )
        if not bool(torch.isfinite(cond_tokens).all()):
            raise ValueError(f"full/full DINO 条件包含 NaN/Inf @ {feature_path}")

    @staticmethod
    def _is_all_black(img_path, threshold=0.01):
        """判断 EXR 图片是否为全黑 (0-1 float32 范围内所有像素接近 0)"""
        if not os.path.exists(img_path):
            return False
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        img = np.asarray(img, dtype=np.float32)
        if img.size == 0:
            return True
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return False
        max_value = float(finite.max())
        return max_value < threshold

    def _has_all_black_reference_pbr(self, obj_path):
        """仅当 3 张参考 PBR EXR 都存在且都全黑时才跳过该样本。"""
        pbr_paths = [
            os.path.join(obj_path, "Albedo_1.exr"),
            os.path.join(obj_path, "Metallic_1.exr"),
            os.path.join(obj_path, "Roughness_1.exr"),
        ]
        if not all((os.path.exists(path) for path in pbr_paths)):
            return False
        return all((self._is_all_black(path) for path in pbr_paths))

    def _init_samples(self, num_views=6, num_points=512, **kwargs):
        """扫描索引入口：rank0 负责扫盘/读缓存，其他 rank 接收广播。"""
        import hashlib
        import pickle

        try:
            import torch.distributed as _dist

            _dist_ready = _dist.is_available() and _dist.is_initialized()
        except Exception:
            _dist = None
            _dist_ready = False
        _rank = _dist.get_rank() if _dist_ready else 0
        _fp = {
            "dataset_class": type(self).__name__,
            "root": self.root_dir,
            "category": getattr(self, "category", None),
            "num_views": getattr(self, "num_views", num_views),
            "require_metadata": getattr(self, "require_metadata", False),
            "require_all_render_views": getattr(
                self, "require_all_render_views", False
            ),
            "require_all_view_plys": getattr(self, "require_all_view_plys", False),
            "cond_dir": getattr(self, "cond_point_features_dir", None),
            "skip_missing_cond": getattr(
                self, "skip_missing_precomputed_cond_point_features", False
            ),
            "dino_dir": getattr(self, "dino_features_dir", None),
            "skip_missing_dino": getattr(self, "skip_missing_dino_features", False),
            "quality_csv": getattr(self, "quality_csv_path", None),
            "prefer_extract_output": getattr(self, "prefer_extract_output", False),
            "full_pc_vis_train_ids": getattr(self, "full_pointcloud_vis_train_ids", ()),
            "excluded_train_obj_ids": getattr(self, "excluded_train_obj_ids", ()),
            "max_samples": getattr(self, "max_samples", None),
            "feature_schema": getattr(self.feature_schema, "name", "pbr_v1"),
            "require_normal": getattr(self, "require_normal", False),
            "normal_field_names": getattr(self, "normal_field_names", ()),
            "normal_valid_field": getattr(self, "normal_valid_field", "normal_valid"),
            "cache_ver": 6,
        }
        _key = hashlib.md5(
            json.dumps(_fp, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        _cache_path = os.path.join(self.root_dir, f".dataset_index_{_key}.pkl")
        _payload = [None, None, None]
        if _rank == 0:
            _cached = None
            if os.path.exists(_cache_path):
                try:
                    with open(_cache_path, "rb") as _f:
                        _cached = pickle.load(_f)
                    print(
                        f"[dataset] 命中索引缓存: {_cache_path} (有效 {len(_cached['obj_dirs'])} / 总 {len(_cached['raw_obj_dirs'])})"
                    )
                except Exception as _e:
                    print(f"[dataset] 读取索引缓存失败，将重新扫描: {_e}")
                    _cached = None
            if _cached is not None:
                _payload = [
                    _cached["obj_dirs"],
                    _cached["raw_obj_dirs"],
                    _cached["filtered_obj_dirs"],
                ]
            else:
                self._scan_obj_dirs_from_disk(
                    num_views=num_views, num_points=num_points, **kwargs
                )
                _payload = [self.obj_dirs, self.raw_obj_dirs, self.filtered_obj_dirs]
                try:
                    _tmp_path = _cache_path + ".tmp"
                    with open(_tmp_path, "wb") as _f:
                        pickle.dump(
                            {
                                "obj_dirs": _payload[0],
                                "raw_obj_dirs": _payload[1],
                                "filtered_obj_dirs": _payload[2],
                            },
                            _f,
                        )
                    os.replace(_tmp_path, _cache_path)
                    print(f"[dataset] 已写入索引缓存: {_cache_path}")
                except Exception as _e:
                    print(f"[dataset] 写索引缓存失败: {_e}")
        if _dist_ready:
            _dist.broadcast_object_list(_payload, src=0)
        self.obj_dirs = _payload[0]
        self.raw_obj_dirs = _payload[1]
        self.filtered_obj_dirs = _payload[2]
        if not self.filtered_obj_dirs:
            self.filtered_obj_dirs = list(self.obj_dirs)
        self._apply_train_obj_id_preferences()

    def _apply_train_obj_id_preferences(self, all_valid_obj_dirs=None):
        if all_valid_obj_dirs is None:
            all_valid_obj_dirs = list(
                getattr(self, "filtered_obj_dirs", None)
                or getattr(self, "obj_dirs", [])
            )
        else:
            all_valid_obj_dirs = list(all_valid_obj_dirs)
        self.filtered_obj_dirs = list(all_valid_obj_dirs)
        valid_set = set(all_valid_obj_dirs)
        fixed_train_ids = [
            obj_id
            for obj_id in self.full_pointcloud_vis_train_ids
            if obj_id in valid_set
        ]
        missing_fixed_train_ids = [
            obj_id
            for obj_id in self.full_pointcloud_vis_train_ids
            if obj_id not in valid_set
        ]
        if missing_fixed_train_ids:
            print(
                f"full pointcloud 可视化训练ID未找到，将自动回退补位: {missing_fixed_train_ids}"
            )
        missing_fixed_test_ids = [
            obj_id
            for obj_id in self.full_pointcloud_vis_test_ids
            if obj_id not in valid_set
        ]
        if missing_fixed_test_ids:
            print(f"full pointcloud 可视化测试ID未找到: {missing_fixed_test_ids}")
        excluded_train_set = set(self.excluded_train_obj_ids)
        excluded_present = [
            obj_id for obj_id in all_valid_obj_dirs if obj_id in excluded_train_set
        ]
        if excluded_present:
            print(f"训练前手工排除测试ID: {excluded_present}")
        remaining_train_obj_dirs = [
            obj_id
            for obj_id in all_valid_obj_dirs
            if obj_id not in excluded_train_set and obj_id not in fixed_train_ids
        ]
        self.obj_dirs = fixed_train_ids + remaining_train_obj_dirs
        if len(self.obj_dirs) == 0:
            raise RuntimeError(
                "手工排除测试ID后训练集为空，请检查 excluded_train_obj_ids 配置。"
            )
        if self.max_samples is not None and self.max_samples < len(self.obj_dirs):
            self.obj_dirs = self.obj_dirs[: self.max_samples]
            print(f"已限制样本数量为: {len(self.obj_dirs)}")
        self.samples = self.obj_dirs

    def _scan_obj_dirs_from_disk(self, num_views=6, num_points=512, **kwargs):
        """在这里实现原本在 __init__ 中的扫描逻辑"""
        allowed_ids = None
        if self.category is not None:
            cat_dir = os.path.join(self.category_glb_dir, self.category)
            if not os.path.isdir(cat_dir):
                raise RuntimeError(f"分类目录不存在: {cat_dir}")
            allowed_ids = set()
            for fname in os.listdir(cat_dir):
                if fname.endswith(".glb"):
                    allowed_ids.add(fname[:-4])
            print(f"分类 '{self.category}' 共有 {len(allowed_ids)} 个 GLB ID")
        quality_blacklist = self._load_quality_blacklist()
        self.raw_obj_dirs = sorted(
            [
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )
        self.obj_dirs = []
        skipped_black = 0
        skipped_category = 0
        skipped_quality = 0
        skipped_missing_ply = 0
        skipped_missing_dino = 0
        skipped_missing_precomputed = 0
        skipped_missing_renders = 0
        skipped_missing_metadata = 0
        skipped_missing_view_plys = 0
        skipped_missing_normal = 0
        missing_ply_examples = []
        missing_dino_examples = []
        missing_precomputed_examples = []
        missing_renders_examples = []
        missing_metadata_examples = []
        missing_view_plys_examples = []
        missing_normal_examples = []
        for d in self.raw_obj_dirs:
            if allowed_ids is not None and d not in allowed_ids:
                skipped_category += 1
                continue
            if quality_blacklist is not None and d in quality_blacklist:
                skipped_quality += 1
                continue
            ply_dir = self._get_ply_dir(d)
            ply_path = os.path.join(ply_dir, "merged_model_render_color.ply")
            if not os.path.exists(ply_path):
                skipped_missing_ply += 1
                if len(missing_ply_examples) < 10:
                    missing_ply_examples.append(d)
                continue
            if self.require_normal:
                try:
                    ply_fields = self._read_ply_vertex_field_names(ply_path)
                except (OSError, ValueError) as exc:
                    skipped_missing_normal += 1
                    if len(missing_normal_examples) < 10:
                        missing_normal_examples.append((d, str(exc)))
                    continue
                missing_normal_fields = [
                    field
                    for field in self.normal_field_names
                    if field not in ply_fields
                ]
                if missing_normal_fields:
                    skipped_missing_normal += 1
                    if len(missing_normal_examples) < 10:
                        missing_normal_examples.append((d, missing_normal_fields))
                    continue
            obj_path = os.path.join(self.root_dir, d)
            if self.require_metadata:
                metadata_path = os.path.join(obj_path, "metadata.json")
                if not os.path.exists(metadata_path):
                    skipped_missing_metadata += 1
                    if len(missing_metadata_examples) < 10:
                        missing_metadata_examples.append(d)
                    continue
            if self.require_all_render_views:
                missing_views = [
                    v
                    for v in range(1, self.num_views + 1)
                    if not os.path.exists(os.path.join(obj_path, f"Render_{v}.png"))
                ]
                if missing_views:
                    skipped_missing_renders += 1
                    if len(missing_renders_examples) < 10:
                        missing_renders_examples.append((d, missing_views[:5]))
                    continue
            if self.require_all_view_plys:
                missing_vp = ViewPointCloudReader(
                    ply_dir, self.prefer_extract_output
                ).missing_views(self.num_views)
                if missing_vp:
                    skipped_missing_view_plys += 1
                    if len(missing_view_plys_examples) < 10:
                        missing_view_plys_examples.append((d, missing_vp[:5]))
                    continue
            if (
                self.cond_point_features_dir is not None
                and self.skip_missing_precomputed_cond_point_features
            ):
                cond_feature_path = self._get_cond_point_features_path(d)
                if cond_feature_path is None or not os.path.exists(cond_feature_path):
                    skipped_missing_precomputed += 1
                    if len(missing_precomputed_examples) < 10:
                        missing_precomputed_examples.append(d)
                    continue
            if self.dino_features_dir is not None and self.skip_missing_dino_features:
                dino_feature_path = self._get_dino_feature_path(d)
                if dino_feature_path is None or not os.path.exists(dino_feature_path):
                    skipped_missing_dino += 1
                    if len(missing_dino_examples) < 10:
                        missing_dino_examples.append(d)
                    continue
            if self._has_all_black_reference_pbr(obj_path):
                skipped_black += 1
                continue
            self.obj_dirs.append(d)
        if allowed_ids is not None:
            print(f"分类过滤跳过: {skipped_category}")
        if quality_blacklist is not None:
            print(f"质量过滤跳过: {skipped_quality}")
        if skipped_missing_ply > 0:
            print(
                f"缺失 merged_model_render_color.ply: {skipped_missing_ply} (示例: {missing_ply_examples})"
            )
        if skipped_missing_metadata > 0:
            print(
                f"缺失 metadata.json: {skipped_missing_metadata} (示例: {missing_metadata_examples})"
            )
        if skipped_missing_renders > 0:
            print(
                f"缺失 Render_*.png (共需 {self.num_views} 张): {skipped_missing_renders} (示例: {missing_renders_examples})"
            )
        if skipped_missing_view_plys > 0:
            print(
                f"缺失 view_XX.ply (共需 {self.num_views} 个): {skipped_missing_view_plys} (示例: {missing_view_plys_examples})"
            )
        if skipped_missing_normal > 0:
            print(
                f"缺失 Blender GT normal 字段: {skipped_missing_normal} (示例: {missing_normal_examples})"
            )
        if skipped_missing_precomputed > 0:
            print(
                f"缺失预计算 cond_point_colors: {skipped_missing_precomputed} (示例: {missing_precomputed_examples})"
            )
        if skipped_missing_dino > 0:
            print(
                f"缺失 DINO 特征: {skipped_missing_dino} (示例: {missing_dino_examples})"
            )
        print(
            f"有效样本数量: {len(self.obj_dirs)} / {len(self.raw_obj_dirs)} (跳过全黑PBR: {skipped_black})"
        )
        if len(self.obj_dirs) == 0:
            raise RuntimeError(f"在 {self.root_dir} 中未找到任何物体文件夹")
        all_valid_obj_dirs = list(self.obj_dirs)
        self._apply_train_obj_id_preferences(all_valid_obj_dirs)

    def __len__(self):
        return len(self.obj_dirs) * self.sample_repeat_factor

    @staticmethod
    def _read_ply_vertex_field_names(ply_path):
        """只读取 PLY header，避免数据集扫描阶段加载整份点云。"""
        field_names = set()
        current_element = None
        with open(ply_path, "rb") as stream:
            first_line = stream.readline()
            if first_line.strip() != b"ply":
                raise ValueError("不是有效的 PLY 文件")
            for _ in range(10000):
                raw_line = stream.readline()
                if not raw_line:
                    raise ValueError("PLY header 缺少 end_header")
                line = raw_line.decode("ascii", errors="strict").strip()
                if line == "end_header":
                    break
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "element":
                    current_element = parts[1]
                elif (
                    current_element == "vertex"
                    and len(parts) >= 3
                    and (parts[0] == "property")
                ):
                    field_names.add(parts[-1])
            else:
                raise ValueError("PLY header 异常大，未找到 end_header")
        return field_names

    @staticmethod
    def _read_ply_vertex_count(ply_path):
        """Read only the declared vertex count from a PLY header."""
        with open(ply_path, "rb") as stream:
            if stream.readline().strip() != b"ply":
                raise ValueError(f"不是有效的 PLY 文件: {ply_path}")
            for _ in range(10000):
                raw_line = stream.readline()
                if not raw_line:
                    raise ValueError(f"PLY header 缺少 end_header: {ply_path}")
                line = raw_line.decode("ascii", errors="strict").strip()
                parts = line.split()
                if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                    count = int(parts[2])
                    if count <= 0:
                        raise ValueError(
                            f"PLY vertex count 非正数: {count} @ {ply_path}"
                        )
                    return count
                if line == "end_header":
                    break
        raise ValueError(f"PLY header 缺少 element vertex: {ply_path}")

    def get_full_pointcloud_sample_costs(self):
        """Return deterministic per-index point counts for rank balancing."""
        costs = []
        for index in range(len(self)):
            obj_id = self.obj_dirs[index % len(self.obj_dirs)]
            ply_path = os.path.join(
                self._get_ply_dir(obj_id), "merged_model_render_color.ply"
            )
            costs.append(self._read_ply_vertex_count(ply_path))
        return costs

    @staticmethod
    def load_full_ply_data(
        ply_path,
        cond_point_dim=18,
        b_scale=None,
        b_offset=None,
        *,
        feature_schema="pbr_v1",
        normal_field_names=("nx", "ny", "nz"),
        normal_valid_field="normal_valid",
        normal_eps=1e-06,
        return_normal_valid_mask=False,
        return_view_source=False,
        point_indices=None,
    ):
        """读取完整 PLY 并提取 schema 对应的点特征。

        默认 schema 的 feats 语义仍固定为
        [albedo(3), roughness(1), metallic(1)]，
        因此只要 PLY 里存在 albedo_red/green/blue，就必须优先读取它们，
        不能再被 cond_point_dim / cond_point_mode 绑到 render RGB 上。
        若数据里没有显式 albedo 字段，才退回 red/green/blue，兼容旧数据。

        ``pbr_normal_world_v1`` 在末尾追加 Blender world normal (nx,ny,nz)。
        无效向量保持为零，且只可通过返回的 mask 在 loss 中排除；这里不做
        几何 normal 回退。
        """
        schema = get_feature_schema(feature_schema)
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]
        field_names = set(vertex.data.dtype.names or ())
        if point_indices is not None:
            vertex = vertex.data[point_indices]
        coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=0).astype(
            np.float32
        )
        if b_scale is not None and b_offset is not None:
            coords = coords
        else:
            coords = coords
        rgb_fields = ("red", "green", "blue")
        albedo_fields = ("albedo_red", "albedo_green", "albedo_blue")
        if set(albedo_fields).issubset(field_names):
            color_fields = albedo_fields
        elif set(rgb_fields).issubset(field_names):
            color_fields = rgb_fields
        else:
            raise ValueError(
                f"PLY {ply_path} 缺少可用颜色字段，找到: {sorted(field_names)}"
            )
        albedo = np.stack([vertex[field] for field in color_fields], axis=0).astype(
            np.float32
        )
        if albedo.size > 0 and float(albedo.max()) > 1.0:
            albedo = albedo / 255.0
        roughness = np.asarray(vertex["roughness"], dtype=np.float32)[None, :]
        metallic = np.asarray(vertex["metallic"], dtype=np.float32)[None, :]
        feats = np.concatenate([albedo, roughness, metallic], axis=0).astype(np.float32)
        normal_valid_mask = None
        if any((group.name == "normal" for group in schema.groups)):
            normal_fields = tuple((str(name) for name in normal_field_names))
            missing = [name for name in normal_fields if name not in field_names]
            if missing:
                raise ValueError(
                    f"PLY {ply_path} 缺少 Blender GT normal 字段 {missing}; 禁止使用几何 normal 回退"
                )
            normals_raw = np.stack(
                [vertex[field] for field in normal_fields], axis=0
            ).astype(np.float32)
            (normals, vector_valid) = normalize_world_normals_numpy(
                normals_raw, axis=0, eps=float(normal_eps)
            )
            normal_valid_mask = vector_valid.astype(np.bool_, copy=False)
            if normal_valid_field in field_names:
                stored_valid = np.asarray(vertex[normal_valid_field]).reshape(-1) > 0
                normal_valid_mask = normal_valid_mask & stored_valid
                normals[:, ~normal_valid_mask] = 0.0
            feats = np.concatenate([feats, normals], axis=0).astype(np.float32)
        view_source = None
        if return_view_source:
            if "view_source" in field_names:
                view_source = np.asarray(vertex["view_source"], dtype=np.int32)
            else:
                view_source = (
                    PBRConditionalDataset._infer_view_source_from_pointcloud_info(
                        ply_path, num_points=int(coords.shape[1])
                    ).astype(np.int32, copy=False)
                )
            if view_source.shape != (coords.shape[1],):
                raise ValueError(
                    f"PLY {ply_path} view_source shape={view_source.shape}, 期望 ({coords.shape[1]},)"
                )
        if feats.shape[0] != schema.feats_dim:
            raise ValueError(
                f"schema {schema.name!r} 期望 {schema.feats_dim} 通道，但 PLY {ply_path} 读取到 {feats.shape[0]} 通道"
            )
        result = [coords, feats]
        if return_normal_valid_mask:
            if normal_valid_mask is None:
                raise ValueError(
                    "return_normal_valid_mask=True 仅适用于包含 normal 的 feature_schema"
                )
            result.append(normal_valid_mask[None, :])
        if return_view_source:
            result.append(view_source)
        return tuple(result)

    @staticmethod
    def load_view_source_data(ply_path, query_coords=None):
        """读取 merged PLY 中的 view_source，并可按 query_coords 做最近邻映射。"""
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]
        field_names = set(vertex.data.dtype.names or ())
        ref_coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(
            np.float32
        )
        if "view_source" in field_names:
            view_source = np.asarray(vertex["view_source"], dtype=np.int64)
        else:
            view_source = PBRConditionalDataset._infer_view_source_from_pointcloud_info(
                ply_path, num_points=int(ref_coords.shape[0])
            )
        if query_coords is None:
            return view_source.copy()
        if (
            query_coords.ndim == 2
            and query_coords.shape[0] == 3
            and (query_coords.shape[1] != 3)
        ):
            query_pts = query_coords.T.astype(np.float32, copy=False)
        else:
            query_pts = query_coords.astype(np.float32, copy=False)
        if query_pts.shape == ref_coords.shape and np.array_equal(
            query_pts, ref_coords
        ):
            return view_source.copy()
        tree = cKDTree(ref_coords)
        (_, indices) = tree.query(query_pts)
        return view_source[indices]

    @staticmethod
    def _infer_view_source_from_pointcloud_info(ply_path, num_points):
        """当 merged PLY 缺少 view_source 时，回退到 pointcloud_info.json 的连续分段信息。"""
        ply_path = os.path.abspath(ply_path)
        ply_dir = os.path.dirname(ply_path)
        candidate_paths = [os.path.join(ply_dir, "pointcloud_info.json")]
        if os.path.basename(ply_dir) == "extract_output":
            candidate_paths.append(
                os.path.join(os.path.dirname(ply_dir), "pointcloud_info.json")
            )
        info_path = next(
            (path for path in candidate_paths if os.path.isfile(path)), None
        )
        if info_path is None:
            raise ValueError(
                f"PLY {ply_path} 缺少 view_source 字段，且未找到 pointcloud_info.json (checked: {candidate_paths})"
            )
        with open(info_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        views = payload.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError(f"{info_path} 缺少有效的 views 列表")
        declared_total = payload.get("total_points")
        if declared_total is not None and int(declared_total) != int(num_points):
            raise ValueError(
                f"{info_path} total_points={declared_total} 与 PLY 点数 {num_points} 不一致 @ {ply_path}"
            )
        view_source = np.zeros((int(num_points),), dtype=np.int64)
        filled = np.zeros((int(num_points),), dtype=bool)
        views_sorted = sorted(views, key=lambda item: int(item.get("start_idx", 0)))
        for idx, item in enumerate(views_sorted):
            if not isinstance(item, dict):
                raise ValueError(f"{info_path} views[{idx}] 不是 dict")
            try:
                view_id = int(item["view_id"])
                start_idx = int(item["start_idx"])
                count = int(item["count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{info_path} views[{idx}] 缺少或包含非法字段: {item}"
                ) from exc
            if view_id <= 0:
                raise ValueError(
                    f"{info_path} views[{idx}] 的 view_id 必须为正整数, 得到 {view_id}"
                )
            if start_idx < 0 or count < 0:
                raise ValueError(
                    f"{info_path} views[{idx}] 的 start_idx/count 非法: start_idx={start_idx}, count={count}"
                )
            end_idx = start_idx + count
            if end_idx > int(num_points):
                raise ValueError(
                    f"{info_path} views[{idx}] 超出点云范围: [{start_idx}, {end_idx}) vs num_points={num_points}"
                )
            if count == 0:
                continue
            if filled[start_idx:end_idx].any():
                raise ValueError(
                    f"{info_path} views[{idx}] 与其他视角区间重叠: [{start_idx}, {end_idx})"
                )
            view_source[start_idx:end_idx] = view_id
            filled[start_idx:end_idx] = True
        if not filled.all():
            missing = np.flatnonzero(~filled)
            preview = missing[:8].tolist()
            raise ValueError(
                f"{info_path} 未覆盖全部点，共缺失 {int((~filled).sum())} 个索引，示例: {preview}"
            )
        return view_source

    @staticmethod
    def _torch_load_with_fallback(path, map_location="cpu"):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except (pickle.UnpicklingError, RuntimeError, EOFError):
            return torch.load(path, map_location=map_location, weights_only=False)

    @staticmethod
    def load_precomputed_cond_point_features(feature_path, expected_dim=None):
        """从 .pt/.npz 读取离线预计算的 cond_point_colors [C, N]。"""
        ext = os.path.splitext(feature_path)[1].lower()
        if ext == ".pt":
            payload = PBRConditionalDataset._torch_load_with_fallback(feature_path)
            if isinstance(payload, dict):
                if "cond_point_colors" in payload:
                    features = payload["cond_point_colors"]
                elif "features" in payload:
                    features = payload["features"]
                else:
                    raise KeyError(
                        f"{feature_path} 缺少 cond_point_colors/features 字段"
                    )
            else:
                features = payload
            features = torch.as_tensor(features)
        elif ext == ".npz":
            payload = np.load(feature_path)
            if "cond_point_colors" not in payload.files:
                raise KeyError(f"{feature_path} 缺少 cond_point_colors 字段")
            features = torch.from_numpy(payload["cond_point_colors"])
        else:
            raise ValueError(f"不支持的 cond_point_colors 文件格式: {feature_path}")
        if features.ndim != 2:
            raise ValueError(
                f"cond_point_colors 必须是 [C, N]，得到 {tuple(features.shape)} @ {feature_path}"
            )
        if expected_dim is not None and features.shape[0] != expected_dim:
            raise ValueError(
                f"cond_point_colors 通道数不匹配: expected {expected_dim}, got {features.shape[0]} @ {feature_path}"
            )
        return features.float().contiguous()

    @staticmethod
    def load_precomputed_topk_cond_features(feature_path):
        """Load Top-K cache written by ``precompute_topk_cond_point_features``.

        Returns packed tensors plus K/view counts and any source-identity
        metadata stored by the precompute tool. Raises if the file does not
        match the ``topk_v1`` format.
        """
        if not feature_path.lower().endswith(".pt"):
            raise ValueError(f"Top-K cache must be .pt, got {feature_path}")
        payload = PBRConditionalDataset._torch_load_with_fallback(feature_path)
        if not isinstance(payload, dict) or payload.get("format") != "topk_v1":
            raise ValueError(
                f"{feature_path} 不是 topk_v1 格式缓存。请用 scripts/tools/precompute_topk_cond_point_features.py 重新生成。"
            )
        required = ("topk_rgb", "topk_conf", "topk_view_id", "k", "num_views")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{feature_path} 缺少字段: {missing}")
        topk_rgb = torch.as_tensor(payload["topk_rgb"]).float().contiguous()
        topk_conf = torch.as_tensor(payload["topk_conf"]).float().contiguous()
        topk_view_id = torch.as_tensor(payload["topk_view_id"]).long().contiguous()
        k = int(payload["k"])
        num_views = int(payload["num_views"])
        if topk_rgb.shape[0] != k or topk_rgb.shape[1] != 3:
            raise ValueError(
                f"topk_rgb 形状异常 {tuple(topk_rgb.shape)}, 期望 [{k}, 3, N]"
            )
        if topk_conf.shape[0] != k or topk_view_id.shape[0] != k:
            raise ValueError(
                f"topk_conf/topk_view_id 通道数不匹配 k={k}: {tuple(topk_conf.shape)} {tuple(topk_view_id.shape)}"
            )
        num_points = topk_rgb.shape[2]
        if topk_conf.shape[1] != num_points or topk_view_id.shape[1] != num_points:
            raise ValueError(
                f"Top-K 缓存各张量点数不一致: rgb={topk_rgb.shape}, conf={topk_conf.shape}, view_id={topk_view_id.shape}"
            )
        return {
            "topk_rgb": topk_rgb,
            "topk_conf": topk_conf,
            "topk_view_id": topk_view_id,
            "k": k,
            "num_views": num_views,
            "object_id": payload.get("object_id"),
            "point_cloud_path": payload.get("point_cloud_path"),
            "num_points": payload.get("num_points"),
            "point_cloud_prefer_extract_output": payload.get(
                "point_cloud_prefer_extract_output"
            ),
            "point_cloud_xyz_fingerprint": payload.get("point_cloud_xyz_fingerprint"),
        }

    @staticmethod
    def compute_ordered_xyz_fingerprint(coords, *, layout):
        """Hash ordered XYZ after canonicalizing to little-endian float32 [N, 3].

        ``layout`` is explicit so a valid three-point ``[3, 3]`` cloud is never
        interpreted ambiguously.  The version and canonical shape are included
        both in the digest header and in the returned metadata.  Consequently,
        identical point content hashes the same after worker-local staging,
        while even a pure point-order permutation changes the digest.
        """
        if torch.is_tensor(coords):
            array = coords.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            array = np.asarray(coords)
        if array.ndim != 2:
            raise ValueError(
                f"XYZ 指纹输入必须是二维数组，得到 shape={tuple(array.shape)}"
            )
        if layout == "channels_first":
            if int(array.shape[0]) != 3:
                raise ValueError(
                    f"channels_first XYZ 指纹输入必须是 [3, N]，得到 {tuple(array.shape)}"
                )
            xyz_n3 = array.T
        elif layout == "points_first":
            if int(array.shape[1]) != 3:
                raise ValueError(
                    f"points_first XYZ 指纹输入必须是 [N, 3]，得到 {tuple(array.shape)}"
                )
            xyz_n3 = array
        else:
            raise ValueError(
                f"XYZ 指纹 layout 必须是 'channels_first' 或 'points_first'，得到 {layout!r}"
            )
        canonical = np.ascontiguousarray(xyz_n3, dtype=np.dtype("<f4"))
        shape = [int(canonical.shape[0]), 3]
        version = PBRConditionalDataset.POINT_CLOUD_XYZ_FINGERPRINT_VERSION
        header = f"gs-pi:{version}\nshape={shape[0]},{shape[1]}\n".encode("ascii")
        digest = hashlib.sha256()
        digest.update(header)
        digest.update(canonical.tobytes(order="C"))
        return {"version": version, "shape": shape, "sha256": digest.hexdigest()}

    @staticmethod
    def _topk_point_cloud_path_identity(point_cloud_path):
        """Return ``(object_id, uses_extract_output)`` without comparing roots.

        Full/full jobs commonly stage the same object tree under a worker-local
        root, so comparing absolute paths would reject a valid cache.  The
        stable identity is the object directory name plus whether the PLY came
        from ``<obj>/`` or ``<obj>/extract_output/``.
        """
        if not isinstance(point_cloud_path, (str, os.PathLike)):
            return (None, None)
        normalized = os.path.normpath(os.fspath(point_cloud_path))
        if (
            not normalized
            or os.path.basename(normalized) != "merged_model_render_color.ply"
        ):
            return (None, None)
        parent = os.path.dirname(normalized)
        uses_extract_output = os.path.basename(parent) == "extract_output"
        object_dir = os.path.dirname(parent) if uses_extract_output else parent
        object_id = os.path.basename(object_dir)
        if not object_id:
            return (None, None)
        return (object_id, uses_extract_output)

    def _validate_full_topk_cache_identity(
        self, payload, *, obj_id, ply_dir, coords_np, full_point_count, feature_path
    ):
        """Reject same-sized Top-K caches built for another PLY identity.

        This is intentionally called only for ``full_pointcloud_training``.
        Historical 512-point paths keep accepting old caches that predate the
        identity metadata.
        """
        if full_point_count is None:
            raise ValueError(
                f"full/full Top-K 缓存校验需要 full_point_count @ {feature_path}"
            )
        if ply_dir is None:
            raise ValueError(
                f"full/full Top-K 缓存校验需要当前 ply_dir @ {feature_path}"
            )
        if coords_np is None:
            raise ValueError(
                f"full/full Top-K 缓存校验需要当前完整 coords @ {feature_path}"
            )
        cache_object_id = payload.get("object_id")
        if cache_object_id is None:
            raise ValueError(f"full/full Top-K 缓存缺少 object_id @ {feature_path}")
        if str(cache_object_id) != str(obj_id):
            raise ValueError(
                f"full/full Top-K 缓存 object_id 不匹配: cache={cache_object_id!r}, current={obj_id!r} @ {feature_path}"
            )
        cache_num_points = payload.get("num_points")
        if cache_num_points is None:
            raise ValueError(f"full/full Top-K 缓存缺少 num_points @ {feature_path}")
        try:
            cache_num_points = int(cache_num_points)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"full/full Top-K 缓存 num_points 非法: {cache_num_points!r} @ {feature_path}"
            ) from exc
        if cache_num_points != int(full_point_count):
            raise ValueError(
                f"full/full Top-K 缓存 num_points 与当前 PLY 不匹配: cache={cache_num_points}, current={int(full_point_count)} @ {feature_path}"
            )
        cache_point_cloud_path = payload.get("point_cloud_path")
        (cache_path_object_id, cache_uses_extract_output) = (
            self._topk_point_cloud_path_identity(cache_point_cloud_path)
        )
        if cache_path_object_id is None:
            raise ValueError(
                f"full/full Top-K 缓存 point_cloud_path 缺失或格式非法: {cache_point_cloud_path!r} @ {feature_path}"
            )
        if cache_path_object_id != str(obj_id):
            raise ValueError(
                f"full/full Top-K 缓存 point_cloud_path 对象不匹配: cache={cache_path_object_id!r}, current={obj_id!r} @ {feature_path}"
            )
        current_ply_path = os.path.join(
            os.fspath(ply_dir), "merged_model_render_color.ply"
        )
        (_, current_uses_extract_output) = self._topk_point_cloud_path_identity(
            current_ply_path
        )
        cache_preference = payload.get("point_cloud_prefer_extract_output")
        if not isinstance(cache_preference, (bool, np.bool_)):
            raise ValueError(
                f"full/full Top-K 缓存缺少或非法的 point_cloud_prefer_extract_output: {cache_preference!r} @ {feature_path}"
            )
        cache_preference = bool(cache_preference)
        if (
            cache_uses_extract_output != current_uses_extract_output
            or cache_preference != current_uses_extract_output
        ):
            current_branch = (
                "extract_output" if current_uses_extract_output else "object_root"
            )
            cache_path_branch = (
                "extract_output" if cache_uses_extract_output else "object_root"
            )
            cache_preference_branch = (
                "extract_output" if cache_preference else "object_root"
            )
            raise ValueError(
                f"full/full Top-K 缓存 PLY 分支不匹配: cache_path={cache_path_branch}, cache_preference={cache_preference_branch}, current={current_branch} @ {feature_path}"
            )
        cache_fingerprint = payload.get("point_cloud_xyz_fingerprint")
        if not isinstance(cache_fingerprint, dict):
            raise ValueError(
                f"full/full Top-K 缓存缺少或非法的 point_cloud_xyz_fingerprint @ {feature_path}。请用当前 PLY 重新生成 Top-K cache。"
            )
        required_fingerprint_fields = ("version", "shape", "sha256")
        missing_fingerprint_fields = [
            key for key in required_fingerprint_fields if key not in cache_fingerprint
        ]
        if missing_fingerprint_fields:
            raise ValueError(
                f"full/full Top-K 缓存 point_cloud_xyz_fingerprint 缺少字段 {missing_fingerprint_fields} @ {feature_path}"
            )
        current_fingerprint = self.compute_ordered_xyz_fingerprint(
            coords_np, layout="channels_first"
        )
        cache_version = cache_fingerprint["version"]
        if cache_version != current_fingerprint["version"]:
            raise ValueError(
                f"full/full Top-K 缓存有序 XYZ 指纹版本不匹配: cache={cache_version!r}, current={current_fingerprint['version']!r} @ {feature_path}"
            )
        raw_cache_shape = cache_fingerprint["shape"]
        if not isinstance(raw_cache_shape, (list, tuple)) or len(raw_cache_shape) != 2:
            raise ValueError(
                f"full/full Top-K 缓存有序 XYZ 指纹 shape 非法: {raw_cache_shape!r} @ {feature_path}"
            )
        try:
            cache_shape = [int(value) for value in raw_cache_shape]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"full/full Top-K 缓存有序 XYZ 指纹 shape 非法: {raw_cache_shape!r} @ {feature_path}"
            ) from exc
        if cache_shape != current_fingerprint["shape"]:
            raise ValueError(
                f"full/full Top-K 缓存有序 XYZ 指纹 shape 不匹配: cache={cache_shape}, current={current_fingerprint['shape']} @ {feature_path}"
            )
        cache_sha256 = cache_fingerprint["sha256"]
        if (
            not isinstance(cache_sha256, str)
            or cache_sha256 != current_fingerprint["sha256"]
        ):
            raise ValueError(
                f"full/full Top-K 缓存有序 XYZ 指纹不匹配: cache_sha256={cache_sha256!r}, current_sha256={current_fingerprint['sha256']!r} @ {feature_path}。对象、点数和 PLY 分支虽一致，但点坐标内容或顺序已变化。"
            )

    @staticmethod
    def pack_topk_cond_point_colors(topk_rgb, topk_conf, topk_view_id):
        """Pack Top-K tensors into a single [K*5, N] channel layout.

        Layout (slot-major):
          channels [0       : 3K]   RGB, flattened as rgb.reshape(K*3, N)
          channels [3K      : 4K]   confidence, one per slot
          channels [4K      : 5K]   view id (as float), one per slot
        """
        (k, three, n) = topk_rgb.shape
        if three != 3:
            raise ValueError(f"topk_rgb must be [K, 3, N], got {tuple(topk_rgb.shape)}")
        rgb_flat = topk_rgb.reshape(k * 3, n).contiguous()
        conf = topk_conf.to(dtype=rgb_flat.dtype).contiguous()
        vid = topk_view_id.to(dtype=rgb_flat.dtype).contiguous()
        return torch.cat([rgb_flat, conf, vid], dim=0).contiguous()

    def load_cond_point_colors(
        self,
        obj_id,
        *,
        coords_np=None,
        ply_dir=None,
        sample_indices=None,
        full_point_count=None,
    ):
        """优先读取离线预计算 cond_point_colors，不存在则回退到在线 KD-Tree 版本。"""
        if self.cond_point_dim <= 0:
            return None
        feature_path = self._get_cond_point_features_path(obj_id)
        cached_record = self._cond_point_features_cache.get(feature_path)
        if cached_record is not None:
            cond_point_colors = cached_record["features"]
            if (
                full_point_count is not None
                and cond_point_colors.shape[1] != full_point_count
            ):
                raise ValueError(
                    f"cached cond points={cond_point_colors.shape[1]} do not match PLY points={full_point_count} @ {feature_path}"
                )
            if sample_indices is not None:
                indices = torch.from_numpy(np.asarray(sample_indices, dtype=np.int64))
                cond_point_colors = cond_point_colors.index_select(1, indices)
            return cond_point_colors.float().contiguous()
        if self.cond_point_mode == "topk":
            if feature_path is None or not os.path.exists(feature_path):
                raise FileNotFoundError(
                    f"cond_point_mode='topk' 需要预计算缓存，缺失: {feature_path}"
                )
            payload = self.load_precomputed_topk_cond_features(feature_path)
            if bool(getattr(self, "full_pointcloud_training", False)):
                self._validate_full_topk_cache_identity(
                    payload,
                    obj_id=obj_id,
                    ply_dir=ply_dir,
                    coords_np=coords_np,
                    full_point_count=full_point_count,
                    feature_path=feature_path,
                )
                if not bool(torch.isfinite(payload["topk_rgb"]).all()):
                    raise ValueError(
                        f"full/full Top-K RGB 包含 NaN/Inf @ {feature_path}"
                    )
                if not bool(torch.isfinite(payload["topk_conf"]).all()):
                    raise ValueError(
                        f"full/full Top-K confidence 包含 NaN/Inf @ {feature_path}"
                    )
                topk_view_id = payload["topk_view_id"]
                if topk_view_id.numel() > 0 and (
                    int(topk_view_id.min()) < 0
                    or int(topk_view_id.max()) >= int(self.num_views)
                ):
                    raise ValueError(
                        f"full/full Top-K view id 超出 [0, num_views) 范围 @ {feature_path}"
                    )
            if payload["k"] != self.cond_point_top_k:
                raise ValueError(
                    f"Top-K 缓存 k={payload['k']} 与训练 cond_point_top_k={self.cond_point_top_k} 不一致 @ {feature_path}"
                )
            if payload["num_views"] != self.num_views:
                raise ValueError(
                    f"Top-K 缓存 num_views={payload['num_views']} 与训练 num_views={self.num_views} 不一致 @ {feature_path}"
                )
            topk_rgb = payload["topk_rgb"]
            topk_conf = payload["topk_conf"]
            topk_view_id = payload["topk_view_id"]
            cache_num_points = topk_rgb.shape[-1]
            if full_point_count is not None and cache_num_points != full_point_count:
                raise ValueError(
                    f"Top-K 缓存点数 {cache_num_points} 与 PLY 点数 {full_point_count} 不一致 @ {feature_path}. 请按当前 prefer_extract_output 配置重新生成缓存。"
                )
            if sample_indices is not None:
                sample_indices_t = torch.from_numpy(
                    np.asarray(sample_indices, dtype=np.int64)
                )
                topk_rgb = topk_rgb.index_select(-1, sample_indices_t)
                topk_conf = topk_conf.index_select(-1, sample_indices_t)
                topk_view_id = topk_view_id.index_select(-1, sample_indices_t)
            cond_point_colors = self.pack_topk_cond_point_colors(
                topk_rgb, topk_conf, topk_view_id
            )
            return cond_point_colors.float().contiguous()
        if feature_path is not None and os.path.exists(feature_path):
            cond_point_colors = self.load_precomputed_cond_point_features(
                feature_path, expected_dim=self.cond_point_dim
            )
            if (
                full_point_count is not None
                and cond_point_colors.shape[1] != full_point_count
            ):
                current_ply_path = None
                if ply_dir is not None:
                    current_ply_path = os.path.join(
                        ply_dir, "merged_model_render_color.ply"
                    )
                feature_source_path = None
                if feature_path.lower().endswith(".pt"):
                    try:
                        payload = torch.load(
                            feature_path, map_location="cpu", weights_only=True
                        )
                        if isinstance(payload, dict):
                            feature_source_path = payload.get("point_cloud_path")
                    except Exception:
                        feature_source_path = None
                raise ValueError(
                    f"预计算 cond_point_colors 点数不匹配: expected {full_point_count}, got {cond_point_colors.shape[1]} @ {feature_path}. current_ply={current_ply_path}, feature_source={feature_source_path}. 这通常是离线预计算时使用的 merged_model_render_color.ply 与训练当前 prefer_extract_output 指向的点云来源不一致；请按训练配置重新生成该对象的 .pt。"
                )
            if sample_indices is not None:
                sample_indices = np.asarray(sample_indices, dtype=np.int64)
                cond_point_colors = cond_point_colors.index_select(
                    1, torch.from_numpy(sample_indices)
                )
            return cond_point_colors.float().contiguous()
        if feature_path is not None and self.require_precomputed_cond_point_features:
            raise FileNotFoundError(f"缺少预计算 cond_point_colors: {feature_path}")
        if coords_np is None or ply_dir is None:
            raise ValueError("在线计算 cond_point_colors 需要提供 coords_np 和 ply_dir")
        cond_point_colors = self.compute_per_point_view_colors(
            coords_np,
            ply_dir,
            self.num_views,
            cond_point_dim=self.cond_point_dim,
            prefer_extract_output=self.prefer_extract_output
            if self.prefer_extract_output is not None
            else None
            if self.cond_point_dim == 24
            else False,
        )
        return torch.from_numpy(cond_point_colors).float()

    def _load_ply_data(
        self,
        ply_path,
        b_scale=None,
        b_offset=None,
        return_indices=False,
        return_normal_valid_mask=False,
        return_normal_relation_edges=False,
        return_view_source=False,
        obj_id=None,
        force_subsample=False,
    ):
        """Load aligned PLY tensors and optionally keep the complete object.

        Historical callers sample ``self.num_points`` with one shared index set.
        ``full_pointcloud_training`` is opt-in and returns all aligned points;
        ``force_subsample`` keeps normalization/statistics helpers bounded.
        """
        ply_key = os.path.realpath(ply_path)
        need_view_source = bool(
            return_view_source
            or getattr(self, "view_source_balanced_sampling_num_views", 0) > 0
            or getattr(self, "normal_view_source_relation_edges_enabled", False)
        )
        view_source_cache = getattr(self, "_view_source_cache", None)
        if view_source_cache is None:
            view_source_cache = {}
            self._view_source_cache = view_source_cache
        view_source = view_source_cache.get(ply_key) if need_view_source else None
        cached = self._ply_data_cache.get(ply_key)
        if cached is not None:
            (coords, feats, normal_valid_mask) = cached
            if return_normal_valid_mask and normal_valid_mask is None:
                raise ValueError(
                    f"cached PLY does not contain normal_valid_mask: {ply_path}"
                )
        else:
            loaded = self.load_full_ply_data(
                ply_path,
                cond_point_dim=self.cond_point_dim,
                b_scale=b_scale,
                b_offset=b_offset,
                feature_schema=self.feature_schema,
                normal_field_names=self.normal_field_names,
                normal_valid_field=self.normal_valid_field,
                normal_eps=self.normal_eps,
                return_normal_valid_mask=return_normal_valid_mask,
                return_view_source=need_view_source,
            )
            if return_normal_valid_mask and need_view_source:
                (coords, feats, normal_valid_mask, view_source) = loaded
            elif return_normal_valid_mask:
                (coords, feats, normal_valid_mask) = loaded
            elif need_view_source:
                (coords, feats, view_source) = loaded
                normal_valid_mask = None
            else:
                (coords, feats) = loaded
                normal_valid_mask = None
            if view_source is not None:
                source_dtype = (
                    np.int8
                    if int(view_source.min()) >= -128 and int(view_source.max()) <= 127
                    else np.int32
                )
                view_source = np.ascontiguousarray(view_source, dtype=source_dtype)
                view_source_cache[ply_key] = view_source
        if need_view_source and view_source is None:
            view_source = self.load_view_source_data(ply_path)
            source_dtype = (
                np.int8
                if int(view_source.min()) >= -128 and int(view_source.max()) <= 127
                else np.int32
            )
            view_source = view_source.astype(source_dtype, copy=False)
            view_source = np.ascontiguousarray(view_source)
            view_source_cache[ply_key] = view_source
        curr_p = coords.shape[1]
        normal_relation_edge_index = None
        normal_relation_edge_valid_mask = None
        use_complete_object = bool(
            getattr(self, "full_pointcloud_training", False)
        ) and (not bool(force_subsample))
        if use_complete_object and return_normal_relation_edges:
            raise ValueError("完整点云训练不支持固定预算 normal relation edge sampling")
        if use_complete_object:
            indices = None
        elif return_normal_relation_edges and getattr(
            self, "normal_view_source_sampling_enabled", False
        ):
            if not self.normal_view_source_sampling_enabled:
                raise ValueError(
                    "return_normal_relation_edges=true 但未启用 normal_view_source_sampling_enabled"
                )
            if obj_id is None:
                raise ValueError("return_normal_relation_edges=true 时必须传入 obj_id")
            (indices, normal_relation_edge_index, normal_relation_edge_valid_mask) = (
                self._sample_normal_view_source_point_indices(
                    obj_id,
                    ply_path,
                    coords,
                    normal_valid_mask if return_normal_valid_mask else None,
                )
            )
        elif getattr(self, "view_source_balanced_sampling_num_views", 0) > 0:
            indices = self._sample_view_source_balanced_point_indices(
                view_source, cache_key=ply_key
            )
        else:
            indices = self._sample_point_indices(
                ply_path,
                coords,
                feats,
                normal_valid_mask if return_normal_valid_mask else None,
            )
        sampled_coords = coords if indices is None else coords[:, indices]
        sampled_feats = feats if indices is None else feats[:, indices]
        sampled_view_source = (
            None
            if view_source is None
            else view_source
            if indices is None
            else view_source[indices]
        )
        sampled_normal_valid_mask = None
        if return_normal_valid_mask:
            sampled_normal_valid_mask = (
                normal_valid_mask if indices is None else normal_valid_mask[:, indices]
            )
        if return_normal_relation_edges and getattr(
            self, "normal_view_source_relation_edges_enabled", False
        ):
            (normal_relation_edge_index, normal_relation_edge_valid_mask) = (
                self._build_sampled_same_view_relation_edges(
                    sampled_coords, sampled_view_source, sampled_normal_valid_mask
                )
            )
        elif return_normal_relation_edges and normal_relation_edge_index is None:
            raise ValueError(
                "return_normal_relation_edges=true，但既未启用旧版 grouped sampling，也未启用 sampled same-view relation edges"
            )
        if return_normal_valid_mask:
            if return_indices:
                result = (
                    sampled_coords,
                    sampled_feats,
                    sampled_normal_valid_mask,
                    indices,
                    curr_p,
                )
                if return_normal_relation_edges:
                    result += (
                        normal_relation_edge_index,
                        normal_relation_edge_valid_mask,
                    )
                if return_view_source:
                    result += (sampled_view_source,)
                return result
            result = (sampled_coords, sampled_feats, sampled_normal_valid_mask)
            if return_view_source:
                result += (sampled_view_source,)
            return result
        if return_indices:
            result = (sampled_coords, sampled_feats, indices, curr_p)
            if return_normal_relation_edges:
                result += (normal_relation_edge_index, normal_relation_edge_valid_mask)
            if return_view_source:
                result += (sampled_view_source,)
            return result
        result = (sampled_coords, sampled_feats)
        if return_view_source:
            result += (sampled_view_source,)
        return result

    def _build_normal_hard_pool(self, ply_path, coords, feats, normal_valid_mask=None):
        """Cache points with the largest local GT-normal variation."""
        cache_key = os.path.realpath(ply_path)
        cached = self._normal_hard_pool_cache.get(cache_key)
        if cached is not None:
            return cached
        normal_slice = self.feature_schema.group("normal").channel_slice
        normals = np.asarray(feats[normal_slice], dtype=np.float32)
        normal_lengths = np.linalg.norm(normals, axis=0)
        valid = np.isfinite(normals).all(axis=0) & (normal_lengths > self.normal_eps)
        valid &= np.isfinite(coords).all(axis=0)
        if normal_valid_mask is not None:
            valid &= np.asarray(normal_valid_mask).reshape(-1).astype(bool)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 2:
            hard_pool = valid_indices.astype(np.int64, copy=False)
            self._normal_hard_pool_cache[cache_key] = hard_pool
            return hard_pool
        seed = zlib.crc32(os.fsencode(cache_key)) & 4294967295
        rng = np.random.default_rng(seed)
        if valid_indices.size > self.normal_hard_max_candidates:
            candidate_indices = rng.choice(
                valid_indices, self.normal_hard_max_candidates, replace=False
            )
        else:
            candidate_indices = valid_indices
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_coords = np.ascontiguousarray(coords[:, candidate_indices].T)
        candidate_normals = np.ascontiguousarray(normals[:, candidate_indices].T)
        candidate_normals /= np.linalg.norm(
            candidate_normals, axis=1, keepdims=True
        ).clip(min=self.normal_eps)
        neighbor_count = min(self.normal_hard_knn, candidate_indices.size - 1)
        tree = cKDTree(candidate_coords)
        (_, neighbor_local_indices) = tree.query(candidate_coords, k=neighbor_count + 1)
        if neighbor_local_indices.ndim == 1:
            neighbor_local_indices = neighbor_local_indices[:, None]
        neighbor_local_indices = neighbor_local_indices[:, 1:]
        neighbor_normals = candidate_normals[neighbor_local_indices]
        local_dot = np.einsum("md,mkd->mk", candidate_normals, neighbor_normals)
        variation = np.mean(1.0 - np.clip(local_dot, -1.0, 1.0), axis=1)
        hard_count = max(
            1, int(np.ceil(candidate_indices.size * self.normal_hard_pool_fraction))
        )
        if hard_count >= candidate_indices.size:
            selected_local = np.arange(candidate_indices.size)
        else:
            selected_local = np.argpartition(variation, -hard_count)[-hard_count:]
        hard_pool = np.ascontiguousarray(
            candidate_indices[selected_local], dtype=np.int64
        )
        self._normal_hard_pool_cache[cache_key] = hard_pool
        self._normal_hard_pool_stats[cache_key] = {
            "valid_points": int(valid_indices.size),
            "candidates": int(candidate_indices.size),
            "hard_pool": int(hard_pool.size),
            "variation_mean": float(variation.mean()),
            "variation_threshold": float(variation[selected_local].min()),
        }
        return hard_pool

    def _sample_point_indices(self, ply_path, coords, feats, normal_valid_mask=None):
        """Mix uniform coverage with cached high-normal-variation points."""
        point_count = int(coords.shape[1])
        if self.normal_hard_sampling_ratio <= 0.0:
            return np.random.choice(
                point_count, self.num_points, replace=point_count < self.num_points
            )
        hard_pool = self._build_normal_hard_pool(
            ply_path, coords, feats, normal_valid_mask=normal_valid_mask
        )
        if hard_pool.size == 0:
            return np.random.choice(
                point_count, self.num_points, replace=point_count < self.num_points
            )
        desired_hard = int(round(self.num_points * self.normal_hard_sampling_ratio))
        if point_count >= self.num_points:
            hard_count = min(desired_hard, int(hard_pool.size))
            hard_indices = np.random.choice(hard_pool, hard_count, replace=False)
            available = np.ones(point_count, dtype=np.bool_)
            available[hard_indices] = False
            uniform_pool = np.flatnonzero(available)
            uniform_count = self.num_points - hard_count
            uniform_indices = np.random.choice(
                uniform_pool, uniform_count, replace=uniform_pool.size < uniform_count
            )
        else:
            hard_count = desired_hard
            hard_indices = np.random.choice(
                hard_pool, hard_count, replace=hard_pool.size < hard_count
            )
            uniform_count = self.num_points - hard_count
            uniform_indices = np.random.choice(point_count, uniform_count, replace=True)
        indices = np.concatenate([hard_indices, uniform_indices]).astype(
            np.int64, copy=False
        )
        np.random.shuffle(indices)
        return indices

    @staticmethod
    def _build_view_source_group_descriptors(view_source):
        """Compactly index contiguous source segments for repeated sampling."""
        source = np.asarray(view_source).reshape(-1)
        if source.size == 0:
            return {}
        boundaries = np.flatnonzero(source[1:] != source[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [source.size]))
        segments = {}
        for start, stop in zip(starts.tolist(), stops.tolist()):
            view_id = int(source[start])
            segments.setdefault(view_id, []).append((int(start), int(stop)))
        descriptors = {}
        for view_id, view_segments in segments.items():
            if len(view_segments) == 1:
                (start, stop) = view_segments[0]
                descriptors[view_id] = ("range", start, stop)
            else:
                descriptors[view_id] = (
                    "indices",
                    np.concatenate(
                        [
                            np.arange(start, stop, dtype=np.int64)
                            for (start, stop) in view_segments
                        ]
                    ),
                )
        return descriptors

    def _sample_view_source_balanced_point_indices(self, view_source, cache_key=None):
        """Choose V source views uniformly and an equal point budget per view."""
        source = np.asarray(view_source).reshape(-1)
        selected_view_count = int(self.view_source_balanced_sampling_num_views)
        points_per_view = int(self.num_points) // selected_view_count
        group_cache = getattr(self, "_view_source_group_cache", None)
        if group_cache is None:
            group_cache = {}
            self._view_source_group_cache = group_cache
        descriptor_key = cache_key if cache_key is not None else id(view_source)
        descriptors = group_cache.get(descriptor_key)
        if descriptors is None:
            descriptors = self._build_view_source_group_descriptors(source)
            group_cache[descriptor_key] = descriptors
        counts = {
            view_id: descriptor[2] - descriptor[1]
            if descriptor[0] == "range"
            else int(descriptor[1].size)
            for (view_id, descriptor) in descriptors.items()
        }
        eligible_views = np.asarray(
            sorted(
                (
                    view_id
                    for (view_id, count) in counts.items()
                    if count >= points_per_view
                )
            )
        )
        if eligible_views.size < selected_view_count:
            raise RuntimeError(
                f"可用 view_source 不足，无法完成等额无放回采样: need_views={selected_view_count}, points_per_view={points_per_view}, eligible={eligible_views.tolist()}, counts={counts}"
            )
        selected_views = np.random.choice(
            eligible_views, selected_view_count, replace=False
        )
        sampled = []
        for view_id in selected_views:
            descriptor = descriptors[int(view_id)]
            if descriptor[0] == "range":
                (_, start, stop) = descriptor
                offsets = np.random.choice(stop - start, points_per_view, replace=False)
                sampled.append(offsets + start)
            else:
                candidates = descriptor[1]
                sampled.append(
                    np.random.choice(candidates, points_per_view, replace=False)
                )
        indices = np.concatenate(sampled).astype(np.int64, copy=False)
        np.random.shuffle(indices)
        return indices

    def _build_sampled_same_view_relation_edges(
        self, sampled_coords, sampled_view_source, sampled_normal_valid_mask=None
    ):
        """Build fixed-size KNN edges inside each already-sampled source view.

        This method consumes no RNG state. B and C therefore select identical
        points and DINO views under the same seed; C differs only by this loss
        graph and its configured residual-relation weight.
        """
        coords = np.asarray(sampled_coords, dtype=np.float32)
        source = np.asarray(sampled_view_source).reshape(-1)
        if coords.ndim != 2 or coords.shape[0] != 3:
            raise ValueError(f"sampled_coords 必须为 [3,N]，得到 {coords.shape}")
        if source.shape != (coords.shape[1],):
            raise ValueError(
                f"sampled_view_source 与 coords 点数不一致: source={source.shape}, coords={coords.shape}"
            )
        neighbor_k = int(self.normal_view_source_neighbor_k)
        anchors_per_view = int(self.normal_view_source_relation_anchors_per_view)
        view_ids = np.unique(source)
        edge_count = int(view_ids.size) * anchors_per_view * neighbor_k
        edge_index = np.empty((2, edge_count), dtype=np.int64)
        edge_valid = np.ones((edge_count,), dtype=np.bool_)
        if sampled_normal_valid_mask is None:
            normal_valid = np.ones((coords.shape[1],), dtype=np.bool_)
        else:
            normal_valid = (
                np.asarray(sampled_normal_valid_mask)
                .reshape(-1)
                .astype(np.bool_, copy=False)
            )
        cursor = 0
        for view_id in view_ids:
            positions = np.flatnonzero(source == view_id)
            if positions.size <= neighbor_k:
                raise RuntimeError(
                    f"same-view relation KNN 点数不足: view={int(view_id)}, points={positions.size}, K={neighbor_k}"
                )
            anchor_offsets = np.linspace(
                0, positions.size - 1, num=anchors_per_view, dtype=np.int64
            )
            anchors = positions[anchor_offsets]
            tree = cKDTree(np.ascontiguousarray(coords[:, positions].T))
            (_, neighbor_offsets) = tree.query(
                np.ascontiguousarray(coords[:, anchors].T), k=neighbor_k + 1
            )
            neighbor_offsets = np.asarray(neighbor_offsets)
            if neighbor_offsets.ndim == 1:
                neighbor_offsets = neighbor_offsets[:, None]
            for anchor, anchor_offset, row in zip(
                anchors.tolist(), anchor_offsets.tolist(), neighbor_offsets
            ):
                row = np.asarray(row, dtype=np.int64)
                row = row[row != int(anchor_offset)][:neighbor_k]
                if row.size != neighbor_k:
                    raise RuntimeError(
                        f"same-view relation KNN 未返回足量非自身邻居: view={int(view_id)}, anchor={int(anchor)}, got={row.size}, K={neighbor_k}"
                    )
                neighbors = positions[row]
                stop = cursor + neighbor_k
                edge_index[0, cursor:stop] = int(anchor)
                edge_index[1, cursor:stop] = neighbors
                edge_valid[cursor:stop] = (
                    normal_valid[int(anchor)] & normal_valid[neighbors]
                )
                cursor = stop
        if cursor != edge_count:
            raise RuntimeError(
                f"same-view relation edge 数量异常: {cursor} vs {edge_count}"
            )
        return (edge_index, edge_valid)

    @classmethod
    def normal_view_source_knn_cache_filename(cls, obj_id, neighbor_k):
        return f"{obj_id}.same_view_knn_k{int(neighbor_k)}.npy"

    def _normal_view_source_knn_cache_paths(self, obj_id):
        filename = self.normal_view_source_knn_cache_filename(
            obj_id, self.normal_view_source_neighbor_k
        )
        array_path = os.path.join(self.normal_view_source_knn_cache_dir, filename)
        metadata_path = os.path.splitext(array_path)[0] + ".json"
        return (array_path, metadata_path)

    def _load_normal_view_source_knn(self, obj_id, ply_path, point_count):
        """Load and validate a mmap-backed, same-view-only full-cloud KNN cache."""
        cache_key = (
            str(obj_id),
            os.path.realpath(ply_path),
            self.normal_view_source_neighbor_k,
        )
        cached = self._normal_view_source_knn_cache.get(cache_key)
        if cached is not None:
            return cached
        (array_path, metadata_path) = self._normal_view_source_knn_cache_paths(obj_id)
        if not os.path.isfile(array_path) or not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"缺少同 view_source KNN 缓存。请先运行 scripts/tools/precompute_normal_view_source_knn.py；期望文件: {array_path} 和 {metadata_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        expected_metadata = {
            "format": self.NORMAL_VIEW_SOURCE_KNN_CACHE_FORMAT,
            "object_id": str(obj_id),
            "point_count": int(point_count),
            "neighbor_k": self.normal_view_source_neighbor_k,
            "same_view_only": True,
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for (key, expected) in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"同 view_source KNN 缓存 metadata 不匹配 @ {metadata_path}: {mismatches}"
            )
        ply_stat = os.stat(ply_path)
        for key, current in (
            ("point_cloud_size", int(ply_stat.st_size)),
            ("point_cloud_mtime_ns", int(ply_stat.st_mtime_ns)),
        ):
            cached_value = metadata.get(key)
            if cached_value is not None and int(cached_value) != current:
                raise ValueError(
                    f"同 view_source KNN 缓存已过期 ({key}: cache={cached_value}, current={current}) @ {array_path}"
                )
        neighbors = np.load(array_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(point_count), self.normal_view_source_neighbor_k)
        if neighbors.shape != expected_shape:
            raise ValueError(
                f"同 view_source KNN 缓存 shape={neighbors.shape}，期望 {expected_shape} @ {array_path}"
            )
        if neighbors.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"同 view_source KNN 缓存必须是整数 dtype，得到 {neighbors.dtype} @ {array_path}"
            )
        cached = (neighbors, metadata)
        self._normal_view_source_knn_cache[cache_key] = cached
        return cached

    def _sample_normal_view_source_point_indices(
        self, obj_id, ply_path, coords, normal_valid_mask=None
    ):
        """Sample anchor+K groups and explicit same-``view_source`` edges.

        The K neighbours come from the full point cloud, not from the other
        randomly selected points.  No cross-view fallback is allowed: if the
        configured number of valid groups cannot be built, sampling fails with
        an actionable error instead of silently changing edge semantics.
        """
        point_count = int(coords.shape[1])
        neighbor_k = self.normal_view_source_neighbor_k
        anchor_count = self.normal_view_source_anchor_count
        (neighbors, cache_metadata) = self._load_normal_view_source_knn(
            obj_id, ply_path, point_count
        )
        if normal_valid_mask is None:
            normal_valid = np.ones(point_count, dtype=np.bool_)
        else:
            normal_valid = (
                np.asarray(normal_valid_mask).reshape(-1).astype(np.bool_, copy=False)
            )
            if normal_valid.shape != (point_count,):
                raise ValueError(
                    f"normal_valid_mask 点数与 PLY 不匹配: mask={normal_valid.shape}, points={point_count}"
                )
        max_distance = None
        if self.normal_view_source_max_distance_ratio is not None:
            bbox_diagonal = cache_metadata.get("bbox_diagonal")
            if bbox_diagonal is None or float(bbox_diagonal) <= 0.0:
                raise ValueError(
                    f"启用 normal_view_source_max_distance_ratio 时，KNN cache metadata 必须包含正数 bbox_diagonal @ {ply_path}"
                )
            max_distance = (
                float(bbox_diagonal) * self.normal_view_source_max_distance_ratio
            )
        grouped_full_indices = []
        selected_full_indices = set()
        edge_count = anchor_count * neighbor_k
        edge_index = np.zeros((2, edge_count), dtype=np.int64)
        edge_valid_mask = np.zeros((edge_count,), dtype=np.bool_)
        max_attempts = max(4096, anchor_count * 256)
        candidate_anchors = np.random.randint(0, point_count, size=max_attempts)
        accepted_groups = 0
        for anchor in candidate_anchors:
            anchor = int(anchor)
            neighbor_row = np.asarray(neighbors[anchor, :neighbor_k], dtype=np.int64)
            if (
                not normal_valid[anchor]
                or np.any(neighbor_row < 0)
                or np.any(neighbor_row >= point_count)
                or np.any(neighbor_row == anchor)
                or (np.unique(neighbor_row).size != neighbor_k)
                or (not normal_valid[neighbor_row].all())
            ):
                continue
            group = [anchor, *(int(value) for value in neighbor_row)]
            if any((value in selected_full_indices for value in group)):
                continue
            group_coords = coords[:, group]
            if not np.isfinite(group_coords).all():
                continue
            if max_distance is not None:
                neighbor_distances = np.linalg.norm(
                    group_coords[:, 1:] - group_coords[:, :1], axis=0
                )
                if not np.isfinite(neighbor_distances).all() or np.any(
                    neighbor_distances > max_distance
                ):
                    continue
            group_start = len(grouped_full_indices)
            grouped_full_indices.extend(group)
            selected_full_indices.update(group)
            edge_start = accepted_groups * neighbor_k
            edge_stop = edge_start + neighbor_k
            edge_index[0, edge_start:edge_stop] = group_start
            edge_index[1, edge_start:edge_stop] = np.arange(
                group_start + 1, group_start + 1 + neighbor_k, dtype=np.int64
            )
            edge_valid_mask[edge_start:edge_stop] = True
            accepted_groups += 1
            if accepted_groups == anchor_count:
                break
        if accepted_groups != anchor_count:
            raise RuntimeError(
                f"无法构造足量的同 view_source 局部分组，且不会回退到跨视角邻居: object={obj_id}, accepted={accepted_groups}/{anchor_count}, K={neighbor_k}, attempts={max_attempts}, max_distance_ratio={self.normal_view_source_max_distance_ratio}. 请检查/重建 KNN 缓存，或降低 anchor_count/放宽距离阈值。"
            )
        grouped_full_indices = np.asarray(grouped_full_indices, dtype=np.int64)
        uniform_count = self.num_points - int(grouped_full_indices.size)
        blocked = set((int(value) for value in grouped_full_indices))
        available_count = point_count - len(blocked)
        if uniform_count == 0:
            uniform_indices = np.empty((0,), dtype=np.int64)
        elif available_count >= uniform_count and available_count > uniform_count * 4:
            selected_uniform = []
            selected_or_blocked = set(blocked)
            while len(selected_uniform) < uniform_count:
                draw_count = max(64, 2 * (uniform_count - len(selected_uniform)))
                candidates = np.random.randint(0, point_count, size=draw_count)
                for candidate in candidates:
                    candidate = int(candidate)
                    if candidate in selected_or_blocked:
                        continue
                    selected_uniform.append(candidate)
                    selected_or_blocked.add(candidate)
                    if len(selected_uniform) == uniform_count:
                        break
            uniform_indices = np.asarray(selected_uniform, dtype=np.int64)
        else:
            available = np.ones(point_count, dtype=np.bool_)
            available[grouped_full_indices] = False
            uniform_pool = np.flatnonzero(available)
            if uniform_pool.size == 0:
                uniform_indices = np.random.choice(
                    point_count, uniform_count, replace=True
                )
            else:
                uniform_indices = np.random.choice(
                    uniform_pool,
                    uniform_count,
                    replace=uniform_pool.size < uniform_count,
                )
            uniform_indices = uniform_indices.astype(np.int64, copy=False)
        sample_indices = np.concatenate([grouped_full_indices, uniform_indices]).astype(
            np.int64, copy=False
        )
        permutation = np.random.permutation(self.num_points)
        inverse_permutation = np.empty(self.num_points, dtype=np.int64)
        inverse_permutation[permutation] = np.arange(self.num_points, dtype=np.int64)
        sample_indices = sample_indices[permutation]
        edge_index = inverse_permutation[edge_index]
        return (sample_indices, edge_index, edge_valid_mask)

    @staticmethod
    def compute_per_point_view_features_projected(
        points, extrinsics, intrinsics, depth_maps, rgb_maps, **kwargs
    ):
        """2D 投影 + 深度匹配版 cond_point_colors，返回 [24, N]。"""
        return build_cond_point_features_2d_projection(
            points=points,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth_maps=depth_maps,
            rgb_maps=rgb_maps,
            **kwargs,
        )

    @staticmethod
    def compute_per_point_view_colors(
        coords_np,
        obj_dir,
        num_views=6,
        dist_thresh=0.02,
        cond_point_dim=18,
        prefer_extract_output=None,
    ):
        """
        coords_np: [3, N] 采样点世界坐标 (numpy)
        obj_dir:   物体目录路径 (含 view_01.ply ~ view_06.ply)
        cond_point_dim: 18 → 仅返回 RGB [18, N]; 24 → RGB + one-hot view_source [24, N]
        返回:      numpy float32, 归一化到 [-1, 1]
        """
        N = coords_np.shape[1]
        query_pts = coords_np.T
        all_colors = []
        best_dists = np.full(N, np.inf, dtype=np.float32)
        view_source = np.zeros(N, dtype=np.int64)
        view_reader = ViewPointCloudReader(obj_dir, prefer_extract_output)
        for vi in range(1, num_views + 1):
            ply_path = PBRConditionalDataset.resolve_view_ply_path(
                obj_dir, vi, prefer_extract_output=prefer_extract_output
            )
            try:
                v = view_reader.read(vi)
            except FileNotFoundError:
                all_colors.append(np.zeros((3, N), dtype=np.float32))
                continue
            field_names = set(v.dtype.names or ())
            view_xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
            if {"red", "green", "blue"}.issubset(field_names):
                color_fields = ("red", "green", "blue")
            elif {"albedo_red", "albedo_green", "albedo_blue"}.issubset(field_names):
                color_fields = ("albedo_red", "albedo_green", "albedo_blue")
            else:
                raise ValueError(
                    f"{ply_path} 缺少 red/green/blue 或 albedo_red/green/albedo_blue 字段"
                )
            view_rgb = np.stack([v[field] for field in color_fields], axis=-1).astype(
                np.float32
            )
            if view_rgb.size > 0 and float(view_rgb.max()) > 1.0:
                view_rgb = view_rgb / 255.0
            tree = cKDTree(view_xyz)
            (dists, indices) = tree.query(query_pts)
            visible = dists <= dist_thresh
            colors = view_rgb[indices]
            colors[~visible] = 0.0
            all_colors.append(colors.T)
            closer = dists < best_dists
            best_dists[closer] = dists[closer]
            view_source[closer] = vi - 1
        rgb = np.concatenate(all_colors, axis=0)
        if cond_point_dim == 24:
            onehot = np.zeros((num_views, N), dtype=np.float32)
            onehot[view_source, np.arange(N)] = 1.0
            pvc = np.concatenate([rgb, onehot], axis=0)
        else:
            pvc = rgb
        pvc = (pvc * 2.0 - 1.0).astype(np.float32)
        return pvc

    def load_light_probe(self, obj_id):
        return None

    def __getitem__(self, idx):
        if not self.obj_dirs:
            raise IndexError("PBRConditionalDataset 中没有可用对象")
        idx = int(idx) % len(self.obj_dirs)
        obj_id = self.obj_dirs[idx]
        ply_dir = self._get_ply_dir(obj_id)
        ply_path = os.path.join(ply_dir, "merged_model_render_color.ply")
        normal_relation_edge_index = None
        normal_relation_edge_valid_mask = None
        view_source = None
        return_relation_edges = bool(
            self.normal_view_source_sampling_enabled
            or getattr(self, "normal_view_source_relation_edges_enabled", False)
        )
        return_view_source = bool(getattr(self, "return_view_source", False))
        if self.return_normal_valid_mask:
            loaded = self._load_ply_data(
                ply_path,
                b_scale=None,
                b_offset=None,
                return_indices=True,
                return_normal_valid_mask=True,
                return_normal_relation_edges=return_relation_edges,
                return_view_source=return_view_source,
                obj_id=obj_id,
            )
            if return_relation_edges:
                (
                    coords,
                    feats,
                    normal_valid_mask,
                    sample_indices,
                    full_point_count,
                    normal_relation_edge_index,
                    normal_relation_edge_valid_mask,
                ) = loaded[:7]
            else:
                (coords, feats, normal_valid_mask, sample_indices, full_point_count) = (
                    loaded[:5]
                )
            if return_view_source:
                view_source = loaded[-1]
        else:
            loaded = self._load_ply_data(
                ply_path,
                b_scale=None,
                b_offset=None,
                return_indices=True,
                return_normal_relation_edges=return_relation_edges,
                return_view_source=return_view_source,
                obj_id=obj_id,
            )
            if return_relation_edges:
                (
                    coords,
                    feats,
                    sample_indices,
                    full_point_count,
                    normal_relation_edge_index,
                    normal_relation_edge_valid_mask,
                ) = loaded[:6]
            else:
                (coords, feats, sample_indices, full_point_count) = loaded[:4]
            if return_view_source:
                view_source = loaded[-1]
        cond_tokens = None
        if self.dino_features_dir is not None:
            feature_path = self._get_dino_feature_path(obj_id)
            if os.path.exists(feature_path):
                cond_tokens = self._load_and_process_dino_features(feature_path)
            elif self.skip_missing_dino_features or (
                getattr(self, "full_pointcloud_training", False)
                and getattr(self, "dino_target_side", None) is not None
            ):
                raise FileNotFoundError(f"缺少 DINO 特征: {feature_path}")
            else:
                cond_tokens = torch.zeros((600, 512))
            self._validate_full_dino_condition(cond_tokens, feature_path=feature_path)
        cond_point_colors = self.load_cond_point_colors(
            obj_id,
            coords_np=coords,
            ply_dir=ply_dir,
            sample_indices=sample_indices,
            full_point_count=full_point_count,
        )
        returned_point_count = int(coords.shape[1])
        if int(feats.shape[1]) != returned_point_count:
            raise ValueError(
                f"coords/feats 点数不一致 @ {obj_id}: {returned_point_count} vs {feats.shape[1]}"
            )
        if (
            self.return_normal_valid_mask
            and int(normal_valid_mask.shape[-1]) != returned_point_count
        ):
            raise ValueError(
                f"normal_valid_mask 点数不一致 @ {obj_id}: {normal_valid_mask.shape[-1]} vs {returned_point_count}"
            )
        if (
            cond_point_colors is not None
            and int(cond_point_colors.shape[-1]) != returned_point_count
        ):
            raise ValueError(
                f"cond_point_colors 点数不一致 @ {obj_id}: {cond_point_colors.shape[-1]} vs {returned_point_count}"
            )
        result = {
            "coords": torch.from_numpy(coords).float(),
            "feats": torch.from_numpy(feats).float(),
            "name": obj_id,
        }
        if hasattr(self, "load_source_pixels"):
            source_uv = self.load_source_pixels(obj_id, sample_indices)
            if source_uv is not None:
                result["source_uv"] = torch.from_numpy(
                    np.array(source_uv, dtype=np.float32)
                )
        if view_source is not None:
            result["view_source"] = torch.from_numpy(
                np.asarray(view_source, dtype=np.int64)
            ).long()
        if self.return_normal_valid_mask:
            result["normal_valid_mask"] = torch.from_numpy(normal_valid_mask).bool()
        if normal_relation_edge_index is not None:
            result["normal_relation_edge_index"] = torch.from_numpy(
                normal_relation_edge_index
            ).long()
            result["normal_relation_edge_valid_mask"] = torch.from_numpy(
                normal_relation_edge_valid_mask
            ).bool()
        if cond_point_colors is not None:
            result["cond_point_colors"] = cond_point_colors.float()
        if cond_tokens is not None:
            result["cond_tokens"] = cond_tokens
        light_probe = self.load_light_probe(obj_id)
        if light_probe is not None:
            result["light_probe"] = light_probe
        if self.full_pointcloud_training:
            result["full_point_count"] = torch.tensor(
                int(full_point_count), dtype=torch.long
            )
        return result

    def preflight_full_pointcloud_training_inputs(self, progress_interval=10):
        """Validate every complete training object before the first GPU step.

        This deliberately exercises the same PLY/DINO/Top-K loaders as
        ``__getitem__``.  Calling it once on rank 0 catches missing, stale,
        reordered, malformed, or non-finite caches before another DDP rank can
        enter an all-reduce.  Historical sampled training never calls this
        opt-in full/full gate.
        """
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        try:
            return self._preflight_full_pointcloud_training_inputs_impl(
                progress_interval=progress_interval
            )
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)

    def _preflight_full_pointcloud_training_inputs_impl(self, progress_interval=10):
        """Implementation separated so RNG state is restored on every exit."""
        if not self.full_pointcloud_training:
            raise RuntimeError(
                "preflight_full_pointcloud_training_inputs 仅适用于 full/full"
            )
        object_ids = list(self.obj_dirs)
        if not object_ids:
            raise RuntimeError("full/full 启动预检发现训练对象列表为空")
        total_points = 0
        min_points = None
        max_points = 0
        interval = max(1, int(progress_interval))
        for index, expected_obj_id in enumerate(object_ids):
            sample = self[index]
            actual_obj_id = str(sample.get("name"))
            if actual_obj_id != str(expected_obj_id):
                raise ValueError(
                    f"full/full 启动预检对象顺序异常: expected={expected_obj_id!r}, actual={actual_obj_id!r}"
                )
            coords = sample.get("coords")
            feats = sample.get("feats")
            if not torch.is_tensor(coords) or coords.ndim != 2 or coords.shape[0] != 3:
                raise ValueError(
                    f"full/full coords 必须为 [3,N] @ {expected_obj_id}, 得到 {getattr(coords, 'shape', None)}"
                )
            point_count = int(coords.shape[1])
            if point_count <= 0:
                raise ValueError(f"full/full 点云为空 @ {expected_obj_id}")
            if (
                not torch.is_tensor(feats)
                or feats.ndim != 2
                or int(feats.shape[1]) != point_count
            ):
                raise ValueError(
                    f"full/full feats 与 coords 未对齐 @ {expected_obj_id}: coords={tuple(coords.shape)}, feats={getattr(feats, 'shape', None)}"
                )
            declared_count = sample.get("full_point_count")
            if declared_count is None or int(declared_count) != point_count:
                raise ValueError(
                    f"full/full full_point_count 未对齐 @ {expected_obj_id}: declared={declared_count}, actual={point_count}"
                )
            tensors_to_check = {"coords": coords, "feats": feats}
            if self.return_normal_valid_mask:
                normal_mask = sample.get("normal_valid_mask")
                if not torch.is_tensor(normal_mask) or tuple(normal_mask.shape) != (
                    1,
                    point_count,
                ):
                    raise ValueError(
                        f"full/full normal_valid_mask 必须为 [1,N] @ {expected_obj_id}, 得到 {getattr(normal_mask, 'shape', None)}"
                    )
            if self.dino_features_dir is not None:
                cond_tokens = sample.get("cond_tokens")
                if not torch.is_tensor(cond_tokens):
                    raise ValueError(f"full/full 缺少 cond_tokens @ {expected_obj_id}")
                tensors_to_check["cond_tokens"] = cond_tokens
            if self.cond_point_dim > 0:
                cond_points = sample.get("cond_point_colors")
                if not torch.is_tensor(cond_points) or tuple(cond_points.shape) != (
                    int(self.cond_point_dim),
                    point_count,
                ):
                    raise ValueError(
                        f"full/full cond_point_colors 必须为 [{self.cond_point_dim},N] @ {expected_obj_id}, 得到 {getattr(cond_points, 'shape', None)}"
                    )
                tensors_to_check["cond_point_colors"] = cond_points
            for tensor_name, tensor in tensors_to_check.items():
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(
                        f"full/full {tensor_name} 包含 NaN/Inf @ {expected_obj_id}"
                    )
            total_points += point_count
            min_points = (
                point_count if min_points is None else min(min_points, point_count)
            )
            max_points = max(max_points, point_count)
            completed = index + 1
            if (
                completed == 1
                or completed % interval == 0
                or completed == len(object_ids)
            ):
                print(
                    f"[full/full preflight] validated={completed}/{len(object_ids)} obj={expected_obj_id} points={point_count}",
                    flush=True,
                )
            del sample
        return {
            "objects": len(object_ids),
            "total_points": int(total_points),
            "min_points": int(min_points),
            "max_points": int(max_points),
        }

    def get_all_coords(self):
        sample_dirs = self.obj_dirs[: min(100, len(self.obj_dirs))]
        all_coords = []
        for d in sample_dirs:
            ply_dir = self._get_ply_dir(d)
            (c, _) = self._load_ply_data(
                os.path.join(ply_dir, "merged_model_render_color.ply"),
                force_subsample=True,
            )
            all_coords.append(c)
        return np.concatenate(all_coords, axis=1)

    def get_all_feats(self):
        sample_dirs = self.obj_dirs[: min(100, len(self.obj_dirs))]
        all_feats = []
        for d in sample_dirs:
            ply_dir = self._get_ply_dir(d)
            (_, f) = self._load_ply_data(
                os.path.join(ply_dir, "merged_model_render_color.ply"),
                force_subsample=True,
            )
            all_feats.append(f)
        return np.concatenate(all_feats, axis=1)
