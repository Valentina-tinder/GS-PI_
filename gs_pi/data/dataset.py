import os.path as osp
import abc
import time

import torch
import numpy as np
from plyfile import PlyData
import pytoml
import os
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from gs_pi.utils import logging


class Sample(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, root):
        return


class CostBalancedDistributedSampler(DistributedSampler):
    """Keep synchronized ranks on samples with comparable compute cost.

    The complete dataset is still shuffled and consumed exactly once per
    epoch when its length is divisible by ``num_replicas``.  Only the grouping
    changes: adjacent-cost samples form a rank-synchronous group, then group
    order and rank assignment are deterministically shuffled each epoch.
    """

    def __init__(self, dataset, sample_costs, **kwargs):
        super().__init__(dataset, **kwargs)
        self.sample_costs = tuple(int(cost) for cost in sample_costs)
        if len(self.sample_costs) != len(dataset):
            raise ValueError(
                "sample_costs length must match dataset length: "
                f"{len(self.sample_costs)} != {len(dataset)}"
            )
        if any(cost <= 0 for cost in self.sample_costs):
            raise ValueError("every distributed sample cost must be positive")

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                repeats = (padding_size + len(indices) - 1) // len(indices)
                indices += (indices * repeats)[:padding_size]
        else:
            indices = indices[: self.total_size]
        indices.sort(key=lambda index: (self.sample_costs[index], index))

        groups = [
            indices[start : start + self.num_replicas]
            for start in range(0, self.total_size, self.num_replicas)
        ]
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            group_order = torch.randperm(len(groups), generator=generator).tolist()
        else:
            group_order = list(range(len(groups)))

        rank_indices = []
        for group_index in group_order:
            group = groups[group_index]
            if self.shuffle:
                assignment = torch.randperm(
                    self.num_replicas,
                    generator=generator,
                ).tolist()
                group = [group[position] for position in assignment]
            rank_indices.append(group[self.rank])
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                "cost-balanced sampler produced an invalid rank length: "
                f"{len(rank_indices)} != {self.num_samples}"
            )
        return iter(rank_indices)


class Dataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    def __init__(self, root, verbose=True, **kwargs):
        self.verbose = verbose

        self.root = None
        self._init_root(root)

        if self.verbose:
            if self.root is not None:
                logging.info(f"Initializing dataset {self.name} from {self.root}")
            else:
                logging.info(f"Initializing dataset {self.name}")

        self.samples = []
        self._init_samples(**kwargs)

        if self.verbose:
            logging.info(f"\tNumber of samples: {len(self)}")
            logging.info(f"Finished initializing dataset {self.name}.")
            logging.info()

    @property
    def name(self):
        return type(self).__name__

    def _init_root(self, root):
        if isinstance(root, str):
            self.root = root
        elif isinstance(root, list):
            self.root = [path for path in root if osp.isdir(path)][0]

    @abc.abstractmethod
    def _init_samples(self):
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_dict = sample.load(root=self.root)
        return sample_dict

    def _get_paths(self):
        return get_paths()

    def _get_path(self, *keys):
        return get_path(*keys)

    def get_loader(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=None,
        indices=None,
        persistent_workers=None,
        prefetch_factor=None,
        distributed_sample_costs=None,
    ):
        import os

        dataset = (
            torch.utils.data.Subset(self, indices) if indices is not None else self
        )
        is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        if is_distributed:
            sampler = (
                CostBalancedDistributedSampler(
                    dataset,
                    distributed_sample_costs,
                    shuffle=shuffle,
                )
                if distributed_sample_costs is not None
                else DistributedSampler(dataset)
            )
            shuffle = False
        else:
            sampler = None

        # --- 新增：强制每个 worker 拥有独立随机状态 ---
        def _default_worker_init_fn(worker_id):
            # 获取当前进程的 seed (DDP 会自动处理每个 rank 不同)
            base_seed = torch.initial_seed() % 2**32
            import numpy as np
            import random

            np.random.seed(base_seed + worker_id)
            random.seed(base_seed + worker_id)

        actual_worker_init_fn = (
            worker_init_fn if worker_init_fn else _default_worker_init_fn
        )
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=actual_worker_init_fn,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = (
                True if persistent_workers is None else persistent_workers
            )
            loader_kwargs["prefetch_factor"] = (
                4 if prefetch_factor is None else prefetch_factor
            )
        dataloader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
        return dataloader


def get_paths():
    rmvd_paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
    home_paths_file = osp.join(osp.expanduser("~"), "rmvd_data_paths.toml")

    if osp.exists(rmvd_paths_file):
        paths_file = rmvd_paths_file
    elif osp.exists(home_paths_file):
        paths_file = home_paths_file
    else:
        raise FileNotFoundError(
            "No paths.toml file found. Please create a paths.toml file as specified in the "
            "rmvd/data/README.md file."
        )

    with open(paths_file, "r") as paths_file:
        return pytoml.load(paths_file)


def get_path(*keys):
    paths = get_paths()
    path = None

    for idx, key in enumerate(keys):
        if key in paths:
            if (
                key in paths
                and (isinstance(paths[key], str) or isinstance(paths[key], list))
                and idx == len(keys) - 1
            ):
                path = paths[key]
            else:
                paths = paths[key]

    return path


class PBRConditionalDataset(Dataset):
    def __init__(self, root_dir, num_views=6, num_points=512, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.num_views = num_views
        self.num_points = num_points

        # 1. 扫描所有 ID 文件夹 (例如 obj_001, obj_002...)
        self.obj_dirs = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        if len(self.obj_dirs) == 0:
            raise RuntimeError(f"在 {root_dir} 中未找到任何物体文件夹")

    def __len__(self):
        return len(self.obj_dirs)

    def _load_ply_data(self, ply_path):
        """读取 PLY 并提取 Coords(6) 和 Feats(5)"""
        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]

        # 提取坐标和法线 (Coords: XYZ + Normal)
        coords = np.stack([v["x"], v["y"], v["z"], v["nx"], v["ny"], v["nz"]], axis=0)

        # 提取材质 (Feats: Albedo RGB + Rough + Metal)
        # 注意：Albedo 在 PLY 里通常是 0-255 的 uchar，需要归一化
        albedo = np.stack([v["red"], v["green"], v["blue"]], axis=0) / 255.0
        feats = np.concatenate(
            [albedo, v["roughness"][None, :], v["metallic"][None, :]], axis=0
        )

        # 采样固定数量的点 (满足 NPCD 的 num_points 要求)
        # 如果点数不够，进行重复采样；如果太多，随机采样
        curr_p = coords.shape[1]
        indices = np.random.choice(
            curr_p, self.num_points, replace=(curr_p < self.num_points)
        )

        return coords[:, indices], feats[:, indices]

    def __getitem__(self, idx):
        # 定位当前物体的文件夹路径
        obj_id = self.obj_dirs[idx]
        curr_dir = os.path.join(self.root_dir, obj_id)

        # 1. 加载 3D 点云 (使用合并后的模型)
        ply_path = os.path.join(curr_dir, "merged_model_render_color.ply")
        coords, feats = self._load_ply_data(ply_path)

        # 2. 加载图像条件 (Input Images)
        images = []
        for v in range(1, self.num_views + 1):
            img_path = os.path.join(curr_dir, f"Render_{v}.png")
            img = Image.open(img_path).convert("RGB")
            img_tensor = (
                torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            )
            images.append(img_tensor)

        # 3. 加载相机参数 (从该文件夹下的 npy 文件加载)
        # 假设您在 makedataset.py 中把这些也存入了每个 ID 文件夹
        intrinsics = np.load(os.path.join(curr_dir, "intrinsics.npy"))
        poses = np.load(os.path.join(curr_dir, "poses.npy"))

        return {
            "coords": torch.from_numpy(coords).float(),
            "feats": torch.from_numpy(feats).float(),
            "input_images": torch.stack(images),  # [V, 3, H, W]
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "extrinsics": torch.from_numpy(poses).float(),
            "obj_idx": torch.tensor(idx, dtype=torch.long),
        }

    # 用于初始化 Normalization 层，可以取第一个样本做近似，或预计算平均值
    def get_all_coords(self):
        return self._load_ply_data(
            os.path.join(
                self.root_dir, self.obj_dirs[0], "merged_model_render_color.ply"
            )
        )[0]

    def get_all_feats(self):
        return self._load_ply_data(
            os.path.join(
                self.root_dir, self.obj_dirs[0], "merged_model_render_color.ply"
            )
        )[1]
