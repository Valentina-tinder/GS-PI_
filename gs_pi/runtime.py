"""Portable data, checkpoint and sampling helpers for the public release."""

from pathlib import Path
import inspect

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from gs_pi.models.diffusion.diffusion_model import DiffusionModel


class NPZDataset(Dataset):
    """One object per NPZ; channel-first arrays in original point order."""

    def __init__(self, root_dir, num_points=512, full_pointcloud_training=False):
        self.paths = sorted(Path(root_dir).glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No .npz objects in {root_dir}")
        self.num_points = int(num_points)
        self.full_pointcloud_training = full_pointcloud_training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            sample = {k: torch.from_numpy(data[k].copy()) for k in data.files}
        coords, feats = sample["coords"], sample["feats"]
        if coords.ndim != 2 or coords.shape[0] != 3 or feats.ndim != 2:
            raise ValueError(f"{path}: expected coords [3,N], feats [F,N]")
        n = coords.shape[1]
        if n == 0 or feats.shape[1] != n or feats.shape[0] not in (5, 8):
            raise ValueError(f"{path}: invalid point count or feature channels")
        for key, value in sample.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{path}: nonfinite {key}")
        if not self.full_pointcloud_training:
            indices = (
                torch.randperm(n)[: self.num_points]
                if n >= self.num_points
                else torch.randint(n, (self.num_points,))
            )
            for key in (
                "coords",
                "feats",
                "cond_point_colors",
                "normal_valid_mask",
                "view_source",
            ):
                if key in sample:
                    sample[key] = sample[key][..., indices]
        sample["name"] = path.stem
        return sample

    def get_all_coords(self):
        return torch.cat([self[i]["coords"] for i in range(min(100, len(self)))], dim=1)

    def get_all_feats(self):
        return torch.cat([self[i]["feats"] for i in range(min(100, len(self)))], dim=1)


def read_config(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def make_dataset(config, split, *, full=False):
    params = dict(config[f"{split}_dataset_params"])
    if full:
        params["full_pointcloud_training"] = True
        params["dino_subsample_views"] = None
    kind = config.get(f"{split}_dataset", "PBRConditionalDataset")
    if kind == "NPZDataset":
        params.pop("dino_subsample_views", None)
        return NPZDataset(**params)
    if kind != "PBRConditionalDataset":
        raise ValueError(f"Unsupported dataset {kind!r}")
    from gs_pi.data import PBRConditionalDataset

    if params.get("light_probe_dir"):
        raise ValueError("Light conditioning is not part of this release")
    dataset = PBRConditionalDataset(**params)
    if len(dataset) == 0:
        raise ValueError("No valid PBR objects found; check paths and required files")
    if dataset.dino_features_dir:
        missing = [
            name
            for name in dataset.obj_dirs
            if not Path(dataset._get_dino_feature_path(name)).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing DINO cache for objects: {missing[:10]}")
    return dataset


def make_model(config):
    params = dict(config["model"])
    # These legacy NPCD wrapper fields do not affect the diffusion network.
    for key in (
        "n_obj",
        "use_view_dir",
        "pointnerf_only",
        "freeze_pointnerf",
        "is_pbr",
    ):
        params.pop(key, None)
    params.setdefault("use_flash_attn", False)
    model = DiffusionModel(**params)
    if not model.fix_coords:
        raise ValueError("This PBR release requires model.fix_coords=true")
    return model


def load_weights(model, path, state_key="model_state_dict"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload[state_key] if state_key in payload else payload
    if not isinstance(state, dict) or not all(
        torch.is_tensor(v) for v in state.values()
    ):
        raise ValueError(f"{path}: missing state key {state_key!r} or invalid weights")
    # Accept original NPCD / compiled / DDP checkpoints as well as raw diffusion.
    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("pointnerf."):
            continue
        key = (
            key.removeprefix("diffusion.")
            .removeprefix("module.")
            .removeprefix("_orig_mod.")
        )
        if key.startswith("image_encoder."):
            continue
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)
    return payload


def model_batch(batch, device):
    accepted = inspect.signature(DiffusionModel.forward).parameters
    return {
        k: v.to(device)
        for k, v in batch.items()
        if k in accepted and torch.is_tensor(v)
    }


@torch.no_grad()
def sample_features(model, batch, steps=50, seed=42):
    if not 1 <= steps <= model.diffusion_process.num_timesteps:
        raise ValueError(
            "DDIM steps must be between 1 and the diffusion timestep count"
        )
    if model.diffusion_process.num_timesteps % steps:
        raise ValueError(
            "DDIM steps must divide the diffusion timestep count (1000), e.g. 20, 50, 100"
        )
    coords = batch["coords"]
    generator = torch.Generator(device=coords.device).manual_seed(seed)
    noise = torch.randn(
        coords.shape[0],
        model.feats_dim,
        coords.shape[-1],
        device=coords.device,
        generator=generator,
    )
    _, normalized = model.diffusion_process.ddim_sample_loop(
        denoise_fn=model.denoiser,
        coords_start=model.coords_normalization(coords),
        feats_start=noise,
        num_steps=steps,
        fix_coords=True,
        eta=0.0,
        cond_tokens=batch.get("cond_tokens"),
        cond_point_colors=batch.get("cond_point_colors"),
        view_source=batch.get("view_source"),
    )
    return model.project_normal_features(
        model.feats_normalization.denormalize(normalized)
    )


def point_metrics(pred, gt, normal_valid_mask=None):
    result = {}
    for name, group in [
        ("albedo", slice(0, 3)),
        ("roughness", slice(3, 4)),
        ("metallic", slice(4, 5)),
    ]:
        error = pred[group].clamp(0, 1) - gt[group].clamp(0, 1)
        result[name + "_mae"] = error.abs().mean().item()
        result[name + "_mse"] = error.square().mean().item()
    if gt.shape[0] == 8:
        g, p = gt[5:8], pred[5:8]
        valid = torch.isfinite(g).all(0) & (g.norm(dim=0) > 1e-6)
        if normal_valid_mask is not None:
            valid &= normal_valid_mask.reshape(-1).bool()
        pred_valid = torch.isfinite(p).all(0) & (p.norm(dim=0) > 1e-6)
        angle = torch.rad2deg(
            torch.acos(torch.nn.functional.cosine_similarity(g, p, dim=0).clamp(-1, 1))
        )
        angle[~pred_valid] = 180
        result["normal_angular_deg"] = (
            angle[valid].mean().item() if valid.any() else None
        )
        result["normal_valid_count"] = valid.sum().item()
    result["point_count"] = gt.shape[-1]
    return result
