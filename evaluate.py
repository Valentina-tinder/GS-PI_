"""DDIM inference, aligned point metrics, and NPZ/PLY export."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch.utils.data import DataLoader

from gs_pi.runtime import (
    load_weights,
    make_dataset,
    make_model,
    model_batch,
    point_metrics,
    read_config,
    sample_features,
)
from gs_pi.utils import set_seed


def export_ply(path, coords, features):
    names = [
        "x",
        "y",
        "z",
        "albedo_red",
        "albedo_green",
        "albedo_blue",
        "roughness",
        "metallic",
    ]
    if features.shape[0] == 8:
        names += ["nx", "ny", "nz"]
    values = np.concatenate((coords, features), axis=0)
    vertices = np.empty(coords.shape[1], dtype=[(key, "f4") for key in names])
    for key, value in zip(names, values):
        vertices[key] = value
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--state-key",
        default="model_state_dict",
        help="Use ema_model_state_dict for public EMA weights",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Evaluate configured num_points instead of a full object",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    config = read_config(args.config)
    model = make_model(config).to(args.device).eval()
    load_weights(model, args.checkpoint, args.state_key)
    dataset = make_dataset(config, "test", full=not args.sampled)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        name = batch["name"][0]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid object name {name!r}")
        tensors = model_batch(batch, args.device)
        if model.denoiser.requires_full_pointcloud_inference and args.sampled:
            raise ValueError("This backbone requires full-object inference")
        seed = (
            args.seed
            + int.from_bytes(
                hashlib.blake2b(name.encode(), digest_size=8).digest(), "little"
            )
        ) % (2**63 - 1)
        pred = sample_features(model, tensors, args.steps, seed)[0].float().cpu()
        if not torch.isfinite(pred).all():
            raise FloatingPointError(f"Nonfinite prediction for {name}")
        pred[:5].clamp_(0, 1)
        coords, gt = batch["coords"][0], batch["feats"][0]
        mask = batch.get("normal_valid_mask")
        record = dict(
            object_id=name, **point_metrics(pred, gt, None if mask is None else mask[0])
        )
        records.append(record)
        np.savez_compressed(
            output / f"{name}.npz",
            coords=coords.numpy(),
            pred_feats=pred.numpy(),
            gt_feats=gt.numpy(),
        )
        export_ply(output / f"{name}.ply", coords.numpy(), pred.numpy())
        print(record, flush=True)
    keys = [
        k
        for k in records[0]
        if k not in ("object_id", "point_count", "normal_valid_count")
    ]
    summary = {
        k: float(np.mean([r[k] for r in records if r[k] is not None]))
        if any(r[k] is not None for r in records)
        else None
        for k in keys
    }
    (output / "metrics.json").write_text(
        json.dumps(
            dict(
                objects=records,
                mean_over_objects=summary,
                checkpoint=str(args.checkpoint),
                state_key=args.state_key,
                ddim_steps=args.steps,
                seed=args.seed,
                full_object=not args.sampled,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
