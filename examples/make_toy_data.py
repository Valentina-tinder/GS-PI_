"""Generate small synthetic objects to verify installation, not model quality."""

import argparse
from pathlib import Path
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/toy")
    args = parser.parse_args()
    root = Path(args.output)
    rng = np.random.default_rng(42)
    for split in ("train", "test"):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            coords = rng.uniform(-1, 1, (3, 24)).astype("float32")
            feats = rng.uniform(0, 1, (5, 24)).astype("float32")
            tokens = rng.normal(size=(4, 32)).astype("float32")
            colors = rng.uniform(-1, 1, (3, 24)).astype("float32")
            np.savez(
                folder / f"object_{index}.npz",
                coords=coords,
                feats=feats,
                cond_tokens=tokens,
                cond_point_colors=colors,
            )
    config = dict(
        train_dataset="NPZDataset",
        test_dataset="NPZDataset",
        train_dataset_params=dict(root_dir=str(root / "train"), num_points=16),
        test_dataset_params=dict(root_dir=str(root / "test"), num_points=16),
        model=dict(
            coords_dim=3,
            feats_dim=5,
            num_points=16,
            width=32,
            heads=4,
            layers=2,
            fix_coords=True,
            fix_coords_training_mode="identity",
            feature_schema="pbr_v1",
            condition_on_images=True,
            cond_point_dim=3,
            backbone_type="transformer",
        ),
        training=dict(
            batch_size=2,
            max_iterations=2,
            learning_rate=0.001,
            ema_decay=0.9,
            save_interval=1,
            log_interval=1,
        ),
    )
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(root / "config.yaml")


if __name__ == "__main__":
    main()
