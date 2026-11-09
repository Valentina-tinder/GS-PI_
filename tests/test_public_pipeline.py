import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from gs_pi.runtime import make_model, point_metrics, sample_features


@pytest.mark.parametrize(
    "backbone",
    [
        "transformer",
        "two_stream",
    ],
)
def test_training_and_ddim(backbone):
    model = make_model(
        {
            "model": dict(
                coords_dim=3,
                feats_dim=5,
                num_points=12,
                width=32,
                heads=4,
                layers=2,
                fix_coords=True,
                cond_point_dim=3,
                backbone_type=backbone,
                z_dim=32,
                num_latents=4,
                num_blocks=2,
                num_compute_layers=1,
            )
        }
    )
    batch = dict(
        coords=torch.randn(2, 3, 12),
        feats=torch.rand(2, 5, 12),
        cond_tokens=torch.randn(2, 4, 32),
        cond_point_colors=torch.randn(2, 3, 12),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.denoiser.output_proj.weight.detach().clone()
    loss, _, _ = model(**batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.denoiser.output_proj.weight)
    model.eval()
    pred = sample_features(model, batch, steps=2)
    assert pred.shape == batch["feats"].shape and torch.isfinite(pred).all()


def test_cli_train_resume_evaluate(tmp_path):
    root = Path(__file__).resolve().parents[1]

    def run(*args):
        subprocess.run(
            [sys.executable, *map(str, args)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    data, output, evaluation = tmp_path / "data", tmp_path / "run", tmp_path / "eval"
    run("examples/make_toy_data.py", "--output", data)
    config = data / "config.yaml"
    run("train.py", "--config", config, "--output", output, "--device", "cpu")
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
    assert checkpoint["iteration"] == 2
    cfg = yaml.safe_load(config.read_text())
    cfg["training"]["max_iterations"] = 3
    config.write_text(yaml.safe_dump(cfg))
    run(
        "train.py",
        "--config",
        config,
        "--output",
        output,
        "--device",
        "cpu",
        "--resume",
        output / "checkpoint.pt",
    )
    resumed = torch.load(output / "checkpoint.pt", weights_only=False)
    assert resumed["iteration"] == 3
    assert not torch.equal(
        checkpoint["model_state_dict"]["denoiser.output_proj.weight"],
        resumed["model_state_dict"]["denoiser.output_proj.weight"],
    )
    run(
        "evaluate.py",
        "--config",
        config,
        "--checkpoint",
        output / "checkpoint.pt",
        "--state-key",
        "ema_model_state_dict",
        "--output",
        evaluation,
        "--device",
        "cpu",
        "--steps",
        "2",
    )
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert len(metrics["objects"]) == 2
    assert all(r["point_count"] == 24 for r in metrics["objects"])
    assert len(list(evaluation.glob("*.ply"))) == 2


def test_invalid_normal_penalty():
    gt = torch.zeros(8, 2)
    gt[7] = 1
    pred = gt.clone()
    pred[5:, 0] = 0
    assert point_metrics(pred, gt)["normal_angular_deg"] == 90


def test_original_pbr_dataset_reads_export(tmp_path):
    from evaluate import export_ply
    from gs_pi.runtime import make_dataset

    folder = tmp_path / "object_0" / "extract_output"
    folder.mkdir(parents=True)
    coords = torch.randn(3, 24).numpy()
    feats = torch.rand(5, 24).numpy()
    export_ply(folder / "merged_model_render_color.ply", coords, feats)
    config = dict(
        test_dataset="PBRConditionalDataset",
        test_dataset_params=dict(
            root_dir=str(tmp_path),
            num_points=16,
            num_views=1,
            cond_point_dim=0,
            require_metadata=False,
            require_all_render_views=False,
            prefer_extract_output=True,
        ),
    )
    dataset = make_dataset(config, "test", full=True)
    item = dataset[0]
    torch.testing.assert_close(item["coords"], torch.from_numpy(coords))
    torch.testing.assert_close(item["feats"], torch.from_numpy(feats))
