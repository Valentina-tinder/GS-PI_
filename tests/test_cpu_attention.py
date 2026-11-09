"""CPU execution must work even when CUDA-only xFormers is importable."""

from types import SimpleNamespace

import pytest
import torch

from gs_pi.models.diffusion.denoisers.transformer import NPCDTransformer
from gs_pi.models.diffusion.denoisers import transformer


@pytest.mark.parametrize("backbone", ["transformer", "two_stream"])
def test_cpu_forward_backward_with_xformers_installed(monkeypatch, backbone):
    def cuda_only(*args, **kwargs):
        raise AssertionError("CPU tensors must not be sent to xFormers")

    monkeypatch.setattr(
        transformer, "xops", SimpleNamespace(memory_efficient_attention=cuda_only)
    )
    torch.manual_seed(9)
    model = NPCDTransformer(
        coords_dim=3,
        feats_dim=8,
        width=16,
        layers=2,
        heads=4,
        backbone_type=backbone,
        use_flash_attn=True,
        z_dim=16,
        num_latents=4,
        num_blocks=2,
        num_compute_layers=1,
    )
    torch.nn.init.normal_(model.output_proj.weight, std=0.05)
    feats = torch.randn(2, 8, 11, requires_grad=True)
    condition = torch.randn(2, 5, 16, requires_grad=True)
    outputs = model(
        torch.randn(2, 3, 11), feats, torch.tensor([1, 7]), cond_tokens=condition
    )
    assert outputs[0].shape == (2, 3, 11)
    assert outputs[1].shape == feats.shape
    sum(output.square().mean() for output in outputs).backward()
    for tensor in (feats, condition):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
