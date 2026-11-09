import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gs_pi"
    / "models"
    / "diffusion"
    / "diffusion_processes"
    / "gaussian_diffusion.py"
)
SPEC = importlib.util.spec_from_file_location("normal_gaussian_diffusion", MODULE_PATH)
DIFFUSION_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DIFFUSION_MODULE)
GaussianDiffusion = DIFFUSION_MODULE.GaussianDiffusion


def _zero_denoiser(coords_t, feats_t, t, **kwargs):
    return torch.zeros_like(coords_t), torch.zeros_like(feats_t)


class _CaptureCoordsDenoiser:
    def __init__(self):
        self.coords_t = None

    def __call__(self, coords_t, feats_t, t, **kwargs):
        self.coords_t = coords_t.detach().clone()
        return torch.zeros_like(coords_t), torch.zeros_like(feats_t)


def test_legacy_five_channel_loss_is_numerically_unchanged():
    torch.manual_seed(4)
    process = GaussianDiffusion(feature_schema="pbr_v1", min_snr_gamma=None)
    coords = torch.randn(2, 3, 7)
    feats = torch.randn(2, 5, 7)
    coords_noise = torch.randn_like(coords)
    feats_noise = torch.randn_like(feats)
    loss, sub_losses, pointwise = process.p_losses(
        _zero_denoiser,
        coords,
        feats,
        torch.tensor([2, 8]),
        coords_noise=coords_noise,
        feats_noise=feats_noise,
    )
    expected = (coords_noise.square() / 2.0).mean() + (
        feats_noise.square() / 2.0
    ).mean()
    torch.testing.assert_close(loss, expected)
    assert set(sub_losses) == {"00_coords_loss", "01_feats_loss"}
    assert pointwise["pointwise_feats_loss"].shape == feats.shape


def test_normal_loss_masks_invalid_points_and_stays_finite():
    torch.manual_seed(9)
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        min_snr_gamma=5.0,
        normal_eps_weight=0.5,
        normal_cos_weight=0.2,
        normal_unit_weight=0.05,
        normal_aux_max_timestep=250,
    )
    coords = torch.randn(2, 3, 11)
    pbr = torch.rand(2, 5, 11)
    normal = F.normalize(torch.randn(2, 3, 11), dim=1)
    feats = torch.cat([pbr, normal], dim=1)
    mask = torch.ones(2, 1, 11, dtype=torch.bool)
    mask[:, :, 0] = False
    feats[:, 5:, 0] = 0.0
    loss, sub_losses, pointwise = process.p_losses(
        _zero_denoiser,
        coords,
        feats,
        torch.tensor([10, 500]),
        fix_coords=True,
        normal_valid_mask=mask,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in sub_losses.values())
    assert not pointwise["normal_valid_mask"][:, :, 0].any()
    assert {
        "01_pbr_eps_loss",
        "02_normal_eps_loss",
        "03_normal_cos_loss",
        "04_normal_unit_loss",
        "05_normal_angle_deg",
    }.issubset(sub_losses)


def test_partial_loss_override_preserves_model_timestep_gate():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_eps_weight=1.0,
        normal_aux_max_timestep=321,
    )
    process.configure_normal_loss(normal_eps_weight=0.25)
    assert process.normal_eps_weight == 0.25
    assert process.normal_aux_max_timestep == 321


def test_fix_coords_identity_matches_inference_coordinate_conditioning():
    coords = torch.tensor(
        [[[0.25, -0.5], [0.75, 0.5], [-0.25, 1.0]]], dtype=torch.float32
    )
    feats = torch.randn(1, 5, 2)
    t = torch.tensor([500])

    identity_denoiser = _CaptureCoordsDenoiser()
    identity_process = GaussianDiffusion(
        feature_schema="pbr_v1",
        fix_coords_training_mode="identity",
    )
    identity_process.p_losses(
        identity_denoiser,
        coords,
        feats,
        t,
        fix_coords=True,
    )
    torch.testing.assert_close(identity_denoiser.coords_t, coords)

    scaled_denoiser = _CaptureCoordsDenoiser()
    scaled_process = GaussianDiffusion(
        feature_schema="pbr_v1",
        fix_coords_training_mode="scaled",
    )
    scaled_process.p_losses(
        scaled_denoiser,
        coords,
        feats,
        t,
        fix_coords=True,
    )
    expected = scaled_process.q_sample(
        data_start=coords,
        t=t,
        noise=torch.zeros_like(coords),
    )
    torch.testing.assert_close(scaled_denoiser.coords_t, expected)
    assert not torch.allclose(scaled_denoiser.coords_t, coords)


def test_ddim_optional_x0_constraints_clip_and_project_normals():
    process = GaussianDiffusion(feature_schema="pbr_normal_world_v1")
    coords = torch.zeros(1, 3, 2)
    feats = torch.tensor([[[2.0, -2.0]] * 5 + [[-2.0, 0.5], [0.25, -2.0], [0.5, 0.25]]])

    _, unconstrained = process.ddim_sample(
        _zero_denoiser,
        coords,
        feats,
        t=500,
        t_prev=-1,
        fix_coords=True,
    )
    _, constrained = process.ddim_sample(
        _zero_denoiser,
        coords,
        feats,
        t=500,
        t_prev=-1,
        fix_coords=True,
        feats_clip_range=(-1.0, 1.0),
        project_normal_x0=True,
    )
    assert unconstrained.abs().max() > 1.0
    assert constrained.abs().max() <= 1.0
    normal_lengths = torch.linalg.vector_norm(constrained[:, 5:8, :], dim=1)
    torch.testing.assert_close(normal_lengths, torch.ones_like(normal_lengths))


def test_normal_tail_loss_targets_worst_fraction():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_cos_tail_fraction=0.5,
    )
    angular_error = torch.tensor([[0.1, 0.2, 0.9, 0.8]])
    mask = torch.ones_like(angular_error, dtype=torch.bool)
    loss = process._normal_tail_loss(angular_error, mask, aux_weights=None)
    torch.testing.assert_close(loss, torch.tensor(0.85))


def test_normal_smoothness_uses_gt_edge_gate_and_is_finite():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_smooth_knn=1,
        normal_smooth_gt_angle_deg=10.0,
    )
    coords = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    gt = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    pred_good = gt.clone()
    pred_noisy = gt.clone()
    pred_noisy[:, :, 1] = torch.tensor([[-1.0, 0.0, 0.0]])
    valid = torch.ones(1, 1, 4, dtype=torch.bool)
    aux = valid[:, 0]
    good_loss = process._normal_smoothness_loss(
        coords, pred_good, gt, valid, aux, aux_weights=None
    )
    noisy_loss = process._normal_smoothness_loss(
        coords, pred_noisy, gt, valid, aux, aux_weights=None
    )
    torch.testing.assert_close(good_loss, torch.tensor(0.0))
    assert torch.isfinite(noisy_loss)
    assert noisy_loss > good_loss


def test_normal_aux_weighting_separates_x0_from_epsilon_min_snr():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_aux_weighting="uniform",
    )
    t = torch.tensor([10, 400])
    legacy_weights = torch.tensor([[[0.01]], [[1.0]]])
    assert process._normal_aux_weights(t, (2, 3, 4), legacy_weights) is None

    process.configure_normal_loss(
        normal_aux_weighting="truncated_snr",
        normal_aux_snr_gamma=5.0,
    )
    weights = process._normal_aux_weights(t, (2, 3, 4), legacy_weights)
    assert weights.shape == (2, 1)
    assert torch.all((weights >= 0.0) & (weights <= 1.0))
    assert weights[0, 0] > weights[1, 0]


def test_robust_x0_weighting_can_stay_uniform_with_legacy_cosine_weights():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_aux_weighting="min_snr",
        normal_robust_aux_weighting="uniform",
    )
    t = torch.tensor([10, 400])
    legacy_weights = torch.tensor([[[0.01]], [[1.0]]])
    aux_weights = process._normal_aux_weights(t, (2, 3, 4), legacy_weights)
    robust_weights = process._normal_robust_aux_weights(
        t, (2, 3, 4), legacy_weights, aux_weights
    )
    torch.testing.assert_close(aux_weights, legacy_weights[:, 0, :])
    assert robust_weights is None


def test_normal_relation_loss_preserves_gt_curvature():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_relation_knn=1,
        normal_relation_gt_angle_deg=45.0,
        normal_relation_beta=0.05,
    )
    coords = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    angles = torch.deg2rad(torch.tensor([0.0, 10.0, 20.0]))
    gt = torch.stack(
        [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=0
    ).unsqueeze(0)
    pred_flat = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    aux = valid[:, 0]
    matched_loss = process._normal_relation_loss(
        coords, gt, gt, valid, aux, aux_weights=None
    )
    flat_loss = process._normal_relation_loss(
        coords, pred_flat, gt, valid, aux, aux_weights=None
    )
    torch.testing.assert_close(matched_loss, torch.tensor(0.0))
    assert torch.isfinite(flat_loss)
    assert flat_loss > matched_loss


def test_normal_x0_huber_has_direct_gradient_at_antipodal_prediction():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_x0_huber_beta=0.1,
    )
    gt = torch.tensor([[[1.0], [0.0], [0.0]]])
    pred = (-gt).clone().requires_grad_(True)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    loss = process._normal_x0_huber_loss(
        pred,
        gt,
        mask[:, 0],
        aux_weights=None,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_normal_residual_relation_penalizes_an_isolated_flip():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_relation_knn=1,
        normal_relation_gt_angle_deg=10.0,
        normal_relation_beta=0.05,
    )
    coords = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    gt = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    pred_good = gt.clone()
    pred_flip = gt.clone()
    pred_flip[:, 0, 1] = -1.0
    pred_flip.requires_grad_(True)
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    aux = valid[:, 0]
    good_loss = process._normal_residual_relation_loss(
        coords, gt, gt, gt, valid, aux, aux_weights=None
    )
    flip_loss = process._normal_residual_relation_loss(
        coords, pred_flip, gt, gt, valid, aux, aux_weights=None
    )
    flip_loss.backward()
    torch.testing.assert_close(good_loss, torch.tensor(0.0))
    assert flip_loss > good_loss
    assert pred_flip.grad is not None
    assert pred_flip.grad.abs().sum() > 0


def test_normal_residual_relation_uses_explicit_edges_and_valid_mask():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_relation_knn=1,
        normal_relation_gt_angle_deg=10.0,
        normal_relation_beta=0.05,
    )
    coords = torch.tensor(
        [[[0.0, 0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    gt = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    pred = gt.clone()
    pred[:, 0, 3] = -1.0
    pred.requires_grad_(True)
    valid = torch.ones(1, 1, 4, dtype=torch.bool)
    aux = valid[:, 0]
    # Only this supplied 0 -> 3 edge is supervised.  Dataset-side sampling can
    # therefore select a full-cloud same-view neighbour independent of the
    # in-batch coordinate KNN.
    edge_index = torch.tensor([[[0], [3]]], dtype=torch.long)
    enabled_loss = process._normal_residual_relation_loss(
        coords,
        pred,
        gt,
        gt,
        valid,
        aux,
        # Timestep-derived auxiliary weights are [B,1] and must broadcast over
        # arbitrary explicit edges rather than being gathered as point weights.
        aux_weights=torch.tensor([[0.5]]),
        normal_relation_edge_index=edge_index,
        normal_relation_edge_valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    masked_loss = process._normal_residual_relation_loss(
        coords,
        pred,
        gt,
        gt,
        valid,
        aux,
        aux_weights=torch.tensor([[0.5]]),
        normal_relation_edge_index=edge_index,
        normal_relation_edge_valid_mask=torch.zeros(1, 1, dtype=torch.bool),
    )
    enabled_loss.backward()

    assert enabled_loss > 0
    torch.testing.assert_close(masked_loss, torch.tensor(0.0))
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_integrated_normal_loss_reports_tail_and_smooth_terms():
    process = GaussianDiffusion(
        feature_schema="pbr_normal_world_v1",
        normal_cos_tail_weight=0.2,
        normal_smooth_weight=0.05,
        normal_x0_huber_weight=0.05,
        normal_residual_relation_weight=0.05,
        normal_aux_max_timestep=400,
    )
    coords = torch.randn(2, 3, 8)
    normal = F.normalize(torch.randn(2, 3, 8), dim=1)
    feats = torch.cat([torch.rand(2, 5, 8), normal], dim=1)
    explicit_edges = torch.tensor(
        [
            [[0, 2], [1, 3]],
            [[4, 6], [5, 7]],
        ],
        dtype=torch.long,
    )
    loss, sub_losses, _ = process.p_losses(
        _zero_denoiser,
        coords,
        feats,
        torch.tensor([10, 20]),
        fix_coords=True,
        normal_relation_edge_index=explicit_edges,
        normal_relation_edge_valid_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert "06_normal_cos_tail_loss" in sub_losses
    assert "07_normal_smooth_loss" in sub_losses
    assert "08_normal_relation_loss" in sub_losses
    assert "09_normal_x0_huber_loss" in sub_losses
    assert "10_normal_residual_relation_loss" in sub_losses
