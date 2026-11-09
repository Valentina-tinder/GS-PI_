import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gs_pi.utils import normal_kl, discretized_gaussian_log_likelihood, mean_flat
from gs_pi.pbr_schema import get_feature_schema


def get_beta_schedule(
    schedule_type, *, num_diffusion_steps, beta_start=None, beta_end=None
):
    if schedule_type == "linear":
        beta_start = (
            1000 / num_diffusion_steps * 0.0001 if beta_start is None else beta_start
        )
        beta_end = 1000 / num_diffusion_steps * 0.02 if beta_end is None else beta_end
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps)

    elif schedule_type == "cosine":
        # Nichol & Dhariwal 2021, Improved DDPM. 对非像素信号（材质/几何）相较 linear 收敛更快。
        s = 0.008
        steps = (
            np.arange(num_diffusion_steps + 1, dtype=np.float64) / num_diffusion_steps
        )
        alphas_cumprod = np.cos((steps + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 1e-8, 0.999)

    else:
        raise NotImplementedError(schedule_type)

    return betas


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        beta_schedule="cosine",
        num_diffusion_steps=1000,
        min_snr_gamma=5.0,
        feature_schema="pbr_v1",
        pbr_eps_weight=1.0,
        normal_eps_weight=1.0,
        normal_cos_weight=0.0,
        normal_cos_tail_weight=0.0,
        normal_cos_tail_fraction=0.1,
        normal_smooth_weight=0.0,
        normal_smooth_knn=8,
        normal_smooth_gt_angle_deg=15.0,
        normal_relation_weight=0.0,
        normal_relation_knn=8,
        normal_relation_gt_angle_deg=30.0,
        normal_relation_beta=0.05,
        normal_x0_huber_weight=0.0,
        normal_x0_huber_beta=0.1,
        normal_residual_relation_weight=0.0,
        normal_robust_aux_weighting="uniform",
        normal_unit_weight=0.0,
        normal_aux_max_timestep=None,
        normal_aux_weighting="min_snr",
        normal_aux_snr_gamma=5.0,
        fix_coords_training_mode="scaled",
    ):
        super().__init__()

        self.beta_schedule = str(beta_schedule)
        self.min_snr_gamma = None if min_snr_gamma is None else float(min_snr_gamma)
        self.fix_coords_training_mode = str(fix_coords_training_mode).strip().lower()
        if self.fix_coords_training_mode not in {"scaled", "identity"}:
            raise ValueError(
                "fix_coords_training_mode 必须是 'scaled' 或 'identity'，"
                f"当前为 {fix_coords_training_mode!r}"
            )
        self.feature_schema = get_feature_schema(feature_schema)
        try:
            normal_group = self.feature_schema.group("normal")
        except KeyError:
            self.normal_slice = None
            self.pbr_stop = self.feature_schema.feats_dim
        else:
            self.normal_slice = normal_group.channel_slice
            self.pbr_stop = normal_group.start
        self.normal_x0_clamp = 4.0
        self.configure_normal_loss(
            pbr_eps_weight=pbr_eps_weight,
            normal_eps_weight=normal_eps_weight,
            normal_cos_weight=normal_cos_weight,
            normal_cos_tail_weight=normal_cos_tail_weight,
            normal_cos_tail_fraction=normal_cos_tail_fraction,
            normal_smooth_weight=normal_smooth_weight,
            normal_smooth_knn=normal_smooth_knn,
            normal_smooth_gt_angle_deg=normal_smooth_gt_angle_deg,
            normal_relation_weight=normal_relation_weight,
            normal_relation_knn=normal_relation_knn,
            normal_relation_gt_angle_deg=normal_relation_gt_angle_deg,
            normal_relation_beta=normal_relation_beta,
            normal_x0_huber_weight=normal_x0_huber_weight,
            normal_x0_huber_beta=normal_x0_huber_beta,
            normal_residual_relation_weight=normal_residual_relation_weight,
            normal_robust_aux_weighting=normal_robust_aux_weighting,
            normal_unit_weight=normal_unit_weight,
            normal_aux_max_timestep=normal_aux_max_timestep,
            normal_aux_weighting=normal_aux_weighting,
            normal_aux_snr_gamma=normal_aux_snr_gamma,
        )

        betas = get_beta_schedule(
            self.beta_schedule, num_diffusion_steps=num_diffusion_steps
        )
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(
            np.float64
        )  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(
            np.append(1.0, alphas_cumprod[:-1])
        ).float()
        betas_t = torch.from_numpy(betas).float()
        alphas_t = torch.from_numpy(alphas).float()

        # nth element of the list is beta_(n+1), e.g. element at index 0 is beta_1 which is used in the distribution q(x_1 | x_0) (equation 2 in the DDPM paper)
        # Register diffusion schedules as non-persistent buffers so model.to(device)
        # moves them once without changing checkpoint compatibility.
        self.register_buffer("betas", betas_t, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev.float(), persistent=False
        )

        self.register_buffer(
            "sqrt_one_minus_betas", torch.sqrt(1.0 - betas_t).float(), persistent=False
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float(), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).float(),
            persistent=False,
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            torch.log(1.0 - alphas_cumprod).float(),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod).float(),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1).float(),
            persistent=False,
        )

        # SNR(t) = alpha_bar / (1 - alpha_bar)，用于 Min-SNR-γ loss 加权
        self.register_buffer(
            "snr", (alphas_cumprod / (1.0 - alphas_cumprod)).float(), persistent=False
        )

        posterior_variance = (
            betas_t * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance", posterior_variance.float(), persistent=False
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(
                torch.cat([posterior_variance[1:2], posterior_variance[1:]])
            ).float(),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (
                betas_t * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            ).float(),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (
                (1.0 - alphas_cumprod_prev)
                * torch.sqrt(alphas_t)
                / (1.0 - alphas_cumprod)
            ).float(),
            persistent=False,
        )

    def configure_normal_loss(
        self,
        *,
        pbr_eps_weight=None,
        normal_eps_weight=None,
        normal_cos_weight=None,
        normal_cos_tail_weight=None,
        normal_cos_tail_fraction=None,
        normal_smooth_weight=None,
        normal_smooth_knn=None,
        normal_smooth_gt_angle_deg=None,
        normal_relation_weight=None,
        normal_relation_knn=None,
        normal_relation_gt_angle_deg=None,
        normal_relation_beta=None,
        normal_x0_huber_weight=None,
        normal_x0_huber_beta=None,
        normal_residual_relation_weight=None,
        normal_robust_aux_weighting=None,
        normal_unit_weight=None,
        normal_aux_max_timestep=None,
        normal_aux_weighting=None,
        normal_aux_snr_gamma=None,
    ):
        """Update normal-loss weights, including from ``diffusion_training`` config."""

        values = {
            "pbr_eps_weight": pbr_eps_weight,
            "normal_eps_weight": normal_eps_weight,
            "normal_cos_weight": normal_cos_weight,
            "normal_cos_tail_weight": normal_cos_tail_weight,
            "normal_smooth_weight": normal_smooth_weight,
            "normal_relation_weight": normal_relation_weight,
            "normal_x0_huber_weight": normal_x0_huber_weight,
            "normal_residual_relation_weight": normal_residual_relation_weight,
            "normal_unit_weight": normal_unit_weight,
        }
        for name, value in values.items():
            if value is None and hasattr(self, name):
                continue
            value = 0.0 if value is None else float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数，当前为 {value!r}")
            setattr(self, name, value)

        bounded_values = {
            "normal_cos_tail_fraction": (normal_cos_tail_fraction, 0.0, 1.0),
            "normal_smooth_gt_angle_deg": (normal_smooth_gt_angle_deg, 0.0, 180.0),
            "normal_relation_gt_angle_deg": (normal_relation_gt_angle_deg, 0.0, 180.0),
        }
        for name, (value, lower, upper) in bounded_values.items():
            if value is None and hasattr(self, name):
                continue
            if value is None:
                defaults = {
                    "normal_cos_tail_fraction": 0.1,
                    "normal_smooth_gt_angle_deg": 15.0,
                    "normal_relation_gt_angle_deg": 30.0,
                }
                value = defaults[name]
            value = float(value)
            if not np.isfinite(value) or not (lower < value <= upper):
                raise ValueError(f"{name} 必须在 ({lower},{upper}]，当前为 {value!r}")
            setattr(self, name, value)

        if normal_smooth_knn is None and hasattr(self, "normal_smooth_knn"):
            pass
        else:
            value = 8 if normal_smooth_knn is None else int(normal_smooth_knn)
            if value < 1:
                raise ValueError(f"normal_smooth_knn 必须 >= 1，当前为 {value}")
            self.normal_smooth_knn = value

        if normal_relation_knn is None and hasattr(self, "normal_relation_knn"):
            pass
        else:
            value = 8 if normal_relation_knn is None else int(normal_relation_knn)
            if value < 1:
                raise ValueError(f"normal_relation_knn 必须 >= 1，当前为 {value}")
            self.normal_relation_knn = value

        if normal_relation_beta is None and hasattr(self, "normal_relation_beta"):
            pass
        else:
            value = (
                0.05 if normal_relation_beta is None else float(normal_relation_beta)
            )
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"normal_relation_beta 必须是正有限数，当前为 {value!r}"
                )
            self.normal_relation_beta = value

        if normal_x0_huber_beta is None and hasattr(self, "normal_x0_huber_beta"):
            pass
        else:
            value = 0.1 if normal_x0_huber_beta is None else float(normal_x0_huber_beta)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"normal_x0_huber_beta 必须是正有限数，当前为 {value!r}"
                )
            self.normal_x0_huber_beta = value

        if normal_aux_weighting is None and hasattr(self, "normal_aux_weighting"):
            pass
        else:
            value = (
                "min_snr" if normal_aux_weighting is None else str(normal_aux_weighting)
            )
            choices = {"min_snr", "uniform", "truncated_snr"}
            if value not in choices:
                raise ValueError(
                    f"normal_aux_weighting 必须是 {sorted(choices)} 之一，当前为 {value!r}"
                )
            self.normal_aux_weighting = value

        if normal_robust_aux_weighting is None and hasattr(
            self, "normal_robust_aux_weighting"
        ):
            pass
        else:
            value = (
                "uniform"
                if normal_robust_aux_weighting is None
                else str(normal_robust_aux_weighting)
            )
            choices = {"inherit", "uniform", "truncated_snr"}
            if value not in choices:
                raise ValueError(
                    "normal_robust_aux_weighting 必须是 "
                    f"{sorted(choices)} 之一，当前为 {value!r}"
                )
            self.normal_robust_aux_weighting = value

        if normal_aux_snr_gamma is None and hasattr(self, "normal_aux_snr_gamma"):
            pass
        else:
            value = 5.0 if normal_aux_snr_gamma is None else float(normal_aux_snr_gamma)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"normal_aux_snr_gamma 必须是正有限数，当前为 {value!r}"
                )
            self.normal_aux_snr_gamma = value

        if normal_aux_max_timestep is None and hasattr(self, "normal_aux_max_timestep"):
            # A partial training-config override should not erase a value that
            # was already supplied in the model config.
            pass
        elif normal_aux_max_timestep is None:
            self.normal_aux_max_timestep = None
        else:
            value = int(normal_aux_max_timestep)
            if value < 0:
                raise ValueError(
                    f"normal_aux_max_timestep 必须 >= 0 或为 None，当前为 {value}"
                )
            self.normal_aux_max_timestep = value

    @staticmethod
    def _masked_mean(values, mask, weights=None):
        mask = mask.to(dtype=values.dtype)
        weighted = values * mask
        if weights is not None:
            weighted = weighted * weights.to(dtype=values.dtype)
        denominator = mask.sum().clamp_min(1.0)
        return weighted.sum() / denominator

    @staticmethod
    def _prepare_normal_valid_mask(normal_valid_mask, normal_gt):
        batch_size, _, num_points = normal_gt.shape
        if normal_valid_mask is None:
            mask = torch.ones(
                (batch_size, 1, num_points),
                dtype=torch.bool,
                device=normal_gt.device,
            )
        else:
            mask = normal_valid_mask.to(device=normal_gt.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask[:, None, :]
            if (
                mask.ndim != 3
                or mask.shape[0] != batch_size
                or mask.shape[-1] != num_points
            ):
                raise ValueError(
                    "normal_valid_mask 应为 [B,N] 或 [B,1,N]，"
                    f"当前 shape={tuple(mask.shape)}"
                )
            if mask.shape[1] not in (1, normal_gt.shape[1]):
                raise ValueError(
                    "normal_valid_mask 的通道数必须是 1 或 3，"
                    f"当前 shape={tuple(mask.shape)}"
                )
            if mask.shape[1] != 1:
                mask = mask.all(dim=1, keepdim=True)

        gt_length = torch.linalg.vector_norm(normal_gt, dim=1, keepdim=True)
        gt_is_valid = torch.isfinite(normal_gt).all(dim=1, keepdim=True)
        gt_is_valid = gt_is_valid & (gt_length > 1e-6)
        return mask & gt_is_valid

    def _normal_tail_loss(self, angular_error, aux_mask, aux_weights):
        """Mean angular error over the worst configurable fraction per object."""

        num_points = angular_error.shape[-1]
        hard_count = max(1, int(math.ceil(num_points * self.normal_cos_tail_fraction)))
        candidates = angular_error.masked_fill(~aux_mask, float("-inf"))
        hard_values = torch.topk(candidates, k=hard_count, dim=-1).values
        hard_valid = torch.isfinite(hard_values)
        hard_values = torch.where(
            hard_valid, hard_values, torch.zeros_like(hard_values)
        )
        weights = None if aux_weights is None else aux_weights.expand(-1, hard_count)
        return self._masked_mean(hard_values, hard_valid, weights=weights)

    def _normal_aux_weights(self, t, x_shape, min_snr_w):
        """Return weights for x0-space normal losses, separate from epsilon MSE.

        ``min_snr`` preserves the historical behavior. ``uniform`` avoids
        reusing epsilon weights for a geometrically different x0 objective.
        ``truncated_snr`` emphasizes reliable low-noise x0 predictions while
        smoothly suppressing high-noise timesteps.
        """

        if self.normal_aux_weighting == "min_snr":
            return None if min_snr_w is None else min_snr_w[:, 0, :]
        if self.normal_aux_weighting == "uniform":
            return None
        snr_t = self._extract(self.snr, t, x_shape)
        return torch.clamp(snr_t / self.normal_aux_snr_gamma, max=1.0)[:, 0, :]

    def _normal_robust_aux_weights(self, t, x_shape, min_snr_w, aux_weights):
        """Weight robust x0 losses independently from legacy cosine/unit terms."""

        if self.normal_robust_aux_weighting == "inherit":
            return aux_weights
        if self.normal_robust_aux_weighting == "uniform":
            return None
        snr_t = self._extract(self.snr, t, x_shape)
        return torch.clamp(snr_t / self.normal_aux_snr_gamma, max=1.0)[:, 0, :]

    def _normal_smoothness_loss(
        self,
        coords_start,
        normal_pred_unit,
        normal_gt_unit,
        valid_mask,
        aux_mask,
        aux_weights,
    ):
        """Encourage nearby points to agree only where GT normals are locally smooth."""

        num_points = coords_start.shape[-1]
        if num_points < 2:
            return normal_pred_unit.sum() * 0.0
        neighbor_count = min(self.normal_smooth_knn, num_points - 1)
        coords_points = coords_start.transpose(1, 2).float()
        with torch.no_grad():
            distances = torch.cdist(coords_points, coords_points)
            diagonal = torch.arange(num_points, device=coords_start.device)
            distances[:, diagonal, diagonal] = float("inf")
            neighbor_indices = torch.topk(
                distances,
                k=neighbor_count,
                dim=-1,
                largest=False,
            ).indices
            batch_indices = torch.arange(
                coords_start.shape[0], device=coords_start.device
            )[:, None, None]
            gt_points = normal_gt_unit.transpose(1, 2)
            gt_neighbors = gt_points[batch_indices, neighbor_indices]
            gt_centers = gt_points[:, :, None, :]
            gt_neighbor_dot = (gt_centers * gt_neighbors).sum(dim=-1).clamp(-1.0, 1.0)
            neighbor_valid = valid_mask[:, 0, :][batch_indices, neighbor_indices]
            gt_cos_threshold = math.cos(math.radians(self.normal_smooth_gt_angle_deg))
            pair_mask = (
                aux_mask[:, :, None]
                & neighbor_valid
                & (gt_neighbor_dot >= gt_cos_threshold)
            )

        pred_points = normal_pred_unit.transpose(1, 2)
        pred_neighbors = pred_points[batch_indices, neighbor_indices]
        pred_centers = pred_points[:, :, None, :]
        pred_neighbor_dot = (pred_centers * pred_neighbors).sum(dim=-1).clamp(-1.0, 1.0)
        pair_weights = None if aux_weights is None else aux_weights[:, :, None]
        return self._masked_mean(
            1.0 - pred_neighbor_dot,
            pair_mask,
            weights=pair_weights,
        )

    def _normal_relation_loss(
        self,
        coords_start,
        normal_pred_unit,
        normal_gt_unit,
        valid_mask,
        aux_mask,
        aux_weights,
    ):
        """Match local predicted angular variation to GT without flattening curvature."""

        num_points = coords_start.shape[-1]
        if num_points < 2:
            return normal_pred_unit.sum() * 0.0
        neighbor_count = min(self.normal_relation_knn, num_points - 1)
        coords_points = coords_start.transpose(1, 2).float()
        with torch.no_grad():
            distances = torch.cdist(coords_points, coords_points)
            diagonal = torch.arange(num_points, device=coords_start.device)
            distances[:, diagonal, diagonal] = float("inf")
            neighbor_indices = torch.topk(
                distances,
                k=neighbor_count,
                dim=-1,
                largest=False,
            ).indices
            batch_indices = torch.arange(
                coords_start.shape[0], device=coords_start.device
            )[:, None, None]
            gt_points = normal_gt_unit.transpose(1, 2)
            gt_neighbors = gt_points[batch_indices, neighbor_indices]
            gt_centers = gt_points[:, :, None, :]
            gt_neighbor_dot = (gt_centers * gt_neighbors).sum(dim=-1).clamp(-1.0, 1.0)
            neighbor_valid = valid_mask[:, 0, :][batch_indices, neighbor_indices]
            gt_cos_threshold = math.cos(math.radians(self.normal_relation_gt_angle_deg))
            pair_mask = (
                aux_mask[:, :, None]
                & neighbor_valid
                & (gt_neighbor_dot >= gt_cos_threshold)
            )

        pred_points = normal_pred_unit.transpose(1, 2)
        pred_neighbors = pred_points[batch_indices, neighbor_indices]
        pred_centers = pred_points[:, :, None, :]
        pred_neighbor_dot = (pred_centers * pred_neighbors).sum(dim=-1).clamp(-1.0, 1.0)
        relation_error = F.smooth_l1_loss(
            1.0 - pred_neighbor_dot,
            1.0 - gt_neighbor_dot,
            reduction="none",
            beta=self.normal_relation_beta,
        )
        pair_weights = None if aux_weights is None else aux_weights[:, :, None]
        return self._masked_mean(relation_error, pair_mask, weights=pair_weights)

    def _normal_x0_huber_loss(
        self,
        normal_pred_raw,
        normal_gt,
        aux_mask,
        aux_weights,
    ):
        """Robust raw-vector x0 loss with a direct gradient at antipodal normals."""

        component_error = F.smooth_l1_loss(
            normal_pred_raw,
            normal_gt,
            reduction="none",
            beta=self.normal_x0_huber_beta,
        )
        component_mask = aux_mask[:, None, :].expand_as(component_error)
        component_weights = None if aux_weights is None else aux_weights[:, None, :]
        return self._masked_mean(
            component_error,
            component_mask,
            weights=component_weights,
        )

    def _normal_residual_relation_loss(
        self,
        coords_start,
        normal_pred_raw,
        normal_gt,
        normal_gt_unit,
        valid_mask,
        aux_mask,
        aux_weights,
        normal_relation_edge_index=None,
        normal_relation_edge_valid_mask=None,
    ):
        """Suppress isolated x0 errors while preserving the GT normal field."""

        num_points = coords_start.shape[-1]
        if num_points < 2:
            return normal_pred_raw.sum() * 0.0
        batch_size = coords_start.shape[0]
        gt_points = normal_gt_unit.transpose(1, 2)
        residual_points = (normal_pred_raw - normal_gt).transpose(1, 2)
        gt_cos_threshold = math.cos(math.radians(self.normal_relation_gt_angle_deg))

        if normal_relation_edge_index is None:
            # Historical path: infer KNN only among the points already sampled
            # into this batch.  Keeping this branch intact preserves old configs
            # and checkpoints exactly.
            neighbor_count = min(self.normal_relation_knn, num_points - 1)
            coords_points = coords_start.transpose(1, 2).float()
            with torch.no_grad():
                distances = torch.cdist(coords_points, coords_points)
                diagonal = torch.arange(num_points, device=coords_start.device)
                distances[:, diagonal, diagonal] = float("inf")
                neighbor_indices = torch.topk(
                    distances,
                    k=neighbor_count,
                    dim=-1,
                    largest=False,
                ).indices
                batch_indices = torch.arange(batch_size, device=coords_start.device)[
                    :, None, None
                ]
                gt_neighbors = gt_points[batch_indices, neighbor_indices]
                gt_neighbor_dot = (
                    (gt_points[:, :, None, :] * gt_neighbors)
                    .sum(dim=-1)
                    .clamp(-1.0, 1.0)
                )
                neighbor_valid = valid_mask[:, 0, :][batch_indices, neighbor_indices]
                pair_mask = (
                    aux_mask[:, :, None]
                    & neighbor_valid
                    & (gt_neighbor_dot >= gt_cos_threshold)
                )

            residual_neighbors = residual_points[batch_indices, neighbor_indices]
            residual_delta = residual_points[:, :, None, :] - residual_neighbors
            pair_weights = (
                None if aux_weights is None else aux_weights[:, :, None, None]
            )
        else:
            # Explicit edge path: the dataset guarantees that both endpoints
            # were selected as one group and that every edge stays within the
            # exact same view_source in the original full point cloud.
            edge_index = normal_relation_edge_index.to(
                device=coords_start.device,
                dtype=torch.long,
            )
            if edge_index.ndim == 2 and batch_size == 1:
                edge_index = edge_index.unsqueeze(0)
            if (
                edge_index.ndim != 3
                or edge_index.shape[0] != batch_size
                or edge_index.shape[1] != 2
            ):
                raise ValueError(
                    "normal_relation_edge_index 应为 [B,2,E]，"
                    f"当前 shape={tuple(edge_index.shape)}"
                )
            if torch.any((edge_index < 0) | (edge_index >= num_points)):
                raise ValueError(
                    "normal_relation_edge_index 包含超出采样点范围的索引: "
                    f"num_points={num_points}"
                )
            edge_count = edge_index.shape[-1]
            if edge_count == 0:
                return normal_pred_raw.sum() * 0.0
            if normal_relation_edge_valid_mask is None:
                explicit_edge_mask = torch.ones(
                    (batch_size, edge_count),
                    device=coords_start.device,
                    dtype=torch.bool,
                )
            else:
                explicit_edge_mask = normal_relation_edge_valid_mask.to(
                    device=coords_start.device,
                    dtype=torch.bool,
                )
                if explicit_edge_mask.ndim == 1 and batch_size == 1:
                    explicit_edge_mask = explicit_edge_mask.unsqueeze(0)
                if explicit_edge_mask.shape != (batch_size, edge_count):
                    raise ValueError(
                        "normal_relation_edge_valid_mask 应为 [B,E]，"
                        f"当前 shape={tuple(explicit_edge_mask.shape)}, "
                        f"期望 {(batch_size, edge_count)}"
                    )

            center_indices = edge_index[:, 0, :]
            neighbor_indices = edge_index[:, 1, :]
            batch_indices = torch.arange(batch_size, device=coords_start.device)[
                :, None
            ]
            with torch.no_grad():
                gt_centers = gt_points[batch_indices, center_indices]
                gt_neighbors = gt_points[batch_indices, neighbor_indices]
                gt_neighbor_dot = (
                    (gt_centers * gt_neighbors).sum(dim=-1).clamp(-1.0, 1.0)
                )
                center_valid = aux_mask.gather(1, center_indices)
                neighbor_valid = valid_mask[:, 0, :].gather(1, neighbor_indices)
                pair_mask = (
                    explicit_edge_mask
                    & center_valid
                    & neighbor_valid
                    & (gt_neighbor_dot >= gt_cos_threshold)
                )

            residual_centers = residual_points[batch_indices, center_indices]
            residual_neighbors = residual_points[batch_indices, neighbor_indices]
            residual_delta = residual_centers - residual_neighbors
            if aux_weights is None:
                pair_weights = None
            elif aux_weights.shape[1] == 1:
                pair_weights = aux_weights[:, :, None]
            else:
                pair_weights = aux_weights.gather(1, center_indices)[:, :, None]

        pair_error = F.smooth_l1_loss(
            residual_delta,
            torch.zeros_like(residual_delta),
            reduction="none",
            beta=self.normal_relation_beta,
        )
        component_mask = pair_mask[..., None].expand_as(pair_error)
        return self._masked_mean(
            pair_error,
            component_mask,
            weights=pair_weights,
        )

    @staticmethod
    def _extract(a, t, x_shape):
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, data_start, t, noise=None):
        if noise is None:
            noise = torch.randn(data_start.shape, device=data_start.device)
        assert noise.shape == data_start.shape

        return (
            self._extract(self.sqrt_alphas_cumprod, t, data_start.shape) * data_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, data_start.shape)
            * noise
        )

    def q_sample_next(self, data_t, t, noise=None):
        if noise is None:
            noise = torch.randn(data_t.shape, device=data_t.device)
        assert noise.shape == data_t.shape

        return (
            self._extract(self.sqrt_one_minus_betas, t, data_t.shape) * data_t
            + self._extract(self.betas, t, data_t.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        denoise_fn,
        coords_t,
        feats_t,
        t,
        coords_clip_range=None,
        feats_clipping_range=None,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        light_probe=None,
        source_uv=None,
    ):
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        denoise_kwargs = {
            "cond_tokens": cond_tokens,
            "cond_point_colors": cond_point_colors,
        }
        if view_source is not None:
            denoise_kwargs["view_source"] = view_source
        if source_uv is not None:
            denoise_kwargs["source_uv"] = source_uv
        if light_probe is not None:
            denoise_kwargs["light_probe"] = light_probe
        model_out_coords, model_out_feats = denoise_fn(
            coords_t, feats_t, t, **denoise_kwargs
        )

        model_variance_coords = self._extract(
            model_variance, t, coords_t.shape
        ) * torch.ones_like(coords_t)
        model_log_variance_coords = self._extract(
            model_log_variance, t, coords_t.shape
        ) * torch.ones_like(coords_t)

        coords_recon = self._predict_xstart_from_eps(
            coords_t, t=t, eps=model_out_coords
        )
        if coords_clip_range is not None:
            coords_clip_min, coords_clip_max = coords_clip_range
            coords_recon = torch.clamp(coords_recon, coords_clip_min, coords_clip_max)
        model_coords_mean, _, _ = self.q_posterior_mean_variance(
            x_start=coords_recon, x_t=coords_t, t=t
        )

        model_variance_feats = self._extract(
            model_variance, t, feats_t.shape
        ) * torch.ones_like(feats_t)
        model_log_variance_feats = self._extract(
            model_log_variance, t, feats_t.shape
        ) * torch.ones_like(feats_t)

        feats_recon = self._predict_xstart_from_eps(feats_t, t=t, eps=model_out_feats)
        if feats_clipping_range is not None:
            feat_clip_min, feat_clip_max = feats_clipping_range
            feats_recon = torch.clamp(feats_recon, feat_clip_min, feat_clip_max)
        model_feats_mean, _, _ = self.q_posterior_mean_variance(
            x_start=feats_recon, x_t=feats_t, t=t
        )

        return (
            model_coords_mean,
            model_variance_coords,
            model_log_variance_coords,
            coords_recon,
            model_feats_mean,
            model_variance_feats,
            model_log_variance_feats,
            feats_recon,
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, x_start):
        assert x_start.shape == x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_start.shape)

    def p_sample(
        self,
        denoise_fn,
        coords_t,
        feats_t,
        t,
        coords_clip_range=None,
        feats_clipping_range=None,
        fix_coords=False,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        light_probe=None,
        source_uv=None,
    ):
        (
            model_coords_mean,
            model_variance_coords,
            model_log_variance_coords,
            coords_recon,
            model_feats_mean,
            model_variance_feats,
            model_log_variance_feats,
            feats_recon,
        ) = self.p_mean_variance(
            denoise_fn,
            coords_t=coords_t,
            feats_t=feats_t,
            t=t,
            coords_clip_range=coords_clip_range,
            feats_clipping_range=feats_clipping_range,
            cond_tokens=cond_tokens,
            cond_point_colors=cond_point_colors,
            view_source=view_source,
            light_probe=light_probe,
            source_uv=source_uv,
        )
        if fix_coords:
            coords_next = coords_t
            coords_recon = coords_t
        else:
            coords_noise = torch.randn_like(coords_t)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(coords_t.shape) - 1)))
            )  # no noise when t == 0
            coords_next = (
                model_coords_mean
                + nonzero_mask
                * torch.exp(0.5 * model_log_variance_coords)
                * coords_noise
            )

        feats_noise = torch.randn_like(feats_t)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(feats_t.shape) - 1)))
        )  # no noise when t == 0
        feats_next = (
            model_feats_mean
            + nonzero_mask * torch.exp(0.5 * model_log_variance_feats) * feats_noise
        )

        return coords_next, coords_recon, feats_next, feats_recon

    def p_sample_loop_trajectory(
        self,
        denoise_fn,
        coords_start,
        feats_start,
        coords_clip_range=None,
        feats_clip_range=None,
        progress=False,
        fix_coords=False,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        light_probe=None,
        source_uv=None,
    ):
        device = coords_start.device

        coords_t = coords_start
        feats_t = feats_start

        coords_ts = [coords_t]
        feats_ts = [feats_t]

        coords_recons = []
        feats_recons = []

        indices = list(
            range(self.num_timesteps)
        )[
            ::-1
        ]  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for t in indices:
            t_batch = torch.tensor([t] * coords_start.shape[0], device=device)
            out = self.p_sample(
                denoise_fn=denoise_fn,
                coords_t=coords_t,
                feats_t=feats_t,
                t=t_batch,
                coords_clip_range=coords_clip_range,
                feats_clipping_range=feats_clip_range,
                fix_coords=fix_coords,
                cond_tokens=cond_tokens,
                cond_point_colors=cond_point_colors,
                view_source=view_source,
                light_probe=light_probe,
                source_uv=source_uv,
            )
            coords_t, coords_recon, feats_t, feats_recon = out
            coords_ts.append(coords_t)
            coords_recons.append(coords_recon)
            feats_ts.append(feats_t)
            feats_recons.append(feats_recon)

        return coords_ts, coords_recons, feats_ts, feats_recons

    def ddim_sample(
        self,
        denoise_fn,
        coords_t,
        feats_t,
        t,
        t_prev,
        fix_coords=False,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        eta=0.0,
        coords_clip_range=None,
        feats_clip_range=None,
        project_normal_x0=False,
        light_probe=None,
        source_uv=None,
    ):
        """单步 DDIM 采样: 从 t 到 t_prev"""
        device = coords_t.device
        B = coords_t.shape[0]
        t_batch = torch.tensor([t] * B, device=device)

        # 预测噪声
        denoise_kwargs = {
            "cond_tokens": cond_tokens,
            "cond_point_colors": cond_point_colors,
        }
        if view_source is not None:
            denoise_kwargs["view_source"] = view_source
        if source_uv is not None:
            denoise_kwargs["source_uv"] = source_uv
        if light_probe is not None:
            denoise_kwargs["light_probe"] = light_probe
        eps_coords, eps_feats = denoise_fn(coords_t, feats_t, t_batch, **denoise_kwargs)

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = (
            self.alphas_cumprod[t_prev] if t_prev >= 0 else coords_t.new_tensor(1.0)
        )

        # 预测 x0
        pred_x0_coords = (coords_t - torch.sqrt(1 - alpha_t) * eps_coords) / torch.sqrt(
            alpha_t
        )
        pred_x0_feats = (feats_t - torch.sqrt(1 - alpha_t) * eps_feats) / torch.sqrt(
            alpha_t
        )
        if coords_clip_range is not None:
            pred_x0_coords = pred_x0_coords.clamp(
                min=coords_clip_range[0], max=coords_clip_range[1]
            )
            eps_coords = (coords_t - torch.sqrt(alpha_t) * pred_x0_coords) / torch.sqrt(
                1 - alpha_t
            )
        if feats_clip_range is not None:
            pred_x0_feats = pred_x0_feats.clamp(
                min=feats_clip_range[0], max=feats_clip_range[1]
            )
        if project_normal_x0 and self.normal_slice is not None:
            normal_x0 = pred_x0_feats[:, self.normal_slice, :]
            projected_normal = F.normalize(normal_x0, dim=1, eps=1e-6)
            pred_x0_feats = pred_x0_feats.clone()
            pred_x0_feats[:, self.normal_slice, :] = projected_normal
        if feats_clip_range is not None or (
            project_normal_x0 and self.normal_slice is not None
        ):
            # DDIM's direction term must agree with the constrained x0.
            eps_feats = (feats_t - torch.sqrt(alpha_t) * pred_x0_feats) / torch.sqrt(
                1 - alpha_t
            )
        # DDIM 方向项
        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        dir_coeff = torch.sqrt(1 - alpha_prev - sigma**2)

        if fix_coords:
            coords_next = coords_t
        else:
            noise_coords = (
                torch.randn_like(coords_t) if t_prev > 0 else torch.zeros_like(coords_t)
            )
            coords_next = (
                torch.sqrt(alpha_prev) * pred_x0_coords
                + dir_coeff * eps_coords
                + sigma * noise_coords
            )

        noise_feats = (
            torch.randn_like(feats_t) if t_prev > 0 else torch.zeros_like(feats_t)
        )
        feats_next = (
            torch.sqrt(alpha_prev) * pred_x0_feats
            + dir_coeff * eps_feats
            + sigma * noise_feats
        )

        return coords_next, feats_next

    def ddim_sample_loop(
        self,
        denoise_fn,
        coords_start,
        feats_start,
        num_steps=50,
        fix_coords=False,
        eta=0.0,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        progress=False,
        coords_clip_range=None,
        feats_clip_range=None,
        project_normal_x0=False,
        force_fp32=False,
        light_probe=None,
        source_uv=None,
    ):
        """DDIM 采样循环: 用 num_steps 步完成 T 步的去噪"""
        device = coords_start.device
        # 均匀选取 timestep 子集
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]  # 从大到小

        coords_t = coords_start
        feats_t = feats_start

        indices = timesteps
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i, t in enumerate(indices):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            autocast_context = (
                nullcontext()
                if force_fp32 or coords_t.device.type != "cuda"
                else torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            )
            with autocast_context:
                coords_t, feats_t = self.ddim_sample(
                    denoise_fn,
                    coords_t,
                    feats_t,
                    t,
                    t_prev,
                    fix_coords=fix_coords,
                    cond_tokens=cond_tokens,
                    cond_point_colors=cond_point_colors,
                    view_source=view_source,
                    eta=eta,
                    light_probe=light_probe,
                    source_uv=source_uv,
                    coords_clip_range=coords_clip_range,
                    feats_clip_range=feats_clip_range,
                    project_normal_x0=project_normal_x0,
                )

        return coords_t, feats_t

    def _vb_terms_bpd(
        self, denoise_fn, coords_start, coords_t, feats_start, feats_t, t
    ):
        (
            model_coords_mean,
            model_variance_coords,
            model_log_variance_coords,
            coords_recon,
            model_feats_mean,
            model_variance_feats,
            model_log_variance_feats,
            feats_recon,
        ) = self.p_mean_variance(denoise_fn, coords_t=coords_t, feats_t=feats_t, t=t)

        true_coords_mean, _, true_coords_log_variance_clipped = (
            self.q_posterior_mean_variance(x_start=coords_start, x_t=coords_t, t=t)
        )
        kl_coords = normal_kl(
            true_coords_mean,
            true_coords_log_variance_clipped,
            model_coords_mean,
            model_log_variance_coords,
        )
        kl_coords = mean_flat(kl_coords) / np.log(
            2.0
        )  # mean over all non-batch dimensions
        decoder_nll_coords = -discretized_gaussian_log_likelihood(
            coords_start,
            means=model_coords_mean,
            log_scales=0.5 * model_log_variance_coords,
        )
        decoder_nll_coords = mean_flat(decoder_nll_coords) / np.log(2.0)
        kl_coords = torch.where((t == 0), decoder_nll_coords, kl_coords)

        true_feats_mean, _, true_feats_log_variance_clipped = (
            self.q_posterior_mean_variance(x_start=feats_start, x_t=feats_t, t=t)
        )
        kl_feats = normal_kl(
            true_feats_mean,
            true_feats_log_variance_clipped,
            model_feats_mean,
            model_log_variance_feats,
        )
        kl_feats = mean_flat(kl_feats) / np.log(
            2.0
        )  # mean over all non-batch dimensions
        decoder_nll_feats = -discretized_gaussian_log_likelihood(
            feats_start,
            means=model_feats_mean,
            log_scales=0.5 * model_log_variance_feats,
        )
        decoder_nll_feats = mean_flat(decoder_nll_feats) / np.log(2.0)
        kl_feats = torch.where((t == 0), decoder_nll_feats, kl_feats)

        return kl_coords, coords_recon, kl_feats, feats_recon

    def p_losses(
        self,
        denoise_fn,
        coords_start,
        feats_start,
        t,
        coords_noise=None,
        feats_noise=None,
        fix_coords=False,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        normal_valid_mask=None,
        normal_relation_edge_index=None,
        normal_relation_edge_valid_mask=None,
        light_probe=None,
        source_uv=None,
    ):
        """
        Training loss calculation
        """
        if fix_coords:
            coords_noise = torch.zeros_like(coords_start)
        elif coords_noise is None:
            coords_noise = torch.randn(
                coords_start.shape, dtype=coords_start.dtype, device=coords_start.device
            )
        N, coords_dim, num_points = coords_start.shape
        _, feats_dim, _ = feats_start.shape
        assert t.shape == torch.Size([N])
        assert (
            coords_noise.shape == coords_start.shape
            and coords_noise.dtype == coords_start.dtype
        )
        if fix_coords and self.fix_coords_training_mode == "identity":
            # Match fix_coords inference, where the normalized coordinates are
            # held fixed instead of being attenuated by sqrt(alpha_bar_t).
            coords_t = coords_start
        else:
            # "scaled" preserves historical checkpoints/training behavior.
            coords_t = self.q_sample(data_start=coords_start, t=t, noise=coords_noise)

        if feats_noise is None:
            feats_noise = torch.randn(
                feats_start.shape, dtype=feats_start.dtype, device=feats_start.device
            )
        assert (
            feats_noise.shape == feats_start.shape
            and feats_noise.dtype == feats_start.dtype
        )
        feats_t = self.q_sample(data_start=feats_start, t=t, noise=feats_noise)
        denoise_kwargs = {
            "cond_tokens": cond_tokens,
            "cond_point_colors": cond_point_colors,
        }
        if view_source is not None:
            denoise_kwargs["view_source"] = view_source
        if source_uv is not None:
            denoise_kwargs["source_uv"] = source_uv
        if light_probe is not None:
            denoise_kwargs["light_probe"] = light_probe
        eps_coords, eps_feats = denoise_fn(coords_t, feats_t, t, **denoise_kwargs)
        pointwise_coords_loss = (coords_noise - eps_coords) ** 2  # [N, 3, num_points]
        pointwise_feats_loss = (
            feats_noise - eps_feats
        ) ** 2  # [N, feat_dim, num_points]

        min_snr_w = None
        if self.min_snr_gamma is not None and self.min_snr_gamma > 0:
            # Min-SNR-γ 加权 (Hang et al. 2023)。设为 None/<=0 可回退到旧版未加权 loss。
            snr_t = self._extract(self.snr, t, coords_start.shape)  # [N,1,1]
            min_snr_w = torch.clamp(snr_t, max=self.min_snr_gamma) / snr_t
            pointwise_coords_loss = pointwise_coords_loss * min_snr_w
            pointwise_feats_loss = pointwise_feats_loss * min_snr_w

        pointwise_coords_loss = (
            pointwise_coords_loss / 2.0
        )  # Divide by 2 to take the average of the coords and the feats losses
        pointwise_feats_loss = (
            pointwise_feats_loss / 2.0
        )  # Divide by 2 to take the average of the coords and the feats losses
        if fix_coords:
            coords_loss = torch.tensor(0.0, device=coords_start.device)
        else:
            coords_loss = pointwise_coords_loss.mean()

        if self.normal_slice is None:
            # Preserve the historical five-channel loss exactly.
            feats_loss = pointwise_feats_loss.mean()
            loss = coords_loss + feats_loss
            pointwise_losses = {
                "pointwise_coords_loss": pointwise_coords_loss,
                "pointwise_feats_loss": pointwise_feats_loss,
            }
            sub_losses = {"00_coords_loss": coords_loss, "01_feats_loss": feats_loss}
            return loss, sub_losses, pointwise_losses

        if feats_dim != self.feature_schema.feats_dim:
            raise ValueError(
                f"schema {self.feature_schema.name!r} 需要 {self.feature_schema.feats_dim} "
                f"通道，loss 收到 {feats_dim} 通道"
            )

        pbr_eps_loss = pointwise_feats_loss[:, : self.pbr_stop, :].mean()
        normal_gt = feats_start[:, self.normal_slice, :]
        valid_mask = self._prepare_normal_valid_mask(normal_valid_mask, normal_gt)
        normal_mask_channels = valid_mask.expand(-1, normal_gt.shape[1], -1)
        normal_eps_loss = self._masked_mean(
            pointwise_feats_loss[:, self.normal_slice, :],
            normal_mask_channels,
        )

        # x0 reconstruction is only used by normal-space auxiliary losses.  The
        # epsilon objective keeps its Min-SNR weighting, while x0 objectives use
        # their independently configured weighting plus an optional timestep gate.
        # Clamp protects direction/unit terms from extreme high-noise estimates;
        # raw-vector robust losses deliberately retain a direct, unclamped gradient.
        feats_x0_pred = self._predict_xstart_from_eps(feats_t, t=t, eps=eps_feats)
        normal_pred_raw = feats_x0_pred[:, self.normal_slice, :]
        normal_pred = normal_pred_raw.clamp(
            -self.normal_x0_clamp,
            self.normal_x0_clamp,
        )
        normal_pred_length = torch.linalg.vector_norm(normal_pred, dim=1)
        normal_pred_unit = F.normalize(normal_pred, dim=1, eps=1e-6)
        normal_gt_unit = F.normalize(normal_gt, dim=1, eps=1e-6)
        normal_dot = (normal_pred_unit * normal_gt_unit).sum(dim=1).clamp(-1.0, 1.0)

        aux_mask = valid_mask[:, 0, :]
        if self.normal_aux_max_timestep is not None:
            aux_mask = aux_mask & (t[:, None] <= self.normal_aux_max_timestep)
        aux_weights = self._normal_aux_weights(t, coords_start.shape, min_snr_w)
        robust_aux_weights = self._normal_robust_aux_weights(
            t,
            coords_start.shape,
            min_snr_w,
            aux_weights,
        )
        normal_cos_loss = self._masked_mean(
            1.0 - normal_dot,
            aux_mask,
            weights=aux_weights,
        )
        zero_aux = normal_pred_unit.sum() * 0.0
        normal_cos_tail_loss = (
            self._normal_tail_loss(1.0 - normal_dot, aux_mask, aux_weights)
            if self.normal_cos_tail_weight > 0.0
            else zero_aux
        )
        normal_smooth_loss = (
            self._normal_smoothness_loss(
                coords_start,
                normal_pred_unit,
                normal_gt_unit,
                valid_mask,
                aux_mask,
                aux_weights,
            )
            if self.normal_smooth_weight > 0.0
            else zero_aux
        )
        normal_relation_loss = (
            self._normal_relation_loss(
                coords_start,
                normal_pred_unit,
                normal_gt_unit,
                valid_mask,
                aux_mask,
                aux_weights,
            )
            if self.normal_relation_weight > 0.0
            else zero_aux
        )
        normal_x0_huber_loss = (
            self._normal_x0_huber_loss(
                normal_pred_raw,
                normal_gt,
                aux_mask,
                robust_aux_weights,
            )
            if self.normal_x0_huber_weight > 0.0
            else zero_aux
        )
        normal_residual_relation_loss = (
            self._normal_residual_relation_loss(
                coords_start,
                normal_pred_raw,
                normal_gt,
                normal_gt_unit,
                valid_mask,
                aux_mask,
                robust_aux_weights,
                normal_relation_edge_index=normal_relation_edge_index,
                normal_relation_edge_valid_mask=(normal_relation_edge_valid_mask),
            )
            if self.normal_residual_relation_weight > 0.0
            else zero_aux
        )
        normal_unit_loss = self._masked_mean(
            (normal_pred_length - 1.0) ** 2,
            aux_mask,
            weights=aux_weights,
        )
        with torch.no_grad():
            normal_angle_deg = self._masked_mean(
                torch.rad2deg(torch.acos(normal_dot.detach())),
                valid_mask[:, 0, :],
            )

        feats_loss = (
            self.pbr_eps_weight * pbr_eps_loss
            + self.normal_eps_weight * normal_eps_loss
            + self.normal_cos_weight * normal_cos_loss
            + self.normal_cos_tail_weight * normal_cos_tail_loss
            + self.normal_smooth_weight * normal_smooth_loss
            + self.normal_relation_weight * normal_relation_loss
            + self.normal_x0_huber_weight * normal_x0_huber_loss
            + self.normal_residual_relation_weight * normal_residual_relation_loss
            + self.normal_unit_weight * normal_unit_loss
        )
        loss = coords_loss + feats_loss

        pointwise_losses = {
            "pointwise_coords_loss": pointwise_coords_loss,
            "pointwise_feats_loss": pointwise_feats_loss,
            "normal_valid_mask": valid_mask,
        }
        sub_losses = {
            "00_coords_loss": coords_loss,
            "01_feats_loss": feats_loss,
            "01_pbr_eps_loss": pbr_eps_loss,
            "02_normal_eps_loss": normal_eps_loss,
            "03_normal_cos_loss": normal_cos_loss,
            "04_normal_unit_loss": normal_unit_loss,
            "05_normal_angle_deg": normal_angle_deg,
            "06_normal_cos_tail_loss": normal_cos_tail_loss,
            "07_normal_smooth_loss": normal_smooth_loss,
            "08_normal_relation_loss": normal_relation_loss,
            "09_normal_x0_huber_loss": normal_x0_huber_loss,
            "10_normal_residual_relation_loss": normal_residual_relation_loss,
        }

        return loss, sub_losses, pointwise_losses

    def _prior_bpd(self, x_start):
        with torch.no_grad():
            N, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(N, dtype=torch.int64, device=x_start.device).fill_(T - 1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(
                mean1=qt_mean,
                logvar1=qt_log_variance,
                mean2=torch.tensor([0.0]).to(qt_mean),
                logvar2=torch.tensor([0.0]).to(qt_log_variance),
            )
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.0)

    def calc_bpd_loop(self, denoise_fn, coords_start, feats_start):
        with torch.no_grad():
            batch_size, T = coords_start.shape[0], self.num_timesteps
            device = coords_start.device

            vbs_coords = []
            xstart_mses_coords = []
            mses_coords = []
            vbs_feats = []
            xstart_mses_feats = []
            mses_feats = []

            for t in list(
                range(self.num_timesteps)
            )[
                ::-1
            ]:  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999
                t_batch = torch.tensor([t] * batch_size, device=device)

                noise_coords = torch.randn_like(coords_start)
                coords_t = self.q_sample(
                    data_start=coords_start, t=t_batch, noise=noise_coords
                )

                noise_feats = torch.randn_like(feats_start)
                feats_t = self.q_sample(
                    data_start=feats_start, t=t_batch, noise=noise_feats
                )

                with torch.no_grad():
                    vb_terms_bpd_result = self._vb_terms_bpd(
                        denoise_fn,
                        coords_start=coords_start,
                        coords_t=coords_t,
                        feats_start=feats_start,
                        feats_t=feats_t,
                        t=t_batch,
                    )

                kl_coords, coords_recon, kl_feats, feats_recon = vb_terms_bpd_result

                vbs_coords.append(kl_coords)
                vbs_feats.append(kl_feats)

                xstart_mse_coords = mean_flat((coords_recon - coords_start) ** 2)
                xstart_mse_feats = mean_flat((feats_recon - feats_start) ** 2)
                xstart_mses_coords.append(xstart_mse_coords)
                xstart_mses_feats.append(xstart_mse_feats)

                eps_coords = self._predict_eps_from_xstart(
                    coords_t, t_batch, coords_recon
                )
                mse_coords = mean_flat((eps_coords - noise_coords) ** 2)
                mses_coords.append(mse_coords)

                eps_feats = self._predict_eps_from_xstart(feats_t, t_batch, feats_recon)
                mse_feats = mean_flat((eps_feats - noise_feats) ** 2)
                mses_feats.append(mse_feats)

            vbs_coords = torch.stack(vbs_coords, dim=1)  # shape N, T
            xstart_mses_coords = torch.stack(xstart_mses_coords, dim=1)
            mses_coords = torch.stack(mses_coords, dim=1)

            prior_bpd_coords = self._prior_bpd(coords_start)
            total_bpd_coords = vbs_coords.sum(dim=1) + prior_bpd_coords

            vbs_feats = torch.stack(vbs_feats, dim=1)
            xstart_mses_feats = torch.stack(xstart_mses_feats, dim=1)
            mses_feats = torch.stack(mses_feats, dim=1)

            prior_bpd_feats = self._prior_bpd(feats_start)
            total_bpd_feats = vbs_feats.sum(dim=1) + prior_bpd_feats

            return (
                total_bpd_coords,
                vbs_coords,
                prior_bpd_coords,
                xstart_mses_coords,
                mses_coords,
                total_bpd_feats,
                vbs_feats,
                prior_bpd_feats,
                xstart_mses_feats,
                mses_feats,
            )
