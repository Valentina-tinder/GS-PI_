import os
import torch
import torch.nn as nn
from gs_pi.utils import to_torch, get_torch_model_device, split_num
from gs_pi.pbr_schema import get_feature_schema
from gs_pi.pbr_normalization import PBRNormalFeatureNormalization
from .diffusion_processes.gaussian_diffusion import GaussianDiffusion
from .denoisers.transformer import NPCDTransformer
import torchvision.models as models
import torch.nn.functional as F


class SimpleImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(512, out_dim)

    def forward(self, images):
        (B, V, C, H, W) = images.shape
        x = images.view(B * V, C, H, W)
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat.view(B, V, -1)


class AdvancedImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, images):
        (B, V, C, H, W) = images.shape
        x = images.view(B * V, C, H, W)
        feat_map = self.backbone(x)
        feat_map = self.proj(feat_map)
        (_, D, h, w) = feat_map.shape
        feat_tokens = feat_map.flatten(2)
        feat_tokens = feat_tokens.transpose(1, 2)
        return feat_tokens.reshape(B, V * h * w, D)


class DINOv2ImageEncoder(nn.Module):
    def __init__(self, output_dim=512, model_size="vits14", proj_seed=0):
        """
        model_size 可选: 'vits14' (小巧极速), 'vitb14' (均衡), 'vitl14' (性能怪兽)
        """
        super().__init__()
        repo_candidates = [os.environ.get("NPCD_DINOV2_REPO")]
        dino_repo = next(
            (
                path
                for path in repo_candidates
                if path and os.path.isfile(os.path.join(path, "hubconf.py"))
            ),
            None,
        )
        if dino_repo is None:
            raise FileNotFoundError(
                f"未找到本地 DINOv2 仓库；请设置 NPCD_DINOV2_REPO。候选路径: {[path for path in repo_candidates if path]}"
            )
        self.dino = torch.hub.load(
            repo_or_dir=dino_repo, model=f"dinov2_{model_size}", source="local"
        )
        dino_dim = self.dino.embed_dim
        self.output_dim = int(output_dim)
        self.dino_dim = int(dino_dim)
        self.proj_seed = None if proj_seed is None else int(proj_seed)
        if self.proj_seed is None:
            self.proj = nn.Linear(dino_dim, output_dim)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.proj_seed)
                self.proj = nn.Linear(dino_dim, output_dim)
        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: [B, V, 3, H, W] (输入的多个视角图，当前是 128x128)
        """
        (B, V, C, H, W) = x.shape
        x_flat = x.view(B * V, C, H, W)
        x_flat = F.interpolate(
            x_flat, size=(140, 140), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            ret = self.dino.forward_features(x_flat)
            patch_tokens = ret["x_norm_patchtokens"]
        out = self.proj(patch_tokens)
        (_, N, D) = out.shape
        out = out.view(B, V * N, D)
        return out


class UnitGaussianNormalization(nn.Module):
    def __init__(self, dim, scale_per_axis=False, clip_per_axis=False):
        super().__init__()
        self.dim = dim
        self.scale_per_axis = scale_per_axis
        self.clip_per_axis = clip_per_axis
        self.register_buffer("min", torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer("max", torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer("shift", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim if scale_per_axis else 1))

    def set_from_all_data(self, data):
        data = to_torch(data, self.shift.device).detach()
        assert data.shape[0] == self.dim
        data = data.reshape(self.dim, -1)
        self.shift[:] = data.mean(axis=1)
        self.scale[:] = data.std(axis=1 if self.scale_per_axis else [0, 1])
        data = (data - self.shift[:, None]) / self.scale[:, None]
        if self.clip_per_axis:
            (self.min, _) = data.min(axis=1)
            (self.max, _) = data.max(axis=1)
        else:
            self.min[:] = data.min()
            self.max[:] = data.max()

    def forward(self, x):
        return (x - self.shift[None, :, None]) / self.scale[None, :, None]

    def denormalize(self, x):
        return x * self.scale[None, :, None] + self.shift[None, :, None]


class MinusOneToOneNormalization(nn.Module):
    def __init__(self, dim, scale_per_axis=False, clip_per_axis=False):
        super().__init__()
        self.dim = dim
        self.scale_per_axis = scale_per_axis
        self.clip_per_axis = clip_per_axis
        self.register_buffer("min", torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer("max", torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer("shift", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim if scale_per_axis else 1))

    def set_from_all_data(self, data):
        data = to_torch(data, self.shift.device).detach()
        assert data.shape[0] == self.dim
        data = data.reshape(self.dim, -1)
        self.shift = (data.min(axis=1)[0] + data.max(axis=1)[0]) / 2.0
        scale = (data.max(axis=1)[0] - data.min(axis=1)[0]) / 2.0
        if not self.scale_per_axis:
            self.scale[:] = scale.max()
        else:
            self.scale = scale
        data = (data - self.shift[:, None]) / self.scale[:, None]
        if self.clip_per_axis:
            (self.min, _) = data.min(axis=1)
            (self.max, _) = data.max(axis=1)
        else:
            self.min[:] = data.min()
            self.max[:] = data.max()

    def forward(self, x):
        return (x - self.shift[None, :, None]) / self.scale[None, :, None]

    def denormalize(self, x):
        return x * self.scale[None, :, None] + self.shift[None, :, None]


class DiffusionModel(nn.Module):
    def __init__(
        self,
        coords_dim,
        feats_dim,
        num_points,
        width,
        layers,
        heads,
        use_flash_attn,
        fix_coords=False,
        condition_on_images=False,
        cond_point_dim=0,
        use_two_stream=False,
        z_dim=768,
        num_latents=256,
        num_blocks=6,
        num_compute_layers=4,
        cond_point_top_k=0,
        num_global_views=0,
        view_id_embed_dim=0,
        topk_view_id_conditioning="embed",
        topk_include_confidence=True,
        beta_schedule="cosine",
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
        normal_projection_eps=1e-06,
        backbone_type=None,
    ):
        super().__init__()
        self.coords_dim = coords_dim
        self.feats_dim = feats_dim
        self.num_points = num_points
        self.feature_schema = get_feature_schema(feature_schema)
        if self.feature_schema.feats_dim != int(feats_dim):
            raise ValueError(
                f"feature_schema={self.feature_schema.name!r} 需要 feats_dim={self.feature_schema.feats_dim}，当前为 {feats_dim}"
            )
        self.normal_projection_eps = float(normal_projection_eps)
        resolved_backbone_type = (
            None
            if backbone_type is None
            else str(backbone_type).strip().lower().replace("-", "_")
        )
        try:
            normal_group = self.feature_schema.group("normal")
        except KeyError:
            normal_feature_start = None
            normal_feature_stop = None
        else:
            normal_feature_start = normal_group.start
            normal_feature_stop = normal_group.stop
        self.diffusion_process = GaussianDiffusion(
            beta_schedule=beta_schedule,
            min_snr_gamma=min_snr_gamma,
            feature_schema=self.feature_schema,
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
            fix_coords_training_mode=fix_coords_training_mode,
        )
        self.denoiser = NPCDTransformer(
            coords_dim=coords_dim,
            feats_dim=feats_dim,
            num_points=num_points,
            cond_point_dim=cond_point_dim,
            width=width,
            layers=layers,
            heads=heads,
            use_flash_attn=use_flash_attn,
            use_two_stream=use_two_stream,
            backbone_type=backbone_type,
            z_dim=z_dim,
            num_latents=num_latents,
            num_blocks=num_blocks,
            num_compute_layers=num_compute_layers,
            cond_point_top_k=cond_point_top_k,
            num_global_views=num_global_views,
            view_id_embed_dim=view_id_embed_dim,
            topk_view_id_conditioning=topk_view_id_conditioning,
            topk_include_confidence=topk_include_confidence,
            normal_feature_start=normal_feature_start,
            normal_feature_stop=normal_feature_stop,
        )
        self.condition_on_images = condition_on_images
        if self.condition_on_images:
            self.image_encoder = None
        self.coords_normalization = UnitGaussianNormalization(dim=coords_dim)
        if any((group.name == "normal" for group in self.feature_schema.groups)):
            self.feats_normalization = PBRNormalFeatureNormalization(
                feature_schema=self.feature_schema
            )
        else:
            self.feats_normalization = MinusOneToOneNormalization(dim=feats_dim)
        self.fix_coords = fix_coords

    def configure_normal_finetune(self, unfreeze_last_n_blocks=4):
        """Prepare normal-only projection gradients before DDP wrapping."""
        return self.denoiser.configure_normal_finetune(
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )

    def compute_loss(
        self,
        coords,
        feats,
        input_images=None,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        normal_valid_mask=None,
        normal_relation_edge_index=None,
        normal_relation_edge_valid_mask=None,
        light_probe=None,
        source_uv=None,
    ):
        coords = self.coords_normalization(coords)
        feats = self.feats_normalization(feats)
        (N, _, _) = coords.shape
        t = torch.randint(
            0, self.diffusion_process.num_timesteps, size=(N,), device=coords.device
        )
        if (
            cond_tokens is None
            and self.condition_on_images
            and (input_images is not None)
        ):
            raise ValueError(
                "Precompute DINO tokens with scripts/tools/extract_features_multiview.py"
            )
        return self.diffusion_process.p_losses(
            denoise_fn=self.denoiser,
            coords_start=coords,
            feats_start=feats,
            t=t,
            fix_coords=self.fix_coords,
            cond_tokens=cond_tokens,
            cond_point_colors=cond_point_colors,
            view_source=view_source,
            light_probe=light_probe,
            source_uv=source_uv,
            normal_valid_mask=normal_valid_mask,
            normal_relation_edge_index=normal_relation_edge_index,
            normal_relation_edge_valid_mask=normal_relation_edge_valid_mask,
        )

    def project_normal_features(self, feats):
        """Project predicted Blender world normals to S2; zero stays invalid."""
        try:
            normal_group = self.feature_schema.group("normal")
        except KeyError:
            return feats
        normal_slice = normal_group.channel_slice
        normals = feats[:, normal_slice, :]
        lengths = torch.linalg.vector_norm(normals, dim=1, keepdim=True)
        valid = torch.isfinite(normals).all(dim=1, keepdim=True)
        valid = valid & (lengths > self.normal_projection_eps)
        projected = torch.where(
            valid,
            normals / lengths.clamp_min(self.normal_projection_eps),
            torch.zeros_like(normals),
        )
        result = feats.clone()
        result[:, normal_slice, :] = projected
        return result

    @torch.no_grad()
    def generate(
        self,
        num,
        batch_size=8,
        progress=True,
        input_coords=None,
        input_images=None,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        light_probe=None,
    ):
        assert not self.training, "Model must be in eval mode for generation"
        device = get_torch_model_device(self)
        coords_out = []
        feats_out = []
        for batch_size in split_num(num, batch_size):
            if input_coords is not None:
                coords_start = self.coords_normalization(input_coords)
                should_fix = True
                generation_num_points = int(coords_start.shape[-1])
            else:
                coords_start = torch.randn(
                    batch_size, self.coords_dim, self.num_points, device=device
                )
                should_fix = False
                generation_num_points = int(self.num_points)
            batch_cond_tokens = None
            if cond_tokens is not None:
                batch_cond_tokens = cond_tokens[:batch_size].to(device)
            elif (
                self.condition_on_images
                and input_images is not None
                and (self.image_encoder is not None)
            ):
                batch_cond_tokens = self.image_encoder(input_images)
            feats_start = torch.randn(
                batch_size, self.feats_dim, generation_num_points, device=device
            )
            batch_cond_point_colors = None
            if cond_point_colors is not None:
                batch_cond_point_colors = cond_point_colors[:batch_size].to(device)
            batch_view_source = None
            if view_source is not None:
                batch_view_source = view_source[:batch_size].to(device)
            (coords_ts, _, feats_ts, _) = (
                self.diffusion_process.p_sample_loop_trajectory(
                    denoise_fn=self.denoiser,
                    coords_start=coords_start,
                    feats_start=feats_start,
                    coords_clip_range=(
                        self.coords_normalization.min,
                        self.coords_normalization.max,
                    ),
                    feats_clip_range=(
                        self.feats_normalization.min,
                        self.feats_normalization.max,
                    ),
                    fix_coords=self.fix_coords or input_coords is not None,
                    cond_tokens=batch_cond_tokens,
                    cond_point_colors=batch_cond_point_colors,
                    view_source=batch_view_source,
                    light_probe=None
                    if light_probe is None
                    else light_probe[:batch_size].to(device),
                    progress=progress,
                )
            )
            coords = coords_ts[-1]
            feats = feats_ts[-1]
            coords = self.coords_normalization.denormalize(coords)
            feats = self.feats_normalization.denormalize(feats)
            feats = self.project_normal_features(feats)
            coords = list(coords.unbind())
            feats = list(feats.unbind())
            coords_out += coords
            feats_out += feats
        return (coords_out, feats_out)

    def forward(
        self,
        coords,
        feats,
        images=None,
        cond_tokens=None,
        cond_point_colors=None,
        view_source=None,
        normal_valid_mask=None,
        normal_relation_edge_index=None,
        normal_relation_edge_valid_mask=None,
        light_probe=None,
        source_uv=None,
    ):
        return self.compute_loss(
            coords,
            feats,
            input_images=images,
            cond_tokens=cond_tokens,
            cond_point_colors=cond_point_colors,
            view_source=view_source,
            light_probe=light_probe,
            source_uv=source_uv,
            normal_valid_mask=normal_valid_mask,
            normal_relation_edge_index=normal_relation_edge_index,
            normal_relation_edge_valid_mask=normal_relation_edge_valid_mask,
        )
