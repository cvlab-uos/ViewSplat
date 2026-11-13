from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import math

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory, camera_head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import camera_normalization, convert_pose_to_4x4, depth_projector
from ...geometry.camera_emb import get_plucker_embedding
from .heads.pose_head import PoseHeadCfg
from .heads.dpt_view_head import DPTViewHead
from ...misc.intrinsics_utils import estimate_intrinsics

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderSPF_ViewSplatCfg:
    name: Literal["spf_viewsplat"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    pose_head: PoseHeadCfg
    use_view_dependent_head: bool
    vd_mlp_hidden_dim: int
    vd_mlp_in_dim: int

    view_dependent_update_scale: bool = True
    view_dependent_update_rotation: bool = True
    view_dependent_update_color: bool = True
    view_dependent_update_opacity: bool = True
    view_dependent_update_mean: bool = True

    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pose_make_baseline_1: bool = True
    pose_make_relative: bool = True
    pose_head_type: str = 'mlp'
    estimating_focal: bool = False
    estimating_pose: bool = True




def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderSPF_ViewSplat(Encoder[EncoderSPF_ViewSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderSPF_ViewSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type
       
        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                            depth_mode=('exp', -inf, inf), conf_mode=None,)
 
            
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        self.set_view_dependent_head(cfg)

        if self.cfg.estimating_pose:
            self.set_pose_head(cfg, cfg.pose_head_type)


    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)


    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)

        elif 'dpt' in head_type:
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")
        
    def set_view_dependent_head(self, cfg: EncoderSPF_ViewSplatCfg):
        if not cfg.use_view_dependent_head:
            self.view_dependent_head = None
            self.view_dependent_head2 = None
            return

        # MLP structure definition: [Pose_In, Hidden, enabled GS offsets]
        offset_dim = self.get_view_dependent_offset_dim(cfg)
        if offset_dim == 0:
            raise ValueError(
                "At least one view-dependent update must be enabled when "
                "use_view_dependent_head is true."
            )
        self.vd_mlp_dims = [
            cfg.vd_mlp_in_dim,
            cfg.vd_mlp_hidden_dim,
            offset_dim,
        ]

        self.view_dependent_head = head_factory('dpt', 'view_mlp', self.backbone, mlp_dims=self.vd_mlp_dims)
        self.view_dependent_head2 = head_factory('dpt', 'view_mlp', self.backbone, mlp_dims=self.vd_mlp_dims)

    def get_view_dependent_offset_dim(self, cfg: EncoderSPF_ViewSplatCfg) -> int:
        offset_dim = 0
        if cfg.view_dependent_update_mean:
            offset_dim += 3
        if cfg.view_dependent_update_opacity:
            offset_dim += 1
        if cfg.view_dependent_update_scale:
            offset_dim += 3
        if cfg.view_dependent_update_rotation:
            offset_dim += 4
        if cfg.view_dependent_update_color:
            offset_dim += self.gaussian_adapter.d_in - 7
        return offset_dim

    def set_pose_head(self, cfg, head_type='mlp'):
        self.pose_head = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)
        self.pose_head2 = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)



    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)
    
    

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target: Optional[dict] = None,
        build_base_gaussians: bool = True,
        visualization_keys: Optional[set[str]] = None,
        defer_view_mlp_projection: bool = False,
    ):
        _, v_cxt, _, h, w = context["image"].shape
        if visualization_dump is None:
            requested_visualizations = set()
        elif visualization_keys is None:
            requested_visualizations = {
                "depth", "scales", "rotations", "means", "opacities"
            }
        else:
            requested_visualizations = set(visualization_keys)

        if target is not None:
            v_tgt = target["image"].shape[1]
            context_target = {
                "image": normalize_image(
                    torch.cat([context["image"], target["image"]], dim=1)
                ),
                "intrinsics": torch.cat(
                    [context["intrinsics"], target["intrinsics"]], dim=1
                ),
            }
            out = self.backbone(context_target, target_num_views=v_tgt)
            dec_feat, dec_feat_w_tgt = out["dec_feat"], out["dec_feat_w_tgt"]
        else:
            v_tgt = 0
            context_input = {
                "image": normalize_image(context["image"]),
                "intrinsics": context["intrinsics"],
            }
            out = self.backbone(context_input)
            dec_feat = out["dec_feat"]

        shape, images = out["shape"], out["images"]

        with torch.amp.autocast("cuda", enabled=False):
            all_mean_res, all_other_params = [], []
            all_vd_params, all_vd_features = [], []
            if self.cfg.estimating_pose:
                all_pose_params, all_pose_params_cwt = [], []

            all_mean_res.append(
                self._downstream_head(
                    1,
                    [tok[:, 0].float() for tok in dec_feat],
                    shape[:, 0],
                )
            )
            for i in range(1, v_cxt):
                all_mean_res.append(
                    self._downstream_head(
                        2,
                        [tok[:, i].float() for tok in dec_feat],
                        shape[:, i],
                    )
                )

            if "dpt" not in self.gs_params_head_type:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")
            gs_res = self.gaussian_param_head(
                [tok[:, 0].float() for tok in dec_feat],
                images[:, 0, :3],
                shape[0, 0].cpu().tolist(),
            )
            all_other_params.append(rearrange(gs_res, "b d h w -> b (h w) d"))
            for i in range(1, v_cxt):
                gs_res = self.gaussian_param_head2(
                    [tok[:, i].float() for tok in dec_feat],
                    images[:, i, :3],
                    shape[0, i].cpu().tolist(),
                )
                all_other_params.append(
                    rearrange(gs_res, "b d h w -> b (h w) d")
                )

            if self.cfg.use_view_dependent_head:
                use_tiled_view_mlp = defer_view_mlp_projection
                view_head = self.view_dependent_head
                view_args = (
                    [tok[:, 0].float() for tok in dec_feat],
                    images[:, 0, :3],
                    shape[0, 0].cpu().tolist(),
                )
                if use_tiled_view_mlp:
                    all_vd_features.append(view_head.forward_features(*view_args))
                else:
                    view_params = view_head(*view_args)
                    all_vd_params.append(
                        rearrange(view_params, "b c h w -> b (h w) c")
                    )

                for i in range(1, v_cxt):
                    view_head = self.view_dependent_head2
                    view_args = (
                        [tok[:, i].float() for tok in dec_feat],
                        images[:, i, :3],
                        shape[0, i].cpu().tolist(),
                    )
                    if use_tiled_view_mlp:
                        all_vd_features.append(view_head.forward_features(*view_args))
                    else:
                        view_params = view_head(*view_args)
                        all_vd_params.append(
                            rearrange(view_params, "b c h w -> b (h w) c")
                        )

            if self.cfg.estimating_pose:
                all_pose_params.append(
                    self.pose_head(
                        [tok[:, 0].float() for tok in dec_feat],
                        shape[0, 0].cpu().tolist(),
                    )
                )
                for i in range(1, v_cxt):
                    all_pose_params.append(
                        self.pose_head2(
                            [tok[:, i].float() for tok in dec_feat],
                            shape[0, i].cpu().tolist(),
                        )
                    )

                if target is not None:
                    all_pose_params_cwt.append(
                        self.pose_head(
                            [tok[:, 0].float() for tok in dec_feat_w_tgt],
                            shape[0, 0].cpu().tolist(),
                        )
                    )
                    for i in range(1, v_cxt + v_tgt):
                        all_pose_params_cwt.append(
                            self.pose_head2(
                                [tok[:, i].float() for tok in dec_feat_w_tgt],
                                shape[0, i].cpu().tolist(),
                            )
                        )

        raw_gaussians = torch.stack(all_other_params, dim=1)

        if self.cfg.estimating_pose:
            poses_enc = torch.stack(all_pose_params, dim=1)
            pred_extrinsics = self.process_pose(poses_enc, v_cxt)
            if target is not None:
                poses_enc_cwt = torch.stack(all_pose_params_cwt, dim=1)
                pred_extrinsics_cwt = self.process_pose(poses_enc_cwt, v_cxt)

        pts_all = torch.stack(
            [mean_result["pts3d"] for mean_result in all_mean_res],
            dim=1,
        )
        pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz")
        context_extrinsics = (
            pred_extrinsics[:, :v_cxt]
            if self.cfg.estimating_pose
            else context["extrinsics"]
        )

        raw_gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        raw_densities = raw_gaussians[..., 0]
        densities = raw_densities.sigmoid().unsqueeze(-1)
        gaussian_parameters = raw_gaussians[..., 1:]
        pts_all = pts_all.unsqueeze(-2)

        need_base_gaussians = build_base_gaussians or bool(
            requested_visualizations & {"scales", "rotations", "opacities"}
        )
        base_gaussians = None
        if need_base_gaussians:
            base_gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(
                    gaussian_parameters,
                    "b v r srf c -> b v r srf () c",
                ),
            )

        if visualization_dump is not None:
            if "depth" in requested_visualizations:
                visualization_dump["depth"] = self.process_depth(
                    context_extrinsics,
                    rearrange(
                        pts_all.squeeze(-2),
                        "b v (h w) xyz -> b v h w xyz",
                        h=h,
                        w=w,
                    ),
                )
            if "means" in requested_visualizations:
                visualization_dump["means"] = rearrange(
                    pts_all,
                    "b v (h w) srf xyz -> b v h w srf xyz",
                    h=h,
                    w=w,
                )
            if "scales" in requested_visualizations:
                visualization_dump["scales"] = rearrange(
                    base_gaussians.scales,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                )
            if "rotations" in requested_visualizations:
                visualization_dump["rotations"] = rearrange(
                    base_gaussians.rotations,
                    "b v r srf spp xyzw -> b (v r srf spp) xyzw",
                )
            if "opacities" in requested_visualizations:
                visualization_dump["opacities"] = rearrange(
                    base_gaussians.opacities,
                    "b v (h w) srf spp -> b v h w srf spp",
                    h=h,
                    w=w,
                )

        mean_map = rearrange(
            pts_all,
            "b v (h w) srf xyz -> b v h w srf xyz",
            h=h,
            w=w,
        ).squeeze(-2)
        if self.cfg.estimating_focal:
            intrinsics = estimate_intrinsics(mean_map, h, w)
            pred_intrinsics = intrinsics.unsqueeze(1).repeat(1, v_cxt, 1, 1)
            pred_intrinsics_cwt = intrinsics.unsqueeze(1).repeat(
                1, v_cxt + v_tgt, 1, 1
            )

        encoder_output = {}
        if build_base_gaussians:
            encoder_output["gaussians"] = Gaussians(
                rearrange(
                    base_gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    base_gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    base_gaussians.rotations,
                    "b v r srf spp i -> b (v r srf spp) i",
                ),
                rearrange(
                    base_gaussians.scales,
                    "b v r srf spp i -> b (v r srf spp) i",
                ),
                rearrange(
                    base_gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    base_gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )

        if self.cfg.use_view_dependent_head:
            if defer_view_mlp_projection:
                encoder_output["vd_mlp_features"] = all_vd_features
            else:
                encoder_output["vd_mlp_params"] = torch.stack(all_vd_params, dim=1)
            encoder_output["vd_mlp_dims"] = self.vd_mlp_dims
            encoder_output["raw_data"] = {
                "pts3d": pts_all,
                "opacities": raw_densities,
                "params": gaussian_parameters,
            }

        if self.cfg.estimating_pose:
            encoder_output["extrinsics"] = {"c": pred_extrinsics}
            if target is not None:
                encoder_output["extrinsics"]["cwt"] = pred_extrinsics_cwt

        if self.cfg.estimating_focal:
            encoder_output["intrinsics"] = {"c": pred_intrinsics}
            if target is not None:
                encoder_output["intrinsics"]["cwt"] = pred_intrinsics_cwt

        return encoder_output

    def process_pose(self, pose_enc, context_views):
        # pose_enc: (b v 9)
        b, v = pose_enc.shape[:2]
        poses = convert_pose_to_4x4(rearrange(pose_enc, "b v ... -> (b v) ..."))
        poses = rearrange(poses, "(b v) ... -> b v ...", b=b, v=v)

        if self.cfg.pose_make_baseline_1:
            a = poses[:, 0, :3, 3]  # [b, 3]
            b = poses[:, context_views - 1, :3, 3]  #  [b, 3]

            scale = (a - b).norm(dim=1, keepdim=True)  # [b, 1]

            poses[:, :, :3, 3] /= scale.unsqueeze(-1)

        if self.cfg.pose_make_relative:
            base_context_pose = poses[:,0] # [b, 4, 4]
            inv_base_context_pose = torch.inverse(base_context_pose)
            poses = inv_base_context_pose[:, None, :, :] @ poses # [b,1,4,4] @ [b,v,4,4]

        return poses      
    
    def process_depth(self, pose, pts3d):
        b, v, h, w, _ = pts3d.shape
        pts3d = rearrange(pts3d, "b v h w c -> (b v) (h w) c")
        pose = rearrange(pose, "b v ... -> (b v) ...")


        depths = depth_projector(pts3d, pose) # (bv, n, 1)
        depths = rearrange(depths, "(b v) (h w) 1 -> b v h w", b=b, v=v, h=h, w=w)
        return depths.contiguous()

    def get_relative_pose_input(self, pred_extrinsics, pts3d):
        """
                                        (World coordinate: Canonical space coordinate)
        pred_extrinsics: (b, 1, 4, 4) - World coordinate-to-Camera coordinate(W2C) -> X_cam = R*X_world + t
                Principle: The camera's optical center is at (0, 0, 0) in the camera coordinate.
                Derivation: 0 = RC + t -> RC = -t -> C = -R^T @ t
                Result: 'cam_centers' represents the exact (x, y, z) coordinates of the camera
        pts3d: (b, 1, n_total, 3) -> Gaussian center location (World coordinate)
        result_pose_input: (b, 1, n_total, 4) -> [direction_vector(3), log_scale_distance(1)]
            (n_total: v_cxt * n_pixels * n_surfaces)
        """
        b, v, n, _ = pts3d.shape

        # Extract Camera Center
        # At W2C matrix R|t, C = -R^T @ t
        R = pred_extrinsics[:, :, :3, :3] # (b, 1, 3, 3)
        t = pred_extrinsics[:, :, :3, 3:4] # (b, 1, 3, 1)
        cam_centers = -torch.matmul(R.transpose(-1, -2), t).squeeze(-1) # (b, 1, 3)

        # Compute relative direction vector and distance
        # cam_centers: (b, 1, 1, 3), pts3d: (b, 1, n, 3)
        vec = cam_centers.unsqueeze(2) - pts3d # (b, 1, n, 3)
        dist = torch.norm(vec, dim=-1, keepdim=True) + 1e-6

        # Set 4D input: Direction(3D) + log scale distance(1D)
        dir_vec = vec / dist # Normalized Direction (b, 1, n, 3)
        log_dist = torch.log(dist) # (b, 1, n, 1)

        pose_input = torch.cat([dir_vec, log_dist], dim=-1) # (b, 1, n, 4)

        return pose_input

    def compute_view_dependent_offset_from_features(self, features, input, dims):
        offsets = []
        for view_index, (view_features, view_input) in enumerate(
            zip(features, input.unbind(dim=1))
        ):
            head = (
                self.view_dependent_head
                if view_index == 0
                else self.view_dependent_head2
            )
            offsets.append(
                head.compute_offset_from_features(
                    view_features,
                    view_input,
                    checkpoint_tiles=self.training,
                )
            )
        return torch.stack(offsets, dim=1)

    def compute_view_dependent_offset(self, params, input, dims):
        return DPTViewHead.compute_offset(params, input, dims)

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
