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

        # MLP structure definition: [Pose_In, Hidden, GS_Offset]
        # # Assuming all parameters include gaussian centers(pts3d): mean, scale, rotation, color, opacity -> 86
        self.vd_mlp_dims = [cfg.vd_mlp_in_dim, cfg.vd_mlp_hidden_dim, self.raw_gs_dim + 3]

        self.view_dependent_head = head_factory('dpt', 'view_mlp', self.backbone, mlp_dims=self.vd_mlp_dims)
        self.view_dependent_head2 = head_factory('dpt', 'view_mlp', self.backbone, mlp_dims=self.vd_mlp_dims)
   
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
    ) :
        device = context["image"].device
        b, v_cxt, _, h, w = context["image"].shape


        if target is not None:
            v_tgt = target["image"].shape[1]
            context_target = {
                "image": normalize_image(torch.cat([context["image"], target["image"]], dim=1)),
                "intrinsics": torch.cat([context["intrinsics"], target["intrinsics"]], dim=1),
            }
            # Encode the context and target images.
            out = self.backbone(context_target, target_num_views=v_tgt)
            dec_feat, dec_feat_w_tgt = out['dec_feat'], out['dec_feat_w_tgt']
        else:
            v_tgt = 0
            context_input = {
                "image": normalize_image(context["image"]),
                "intrinsics": context["intrinsics"],
            }
            # Encode the context images.
            out = self.backbone(context_input)
            dec_feat = out['dec_feat']

        shape, images = out["shape"], out["images"]
        
        with torch.amp.autocast('cuda', enabled=False):
            all_mean_res, all_other_params = [], []
            if self.cfg.estimating_pose:
                all_pose_params, all_pose_params_cwt = [], []

            
            # Pts3d head (context only)
            res1 = self._downstream_head(1, [tok[:, 0].float() for tok in dec_feat], shape[:, 0])
            all_mean_res.append(res1)
            for i in range(1, v_cxt):
                res2 = self._downstream_head(2, [tok[:, i].float() for tok in dec_feat], shape[:, i])
                all_mean_res.append(res2)
            
            # Gaussian parameter head (context only)
            if 'dpt' in self.gs_params_head_type:
                GS_res1 = self.gaussian_param_head([tok[:, 0].float() for tok in dec_feat], images[:, 0, :3], shape[0, 0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1)
                for i in range(1, v_cxt):
                    GS_res2 = self.gaussian_param_head2([tok[:, i].float() for tok in dec_feat], images[:, i, :3], shape[0, i].cpu().tolist())
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
                    all_other_params.append(GS_res2)
            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")
            
            # View dependent head (context only)
            all_vd_params = []
            if self.cfg.use_view_dependent_head:
                vd_params1 = self.view_dependent_head([tok[:, 0].float() for tok in dec_feat], images[:, 0, :3], shape[0, 0].cpu().tolist())
                # vd_params1 shape: [B, total_params, H, W]
                all_vd_params.append(rearrange(vd_params1, "b c h w -> b (h w) c"))
                for i in range(1, v_cxt):
                    vd_params2 = self.view_dependent_head2([tok[:, i].float() for tok in dec_feat], images[:, i, :3], shape[0, i].cpu().tolist())
                    all_vd_params.append(rearrange(vd_params2, "b c h w -> b (h w) c"))

            # Pose head
            if self.cfg.estimating_pose:
                # Context views
                pose_res1 = self.pose_head([tok[:, 0].float() for tok in dec_feat], shape[0, 0].cpu().tolist()) # (16, 9)
                all_pose_params.append(pose_res1)
                for i in range(1, v_cxt):
                    pose_res2 = self.pose_head2([tok[:, i].float() for tok in dec_feat], shape[0, i].cpu().tolist()) # (16, 9)
                    all_pose_params.append(pose_res2)

                # Context + target views
                if target is not None:
                    pose_res1 = self.pose_head([tok[:, 0].float() for tok in dec_feat_w_tgt], shape[0, 0].cpu().tolist()) # (16, 9)
                    all_pose_params_cwt.append(pose_res1)
                    for i in range(1, v_cxt + v_tgt):
                        pose_res2 = self.pose_head2([tok[:, i].float() for tok in dec_feat_w_tgt], shape[0, i].cpu().tolist()) # (16, 9)
                        all_pose_params_cwt.append(pose_res2)
            

            
            
        gaussians = torch.stack(all_other_params, dim=1) # [b, v, 65536, 83]
        
        if self.cfg.estimating_pose:
            poses_enc = torch.stack(all_pose_params, dim=1) # (b, v 9)
            pred_extrinsics = self.process_pose(poses_enc, v_cxt) # (b, v, 4, 4)

            if target is not None:
                poses_enc_cwt = torch.stack(all_pose_params_cwt, dim=1) # (b, v + v2, 9)
                pred_extrinsics_cwt = self.process_pose(poses_enc_cwt, v_cxt) # (b, v + v2, 4, 4)



        pts_all = [all_mean_res_i['pts3d'] for all_mean_res_i in all_mean_res]
        pts_all = torch.stack(pts_all, dim=1) # [b, v, h, w, 3]
        pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz")
        context_extrinsics = pred_extrinsics[:, :v_cxt] if self.cfg.estimating_pose else context["extrinsics"]
        depths_per_view = self.process_depth(context_extrinsics, rearrange(pts_all, "b v (h w) xyz -> b v h w xyz", h=h, w=w)) # depth for each cam, (b, v, h, w)
            

        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces) # for cfg.num_surfaces
        raw_densities = gaussians[..., 0]
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        gaussian_parameters = gaussians[..., 1:]
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces [B, V_cxt, N, S, 3]


        # Convert the features and depths into Gaussians.
        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussian_parameters, "b v r srf c -> b v r srf () c"),
        )


        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view

            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            ) 
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ) 

            
        if self.cfg.estimating_focal:
            intrinsics = estimate_intrinsics(rearrange(gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w).squeeze(-2), h, w)
            pred_intrinsics = intrinsics.unsqueeze(1).repeat(1, v_cxt, 1, 1)
            pred_intrinsics_cwt = intrinsics.unsqueeze(1).repeat(1, v_cxt+v_tgt, 1, 1)


        encoder_output = dict()


        encoder_output["gaussians"] = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.rotations,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                gaussians.scales,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            )
        )

        if self.cfg.use_view_dependent_head:
            encoder_output["vd_mlp_params"] = torch.stack(all_vd_params, dim=1) # [B, V_cxt, N, total_params]
            encoder_output["vd_mlp_dims"] = self.vd_mlp_dims

            # Raw data for refinement (Base value before add offsets)
            encoder_output["raw_data"] = {
                "pts3d": pts_all,
                "opacities": raw_densities,
                "params": gaussian_parameters
            }

        if self.cfg.estimating_pose:
            encoder_output['extrinsics'] = {"c": pred_extrinsics}
            if target is not None:
                encoder_output['extrinsics']['cwt'] = pred_extrinsics_cwt

        if self.cfg.estimating_focal:
            encoder_output['intrinsics'] = {"c": pred_intrinsics}
            if target is not None:
                encoder_output['intrinsics']['cwt'] = pred_intrinsics_cwt

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
