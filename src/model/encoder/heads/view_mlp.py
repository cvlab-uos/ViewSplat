from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def compute_view_mlp_offset(
    view_mlp_params: Tensor,
    pose_input: Tensor,
    mlp_dims: Sequence[int],
) -> Tensor:
    """Evaluate the per-pixel MLP represented by view_mlp_params.

    Args:
        view_mlp_params: [B, V, C, H, W] dynamic MLP parameters.
        pose_input: [B, V, H*W, in_dim] inputs to the dynamic MLP.
    """
    batch, num_views, _, height, width = view_mlp_params.shape
    in_dim, hidden_dim, out_dim = mlp_dims

    flat_params = rearrange(
        view_mlp_params,
        "b v c h w -> (b v h w) c",
    )
    flat_input = rearrange(pose_input, "b v n d -> (b v n) d 1")

    first_weight_end = in_dim * hidden_dim
    first_bias_end = first_weight_end + hidden_dim
    second_weight_end = first_bias_end + hidden_dim * out_dim

    weight1 = flat_params[:, :first_weight_end].view(-1, hidden_dim, in_dim)
    bias1 = flat_params[:, first_weight_end:first_bias_end].view(
        -1, hidden_dim, 1
    )
    hidden = F.relu(torch.bmm(weight1, flat_input) + bias1)

    weight2 = flat_params[:, first_bias_end:second_weight_end].view(
        -1, out_dim, hidden_dim
    )
    bias2 = flat_params[:, second_weight_end:].view(-1, out_dim, 1)
    offset = torch.bmm(weight2, hidden) + bias2

    return rearrange(
        offset.squeeze(-1),
        "(b v n) d -> b v n d",
        b=batch,
        v=num_views,
        n=height * width,
    )


def compute_view_mlp_offset_tiled(
    features: Tensor,
    pose_input: Tensor,
    projection: nn.Conv2d,
    mlp_dims: Sequence[int],
    *,
    tile_rows: int = 16,
    checkpoint_tiles: bool = True,
) -> Tensor:
    """Project and evaluate a per-pixel MLP without materializing its full map.

    features is the activation immediately before the 1x1 convolution that
    emits all dynamic MLP parameters. Each row tile is projected and consumed
    immediately. During training, checkpointing retains only the compact input
    feature and recomputes the large parameter tile in backward.

    Args:
        features: [B, C, H, W] pre-projection features for one view.
        pose_input: [B, H*W, in_dim] inputs for the same view.
    """
    batch, _, height, width = features.shape
    pose_map = rearrange(
        pose_input,
        "b (h w) d -> b h w d",
        h=height,
        w=width,
    )

    def project_and_evaluate(feature_tile: Tensor, pose_tile: Tensor) -> Tensor:
        params = projection(feature_tile)
        params = rearrange(params, "b c h w -> b () c h w")
        inputs = rearrange(pose_tile, "b h w d -> b () (h w) d")
        offsets = compute_view_mlp_offset(params, inputs, mlp_dims)
        return offsets[:, 0]

    outputs = []
    use_checkpoint = checkpoint_tiles and torch.is_grad_enabled() and (
        features.requires_grad
        or any(parameter.requires_grad for parameter in projection.parameters())
    )
    for row_start in range(0, height, tile_rows):
        row_end = min(row_start + tile_rows, height)
        feature_tile = features[:, :, row_start:row_end]
        pose_tile = pose_map[:, row_start:row_end]
        if use_checkpoint:
            tile_output = checkpoint(
                project_and_evaluate,
                feature_tile,
                pose_tile,
                use_reentrant=False,
            )
        else:
            tile_output = project_and_evaluate(feature_tile, pose_tile)
        outputs.append(tile_output)

    return torch.cat(outputs, dim=1).view(batch, height * width, -1)
