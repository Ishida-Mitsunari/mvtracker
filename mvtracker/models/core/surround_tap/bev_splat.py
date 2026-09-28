"""Coarse BEV feature splat + patch sampling for SurroundTAP.

Projects multi-view RGB-D encoder features onto an ego-XY grid (mean pool),
then samples a local BEV patch around each track estimate.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd


def _bev_grid_size(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: float,
) -> Tuple[int, int]:
    hb = int(round((x_max - x_min) / resolution))
    wb = int(round((y_max - y_min) / resolution))
    return max(hb, 1), max(wb, 1)


def splat_pointcloud_to_bev(
    xyz: torch.Tensor,
    fvec: torch.Tensor,
    valid: torch.Tensor,
    B: int,
    S: int,
    x_min: float = -51.2,
    x_max: float = 51.2,
    y_min: float = -51.2,
    y_max: float = 51.2,
    resolution: float = 1.6,
) -> torch.Tensor:
    """Mean-pool already-lifted feature points onto an ego-XY BEV.

    Args:
        xyz / fvec / valid: (B*S, P, 3/C/1-or-bool) from ``init_pointcloud_from_rgbd``.
        B, S: batch and window length used to reshape the output.

    Returns:
        bev: (B, S, C, Hb, Wb)
    """
    bs, n_pts, C = fvec.shape
    assert xyz.shape == (bs, n_pts, 3)
    assert valid.shape[:2] == (bs, n_pts)
    assert bs == B * S
    Hb, Wb = _bev_grid_size(x_min, x_max, y_min, y_max, resolution)
    device = fvec.device
    dtype = fvec.dtype

    x = xyz[..., 0]
    y = xyz[..., 1]
    ix = ((x - x_min) / resolution).long()
    iy = ((y - y_min) / resolution).long()
    valid_flat = valid.reshape(bs, n_pts).bool()
    in_bounds = (ix >= 0) & (ix < Hb) & (iy >= 0) & (iy < Wb) & valid_flat
    flat_idx = (ix * Wb + iy).clamp(0, Hb * Wb - 1)

    feat_sum = fvec.new_zeros(bs, Hb * Wb, C)
    count = fvec.new_zeros(bs, Hb * Wb, 1)
    ones = torch.ones(bs, n_pts, 1, device=device, dtype=dtype)

    batch_arange = torch.arange(bs, device=device)[:, None].expand(bs, n_pts)
    mask = in_bounds
    if mask.any():
        b_i = batch_arange[mask]
        c_i = flat_idx[mask]
        f_i = fvec[mask]
        o_i = ones[mask]
        lin = b_i * (Hb * Wb) + c_i
        feat_sum_flat = feat_sum.reshape(bs * Hb * Wb, C)
        count_flat = count.reshape(bs * Hb * Wb, 1)
        feat_sum_flat.index_add_(0, lin, f_i)
        count_flat.index_add_(0, lin, o_i)
        feat_sum = feat_sum_flat.view(bs, Hb * Wb, C)
        count = count_flat.view(bs, Hb * Wb, 1)

    bev = feat_sum / count.clamp_min(1.0)
    return bev.view(B, S, Hb, Wb, C).permute(0, 1, 4, 2, 3).contiguous()


def splat_rgbd_to_bev(
    fmaps: torch.Tensor,
    depths: torch.Tensor,
    intrs: torch.Tensor,
    extrs: torch.Tensor,
    stride: int,
    x_min: float = -51.2,
    x_max: float = 51.2,
    y_min: float = -51.2,
    y_max: float = 51.2,
    resolution: float = 1.6,
) -> torch.Tensor:
    """Mean-pool multi-view feature points onto a coarse ego-XY BEV.

    Args:
        fmaps: (B, V, S, C, H, W) image features at encoder stride.
        depths: (B, V, S, 1, H, W) metric depth aligned with fmaps.
        intrs / extrs: camera matrices for the same window.

    Returns:
        bev: (B, S, C, Hb, Wb) where Hb/Wb follow ``resolution`` over XY.
    """
    B, V, S, C, H, W = fmaps.shape
    assert depths.shape == (B, V, S, 1, H, W)

    xyz, fvec, valid = init_pointcloud_from_rgbd(
        fmaps=fmaps,
        depths=depths,
        intrs=intrs,
        extrs=extrs,
        stride=stride,
        level=0,
        return_validity_mask=True,
    )
    return splat_pointcloud_to_bev(
        xyz,
        fvec,
        valid,
        B=B,
        S=S,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=resolution,
    )


def sample_bev_patch_corr(
    bev: torch.Tensor,
    coords_xyz: torch.Tensor,
    radius: int,
    x_min: float = -51.2,
    x_max: float = 51.2,
    y_min: float = -51.2,
    y_max: float = 51.2,
    resolution: float = 1.6,
) -> torch.Tensor:
    """Sample a (2r+1)^2 BEV neighborhood around each 3D track estimate.

    Args:
        bev: (B, S, C, Hb, Wb)
        coords_xyz: (B, S, N, 3) current track estimates (ego/world XY used).

    Returns:
        corr: (B, S, N, (2r+1)^2 * C) flattened local BEV features.
    """
    B, S, C, Hb, Wb = bev.shape
    assert coords_xyz.shape[:3] == (B, S, coords_xyz.shape[2])
    N = coords_xyz.shape[2]
    r = int(radius)
    k = 2 * r + 1

    # Pixel centers in BEV index space
    fx = (coords_xyz[..., 0] - x_min) / resolution  # (B,S,N)
    fy = (coords_xyz[..., 1] - y_min) / resolution

    # Offsets
    oy = torch.arange(-r, r + 1, device=bev.device, dtype=bev.dtype)
    ox = torch.arange(-r, r + 1, device=bev.device, dtype=bev.dtype)
    oy, ox = torch.meshgrid(oy, ox, indexing="ij")  # (k,k)
    ox = ox.reshape(1, 1, 1, k * k)
    oy = oy.reshape(1, 1, 1, k * k)

    sample_x = fx[..., None] + ox  # (B,S,N,k^2)
    sample_y = fy[..., None] + oy

    # grid_sample expects x→W, y→H in [-1,1]
    gx = sample_x / max(Wb - 1, 1) * 2 - 1
    gy = sample_y / max(Hb - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)  # (B,S,N,k^2,2)

    bev_flat = bev.reshape(B * S, C, Hb, Wb)
    grid_flat = grid.reshape(B * S, N, k * k, 2)
    sampled = F.grid_sample(
        bev_flat,
        grid_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B*S, C, N, k^2)
    sampled = sampled.permute(0, 2, 3, 1).reshape(B, S, N, k * k * C)
    return sampled
