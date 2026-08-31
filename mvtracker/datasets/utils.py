# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Optional, List

import numpy as np
import png
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from mvtracker.utils.basic import to_homogeneous, from_homogeneous


@dataclass(eq=False)
class Datapoint:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    segmentation: torch.Tensor  # B, S, 1, H, W

    # optional data
    videodepth: Optional[torch.Tensor] = None  # B, S, 1, H, W
    videodepthconf: Optional[torch.Tensor] = None  # B, S, 1, H, W
    feats: Optional[torch.Tensor] = None  # B, S, C, H_strided, W_strided
    valid: Optional[torch.Tensor] = None  # B, S, N
    seq_name: Optional[List[str]] = None  # B
    intrs: Optional[torch.Tensor] = torch.eye(3).unsqueeze(0)  # B, 3, 3

    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    query_points_3d: Optional[torch.Tensor] = None  # TapVID evaluation format

    trajectory: Optional[torch.Tensor] = None  # B, S, N, 2
    visibility: Optional[torch.Tensor] = None  # B, S, N
    trajectory_3d: Optional[torch.Tensor] = None  # B, S, 4, 4
    trajectory_category: Optional[torch.Tensor] = None  # B, S, 1
    extrs: Optional[torch.Tensor] = None  # B, S, 4, 4

    track_upscaling_factor: Optional[float] = 1.0

    novel_video: Optional[torch.Tensor] = None  # B, S, C, H, W
    novel_intrs: Optional[torch.Tensor] = torch.eye(3).unsqueeze(0)  # B, 3, 3
    novel_extrs: Optional[torch.Tensor] = None  # B, S, 4, 4

    # NuscTrack / explicit query-view metadata (per point, no batch dim before collate)
    query_points_view: Optional[torch.Tensor] = None  # N  (query camera index)
    is_background: Optional[torch.Tensor] = None  # N
    category_id: Optional[torch.Tensor] = None  # N


def collate_fn(batch):
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    videodepth = torch.stack([b.videodepth for b, _ in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    intrs = torch.stack([b.intrs for b, _ in batch], dim=0)

    videodepthconf = (
        torch.stack([b.videodepthconf for b, _ in batch], dim=0)
        if batch[0][0].videodepthconf is not None
        else None
    )
    feats = (
        torch.stack([b.feats for b, _ in batch], dim=0)
        if batch[0][0].feats is not None
        else None
    )
    trajectory = (
        torch.stack([b.trajectory for b, _ in batch], dim=0)
        if batch[0][0].trajectory is not None
        else None
    )
    valid = (
        torch.stack([b.valid for b, _ in batch], dim=0)
        if batch[0][0].valid is not None
        else None
    )
    visibility = (
        torch.stack([b.visibility for b, _ in batch], dim=0)
        if batch[0][0].visibility is not None
        else None
    )
    trajectory_3d = (
        torch.stack([b.trajectory_3d for b, _ in batch], dim=0)
        if batch[0][0].trajectory_3d is not None
        else None
    )
    extrs = (
        torch.stack([b.extrs for b, _ in batch], dim=0)
        if batch[0][0].extrs is not None
        else None
    )
    query_points = (
        torch.stack([b.query_points for b, _ in batch], dim=0)
        if batch[0][0].query_points is not None
        else None
    )
    query_points_3d = (
        torch.stack([b.query_points_3d for b, _ in batch], dim=0)
        if batch[0][0].query_points_3d is not None
        else None
    )
    query_points_view = (
        torch.stack([b.query_points_view for b, _ in batch], dim=0)
        if batch[0][0].query_points_view is not None
        else None
    )
    is_background = (
        torch.stack([b.is_background for b, _ in batch], dim=0)
        if batch[0][0].is_background is not None
        else None
    )
    category_id = (
        torch.stack([b.category_id for b, _ in batch], dim=0)
        if batch[0][0].category_id is not None
        else None
    )

    track_upscaling_factor = batch[0][0].track_upscaling_factor

    novel_video = None
    novel_intrs = None
    novel_extrs = None
    if batch[0][0].novel_video is not None:
        novel_video = torch.stack([b.novel_video for b, _ in batch], dim=0)
        novel_intrs = torch.stack([b.novel_intrs for b, _ in batch], dim=0)
        novel_extrs = torch.stack([b.novel_extrs for b, _ in batch], dim=0)

    return (
        Datapoint(
            video=video,
            videodepth=videodepth,
            videodepthconf=videodepthconf,
            feats=feats,
            segmentation=segmentation,
            trajectory=trajectory,
            trajectory_3d=trajectory_3d,
            visibility=visibility,
            valid=valid,
            seq_name=seq_name,
            intrs=intrs,
            extrs=extrs,
            query_points=query_points,
            query_points_3d=query_points_3d,
            query_points_view=query_points_view,
            is_background=is_background,
            category_id=category_id,
            track_upscaling_factor=track_upscaling_factor,
            novel_video=novel_video,
            novel_intrs=novel_intrs,
            novel_extrs=novel_extrs
        ),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


def read_json(filename: str) -> Any:
    with open(filename, "r") as fp:
        return json.load(fp)


def read_tiff(filename: str) -> np.ndarray:
    import imageio
    img = imageio.v2.imread(pathlib.Path(filename).read_bytes(), format="tiff")
    if img.ndim == 2:
        img = img[:, :, None]
    return img


def read_png(filename: str, rescale_range=None) -> np.ndarray:
    png_reader = png.Reader(bytes=pathlib.Path(filename).read_bytes())
    width, height, pngdata, info = png_reader.read()
    del png_reader

    bitdepth = info["bitdepth"]
    if bitdepth == 8:
        dtype = np.uint8
    elif bitdepth == 16:
        dtype = np.uint16
    else:
        raise NotImplementedError(f"Unsupported bitdepth: {bitdepth}")

    plane_count = info["planes"]
    pngdata = np.vstack(list(map(dtype, pngdata)))
    if rescale_range is not None:
        minv, maxv = rescale_range
        pngdata = pngdata / 2 ** bitdepth * (maxv - minv) + minv

    return pngdata.reshape((height, width, plane_count))

def transform_scene(
        transformation_scale: float = 1.0,
        transformation_rotation: torch.Tensor = torch.eye(3, dtype=torch.float32),
        transformation_translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),

        depth: torch.Tensor = None,  # [V,T,1,H,W]
        extrs: torch.Tensor = None,  # [V,T,3,4] world->cam

        query_points: torch.Tensor = None,  # [N,4] (t, x, y, z) in world
        traj3d_world: torch.Tensor = None,  # [T,N,3]
        traj2d_w_z: torch.Tensor = None,  # [V,T,N,3] (x_px, y_px, z_cam)
):
    """
    Make the world space `transformation_scale` larger, then rotate it by `transformation_rotation`,
    then translate it by `transformation_translation`. In other words, apply the following transformation:
    X_world' = transformation_translation + transformation_rotation @ (transformation_scale * X_world).

    Implemented as:
      - depth (z_cam) *= scale
      - extrinsics: scale translation by 'scale', then right-multiply by rigid inverse
      - query/world trajectories: scale then rigid
      - traj2d_w_z: only z scaled; (x,y) unchanged
    """
    is_rot_orthonormal = torch.allclose(
        transformation_rotation @ transformation_rotation.T,
        torch.eye(3, dtype=transformation_rotation.dtype, device=transformation_rotation.device),
        atol=1e-3,
    )
    assert is_rot_orthonormal, "The rotation matrix should be orthonormal."

    Rt = torch.eye(4, dtype=transformation_rotation.dtype, device=transformation_rotation.device)
    Rt[:3, :3] = transformation_rotation
    Rt[:3, 3] = transformation_translation

    # Transform depth
    if depth is not None:
        depth_trans = depth * transformation_scale
    else:
        depth_trans = None

    # Transform extrinsics
    if extrs is not None:
        n_views, n_frames, _, _ = extrs.shape
        assert extrs.shape == (n_views, n_frames, 3, 4)
        src_dtype = extrs.dtype
        extrs = extrs.type(Rt.dtype)
        extrs_trans_square = torch.eye(4, dtype=extrs.dtype, device=extrs.device).repeat(n_views, n_frames, 1, 1)
        extrs_trans_square[:, :, :3, :3] = extrs[:, :, :3, :3]
        extrs_trans_square[:, :, :3, 3] = extrs[:, :, :3, 3] * transformation_scale
        extrs_trans_square = torch.einsum('ABki,ij->ABkj', extrs_trans_square, torch.inverse(Rt))
        extrs_trans = extrs_trans_square[..., :3, :]
        extrs_trans = extrs_trans.type(src_dtype)
    else:
        extrs_trans = None

    # Transform query points
    if query_points is not None:
        n_tracks = query_points.shape[0]
        assert query_points.shape == (n_tracks, 4)
        src_dtype = query_points.dtype
        query_points = query_points.type(Rt.dtype)
        query_points_xyz_scaled_homo = to_homogeneous(query_points[..., 1:4] * transformation_scale)
        query_points_xyz_trans_homo = torch.einsum('ij,Nj->Ni', Rt, query_points_xyz_scaled_homo)
        query_points_xyz_trans = from_homogeneous(query_points_xyz_trans_homo)
        query_points_trans = torch.cat([query_points[..., :1], query_points_xyz_trans], dim=-1)
        query_points_trans = query_points_trans.type(src_dtype)
    else:
        query_points_trans = None

    # Transform 3D trajectories
    if traj3d_world is not None:
        n_frames, n_tracks, _ = traj3d_world.shape
        assert traj3d_world.shape == (n_frames, n_tracks, 3)
        src_dtype = traj3d_world.dtype
        traj3d_world = traj3d_world.type(Rt.dtype)
        traj3d_world_scaled_homo = to_homogeneous(traj3d_world * transformation_scale)
        traj3d_world_trans_homo = torch.einsum('ij,TNj->TNi', Rt, traj3d_world_scaled_homo)
        traj3d_world_trans = from_homogeneous(traj3d_world_trans_homo)
        traj3d_world_trans = traj3d_world_trans.type(src_dtype)
    else:
        traj3d_world_trans = None

    # Transform 2D+depth trajectories
    if traj2d_w_z is not None:
        n_views, n_frames, n_tracks, _ = traj2d_w_z.shape
        assert traj2d_w_z.shape == (n_views, n_frames, n_tracks, 3)
        traj2d_w_z_trans = traj2d_w_z.clone()
        traj2d_w_z_trans[:, :, :, 2] *= transformation_scale
    else:
        traj2d_w_z_trans = None

    return depth_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans


def add_camera_noise(intrs, extrs, noise_std_intr=0.01, noise_std_extr=0.001, rnd=np.random):
    """
    Add small Gaussian noise to intrinsic and extrinsic camera parameters.

    Args:
        intrs (np.ndarray): (V, T, 3, 3) intrinsic matrices.
        extrs (np.ndarray): (V, T, 3, 4) extrinsic matrices.
        noise_std_intr (float): Standard deviation of intrinsic matrix noise.
        noise_std_extr (float): Standard deviation of extrinsic matrix noise.
        rnd (module): Random number generator (e.g., np.random or torch).

    Returns:
        intrs (same type as input): Noisy intrinsic matrices.
        extrs (same type as input): Noisy extrinsic matrices.
    """
    V, T, _, _ = intrs.shape
    assert isinstance(intrs, np.ndarray)
    assert intrs.shape == (V, T, 3, 3)
    assert extrs.shape == (V, T, 3, 4)

    intrs, extrs = intrs.copy(), extrs.copy()

    intrs += rnd.normal(0, noise_std_intr, size=intrs.shape)
    extrs += rnd.normal(0, noise_std_extr, size=extrs.shape)

    return intrs, extrs


def aug_depth(depth, grid=(8, 8), scale=(0.7, 1.3), shift=(-0.1, 0.1),
              gn_kernel=(7, 7), gn_sigma=(2.0, 2.0), generator=None):
    """
    Augment depth for training.
    """
    B, T, H, W = depth.shape
    msk = (depth != 0)

    # fallback to global generator if none is provided
    gen = generator if generator is not None else torch.default_generator

    # generate scale and shift maps
    H_s, W_s = grid
    scale_map = (torch.rand(B, T, H_s, W_s, device=depth.device, generator=gen) * (scale[1] - scale[0]) + scale[0])
    shift_map = (torch.rand(B, T, H_s, W_s, device=depth.device, generator=gen) * (shift[1] - shift[0]) + shift[0])

    # scale and shift the depth map
    scale_map = F.interpolate(scale_map, (H, W), mode='bilinear', align_corners=True)
    shift_map = F.interpolate(shift_map, (H, W), mode='bilinear', align_corners=True)

    # local scale and shift the depth
    depth[msk] = (depth[msk] * scale_map[msk]) + shift_map[msk] * (depth[msk].mean())

    # gaussian blur
    depth = TF.gaussian_blur(depth, kernel_size=gn_kernel, sigma=gn_sigma)
    depth[~msk] = 0

    return depth


def align_umeyama(model, data, known_scale=False, yaw_only=False):
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0 / n * np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0 / sigma2 * np.trace(np.dot(D_svd, S))

    t = mu_M - s * np.dot(R, mu_D)

    return s, R, t


def get_camera_center(extr):
    R = extr[:, :3]
    t = extr[:, 3]
    return -R.T @ t


def apply_sim3_to_extrinsics(vggt_extrinsics, s, R_align, t_align):
    aligned_extrinsics = []
    R_inv = R_align.T
    t_inv = -R_inv @ t_align / s
    for extr in vggt_extrinsics:
        extr_h = np.eye(4)
        extr_h[:3, :4] = extr
        sim3_inv = np.eye(4)
        sim3_inv[:3, :3] = R_inv / s
        sim3_inv[:3, 3] = t_inv
        aligned = extr_h @ sim3_inv
        aligned_extrinsics.append(aligned[:3, :])
    return aligned_extrinsics


def get_best_yaw(C):
    """
    maximize trace(Rz(theta) * C)
    """
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


def rot_z(theta):
    R = rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R
