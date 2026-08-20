"""NuScenes / NuscTrack adapter for MVTracker.

Reads BEVTracker's clip pkls (`bev_traj_infos_*.pkl`) plus nuScenes keyframe
RGB / LiDAR / calibration, and returns MVTracker's `Datapoint`:

  video           [V, T, 3, H, W]
  videodepth      [V, T, 1, H, W]   (LiDAR projected + nearest completion)
  trajectory      [V, T, N, 3]      (pixel xy + camera z)
  trajectory_3d   [T, N, 3]         (per-frame ego, optionally scaled)
  visibility      [V, T, N]
  query_points_3d [N, 4]            (t, x, y, z) in the same ego frame
  extrs           [V, T, 3, 4]      world(ego)->camera
  intrs           [V, T, 3, 3]

Scene coordinates stay in the nuScenes ego frame (same as BEVTracker). A
uniform scale (default 0.25) is applied so the metric range is closer to
MV-Kubric; `track_upscaling_factor` converts predictions back to meters
for NuscTrack metrics (3D-AJ / APD / OA / MTE).
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import time
import warnings

import cv2
import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator
from torch.utils.data import Dataset

from mvtracker.datasets.utils import Datapoint, transform_scene

CAMERA_TYPES = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

TRAJ_DIR_ALIASES = {
    "omnidc": "processed_track_24_omnidc",
    "nearest": "processed_track_24_nearest",
    "allcate": "processed_track_24_all_cate",
}

NATIVE_H, NATIVE_W = 900, 1600
SLIM_CACHE_VERSION = "v1"


def _quat_wxyz_to_rot(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _cam2ego_to_extr(R_c2e: np.ndarray, t_c2e: np.ndarray) -> np.ndarray:
    """Camera-to-ego (R, t) -> 3x4 world(ego)-to-camera extrinsics."""
    R_e2c = R_c2e.T
    t_e2c = -R_e2c @ t_c2e
    extr = np.concatenate([R_e2c, t_e2c[:, None]], axis=1)
    return extr.astype(np.float32)


def _load_lidar_xyz(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    for dim in (5, 4, 6):
        if raw.size % dim == 0:
            return raw.reshape(-1, dim)[:, :3]
    raise ValueError(f"Unexpected lidar size {raw.size} in {path}")


def _complete_depth_nearest(uvz: np.ndarray, height: int, width: int) -> np.ndarray:
    """Sparse (u, v, z) -> dense depth by nearest-neighbour interpolation."""
    depth = np.zeros((height, width), dtype=np.float32)
    if uvz is None or len(uvz) == 0:
        return depth
    interpolator = NearestNDInterpolator(uvz[:, :2], uvz[:, 2])
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    filled = interpolator(xs, ys)
    if filled is None:
        return depth
    return np.nan_to_num(filled, nan=0.0).astype(np.float32)


def _project_lidar_to_depth(
    lidar_xyz: np.ndarray,
    lidar2ego_R: np.ndarray,
    lidar2ego_t: np.ndarray,
    extr: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if lidar_xyz.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)
    xyz_ego = lidar_xyz @ lidar2ego_R.T + lidar2ego_t[None]
    xyz_h = np.concatenate([xyz_ego, np.ones((xyz_ego.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_cam = (extr.astype(np.float32) @ xyz_h.T).T
    z = xyz_cam[:, 2]
    valid = z > 1.0
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.float32)
    u = K[0, 0] * xyz_cam[:, 0] / np.clip(z, 1e-6, None) + K[0, 2]
    v = K[1, 1] * xyz_cam[:, 1] / np.clip(z, 1e-6, None) + K[1, 2]
    in_img = valid & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    if not np.any(in_img):
        return np.zeros((height, width), dtype=np.float32)
    uvz = np.stack([u[in_img], v[in_img], z[in_img]], axis=1)
    return _complete_depth_nearest(uvz, height, width)


class NuScenesMultiViewDataset(Dataset):
    """Load NuscTrack clips into MVTracker's multi-view Datapoint format."""

    @staticmethod
    def from_name(dataset_name: str, dataset_root: str, cfg=None):
        """Parse names such as:

        - nuscenes-multiview
        - nuscenes-multiview-omnidc-val
        - nuscenes-multiview-nearest-train
        - nuscenes-multiview-omnidc-val-n384
        """
        non_parsed = dataset_name.replace("nuscenes-multiview", "", 1)

        traj_alias = "omnidc"
        match = re.match(r"-(omnidc|nearest|allcate)", non_parsed)
        if match is not None:
            traj_alias = match.group(1)
            non_parsed = non_parsed.replace(match.group(0), "", 1)

        split = "val"
        match = re.match(r"-(train|val)", non_parsed)
        if match is not None:
            split = match.group(1)
            non_parsed = non_parsed.replace(match.group(0), "", 1)

        traj_per_sample = 384
        match = re.match(r"-n(\d+)", non_parsed)
        if match is not None:
            traj_per_sample = int(match.group(1))
            non_parsed = non_parsed.replace(match.group(0), "", 1)

        use_cached_tracks = split == "val"
        if non_parsed.startswith("-cached"):
            use_cached_tracks = True
            non_parsed = non_parsed.replace("-cached", "", 1)
        elif non_parsed.startswith("-nocache"):
            use_cached_tracks = False
            non_parsed = non_parsed.replace("-nocache", "", 1)

        assert non_parsed == "", f"Unparsed part of the dataset name: {non_parsed}"

        nusc_root = dataset_root
        if not os.path.isdir(os.path.join(nusc_root, "samples")):
            fallback = "/home/yangyi/nuscenes"
            if os.path.isdir(os.path.join(fallback, "samples")):
                logging.warning(
                    "datasets.root=%s has no nuScenes samples/; using %s",
                    dataset_root,
                    fallback,
                )
                nusc_root = fallback

        image_size = (432, 768)
        if cfg is not None:
            nusc_cfg = getattr(cfg.datasets, "nuscenes", None)
            if nusc_cfg is not None and getattr(nusc_cfg, "image_size", None) is not None:
                image_size = tuple(nusc_cfg.image_size)

        return NuScenesMultiViewDataset(
            data_root=nusc_root,
            traj_dir=os.path.join(nusc_root, TRAJ_DIR_ALIASES[traj_alias]),
            split=split,
            traj_per_sample=traj_per_sample,
            image_size=image_size,
            use_cached_tracks=use_cached_tracks,
            seed=72 if split == "val" else None,
        )

    def __init__(
        self,
        data_root: str,
        traj_dir: str,
        split: str = "val",
        info_prefix: str = "tap3d-nuscenes",
        traj_per_sample: int = 384,
        image_size=(432, 768),
        scene_scale: float = 0.25,
        seed: int | None = 72,
        max_videos: int | None = None,
        use_cached_tracks: bool = True,
        cache_dir: str | None = None,
        depth_cache_dir: str | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.traj_dir = traj_dir
        self.split = split
        self.traj_per_sample = traj_per_sample
        self.image_h, self.image_w = int(image_size[0]), int(image_size[1])
        self.scene_scale = float(scene_scale)
        self.seed = seed
        self.use_cached_tracks = use_cached_tracks
        self.cache_dir = cache_dir or os.path.join(traj_dir, ".mvtracker_cache")
        self.depth_cache_dir = depth_cache_dir or os.path.join(
            data_root, "mvtracker_depth_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.depth_cache_dir, exist_ok=True)

        info_path = os.path.join(data_root, f"{info_prefix}_infos_{split}.pkl")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(
                f"Missing {info_path}. Generate it with BEVTracker "
                "tools/create_data_tap.py --gen-infos."
            )
        with open(info_path, "rb") as f:
            packed = pickle.load(f)
        self.scene_infos = {info["name"]: info for info in packed["infos"]}

        clips = []
        for scene_name in sorted(self.scene_infos.keys()):
            for suffix, is_former in (("former", True), ("later", False)):
                pkl_path = os.path.join(traj_dir, f"bev_traj_infos_{scene_name}_{suffix}.pkl")
                if os.path.isfile(pkl_path):
                    clips.append(
                        {
                            "scene_name": scene_name,
                            "is_former": is_former,
                            "pkl_path": pkl_path,
                            "seq_name": f"{scene_name}_{suffix}",
                        }
                    )
        if max_videos is not None:
            clips = clips[:max_videos]
        if not clips:
            raise FileNotFoundError(
                f"No bev_traj_infos_*.pkl clips found in {traj_dir} "
                f"for split={split} scenes {sorted(self.scene_infos)}"
            )
        self.clips = clips
        self.getitem_calls = 0
        logging.info(
            "NuScenesMultiViewDataset split=%s n_clips=%d traj_dir=%s size=%dx%d scale=%.3f",
            split,
            len(self.clips),
            traj_dir,
            self.image_h,
            self.image_w,
            self.scene_scale,
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        start = time.time()
        sample = self._getitem_helper(index)
        self.getitem_calls += 1
        if self.getitem_calls < 8:
            logging.info(
                "Loading %s took %.2fs (call %d)",
                self.clips[index]["seq_name"],
                time.time() - start,
                self.getitem_calls,
            )
        return sample, True

    def _clip_slice(self, scene_len: int, is_former: bool, clip_len: int):
        if is_former:
            return 0, min(clip_len, scene_len)
        start = max(0, scene_len - clip_len)
        return start, scene_len

    def _ensure_slim_tracks(self, clip) -> dict:
        cache_path = os.path.join(
            self.cache_dir, f"{clip['seq_name']}--tracks-{SLIM_CACHE_VERSION}.npz"
        )
        if os.path.isfile(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            return {k: data[k] for k in data.files}

        logging.info("Converting %s to slim track cache (slow, once)", clip["pkl_path"])
        with open(clip["pkl_path"], "rb") as f:
            payload = pickle.load(f)
        info = payload["traj_info"]
        traj = np.asarray(info["traj"], dtype=np.float32)
        valid = np.asarray(info["valid"], dtype=bool)
        n_tracks, n_frames, _ = traj.shape
        vis = np.zeros((len(CAMERA_TYPES), n_frames, n_tracks), dtype=bool)
        tracks2d = np.zeros((len(CAMERA_TYPES), n_frames, n_tracks, 2), dtype=np.float32)
        for n in range(n_tracks):
            for v, cam in enumerate(CAMERA_TYPES):
                vis[v, :, n] = np.asarray(info["visibility"][n][cam], dtype=bool)
                tracks2d[v, :, n] = np.asarray(info["tracks2d"][n][cam], dtype=np.float32)
        query_t = np.array([qp["coord"][0] for qp in info["query_point"]], dtype=np.int32)
        query_xy = np.stack(
            [np.asarray(qp["coord"][1:], dtype=np.float32) for qp in info["query_point"]],
            axis=0,
        )
        cam_to_idx = {c: i for i, c in enumerate(CAMERA_TYPES)}
        query_cam = np.array(
            [cam_to_idx[qp["camera"]] for qp in info["query_point"]], dtype=np.int32
        )
        # Camera extrinsics are identical across tracks; take the first.
        cam2ego_rot = np.stack(
            [np.asarray(info["cam2ego_rot"][0][cam], dtype=np.float32) for cam in CAMERA_TYPES],
            axis=0,
        )
        cam2ego_trans = np.stack(
            [np.asarray(info["cam2ego_trans"][0][cam], dtype=np.float32) for cam in CAMERA_TYPES],
            axis=0,
        )
        packed = dict(
            traj=traj,
            valid=valid,
            vis=vis,
            tracks2d=tracks2d,
            query_t=query_t,
            query_xy=query_xy,
            query_cam=query_cam,
            cam2ego_rot=cam2ego_rot,
            cam2ego_trans=cam2ego_trans,
        )
        np.savez_compressed(cache_path, **packed)
        logging.info("Wrote slim track cache %s", cache_path)
        return packed

    def _load_rgb_intr_lidar(self, scene_info, start: int, end: int):
        n_frames = end - start
        n_views = len(CAMERA_TYPES)
        rgbs = np.zeros((n_views, n_frames, 3, self.image_h, self.image_w), dtype=np.float32)
        intrs = np.zeros((n_views, n_frames, 3, 3), dtype=np.float32)
        lidar_xyz = []
        lidar2ego_R = np.zeros((n_frames, 3, 3), dtype=np.float32)
        lidar2ego_t = np.zeros((n_frames, 3), dtype=np.float32)
        sx = self.image_w / float(NATIVE_W)
        sy = self.image_h / float(NATIVE_H)

        for t, sample_i in enumerate(range(start, end)):
            sample = scene_info["sample_infos"][sample_i]
            lidar2ego_R[t] = _quat_wxyz_to_rot(sample["lidar2ego_rotation"])
            lidar2ego_t[t] = np.asarray(sample["lidar2ego_translation"], dtype=np.float32)
            lidar_path = sample["lidar_path"]
            if os.path.isfile(lidar_path):
                lidar_xyz.append(_load_lidar_xyz(lidar_path))
            else:
                warnings.warn(f"Missing lidar {lidar_path}; depth will be empty for t={t}")
                lidar_xyz.append(np.zeros((0, 3), dtype=np.float32))
            for v, cam in enumerate(CAMERA_TYPES):
                cam_info = sample["cams"][cam]
                img_bgr = cv2.imread(cam_info["data_path"], cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise FileNotFoundError(cam_info["data_path"])
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[0] != self.image_h or img_rgb.shape[1] != self.image_w:
                    img_rgb = cv2.resize(
                        img_rgb, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA
                    )
                rgbs[v, t] = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32)
                K = np.asarray(cam_info["cam_intrinsic"], dtype=np.float32).copy()
                K[0, 0] *= sx
                K[0, 2] *= sx
                K[1, 1] *= sy
                K[1, 2] *= sy
                intrs[v, t] = K
        return rgbs, intrs, lidar_xyz, lidar2ego_R, lidar2ego_t

    def _load_or_build_depths(
        self,
        clip,
        extrs: np.ndarray,
        intrs: np.ndarray,
        lidar_xyz,
        lidar2ego_R: np.ndarray,
        lidar2ego_t: np.ndarray,
    ) -> np.ndarray:
        n_views, n_frames = extrs.shape[:2]
        cache_path = os.path.join(
            self.depth_cache_dir,
            f"{clip['seq_name']}--{self.image_h}x{self.image_w}.npz",
        )
        if os.path.isfile(cache_path):
            return np.load(cache_path)["depths"].astype(np.float32)

        logging.info("Building LiDAR depths for %s (cached afterwards)", clip["seq_name"])
        depths = np.zeros((n_views, n_frames, 1, self.image_h, self.image_w), dtype=np.float32)
        for t in range(n_frames):
            for v in range(n_views):
                depths[v, t, 0] = _project_lidar_to_depth(
                    lidar_xyz[t],
                    lidar2ego_R[t],
                    lidar2ego_t[t],
                    extrs[v, t],
                    intrs[v, t],
                    self.image_h,
                    self.image_w,
                )
        np.savez_compressed(cache_path, depths=depths.astype(np.float16))
        return depths

    def _sample_tracks(self, tracks: dict, generator: torch.Generator):
        vis = torch.from_numpy(tracks["vis"])  # [V, T, N]
        valid = torch.from_numpy(tracks["valid"])  # [N, T]
        vis_any = vis.any(dim=0)  # [T, N]
        vis_any = vis_any & valid.transpose(0, 1)
        visible_enough = vis_any.sum(dim=0) >= 2
        valid_idx = visible_enough.nonzero(as_tuple=False)[:, 0]
        if len(valid_idx) == 0:
            raise RuntimeError("No tracks with >=2 visible valid frames")

        n_want = self.traj_per_sample if self.traj_per_sample is not None else len(valid_idx)
        perm = torch.randperm(len(valid_idx), generator=generator)
        if len(valid_idx) >= n_want:
            chosen = valid_idx[perm[:n_want]]
        else:
            extra = perm[torch.randint(0, len(valid_idx), (n_want - len(valid_idx),), generator=generator)]
            chosen = torch.cat([valid_idx, valid_idx[extra]], dim=0)

        idx = chosen.numpy()
        traj3d = torch.from_numpy(tracks["traj"][idx]).permute(1, 0, 2).float()  # [T, N, 3]
        vis_s = torch.from_numpy(tracks["vis"][:, :, idx]).bool()
        valid_s = torch.from_numpy(tracks["valid"][idx]).transpose(0, 1).bool()  # [T, N]
        tracks2d = torch.from_numpy(tracks["tracks2d"][:, :, idx]).float()
        query_t = torch.from_numpy(tracks["query_t"][idx]).long()
        vis_s = vis_s & valid_s[None]
        return traj3d, vis_s, valid_s, tracks2d, query_t, idx

    def _getitem_helper(self, index: int) -> Datapoint:
        if self.seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        else:
            seed = self.seed
        rnd = torch.Generator().manual_seed(int(seed) + index)

        clip = self.clips[index]
        scene_info = self.scene_infos[clip["scene_name"]]
        scene_len = int(scene_info["scene_len"])
        tracks = self._ensure_slim_tracks(clip)
        n_frames_traj = int(tracks["traj"].shape[1])
        start, end = self._clip_slice(scene_len, clip["is_former"], n_frames_traj)
        if end - start != n_frames_traj:
            warnings.warn(
                f"{clip['seq_name']}: scene_len={scene_len} clip_len={n_frames_traj} "
                f"slice=[{start},{end}) resized to match traj"
            )
            end = start + n_frames_traj

        n_views = len(CAMERA_TYPES)
        n_frames = n_frames_traj

        cam2ego_rot = tracks["cam2ego_rot"]
        cam2ego_trans = tracks["cam2ego_trans"]
        extrs = np.zeros((n_views, n_frames, 3, 4), dtype=np.float32)
        for v in range(n_views):
            for t in range(n_frames):
                extrs[v, t] = _cam2ego_to_extr(cam2ego_rot[v, t], cam2ego_trans[v, t])

        rgbs, intrs, lidar_xyz, lidar2ego_R, lidar2ego_t = self._load_rgb_intr_lidar(
            scene_info, start, end
        )
        depths = self._load_or_build_depths(
            clip, extrs, intrs, lidar_xyz, lidar2ego_R, lidar2ego_t
        )

        sample_cache = os.path.join(
            self.cache_dir,
            f"{clip['seq_name']}--sample-seed{seed}-n{self.traj_per_sample}.npz",
        )
        if self.use_cached_tracks and os.path.isfile(sample_cache):
            cached = np.load(sample_cache)
            traj3d = torch.from_numpy(cached["traj3d"]).float()
            vis = torch.from_numpy(cached["vis"]).bool()
            valids = torch.from_numpy(cached["valids"]).bool()
            tracks2d = torch.from_numpy(cached["tracks2d"]).float()
            query_t = torch.from_numpy(cached["query_t"]).long()
        else:
            traj3d, vis, valids, tracks2d, query_t, _ = self._sample_tracks(tracks, rnd)
            if self.use_cached_tracks:
                np.savez_compressed(
                    sample_cache,
                    traj3d=traj3d.numpy(),
                    vis=vis.numpy(),
                    valids=valids.numpy(),
                    tracks2d=tracks2d.numpy(),
                    query_t=query_t.numpy(),
                )

        n_tracks = traj3d.shape[1]
        sx = self.image_w / float(NATIVE_W)
        sy = self.image_h / float(NATIVE_H)
        tracks2d = tracks2d.clone()
        tracks2d[..., 0] *= sx
        tracks2d[..., 1] *= sy

        # Camera-space z via world-to-camera extrinsics.
        traj_h = torch.cat([traj3d, torch.ones_like(traj3d[..., :1])], dim=-1)
        extrs_t = torch.from_numpy(extrs)
        xyz_cam = torch.einsum("vtij,tnj->vtni", extrs_t, traj_h)
        cam_z = xyz_cam[..., 2]
        traj2d_w_z = torch.cat([tracks2d, cam_z[..., None]], dim=-1)

        query_xyz = traj3d[query_t, torch.arange(n_tracks)]
        query_points = torch.cat([query_t[:, None].float(), query_xyz], dim=1)

        vis = vis.clone()
        for n in range(n_tracks):
            t0 = int(query_t[n].item())
            vis[:, :t0, n] = False

        rgbs_t = torch.from_numpy(rgbs).float()
        depths_t = torch.from_numpy(depths).float()
        intrs_t = torch.from_numpy(intrs).float()

        scale = self.scene_scale
        rot = torch.eye(3, dtype=torch.float32)
        translate = torch.zeros(3, dtype=torch.float32)
        (
            depths_trans,
            extrs_trans,
            query_points_trans,
            traj3d_trans,
            traj2d_w_z_trans,
        ) = transform_scene(scale, rot, translate, depths_t, extrs_t, query_points, traj3d, traj2d_w_z)

        segs = torch.ones((n_frames, 1, self.image_h, self.image_w))
        return Datapoint(
            video=rgbs_t,
            videodepth=depths_trans,
            feats=None,
            segmentation=segs,
            trajectory=traj2d_w_z_trans,
            trajectory_3d=traj3d_trans,
            trajectory_category=None,
            visibility=vis,
            valid=valids,
            seq_name=clip["seq_name"],
            intrs=intrs_t,
            extrs=extrs_trans,
            query_points=None,
            query_points_3d=query_points_trans,
            track_upscaling_factor=1.0 / scale,
        )
