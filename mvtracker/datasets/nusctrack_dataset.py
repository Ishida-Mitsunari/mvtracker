"""NuscTrack clips + UniDepthV2 depths for MVTracker.

Reads:
  - RGB keyframes from nuScenes ``samples/CAM_*``
  - metric camera-Z dumps under ``{nusc_root}/unidepthv2/`` (same relative path, ``.npy``)
  - TAP clips ``processed_track_24_omnidc/bev_traj_infos_{scene}_{former|later}.pkl``

Queries are the NuscTrack 2D points ``(cam, t, x, y)`` in original pixels, lifted
with UniDepth Z and the original K / cam2ego. GT 3D is **not** used as the query.
Ego is treated as world; scene normalization is disabled (``track_upscaling_factor=1``).

Incomplete UniDepth coverage is skipped until the dump finishes.
"""
from __future__ import annotations

import hashlib
import pickle
import re
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mvtracker.datasets.utils import Datapoint

CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
CAM_TO_IDX = {name: i for i, name in enumerate(CAMERAS)}

DEFAULT_NUSC_ROOT = "/share/tgp/yangyi/nuscenes"
CLIP_NAME_RE = re.compile(r"bev_traj_infos_(scene-\d+)_(former|later)$")

CATEGORY_IDX = {
    0: "noise",
    1: "animal",
    2: "human.pedestrian.adult",
    3: "human.pedestrian.child",
    4: "human.pedestrian.construction_worker",
    5: "human.pedestrian.personal_mobility",
    6: "human.pedestrian.police_officer",
    7: "human.pedestrian.stroller",
    8: "human.pedestrian.wheelchair",
    9: "movable_object.barrier",
    10: "movable_object.debris",
    11: "movable_object.pushable_pullable",
    12: "movable_object.trafficcone",
    13: "static_object.bicycle_rack",
    14: "vehicle.bicycle",
    15: "vehicle.bus.bendy",
    16: "vehicle.bus.rigid",
    17: "vehicle.car",
    18: "vehicle.construction",
    19: "vehicle.emergency.ambulance",
    20: "vehicle.emergency.police",
    21: "vehicle.motorcycle",
    22: "vehicle.trailer",
    23: "vehicle.truck",
    24: "flat.driveable_surface",
    25: "flat.other",
    26: "flat.sidewalk",
    27: "flat.terrain",
    28: "static.manmade",
    29: "static.other",
    30: "static.vegetation",
    31: "vehicle.ego",
}
CATEGORY_NAME_TO_ID = {name: idx for idx, name in CATEGORY_IDX.items()}


def _stable_seed_int(*parts) -> int:
    """Clip-stable seed; do not use Python hash() (randomized per process)."""
    payload = "\0".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little")


def sample_traj_indices(n, bg_flags, n_traj, traj_source, bg_ratio, rng):
    """Stratified instance/background subsample. Same policy as BEVTracker LoadTraj."""
    bg_flags = np.asarray(bg_flags, dtype=bool).reshape(-1)
    if bg_flags.size != n:
        bg_flags = np.zeros(n, dtype=bool)
    inst = np.flatnonzero(~bg_flags)
    bg = np.flatnonzero(bg_flags)

    def take(pool, k):
        k = min(int(k), len(pool))
        if k <= 0:
            return pool[:0]
        chosen = rng.choice(len(pool), k, replace=False)
        return pool[chosen]

    if traj_source == "instance":
        indices = take(inst, min(n_traj, len(inst)))
    elif traj_source == "background":
        indices = take(bg, min(n_traj, len(bg)))
    elif traj_source == "all":
        n_choice = min(int(n_traj), int(n))
        if n_choice <= 0:
            return np.array([], dtype=np.int64)
        n_bg = int(round(n_choice * float(bg_ratio)))
        n_bg = min(max(n_bg, 0), n_choice, len(bg))
        n_inst = n_choice - n_bg
        if n_inst > len(inst):
            extra = n_inst - len(inst)
            n_inst = len(inst)
            n_bg = min(n_bg + extra, len(bg))
        if n_bg > len(bg):
            extra = n_bg - len(bg)
            n_bg = len(bg)
            n_inst = min(n_inst + extra, len(inst))
        parts = []
        if n_inst:
            parts.append(take(inst, n_inst))
        if n_bg:
            parts.append(take(bg, n_bg))
        indices = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    else:
        raise ValueError("unknown traj_source: {}".format(traj_source))
    return np.sort(np.asarray(indices, dtype=np.int64))


def _as_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_to_mat(quat_wxyz, trans) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _quat_wxyz_to_mat(quat_wxyz)
    mat[:3, 3] = np.asarray(trans, dtype=np.float32).reshape(3)
    return mat


def _samples_rel(data_path: str) -> Path:
    parts = Path(data_path).parts
    if "samples" in parts:
        idx = parts.index("samples")
        return Path(*parts[idx:])
    p = Path(data_path)
    return Path("samples") / p.parent.name / p.name


def _resolve_jpeg(data_path: str, nusc_root: Path) -> Path:
    p = Path(data_path)
    if p.is_file():
        return p
    candidates = [
        nusc_root / data_path,
        nusc_root / _samples_rel(data_path),
        Path(data_path),
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"JPEG not found for {data_path}")


def _depth_npy(data_path: str, unidepth_root: Path) -> Path:
    return (unidepth_root / _samples_rel(data_path)).with_suffix(".npy")


def _scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = np.array(K, dtype=np.float32, copy=True)
    out[0, :] *= sx
    out[1, :] *= sy
    return out


def _bilinear_depth(depth: np.ndarray, u: float, v: float) -> float:
    """Sample camera-Z at pixel (u, v) = (x, y)."""
    h, w = depth.shape
    if h < 1 or w < 1:
        return float("nan")
    u = float(np.clip(u, 0.0, w - 1.0))
    v = float(np.clip(v, 0.0, h - 1.0))
    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = u - x0
    dy = v - y0
    z00 = float(depth[y0, x0])
    z10 = float(depth[y0, x1])
    z01 = float(depth[y1, x0])
    z11 = float(depth[y1, x1])
    z = z00 * (1.0 - dx) * (1.0 - dy) + z10 * dx * (1.0 - dy) + z01 * (1.0 - dx) * dy + z11 * dx * dy
    return z


def _unproject_pixel(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) / max(fx, 1e-6) * z
    y = (v - cy) / max(fy, 1e-6) * z
    return np.array([x, y, z], dtype=np.float32)


def _category_id(name) -> int:
    if name is None:
        return -1
    if isinstance(name, (int, np.integer)):
        return int(name)
    return CATEGORY_NAME_TO_ID.get(str(name), -1)


def _resolve_nusc_root(dataset_root: Optional[str], nusc_root: Optional[str]) -> Path:
    if nusc_root:
        return Path(nusc_root)
    if dataset_root:
        cand = Path(dataset_root) / "nuscenes"
        if cand.is_dir():
            return cand
        if Path(dataset_root).is_dir() and (Path(dataset_root) / "samples").is_dir():
            return Path(dataset_root)
    if Path(DEFAULT_NUSC_ROOT).is_dir():
        return Path(DEFAULT_NUSC_ROOT)
    raise FileNotFoundError(
        "Cannot find nuScenes root. Pass nusc_root= or put the dataset under "
        f"{DEFAULT_NUSC_ROOT} or {{dataset_root}}/nuscenes."
    )


class NuscTrackDataset(Dataset):
    """Eval-style NuscTrack loader for MVTracker (no scene-radius normalization)."""

    @staticmethod
    def from_name(dataset_name: str, dataset_root: str, n_traj: Optional[int] = None):
        """
        Names:
          - nusctrack / nusctrack-val
          - nusctrack-train
          - nusctrack-val-max2
          - nusctrack-val-2dpt
        """
        if not dataset_name.startswith("nusctrack"):
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
        rest = dataset_name[len("nusctrack") :]
        split = "val"
        max_clips = None
        if rest.startswith("-train"):
            split = "train"
            rest = rest[len("-train") :]
        elif rest.startswith("-val"):
            split = "val"
            rest = rest[len("-val") :]
        if rest.startswith("-max"):
            match = re.match(r"-max(\d+)", rest)
            if match is None:
                raise ValueError(f"Bad max-clips suffix in {dataset_name}")
            max_clips = int(match.group(1))
            rest = rest[match.end() :]
        if rest.startswith("-2dpt"):
            rest = rest[len("-2dpt") :]
        if rest not in ("",):
            raise ValueError(f"Unparsed NuscTrack dataset suffix: {rest!r} in {dataset_name}")
        kwargs = dict(
            nusc_root=_resolve_nusc_root(dataset_root, None),
            split=split,
            max_clips=max_clips,
        )
        if n_traj is not None:
            kwargs["n_traj"] = int(n_traj)
        return NuscTrackDataset(**kwargs)

    def __init__(
        self,
        nusc_root: str | Path = DEFAULT_NUSC_ROOT,
        split: str = "val",
        traj_dir: str = "processed_track_24_omnidc",
        unidepth_dir: str = "unidepthv2",
        ann_file: Optional[str] = None,
        n_traj: int = 384,
        bg_ratio: float = 0.5,
        traj_source: str = "all",
        traj_sample_seed: int = 0,
        image_size: Optional[Tuple[int, int]] = (432, 768),
        skip_incomplete: bool = True,
        max_clips: Optional[int] = None,
        require_unidepth: bool = True,
    ):
        super().__init__()
        if traj_source not in ("all", "instance", "background"):
            raise ValueError(f"traj_source must be all/instance/background, got {traj_source}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split {split}")

        self.nusc_root = Path(nusc_root)
        self.split = split
        self.traj_root = self.nusc_root / traj_dir
        self.unidepth_root = self.nusc_root / unidepth_dir
        self.n_traj = int(n_traj)
        self.bg_ratio = float(bg_ratio)
        self.traj_source = traj_source
        self.traj_sample_seed = int(traj_sample_seed)
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self.skip_incomplete = bool(skip_incomplete)
        self.require_unidepth = bool(require_unidepth)

        if ann_file is None:
            ann_path = self.nusc_root / f"tap3d-nuscenes_infos_{split}.pkl"
        else:
            ann_path = Path(ann_file)
            if not ann_path.is_absolute():
                ann_path = self.nusc_root / ann_file
        if not ann_path.is_file():
            raise FileNotFoundError(f"NuscTrack ann file not found: {ann_path}")
        if not self.traj_root.is_dir():
            raise FileNotFoundError(f"NuscTrack traj dir not found: {self.traj_root}")
        if self.require_unidepth and not self.unidepth_root.is_dir():
            raise FileNotFoundError(f"UniDepth dir not found: {self.unidepth_root}")

        with open(ann_path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "infos" in payload:
            scene_infos = payload["infos"]
        elif isinstance(payload, list):
            scene_infos = payload
        else:
            raise TypeError(f"unexpected ann pkl: {type(payload)}")

        self.scene_infos = {info["name"]: info for info in scene_infos}
        npy_rel = self._index_unidepth() if self.require_unidepth else None
        self.clips = self._discover_clips(npy_rel)
        if max_clips is not None:
            self.clips = self.clips[: int(max_clips)]
        if not self.clips:
            warnings.warn(
                f"NuscTrackDataset: no clips for split={split} under {self.traj_root} "
                f"(UniDepth skip_incomplete={self.skip_incomplete}, indexed npy="
                f"{0 if npy_rel is None else len(npy_rel)})."
            )
        else:
            print(
                f"NuscTrackDataset: {len(self.clips)} clips "
                f"(split={split}, unidepth={self.unidepth_root}, image_size={self.image_size})"
            )

    def _index_unidepth(self) -> set:
        indexed = set()
        samples_dir = self.unidepth_root / "samples"
        if not samples_dir.is_dir():
            return indexed
        for cam_dir in samples_dir.iterdir():
            if not cam_dir.is_dir():
                continue
            for npy in cam_dir.glob("*.npy"):
                indexed.add(f"samples/{cam_dir.name}/{npy.name}")
        return indexed

    def _scene_depth_complete(self, scene_name: str, npy_rel: Optional[set]) -> bool:
        if npy_rel is None:
            return True
        info = self.scene_infos.get(scene_name)
        if info is None:
            return False
        for sample in info["sample_infos"]:
            for cam in CAMERAS:
                cam_data = sample["cams"][cam]
                rel = _depth_npy(cam_data["data_path"], self.unidepth_root)
                key = f"samples/{rel.parent.name}/{rel.name}"
                if key not in npy_rel:
                    return False
        return True

    def _discover_clips(self, npy_rel: Optional[set]) -> List[dict]:
        clips = []
        skipped_incomplete = 0
        skipped_other_split = 0
        for pkl_path in sorted(self.traj_root.glob("bev_traj_infos_*.pkl")):
            match = CLIP_NAME_RE.match(pkl_path.stem)
            if match is None:
                continue
            scene_name, half = match.group(1), match.group(2)
            if scene_name not in self.scene_infos:
                skipped_other_split += 1
                continue
            if self.skip_incomplete and self.require_unidepth:
                if not self._scene_depth_complete(scene_name, npy_rel):
                    skipped_incomplete += 1
                    continue
            clips.append(
                {
                    "pkl": pkl_path,
                    "scene_name": scene_name,
                    "is_former": half == "former",
                    "seq_name": f"{scene_name}_{half}",
                }
            )
        if skipped_incomplete:
            print(
                f"NuscTrackDataset: skipped {skipped_incomplete} clips without full UniDepth "
                f"({skipped_other_split} belonged to another split)."
            )
        elif skipped_other_split:
            print(f"NuscTrackDataset: ignored {skipped_other_split} clips not in split={self.split}.")
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        datapoint = self._load_clip(clip, index)
        return datapoint, True

    def _load_clip(self, clip: dict, index: int) -> Datapoint:
        with open(clip["pkl"], "rb") as f:
            payload = pickle.load(f)
        traj_info = payload.get("traj_info", payload)
        scene_name = payload.get("scene_name", clip["scene_name"])
        is_former = bool(payload.get("is_former", clip["is_former"]))
        scene = self.scene_infos[scene_name]
        sample_infos = scene["sample_infos"]
        scene_len = len(sample_infos)

        traj = _as_numpy(traj_info["traj"]).astype(np.float32)
        if traj.ndim != 3:
            raise RuntimeError(f"{clip['pkl']}: traj shape {traj.shape}")
        n_total, seq_len, _ = traj.shape
        if is_former:
            start, end = 0, seq_len
        else:
            start, end = scene_len - seq_len, scene_len
        if start < 0 or end > scene_len:
            raise RuntimeError(
                f"{clip['seq_name']}: clip frames [{start}, {end}) vs scene_len={scene_len}"
            )
        samples = sample_infos[start:end]
        if len(samples) != seq_len:
            raise RuntimeError(
                f"{clip['seq_name']}: expected {seq_len} samples, got {len(samples)}"
            )

        rgbs_v, depths_v, depths_orig_v, K_orig_v, K_v, extrs_v = self._load_rgbd_calib(samples)

        n_views, n_frames, h, w, _ = rgbs_v.shape
        if n_frames != seq_len:
            raise RuntimeError(f"{clip['seq_name']}: RGB T={n_frames} vs traj T={seq_len}")

        bg_flags = traj_info.get("is_background")
        if bg_flags is None or len(bg_flags) == 0:
            bg_flags = np.zeros(n_total, dtype=bool)
        else:
            bg_flags = np.asarray(bg_flags, dtype=bool).reshape(-1)
            if bg_flags.size != n_total:
                bg_flags = np.zeros(n_total, dtype=bool)

        if n_total == 0:
            raise RuntimeError(f"NuscTrack empty traj clip {clip['seq_name']}")

        if self.split == "train":
            rng = np.random.RandomState(
                int((torch.initial_seed() + index) % (2 ** 32))
            )
        else:
            rng = np.random.RandomState(
                _stable_seed_int(
                    self.traj_sample_seed,
                    scene_name,
                    int(bool(is_former)),
                    self.traj_source,
                    "{:.6f}".format(self.bg_ratio),
                    self.n_traj,
                )
            )
        indices = sample_traj_indices(
            n_total, bg_flags, self.n_traj, self.traj_source, self.bg_ratio, rng
        )
        if indices.size == 0:
            indices = np.array([0], dtype=np.int64)
            valid_force_false = True
        else:
            valid_force_false = False

        valid = _as_numpy(traj_info["valid"]).astype(bool)
        if valid.shape != (n_total, seq_len):
            raise RuntimeError(f"{clip['seq_name']}: valid {valid.shape} vs traj {(n_total, seq_len)}")
        finite_xyz = np.isfinite(traj).all(axis=-1)
        valid = valid & finite_xyz
        if valid_force_false:
            valid = np.zeros_like(valid)

        h_orig = depths_orig_v.shape[2]
        w_orig = depths_orig_v.shape[3]
        sx = w / float(w_orig)
        sy = h / float(h_orig)

        query_points_3d = []
        query_points_2d = []
        query_points_view = []
        traj2d = []
        vis = []
        cats = traj_info.get("category")
        category_ids = []
        query_points_meta = traj_info["query_point"]
        visibility_meta = traj_info["visibility"]
        tracks2d_meta = traj_info.get("tracks2d")

        for idx in indices:
            qp = query_points_meta[idx]
            cam_name = qp["camera"]
            if cam_name not in CAM_TO_IDX:
                raise KeyError(f"unknown query camera {cam_name}")
            view_idx = CAM_TO_IDX[cam_name]
            coord = np.asarray(qp["coord"], dtype=np.float32).reshape(-1)
            t_q = int(np.round(coord[0]))
            u_q = float(coord[1])
            v_q = float(coord[2])
            t_q = int(np.clip(t_q, 0, seq_len - 1))
            z_q = _bilinear_depth(depths_orig_v[view_idx, t_q], u_q, v_q)
            if not np.isfinite(z_q) or z_q <= 1e-4:
                z_q = 1.0
            xyz_cam = _unproject_pixel(u_q, v_q, z_q, K_orig_v[view_idx, t_q])
            xyz_cam_h = np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0], dtype=np.float32)
            # extrs is world(ego)->cam, so cam->ego is the inverse
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = extrs_v[view_idx, t_q]
            c2w = np.linalg.inv(w2c)
            xyz_ego = c2w[:3, :] @ xyz_cam_h
            query_points_3d.append(np.array([t_q, xyz_ego[0], xyz_ego[1], xyz_ego[2]], dtype=np.float32))
            query_points_view.append(view_idx)
            # TAP-Vid 2D query is (t, y, x) in the tensor resolution
            query_points_2d.append(np.array([t_q, v_q * sy, u_q * sx], dtype=np.float32))

            vis_i = np.zeros((n_views, seq_len), dtype=np.float32)
            vis_dict = visibility_meta[idx]
            for v, cam in enumerate(CAMERAS):
                if cam in vis_dict:
                    vis_i[v] = _as_numpy(vis_dict[cam]).astype(np.float32).reshape(-1)[:seq_len]
            vis.append(vis_i)

            xy_i = np.zeros((n_views, seq_len, 2), dtype=np.float32)
            if tracks2d_meta is not None:
                t2d = tracks2d_meta[idx]
                for v, cam in enumerate(CAMERAS):
                    if cam in t2d:
                        xy = _as_numpy(t2d[cam]).astype(np.float32).reshape(-1, 2)[:seq_len]
                        xy_i[v, :, 0] = xy[:, 0] * sx
                        xy_i[v, :, 1] = xy[:, 1] * sy
            traj2d.append(xy_i)
            category_ids.append(_category_id(None if cats is None or idx >= len(cats) else cats[idx]))

        traj3d = torch.from_numpy(np.nan_to_num(traj[indices], nan=0.0, posinf=0.0, neginf=0.0))
        # (N,T,3) -> (T,N,3)
        traj3d = traj3d.permute(1, 0, 2).contiguous()
        valid_t = torch.from_numpy(valid[indices].T.copy())  # (T,N)
        vis_t = torch.from_numpy(np.stack(vis, axis=-1))  # (V,T,N)
        traj2d_np = np.stack(traj2d, axis=2)  # (V,T,N,2)

        # Camera Z of GT ego points (for the 2D+Z trajectory field)
        xyz_ego_h = np.concatenate(
            [traj3d.numpy(), np.ones((seq_len, len(indices), 1), dtype=np.float32)], axis=-1
        )
        z_cam = np.zeros((n_views, seq_len, len(indices), 1), dtype=np.float32)
        for v in range(n_views):
            # extrs[v,t]: 3x4 world->cam
            cam_xyz = np.einsum("tij,tnj->tni", extrs_v[v], xyz_ego_h)
            z_cam[v, :, :, 0] = cam_xyz[..., 2]
        traj2d_w_z = np.concatenate([traj2d_np, z_cam], axis=-1)

        query_points_3d_t = torch.from_numpy(np.stack(query_points_3d, axis=0))
        query_points_2d_t = torch.from_numpy(np.stack(query_points_2d, axis=0))
        query_points_view_t = torch.tensor(query_points_view, dtype=torch.long)
        is_background_t = torch.from_numpy(bg_flags[indices].copy())
        category_id_t = torch.tensor(category_ids, dtype=torch.long)

        rgbs = torch.from_numpy(rgbs_v).permute(0, 1, 4, 2, 3).float()
        depths = torch.from_numpy(depths_v)[:, :, None].float()
        segs = torch.ones((n_frames, 1, h, w), dtype=torch.float32)
        intrs = torch.from_numpy(K_v).float()
        extrs = torch.from_numpy(extrs_v).float()

        return Datapoint(
            video=rgbs,
            videodepth=depths,
            segmentation=segs,
            trajectory=torch.from_numpy(traj2d_w_z).float(),
            trajectory_3d=traj3d.float(),
            visibility=vis_t.bool(),
            valid=valid_t.bool(),
            seq_name=clip["seq_name"],
            intrs=intrs,
            extrs=extrs,
            query_points=query_points_2d_t,
            query_points_3d=query_points_3d_t,
            query_points_view=query_points_view_t,
            is_background=is_background_t,
            category_id=category_id_t,
            track_upscaling_factor=1.0,
        )

    def _load_rgbd_calib(self, samples: Sequence[dict]):
        n_frames = len(samples)
        rgbs, depths, depths_orig, K_orig, K_out, extrs = [], [], [], [], [], []
        target_hw = self.image_size  # (H, W) or None

        for cam in CAMERAS:
            rgb_t, depth_t, depth_orig_t, k_orig_t, k_t, extr_t = [], [], [], [], [], []
            for sample in samples:
                cam_data = sample["cams"][cam]
                jpeg = _resolve_jpeg(cam_data["data_path"], self.nusc_root)
                rgb = cv2.imread(str(jpeg), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(f"failed to read RGB {jpeg}")
                rgb = rgb[:, :, ::-1].copy()
                h0, w0 = rgb.shape[:2]
                if self.require_unidepth:
                    npy = _depth_npy(cam_data["data_path"], self.unidepth_root)
                    if not npy.is_file():
                        raise FileNotFoundError(f"UniDepth missing: {npy}")
                    depth = np.load(npy).astype(np.float32)
                    if depth.ndim != 2:
                        depth = np.squeeze(depth)
                    if depth.shape != (h0, w0):
                        raise RuntimeError(
                            f"UniDepth shape {depth.shape} != RGB {(h0, w0)} for {npy}"
                        )
                else:
                    depth = np.zeros((h0, w0), dtype=np.float32)

                K = np.asarray(cam_data["cam_intrinsic"], dtype=np.float32).reshape(3, 3)
                c2e = _pose_to_mat(cam_data["sensor2ego_rotation"], cam_data["sensor2ego_translation"])
                w2c = np.linalg.inv(c2e).astype(np.float32)

                depth_orig_t.append(depth)
                k_orig_t.append(K)

                if target_hw is not None and (h0, w0) != tuple(target_hw):
                    ht, wt = target_hw
                    sx = wt / float(w0)
                    sy = ht / float(h0)
                    rgb = cv2.resize(rgb, (wt, ht), interpolation=cv2.INTER_LINEAR)
                    depth = cv2.resize(depth, (wt, ht), interpolation=cv2.INTER_LINEAR)
                    K = _scale_K(K, sx, sy)

                rgb_t.append(rgb)
                depth_t.append(depth)
                k_t.append(K)
                extr_t.append(w2c[:3, :])

            rgbs.append(np.stack(rgb_t, axis=0))
            depths.append(np.stack(depth_t, axis=0))
            depths_orig.append(np.stack(depth_orig_t, axis=0))
            K_orig.append(np.stack(k_orig_t, axis=0))
            K_out.append(np.stack(k_t, axis=0))
            extrs.append(np.stack(extr_t, axis=0))

        return (
            np.stack(rgbs, axis=0),
            np.stack(depths, axis=0),
            np.stack(depths_orig, axis=0),
            np.stack(K_orig, axis=0),
            np.stack(K_out, axis=0),
            np.stack(extrs, axis=0),
        )
