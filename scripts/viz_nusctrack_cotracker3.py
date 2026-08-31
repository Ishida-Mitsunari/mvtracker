#!/usr/bin/env python3
"""Visualize CoTracker3 on one NuscTrack clip (CAM_FRONT 2D + ego 3D).

Runs the same eval pipeline as ``eval_nusctrack_parallel`` (per-query-cam
CoTracker3 + UniDepth lift). Draws only tracks whose query camera is CAM_FRONT.

Outputs under ``--out-dir``:
  cam_front_native2d_gt.mp4 / _pred.mp4 / _overlay.mp4
  cam_front_proj3d_overlay.mp4
  tracks_3d.mp4, tracks_bev.mp4
  summary.png
  tracks.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mvtracker.datasets.nusctrack_dataset import CAM_TO_IDX, CAMERAS, NuscTrackDataset
from mvtracker.models.core.model_utils import world_space_to_pixel_xy_and_camera_z
from mvtracker.models.core.monocular_baselines import (
    CoTrackerOfflineWrapper,
    MonocularToMultiViewAdapter,
)
from mvtracker.models.evaluation_predictor_3dpt import EvaluationPredictor

CAM_FRONT = "CAM_FRONT"
CAM_FRONT_IDX = CAM_TO_IDX[CAM_FRONT]
GT_BGR = (40, 200, 70)
PRED_BGR = (50, 60, 255)
GT_RGB = (0.16, 0.78, 0.27)
PRED_RGB = (0.90, 0.22, 0.18)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene-0003")
    p.add_argument("--half", default="former", choices=["former", "later", "both"])
    p.add_argument("--dataset", default="nusctrack-val")
    p.add_argument("--dataset-root", default="/share/tgp/yangyi/nuscenes")
    p.add_argument(
        "--out-dir",
        default="/share/tgp/yangyi/mvtracker/logs/cotracker3_offline_nusctrack_val/viz",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-instance", type=int, default=24)
    p.add_argument("--max-background", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--leave-trace", type=int, default=12)
    p.add_argument("--min-cam-z", type=float, default=0.5)
    return p.parse_args()


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def subsample_tracks(is_bg, max_inst, max_bg, seed):
    is_bg = np.asarray(is_bg, dtype=bool).reshape(-1)
    inst = np.flatnonzero(~is_bg)
    bg = np.flatnonzero(is_bg)
    rng = np.random.RandomState(seed)

    def take(pool, k):
        if k <= 0 or len(pool) == 0:
            return pool[:0]
        k = min(int(k), len(pool))
        return np.sort(pool[rng.choice(len(pool), k, replace=False)])

    parts = [take(inst, max_inst), take(bg, max_bg)]
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.arange(len(is_bg), dtype=np.int64)
    return np.sort(np.concatenate(parts))


def rainbow_colors(n):
    cmap = plt.get_cmap("hsv", max(n, 1))
    rgb = np.array([cmap(i)[:3] for i in range(n)], dtype=np.float32)
    bgr = (rgb[:, ::-1] * 255).astype(np.uint8)
    return bgr, rgb


def draw_tracks_on_rgb(
    frames_rgb,
    xy,
    vis,
    colors_bgr,
    leave_trace=12,
    radius=6,
    thickness=2,
    query_t=None,
):
    """frames_rgb: T,H,W,3 uint8 RGB; xy T,N,2; vis T,N; colors N,3 BGR."""
    T, H, W, _ = frames_rgb.shape
    N = xy.shape[1]
    out = []
    for t in range(T):
        bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        t0 = max(0, t - leave_trace)
        for n in range(N):
            color = tuple(int(c) for c in colors_bgr[n])
            pts = []
            for tt in range(t0, t + 1):
                if not vis[tt, n]:
                    pts = []
                    continue
                u, v = xy[tt, n]
                if not np.isfinite(u) or not np.isfinite(v):
                    pts = []
                    continue
                pts.append((int(round(u)), int(round(v))))
            if len(pts) >= 2:
                cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
            if vis[t, n] and np.isfinite(xy[t, n]).all():
                u, v = int(round(xy[t, n, 0])), int(round(xy[t, n, 1]))
                cv2.circle(overlay, (u, v), radius, color, -1, cv2.LINE_AA)
                cv2.circle(overlay, (u, v), radius, (20, 20, 20), 1, cv2.LINE_AA)
                if query_t is not None and int(query_t[n]) == t:
                    cv2.circle(overlay, (u, v), radius + 4, (255, 255, 255), 2, cv2.LINE_AA)
        blended = cv2.addWeighted(overlay, 0.85, bgr, 0.15, 0)
        out.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


def draw_overlay_gt_pred(frames_rgb, xy_gt, vis_gt, xy_pr, vis_pr, leave_trace=12, query_t=None):
    T, H, W, _ = frames_rgb.shape
    N = xy_gt.shape[1]
    out = []
    for t in range(T):
        bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        t0 = max(0, t - leave_trace)
        for n in range(N):
            for xy, vis, color, thick, rad in (
                (xy_gt, vis_gt, GT_BGR, 2, 5),
                (xy_pr, vis_pr, PRED_BGR, 2, 4),
            ):
                pts = []
                for tt in range(t0, t + 1):
                    if not vis[tt, n]:
                        pts = []
                        continue
                    u, v = xy[tt, n]
                    if not np.isfinite(u) or not np.isfinite(v):
                        pts = []
                        continue
                    pts.append((int(round(u)), int(round(v))))
                if len(pts) >= 2:
                    cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, color, thick, cv2.LINE_AA)
                if vis[t, n] and np.isfinite(xy[t, n]).all():
                    u, v = int(round(xy[t, n, 0])), int(round(xy[t, n, 1]))
                    cv2.circle(overlay, (u, v), rad, color, -1, cv2.LINE_AA)
                    if query_t is not None and int(query_t[n]) == t and color == GT_BGR:
                        cv2.circle(overlay, (u, v), rad + 4, (255, 255, 255), 2, cv2.LINE_AA)
        blended = cv2.addWeighted(overlay, 0.85, bgr, 0.15, 0)
        out.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


def put_banner(frames, text):
    out = []
    for fr in frames:
        img = fr.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            img,
            text,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out.append(img)
    return np.stack(out, axis=0)


def hstack_videos(*videos):
    h = min(v.shape[1] for v in videos)
    resized = []
    for v in videos:
        if v.shape[1] == h:
            resized.append(v)
            continue
        scale = h / float(v.shape[1])
        w = int(round(v.shape[2] * scale))
        resized.append(
            np.stack([cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA) for fr in v], axis=0)
        )
    return np.concatenate(resized, axis=2)


def write_mp4(path, frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    if w % 2:
        frames = frames[:, :, : w - 1]
        w -= 1
    if h % 2:
        frames = frames[:, : h - 1]
        h -= 1
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {path}")
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def load_cam_front_orig(dataset: NuscTrackDataset, clip: dict):
    scene = dataset.scene_infos[clip["scene_name"]]
    sample_infos = scene["sample_infos"]
    scene_len = len(sample_infos)
    with open(clip["pkl"], "rb") as f:
        payload = pickle_load(f)
    traj_info = payload.get("traj_info", payload)
    traj = np.asarray(traj_info["traj"])
    seq_len = traj.shape[1]
    is_former = bool(payload.get("is_former", clip["is_former"]))
    start = 0 if is_former else scene_len - seq_len
    samples = sample_infos[start : start + seq_len]
    rgbs, Ks = [], []
    for sample in samples:
        cam = sample["cams"][CAM_FRONT]
        from mvtracker.datasets.nusctrack_dataset import _resolve_jpeg

        jpeg = _resolve_jpeg(cam["data_path"], dataset.nusc_root)
        bgr = cv2.imread(str(jpeg), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(jpeg)
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        Ks.append(np.asarray(cam["cam_intrinsic"], dtype=np.float32).reshape(3, 3))
    return np.stack(rgbs, axis=0), np.stack(Ks, axis=0)


def pickle_load(f):
    import pickle

    return pickle.load(f)


def project_ego(xyz, K, extr, min_z, h, w):
    """xyz T,N,3; K T,3,3; extr T,3,4 -> xy T,N,2 and in_fov T,N."""
    xyz_t = torch.from_numpy(np.asarray(xyz, dtype=np.float32))
    K_t = torch.from_numpy(np.asarray(K, dtype=np.float32))
    extr_t = torch.from_numpy(np.asarray(extr, dtype=np.float32))
    xy, z = world_space_to_pixel_xy_and_camera_z(xyz_t, K_t, extr_t)
    xy = xy.numpy()
    z = z.numpy()[..., 0]
    in_fov = (
        np.isfinite(xy).all(axis=-1)
        & np.isfinite(z)
        & (z > float(min_z))
        & (xy[..., 0] >= 0)
        & (xy[..., 0] < w)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < h)
    )
    xy = xy.copy()
    xy[~in_fov] = np.nan
    return xy, in_fov


def render_3d_frames(gt, pred, valid, query_t, leave_trace=12, mode="3d"):
    """gt/pred T,N,3; valid T,N. Returns RGB uint8 frames."""
    T, N, _ = gt.shape
    finite = valid & np.isfinite(gt).all(-1)
    pts = gt[finite]
    if pts.size == 0:
        pts = gt.reshape(-1, 3)
    lo = np.nanpercentile(pts, 2, axis=0)
    hi = np.nanpercentile(pts, 98, axis=0)
    span = np.maximum(hi - lo, 1.0)
    mid = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(span)) * 1.15
    lim = np.stack([mid - half, mid + half], axis=0)

    frames = []
    for t in range(T):
        fig = Figure(figsize=(7.2, 6.4), dpi=120)
        canvas = FigureCanvasAgg(fig)
        if mode == "3d":
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim(lim[0, 0], lim[1, 0])
            ax.set_ylim(lim[0, 1], lim[1, 1])
            ax.set_zlim(lim[0, 2], lim[1, 2])
            ax.set_xlabel("x forward (m)")
            ax.set_ylabel("y left (m)")
            ax.set_zlabel("z up (m)")
            ax.view_init(elev=22, azim=-70)
        else:
            ax = fig.add_subplot(111)
            ax.set_xlim(lim[0, 1], lim[1, 1])
            ax.set_ylim(lim[0, 0], lim[1, 0])
            ax.set_xlabel("y left (m)")
            ax.set_ylabel("x forward (m)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.25)
            ax.scatter([0], [0], c="k", s=28, zorder=5, marker="x")

        t0 = max(0, t - leave_trace)
        for n in range(N):
            q = int(query_t[n])
            sl = slice(max(t0, q), t + 1)
            if sl.stop - sl.start < 1:
                continue
            g = gt[sl, n]
            p = pred[sl, n]
            m = valid[sl, n] & np.isfinite(g).all(-1)
            if m.sum() >= 1:
                gg = g.copy()
                if mode == "3d":
                    ax.plot(gg[m, 0], gg[m, 1], gg[m, 2], color=GT_RGB, lw=1.4, alpha=0.9)
                    ax.scatter(gg[m][-1, 0], gg[m][-1, 1], gg[m][-1, 2], color=GT_RGB, s=10)
                else:
                    ax.plot(gg[m, 1], gg[m, 0], color=GT_RGB, lw=1.6, alpha=0.9)
                    ax.scatter(gg[m][-1, 1], gg[m][-1, 0], color=GT_RGB, s=14, zorder=4)
            mpr = (np.arange(T)[sl] >= q) & np.isfinite(p).all(-1)
            if mpr.sum() >= 1:
                pp = p.copy()
                if mode == "3d":
                    ax.plot(pp[mpr, 0], pp[mpr, 1], pp[mpr, 2], color=PRED_RGB, lw=1.4, alpha=0.85)
                    ax.scatter(pp[mpr][-1, 0], pp[mpr][-1, 1], pp[mpr][-1, 2], color=PRED_RGB, s=10)
                else:
                    ax.plot(pp[mpr, 1], pp[mpr, 0], color=PRED_RGB, lw=1.6, alpha=0.85)
                    ax.scatter(pp[mpr][-1, 1], pp[mpr][-1, 0], color=PRED_RGB, s=14, zorder=4)

        ax.set_title("GT green  |  Pred red   t={}/{}".format(t, T - 1), fontsize=10)
        fig.tight_layout()
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(buf)
    return np.stack(frames, axis=0)


def render_summary_png(
    path,
    overlay_native,
    overlay_proj,
    bev_frame,
    xyz_frame,
    title,
):
    fig = plt.figure(figsize=(16, 10), dpi=140)
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    for ax, img, cap in zip(
        axes,
        [overlay_native, overlay_proj, bev_frame, xyz_frame],
        [
            "CAM_FRONT native 2D (GT green / Pred red)",
            "CAM_FRONT 3D→2D projection",
            "Ego BEV",
            "Ego 3D",
        ],
    ):
        ax.imshow(img)
        ax.set_title(cap, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def find_clip(dataset, scene, half):
    for i, clip in enumerate(dataset.clips):
        if clip["scene_name"] == scene and clip["seq_name"].endswith("_" + half):
            return i, clip
    raise KeyError(
        "clip {}_{} not in {} ({} clips). first={}".format(
            scene,
            half,
            dataset.split,
            len(dataset.clips),
            [c["seq_name"] for c in dataset.clips[:8]],
        )
    )


def run_clip(args, dataset, half):
    idx, clip = find_clip(dataset, args.scene, half)
    print("Loaded clip index={} seq={}".format(idx, clip["seq_name"]))
    datapoint, _ = dataset[idx]

    device = torch.device("cuda:0")
    rgbs = datapoint.video[None].to(device)
    depths = datapoint.videodepth[None].to(device)
    query_points_3d = datapoint.query_points_3d[None].to(device)
    query_points_view = datapoint.query_points_view[None].to(device)
    intrs = datapoint.intrs[None].to(device)
    extrs = datapoint.extrs[None].to(device)

    print("Loading CoTracker3 …")
    inner = CoTrackerOfflineWrapper(model_name="cotracker3_offline", grid_size=10)
    model = MonocularToMultiViewAdapter(inner).to(device).eval()
    predictor = EvaluationPredictor(
        multiview_model=model,
        interp_shape=None,
        visibility_threshold=0.5,
        grid_size=0,
        n_grids_per_view=1,
        local_grid_size=0,
        local_extent=50,
        single_point=False,
        sift_size=0,
        num_uniformly_sampled_pts=0,
        n_iters=4,
    )
    with torch.no_grad():
        results = predictor(
            rgbs=rgbs,
            depths=depths,
            query_points_3d=query_points_3d,
            query_points_view=query_points_view,
            intrs=intrs,
            extrs=extrs,
        )

    qview = _to_np(datapoint.query_points_view).reshape(-1)
    front = np.flatnonzero(qview == CAM_FRONT_IDX)
    if front.size == 0:
        raise RuntimeError("no CAM_FRONT queries in {}".format(clip["seq_name"]))

    is_bg = _to_np(datapoint.is_background).reshape(-1)[front]
    keep_local = subsample_tracks(is_bg, args.max_instance, args.max_background, args.seed)
    keep = front[keep_local]
    print(
        "CAM_FRONT queries: {} (inst={}, bg={}); drawing {}".format(
            front.size,
            int((~is_bg).sum()),
            int(is_bg.sum()),
            keep.size,
        )
    )

    pred_xyz = _to_np(results["traj_e"])[0][:, keep]  # T,N,3
    pred_vis = _to_np(results["vis_e"])[0][:, keep]
    if pred_vis.dtype != bool:
        pred_vis = pred_vis >= 0.5
    pred_xy_native = _to_np(results["traj2d_e"])[0][:, keep]  # T,N,2  tensor res

    gt_xyz = _to_np(datapoint.trajectory_3d)[:, keep]
    gt_valid = _to_np(datapoint.valid)[:, keep].astype(bool)
    vis_all = _to_np(datapoint.visibility)  # V,T,N_all
    gt_vis_front = vis_all[CAM_FRONT_IDX][:, keep].astype(bool)
    # Slice view first, then time x tracks — mixed advanced indexing would swap T/N.
    gt_xy_native = _to_np(datapoint.trajectory)[CAM_FRONT_IDX][:, keep, :2]
    query_t = _to_np(datapoint.query_points_3d)[keep, 0].astype(int)
    is_bg_draw = _to_np(datapoint.is_background).reshape(-1)[keep]

    # video is V, T, C, H, W at the eval tensor resolution (432x768)
    _, T, _, H, W = datapoint.video.shape
    after_q = np.arange(T)[:, None] >= query_t[None, :]

    # Native 2D: CoTracker pixels in tensor resolution, OOB = invisible
    def in_image(xy, h, w):
        return (
            np.isfinite(xy).all(-1)
            & (xy[..., 0] >= 0)
            & (xy[..., 0] < w)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] < h)
        )

    vis_pred_2d = pred_vis & after_q & in_image(pred_xy_native, H, W)
    vis_gt_2d = gt_vis_front & gt_valid & after_q & in_image(gt_xy_native, H, W)

    orig_rgb, K_orig = load_cam_front_orig(dataset, clip)
    H0, W0 = orig_rgb.shape[1], orig_rgb.shape[2]
    sx, sy = W0 / float(W), H0 / float(H)

    def scale_xy(xy):
        out = xy.copy()
        out[..., 0] *= sx
        out[..., 1] *= sy
        return out

    xy_gt_orig = scale_xy(gt_xy_native)
    xy_pr_orig = scale_xy(pred_xy_native)
    vis_pred_2d_orig = pred_vis & after_q & in_image(xy_pr_orig, H0, W0)
    vis_gt_2d_orig = gt_vis_front & gt_valid & after_q & in_image(xy_gt_orig, H0, W0)

    extr_front = _to_np(datapoint.extrs)[CAM_FRONT_IDX]  # T,3,4
    xy_gt_proj, fov_gt = project_ego(gt_xyz, K_orig, extr_front, args.min_cam_z, H0, W0)
    xy_pr_proj, fov_pr = project_ego(pred_xyz, K_orig, extr_front, args.min_cam_z, H0, W0)
    vis_gt_proj = gt_valid & after_q & fov_gt
    vis_pr_proj = after_q & fov_pr & np.isfinite(pred_xyz).all(-1)

    colors_bgr, _ = rainbow_colors(keep.size)
    out_dir = Path(args.out_dir) / clip["seq_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_native = put_banner(
        draw_tracks_on_rgb(orig_rgb, xy_gt_orig, vis_gt_2d_orig, colors_bgr, args.leave_trace, query_t=query_t),
        "{}  CAM_FRONT  GT native 2D  (N={})".format(clip["seq_name"], keep.size),
    )
    pr_native = put_banner(
        draw_tracks_on_rgb(orig_rgb, xy_pr_orig, vis_pred_2d_orig, colors_bgr, args.leave_trace, query_t=query_t),
        "{}  CAM_FRONT  CoTracker3 native 2D".format(clip["seq_name"]),
    )
    overlay_native = put_banner(
        draw_overlay_gt_pred(
            orig_rgb, xy_gt_orig, vis_gt_2d_orig, xy_pr_orig, vis_pred_2d_orig, args.leave_trace, query_t
        ),
        "{}  CAM_FRONT native 2D   GT=green  Pred=red".format(clip["seq_name"]),
    )
    overlay_proj = put_banner(
        draw_overlay_gt_pred(
            orig_rgb, xy_gt_proj, vis_gt_proj, xy_pr_proj, vis_pr_proj, args.leave_trace, query_t
        ),
        "{}  CAM_FRONT 3D projected   GT=green  Pred=red (UniDepth lift)".format(clip["seq_name"]),
    )

    side_native = hstack_videos(gt_native, pr_native)
    write_mp4(out_dir / "cam_front_native2d_gt_vs_pred.mp4", side_native, args.fps)
    write_mp4(out_dir / "cam_front_native2d_overlay.mp4", overlay_native, args.fps)
    write_mp4(out_dir / "cam_front_proj3d_overlay.mp4", overlay_proj, args.fps)

    xyz_vid = render_3d_frames(gt_xyz, pred_xyz, gt_valid & after_q, query_t, args.leave_trace, mode="3d")
    bev_vid = render_3d_frames(gt_xyz, pred_xyz, gt_valid & after_q, query_t, args.leave_trace, mode="bev")
    write_mp4(out_dir / "tracks_3d.mp4", xyz_vid, args.fps)
    write_mp4(out_dir / "tracks_bev.mp4", bev_vid, args.fps)

    mid = T // 2
    last = T - 1
    cv2.imwrite(str(out_dir / "cam_front_native2d_overlay_t{:02d}.png".format(mid)), cv2.cvtColor(overlay_native[mid], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "cam_front_proj3d_overlay_t{:02d}.png".format(mid)), cv2.cvtColor(overlay_proj[mid], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "cam_front_native2d_overlay_last.png"), cv2.cvtColor(overlay_native[last], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "cam_front_proj3d_overlay_last.png"), cv2.cvtColor(overlay_proj[last], cv2.COLOR_RGB2BGR))

    render_summary_png(
        out_dir / "summary.png",
        overlay_native[mid],
        overlay_proj[mid],
        bev_vid[mid],
        xyz_vid[mid],
        "CoTracker3 + UniDepth  |  {}  |  CAM_FRONT queries  |  inst {} / bg {}".format(
            clip["seq_name"],
            int((~is_bg_draw).sum()),
            int(is_bg_draw.sum()),
        ),
    )

    np.savez_compressed(
        out_dir / "tracks.npz",
        seq_name=clip["seq_name"],
        keep=keep,
        query_t=query_t,
        is_background=is_bg_draw,
        pred_xyz=pred_xyz,
        pred_vis=pred_vis,
        pred_xy_native=pred_xy_native,
        gt_xyz=gt_xyz,
        gt_valid=gt_valid,
        gt_vis_front=gt_vis_front,
        gt_xy_native=gt_xy_native,
    )
    print("Wrote", out_dir)
    return out_dir


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)
    dataset = NuscTrackDataset.from_name(args.dataset, args.dataset_root)
    halves = ["former", "later"] if args.half == "both" else [args.half]
    for half in halves:
        run_clip(args, dataset, half)


if __name__ == "__main__":
    main()
