import json
import logging
import os
import re
import time
import warnings
from collections import namedtuple
from typing import Iterable
from typing import Optional

import imageio
import matplotlib.cm as cm
import numpy as np
import rerun as rr
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mvtracker.datasets.utils import dataclass_to_cuda_
from mvtracker.evaluation.metrics import compute_tapvid_metrics_original, evaluate_predictions
from mvtracker.models.core.model_utils import world_space_to_pixel_xy_and_camera_z, \
    pixel_xy_and_camera_z_to_world_space, init_pointcloud_from_rgbd
from mvtracker.utils.visualizer_mp4 import log_mp4_track_viz
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun, log_tracks_to_rerun


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def kmeans_sample(pts, count):
    """
    Given (N, 3) torch tensor of 3D points, return (count, 3) tensor of kmeans centers.
    """
    if len(pts) <= count:
        return pts

    logging.info(f"Computing k-means (k={count}, N={len(pts)})...")

    start = time.time()
    with threadpool_limits(limits=1):
        pts_np = pts.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=count, n_init='auto', random_state=0).fit(pts_np)
    duration = time.time() - start

    logging.info(f"K-means clustering completed in {duration:.2f} seconds.")
    centers = torch.tensor(kmeans.cluster_centers_, dtype=pts.dtype, device=pts.device)
    return centers


def evaluate_3dpt(
        gt_tracks,
        gt_visibilities,
        pred_tracks,
        pred_visibilities,
        evaluation_setting,
        track_upscaling_factor,
        query_points=None,
        prefix="3dpt",
        verbose=True,
        add_per_track_results=True,
):
    n_frames, n_tracks, n_point_dim = gt_tracks.shape
    assert gt_tracks.shape == pred_tracks.shape
    assert gt_visibilities.shape == (n_frames, n_tracks)
    assert pred_visibilities.shape == (n_frames, n_tracks)

    if query_points is None:
        query_points_frame_id = gt_visibilities.argmax(axis=0)
        query_points_xyz = gt_tracks[query_points_frame_id, np.arange(gt_tracks.shape[1]), :]
        query_points = np.concatenate([query_points_frame_id[:, None], query_points_xyz], axis=-1)
    else:
        query_points_frame_id = query_points[:, 0].astype(int)
        query_points_xyz = query_points[:, 1:]

    if evaluation_setting == "kubric-multiview":
        assert n_point_dim == 3
        distance_thresholds = [0.05, 0.1, 0.2, 0.4, 0.8]  # The scale is non-metric
        survival_distance_threshold = 0.5  # 50 cm
        static_threshold = 0.01  # < 1 cm
        dynamic_threshold = 0.1  # > 10 cm
        very_dynamic_threshold = 2.0  # > 2 m
    elif evaluation_setting == "dexycb-multiview":
        assert n_point_dim == 3
        distance_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]  # 1 cm, 2 cm, 5 cm, 10 cm, 20 cm
        survival_distance_threshold = 0.1  # 10 cm
        static_threshold = 0.01  # < 1 cm
        dynamic_threshold = 0.1  # > 10 cm
        very_dynamic_threshold = 0.5  # > 50 cm
    elif evaluation_setting == "panoptic-multiview":
        assert n_point_dim == 3
        distance_thresholds = [0.05, 0.10, 0.20, 0.40]  # from 5 cm to 80 cm
        survival_distance_threshold = 1.0  # 1 m
        static_threshold = None
        dynamic_threshold = None
        very_dynamic_threshold = None
    elif evaluation_setting == "tapvid2d":
        assert n_point_dim == 2
        distance_thresholds = [1, 2, 4, 8, 16]  # pixels
        survival_distance_threshold = 50
        static_threshold = None
        dynamic_threshold = None
        very_dynamic_threshold = None
    elif evaluation_setting == "2dpt_ablation":
        assert n_point_dim == 2
        distance_thresholds = [1, 2, 4, 8, 16]  # pixels
        survival_distance_threshold = 50
        static_threshold = 1
        dynamic_threshold = 1
        very_dynamic_threshold = 50
    else:
        raise NotImplementedError

    if verbose:
        logging.info(f"n_frames: {n_frames}, n_tracks: {n_tracks}")
        logging.info(f"GT TRACKS (min, max): {gt_tracks.min()}, {gt_tracks.max()}")
        logging.info(f"query_poits_xyz (min, max): {query_points_xyz.min()}, {query_points_xyz.max()}")

    df_model, df_model_per_track = evaluate_predictions(
        gt_tracks * track_upscaling_factor,
        gt_visibilities,
        pred_tracks * track_upscaling_factor,
        ~pred_visibilities,
        np.concatenate([query_points[:, 0:1], query_points[:, 1:] * track_upscaling_factor], axis=-1),
        distance_thresholds=distance_thresholds,
        survival_distance_threshold=survival_distance_threshold,
        static_threshold=static_threshold,
        dynamic_threshold=dynamic_threshold,
        very_dynamic_threshold=very_dynamic_threshold,
    )

    if verbose:
        logging.info(f"DF Model:\n{df_model}")
        logging.info(f"DF Model:\n{df_model.loc[['average_pts_within_thresh', 'survival']]}")

    # Save to results_dict
    results_dict = {}

    # For dynamic points, report all metrics
    for point_type in ["dynamic-static-mean", "dynamic", "very_dynamic", "static", "any"]:
        if f'all_{point_type}' not in df_model.columns:
            continue
        for metric in sorted(df_model.index):
            results_dict[f'{prefix}/model__{metric}__{point_type}'] = df_model.loc[metric, f'all_{point_type}']

    # For other point types, report only selected metrics
    for point_type in []:
        if f'all_{point_type}' not in df_model.columns:
            continue
        for metric in ["average_pts_within_thresh", "survival", "occlusion_accuracy", "average_jaccard"]:
            results_dict[f'{prefix}/model__{metric}__{point_type}'] = df_model.loc[metric, f'all_{point_type}']

    for k in results_dict:
        results_dict[k] = results_dict[k].item()

    if verbose:
        logging.info(f"3DPT results:\n{results_dict}")

    if add_per_track_results:
        results_dict[f'{prefix}/model__per_track_results'] = df_model_per_track

    return results_dict


class Evaluator:
    def __init__(
            self,
            rerun_viz_indices: Optional[Iterable[int]] = None,
            forward_pass_log_indices: Optional[Iterable[int]] = None,
            mp4_track_viz_indices: Optional[Iterable[int]] = (0, 3, 4, 5),
    ) -> None:
        """
        Initializes the Evaluator.

        Parameters
        ----------
        rerun_viz_indices : Optional[Iterable[int]]
            Indices of datapoints for which rerun 3D visualizations should be saved.
            If None, no rerun visualizations will be logged.

        forward_pass_log_indices : Optional[Iterable[int]]
            Indices of datapoints for which debug logs from the model's forward pass should be saved.
            If None, no forward pass debug logs will be generated.

        mp4_track_viz_indices : Optional[Iterable[int]]
            Indices of datapoints for which 2D trajectory visualizations (MP4 videos) should be saved.
            If None, MP4 visualizations will not be generated.
        """
        self.rerun_viz_indices = rerun_viz_indices
        self.forward_pass_log_indices = forward_pass_log_indices
        self.mp4_track_viz_indices = mp4_track_viz_indices

        if self.rerun_viz_indices is None:
            self.rerun_viz_indices = []
        if self.forward_pass_log_indices is None:
            self.forward_pass_log_indices = []
        if self.mp4_track_viz_indices is None:
            self.mp4_track_viz_indices = []

    @torch.no_grad()
    def evaluate_sequence(
            self,
            model,
            test_dataloader,
            dataset_name,
            log_dir,
            writer: Optional[SummaryWriter] = None,
            step: Optional[int] = 0,
    ):
        metrics = {}
        assert len(test_dataloader) > 0
        total_fps = 0.0
        count = 0
        for datapoint_idx, datapoint in enumerate(tqdm(test_dataloader)):
            should_save_mp4_viz = datapoint_idx in self.mp4_track_viz_indices
            should_save_forward_pass_logs = datapoint_idx in self.forward_pass_log_indices
            should_save_rerun_viz = datapoint_idx in self.rerun_viz_indices

            # Hotfix for debugging: Load an edge-case datapoint directly from disk
            if False:
                # Batch 10060
                datapoint = torch.load("logs/debug/ablation-E07/mvtracker-ptv3-512/crash_batch_step_010060.pt",
                                       map_location="cuda:0")
                # (datapoint.videodepth > 0).float().mean() --> 0

                # Batch 8145
                datapoint = torch.load("logs/ablation-E07/mvtracker-ptv3-512-2/crash_batch_step_008145.pt",
                                       map_location="cuda:0")
                datapoint.videodepth = datapoint.videodepth.clip(0.0, 1000.0)

                should_save_mp4_viz = True
                should_save_rerun_viz = True
                should_save_forward_pass_logs = False
                model.model.use_ptv3 = False

            if isinstance(datapoint, tuple) or isinstance(datapoint, list) and len(datapoint) == 2:
                datapoint, gotit = datapoint
                if not all(gotit):
                    logging.warning("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(datapoint)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Per view data
            rgbs = datapoint.video
            depths = datapoint.videodepth
            depths_conf = datapoint.videodepthconf
            image_features = datapoint.feats
            intrs = datapoint.intrs
            extrs = datapoint.extrs
            gt_trajectories_2d_pixelspace_w_z_cameraspace = datapoint.trajectory
            gt_visibilities_per_view = datapoint.visibility

            query_points_2d = (datapoint.query_points.clone().float().to(device)
                               if datapoint.query_points is not None else None)
            query_points_3d = (datapoint.query_points_3d.clone().float().to(device)
                               if datapoint.query_points_3d is not None else None)

            # Non-per-view data
            gt_trajectories_3d_worldspace = datapoint.trajectory_3d
            valid_tracks_per_frame = datapoint.valid
            track_upscaling_factor = datapoint.track_upscaling_factor
            seq_name = datapoint.seq_name[0]

            # Novel view data
            novel_rgbs = datapoint.novel_video
            novel_intrs = datapoint.novel_intrs
            novel_extrs = datapoint.novel_extrs

            batch_size, num_views, num_frames, _, height, width = rgbs.shape

            # For generic datasets without labels, we will try sampling queries from depthmap points and around origin
            no_tracking_labels = False
            if query_points_2d is None and query_points_3d is None:
                no_tracking_labels = True
                assert batch_size == 1
                assert gt_trajectories_2d_pixelspace_w_z_cameraspace is None
                assert gt_visibilities_per_view is None
                assert gt_trajectories_3d_worldspace is None
                assert valid_tracks_per_frame is None

                assert depths is not None
                assert depths_conf is not None

                # Config: (frame_idx, z_min, z_max, count)
                if "selfcap" in dataset_name:
                    sampling_spec = [
                        (0, -0.1, 0.2, 1.8, 100, ""),
                        (0, 0.2, 2.1, 1.8, 200, ""),
                        # (0, 0.2, 2.1, 1.8, 200, "kmeans"),
                        (36, 0.2, 2.1, 1.8, 200, ""),
                        (120, 0.2, 2.1, 1.8, 200, ""),
                    ]
                    x0, y0, zmin, zmax, radius = 0.25, 0.7, -0.15, 3.6, 1.8
                    xyz, _ = init_pointcloud_from_rgbd(
                        fmaps=depths_conf,
                        depths=depths,
                        intrs=intrs,
                        extrs=extrs,
                        stride=1,
                        level=0,
                        depth_interp_mode="N/A",
                    )
                    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
                    x -= x0
                    y -= y0
                    mask = (x ** 2 + y ** 2 < radius ** 2) & (z >= zmin) & (z <= zmax)
                    mask = mask.reshape(batch_size, num_frames, num_views, height, width).permute(0, 2, 1, 3, 4)
                    # depths[~mask[:, :, :, None, :, :]] = 0.0
                    # depths[depths_conf < 5] = 0.0
                    depths_conf[~mask[:, :, :, None, :, :]] = 2.0

                elif "4d-dress" in dataset_name:
                    sampling_spec = [
                        # (0, -10, +10, 10, 1500, ""),
                        # (0, -10, +10, 10, 500, ""),
                        (0, -10, +10, 10, 300, "kmeans"),
                        # (72, -10, +10, 10, 500, "kmeans"),
                    ]

                elif "hi4d" in dataset_name:
                    sampling_spec = [
                        (0, -np.inf, +np.inf, np.inf, 1000, ""),
                    ]

                else:
                    sampling_spec = [
                        (0, -0.1, +4.2, 2.1, 1000, "kmeans"),
                    ]

                depth_conf_threshold = 0.9
                query_list = []

                for t, zmin, zmax, radius, count, method in sampling_spec:
                    if t >= num_frames:
                        continue  # skip invalid timestep

                    dmap = depths[:, :, t:t + 1]
                    conf = depths_conf[:, :, t:t + 1]
                    xyz, c = init_pointcloud_from_rgbd(
                        fmaps=conf,
                        depths=dmap,
                        intrs=intrs[:, :, t:t + 1],
                        extrs=extrs[:, :, t:t + 1],
                        stride=1,
                        level=0,
                        depth_interp_mode="N/A",
                    )
                    xyz = xyz[0]  # (N, 3)
                    conf = c[0, :, 0]  # (N,)
                    valid = conf > depth_conf_threshold
                    pts = xyz[valid]
                    if pts.numel() == 0:
                        continue

                    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
                    mask = (x ** 2 + y ** 2 < radius ** 2) & (z >= zmin) & (z <= zmax)
                    pts = pts[mask]
                    if pts.numel() == 0:
                        continue

                    if len(pts) >= count:
                        if method == "":
                            pts = pts[torch.randperm(len(pts))[:count]]
                        elif method == "kmeans":
                            pts = kmeans_sample(pts, count)
                        else:
                            raise NotImplementedError

                    t_col = torch.full((len(pts), 1), float(t), device=pts.device)
                    query_list.append(torch.cat([t_col, pts], dim=1))

                # Finalize query points
                query_points_3d = torch.cat(query_list, dim=0)[None]  # (1, N, 4)

                # Dummy GT trajectory
                num_points = query_points_3d.shape[1]
                gt_trajectories_3d_worldspace = query_points_3d[:, None, :, 1:].repeat(1, num_frames, 1, 1)
                gt_trajectories_2d_pixelspace_w_z_cameraspace = torch.stack([
                    torch.cat(world_space_to_pixel_xy_and_camera_z(
                        world_xyz=gt_trajectories_3d_worldspace[0],
                        intrs=intrs[0, view_idx],
                        extrs=extrs[0, view_idx],
                    ), dim=-1)
                    for view_idx in range(num_views)
                ], dim=0).unsqueeze(0)
                d = query_points_3d.device
                gt_visibilities_per_view = torch.ones((batch_size, num_views, num_frames, num_points), dtype=bool).to(d)
                valid_tracks_per_frame = torch.ones((batch_size, num_frames, num_points), dtype=bool).to(d)

            if no_tracking_labels and not any([should_save_mp4_viz,
                                               should_save_rerun_viz,
                                               should_save_forward_pass_logs]):
                continue

            # Assert shapes of per-view data
            num_points = gt_trajectories_2d_pixelspace_w_z_cameraspace.shape[3]
            assert depths is not None, "Depth is required for evaluation."
            assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
            assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
            assert depths_conf is None or depths_conf.shape == (batch_size, num_views, num_frames, 1, height, width)
            assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
            assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)
            assert gt_trajectories_2d_pixelspace_w_z_cameraspace.shape == (
                batch_size, num_views, num_frames, num_points, 3)
            assert gt_visibilities_per_view.shape == (batch_size, num_views, num_frames, num_points)

            # Assert shapes of non-per-view data
            assert query_points_3d.shape == (batch_size, num_points, 4)
            assert gt_trajectories_3d_worldspace.shape == (batch_size, num_frames, num_points, 3)
            assert valid_tracks_per_frame.shape == (batch_size, num_frames, num_points)

            # Dump the RGBs and depths to disk
            if should_save_rerun_viz:
                for v in range(num_views):
                    rgb_path = os.path.join(log_dir, f"rgbs__{dataset_name}--seq-{datapoint_idx}__view-{v}.mp4")
                    depth_path = os.path.join(log_dir, f"depths__{dataset_name}--seq-{datapoint_idx}__view-{v}.mp4")
                    conf_path = os.path.join(log_dir, f"depth_confs__{dataset_name}--seq-{datapoint_idx}__view-{v}.mp4")

                    # Precompute global min/max
                    d_all = depths[0, v, :, 0].reshape(-1, height, width).cpu().numpy()
                    d_min, d_max = d_all.min(), d_all.max()
                    if depths_conf is not None:
                        c_all = depths_conf[0, v, :, 0].reshape(-1, height, width).cpu().numpy()
                        c_min, c_max = c_all.min(), c_all.max()

                    # Colormaps
                    depth_cmap = cm.get_cmap("turbo")
                    conf_cmap = cm.get_cmap("inferno")

                    rgb_video, depth_video, conf_video = [], [], []
                    for t in range(num_frames):
                        rgb = (rgbs[0, v, t].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
                        rgb_video.append(rgb)

                        d = depths[0, v, t, 0].cpu().numpy()
                        d_norm = (d - d_min) / (d_max - d_min + 1e-5)
                        depth_color = (depth_cmap(d_norm)[..., :3] * 255).astype(np.uint8)
                        depth_video.append(depth_color)

                        if depths_conf is not None:
                            c = depths_conf[0, v, t, 0].cpu().numpy()
                            c_norm = (c - c_min) / (c_max - c_min + 1e-5)
                            conf_color = (conf_cmap(c_norm)[..., :3] * 255).astype(np.uint8)
                            conf_video.append(conf_color)

                    if "selfcap-v1" in dataset_name:
                        fps = 12
                    elif "4d-dress" in dataset_name or "egoexo4d" in dataset_name:
                        fps = 30
                    else:
                        fps = 12
                    imageio.mimsave(rgb_path, rgb_video, fps=fps)
                    imageio.mimsave(depth_path, depth_video, fps=fps)
                    if depths_conf is not None:
                        imageio.mimsave(conf_path, conf_video, fps=fps)

            # Run the model
            fwd_kwargs = {
                "rgbs": rgbs,
                "depths": depths,
                "image_features": image_features,
                "query_points_3d": query_points_3d,
                "intrs": intrs,
                "extrs": extrs,
                "save_debug_logs": should_save_forward_pass_logs,
                "debug_logs_path": os.path.join(
                    log_dir, f"forward_pass__eval_{dataset_name}_step-{step}_seq-{datapoint_idx}",
                ),
                "save_rerun_logs": should_save_rerun_viz,
                "save_rerun_logs_output_rrd_path": os.path.join(
                    log_dir, f"rerun__{dataset_name}--seq-{datapoint_idx}--name-{seq_name}--fwd.rrd"
                ),

            }
            if "2dpt" in dataset_name:
                assert batch_size == 1
                query_timestep = query_points_3d[0, :, 0].cpu().numpy().astype(int)
                query_points_view = gt_visibilities_per_view.argmax(dim=1)[0, query_timestep, torch.arange(num_points)]
                fwd_kwargs["query_points_view"] = query_points_view[None]

            start_time = time.time()
            if "shape_of_motion" in log_dir or "dynamic_3dgs" in log_dir:
                if "dynamic_3dgs" in log_dir:
                    cached_output_path = os.path.join(log_dir, f"step-0_seq-{seq_name}_tracks.npz")
                else:
                    cached_output_path = os.path.join(log_dir, f"step-{step}_seq-{seq_name}_tracks.npz")
                cached_output_path = re.sub(r"-novelviews\d+(_\d+)*", "", cached_output_path)
                assert os.path.exists(cached_output_path), cached_output_path
                cached_data = np.load(cached_output_path)
                if "dynamic_3dgs" in log_dir:
                    results = {
                        "traj_e": torch.from_numpy(cached_data["pred_trajectories_3d"]).to(device)[None],
                        "vis_e": torch.from_numpy(cached_data["pred_visibilities_any_view"]).to(device).any(1),
                    }
                else:
                    results = {
                        "traj_e": torch.from_numpy(cached_data["pred_trajectories_3d"]).to(device),
                        "vis_e": torch.from_numpy(cached_data["pred_visibilities_any_view"]).to(device),
                    }
            else:
                results = model(**fwd_kwargs)

            end_time = time.time()
            frames_processed = batch_size * num_frames
            elapsed = end_time - start_time
            fps = frames_processed / elapsed
            logging.info(f"[Datapoint {datapoint_idx}] FPS: {fps:.1f}")
            total_fps += fps
            count += 1

            pred_trajectories = results["traj_e"]
            pred_visibilities = results["vis_e"]
            pred_trajectories_2d = results["traj2d_e"] if "traj2d_e" in results else None
            assert "strided" not in dataset_name, "Strided evaluation is not supported yet."

            # Determine the evaluation setting
            if "kubric" in dataset_name:
                evaluation_setting = "kubric-multiview"
            elif "panoptic-multiview" in dataset_name:
                evaluation_setting = "panoptic-multiview"
            elif "dex-ycb" in dataset_name:
                evaluation_setting = "dexycb-multiview"
            elif "tapvid2d" in dataset_name:
                evaluation_setting = "tapvid2d"
            elif no_tracking_labels:
                evaluation_setting = "no-tracking-labels"
            else:
                raise NotImplementedError

            # Invert the intrinsics and extrinsics matrices
            intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
            extrs_square = torch.eye(4).to(extrs.device)[None].repeat(batch_size, num_views, num_frames, 1, 1)
            extrs_square[:, :, :, :3, :] = extrs
            extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)
            assert intrs_inv.shape == (batch_size, num_views, num_frames, 3, 3)
            assert extrs_inv.shape == (batch_size, num_views, num_frames, 4, 4)

            # Project the predictions to pixel space for visualization
            pred_trajectories_pixel_xy_camera_z_per_view = torch.stack([
                torch.cat(world_space_to_pixel_xy_and_camera_z(
                    world_xyz=pred_trajectories[0],
                    intrs=intrs[0, view_idx],
                    extrs=extrs[0, view_idx],
                ), dim=-1)
                for view_idx in range(num_views)
            ], dim=0)
            for view_idx in range(num_views):
                pred_trajectories_reproduced = pixel_xy_and_camera_z_to_world_space(
                    pixel_xy=pred_trajectories_pixel_xy_camera_z_per_view[view_idx, :, :, :2],
                    camera_z=pred_trajectories_pixel_xy_camera_z_per_view[view_idx, :, :, 2:],
                    intrs_inv=intrs_inv[0, view_idx],
                    extrs_inv=extrs_inv[0, view_idx],
                )
                if not torch.allclose(pred_trajectories_reproduced, pred_trajectories, atol=1):
                    warnings.warn(f"Reprojection of the predicted trajectories failed: "
                                  f"view_idx={view_idx}, "
                                  f"max_diff={torch.max(torch.abs(pred_trajectories_reproduced - pred_trajectories))}")
            pred_trajectories_pixel_xy_camera_z_per_view = pred_trajectories_pixel_xy_camera_z_per_view[None]

            # Compute 3D metrics
            gt_visibilities_any_view = gt_visibilities_per_view.any(dim=1)
            assert gt_visibilities_any_view.any(dim=1).all(), "All points should be visible in at least one view."
            per_track_results = None
            if evaluation_setting in ["kubric-multiview", "panoptic-multiview", "dexycb-multiview"]:
                eval_3dpt_results_dict = evaluate_3dpt(
                    gt_tracks=gt_trajectories_3d_worldspace[0].cpu().numpy(),
                    gt_visibilities=gt_visibilities_any_view[0].cpu().numpy(),
                    query_points=query_points_3d[0].cpu().numpy(),
                    pred_tracks=pred_trajectories[0].cpu().numpy(),
                    pred_visibilities=pred_visibilities[0].cpu().numpy(),
                    evaluation_setting=evaluation_setting,
                    track_upscaling_factor=track_upscaling_factor,
                    prefix=f"eval_{dataset_name}",
                    add_per_track_results=should_save_rerun_viz,
                    verbose=False,
                )
                if should_save_rerun_viz:
                    per_track_results = eval_3dpt_results_dict[f'eval_{dataset_name}/model__per_track_results']
                    del eval_3dpt_results_dict[f'eval_{dataset_name}/model__per_track_results']
                metrics[datapoint_idx] = eval_3dpt_results_dict

                if "2dpt" in dataset_name:
                    assert batch_size == 1
                    if pred_trajectories_2d is None:
                        pred_trajectories_2d = pred_trajectories_pixel_xy_camera_z_per_view[:, :, :, :, :2]
                    _rescale_to_256x256 = np.array([256, 256]) / np.array([width, height])
                    _metrics = {}
                    for view_idx in range(num_views):
                        track_mask = (query_points_view == view_idx).cpu().numpy()
                        if track_mask.sum() == 0:
                            continue
                        _n_tracks = track_mask.sum()
                        _gt_tracks = gt_trajectories_2d_pixelspace_w_z_cameraspace[0, view_idx, :, track_mask, :2]
                        _gt_tracks = _gt_tracks.cpu().numpy()
                        _gt_visibilities = gt_visibilities_per_view[0, view_idx, :, track_mask].cpu().bool().numpy()
                        _query_t = query_timestep[track_mask]
                        _query_xy = _gt_tracks[_query_t, np.arange(_n_tracks)]
                        _query = np.concatenate([_query_t[:, None], _query_xy], axis=-1)
                        _pred_tracks = pred_trajectories_2d[0, view_idx, :, track_mask].cpu().numpy()
                        _pred_visibilities = np.zeros_like(_gt_visibilities)
                        assert _gt_visibilities[_query_t, np.arange(_n_tracks)].all()
                        eval_2dpt_results_dict = evaluate_3dpt(
                            gt_tracks=_gt_tracks,
                            gt_visibilities=_gt_visibilities,
                            query_points=_query,
                            pred_tracks=_pred_tracks,
                            pred_visibilities=_pred_visibilities,
                            evaluation_setting="2dpt_ablation",
                            track_upscaling_factor=_rescale_to_256x256,
                            prefix=f"eval_{dataset_name}",
                            add_per_track_results=False,
                            verbose=False,
                        )
                        tapvid2d_original_metrics = compute_tapvid_metrics_original(
                            query_points=np.concatenate([_query_t[:, None], _query_xy * _rescale_to_256x256], axis=-1),
                            gt_occluded=~_gt_visibilities[None].transpose(0, 2, 1),
                            gt_tracks=_gt_tracks[None].transpose(0, 2, 1, 3) * _rescale_to_256x256,
                            pred_occluded=~_pred_visibilities[None].transpose(0, 2, 1),
                            pred_tracks=_pred_tracks[None].transpose(0, 2, 1, 3) * _rescale_to_256x256,
                            query_mode="first",
                        )
                        tapvid2d_original_metrics = {
                            f"eval_{dataset_name}/model__tapvid2d_{k}":
                                (tapvid2d_original_metrics[k] * 100).round(2).item()
                            for k in sorted(tapvid2d_original_metrics)
                        }
                        _metrics[view_idx] = {}
                        _metrics[view_idx].update(eval_2dpt_results_dict)
                        _metrics[view_idx].update(tapvid2d_original_metrics)
                        _metrics[view_idx] = {
                            k.replace("model__", "model__2dpt__"): v
                            for k, v in _metrics[view_idx].items()
                            if "jaccard" not in k and "occlusion" not in k
                        }
                    _metrics_avg = {}
                    for k in _metrics[next(iter(_metrics.keys()))]:
                        _metrics_avg[k] = np.mean([
                            _metrics[view_idx][k]
                            for view_idx in _metrics
                            if k in _metrics[view_idx]
                        ]).round(2)

                    metrics[datapoint_idx].update(_metrics_avg)
                    for view_idx in _metrics:
                        metrics[datapoint_idx].update({
                            f"{k}__view-{view_idx}": v
                            for k, v in _metrics[view_idx].items()
                        })

            # Compute 2D metrics
            elif evaluation_setting in ["tapvid2d"]:
                assert num_views == 1
                if pred_trajectories_2d is None:
                    pred_trajectories_2d = pred_trajectories_pixel_xy_camera_z_per_view[:, :, :, :, :2]
                eval_2dpt_results_dict = evaluate_3dpt(
                    gt_tracks=gt_trajectories_2d_pixelspace_w_z_cameraspace[0, 0, :, :, :2].cpu().numpy(),
                    gt_visibilities=gt_visibilities_per_view[0, 0].cpu().bool().numpy(),
                    query_points=query_points_2d[0].cpu().numpy(),
                    pred_tracks=pred_trajectories_2d[0, 0].cpu().numpy(),
                    pred_visibilities=pred_visibilities[0].cpu().numpy(),
                    evaluation_setting=evaluation_setting,
                    track_upscaling_factor=track_upscaling_factor,
                    prefix=f"eval_{dataset_name}",
                    add_per_track_results=should_save_rerun_viz,
                    verbose=False,
                )
                if should_save_rerun_viz:
                    per_track_results = eval_2dpt_results_dict[f'eval_{dataset_name}/model__per_track_results']
                    del eval_2dpt_results_dict[f'eval_{dataset_name}/model__per_track_results']
                metrics[datapoint_idx] = eval_2dpt_results_dict

                tapvid2d_original_metrics = compute_tapvid_metrics_original(
                    query_points_2d[0].cpu().numpy(),
                    torch.logical_not(gt_visibilities_per_view[:, 0].clone().permute(0, 2, 1)).cpu().numpy(),
                    gt_trajectories_2d_pixelspace_w_z_cameraspace[:, 0, :, :, :2].clone().permute(0, 2, 1,
                                                                                                  3).cpu().numpy(),
                    torch.logical_not(pred_visibilities.clone().permute(0, 2, 1)).cpu().numpy(),
                    pred_trajectories_2d[:, 0].permute(0, 2, 1, 3).cpu().numpy(),
                    query_mode="first",
                )
                tapvid2d_original_metrics = {
                    f"eval_{dataset_name}/model__tapvid2d_{k}": (tapvid2d_original_metrics[k] * 100).round(2).item()
                    for k in sorted(tapvid2d_original_metrics)
                }
                metrics[datapoint_idx].update(tapvid2d_original_metrics)

            elif evaluation_setting in ["no-tracking-labels"]:
                metrics[datapoint_idx] = {}

            np.savez(
                os.path.join(log_dir, f"step-{step}_seq-{seq_name}_tracks.npz"),
                gt_trajectories_2d=gt_trajectories_2d_pixelspace_w_z_cameraspace.cpu().numpy(),
                gt_trajectories_3d=gt_trajectories_3d_worldspace.cpu().numpy(),
                gt_visibilities_per_view=gt_visibilities_per_view.cpu().numpy(),
                gt_visibilities_any_view=gt_visibilities_any_view.cpu().numpy(),
                pred_trajectories_2d=pred_trajectories_pixel_xy_camera_z_per_view.cpu().numpy(),
                pred_trajectories_3d=pred_trajectories.cpu().numpy(),
                pred_visibilities_any_view=pred_visibilities.cpu().numpy(),
                query_points_2d=query_points_2d.cpu().numpy() if query_points_2d is not None else None,
                query_points_3d=query_points_3d.cpu().numpy(),
                track_upscaling_factor=track_upscaling_factor,
            )

            # Visualize the results with rerun.io
            viz_fps = 30
            if "panoptic" in dataset_name:
                viz_fps = 30
            elif "dex" in dataset_name:
                viz_fps = 10
            elif "kubric" in dataset_name:
                viz_fps = 12

            if should_save_rerun_viz:
                # Log the visualizations to rerun
                if "mvtracker" in log_dir:
                    method_id = 0
                    method_name = "MVTracker"

                elif "spatracker_mono" in log_dir:
                    method_id = 1
                    method_name = "SpatialTrackerV1"

                elif "tapip3d" in log_dir:
                    method_id = 2
                    method_name = "TAPIP3D"

                elif "spatracker_multi" in log_dir:
                    method_id = 3
                    method_name = "Triplane"

                else:
                    method_id = None
                    method_name = "x"

                if "panoptic" in dataset_name:
                    sphere_radius = 12
                else:
                    sphere_radius = 6.0

                max_tracks = None
                if "dress" in dataset_name:
                    max_tracks = 300
                elif "panoptic" in dataset_name:
                    max_tracks = 100
                elif "kubric" in dataset_name or "dex-ycb" in dataset_name:
                    max_tracks = 36

                LogConfig = namedtuple("LogConfig", [
                    "suffix", "method_id", "max_tracks", "track_batch_size", "sphere_radius",
                    "conf_thrs", "log_only_confident_pc", "memory_lightweight_logging"
                ])
                log_configs = [
                    LogConfig(
                        suffix="",
                        method_id=None,
                        max_tracks=None,
                        track_batch_size=50,
                        sphere_radius=None,
                        conf_thrs=[1.0, 5.0],
                        log_only_confident_pc=False,
                        memory_lightweight_logging=False,
                    ),
                    LogConfig(
                        suffix=".comparisons",
                        method_id=method_id,
                        max_tracks=100,
                        track_batch_size=50,
                        sphere_radius=None,
                        conf_thrs=[1.0, 5.0],
                        log_only_confident_pc=False,
                        memory_lightweight_logging=True,
                    ),
                    LogConfig(
                        suffix=".lightweight",
                        method_id=None,
                        max_tracks=max_tracks,
                        track_batch_size=50,
                        sphere_radius=sphere_radius,
                        conf_thrs=[5.0],
                        log_only_confident_pc=True,
                        memory_lightweight_logging=True,
                    ),
                    LogConfig(
                        suffix=".lightweight.comparisons",
                        method_id=method_id,
                        max_tracks=50,
                        track_batch_size=50,
                        sphere_radius=sphere_radius,
                        conf_thrs=[5.0],
                        log_only_confident_pc=True,
                        memory_lightweight_logging=True,
                    ),
                ]

                for cfg in log_configs:
                    logfile_name = f"rerun__{dataset_name}--seq-{datapoint_idx}--name-{seq_name}--eval{cfg.suffix}.rrd"
                    rr.init("3dpt", recording_id="v0.16")

                    if cfg.method_id is None or cfg.method_id == 0:
                        log_pointclouds_to_rerun(
                            dataset_name=dataset_name,
                            datapoint_idx=datapoint_idx,
                            rgbs=rgbs,
                            depths=depths,
                            intrs=intrs,
                            extrs=extrs,
                            depths_conf=depths_conf,
                            conf_thrs=cfg.conf_thrs,
                            log_only_confident_pc=cfg.log_only_confident_pc,
                            radii=-2.45,
                            fps=viz_fps,
                            bbox_crop=None,
                            sphere_radius_crop=cfg.sphere_radius,
                            sphere_center_crop=np.array([0, 0, 0]),
                            log_rgb_image=not cfg.memory_lightweight_logging,
                            log_depthmap_as_image_v1=False,
                            log_depthmap_as_image_v2=False,
                            log_camera_frustrum=True,
                            log_rgb_pointcloud=True,
                        )

                    log_tracks_to_rerun(
                        dataset_name=dataset_name,
                        datapoint_idx=datapoint_idx,
                        predictor_name=method_name,
                        gt_trajectories_3d_worldspace=None if no_tracking_labels else gt_trajectories_3d_worldspace,
                        gt_visibilities_any_view=None if no_tracking_labels else gt_visibilities_any_view,
                        query_points_3d=query_points_3d,
                        pred_trajectories=pred_trajectories,
                        pred_visibilities=pred_visibilities,
                        per_track_results=per_track_results,
                        radii_scale=1.0,
                        fps=viz_fps,
                        sphere_radius_crop=cfg.sphere_radius,
                        sphere_center_crop=np.array([0, 0, 0]),
                        log_per_interval_results=False,
                        max_tracks_to_log=cfg.max_tracks,
                        track_batch_size=cfg.track_batch_size,
                        method_id=cfg.method_id,
                        memory_lightweight_logging=cfg.memory_lightweight_logging,
                    )

                    rr_rrd_path = os.path.join(log_dir, logfile_name)
                    rr.save(rr_rrd_path)
                    logging.info(f"Saved Rerun recording to: {rr_rrd_path}")

            # Visualize the results as mp4
            if should_save_mp4_viz:
                log_mp4_track_viz(
                    log_dir=log_dir,
                    dataset_name=dataset_name,
                    datapoint_idx=datapoint_idx,
                    rgbs=rgbs,
                    intrs=intrs,
                    extrs=extrs,
                    gt_trajectories=gt_trajectories_3d_worldspace,
                    gt_visibilities=gt_visibilities_any_view,
                    pred_trajectories=pred_trajectories,
                    pred_visibilities=pred_visibilities,
                    query_points_3d=query_points_3d,
                    step=step,
                    prefix="comparison__v4a-train__",
                    max_tracks_to_visualize=36,
                    max_individual_tracks_to_visualize=6,
                )
                if novel_rgbs is not None:
                    log_mp4_track_viz(
                        log_dir=log_dir,
                        dataset_name=dataset_name,
                        datapoint_idx=datapoint_idx,
                        rgbs=novel_rgbs,
                        intrs=novel_intrs,
                        extrs=novel_extrs,
                        gt_trajectories=gt_trajectories_3d_worldspace,
                        gt_visibilities=gt_visibilities_any_view,
                        pred_trajectories=pred_trajectories,
                        pred_visibilities=pred_visibilities,
                        query_points_3d=query_points_3d,
                        step=step,
                        prefix="comparison__v4b-novel__",
                        max_tracks_to_visualize=36,
                        max_individual_tracks_to_visualize=0,
                    )

            metrics[datapoint_idx]["fps"] = fps

            try:
                params_total = sum(p.numel() for p in model.parameters())
                params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                params_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
                metrics[datapoint_idx]["params_total"] = params_total
                metrics[datapoint_idx]["params_trainable"] = params_trainable
                metrics[datapoint_idx]["params_non_trainable"] = params_non_trainable
            except Exception as e:
                logging.info(f"Error calculating model parameters: {e}")

        # Compute average
        if count > 0:
            avg_fps = total_fps / count
            logging.info(f"\nAverage FPS across {count} datapoints: {avg_fps:.1f}")
        else:
            logging.warning("No datapoints were processed.")

        return metrics
