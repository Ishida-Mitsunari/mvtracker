import logging
import os
import sys
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from mvtracker.datasets.utils import transform_scene
from mvtracker.models.core.model_utils import bilinear_sample2d, pixel_xy_and_camera_z_to_world_space
from mvtracker.utils.visualizer_mp4 import Visualizer

_COTRACKER_LOCAL_REPOS = (
    "/share/tgp/yangyi/co-tracker",
    os.path.expanduser("~/.cache/torch/hub/facebookresearch_co-tracker_main"),
)


def _load_cotracker_hub(model_name: str):
    """Load official CoTracker weights; prefer a local clone so eval does not need GitHub."""
    last_err = None
    for repo in _COTRACKER_LOCAL_REPOS:
        hubconf = os.path.join(repo, "hubconf.py")
        if not os.path.isfile(hubconf):
            continue
        try:
            return torch.hub.load(repo, model_name, source="local", trust_repo=True)
        except Exception as exc:
            last_err = exc
            logging.warning("Local CoTracker load from %s failed: %s", repo, exc)
    try:
        return torch.hub.load("facebookresearch/co-tracker", model_name, trust_repo=True)
    except Exception as exc:
        if last_err is not None:
            raise exc from last_err
        raise


class CoTrackerOfflineWrapper(nn.Module):
    def __init__(self, model_name="cotracker3_offline", grid_size=10):
        super(CoTrackerOfflineWrapper, self).__init__()
        self.grid_size = grid_size
        self.cotracker = _load_cotracker_hub(model_name)

    def forward(self, rgbs, queries, **kwargs):
        T, _, H, W = rgbs.shape
        N, _ = queries.shape

        assert rgbs.shape == (T, 3, H, W)
        assert queries.shape == (N, 3)

        # Forward pass: https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/predictor.py#L36
        pred_tracks, pred_visibility = self.cotracker(
            video=rgbs[None].float(),
            queries=queries[None].float(),
            grid_size=self.grid_size,
        )
        vis = pred_visibility[0, :, :N]
        if vis.ndim > 2:
            vis = vis.squeeze(-1)
        return {"traj_2d": pred_tracks[0, :, :N, :2], "vis": vis}


class CoTrackerOnlineWrapper(nn.Module):
    def __init__(self, model_name="cotracker3_online", grid_size=10):
        super(CoTrackerOnlineWrapper, self).__init__()
        self.grid_size = grid_size
        self.cotracker = _load_cotracker_hub(model_name)

    def forward(self, rgbs, queries, **kwargs):
        T, _, H, W = rgbs.shape
        N, _ = queries.shape

        assert rgbs.shape == (T, 3, H, W)
        assert queries.shape == (N, 3)

        # Forward pass: https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/predictor.py#L230
        self.cotracker(
            video_chunk=rgbs[None].float(),
            queries=queries[None].float(),
            grid_size=self.grid_size,
            is_first_step=True,
        )
        for t in range(0, T - self.cotracker.step, self.cotracker.step):
            pred_tracks, pred_visibility = self.cotracker(video_chunk=rgbs[None, t: t + self.cotracker.step * 2])

        vis = pred_visibility[0, :, :N]
        if vis.ndim > 2:
            vis = vis.squeeze(-1)
        return {"traj_2d": pred_tracks[0, :, :N, :2], "vis": vis}


class SpaTrackerV2Wrapper(nn.Module):
    """
    Environment setup:
    ```bash
    git clone https://github.com/henry123-boy/SpaTrackerV2.git ../spatialtrackerv2
    cd ../spatialtrackerv2
    git checkout 1673230
    git submodule update --init --recursive
    pip install pycolmap==3.11.1
    pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d
    pip install pyceres==2.4

    # Update the threshold for weighted_procrustes_torch from 1e-3 to 5e-3
    sed -i 's/(torch.det(R) - 1).abs().max() < 1e-3/(torch.det(R) - 1).abs().max() < 5e-3/' ./models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py

    # Verify the change: this should print a line with 5e-3
    cat ./models/SpaTrackV2/models/tracker3D/spatrack_modules/utils.py | grep "(torch.det(R) - 1).abs().max()"
    ```
    """

    def __init__(
            self,
            model_type="offline",
            vo_points=756,
    ):
        super(SpaTrackerV2Wrapper, self).__init__()

        spatrack_root = os.environ.get("SPATRACKERV2_ROOT", "/share/tgp/yangyi/spatialtrackerv2")
        if spatrack_root not in sys.path:
            sys.path.insert(0, spatrack_root)
        import utils3d

        def _spatrack_depth_edge(depth, rtol=None, atol=None, mask=None, **kwargs):
            # Current utils3d renamed depth_edge -> depth_map_edge and @batched(2)
            # treats the last two dims as H,W. SpaTrackerV2 still passes (B,1,H,W);
            # leaving the channel in the batch dims broadcasts to (B,B,B,H,W).
            squeeze_c = depth.ndim == 4 and depth.shape[1] == 1
            if squeeze_c:
                depth = depth[:, 0]
                if mask is not None and mask.ndim == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0]
            edge = utils3d.torch.depth_map_edge(depth, rtol=rtol, atol=atol, mask=mask, **kwargs)
            if squeeze_c:
                edge = edge[:, None]
            return edge

        utils3d.torch.depth_edge = _spatrack_depth_edge
        from models.SpaTrackV2.models.predictor import Predictor
        if model_type == "offline":
            self.model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
        elif model_type == "online":
            self.model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.model.spatrack.track_num = vo_points  # the track_num is the number of points in the grid
        self.model.eval()
        self.model.to("cuda")

    def forward(self, rgbs, depths, queries, queries_xyz_worldspace, intrs, extrs, **kwargs):
        T, _, H, W = rgbs.shape
        N, _ = queries.shape

        assert rgbs.shape == (T, 3, H, W)
        assert depths.shape == (T, 1, H, W)
        assert intrs.shape == (T, 3, 3)
        assert extrs.shape == (T, 3, 4)
        assert queries.shape == (N, 3)
        assert queries_xyz_worldspace.shape == (N, 4)

        extrs_square = torch.eye(4).to(extrs.device)[None].repeat(T, 1, 1)
        extrs_square[:, :3, :] = extrs

        # Transform the extrinsics so that the camera is in the origin, and later revert.
        transform = extrs_square[0]
        transform_inv = torch.inverse(transform)
        extrs, queries_xyz_worldspace = extrs.clone(), queries_xyz_worldspace.clone()
        (
            _, extrs, queries_xyz_worldspace, _, _
        ) = transform_scene(1, transform[:3, :3], transform[:3, 3], None, extrs[None], queries_xyz_worldspace)
        extrs = extrs[0]
        extrs_square[:, :3, :] = extrs

        # Check if the camera is fixed
        extrs_delta = torch.linalg.norm(extrs - extrs[0], dim=(1, 2))
        fixed_cam = (extrs_delta < 1e-3).all().item()

        # Run inference
        extrs_inv = torch.inverse(extrs_square)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                c2w_traj, intrs, point_map, conf_depth,
                track3d_pred, track2d_pred, vis_pred, conf_pred, video
            ) = self.model.forward(rgbs.cpu(), depth=depths.squeeze(1).cpu().numpy(),
                                   intrs=intrs.cpu(), extrs=extrs_inv.cpu().numpy(),
                                   queries=queries.cpu().numpy(), queries_3d=queries_xyz_worldspace.cpu().numpy(),
                                   fps=1, full_point=True, iters_track=4,
                                   query_no_BA=True, fixed_cam=fixed_cam, stage=1, unc_metric=None,
                                   support_frame=T - 1, replace_ratio=0.2)

        trajectories_3d = (
                torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3].to(track3d_pred.device), track3d_pred[:, :, :3])
                + c2w_traj[:, :3, 3][:, None, :].to(track3d_pred.device)
        )
        (
            _, _, _, trajectories_3d, _
        ) = transform_scene(1, transform_inv[:3, :3], transform_inv[:3, 3], None, None, None, trajectories_3d, None)
        visibilities = vis_pred.squeeze(2)

        assert trajectories_3d.shape == (T, N, 3)
        assert visibilities.shape == (T, N)

        return {"traj_2d": None, "traj_3d_worldspace": trajectories_3d, "vis": visibilities}


class LocoTrackWrapper(nn.Module):
    """
    Environment setup:
    ```sh
    git clone https://github.com/cvlab-kaist/locotrack ../locotrack
    cd ../locotrack
    find ./locotrack_pytorch -type f -name "*.py" -exec sed -i 's/\bimport models\b/import locotrack_pytorch.models/g' {} \;
    find ./locotrack_pytorch -type f -name "*.py" -exec sed -i 's/\bfrom models\b/from locotrack_pytorch.models/g' {} \;
    cd ../spatialtracker
    ```
    """

    def __init__(self, model_size="base"):
        super(LocoTrackWrapper, self).__init__()
        sys.path.append("../locotrack")
        from locotrack_pytorch.models.locotrack_model import load_model
        self.model = load_model(model_size=model_size).cuda()
        self.model.eval()

    def forward(self, rgbs, queries, **kwargs):
        T, _, H, W = rgbs.shape
        N, _ = queries.shape

        assert (H, W) == (256, 256), f"LocoTrack only supports (256, 256) images, but got ({H}, {W})"
        assert rgbs.shape == (T, 3, H, W)
        assert queries.shape == (N, 3)

        # Forward pass: https://github.com/cvlab-kaist/locotrack/blob/6f3f9cad46b06c3de9c38fbf21006271056baf45/locotrack_pytorch/models/locotrack_model.py#L323
        video = (rgbs.permute(0, 2, 3, 1)[None] / 255.0) * 2 - 1
        queries_tyx = torch.stack([queries[:, 0], queries[:, 2], queries[:, 1]], dim=1)[None]
        # queries_tyx = queries_tyx / torch.tensor([1, H, W], dtype=queries_tyx.dtype, device=queries_tyx.device)

        with torch.no_grad():
            output = self.model(video=video, query_points=queries_tyx)
        pred_occ = torch.sigmoid(output['occlusion'])
        if 'expected_dist' in output:
            pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(output['expected_dist']))
        pred_occ = (pred_occ > 0.5)[0]

        trajectories_2d = output['tracks'][0].permute(1, 0, 2)
        # trajectories_2d[..., 0] *= W
        # trajectories_2d[..., 1] *= H
        visibilities = ~pred_occ.permute(1, 0)

        if torch.isnan(trajectories_2d).any():
            warnings.warn(
                f"Found {torch.isnan(trajectories_2d).sum()}/{trajectories_2d.numel()} NaN values in trajectories_2d. Setting them to 0.")
            trajectories_2d[trajectories_2d.isnan()] = 0
        if torch.isnan(visibilities).any():
            warnings.warn(
                f"Found {torch.isnan(visibilities).sum()}/{visibilities.numel()} NaN values in visibilities. Setting them to 1.")
            visibilities[visibilities.isnan()] = 1

        return {"traj_2d": trajectories_2d, "vis": visibilities}


class TAPTRWrapper(nn.Module):
    pass


class TAPIRWrapper(nn.Module):
    pass


class PIPSWrapper(nn.Module):
    pass


class PIPSPlusPlusWrapper(nn.Module):
    pass


class SceneTrackerWrapper(nn.Module):
    """
    Environment setup:
    ```sh
    wget --directory-prefix=checkpoints https://huggingface.co/wwcreator/SceneTracker/resolve/main/scenetracker-odyssey-200k.pth
    git clone https://github.com/wwsource/SceneTracker.git ../scenetracker

    python eval.py experiment_path=logs/scenetracker model=scenetracker

    ```
    """

    def __init__(
            self,
            ckpt="checkpoints/scenetracker-odyssey-200k.pth",
            return_2d_track=False,
    ):
        super(SceneTrackerWrapper, self).__init__()

        sys.path.append("../scenetracker/")
        from model.model_scenetracker import SceneTracker
        model = SceneTracker()
        pre_replace_list = [['module.', '']]
        checkpoint = torch.load(ckpt)
        for l in pre_replace_list:
            checkpoint = {k.replace(l[0], l[1]): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
        model.eval().cuda()

        self.return_2d_track = return_2d_track
        self.model = model

    def forward(self, rgbs, depths, queries_with_z, **kwargs):
        T, _, H, W = rgbs.shape
        N, _ = queries_with_z.shape

        assert rgbs.shape == (T, 3, H, W)
        assert depths.shape == (T, 1, H, W)
        assert queries_with_z.shape == (N, 4)

        trajs_uv_e, trajs_z_e, _, _ = self.model.infer(
            self.model,
            input_list=[
                rgbs[None].float(),
                depths[None].float(),
                queries_with_z[None].float(),
            ],
            iters=4,
            is_train=False,
        )

        trajectories_2d = trajs_uv_e[0].type(queries_with_z.dtype)
        trajectories_z = trajs_z_e[0].type(queries_with_z.dtype)
        visibilities = torch.zeros_like(trajectories_2d[..., 0], dtype=torch.bool)

        if self.return_2d_track:
            return {"traj_2d": trajectories_2d, "vis": visibilities}
        else:
            return {"traj_2d": trajectories_2d, "traj_z": trajectories_z, "vis": visibilities}


class DELTAWrapper(nn.Module):
    """
    Environment setup:
    ```sh
    mkdir -p ./checkpoints/
    gdown --fuzzy https://drive.google.com/file/d/18d5M3nl3AxbG4ZkT7wssvMXZXbmXrnjz/view?usp=sharing -O ./checkpoints/ # 3D ckpt
    gdown --fuzzy https://drive.google.com/file/d/1S_T7DzqBXMtr0voRC_XUGn1VTnPk_7Rm/view?usp=sharing -O ./checkpoints/ # 2D ckpt
    git clone --recursive https://github.com/snap-research/DELTA_densetrack3d ../delta
    pip install jaxtyping

    python eval.py experiment_path=logs/delta model=delta
    ```
    """

    def __init__(
            self,
            ckpt="checkpoints/densetrack3d.pth",
            upsample_factor=4,
            grid_size=20,
            return_2d_track=False,
    ):
        super(DELTAWrapper, self).__init__()

        self.grid_size = grid_size
        self.return_2d_track = return_2d_track

        sys.path.append("../delta")
        from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
        from densetrack3d.models.predictor.predictor import Predictor3D
        model = DenseTrack3D(
            stride=4,
            window_len=16,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=(384, 512),
            upsample_factor=upsample_factor
        )
        with open(ckpt, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        predictor = Predictor3D(model=model)
        predictor = predictor.eval().cuda()
        self.model = model
        self.predictor = predictor

    def forward(self, rgbs, depths, queries, **kwargs):
        T, _, H, W = rgbs.shape
        N, _ = queries.shape

        assert rgbs.shape == (T, 3, H, W)
        assert depths.shape == (T, 1, H, W)
        assert queries.shape == (N, 3)

        out_dict = self.predictor(
            rgbs[None],
            depths[None],
            queries=queries[None],
            segm_mask=None,
            grid_size=self.grid_size,
            grid_query_frame=0,
            backward_tracking=False,
            predefined_intrs=None
        )

        trajectories_2d = out_dict["trajs_uv"][0]
        trajectories_z = out_dict["trajs_depth"][0]
        trajectories_3d = out_dict["trajs_3d_dict"]["coords"][0]
        visibilities = out_dict["vis"][0]

        if self.return_2d_track:
            return {"traj_2d": trajectories_2d, "vis": visibilities}
        else:
            return {"traj_2d": trajectories_2d, "traj_z": trajectories_z, "vis": visibilities}


class TAPIP3DWrapper(nn.Module):
    """
    Environment setup:
    ```sh
    wget --directory-prefix=checkpoints https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
    git clone git@github.com:zbw001/TAPIP3D.git ../tapip3d
    cd ../tapip3d
    git checkout 9359ae236f16a58a103dc1c55ad1919360dc6f8b
    cd third_party/pointops2
    LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
    cd ../..
    """

    def __init__(
            self,
            ckpt="checkpoints/tapip3d_final.pth",
            num_iters=6,
            grid_size=8,
            resolution_factor=2,
            transform_to_camera_space=False,
    ):
        super(TAPIP3DWrapper, self).__init__()

        self.num_iters = num_iters
        self.support_grid_size = grid_size
        self.resolution_factor = resolution_factor
        self.transform_to_camera_space = transform_to_camera_space

        tapip3d_root = os.environ.get("TAPIP3D_ROOT", "/share/tgp/yangyi/tapip3d")
        if tapip3d_root not in sys.path:
            sys.path.insert(0, tapip3d_root)
        # CoTrackerCNNEncoder would torch.hub.load facebook/co-tracker just to
        # init fnet; TAPIP3D's checkpoint then overwrites those weights.
        import models.encoders.cotracker_cnn as _tapip_cnn

        _orig_enc_init = _tapip_cnn.CoTrackerCNNEncoder.__init__

        def _init_without_hub(self, resolution, output_dim, stride, pretrained, freeze_mode):
            return _orig_enc_init(self, resolution, output_dim, stride, False, freeze_mode)

        _tapip_cnn.CoTrackerCNNEncoder.__init__ = _init_without_hub
        from utils.inference_utils import load_model
        self.model = load_model(ckpt)
        self.model.cuda()

        inference_res = (
            int(self.model.image_size[0] * np.sqrt(self.resolution_factor)),
            int(self.model.image_size[1] * np.sqrt(self.resolution_factor)),
        )
        self.model.set_image_size(inference_res)

    def forward(self, rgbs, depths, intrs, extrs, queries_xyz_worldspace, **kwargs):
        T, _, H, W = rgbs.shape
        N, _ = queries_xyz_worldspace.shape

        assert rgbs.shape == (T, 3, H, W)
        assert depths.shape == (T, 1, H, W)
        assert intrs.shape == (T, 3, 3)
        assert extrs.shape == (T, 3, 4)
        assert queries_xyz_worldspace.shape == (N, 4)

        extrs_square = torch.eye(4).to(extrs.device)[None].repeat(T, 1, 1)
        extrs_square[:, :3, :] = extrs

        # Transform the extrinsics (and query points) so that
        # the camera is in the origin, and later revert.
        # But it's about the same performance either way.
        if self.transform_to_camera_space:
            T = extrs_square[0]
            T_inv = torch.inverse(T)
            extrs = extrs.clone()
            (
                _, extrs, queries_xyz_worldspace, _, _
            ) = transform_scene(1, T[:3, :3], T[:3, 3], None, extrs[None], queries_xyz_worldspace, None, None)
            extrs = extrs[0]
            extrs_square[:, :3, :] = extrs

        # Run inference
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trajectories_3d, visibilities = TAPIP3DWrapper.inference(
                model=self.model,
                video=rgbs / 255.0,
                depths=depths.squeeze(1),
                intrinsics=intrs,
                extrinsics=extrs_square,
                query_point=queries_xyz_worldspace,
                num_iters=self.num_iters,
                grid_size=self.support_grid_size,
            )

        if self.transform_to_camera_space:
            (
                _, _, _, trajectories_3d, _
            ) = transform_scene(1, T_inv[:3, :3], T_inv[:3, 3], None, None, None, trajectories_3d, None)

        if N == 1:
            trajectories_3d = trajectories_3d.unsqueeze(1)
            visibilities = visibilities.unsqueeze(1)
        assert trajectories_3d.shape == (T, N, 3)
        assert visibilities.shape == (T, N)

        return {"traj_2d": None, "traj_3d_worldspace": trajectories_3d.clone(), "vis": visibilities.clone()}

    @staticmethod
    @torch.no_grad()
    def inference(
            *,
            model: torch.nn.Module,
            video: torch.Tensor,
            depths: torch.Tensor,
            intrinsics: torch.Tensor,
            extrinsics: torch.Tensor,
            query_point: torch.Tensor,
            num_iters: int = 6,
            grid_size: int = 8,
            bidrectional: bool = True,
            vis_threshold=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from utils.inference_utils import _inference_with_grid
        from einops import repeat

        _depths = depths.clone()
        _depths = _depths[_depths > 0].reshape(-1)
        q25 = torch.kthvalue(_depths, int(0.25 * len(_depths))).values
        q75 = torch.kthvalue(_depths, int(0.75 * len(_depths))).values
        iqr = q75 - q25
        _depth_roi = torch.tensor(
            [1e-7, (q75 + 1.5 * iqr).item()],
            dtype=torch.float32,
            device=video.device
        )

        T, C, H, W = video.shape
        assert depths.shape == (T, H, W)
        N = query_point.shape[0]

        model.set_image_size((H, W))

        preds, _ = _inference_with_grid(
            model=model,
            video=video[None],
            depths=depths[None],
            intrinsics=intrinsics[None],
            extrinsics=extrinsics[None],
            query_point=query_point[None],
            num_iters=num_iters,
            depth_roi=_depth_roi,
            grid_size=grid_size
        )

        if bidrectional and not model.bidirectional and (query_point[..., 0] > 0).any():
            preds_backward, _ = _inference_with_grid(
                model=model,
                video=video[None].flip(dims=(1,)),
                depths=depths[None].flip(dims=(1,)),
                intrinsics=intrinsics[None].flip(dims=(1,)),
                extrinsics=extrinsics[None].flip(dims=(1,)),
                query_point=torch.cat([T - 1 - query_point[..., :1], query_point[..., 1:]], dim=-1)[None],
                num_iters=num_iters,
                depth_roi=_depth_roi,
                grid_size=grid_size,
            )
            preds.coords = torch.where(
                repeat(torch.arange(T, device=video.device), 't -> b t n 3', b=1, n=N) < repeat(
                    query_point[..., 0][None], 'b n -> b t n 3', t=T, n=N),
                preds_backward.coords.flip(dims=(1,)),
                preds.coords
            )
            preds.visibs = torch.where(
                repeat(torch.arange(T, device=video.device), 't -> b t n', b=1, n=N) < repeat(
                    query_point[..., 0][None], 'b n -> b t n', t=T, n=N),
                preds_backward.visibs.flip(dims=(1,)),
                preds.visibs
            )

        coords, visib_logits = preds.coords, preds.visibs
        visibs = torch.sigmoid(visib_logits)
        if vis_threshold is not None:
            visibs = visibs >= vis_threshold
        return coords.squeeze(), visibs.squeeze()


class MonocularToMultiViewAdapter(nn.Module):
    def __init__(self, model, **kwargs):
        super(MonocularToMultiViewAdapter, self).__init__()
        self.model = model

    def forward(
            self,
            rgbs,
            depths,
            query_points,
            intrs,
            extrs,
            save_debug_logs=False,
            debug_logs_path="",
            query_points_view=None,
            **kwargs,
    ):
        batch_size, num_views, num_frames, _, height, width = rgbs.shape
        _, num_points, _ = query_points.shape

        assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
        assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
        assert query_points.shape == (batch_size, num_points, 4)
        assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
        assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)

        # Project the queries to each view
        query_points_t = query_points[:, :, :1].long()
        query_points_xyz_worldspace = query_points[:, :, 1:]

        query_points_xy_pixelspace_per_view = query_points.new_zeros((batch_size, num_views, num_points, 2))
        query_points_z_cameraspace_per_view = query_points.new_zeros((batch_size, num_views, num_points, 1))
        for batch_idx in range(batch_size):
            for t in query_points_t[batch_idx].unique():
                query_points_t_mask = query_points_t[batch_idx].squeeze(-1) == t
                point_3d_world = query_points_xyz_worldspace[batch_idx][query_points_t_mask]

                # World to camera space
                point_4d_world_homo = torch.cat(
                    [point_3d_world, point_3d_world.new_ones(point_3d_world[..., :1].shape)], -1)
                point_3d_camera = torch.einsum('Aij,Bj->ABi', extrs[batch_idx, :, t, :, :], point_4d_world_homo[:, :])

                # Camera to pixel space
                point_2d_pixel_homo = torch.einsum('Aij,ABj->ABi', intrs[batch_idx, :, t, :, :], point_3d_camera[:, :])
                point_2d_pixel = point_2d_pixel_homo[..., :2] / point_2d_pixel_homo[..., 2:]

                query_points_xy_pixelspace_per_view[batch_idx, :, query_points_t_mask] = point_2d_pixel
                query_points_z_cameraspace_per_view[batch_idx, :, query_points_t_mask] = point_3d_camera[..., -1:]

        # Estimate occlusion mask in each view based on depth maps
        query_points_depth_in_view = query_points.new_zeros((batch_size, num_views, num_points, 1))
        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                for t in query_points_t[batch_idx].unique():
                    query_points_t_mask = query_points_t[batch_idx].squeeze(-1) == t
                    interpolated_depth = bilinear_sample2d(
                        im=depths[batch_idx, view_idx, t][None],
                        x=query_points_xy_pixelspace_per_view[batch_idx, view_idx, query_points_t_mask, 0][None],
                        y=query_points_xy_pixelspace_per_view[batch_idx, view_idx, query_points_t_mask, 1][None],
                    )[0].permute(1, 0).type(query_points.dtype)
                    query_points_depth_in_view[batch_idx, view_idx, query_points_t_mask] = interpolated_depth

        query_points_depth_in_view_masked = query_points_depth_in_view.clone()
        query_points_outside_of_view_box = (
                (query_points_xy_pixelspace_per_view[..., 0] < 0) |
                (query_points_xy_pixelspace_per_view[..., 0] >= width) |
                (query_points_xy_pixelspace_per_view[..., 1] < 0) |
                (query_points_xy_pixelspace_per_view[..., 1] >= height) |
                (query_points_z_cameraspace_per_view[..., 0] < 0)
        )
        if query_points_outside_of_view_box.all(1).any():
            warnings.warn(f"There are some query points that are outside of the frame of every view: "
                          f"{query_points_xy_pixelspace_per_view[query_points_outside_of_view_box.all(1)[:, None, :].repeat(1, num_views, 1)].reshape(num_views, -1, 2).permute(1, 0, 2)}")
        query_points_depth_in_view_masked[query_points_outside_of_view_box] = -1e4
        query_points_best_visibility_view = (
                query_points_depth_in_view_masked - query_points_z_cameraspace_per_view).argmax(1)
        query_points_best_visibility_view = query_points_best_visibility_view.squeeze(-1)

        if query_points_view is not None:
            if query_points_view.dim() == 1:
                query_points_view = query_points_view[None]
            query_points_best_visibility_view = query_points_view.long()
            logging.info(f"Using the provided query_points_view instead of the estimated one")

        assert batch_size == 1, "Batch size > 1 is not supported yet"
        batch_idx = 0

        # Call the 2D tracker for each view
        traj_e_per_view = {}
        traj2d_e_per_view = {}
        vis_e_per_view = {}
        for view_idx in range(num_views):
            track_mask = query_points_best_visibility_view[batch_idx] == view_idx
            if track_mask.sum() == 0:
                continue

            view_rgbs = rgbs[batch_idx, view_idx]
            view_depths = depths[batch_idx, view_idx]
            view_intrs = intrs[batch_idx, view_idx]
            view_extrs = extrs[batch_idx, view_idx]
            view_query_points = torch.concat([
                query_points_t[batch_idx, :, :][track_mask],
                query_points_xy_pixelspace_per_view[batch_idx, view_idx, :, :][track_mask],
            ], dim=-1)
            view_query_points_with_z = torch.concat([
                query_points_t[batch_idx, :, :][track_mask],
                query_points_xy_pixelspace_per_view[batch_idx, view_idx, :, :][track_mask],
                query_points_z_cameraspace_per_view[batch_idx, view_idx, :][track_mask],
            ], dim=-1)
            view_query_points_xyz_worldspace = torch.concat([
                query_points_t[batch_idx, :, :][track_mask],
                query_points_xyz_worldspace[batch_idx, :][track_mask],
            ], dim=-1)

            results = self.model(
                rgbs=view_rgbs,
                depths=view_depths,
                intrs=view_intrs,
                extrs=view_extrs,
                queries=view_query_points,
                queries_with_z=view_query_points_with_z,
                queries_xyz_worldspace=view_query_points_xyz_worldspace,
            )
            view_traj_2d = results.get("traj_2d")
            view_traj_e = view_traj_2d
            view_vis_e = results["vis"]

            if save_debug_logs and view_traj_e is not None:
                visualizer = Visualizer(
                    save_dir=debug_logs_path,
                    pad_value=16,
                    fps=12,
                    show_first_frame=0,
                    tracks_leave_trace=3,
                )
                visualizer.visualize(
                    video=view_rgbs[None].cpu(),
                    tracks=view_traj_e[None].cpu(),
                    visibility=view_vis_e[None].cpu(),
                    filename=f"view_{view_idx}.mp4",
                    query_frame=query_points_t[batch_idx, :, 0][track_mask][None],
                    save_video=True,
                )

            # Project the trajectories to the world space
            if "traj_3d_worldspace" in results:
                view_traj_e = results["traj_3d_worldspace"]
            else:
                if "traj_z" in results:
                    view_camera_z = results["traj_z"]
                else:
                    view_camera_z = bilinear_sampler(view_depths, view_traj_e.reshape(num_frames, -1, 1, 2))[:, 0, :, :]

                view_intrs = intrs[batch_idx, view_idx]
                view_extrs = extrs[batch_idx, view_idx]
                intrs_inv = torch.inverse(view_intrs.float())
                view_extrs_square = torch.eye(4).to(view_extrs.device)[None].repeat(num_frames, 1, 1)
                view_extrs_square[:, :3, :] = view_extrs
                extrs_inv = torch.inverse(view_extrs_square.float())
                view_traj_e = pixel_xy_and_camera_z_to_world_space(
                    pixel_xy=view_traj_e[..., :2].float(),
                    camera_z=view_camera_z.float(),
                    intrs_inv=intrs_inv,
                    extrs_inv=extrs_inv,
                )

            # Set the trajectory to (0,0,0) for the timesteps before the query timestep
            for point_idx, t in enumerate(query_points_t[batch_idx, :, :].squeeze(-1)[track_mask]):
                view_traj_e[:t, point_idx, :] = 0.0
                if view_traj_2d is not None:
                    view_traj_2d[:t, point_idx, :] = 0.0

            traj_e_per_view[view_idx] = view_traj_e[None]
            vis_e_per_view[view_idx] = view_vis_e[None]
            if view_traj_2d is not None:
                traj2d_e_per_view[view_idx] = view_traj_2d[None]

        # Merging the results from all views
        views_to_keep = list(traj_e_per_view.keys())
        traj_e = torch.cat([traj_e_per_view[view_idx] for view_idx in views_to_keep], dim=2)
        vis_e = torch.cat([vis_e_per_view[view_idx] for view_idx in views_to_keep], dim=2)
        traj2d_e = None
        if len(traj2d_e_per_view) == len(views_to_keep) and len(views_to_keep) > 0:
            traj2d_e = torch.cat([traj2d_e_per_view[view_idx] for view_idx in views_to_keep], dim=2)

        # Sort the traj_e and vis_e based on the original indices, since concatenating the results from all views
        # will first put the results from the first view, then the results from the second view, and so on.
        # But we want to keep the trajectories order to match the original query points order.
        sort_inds = []
        for view_idx in views_to_keep:
            track_mask = query_points_best_visibility_view[batch_idx] == view_idx
            if track_mask.sum() == 0:
                continue
            global_indices = torch.nonzero(track_mask).squeeze(-1)
            sort_inds += [global_indices]
        sort_inds = torch.cat(sort_inds, dim=0)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)

        # Use the inv_sort_inds to sort the traj_e and vis_e
        traj_e = traj_e[:, :, inv_sort_inds]
        vis_e = vis_e[:, :, inv_sort_inds]
        if traj2d_e is not None:
            traj2d_e = traj2d_e[:, :, inv_sort_inds]

        # Save to results
        results = {"traj_e": traj_e, "vis_e": vis_e}
        if traj2d_e is not None:
            results["traj2d_e"] = traj2d_e
        return results


# From https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/models/core/model_utils.py#L286
def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )
