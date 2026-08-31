# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

torch.set_float32_matmul_precision('high')

from lightning.fabric.wrappers import _unwrap_objects
from mvtracker.datasets.generic_scene_dataset import GenericSceneDataset

from torch.utils.tensorboard import SummaryWriter
import gpustat
import json
import threading
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch.optim as optim
import wandb
from lightning.fabric import Fabric
from lightning.fabric.utilities import AttributeDict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import signal, sys

from mvtracker.datasets import KubricMultiViewDataset
from mvtracker.datasets import TapVidDataset
from mvtracker.datasets import kubric_multiview_dataset
from mvtracker.datasets.dexycb_multiview_dataset import DexYCBMultiViewDataset
from mvtracker.datasets.panoptic_studio_multiview_dataset import PanopticStudioMultiViewDataset
from mvtracker.datasets.utils import collate_fn, dataclass_to_cuda_
from mvtracker.models.core.losses import balanced_ce_loss, sequence_loss_3d
from mvtracker.models.core.model_utils import world_space_to_pixel_xy_and_camera_z, pixel_xy_and_camera_z_to_world_space
from mvtracker.models.evaluation_predictor_3dpt import EvaluationPredictor as EvaluationPredictor3D
from mvtracker.utils.visualizer_mp4 import MultiViewVisualizer, Visualizer
from mvtracker.cli.utils import extras
from mvtracker.cli.utils.helpers import maybe_close_wandb

import logging
import os

import torch
import time
from collections import deque
from torchdata.stateful_dataloader import StatefulDataLoader


def fetch_optimizer(trainer_cfg, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=trainer_cfg.lr, weight_decay=trainer_cfg.wdecay)
    if trainer_cfg.anneal_strategy in ["linear", "cos"]:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            trainer_cfg.lr,
            trainer_cfg.num_steps + 100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy=trainer_cfg.anneal_strategy,
        )
    elif trainer_cfg.anneal_strategy == "restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=trainer_cfg.lr / 1000,
        )

    return optimizer, scheduler


def forward_batch_multi_view(batch, model, cfg, step, train_iters, gamma, save_debug_logs=False, debug_logs_path=''):
    # Per view data
    rgbs = batch.video
    depths = batch.videodepth
    image_features = batch.feats
    intrs = batch.intrs
    extrs = batch.extrs
    gt_trajectories_2d_pixelspace_w_z_cameraspace = batch.trajectory
    gt_visibilities_per_view = batch.visibility
    query_points_3d = batch.query_points_3d

    # Non-per-view data
    gt_trajectories_3d_worldspace = batch.trajectory_3d
    valid_tracks_per_frame = batch.valid
    track_upscaling_factor = batch.track_upscaling_factor

    batch_size, num_views, num_frames, _, height, width = rgbs.shape
    num_points = gt_trajectories_2d_pixelspace_w_z_cameraspace.shape[3]

    # Assert shapes of per-view data
    assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
    assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
    assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
    assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)
    assert gt_trajectories_2d_pixelspace_w_z_cameraspace.shape == (batch_size, num_views, num_frames, num_points, 3)
    assert gt_visibilities_per_view.shape == (batch_size, num_views, num_frames, num_points)

    # Assert shapes of non-per-view data
    assert query_points_3d.shape == (batch_size, num_points, 4)
    assert gt_trajectories_3d_worldspace.shape == (batch_size, num_frames, num_points, 3)
    assert valid_tracks_per_frame.shape == (batch_size, num_frames, num_points)

    gt_visibilities_any_view = gt_visibilities_per_view.any(dim=1)
    assert gt_visibilities_any_view.any(dim=1).all(), "All points should be visible at in least one frame."

    for batch_idx in range(batch_size):
        for point_idx in range(num_points):
            t = query_points_3d[batch_idx, point_idx, 0].long().item()
            valid_tracks_per_frame[batch_idx, :t, point_idx] = False

    # Run the model
    results = model(
        rgbs=rgbs,
        depths=depths,
        image_features=image_features,
        query_points=query_points_3d,
        iters=train_iters,
        is_train=True,
        intrs=intrs,
        extrs=extrs,
        save_debug_logs=save_debug_logs,
        debug_logs_path=debug_logs_path,
    )
    pred_trajectories = results["traj_e"]
    pred_visibilities = results["vis_e"]
    vis_predictions = results["train_data"]["vis_predictions"]
    coord_predictions = results["train_data"]["coord_predictions"]
    p_idx_end_list = results["train_data"]["p_idx_end_list"]
    sort_inds = results["train_data"]["sort_inds"]

    # Prepare the ground truth for the loss functions,
    # which expect the data to be in the sliding-window
    vis_gts = []
    traj_gts = []
    valids_gts = []
    query_points_t_min = query_points_3d[:, :, 0].long().min()
    for i, wind_p_idx_end in enumerate(p_idx_end_list):
        gt_visibilities_any_view_sorted = gt_visibilities_any_view[:, :, sort_inds]
        gt_trajectories_3d_worldspace_sorted = gt_trajectories_3d_worldspace[:, :, sort_inds]
        valid_tracks_per_frame_sorted = valid_tracks_per_frame[:, :, sort_inds]
        ind = query_points_t_min + i * (cfg.model.sliding_window_len // 2)
        vis_gts.append(gt_visibilities_any_view_sorted[:, ind: ind + cfg.model.sliding_window_len, :wind_p_idx_end])
        traj_gts.append(
            gt_trajectories_3d_worldspace_sorted[:, ind: ind + cfg.model.sliding_window_len, :wind_p_idx_end])
        valids_gts.append(valid_tracks_per_frame_sorted[:, ind: ind + cfg.model.sliding_window_len, :wind_p_idx_end])

    # Compute the losses
    logging.info(f"[DEBUG] "
                 f"{step=} "
                 f"{track_upscaling_factor=} "
                 f"{coord_predictions[0][0][0, 0, 0]=} "
                 f"{coord_predictions[-1][0][0, 0, 0]=} "
                 f"{vis_predictions[0][0, 0, 0]=} "
                 f"{vis_predictions[-1][0, 0, 0]=}")
    xyz_loss = sequence_loss_3d(coord_predictions, traj_gts, vis_gts, valids_gts, gamma) * track_upscaling_factor
    vis_loss = balanced_ce_loss(vis_predictions, vis_gts, valids_gts)

    # Compute 3DPT metrics
    # eval_3dpt_results_dict = evaluate_3dpt(
    #     gt_tracks=gt_trajectories_3d_worldspace[0].cpu().numpy(),
    #     gt_visibilities=gt_visibilities_any_view[0].cpu().numpy(),
    #     pred_tracks=pred_trajectories[0].detach().cpu().numpy(),
    #     pred_visibilities=(pred_visibilities[0] > 0.5).detach().cpu().numpy(),
    #     evaluation_setting="kubric-multiview",
    #     track_upscaling_factor=track_upscaling_factor,
    #     prefix="train_3dpt",
    #     verbose=False,
    #     query_points=query_points_3d[0].cpu().numpy(),
    # )

    # Invert the intrinsics and extrinsics matrices
    intrs_inv = torch.inverse(intrs.float())
    extrs_square = torch.eye(4).to(extrs.device)[None].repeat(batch_size, num_views, num_frames, 1, 1)
    extrs_square[:, :, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float())

    # Project the predictions to pixel space
    pred_trajectories = pred_trajectories[0].detach()
    pred_trajectories_pixel_xy_camera_z_per_view = torch.stack([
        torch.cat(world_space_to_pixel_xy_and_camera_z(
            world_xyz=pred_trajectories,
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

    logging.info(
        f"{step=}, "
        f"seq={batch.seq_name}, "
        f"{xyz_loss.item()=}, "
        f"{vis_loss.item()=}, "
    )

    output = {
        "flow": {
            "loss": xyz_loss * 1.0,
            "predictions": pred_trajectories_pixel_xy_camera_z_per_view,
            "predictions_worldspace": pred_trajectories,
        },
        "visibility": {
            "loss": vis_loss * cfg.trainer.visibility_loss_weight,
            "predictions": pred_visibilities[0].detach(),
        },
        # "metrics": {
        #     k: v
        #     for k, v in eval_3dpt_results_dict.items()
        #     if "per_track" not in k
        # },
    }
    return output


def run_test_eval(cfg, evaluator, model, dataloaders, writer, step):
    if len(dataloaders) == 0:
        return

    logging.info(f"Eval – GPU usage A: {gpustat.new_query()}")

    log_dir = cfg.experiment_path
    model.eval()
    for ds_name, dataloader in dataloaders:
        if ds_name.startswith("kubric"):
            predictor_settings = cfg.evaluation.predictor_settings["kubric"]
        elif ds_name.startswith("dex-ycb"):
            predictor_settings = cfg.evaluation.predictor_settings["dex_ycb"]
        elif ds_name.startswith("panoptic"):
            predictor_settings = cfg.evaluation.predictor_settings["panoptic"]
        elif ds_name.startswith("tapvid2d-davis"):
            predictor_settings = cfg.evaluation.predictor_settings["tapvid2d-davis"]
        else:
            predictor_settings = cfg.evaluation.predictor_settings["generic"]
            logging.info(f"Using generic predictor settings for dataset with name {ds_name}")

        predictor = EvaluationPredictor3D(
            multiview_model=model,
            interp_shape=cfg.evaluation.interp_shape,
            single_point="single" in ds_name,
            n_iters=cfg.evaluation.eval_iters,
            **predictor_settings
        )

        log_dir_ds = os.path.join(log_dir, f"eval_{ds_name}")
        os.makedirs(log_dir_ds, exist_ok=True)

        if cfg.evaluation.consume_model_stats and hasattr(model, "init_stats"):
            model.init_stats()
        metrics = evaluator.evaluate_sequence(
            model=predictor,
            test_dataloader=dataloader,
            dataset_name=ds_name,
            writer=writer,
            step=step,
            log_dir=log_dir_ds,
        )
        if cfg.evaluation.consume_model_stats and hasattr(model, "consume_stats"):
            model.consume_stats()

        metrics_to_log = {
            k: np.nanmean([v[k] for v in metrics.values() if k in v]).round(2)
            for k in metrics[0].keys()
        }
        for k, v in metrics_to_log.items():
            writer.add_scalar(k, v, step)

        with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.max_colwidth', None,
                'display.width', None,
        ):
            logging.info(f"Per-sequence Metrics for {ds_name}: {pd.DataFrame(metrics)}")
            logging.info(f"Average metrics for {ds_name}: {json.dumps(metrics_to_log, indent=4)}")

        # Save metrics to csv
        if log_dir_ds is not None:
            df = pd.DataFrame(metrics)
            df = df.T
            assert df.map(lambda x: (len(x) == 1) if isinstance(x, np.ndarray) else True).all().all()
            df = df.map(lambda x: x[0] if isinstance(x, np.ndarray) or isinstance(x, list) else x)
            df.to_csv(f"{log_dir_ds}/step-{step}_metrics.csv")

            df = pd.DataFrame(metrics_to_log, index=["score"])
            df = df.T
            df.to_csv(f"{log_dir_ds}/step-{step}_metrics_avg.csv")
            logging.info(f"Saved metrics to {log_dir_ds}/step-{step}_metrics_avg.csv")
        # logging.info(f"Eval – GPU usage (after {ds_name}): {gpustat.new_query()}")

    # logging.info(f"Eval – GPU usage B: {gpustat.new_query()}")
    del predictor
    del metrics
    # logging.info(f"Eval – GPU usage C: {gpustat.new_query()}")
    torch.cuda.empty_cache()
    # logging.info(f"Eval – GPU usage D: {gpustat.new_query()}")

    model.train()


def augment_train_iters(train_iters: int, current_step: int, warmup_steps: int = 1000) -> int:
    """
    Adaptive iteration scheduler with warmup:
    - During warmup_steps: always return 1
    - After warmup:
        - 10% chance: return 1
        - 15% chance: return random int in [2, train_iters - 1]
        - 75% chance: return train_iters
    """
    if current_step < warmup_steps or train_iters <= 1:
        return 1

    rng = torch.Generator().manual_seed(current_step)
    p = torch.rand(1, generator=rng).item()

    if p < 0.10:
        return 1
    elif p < 0.25 and train_iters > 2:
        mid_candidates = list(range(2, train_iters))
        idx = torch.randint(len(mid_candidates), (1,), generator=rng).item()
        return mid_candidates[idx]
    else:
        return train_iters


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
@maybe_close_wandb
def main(cfg: DictConfig):
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)
    Path(cfg.experiment_path).mkdir(exist_ok=True, parents=True)

    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    devices = int(os.environ.get("SLURM_GPUS_PER_NODE", torch.cuda.device_count()))
    logging.info(f"SLURM job num nodes: {num_nodes}")
    logging.info(f"SLURM tasks per node (devices): {devices}")

    from lightning.fabric.strategies import DDPStrategy
    fabric = Fabric(
        num_nodes=num_nodes,
        devices=devices,
        precision=cfg.trainer.precision,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    fabric.launch()
    fabric.seed_everything(cfg.reproducibility.seed, workers=True)
    if cfg.reproducibility.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.autograd.set_detect_anomaly(True)

    if cfg.logging.get("log_wandb", False) and fabric.global_rank == 0:
        exp_name = cfg.experiment_path.replace("./logs/", "").replace("/", "_").replace("\\", "_")
        wandb.init(
            project=cfg.logging.wandb_project,
            name=exp_name,
            tags=cfg.logging.get("tags", []),
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    original_numpy = torch.Tensor.numpy

    def patched_numpy(self, *args, **kwargs):
        if self.dtype == torch.bfloat16:
            return original_numpy(self.float(), *args, **kwargs)
        return original_numpy(self, *args, **kwargs)

    torch.Tensor.numpy = patched_numpy

    eval_dataloaders = []
    for dataset_name in cfg.datasets.eval.names:
        if dataset_name.startswith("tapvid2d-davis-"):
            eval_dataset = TapVidDataset.from_name(dataset_name, cfg.datasets.root)
        elif dataset_name.startswith("kubric-multiview-v3-25views"):
            kubric_kwargs = {
                "data_root": os.path.join(cfg.datasets.root, "kubric_multiview_003", "kubric_25_view"),
                "seq_len": 24,
                "traj_per_sample": 200,
                "seed": 72,
                "sample_vis_1st_frame": True,
                "tune_per_scene": False,
                "max_videos": 30,
                "use_duster_depths": False,
                "duster_views": None,
                "clean_duster_depths": False,
                "views_to_return": list(range(20)),
                "novel_views": list(range(20, 25)),
                "num_views": -1,
                "depth_noise_std": 0,
            }
            eval_dataset = KubricMultiViewDataset(**kubric_kwargs)
        elif dataset_name.startswith("kubric-multiview-v3"):
            eval_dataset = KubricMultiViewDataset.from_name(dataset_name, cfg.datasets.root, cfg)
        elif dataset_name.startswith("panoptic-multiview"):
            eval_dataset = PanopticStudioMultiViewDataset.from_name(dataset_name, cfg.datasets.root)
        elif dataset_name.startswith("dex-ycb-multiview"):
            eval_dataset = DexYCBMultiViewDataset.from_name(dataset_name, cfg.datasets.root)
        elif dataset_name == "egoexo4d":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/egoexo4d-processed/maxframes-300_downsample-1_downscale-512/",
                drop_first_n_frames=44,
            )
        elif dataset_name == "4d-dress":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/4d-dress-processed-resized-512-selection",
                use_duster_depths=False,
            )
        elif dataset_name == "hi4d":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/hi4d-processed-resized-512",
                use_duster_depths=False,
                use_vggt_depths_with_aligned_cameras=True,
            )
        elif dataset_name == "selfcap-v1":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-10_downscale-512/",
                drop_first_n_frames=72,
            )
        elif dataset_name == "selfcap-v2":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-True_startframe-90_maxframes-256_downsample-10_downscale-512/",
                drop_first_n_frames=72,
            )
        elif dataset_name == "selfcap-v3":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-20_downscale-512/",
                drop_first_n_frames=36,
            )
        elif dataset_name == "selfcap-v4":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-30_downscale-512/",
                drop_first_n_frames=24,
            )
        elif dataset_name == "selfcap-v5":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-5_downscale-512/",
                drop_first_n_frames=144,
            )
        elif dataset_name == "selfcap-v6":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-2560_downsample-10_downscale-512/",
                drop_first_n_frames=44,
            )
        elif dataset_name == "selfcap-v7":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-4-seq-False_startframe-90_maxframes-256_downsample-10_downscale-512/",
                drop_first_n_frames=72,
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for evaluation.")
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.datasets.eval.num_workers,
            collate_fn=collate_fn,
        )
        eval_dataloaders.append((dataset_name, eval_dataloader))

    # # Let each rank handle a subset of the evaluation dataloaders
    # eval_dataloaders_for_rank = []
    # for idx, (dset_name, dset_loader) in enumerate(eval_dataloaders):
    #     if (idx % fabric.world_size) == fabric.global_rank:
    #         eval_dataloaders_for_rank.append((dset_name, fabric.setup_dataloaders(dset_loader)))
    # eval_dataloaders = eval_dataloaders_for_rank

    train_viz_save_dir = os.path.join(cfg.experiment_path, f"train_{cfg.datasets.train.name}")
    os.makedirs(train_viz_save_dir, exist_ok=True)
    visualizer = MultiViewVisualizer(
        save_dir=train_viz_save_dir,
        pad_value=16,
        fps=12,
        show_first_frame=0,
        tracks_leave_trace=0,
    )

    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)

    if cfg.modes.do_initial_static_pretrain and not cfg.modes.eval_only:
        pretraining_datasets = [
            kubric_multiview_dataset.KubricMultiViewDataset(
                data_root=os.path.join(cfg.datasets.root, "kubric_multiview_003", "train"),
                traj_per_sample=cfg.datasets.train.traj_per_sample,
                ratio_dynamic=0.1,
                ratio_very_dynamic=0.0,
                num_views=4,
                enable_cropping_augs=cfg.augmentations.cropping,

                seq_len=seq_len,
                static_cropping=static_cropping,
                max_videos=max_videos,
            )
            for seq_len, static_cropping, max_videos in [
                (12, True, 500),
                (18, True, 500),
                (24, True, 1000),
                (24, False, 2000),
            ]
        ]
        pretraining_dataset = torch.utils.data.ConcatDataset(pretraining_datasets)
        pretraining_dataloader = StatefulDataLoader(
            pretraining_dataset,
            batch_size=cfg.datasets.train.batch_size,
            shuffle=False,
            num_workers=cfg.datasets.train.num_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            collate_fn=collate_fn,
            drop_last=True,
            in_order=cfg.reproducibility.deterministic,
        )
        pretraining_dataloader = fabric.setup_dataloaders(pretraining_dataloader)
    else:
        pretraining_dataloader = None

    if cfg.modes.eval_only:
        train_dataset = None
    elif cfg.datasets.train.name.startswith("kubric-multiview-v3"):
        train_dataset = KubricMultiViewDataset.from_name(cfg.datasets.train.name, cfg.datasets.root, cfg, fabric)
    else:
        raise ValueError(f"Dataset {cfg.datasets.train.name} not supported for training")

    if not cfg.modes.eval_only:
        train_loader = StatefulDataLoader(
            train_dataset,
            batch_size=cfg.datasets.train.batch_size,
            shuffle=True,
            num_workers=cfg.datasets.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=4 if cfg.datasets.train.num_workers > 0 else None,
            in_order=cfg.reproducibility.deterministic,
        )
        # eval_dataloaders += [("kubric-multiview-v3-training", train_loader)]
        train_loader = fabric.setup_dataloaders(train_loader)
        logging.info(f"LEN TRAIN LOADER={len(train_loader)}")
        num_epochs = cfg.trainer.num_steps // len(train_loader) + 1 + (1 if cfg.modes.do_initial_static_pretrain else 0)
        if cfg.modes.do_initial_static_pretrain:
            cfg.trainer.num_steps += len(pretraining_dataloader)
    else:
        train_loader = None
        num_epochs = None

    epoch = -1
    total_steps = 0

    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model.cuda()
    optimizer, scheduler = fetch_optimizer(cfg.trainer, model)
    model, optimizer = fabric.setup(model, optimizer)

    folder_ckpts = [
        f
        for f in os.listdir(cfg.experiment_path)
        if f.endswith(".pth")
           and not os.path.isdir(f)
           and not "final" in f
           and not "unwrap_model" in f
           and not "unwrap_module" in f
    ]
    logging.info(f"Found {len(folder_ckpts)} checkpoints: {folder_ckpts}")
    if len(folder_ckpts) > 0:
        # We can load this checkpoint directly since we have saved it during training
        ckpt_name = sorted(folder_ckpts)[-1]
        experiment_path = os.path.join(cfg.experiment_path, ckpt_name)
        state = AttributeDict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
        )
        logging.info(f"Total steps before loading checkpoint: {total_steps}")
        fabric.load(experiment_path, state)
        total_steps = state.total_steps  # Integers are immutable, so they cannot be changed inplace
        if train_loader is not None:
            epoch = total_steps // len(train_loader) - 1
        logging.info(f"Loaded checkpoint {experiment_path} (total_steps={total_steps})")
        logging.info(f"Total steps after loading checkpoint: {total_steps}")

    elif cfg.restore_ckpt_path is not None:
        restore_ckpt_path = cfg.restore_ckpt_path
        assert restore_ckpt_path.endswith(".pth")
        logging.info(f"Restoring pre-trained weights from {os.path.abspath(restore_ckpt_path)}")
        training_ckpt = "total_steps" in torch.load(restore_ckpt_path)
        if training_ckpt:
            # Loading a checkpoint saved by fabric during training
            logging.info("Trying to load as a training checkpoint...")
            state = AttributeDict(model=model)
            try:
                fabric.load(restore_ckpt_path, state, strict=True)
            except RuntimeError as e:
                logging.warning(f"Failed to load weights with from {restore_ckpt_path} with strict=True: {e}. "
                                f"Trying again with strict=False.")
                fabric.load(restore_ckpt_path, state, strict=False)
            logging.info(f"Loaded checkpoint {restore_ckpt_path}")
        else:
            fabric.load_raw(restore_ckpt_path, model)

    tb_writer = SummaryWriter(log_dir=os.path.join(cfg.experiment_path, f"runs_{fabric.global_rank}"))
    if cfg.modes.eval_only or cfg.modes.validate_at_start:
        run_test_eval(cfg, evaluator, model, eval_dataloaders, tb_writer, total_steps - 1)
        fabric.barrier()
        if cfg.modes.eval_only:
            return

    total_durations = deque()
    dataloader_durations = deque()
    fwd_durations = deque()
    sync_durations = deque()
    bwd_durations = deque()
    timing_log_freq = 100

    def handle_sigterm(signum, frame):
        logging.error(f"Signal {signum} received, saving checkpoint and exiting...")
        ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
        save_path = Path(f"{cfg.experiment_path}/model_{ckpt_iter}.pth")
        state = AttributeDict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps + 1,
        )
        fabric.save(save_path, state)
        logging.info(f"Saved checkpoint to {save_path}. Waiting for all ranks to finish...")
        fabric.barrier()
        logging.info(f"Calling sys.exit(0) now.")
        sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    logging.info(f"Registered signal handlers for SIGUSR1 and SIGTERM.")

    model.train()
    should_keep_training = True if cfg.trainer.num_steps > 0 else False
    total_batches_loaded = 0
    total_batches_failed = 0
    if fabric.global_rank == 0:
        tqdm_total_steps = tqdm(
            total=cfg.trainer.num_steps,
            desc=f"Total Training Progress (rank={fabric.global_rank})",
            unit="batch",
            initial=total_steps,
            position=0,
        )
    threads = []
    had_run_pretraining_epoch = cfg.modes.do_initial_static_pretrain and total_steps > len(pretraining_dataloader)
    logging.info(f"{total_steps=}, {epoch=}/{num_epochs}, {had_run_pretraining_epoch=}")
    while should_keep_training:
        epoch += 1
        i_batch = -1

        if cfg.modes.do_initial_static_pretrain and not had_run_pretraining_epoch:
            had_run_pretraining_epoch = True
            data_iter = iter(pretraining_dataloader)
            n_batches = len(pretraining_dataloader)
        else:
            data_iter = iter(train_loader)
            n_batches = len(train_loader)
        if fabric.global_rank == 0:
            tqdm_epoch = tqdm(total=n_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", position=1)

        while i_batch < n_batches:
            start_time_1 = time.time()
            logging.info(f"Gonna load batch {i_batch + 1}/{n_batches} (rank={fabric.global_rank})")
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                n_batches = len(train_loader)
                batch = next(data_iter)

            batch, gotit = batch
            total_batches_loaded += 1

            if cfg.modes.debugging_hotfix_datapoint_path is not None:
                logging.info(f"Debugging hotfix: loading batch from {cfg.modes.debugging_hotfix_datapoint_path}")
                batch = torch.load(cfg.modes.debugging_hotfix_datapoint_path, map_location="cuda:0")
                logging.info(f"Debugging hotfix: loaded batch {batch.seq_name} "
                             f"with {len(batch.video)} views and {batch.video.shape[2]} frames")

            if not all(gotit):
                total_batches_failed += 1
                logging.info(f"batch is None: "
                             f"failed {total_batches_failed} / {total_batches_loaded} "
                             f"({total_batches_failed / total_batches_loaded * 100:.2f}%) batches")
                continue

            i_batch += 1
            dataclass_to_cuda_(batch)
            assert model.training

            start_time_2 = time.time()
            dataloader_duration = start_time_2 - start_time_1
            logging.info(f"Datapoint: {batch.seq_name} (Waited for {dataloader_duration:>5.2f}s)")

            train_iters = cfg.trainer.train_iters
            if cfg.trainer.augment_train_iters:
                train_iters = augment_train_iters(train_iters, total_steps, cfg.trainer.augment_train_iters_warmup)
            optimizer.zero_grad()

            try:
                output = forward_batch_multi_view(
                    batch=batch,
                    model=model,
                    cfg=cfg,
                    step=total_steps,
                    train_iters=train_iters,
                    gamma=cfg.trainer.gamma,
                    save_debug_logs=(
                            ((total_steps % cfg.trainer.viz_freq) == (cfg.trainer.viz_freq - 1))
                            or (total_steps in [0, 10, 100, cfg.trainer.num_steps - 1])
                    ),
                    debug_logs_path=os.path.join(
                        cfg.experiment_path,
                        f'forward_pass__train_step-{total_steps}_global_rank-{fabric.global_rank}'
                    ),
                )
            except Exception as e:
                logging.critical(f"Forward pass crashed at step {total_steps}: {e}")

                # Save current checkpoint
                save_path = Path(f"{cfg.experiment_path}/test_{total_steps:06d}.pth")
                state = AttributeDict(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    total_steps=total_steps + 1,
                )
                fabric._strategy.checkpoint_io.save_checkpoint(
                    checkpoint=fabric._strategy._convert_stateful_objects_in_state(_unwrap_objects(state), filter={}),
                    path=save_path,
                )
                logging.info(f"Saved crash checkpoint to {save_path}")

                # Save the batch
                batch_path = Path(f"{cfg.experiment_path}/crash_batch_step_{total_steps:06d}.pt")
                try:
                    torch.save(batch, batch_path)
                    logging.info(f"Saved crashing batch to {batch_path}")
                except Exception as batch_exc:
                    logging.error(f"Failed to save crashing batch as .pt: {batch_exc}")

                raise  # re-raise to crash the job after saving artifacts

            loss = torch.tensor(0.0).cuda()
            for k, v in output.items():
                if k == "metrics":
                    for metric_name, metric_value in v.items():
                        tb_writer.add_scalar(metric_name, metric_value, total_steps)
                elif "loss" in v:
                    loss += v["loss"]
                    tb_writer.add_scalar(f"live_{k}_loss", v["loss"].item(), total_steps)
                else:
                    raise ValueError(f"Unknown key {k} in output")

            start_time_3 = time.time()
            fwd_duration = start_time_3 - start_time_2

            fabric.barrier()

            start_time_4 = time.time()
            sync_duration = start_time_4 - start_time_3

            fabric.backward(loss)
            # Log a limited number of grad + optimizer state pairs, also log current learning rate
            if (total_steps <= 10) or (total_steps % cfg.trainer.viz_freq == 0):
                log_limit = 5
                logged = 0
                prefix = f"[DEBUG] [RANK={fabric.global_rank:03d}]"
                logging.info(f"{prefix} RNG seed: {torch.initial_seed()}")
                logging.info(f"{prefix} Step={total_steps} – Gradients and Optimizer State")
                for name, param in model.named_parameters():
                    if param.grad is not None and param in optimizer.state:
                        state = optimizer.state[param]
                        exp_avg_norm = state['exp_avg'].norm().item() if 'exp_avg' in state else float('nan')
                        exp_avg_sq_norm = state['exp_avg_sq'].norm().item() if 'exp_avg_sq' in state else float('nan')
                        grad_norm = param.grad.norm().item()
                        logging.info(
                            f"{prefix} Param: {name:<60s} | "
                            f"grad_norm={grad_norm:>14.9f} | "
                            f"exp_avg_norm={exp_avg_norm:>14.9f} | "
                            f"exp_avg_sq_norm={exp_avg_sq_norm:>14.9f}"
                        )
                        logged += 1
                        if logged >= log_limit:
                            break
                for name, param in model.named_parameters():
                    if param.grad_fn:
                        print(f"{prefix} {name} grad_fn: {param.grad_fn}")
                logging.info(f"{prefix} LR at step {total_steps}: {scheduler.get_last_lr()}")
            fabric.clip_gradients(model, optimizer, clip_val=cfg.trainer.grad_clip)
            optimizer.step()
            scheduler.step()

            start_time_5 = time.time()
            bwd_duration = start_time_5 - start_time_4

            if fabric.global_rank == 0:
                if (total_steps % cfg.trainer.viz_freq == 0) or (
                        total_steps == cfg.trainer.num_steps - 1) or total_steps in [0, 10, 100]:
                    logging.info(f"Creating training viz logs (rank: {fabric.global_rank}, step: {total_steps})")
                    video = batch.video.clone().cpu()
                    video_depth = batch.videodepth.clone().cpu()
                    gt_viz, vector_colors = visualizer.visualize(
                        video=video,
                        video_depth=video_depth,
                        tracks=batch.trajectory.clone().cpu(),
                        visibility=batch.visibility.clone().cpu(),
                        query_frame=batch.query_points_3d[..., 0].long().clone().cpu(),
                        filename="train_gt_traj",
                        writer=tb_writer,
                        step=total_steps,
                        save_video=False,
                    )
                    pred_viz, _ = visualizer.visualize(
                        video=video,
                        video_depth=video_depth,
                        tracks=output["flow"]["predictions"][None].cpu(),
                        visibility=(output["visibility"]["predictions"][None] > 0.5).cpu(),
                        query_frame=batch.query_points_3d[..., 0].long().clone().cpu(),
                        filename="train_pred_traj",
                        writer=tb_writer,
                        step=total_steps,
                        save_video=False,
                    )
                    viz = torch.cat([gt_viz[..., :gt_viz.shape[-1] // 2], pred_viz], dim=-1)
                    thread = threading.Thread(
                        target=Visualizer.save_video,
                        args=(viz, visualizer.save_dir, f"train", tb_writer, visualizer.fps, total_steps)
                    )
                    thread.start()
                    threads.append(thread)

                if len(output) > 1:
                    tb_writer.add_scalar(f"live_total_loss", loss.item(), total_steps)
                tb_writer.add_scalar(f"learning_rate", optimizer.param_groups[0]["lr"], total_steps)

            if total_steps % cfg.trainer.save_ckpt_freq == 0:
                ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                save_path = Path(f"{cfg.experiment_path}/model_{ckpt_iter}.pth")
                logging.info(f"Saving file {save_path}")
                state = AttributeDict(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    total_steps=total_steps + 1,
                )
                fabric.save(save_path, state)

            if total_steps % cfg.trainer.eval_freq == 0 and total_steps > 1:
                run_test_eval(cfg, evaluator, model, eval_dataloaders, tb_writer, total_steps)
                fabric.barrier()

            total_steps += 1
            if fabric.global_rank == 0:
                tqdm_epoch.update(1)
                tqdm_total_steps.update(1)
                tqdm_epoch.set_postfix(
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    train_iters=cfg.trainer.train_iters,
                    gamma=cfg.trainer.gamma,
                    seq_name=batch.seq_name,
                )

            total_duration = time.time() - start_time_1
            logging.info(
                f"[timing:{total_steps:06d}] "
                f"Total: {total_duration:>6.2f}s | "
                f"Data: {dataloader_duration:>6.2f}s | "
                f"Fwd: {fwd_duration:>6.2f}s | "
                f"Sync: {sync_duration:>6.2f}s | "
                f"Bwd: {bwd_duration:>6.2f}s | "
            )
            if fabric.global_rank == 0:
                dataloader_durations.append(dataloader_duration)
                fwd_durations.append(fwd_duration)
                sync_durations.append(sync_duration)
                bwd_durations.append(bwd_duration)
                total_durations.append(total_duration)

                tb_writer.add_scalar(f"timing/step", total_duration, total_steps)
                tb_writer.add_scalar(f"timing/only_fwd", fwd_durations[-1], total_steps)
                tb_writer.add_scalar(f"timing/only_sync", sync_durations[-1], total_steps)
                tb_writer.add_scalar(f"timing/only_bwd", bwd_durations[-1], total_steps)
                tb_writer.add_scalar(f"timing/only_dataloader", dataloader_duration, total_steps)

                if len(total_durations) >= timing_log_freq:
                    total_durations_np = np.array(total_durations)
                    fwd_durations_np = np.array(fwd_durations)
                    sync_durations_np = np.array(sync_durations)
                    bwd_durations_np = np.array(bwd_durations)
                    dataloader_durations_np = np.array(dataloader_durations)

                    total_duration_mean = np.mean(total_durations_np)
                    fwd_duration_mean = np.mean(fwd_durations_np)
                    sync_duration_mean = np.mean(sync_durations_np)
                    bwd_duration_mean = np.mean(bwd_durations_np)
                    dataloader_duration_mean = np.mean(dataloader_durations_np)

                    total_duration_median = np.median(total_durations_np)
                    fwd_duration_median = np.median(fwd_durations_np)
                    sync_duration_median = np.median(sync_durations_np)
                    bwd_duration_median = np.median(bwd_durations_np)
                    dataloader_duration_median = np.median(dataloader_durations_np)

                    total_duration_std = np.std(total_durations_np)
                    fwd_duration_std = np.std(fwd_durations_np)
                    sync_duration_std = np.std(sync_durations_np)
                    bwd_duration_std = np.std(bwd_durations_np)
                    dataloader_duration_std = np.std(dataloader_durations_np)

                    tb_writer.add_scalar("timing/step_mean", total_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/step_median", total_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_fwd_mean", fwd_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_fwd_median", fwd_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_sync_mean", sync_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_sync_median", sync_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_bwd_mean", bwd_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_bwd_median", bwd_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_dataloader_mean", dataloader_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_dataloader_median", dataloader_duration_median, total_steps)

                    logging.info(
                        f"[timing:total] "
                        f"Mean: {total_duration_mean:>6.2f}s | "
                        f"Median: {total_duration_median:>6.2f}s | "
                        f"Std: {total_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:fwd]   "
                        f"Mean: {fwd_duration_mean:>6.2f}s | "
                        f"Median: {fwd_duration_median:>6.2f}s | "
                        f"Std: {fwd_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:sync]  "
                        f"Mean: {sync_duration_mean:>6.2f}s | "
                        f"Median: {sync_duration_median:>6.2f}s | "
                        f"Std: {sync_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:bwd]   "
                        f"Mean: {bwd_duration_mean:>6.2f}s | "
                        f"Median: {bwd_duration_median:>6.2f}s | "
                        f"Std: {bwd_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:datal] "
                        f"Mean: {dataloader_duration_mean:>6.2f}s | "
                        f"Median: {dataloader_duration_median:>6.2f}s | "
                        f"Std: {dataloader_duration_std:6.2f}s"
                    )

                    total_durations.clear()
                    fwd_durations.clear()
                    sync_durations.clear()
                    bwd_durations.clear()
                    dataloader_durations.clear()

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

        if fabric.global_rank == 0:
            tqdm_epoch.close()

    if fabric.global_rank == 0:
        tqdm_total_steps.close()
    logging.info("FINISHED TRAINING")

    save_path = f"{cfg.experiment_path}/model_final.pth"
    logging.info(f"Saving file {save_path}")
    state = AttributeDict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        total_steps=total_steps,
    )
    fabric.save(save_path, state)
    run_test_eval(cfg, evaluator, model, eval_dataloaders, tb_writer, total_steps)
    for thread in threads:
        thread.join()
    tb_writer.flush()
    tb_writer.close()
    fabric.barrier()


if __name__ == "__main__":
    main()
