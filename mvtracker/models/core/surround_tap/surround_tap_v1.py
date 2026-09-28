"""SurroundTAP v1: BEV coarse XY update, then stock MVTracker kNN.

Does not go through SurroundTAP v0 (no fcorrs + bev_tok add).
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import torch

from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
from mvtracker.models.core.mvtracker.mvtracker import MVTracker
from mvtracker.models.core.surround_tap.bev_splat import (
    sample_bev_patch_corr,
    splat_pointcloud_to_bev,
)
from mvtracker.models.core.surround_tap.coarse_head import CoarseBEVHead


class SurroundTAPV1(MVTracker):
    """MVTracker + staged coarse BEV head (local patch attention, XY only)."""

    def __init__(
            self,
            sliding_window_len=12,
            stride=4,
            normalize_scene_in_fwd_pass=False,
            fmaps_dim=128,
            add_space_attn=True,
            num_heads=6,
            hidden_size=256,
            space_depth=6,
            time_depth=6,
            num_virtual_tracks=64,
            use_flash_attention=True,
            corr_n_groups=1,
            corr_n_levels=4,
            corr_neighbors=16,
            corr_add_neighbor_offset=True,
            corr_add_neighbor_xyz=False,
            corr_filter_invalid_depth=False,
            bev_x_min: float = -51.2,
            bev_x_max: float = 51.2,
            bev_y_min: float = -51.2,
            bev_y_max: float = 51.2,
            bev_resolutions: Sequence[float] = (1.6, 0.8),
            bev_corr_radius: int = 4,
            coarse_detach: bool = True,
            coarse_attn_heads: int = 4,
            **kwargs,
    ):
        if kwargs:
            logging.warning("SurroundTAPV1 unused kwargs: %s", sorted(kwargs.keys()))
        super().__init__(
            sliding_window_len=sliding_window_len,
            stride=stride,
            normalize_scene_in_fwd_pass=normalize_scene_in_fwd_pass,
            fmaps_dim=fmaps_dim,
            add_space_attn=add_space_attn,
            num_heads=num_heads,
            hidden_size=hidden_size,
            space_depth=space_depth,
            time_depth=time_depth,
            num_virtual_tracks=num_virtual_tracks,
            use_flash_attention=use_flash_attention,
            corr_n_groups=corr_n_groups,
            corr_n_levels=corr_n_levels,
            corr_neighbors=corr_neighbors,
            corr_add_neighbor_offset=corr_add_neighbor_offset,
            corr_add_neighbor_xyz=corr_add_neighbor_xyz,
            corr_filter_invalid_depth=corr_filter_invalid_depth,
        )
        self.bev_x_min = float(bev_x_min)
        self.bev_x_max = float(bev_x_max)
        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)
        self.bev_resolutions: List[float] = [float(r) for r in bev_resolutions]
        self.bev_corr_radius = int(bev_corr_radius)
        self.coarse_detach = bool(coarse_detach)
        self.coarse_head = CoarseBEVHead(dim=self.latent_dim, num_heads=coarse_attn_heads)
        self._coarse_coords_buf: List[torch.Tensor] = []

    def _pad_time(self, coords, vis_init, track_mask, S):
        if coords.shape[1] < S:
            coords = torch.cat([coords, coords[:, -1].repeat(1, S - coords.shape[1], 1, 1)], dim=1)
            vis_init = torch.cat([vis_init, vis_init[:, -1].repeat(1, S - vis_init.shape[1], 1, 1)], dim=1)
        if track_mask.shape[1] < S:
            track_mask = torch.cat([
                track_mask,
                torch.zeros_like(track_mask[:, 0]).repeat(1, S - track_mask.shape[1], 1, 1),
            ], dim=1)
        return coords, vis_init, track_mask

    def _coarse_update(self, fmaps, depths, intrs, extrs, coords, feat):
        B, V, S, C, H, W = fmaps.shape
        xyz, fvec, valid = init_pointcloud_from_rgbd(
            fmaps=fmaps,
            depths=depths,
            intrs=intrs,
            extrs=extrs,
            stride=self.stride,
            level=0,
            return_validity_mask=True,
        )
        patch_tokens = []
        k = 2 * self.bev_corr_radius + 1
        for res in self.bev_resolutions:
            bev = splat_pointcloud_to_bev(
                xyz,
                fvec,
                valid,
                B=B,
                S=S,
                x_min=self.bev_x_min,
                x_max=self.bev_x_max,
                y_min=self.bev_y_min,
                y_max=self.bev_y_max,
                resolution=res,
            )
            raw = sample_bev_patch_corr(
                bev=bev,
                coords_xyz=coords,
                radius=self.bev_corr_radius,
                x_min=self.bev_x_min,
                x_max=self.bev_x_max,
                y_min=self.bev_y_min,
                y_max=self.bev_y_max,
                resolution=res,
            )
            patch_tokens.append(raw.view(B, S, coords.shape[2], k * k, C))
        patches = torch.cat(patch_tokens, dim=3)
        dxy = self.coarse_head(feat, patches)
        coords = coords.clone()
        coords[..., :2] = coords[..., :2] + dxy
        return coords

    def forward_iteration(
            self,
            fmaps,
            depths,
            intrs,
            extrs,
            coords_init,
            vis_init,
            track_mask,
            iters=4,
            feat_init=None,
            save_debug_logs=False,
            debug_logs_path="",
            debug_logs_prefix="",
            debug_logs_window_idx=None,
            save_rerun_logs: bool = False,
            rerun_fmap_coloring_fn=None,
    ):
        B, V, S, D, H, W = fmaps.shape
        coords, vis_init, track_mask = self._pad_time(coords_init, vis_init, track_mask, S)
        if feat_init is None:
            feat = coords.new_zeros(B, S, coords.shape[2], self.latent_dim)
        elif feat_init.shape[1] < S:
            feat = torch.cat([feat_init, feat_init[:, -1].repeat(1, S - feat_init.shape[1], 1, 1)], dim=1)
        else:
            feat = feat_init

        coords = self._coarse_update(fmaps, depths, intrs, extrs, coords, feat)
        if getattr(self, "is_train", False):
            self._coarse_coords_buf.append(coords.clone())
        if self.coarse_detach:
            coords = coords.detach()

        return super().forward_iteration(
            fmaps=fmaps,
            depths=depths,
            intrs=intrs,
            extrs=extrs,
            coords_init=coords,
            vis_init=vis_init,
            track_mask=track_mask,
            iters=iters,
            feat_init=feat_init,
            save_debug_logs=save_debug_logs,
            debug_logs_path=debug_logs_path,
            debug_logs_prefix=debug_logs_prefix,
            debug_logs_window_idx=debug_logs_window_idx,
            save_rerun_logs=save_rerun_logs,
            rerun_fmap_coloring_fn=rerun_fmap_coloring_fn,
        )

    def forward(self, *args, **kwargs):
        self._coarse_coords_buf = []
        results = super().forward(*args, **kwargs)
        if getattr(self, "is_train", False) and results.get("train_data") is not None:
            aligned = []
            coord_preds = results["train_data"]["coord_predictions"]
            for i, coarse in enumerate(self._coarse_coords_buf):
                sl = coord_preds[i][0].shape[1]
                n_win = coord_preds[i][0].shape[2]
                aligned.append(coarse[:, :sl, :n_win])
            results["train_data"]["coarse_coords"] = aligned
        return results
