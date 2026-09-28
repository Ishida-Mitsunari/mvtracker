"""SurroundTAP: MVTracker fine kNN + coarse depth-splat BEV correlation.

Keeps the stock MVTracker forward / train_data contract so existing
NuscTrack training loss code can be reused later. New BEV path is additive
and fused into the same UpdateFormer input dim (Kubric weights loadable with
strict=False; fuse MLP / BEV params are new).
"""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from mvtracker.models.core.embeddings import (
    get_3d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid,
    get_3d_embedding,
)
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
from mvtracker.models.core.mvtracker.mvtracker import MVTracker, PointcloudCorrBlock
from mvtracker.models.core.surround_tap.bev_splat import (
    sample_bev_patch_corr,
    splat_rgbd_to_bev,
)


class SurroundTAP(MVTracker):
    """MVTracker subclass with an extra coarse BEV correlation branch."""

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
            # Coarse BEV
            bev_x_min: float = -51.2,
            bev_x_max: float = 51.2,
            bev_y_min: float = -51.2,
            bev_y_max: float = 51.2,
            bev_resolution: float = 1.6,
            bev_corr_radius: int = 3,
            bev_fuse: str = "add",  # add | concat_proj
            **kwargs,
    ):
        if kwargs:
            logging.warning("SurroundTAP unused kwargs: %s", sorted(kwargs.keys()))
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
        self.bev_resolution = float(bev_resolution)
        self.bev_corr_radius = int(bev_corr_radius)
        self.bev_fuse = str(bev_fuse)

        knn_lrr = (
            self.corr_neighbors
            * self.corr_n_levels
            * (
                self.corr_n_groups
                + 3 * self.corr_add_neighbor_offset
                + 3 * self.corr_add_neighbor_xyz
                + self.corr_pos_emb_size
            )
        )
        patch = (2 * self.bev_corr_radius + 1) ** 2
        bev_raw_dim = patch * self.latent_dim
        self.knn_lrr = knn_lrr
        self.bev_raw_dim = bev_raw_dim
        # Project BEV patch features into the same LRR dim as kNN corr, then add.
        self.bev_corr_proj = nn.Sequential(
            nn.Linear(bev_raw_dim, knn_lrr),
            nn.GELU(),
            nn.Linear(knn_lrr, knn_lrr),
        )

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
            rerun_fmap_coloring_fn: Optional[Callable] = None,
    ):
        B, V, S, D, H, W = fmaps.shape
        N = coords_init.shape[2]
        device = fmaps.device
        if coords_init.shape[1] < S:
            coords = torch.cat([coords_init, coords_init[:, -1].repeat(1, S - coords_init.shape[1], 1, 1)], dim=1)
            vis_init = torch.cat([vis_init, vis_init[:, -1].repeat(1, S - vis_init.shape[1], 1, 1)], dim=1)
        else:
            coords = coords_init.clone()
        if track_mask.shape[1] < S:
            track_mask = torch.cat([
                track_mask,
                torch.zeros_like(track_mask[:, 0]).repeat(1, S - track_mask.shape[1], 1, 1),
            ], dim=1)
        assert B == 1
        assert D == self.latent_dim
        assert fmaps.shape == (B, V, S, D, H, W)
        assert depths.shape == (B, V, S, 1, H, W)
        assert intrs.shape == (B, V, S, 3, 3)
        assert extrs.shape == (B, V, S, 3, 4)
        assert coords.shape == (B, S, N, 3)
        assert vis_init.shape == (B, S, N, 1)
        assert track_mask.shape == (B, S, N, 1)
        assert feat_init is None or feat_init.shape == (B, S, N, self.latent_dim)
        assert track_mask.any(1).all(), "All points should be requested to be tracked at least for one frame"

        fcorr_fns = {}
        for lvl in range(self.corr_n_levels):
            pc = init_pointcloud_from_rgbd(
                fmaps=fmaps,
                depths=depths,
                intrs=intrs,
                extrs=extrs,
                stride=self.stride,
                level=lvl,
                return_validity_mask=self.corr_filter_invalid_depth or save_rerun_logs,
            )
            if self.corr_filter_invalid_depth or save_rerun_logs:
                pc_xyz, pc_fvec, pc_valid = pc
            else:
                pc_xyz, pc_fvec = pc
                pc_valid = None
            fcorr_fns[lvl] = PointcloudCorrBlock(
                k=self.corr_neighbors,
                groups=self.corr_n_groups,
                xyz=pc_xyz,
                fvec=pc_fvec,
                filter_invalid=self.corr_filter_invalid_depth,
                valid=pc_valid,
                corr_add_neighbor_offset=self.corr_add_neighbor_offset,
                corr_add_neighbor_xyz=self.corr_add_neighbor_xyz,
                rerun_fmap_coloring_fn=rerun_fmap_coloring_fn,
            )

        # Coarse BEV once per window (features fixed; sampling follows coords).
        bev = splat_rgbd_to_bev(
            fmaps=fmaps,
            depths=depths,
            intrs=intrs,
            extrs=extrs,
            stride=self.stride,
            x_min=self.bev_x_min,
            x_max=self.bev_x_max,
            y_min=self.bev_y_min,
            y_max=self.bev_y_max,
            resolution=self.bev_resolution,
        )

        embed_dim = self.updateformer_input_dim
        if embed_dim % 6 != 0:
            embed_dim += 6 - (embed_dim % 6)
        pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, coords[:, 0:1]).float()[:, 0].permute(0, 2, 1)
        if embed_dim > self.updateformer_input_dim:
            pos_embed = pos_embed[:, :self.updateformer_input_dim, :]
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1) / S
        embed_dim = self.updateformer_input_dim
        if embed_dim % 2 != 0:
            embed_dim += 2 - (embed_dim % 2)
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        if embed_dim > self.updateformer_input_dim:
            times_embed = times_embed[:, :, :self.updateformer_input_dim]

        coord_predictions = []
        ffeats = feat_init.clone()
        track_mask_and_vis = torch.cat([track_mask, vis_init], dim=3).permute(0, 2, 1, 3).reshape(B * N, S, 2)

        for it in range(iters):
            coords = coords.detach()

            fcorrs = []
            for lvl in range(self.corr_n_levels):
                fcorr_fn = fcorr_fns[lvl]
                fcorrs_level = (
                    fcorr_fn
                    .corr_sample(
                        targets=ffeats.reshape(B * S, N, self.latent_dim),
                        coords_world_xyz=coords.reshape(B * S, N, 3),
                        save_debug_logs=False,
                        debug_logs_path=debug_logs_path,
                        debug_logs_prefix=debug_logs_prefix + f"__iter_{it}__pyramid_level_{lvl}",
                        save_rerun_logs=save_rerun_logs,
                    )
                    .reshape(B, S, N, -1)
                )
                fcorrs.append(fcorrs_level)
                if self.stats_pyramid is not None:
                    self.stats_pyramid[(lvl, it)] += [
                        np.linalg.norm(fcorrs_level.reshape(-1, 4)[:, 1:].detach().cpu().numpy(), axis=-1)
                    ]
            fcorrs = torch.cat(fcorrs, dim=-1)
            LRR = fcorrs.shape[3]
            assert LRR == self.knn_lrr, f"LRR mismatch: {LRR} vs {self.knn_lrr}"

            bev_raw = sample_bev_patch_corr(
                bev=bev,
                coords_xyz=coords,
                radius=self.bev_corr_radius,
                x_min=self.bev_x_min,
                x_max=self.bev_x_max,
                y_min=self.bev_y_min,
                y_max=self.bev_y_max,
                resolution=self.bev_resolution,
            )
            assert bev_raw.shape[-1] == self.bev_raw_dim
            bev_tok = self.bev_corr_proj(bev_raw)  # (B,S,N,LRR)
            fcorrs = fcorrs + bev_tok

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)

            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_ = get_3d_embedding(flows_, self.flow_embed_dim, cat_coords=True)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            transformer_input = torch.cat([flows_, fcorrs_, ffeats_, track_mask_and_vis], dim=2)
            assert transformer_input.shape[-1] == self.updateformer_input_dim
            x = transformer_input + pos_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta = self.updateformer(x)
            delta = rearrange(delta, " b n t d -> (b n) t d")

            d_coord = delta[:, :, :3].reshape(B, N, S, 3).permute(0, 2, 1, 3)
            d_feats = delta[:, :, 3:self.latent_dim + 3]
            d_feats = self.ffeats_norm(d_feats.view(-1, self.latent_dim))
            d_feats = self.ffeats_updater(d_feats).view(B, N, S, self.latent_dim).permute(0, 2, 1, 3)

            coords = coords + d_coord
            ffeats = ffeats + d_feats

            if torch.isnan(coords).any():
                logging.error("Got NaN values in coords (SurroundTAP)")
                raise RuntimeError("NaN in SurroundTAP coords")

            coord_predictions.append(coords.clone())

        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
        return coord_predictions, vis_e, feat_init
