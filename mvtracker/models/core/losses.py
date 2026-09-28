# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from mvtracker.models.core.model_utils import reduce_masked_mean

EPS = 1e-6

sigma = 3
x_grid = torch.arange(-7, 8, 1)
y_grid = torch.arange(-7, 8, 1)
x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing="ij")
gridxy = torch.stack([x_grid, y_grid], dim=-1).float()
gs_kernel = torch.exp(-torch.sum(gridxy ** 2, dim=-1) / (2 * sigma ** 2))


def balanced_ce_loss(pred, gt, valid=None):
    total_balanced_loss = 0.0
    for j in range(len(gt)):
        B, S, N = gt[j].shape
        # pred and gt are the same shape
        for (a, b) in zip(pred[j].size(), gt[j].size()):
            assert a == b  # some shape mismatch!
        # if valid is not None:
        for (a, b) in zip(pred[j].size(), valid[j].size()):
            assert a == b  # some shape mismatch!

        pos = (gt[j] > 0.95).float()
        neg = (gt[j] < 0.05).float()

        label = pos * 2.0 - 1.0
        a = -label * pred[j]
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

        pos_loss = reduce_masked_mean(loss, pos * valid[j])
        neg_loss = reduce_masked_mean(loss, neg * valid[j])

        balanced_loss = pos_loss + neg_loss
        total_balanced_loss += balanced_loss
    return total_balanced_loss


def sequence_loss_3d(flow_preds, flow_gt, vis, valids, gamma=0.8, dmin=0.1, dmax=65, Dz=128):
    """Loss function defined over sequence of flow predictions with z component post-processing"""
    total_flow_loss = 0.0
    J = len(flow_gt)
    for j in range(J):
        B, S, N, D = flow_gt[j].shape
        assert D == 3
        B, S1, N = vis[j].shape
        B, S2, N = valids[j].shape
        assert S == S1
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            flow_gt_j = flow_gt[j].clone()
            flow_pred[..., 2] = (flow_pred[..., 2] - dmin) / (dmax - dmin) * Dz
            flow_gt_j[..., 2] = (flow_gt_j[..., 2] - dmin) / (dmax - dmin) * Dz
            i_loss = (flow_pred - flow_gt_j).abs()  # B, S, N, 3
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            flow_loss += i_weight * reduce_masked_mean(i_loss, valids[j])
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss / float(J)
    return total_flow_loss


def coarse_xy_huber_loss(coarse_preds, traj_gts, valids, query_gts, delta=0.8, disp_thresh=2.0):
    """Huber on XY only, masked by valid and large GT displacement from the query frame.

    Args:
        coarse_preds: list of (B, S, N, 3) coarse window predictions.
        traj_gts / valids: sliding-window GT aligned with ``sequence_loss_3d``.
        query_gts: list of (B, N, 3) GT xyz at each track's query frame (same N as the window).
        delta: Huber / SmoothL1 beta in metres.
        disp_thresh: min GT XY displacement from the query frame (metres).
    """
    total = 0.0
    j = 0
    for pred, gt, valid, qgt in zip(coarse_preds, traj_gts, valids, query_gts):
        assert pred.shape == gt.shape
        pred_xy = pred[..., :2]
        gt_xy = gt[..., :2]
        q_xy = qgt[..., :2][:, None, :, :]
        disp = (gt_xy - q_xy).norm(dim=-1)
        mask = valid.float() * (disp > disp_thresh).float()
        per = F.smooth_l1_loss(pred_xy, gt_xy, beta=delta, reduction="none").mean(dim=-1)
        total = total + reduce_masked_mean(per, mask)
        j += 1
    if j == 0:
        return torch.zeros((), device=traj_gts[0].device if traj_gts else "cpu")
    return total / float(j)
