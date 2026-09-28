#!/usr/bin/env python3
"""Single-GPU smoke for SurroundTAP v1 (coarse head + frozen-backbone grads).

Usage:
  cd /share/tgp/yangyi/mvtracker
  export PYTHONPATH=/share/tgp/yangyi/mvtracker
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_surround_tap_v1.py \\
    --ckpt logs/mvtracker_nusctrack_ft/model_final.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import torch
from torch.utils.data import DataLoader


def _gib(x: int) -> float:
    return x / (1024 ** 3)


def _peak_reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_report(tag: str):
    if not torch.cuda.is_available():
        logging.info("[%s] CUDA unavailable", tag)
        return None
    torch.cuda.synchronize()
    alloc = _gib(torch.cuda.max_memory_allocated())
    reserved = _gib(torch.cuda.max_memory_reserved())
    logging.info("[%s] peak allocated=%.2f GiB  reserved=%.2f GiB", tag, alloc, reserved)
    return alloc, reserved


def _load_ckpt(model: torch.nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
        total_steps = ckpt.get("total_steps")
    else:
        state, total_steps = ckpt, None
    if state and next(iter(state)).startswith("module."):
        state = {k[len("module.") :]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info(
        "Loaded %s (total_steps=%s) missing=%d unexpected=%d",
        path,
        total_steps,
        len(missing),
        len(unexpected),
    )
    if missing:
        logging.info("  missing sample: %s", missing[:16])
    return missing, unexpected


def _batch_from_loader(loader):
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        datapoint, gotit = batch
        if not all(gotit):
            raise RuntimeError("first batch marked invalid")
    else:
        datapoint = batch
    return datapoint


def _to_cuda(datapoint):
    from mvtracker.datasets.utils import dataclass_to_cuda_

    if torch.cuda.is_available():
        dataclass_to_cuda_(datapoint)
    return datapoint


def _freeze_except_coarse(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("coarse_")
    n_on = sum(p.requires_grad for p in model.parameters())
    logging.info("trainable tensors (coarse_*): %s", n_on)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/share/tgp/yangyi/mvtracker/logs/mvtracker_nusctrack_ft/model_final.pth",
    )
    parser.add_argument("--dataset", default="nusctrack-val-max2")
    parser.add_argument("--dataset-root", default="./datasets")
    parser.add_argument("--iters", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    from mvtracker.datasets.nusctrack_dataset import NuscTrackDataset
    from mvtracker.datasets.utils import collate_fn
    from mvtracker.models.core.surround_tap.surround_tap_v1 import SurroundTAPV1

    ds = NuscTrackDataset.from_name(args.dataset, args.dataset_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    datapoint = _to_cuda(_batch_from_loader(loader))
    logging.info("clip=%s video=%s", getattr(datapoint, "seq_name", "?"), tuple(datapoint.video.shape))

    model = SurroundTAPV1(
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
        bev_resolutions=(1.6, 0.8),
        bev_corr_radius=4,
        coarse_detach=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_coarse = sum(p.numel() for n, p in model.named_parameters() if n.startswith("coarse_"))
    logging.info("SurroundTAPV1 params: %.2f M (coarse_head %.3f M)", n_params / 1e6, n_coarse / 1e6)
    _load_ckpt(model, args.ckpt)
    _freeze_except_coarse(model)
    if torch.cuda.is_available():
        model = model.cuda()

    rgbs = datapoint.video
    depths = datapoint.videodepth
    if rgbs.ndim == 5:
        rgbs = rgbs.unsqueeze(0)
        depths = depths.unsqueeze(0)
        query = datapoint.query_points_3d.unsqueeze(0)
        intrs = datapoint.intrs.unsqueeze(0)
        extrs = datapoint.extrs.unsqueeze(0)
    else:
        query = datapoint.query_points_3d
        intrs = datapoint.intrs
        extrs = datapoint.extrs

    logging.info("=== eval forward ===")
    model.eval()
    _peak_reset()
    with torch.no_grad():
        out = model(rgbs=rgbs, depths=depths, query_points=query, intrs=intrs, extrs=extrs, iters=args.iters, is_train=False)
    _peak_report("v1_eval")
    logging.info("traj_e=%s vis_e=%s", tuple(out["traj_e"].shape), tuple(out["vis_e"].shape))

    logging.info("=== train forward + backward (coarse only) ===")
    model.train()
    _peak_reset()
    out = model(rgbs=rgbs, depths=depths, query_points=query, intrs=intrs, extrs=extrs, iters=args.iters, is_train=True)
    assert "coarse_coords" in out["train_data"], "missing train_data[coarse_coords]"
    loss = out["traj_e"].float().pow(2).mean()
    for cc in out["train_data"]["coarse_coords"]:
        loss = loss + cc.float().pow(2).mean()
    loss.backward()

    coarse_grad = 0.0
    frozen_grad = 0.0
    for name, param in model.named_parameters():
        g = 0.0 if param.grad is None else param.grad.detach().abs().sum().item()
        if name.startswith("coarse_"):
            coarse_grad += g
        else:
            frozen_grad += g
    logging.info("grad_l1 coarse_head=%.6f frozen=%.6f (frozen should be 0)", coarse_grad, frozen_grad)
    if coarse_grad <= 0:
        raise RuntimeError("coarse_head received no gradient")
    if frozen_grad > 0:
        raise RuntimeError(f"frozen backbone received gradient {frozen_grad}")
    _peak_report("v1_train")
    logging.info("SMOKE_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("smoke failed")
        sys.exit(1)
