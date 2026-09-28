#!/usr/bin/env python3
"""Single-GPU forward smoke for SurroundTAP (+ optional stock MVTracker memory compare).

Usage:
  cd /share/tgp/yangyi/mvtracker
  export PYTHONPATH=/share/tgp/yangyi/mvtracker
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_surround_tap_forward.py \\
    --ckpt checkpoints/mvtracker_200000_june2025.pth
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
        return
    torch.cuda.synchronize()
    alloc = _gib(torch.cuda.max_memory_allocated())
    reserved = _gib(torch.cuda.max_memory_reserved())
    logging.info("[%s] peak allocated=%.2f GiB  reserved=%.2f GiB", tag, alloc, reserved)
    return alloc, reserved


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _load_ckpt(model: torch.nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
        total_steps = ckpt.get("total_steps")
    else:
        state, total_steps = ckpt, None
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info(
        "Loaded %s (total_steps=%s) missing=%d unexpected=%d",
        path,
        total_steps,
        len(missing),
        len(unexpected),
    )
    if missing:
        logging.info("  missing sample: %s", missing[:12])
    if unexpected:
        logging.info("  unexpected sample: %s", unexpected[:12])
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


def _run_forward(model, datapoint, iters: int, is_train: bool, tag: str):
    model.train(is_train)
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

    _peak_reset()
    t0 = time.time()
    with torch.set_grad_enabled(is_train):
        out = model(
            rgbs=rgbs,
            depths=depths,
            query_points=query,
            intrs=intrs,
            extrs=extrs,
            iters=iters,
            is_train=is_train,
        )
        if is_train:
            # touch grads lightly so backward path is not required for smoke
            loss = out["traj_e"].float().pow(2).mean() + out["vis_e"].float().pow(2).mean()
            if "train_data" in out:
                td = out["train_data"]
                for k in ("vis_predictions", "coord_predictions", "p_idx_end_list", "sort_inds"):
                    assert k in td, f"missing train_data[{k}]"
            loss.backward()
            model.zero_grad(set_to_none=True)
    elapsed = time.time() - t0
    peaks = _peak_report(tag)
    logging.info(
        "[%s] traj_e=%s vis_e=%s elapsed=%.2fs keys=%s",
        tag,
        tuple(out["traj_e"].shape),
        tuple(out["vis_e"].shape),
        elapsed,
        sorted(out.keys()),
    )
    return peaks, out


def build_models(args):
    from mvtracker.models.core.mvtracker.mvtracker import MVTracker
    from mvtracker.models.core.surround_tap.surround_tap import SurroundTAP

    common = dict(
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
    )
    surround = SurroundTAP(
        **common,
        bev_x_min=-51.2,
        bev_x_max=51.2,
        bev_y_min=-51.2,
        bev_y_max=51.2,
        bev_resolution=1.6,
        bev_corr_radius=3,
    )
    baseline = MVTracker(**common)
    logging.info("SurroundTAP params: %.2f M", _count_params(surround) / 1e6)
    logging.info("MVTracker  params: %.2f M", _count_params(baseline) / 1e6)
    if args.ckpt:
        _load_ckpt(surround, args.ckpt)
        _load_ckpt(baseline, args.ckpt)
    if torch.cuda.is_available():
        surround = surround.cuda()
        baseline = baseline.cuda()
    return surround, baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/share/tgp/yangyi/mvtracker/checkpoints/mvtracker_200000_june2025.pth",
        help="Official / Fabric checkpoint (strict=False). Empty string to skip.",
    )
    parser.add_argument("--dataset", default="nusctrack-val-max2")
    parser.add_argument("--dataset-root", default="./datasets")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if args.ckpt == "":
        args.ckpt = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("cuda_available=%s device_count=%s", torch.cuda.is_available(), torch.cuda.device_count())

    from mvtracker.datasets.nusctrack_dataset import NuscTrackDataset
    from mvtracker.datasets.utils import collate_fn

    ds = NuscTrackDataset.from_name(args.dataset, args.dataset_root)
    logging.info("dataset=%s n_clips=%s", args.dataset, len(ds))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    datapoint = _to_cuda(_batch_from_loader(loader))
    logging.info(
        "clip=%s video=%s depth=%s query=%s",
        getattr(datapoint, "seq_name", "?"),
        tuple(datapoint.video.shape),
        tuple(datapoint.videodepth.shape),
        tuple(datapoint.query_points_3d.shape),
    )

    surround, baseline = build_models(args)

    logging.info("=== SurroundTAP eval forward ===")
    p_eval, _ = _run_forward(surround, datapoint, args.iters, is_train=False, tag="surround_eval")

    logging.info("=== SurroundTAP train forward (+tiny backward) ===")
    p_train, _ = _run_forward(surround, datapoint, args.iters, is_train=True, tag="surround_train")

    if not args.skip_baseline:
        logging.info("=== Stock MVTracker eval forward (memory baseline) ===")
        del surround
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        p_base, _ = _run_forward(baseline, datapoint, args.iters, is_train=False, tag="mvtracker_eval")
        if p_eval and p_base:
            logging.info(
                "Delta Surround - MVTracker (eval allocated): %+.2f GiB",
                p_eval[0] - p_base[0],
            )

    logging.info("SMOKE_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("smoke failed")
        sys.exit(1)
