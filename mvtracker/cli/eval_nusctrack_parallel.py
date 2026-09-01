"""Clip-sharded multi-GPU NuscTrack eval (independent processes, no DDP).

Each rank loads the model once, runs a disjoint subset of clips, writes a shard.
The parent process micro-averages tracks and dumps the protocol report.

``--gpus`` are **physical** CUDA indices. Do not wrap this launcher in
``CUDA_VISIBLE_DEVICES`` unless those ids already match ``--gpus``; the worker
re-sets ``CUDA_VISIBLE_DEVICES`` to the physical id.

Do **not** use Fabric DDP (``python -m mvtracker.cli.eval`` with several visible
GPUs): the eval dataloader is not sharded, so every rank would repeat the full val.

Example (pin TAPIP3D to physical GPU 2 while other cards train)::

  python -m mvtracker.cli.eval_nusctrack_parallel \\
    --model tapip3d --dataset nusctrack-val-max2 --gpus 2

Example (MVTracker, clip-sharded across free cards)::

  python -m mvtracker.cli.eval_nusctrack_parallel \\
    --model mvtracker --dataset nusctrack-val --gpus 0,1,3 \\
    --ckpt /share/tgp/yangyi/mvtracker/checkpoints/mvtracker_200000_june2025.pth \\
    --log-dir /share/tgp/yangyi/mvtracker/logs/mvtracker_nusctrack_zeroshot_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

DEFAULT_MVTRACKER_CKPT = (
    "/share/tgp/yangyi/mvtracker/checkpoints/mvtracker_200000_june2025.pth"
)

METHOD_HEADERS = {
    "cotracker3": [
        "method: CoTracker3 offline (official scaled_offline.pth)",
        "pipeline: per-query-camera monocular 2D TAP + UniDepthV2 lift to ego 3D",
        "depth: UniDepthV2 (camera Z, lifted with the same K / cam2ego as the video)",
    ],
    "tapip3d": [
        "method: TAPIP3D (official tapip3d_final.pth, RGB-D)",
        "pipeline: per-query-camera TAPIP3D + UniDepthV2 as input depth (ego 3D out)",
        "depth: UniDepthV2 RGB-D (not GT xyz query; not post-hoc 2D lift)",
        "interp_shape: 384x512 (TAPIP3D official eval resolution)",
    ],
    "spatrackerv2": [
        "method: SpaTrackerV2 offline (Yuxihenry/SpatialTrackerV2-Offline)",
        "pipeline: per-query-camera SpaTrackerV2 + UniDepthV2 as input depth (ego 3D out)",
        "depth: UniDepthV2 RGB-D (not GT xyz query; query_no_BA=True)",
        "interp_shape: 384x512 (same as TAPIP3D smoke resolution)",
    ],
    "mvtracker": [
        "method: MVTracker (multi-view 3D TAP, NuscTrack protocol)",
        "pipeline: 6-view RGB-D, UniDepthV2 as input depth, queries lifted to ego 3D",
        "depth: UniDepthV2 (camera Z, lifted with the same K / cam2ego as the video)",
        "interp_shape: native 432x768 (no Kubric 384x512 resize)",
    ],
}


def _parse_gpus(spec: str):
    spec = spec.strip()
    if spec.lower() in ("all", ""):
        n = torch.cuda.device_count()
        if n <= 0:
            raise RuntimeError("No CUDA GPUs visible")
        return list(range(n))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def _mvtracker_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"], ckpt.get("total_steps")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("total_steps")
    return ckpt, None


def _build_mvtracker(ckpt_path: str):
    from mvtracker.models.core.mvtracker.mvtracker import MVTracker

    # Must match configs/model/mvtracker.yaml (hidden_size=256, not the class default 384).
    model = MVTracker(
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
    state, total_steps = _mvtracker_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    logging.info(
        "Loaded MVTracker from %s (total_steps=%s, missing=%s, unexpected=%s)",
        ckpt_path,
        total_steps,
        len(missing),
        len(unexpected),
    )
    return model


def _build_model(method: str, ckpt: str | None = None):
    from mvtracker.models.core.monocular_baselines import (
        CoTrackerOfflineWrapper,
        MonocularToMultiViewAdapter,
        SpaTrackerV2Wrapper,
        TAPIP3DWrapper,
    )

    if method == "mvtracker":
        if not ckpt:
            raise ValueError("--ckpt is required for --model mvtracker")
        logging.info("Loading MVTracker weights from %s", ckpt)
        return _build_mvtracker(ckpt)
    if method == "cotracker3":
        logging.info("Loading CoTracker3 offline weights")
        inner = CoTrackerOfflineWrapper(model_name="cotracker3_offline", grid_size=10)
        return MonocularToMultiViewAdapter(inner)
    if method == "tapip3d":
        ckpt = ckpt or os.environ.get(
            "TAPIP3D_CKPT",
            "/share/tgp/yangyi/mvtracker/checkpoints/tapip3d_final.pth",
        )
        logging.info("Loading TAPIP3D weights from %s", ckpt)
        inner = TAPIP3DWrapper(
            ckpt=ckpt,
            num_iters=6,
            grid_size=8,
            resolution_factor=1,
        )
        return MonocularToMultiViewAdapter(inner)
    if method == "spatrackerv2":
        logging.info("Loading SpaTrackerV2 offline weights (HF: Yuxihenry/SpatialTrackerV2-Offline)")
        inner = SpaTrackerV2Wrapper(model_type="offline", vo_points=756)
        return MonocularToMultiViewAdapter(inner)
    raise ValueError(f"unknown --model {method}")


def _predictor_kwargs(method: str):
    # CoTracker: native NuscTrack 432x768. TAPIP3D / SpaTrackerV2: 384x512.
    interp_shape = (384, 512) if method in ("tapip3d", "spatrackerv2") else None
    return dict(
        interp_shape=interp_shape,
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


def _worker(rank: int, world_size: int, gpus, args_dict: dict):
    gpu = gpus[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Re-bind CUDA after restricting visibility.
    torch.cuda.set_device(0)

    log_dir = Path(args_dict["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank{rank}/gpu{gpu}] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"rank{rank}.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("Worker start. world_size=%s clip stride from %s method=%s", world_size, rank, args_dict["method"])

    from mvtracker.datasets.nusctrack_dataset import NuscTrackDataset
    from mvtracker.datasets.utils import collate_fn
    from mvtracker.evaluation.evaluator_3dpt import Evaluator
    from mvtracker.models.evaluation_predictor_3dpt import EvaluationPredictor

    lock_path = log_dir / "model_load.lock"
    lock_path.touch(exist_ok=True)
    import fcntl

    with open(lock_path, "r+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            model = _build_model(args_dict["method"], args_dict.get("ckpt"))
            model.cuda()
            model.eval()
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    dataset = NuscTrackDataset.from_name(args_dict["dataset"], args_dict["dataset_root"])
    indices = list(range(rank, len(dataset), world_size))
    logging.info("Assigned %s / %s clips: %s", len(indices), len(dataset), indices[:8])
    shard = Subset(dataset, indices)
    loader = DataLoader(
        shard,
        batch_size=1,
        shuffle=False,
        num_workers=int(args_dict["num_workers"]),
        collate_fn=collate_fn,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    predictor = EvaluationPredictor(
        multiview_model=model,
        **_predictor_kwargs(args_dict["method"]),
    )
    evaluator = Evaluator(
        rerun_viz_indices=None,
        forward_pass_log_indices=None,
        mp4_track_viz_indices=None,
        save_tracks_npz=False,
        dump_nusctrack_metrics=False,
    )
    t0 = time.time()
    metrics = evaluator.evaluate_sequence(
        model=predictor,
        test_dataloader=loader,
        dataset_name=args_dict["dataset"],
        log_dir=str(log_dir / f"rank{rank}"),
        writer=None,
        step=-1,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        logging.info("Peak CUDA memory over shard: allocated=%.2f GiB reserved=%.2f GiB", peak_gb, reserved_gb)
        metrics["_peak_mem_allocated_gb"] = peak_gb
        metrics["_peak_mem_reserved_gb"] = reserved_gb
    records = metrics.pop("__nusctrack_records__", [])
    clip_metrics = {k: v for k, v in metrics.items() if isinstance(k, int)}
    shard_path = log_dir / f"shard_rank{rank}.pt"
    torch.save(
        {
            "rank": rank,
            "gpu": gpu,
            "indices": indices,
            "n_clips": len(indices),
            "elapsed_sec": time.time() - t0,
            "records": records,
            "clip_metrics": clip_metrics,
            "peak_mem_allocated_gb": metrics.get("_peak_mem_allocated_gb"),
            "peak_mem_reserved_gb": metrics.get("_peak_mem_reserved_gb"),
        },
        shard_path,
    )
    logging.info("Wrote %s (%s clips, %.1fs)", shard_path, len(indices), time.time() - t0)


def _aggregate(log_dir: Path, dataset_name: str, world_size: int, method: str, header_extra=None):
    from mvtracker.evaluation.nusctrack_eval import (
        THRESHOLDS,
        aggregate_track_evals,
        dump_tap_metrics,
        format_tap_metrics,
    )

    records = []
    clip_metrics = {}
    shard_meta = []
    for rank in range(world_size):
        path = log_dir / f"shard_rank{rank}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing shard {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        for local_i, summary in payload["clip_metrics"].items():
            seq = summary.get("seq_name", f"rank{rank}_{local_i}")
            clip_metrics[seq] = summary
        shard_meta.append(
            {
                "rank": payload["rank"],
                "gpu": payload["gpu"],
                "n_clips": payload["n_clips"],
                "elapsed_sec": payload["elapsed_sec"],
                "indices": payload["indices"],
                "peak_mem_allocated_gb": payload.get("peak_mem_allocated_gb"),
                "peak_mem_reserved_gb": payload.get("peak_mem_reserved_gb"),
            }
        )

    dataset_metrics = aggregate_track_evals(records)
    header = [
        "NuscTrack / xTAP3D evaluation",
        "dataset: {}".format(dataset_name),
        *METHOD_HEADERS[method],
        "3D thresholds (m): {}".format(THRESHOLDS),
        "2D thresholds (px @ 256x256): 1, 2, 4, 8, 16",
        "scale rescale: none",
        "query: (cam, t, x, y) + UniDepth unproject (not GT xyz)",
        "aggregation: track micro-average (not clip-equal)",
        "n_clips: {}".format(len(records)),
        "parallel: {} GPU shards".format(world_size),
    ]
    if header_extra:
        header.extend(header_extra)
    txt_path, json_path = dump_tap_metrics(
        dataset_metrics,
        str(log_dir / "nusctrack_metrics"),
        header_lines=header,
    )

    if clip_metrics:
        df = pd.DataFrame(clip_metrics).T
        df.to_csv(log_dir / "clip_metrics.csv")
    pd.DataFrame([metrics_to_row(dataset_metrics)], index=["score"]).T.to_csv(
        log_dir / "nusctrack_metrics_avg.csv"
    )
    with open(log_dir / "shards.json", "w") as f:
        json.dump(shard_meta, f, indent=2)
        f.write("\n")

    summary = format_tap_metrics(dataset_metrics)
    logging.info("Wrote %s\n%s", txt_path, summary)
    return txt_path, json_path


def metrics_to_row(metrics: dict):
    row = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue
        row[k] = v
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nusctrack-val")
    parser.add_argument("--dataset-root", default="./datasets")
    parser.add_argument(
        "--log-dir",
        default="/share/tgp/yangyi/mvtracker/logs/cotracker3_offline_nusctrack_val",
    )
    parser.add_argument("--gpus", default="all", help="Physical GPU ids, comma-separated, or 'all'")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model",
        default="cotracker3",
        choices=sorted(METHOD_HEADERS.keys()),
        help="NuscTrack baseline to run",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Weight path for --model mvtracker (official june2025 or Fabric model_*.pth).",
    )
    args = parser.parse_args()
    if args.model == "mvtracker" and not args.ckpt:
        args.ckpt = DEFAULT_MVTRACKER_CKPT

    gpus = _parse_gpus(args.gpus)
    world_size = len(gpus)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [parent] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "parent.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("Launching %s workers on physical GPUs %s method=%s", world_size, gpus, args.model)
    logging.info("log_dir=%s dataset=%s", log_dir, args.dataset)

    args_dict = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "log_dir": str(log_dir),
        "num_workers": args.num_workers,
        "method": args.model,
        "ckpt": args.ckpt,
    }
    ctx = torch.multiprocessing.get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, gpus, args_dict), daemon=False)
        p.start()
        procs.append(p)
    failures = []
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            failures.append((rank, p.exitcode))
    if failures:
        raise RuntimeError(f"worker failures: {failures}")

    extra = [f"ckpt: {args.ckpt}"] if args.ckpt else None
    txt_path, json_path = _aggregate(
        log_dir, args.dataset, world_size, args.model, header_extra=extra
    )
    print(txt_path)
    print(json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
