"""Clip-sharded multi-GPU NuscTrack eval (independent processes, no DDP).

Each rank loads CoTracker once, runs a disjoint subset of clips, writes a shard.
The parent process micro-averages tracks and dumps the protocol report.

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
  python -m mvtracker.cli.eval_nusctrack_parallel \\
    --dataset nusctrack-val \\
    --gpus 0,1,2,3,4,5,6,7
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


def _parse_gpus(spec: str):
    spec = spec.strip()
    if spec.lower() in ("all", ""):
        n = torch.cuda.device_count()
        if n <= 0:
            raise RuntimeError("No CUDA GPUs visible")
        return list(range(n))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


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
    logging.info("Worker start. world_size=%s clip stride from %s", world_size, rank)

    from mvtracker.datasets.nusctrack_dataset import NuscTrackDataset
    from mvtracker.datasets.utils import collate_fn
    from mvtracker.evaluation.evaluator_3dpt import Evaluator
    from mvtracker.models.core.monocular_baselines import (
        CoTrackerOfflineWrapper,
        MonocularToMultiViewAdapter,
    )
    from mvtracker.models.evaluation_predictor_3dpt import EvaluationPredictor

    lock_path = log_dir / "cotracker_hub.lock"
    lock_path.touch(exist_ok=True)
    import fcntl

    with open(lock_path, "r+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            logging.info("Loading CoTracker3 offline weights")
            inner = CoTrackerOfflineWrapper(model_name="cotracker3_offline", grid_size=10)
            model = MonocularToMultiViewAdapter(inner)
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
        },
        shard_path,
    )
    logging.info("Wrote %s (%s clips, %.1fs)", shard_path, len(indices), time.time() - t0)


def _aggregate(log_dir: Path, dataset_name: str, world_size: int, header_extra=None):
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
            }
        )

    dataset_metrics = aggregate_track_evals(records)
    header = [
        "NuscTrack / xTAP3D evaluation",
        "dataset: {}".format(dataset_name),
        "method: CoTracker3 offline (official scaled_offline.pth)",
        "pipeline: per-query-camera monocular 2D TAP + UniDepthV2 lift to ego 3D",
        "depth: UniDepthV2 (camera Z, lifted with the same K / cam2ego as the video)",
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
    parser.add_argument("--gpus", default="all", help="Comma-separated GPU ids, or 'all'")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

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
    logging.info("Launching %s workers on GPUs %s", world_size, gpus)
    logging.info("log_dir=%s dataset=%s", log_dir, args.dataset)

    args_dict = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "log_dir": str(log_dir),
        "num_workers": args.num_workers,
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

    txt_path, json_path = _aggregate(log_dir, args.dataset, world_size)
    print(txt_path)
    print(json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
