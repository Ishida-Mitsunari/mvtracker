#!/usr/bin/env bash
# Clip-sharded MVTracker eval on NuscTrack (independent processes, not Fabric DDP).
#
# --gpus are physical CUDA indices. Do not wrap CUDA_VISIBLE_DEVICES unless
# those ids already match --gpus.
#
# Usage:
#   unset CUDA_VISIBLE_DEVICES
#   bash scripts/nusctrack_eval_mvtracker.sh \
#     --ckpt /share/tgp/yangyi/mvtracker/logs/mvtracker_nusctrack_ft/model_004500.pth \
#     --gpus 2 \
#     --dataset nusctrack-val-max2
#
# Full val after the FT job releases cards:
#   bash scripts/nusctrack_eval_mvtracker.sh \
#     --ckpt /share/tgp/yangyi/mvtracker/logs/mvtracker_nusctrack_ft/model_final.pth \
#     --gpus 0,1,3,4,5,6,7 \
#     --dataset nusctrack-val
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${ROOT}/checkpoints/mvtracker_200000_june2025.pth"
DATASET="nusctrack-val"
GPUS=""
LOG_DIR=""
NUM_WORKERS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$GPUS" ]]; then
  echo "Set --gpus to physical GPU ids (e.g. 0,1,3). Do not pass cards still used by training." >&2
  exit 1
fi
if [[ -z "$LOG_DIR" ]]; then
  if [[ "$DATASET" == "nusctrack-val-max2" ]]; then
    LOG_DIR="${ROOT}/logs/mvtracker_nusctrack_eval_max2"
  else
    LOG_DIR="${ROOT}/logs/mvtracker_nusctrack_eval"
  fi
fi

exec .venv/bin/python -m mvtracker.cli.eval_nusctrack_parallel \
  --model mvtracker \
  --dataset "$DATASET" \
  --ckpt "$CKPT" \
  --gpus "$GPUS" \
  --log-dir "$LOG_DIR" \
  --num-workers "$NUM_WORKERS"
