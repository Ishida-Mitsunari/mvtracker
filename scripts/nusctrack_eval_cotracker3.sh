#!/usr/bin/env bash
# Clip-sharded CoTracker3 eval on NuscTrack (not DDP).
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash scripts/nusctrack_eval_cotracker3.sh          # full val
#   CUDA_VISIBLE_DEVICES=0 bash scripts/nusctrack_eval_cotracker3.sh nusctrack-val-max2   # smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
DATASET="${1:-nusctrack-val}"
if [[ "$DATASET" == "nusctrack-val-max2" ]]; then
  LOG_DIR="${ROOT}/logs/cotracker3_offline_nusctrack"
else
  LOG_DIR="${ROOT}/logs/cotracker3_offline_nusctrack_val"
fi
GPUS="${CUDA_VISIBLE_DEVICES:-all}"
exec .venv/bin/python -m mvtracker.cli.eval_nusctrack_parallel \
  --dataset "$DATASET" \
  --gpus "$GPUS" \
  --log-dir "$LOG_DIR"
