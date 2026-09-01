#!/usr/bin/env bash
# Train MVTracker from scratch on NuscTrack (no Kubric weights).
# Usage (after the FT job releases GPUs):
#   tmux new -s yangyi_mvtracker_scratch
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/nusctrack_scratch.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the GPUs to use (skip busy cards)." >&2
  exit 1
fi
exec .venv/bin/python -m mvtracker.cli.train +experiment=mvtracker_nusctrack_scratch "$@"
