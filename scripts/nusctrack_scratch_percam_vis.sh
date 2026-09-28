#!/usr/bin/env bash
# Train MVTracker from scratch on NuscTrack with per-camera visibility.
# Does not load Kubric / FT / previous any-view scratch weights.
# Usage:
#   tmux new -s yy
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/nusctrack_scratch_percam_vis.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the GPUs to use (skip busy cards)." >&2
  exit 1
fi
exec .venv/bin/python -m mvtracker.cli.train +experiment=mvtracker_nusctrack_scratch_percam_vis "$@"
