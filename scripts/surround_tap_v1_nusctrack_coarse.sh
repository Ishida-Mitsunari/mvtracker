#!/usr/bin/env bash
# Train SurroundTAP v1 coarse head only (Fabric DDP). Independent of v0 FT.
# Usage:
#   tmux attach -t yangyi
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/surround_tap_v1_nusctrack_coarse.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the GPUs to use (skip busy cards)." >&2
  exit 1
fi
exec .venv/bin/python -m mvtracker.cli.train +experiment=surround_tap_v1_nusctrack_coarse "$@"
