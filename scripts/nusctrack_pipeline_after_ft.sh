#!/usr/bin/env bash
# After the current NuscTrack FT job exits, run:
#   1) 8-GPU clip-sharded eval of FT MVTracker
#   2) 8-GPU clip-sharded eval of official Kubric (zero-shot) MVTracker
#   3) 8-GPU scratch train (no Kubric weights)
#   4) 8-GPU clip-sharded eval of scratch MVTracker
#
# Weights stay in separate dirs; eval metrics go to separate log dirs.
# Launch (detached):
#   tmux new -s yangyi_mvtracker_pipeline -c /share/tgp/yangyi/mvtracker \
#     "bash scripts/nusctrack_pipeline_after_ft.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
POLL_SEC="${POLL_SEC:-60}"
N_TRAIN_CLIPS="${N_TRAIN_CLIPS:-973}"   # 7-GPU FT loader 139 * 7
N_EPOCHS="${N_EPOCHS:-64}"

FT_DIR="${ROOT}/logs/mvtracker_nusctrack_ft"
FT_CKPT="${FT_DIR}/model_final.pth"
OFFICIAL_CKPT="${ROOT}/checkpoints/mvtracker_200000_june2025.pth"
SCRATCH_DIR="${ROOT}/logs/mvtracker_nusctrack_scratch"
SCRATCH_CKPT="${SCRATCH_DIR}/model_final.pth"

EVAL_FT_DIR="${ROOT}/logs/mvtracker_nusctrack_ft_eval"
EVAL_ZS_DIR="${ROOT}/logs/mvtracker_nusctrack_zeroshot_eval"
EVAL_SCRATCH_DIR="${ROOT}/logs/mvtracker_nusctrack_scratch_eval"

LOG="${ROOT}/logs/nusctrack_pipeline_after_ft.log"
STATUS="${ROOT}/logs/nusctrack_pipeline_status.txt"
LOCK="${ROOT}/logs/nusctrack_pipeline_after_ft.lock"
mkdir -p "${ROOT}/logs"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Pipeline already running (lock $LOCK)." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

n_gpus() {
  tr ',' '\n' <<<"$GPUS" | grep -c .
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

write_status() {
  printf '%s\n' "$*" >>"$STATUS"
}

ft_train_running() {
  pgrep -f '[p]ython.*experiment=mvtracker_nusctrack_ft' >/dev/null 2>&1
}

wait_ft_done() {
  log "Waiting for FT train processes to exit (poll ${POLL_SEC}s)."
  while ft_train_running; do
    log "FT still running (latest ckpts in $FT_DIR)."
    sleep "$POLL_SEC"
  done
  log "FT python processes are gone."
  local i
  for i in 1 2 3 4 5 6; do
    if [[ -f "$FT_CKPT" ]]; then
      break
    fi
    log "Waiting for $FT_CKPT ($i/6)."
    sleep 10
  done
  if [[ ! -f "$FT_CKPT" ]]; then
    log "ERROR: FT finished without $FT_CKPT — not starting the pipeline."
    write_status "FAIL $(date '+%F %T') missing $FT_CKPT"
    exit 1
  fi
  log "Found FT checkpoint: $FT_CKPT ($(du -h "$FT_CKPT" | awk '{print $1}'))"
}

settle_gpus() {
  # 48GB cards; MVTracker eval/train ~16–25GB. A few GB leftover on GPU 2 is fine.
  sleep 20
  log "Proceeding on GPUs [$GPUS] (not waiting for leftover occupancy to reach 0)."
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    | while read -r line; do log "  gpu $line"; done
}

clear_eval_shards() {
  local dir="$1"
  mkdir -p "$dir"
  rm -f "$dir"/shard_rank*.pt "$dir"/model_load.lock
}

run_eval() {
  local name="$1" ckpt="$2" out="$3"
  log "=== START eval $name ==="
  write_status "START eval_$name $(date '+%F %T') ckpt=$ckpt out=$out"
  if [[ ! -f "$ckpt" ]]; then
    log "ERROR: missing ckpt $ckpt"
    write_status "FAIL eval_$name missing ckpt"
    return 1
  fi
  clear_eval_shards "$out"
  unset CUDA_VISIBLE_DEVICES
  bash "$ROOT/scripts/nusctrack_eval_mvtracker.sh" \
    --ckpt "$ckpt" \
    --gpus "$GPUS" \
    --dataset nusctrack-val \
    --log-dir "$out"
  if [[ ! -f "$out/nusctrack_metrics.txt" ]]; then
    log "ERROR: eval $name did not write $out/nusctrack_metrics.txt"
    write_status "FAIL eval_$name no metrics"
    return 1
  fi
  log "=== OK eval $name → $out/nusctrack_metrics.txt ==="
  write_status "OK eval_$name $(date '+%F %T')"
}

run_scratch_train() {
  local ng steps_per_epoch num_steps
  ng="$(n_gpus)"
  steps_per_epoch=$(( N_TRAIN_CLIPS / ng ))
  if (( steps_per_epoch < 1 )); then
    log "ERROR: steps_per_epoch=0 (N_TRAIN_CLIPS=$N_TRAIN_CLIPS ng=$ng)"
    return 1
  fi
  num_steps=$(( N_EPOCHS * steps_per_epoch ))
  log "=== START scratch train: ${ng} GPUs, ~${N_EPOCHS} epochs, num_steps=${num_steps} ==="
  write_status "START scratch_train $(date '+%F %T') gpus=$GPUS num_steps=$num_steps"

  mkdir -p "$SCRATCH_DIR"
  if compgen -G "$SCRATCH_DIR"'/model_*.pth' >/dev/null; then
    log "ERROR: $SCRATCH_DIR already has model_*.pth; refuse to resume as scratch."
    write_status "FAIL scratch_train nonempty $SCRATCH_DIR"
    return 1
  fi

  export CUDA_VISIBLE_DEVICES="$GPUS"
  bash "$ROOT/scripts/nusctrack_scratch.sh" "trainer.num_steps=${num_steps}"
  unset CUDA_VISIBLE_DEVICES
  if [[ ! -f "$SCRATCH_CKPT" ]]; then
    log "ERROR: scratch train finished without $SCRATCH_CKPT"
    write_status "FAIL scratch_train missing final ckpt"
    return 1
  fi
  log "=== OK scratch train → $SCRATCH_CKPT ==="
  write_status "OK scratch_train $(date '+%F %T')"
}

: >"$STATUS"
write_status "PIPELINE START $(date '+%F %T') gpus=$GPUS"
log "Pipeline start. ROOT=$ROOT GPUS=$GPUS"
log "FT weights:        $FT_DIR"
log "Official weights:  $OFFICIAL_CKPT"
log "Scratch weights:   $SCRATCH_DIR"
log "Eval FT:           $EVAL_FT_DIR/nusctrack_metrics.txt"
log "Eval zero-shot:    $EVAL_ZS_DIR/nusctrack_metrics.txt"
log "Eval scratch:      $EVAL_SCRATCH_DIR/nusctrack_metrics.txt"

fail=0
wait_ft_done
settle_gpus

run_eval ft "$FT_CKPT" "$EVAL_FT_DIR" || fail=1
run_eval zeroshot "$OFFICIAL_CKPT" "$EVAL_ZS_DIR" || fail=1

if ! run_scratch_train; then
  fail=1
  log "Skipping scratch eval because scratch train failed."
  write_status "SKIP eval_scratch"
else
  settle_gpus
  run_eval scratch "$SCRATCH_CKPT" "$EVAL_SCRATCH_DIR" || fail=1
fi

if (( fail == 0 )); then
  log "PIPELINE COMPLETE"
  write_status "PIPELINE COMPLETE $(date '+%F %T')"
else
  log "PIPELINE FINISHED WITH FAILURES — see $STATUS and $LOG"
  write_status "PIPELINE FAIL $(date '+%F %T')"
  exit 1
fi
