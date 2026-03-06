#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/generate_candidates_parallel.sh
#
# Launches 8 independent workers, one per GPU, each handling 1/8 of the prompts.
# After all workers finish, merges shard files into the final JSONL outputs.
#
# Usage:
#   bash scripts/generate_candidates_parallel.sh [--num-gpus N] [--split both|train|val]
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NUM_GPUS=8
SPLIT="both"
CONFIG="configs/generation.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --split)    SPLIT="$2";    shift 2 ;;
    --config)   CONFIG="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dmapo
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:-}"

mkdir -p logs

echo "Launching ${NUM_GPUS} workers (split=${SPLIT}) …"

PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
  LOG="logs/stage2_gpu${i}_$(date +%Y%m%d_%H%M%S).log"
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_candidates.py \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --shard "$i" \
    --num-shards "$NUM_GPUS" \
    --gpu "$i" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "  GPU ${i}: PID ${PIDS[-1]}  log → ${LOG}"
done

echo ""
echo "All workers launched. Waiting for completion…"
echo "(Monitor with: watch -n10 'wc -l data/processed/candidates_train_shard*.jsonl | tail -1')"
echo ""

# ── Wait for all workers ──────────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  if wait "$pid"; then
    echo "  ✓ GPU ${i} (PID ${pid}) done"
  else
    echo "  ✗ GPU ${i} (PID ${pid}) FAILED — check logs/stage2_gpu${i}_*.log"
    FAILED=1
  fi
done

if [[ $FAILED -eq 1 ]]; then
  echo "One or more workers failed. Aborting merge." >&2
  exit 1
fi

# ── Merge shards ──────────────────────────────────────────────────────────────
echo ""
echo "Merging shard files…"

merge() {
  local stem="$1"       # e.g. data/processed/candidates_train
  local final="${stem}.jsonl"
  local shards=( "${stem}_shard"[0-9][0-9].jsonl )

  if [[ ${#shards[@]} -eq 0 ]] || [[ ! -e "${shards[0]}" ]]; then
    echo "  No shards found for ${stem} — skipping"
    return
  fi

  # Sort shards by shard index so order is deterministic
  cat "${shards[@]}" > "$final"
  local n
  n=$(wc -l < "$final")
  echo "  ✓ ${final}  (${n} lines from ${#shards[@]} shards)"
  rm -f "${shards[@]}"
}

if [[ "$SPLIT" == "train" || "$SPLIT" == "both" ]]; then
  merge "data/processed/candidates_train"
fi
if [[ "$SPLIT" == "val" || "$SPLIT" == "both" ]]; then
  merge "data/processed/candidates_val"
fi

echo ""
echo "Stage 2 complete."
