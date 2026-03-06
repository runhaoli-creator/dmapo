#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/score_candidates_parallel.sh
#
# Launches 8 workers (one per GPU) to shard-score all candidates.
# After all finish, merges shard files and runs confidence gating on merged output.
#
# Usage:
#   bash scripts/score_candidates_parallel.sh [--num-gpus N] [--split both|train|val]
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NUM_GPUS=8
SPLIT="both"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --split)    SPLIT="$2";    shift 2 ;;
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

echo "Launching ${NUM_GPUS} scoring workers (split=${SPLIT}) …"

PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
  LOG="logs/stage3_gpu${i}_$(date +%Y%m%d_%H%M%S).log"
  CUDA_VISIBLE_DEVICES=$i python scripts/score_candidates.py \
    --judges-config configs/judges.yaml \
    --arbitration-config configs/arbitration.yaml \
    --split "$SPLIT" \
    --shard "$i" \
    --num-shards "$NUM_GPUS" \
    --gpu "$i" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "  GPU ${i}: PID ${PIDS[-1]}  log → ${LOG}"
done

echo ""
echo "All scoring workers launched. Waiting for completion…"
echo "(Monitor: watch -n30 'wc -l data/processed/scored_*_shard*.jsonl 2>/dev/null | tail -1')"
echo ""

# ── Wait ──────────────────────────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  if wait "$pid"; then
    echo "  ✓ GPU ${i} (PID ${pid}) done"
  else
    echo "  ✗ GPU ${i} (PID ${pid}) FAILED — check logs/stage3_gpu${i}_*.log"
    FAILED=1
  fi
done

if [[ $FAILED -eq 1 ]]; then
  echo "One or more workers failed. Aborting merge." >&2
  exit 1
fi

# ── Merge scored shards ──────────────────────────────────────────────────────
echo ""
echo "Merging scored shard files…"

merge() {
  local stem="$1"
  local final="${stem}.jsonl"
  local shards=( "${stem}_shard"[0-9][0-9].jsonl )

  if [[ ${#shards[@]} -eq 0 ]] || [[ ! -e "${shards[0]}" ]]; then
    echo "  No shards for ${stem} — skipping"
    return
  fi

  cat "${shards[@]}" > "$final"
  local n
  n=$(wc -l < "$final")
  echo "  ✓ ${final}  (${n} lines from ${#shards[@]} shards)"
  rm -f "${shards[@]}"
}

merge_gated() {
  local stem="$1"
  local final="${stem}.jsonl"
  local shards=( "${stem}_shard"[0-9][0-9].jsonl )

  if [[ ${#shards[@]} -eq 0 ]] || [[ ! -e "${shards[0]}" ]]; then
    echo "  No gated shards for ${stem} — skipping"
    return
  fi

  cat "${shards[@]}" > "$final"
  local n
  n=$(wc -l < "$final")
  echo "  ✓ ${final}  (${n} lines from ${#shards[@]} shards)"
  rm -f "${shards[@]}"
}

if [[ "$SPLIT" == "train" || "$SPLIT" == "both" ]]; then
  merge "data/processed/scored_train"
  merge_gated "data/processed/gated_train"
fi
if [[ "$SPLIT" == "val" || "$SPLIT" == "both" ]]; then
  merge "data/processed/scored_val"
  merge_gated "data/processed/gated_val"
fi

echo ""
echo "=== Final dataset stats ==="
for f in data/processed/scored_train.jsonl data/processed/scored_val.jsonl data/processed/gated_train.jsonl data/processed/gated_val.jsonl; do
  if [[ -f "$f" ]]; then
    printf "  %-45s %s lines\n" "$f" "$(wc -l < "$f")"
  fi
done

echo ""
echo "Stage 3 complete."
