#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/run_pipeline.sh  –  DMAPO full pipeline runner
#
# Usage:
#   bash scripts/run_pipeline.sh [OPTIONS]
#
# Options:
#   --from N          Start from stage N (default: 1)
#   --to N            Stop after stage N (default: 7)
#   --stage N         Run only stage N
#   --skip-generation Skip perplexity/win-rate in eval (faster, no model load)
#   --dry-run         Quick end-to-end check: 20 samples, stages 1-4 only
#   --max-samples N   Override max samples for stages 1 & 2 (implies --to 4)
#   --help            Show this help
#
# Stages:
#   1  prepare_prompts
#   2  generate_candidates
#   3  score_candidates
#   4  build_kto_dataset
#   5  train_kto
#   6  train_dpo          (optional baseline — set --to 5 to skip)
#   7  run_eval
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

START_STAGE=1
END_STAGE=7
SKIP_GEN=""
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)           START_STAGE="$2"; shift 2 ;;
    --to)             END_STAGE="$2"; shift 2 ;;
    --stage)          START_STAGE="$2"; END_STAGE="$2"; shift 2 ;;
    --skip-generation) SKIP_GEN="--skip-generation"; shift ;;
    --dry-run)        MAX_SAMPLES=20; END_STAGE=4; shift ;;
    --max-samples)    MAX_SAMPLES="$2"; END_STAGE=4; shift 2 ;;
    --help)
      sed -n '2,32p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Conda activation ──────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate dmapo 2>/dev/null || true
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="logs/pipeline_${TS}.log"

_log()  { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$LOGFILE"; }
_skip() { _log "⏭  Skipping stage $1: $2"; }
_ok()   { _log "✓  Stage $1 complete"; }
_run()  {
  local n="$1" name="$2"; shift 2
  if [[ $n -ge $START_STAGE && $n -le $END_STAGE ]]; then
    _log "━━━ Stage ${n}: ${name} ━━━"
    "$@" 2>&1 | tee -a "$LOGFILE"
    _ok "$n"
  else
    _skip "$n" "$name"
  fi
}

DRY_TAG=""
[[ -n "$MAX_SAMPLES" ]] && DRY_TAG=" | dry-run (max-samples=${MAX_SAMPLES})"
_log "DMAPO pipeline | stages ${START_STAGE}–${END_STAGE}${DRY_TAG} | log → ${LOGFILE}"

# ── Stage 1: Prepare prompts ──────────────────────────────────────────────────
_run 1 "Prepare prompts" \
  python scripts/prepare_prompts.py --config configs/data.yaml \
  ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}

# ── Stage 2: Generate candidates ─────────────────────────────────────────────
_run 2 "Generate candidates" \
  python scripts/generate_candidates.py --config configs/generation.yaml \
  ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}

# ── Stage 3: Score candidates ─────────────────────────────────────────────────
_run 3 "Score candidates (judges + critic + gating)" \
  python scripts/score_candidates.py \
    --judges-config configs/judges.yaml \
    --arbitration-config configs/arbitration.yaml

# ── Stage 4: Build KTO/DPO datasets ──────────────────────────────────────────
_run 4 "Build KTO/DPO datasets" \
  python scripts/build_kto_dataset.py \
    --training-config configs/training.yaml \
    --arbitration-config configs/arbitration.yaml

# ── Stage 5: Train KTO ───────────────────────────────────────────────────────
_run 5 "Train KTO policy" \
  python scripts/train_kto.py --config configs/training.yaml

# ── Stage 6: Train DPO (optional baseline) ───────────────────────────────────
_run 6 "Train DPO baseline" \
  python scripts/train_dpo.py --config configs/training.yaml

# ── Stage 7: Evaluate ────────────────────────────────────────────────────────
_run 7 "Evaluate" \
  python scripts/run_eval.py \
    --eval-config configs/eval.yaml \
    --training-config configs/training.yaml \
    --arbitration-config configs/arbitration.yaml \
    --judges-config configs/judges.yaml \
    $SKIP_GEN

_log "Pipeline complete. Outputs in outputs/ and logs/"
