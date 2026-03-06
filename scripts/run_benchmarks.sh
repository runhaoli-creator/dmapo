#!/usr/bin/env bash
# scripts/run_benchmarks.sh
# ─────────────────────────
# Full benchmark pipeline:
#   1. Generate responses for Base / DPO-baseline / DMAPO on MT-Bench & AlpacaEval
#   2. Judge with GPT-4o
#   3. Print comparison table
#
# Prerequisites: OPENAI_API_KEY, HF_TOKEN, conda env dmapo
#
# Usage:
#   bash scripts/run_benchmarks.sh              # full run
#   bash scripts/run_benchmarks.sh --skip-dpo   # skip DPO baseline training
#   bash scripts/run_benchmarks.sh --judge-only  # skip generation, just judge

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DMAPO_ADAPTER="outputs/dmapo_policy"
DPO_ADAPTER="outputs/dpo_baseline"
BENCH_DIR="outputs/bench"

mkdir -p "$BENCH_DIR"

SKIP_DPO=false
JUDGE_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-dpo) SKIP_DPO=true ;;
        --judge-only) JUDGE_ONLY=true ;;
    esac
done

# ── Step 1: Train DPO baseline (if needed) ──────────────────────────────────
if [[ "$SKIP_DPO" == false && "$JUDGE_ONLY" == false ]] && [[ ! -f "$DPO_ADAPTER/adapter_config.json" ]]; then
    echo "═══ Training DPO baseline ═══"
    CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo_baseline.py \
        --training-config configs/training.yaml \
        --output-dir "$DPO_ADAPTER"
fi

# ── Step 2: Generate on benchmarks (3 models × 2 benchmarks = 6 runs) ───────
if [[ "$JUDGE_ONLY" == false ]]; then

    echo "═══ Generating: Base model on MT-Bench ═══"
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_generate.py \
        --model "$BASE_MODEL" --bench mt_bench \
        --output "$BENCH_DIR/base_mt_bench.jsonl" &
    PID_BASE_MT=$!

    echo "═══ Generating: Base model on AlpacaEval ═══"
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_generate.py \
        --model "$BASE_MODEL" --bench alpaca_eval \
        --output "$BENCH_DIR/base_alpaca_eval.jsonl" &
    PID_BASE_AE=$!

    echo "═══ Generating: DMAPO on MT-Bench ═══"
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_generate.py \
        --model "$BASE_MODEL" --adapter "$DMAPO_ADAPTER" --bench mt_bench \
        --output "$BENCH_DIR/dmapo_mt_bench.jsonl" &
    PID_DMAPO_MT=$!

    echo "═══ Generating: DMAPO on AlpacaEval ═══"
    CUDA_VISIBLE_DEVICES=3 python scripts/bench_generate.py \
        --model "$BASE_MODEL" --adapter "$DMAPO_ADAPTER" --bench alpaca_eval \
        --output "$BENCH_DIR/dmapo_alpaca_eval.jsonl" &
    PID_DMAPO_AE=$!

    if [[ -f "$DPO_ADAPTER/adapter_config.json" ]]; then
        echo "═══ Generating: DPO-baseline on MT-Bench ═══"
        CUDA_VISIBLE_DEVICES=4 python scripts/bench_generate.py \
            --model "$BASE_MODEL" --adapter "$DPO_ADAPTER" --bench mt_bench \
            --output "$BENCH_DIR/dpo_mt_bench.jsonl" &
        PID_DPO_MT=$!

        echo "═══ Generating: DPO-baseline on AlpacaEval ═══"
        CUDA_VISIBLE_DEVICES=5 python scripts/bench_generate.py \
            --model "$BASE_MODEL" --adapter "$DPO_ADAPTER" --bench alpaca_eval \
            --output "$BENCH_DIR/dpo_alpaca_eval.jsonl" &
        PID_DPO_AE=$!
    fi

    echo "Waiting for all generation jobs …"
    wait $PID_BASE_MT $PID_BASE_AE $PID_DMAPO_MT $PID_DMAPO_AE
    [[ -v PID_DPO_MT ]] && wait $PID_DPO_MT $PID_DPO_AE
    echo "All generation complete."
fi

# ── Step 3: Judge with GPT-4o ────────────────────────────────────────────────
echo "═══ Judging with GPT-4o ═══"

for model_tag in base dmapo dpo; do
    for bench in mt_bench alpaca_eval; do
        input="$BENCH_DIR/${model_tag}_${bench}.jsonl"
        output="$BENCH_DIR/${model_tag}_${bench}_judged.jsonl"
        [[ ! -f "$input" ]] && continue
        [[ -f "$output" ]] && echo "  ✓ $output already exists, skipping" && continue
        echo "  Judging: $model_tag / $bench"
        python scripts/bench_judge.py --bench "$bench" --input "$input" --output "$output"
    done
done

# ── Step 4: Summary table ───────────────────────────────────────────────────
echo ""
echo "═══ Final Results ═══"
python scripts/bench_summary.py --bench-dir "$BENCH_DIR"
