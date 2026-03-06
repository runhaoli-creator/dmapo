#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all_baselines.sh — Train all baselines, generate + judge on all benchmarks
#
# Phase 1: Train 4 new baselines (SFT, KTO, ORPO, SimPO)   [~2-3 hrs total]
# Phase 2: Generate on 3 benchmarks × 7 models              [~4-5 hrs total]
# Phase 3: Judge MT-Bench + AlpacaEval × 7 models           [~3-4 hrs total]
# Phase 4: Run IFEval (rule-based, no judge) × 7 models     [~2-3 hrs total]
# Phase 5: Compile final summary tables
#
# Assign GPUs: 0-7 available. Run training sequentially (needs full GPU),
# then parallelize generation + judging.
#
# Usage:
#   bash scripts/run_all_baselines.sh          # run everything
#   bash scripts/run_all_baselines.sh phase2   # skip training, start from gen
#   bash scripts/run_all_baselines.sh phase3   # skip to judging
#   bash scripts/run_all_baselines.sh phase4   # skip to IFEval
#   bash scripts/run_all_baselines.sh phase5   # skip to summary
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PY=/home/kztrgg/miniconda3/envs/dmapo/bin/python
export PYTHONPATH="/home/kztrgg/dmapo/src"
export HF_TOKEN="${HF_TOKEN:-}"
cd /home/kztrgg/dmapo

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
BENCH_DIR="outputs/bench"
LOG_DIR="outputs/bench/logs"
mkdir -p "$BENCH_DIR" "$LOG_DIR"

PHASE="${1:-all}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Train baselines (sequential — each needs ~15 GB VRAM)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PHASE" == "all" || "$PHASE" == "phase1" ]]; then
    echo "═══ PHASE 1: Training baselines ═══"

    # SFT baseline (GPU 0)
    if [[ ! -f outputs/sft_baseline/adapter_config.json ]]; then
        echo "[$(date)] Training SFT baseline …"
        CUDA_VISIBLE_DEVICES=0 $PY scripts/train_sft_baseline.py \
            --training-config configs/training.yaml \
            --output-dir outputs/sft_baseline \
            --max-samples 10000 \
            2>&1 | tee "$LOG_DIR/train_sft.log"
    else
        echo "[$(date)] SFT baseline already exists, skipping."
    fi

    # KTO baseline (GPU 0)
    if [[ ! -f outputs/kto_baseline/adapter_config.json ]]; then
        echo "[$(date)] Training KTO baseline …"
        CUDA_VISIBLE_DEVICES=0 $PY scripts/train_kto_baseline.py \
            --training-config configs/training.yaml \
            --output-dir outputs/kto_baseline \
            --max-samples 10000 \
            2>&1 | tee "$LOG_DIR/train_kto.log"
    else
        echo "[$(date)] KTO baseline already exists, skipping."
    fi

    # ORPO baseline (GPU 0)
    if [[ ! -f outputs/orpo_baseline/adapter_config.json ]]; then
        echo "[$(date)] Training ORPO baseline …"
        CUDA_VISIBLE_DEVICES=0 $PY scripts/train_orpo_baseline.py \
            --training-config configs/training.yaml \
            --output-dir outputs/orpo_baseline \
            --max-samples 10000 \
            2>&1 | tee "$LOG_DIR/train_orpo.log"
    else
        echo "[$(date)] ORPO baseline already exists, skipping."
    fi

    # SimPO baseline (GPU 0)
    if [[ ! -f outputs/simpo_baseline/adapter_config.json ]]; then
        echo "[$(date)] Training SimPO baseline …"
        CUDA_VISIBLE_DEVICES=0 $PY scripts/train_simpo_baseline.py \
            --training-config configs/training.yaml \
            --output-dir outputs/simpo_baseline \
            --max-samples 10000 \
            2>&1 | tee "$LOG_DIR/train_simpo.log"
    else
        echo "[$(date)] SimPO baseline already exists, skipping."
    fi

    echo "[$(date)] Phase 1 complete."
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Generate responses (parallel across GPUs)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PHASE" == "all" || "$PHASE" == "phase2" ]]; then
    echo "═══ PHASE 2: Generating responses ═══"

    # Models: base(done), dpo(done), dmapo(done), sft, kto, orpo, simpo
    # Benchmarks: mt_bench, alpaca_eval (IFEval done in phase 4)
    # Only generate for new baselines — existing ones already done

    declare -A ADAPTERS=(
        ["sft"]="outputs/sft_baseline"
        ["kto"]="outputs/kto_baseline"
        ["orpo"]="outputs/orpo_baseline"
        ["simpo"]="outputs/simpo_baseline"
    )

    GPU=0
    PIDS=()
    for tag in sft kto orpo simpo; do
        adapter="${ADAPTERS[$tag]}"
        for bench in mt_bench alpaca_eval; do
            outfile="$BENCH_DIR/${tag}_${bench}.jsonl"
            if [[ -f "$outfile" ]]; then
                echo "  [skip] $outfile exists"
                continue
            fi
            echo "  [GPU $GPU] Generating $tag $bench …"
            CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/bench_generate.py \
                --model "$MODEL" --adapter "$adapter" \
                --bench "$bench" --output "$outfile" \
                > "$LOG_DIR/gen_${tag}_${bench}.log" 2>&1 &
            PIDS+=($!)
            GPU=$(( (GPU + 1) % 8 ))
        done
    done

    # Wait for all generation jobs
    echo "  Waiting for ${#PIDS[@]} generation jobs …"
    for pid in "${PIDS[@]}"; do
        wait "$pid" && echo "  PID $pid done" || echo "  PID $pid FAILED"
    done
    echo "[$(date)] Phase 2 complete."
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Judge MT-Bench + AlpacaEval (parallel across GPUs)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PHASE" == "all" || "$PHASE" == "phase3" ]]; then
    echo "═══ PHASE 3: Judging (Qwen3-8B) ═══"

    GPU=0
    PIDS=()
    for tag in sft kto orpo simpo; do
        for bench in mt_bench alpaca_eval; do
            infile="$BENCH_DIR/${tag}_${bench}.jsonl"
            outfile="$BENCH_DIR/${tag}_${bench}_judged.jsonl"
            if [[ -f "$outfile" ]]; then
                echo "  [skip] $outfile exists"
                continue
            fi
            if [[ ! -f "$infile" ]]; then
                echo "  [WARN] $infile not found, skipping"
                continue
            fi
            echo "  [GPU $GPU] Judging $tag $bench …"
            CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/bench_judge_local.py \
                --bench "$bench" --input "$infile" --output "$outfile" \
                > "$LOG_DIR/judge_${tag}_${bench}.log" 2>&1 &
            PIDS+=($!)
            GPU=$(( (GPU + 1) % 8 ))
        done
    done

    echo "  Waiting for ${#PIDS[@]} judging jobs …"
    for pid in "${PIDS[@]}"; do
        wait "$pid" && echo "  PID $pid done" || echo "  PID $pid FAILED"
    done
    echo "[$(date)] Phase 3 complete."
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: IFEval (rule-based — generate + evaluate in one script)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PHASE" == "all" || "$PHASE" == "phase4" ]]; then
    echo "═══ PHASE 4: IFEval ═══"

    declare -A ALL_ADAPTERS=(
        ["base"]=""
        ["dpo"]="outputs/dpo_baseline"
        ["dmapo"]="outputs/dmapo_policy"
        ["sft"]="outputs/sft_baseline"
        ["kto"]="outputs/kto_baseline"
        ["orpo"]="outputs/orpo_baseline"
        ["simpo"]="outputs/simpo_baseline"
    )

    GPU=0
    PIDS=()
    for tag in base dpo dmapo sft kto orpo simpo; do
        outfile="$BENCH_DIR/${tag}_ifeval.jsonl"
        if [[ -f "$outfile" ]]; then
            echo "  [skip] $outfile exists"
            continue
        fi
        adapter="${ALL_ADAPTERS[$tag]}"
        adapter_flag=""
        if [[ -n "$adapter" ]]; then
            adapter_flag="--adapter $adapter"
        fi
        echo "  [GPU $GPU] IFEval $tag …"
        CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/bench_ifeval.py \
            --model "$MODEL" $adapter_flag \
            --output "$outfile" \
            > "$LOG_DIR/ifeval_${tag}.log" 2>&1 &
        PIDS+=($!)
        GPU=$(( (GPU + 1) % 8 ))
    done

    echo "  Waiting for ${#PIDS[@]} IFEval jobs …"
    for pid in "${PIDS[@]}"; do
        wait "$pid" && echo "  PID $pid done" || echo "  PID $pid FAILED"
    done
    echo "[$(date)] Phase 4 complete."
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Compile summary
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PHASE" == "all" || "$PHASE" == "phase5" ]]; then
    echo "═══ PHASE 5: Summary ═══"
    $PY scripts/bench_summary_full.py --bench-dir "$BENCH_DIR"
    echo "[$(date)] All done! Results in outputs/paper/"
fi

echo "════════════════════════════════════════════════════════════"
echo "  All phases complete. Check outputs/paper/ for tables."
echo "════════════════════════════════════════════════════════════"
