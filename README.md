# DMAPO — Data-centric Multi-Agent Preference Optimization

<p align="center">
  <strong>Quality over Quantity: Aligning LLMs with 5× Less Data via Multi-Agent Consensus</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#results">Results</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

DMAPO is a **data-centric** alignment pipeline that constructs high-quality preference training data through multi-agent LLM evaluation and aggressive confidence gating. Instead of collecting more noisy preference pairs, DMAPO invests computation in filtering—retaining only examples on which three independent LLM judge agents reach strong consensus.

**Key insight:** With only **1,871 training examples** (3.45% acceptance rate from 54k candidates), DMAPO outperforms all baselines trained on 10k–20k examples across all four benchmarks—and is the **only** method that improves over the base model on MT-Bench.

### Method at a Glance

```
          ┌──────────────────────────────────────────────────────────────┐
          │                   DMAPO Pipeline                            │
          │                                                              │
          │  Stage 1: Prompt Collection (UltraFeedback + HelpSteer2)    │
          │      ↓ 14,272 prompts                                       │
          │  Stage 2: On-Policy Candidate Generation (k=4)              │
          │      ↓ 54,236 candidates                                    │
          │  Stage 3: Multi-Agent Scoring                               │
          │      ├─ Helpfulness Judge (Qwen3-8B)                        │
          │      ├─ Factuality Judge  (Qwen3-8B)                        │
          │      └─ Conciseness Judge (Qwen3-8B)                        │
          │      ↓                                                       │
          │  Stage 4: Process Critic (reasoning flaw detection)         │
          │      ↓                                                       │
          │  Stage 5: Confidence Gating (3.45% acceptance)              │
          │      ↓ 1,871 high-quality examples                          │
          │  Stage 6: KTO Policy Training                               │
          │      ↓                                                       │
          │  ✓ Aligned Model                                            │
          └──────────────────────────────────────────────────────────────┘
```

---

## Results

All methods fine-tune **Mistral-7B-Instruct-v0.2** with identical LoRA configuration (rank 16, α=32).

### Main Results

| Method | Training Data | MT-Bench ↑ | AlpacaEval 2.0 ↑ | IFEval ↑ | Internal Win-Rate ↑ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Base (pretrained) | — | 7.41 | 96.0% | 41.2% | — |
| + SFT | 10k | 7.18 | 95.4% | 42.5% | 62.0% |
| + DPO | 10k | 7.08 | 95.7% | 43.8% | 65.1% |
| + KTO | 20k | 7.22 | 95.5% | 43.1% | 67.4% |
| + ORPO | 10k | 7.15 | 95.3% | 42.0% | 63.6% |
| + SimPO | 10k | 7.25 | 95.8% | 44.2% | 68.2% |
| **+ DMAPO (ours)** | **1.9k** | **7.62** | **96.3%** | **46.8%** | **85.3%** |

> **Key finding:** All baselines *degrade* MT-Bench performance below the base model. DMAPO is the only method that improves it — demonstrating that data quality dominates data quantity in preference optimization.

### Quality-Gated Data Statistics

| Statistic | Value |
|-----------|-------|
| Source prompts | 14,272 |
| Candidates generated | 54,236 |
| Acceptance rate | 3.45% |
| Gated training examples | 1,871 (951 desirable / 920 undesirable) |
| Desirable mean score | 9.23 ± 1.09 |
| Undesirable mean score | 2.42 ± 1.10 |
| Quality gap | ~6.8 points |
| Inter-judge Cohen's κ | 0.64 (mean) |

---

## Pipeline

### Six-Stage Architecture

```
Prompts (UltraFeedback + HelpSteer2)
    │
    ▼  Stage 1: prepare_prompts.py
Normalised JSONL  →  data/processed/all_prompts.jsonl
    │
    ▼  Stage 2: generate_candidates.py
Candidates JSONL  →  data/processed/candidates_{train,val}.jsonl
    │
    ▼  Stage 3: score_candidates.py
         ├─ Helpfulness judge
         ├─ Factuality judge
         ├─ Conciseness judge
         ├─ Process critic  (reasoning penalty α=0.15)
         └─ Confidence gate (variance + score-gap filtering)
Gated JSONL  →  data/processed/gated_{train,val}.jsonl
    │
    ▼  Stage 4: build_kto_dataset.py
KTO JSONL  →  data/processed/kto_{train,val}.jsonl
DPO JSONL  →  data/processed/dpo_{train,val}.jsonl
    │
    ▼  Stage 5: train_kto.py  (or train_dpo.py for baseline)
Policy model  →  outputs/dmapo_policy/
    │
    ▼  Stage 6: run_eval.py
outputs/eval/metrics.json  •  summary.csv  •  report.md
```

---

## Project Structure

```
dmapo/
├── configs/
│   ├── data.yaml            # Prompt datasets, splits, output paths
│   ├── generation.yaml      # Candidate generation model & sampling params
│   ├── judges.yaml          # Judge models, critic config, scoring params
│   ├── arbitration.yaml     # Variance & quality gating thresholds
│   ├── training.yaml        # Model, LoRA, KTO/DPO hyperparams
│   └── eval.yaml            # Evaluation settings & output paths
├── scripts/
│   ├── prepare_prompts.py         # Stage 1: prompt collection
│   ├── generate_candidates.py     # Stage 2: on-policy generation
│   ├── generate_candidates_parallel.sh
│   ├── score_candidates.py        # Stage 3: multi-agent scoring + gating
│   ├── score_candidates_parallel.sh
│   ├── build_kto_dataset.py       # Stage 4: build preference datasets
│   ├── train_kto.py               # Stage 5: KTO training (default)
│   ├── train_dpo.py               # Stage 5: DPO training (baseline)
│   ├── train_dpo_baseline.py      # DPO baseline on raw data
│   ├── train_kto_baseline.py      # KTO baseline on raw data
│   ├── train_sft_baseline.py      # SFT baseline
│   ├── train_orpo_baseline.py     # ORPO baseline
│   ├── train_simpo_baseline.py    # SimPO baseline
│   ├── run_eval.py                # Stage 6: evaluation
│   ├── run_pipeline.sh            # Full pipeline orchestrator
│   ├── run_all_baselines.sh       # Run all baseline experiments
│   ├── run_benchmarks.sh          # Run benchmark suite
│   ├── bench_generate.py          # Benchmark: generation
│   ├── bench_judge.py             # Benchmark: LLM judge eval
│   ├── bench_judge_local.py       # Benchmark: local judge eval
│   ├── bench_ifeval.py            # Benchmark: IFEval
│   ├── bench_summary.py           # Benchmark: summary
│   └── bench_summary_full.py      # Benchmark: full summary
├── src/dmapo/
│   ├── data/
│   │   ├── loader.py              # HF dataset loading & normalisation
│   │   ├── generator.py           # Candidate generation engine
│   │   ├── prepare_prompts.py
│   │   ├── generate_candidates.py
│   │   └── build_dataset.py
│   ├── judges/
│   │   ├── base_judge.py          # Abstract base judge
│   │   ├── helpfulness.py         # Helpfulness scoring (1-10)
│   │   ├── factuality.py          # Factuality scoring (1-10)
│   │   ├── conciseness.py         # Conciseness scoring (1-10)
│   │   ├── judge_pool.py          # Multi-judge orchestrator
│   │   └── scorer.py              # Unified scoring engine
│   ├── critics/
│   │   └── process_critic.py      # Reasoning flaw detector
│   ├── arbitration/
│   │   ├── confidence_gate.py     # Variance + quality gating
│   │   └── gating.py              # Gate logic utilities
│   ├── training/
│   │   ├── dataset_builder.py     # KTO & DPO dataset construction
│   │   ├── trainer.py             # Model loading / LoRA utils
│   │   └── train.py               # Training loop
│   └── eval/
│       ├── metrics.py             # All metric functions
│       └── evaluate.py            # Evaluation orchestrator
├── environment.yml                # Conda environment spec
├── requirements.txt               # Pip dependencies
└── README.md
```

---

## Setup

### Option A — Conda (recommended)

```bash
git clone https://github.com/<your-username>/dmapo.git
cd dmapo

conda env create -f environment.yml
conda activate dmapo
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### Option B — Pip

```bash
git clone https://github.com/<your-username>/dmapo.git
cd dmapo

conda create -n dmapo python=3.10 -y
conda activate dmapo
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### Requirements

- Python 3.10+
- PyTorch 2.1+
- 8× NVIDIA GPUs recommended (tested on RTX PRO 6000 Blackwell, ~760 GB VRAM total)
- Single-GPU training is supported (adjust `gradient_accumulation_steps` accordingly)

---

## Usage

### Full Pipeline (all stages)

```bash
bash scripts/run_pipeline.sh
```

### Stage-by-Stage

```bash
# Stage 1 — Download and normalise prompts
python scripts/prepare_prompts.py --config configs/data.yaml

# Stage 2 — Generate candidate responses (k=4 per prompt)
python scripts/generate_candidates.py --config configs/generation.yaml
# Or parallel across GPUs:
bash scripts/generate_candidates_parallel.sh

# Stage 3 — Score candidates (3 judges + process critic + confidence gate)
python scripts/score_candidates.py \
  --judges-config configs/judges.yaml \
  --arbitration-config configs/arbitration.yaml
# Or parallel:
bash scripts/score_candidates_parallel.sh

# Stage 4 — Build KTO and DPO preference datasets
python scripts/build_kto_dataset.py \
  --training-config configs/training.yaml \
  --arbitration-config configs/arbitration.yaml

# Stage 5 — Train DMAPO policy (KTO)
python scripts/train_kto.py --config configs/training.yaml
# Multi-GPU:
accelerate launch scripts/train_kto.py --config configs/training.yaml

# Stage 6 — Evaluate
python scripts/run_eval.py \
  --eval-config configs/eval.yaml \
  --training-config configs/training.yaml \
  --arbitration-config configs/arbitration.yaml \
  --judges-config configs/judges.yaml
```

### Run All Baselines

```bash
bash scripts/run_all_baselines.sh
```

### Run Benchmark Suite

```bash
bash scripts/run_benchmarks.sh
```

### Partial Pipeline

```bash
# Resume from stage 3
bash scripts/run_pipeline.sh --from 3

# Run up to stage 5 only
bash scripts/run_pipeline.sh --to 5

# Fast eval (skip generation metrics)
bash scripts/run_pipeline.sh --from 7 --skip-generation
```

---

## Configuration

All hyperparameters are controlled via YAML configs in `configs/`:

| Config | Key Parameters |
|--------|---------------|
| `data.yaml` | Source datasets, splits, output paths |
| `generation.yaml` | Model, k=4 candidates, T=0.8, top_p=0.95 |
| `judges.yaml` | Judge model (Qwen3-8B), critic penalty α=0.15 |
| `arbitration.yaml` | Variance threshold τ=2.5, quality gates (≥7.0 / ≤4.0) |
| `training.yaml` | LoRA r=16/α=32, KTO β=0.1, LR=5e-5, cosine schedule |
| `eval.yaml` | Evaluation metrics and output paths |

---

## Datasets

DMAPO uses publicly available datasets from HuggingFace. **No dataset files are included in this repository**—they are downloaded automatically during Stage 1.

| Dataset | HuggingFace Hub | Samples Used | Role |
|---------|----------------|:---:|------|
| **UltraFeedback** | [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback) | 10,000 | Diverse instruction-following prompts |
| **HelpSteer2** | [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | 5,000 | Helpfulness-focused prompts |

After deduplication and normalization, this yields **14,272 unique prompts** (95% train / 5% validation).

### Models Used

| Model | HuggingFace Hub | Role |
|-------|----------------|------|
| **Mistral-7B-Instruct-v0.2** | [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | Policy backbone (generation + training) |
| **Qwen3-8B** | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Judge agents + process critic |

---

## Outputs

When you run the full pipeline, the following artifacts are generated:

| Path | Description |
|------|-------------|
| `data/processed/all_prompts.jsonl` | Normalised prompt set |
| `data/processed/candidates_{train,val}.jsonl` | Generated candidates |
| `data/processed/scored_{train,val}.jsonl` | Multi-agent scored candidates |
| `data/processed/gated_{train,val}.jsonl` | Confidence-gated examples |
| `data/processed/kto_{train,val}.jsonl` | KTO training set |
| `data/processed/dpo_{train,val}.jsonl` | DPO training set |
| `outputs/dmapo_policy/` | Trained DMAPO policy (LoRA merged) |
| `outputs/eval/metrics.json` | Evaluation metrics |
| `outputs/eval/summary.csv` | Tabular summary |
| `outputs/eval/report.md` | Markdown evaluation report |
| `logs/pipeline_<timestamp>.log` | Full pipeline log |

> **Note:** Model checkpoints, processed datasets, and evaluation outputs are excluded from this repository via `.gitignore`. Run the pipeline to regenerate them.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{dmapo2025,
  title   = {DMAPO: Data-centric Multi-Agent Preference Optimization},
  author  = {Anonymous},
  year    = {2025},
  note    = {Under review at NeurIPS 2025}
}
```

---

## License

This project is released under the MIT License.
