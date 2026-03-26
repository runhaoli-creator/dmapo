# DMAPO

## Overview
- **What**: Direct Multi-Agent Policy Optimization (DMAPO) — research project
- **Target venue**: COLM 2025
- **Repo**: local
- **Collaborators**: [fill in]

## Tech Stack
- Python, PyTorch

## Project Structure
```
├── configs/          # YAML configs (arbitration, data, eval, generation, judges, training)
├── src/dmapo/        # Main source code
├── scripts/          # Training, eval, benchmark scripts
├── paper/            # DO NOT TOUCH — original paper files
├── paper_runhao/     # Active paper drafts — edit dmapo_complete_neurips.tex only
├── outputs/          # Training outputs (gitignored)
└── STATUS.md         # Project status
```

## Key Commands
```bash
# Train DPO
python scripts/train_dpo.py

# Train KTO
python scripts/train_kto.py

# Run full pipeline
bash scripts/run_pipeline.sh

# Run benchmarks
bash scripts/run_benchmarks.sh

# Run all baselines
bash scripts/run_all_baselines.sh

# Evaluate
python scripts/run_eval.py
```

## Paper Rules
- **Only edit**: `paper_runhao/dmapo_complete_neurips.tex`
- **Never touch**: anything in `paper/` directory

## Code Conventions
- Follow existing style in this repo
- Config changes go in configs/, not hardcoded
- Every experiment must log: config, seed, git hash
- Results files go in results/ (git tracked), large files in outputs/ (gitignored)

## Current Focus
First draft complete. Need to improve writing quality and add more experiments for COLM 2025.
