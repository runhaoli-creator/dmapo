# DMAPO Paper — Remaining Modifications for COLM 2026

Items already fixed in the tex file have been removed. Only outstanding work remains.

---

## Tier 1: Must-Do (rejection risk without these)

### 1. Multi-Seed Experiments
- **Problem**: All results are single seed (seed=42). No reviewer will trust these numbers.
- **Action**: Run 3 seeds minimum (42, 123, 456) for DMAPO and ALL baselines. Report mean +/- std in all tables.
- **Owner**: Experiment team
- **Priority**: Start ASAP — this is the most time-consuming fix

### 2. Replace Figure Placeholders with Real Figures
Placeholder `\fbox` blocks have been added to the tex file. Replace them with actual figures:
- **Fig 1 (pipeline)**: Diagram of the 6 stages. Location: after `\label{sec:method}`, search for `fig:pipeline`.
- **Fig 2 (score distribution)**: Histogram before vs after gating. Location: after confidence gating section, search for `fig:score_dist`.
- **Fig 3 (training dynamics)**: Loss curves and reward margin for DMAPO vs DPO/KTO baselines. This one still needs to be added to the tex — insert after main results table.
- **Owner**: Paper team (Fig 1 = diagram tool), Experiment team (Fig 2-3 = plot from logs)

### 3. Deepen the Analysis ("Why" Section)
- **Problem**: Section 5 (Discussion) is surface-level. COLM wants insight, not just numbers.
- **Add qualitative examples**: Show 2-3 concrete cases where the gating correctly filtered a noisy "preferred" example from UltraFeedback. Makes the noise claim tangible.
- **Add behavioral analysis**: How do DMAPO outputs differ from DPO outputs? Are they shorter? More structured? Less hedging? Compare generation characteristics.
- **Add failure analysis**: Where does DMAPO still fail? What prompt types does the gating struggle with? (e.g., creative writing — consistent with the Writing category weakness now acknowledged in per-category analysis)
- **Owner**: Paper team + experiment team (need generation samples)

---

## Tier 2: Strongly Recommended (separates accept from borderline)

### 4. Data Filtering Baselines (critical ablation gap)
- **Problem**: Is the gain from multi-agent consensus, or from just training on fewer examples?
- **Add these baselines** (all using 1.9k examples to match DMAPO):
  - Random 1.9k subset from the full 54k candidates
  - Top-1.9k by single holistic judge (no multi-agent)
  - Top-1.9k by mean of 3 judges WITHOUT variance gate
- **Why**: If random-1.9k performs similarly, the whole story collapses. If it doesn't, this is your strongest evidence for multi-agent consensus.
- **Owner**: Experiment team

### 5. Second Model
- **Problem**: Mistral-7B only — reviewers will question generalizability.
- **Action**: Run the full pipeline on at least one more model: Llama-3-8B or Qwen2.5-7B. Even just Table 1 equivalent (main results on 4 benchmarks) is sufficient.
- **Owner**: Experiment team

### 6. Verify AlpacaEval Numbers
- **Problem**: 96.3% for a 7B model is suspiciously high. Raw win rate vs LC win rate matters.
- **Action**: Confirm which metric variant is reported. If raw win rate, switch to length-controlled (LC) win rate — that's the standard. Re-run if needed.
- **Owner**: Experiment team

### 7. More Recent Baselines
- **Problem**: Current baselines (DPO, KTO, ORPO, SimPO) are 2023-2024. COLM reviewers expect newer methods.
- **Consider adding**: SPPO (self-play preference optimization), Online/Iterative DPO, or other 2024-2025 methods relevant to data-centric preference learning.
- **Owner**: Experiment team

---

## Suggested Execution Order

```
Week 1:  Start multi-seed runs (#1) + data filtering baselines (#4)
         Start second model runs if feasible (#5)
Week 2:  Generate real figures (#2) from experiment logs
         Verify AlpacaEval metric (#6)
Week 3:  Write deeper analysis section (#3) — needs generation samples
         Update all tables with mean +/- std
Week 4:  Final polish, proofread, submit
```

---

## Quick Reference

| # | Issue | Severity | Owner |
|---|-------|----------|-------|
| 1 | Single-seed results | Major | Experiment |
| 2 | Figure placeholders need real figures | Major | Paper + Experiment |
| 3 | Shallow analysis/discussion | Major | Paper + Experiment |
| 4 | Missing data filtering baselines | Major | Experiment |
| 5 | Single model only | Medium | Experiment |
| 6 | AlpacaEval metric unclear | Medium | Experiment |
| 7 | No recent baselines | Medium | Experiment |

---

## Already Fixed (for reference)

The following were fixed directly in `dmapo_complete_neurips.tex`:
- Switched from NeurIPS 2025 to COLM 2026 style (`colm2026_conference.sty`)
- Removed `[final]` flag (now anonymous `[submission]` mode)
- Removed conflicting `\usepackage[margin=1in]{geometry}`
- Fixed all 6 broken/missing citations (added bibitems, fixed keys)
- Replaced wrong data-centric AI citation (Andrychowicz -> Zha et al.)
- Rewrote DPO-IPO sentence to correctly describe SimPO
- Fixed bare `\cite{}` at sentence starts (added author names)
- Removed unused Gao et al. bibitem
- Fixed Table 1 column spec (removed phantom leading `&`, corrected to 6 columns)
- Fixed "5x" -> "5-10x" in abstract, contributions, and conclusion
- Acknowledged Writing category weakness in per-category analysis
- Added ethics/broader impact statement (replaced empty Appendix D)
- Moved hardware detail from method to appendix
- Added process critic mechanism description (structured output parsing)
- Added pipeline figure placeholder (Fig 1) and score distribution placeholder (Fig 2)
