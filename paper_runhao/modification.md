# DMAPO Paper Review — modification.md

## Summary (Phase 3 — Final Assessment)

### What was improved
- **Abstract**: Removed informal language ("rife with"); "We propose" → "\ours{} addresses"; tighter phrasing throughout
- **Introduction**: Consolidated 4 contributions to 3 with noun-phrase style; removed "We introduce/show/find" openers
- **Related Work**: Removed "To our knowledge" hedge; tightened confidence gating paragraph
- **Method §3.1-3.6**: Eliminated 8+ "We ..." sentence openers (e.g., "We combine" → "Prompts are drawn from", "We sample" → "Sampling from", "We fine-tune" → "The policy is fine-tuned with"); passive/active-noun constructions throughout
- **Results §4.2**: Converted 4-item bullet list restating Table 1 into flowing prose with analytical connections (IFEval rules out judge bias, internal win-rate quantifies preference signal)
- **Discussion §5.1**: Rewritten with analytical depth — now references Table 4 directly to explain the U-shaped acceptance-rate curve, identifies two regimes (noise-dominated vs. data-starved), connects 6.8-point quality gap to reward margin (10.96 vs. 1.88)
- **Conclusion**: Removed "We presented" and "We believe"; focused on practical implication and broader lesson about data quality vs. quantity
- **Throughout**: Reduced "We ..." sentence monotony across all sections

### What remains (cannot fix without experiments)
- **[E1]** All results are single seed (42) — need 3+ seeds with mean±std
- **[E2]** Both figures are placeholder text boxes — need real pipeline diagram and score distribution histogram
- **[E3]** Missing training dynamics figure (loss curves, reward margin over steps)
- **[E4]** Single model only (Mistral-7B) — need at least one more (e.g., Llama-3-8B)
- **[E5]** Missing data filtering baselines (random-1.9k, single-judge top-1.9k) to rule out trivial explanations
- **[E6]** AlpacaEval 96.3% may be raw win rate rather than length-controlled — needs verification
- **[E7]** No recent 2025 baselines (e.g., SPPO, Online DPO)
- **[E8]** No qualitative examples showing what the gating filters look like in practice

### Overall readiness
The writing quality meets COLM standards. The paper is well-structured with a clear story: noisy data → multi-agent filtering → better models with less data. The method is well-motivated, ablations are informative, and the Discussion now provides genuine analytical insight. The main blockers for submission are: (1) multi-seed experiments, (2) real figures, and (3) a second backbone model to demonstrate generality.

---

## Phase 1: Initial Review (2026-03-26)

---

### [WRITING] Issues

1. **W1** (Abstract, L52): "rife with mislabeled pairs" — informal register for a conference paper ✅ FIXED
2. **W2** (Abstract, L54): "We propose \ours{}" — formulaic opener ✅ FIXED
3. **W3** (Intro, L89-95): Contributions list with 4 items all starting "We introduce/show/find" — repetitive ✅ FIXED (consolidated to 3 items, noun-phrase style)
4. **W4** (Related Work, L124): "To our knowledge, none of these works" — hedged overclaim ✅ FIXED
5. **W5** (Method §3.1, L154): "We combine prompts from two complementary sources" — "We" opener ✅ FIXED
6. **W6** (Method §3.2, L166): "We sample from the same model" — "We" opener ✅ FIXED
7. **W7** (Method §3.2, L168): "To maximize throughput, we tile prompts" — wordy ✅ FIXED
8. **W8** (Method §3.6, L245): "We fine-tune the policy using KTO" — "We" opener ✅ FIXED
9. **W9** (Results §4.2, L328-334): 4-item bullet list restates Table 1 numbers with minimal insight ✅ FIXED (converted to prose)
10. **W10** (Discussion §5.1, L477-483): First two paragraphs partially repeat intro content about noisy data ✅ FIXED (rewritten around ablation table analysis)
11. **W11** (Conclusion, L521): "We presented \ours{}" — formulaic opener ✅ FIXED
12. **W12** (Conclusion, L526): "We believe these findings point to a broader opportunity" — hedging ✅ FIXED
13. **W13** (Setup §4.1, L282): "We compare against:" — "We" opener ✅ FIXED

### [STRUCTURE] Issues

14. **S1**: §4.6 Pipeline Statistics (Table 6) sits oddly between ablation and discussion — could be moved to appendix or merged with §3 Method. NOT FIXED (minor)
15. **S2**: §5.3 Computational Efficiency is only 2 sentences — too thin for its own subsection. NOT FIXED (minor)
16. **S3**: §5 Discussion has 4 subsections but §5.2 Role of the Process Critic (3 sentences) and §5.3 Computational Efficiency (2 sentences) are very thin. Could merge into a single analysis paragraph. NOT FIXED (minor)
17. **S4**: No qualitative examples anywhere in the paper — the gating mechanism is abstract without seeing concrete filtered examples. **[EXPERIMENT]**

### [FORMAT] Issues

18. **F1**: Figure 1 (pipeline diagram) is a placeholder `\fbox{}` text box. **[EXPERIMENT]**
19. **F2**: Figure 2 (score distribution) is a placeholder `\fbox{}` text box. **[EXPERIMENT]**
20. **F3**: No Figure 3 (training dynamics / loss curves) — would strengthen the training analysis in §3.6. **[EXPERIMENT]**
21. **F4**: Table 1 uses `\resizebox{0.9\textwidth}` which creates inconsistent column spacing — minor visual issue. NOT FIXED (cosmetic)

### [DATA] Inconsistencies

22. **D1**: AlpacaEval 2.0 reports 96.3% for a 7B model — suspiciously high. May be raw win rate rather than length-controlled (LC) win rate. LC is the standard. **[EXPERIMENT — needs verification]**
23. **D2**: Abstract says "5-10× fewer training examples than any baseline" — SFT uses 10k, KTO uses 20k, DMAPO uses 1.9k. Ratio is 5.3-10.7×. Correct.
24. **D3**: Table 2 per-category: Extraction shows Base at 7.90 (best), DMAPO at 7.85 — DMAPO slightly degrades Extraction. Not mentioned in per-category analysis text. Minor.
25. **D4**: Pipeline Statistics (Table 6) shows 951 desirable / 920 undesirable — roughly balanced (50.8/49.2%). Good data balance, not discussed. Minor.

### [EXPERIMENT] Issues (Cannot Fix)

26. **E1**: Single seed (42) for all results — no reviewer will trust without multi-seed statistics
27. **E2**: Both figures are placeholder text boxes — need actual diagrams
28. **E3**: Missing training dynamics figure (loss curves, reward margin vs. steps)
29. **E4**: Single backbone model (Mistral-7B only) — generalizability concern
30. **E5**: Missing critical data-filtering baselines: random-1.9k subset, single-judge top-1.9k, mean-of-3 without variance gate — needed to isolate multi-agent consensus contribution
31. **E6**: AlpacaEval metric variant unclear (raw vs. LC win rate)
32. **E7**: No recent 2025 baselines (SPPO, Online DPO, etc.)
33. **E8**: No qualitative examples of filtered vs. retained data — the gating story is abstract
34. **E9**: No analysis of what types of prompts get filtered out vs. retained

---

## Priority for Remaining Work

### Tier 1: Must-Do (rejection risk)
- **E1**: Multi-seed experiments (3+ seeds for DMAPO and all baselines)
- **E2+E3**: Replace figure placeholders with real pipeline diagram, score distribution, and training dynamics plots
- **E5**: Data filtering baselines (random-1.9k, single-judge-1.9k, no-variance-gate)

### Tier 2: Strongly Recommended
- **E4**: Second backbone model (Llama-3-8B or Qwen2.5-7B)
- **E6**: Verify AlpacaEval metric (raw vs. LC win rate)
- **E8**: Add 2-3 qualitative examples of gated vs. discarded data

### Tier 3: Nice to Have
- **E7**: Add 1-2 recent 2025 baselines
- **S1-S3**: Minor structural cleanup (merge thin subsections)

---

## Already Fixed (for reference)

The following were fixed in prior sessions directly in `dmapo_complete_neurips.tex`:
- Switched from NeurIPS 2025 to COLM 2026 style (`colm2026_conference.sty`)
- Removed `[final]` flag (now anonymous `[submission]` mode)
- Removed conflicting `\usepackage[margin=1in]{geometry}`
- Fixed all 6 broken/missing citations (added bibitems, fixed keys)
- Replaced wrong data-centric AI citation (Andrychowicz → Zha et al.)
- Rewrote DPO-IPO sentence to correctly describe SimPO
- Fixed bare `\cite{}` at sentence starts (added author names)
- Removed unused Gao et al. bibitem
- Fixed Table 1 column spec (removed phantom leading `&`, corrected to 6 columns)
- Fixed "5x" → "5-10x" in abstract, contributions, and conclusion
- Acknowledged Writing category weakness in per-category analysis
- Added ethics/broader impact statement (replaced empty Appendix D)
- Moved hardware detail from method to appendix
- Added process critic mechanism description (structured output parsing)
- Added pipeline figure placeholder (Fig 1) and score distribution placeholder (Fig 2)

The following were fixed in the current writing revision pass:
- Tightened abstract, removed informal language
- Consolidated contributions from 4 to 3 items
- Removed "To our knowledge" hedge in Related Work
- Eliminated 8+ "We ..." sentence openers across Method sections
- Converted Results §4.2 bullet list to analytical prose
- Rewrote Discussion §5.1 with ablation-grounded analysis (U-shaped curve, two regimes)
- Rewrote Conclusion to avoid repeating intro
