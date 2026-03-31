# DMAPO — Open Problems & Reviewer Risk Assessment

Last updated: 2026-03-30

---

## ✅ Fixed (this session)

| # | Issue | Fix applied |
|---|-------|-------------|
| F1 | Teaser caption said "Six stages" (should be seven) | → "Seven stages" |
| F2 | `wang2024selfinstruct` year 2024 in bib (should be 2023, ACL) | → `@inproceedings`, year 2023 |
| F3 | Orphan `\vspace{0.3em}` before `\end{table}` | Removed |
| F4 | Duplicate "Single epoch" paragraph | Removed (prior session) |
| F5 | Llama table MT-Bench bold/underline reversed (DMAPO 7.80 was underlined, SimPO 7.72 was bold) | Swapped correctly |
| F6 | Llama table AE WR second-best (SPPO 96.7) missing `\underline` | Added |
| F7 | "five benchmarks" ↔ "four benchmarks" contradiction in abstract and intro | Standardized to "four benchmarks" throughout |
| F8 | Line 294: "same 10k training budget" implied DMAPO uses 10k | Reworded clearly |
| F9 | Teaser Panel B caption said "MT-Bench results" but shows win-rate | → "Benchmark comparison" |
| F10 | Table 6 "Margin" column undefined | Added definition in caption |
| F11 | Training dynamics figure caption hid that ORPO/SPPO/REINFORCE++ are absent | Added "(representative subset; see Table for all methods)" |
| F12 | Contribution 4 "four benchmarks plus AlpacaEval LC" confusing phrasing | Rewritten as complete enumeration |
| F13 | Limitations section appeared mid-Analysis (before Score distribution, Training dynamics) | Moved to end of Analysis (after Qualitative examples) |
| F14 | Double space on line 357 | Fixed |

---

## 🔴 Critical (likely rejection if unaddressed)

### 1. Circular Evaluation — Qwen3-8B as both training filter and MT-Bench judge
The same Qwen3-8B model used to select training data (Stage 3 scoring) is also used to evaluate MT-Bench scores (`Section 4 Experimental Setup`, lines 206 and 240). Any systematic stylistic preference of Qwen3-8B will be baked into the training data **and** rewarded at evaluation time. This is a textbook circular evaluation setup.

**Fix**: Acknowledge explicitly in Section 4.5 Limitations. Add a cross-validation note: e.g., re-run MT-Bench with a second independent judge (GPT-4o or Claude) and show scores are consistent. Alternatively, re-label the judge-independence issue as a limitation and cite that AlpacaEval and IFEval are fully independent (GPT-4 and rule-based respectively).

### 2. pic.png figure still has wrong values
`pic.png` Panel B still shows the old DMAPO MT-Bench score (7.62), while the paper reports **7.50**. Also contains Chinese "已接受" text, "Train onbinary labels" typo, and is missing SPPO/REINFORCE++ bars.
**Fix**: Regenerate pic.png with correct values.

---

## 🟠 Major (significantly weakens credibility)

### 3. Cohen's Kappa methodology is incorrect
Table 5 computes Cohen's κ between pairs of judges (Help.–Fact., Help.–Conc., Fact.–Conc.). But these three judges evaluate **different dimensions** — κ is a measure of agreement between raters evaluating the **same criterion**. What is actually being computed here is cross-dimension score correlation, not inter-rater agreement. Reviewers with statistics background will flag this immediately.

**Fix**: Rename to "cross-dimension Pearson correlation" or redesign: have two independent judges evaluate the same dimension (e.g., two helpfulness judges) and report κ between them.

### 4. Process Critic section severely underdescribed
Section 3.4 is a single sentence. Missing:
- What model performs the critic role (same Qwen3-8B? a different model?)
- What the critic prompt looks like
- How "flaw severity" is operationalized (continuous score? discrete levels?)
- How the α=0.15 penalty is applied (to which score? before or after aggregation? per flaw or once?)

Without this, Stage 4 is not reproducible.

**Fix**: Expand Process Critic to a proper subsection with the prompt template and a worked example in the Appendix.

### 5. Math category regression unaddressed
In Table 2 (MT-Bench per-category): Base = **6.45**, DMAPO = **5.70** (−0.75 regression). This is the largest single-category drop of any method relative to base. The paper only claims "DMAPO achieves the best reasoning score" without mentioning the Math regression. Reviewers will spot it immediately in Table 2.

**Fix**: Add a sentence in the Table 2 discussion: acknowledge the Math regression and offer a hypothesis (e.g., KTO on concise data may suppress step-by-step elaboration that Math questions require).

### 6. No-Variance-Gate ablation has a confound
"No-Variance-Gate (77.5%) performs worse than Random despite using 2.3× more data (4,297 vs. 1,871)" is presented as validating the variance gate. But the two conditions differ in **both** the filtering strategy and the dataset size — you cannot isolate the effect of variance gating from the effect of dataset size.

**Fix**: Add a row "No-Variance-Gate (1,871 subsampled)" to hold dataset size constant.

### 7. Llama table missing ORPO and REINFORCE++
Table 9 (Llama-3.1-8B results) has only 4 baselines vs Table 1's 7. ORPO and REINFORCE++ are absent. Reviewers will ask whether these were excluded because they performed better than DMAPO on Llama.

**Fix**: Add ORPO and REINFORCE++ rows, or add a caption note explaining the exclusion.

---

## 🟡 Moderate (needs explanation or is a credibility risk)

### 8. AlpacaEval 98.0% raw win-rate is extraordinarily high
A fine-tuned 7B model achieving 98.0% raw win-rate vs text-davinci-003 is higher than most GPT-4-class results. This will draw immediate scrutiny — reviewers will suspect length inflation or judge manipulation.

**Fix**: The AlpacaEval LC (95.5%) is only −2.5 pp from raw (262 avg tokens), which is already discussed. Consider adding a sentence explicitly noting that the raw WR is high relative to literature and pointing to the LC result as the more conservative number.

### 9. Internal win-rate uses log-probability comparison (non-standard)
Win-rate is computed via log-probability comparison against the base model, not by generating actual outputs and using a judge. Log-prob win-rate is biased toward shorter responses and is not directly comparable to standard AlpacaEval-style win-rates.

**Fix**: Note this explicitly in Section 4.4 Evaluation. Ideally add a sentence validating that log-prob WR correlates with judge-based WR on a small sample.

### 10. Training dynamics figure missing ORPO, SPPO, REINFORCE++
The figure caption (now updated to say "representative subset") still shows only 5 of 8 methods, and the table already covers all of them. Consider either adding the missing curves or removing the figure entirely in favor of Table 6.

---

## Summary table

| # | Issue | Severity | Fix effort |
|---|-------|----------|------------|
| 1 | Circular eval (Qwen3-8B judge = training judge) | 🔴 Critical | Low (add disclaimer + cross-check) |
| 2 | pic.png wrong values | 🔴 Critical | Low (regenerate figure) |
| 3 | Cohen's κ methodology wrong | 🟠 Major | Medium (reframe or redesign) |
| 4 | Process Critic underdescribed | 🟠 Major | Low (add detail) |
| 5 | Math category regression unmentioned | 🟠 Major | Low (add one sentence) |
| 6 | No-Variance-Gate confound | 🟠 Major | High (new experiment row) |
| 7 | Llama table missing baselines | 🟠 Major | Medium (add rows or note) |
| 8 | AlpacaEval 98.0% needs more justification | 🟡 Moderate | Low (add sentence) |
| 9 | Internal WR uses log-prob (non-standard) | 🟡 Moderate | Low (add note) |
| 10 | Training dynamics figure incomplete | 🟡 Moderate | Low (add note or remove) |
