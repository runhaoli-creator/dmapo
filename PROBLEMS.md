# DMAPO — Remaining Issue

Last updated: 2026-03-31

All tex/writing/structural issues resolved. Paper fits COLM 9-page limit. Only remaining item:

---

## 🔴 Teaser figure (DMAPO(1).pdf / pic.png) needs correction

### Panel A — Pipeline diagram

| Problem | Fix |
|---------|-----|
| "Train **onbinary** labels" | → "Train **on binary** labels" (add space) |
| Green "已接受" Chinese text in Confidence Gate box | Remove completely |

### Panel B — Bar chart values are wrong

Current values vs. correct values from Table 1:

| Method | Current (WRONG) | Correct (Table 1) | Status |
|--------|----------------|--------------------|--------|
| SFT (10K) | 7.18 | **6.71** | ❌ way off |
| DPO (10K) | 7.08 | **7.08** | ✅ |
| KTO (20K) | 7.22 | **7.25** | ❌ |
| ORPO (10K) | 7.15 | **7.42** | ❌ way off |
| SimPO (10K) | 7.25 | **7.23** | ❌ |
| SPPO (60K) | — | **7.28** | ❌ missing |
| REINFORCE++ (10K) | — | **7.35** | ❌ missing |
| DMAPO (1.9K) | 7.50 | **7.50** | ✅ |
| Base (dashed line) | 7.41 | **7.41** | ✅ |

### Panel B — Title and annotations are wrong

| Problem | Fix |
|---------|-----|
| Title: "Only DMAPO Improves Over Pretrained Base" | Wrong — ORPO (7.42) also beats Base (7.41). Change to **"MT-Bench Scores (Pretrained Base = 7.41)"** |
| Red annotation: "All baselines degrade below pretrained base" | Wrong — ORPO does not degrade. Remove or change to "Most baselines degrade" |
| Green annotation: "Only DMAPO improves — with 5× less data" | Needs update since ORPO also improves. Change to **"DMAPO improves most — with 5× less data"** |

### Summary of bar chart after fix

```
SFT         6.71  (10K)   red bar, below base
DPO         7.08  (10K)   red bar, below base
KTO         7.25  (20K)   red bar, below base
SPPO        7.28  (60K)   red bar, below base    ← NEW
SimPO       7.23  (10K)   red bar, below base
REINFORCE++ 7.35  (10K)   red bar, below base    ← NEW
ORPO        7.42  (10K)   green/teal bar, ABOVE base  ← now above line!
DMAPO       7.50  (1.9K)  green bar, ABOVE base

Base line at 7.41 (dashed horizontal)
```

Note: With correct values, both ORPO and DMAPO are above the base line. The figure should reflect this (two green bars above the dashed line).
