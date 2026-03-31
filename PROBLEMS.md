# DMAPO — Open Problems

Last updated: 2026-03-31

All tex/writing/technical issues have been resolved. The only remaining item is the teaser figure.

---

## 🔴 pic.png needs to be regenerated

The current `pic.png` is outdated and inconsistent with the paper. It must be regenerated before submission.

### Panel A (Pipeline diagram)

| Element | Current (wrong) | Required |
|---------|-----------------|----------|
| KTO box text | "Train onbinary labels" | "Train on binary labels" (missing space) |
| Background | Chinese "已接受" watermark visible | Remove all watermarks |

### Panel B (Bar chart)

| Element | Current (wrong) | Required |
|---------|-----------------|----------|
| DMAPO bar value | **7.62** | **7.50** |
| ORPO bar value | **7.15** | **7.42** |
| SFT bar value | **7.18** | **6.71** |
| DPO bar value | **7.08** | **7.08** (OK) |
| KTO bar value | **7.22** | **7.25** |
| SimPO bar value | **7.25** | **7.23** |
| Missing bars | — | Add **SPPO (7.28)** and **REINFORCE++ (7.35)** |
| Panel title | "Only DMAPO Improves Over Pretrained Base" | Incorrect — ORPO (7.42) also beats Base (7.41). Change to e.g. "MT-Bench Scores (Pretrained Base = 7.41)" |
| Base line label | "Pretrained Base (7.41)" | OK, keep |

### Correct bar values (from Table 1)

```
SFT        6.71  (10k)
DPO        7.08  (10k)
KTO        7.25  (20k)
ORPO       7.42  (10k)
SimPO      7.23  (10k)
SPPO       7.28  (60k)   ← NEW
REINFORCE++7.35  (10k)   ← NEW
DMAPO      7.50  (1.9k)  ← HIGHEST
Base line: 7.41 (dashed horizontal)
```

### Style notes
- Color DMAPO bar green (distinct from baselines)
- Keep dashed horizontal line at Base = 7.41
- Bars below base line in red/orange, bars above in green
- Include dataset size labels below each method name
- No Chinese text anywhere
- Export at high resolution (≥300 DPI)
