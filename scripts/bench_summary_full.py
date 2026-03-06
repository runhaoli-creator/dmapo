"""
scripts/bench_summary_full.py
──────────────────────────────
Parse ALL judged outputs (7 models × 3 benchmarks) and produce paper-ready
tables in .tex and .txt format.

Usage:
    python scripts/bench_summary_full.py --bench-dir outputs/bench
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MODELS = [
    ("Base (Mistral-7B-v0.2)", "base"),
    ("+ SFT", "sft"),
    ("+ DPO", "dpo"),
    ("+ KTO", "kto"),
    ("+ ORPO", "orpo"),
    ("+ SimPO", "simpo"),
    ("+ DMAPO (ours)", "dmapo"),
]

MT_CATEGORIES = ["coding", "extraction", "humanities", "math",
                  "reasoning", "roleplay", "stem", "writing"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def mt_bench_score(records: list[dict]) -> dict:
    if not records:
        return {"avg": None, "categories": {}}
    valid = [r for r in records if (r.get("avg_score") or 0) > 0]
    if not valid:
        return {"avg": None, "categories": {}}
    avg = statistics.mean(r["avg_score"] for r in valid)
    cats: dict[str, list[float]] = {}
    for r in valid:
        cats.setdefault(r.get("category", "unk"), []).append(r["avg_score"])
    cat_avgs = {c: round(statistics.mean(s), 2) for c, s in sorted(cats.items())}
    return {"avg": round(avg, 2), "categories": cat_avgs, "n_valid": len(valid)}


def alpaca_eval_score(records: list[dict]) -> dict:
    if not records:
        return {"win_rate": None}
    wins = sum(1 for r in records if r.get("winner") == "A")
    ties = sum(1 for r in records if r.get("winner") in ("TIE", "tie"))
    losses = sum(1 for r in records if r.get("winner") == "B")
    total = len(records)
    wr = round(wins / total * 100, 1) if total else None
    return {"win_rate": wr, "wins": wins, "ties": ties, "losses": losses, "total": total}


def ifeval_score(bench_dir: Path, tag: str) -> dict:
    # Try summary file first
    summary_path = bench_dir / f"{tag}_ifeval.summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        return {
            "prompt_acc": data.get("prompt_accuracy"),
            "inst_acc": data.get("instruction_accuracy"),
            "prompt_pass": data.get("prompt_pass", 0),
            "prompt_total": data.get("prompt_total", 0),
        }
    # Fallback: parse JSONL
    records = load_jsonl(bench_dir / f"{tag}_ifeval.jsonl")
    if not records:
        return {"prompt_acc": None, "inst_acc": None}
    prompt_pass = sum(1 for r in records if r.get("eval", {}).get("all_passed"))
    inst_pass = 0
    inst_total = 0
    for r in records:
        for ir in r.get("eval", {}).get("instruction_results", []):
            inst_total += 1
            if ir.get("passed"):
                inst_pass += 1
    n = len(records)
    return {
        "prompt_acc": round(prompt_pass / n * 100, 1) if n else None,
        "inst_acc": round(inst_pass / inst_total * 100, 1) if inst_total else None,
        "prompt_pass": prompt_pass,
        "prompt_total": n,
    }


def fmt(val, suffix="", default="\\xxxx"):
    if val is None:
        return default
    return f"{val}{suffix}"


def fmt_txt(val, suffix="", default="xxxx"):
    if val is None:
        return default
    return f"{val}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", default="outputs/bench")
    args = parser.parse_args()
    bench_dir = Path(args.bench_dir)
    paper_dir = Path("outputs/paper")
    paper_dir.mkdir(parents=True, exist_ok=True)

    # ── Gather all results ───────────────────────────────────────────────────
    results = {}
    for name, tag in MODELS:
        mt = mt_bench_score(load_jsonl(bench_dir / f"{tag}_mt_bench_judged.jsonl"))
        ae = alpaca_eval_score(load_jsonl(bench_dir / f"{tag}_alpaca_eval_judged.jsonl"))
        ie = ifeval_score(bench_dir, tag)
        results[tag] = {"name": name, "mt": mt, "ae": ae, "ie": ie}

    # Internal eval
    internal = {}
    metrics_path = Path("outputs/eval/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        gen = m.get("generation", {})
        internal = {
            "win_rate": round(gen.get("win_rate", 0) * 100, 1),
            "ppl_policy": gen.get("perplexity_policy"),
            "ppl_base": gen.get("perplexity_base"),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Generate LaTeX tables
    # ══════════════════════════════════════════════════════════════════════════
    tex_lines = []

    # --- Table 1: Main results ---
    tex_lines.append(r"""% Table 1: Main Results
\begin{table*}[t]
\centering
\caption{Main results. All methods use Mistral-7B-Instruct-v0.2 with LoRA ($r{=}16$).
Baselines train on UltraFeedback-binarized (10k). DMAPO trains on quality-gated data (1,871).
Judge: Qwen3-8B. Best in \textbf{bold}, second \underline{underlined}.}
\label{tab:main_results}
\vspace{0.5em}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c c c}
\toprule
\textbf{Method} & \textbf{Train $|\mathcal{D}|$} & \textbf{MT-Bench} $\uparrow$ & \textbf{AlpacaEval WR\%} $\uparrow$ & \textbf{IFEval Acc\%} $\uparrow$ & \textbf{Internal WR\%} $\uparrow$ \\
\midrule""")

    train_sizes = {"base": "---", "sft": "10k", "dpo": "10k", "kto": "20k",
                   "orpo": "10k", "simpo": "10k", "dmapo": "1,871"}

    for name, tag in MODELS:
        r = results[tag]
        mt_s = fmt(r["mt"]["avg"])
        ae_s = fmt(r["ae"]["win_rate"])
        ie_s = fmt(r["ie"]["prompt_acc"])
        if tag == "dmapo":
            iw = fmt(internal.get("win_rate"))
        elif tag == "base":
            iw = "---"
        else:
            iw = "\\xxxx"
        ts = train_sizes.get(tag, "10k")

        if tag == "dmapo":
            tex_lines.append(f"\\midrule")
            tex_lines.append(f"\\textbf{{{name}}} & \\textbf{{{ts}}} & {mt_s} & {ae_s} & {ie_s} & {iw} \\\\")
        elif tag == "base":
            tex_lines.append(f"{name} & {ts} & {mt_s} & {ae_s} & {ie_s} & {iw} \\\\")
            tex_lines.append(f"\\midrule")
        else:
            tex_lines.append(f"{name} & {ts} & {mt_s} & {ae_s} & {ie_s} & {iw} \\\\")

    tex_lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}
""")

    # --- Table 2: MT-Bench categories ---
    tex_lines.append(r"""% Table 2: MT-Bench Per-Category
\begin{table*}[t]
\centering
\caption{MT-Bench per-category scores (Qwen3-8B judge, 10 questions/category, scale 1--10).}
\label{tab:mt_bench_categories}
\vspace{0.5em}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l cccccccc c}
\toprule
\textbf{Method} & \textbf{Coding} & \textbf{Extract.} & \textbf{Human.} & \textbf{Math} & \textbf{Reason.} & \textbf{Roleplay} & \textbf{STEM} & \textbf{Writing} & \textbf{Avg.} \\
\midrule""")

    for name, tag in MODELS:
        cats = results[tag]["mt"]["categories"]
        avg = results[tag]["mt"]["avg"]
        cat_vals = [fmt(cats.get(c), default="\\xxxx") for c in MT_CATEGORIES]
        avg_s = fmt(avg)
        row = " & ".join(cat_vals) + f" & {avg_s}"
        prefix = f"\\textbf{{{name}}}" if tag == "dmapo" else name
        sep = "\\midrule\n" if tag == "dmapo" else ""
        tex_lines.append(f"{sep}{prefix} & {row} \\\\")

    tex_lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}
""")

    # --- Table 3: AlpacaEval detailed ---
    tex_lines.append(r"""% Table 3: AlpacaEval Detailed
\begin{table}[t]
\centering
\caption{AlpacaEval~2.0 detailed results (Qwen3-8B judge, 805 instructions, vs text-davinci-003).}
\label{tab:alpaca_eval}
\vspace{0.5em}
\begin{tabular}{l cccc}
\toprule
\textbf{Method} & \textbf{WR (\%)} $\uparrow$ & \textbf{Wins} & \textbf{Ties} & \textbf{Losses} \\
\midrule""")

    for name, tag in MODELS:
        ae = results[tag]["ae"]
        wr = fmt(ae.get("win_rate"))
        w = fmt(ae.get("wins"), default="\\xxxx")
        t = fmt(ae.get("ties"), default="\\xxxx")
        l = fmt(ae.get("losses"), default="\\xxxx")
        prefix = f"\\textbf{{{name}}}" if tag == "dmapo" else name
        sep = "\\midrule\n" if tag == "dmapo" else ""
        tex_lines.append(f"{sep}{prefix} & {wr} & {w} & {t} & {l} \\\\")

    tex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # --- Table 4: IFEval ---
    tex_lines.append(r"""% Table 4: IFEval
\begin{table}[t]
\centering
\caption{IFEval results (rule-based evaluation, prompt-level and instruction-level accuracy).}
\label{tab:ifeval}
\vspace{0.5em}
\begin{tabular}{l cc}
\toprule
\textbf{Method} & \textbf{Prompt Acc (\%)} $\uparrow$ & \textbf{Instruction Acc (\%)} $\uparrow$ \\
\midrule""")

    for name, tag in MODELS:
        ie = results[tag]["ie"]
        pa = fmt(ie.get("prompt_acc"))
        ia = fmt(ie.get("inst_acc"))
        prefix = f"\\textbf{{{name}}}" if tag == "dmapo" else name
        sep = "\\midrule\n" if tag == "dmapo" else ""
        tex_lines.append(f"{sep}{prefix} & {pa} & {ia} \\\\")

    tex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # --- Table 5: Pipeline Stats (static) ---
    tex_lines.append(r"""% Table 5: Pipeline Statistics
\begin{table}[t]
\centering
\caption{DMAPO pipeline statistics.}
\label{tab:pipeline_stats}
\vspace{0.5em}
\begin{tabular}{l r}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Source prompts (UltraFeedback + HelpSteer2)  & 14,272 \\
Candidates generated ($k{=}4$ per prompt)    & 54,236 \\
Multi-agent judges                            & 3 \\
\quad Helpfulness (mean $\pm$ std)            & $5.50 \pm 0.91$ \\
\quad Factuality (mean $\pm$ std)             & $5.49 \pm 0.99$ \\
\quad Conciseness (mean $\pm$ std)            & $5.52 \pm 1.00$ \\
Quality-gated (train / val)                   & 1,871 / 129 \\
\quad Desirable / Undesirable                 & 951+63 / 920+66 \\
Acceptance rate                               & 3.45\% \\
Desirable avg score                           & $9.23 \pm 1.09$ \\
Undesirable avg score                         & $2.42 \pm 1.10$ \\
\bottomrule
\end{tabular}
\end{table}
""")

    # --- Table 6: Ablation ---
    tex_lines.append(r"""% Table 6: Ablation Study
\begin{table}[t]
\centering
\caption{Ablation: effect of multi-agent quality gating.}
\label{tab:ablation_gating}
\vspace{0.5em}
\begin{tabular}{l r cc}
\toprule
\textbf{Setting} & $|\mathcal{D}|$ & \textbf{MT-Bench} $\uparrow$ & \textbf{Internal WR\%} $\uparrow$ \\
\midrule
KTO (no gating)       & \xxxx & \xxxx & \xxxx \\
KTO (1-judge gating)  & \xxxx & \xxxx & \xxxx \\
KTO (2-judge gating)  & \xxxx & \xxxx & \xxxx \\
\textbf{KTO (3-judge = DMAPO)} & \textbf{1,871} & \textbf{7.30} & \textbf{85.3} \\
\bottomrule
\end{tabular}
\end{table}
""")

    tex_content = "\n".join(tex_lines)
    tex_path = paper_dir / "tables.tex"
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"Saved LaTeX tables → {tex_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Generate TXT tables
    # ══════════════════════════════════════════════════════════════════════════
    txt_lines = []
    txt_lines.append("=" * 80)
    txt_lines.append("  DMAPO — Paper-Ready Results (Full)")
    txt_lines.append("=" * 80)
    txt_lines.append("")

    # Main results
    txt_lines.append("TABLE 1: MAIN RESULTS")
    txt_lines.append("-" * 80)
    txt_lines.append(f"{'Method':<30} {'|D|':>6} {'MT-B':>6} {'AE%':>6} {'IFE%':>6} {'Int%':>6}")
    txt_lines.append("-" * 80)
    for name, tag in MODELS:
        r = results[tag]
        ts = train_sizes.get(tag, "10k")
        mt_s = fmt_txt(r["mt"]["avg"])
        ae_s = fmt_txt(r["ae"]["win_rate"])
        ie_s = fmt_txt(r["ie"]["prompt_acc"])
        if tag == "dmapo":
            iw = fmt_txt(internal.get("win_rate"))
        elif tag == "base":
            iw = "---"
        else:
            iw = "xxxx"
        txt_lines.append(f"{name:<30} {ts:>6} {mt_s:>6} {ae_s:>6} {ie_s:>6} {iw:>6}")
    txt_lines.append("")

    # MT-Bench categories
    txt_lines.append("TABLE 2: MT-BENCH PER-CATEGORY")
    txt_lines.append("-" * 100)
    hdr = f"{'Method':<20}" + "".join(f" {c[:7]:>7}" for c in MT_CATEGORIES) + f" {'Avg':>7}"
    txt_lines.append(hdr)
    txt_lines.append("-" * 100)
    for name, tag in MODELS:
        cats = results[tag]["mt"]["categories"]
        avg = results[tag]["mt"]["avg"]
        vals = [fmt_txt(cats.get(c)) for c in MT_CATEGORIES]
        row = f"{name:<20}" + "".join(f" {v:>7}" for v in vals) + f" {fmt_txt(avg):>7}"
        txt_lines.append(row)
    txt_lines.append("")

    # AlpacaEval
    txt_lines.append("TABLE 3: ALPACAEVAL 2.0")
    txt_lines.append("-" * 60)
    txt_lines.append(f"{'Method':<30} {'WR%':>6} {'W':>5} {'T':>4} {'L':>5}")
    txt_lines.append("-" * 60)
    for name, tag in MODELS:
        ae = results[tag]["ae"]
        wr = fmt_txt(ae.get("win_rate"))
        w = fmt_txt(ae.get("wins"))
        t = fmt_txt(ae.get("ties"))
        l = fmt_txt(ae.get("losses"))
        txt_lines.append(f"{name:<30} {wr:>6} {w:>5} {t:>4} {l:>5}")
    txt_lines.append("")

    # IFEval
    txt_lines.append("TABLE 4: IFEVAL")
    txt_lines.append("-" * 60)
    txt_lines.append(f"{'Method':<30} {'Prompt%':>8} {'Instr%':>8}")
    txt_lines.append("-" * 60)
    for name, tag in MODELS:
        ie = results[tag]["ie"]
        pa = fmt_txt(ie.get("prompt_acc"))
        ia = fmt_txt(ie.get("inst_acc"))
        txt_lines.append(f"{name:<30} {pa:>8} {ia:>8}")
    txt_lines.append("")

    txt_content = "\n".join(txt_lines)
    txt_path = paper_dir / "tables.txt"
    with open(txt_path, "w") as f:
        f.write(txt_content)
    print(f"Saved TXT tables → {txt_path}")

    # ── Save JSON summary ────────────────────────────────────────────────────
    json_path = paper_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON summary → {json_path}")

    # ── Print to console ─────────────────────────────────────────────────────
    print("\n" + txt_content)


if __name__ == "__main__":
    main()
