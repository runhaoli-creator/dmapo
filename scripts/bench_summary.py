"""
scripts/bench_summary.py
────────────────────────
Parse judged outputs and print a paper-ready comparison table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


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
    valid = [r for r in records if r.get("avg_score", -1) > 0]
    if not valid:
        return {"avg": None, "categories": {}}
    avg = statistics.mean(r["avg_score"] for r in valid)
    cats: dict[str, list[float]] = {}
    for r in valid:
        cats.setdefault(r["category"], []).append(r["avg_score"])
    cat_avgs = {c: round(statistics.mean(s), 2) for c, s in sorted(cats.items())}
    return {"avg": round(avg, 2), "categories": cat_avgs}


def alpaca_eval_score(records: list[dict]) -> dict:
    if not records:
        return {"win_rate": None, "wins": 0, "ties": 0, "losses": 0, "total": 0}
    wins = sum(1 for r in records if r.get("winner") == "A")
    ties = sum(1 for r in records if r.get("winner") in ("TIE", "tie"))
    losses = sum(1 for r in records if r.get("winner") == "B")
    total = len(records)
    wr = round(wins / total * 100, 1) if total else 0
    return {"win_rate": wr, "wins": wins, "ties": ties, "losses": losses, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", default="outputs/bench")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)

    models = [
        ("Base (Mistral-7B-Instruct-v0.2)", "base"),
        ("+ DPO baseline (UF-binarized)", "dpo"),
        ("+ DMAPO (ours)", "dmapo"),
    ]

    # ── MT-Bench ─────────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                  MT-Bench (Qwen3-8B judge)                  ║")
    print("╠════════════════════════════════════════╦═════════════════════╣")
    print("║ Model                                  ║ Score (1-10)       ║")
    print("╠════════════════════════════════════════╬═════════════════════╣")

    mt_results = {}
    for name, tag in models:
        records = load_jsonl(bench_dir / f"{tag}_mt_bench_judged.jsonl")
        result = mt_bench_score(records)
        mt_results[tag] = result
        score_str = f"{result['avg']:.2f}" if result["avg"] is not None else "N/A"
        print(f"║ {name:<38} ║ {score_str:>19} ║")

    print("╚════════════════════════════════════════╩═════════════════════╝")

    # Per-category breakdown for MT-Bench
    all_cats = set()
    for r in mt_results.values():
        all_cats.update(r.get("categories", {}).keys())
    if all_cats:
        print("\nMT-Bench per-category breakdown:")
        header = f"{'Category':<14}" + "".join(f" {n[:10]:>10}" for n, _ in models)
        print(header)
        print("-" * len(header))
        for cat in sorted(all_cats):
            row = f"{cat:<14}"
            for _, tag in models:
                s = mt_results.get(tag, {}).get("categories", {}).get(cat)
                row += f" {s:>10.2f}" if s is not None else f" {'N/A':>10}"
            print(row)

    # ── AlpacaEval ───────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║             AlpacaEval 2.0 (Qwen3-8B judge)                 ║")
    print("╠════════════════════════════════════════╦═════════════════════╣")
    print("║ Model                                  ║ Win-Rate vs Ref    ║")
    print("╠════════════════════════════════════════╬═════════════════════╣")

    for name, tag in models:
        records = load_jsonl(bench_dir / f"{tag}_alpaca_eval_judged.jsonl")
        result = alpaca_eval_score(records)
        wr_str = f"{result['win_rate']:.1f}%" if result["win_rate"] is not None else "N/A"
        detail = f"({result['wins']}W/{result['ties']}T/{result['losses']}L)" if result["total"] else ""
        print(f"║ {name:<38} ║ {wr_str:>7} {detail:>11} ║")

    print("╚════════════════════════════════════════╩═════════════════════╝")

    # ── Internal Win-Rate ────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              Internal Eval (gated val set)                   ║")
    print("╠════════════════════════════════════════╦═════════════════════╣")
    print("║ Model                                  ║ Win-Rate vs Base   ║")
    print("╠════════════════════════════════════════╬═════════════════════╣")

    internal = Path("outputs/eval/metrics.json")
    if internal.exists():
        with open(internal) as f:
            m = json.load(f)
        gen = m.get("generation", {})
        wr = gen.get("win_rate", 0) * 100
        print(f"║ {'Base (Mistral-7B-Instruct-v0.2)':<38} ║ {'—':>19} ║")
        print(f"║ {'+ DMAPO (ours)':<38} ║ {wr:>18.1f}% ║")

    print("╚════════════════════════════════════════╩═════════════════════╝")

    # ── Save JSON ────────────────────────────────────────────────────────────
    summary = {"mt_bench": mt_results, "alpaca_eval": {}, "internal": {}}
    out_path = bench_dir / "benchmark_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary → {out_path}")


if __name__ == "__main__":
    main()
