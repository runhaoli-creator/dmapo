"""
scripts/run_eval.py
────────────────────
Stage 6 – Evaluate pipeline outputs.

Produces:
  outputs/eval/metrics.json      – all numeric metrics
  outputs/eval/summary.csv       – per-split tabular summary
  outputs/eval/report.md         – human-readable markdown report

Usage:
    python scripts/run_eval.py --eval-config configs/eval.yaml \
                                --training-config configs/training.yaml \
                                --arbitration-config configs/arbitration.yaml \
                                --judges-config configs/judges.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import orjson
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmapo.eval.metrics import (
    label_distribution,
    abstention_rate,
    judge_variance_stats,
    response_length_stats,
    final_score_stats,
    per_judge_stats,
    compute_perplexity,
    compute_win_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def run_statistical_metrics(
    scored_train: list[dict],
    gated_train: list[dict],
    scored_val: list[dict],
    gated_val: list[dict],
) -> dict:
    results = {}
    for split, scored, gated in [
        ("train", scored_train, gated_train),
        ("val", scored_val, gated_val),
    ]:
        if not scored and not gated:
            continue
        results[split] = {
            "label_distribution": label_distribution(gated) if gated else {},
            "abstention_rate": abstention_rate(scored, gated) if scored else {},
            "judge_variance_stats": judge_variance_stats(scored) if scored else {},
            "response_length_stats": response_length_stats(gated) if gated else {},
            "final_score_stats": final_score_stats(gated) if gated else {},
            "per_judge_stats": per_judge_stats(scored) if scored else {},
        }
    return results


def run_generation_metrics(
    eval_cfg: dict,
    train_cfg: dict,
    gated_val: list[dict],
) -> dict:
    policy_dir = eval_cfg["input"]["policy_dir"]
    base_model_name = eval_cfg["input"]["base_model"]
    batch_size = eval_cfg.get("batch_size", 4)
    max_new_tokens = eval_cfg.get("max_new_tokens", 256)
    max_prompts = eval_cfg.get("max_eval_prompts", 200)

    if not Path(policy_dir).exists():
        log.warning("Policy dir not found: %s — skipping generation metrics", policy_dir)
        return {}
    if not gated_val:
        log.warning("No gated val records — skipping generation metrics")
        return {}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(train_cfg["model"].get("torch_dtype", "bfloat16"), torch.bfloat16)

    log.info("Loading tokenizer from %s …", policy_dir)
    tokenizer = AutoTokenizer.from_pretrained(policy_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model %s …", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, device_map="auto"
    )
    base_model.eval()

    log.info("Loading LoRA adapter from %s and merging …", policy_dir)
    policy_model = PeftModel.from_pretrained(base_model, policy_dir)
    policy_model = policy_model.merge_and_unload()
    policy_model.eval()

    # Reload base model (the merged call modified it in-place for some PEFT versions)
    log.info("Reloading base model for comparison …")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, device_map="auto"
    )
    base_model.eval()

    eval_records = gated_val[:max_prompts]
    prompts = [r["prompt"] for r in eval_records]

    # Perplexity on chosen (desirable) completions
    chosen_texts = [
        r["prompt"] + "\n" + r["response"]
        for r in eval_records
        if r.get("gate_label") == "desirable"
    ]

    log.info("Computing perplexity on %d chosen completions …", len(chosen_texts))
    ppl_results = {}
    if chosen_texts:
        ppl_policy = compute_perplexity(policy_model, tokenizer, chosen_texts, batch_size=batch_size)
        ppl_base = compute_perplexity(base_model, tokenizer, chosen_texts, batch_size=batch_size)
        ppl_results = {
            "perplexity_policy": round(ppl_policy, 4),
            "perplexity_base": round(ppl_base, 4),
            "perplexity_improvement": round(ppl_base - ppl_policy, 4),
        }
        log.info("PPL — policy: %.2f  base: %.2f", ppl_policy, ppl_base)

    log.info("Computing win-rate on %d prompts …", len(prompts))
    win_results = compute_win_rate(
        policy_model, base_model, tokenizer, prompts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    log.info("Win-rate: %.2f%%", win_results["win_rate"] * 100)

    return {**ppl_results, **win_results}


def write_csv(path: Path, results: dict) -> None:
    rows = []
    for split, metrics in results.items():
        if split in ("generation",):
            continue
        ld = metrics.get("label_distribution", {})
        ar = metrics.get("abstention_rate", {})
        vs = metrics.get("judge_variance_stats", {})
        fs = metrics.get("final_score_stats", {}).get("all", {})
        rows.append({
            "split": split,
            "total_scored": ar.get("total_scored", ""),
            "kept": ar.get("kept", ""),
            "filtered_out": ar.get("filtered_out", ""),
            "keep_rate": ar.get("keep_rate", ""),
            "n_desirable": ld.get("counts", {}).get("desirable", ""),
            "n_undesirable": ld.get("counts", {}).get("undesirable", ""),
            "mean_final_score": fs.get("mean", ""),
            "mean_judge_variance": vs.get("mean_variance", ""),
        })

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV summary → %s", path)


def write_markdown(path: Path, results: dict, gen_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# DMAPO Evaluation Report",
        f"\n_Generated: {ts}_\n",
        "## Dataset Statistics\n",
    ]

    for split in ("train", "val"):
        if split not in results:
            continue
        m = results[split]
        ld = m.get("label_distribution", {})
        ar = m.get("abstention_rate", {})
        vs = m.get("judge_variance_stats", {})
        fs_all = m.get("final_score_stats", {}).get("all", {})
        fs_des = m.get("final_score_stats", {}).get("desirable", {})
        fs_und = m.get("final_score_stats", {}).get("undesirable", {})
        rl = m.get("response_length_stats", {})

        lines += [
            f"### {split.capitalize()} Split\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total scored | {ar.get('total_scored', 'N/A')} |",
            f"| Kept (des + undes) | {ar.get('kept', 'N/A')} |",
            f"| Keep rate | {ar.get('keep_rate', 'N/A')} |",
            f"| Filtered out | {ar.get('filtered_out', 'N/A')} |",
            f"| Desirable | {ld.get('counts', {}).get('desirable', 'N/A')} ({ld.get('fractions', {}).get('desirable', '')}) |",
            f"| Undesirable | {ld.get('counts', {}).get('undesirable', 'N/A')} ({ld.get('fractions', {}).get('undesirable', '')}) |",
            f"| Mean final score | {fs_all.get('mean', 'N/A')} ± {fs_all.get('std', '')} |",
            f"| Desirable score | {fs_des.get('mean', 'N/A')} ± {fs_des.get('std', '')} |",
            f"| Undesirable score | {fs_und.get('mean', 'N/A')} ± {fs_und.get('std', '')} |",
            f"| Mean judge variance | {vs.get('mean_variance', 'N/A')} |",
            f"| Avg response length (all) | {rl.get('all', {}).get('mean', 'N/A')} words |",
            "",
        ]

        # Per-judge stats table
        pj = m.get("per_judge_stats", {})
        if pj:
            lines += ["**Per-Judge Score Statistics**\n", "| Judge | Mean | Std | N |", "|-------|------|-----|---|"]
            for jname, jstats in pj.items():
                lines.append(f"| {jname} | {jstats.get('mean')} | {jstats.get('std')} | {jstats.get('n')} |")
            lines.append("")

    if gen_metrics:
        lines += [
            "## Generation-Based Metrics\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Policy perplexity | {gen_metrics.get('perplexity_policy', 'N/A')} |",
            f"| Base perplexity | {gen_metrics.get('perplexity_base', 'N/A')} |",
            f"| PPL improvement (↓) | {gen_metrics.get('perplexity_improvement', 'N/A')} |",
            f"| Win-rate vs base | {gen_metrics.get('win_rate', 'N/A')} ({gen_metrics.get('wins', '')}/{gen_metrics.get('n_prompts', '')}) |",
            "",
        ]

    with path.open("w") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Markdown report → %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 – Evaluation")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--arbitration-config", default="configs/arbitration.yaml")
    parser.add_argument("--judges-config", default="configs/judges.yaml")
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip perplexity/win-rate (no model loading needed)",
    )
    args = parser.parse_args()

    with open(args.eval_config) as fh:
        eval_cfg = yaml.safe_load(fh)
    with open(args.training_config) as fh:
        train_cfg = yaml.safe_load(fh)
    with open(args.arbitration_config) as fh:
        arb_cfg = yaml.safe_load(fh)
    with open(args.judges_config) as fh:
        judges_cfg = yaml.safe_load(fh)

    out_dir = Path(eval_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    scored_train = load_jsonl(Path(judges_cfg["output"]["train"]))
    scored_val = load_jsonl(Path(judges_cfg["output"]["val"]))
    gated_train = load_jsonl(Path(arb_cfg["output"]["train"]))
    gated_val = load_jsonl(Path(arb_cfg["output"]["val"]))

    log.info(
        "Loaded: scored_train=%d  scored_val=%d  gated_train=%d  gated_val=%d",
        len(scored_train), len(scored_val), len(gated_train), len(gated_val),
    )

    # Statistical metrics
    stat_results = run_statistical_metrics(scored_train, gated_train, scored_val, gated_val)

    # Generation-based metrics
    gen_metrics = {}
    if not args.skip_generation:
        gen_metrics = run_generation_metrics(eval_cfg, train_cfg, gated_val)

    # Combine and save
    all_results = {**stat_results}
    if gen_metrics:
        all_results["generation"] = gen_metrics

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(all_results, fh, indent=2)
    log.info("Metrics JSON → %s", metrics_path)

    write_csv(out_dir / "summary.csv", stat_results)
    write_markdown(out_dir / "report.md", stat_results, gen_metrics)

    log.info("Evaluation complete. Outputs in %s", out_dir)

    # Print quick summary
    for split in ("train", "val"):
        if split in stat_results:
            ar = stat_results[split].get("abstention_rate", {})
            ld = stat_results[split].get("label_distribution", {})
            print(f"\n── {split.upper()} ─────────────────────────────────────")
            print(f"  Kept: {ar.get('kept', 'N/A')} / {ar.get('total_scored', 'N/A')}  "
                  f"(keep rate: {ar.get('keep_rate', 'N/A')})")
            print(f"  Desirable: {ld.get('counts', {}).get('desirable', 'N/A')}  "
                  f"Undesirable: {ld.get('counts', {}).get('undesirable', 'N/A')}")
    if gen_metrics:
        print(f"\n── GENERATION METRICS ──────────────────────────────")
        print(f"  Win-rate: {gen_metrics.get('win_rate', 'N/A')}")
        print(f"  PPL (policy / base): {gen_metrics.get('perplexity_policy', 'N/A')} / "
              f"{gen_metrics.get('perplexity_base', 'N/A')}")


if __name__ == "__main__":
    main()
