"""Evaluate the fine-tuned model and build the before/after comparison.

Runs the same harness as baseline.py but with the LoRA adapter loaded, then
writes eval/results.md comparing against the stored baseline numbers.

Usage:
    python -m merchant_llm.evaluate --adapter models/adapter
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from merchant_llm.baseline import print_metrics, run_eval
from merchant_llm.config import DEFAULT_MODEL

METRIC_LABELS = {
    "json_validity": "JSON validity",
    "category_accuracy": "Category accuracy",
    "name_exact_match": "Merchant name exact match",
    "both_correct": "Both fields correct",
}


def build_results_md(baseline: dict, tuned: dict) -> str:
    base_m, tuned_m = baseline["metrics"], tuned["metrics"]
    lines = [
        "# Before/after evaluation",
        "",
        f"Base model: `{baseline['model']}` | adapter: `{tuned['adapter']}`",
        "",
        f"Test set: {tuned_m['n']} held-out synthetic examples ({tuned['data']}), "
        f"greedy decoding, max_new_tokens={tuned['decode']['max_new_tokens']}, "
        f"device: {tuned['device']}. Evaluated on {date.today().isoformat()}.",
        "",
        "| Metric | Base model (zero-shot) | LoRA fine-tuned | Delta |",
        "|---|---|---|---|",
    ]
    for key, label in METRIC_LABELS.items():
        b, t = base_m[key] * 100, tuned_m[key] * 100
        lines.append(f"| {label} | {b:.1f}% | {t:.1f}% | {t - b:+.1f} pp |")
    lines += [
        "",
        "Malformed or missing JSON counts as wrong on every metric, for both models.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default="models/adapter")
    parser.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--baseline", type=Path, default=Path("eval/baseline.json"))
    parser.add_argument("--out-json", type=Path, default=Path("eval/finetuned.json"))
    parser.add_argument("--out-md", type=Path, default=Path("eval/results.md"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        raise SystemExit(
            f"{args.baseline} not found; run `python -m merchant_llm.baseline` first "
            "so the comparison uses real baseline numbers"
        )
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    tuned = run_eval(
        args.model,
        args.data,
        adapter_path=args.adapter,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        device=args.device,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(tuned, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(build_results_md(baseline, tuned), encoding="utf-8")

    print_metrics(f"zero-shot baseline: {baseline['model']}", baseline["metrics"])
    print_metrics(f"fine-tuned: {args.model} + {args.adapter}", tuned["metrics"])
    print(f"\nwrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
