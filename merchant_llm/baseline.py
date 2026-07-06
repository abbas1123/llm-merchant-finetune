"""Zero-shot evaluation of the base model on the test split.

Writes eval/baseline.json so evaluate.py can build the before/after table.

Usage:
    python -m merchant_llm.baseline --data data/test.jsonl --out eval/baseline.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from merchant_llm.data import read_jsonl
from merchant_llm.inference import DEFAULT_MODEL, generate_all, load_model
from merchant_llm.parsing import score_outputs


def run_eval(
    model_name: str,
    data_path: Path,
    adapter_path: str | None = None,
    batch_size: int = 32,
    max_new_tokens: int = 64,
    limit: int | None = None,
    device: str = "auto",
) -> dict:
    """Shared harness used by both baseline.py and evaluate.py."""
    examples = read_jsonl(data_path)
    if limit:
        examples = examples[:limit]
    print(f"loading {model_name}" + (f" + adapter {adapter_path}" if adapter_path else ""))
    model, tokenizer, device = load_model(model_name, adapter_path, device)
    print(f"evaluating {len(examples)} examples on {device}")
    start = time.perf_counter()
    outputs = generate_all(
        model, tokenizer, [ex["raw"] for ex in examples], device, batch_size, max_new_tokens
    )
    elapsed = time.perf_counter() - start
    metrics, records = score_outputs(examples, outputs)
    return {
        "model": model_name,
        "adapter": adapter_path,
        "data": str(data_path),
        "device": device,
        "decode": {"strategy": "greedy", "max_new_tokens": max_new_tokens},
        "eval_seconds": round(elapsed, 1),
        "metrics": metrics,
        "sample_predictions": records[:10],
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title} (n={metrics['n']})")
    for key in ("json_validity", "category_accuracy", "name_exact_match", "both_correct"):
        print(f"  {key:<22} {metrics[key] * 100:6.1f}%")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/baseline.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    result = run_eval(
        args.model,
        args.data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_metrics(f"zero-shot baseline: {args.model}", result["metrics"])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
