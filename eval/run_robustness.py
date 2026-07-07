"""Robustness evaluation of the output-parsing layer.

The accuracy numbers in results.md depend on `parse_prediction` turning real,
messy model output into a clean prediction. A 0.5B model does not always emit a
bare JSON object: it wraps it in prose, fences it in markdown, adds extra keys,
or - worse - emits something that only looks like JSON. This harness stress-tests
that layer with two families of perturbations and reports, without needing the
model or a GPU:

* extraction: messy-but-recoverable outputs should still parse to the right fields
* safe rejection: genuinely malformed outputs should return None (counted wrong),
  never a crash or a hallucinated field

    python eval/run_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merchant_llm.parsing import normalize_name, parse_prediction  # noqa: E402

REPORT = Path(__file__).resolve().parent / "robustness_report.md"

GOOD = '{"merchant_name": "GridPoint Energy", "category": "utilities"}'

# Messy-but-recoverable outputs: the right prediction is in there somewhere.
RECOVERABLE: dict[str, str] = {
    "clean": GOOD,
    "prose_wrapped": f"Sure! Here is the result:\n{GOOD}\nHope this helps.",
    "markdown_fence": f"```json\n{GOOD}\n```",
    "extra_keys": (
        '{"merchant_name": "GridPoint Energy", "category": "utilities", "confidence": 0.91}'
    ),
    "whitespace_noise": '{  "merchant_name" :  "GridPoint Energy" ,  "category" : "utilities"  }',
    "decoy_empty_object": f"Thinking... {{}} then the answer: {GOOD}",
    "category_uppercase": '{"merchant_name": "GridPoint Energy", "category": "UTILITIES"}',
}

# Genuinely malformed: the parser must reject these (return None), not guess.
MALFORMED: dict[str, str] = {
    "single_quotes": "{'merchant_name': 'GridPoint Energy', 'category': 'utilities'}",
    "truncated": '{"merchant_name": "GridPoint Ener',
    "non_string_value": '{"merchant_name": 123, "category": "utilities"}',
    "missing_field": '{"merchant_name": "GridPoint Energy"}',
    "empty": "",
    "prose_only": "I think this is an energy company but I am not sure.",
}

EXPECTED_NAME = "GridPoint Energy"
EXPECTED_CATEGORY = "utilities"


def evaluate() -> dict:
    recoverable_hits = 0
    recoverable_rows = []
    for label, text in RECOVERABLE.items():
        parsed = parse_prediction(text)
        ok = (
            parsed is not None
            and normalize_name(parsed["merchant_name"]) == normalize_name(EXPECTED_NAME)
            and parsed["category"].lower() == EXPECTED_CATEGORY
        )
        recoverable_hits += ok
        recoverable_rows.append((label, ok))

    rejected = 0
    malformed_rows = []
    for label, text in MALFORMED.items():
        parsed = parse_prediction(text)
        safe = parsed is None
        rejected += safe
        malformed_rows.append((label, safe))

    return {
        "extraction_rate": recoverable_hits / len(RECOVERABLE),
        "safe_rejection_rate": rejected / len(MALFORMED),
        "recoverable_rows": recoverable_rows,
        "malformed_rows": malformed_rows,
    }


def render(res: dict) -> str:
    def rows(items, ok_label, bad_label):
        return "\n".join(
            f"| `{label}` | {ok_label if ok else bad_label} |" for label, ok in items
        )

    return (
        "# Output-parsing robustness\n\n"
        f"- Extraction on messy-but-recoverable outputs: "
        f"**{res['extraction_rate']:.0%}** ({len(res['recoverable_rows'])} cases)\n"
        f"- Safe rejection of malformed outputs: "
        f"**{res['safe_rejection_rate']:.0%}** ({len(res['malformed_rows'])} cases)\n\n"
        "## Messy-but-recoverable (should extract the right prediction)\n\n"
        "| perturbation | extracted |\n|---|---|\n"
        + rows(res["recoverable_rows"], "yes", "NO")
        + "\n\n## Malformed (should return None, never guess)\n\n"
        "| perturbation | safely rejected |\n|---|---|\n"
        + rows(res["malformed_rows"], "yes", "NO")
        + "\n\nReproduce with `python eval/run_robustness.py`. The parser tolerates prose, "
        "markdown fences, extra keys and whitespace around a valid object, but treats "
        "single-quoted, truncated or wrong-typed output as invalid rather than guessing.\n"
    )


def main() -> int:
    res = evaluate()
    report = render(res)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    # Fail loudly if the parser regresses.
    return 0 if res["extraction_rate"] == 1.0 and res["safe_rejection_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
