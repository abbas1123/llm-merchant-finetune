"""Parsing and scoring of model outputs.

Single-pass, no retries: if the model output does not contain a JSON object
with string values for ``merchant_name`` and ``category``, the prediction
counts as invalid (and therefore wrong on both fields).

Kept free of torch/transformers imports so it can be unit-tested in CI.
"""

from __future__ import annotations

import json
import re
from typing import Any

from merchant_llm.categories import CATEGORY_SET

_DECODER = json.JSONDecoder()
_APOSTROPHES = re.compile(r"['’]")
_NAME_JUNK = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def parse_prediction(text: str) -> dict[str, str] | None:
    """Extract the first JSON object with string merchant_name/category keys.

    Tolerates extra prose around the object (models love to add it) but not a
    malformed object itself. Returns ``None`` when nothing usable is found.
    """
    if not text:
        return None
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if _is_valid(obj):
            return {
                "merchant_name": obj["merchant_name"].strip(),
                "category": obj["category"].strip(),
            }
    return None


def _is_valid(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("merchant_name"), str)
        and isinstance(obj.get("category"), str)
    )


def normalize_name(name: str) -> str:
    """Case/punctuation-insensitive form used for exact-match comparison."""
    cleaned = _APOSTROPHES.sub("", name.lower().replace("&", " and "))
    cleaned = _NAME_JUNK.sub(" ", cleaned)
    return _SPACES.sub(" ", cleaned).strip()


def score_outputs(
    examples: list[dict[str, str]], outputs: list[str]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Score raw model outputs against labeled examples.

    Returns (metrics, per-example records). Metrics:
      - json_validity: share of outputs containing a well-formed prediction object
      - category_accuracy: exact category match (invalid output counts as wrong)
      - name_exact_match: normalized merchant-name match (invalid counts as wrong)
      - both_correct: category and name both right
    """
    if len(examples) != len(outputs):
        raise ValueError(f"got {len(examples)} examples but {len(outputs)} outputs")
    if not examples:
        raise ValueError("nothing to score")

    n = len(examples)
    valid = category_ok = name_ok = both_ok = 0
    records: list[dict[str, Any]] = []
    for example, output in zip(examples, outputs, strict=True):
        parsed = parse_prediction(output)
        cat_hit = name_hit = False
        if parsed is not None:
            valid += 1
            cat_hit = (
                parsed["category"].lower() in CATEGORY_SET
                and parsed["category"].lower() == example["category"]
            )
            name_hit = normalize_name(parsed["merchant_name"]) == normalize_name(
                example["merchant_name"]
            )
        category_ok += cat_hit
        name_ok += name_hit
        both_ok += cat_hit and name_hit
        records.append(
            {
                "raw": example["raw"],
                "expected": {
                    "merchant_name": example["merchant_name"],
                    "category": example["category"],
                },
                "output": output,
                "parsed": parsed,
                "category_correct": cat_hit,
                "name_correct": name_hit,
            }
        )

    metrics = {
        "n": n,
        "json_validity": valid / n,
        "category_accuracy": category_ok / n,
        "name_exact_match": name_ok / n,
        "both_correct": both_ok / n,
    }
    return metrics, records
