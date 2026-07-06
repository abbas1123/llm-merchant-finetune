"""Prompt construction for the normalization task.

Kept free of torch/transformers imports so it can be unit-tested in CI.
"""

from __future__ import annotations

import json

from merchant_llm.categories import CATEGORIES

SYSTEM_PROMPT = (
    "You normalize raw card transaction descriptors. Given a raw merchant string from a "
    "card statement, identify the clean merchant name and its spending category. "
    'Respond with only a JSON object of the form {"merchant_name": "...", "category": "..."}. '
    "The category must be exactly one of: " + ", ".join(CATEGORIES) + ". "
    "Do not output anything except the JSON object."
)


def build_messages(raw: str) -> list[dict[str, str]]:
    """Chat messages for inference (system + user, no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
    ]


def target_json(merchant_name: str, category: str) -> str:
    """Canonical assistant completion used as the training target."""
    return json.dumps(
        {"merchant_name": merchant_name, "category": category}, ensure_ascii=False
    )


def build_training_messages(example: dict[str, str]) -> list[dict[str, str]]:
    """Full chat (system + user + assistant) for supervised fine-tuning."""
    return [
        *build_messages(example["raw"]),
        {
            "role": "assistant",
            "content": target_json(example["merchant_name"], example["category"]),
        },
    ]
