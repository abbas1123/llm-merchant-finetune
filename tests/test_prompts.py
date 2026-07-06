import json

from merchant_llm.categories import CATEGORIES
from merchant_llm.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_training_messages,
    target_json,
)


def test_system_prompt_lists_every_category():
    for category in CATEGORIES:
        assert category in SYSTEM_PROMPT


def test_build_messages_structure():
    messages = build_messages("SQ *BEAN THEORY AUSTIN")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "SQ *BEAN THEORY AUSTIN"


def test_target_json_round_trips():
    target = target_json("Bean Theory", "coffee")
    assert json.loads(target) == {"merchant_name": "Bean Theory", "category": "coffee"}
    # key order is fixed so training targets are byte-identical across runs
    assert target == '{"merchant_name": "Bean Theory", "category": "coffee"}'


def test_training_messages_end_with_assistant_target():
    example = {"raw": "TST* RUSTIC FORK", "merchant_name": "Rustic Fork", "category": "restaurants"}
    messages = build_training_messages(example)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert json.loads(messages[-1]["content"]) == {
        "merchant_name": "Rustic Fork",
        "category": "restaurants",
    }
