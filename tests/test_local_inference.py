"""Tests that need the model weights on disk. Skipped unless MERCHANT_LOCAL_TESTS=1."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.local


@pytest.fixture(scope="module")
def base_model():
    from merchant_llm.config import DEFAULT_MODEL
    from merchant_llm.inference import load_model

    return load_model(DEFAULT_MODEL)


def test_base_model_generates_text(base_model):
    from merchant_llm.inference import generate_batch

    model, tokenizer, device = base_model
    outputs = generate_batch(model, tokenizer, ["SQ *BEAN THEORY AUSTIN TX"], device)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str) and outputs[0].strip()


@pytest.mark.skipif(
    not Path("models/adapter/adapter_config.json").exists(),
    reason="no trained adapter on disk",
)
def test_adapter_output_parses():
    from merchant_llm.config import DEFAULT_MODEL
    from merchant_llm.inference import generate_batch, load_model
    from merchant_llm.parsing import parse_prediction

    model, tokenizer, device = load_model(DEFAULT_MODEL, "models/adapter")
    outputs = generate_batch(model, tokenizer, ["WALGREENS #4421 CHICAGO IL"], device)
    parsed = parse_prediction(outputs[0])
    assert parsed is not None
    assert parsed["category"]
