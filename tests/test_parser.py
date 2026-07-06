import pytest

from merchant_llm.parsing import normalize_name, parse_prediction, score_outputs


def test_parses_clean_json():
    parsed = parse_prediction('{"merchant_name": "Bean Theory", "category": "coffee"}')
    assert parsed == {"merchant_name": "Bean Theory", "category": "coffee"}


def test_parses_json_with_surrounding_prose():
    text = 'Sure! Here is the result:\n```json\n{"merchant_name": "Shell", "category": "fuel"}\n```'
    assert parse_prediction(text) == {"merchant_name": "Shell", "category": "fuel"}


def test_skips_broken_braces_before_valid_object():
    text = '{oops not json} then {"merchant_name": "Netflix", "category": "streaming"} trailing'
    assert parse_prediction(text) == {"merchant_name": "Netflix", "category": "streaming"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "the merchant is Shell (fuel)",
        '{"merchant_name": "Shell", "category": ',
        '{"merchant": "Shell", "category": "fuel"}',
        '{"merchant_name": "Shell"}',
        '{"merchant_name": 42, "category": "fuel"}',
        '{"merchant_name": "Shell", "category": ["fuel"]}',
    ],
)
def test_rejects_invalid_outputs(text):
    assert parse_prediction(text) is None


def test_normalize_name():
    assert normalize_name("  Trader Joe's ") == "trader joes"
    assert normalize_name("EMBER & OAK") == "ember and oak"
    assert normalize_name("Bowl-O-Rama  Lanes") == "bowl o rama lanes"


def _example(raw, name, category):
    return {"raw": raw, "merchant_name": name, "category": category}


def test_score_outputs_counts_correctly():
    examples = [
        _example("SQ *BEAN THEORY", "Bean Theory", "coffee"),
        _example("SHELL 4421 HOUSTON TX", "Shell", "fuel"),
        _example("NETFLIX.COM", "Netflix", "streaming"),
        _example("WALGREENS #221", "Walgreens", "pharmacy"),
    ]
    outputs = [
        '{"merchant_name": "bean theory", "category": "coffee"}',  # both right (case-insensitive)
        '{"merchant_name": "Shell", "category": "grocery"}',  # name right, category wrong
        "NETFLIX is a streaming service",  # malformed
        '{"merchant_name": "Walgreens", "category": "not_a_category"}',  # invalid category
    ]
    metrics, records = score_outputs(examples, outputs)
    assert metrics["n"] == 4
    assert metrics["json_validity"] == pytest.approx(3 / 4)
    assert metrics["category_accuracy"] == pytest.approx(1 / 4)
    assert metrics["name_exact_match"] == pytest.approx(3 / 4)
    assert metrics["both_correct"] == pytest.approx(1 / 4)
    assert records[2]["parsed"] is None
    assert records[3]["category_correct"] is False


def test_score_outputs_validates_lengths():
    with pytest.raises(ValueError):
        score_outputs([_example("A", "A", "fuel")], [])
    with pytest.raises(ValueError):
        score_outputs([], [])
