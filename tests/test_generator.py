import json

from merchant_llm.categories import CATEGORY_SET
from merchant_llm.data.generate import MERCHANTS, generate_examples, generate_splits, main


def test_deterministic_for_same_seed():
    a = generate_splits(train=200, val=40, test=40, seed=7)
    b = generate_splits(train=200, val=40, test=40, seed=7)
    assert a == b


def test_different_seed_changes_data():
    a = generate_examples(100, seed=7)
    b = generate_examples(100, seed=8)
    assert a != b


def test_schema_and_labels():
    for row in generate_examples(300, seed=11):
        assert set(row) == {"raw", "merchant_name", "category"}
        assert isinstance(row["raw"], str) and len(row["raw"]) >= 4
        assert row["category"] in CATEGORY_SET
        assert row["merchant_name"] in MERCHANTS[row["category"]]


def test_raw_strings_unique():
    rows = generate_examples(500, seed=3)
    raws = [row["raw"] for row in rows]
    assert len(set(raws)) == len(raws)


def test_splits_are_disjoint_and_sized():
    splits = generate_splits(train=300, val=60, test=60, seed=5)
    assert len(splits["train"]) == 300
    assert len(splits["val"]) == 60
    assert len(splits["test"]) == 60
    train = {r["raw"] for r in splits["train"]}
    val = {r["raw"] for r in splits["val"]}
    test = {r["raw"] for r in splits["test"]}
    assert not train & val
    assert not train & test
    assert not val & test


def test_cli_writes_expected_files(tmp_path):
    main(
        [
            "--out", str(tmp_path),
            "--train", "40",
            "--val", "10",
            "--test", "10",
            "--seed", "3",
            "--sample-size", "5",
        ]
    )  # fmt: skip
    for name, expected in [("train", 40), ("val", 10), ("test", 10), ("sample", 5)]:
        lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == expected
        json.loads(lines[0])
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 3
    assert meta["splits"] == {"train": 40, "val": 10, "test": 10}
