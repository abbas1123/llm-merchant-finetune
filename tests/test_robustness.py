from eval.run_robustness import evaluate


def test_parser_recovers_all_messy_outputs():
    res = evaluate()
    failures = [label for label, ok in res["recoverable_rows"] if not ok]
    assert res["extraction_rate"] == 1.0, f"failed to extract: {failures}"


def test_parser_safely_rejects_all_malformed_outputs():
    res = evaluate()
    leaks = [label for label, ok in res["malformed_rows"] if not ok]
    assert res["safe_rejection_rate"] == 1.0, f"did not reject: {leaks}"
