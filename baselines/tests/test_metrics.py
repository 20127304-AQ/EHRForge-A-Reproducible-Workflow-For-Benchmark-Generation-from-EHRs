from ehrforge_repro.metrics import (
    evidence_recall_at_k,
    exact_coverage_at_k,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_normalize_answer() -> None:
    assert normalize_answer("The Patient's answer.") == "patients answer"


def test_exact_match() -> None:
    assert exact_match("The answer", "answer") == 1.0
    assert exact_match("answer one", "answer two") == 0.0


def test_token_f1() -> None:
    assert token_f1("alpha beta", "alpha beta") == 1.0
    assert token_f1("alpha", "alpha beta") == 2 / 3
    assert token_f1("", "") == 1.0
    assert token_f1("", "alpha") == 0.0


def test_retrieval_recall_partial_credit() -> None:
    assert evidence_recall_at_k([2, 7, 9], [2, 4, 8, 10], 3) == 0.25
    assert evidence_recall_at_k([4, 8, 10], [4, 8, 10], 2) == 2 / 3


def test_exact_coverage() -> None:
    assert exact_coverage_at_k([2, 4, 8], [2, 4], 2) == 1.0
    assert exact_coverage_at_k([2, 8, 4], [2, 4], 2) == 0.0
