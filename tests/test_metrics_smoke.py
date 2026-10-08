"""Smoke tests of the SHIPPED metric code without restricted data.
Script-style modules (part12_metrics.py, aggregate.py) execute their pipeline at import, so the functions under test
are extracted from the shipped source with `ast` (function bodies are the shipped code, byte for byte) and run on
tiny synthetic inputs. Library-style modules (judge_common.py, step3_common.py) are imported directly.
"""
import ast
import os
import sys
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np
import pytest

REL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(REL, "evaluation")


def extract(path, names, ns):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), f"missing functions in {path}"
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


# ------------------------------------------------------------------ step 2: exact random baselines
@pytest.fixture(scope="module")
def rb():
    ns = extract(os.path.join(EV, "02_retrieval_metrics/part12_metrics.py"), ["rand_baseline"],
                 {"KS": [5, 10, 20], "Fraction": Fraction, "comb": comb, "np": np, "lru_cache": lru_cache})
    return ns["rand_baseline"]


@pytest.mark.parametrize("N,G", [(1, 1), (3, 1), (5, 2), (7, 3), (8, 8), (12, 2), (25, 4)])
def test_random_baseline_closed_forms_vs_enumeration(rb, N, G):
    pos = list(combinations(range(1, N + 1), G)); n = len(pos)
    got = rb(N, G, G)
    for k in (5, 10, 20):
        assert got[f"hit@{k}"] == pytest.approx(sum(any(p <= k for p in c) for c in pos) / n, abs=1e-12)
        assert got[f"recall@{k}"] == pytest.approx(sum(sum(p <= k for p in c) / G for c in pos) / n, abs=1e-12)
        assert got[f"coverage@{k}"] == pytest.approx(sum(all(p <= k for p in c) for c in pos) / n, abs=1e-12)
    assert got["mrr"] == pytest.approx(sum(1 / min(c) for c in pos) / n, abs=1e-12)
    assert got["last_gold_rank"] == pytest.approx(sum(max(c) for c in pos) / n, abs=1e-12)


def test_random_baseline_out_of_window_gold_counts_as_miss(rb):
    r = rb(10, 1, 2)
    assert r["coverage@5"] == 0.0 and r["recall@5"] == pytest.approx(0.5 * 0.5)


# ------------------------------------------------------------------ step 4: bootstrap helpers (protocol v1.4)
@pytest.fixture(scope="module")
def agg():
    return extract(os.path.join(EV, "04_scoring/aggregate.py"), ["counts", "boot", "ci", "pval", "holm"], {"np": np, "B": 1000})


def test_bootstrap_counts_seed42_and_paired(agg):
    C = agg["counts"](7)
    assert C.shape == (1000, 7) and (C.sum(1) == 7).all()
    assert np.array_equal(C, agg["counts"](7))                    # fixed seed 42 -> identical resamples for every run


def test_boot_point_estimate_and_ci(agg):
    M = np.array([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    C = agg["counts"](7)
    pt, bt, n = agg["boot"](M, np.ones(7), C)
    assert pt == pytest.approx([5 / 7, 3 / 7]) and list(n) == [7, 7]
    lo, hi = agg["ci"](bt[:, 0] - bt[:, 1])
    assert lo <= pt[0] - pt[1] <= hi
    assert 0 < agg["pval"](bt[:, 0] - bt[:, 1]) <= 1
    assert list(agg["holm"]([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


# ------------------------------------------------------------------ token F1 / EM and judge parsing
@pytest.fixture(scope="module")
def jc():
    sys.path.insert(0, os.path.join(EV, "04_scoring"))
    import judge_common
    return judge_common


def test_squad_f1_em(jc):
    assert jc.exact_match_score("The fictional dose was increased.", "fictional dose was increased") == 1.0
    assert jc.f1_score("dose increased", "the dose was increased") == pytest.approx(2 * (2 / 2) * (2 / 3) / (2 / 2 + 2 / 3))
    assert jc.f1_score("", "anything") == 0


def test_step3_f1_matches_step4_f1(jc):
    sys.path.insert(0, os.path.join(EV, "03_readers"))
    import step3_common
    for p, g in [("dose increased", "the dose was increased"), ("abc", "xyz"), ("a b c", "a b c")]:
        assert step3_common.f1_em(p, g)[0] == pytest.approx(jc.f1_score(p, g))


@pytest.mark.parametrize("text,exp", [("Feedback ... [RESULT] 4", (4, True)), ("[RESULT] 2 [RESULT] 5", (5, True)),
                                      ("[RESULT] 7", (None, False)), ("no result", (None, False)), ("[RESULT] 3.5", (None, False))])
def test_judge_parse_score(jc, text, exp):
    assert jc.parse_score(text) == exp


def test_judge_prompt_has_question_reference_and_no_context(jc):
    pytest.importorskip("prometheus_eval")  # prompt templates come from prometheus-eval==0.1.20 (judge image only)
    p = jc.build_prompt("Invented question?", "invented prediction", "invented gold")
    assert "Invented question?" in p and "invented gold" in p and "invented prediction" in p
