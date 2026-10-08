"""Tiny SYNTHETIC fixture: 3 invented patients, invented note text, dates in the year 2099. No real data.

Layout written under <root>/ mirrors the original project layout used by ehr_data.py:
  data/dataset.csv, data/corpus.json (jsonl), shared/canonical/{ehr_data.py, sampled_window.py, protocol*.yaml}

Evidence items carry BOTH a chunk-relative `visit_index` (deliberately different from the full-timeline index) and the
full-timeline `original_visit_index`, so the visit_index bug of the pre-fix code is observable in tests.
"""
import json
import os
import shutil

REL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # release root


def _visits(pid, n):
    return [{"visit_datetime": f"2099-01-{d + 1:02d} 08:00:00",
             "text": f"Synthetic note {pid}-{d}: invented observation number {d}; fictional vitals stable; plan continue."}
            for d in range(n)]


CORPUS = {101: _visits(101, 6), 102: _visits(102, 40), 103: _visits(103, 3)}


def _ev(pid, ovi, chunk_idx):
    return {"visit_index": chunk_idx, "original_visit_index": ovi,
            "visit_datetime": CORPUS[pid][ovi]["visit_datetime"],
            "evidence_snippet": f"invented observation number {ovi}"}


ROWS = [
    # person_id, qa_index, strategy, evidence (ovi, chunk-relative visit_index), reasoning_type, difficulty, visit_group
    (101, 0, "full_timeline_2_10", [(1, 1), (4, 4)], "trend", "easy", "2-10 visits"),
    (101, 1, "full_timeline_2_10", [(2, 2), (2, 2)], "before_after", "medium", "2-10 visits"),          # duplicate evidence
    (102, 0, "local_chunk_25_40_size_30_overlap_5", [(30, 5), (27, 2)], "progression", "hard", "11-100 visits"),
    (102, 1, "u_shape_first_12_last_18", [(3, 3), (38, 28)], "comparison_across_visits", "medium", "11-100 visits"),
    (103, 0, "full_timeline_2_10", [(0, 0)], "first_last_occurrence", "easy", "2-10 visits"),
]


def build(root):
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    import pandas as pd
    recs = []
    for pid, qi, strat, ev, rt, dif, vg in ROWS:
        recs.append(dict(person_id=pid, qa_index=qi, num_total_visits=len(CORPUS[pid]), visit_group=vg,
                         timeline_sampling_strategy=strat, question=f"Invented question {pid}-{qi}?",
                         answer=f"Invented answer {pid}-{qi}.", reasoning_type=rt, difficulty=dif,
                         evidence=json.dumps([_ev(pid, o, c) for o, c in ev])))
    pd.DataFrame(recs).to_csv(os.path.join(root, "data", "dataset.csv"), index=False)
    with open(os.path.join(root, "data", "corpus.json"), "w", encoding="utf-8") as f:
        for pid, v in CORPUS.items():
            f.write(json.dumps({"person_id": pid, "visits": v}) + "\n")
    canon = os.path.join(root, "shared", "canonical")
    os.makedirs(canon, exist_ok=True)
    for fn in ("ehr_data.py", "sampled_window.py"):
        shutil.copyfile(os.path.join(REL, "evaluation", "common", fn), os.path.join(canon, fn))
    for fn in os.listdir(os.path.join(REL, "evaluation", "protocol")):
        if fn.startswith("protocol"):
            shutil.copyfile(os.path.join(REL, "evaluation", "protocol", fn), os.path.join(canon, fn))
    return root


class CharTokenizer:
    """Deterministic stand-in for the Qwen budget tokenizer: one token per character (tests only)."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, s, add_special_tokens=False):
        return self._Enc([ord(c) for c in s])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)
