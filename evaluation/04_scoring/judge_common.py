"""Shared, dependency-light definitions for Step 4 (protocol v1.4): judge prompt building, parsing, EM/F1.
Imported locally (venv) and inside Modal containers. Prompt strings are imported from the pinned
prometheus-eval==0.1.20 package (verbatim), conversation template from fschat==0.2.36 ('mistral')."""
import re
import string
import collections

PROMETHEUS_REPO = "prometheus-eval/prometheus-8x7b-v2.0"
PROMETHEUS_SHA = "2db013b60e3e91f7a06113e436410899768e8228"
INSTRUCTION_PREFIX = "The following question is about a patient's longitudinal clinical notes. "

RUBRIC = {
    "criteria": ("Does the response give the same clinically relevant answer as the reference answer: the same findings, "
                 "values and medications, and the same temporal relations (order, trend direction, recurrence, timing)? "
                 "Ignore wording, length and style. Extra detail is penalised only if it contradicts the reference; "
                 "length alone must never raise the score."),
    "score1_description": "incorrect, contradicts the reference, irrelevant, empty, or says unanswerable/not stated when the reference gives an answer.",
    "score2_description": "mostly incorrect; little overlap, or the core temporal claim is wrong.",
    "score3_description": "partly correct; some key elements right but an important one (trend direction, time point, value, medication) missing or misstated.",
    "score4_description": "core answer correct; a secondary detail is omitted, but there is no clinically meaningful error.",
    "score5_description": "equivalent to the reference; all key facts and temporal relations correct, no contradiction.",
}

SAMPLING = dict(temperature=0.0, top_p=1.0, max_tokens=1024, seed=42, n=1)


def build_prompt(question: str, prediction: str, gold: str, rubric: dict = RUBRIC) -> str:
    """Exactly what prometheus_eval.PrometheusEval.absolute_grade + VLLM.completions send to vLLM
    (ABSOLUTE_PROMPT with reference, ABS_SYSTEM_PROMPT, fastchat 'mistral' template, then .strip())."""
    from prometheus_eval.prompts import ABSOLUTE_PROMPT, ABS_SYSTEM_PROMPT, SCORE_RUBRIC_TEMPLATE
    from fastchat.conversation import get_conv_template
    rub = SCORE_RUBRIC_TEMPLATE.format(**rubric)
    content = ABSOLUTE_PROMPT.format(instruction=INSTRUCTION_PREFIX + question, response=prediction.strip(),
                                     reference_answer=gold, rubric=rub)
    conv = get_conv_template("mistral")
    conv.set_system_message(ABS_SYSTEM_PROMPT)
    conv.append_message(conv.roles[0], content)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt().strip()


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_score(text: str):
    """Integer after the LAST '[RESULT]'. Returns (score or None, valid_flag).
    Rule: take the text after the last '[RESULT]'; the first numeric token there must be an integer in 1..5."""
    if text is None or "[RESULT]" not in text:
        return None, False
    tail = text.rsplit("[RESULT]", 1)[1]
    m = _NUM.search(tail)
    if not m:
        return None, False
    tok = m.group(0)
    if not re.fullmatch(r"[+]?\d+", tok):
        return None, False
    v = int(tok)
    if 1 <= v <= 5:
        return v, True
    return None, False


# ---------------- SQuAD v1.1 official normalize_answer / EM / F1 (single gold) ----------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))
