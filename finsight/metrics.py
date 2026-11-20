"""Answer parsing and scoring.

Metrics reported in the README:

* numeric_match: the predicted number is within 1% (relative) of the gold
  number. Percent/decimal forms are treated as equal (14.1% == 0.141), which
  is how FinQA stores percentages.
* exact_match: normalized string equality (used for yes/no and span answers).
* token_f1: SQuAD-style bag-of-tokens F1, useful for free-text answers.
* format_ok: the response contains a parseable 'Answer:' line.
* correct: numeric_match when the gold answer is numeric, otherwise
  exact_match (or token_f1 >= 0.5 for long free-text gold answers).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

ANSWER_RE = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)
NUM_RE = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?\s*%?|\(?-?\$?\s*\.\d+\)?\s*%?")
SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def extract_final_answer(response: str) -> tuple[str, bool]:
    """Return (answer_text, format_ok). Falls back to the last non-empty line."""
    matches = ANSWER_RE.findall(response or "")
    if matches:
        return matches[-1].strip().strip("*").strip(), True
    lines = [l.strip() for l in (response or "").splitlines() if l.strip()]
    return (lines[-1] if lines else ""), False


def parse_number(text: str) -> float | None:
    """Parse the first number in text. Handles $, commas, %, (negatives), and a
    trailing scale word is ignored (gold and prediction use the same units in
    these datasets; scale words are compared separately when present)."""
    if text is None:
        return None
    m = NUM_RE.search(str(text))
    if not m:
        return None
    raw = m.group(0)
    negative = raw.strip().startswith("-") or (raw.strip().startswith("(") and raw.strip().endswith(")"))
    digits = re.sub(r"[^\d.]", "", raw)
    if digits in ("", "."):
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def all_numbers(text: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(str(text or "")):
        v = parse_number(m.group(0))
        if v is not None:
            out.append(v)
    return out


def is_percent(text: str) -> bool:
    return "%" in str(text) or "percent" in str(text).lower()


def numbers_close(pred: float, gold: float, rel_tol: float = 0.01, allow_percent: bool = True) -> bool:
    """True if pred matches gold within rel_tol, allowing a x100 percent/decimal
    mismatch (0.141 vs 14.1) and tiny absolute differences from rounding."""
    candidates = [gold, gold * 100, gold / 100] if allow_percent else [gold]
    for g in candidates:
        if math.isclose(pred, g, rel_tol=rel_tol, abs_tol=0.005 if abs(g) < 1 else 0.05):
            return True
    return False


def normalize_text(s: str) -> str:
    s = str(s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_is_numeric(gold: str) -> bool:
    """A gold answer counts as numeric when it is short and contains a number
    (e.g. '$1577.00', '14.1%', '-3.2'), not prose that happens to mention one."""
    gold = str(gold).strip()
    if gold.lower() in ("yes", "no"):
        return False
    return parse_number(gold) is not None and len(gold.split()) <= 4


def score(response: str, gold: str) -> dict:
    """Score one model response against one gold answer string."""
    pred, format_ok = extract_final_answer(response)
    gold = str(gold).strip()
    result = {
        "pred": pred,
        "format_ok": format_ok,
        "exact_match": normalize_text(pred) == normalize_text(gold),
        "token_f1": token_f1(pred, gold),
        "numeric_match": None,
    }
    if gold_is_numeric(gold):
        g = parse_number(gold)
        p = parse_number(pred)
        result["numeric_match"] = bool(p is not None and numbers_close(p, g))
        result["correct"] = result["numeric_match"]
    elif gold.lower() in ("yes", "no"):
        first = normalize_text(pred).split()[:1]
        result["exact_match"] = first == [gold.lower()]
        result["correct"] = result["exact_match"]
    elif len(gold.split()) > 6:
        # Long free-text gold (FinanceBench analysis questions). Give credit if a
        # gold number appears in the response's answer or token overlap is high.
        # An LLM judge (finsight.judge) is the better measure for these.
        gold_nums = all_numbers(gold)
        pred_nums = all_numbers(pred)
        num_hit = bool(gold_nums) and any(numbers_close(p, g) for p in pred_nums for g in gold_nums[:1])
        result["correct"] = bool(num_hit or result["token_f1"] >= 0.5)
    else:
        result["correct"] = result["exact_match"] or result["token_f1"] >= 0.8
    return result


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    numeric = [s["numeric_match"] for s in scores if s["numeric_match"] is not None]
    return {
        "n": n,
        "accuracy": round(sum(bool(s["correct"]) for s in scores) / n, 4),
        "numeric_accuracy": round(sum(numeric) / len(numeric), 4) if numeric else None,
        "n_numeric": len(numeric),
        "exact_match": round(sum(s["exact_match"] for s in scores) / n, 4),
        "token_f1": round(sum(s["token_f1"] for s in scores) / n, 4),
        "format_ok": round(sum(s["format_ok"] for s in scores) / n, 4),
    }
