import json
from pathlib import Path

import pytest

from finsight.data import convert_finqa, convert_tatqa, execute_finqa_program, safe_eval
from finsight.metrics import score
from finsight.prompts import table_to_markdown

TABLE = [["", "2015", "2014"], ["net revenue", "$ 5829", "$ 5735"], ["other", "-14 ( 14 )", "10"]]


def test_execute_finqa_program_with_references():
    value, steps = execute_finqa_program("subtract(5829, 5735), divide(#0, 5735)", TABLE)
    assert abs(value - 94 / 5735) < 1e-9
    assert steps[0] == "Step 1: 5,829 - 5,735 = 94"


def test_execute_finqa_table_and_const_ops():
    value, _ = execute_finqa_program("table_sum(net revenue, none)", TABLE)
    assert value == 5829 + 5735
    value, _ = execute_finqa_program("multiply(0.5, const_100)", TABLE)
    assert value == 50
    value, _ = execute_finqa_program("greater(5829, 5735)", TABLE)
    assert value is True


def test_safe_eval_rejects_code():
    assert safe_eval("[(2.9+2.9)/2] - [(2.7+2.7)/2]") == pytest.approx(0.2)
    with pytest.raises(Exception):
        safe_eval("__import__('os').system('ls')")


def test_table_to_markdown():
    md = table_to_markdown([["", "2019"], ["Revenue", "10"]])
    assert md.splitlines()[0] == "|  | 2019 |" and md.splitlines()[2] == "| Revenue | 10 |"


def _finqa_row(program, exe_ans, answer):
    return {"id": "X/2016/page_1.pdf-1", "pre_text": ["revenue rose ."], "post_text": [], "table": TABLE,
            "table_ori": TABLE, "qa": {"question": "what was the change?", "program": program,
                                       "exe_ans": exe_ans, "answer": answer}}


def test_convert_finqa_verifies_and_formats_percent():
    rows = [
        _finqa_row("subtract(5829, 5735), divide(#0, 5735)", 0.01639, "1.6%"),
        _finqa_row("subtract(5829, 5735)", 100.0, "100"),  # program disagrees with gold: dropped
    ]
    out, drops = convert_finqa(rows, "train")
    assert len(out) == 1 and drops["unverified"] == 1
    assert out[0]["answer"] == "1.6%"
    assert out[0]["response"].endswith("Answer: 1.6%")
    # The training target must score as correct against its own gold answer.
    assert score(out[0]["response"], out[0]["answer"])["correct"]


def test_convert_tatqa_arithmetic_and_spans():
    doc = {"table": {"uid": "t1", "table": [["", "2019"], ["Rate", "2.9"]]},
           "paragraphs": [{"uid": "p", "order": 1, "text": "Assumptions."}],
           "questions": [
               {"uid": "q1", "question": "Average rate?", "answer": 2.9, "derivation": "(2.9+2.9)/2",
                "answer_type": "arithmetic", "scale": "percent"},
               {"uid": "q2", "question": "What items?", "answer": ["Rate"], "derivation": "",
                "answer_type": "span", "scale": ""},
           ]}
    out, _ = convert_tatqa([doc], "train")
    assert [o["answer"] for o in out] == ["2.9%", "Rate"]
    assert out[0]["response"].startswith("Calculation: (2.9+2.9)/2 = 2.9")


@pytest.mark.skipif(not Path("data/processed/train.jsonl").exists(), reason="run python -m finsight.data first")
def test_built_targets_score_as_correct():
    """Every training target should get full marks from our own metric (sanity check of labels + metric)."""
    rows = [json.loads(l) for l in Path("data/processed/train.jsonl").read_text().splitlines()[:2000]]
    wrong = [r for r in rows if not score(r["response"], r["answer"])["correct"]]
    assert len(wrong) / len(rows) < 0.01, wrong[:3]
