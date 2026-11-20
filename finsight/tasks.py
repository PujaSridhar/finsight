"""The two tasks this repo trains on, behind one interface used by training, comparison, serving and CI.

  earnings  the product: earnings release -> JSON of key GAAP figures (finsight/earnings.py)
  qa        optional warm-up stage: numerical questions over 10-K excerpts from FinQA and TAT-QA
            (finsight/data.py), kept to test whether that practice helps extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Task:
    name: str
    data_dir: str
    test_sets: dict[str, str]
    messages: Callable[[dict], list[dict]]          # prompt for one example
    target: Callable[[dict], str]                   # training target for one example
    score: Callable[[str, dict], dict]              # (response, example) -> scores, incl. "correct"
    aggregate: Callable[[list[dict]], dict]
    retrieval_text: Callable[[dict], str]           # what the few-shot retriever matches on
    record_keys: tuple[str, ...]
    demo_chars: int | None = None                   # truncate few-shot demos' documents (None = keep whole)
    max_new_tokens: int = 128


def _earnings() -> Task:
    from . import earnings as e

    return Task(
        name="earnings", data_dir="data/earnings", test_sets={"earnings": "test_earnings.jsonl"},
        messages=lambda ex: e.build_messages(ex["text"]),
        target=lambda ex: ex["target"],
        score=e.score, aggregate=e.aggregate,
        retrieval_text=lambda ex: ex["text"][:1500],
        record_keys=("id", "source", "ticker", "filed", "url", "label"),
        demo_chars=None, max_new_tokens=128,  # the JSON answer is ~50 tokens
    )


def _qa() -> Task:
    from . import metrics
    from .evaluate import TEST_SETS
    from .prompts import build_messages

    return Task(
        name="qa", data_dir="data/processed", test_sets=TEST_SETS,
        messages=lambda ex: build_messages(ex["context"], ex["question"]),
        target=lambda ex: ex["response"],
        score=lambda resp, ex: metrics.score(resp, ex["answer"]), aggregate=metrics.aggregate,
        retrieval_text=lambda ex: ex["question"] + "\n" + ex["context"][:300],
        record_keys=("id", "source", "question", "answer", "question_type", "answer_type"),
        demo_chars=1200, max_new_tokens=384,
    )


def get(name: str) -> Task:
    return {"earnings": _earnings, "qa": _qa}[name]()
