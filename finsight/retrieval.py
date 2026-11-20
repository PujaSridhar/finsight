"""Retrieval for the RAG baseline: BM25 over the solved training examples.

The strongest no-training alternative to fine-tuning is to show the base model
worked examples at inference time. For each test input we retrieve the k most
similar training examples (BM25) and put them in the prompt as earlier chat
turns, each with its verified answer. If this baseline closes most of the gap,
fine-tuning is not earning its keep; if it does, it still costs a longer prompt
on every request, which shows up in latency and cost per query.

Pure Python, no extra dependencies, so it also runs on a laptop.
"""

from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the to was were what which
with this how did does do during year years million thousand""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg_len = sum(self.lens) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.index: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.index.setdefault(t, []).append(i)

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.index[t]:
                tf = self.tfs[i][t]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / denom
        return out


class ExampleRetriever:
    """Retrieve solved training examples similar to a new input. `text_fn` says what to match on."""

    def __init__(self, train: list[dict], text_fn):
        self.train, self.text_fn = train, text_fn
        self.bm25 = BM25([tokenize(text_fn(r)) for r in train])

    def retrieve(self, example: dict, k: int = 3, exclude_ids: set[str] | None = None) -> list[dict]:
        ranked = sorted(self.bm25.scores(tokenize(self.text_fn(example))).items(), key=lambda kv: -kv[1])
        picked, seen = [], set()
        for i, _ in ranked:
            r = self.train[i]
            key = self.text_fn(r)[:200].lower()
            if (exclude_ids and r["id"] in exclude_ids) or key in seen:  # skip near-duplicate templates
                continue
            seen.add(key)
            picked.append(r)
            if len(picked) == k:
                break
        return picked


def truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] + "\n[...]"


def build_fewshot_messages(task, example: dict, demos: list[dict], demo_chars: int | None = None) -> list[dict]:
    """Each retrieved demo is an earlier user/assistant turn, then the real input.

    demo_chars truncates the demos' documents (QA excerpts are long and only the worked answer matters there);
    earnings demos are kept whole, because a demo whose answer is not in its text would teach guessing.
    """
    messages: list[dict] = []
    for d in demos:
        shown = {k: truncate(v, demo_chars) if k in ("text", "context") else v for k, v in d.items()}
        messages += task.messages(shown) + [{"role": "assistant", "content": task.target(d)}]
    return messages + task.messages(example)
