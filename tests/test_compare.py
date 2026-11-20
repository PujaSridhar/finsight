"""Retrieval and the comparison harness, with every model replaced by a stand-in (no GPU, no API key)."""

import json

from finsight import compare, tasks
from finsight.earnings import target_json
from finsight.retrieval import BM25, ExampleRetriever, build_fewshot_messages, tokenize

QA_TRAIN = [
    {"id": "t1", "context": "Net revenues rose.", "question": "what is the growth rate in net revenues in 2018?",
     "response": "Step 1: ...\nAnswer: 11.9%"},
    {"id": "t2", "context": "Leases.", "question": "what portion of operating leases is due in 2020?",
     "response": "Answer: 30%"},
    {"id": "t3", "context": "Net revenues fell.", "question": "what is the growth rate in net revenues in 2017?",
     "response": "Answer: -2%"},
]
LABEL = {"period_end": "2024-09-28", "revenue": 1_234_000_000, "net_income": 56_000_000, "eps_diluted": 0.97}


def test_tokenize_keeps_numbers_and_drops_stopwords():
    assert tokenize("What is the growth in 2018 of 3.5%?") == ["growth", "2018", "3.5"]


def test_bm25_prefers_matching_terms():
    s = BM25([["revenue", "growth"], ["lease", "due"], ["revenue"]]).scores(["revenue", "growth"])
    assert max(s, key=s.get) == 0 and 1 not in s


def test_retriever_ranks_similar_inputs_and_skips_excluded():
    task = tasks.get("qa")
    r = ExampleRetriever(QA_TRAIN, task.retrieval_text)
    query = {"question": "what was the growth rate in net revenues in 2019?", "context": ""}
    assert {d["id"] for d in r.retrieve(query, k=2)} == {"t1", "t3"}
    query = {"question": "growth rate in net revenues and leases", "context": ""}
    assert [d["id"] for d in r.retrieve(query, k=2, exclude_ids={"t1", "t3"})] == ["t2"]


def test_fewshot_messages_alternate_and_end_with_the_input():
    task = tasks.get("earnings")
    demos = [{"id": "d1", "text": "Release one", "target": target_json(LABEL)}]
    msgs = build_fewshot_messages(task, {"text": "The new release"}, demos)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == demos[0]["target"] and msgs[-1]["content"].endswith("The new release")


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_run_scores_records_usage_and_resumes(tmp_path):
    task = tasks.get("earnings")
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"id": f"r{i}", "source": "earnings", "text": "release", "label": {**LABEL, "eps_diluted": 0.97 + i},
             "distractors": {}} for i in range(5)]
    _write(data / "test_earnings.jsonl", rows)
    out = tmp_path / "out"
    loads = {"n": 0}

    def loader():
        loads["n"] += 1
        return lambda convs: [{"response": target_json(LABEL), "input_tokens": 2500, "output_tokens": 50,
                               "latency_s": 2.0} for _ in convs]

    m = compare.run(task, "hosted", loader, data, out, ["earnings"], None, model_label="claude-opus-5-5")
    assert m["earnings"]["n"] == 5 and m["earnings"]["accuracy"] == 0.2 and m["earnings"]["revenue"] == 1.0
    assert m["_cost_latency"]["p50_latency_s"] == 2.0 and m["_cost_latency"]["model"] == "claude-opus-5-5"

    compare.run(task, "hosted", loader, data, out, ["earnings"], None, model_label="claude-opus-5-5")
    assert loads["n"] == 1  # nothing left to answer, so the backend is not loaded again

    pricing = {"hosted": {"claude-opus-5-5": {"input_per_mtok": 4.0, "output_per_mtok": 20.0}}}
    assert abs(compare.cost_per_query("hosted", m["_cost_latency"], pricing, None) - 0.011) < 1e-9
    assert compare.cost_per_query("base", {"n": 5, "mean_latency_s": 3.6}, {}, 1.0) == 0.001
    assert compare.cost_per_query("base", {"n": 5, "mean_latency_s": 3.6}, {}, None) is None

    text = compare.report(task, out, tmp_path / "missing.yaml")
    assert "| hosted (claude-opus-5-5) | 20.0% (n=5) | 100.0% | 100.0% | 100.0% | 20.0% | n/a | 2.00 s |" in text
    assert "other 4" in text


def test_report_uses_gpu_price_for_local_cost(tmp_path):
    (tmp_path / "metrics_finetuned.json").write_text(json.dumps({
        "earnings": {"n": 100, "accuracy": 0.6, "errors": {}},
        "_cost_latency": {"n": 100, "mean_input_tokens": 2400, "mean_output_tokens": 50,
                          "mean_latency_s": 4.0, "p50_latency_s": 3.5, "p95_latency_s": 7.0, "model": "m"}}))
    text = compare.report(tasks.get("earnings"), tmp_path, tmp_path / "missing.yaml", gpu_hourly_usd=0.9)
    assert "| finetuned (m) | 60.0% (n=100) |" in text and "| $1.00 | 3.50 s | 7.00 s | 2400 / 50 |" in text
