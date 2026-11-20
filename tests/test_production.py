"""Serving API, eval gate and monitoring, with the model replaced by stand-ins (no GPU)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from finsight import ci_eval, monitor, tasks
from finsight.earnings import target_json
from finsight.serve import VLLMBackend, create_app

LABEL = {"period_end": "2024-09-28", "revenue": 1_234_000_000, "net_income": 56_000_000, "eps_diluted": 0.97}


class FakeBackend:
    def __init__(self, response=target_json(LABEL), fail=False):
        self.response, self.fail, self.seen = response, fail, []

    def complete(self, messages):
        if self.fail:
            raise TimeoutError
        self.seen.append(messages)
        return {"response": self.response, "input_tokens": 2500, "output_tokens": 50}

    def ready(self):
        return not self.fail


def test_extract_from_text_returns_fields_timing_and_logs_without_content(tmp_path):
    log = tmp_path / "requests.jsonl"
    backend = FakeBackend()
    client = TestClient(create_app(backend, log))
    r = client.post("/v1/extract", json={"text": "ACME reports Q3 results. Revenue | $1,234"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"] == LABEL and body["valid_json"] and set(body["timing_ms"]) == {"fetch", "model", "total"}
    rec = json.loads(log.read_text())
    assert rec["format_ok"] and rec["source"] == "text" and "response" not in rec and "url" not in rec
    assert backend.seen[0][0]["content"].endswith("Revenue | $1,234")


def test_extract_fetches_only_sec_urls(tmp_path):
    fetched = []
    client = TestClient(create_app(FakeBackend(), None, fetch_url=lambda u: fetched.append(u) or "release text"))
    ok = client.post("/v1/extract", json={"url": "https://www.sec.gov/Archives/edgar/data/1/x.htm"})
    assert ok.status_code == 200 and fetched == ["https://www.sec.gov/Archives/edgar/data/1/x.htm"]
    assert client.post("/v1/extract", json={"url": "https://example.com/x.htm"}).status_code == 422
    assert client.post("/v1/extract", json={}).status_code == 422
    assert client.post("/v1/extract", json={"text": "a", "url": "https://www.sec.gov/x"}).status_code == 422


def test_backend_errors_are_502_and_logged(tmp_path):
    log = tmp_path / "requests.jsonl"
    client = TestClient(create_app(FakeBackend(fail=True), log))
    assert client.post("/v1/extract", json={"text": "release"}).status_code == 502
    assert client.get("/health").status_code == 503
    assert json.loads(log.read_text())["error"] == "TimeoutError"


def test_unparseable_output_returns_null_fields():
    client = TestClient(create_app(FakeBackend(response="I am not sure."), None))
    body = client.post("/v1/extract", json={"text": "release"}).json()
    assert body["fields"] is None and not body["valid_json"]


def test_vllm_backend_sends_greedy_request_to_the_lora_model():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": " {} "}}],
                                         "usage": {"prompt_tokens": 40, "completion_tokens": 3}})

    backend = VLLMBackend("http://vllm", "finsight", transport=httpx.MockTransport(handler))
    out = backend.complete([{"role": "user", "content": "hi"}])
    assert out == {"response": "{}", "input_tokens": 40, "output_tokens": 3}
    assert seen["model"] == "finsight" and seen["temperature"] == 0


GATE = {"min_format_ok": 0.95, "max_accuracy_drop": 0.08, "rescore_tolerance": 0.005}
EXAMPLES = [{"id": f"r{i}", "text": "release", "label": {**LABEL, "eps_diluted": 0.97 + i}} for i in range(10)]
TASK = tasks.get("earnings")


def test_gate_passes_a_matching_model_and_fails_a_regression():
    good = ci_eval.smoke_eval(TASK, lambda m: target_json(LABEL), EXAMPLES)  # 1 of 10 right
    assert good["metrics"]["accuracy"] == 0.1 and ci_eval.check(good, None, GATE) == []
    assert ci_eval.check(good, good, GATE) == []
    strong = {"metrics": {"n": 10, "accuracy": 0.5, "format_ok": 1.0}, "per_input": {"r0": True}}
    broken = ci_eval.smoke_eval(TASK, lambda m: "no idea", EXAMPLES)
    failures = ci_eval.check(broken, strong, GATE)
    assert len(failures) == 2 and "parsed" in failures[0] and "fell" in failures[1]


def test_gate_counts_endpoint_errors_as_wrong():
    def boom(messages):
        raise ConnectionError

    assert ci_eval.smoke_eval(TASK, boom, EXAMPLES[:2])["metrics"]["accuracy"] == 0


@pytest.mark.parametrize("saved_acc, expect_fail", [(0.5, False), (0.9, True)])
def test_rescore_flags_published_numbers_that_move(tmp_path, saved_acc, expect_fail):
    (tmp_path / "predictions").mkdir()
    preds = [{"id": "a", "response": "Answer: 1", "answer": "1"}, {"id": "b", "response": "Answer: 2", "answer": "3"}]
    (tmp_path / "predictions" / "finetuned_finqa.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    (tmp_path / "metrics_finetuned.json").write_text(json.dumps({"finqa": {"accuracy": saved_acc}}))
    assert bool(ci_eval.rescore(tasks.get("qa"), tmp_path, 0.005)) is expect_fail


def test_rescore_works_on_earnings_predictions(tmp_path):
    (tmp_path / "predictions").mkdir()
    preds = [{"id": "a", "response": target_json(LABEL), "label": LABEL}]
    (tmp_path / "predictions" / "finetuned_earnings.jsonl").write_text(json.dumps(preds[0]) + "\n")
    (tmp_path / "metrics_finetuned.json").write_text(json.dumps({"earnings": {"accuracy": 1.0}}))
    assert ci_eval.rescore(TASK, tmp_path, 0.005) == []


def test_monitor_summarizes_by_day():
    recs = [{"ts": "2026-11-20T10:00:00", "latency_ms": 100, "format_ok": True, "context_chars": 900},
            {"ts": "2026-11-20T11:00:00", "latency_ms": 300, "format_ok": False, "context_chars": 1100},
            {"ts": "2026-11-20T12:00:00", "error": "TimeoutError", "latency_ms": 60000, "context_chars": 1000},
            {"ts": "2026-11-21T09:00:00", "latency_ms": 200, "format_ok": True, "context_chars": 800}]
    day1, day2 = monitor.summarize(recs)
    assert day1["requests"] == 3 and day1["error_rate"] == round(1 / 3, 4) and day1["format_ok"] == 0.5
    assert day1["p95_ms"] == 300 and day2["requests"] == 1
