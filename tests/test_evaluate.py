"""End-to-end check of the evaluation and report code with the model mocked out (no GPU needed)."""

import json

import finsight.inference as inference
from finsight import evaluate


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_evaluate_and_report_with_mock_model(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"id": f"q{i}", "source": "finqa", "context": "c", "question": f"q{i}", "answer": str(i)} for i in range(10)]
    _write(data / "test_finqa.jsonl", rows)

    calls = {"load": 0}

    def fake_load(model_id, adapter=None):
        calls["load"] += 1
        return object(), object()

    def make_generate(n_right):
        def fake_generate(model, tokenizer, examples, max_new_tokens=256, batch_size=8):
            return [f"Step 1: ...\nAnswer: {e['answer']}" if int(e["answer"]) < n_right else "Answer: -1"
                    for e in examples]
        return fake_generate

    monkeypatch.setattr(inference, "load", fake_load)
    out = tmp_path / "results"

    monkeypatch.setattr(inference, "generate", make_generate(3))
    evaluate.run("base", None, "m", data, out, ["finqa"], None, 4, 32)
    monkeypatch.setattr(inference, "generate", make_generate(8))
    evaluate.run("finetuned", "a", "m", data, out, ["finqa"], None, 4, 32)

    assert json.loads((out / "metrics_base.json").read_text())["finqa"]["accuracy"] == 0.3
    assert json.loads((out / "metrics_finetuned.json").read_text())["finqa"]["accuracy"] == 0.8

    # Re-running resumes: nothing left to predict, so the model is not loaded again.
    evaluate.run("finetuned", "a", "m", data, out, ["finqa"], None, 4, 32)
    assert calls["load"] == 2

    text = evaluate.report(out)
    assert "| finqa | 10 | 30.0% | 80.0% | +50.0 pts |" in text
