"""Optional: LLM-as-judge for FinanceBench's free-text answers.

About two thirds of FinanceBench answers are analyst-style sentences ("No, the
operating margin declined from 36.8% to 34.6%..."), so string metrics
under-count correct answers. This script asks Claude to grade each prediction
against the gold answer and writes a verdict per item.

    export ANTHROPIC_API_KEY=...        # or `ant auth login`
    python -m finsight.judge results/predictions/finetuned_financebench.jsonl

Costs a few cents for 150 items. Skip it if you don't have API access; the
README reports the automatic metrics either way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

JUDGE_PROMPT = """You are grading answers to financial questions about company filings.

Question: {question}

Gold answer: {gold}

Model answer: {pred}

The model answer is correct if it reaches the same conclusion as the gold answer and any
key figures agree (small rounding differences are fine). Extra explanation is fine;
contradicting the gold answer or giving the wrong figure is not."""

SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


def judge_file(pred_path: Path, model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    rows = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
    out_path = pred_path.with_name(pred_path.stem + "_judged.jsonl")
    done = {}
    if out_path.exists():
        done = {r["id"]: r for r in map(json.loads, out_path.read_text().splitlines()) if r.get("judge_correct") is not None}

    with out_path.open("w") as f:
        for r in rows:
            if r["id"] in done:
                f.write(json.dumps(done[r["id"]]) + "\n")
                continue
            prompt = JUDGE_PROMPT.format(question=r["question"], gold=r["answer"], pred=r["response"])
            verdict, reason = None, ""
            try:
                resp = client.beta.messages.create(
                    model=model,
                    max_tokens=1024,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                    messages=[{"role": "user", "content": prompt}],
                )
                if resp.stop_reason == "refusal":
                    reason = "judge declined"
                else:
                    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
                    verdict, reason = bool(data["correct"]), data["reason"]
            except anthropic.APIStatusError as e:
                reason = f"API error {e.status_code}"
            except anthropic.APIConnectionError:
                reason = "connection error"
            r = {**r, "judge_correct": verdict, "judge_reason": reason}
            f.write(json.dumps(r) + "\n")
            f.flush()

    judged = [json.loads(l) for l in out_path.read_text().splitlines()]
    graded = [r for r in judged if r["judge_correct"] is not None]
    summary = {
        "file": str(pred_path),
        "judge_model": model,
        "n_graded": len(graded),
        "judge_accuracy": round(sum(r["judge_correct"] for r in graded) / len(graded), 4) if graded else None,
    }
    pred_path.with_name(pred_path.stem + "_judge_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", type=Path)
    ap.add_argument("--model", default="claude-opus-5-5")
    args = ap.parse_args()
    for p in args.predictions:
        print(json.dumps(judge_file(p, args.model), indent=2))


if __name__ == "__main__":
    main()
