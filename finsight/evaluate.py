"""Phase 3: evaluate the base model and the fine-tuned adapter on held-out sets.

    # base model, zero-shot, same prompt
    python -m finsight.evaluate --tag base
    # fine-tuned
    python -m finsight.evaluate --tag finetuned --adapter outputs/finsight-mistral-7b-qlora/final
    # comparison table -> results/RESULTS.md
    python -m finsight.evaluate --report

Test sets (none seen in training):
  finqa         300 questions sampled from the FinQA test split (in-domain)
  tatqa         300 questions from the TAT-QA dev split (in-domain, unseen filings)
  financebench  all 150 open-source FinanceBench questions (out-of-domain)

Predictions are written as they are produced and skipped on re-run, so a
Colab disconnect only loses the current batch.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import aggregate, score

TEST_SETS = {"finqa": "test_finqa.jsonl", "tatqa": "test_tatqa.jsonl", "financebench": "test_financebench.jsonl"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def run(tag: str, adapter: str | None, model_id: str, data_dir: Path, out_dir: Path, sets: list[str],
        limit: int | None, batch_size: int, max_new_tokens: int) -> None:
    from .inference import generate, load

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    for name in sets:
        rows = read_jsonl(data_dir / TEST_SETS[name])[:limit]
        pred_path = pred_dir / f"{tag}_{name}.jsonl"
        done = {r["id"] for r in read_jsonl(pred_path)}
        todo = [r for r in rows if r["id"] not in done]
        print(f"[{tag}] {name}: {len(done)} done, {len(todo)} to go")
        if todo and model is None:
            model, tokenizer = load(model_id, adapter)  # only load the 7B model if there is work to do
        with pred_path.open("a") as f:
            for i in range(0, len(todo), batch_size):
                chunk = todo[i:i + batch_size]
                for ex, resp in zip(chunk, generate(model, tokenizer, chunk, max_new_tokens, batch_size)):
                    rec = {k: ex.get(k) for k in ("id", "source", "question", "answer", "question_type", "answer_type")}
                    rec["response"] = resp
                    rec.update(score(resp, ex["answer"]))
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  {min(i + batch_size, len(todo))}/{len(todo)}", end="\r")
    summarize(tag, out_dir)


def summarize(tag: str, out_dir: Path) -> dict:
    metrics = {}
    for name in TEST_SETS:
        preds = read_jsonl(out_dir / "predictions" / f"{tag}_{name}.jsonl")
        if not preds:
            continue
        metrics[name] = aggregate(preds)
        # Breakdowns that are worth showing in the README.
        key = "question_type" if name == "financebench" else ("answer_type" if name == "tatqa" else None)
        if key:
            groups = defaultdict(list)
            for p in preds:
                groups[p.get(key) or "unknown"].append(p)
            metrics[name]["by_" + key] = {k: aggregate(v)["accuracy"] for k, v in sorted(groups.items())}
        judge = out_dir / "predictions" / f"{tag}_{name}_judge_summary.json"
        if judge.exists():
            metrics[name]["judge_accuracy"] = json.loads(judge.read_text())["judge_accuracy"]
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def report(out_dir: Path) -> str:
    base = json.loads((out_dir / "metrics_base.json").read_text())
    ft = json.loads((out_dir / "metrics_finetuned.json").read_text())
    lines = ["# FinSight results", "",
             "Mistral-7B-Instruct-v0.3, zero-shot vs. after QLoRA fine-tuning, same prompt, greedy decoding.", "",
             "| Test set | n | Base accuracy | Fine-tuned accuracy | Change | Base format OK | Fine-tuned format OK |",
             "|---|---|---|---|---|---|---|"]
    for name in TEST_SETS:
        if name in base and name in ft:
            b, f = base[name], ft[name]
            delta = f"{100 * (f['accuracy'] - b['accuracy']):+.1f} pts"
            lines.append(f"| {name} | {f['n']} | {pct(b['accuracy'])} | {pct(f['accuracy'])} | {delta} | "
                         f"{pct(b['format_ok'])} | {pct(f['format_ok'])} |")
    if "financebench" in ft and "judge_accuracy" in ft["financebench"]:
        lines += ["", f"FinanceBench LLM-judge accuracy: base {pct(base['financebench'].get('judge_accuracy'))}, "
                      f"fine-tuned {pct(ft['financebench'].get('judge_accuracy'))}."]
    for name, key in (("tatqa", "by_answer_type"), ("financebench", "by_question_type")):
        if name in ft and key in ft[name]:
            lines += ["", f"**{name} by {key[3:].replace('_', ' ')}**", "", "| Type | Base | Fine-tuned |", "|---|---|---|"]
            for k, v in ft[name][key].items():
                lines.append(f"| {k} | {pct(base[name].get(key, {}).get(k))} | {pct(v)} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "RESULTS.md").write_text(text)
    print(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="finetuned", help="name for this run, e.g. base or finetuned")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter; omit for the base model")
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--sets", default="finqa,tatqa,financebench")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N of each set")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--report", action="store_true", help="only build results/RESULTS.md from saved metrics")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.report:
        report(out_dir)
        return
    run(args.tag, args.adapter, args.model_id, Path(args.data_dir), out_dir, args.sets.split(","),
        args.limit, args.batch_size, args.max_new_tokens)


if __name__ == "__main__":
    main()
