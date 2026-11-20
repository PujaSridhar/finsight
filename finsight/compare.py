"""Compare FinSight with the alternatives on accuracy, cost per release and latency.

Four systems read the same held-out earnings releases (from companies never
seen in training) with the same instruction:

  base        Mistral-7B-Instruct, zero-shot: what you get without training
  rag         Mistral-7B-Instruct + the 2 most similar training releases and their
              answers retrieved into the prompt (BM25, finsight/retrieval.py)
  finetuned   Mistral-7B-Instruct + the FinSight LoRA adapter, zero-shot
  hosted      a large hosted model through its API (default: Claude), zero-shot

    python -m finsight.compare --system base
    python -m finsight.compare --system rag
    python -m finsight.compare --system finetuned --adapter outputs/finsight-earnings/final
    ANTHROPIC_API_KEY=... python -m finsight.compare --system hosted    # optional, roughly $1 for 100 releases
    python -m finsight.compare --report --gpu-hourly-usd 0.35           # put your GPU's real price here

Local systems run one request at a time by default, so latency_s is what one
release takes on your GPU. Predictions are resumable and go to results/compare/.
--task qa runs the same comparison on the FinQA / TAT-QA warm-up task.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from . import tasks
from .earnings import FIELDS

LOCAL_SYSTEMS = ("base", "rag", "finetuned")
SYSTEMS = LOCAL_SYSTEMS + ("hosted",)
DEFAULT_HOSTED_MODEL = "claude-opus-5-5"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


# ---------------------------------------------------------------------------
# Backends: each takes a list of chats and returns {response, input_tokens, output_tokens, latency_s} per chat.

def local_backend(model_id: str, adapter: str | None, max_new_tokens: int, batch_size: int):
    from . import inference

    model, tokenizer = inference.load(model_id, adapter)

    def answer(conversations: list[list[dict]]) -> list[dict]:
        return inference.generate_with_stats(model, tokenizer, conversations, max_new_tokens, batch_size)

    return answer


def hosted_backend(model: str, max_tokens: int = 1024):
    """Claude via the Anthropic SDK. Reads ANTHROPIC_API_KEY from the environment; nothing else is sent."""
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the hosted baseline (it is optional; skip it otherwise).")
    client = anthropic.Anthropic()

    def answer(conversations: list[list[dict]]) -> list[dict]:
        out = []
        for messages in conversations:
            start = time.perf_counter()
            try:
                resp = client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                out.append({"response": "", "error": type(e).__name__, "latency_s": time.perf_counter() - start,
                            "input_tokens": 0, "output_tokens": 0})
                continue
            text = "" if resp.stop_reason == "refusal" else "".join(b.text for b in resp.content if b.type == "text")
            out.append({"response": text.strip(), "latency_s": time.perf_counter() - start,
                        "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                        "served_model": resp.model, "stop_reason": resp.stop_reason})
        return out

    return answer


# ---------------------------------------------------------------------------

def conversations_for(task, system: str, examples: list[dict], retriever=None, k: int = 2) -> list[list[dict]]:
    if system != "rag":
        return [task.messages(e) for e in examples]
    from .retrieval import build_fewshot_messages

    return [build_fewshot_messages(task, e, retriever.retrieve(e, k=k), task.demo_chars) for e in examples]


def run(task, system: str, loader, data_dir: Path, out_dir: Path, sets: list[str], limit: int | None,
        chunk_size: int = 8, retriever=None, k: int = 2, model_label: str = "") -> dict:
    """Answer every test input not answered yet, score it, and write resumable predictions.

    `loader()` returns a backend; it is called only if something is left to answer, so a finished run never
    loads the 7B model again.
    """
    answer = None
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name in sets:
        rows = read_jsonl(data_dir / task.test_sets[name])[:limit]
        pred_path = pred_dir / f"{system}_{name}.jsonl"
        done = {r["id"] for r in read_jsonl(pred_path)}
        todo = [r for r in rows if r["id"] not in done]
        print(f"[{system}] {name}: {len(done)} done, {len(todo)} to go")
        if todo and answer is None:
            answer = loader()
        with pred_path.open("a") as f:
            for i in range(0, len(todo), chunk_size):
                chunk = todo[i:i + chunk_size]
                for ex, res in zip(chunk, answer(conversations_for(task, system, chunk, retriever, k))):
                    rec = {key: ex.get(key) for key in task.record_keys}
                    rec.update(res, system=system, model=model_label)
                    rec.update(task.score(res["response"], ex))
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  {min(i + chunk_size, len(todo))}/{len(todo)}", end="\r")
    return summarize(task, system, out_dir)


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(task, system: str, out_dir: Path) -> dict:
    metrics, all_preds = {}, []
    for name in task.test_sets:
        preds = read_jsonl(out_dir / "predictions" / f"{system}_{name}.jsonl")
        if preds:
            metrics[name] = task.aggregate(preds)
            all_preds += preds
    lat = [p["latency_s"] for p in all_preds if p.get("latency_s") is not None and not p.get("error")]
    n = len(all_preds)
    metrics["_cost_latency"] = {
        "n": n,
        "errors": sum(1 for p in all_preds if p.get("error")),
        "mean_input_tokens": statistics.fmean(p.get("input_tokens", 0) for p in all_preds) if n else None,
        "mean_output_tokens": statistics.fmean(p.get("output_tokens", 0) for p in all_preds) if n else None,
        "mean_latency_s": statistics.fmean(lat) if lat else None,
        "p50_latency_s": _percentile(lat, 0.5),
        "p95_latency_s": _percentile(lat, 0.95),
        "model": next((p.get("model") for p in all_preds if p.get("model")), None),
    }
    (out_dir / f"metrics_{system}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def cost_per_query(system: str, cl: dict, pricing: dict, gpu_hourly_usd: float | None) -> float | None:
    if not cl.get("n"):
        return None
    if system == "hosted":
        price = (pricing.get("hosted") or {}).get(cl.get("model") or "")
        if not price:
            return None
        return (cl["mean_input_tokens"] * price["input_per_mtok"]
                + cl["mean_output_tokens"] * price["output_per_mtok"]) / 1e6
    if gpu_hourly_usd is None or cl.get("mean_latency_s") is None:
        return None
    return gpu_hourly_usd * cl["mean_latency_s"] / 3600


def load_pricing(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def report(task, out_dir: Path, pricing_path: Path = Path("configs/pricing.yaml"),
           gpu_hourly_usd: float | None = None) -> str:
    pricing = load_pricing(pricing_path)
    if gpu_hourly_usd is None:
        gpu_hourly_usd = pricing.get("gpu_hourly_usd")
    metrics = {s: json.loads((out_dir / f"metrics_{s}.json").read_text())
               for s in SYSTEMS if (out_dir / f"metrics_{s}.json").exists()}
    sets = [n for n in task.test_sets if any(n in m for m in metrics.values())]
    field_cols = list(FIELDS) if task.name == "earnings" else []
    head = ["System"] + [f"{n} accuracy" for n in sets] + field_cols + \
           ["Cost per 1K", "p50 latency", "p95 latency", "Tokens in / out"]
    lines = [f"# FinSight comparison ({task.name})", "",
             "Same held-out inputs and instruction for every system; greedy decoding for the local models. "
             "Accuracy is the share of releases with every field right"
             + (" (field columns: share right per field)" if field_cols else "") + ". Local latency is one "
             "request at a time; local cost is GPU price times GPU seconds, hosted cost is the provider's token "
             "price.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s, m in metrics.items():
        cl = m.get("_cost_latency", {})
        cpq = cost_per_query(s, cl, pricing, gpu_hourly_usd)
        cells = [f"{pct(m[n]['accuracy'])} (n={m[n]['n']})" if n in m else "–" for n in sets]
        first = m.get(sets[0], {}) if sets else {}
        cells += [pct(first.get(f)) for f in field_cols]
        cells += ["n/a" if cpq is None else f"${1000 * cpq:.2f}"]
        cells += [f"{cl[k]:.2f} s" if cl.get(k) is not None else "–" for k in ("p50_latency_s", "p95_latency_s")]
        cells += [f"{cl['mean_input_tokens']:.0f} / {cl['mean_output_tokens']:.0f}"
                  if cl.get("mean_input_tokens") is not None else "–"]
        label = s + (f" ({cl['model']})" if cl.get("model") else "")
        lines.append("| " + " | ".join([label] + cells) + " |")
    if field_cols:
        lines += ["", "**Where the errors come from** (count of wrong fields by cause)", "",
                  "| System | Errors |", "|---|---|"]
        for s, m in metrics.items():
            errs = m.get(sets[0], {}).get("errors", {}) if sets else {}
            lines.append(f"| {s} | " + (", ".join(f"{k} {v}" for k, v in errs.items()) or "none") + " |")
    lines += ["", f"GPU price used for local cost: "
                  f"{'not set' if gpu_hourly_usd is None else f'${gpu_hourly_usd}/hour'}. "
                  "With 100 test releases, accuracy carries roughly ±5 points of sampling error."]
    text = "\n".join(lines) + "\n"
    (out_dir / "COMPARISON.md").write_text(text)
    print(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["earnings", "qa"], default="earnings")
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (required for --system finetuned)")
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--hosted-model", default=DEFAULT_HOSTED_MODEL)
    ap.add_argument("--data-dir", default=None, help="defaults to the task's data directory")
    ap.add_argument("--out-dir", default=None, help="defaults to results/compare (earnings) or results/compare_qa")
    ap.add_argument("--sets", default=None, help="comma-separated test sets; defaults to all of the task's")
    ap.add_argument("--limit", type=int, default=100, help="inputs per test set")
    ap.add_argument("--k", type=int, default=2, help="retrieved examples for --system rag")
    ap.add_argument("--batch-size", type=int, default=1, help="1 measures real per-request latency")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="only rebuild the comparison table")
    ap.add_argument("--pricing", default="configs/pricing.yaml")
    ap.add_argument("--gpu-hourly-usd", type=float, default=None, help="overrides gpu_hourly_usd in the pricing file")
    args = ap.parse_args()
    task = tasks.get(args.task)
    data_dir = Path(args.data_dir or task.data_dir)
    out_dir = Path(args.out_dir or ("results/compare" if task.name == "earnings" else "results/compare_qa"))
    if args.report:
        report(task, out_dir, Path(args.pricing), args.gpu_hourly_usd)
        return
    if not args.system:
        ap.error("--system is required unless --report is given")
    if args.system == "finetuned" and not args.adapter:
        ap.error("--system finetuned needs --adapter")

    retriever = None
    if args.system == "rag":
        from .retrieval import ExampleRetriever

        retriever = ExampleRetriever(read_jsonl(data_dir / "train.jsonl"), task.retrieval_text)
    max_new = args.max_new_tokens or task.max_new_tokens
    if args.system == "hosted":
        label = args.hosted_model

        def loader():
            return hosted_backend(args.hosted_model)
    else:
        adapter = args.adapter if args.system == "finetuned" else None
        label = args.model_id + (" + FinSight adapter" if adapter else "")

        def loader():
            return local_backend(args.model_id, adapter, max_new, args.batch_size)
    sets = args.sets.split(",") if args.sets else list(task.test_sets)
    run(task, args.system, loader, data_dir, out_dir, sets, args.limit, chunk_size=max(args.batch_size, 8),
        retriever=retriever, k=args.k, model_label=label)
    report(task, out_dir, Path(args.pricing), args.gpu_hourly_usd)


if __name__ == "__main__":
    main()
