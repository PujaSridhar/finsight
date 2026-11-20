"""Eval gate for CI: catch changes that silently move the numbers or break the model.

    python -m finsight.ci_eval --rescore                         # CPU, always runs in CI
    python -m finsight.ci_eval --endpoint http://host:8000       # small eval against a served model
    python -m finsight.ci_eval --endpoint http://host:8000 --write-baseline   # record the promoted model

Exit code 1 means the gate failed; the reasons are printed. Thresholds live in
configs/eval_gate.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import tasks

# (task, results directory) pairs whose saved predictions and metrics are re-scored.
RESULT_DIRS = (("earnings", "results/compare"), ("qa", "results"), ("qa", "results/compare_qa"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def load_gate(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def rescore(task, results_dir: Path, tolerance: float) -> list[str]:
    """Re-score saved predictions with today's scoring code and compare with the saved metrics files."""
    failures = []
    for metrics_path in sorted(results_dir.glob("metrics_*.json")):
        tag = metrics_path.stem[len("metrics_"):]
        saved = json.loads(metrics_path.read_text())
        for name in task.test_sets:
            preds = read_jsonl(results_dir / "predictions" / f"{tag}_{name}.jsonl")
            if not preds or name not in saved:
                continue
            new = task.aggregate([task.score(p["response"], p) for p in preds])["accuracy"]
            old = saved[name]["accuracy"]
            status = "ok" if abs(new - old) <= tolerance else "CHANGED"
            print(f"rescore {results_dir}/{tag}/{name}: saved {old:.4f}, now {new:.4f} [{status}]")
            if status != "ok":
                failures.append(f"{results_dir}/{tag}/{name} accuracy moved from {old:.4f} to {new:.4f} under the "
                                "current scoring code; regenerate the metrics and update the README, or revert")
    return failures


def gate_sample(task, data_dir: Path, n: int) -> list[dict]:
    """A fixed sample: the first n inputs of each test set (shuffled once, with a fixed seed, at build time)."""
    return [r for name in task.test_sets for r in read_jsonl(data_dir / task.test_sets[name])[:n]]


def smoke_eval(task, complete, examples: list[dict]) -> dict:
    """complete(messages) -> response text. Returns metrics and per-input correctness."""
    scores, per_input = [], {}
    for ex in examples:
        try:
            resp = complete(task.messages(ex))
        except Exception as e:  # count an endpoint failure as a wrong answer, and say why
            print(f"  {ex['id']}: endpoint error {type(e).__name__}")
            resp = ""
        s = task.score(resp, ex)
        scores.append(s)
        per_input[ex["id"]] = s["correct"]
    return {"metrics": task.aggregate(scores), "per_input": per_input}


def check(result: dict, baseline: dict | None, gate: dict) -> list[str]:
    m, failures = result["metrics"], []
    line = f"gate: accuracy {m['accuracy']:.3f}, valid output {m['format_ok']:.3f} (n={m['n']})"
    if baseline:
        line += f"; promoted model {baseline['metrics']['accuracy']:.3f}"
    print(line)
    if m["format_ok"] < gate["min_format_ok"]:
        failures.append(f"only {m['format_ok']:.0%} of outputs parsed (gate {gate['min_format_ok']:.0%})")
    if baseline:
        drop = baseline["metrics"]["accuracy"] - m["accuracy"]
        if drop > gate["max_accuracy_drop"]:
            failures.append(f"accuracy fell {drop:.1%} below the promoted model "
                            f"(gate allows {gate['max_accuracy_drop']:.0%})")
        regressed = [q for q, ok in baseline.get("per_input", {}).items() if ok and result["per_input"].get(q) is False]
        if regressed:
            print(f"inputs the promoted model got right and this one did not: {', '.join(regressed[:10])}")
    else:
        print("no baseline recorded yet: reporting only (record one with --write-baseline)")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", default="configs/eval_gate.yaml")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--endpoint", default=None, help="OpenAI-compatible server, e.g. vLLM from scripts/serve_vllm.sh")
    ap.add_argument("--served-model", default="finsight")
    ap.add_argument("--data-dir", default="data/earnings")
    ap.add_argument("--baseline", default="results/ci_baseline.json")
    ap.add_argument("--sample", default="results/ci_sample.jsonl",
                    help="the fixed releases the gate uses; written with the baseline, then committed")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()
    if not (args.rescore or args.endpoint):
        ap.error("give --rescore and/or --endpoint")
    gate = load_gate(Path(args.gate))
    failures: list[str] = []

    if args.rescore:
        for task_name, d in RESULT_DIRS:
            failures += rescore(tasks.get(task_name), Path(d), gate["rescore_tolerance"])
    if args.endpoint:
        from .serve import VLLMBackend

        task = tasks.get("earnings")
        backend = VLLMBackend(args.endpoint, args.served_model, api_key=os.environ.get("FINSIGHT_EVAL_TOKEN"))
        sample_path = Path(args.sample)
        if sample_path.exists() and not args.write_baseline:
            examples = read_jsonl(sample_path)  # the same releases the baseline was measured on
        else:
            examples = gate_sample(task, Path(args.data_dir), gate["n_per_set"])
        if not examples:
            sys.exit(f"no test data in {args.data_dir}; build it with python -m finsight.earnings")
        print(f"smoke eval: {len(examples)} releases against {args.endpoint} ({args.served_model})")
        result = smoke_eval(task, lambda msgs: backend.complete(msgs)["response"], examples)
        baseline_path = Path(args.baseline)
        if args.write_baseline:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(result, indent=2))
            sample_path.write_text("".join(json.dumps(r) + "\n" for r in examples))
            print(f"wrote {baseline_path} and {sample_path}; commit both")
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        failures += check(result, baseline, gate)

    if failures:
        print("\nEVAL GATE FAILED")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\neval gate passed")


if __name__ == "__main__":
    main()
