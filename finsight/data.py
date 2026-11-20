"""Phase 1 (v2): build the FinSight instruction dataset.

Why v2: the first notebook trained on FinanceBench, which is a 150-question
*evaluation benchmark*, and its EDGAR step returned no passages, so it ended
with 68 training examples. v2 instead:

* trains on FinQA (Chen et al., 2021) and TAT-QA (Zhu et al., 2021): ~19K
  human-annotated questions over real 10-K tables and text;
* builds step-by-step targets from the human-written FinQA programs and
  TAT-QA derivations, and keeps an example only if re-executing that
  calculation reproduces the gold answer ("verified labels");
* splits TAT-QA by document, so no filing appears in both train and val;
* keeps FinanceBench completely out of training as an out-of-domain test set.

Run (CPU only, ~1 minute plus downloads):

    python -m finsight.data --out data/processed

Outputs: train.jsonl, val.jsonl, test_finqa.jsonl, test_tatqa.jsonl,
test_financebench.jsonl and stats.json. Every record has
{"id", "source", "context", "question", "response", "answer"}; "response" is
the training target and "answer" the gold string used for scoring.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import numbers_close, parse_number
from .prompts import build_context

SOURCES = {
    # Original releases on GitHub (stable, no Hugging Face account needed).
    "finqa_train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "finqa_dev": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "finqa_test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    "tatqa_train": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json",
    "tatqa_dev": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json",
    "financebench": "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl",
}

MAX_CONTEXT_CHARS = 6500  # ~1.7K Mistral tokens; keeps prompt+answer under 2048 tokens


# ───────────────────────────── download ─────────────────────────────

def download(name: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = SOURCES[name]
    path = raw_dir / (f"{name}.jsonl" if url.endswith(".jsonl") else f"{name}.json")
    if not path.exists():
        print(f"  downloading {name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "finsight-data-builder"})
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
    return path


def load_json(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())


# ───────────────────────────── formatting helpers ─────────────────────────────

def fmt_num(x: float) -> str:
    """Readable number: up to 5 significant decimals, no trailing zeros."""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return f"{int(round(x)):,}" if abs(x) >= 1000 else str(int(round(x)))
    s = f"{x:.5f}".rstrip("0").rstrip(".")
    return s


def fmt_answer(x: float) -> str:
    """Final-answer display: 2 decimals, or 4 for values below 1."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".") if abs(x) >= 1 else f"{x:.4f}".rstrip("0").rstrip(".")


def agrees_when_rounded(shown: str, shown_v: float, gold: float, allow_percent: bool = True) -> bool:
    """True if `shown` equals `gold` rounded to the decimals `shown` displays
    (or gold x 100, for percentages stored as decimals)."""
    m = re.search(r"\.(\d+)", shown)
    decimals = len(m.group(1)) if m else 0
    tol = 0.5 * 10 ** -decimals + 1e-9
    targets = [gold, gold * 100] if allow_percent else [gold]
    return any(abs(shown_v - t) <= tol for t in targets)


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    # FinQA's text was tokenized; undo the most visible artifacts.
    s = re.sub(r" ([,.;:%)])", r"\1", s).replace("( ", "(").replace("$ ", "$")
    # Mangled unicode in the FinQA release: "management 2019s" -> "management's", 201c/201d -> quotes.
    s = re.sub(r"(?<=[a-z0-9)]) 2019s\b", "'s", s)
    s = re.sub(r"\b201c(?=\w)", '"', s)
    s = re.sub(r"\s?\b201d(?=[\s),.;:]|$)", '"', s)
    return s


# ───────────────────────────── FinQA ─────────────────────────────

FINQA_BINARY = {
    "add": (operator.add, "+"),
    "subtract": (operator.sub, "-"),
    "multiply": (operator.mul, "×"),
    "divide": (operator.truediv, "/"),
    "exp": (operator.pow, "^"),
}
TABLE_OPS = {"table_max": max, "table_min": min, "table_sum": sum,
             "table_average": lambda xs: sum(xs) / len(xs)}
STEP_RE = re.compile(r"(\w+)\(([^)]*)\)")


def _finqa_arg(arg: str, results: list[float]) -> float:
    arg = arg.strip()
    if arg.startswith("#"):
        return results[int(arg[1:])]
    if arg.startswith("const_"):
        c = arg[len("const_"):]
        return -1.0 if c == "m1" else float(c)
    v = parse_number(arg.replace("%", ""))
    if v is None:
        raise ValueError(f"bad arg {arg!r}")
    return v / 100 if arg.endswith("%") else v


def _row_numbers(table: list[list[str]], row_name: str) -> list[float]:
    for row in table:
        if row and row[0].strip().lower() == row_name.strip().lower():
            nums = [parse_number(c) for c in row[1:]]
            nums = [n for n in nums if n is not None]
            if nums:
                return nums
    raise ValueError(f"row {row_name!r} not found")


def execute_finqa_program(program: str, table: list[list[str]]) -> tuple[float | bool, list[str]]:
    """Execute a FinQA DSL program and return (result, human-readable steps)."""
    results: list = []
    steps: list[str] = []
    for i, (op, args) in enumerate(STEP_RE.findall(program)):
        parts = [a.strip() for a in args.split(",")]
        if op in FINQA_BINARY:
            fn, sym = FINQA_BINARY[op]
            a, b = (_finqa_arg(p, results) for p in parts[:2])
            res = fn(a, b)
            steps.append(f"Step {i + 1}: {fmt_num(a)} {sym} {fmt_num(b)} = {fmt_num(res)}")
        elif op == "greater":
            a, b = (_finqa_arg(p, results) for p in parts[:2])
            res = a > b
            steps.append(f"Step {i + 1}: Is {fmt_num(a)} greater than {fmt_num(b)}? {'yes' if res else 'no'}")
        elif op in TABLE_OPS:
            nums = _row_numbers(table, parts[0])
            res = TABLE_OPS[op](nums)
            label = op.replace("table_", "")
            steps.append(f"Step {i + 1}: {label} of '{parts[0]}' ({', '.join(fmt_num(n) for n in nums)}) = {fmt_num(res)}")
        else:
            raise ValueError(f"unknown op {op}")
        results.append(res)
    if not results:
        raise ValueError("empty program")
    return results[-1], steps


def convert_finqa(rows: list[dict], split: str) -> tuple[list[dict], Counter]:
    out, drops = [], Counter()
    for r in rows:
        qa = r["qa"]
        gold = qa.get("exe_ans")
        context = build_context(
            clean_text(" ".join(r["pre_text"])), r.get("table_ori") or r["table"], clean_text(" ".join(r["post_text"]))
        )
        if len(context) > MAX_CONTEXT_CHARS:
            drops["context_too_long"] += 1
            continue
        try:
            value, steps = execute_finqa_program(qa["program"], r["table"])
        except Exception:
            drops["program_failed"] += 1
            continue

        # Verify: re-executed program must reproduce the dataset's executed answer.
        if isinstance(gold, str):  # yes/no questions
            if gold not in ("yes", "no") or (("yes" if value else "no") != gold):
                drops["unverified"] += 1
                continue
            answer = gold
        else:
            if isinstance(value, bool) or not math.isclose(value, float(gold), rel_tol=1e-3, abs_tol=1e-4):
                drops["unverified"] += 1
                continue
            g = float(gold)
            shown = str(qa.get("answer", "")).strip()
            shown_v = parse_number(shown)
            if shown and shown_v is not None and agrees_when_rounded(shown, shown_v, g):
                # The annotator's display string (e.g. "14.1%") matches the
                # executed value up to its own rounding, so use it.
                answer = shown
                if "%" in shown and not agrees_when_rounded(shown, shown_v, g, allow_percent=False):
                    steps.append(f"As a percentage: {fmt_num(g)} × 100 = {shown}")
            elif ("percent" in qa["question"].lower() and abs(g) < 10
                  and STEP_RE.findall(qa["program"])[-1][0] == "divide"):
                answer = f"{fmt_answer(g * 100)}%"
                steps.append(f"As a percentage: {fmt_num(g)} × 100 = {answer}")
            else:
                answer = fmt_answer(g)
        out.append({
            "id": f"finqa-{split}-{r['id']}",
            "source": "finqa",
            "context": context,
            "question": clean_text(qa["question"]),
            "response": "\n".join(steps + [f"Answer: {answer}"]),
            "answer": answer,
        })
    return out, drops


# ───────────────────────────── TAT-QA ─────────────────────────────

_ALLOWED_AST = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult,
                ast.Div, ast.Pow, ast.USub, ast.UAdd)


def safe_eval(expr: str) -> float:
    expr = expr.replace("[", "(").replace("]", ")").replace(",", "").replace("$", "").replace("%", "")
    expr = expr.replace("×", "*")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise ValueError("unsafe expression")
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}))


def with_scale(ans: str, scale: str) -> str:
    if scale == "percent":
        return ans if ans.endswith("%") else f"{ans}%"
    if scale in ("thousand", "million", "billion"):
        return f"{ans} {scale}"
    return ans


def convert_tatqa(docs: list[dict], split: str) -> tuple[list[dict], Counter]:
    out, drops = [], Counter()
    for d in docs:
        paragraphs = [p["text"] for p in sorted(d["paragraphs"], key=lambda p: p["order"])]
        context = build_context("", d["table"]["table"], "\n".join(paragraphs))
        if len(context) > MAX_CONTEXT_CHARS:
            drops["context_too_long"] += len(d["questions"])
            continue
        for q in d["questions"]:
            atype, ans, deriv, scale = q["answer_type"], q["answer"], q.get("derivation", ""), q.get("scale", "")
            lines = []
            if atype == "arithmetic":
                try:
                    v = safe_eval(deriv)
                    g = float(ans)
                except Exception:
                    drops["unverified"] += 1
                    continue
                if not numbers_close(v, g, rel_tol=0.01):
                    drops["unverified"] += 1
                    continue
                times_100 = " × 100" if not numbers_close(v, g, rel_tol=0.01, allow_percent=False) else ""
                lines.append(f"Calculation: {deriv}{times_100} = {fmt_num(g)}")
                answer = with_scale(fmt_num(g), scale)
            elif atype == "count":
                items = [s.strip() for s in deriv.split("##") if s.strip()]
                if items and str(len(items)) == str(ans).strip():
                    lines.append("Items counted: " + "; ".join(items))
                answer = str(ans).strip()
            elif atype in ("span", "multi-span"):
                spans = ans if isinstance(ans, list) else [ans]
                spans = [str(s).strip() for s in spans if str(s).strip()]
                if not spans:
                    drops["empty"] += 1
                    continue
                answer = "; ".join(with_scale(s, scale) if parse_number(s) is not None and len(s.split()) == 1 else s
                                   for s in spans)
            else:
                drops["unknown_type"] += 1
                continue
            out.append({
                "id": f"tatqa-{split}-{q['uid']}",
                "source": "tatqa",
                "doc_id": d["table"]["uid"],
                "context": context,
                "question": q["question"].strip(),
                "response": "\n".join(lines + [f"Answer: {answer}"]),
                "answer": answer,
                "answer_type": atype,
            })
    return out, drops


# ───────────────────────────── FinanceBench (test only) ─────────────────────────────

def convert_financebench(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        ev = r.get("evidence") or []
        text = "\n\n".join(e.get("evidence_text", "") for e in ev if isinstance(e, dict)).strip()
        context = f"Source: {r.get('doc_name', '')}\n\n{text[:MAX_CONTEXT_CHARS]}"
        out.append({
            "id": r["financebench_id"],
            "source": "financebench",
            "context": context,
            "question": r["question"].strip(),
            "response": None,
            "answer": str(r["answer"]).strip(),
            "question_type": r.get("question_type"),
            "company": r.get("company"),
        })
    return out


# ───────────────────────────── build ─────────────────────────────

def dedupe(examples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for ex in examples:
        key = (ex["question"].lower(), ex["context"][:500])
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(out_dir: Path, raw_dir: Path, seed: int = 42, max_train: int | None = None,
          tatqa_val_frac: float = 0.05, n_val: int = 400, n_test: int = 300) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading sources (cached after the first run)...")
    raw = {k: load_json(download(k, raw_dir)) for k in SOURCES}

    print("Converting FinQA...")
    fq_train, d1 = convert_finqa(raw["finqa_train"], "train")
    fq_dev, d2 = convert_finqa(raw["finqa_dev"], "dev")
    fq_test, d3 = convert_finqa(raw["finqa_test"], "test")
    print("Converting TAT-QA...")
    tat_train_all, d4 = convert_tatqa(raw["tatqa_train"], "train")
    tat_dev, d5 = convert_tatqa(raw["tatqa_dev"], "dev")

    # Document-level split of TAT-QA train into train/val (no filing in both).
    doc_ids = sorted({ex["doc_id"] for ex in tat_train_all})
    rng.shuffle(doc_ids)
    val_docs = set(doc_ids[: int(len(doc_ids) * tatqa_val_frac)])
    tat_train = [e for e in tat_train_all if e["doc_id"] not in val_docs]
    tat_val = [e for e in tat_train_all if e["doc_id"] in val_docs]

    train = dedupe(fq_train + tat_train)
    rng.shuffle(train)
    if max_train:
        train = train[:max_train]

    val = rng.sample(fq_dev, min(n_val // 2, len(fq_dev))) + rng.sample(tat_val, min(n_val // 2, len(tat_val)))
    rng.shuffle(val)
    test_finqa = rng.sample(fq_test, min(n_test, len(fq_test)))
    test_tatqa = rng.sample(tat_dev, min(n_test, len(tat_dev)))
    test_fb = convert_financebench(raw["financebench"])

    # Leakage check: no test question/context pair may appear in train.
    train_keys = {(e["question"].lower(), e["context"][:500]) for e in train}
    for name, test in [("finqa", test_finqa), ("tatqa", test_tatqa), ("financebench", test_fb)]:
        overlap = sum((e["question"].lower(), e["context"][:500]) in train_keys for e in test)
        assert overlap == 0, f"{overlap} {name} test examples leak into train"

    splits = {"train": train, "val": val, "test_finqa": test_finqa, "test_tatqa": test_tatqa,
              "test_financebench": test_fb}
    for name, rows in splits.items():
        write_jsonl(rows, out_dir / f"{name}.jsonl")

    stats = {
        "seed": seed,
        "sizes": {k: len(v) for k, v in splits.items()},
        "train_by_source": dict(Counter(e["source"] for e in train)),
        "train_tatqa_by_answer_type": dict(Counter(e.get("answer_type") for e in train if e["source"] == "tatqa")),
        "avg_context_chars_train": round(sum(len(e["context"]) for e in train) / len(train)),
        "avg_response_chars_train": round(sum(len(e["response"]) for e in train) / len(train)),
        "dropped": {
            "finqa_train": dict(d1), "finqa_dev": dict(d2), "finqa_test": dict(d3),
            "tatqa_train": dict(d4), "tatqa_dev": dict(d5),
        },
        "source_counts_before_filtering": {
            "finqa_train": len(raw["finqa_train"]), "finqa_test": len(raw["finqa_test"]),
            "tatqa_train_questions": sum(len(d["questions"]) for d in raw["tatqa_train"]),
            "financebench": len(raw["financebench"]),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None, help="cap the training set size")
    ap.add_argument("--n-test", type=int, default=300, help="examples sampled per FinQA/TAT-QA test set")
    args = ap.parse_args()
    stats = build(Path(args.out), Path(args.raw), seed=args.seed, max_train=args.max_train, n_test=args.n_test)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
