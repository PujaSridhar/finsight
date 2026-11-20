"""The FinSight task: read a quarterly earnings release, return its key GAAP figures as JSON.

    export SEC_USER_AGENT="Your Name your.email@example.com"
    python -m finsight.earnings --companies 5            # quick check, ~2 minutes
    python -m finsight.earnings                          # full build, ~150 companies, 30-60 minutes on CPU

Input: the press release a company files with the SEC on Form 8-K (Item 2.02,
usually Exhibit 99.1) when it reports a quarter.

Output (the schema every system is held to):

    {"period_end": "2024-09-28", "revenue": 94930000000, "net_income": 14736000000, "eps_diluted": 0.97}

Labels come for free: weeks later the company files the same quarter's
figures in its 10-Q or 10-K in machine-readable XBRL, which the SEC publishes
through its companyfacts API. For each release we take the XBRL figures for
the quarter it reports and keep the example only if those figures also appear
in the release text the model reads, which catches period mismatches,
restatements and releases that report something else.

Splits are by company: test companies never appear in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

FIELDS = ("period_end", "revenue", "net_income", "eps_diluted")
MONEY_FIELDS = ("revenue", "net_income")

INSTRUCTION = (
    "You extract figures from a company's quarterly earnings press release. Report the figures for the most "
    "recent three-month quarter the release covers, not the year-ago quarter and not year-to-date totals. "
    "Report GAAP figures, not adjusted or non-GAAP ones. Give amounts in US dollars as plain numbers "
    "(a table 'in millions' showing 1,234 means 1234000000). Use null for anything the release does not state.\n"
    "Reply with only a JSON object with these keys:\n"
    '  "period_end": last day of the quarter, YYYY-MM-DD\n'
    '  "revenue": total revenue (net sales) for the quarter\n'
    '  "net_income": GAAP net income (loss) attributable to the company; negative for a loss\n'
    '  "eps_diluted": GAAP diluted earnings (loss) per share'
)

# XBRL tags tried in order. Companies use different revenue tags; banks and insurers often use none of
# these, and are skipped rather than mislabeled.
XBRL_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}
QUARTER_DAYS = (80, 100)
RELEASE_LAG_DAYS = (5, 75)   # releases come 2-10 weeks after the quarter ends


# ---------------------------------------------------------------------------
# Prompt and output parsing

def build_messages(text: str, target: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n### Earnings release\n{text.strip()}"}]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def target_json(label: dict) -> str:
    return json.dumps({k: label.get(k) for k in FIELDS})


def parse_output(response: str) -> dict | None:
    """The first JSON object in the response (models sometimes wrap it in prose or ``` fences)."""
    if not response:
        return None
    start = response.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(response)):
            depth += {"{": 1, "}": -1}.get(response[i], 0)
            if depth == 0:
                try:
                    obj = json.loads(response[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    break
        start = response.find("{", start + 1)
    return None


def to_number(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    scale = 1.0
    for word, mult in (("billion", 1e9), ("million", 1e6), ("thousand", 1e3)):
        if s.lower().endswith(word):
            s, scale = s[: -len(word)].strip(), mult
    try:
        x = float(s) * scale
    except ValueError:
        return None
    return -x if neg else x


def to_date(v) -> str | None:
    if not isinstance(v, str):
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", v)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Scoring

def field_correct(field: str, pred, gold) -> bool:
    if gold is None:
        return pred is None
    if field == "period_end":
        return to_date(pred) == gold
    p = to_number(pred)
    if p is None:
        return False
    if field == "eps_diluted":
        return abs(p - gold) <= 0.005 + 1e-9
    return abs(p - gold) <= 0.005 * abs(gold)


def classify_error(field: str, pred, gold, distractors: dict, text: str = "") -> str:
    """Why a field is wrong, for the error analysis in the README."""
    if field_correct(field, pred, gold):
        return "correct"
    if pred is None:
        return "missing"
    if field == "period_end":
        return "wrong_period"
    p = to_number(pred)
    if p is None:
        return "unparseable"
    for kind, value in (distractors.get(field) or {}).items():
        if value is not None and field_correct(field, p, value):
            return kind  # "prior_year_quarter" or "year_to_date"
    if gold and p and any(abs(p / gold - r) < 0.01 * r for r in (1e3, 1e6, 1e-3, 1e-6)):
        return "scale"
    if gold and abs(p + gold) <= 0.005 * abs(gold) + 0.005:
        return "sign"
    if field == "eps_diluted" and text and _near_non_gaap(text, p):
        return "likely_non_gaap"
    return "other"


def _near_non_gaap(text: str, value: float) -> bool:
    s = f"{abs(value):.2f}"
    for m in re.finditer(re.escape(s), text):
        window = text[max(0, m.start() - 250): m.end() + 100].lower()
        if "adjusted" in window or "non-gaap" in window or "non gaap" in window:
            return True
    return False


def score(response: str, example: dict) -> dict:
    gold, out = example["label"], parse_output(response)
    result = {"valid_json": out is not None}
    pred = out or {}
    for f in FIELDS:
        result[f] = field_correct(f, pred.get(f), gold.get(f))
        result[f + "_error"] = classify_error(f, pred.get(f), gold.get(f), example.get("distractors", {}),
                                              example.get("text", ""))
    result["all_correct"] = all(result[f] for f in FIELDS)
    result["correct"] = result["all_correct"]
    return result


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if not n:
        return {"n": 0}
    out = {"n": n, "accuracy": sum(s["all_correct"] for s in scores) / n,
           "valid_json": sum(s["valid_json"] for s in scores) / n}
    out["format_ok"] = out["valid_json"]
    for f in FIELDS:
        out[f] = sum(s[f] for s in scores) / n
    errors = Counter(s[f + "_error"] for s in scores for f in FIELDS if s[f + "_error"] != "correct")
    out["errors"] = dict(errors.most_common())
    return out


# ---------------------------------------------------------------------------
# Labels from XBRL

def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def tag_facts(companyfacts: dict, field: str) -> list[tuple[str, list[dict]]]:
    """[(tag, facts with a start date)] for the field's XBRL tags, in priority order."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    unit = "USD/shares" if field == "eps_diluted" else "USD"
    out = []
    for tag in XBRL_TAGS[field]:
        facts = [f for f in gaap.get(tag, {}).get("units", {}).get(unit, []) if f.get("start") and f.get("end")]
        if facts:
            out.append((tag, facts))
    return out


def is_quarter(f: dict) -> bool:
    return QUARTER_DAYS[0] <= _days(f["start"], f["end"]) <= QUARTER_DAYS[1]


def _first_filed(facts: list[dict], filed_after: str = "") -> dict | None:
    """The fact as first reported on or after the release (the original 10-Q/10-K, before any restatement)."""
    cands = [f for f in facts if f.get("filed", "") >= filed_after]
    return min(cands, key=lambda f: f["filed"]) if cands else None


def _shift_year(d: str, years: int) -> date:
    x = date.fromisoformat(d)
    try:
        return x.replace(year=x.year + years)
    except ValueError:  # Feb 29
        return x.replace(year=x.year + years, day=28)


def label_for_release(companyfacts: dict, filed: str) -> tuple[dict | None, dict, str]:
    """XBRL label for the quarter an earnings release (filed on `filed`) reports.

    Returns (label or None, distractors, reason). Companies switch revenue tags over the years, so for each
    field the highest-priority tag that has a fact for that quarter wins. Distractors are the same figures for
    the year-ago quarter and the year-to-date period, which releases print right next to the right ones; they
    let the evaluation say *how* a model went wrong.
    """
    filed_d = date.fromisoformat(filed)
    ends = {f["end"] for _, facts in tag_facts(companyfacts, "revenue") for f in facts
            if is_quarter(f) and RELEASE_LAG_DAYS[0] <= (filed_d - date.fromisoformat(f["end"])).days
            <= RELEASE_LAG_DAYS[1]}
    if not ends:
        return None, {}, "no_quarter"
    end = max(ends)
    label, distractors = {"period_end": end}, {}
    prior_end = _shift_year(end, -1)
    for field in ("revenue", "net_income", "eps_diluted"):
        for _, facts in tag_facts(companyfacts, field):
            q = [f for f in facts if is_quarter(f)]
            fact = _first_filed([f for f in q if f["end"] == end], filed)
            if fact is None:
                continue
            prior = _first_filed([f for f in q if abs((date.fromisoformat(f["end"]) - prior_end).days) <= 7])
            ytd = _first_filed([f for f in facts if f["end"] == end and f["start"] < fact["start"]
                                and not is_quarter(f)])
            label[field] = fact["val"]
            distractors[field] = {"prior_year_quarter": prior["val"] if prior else None,
                                  "year_to_date": ytd["val"] if ytd else None}
            break
        else:
            return None, {}, f"no_{field}"
    return label, distractors, "ok"


def number_forms(value: float, field: str) -> list[str]:
    """How a figure is printed in a release's tables: 94,930 (millions), 94,930,000 (thousands), 0.97 (EPS)."""
    v = abs(value)
    if field == "eps_diluted":
        return [f"{v:.2f}"]
    forms = []
    for scale in (1e6, 1e3, 1):
        x = v / scale
        if x >= 1 and abs(x - round(x)) < 1e-6:
            forms.append(f"{round(x):,}")
    return forms  # rounded headline forms ("$1.2 billion") are too loose to verify a label with


def verify(label: dict, text: str) -> list[str]:
    """Fields of the label whose value cannot be found in the release text (empty list = verified)."""
    missing = []
    for field in ("revenue", "net_income", "eps_diluted"):
        forms = number_forms(label[field], field)
        if not any(re.search(r"(?<![\d.,])" + re.escape(f) + r"(?![\d,]|\.\d)", text) for f in forms):
            missing.append(field)
    return missing


# ---------------------------------------------------------------------------
# Dataset build

def company_split(cik: int, val_frac: float = 0.05, test_frac: float = 0.15) -> str:
    h = int(hashlib.sha1(str(cik).encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if h < test_frac else ("val" if h < test_frac + val_frac else "train")


def build_records_for_company(client, company: dict, since: str, max_chars: int) -> tuple[list[dict], Counter]:
    from .edgar import release_excerpt

    reasons: Counter = Counter()
    cik = company["cik"]
    try:
        facts = client.companyfacts(cik)
        filings = client.earnings_8ks(cik, since)
    except Exception as e:  # a company with no XBRL facts returns 404
        reasons[f"company_{type(e).__name__}"] += 1
        return [], reasons
    records = []
    for filing in filings:
        label, distractors, why = label_for_release(facts, filing["filed"])
        if label is None:
            reasons[why] += 1
            continue
        try:
            url = client.release_url(cik, filing["accession"])
            if not url:
                reasons["no_exhibit"] += 1
                continue
            text = release_excerpt(client.release_text(url), max_chars)
        except Exception as e:
            reasons[f"download_{type(e).__name__}"] += 1
            continue
        missing = verify(label, text)
        if missing:
            reasons["unverified_" + "+".join(missing)] += 1
            continue
        reasons["kept"] += 1
        records.append({"id": f"{company['ticker']}-{filing['accession']}", "source": "earnings",
                        "cik": cik, "ticker": company["ticker"], "company": company["title"],
                        "filed": filing["filed"], "url": url, "text": text, "label": label,
                        "distractors": distractors, "target": target_json(label)})
    return records, reasons


def build(out_dir: Path, raw_dir: Path, n_companies: int, since: str, max_chars: int, seed: int = 42,
          n_test: int = 300) -> dict:
    from .edgar import SECClient

    client = SECClient(raw_dir / "sec")
    companies = client.companies()[:n_companies]
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    reasons: Counter = Counter()
    for i, c in enumerate(companies, 1):
        recs, why = build_records_for_company(client, c, since, max_chars)
        reasons.update(why)
        splits[company_split(c["cik"])].extend(recs)
        print(f"[{i}/{len(companies)}] {c['ticker']}: {len(recs)} releases kept "
              f"(running total {sum(map(len, splits.values()))})")
    rng = random.Random(seed)
    for rows in splits.values():
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)
    splits["test"] = splits["test"][:n_test]
    assert not ({r["cik"] for r in splits["train"]} & {r["cik"] for r in splits["test"]}), "company leak"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"train": "train.jsonl", "val": "val.jsonl", "test": "test_earnings.jsonl"}
    for split, rows in splits.items():
        (out_dir / names[split]).write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {"companies": len(companies), "since": since, "max_chars": max_chars,
             "examples": {s: len(r) for s, r in splits.items()},
             "companies_per_split": {s: len({r["cik"] for r in rows}) for s, rows in splits.items()},
             "outcomes": dict(reasons.most_common())}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    from .edgar import EXCERPT_CHARS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/earnings")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--companies", type=int, default=150, help="first N companies in SEC's ticker file")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--max-chars", type=int, default=EXCERPT_CHARS, help="release excerpt length the model reads")
    ap.add_argument("--n-test", type=int, default=300)
    args = ap.parse_args()
    build(Path(args.out), Path(args.raw), args.companies, args.since, args.max_chars, n_test=args.n_test)


if __name__ == "__main__":
    main()
