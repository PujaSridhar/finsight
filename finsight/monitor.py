"""Summarize the API's request log by day: volume, errors, latency, answer format, input size.

    python -m finsight.monitor logs/requests.jsonl

The signals, and why each one matters, are described in docs/PRODUCTION.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(records: list[dict]) -> list[dict]:
    days: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        days[r["ts"][:10]].append(r)
    rows = []
    for day, rs in sorted(days.items()):
        ok = [r for r in rs if not r.get("error")]
        lat = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
        rows.append({
            "day": day,
            "requests": len(rs),
            "error_rate": round(1 - len(ok) / len(rs), 4),
            "p50_ms": _pct(lat, 0.5),
            "p95_ms": _pct(lat, 0.95),
            "format_ok": round(sum(bool(r.get("format_ok")) for r in ok) / len(ok), 4) if ok else None,
            "median_context_chars": _pct([r["context_chars"] for r in rs], 0.5),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    args = ap.parse_args()
    records = [json.loads(l) for l in args.log.read_text().splitlines() if l.strip()]
    cols = ["day", "requests", "error_rate", "p50_ms", "p95_ms", "format_ok", "median_context_chars"]
    print(" | ".join(cols))
    for row in summarize(records):
        print(" | ".join("" if row[c] is None else str(row[c]) for c in cols))


if __name__ == "__main__":
    main()
