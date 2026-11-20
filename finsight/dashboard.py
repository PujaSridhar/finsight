"""Build the public FinSight live page from the live/ log: one static HTML file plus the numbers as JSON.

    python -m finsight.dashboard --live live --out site
    python -m http.server -d site 8000      # look at it locally

Every number on the page is computed here from releases.jsonl and scores.jsonl; there are no hand-entered
results. The page is plain HTML with inline SVG charts and no scripts, so GitHub Pages can serve it as is.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from .earnings import FIELDS
from .live import Store

FIELD_NAMES = {"revenue": "Revenue", "net_income": "Net income", "eps_diluted": "Diluted EPS",
               "period_end": "Period end"}


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def week_of(d: str) -> str:
    y, w, _ = date.fromisoformat(d).isocalendar()
    return f"{y}-W{w:02d}"


def compute(releases: list[dict], scores: list[dict], now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    by_acc = {s["accession"]: s for s in scores}
    answered = [r for r in releases if not r.get("error")]
    scored = [by_acc[r["accession"]] | {"filed": r["filed"]} for r in answered
              if by_acc.get(r["accession"], {}).get("status") == "scored"]
    verified = [s for s in scored if s.get("verified_in_text")]
    live = [r for r in answered if r.get("mode") == "live"]

    def acc_block(rows):
        n = len(rows)
        if not n:
            return {"n": 0}
        out = {"n": n, "all_correct": sum(s["all_correct"] for s in rows) / n}
        for f in FIELDS:
            out[f] = sum(s[f] for s in rows) / n
        return out

    weeks = defaultdict(lambda: {"answered": 0, "verified": []})
    for r in answered:
        weeks[week_of(r["filed"])]["answered"] += 1
    for s in verified:
        weeks[week_of(s["filed"])]["verified"].append(s)
    weekly = [{"week": w, "answered": v["answered"], "scored": len(v["verified"]),
               "all_correct": (sum(s["all_correct"] for s in v["verified"]) / len(v["verified"]))
               if v["verified"] else None} for w, v in sorted(weeks.items())]

    errors = defaultdict(int)
    for s in verified:
        for f in FIELDS:
            if s.get(f + "_error", "correct") != "correct":
                errors[s[f + "_error"]] += 1

    model_ms = [r.get("model_ms") for r in answered]
    recent = []
    for r in sorted(answered, key=lambda r: r["answered_at"], reverse=True)[:25]:
        s = by_acc.get(r["accession"])
        status = ("right" if s["all_correct"] else "wrong: " + ", ".join(FIELD_NAMES[f] for f in FIELDS if not s[f])) \
            if s and s.get("status") == "scored" else ("can't be scored" if s else "waiting for the 10-Q")
        recent.append({"company": r["company"], "accepted_at": r.get("accepted_at") or r["filed"],
                       "fields": r.get("fields"), "status": status, "model_ms": r.get("model_ms"),
                       "url": r.get("url")})
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "counts": {"answered": len(answered), "live": len(live),
                   "no_release_document": sum(r.get("error") == "no_release_document" for r in releases),
                   "scored": len(scored), "verified": len(verified),
                   "unscorable": sum(s.get("status") == "unscorable" for s in scores),
                   "waiting": len(answered) - len(scored) - sum(by_acc.get(r["accession"], {}).get("status")
                                                               == "unscorable" for r in answered),
                   "valid_json": (sum(bool(r.get("valid_json")) for r in answered) / len(answered))
                   if answered else None},
        "accuracy_verified": acc_block(verified),
        "accuracy_all_scored": acc_block(scored),
        "latency": {"model_ms_p50": pct(model_ms, 0.5), "model_ms_p95": pct(model_ms, 0.95),
                    "detect_s_p50": pct([r.get("detect_lag_s") for r in live], 0.5),
                    "detect_s_p95": pct([r.get("detect_lag_s") for r in live], 0.95),
                    "end_to_end_s_p50": pct([r.get("end_to_end_s") for r in live], 0.5),
                    "end_to_end_s_p95": pct([r.get("end_to_end_s") for r in live], 0.95)},
        "errors": dict(sorted(errors.items(), key=lambda kv: -kv[1])),
        "weekly": weekly,
        "recent": recent,
        "models": sorted({r.get("model") for r in answered if r.get("model")}),
    }


# ---------------------------------------------------------------------------
# HTML

CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6a66; --line:#e3e2de; --card:#ffffff; --accent:#2f6fde;
        --good:#1f8a4c; --bad:#c2410c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141413; --fg:#ecebe7; --muted:#a3a29d; --line:#2e2d2a; --card:#1d1c1a; --accent:#7aa7ff;
          --good:#5ccf8a; --bad:#ff8a5c; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 36px 0 12px; }
p { color: var(--muted); margin: 6px 0; } a { color: var(--accent); }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.card .v { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.card .k { color: var(--muted); font-size: 13px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 14px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; }
.right { color: var(--good); } .wrong { color: var(--bad); }
svg text { fill: var(--muted); font-size: 11px; }
"""


def _fmt_money(v) -> str:
    if not isinstance(v, (int, float)):
        return "–"
    a = abs(v)
    s = f"{a / 1e9:.2f}B" if a >= 1e9 else (f"{a / 1e6:.1f}M" if a >= 1e6 else f"{a:,.0f}")
    return ("-" if v < 0 else "") + "$" + s


def _fmt_pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _fmt_s(x, unit="s") -> str:
    if x is None:
        return "–"
    return f"{x / 1000:.2f} s" if unit == "ms" else (f"{x:.0f} s" if x < 120 else f"{x / 60:.1f} min")


def _weekly_svg(weekly: list[dict]) -> str:
    if not weekly:
        return "<p>No releases yet.</p>"
    w, h, pad = 900, 200, 30
    n, top = len(weekly), max(x["answered"] for x in weekly) or 1
    bw = (w - 2 * pad) / n
    parts = [f'<svg viewBox="0 0 {w} {h + 40}" width="100%" role="img" '
             'aria-label="Releases answered per week, with share fully right">']
    pts = []
    for i, x in enumerate(weekly):
        bh = (h - pad) * x["answered"] / top
        xpos = pad + i * bw
        parts.append(f'<rect x="{xpos + bw * 0.15:.1f}" y="{h - bh:.1f}" width="{bw * 0.7:.1f}" height="{bh:.1f}" '
                     f'rx="2" fill="var(--line)"><title>{x["week"]}: {x["answered"]} answered</title></rect>')
        if x["all_correct"] is not None:
            pts.append((xpos + bw / 2, h - (h - pad) * x["all_correct"]))
        if n <= 16 or i % max(1, n // 12) == 0:
            parts.append(f'<text x="{xpos + bw / 2:.1f}" y="{h + 16}" text-anchor="middle">{x["week"][5:]}</text>')
    if pts:
        parts.append('<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="'
                     + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        parts += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.5" fill="var(--accent)"/>' for a, b in pts]
    parts.append(f'<text x="{pad}" y="{h + 34}">Grey bars: releases answered per week (max {top}). '
                 'Blue: share with all four fields right, scored releases only (0 to 100%).</text></svg>')
    return "".join(parts)


def render(stats: dict, repo_url: str | None = None) -> str:
    e = html.escape
    c, lat, av = stats["counts"], stats["latency"], stats["accuracy_verified"]
    cards = [
        (str(c["answered"]), "releases answered"),
        (_fmt_pct(av.get("all_correct")) if av["n"] else "–", f"all 4 fields right (n={av['n']})"),
        (_fmt_s(lat["model_ms_p50"], "ms"), "model time, median"),
        (_fmt_s(lat["detect_s_p50"]), "SEC filing to release seen, median"),
        (str(c["waiting"]), "waiting for the 10-Q"),
    ]
    per_field = "".join(f"<tr><td>{FIELD_NAMES[f]}</td><td>{_fmt_pct(av.get(f))}</td>"
                        f"<td>{_fmt_pct(stats['accuracy_all_scored'].get(f))}</td></tr>" for f in FIELDS)
    errs = "".join(f"<tr><td>{e(k.replace('_', ' '))}</td><td>{v}</td></tr>" for k, v in stats["errors"].items()) \
        or '<tr><td colspan="2">None yet.</td></tr>'
    rows = []
    for r in stats["recent"]:
        f = r["fields"] or {}
        cls = "right" if r["status"] == "right" else ("wrong" if r["status"].startswith("wrong") else "")
        name = e(r["company"])
        if r.get("url"):
            name = f'<a href="{e(r["url"])}">{name}</a>'
        rows.append(f"<tr><td>{name}</td><td>{e(str(r['accepted_at'])[:16].replace('T', ' '))}</td>"
                    f"<td>{e(str(f.get('period_end') or '–'))}</td><td>{_fmt_money(f.get('revenue'))}</td>"
                    f"<td>{_fmt_money(f.get('net_income'))}</td>"
                    f"<td>{e(str(f.get('eps_diluted') if f.get('eps_diluted') is not None else '–'))}</td>"
                    f"<td>{_fmt_s(r['model_ms'], 'ms')}</td><td class=\"{cls}\">{e(r['status'])}</td></tr>")
    recent = "".join(rows) or '<tr><td colspan="8">No releases yet.</td></tr>'
    repo = f' · <a href="{e(repo_url)}">code and method</a>' if repo_url else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinSight live</title><style>{CSS}</style></head><body><main>
<h1>FinSight live</h1>
<p>A fine-tuned Mistral-7B reads each new US earnings press release as it is filed with the SEC and returns revenue,
net income, diluted EPS and the quarter as JSON. Weeks later, when the company files the same quarter in its 10-Q
or 10-K, every answer is checked automatically against those official figures.</p>
<p>Updated {e(stats['generated_at'][:16].replace('T', ' '))} UTC · model: {e(', '.join(stats['models']) or '–')}{repo}</p>
<div class="cards">{''.join(f'<div class="card"><div class="v">{v}</div><div class="k">{e(k)}</div></div>' for v, k in cards)}</div>

<h2>Accuracy over time</h2>
{_weekly_svg(stats['weekly'])}

<h2>Accuracy by field</h2>
<div class="scroll"><table><tr><th>Field</th><th>Figures printed in the release (n={av['n']})</th>
<th>All scored (n={stats['accuracy_all_scored']['n']})</th></tr>{per_field}</table></div>
<p>The first column counts releases where the official figures appear in the text the model read, which matches how
the model was tested. The second also counts releases whose official figure is not printed in the release (a
different revenue line, a later restatement), where even a perfect reader can be scored wrong.</p>

<h2>Why answers were wrong</h2>
<div class="scroll"><table><tr><th>Cause</th><th>Fields</th></tr>{errs}</table></div>

<h2>Speed</h2>
<div class="scroll"><table><tr><th></th><th>Median</th><th>95th percentile</th></tr>
<tr><td>Model time per release</td><td>{_fmt_s(lat['model_ms_p50'], 'ms')}</td><td>{_fmt_s(lat['model_ms_p95'], 'ms')}</td></tr>
<tr><td>SEC acceptance to release seen (live only, n={c['live']})</td><td>{_fmt_s(lat['detect_s_p50'])}</td><td>{_fmt_s(lat['detect_s_p95'])}</td></tr>
<tr><td>SEC acceptance to answer (live only)</td><td>{_fmt_s(lat['end_to_end_s_p50'])}</td><td>{_fmt_s(lat['end_to_end_s_p95'])}</td></tr>
</table></div>
<p>Detection speed depends on how often the watcher polls EDGAR, and is measured from SEC acceptance, not from the
newswire, which can be a few minutes earlier. Releases picked up after the watcher was off, and past days filled in
later, are answered and scored but left out of these two rows.</p>

<h2>Latest releases</h2>
<div class="scroll"><table><tr><th>Company</th><th>Accepted (ET)</th><th>Quarter end</th><th>Revenue</th>
<th>Net income</th><th>EPS</th><th>Model time</th><th>Check</th></tr>{recent}</table></div>

<h2>Limits</h2>
<p>A research prototype, not a trading system or advice. Fourth quarters, banks and insurers usually cannot be scored
automatically ({c['unscorable']} so far), because their quarterly figures are not tagged the standard way.
{c['no_release_document']} earnings filings had no press release exhibit to read. Answers that did not parse as
JSON: {_fmt_pct(None if c['valid_json'] is None else 1 - c['valid_json'])}.</p>
</main></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", default="live")
    ap.add_argument("--out", default="site")
    ap.add_argument("--repo-url", default="https://github.com/PujaSridhar/finsight")
    args = ap.parse_args()
    store = Store(Path(args.live))
    stats = compute(store.releases, store.scores)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(stats, args.repo_url))
    (out / "data.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out / 'index.html'}: {stats['counts']}")


if __name__ == "__main__":
    main()
