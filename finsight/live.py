"""Live run: watch SEC EDGAR for new earnings releases, answer each one, and score the answers weeks later.

    export SEC_USER_AGENT="Your Name your.email@example.com"
    python -m finsight.live poll --model-url http://localhost:8000          # one pass over the latest filings
    python -m finsight.live poll --model-url ... --loop --interval 60       # keep watching
    python -m finsight.live backfill --date 2026-11-19 --model-url ...      # a past day, from EDGAR's daily index
    python -m finsight.live score                                           # check answers against filed XBRL
    python -m finsight.dashboard                                            # build the public page

Everything lands in plain files under live/ (one JSON object per line), so the log is easy to diff, commit and
audit:

    releases.jsonl   one line per earnings release answered: the model's JSON, SEC acceptance time, when it was
                     seen, how long the model took
    scores.jsonl     one line per release once it is final: right or wrong per field against the figures the
                     company later filed in XBRL, or why it could not be scored
    pending.json     releases seen but not answered yet (model endpoint down, say); retried on the next poll
    state.json       where the last poll stopped, and 8-Ks already checked and found not to be earnings

Times: `accepted_at` is when the SEC accepted the filing (Eastern time, from EDGAR). Many companies put the same
release on a newswire a few minutes earlier, so `detect_lag_s` measures speed from EDGAR, not from the wire.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import earnings
from .edgar import CURRENT_FEED_URL, DAILY_INDEX_URL, EXCERPT_CHARS, SECClient, release_excerpt

ATOM = "{http://www.w3.org/2005/Atom}"
ACC_RE = re.compile(r"accession-number=(\d{10}-\d{2}-\d{6})")
CIK_RE = re.compile(r"\((\d{4,10})\)")
ITEM_RE = re.compile(r"Item\s+(\d+\.\d+)")
MIN_SCORE_AGE_DAYS = 3      # 10-Qs usually come 1 to 4 weeks after the release
MAX_SCORE_AGE_DAYS = 120    # after this, a release with no matching XBRL quarter is marked unscorable
CHECKED_KEEP = 5000


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Finding earnings releases

def parse_feed(xml_text: str) -> list[dict]:
    """Entries of EDGAR's latest-filings Atom feed: accession, cik, company, form, accepted_at, items.

    `items` is the list of 8-K item numbers when the entry's summary lists them, or None when it does not
    (the caller then checks the filing itself). A filing with co-registrants appears once per registrant; only
    the first is kept.
    """
    root = ET.fromstring(xml_text)
    out, seen = [], set()
    for e in root.iter(ATOM + "entry"):
        title = (e.findtext(ATOM + "title") or "").strip()
        acc = ACC_RE.search(e.findtext(ATOM + "id") or "")
        cat = e.find(ATOM + "category")
        form = cat.get("term") if cat is not None else title.split(" - ")[0]
        accepted = (e.findtext(ATOM + "updated") or "").strip()
        if not acc or acc.group(1) in seen or not accepted:
            continue
        seen.add(acc.group(1))
        cik = CIK_RE.search(title)
        company = re.sub(r"\s*\(\d{4,10}\).*$", "", title.split(" - ", 1)[-1]).strip()
        summary = html.unescape(e.findtext(ATOM + "summary") or "")
        items = ITEM_RE.findall(summary)
        out.append({"accession": acc.group(1), "cik": int(cik.group(1)) if cik else None, "company": company,
                    "form": form, "accepted_at": accepted,
                    "items": items or None})
    return out


def parse_daily_index(text: str) -> list[dict]:
    """8-K rows of an EDGAR daily form index (fixed-width text): accession, cik, company, filed."""
    out = []
    for line in text.splitlines():
        if not line.startswith("8-K "):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5 or parts[0] != "8-K":
            continue
        company, cik, filed, path = parts[1], parts[-3], parts[-2], parts[-1]
        acc = Path(path).stem
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", acc):
            continue
        out.append({"accession": acc, "cik": int(cik), "company": company, "form": "8-K",
                    "filed": f"{filed[:4]}-{filed[4:6]}-{filed[6:]}", "accepted_at": None, "items": None})
    return out


def items_from_index(index_html: str) -> list[str] | None:
    """8-K item numbers listed on a filing's index page, or None if the page lists none."""
    found = ITEM_RE.findall(html.unescape(re.sub(r"<[^>]+>", " ", index_html)))
    return found or None


def items_from_submissions(submissions: dict, accession: str) -> list[str] | None:
    recent = submissions.get("filings", {}).get("recent", {})
    for i, acc in enumerate(recent.get("accessionNumber", [])):
        if acc == accession:
            items = (recent.get("items") or [""] * (i + 1))[i] or ""
            return [x.strip() for x in items.split(",") if x.strip()]
    return None


def filed_date(entry: dict) -> str:
    if entry.get("filed"):
        return entry["filed"]
    return parse_time(entry["accepted_at"]).date().isoformat()  # accepted_at is Eastern time, as EDGAR dates are


# ---------------------------------------------------------------------------
# Store

class Store:
    """The live/ folder. Append-only JSONL for results, small JSON files for state."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _read_jsonl(self, name: str) -> list[dict]:
        p = self.root / name
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def _append(self, name: str, rec: dict) -> None:
        with (self.root / name).open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def _read_json(self, name: str, default):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    def _write_json(self, name: str, obj) -> None:
        tmp = self.root / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.root / name)

    releases = property(lambda self: self._read_jsonl("releases.jsonl"))
    scores = property(lambda self: self._read_jsonl("scores.jsonl"))

    def add_release(self, rec: dict) -> None:
        self._append("releases.jsonl", rec)

    def add_score(self, rec: dict) -> None:
        self._append("scores.jsonl", rec)

    @property
    def pending(self) -> dict:
        return self._read_json("pending.json", {})

    def set_pending(self, pending: dict) -> None:
        self._write_json("pending.json", pending)

    @property
    def state(self) -> dict:
        return self._read_json("state.json", {"checked": [], "newest_accepted": None})

    def set_state(self, state: dict) -> None:
        state["checked"] = state.get("checked", [])[-CHECKED_KEEP:]
        self._write_json("state.json", state)


# ---------------------------------------------------------------------------
# Watcher

class Watcher:
    """Finds new earnings 8-Ks and answers them. `extract(text)` is finsight.serve.extract bound to a backend."""

    def __init__(self, client: SECClient, store: Store, extract, model_name: str, max_pages: int = 5,
                 now=utcnow, log=print):
        self.client, self.store, self.extract = client, store, extract
        self.model_name, self.max_pages, self.now, self.log = model_name, max_pages, now, log

    def is_earnings(self, entry: dict, checked: set) -> tuple[bool, str | None]:
        """(is an Item 2.02 8-K, its index page if fetched). Uses the feed's item list when it has one."""
        if entry["form"] != "8-K":
            return False, None
        if entry["items"] is not None:
            return "2.02" in entry["items"], None
        if entry["accession"] in checked:
            return False, None
        index_html = self.client.filing_index(entry["cik"], entry["accession"])
        items = items_from_index(index_html)
        if items is None:
            items = items_from_submissions(self.client.submissions(entry["cik"], cache=False), entry["accession"])
        if items is None:  # not listed anywhere yet; look again next time
            return False, None
        if "2.02" not in items:
            checked.add(entry["accession"])
        return "2.02" in items, index_html

    def feed_entries(self, newest: str | None) -> list[dict]:
        """New feed entries since the last poll, paging back until the last poll's newest filing."""
        out, stop = [], parse_time(newest) if newest else None
        for page in range(self.max_pages):
            entries = parse_feed(self.client.get(CURRENT_FEED_URL.format(start=page * 100), cache=False)
                                 .decode("utf-8", "replace"))
            out.extend(entries)
            if not entries or stop is None or any(parse_time(e["accepted_at"]) <= stop for e in entries):
                break
        if stop is not None:
            out = [e for e in out if parse_time(e["accepted_at"]) > stop - timedelta(minutes=5)]
        return out

    def detect(self, entries: list[dict], mode: str) -> int:
        """Add new earnings releases among `entries` to pending. Returns how many were added."""
        known = {r["accession"] for r in self.store.releases}
        pending, state = self.store.pending, self.store.state
        checked = set(state.get("checked", []))
        added = 0
        for e in entries:
            acc = e["accession"]
            if acc in known or acc in pending or e["cik"] is None:
                continue
            try:
                ok, index_html = self.is_earnings(e, checked)
                if not ok:
                    continue
                url = self.client.release_url(e["cik"], acc, index_html)
            except Exception as err:  # one bad filing must not stop the watch; it is seen again next poll
                self.log(f"skip {acc} for now: {type(err).__name__}: {err}")
                continue
            pending[acc] = {"accession": acc, "cik": e["cik"], "company": e["company"],
                            "accepted_at": e.get("accepted_at"), "filed": filed_date(e), "url": url,
                            "detected_at": iso(self.now()), "mode": mode, "attempts": 0}
            added += 1
        self.store.set_pending(pending)
        state["checked"] = sorted(checked)
        self.store.set_state(state)
        return added

    def answer_pending(self, max_attempts: int = 20) -> int:
        """Run the model on every pending release. Returns how many were answered."""
        pending, answered = self.store.pending, 0
        for acc, p in sorted(pending.items(), key=lambda kv: kv[1]["detected_at"]):
            rec = {k: p[k] for k in ("accession", "cik", "company", "accepted_at", "filed", "url", "detected_at",
                                     "mode")}
            rec["model"] = self.model_name
            if p["url"] is None:
                rec.update(error="no_release_document", fields=None, valid_json=False)
            else:
                try:
                    text = self.client.release_text(p["url"])
                    out = self.extract(text)
                except Exception as err:
                    p["attempts"] += 1
                    p["last_error"] = f"{type(err).__name__}: {err}"[:300]
                    self.log(f"{acc}: model or fetch failed ({p['last_error']}), will retry")
                    if p["attempts"] < max_attempts:
                        continue
                    rec.update(error="gave_up: " + p["last_error"], fields=None, valid_json=False)
                else:
                    rec.update(fields=out["fields"], valid_json=out["valid_json"], model_ms=out["model_ms"],
                               input_tokens=out.get("input_tokens"), output_tokens=out.get("output_tokens"),
                               excerpt_chars=out["excerpt_chars"])
            done = self.now()
            rec["answered_at"] = iso(done)
            if rec["mode"] == "live" and rec["accepted_at"]:
                acc_t = parse_time(rec["accepted_at"])
                rec["detect_lag_s"] = round((parse_time(rec["detected_at"]) - acc_t).total_seconds(), 1)
                rec["end_to_end_s"] = round((done - acc_t).total_seconds(), 1)
            self.store.add_release(rec)
            del pending[acc]
            self.store.set_pending(pending)
            answered += 1
            self.log(f"{rec['company']}: {rec.get('fields') or rec.get('error')}")
        self.store.set_pending(pending)
        return answered

    def poll(self, max_gap_minutes: float = 45) -> dict:
        """One pass over the latest filings. Releases found after a gap (the first poll, or a watcher that was
        off for a while) are marked catch_up rather than live, so they don't count toward detection speed."""
        state = self.store.state
        last = state.get("last_poll_at")
        live = bool(last) and (self.now() - parse_time(last)) <= timedelta(minutes=max_gap_minutes)
        entries = self.feed_entries(state.get("newest_accepted") if live else None)
        added = self.detect(entries, "live" if live else "catch_up")
        state = self.store.state
        if entries:
            top = max(entries, key=lambda e: parse_time(e["accepted_at"]))["accepted_at"]
            if not state.get("newest_accepted") or parse_time(top) > parse_time(state["newest_accepted"]):
                state["newest_accepted"] = top
        state["last_poll_at"] = iso(self.now())
        self.store.set_state(state)
        answered = self.answer_pending()
        return {"mode": "live" if live else "catch_up", "feed_entries": len(entries), "new_releases": added,
                "answered": answered,
                "still_pending": len(self.store.pending)}

    def backfill(self, day: date) -> dict:
        url = DAILY_INDEX_URL.format(year=day.year, qtr=(day.month - 1) // 3 + 1, ymd=day.strftime("%Y%m%d"))
        entries = parse_daily_index(self.client.get(url).decode("latin-1"))
        added = self.detect(entries, "backfill")
        return {"index_8ks": len(entries), "new_releases": added, "answered": self.answer_pending()}


# ---------------------------------------------------------------------------
# Delayed scoring

def score_release(rec: dict, companyfacts: dict, text: str | None, today: date) -> dict | None:
    """Final score for one answered release, or None if its figures may still be on the way."""
    age = (today - date.fromisoformat(rec["filed"])).days
    label, distractors, reason = earnings.label_for_release(companyfacts, rec["filed"]) if companyfacts else \
        (None, {}, "no_xbrl")
    if label is None:
        if age < MAX_SCORE_AGE_DAYS:
            return None
        return {"accession": rec["accession"], "status": "unscorable", "reason": reason}
    excerpt = release_excerpt(text) if text else ""
    example = {"label": label, "distractors": distractors, "text": excerpt}
    s = earnings.score(json.dumps(rec["fields"]) if rec.get("fields") else "", example)
    out = {"accession": rec["accession"], "status": "scored", "label": label,
           "verified_in_text": bool(excerpt) and not earnings.verify(label, excerpt),
           "all_correct": s["all_correct"], "valid_json": s["valid_json"]}
    for f in earnings.FIELDS:
        out[f] = s[f]
        out[f + "_error"] = s[f + "_error"]
    return out


def score_all(client: SECClient, store: Store, today: date | None = None, log=print) -> dict:
    today = today or utcnow().date()
    done = {s["accession"] for s in store.scores}
    todo = [r for r in store.releases if r["accession"] not in done and not r.get("error")
            and (today - date.fromisoformat(r["filed"])).days >= MIN_SCORE_AGE_DAYS]
    facts_by_cik, counts = {}, {"checked": 0, "scored": 0, "unscorable": 0, "waiting": 0}
    for rec in todo:
        counts["checked"] += 1
        try:
            if rec["cik"] not in facts_by_cik:
                try:
                    facts_by_cik[rec["cik"]] = client.companyfacts(rec["cik"], cache=False)
                except Exception as err:
                    if getattr(err, "code", None) != 404:  # 404: the company has no XBRL facts at all
                        raise
                    facts_by_cik[rec["cik"]] = {}
            text = client.release_text(rec["url"]) if rec.get("url") else None
            s = score_release(rec, facts_by_cik[rec["cik"]], text, today)
        except Exception as err:
            log(f"{rec['accession']}: could not score yet ({type(err).__name__})")
            counts["waiting"] += 1
            continue
        if s is None:
            counts["waiting"] += 1
            continue
        s["scored_at"] = iso(utcnow())
        store.add_score(s)
        counts[s["status"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Merging logs from two runners (say, a Colab session and the scheduled job)

def merge(store: Store, other: Store) -> dict:
    """Add the other folder's releases and scores that this one lacks. For a release answered by both, the
    earlier answer is kept, since the live log is about what was known first."""
    mine = {r["accession"]: r for r in store.releases}
    rows = dict(mine)
    for r in other.releases:
        if r["accession"] not in rows or r["answered_at"] < rows[r["accession"]]["answered_at"]:
            rows[r["accession"]] = r
    scores = {s["accession"]: s for s in other.scores}
    scores.update({s["accession"]: s for s in store.scores})
    added = len(rows) - len(mine)
    (store.root / "releases.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in
                                                       sorted(rows.values(), key=lambda r: r["answered_at"])))
    (store.root / "scores.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in
                                                     sorted(scores.values(), key=lambda s: s.get("scored_at", ""))))
    pending = {**other.pending, **store.pending}
    store.set_pending({k: v for k, v in pending.items() if k not in rows})
    return {"releases_added": added, "releases": len(rows), "scores": len(scores)}


# ---------------------------------------------------------------------------
# CLI

def make_extract(args):
    from .serve import VLLMBackend, extract

    url = args.model_url or os.environ.get("FINSIGHT_MODEL_URL")
    if not url:
        raise SystemExit("Give --model-url (an OpenAI-compatible endpoint serving the FinSight adapter, such as "
                         "scripts/serve_vllm.sh) or set FINSIGHT_MODEL_URL.")
    backend = VLLMBackend(url, model=args.model_name, timeout=args.timeout,
                          api_key=os.environ.get(args.api_key_env) or None)
    return lambda text: extract(backend, text, args.max_chars)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-dir", default="live")
    ap.add_argument("--cache-dir", default="data/raw/sec")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("poll", "backfill"):
        p = sub.add_parser(name)
        p.add_argument("--model-url", default=None)
        p.add_argument("--model-name", default=os.environ.get("FINSIGHT_MODEL_NAME", "finsight"),
                       help="served model name: the LoRA name in vLLM (finsight) or what your host calls it")
        p.add_argument("--api-key-env", default="FINSIGHT_MODEL_KEY", help="env var holding the endpoint's key")
        p.add_argument("--timeout", type=float, default=300, help="seconds; allows for a serverless cold start")
        p.add_argument("--max-chars", type=int, default=EXCERPT_CHARS)
    sub.choices["poll"].add_argument("--loop", action="store_true")
    sub.choices["poll"].add_argument("--interval", type=float, default=60, help="seconds between polls")
    sub.choices["poll"].add_argument("--max-hours", type=float, default=None)
    sub.choices["backfill"].add_argument("--date", required=True, help="YYYY-MM-DD, a day already past")
    sub.add_parser("score")
    m = sub.add_parser("merge")
    m.add_argument("other", help="another live/ folder to merge into this one")
    args = ap.parse_args()

    store = Store(Path(args.live_dir))
    if args.cmd == "merge":
        print(merge(store, Store(Path(args.other))))
        return
    client = SECClient(Path(args.cache_dir))
    if args.cmd == "score":
        print(score_all(client, store))
        return
    watcher = Watcher(client, store, make_extract(args), args.model_name)
    if args.cmd == "backfill":
        print(watcher.backfill(date.fromisoformat(args.date)))
        return
    deadline = time.monotonic() + 3600 * args.max_hours if args.max_hours else None
    while True:
        started = time.monotonic()
        try:
            print(iso(utcnow()), watcher.poll(), flush=True)
        except Exception as err:  # keep watching through a bad poll (EDGAR hiccup, network blip)
            print(iso(utcnow()), f"poll failed: {type(err).__name__}: {err}", flush=True)
        if not args.loop or (deadline and time.monotonic() > deadline):
            break
        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
