"""Live watcher, delayed scorer and dashboard, with EDGAR and the model replaced by stand-ins (no network, no GPU)."""

import json
from datetime import date, datetime, timedelta, timezone

from test_earnings import FACTS, INDEX_HTML, RELEASE_HTML

from finsight import dashboard, live
from finsight.edgar import html_to_text

FEED = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Latest Filings</title>
<entry>
<title>8-K - ACME CORP (0000000001) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1/000000000124000001/0000000001-24-000001-index.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2024-10-24 &lt;b&gt;AccNo:&lt;/b&gt; 0000000001-24-000001 &lt;b&gt;Size:&lt;/b&gt; 1 MB&lt;br&gt;Item 2.02: Results of Operations and Financial Condition&lt;br&gt;Item 9.01: Financial Statements and Exhibits</summary>
<updated>2024-10-24T16:05:10-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0000000001-24-000001</id>
</entry>
<entry>
<title>8-K - ACME HOLDINGS (0000000009) (Filer)</title>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2024-10-24 &lt;b&gt;AccNo:&lt;/b&gt; 0000000001-24-000001</summary>
<updated>2024-10-24T16:05:10-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0000000001-24-000001</id>
</entry>
<entry>
<title>8-K - WIDGET INC (0000000002) (Filer)</title>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2024-10-24 &lt;b&gt;AccNo:&lt;/b&gt; 0000000002-24-000007&lt;br&gt;Item 5.02: Departure of Directors</summary>
<updated>2024-10-24T16:01:00-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0000000002-24-000007</id>
</entry>
<entry>
<title>8-K - GADGET CO (0000000003) (Filer)</title>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2024-10-24 &lt;b&gt;AccNo:&lt;/b&gt; 0000000003-24-000002</summary>
<updated>2024-10-24T16:00:00-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0000000003-24-000002</id>
</entry>
</feed>"""

DAILY_INDEX = """Description:           Daily Index of EDGAR Dissemination Feed by Form Type
Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
10-Q        SOMETHING ELSE INC                                            5           20241024    edgar/data/5/0000000005-24-000001.txt
8-K         ACME CORP                                                     1           20241024    edgar/data/1/0000000001-24-000001.txt
8-K/A       ACME CORP                                                     1           20241024    edgar/data/1/0000000001-24-000009.txt
"""

ACCEPTED = datetime(2024, 10, 24, 20, 5, 10, tzinfo=timezone.utc)   # 16:05:10 Eastern (EDT)
ANSWER = {"period_end": "2024-09-28", "revenue": 1_234_000_000, "net_income": -56_000_000, "eps_diluted": 0.97}


class FakeClient:
    def __init__(self, feed=FEED, gadget_items="Item 8.01: Other Events"):
        self.feed, self.gadget_items, self.calls = feed, gadget_items, []

    def get(self, url, cache=True):
        self.calls.append(url)
        if "getcurrent" in url:
            return self.feed.encode() if "start=0&" in url else b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
        if "daily-index" in url:
            return DAILY_INDEX.encode()
        raise AssertionError(url)

    def filing_index(self, cik, accession):
        self.calls.append(f"index {accession}")
        items = {1: "Item 2.02: Results of Operations and Financial Condition", 3: self.gadget_items}.get(cik, "")
        return f'<div class="infoHead">Items</div><div class="info">{items}</div>' + INDEX_HTML

    def submissions(self, cik, cache=True):
        return {"filings": {"recent": {"accessionNumber": [], "items": []}}}

    def release_url(self, cik, accession, index_html=None):
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/ex991.htm"

    def release_text(self, url):
        return html_to_text(RELEASE_HTML)

    def companyfacts(self, cik, cache=True):
        return FACTS


def watcher(tmp_path, client=None, extract=None, now=ACCEPTED + timedelta(seconds=40)):
    store = live.Store(tmp_path / "live")
    extract = extract or (lambda text: {"fields": ANSWER, "valid_json": True, "model_ms": 850.0,
                                        "input_tokens": 2500, "output_tokens": 50, "excerpt_chars": len(text)})
    clock = {"t": now}
    w = live.Watcher(client or FakeClient(), store, extract, "finsight", now=lambda: clock["t"], log=lambda *a: None)
    return w, store, clock


def test_parse_feed_reads_items_and_drops_co_registrant_duplicates():
    entries = live.parse_feed(FEED)
    assert [e["accession"] for e in entries] == ["0000000001-24-000001", "0000000002-24-000007",
                                                 "0000000003-24-000002"]
    assert entries[0]["items"] == ["2.02", "9.01"] and entries[0]["cik"] == 1 and entries[0]["company"] == "ACME CORP"
    assert entries[2]["items"] is None   # no item list in the summary: the watcher checks the filing itself
    assert live.filed_date(entries[0]) == "2024-10-24"


def test_item_lookup_falls_back_from_index_page_to_submissions():
    assert live.items_from_index('<div class="info">Item 2.02: Results<br>Item 9.01: Exhibits</div>') == ["2.02", "9.01"]
    assert live.items_from_index(INDEX_HTML) is None
    subs = {"filings": {"recent": {"accessionNumber": ["a", "b"], "items": ["5.02", "2.02,9.01"]}}}
    assert live.items_from_submissions(subs, "b") == ["2.02", "9.01"] and live.items_from_submissions(subs, "c") is None


def test_parse_daily_index_keeps_plain_8ks():
    rows = live.parse_daily_index(DAILY_INDEX)
    assert rows == [{"accession": "0000000001-24-000001", "cik": 1, "company": "ACME CORP", "form": "8-K",
                     "filed": "2024-10-24", "accepted_at": None, "items": None}]


def test_first_poll_is_catch_up_then_live_polls_time_detection(tmp_path):
    w, store, clock = watcher(tmp_path)
    out = w.poll()
    assert out["mode"] == "catch_up" and out["new_releases"] == 1 and out["answered"] == 1
    rec = store.releases[0]
    assert rec["fields"] == ANSWER and rec["model_ms"] == 850.0 and "detect_lag_s" not in rec
    # Gadget had no items in the feed, so its index page was read once and remembered as not earnings.
    assert "0000000003-24-000002" in store.state["checked"]

    newer = FEED.replace("0000000001-24-000001", "0000000001-24-000002").replace("16:05:10", "16:20:00")
    w.client = FakeClient(newer)
    clock["t"] = datetime(2024, 10, 24, 20, 20, 45, tzinfo=timezone.utc)
    out = w.poll()
    assert out["mode"] == "live" and out["new_releases"] == 1
    rec = store.releases[1]
    assert rec["mode"] == "live" and rec["detect_lag_s"] == 45.0 and rec["end_to_end_s"] == 45.0
    assert not any("index 0000000003" in c for c in w.client.calls)   # not re-checked


def test_polls_skip_what_is_already_answered(tmp_path):
    w, store, _ = watcher(tmp_path)
    w.poll()
    assert w.poll()["new_releases"] == 0 and len(store.releases) == 1


def test_model_failure_keeps_the_release_pending_and_retries(tmp_path):
    def down(text):
        raise ConnectionError("endpoint asleep")

    w, store, _ = watcher(tmp_path, extract=down)
    out = w.poll()
    assert out["answered"] == 0 and out["still_pending"] == 1 and store.releases == []
    assert store.pending["0000000001-24-000001"]["attempts"] == 1
    w.extract = lambda text: {"fields": ANSWER, "valid_json": True, "model_ms": 900.0, "excerpt_chars": 10}
    assert w.answer_pending() == 1 and store.pending == {} and store.releases[0]["fields"] == ANSWER


def test_backfill_uses_daily_index_and_is_not_counted_as_live(tmp_path):
    w, store, _ = watcher(tmp_path)
    out = w.backfill(date(2024, 10, 24))
    assert out == {"index_8ks": 1, "new_releases": 1, "answered": 1}
    assert store.releases[0]["mode"] == "backfill" and "detect_lag_s" not in store.releases[0]


def test_score_waits_then_scores_against_xbrl(tmp_path):
    w, store, _ = watcher(tmp_path)
    w.poll()
    client = FakeClient()
    assert live.score_all(client, store, today=date(2024, 10, 25), log=lambda *a: None)["checked"] == 0  # too soon
    out = live.score_all(client, store, today=date(2024, 11, 5), log=lambda *a: None)
    assert out["scored"] == 1
    s = store.scores[0]
    assert s["all_correct"] and s["verified_in_text"] and s["label"]["eps_diluted"] == 0.97
    assert live.score_all(client, store, today=date(2024, 11, 6))["checked"] == 0   # final scores are not redone


def test_score_marks_wrong_fields_and_gives_up_on_unlabelable_releases(tmp_path):
    wrong = {**ANSWER, "eps_diluted": 1.10}
    w, store, _ = watcher(tmp_path, extract=lambda t: {"fields": wrong, "valid_json": True, "model_ms": 1.0,
                                                       "excerpt_chars": 1})
    w.poll()
    s = live.score_release(store.releases[0], FACTS, html_to_text(RELEASE_HTML), date(2024, 11, 5))
    assert not s["all_correct"] and s["revenue"] and s["eps_diluted_error"] == "likely_non_gaap"
    rec = store.releases[0]
    assert live.score_release(rec, {}, None, date(2024, 11, 5)) is None                  # may still come
    assert live.score_release(rec, {}, None, date(2025, 6, 1))["status"] == "unscorable"   # gave up


def test_merge_keeps_the_earlier_answer(tmp_path):
    a, b = live.Store(tmp_path / "a"), live.Store(tmp_path / "b")
    a.add_release({"accession": "x", "answered_at": "2024-10-24T20:10:00Z", "model": "colab"})
    b.add_release({"accession": "x", "answered_at": "2024-10-24T20:05:00Z", "model": "action"})
    b.add_release({"accession": "y", "answered_at": "2024-10-24T21:00:00Z", "model": "action"})
    assert live.merge(a, b) == {"releases_added": 1, "releases": 2, "scores": 0}
    assert [r["model"] for r in a.releases] == ["action", "action"]


def test_dashboard_computes_and_renders_from_the_log(tmp_path):
    w, store, clock = watcher(tmp_path)
    w.poll()
    newer = FEED.replace("0000000001-24-000001", "0000000001-24-000002").replace("16:05:10", "16:20:00")
    w.client, clock["t"] = FakeClient(newer), datetime(2024, 10, 24, 20, 21, 0, tzinfo=timezone.utc)
    w.poll()
    live.score_all(FakeClient(), store, today=date(2024, 11, 5), log=lambda *a: None)
    stats = dashboard.compute(store.releases, store.scores)
    assert stats["counts"]["answered"] == 2 and stats["counts"]["live"] == 1 and stats["counts"]["waiting"] == 0
    assert stats["accuracy_verified"] == {"n": 2, "all_correct": 1.0, "period_end": 1.0, "revenue": 1.0,
                                          "net_income": 1.0, "eps_diluted": 1.0}
    assert stats["latency"]["detect_s_p50"] == 60.0 and stats["weekly"][0]["scored"] == 2
    page = dashboard.render(stats, "https://github.com/PujaSridhar/finsight")
    assert "<svg" in page and "ACME CORP" in page and "$1.23B" in page and "<script" not in page
    json.dumps(stats)


def test_dashboard_handles_an_empty_log():
    stats = dashboard.compute([], [])
    assert stats["counts"]["answered"] == 0 and "No releases yet." in dashboard.render(stats)
