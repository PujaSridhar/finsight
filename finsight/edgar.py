"""SEC EDGAR access: company list, 8-K earnings releases and XBRL facts.

EDGAR is free but asks every client to identify itself and stay under 10
requests a second (https://www.sec.gov/os/accessing-edgar-data). Set your own
name and email before running anything that downloads:

    export SEC_USER_AGENT="Your Name your.email@example.com"

Responses are cached under data/raw/sec/, so a re-run only fetches what is new.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
SEC_HOSTS = ("www.sec.gov", "sec.gov", "data.sec.gov")


class SECClient:
    def __init__(self, cache_dir: Path = Path("data/raw/sec"), user_agent: str | None = None,
                 max_per_second: float = 8.0):
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT", "").strip()
        if not self.user_agent or "@" not in self.user_agent:
            raise SystemExit('Set SEC_USER_AGENT to your name and email first, e.g.\n'
                             '  export SEC_USER_AGENT="Your Name your.email@example.com"\n'
                             "The SEC requires it of every client (it is sent only to sec.gov).")
        self.cache_dir = Path(cache_dir)
        self.min_interval = 1.0 / max_per_second
        self._last = 0.0

    def _cache_path(self, url: str) -> Path:
        key = re.sub(r"^https://", "", url).replace("/", "__")
        return self.cache_dir / key[:200]

    def get(self, url: str, retries: int = 4) -> bytes:
        import urllib.error
        import urllib.request

        path = self._cache_path(url)
        if path.exists():
            return path.read_bytes()
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt * (5 if e.code in (403, 429) else 1))  # back off if throttled
            except urllib.error.URLError:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    def get_json(self, url: str) -> dict:
        return json.loads(self.get(url))

    def companies(self) -> list[dict]:
        """[{cik, ticker, title}] in the order of SEC's ticker file."""
        data = self.get_json(TICKERS_URL)
        return [{"cik": int(v["cik_str"]), "ticker": v["ticker"], "title": v["title"]} for v in data.values()]

    def earnings_8ks(self, cik: int, since: str = "2019-01-01") -> list[dict]:
        return earnings_8ks_from_submissions(self.get_json(SUBMISSIONS_URL.format(cik=cik)), since)

    def companyfacts(self, cik: int) -> dict:
        return self.get_json(COMPANYFACTS_URL.format(cik=cik))

    def release_url(self, cik: int, accession: str) -> str | None:
        base = ARCHIVE_URL.format(cik=cik, acc_nodash=accession.replace("-", ""))
        index_html = self.get(base + f"{accession}-index.htm").decode("utf-8", "replace")
        doc = find_release_document(index_html)
        return base + doc if doc else None

    def release_text(self, url: str) -> str:
        return html_to_text(self.get(url).decode("utf-8", "replace"))


def earnings_8ks_from_submissions(submissions: dict, since: str = "2019-01-01") -> list[dict]:
    """8-K (and 8-K/A excluded) filings whose items include 2.02, Results of Operations and Financial Condition."""
    recent = submissions["filings"]["recent"]
    out = []
    for i, form in enumerate(recent["form"]):
        items = recent.get("items", [""] * len(recent["form"]))[i] or ""
        if form == "8-K" and "2.02" in items.split(",") and recent["filingDate"][i] >= since:
            out.append({"accession": recent["accessionNumber"][i], "filed": recent["filingDate"][i]})
    return out


class _IndexParser(HTMLParser):
    """Rows of the filing index table: [seq, description, document, type, size]."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._href, self._in_td = [], None, "", None, False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._in_td, self._cell, self._href = True, "", None
        elif tag == "a" and self._in_td:
            self._href = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag == "td" and self._row is not None:
            self._row.append((self._cell.strip(), self._href))
            self._in_td = False
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._in_td:
            self._cell += data


def find_release_document(index_html: str) -> str | None:
    """File name of the press release (EX-99.1, else the first EX-99.x) from a filing's -index.htm page."""
    p = _IndexParser()
    p.feed(index_html)
    candidates = []
    for row in p.rows:
        texts = [t for t, _ in row]
        href = next((h for _, h in row if h and re.search(r"\.(htm|html|txt)$", h, re.I)), None)
        doc_type = next((t for t in texts if re.fullmatch(r"EX-99(\.\d+)?", t.upper())), None)
        if href and doc_type:
            candidates.append((0 if doc_type.upper() in ("EX-99.1", "EX-99") else 1, href.rsplit("/", 1)[-1]))
    return min(candidates)[1] if candidates else None


class _TextParser(HTMLParser):
    """HTML to plain text that keeps table rows on one line with cells separated by ' | '."""

    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}

    def __init__(self):
        super().__init__()
        self.out, self._cells, self._cell, self._skip = [], None, None, 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "tr":
            self._cells = []
        elif tag in ("td", "th") and self._cells is not None:
            self._cell = ""
        elif tag in self.BLOCK and self._cell is None:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th") and self._cells is not None and self._cell is not None:
            self._cells.append(" ".join(self._cell.split()))
            self._cell = None
        elif tag == "tr" and self._cells is not None:
            row = merge_cells(self._cells)
            if row:
                self.out.append("\n" + " | ".join(row) + "\n")
            self._cells = None
        elif tag in self.BLOCK and self._cell is None:
            self.out.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._cell is not None:
            self._cell += data
        else:
            self.out.append(data)


def merge_cells(cells: list[str]) -> list[str]:
    """Drop empty cells and glue the '$', '(' and ')' / '%' fragments that filings put in their own cells."""
    merged: list[str] = []
    for c in (c.strip() for c in cells):
        if not c:
            continue
        if merged and (merged[-1] in ("$", "(", "$(") or c in (")", "%", ")%", "%)")):
            merged[-1] = (merged[-1] + c).replace("$ ", "$")
        else:
            merged.append(c)
    return merged


def html_to_text(raw: str) -> str:
    p = _TextParser()
    p.feed(raw)
    text = html.unescape("".join(p.out)).replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


INCOME_STATEMENT_RE = re.compile(
    r"(condensed\s+)?(consolidated\s+)?(statements?\s+of\s+(consolidated\s+)?(operations|income|earnings)"
    r"|income\s+statements?)", re.I)


def release_excerpt(text: str, max_chars: int = 9000, head_chars: int = 3000) -> str:
    """What the model reads: the release's opening (headline figures) plus its income statement.

    A release runs 20K-100K characters; the figures we extract are in the first page and the income statement.
    Cutting to those keeps the prompt under ~2,500 tokens, which is what makes a 7B model fast and fits a T4.
    The same cut is used in training, evaluation, serving and the demo.
    """
    if len(text) <= max_chars:
        return text
    head = text[:head_chars].rsplit("\n", 1)[0]
    m = INCOME_STATEMENT_RE.search(text, len(head))
    start = m.start() if m else len(head)
    body = text[start:start + max_chars - len(head) - 10].rsplit("\n", 1)[0]
    return f"{head}\n[...]\n{body}"


def is_sec_url(url: str) -> bool:
    from urllib.parse import urlparse

    u = urlparse(url)
    return u.scheme == "https" and u.hostname in SEC_HOSTS
