"""The earnings task: EDGAR parsing, XBRL labels, verification and scoring, on small hand-made fixtures."""

import json

import pytest

from finsight import earnings as e
from finsight import edgar

RELEASE_HTML = """<html><body>
<p>ACME CORP REPORTS THIRD QUARTER 2024 RESULTS</p>
<p>Revenue of $1.2 billion; adjusted EPS of $1.10. GAAP diluted EPS of $0.97.</p>
<table>
<tr><td></td><td>Three Months Ended</td><td></td></tr>
<tr><td>(in millions, except per share)</td><td>Sep 28, 2024</td><td>Sep 30, 2023</td></tr>
<tr><td>Net revenue</td><td>$</td><td>1,234</td><td>$</td><td>1,100</td></tr>
<tr><td>Net income (loss)</td><td>$</td><td>(56</td><td>)</td><td>$</td><td>40</td></tr>
<tr><td>Diluted earnings per share</td><td>$</td><td>0.97</td><td>$</td><td>0.70</td></tr>
</table>
<script>var x = 1;</script>
</body></html>"""

INDEX_HTML = """<table class="tableFile">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/1/000000000124000001/acme-8k.htm">acme-8k.htm</a></td><td>8-K</td><td>30000</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/1/000000000124000001/acme-ex991.htm">acme-ex991.htm</a></td><td>EX-99.1</td><td>90000</td></tr>
<tr><td>3</td><td>SLIDES</td><td><a href="/Archives/edgar/data/1/000000000124000001/acme-ex992.htm">acme-ex992.htm</a></td><td>EX-99.2</td><td>9000</td></tr>
</table>"""


def fact(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


FACTS = {"facts": {"us-gaap": {
    # The company moved from SalesRevenueNet to the ASC 606 tag; both should be found.
    "SalesRevenueNet": {"units": {"USD": [fact("2022-07-01", "2022-09-30", 1_000_000_000, "2022-11-01")]}},
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        fact("2023-07-02", "2023-09-30", 1_100_000_000, "2023-11-01"),
        fact("2024-06-30", "2024-09-28", 1_234_000_000, "2024-11-01"),
        fact("2024-06-30", "2024-09-28", 1_230_000_000, "2025-11-01"),   # later restatement: ignored
        fact("2023-12-31", "2024-09-28", 3_500_000_000, "2024-11-01"),   # nine months (year to date)
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        fact("2023-07-02", "2023-09-30", 40_000_000, "2023-11-01"),
        fact("2024-06-30", "2024-09-28", -56_000_000, "2024-11-01")]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [
        fact("2023-07-02", "2023-09-30", 0.70, "2023-11-01"),
        fact("2024-06-30", "2024-09-28", 0.97, "2024-11-01")]}},
}}}

SUBMISSIONS = {"filings": {"recent": {
    "accessionNumber": ["0000000001-24-000001", "0000000001-24-000002", "0000000001-18-000003"],
    "filingDate": ["2024-10-24", "2024-09-01", "2018-10-20"],
    "form": ["8-K", "8-K", "8-K"],
    "items": ["2.02,9.01", "5.02", "2.02"],
}}}


def test_html_to_text_keeps_table_rows_and_glues_dollar_and_paren_cells():
    text = edgar.html_to_text(RELEASE_HTML)
    assert "Net revenue | $1,234 | $1,100" in text
    assert "Net income (loss) | $(56) | $40" in text
    assert "var x" not in text


def test_find_release_document_prefers_ex_99_1():
    assert edgar.find_release_document(INDEX_HTML) == "acme-ex991.htm"


def test_earnings_8ks_keeps_item_2_02_since_date():
    assert edgar.earnings_8ks_from_submissions(SUBMISSIONS, "2019-01-01") == [
        {"accession": "0000000001-24-000001", "filed": "2024-10-24"}]


def test_release_excerpt_keeps_head_and_income_statement():
    text = "Headline numbers.\n" + "filler line\n" * 2000 + "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\n" \
           "Revenue | 1,234\n" + "more\n" * 2000
    ex = edgar.release_excerpt(text, max_chars=3000, head_chars=500)
    assert ex.startswith("Headline numbers.") and "STATEMENTS OF OPERATIONS\nRevenue | 1,234" in ex
    assert len(ex) <= 3000


def test_is_sec_url():
    assert edgar.is_sec_url("https://www.sec.gov/Archives/edgar/data/1/x.htm")
    assert not edgar.is_sec_url("https://evil.example/www.sec.gov/x")
    assert not edgar.is_sec_url("http://www.sec.gov/x")


def test_sec_client_refuses_to_run_without_a_user_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit):
        edgar.SECClient(tmp_path)


def test_label_for_release_uses_original_filing_and_finds_distractors():
    label, distractors, why = e.label_for_release(FACTS, "2024-10-24")
    assert why == "ok"
    assert label == {"period_end": "2024-09-28", "revenue": 1_234_000_000, "net_income": -56_000_000,
                     "eps_diluted": 0.97}
    assert distractors["revenue"] == {"prior_year_quarter": 1_100_000_000, "year_to_date": 3_500_000_000}
    assert distractors["eps_diluted"]["prior_year_quarter"] == 0.70


def test_label_for_release_skips_when_no_quarter_matches():
    assert e.label_for_release(FACTS, "2024-06-01")[2] == "no_quarter"


def test_verify_finds_scaled_figures_in_text():
    text = edgar.html_to_text(RELEASE_HTML)
    label = {"revenue": 1_234_000_000, "net_income": -56_000_000, "eps_diluted": 0.97}
    assert e.verify(label, text) == []
    assert e.verify({**label, "revenue": 1_235_000_000}, text) == ["revenue"]


GOLD = {"label": {"period_end": "2024-09-28", "revenue": 1_234_000_000, "net_income": -56_000_000,
                  "eps_diluted": 0.97},
        "distractors": {"revenue": {"prior_year_quarter": 1_100_000_000, "year_to_date": 3_500_000_000},
                        "eps_diluted": {"prior_year_quarter": 0.70}},
        "text": "GAAP diluted EPS of $0.97. Adjusted EPS of $1.10 for the quarter."}


def test_target_scores_as_correct():
    s = e.score(e.target_json(GOLD["label"]), GOLD)
    assert s["all_correct"] and s["valid_json"]


def test_score_explains_wrong_fields():
    resp = ('Here you go:\n```json\n{"period_end": "2024-09-28", "revenue": 1100000000, '
            '"net_income": "-56 million", "eps_diluted": 1.10}\n```')
    s = e.score(resp, GOLD)
    assert s["valid_json"] and s["period_end"] and s["net_income"] and not s["all_correct"]
    assert s["revenue_error"] == "prior_year_quarter"
    assert s["eps_diluted_error"] == "likely_non_gaap"
    s = e.score('{"period_end": "2024-09-28", "revenue": 1234000, "net_income": null, "eps_diluted": 0.97}', GOLD)
    assert s["revenue_error"] == "scale" and s["net_income_error"] == "missing"
    assert not e.score("I could not find it.", GOLD)["valid_json"]


def test_aggregate_counts_fields_and_errors():
    scores = [e.score(e.target_json(GOLD["label"]), GOLD), e.score('{"revenue": 3500000000}', GOLD)]
    agg = e.aggregate(scores)
    assert agg["accuracy"] == 0.5 and agg["revenue"] == 0.5 and agg["errors"]["year_to_date"] == 1


def test_build_records_end_to_end_with_a_fake_client():
    class FakeClient:
        def companyfacts(self, cik):
            return FACTS

        def earnings_8ks(self, cik, since):
            return edgar.earnings_8ks_from_submissions(SUBMISSIONS, since)

        def release_url(self, cik, acc):
            return "https://www.sec.gov/Archives/edgar/data/1/x/acme-ex991.htm"

        def release_text(self, url):
            return edgar.html_to_text(RELEASE_HTML)

    recs, reasons = e.build_records_for_company(FakeClient(), {"cik": 1, "ticker": "ACME", "title": "Acme"},
                                                "2019-01-01", 9000)
    assert reasons["kept"] == 1 and len(recs) == 1
    r = recs[0]
    assert json.loads(r["target"])["eps_diluted"] == 0.97
    assert e.score(r["target"], r)["all_correct"]


def test_company_split_is_stable_and_covers_all_splits():
    splits = {e.company_split(cik) for cik in range(1, 500)}
    assert splits == {"train", "val", "test"} and e.company_split(320193) == e.company_split(320193)
