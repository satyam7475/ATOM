"""
ATOM -- Sprint D3 focused tests.

Covers:
    1. NL parser recognizes kind + time scope + keywords.
    2. mdfind query compiler produces valid predicate expressions.
    3. Intent matcher triggers smart_find_file for realistic phrasings.
    4. ``find_files_for_text`` composes a spoken summary from a stubbed
       Spotlight engine.
"""

from __future__ import annotations

from unittest.mock import patch

from core.intent_engine import file_intents
from core.proactive.file_finder import (
    FileQuery,
    build_mdfind_query,
    find_files_for_text,
    parse_file_query,
)


def test_parse_invoice_pdf_last_week() -> None:
    q = parse_file_query("find the invoice PDF from last week")
    assert q.kind == "pdf"
    assert q.since_days == 14
    assert "invoice" in q.keywords
    assert "last" not in q.keywords
    assert "week" not in q.keywords


def test_parse_screenshot_yesterday() -> None:
    q = parse_file_query("find screenshots from yesterday")
    assert q.kind == "screenshot"
    assert q.since_days == 2


def test_parse_generic_pdfs_this_month() -> None:
    q = parse_file_query("show me PDFs from this month")
    assert q.kind == "pdf"
    assert q.since_days == 30
    assert q.keywords == []


def test_parse_n_days_ago() -> None:
    q = parse_file_query("find the receipt from 3 days ago")
    assert q.since_days is not None and q.since_days >= 3
    assert "receipt" in q.keywords


def test_parse_no_scope_returns_just_keywords() -> None:
    q = parse_file_query("find the ATOM architecture diagram")
    assert q.since_days is None
    assert "atom" in q.keywords or "architecture" in q.keywords


def test_build_mdfind_query_includes_predicates() -> None:
    q = FileQuery(keywords=["invoice"], kind="pdf", since_days=7)
    md = build_mdfind_query(q)
    assert "com.adobe.pdf" in md
    assert "kMDItemFSContentChangeDate" in md
    assert "$time.iso(-7d)" in md
    assert "invoice" in md


def test_build_mdfind_query_keyword_only() -> None:
    q = FileQuery(keywords=["quarterly-report"])
    md = build_mdfind_query(q)
    assert "quarterly-report" in md
    assert "$time.iso" not in md


def test_build_mdfind_query_empty() -> None:
    assert build_mdfind_query(FileQuery()) == ""


def test_intent_matches_invoice_pdf_last_week() -> None:
    res = file_intents.check("find the invoice PDF from last week")
    assert res is not None
    assert res.intent == "smart_find_file"
    assert res.action == "smart_find_file"
    assert "invoice" in (res.action_args or {}).get("query", "").lower()


def test_intent_matches_screenshots_yesterday() -> None:
    res = file_intents.check("find screenshots from yesterday")
    assert res is not None
    assert res.intent == "smart_find_file"


def test_intent_plain_find_falls_through_when_no_kind_or_time() -> None:
    res = file_intents.check("find that file I saved")
    assert res is None or res.intent != "smart_find_file"


def test_find_files_summary_with_hits() -> None:
    hits = [
        {"path": "/Users/satyam/Documents/invoice_april.pdf"},
        {"path": "/Users/satyam/Downloads/old_invoice.pdf"},
    ]

    class _Engine:
        def search(self, *_a, **_kw):
            return hits

    with patch(
        "core.proactive.file_finder.SpotlightEngine", create=True, new=None,
    ):
        pass

    with patch("core.macos.spotlight_engine.SpotlightEngine", return_value=_Engine()):
        summary, paths = find_files_for_text(
            "find the invoice PDF from last week",
            limit=5,
        )

    assert "invoice_april.pdf" in summary
    assert len(paths) == 2
    assert "Found 2" in summary or "Found two" in summary.lower()


def test_find_files_summary_no_hits() -> None:
    class _Engine:
        def search(self, *_a, **_kw):
            return []

    with patch("core.macos.spotlight_engine.SpotlightEngine", return_value=_Engine()):
        summary, paths = find_files_for_text(
            "find the invoice PDF from last week", limit=5,
        )

    assert paths == []
    assert "couldn't" in summary.lower() or "no" in summary.lower()


def test_find_files_for_empty_text() -> None:
    summary, paths = find_files_for_text("   ", limit=5)
    assert paths == []
    assert len(summary) > 0


if __name__ == "__main__":
    test_parse_invoice_pdf_last_week()
    test_parse_screenshot_yesterday()
    test_parse_generic_pdfs_this_month()
    test_parse_n_days_ago()
    test_parse_no_scope_returns_just_keywords()
    test_build_mdfind_query_includes_predicates()
    test_build_mdfind_query_keyword_only()
    test_build_mdfind_query_empty()
    test_intent_matches_invoice_pdf_last_week()
    test_intent_matches_screenshots_yesterday()
    test_intent_plain_find_falls_through_when_no_kind_or_time()
    test_find_files_summary_with_hits()
    test_find_files_summary_no_hits()
    test_find_files_for_empty_text()
    print("[D3] All file finder tests passed.")
