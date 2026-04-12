from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.query_policy import (  # noqa: E402
    ResponseMode,
    classify_response_mode,
    detect_response_language,
    should_export_report,
    summarize_report,
)
from core.quick_replies import try_quick_reply  # noqa: E402


def test_response_mode_classification() -> None:
    assert classify_response_mode("how are you") is ResponseMode.SHORT
    assert classify_response_mode("what is docker") is ResponseMode.SHORT
    assert classify_response_mode("explain properly what is docker") is ResponseMode.DETAIL
    assert classify_response_mode("research best browser for coding on mac") is ResponseMode.REPORT
    assert classify_response_mode("mujhe short me batao unified memroy kya hota hai") is ResponseMode.SHORT
    assert classify_response_mode("samjhao proper unified memroy kya hai") is ResponseMode.DETAIL


def test_quick_reply_skips_explicit_detail_requests() -> None:
    assert try_quick_reply("who are you", {}) is not None
    assert try_quick_reply("who are you explain properly", {}) is None
    assert try_quick_reply("how are yuo", {}) is not None
    assert try_quick_reply("atom mujhe ek chota joke batao", {}) is not None
    assert try_quick_reply("compare safari and arc for coding on a macbook air", {}) is not None
    assert try_quick_reply("hinglish me samjha do ki optimal aur full performance me farak kya hai", {}) is not None


def test_quick_reply_does_not_match_longer_reasoning_question() -> None:
    cfg = {
        "assistant_brain": {
            "quick_replies_enabled": True,
            "quick_replies": {
                "what should i do": "Check your goals, Boss. Say 'show goals' to see what's pending.",
            },
        },
    }
    assert (
        try_quick_reply(
            "if i have 90 minutes and tasks of 30 45 and 60 minutes what should i do first",
            cfg,
        )
        is None
    )


def test_report_helpers() -> None:
    report = (
        "Safari is best for battery life and text rendering. "
        "Chrome and Arc win on extension compatibility and dev tools. "
        "If you want the best balance on a MacBook Air, Safari or Arc are usually "
        "the strongest picks depending on workflow. "
        "For long coding sessions, memory pressure and battery life matter as much "
        "as extension depth."
    )

    summary = summarize_report(report, max_sentences=2, max_chars=120)
    assert "Safari is best for battery life" in summary
    assert len(summary) <= 123
    assert should_export_report("research best browser for coding on mac", report) is True
    assert should_export_report("what is docker", report) is False


def test_response_language_detection() -> None:
    assert detect_response_language("reply in english", previous="hinglish") == "english"
    assert detect_response_language("hinglish me samjha do", previous="english") == "hinglish"
    assert detect_response_language("mujhe batao cpu spike kyu hota hai", previous="english") == "hinglish"
    assert detect_response_language("aur isme kya", previous="hinglish") == "hinglish"
