"""Focused tests for Sprint A4: timeline recall intent + helpers."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


def _make_timeline(events: list[tuple[str, float]]):
    """Build an isolated TimelineMemory with the given (text, ts) events."""
    from core.memory import timeline_memory as tm_mod
    from core.memory.timeline_memory import TimelineMemory

    tmp = Path(tempfile.mkdtemp()) / "tl.json"
    with mock.patch.object(tm_mod, "_PERSIST_PATH", tmp):
        tl = TimelineMemory(max_events=500)
        for text, ts in events:
            tl.append_event("user_query", {"text": text}, timestamp=ts)
        return tl, tmp


class TimelineSearchTests(unittest.TestCase):
    def test_search_filters_by_keyword_and_window(self) -> None:
        now = time.time()
        tl, tmp = _make_timeline(
            [
                ("can you find last months invoice", now - 26 * 3600),
                ("how is my cpu doing", now - 5 * 3600),
                ("can we revisit the invoice", now - 10 * 60),
            ],
        )
        try:
            all_invoice = tl.search_user_queries("invoice")
            self.assertEqual(len(all_invoice), 2)

            recent_invoice = tl.search_user_queries(
                "invoice", since_ts=now - 3600,
            )
            self.assertEqual(len(recent_invoice), 1)
            self.assertIn("revisit", recent_invoice[0].data["text"])

            hits_blank = tl.search_user_queries("")
            self.assertEqual(len(hits_blank), 3)

            no_hits = tl.search_user_queries("zebra")
            self.assertEqual(no_hits, [])
        finally:
            tmp.unlink(missing_ok=True)

    def test_recall_summary_describes_latest(self) -> None:
        now = time.time()
        tl, tmp = _make_timeline(
            [("remind me about that billing thing", now - 4 * 3600)],
        )
        try:
            msg = tl.recall_user_queries_summary("billing")
            self.assertIn("billing", msg)
            self.assertIn("once", msg.lower())
            self.assertIn("remind me about that billing thing", msg)
        finally:
            tmp.unlink(missing_ok=True)


class MemoryRecallIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        from core.intent_engine import memory_recall_intents as mri

        self.mri = mri
        now = time.time()
        self.tl, self.tmp = _make_timeline(
            [
                ("can you find last months invoice", now - 26 * 3600),
                ("what time is it", now - 24 * 3600),
                ("how is my cpu looking", now - 5 * 3600),
                ("remind me about that billing thing", now - 4 * 3600),
                ("should i deploy today", now - 30 * 60),
                ("can we revisit the invoice?", now - 10 * 60),
            ],
        )
        mri.set_timeline(self.tl)

    def tearDown(self) -> None:
        self.mri.set_timeline(None)
        self.tmp.unlink(missing_ok=True)

    def test_matches_about_X_with_scope(self) -> None:
        r = self.mri.check("what did i ask yesterday about billing?")
        self.assertIsNotNone(r)
        self.assertEqual(r.intent, "memory_recall")
        self.assertIn("billing", r.response.lower())

    def test_matches_did_i_mention_short_form(self) -> None:
        r = self.mri.check("did i mention the invoice this morning")
        self.assertIsNotNone(r)
        self.assertEqual(r.intent, "memory_recall")
        self.assertIn("invoice", r.response.lower())

    def test_matches_topicless_what_we_talked(self) -> None:
        r = self.mri.check("remind me what we talked about today")
        self.assertIsNotNone(r)
        self.assertEqual(r.intent, "memory_recall")
        self.assertIn("today", r.response.lower())

    def test_non_recall_query_returns_none(self) -> None:
        r = self.mri.check("what's the cpu doing right now")
        self.assertIsNone(r)

    def test_handler_safe_when_timeline_unset(self) -> None:
        self.mri.set_timeline(None)
        r = self.mri.check("what did i ask yesterday about billing")
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
