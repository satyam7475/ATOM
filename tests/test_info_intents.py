from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent_engine import info_intents


def test_date_and_top_process_queries_match_info_intents() -> None:
    date_result = info_intents.check("what date is it")
    assert date_result is not None
    assert date_result.intent == "date"

    top_result = info_intents.check("top processes")
    assert top_result is not None
    assert top_result.intent == "top_processes"
