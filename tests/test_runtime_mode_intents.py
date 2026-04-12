from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent_engine import os_intents
from core.intent_engine import runtime_mode_intents


def test_runtime_mode_intents_understand_m5_mode_names() -> None:
    result = runtime_mode_intents.check("switch to optimal mode")
    assert result is not None
    assert result.action_args["profile"] == "optimal"

    result = runtime_mode_intents.check("enable full performance mode")
    assert result is not None
    assert result.action_args["profile"] == "full_performance"


def test_performance_mode_intents_map_new_and_legacy_names() -> None:
    result = os_intents.check("switch to full performance mode")
    assert result is not None
    assert result.action_args["mode"] == "full_performance"

    result = os_intents.check("switch to auto mode")
    assert result is not None
    assert result.action_args["mode"] == "auto"

    result = os_intents.check("switch to lite mode")
    assert result is not None
    assert result.action_args["mode"] == "optimal"


def test_runtime_mode_intents_do_not_hijack_descriptive_questions() -> None:
    result = runtime_mode_intents.check(
        "give me a detailed answer about how optimal mode should differ from full performance mode",
    )
    assert result is None
