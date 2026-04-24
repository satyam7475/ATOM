"""iPhone-driven hints: presence + named triggers.

Pure unit tests over :py:class:`core.proactive_awareness.ProactiveAwareness`.
The bus/wiring layer is covered by ``test_bridge_event_wiring``.

Run: ``python3 -m pytest tests/test_iphone_proactive.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.proactive_awareness import ProactiveAwareness  # noqa: E402


def _pa(*, enabled: bool = True) -> ProactiveAwareness:
    return ProactiveAwareness(config={"features": {"proactive_awareness": enabled}})


# ────────────────────────────────────────────
# presence
# ────────────────────────────────────────────

def test_presence_at_desk_greets_boss() -> None:
    pa = _pa()
    hint = pa.handle_iphone_presence("at_desk")
    assert hint is not None
    assert "Boss" in hint


def test_presence_disabled_returns_none() -> None:
    pa = _pa(enabled=False)
    assert pa.handle_iphone_presence("at_desk") is None


def test_presence_unknown_state_returns_none() -> None:
    pa = _pa()
    assert pa.handle_iphone_presence("dancing") is None
    assert pa.handle_iphone_presence("") is None


def test_presence_same_state_cooldown_blocks_duplicate() -> None:
    pa = _pa()
    h1 = pa.handle_iphone_presence("at_desk")
    h2 = pa.handle_iphone_presence("at_desk")
    assert h1 is not None
    assert h2 is None, "same-state within cooldown must not re-fire"


def test_presence_different_state_bypasses_cooldown() -> None:
    pa = _pa()
    assert pa.handle_iphone_presence("at_desk") is not None
    assert pa.handle_iphone_presence("leaving") is not None


def test_presence_at_desk_counts_as_daily_greeting() -> None:
    pa = _pa()
    pa.handle_iphone_presence("at_desk")
    # check_greeting should now stay quiet because Boss was already
    # greeted via iPhone this day.
    assert pa.check_greeting() is None


def test_presence_is_case_and_whitespace_insensitive() -> None:
    pa = _pa()
    assert pa.handle_iphone_presence("  At_Desk  ") is not None


# ────────────────────────────────────────────
# triggers
# ────────────────────────────────────────────

def test_trigger_morning_routine_returns_envelope() -> None:
    pa = _pa()
    envelope = pa.handle_iphone_trigger("morning_routine")
    assert envelope is not None
    assert envelope["trigger"] == "morning_routine"
    assert "morning" in envelope["ack"].lower()


def test_trigger_unknown_name_uses_generic_ack() -> None:
    pa = _pa()
    envelope = pa.handle_iphone_trigger("call_friend")
    assert envelope is not None
    assert envelope["trigger"] == "call_friend"
    assert "call friend" in envelope["ack"].lower()


def test_trigger_empty_name_returns_none() -> None:
    pa = _pa()
    assert pa.handle_iphone_trigger("") is None
    assert pa.handle_iphone_trigger("   ") is None


def test_trigger_disabled_engine_returns_none() -> None:
    pa = _pa(enabled=False)
    assert pa.handle_iphone_trigger("morning_routine") is None


def test_trigger_same_name_cooldown_blocks_duplicate() -> None:
    pa = _pa()
    e1 = pa.handle_iphone_trigger("focus_on")
    e2 = pa.handle_iphone_trigger("focus_on")
    assert e1 is not None
    assert e2 is None


def test_trigger_different_names_do_not_share_cooldown() -> None:
    pa = _pa()
    assert pa.handle_iphone_trigger("focus_on") is not None
    assert pa.handle_iphone_trigger("focus_off") is not None
    assert pa.handle_iphone_trigger("morning_routine") is not None


def test_trigger_args_accepted_but_ignored_for_now() -> None:
    pa = _pa()
    envelope = pa.handle_iphone_trigger(
        "morning_routine",
        args={"verbose": True, "include_weather": False},
    )
    assert envelope is not None
    assert envelope["trigger"] == "morning_routine"


def test_trigger_case_insensitive() -> None:
    pa = _pa()
    envelope = pa.handle_iphone_trigger("  Morning_Routine  ")
    assert envelope is not None
    assert envelope["trigger"] == "morning_routine"
