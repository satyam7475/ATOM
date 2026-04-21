"""Regression tests for System Control v1 — Phase A + B + C.

Covers:
  1. New voice regexes classify to the right action
  2. Every newly-wired action name is dispatchable (router _ACTION_DISPATCH
     / _LATE_DISPATCH) or handled through the ``intent_classified`` bus
     by a known wiring module.
  3. Security tiers match expected risk bands (read-only ≤ 1, destructive ≥ 4).
  4. Persistent SystemProfile bootstraps and persists; structured prompt
     builder injects a ``[MACHINE] …`` line into the context layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.intent_engine import IntentEngine
from core.router.router import Router
from core.security_tiers import (
    action_tier,
    is_escalatable,
    max_tier_for_security_mode,
    tier_allowed,
)
from core.system_profile import SystemProfile


# ── Phase A.2: voice regexes ────────────────────────────────────────


_INTENT_CASES: list[tuple[str, str]] = [
    ("is chrome running", "find_process_by_name"),
    ("find process called spotify", "find_process_by_name"),
    ("show me details for pid 4521", "get_process_details"),
    ("make pid 1234 high priority", "set_process_priority"),
    ("slow down pid 4321", "set_process_priority"),
    ("boost pid 8800", "set_process_priority"),
    ("show me the open ports", "get_open_ports"),
    ("what ports are listening", "get_open_ports"),
    ("scan wifi networks", "get_wifi_networks"),
    ("show nearby wifi", "get_wifi_networks"),
    ("find the biggest files", "find_large_files"),
    ("analyze temp files", "analyze_temp_files"),
    ("how much junk do I have", "analyze_temp_files"),
    ("optimize for atom", "optimize_for_atom"),
    ("free up ram for yourself", "optimize_for_atom"),
    ("describe the focused element", "describe_focused_element"),
    ("what is focused right now", "describe_focused_element"),
    ("what has focus", "describe_focused_element"),
    ("read the focused field", "read_focused_text"),
    ("type hello into the focused field", "set_focused_text"),
    ("fill the focused field with world", "set_focused_text"),
    ("click the submit button", "click_ui_element"),
    ("press the cancel link", "click_ui_element"),
]


@pytest.mark.parametrize("text,expected_action", _INTENT_CASES)
def test_new_voice_intents_classify(text: str, expected_action: str) -> None:
    engine = IntentEngine()
    result = engine.classify(text)
    assert result.action == expected_action, (
        f"'{text}' -> intent={result.intent} action={result.action} "
        f"(expected action={expected_action})"
    )


# ── Phase A reachability: dispatchable or bus-handled ──────────────


_BUS_HANDLED_INTENTS = frozenset({
    "diagnose_failure", "fix_it", "fix_all", "module_health",
    "read_own_code", "explain_module", "search_code",
    "security_status", "security_lockdown",
    "voice_enroll", "voice_verify", "voice_reset", "voice_auth_status",
    "behavior_auth_status",
    "weather_report", "news_headlines", "world_clock", "daily_briefing",
    "temporal_info", "world_status",
    "goal_create", "goal_show", "goal_progress", "goal_decompose",
    "goal_log_progress", "goal_complete_step", "goal_pause",
    "goal_resume", "goal_abandon",
    "prediction", "mode_switch",
    "cognitive_behavior_report", "scheduling_advice",
    "brain_remember", "brain_recall", "brain_preferences",
    "self_optimize",
})


_SYSTEM_CONTROL_V1_ACTIONS = frozenset({
    "find_process_by_name", "get_process_details", "set_process_priority",
    "get_open_ports", "get_wifi_networks",
    "find_large_files", "analyze_temp_files", "optimize_for_atom",
    "describe_focused_element", "read_focused_text", "set_focused_text",
    "click_ui_element",
})


def test_system_control_v1_actions_all_dispatchable() -> None:
    """Every new v1 action must resolve via _ACTION_DISPATCH or _LATE_DISPATCH."""
    action_table = dict(Router._ACTION_DISPATCH)
    late_table = dict(Router._LATE_DISPATCH)
    missing: list[str] = []
    for a in _SYSTEM_CONTROL_V1_ACTIONS:
        if a not in action_table and a not in late_table:
            missing.append(a)
    assert not missing, (
        "These System Control v1 actions have no router dispatch entry: "
        f"{missing}"
    )


# ── Phase C: security tier assignments ─────────────────────────────


_TIER_EXPECTATIONS: list[tuple[str, int]] = [
    ("find_process_by_name", 1),
    ("get_process_details", 1),
    ("get_open_ports", 1),
    ("get_wifi_networks", 1),
    ("find_large_files", 1),
    ("analyze_temp_files", 1),
    ("describe_focused_element", 1),
    ("read_focused_text", 1),
    ("set_focused_text", 3),
    ("click_ui_element", 3),
    ("set_process_priority", 4),
    ("optimize_for_atom", 4),
    ("security_lockdown", 4),
]


@pytest.mark.parametrize("action,expected_tier", _TIER_EXPECTATIONS)
def test_security_tier_for_action(action: str, expected_tier: int) -> None:
    assert action_tier(action) == expected_tier, (
        f"{action}: expected tier {expected_tier}, got {action_tier(action)}"
    )


def test_read_only_actions_allowed_in_strict_mode() -> None:
    cap = max_tier_for_security_mode("strict")
    for a in (
        "find_process_by_name", "get_open_ports", "get_wifi_networks",
        "find_large_files", "analyze_temp_files",
        "describe_focused_element", "read_focused_text",
    ):
        ok, reason = tier_allowed(a, cap)
        assert ok, f"{a} should be allowed in strict mode ({reason})"


def test_power_actions_escalatable_in_strict_mode() -> None:
    cap = max_tier_for_security_mode("strict")
    for a in ("set_process_priority", "optimize_for_atom", "security_lockdown"):
        ok, _ = tier_allowed(a, cap)
        assert not ok, f"{a} should require escalation in strict mode"
        assert is_escalatable(a, cap), (
            f"{a} must be marked escalatable so the UX can prompt Boss"
        )


# ── Phase B: SystemProfile ─────────────────────────────────────────


def test_system_profile_bootstrap_without_scanner(tmp_path: Path) -> None:
    sp = SystemProfile(config={}, scanner=None,
                       path=tmp_path / "system_profile.json")
    compact = sp.get_compact_context()
    assert compact.startswith("[MACHINE]")
    assert len(compact) <= 230
    assert (tmp_path / "system_profile.json").exists()


def test_system_profile_refresh_from_scanner(tmp_path: Path) -> None:
    path = tmp_path / "system_profile.json"

    class FakeScanner:
        last_scan = {
            "system": {
                "os": "macOS 15.0",
                "cpu": "Apple M5",
                "ram_total_gb": 16.0,
                "ram_available_gb": 6.2,
            },
            "disks": [
                {"mount": "/", "total_gb": 460.0, "free_gb": 128.5,
                 "percent_used": 72.1},
            ],
            "health": {"overall": 82},
        }

    sp = SystemProfile(config={}, scanner=FakeScanner(), path=path)
    sp.invalidate()
    compact = sp.get_compact_context()

    assert compact.startswith("[MACHINE]")
    assert "Apple M5" in compact
    assert "16" in compact  # RAM total
    assert "460" in compact  # Disk total
    assert "82" in compact  # health score

    persisted = json.loads(path.read_text())
    assert persisted["machine"]["cpu"] == "Apple M5"
    assert persisted["storage"]["free_gb"] == 128.5


def test_structured_prompt_injects_machine_line(tmp_path: Path) -> None:
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    class FakeScanner:
        last_scan = {
            "system": {"os": "macOS 15.0", "cpu": "Apple M5", "ram_total_gb": 16},
            "disks": [{"mount": "/", "total_gb": 460, "free_gb": 128}],
            "health": {"overall": 82},
        }

    sp = SystemProfile(config={}, scanner=FakeScanner(),
                       path=tmp_path / "sp.json")

    pb = StructuredPromptBuilder(config={"owner_name": "Boss", "project": "ATOM"})
    pb.set_system_profile_provider(sp)

    prompt = pb.build(
        "what time is it",
        memory_summaries=[],
        document_context=None,
        context={"active_app": "Cursor"},
    )
    assert "[MACHINE]" in prompt, (
        "StructuredPromptBuilder must inject the [MACHINE] line "
        "into the context layer when a system profile provider is set."
    )
    assert "Apple M5" in prompt


def test_prompt_builder_without_profile_still_works() -> None:
    """No profile provider wired → build() must still return a prompt."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
    pb = StructuredPromptBuilder(config={"owner_name": "Boss", "project": "ATOM"})
    prompt = pb.build(
        "hello",
        memory_summaries=[],
        document_context=None,
        context=None,
    )
    assert "BOSS:" in prompt
    assert "[MACHINE]" not in prompt


# ── Regression guard: existing intents must still classify ─────────


_EXISTING_INTENTS: list[tuple[str, str]] = [
    ("kill process firefox", "kill_process"),
    ("what time is it", None),  # info intent, no action
    ("scroll down", "scroll_down"),
    ("lock it", "lock_screen"),
    ("take a screenshot", "screenshot"),
    ("remember that I like coffee", "brain_remember"),
    ("show my goals", "goal_show"),
    ("security status", "security_status"),
    ("enroll my voice", "voice_enroll"),
]


@pytest.mark.parametrize("text,expected_action", _EXISTING_INTENTS)
def test_existing_intents_regression(text: str, expected_action: str | None) -> None:
    engine = IntentEngine()
    result = engine.classify(text)
    if expected_action is None:
        assert result.intent != "fallback", f"'{text}' should not fall back to LLM"
    else:
        assert result.action == expected_action, (
            f"'{text}' -> action={result.action}, expected {expected_action}"
        )
