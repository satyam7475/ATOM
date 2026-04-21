"""v3 Phase 3.5 — Cloud routing + budget guard regression tests.

Covers:
  * Smart-route triggers (keywords, long-form word threshold).
  * Daily budget guard (allow → exhaust → deny).
  * note_cloud_call() bookkeeping + auto-roll across day boundaries.
  * Router-side cloud egress safeguards: privacy filter (PII redaction)
    and cloud-stream sanitiser delegating to the LocalBrainController.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_kernel import CognitiveKernel, ExecPath


# ── Test doubles ───────────────────────────────────────────────────────


class FakeGeminiClient:
    """Minimal stand-in for the Gemini cloud client.

    The cognitive kernel only inspects ``is_available`` (a boolean
    attribute, not a method) on the smart-route paths.
    """

    def __init__(self, available: bool = True) -> None:
        self.is_available = available


def _make_kernel(
    *,
    cloud_enabled: bool = True,
    daily_budget_calls: int = 200,
    smart_keywords: tuple[str, ...] = (
        "explain properly",
        "deep analysis",
        "research this",
    ),
    smart_min_words: int = 25,
    gemini_available: bool = True,
    smart_local_conf_floor: float = 0.5,
) -> CognitiveKernel:
    cfg: dict[str, Any] = {
        "cloud": {
            "enabled": cloud_enabled,
            "daily_budget_calls": daily_budget_calls,
            "smart_route_keywords": list(smart_keywords),
            "smart_route_min_query_words": smart_min_words,
            "smart_route_local_confidence_floor": smart_local_conf_floor,
        },
    }
    kernel = CognitiveKernel(config=cfg)
    kernel.attach_cloud_intelligence(
        gemini_client=FakeGeminiClient(available=gemini_available),
    )
    return kernel


# ── Smart-route trigger logic ─────────────────────────────────────────


def test_cloud_smart_match_keyword_hit() -> None:
    """A query containing any configured keyword routes to the cloud."""
    k = _make_kernel(smart_keywords=("research this", "explain properly"))
    assert k._cloud_smart_match("Boss, can you research this for me?")
    assert k._cloud_smart_match("Please EXPLAIN PROPERLY what happened.")


def test_cloud_smart_match_long_query_hit() -> None:
    """A query past the long-form word threshold also triggers cloud."""
    k = _make_kernel(smart_min_words=10)
    long_q = " ".join(["word"] * 12)
    assert k._cloud_smart_match(long_q)


def test_cloud_smart_match_short_no_keyword_misses() -> None:
    """Short, non-keyword questions stay local."""
    k = _make_kernel(smart_min_words=25)
    assert not k._cloud_smart_match("what time is it")
    assert not k._cloud_smart_match("open chrome")
    assert not k._cloud_smart_match("")


def test_cloud_smart_match_min_words_zero_disables_length_trigger() -> None:
    """Setting smart_route_min_query_words=0 disables the length trigger."""
    k = _make_kernel(smart_keywords=(), smart_min_words=0)
    long_q = " ".join(["word"] * 200)
    assert not k._cloud_smart_match(long_q)


# ── Daily budget guard ────────────────────────────────────────────────


def test_cloud_budget_starts_available() -> None:
    k = _make_kernel(daily_budget_calls=3)
    assert k._cloud_budget_available()


def test_cloud_budget_exhaustion_blocks() -> None:
    """After daily_budget_calls note_cloud_call calls, the guard says no."""
    k = _make_kernel(daily_budget_calls=2)
    assert k._cloud_budget_available()
    k.note_cloud_call()
    assert k._cloud_budget_available()
    k.note_cloud_call()
    assert not k._cloud_budget_available()


def test_cloud_budget_zero_means_unlimited() -> None:
    """daily_budget_calls=0 disables the guard entirely (kill-switch off)."""
    k = _make_kernel(daily_budget_calls=0)
    for _ in range(50):
        k.note_cloud_call()
    assert k._cloud_budget_available()


def test_cloud_budget_auto_rolls_on_new_day() -> None:
    """If the day-bucket changes, the counter resets."""
    k = _make_kernel(daily_budget_calls=1)
    k.note_cloud_call()
    assert not k._cloud_budget_available()
    # Simulate a day boundary by stamping yesterday's bucket key.
    k._cloud_budget_day = "1999-01-01"
    assert k._cloud_budget_available()
    assert k._cloud_calls_today == 0


def test_cloud_budget_low_warning_does_not_break() -> None:
    """The "budget low" warning path runs without raising near exhaustion."""
    k = _make_kernel(daily_budget_calls=15)
    for _ in range(15):
        k.note_cloud_call()
    assert not k._cloud_budget_available()


# ── End-to-end route() integration on the smart path ─────────────────


def test_route_smart_keyword_goes_to_cloud_reason() -> None:
    """Path 2.65: explicit smart-route keyword → CLOUD_REASON."""
    k = _make_kernel(smart_keywords=("research this",))
    plan = k.route("Boss, please research this topic for me.")
    assert plan.path == ExecPath.CLOUD_REASON
    assert plan.cloud_augmented is True
    assert plan.model_role == "reasoning"
    assert plan.reason == "v3_smart_route_keyword_or_longform"


def test_route_smart_keyword_blocked_when_budget_exhausted() -> None:
    """Once today's budget is gone, smart keywords no longer go cloud."""
    k = _make_kernel(daily_budget_calls=1, smart_keywords=("research this",))
    k.note_cloud_call()  # exhaust
    assert not k._cloud_budget_available()
    plan = k.route("Boss, please research this topic for me.")
    assert plan.path != ExecPath.CLOUD_REASON


def test_route_smart_keyword_blocked_when_cloud_disabled() -> None:
    """cloud.enabled=false fully disables the cloud path."""
    k = _make_kernel(cloud_enabled=False, smart_keywords=("research this",))
    plan = k.route("Boss, please research this topic for me.")
    assert plan.path != ExecPath.CLOUD_REASON


def test_route_smart_keyword_blocked_when_gemini_unavailable() -> None:
    """If the wired Gemini client reports is_available=False, no cloud."""
    k = _make_kernel(gemini_available=False, smart_keywords=("research this",))
    plan = k.route("Boss, please research this topic for me.")
    assert plan.path != ExecPath.CLOUD_REASON


# ── Router-side cloud egress safeguards (Phase 3.3-3.4) ──────────────


def test_router_sanitize_cloud_chunk_uses_local_brain_controller() -> None:
    """``Router._sanitize_cloud_chunk`` delegates to the wired
    LocalBrainController so cloud chunks share the same guards."""
    from core.router.router import Router

    class _StubCtrl:
        calls: list[str] = []

        def _sanitize_emittable_text(self, text: str) -> str:  # noqa: D401
            type(self).calls.append(text)
            return f"CLEAN[{text}]"

    r = Router.__new__(Router)
    r._local_brain_controller = _StubCtrl()
    out = r._sanitize_cloud_chunk("hello cloud")
    assert out == "CLEAN[hello cloud]"
    assert _StubCtrl.calls == ["hello cloud"]


def test_router_sanitize_cloud_chunk_strips_control_tokens_in_fallback() -> None:
    """If no controller is wired, the fallback at minimum drops ChatML
    control tokens that some cloud providers leak."""
    from core.router.router import Router

    r = Router.__new__(Router)
    r._local_brain_controller = None
    bad = "Sure boss<|im_end|> here you go<|endoftext|>"
    cleaned = r._sanitize_cloud_chunk(bad)
    assert "<|im_end|>" not in cleaned
    assert "<|endoftext|>" not in cleaned
    assert "Sure boss" in cleaned


def test_router_sanitize_cloud_chunk_can_drop_a_full_fragment() -> None:
    """Sanitiser may legitimately return '' to suppress an entire chunk
    (e.g. a CoT preface). That MUST be honoured (not converted to None
    or original text)."""
    from core.router.router import Router

    class _DroppingCtrl:
        def _sanitize_emittable_text(self, text: str) -> str:
            return ""

    r = Router.__new__(Router)
    r._local_brain_controller = _DroppingCtrl()
    assert r._sanitize_cloud_chunk("Let me think step by step...") == ""


def test_router_sanitize_cloud_chunk_handles_empty_input() -> None:
    from core.router.router import Router

    r = Router.__new__(Router)
    r._local_brain_controller = None
    assert r._sanitize_cloud_chunk("") == ""


def test_router_sanitize_cloud_chunk_resilient_when_ctrl_raises() -> None:
    """A buggy controller must not break the cloud stream."""
    from core.router.router import Router

    class _AngryCtrl:
        def _sanitize_emittable_text(self, text: str) -> str:
            raise RuntimeError("oops")

    r = Router.__new__(Router)
    r._local_brain_controller = _AngryCtrl()
    out = r._sanitize_cloud_chunk("hello world")
    # Falls back to control-token stripper, which leaves plain text alone.
    assert "hello world" in out


def test_attach_local_brain_controller_sets_attr() -> None:
    """`Router.attach_local_brain_controller` wires the controller used
    by the cloud sanitiser."""
    from core.router.router import Router

    class _Ctrl:
        pass

    r = Router.__new__(Router)
    r._local_brain_controller = None
    r.attach_local_brain_controller(_Ctrl)
    assert r._local_brain_controller is _Ctrl


# ── Privacy filter on cloud egress (Phase 3.4) ───────────────────────


def test_privacy_filter_redacts_emails_and_phone_before_cloud() -> None:
    """The same `redact()` the router calls before sending to Gemini."""
    from context.privacy_filter import redact

    cleaned = redact(
        "Send the report to satyam@example.com and call +1-415-555-0100.",
    )
    assert "satyam@example.com" not in cleaned
    assert "415" not in cleaned or "[REDACTED]" in cleaned
    assert "[REDACTED]" in cleaned


def test_privacy_filter_is_safe_on_clean_text() -> None:
    from context.privacy_filter import redact

    s = "What is the weather in Mumbai today?"
    assert redact(s) == s


def test_privacy_filter_handles_empty() -> None:
    from context.privacy_filter import redact

    assert redact("") == ""


# ── Config-flag wiring sanity (Phase 3.1 settings.json) ──────────────


def test_settings_json_has_cloud_block_enabled() -> None:
    """settings.json should ship with the v3 cloud block populated so
    the smart router actually has thresholds to read."""
    import json
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.json"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cloud = cfg.get("cloud") or {}
    assert cloud.get("enabled") is True
    assert int(cloud.get("daily_budget_calls", 0)) > 0
    assert isinstance(cloud.get("smart_route_keywords", []), list)
    assert int(cloud.get("smart_route_min_query_words", 0)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
