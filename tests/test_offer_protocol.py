"""ATOM -- regression suite for the Jarvis Offer Protocol (Sprint J).

Pins the contract Boss is paying for:

  Turn 1   Boss:  "How do I open Chrome?"
  Turn 1   ATOM:  "Open Launchpad and click Chrome.
                   Want me to open Chrome for you, Boss?"     <- offer
  Turn 2   Boss:  "yes"
  Turn 2   ATOM:  *executes open_app(Google Chrome)*           <- payoff

This file is the safety net that keeps that loop alive. It exercises:

  * ``OfferRegistry`` -- single slot, TTL, eviction, stats.
  * ``synthesize_offer`` -- the regex matrix that turns explainer
    queries into ``(action, args, offer_text)`` triples (and the
    negatives that must NOT trigger an offer).
  * ``LocalBrainController._append_offer_to_reply`` -- the gluing
    helper that adds the offer line to an LLM answer with the right
    punctuation and dedupes any LLM-emitted offer to avoid double-asks.
  * ``Router._maybe_consume_pending_offer`` -- the pre-classify hook
    that fires the staged action on a bare "yes", politely cancels on
    "no", and falls through on a topic switch.

If any of these break, ATOM regresses to chatbox mode and Boss notices
within one conversation. We catch it here first.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from core.cognitive.offer_synthesizer import (
    OfferProposal,
    is_explainer_query,
    synthesize_offer,
)
from core.router.offer_registry import (
    OfferRegistry,
    PendingOffer,
    get_offer_registry,
    reset_offer_registry,
)


# ── Section 1: OfferRegistry semantics ─────────────────────────────


class TestOfferRegistry:
    """Pin the single-slot, TTL-bounded behaviour."""

    def setup_method(self) -> None:
        # Each test gets a fresh registry so counters are predictable.
        self.reg = OfferRegistry(default_ttl_s=60.0)

    def test_empty_registry_has_no_pending(self) -> None:
        assert not self.reg.has_pending
        assert self.reg.peek() is None
        assert self.reg.consume() is None

    def test_stash_and_peek_returns_same_offer(self) -> None:
        offer = self.reg.stash(
            action="open_app",
            args={"app_name": "Chrome"},
            offer_text="Want me to open Chrome, Boss?",
            source_query="how do I open chrome",
        )
        peeked = self.reg.peek()
        assert peeked is offer
        # peek must NOT consume.
        assert self.reg.peek() is offer

    def test_consume_returns_offer_then_clears(self) -> None:
        self.reg.stash(action="weather", args={"city": "Mumbai"},
                       offer_text="Want the weather, Boss?")
        first = self.reg.consume()
        assert first is not None
        assert first.action == "weather"
        # Slot now empty.
        assert self.reg.consume() is None
        assert not self.reg.has_pending

    def test_new_stash_evicts_previous(self) -> None:
        self.reg.stash(action="open_app", args={"app_name": "Chrome"},
                       offer_text="Open Chrome?")
        self.reg.stash(action="weather", args={"city": "Mumbai"},
                       offer_text="Pull weather?")
        live = self.reg.peek()
        assert live is not None
        assert live.action == "weather"

    def test_clear_drops_offer_silently(self) -> None:
        self.reg.stash(action="lock_screen", args={}, offer_text="Lock?")
        assert self.reg.has_pending
        self.reg.clear(reason="topic-switch")
        assert not self.reg.has_pending

    def test_expired_offer_auto_evicts_on_peek(self) -> None:
        # 1 ms TTL -- guarantees expiry by the time we peek.
        reg = OfferRegistry(default_ttl_s=0.001)
        reg.stash(action="screenshot", args={}, offer_text="Snap?")
        time.sleep(0.005)
        assert reg.peek() is None
        assert not reg.has_pending

    def test_consume_returns_none_on_expired(self) -> None:
        reg = OfferRegistry(default_ttl_s=0.001)
        reg.stash(action="mute", args={}, offer_text="Mute?")
        time.sleep(0.005)
        assert reg.consume() is None

    def test_stash_rejects_blank_action(self) -> None:
        with pytest.raises(ValueError):
            self.reg.stash(action="", args={}, offer_text="x")

    def test_stash_rejects_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError):
            self.reg.stash(action="x", args={}, offer_text="y", ttl_s=0)
        with pytest.raises(ValueError):
            self.reg.stash(action="x", args={}, offer_text="y", ttl_s=-1)

    def test_args_are_copied_not_referenced(self) -> None:
        # Mutating the caller's dict must NOT affect the staged offer.
        original = {"app_name": "Chrome"}
        self.reg.stash(action="open_app", args=original, offer_text="?")
        original["app_name"] = "Tampered"
        live = self.reg.peek()
        assert live is not None
        assert live.args["app_name"] == "Chrome"

    def test_stats_reflect_lifecycle(self) -> None:
        s0 = self.reg.stats()
        assert s0["stashed_total"] == 0 and s0["consumed_total"] == 0
        self.reg.stash(action="weather", args={}, offer_text="?")
        s1 = self.reg.stats()
        assert s1["stashed_total"] == 1
        assert s1["has_pending"] is True
        assert s1["current_action"] == "weather"
        self.reg.consume()
        s2 = self.reg.stats()
        assert s2["consumed_total"] == 1
        assert s2["has_pending"] is False

    def test_singleton_get_offer_registry_is_stable(self) -> None:
        a = get_offer_registry()
        b = get_offer_registry()
        assert a is b

    def test_reset_offer_registry_returns_fresh(self) -> None:
        reset_offer_registry()
        a = get_offer_registry()
        a.stash(action="x", args={}, offer_text="y")
        b = reset_offer_registry()
        assert b is not a
        assert not b.has_pending

    def test_pending_offer_remaining_s_decreases(self) -> None:
        offer = PendingOffer(
            action="x", args={}, offer_text="y",
            expires_at=time.monotonic() + 5.0,
        )
        r1 = offer.remaining_s()
        time.sleep(0.01)
        r2 = offer.remaining_s()
        assert r2 < r1
        assert not offer.is_expired


# ── Section 2: Synthesizer pattern matrix ──────────────────────────


# Each row: (query, expected_action, expected_args_subset)
# args_subset is a partial match -- the proposal may include MORE keys
# than the minimum we assert on. Use empty dict to mean "no args required".
_POSITIVE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("how do I open chrome",                 "open_app",        {"app_name": "Google Chrome"}),
    ("how to open spotify",                  "open_app",        {"app_name": "Spotify"}),
    ("how can I launch vscode",              "open_app",        {"app_name": "Visual Studio Code"}),
    ("how do I close chrome",                "close_app",       {"app_name": "Google Chrome"}),
    ("what is the weather in mumbai",        "weather",         {"city": "mumbai"}),
    ("what's the weather like in delhi",     "weather",         {"city": "delhi"}),
    ("what's the weather",                   "weather",         {}),
    ("what's my battery level",              "resource_report", {}),
    ("how much ram am I using",              "resource_report", {}),
    ("what is my wifi status",               "wifi_status",     {}),
    ("am I connected to wifi",               "wifi_status",     {}),
    ("what's the news today",                "news_headlines",  {}),
    ("how do I take a screenshot",           "screenshot",      {}),
    ("how do I lock my screen",              "lock_screen",     {}),
    ("how to play music",                    "music_play",      {}),
    ("how do I increase the volume",         "set_volume",      {"level": 80}),
    ("how do I lower the volume",            "set_volume",      {"level": 30}),
    ("how do I mute my mac",                 "mute",            {}),
    ("how do I adjust the brightness",       "set_brightness",  {"level": 80}),
    ("what is on my screen",                 "screen_describe", {}),
    ("what do you see",                      "vision_describe", {}),
    ("how do I set a reminder for tea",      "set_reminder",    {"text": "tea"}),
    ("what's on my plate today",             "whats_on_my_plate", {}),
    ("what's my day looking like",           "daily_briefing",  {}),
    ("tell me about Tesla",                  "research_topic",  {"topic": "Tesla"}),
    ("what is quantum computing",            "research_topic",  {"topic": "quantum computing"}),
]


_NEGATIVE_CASES: list[str] = [
    "",                       # empty
    "play music",             # already an action verb, not an explainer
    "open chrome",            # already an action verb
    "set the volume to 50",   # action, not explainer
    "yes",                    # confirmation token
    "no thank you",           # contains "thank" but synthesizer requires explainer prefix
    "don't tell me about that",  # negative-intent hint short-circuits
    "no need to explain",        # negative-intent hint short-circuits
    "skip it for now",           # negative-intent hint
]


class TestSynthesizer:

    @pytest.mark.parametrize("query,expected_action,expected_args", _POSITIVE_CASES)
    def test_positive_matrix(
        self, query: str, expected_action: str, expected_args: dict[str, Any],
    ) -> None:
        # Long-enough response so skip_if_short_response doesn't trigger.
        proposal = synthesize_offer(query, response="x" * 30)
        assert proposal is not None, (
            f"Expected an offer for {query!r}, got None"
        )
        assert proposal.action == expected_action, (
            f"Wrong action for {query!r}: got {proposal.action!r}"
        )
        for k, v in expected_args.items():
            assert proposal.args.get(k) == v, (
                f"Wrong arg {k!r} for {query!r}: "
                f"got {proposal.args.get(k)!r}, expected {v!r}"
            )
        assert proposal.offer_text.endswith("?"), (
            f"Offer text must end with '?', got {proposal.offer_text!r}"
        )
        assert "Boss" in proposal.offer_text, (
            "Offer text must address Boss"
        )

    @pytest.mark.parametrize("query", _NEGATIVE_CASES)
    def test_negative_matrix(self, query: str) -> None:
        proposal = synthesize_offer(query, response="x" * 30)
        assert proposal is None, (
            f"Did NOT expect an offer for {query!r}, got {proposal}"
        )

    def test_short_response_skips_offer(self) -> None:
        # "ok" / "got it" style replies don't earn an offer follow-up.
        proposal = synthesize_offer(
            "how do I open chrome", response="ok",
            skip_if_short_response=True,
        )
        assert proposal is None

    def test_short_response_can_be_overridden(self) -> None:
        proposal = synthesize_offer(
            "how do I open chrome", response="ok",
            skip_if_short_response=False,
        )
        assert proposal is not None

    def test_offer_proposal_requires_action(self) -> None:
        with pytest.raises(ValueError):
            OfferProposal(action="", offer_text="?")

    def test_offer_proposal_requires_offer_text(self) -> None:
        with pytest.raises(ValueError):
            OfferProposal(action="open_app", offer_text="")

    def test_explainer_predicate(self) -> None:
        assert is_explainer_query("how do I open chrome")
        assert is_explainer_query("what is the weather")
        assert is_explainer_query("tell me about quantum")
        assert is_explainer_query("explain what is python")
        assert not is_explainer_query("open chrome")
        assert not is_explainer_query("yes")
        assert not is_explainer_query("")


# ── Section 3: _append_offer_to_reply (LLM glue) ───────────────────


class TestAppendOfferToReply:

    @pytest.fixture(autouse=True)
    def _import_method(self) -> None:
        from cursor_bridge.local_brain_controller import LocalBrainController
        self.glue = LocalBrainController._append_offer_to_reply

    def test_appends_offer_with_period_when_missing(self) -> None:
        out = self.glue("Open Launchpad and click Chrome",
                        "Want me to open Chrome, Boss?")
        assert out == "Open Launchpad and click Chrome. Want me to open Chrome, Boss?"

    def test_keeps_existing_terminal_punctuation(self) -> None:
        out = self.glue("Open Launchpad and click Chrome.",
                        "Want me to open Chrome, Boss?")
        assert out == "Open Launchpad and click Chrome. Want me to open Chrome, Boss?"

    def test_handles_question_mark_terminal(self) -> None:
        out = self.glue("Have you tried Spotlight?",
                        "Want me to do that, Boss?")
        # A question reply doesn't get a period spliced in.
        assert out == "Have you tried Spotlight? Want me to do that, Boss?"

    def test_skips_double_offer_when_llm_already_asked(self) -> None:
        # Model already proposed action -- don't echo a second time.
        body = "Open Launchpad and pin it. Want me to do that for you?"
        out = self.glue(body, "Want me to open Chrome, Boss?")
        assert out == body

    def test_skips_double_offer_should_i_phrasing(self) -> None:
        body = "Open Launchpad. Should I do that?"
        out = self.glue(body, "Want me to open Chrome, Boss?")
        assert out == body

    def test_empty_reply_returns_offer_alone(self) -> None:
        out = self.glue("", "Want me to do that, Boss?")
        assert out == "Want me to do that, Boss?"

    def test_empty_offer_returns_reply_alone(self) -> None:
        out = self.glue("Some answer.", "")
        assert out == "Some answer."


# ── Section 4: Router pre-classify offer hook ──────────────────────


class _RouterStub:
    """Duck-typed stand-in for ``Router`` -- only the surface
    ``_maybe_consume_pending_offer`` actually touches.

    We deliberately avoid constructing a full ``Router`` (which pulls
    in the brain, security policy, bus, etc.) because the contract we
    care about is local: given a pending offer + an utterance, do the
    right thing and don't touch anything else.
    """

    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []
        self.executed: list[Any] = []
        self.execute_should_raise: Exception | None = None

    def _emit_response(self, text: str, **kw: Any) -> None:
        self.emitted.append((text, kw))

    async def _execute_action(self, result: Any) -> None:
        if self.execute_should_raise is not None:
            raise self.execute_should_raise
        self.executed.append(result)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each router test starts with an empty singleton registry."""
    reset_offer_registry()
    yield
    reset_offer_registry()


class TestRouterOfferConsume:

    def setup_method(self) -> None:
        from core.router.router import Router
        self.method = Router._maybe_consume_pending_offer

    def _stash_offer(self, action: str = "open_app",
                     args: dict | None = None) -> PendingOffer:
        reg = get_offer_registry()
        return reg.stash(
            action=action,
            args=args or {"app_name": "Google Chrome"},
            offer_text="Want me to open Chrome, Boss?",
            source_query="how do I open chrome",
            source_response="Open Launchpad and click Chrome.",
        )

    def test_no_pending_returns_false(self) -> None:
        stub = _RouterStub()
        handled = asyncio.run(self.method(stub, "yes"))
        assert handled is False
        assert stub.emitted == []
        assert stub.executed == []

    def test_confirm_executes_staged_action(self) -> None:
        offer = self._stash_offer()
        stub = _RouterStub()
        handled = asyncio.run(self.method(stub, "yes"))
        assert handled is True
        assert len(stub.executed) == 1
        result = stub.executed[0]
        assert result.intent == "confirm_offer"
        assert result.action == offer.action
        assert result.action_args == offer.args
        # Offer was popped.
        assert not get_offer_registry().has_pending

    def test_deny_emits_polite_cancellation(self) -> None:
        self._stash_offer()
        stub = _RouterStub()
        handled = asyncio.run(self.method(stub, "no"))
        assert handled is True
        assert stub.executed == []
        assert len(stub.emitted) == 1
        text, _ = stub.emitted[0]
        assert "leaving it" in text.lower() or "cancelled" in text.lower()
        assert not get_offer_registry().has_pending

    def test_topic_switch_does_not_consume_offer(self) -> None:
        # User asks something completely unrelated -- the offer must
        # survive (TTL handles eviction) so we don't false-confirm.
        self._stash_offer()
        stub = _RouterStub()
        handled = asyncio.run(self.method(stub, "what time is it"))
        assert handled is False
        assert stub.executed == []
        assert stub.emitted == []
        # Offer is still alive.
        assert get_offer_registry().has_pending

    def test_dominant_confirmation_token_works(self) -> None:
        # "okay sure yes" should be treated as confirm even though it's
        # multi-word, via _is_confirm_dominant.
        self._stash_offer()
        stub = _RouterStub()
        handled = asyncio.run(self.method(stub, "okay sure yes"))
        assert handled is True
        assert len(stub.executed) == 1

    def test_hindi_confirmation_works(self) -> None:
        self._stash_offer()
        stub = _RouterStub()
        handled = asyncio.run(self.method(stub, "haan kar do"))
        assert handled is True
        assert len(stub.executed) == 1

    def test_execution_error_emits_recovery_message(self) -> None:
        self._stash_offer()
        stub = _RouterStub()
        stub.execute_should_raise = RuntimeError("dispatch boom")
        handled = asyncio.run(self.method(stub, "yes"))
        assert handled is True
        # Offer was still consumed (no zombie).
        assert not get_offer_registry().has_pending
        # User got a graceful recovery message instead of silence.
        assert len(stub.emitted) == 1
        text, _ = stub.emitted[0]
        assert "didn't go through" in text.lower() or "try again" in text.lower()

    def test_expired_offer_falls_through(self) -> None:
        # Stash with a microscopic TTL so it expires before the call.
        reg = reset_offer_registry(default_ttl_s=0.001)
        reg.stash(
            action="open_app", args={"app_name": "Chrome"},
            offer_text="Open?", source_query="how do I open chrome",
        )
        time.sleep(0.005)
        stub = _RouterStub()
        handled = asyncio.run(self.method(stub, "yes"))
        # Expired offer is invisible; "yes" gets no special handling.
        assert handled is False
        assert stub.executed == []


# ── Section 5: Persona contract carries the offer rule ─────────────


class TestPersonaContract:

    def test_system_prompt_mentions_proactive_offer(self) -> None:
        from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
        builder = StructuredPromptBuilder(config={
            "owner": {"name": "Satyam"},
            "brain": {"n_ctx": 8192, "max_tokens": 256},
            "developer": {},
        })
        sys_layer = builder._build_system_layer()
        # Cue words a downstream prompt-rule audit can grep for.
        assert "PROACTIVE OFFER" in sys_layer
        # The rule must NOT instruct the model to fabricate the offer
        # itself (the runtime synthesizer owns that copy).
        assert "runtime appends" in sys_layer
