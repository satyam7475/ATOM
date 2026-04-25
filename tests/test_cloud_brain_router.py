"""Sprint M1 -- confidence-gated cloud brain router."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.cloud.cloud_brain_router import (
    CloudBrainConfig,
    CloudBrainRouter,
    CloudDecision,
    CloudResult,
)

# Async tests inside this file declare pytest.mark.asyncio individually
# so we don't trip the global-marker warning on the synchronous tests.


class _StubGemini:
    is_available = True

    def __init__(self, *, reply: tuple[str, bool] = ("the cloud answer", True)) -> None:
        self._reply = reply
        self.calls: list[tuple[str, str | None]] = []

    async def ask_reasoning(self, query: str, *, system_instruction: str | None = None) -> tuple[str, bool]:
        self.calls.append(("reasoning", query))
        return self._reply

    async def ask_buddy(self, query: str, *, system_instruction: str | None = None) -> tuple[str, bool]:
        self.calls.append(("buddy", query))
        return self._reply


# ── classify -------------------------------------------------------


def test_classify_blocks_when_no_provider() -> None:
    r = CloudBrainRouter()
    assert r.classify("think hard about this").use_cloud is False
    assert r.classify("hello").use_cloud is False


def test_classify_routes_deep_prefix() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini())
    out = r.classify("think hard: should I refactor router.py?")
    assert isinstance(out, CloudDecision)
    assert out.use_cloud and out.profile == "deep"
    assert out.reason == "deep_prefix"


def test_classify_routes_explicit_deep_token() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini())
    assert r.classify("deep: explain monad transformers").use_cloud is True


def test_classify_blocks_light_intents() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini())
    for q in ("hi", "thanks", "play music", "open chrome"):
        assert r.classify(q).use_cloud is False, q


def test_classify_routes_long_reasoning_question() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini())
    long_q = (
        "Compare swift and python for high-performance signal "
        "processing on Apple Silicon, including SIMD differences"
    )
    out = r.classify(long_q)
    assert out.use_cloud is True
    assert out.reason == "reasoning_pattern"


def test_classify_routes_after_local_failure() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini())
    out = r.classify("what time is it", local_response="", local_failed=True)
    assert out.use_cloud is True


def test_classify_quota_exhausted_blocks() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini(), config=CloudBrainConfig(daily_quota=1))
    r._stats.requests_today = 1
    out = r.classify("think hard about this")
    assert out.use_cloud is False
    assert out.reason == "daily_quota_exhausted"


def test_classify_records_decisions_metrics() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini())
    r.classify("hi")
    r.classify("think hard about this")
    assert any(k.startswith("local:") for k in r.stats()["decisions"])
    assert any(k.startswith("cloud:") for k in r.stats()["decisions"])


# ── escalate -------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_escalate_returns_cloud_text() -> None:
    g = _StubGemini(reply=("Friday-class answer", True))
    r = CloudBrainRouter(gemini_client=g)
    res = await r.maybe_escalate("think hard about this", deep_hint=True)
    assert isinstance(res, CloudResult)
    assert res.text == "Friday-class answer"
    assert res.provider == "gemini"
    assert res.fallback_to_local is False


@pytest.mark.asyncio
async def test_maybe_escalate_returns_none_when_classifier_blocks() -> None:
    r = CloudBrainRouter(gemini_client=_StubGemini())
    assert await r.maybe_escalate("hi") is None


@pytest.mark.asyncio
async def test_maybe_escalate_marks_failure_when_cloud_returns_empty() -> None:
    g = _StubGemini(reply=("", False))
    r = CloudBrainRouter(gemini_client=g)
    res = await r.maybe_escalate("deep: very deep question")
    assert res is not None
    assert res.text == ""
    assert res.fallback_to_local is True


@pytest.mark.asyncio
async def test_maybe_escalate_handles_provider_exception() -> None:
    class _Boom:
        is_available = True

        async def ask_reasoning(self, *_args: Any, **_kw: Any) -> tuple[str, bool]:
            raise RuntimeError("network exploded")

        async def ask_buddy(self, *_args: Any, **_kw: Any) -> tuple[str, bool]:
            raise RuntimeError("network exploded")

    r = CloudBrainRouter(gemini_client=_Boom())
    res = await r.maybe_escalate("deep: explain quantum")
    assert res is not None
    assert res.fallback_to_local is True
    assert "network exploded" in (res.error or "")


def test_fallback_only_mode_skips_proactive_escalation() -> None:
    g = _StubGemini()
    r = CloudBrainRouter(gemini_client=g, config=CloudBrainConfig(fallback_only=True))
    long_q = (
        "Compare swift and python for high-performance signal "
        "processing on Apple Silicon, including SIMD differences"
    )
    out = r.classify(long_q)
    assert out.use_cloud is False
    assert out.reason == "fallback_only_mode"


def test_fallback_only_mode_still_escalates_after_local_failure() -> None:
    g = _StubGemini()
    r = CloudBrainRouter(gemini_client=g, config=CloudBrainConfig(fallback_only=True))
    out = r.classify("anything", local_response="", local_failed=True)
    assert out.use_cloud is True
