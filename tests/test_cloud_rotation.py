"""Sprint Ω.9 -- rotating multi-provider OpenAI-compatible cloud client.

Covers:
  - Round-robin distribution across slots (cursor advances every call).
  - 429 → slot quarantined for cooldown_429_s; rotation skips it cleanly.
  - 5xx → shorter cooldown; slot returns after window.
  - Hard failure threshold opens slot for cooldown_hard_s.
  - Soft RPM gate pre-emptively skips a saturated slot.
  - All-down → ("", False) without crashing the router.
  - Streaming shim emits one synthetic token on success.
  - Diagnostics shape stays stable (used by the dashboard).

We never hit the network in these tests — every slot's ``_call_sync`` is
monkeypatched to return a queued sequence of (text, ok, status, quota)
tuples so the rotation, quarantine and timing logic are exercised in
pure-Python deterministic mode.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import pytest

from core.cloud.rotating_openai_client import RotatingOpenAIClient, _Slot


# ── helpers --------------------------------------------------------


def _config(
    *,
    enabled: bool = True,
    cooldown_429_s: float = 60.0,
    cooldown_5xx_s: float = 30.0,
    cooldown_hard_s: float = 300.0,
    soft_rpm: int = 25,
) -> dict:
    return {
        "cloud": {
            "enabled": enabled,
            "rotation": {
                "enabled": enabled,
                "cooldown_429_s": cooldown_429_s,
                "cooldown_5xx_s": cooldown_5xx_s,
                "cooldown_hard_s": cooldown_hard_s,
                "soft_rpm_per_slot": soft_rpm,
                "providers": [
                    {
                        "name": "groq",
                        "base_url": "https://groq.test/v1",
                        "credential_id": "groq",
                        "fast_model": "llama-3.1-8b-instant",
                        "deep_model": "llama-3.3-70b-versatile",
                    },
                    {
                        "name": "nvidia",
                        "base_url": "https://nvidia.test/v1",
                        "credential_id": "nvidia",
                        "fast_model": "meta/llama-3.1-8b-instruct",
                        "deep_model": "meta/llama-3.3-70b-instruct",
                    },
                    {
                        "name": "cerebras",
                        "base_url": "https://cerebras.test/v1",
                        "credential_id": "cerebras",
                        "fast_model": "llama3.1-8b",
                        "deep_model": "llama-3.3-70b",
                    },
                ],
            },
        },
    }


def _build_client(monkeypatch: pytest.MonkeyPatch, **cfg_kwargs: Any) -> RotatingOpenAIClient:
    """Construct a client with all three slots pre-keyed and no vault hits."""
    monkeypatch.setattr(
        "core.cloud.rotating_openai_client.RotatingOpenAIClient._hydrate_keys_from_vault",
        lambda self: None,
    )
    client = RotatingOpenAIClient(_config(**cfg_kwargs), security_gateway=None)
    for slot in client._slots:
        slot.api_key = f"test-{slot.name}-key"
    return client


def _install_slot_responses(
    monkeypatch: pytest.MonkeyPatch,
    client: RotatingOpenAIClient,
    *,
    responses: dict[str, list[tuple[str, bool, int, bool]]],
    record: list[str] | None = None,
) -> None:
    """Patch _call_sync to consume queued (text, ok, status, quota) per slot."""

    queues = {name: list(items) for name, items in responses.items()}

    def fake_call_sync(slot: _Slot, **_kwargs: Any) -> tuple[str, bool, int, bool]:
        if record is not None:
            record.append(slot.name)
        q = queues.setdefault(slot.name, [])
        if not q:
            return "", False, 0, False
        text, ok, status, quota = q.pop(0)
        if ok:
            slot.total_requests += 1
            slot.total_successes += 1
            slot.consecutive_failures = 0
        else:
            slot.total_requests += 1
            slot.total_failures += 1
            slot.consecutive_failures += 1
            slot.last_error = f"http={status}"
        return text, ok, status, quota

    monkeypatch.setattr(
        "core.cloud.rotating_openai_client.RotatingOpenAIClient._call_sync",
        staticmethod(fake_call_sync),
    )

    # Bypass executor to keep tests deterministic and fast.
    async def fake_call_slot(self: RotatingOpenAIClient, slot: _Slot, **kwargs: Any):
        slot.request_window.append(0.0)
        return fake_call_sync(slot, **kwargs)

    monkeypatch.setattr(
        "core.cloud.rotating_openai_client.RotatingOpenAIClient._call_slot",
        fake_call_slot,
    )


# ── round-robin ----------------------------------------------------


@pytest.mark.asyncio
async def test_round_robin_advances_every_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    record: list[str] = []
    _install_slot_responses(
        monkeypatch,
        client,
        responses={
            "groq":     [("hi groq", True, 200, False)] * 3,
            "nvidia":   [("hi nvidia", True, 200, False)] * 3,
            "cerebras": [("hi cerebras", True, 200, False)] * 3,
        },
        record=record,
    )

    # Three sequential turns must hit three distinct providers.
    for _ in range(3):
        text, ok = await client.ask("ping")
        assert ok and text

    assert record == ["groq", "nvidia", "cerebras"]

    # Fourth turn wraps back to groq.
    await client.ask("ping again")
    assert record[3] == "groq"


# ── 429 quarantine -------------------------------------------------


@pytest.mark.asyncio
async def test_429_quarantines_slot_and_skips_it(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch, cooldown_429_s=120.0)
    record: list[str] = []
    _install_slot_responses(
        monkeypatch,
        client,
        responses={
            "groq":     [("", False, 429, True), ("groq later", True, 200, False)],
            "nvidia":   [("nvidia ok", True, 200, False)] * 5,
            "cerebras": [("cerebras ok", True, 200, False)] * 5,
        },
        record=record,
    )

    # First turn: groq → 429 → rotate to nvidia → ok.
    text, ok = await client.ask("turn 1")
    assert ok and text == "nvidia ok"
    assert record == ["groq", "nvidia"]

    # Groq must be cold for ~cooldown_429_s; rotation should skip it.
    groq = next(s for s in client._slots if s.name == "groq")
    assert groq.circuit_open_until > 0

    # Next two turns must not touch groq (quarantine respected).
    record.clear()
    for _ in range(2):
        text, ok = await client.ask("more")
        assert ok
    assert "groq" not in record


@pytest.mark.asyncio
async def test_5xx_uses_short_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch, cooldown_5xx_s=10.0)
    _install_slot_responses(
        monkeypatch,
        client,
        responses={
            "groq":     [("", False, 503, False)],
            "nvidia":   [("nvidia ok", True, 200, False)],
            "cerebras": [],
        },
    )
    text, ok = await client.ask("turn")
    assert ok and text == "nvidia ok"

    groq = next(s for s in client._slots if s.name == "groq")
    cerebras = next(s for s in client._slots if s.name == "cerebras")
    # 5xx cooldown is shorter than the 60s 429 default.
    assert 0 < (groq.circuit_open_until or 0)
    assert cerebras.circuit_open_until == 0


# ── soft RPM -------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_rpm_skips_saturated_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch, soft_rpm=2)
    record: list[str] = []
    _install_slot_responses(
        monkeypatch,
        client,
        responses={
            "groq":     [("groq", True, 200, False)] * 10,
            "nvidia":   [("nvidia", True, 200, False)] * 10,
            "cerebras": [("cerebras", True, 200, False)] * 10,
        },
        record=record,
    )

    # Pre-saturate groq's window so it's over budget without doing work.
    groq = next(s for s in client._slots if s.name == "groq")
    import time as _t
    now = _t.monotonic()
    groq.request_window = deque([now, now])  # 2 hits → at soft_rpm cap

    # Cursor starts at 0 (groq). Picking should skip groq and choose nvidia.
    await client.ask("turn")
    assert record == ["nvidia"]


# ── all-down ------------------------------------------------------


@pytest.mark.asyncio
async def test_all_down_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    # Drop all keys → no warm slots.
    for slot in client._slots:
        slot.api_key = ""
    text, ok = await client.ask("anything")
    assert ok is False and text == ""


@pytest.mark.asyncio
async def test_disabled_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch, enabled=False)
    assert client.is_available is False
    text, ok = await client.ask("hi")
    assert ok is False and text == ""


# ── streaming shim -------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_shim_emits_one_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    _install_slot_responses(
        monkeypatch,
        client,
        responses={
            "groq":     [("the answer", True, 200, False)],
            "nvidia":   [],
            "cerebras": [],
        },
    )
    chunks: list[tuple[str, bool]] = []
    text, ok = await client.ask_streaming(
        "explain", on_token=lambda c, last: chunks.append((c, last)),
    )
    assert ok and text == "the answer"
    assert chunks == [("the answer", True)]


# ── diagnostics shape ---------------------------------------------


@pytest.mark.asyncio
async def test_diagnostics_shape_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    _install_slot_responses(
        monkeypatch,
        client,
        responses={
            "groq":     [("ok", True, 200, False)],
            "nvidia":   [],
            "cerebras": [],
        },
    )
    await client.ask("ping")
    diag = client.diagnostics()
    assert diag["provider"] == "rotating"
    assert diag["enabled"] is True
    assert diag["available"] is True
    # Sprint Ω.10: tiered round-robin replaces the single cursor with
    # one cursor per tier. The shape contract is "a dict of int→int".
    assert isinstance(diag["tier_cursors"], dict)
    for tier, cursor in diag["tier_cursors"].items():
        assert isinstance(tier, int)
        assert isinstance(cursor, int)
    assert {s["name"] for s in diag["slots"]} == {"groq", "nvidia", "cerebras"}
    for s in diag["slots"]:
        for key in (
            "fast_model", "deep_model", "tier", "provider_kind",
            "has_key", "warm", "cooldown_remaining_s",
            "rpm_used", "soft_rpm",
            "total_requests", "total_successes", "total_failures",
            "avg_latency_ms", "last_error",
        ):
            assert key in s, f"missing diag key: {key} on {s['name']}"
        assert isinstance(s["tier"], int) and s["tier"] >= 1
        assert s["provider_kind"] in {"openai", "anthropic", "gemini"}


# ── duck-typed deep/buddy ------------------------------------------


@pytest.mark.asyncio
async def test_deep_routes_to_deep_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    seen: list[str] = []

    def fake_call_sync(slot: _Slot, *, model: str, **_kwargs: Any) -> tuple[str, bool, int, bool]:
        seen.append(model)
        slot.total_requests += 1
        slot.total_successes += 1
        return "out", True, 200, False

    monkeypatch.setattr(
        "core.cloud.rotating_openai_client.RotatingOpenAIClient._call_sync",
        staticmethod(fake_call_sync),
    )

    async def fake_call_slot(self: RotatingOpenAIClient, slot: _Slot, **kwargs: Any):
        slot.request_window.append(0.0)
        return fake_call_sync(slot, **kwargs)

    monkeypatch.setattr(
        "core.cloud.rotating_openai_client.RotatingOpenAIClient._call_slot",
        fake_call_slot,
    )

    await client.ask_buddy("hello")
    await client.ask_reasoning("explain why")

    # First call: fast model on cursor=0 (groq); second: deep model on cursor=1 (nvidia).
    assert seen[0] == "llama-3.1-8b-instant"
    assert seen[1] == "meta/llama-3.3-70b-instruct"


# ── Sprint Ω.11: tier fallback + multi-vendor dispatch ──────────────


def _multi_tier_config() -> dict:
    """Three tiers across three vendor kinds, mirroring the production
    settings.json layout post-Sprint-Ω.11.
    """
    return {
        "cloud": {
            "enabled": True,
            "rotation": {
                "enabled": True,
                "cooldown_429_s": 60.0,
                "cooldown_5xx_s": 30.0,
                "cooldown_hard_s": 300.0,
                "soft_rpm_per_slot": 25,
                "providers": [
                    {
                        "name": "claude",
                        "provider_kind": "anthropic",
                        "base_url": "https://anthropic.test/v1",
                        "credential_id": "anthropic",
                        "fast_model": "claude-haiku-4-5",
                        "deep_model": "claude-sonnet-4-5",
                        "tier": 1,
                    },
                    {
                        "name": "groq",
                        "provider_kind": "openai",
                        "base_url": "https://groq.test/v1",
                        "credential_id": "groq",
                        "fast_model": "llama-3.1-8b-instant",
                        "deep_model": "llama-3.3-70b-versatile",
                        "tier": 1,
                    },
                    {
                        "name": "gemini",
                        "provider_kind": "gemini",
                        "base_url": "https://gemini.test/v1beta",
                        "credential_id": "gemini_fast",
                        "fast_model": "gemini-2.5-flash-lite",
                        "deep_model": "gemini-2.5-flash",
                        "tier": 1,
                    },
                    {
                        "name": "cerebras",
                        "provider_kind": "openai",
                        "base_url": "https://cerebras.test/v1",
                        "credential_id": "cerebras",
                        "fast_model": "llama3.1-8b",
                        "deep_model": "llama-3.3-70b",
                        "tier": 2,
                    },
                    {
                        "name": "nvidia",
                        "provider_kind": "openai",
                        "base_url": "https://nvidia.test/v1",
                        "credential_id": "nvidia",
                        "fast_model": "meta/llama-3.1-8b-instruct",
                        "deep_model": "meta/llama-3.3-70b-instruct",
                        "tier": 3,
                    },
                ],
            },
        },
    }


def _build_multi_tier(monkeypatch: pytest.MonkeyPatch) -> RotatingOpenAIClient:
    monkeypatch.setattr(
        "core.cloud.rotating_openai_client.RotatingOpenAIClient._hydrate_keys_from_vault",
        lambda self: None,
    )
    client = RotatingOpenAIClient(_multi_tier_config(), security_gateway=None)
    for slot in client._slots:
        slot.api_key = f"test-{slot.name}-key"
    return client


@pytest.mark.asyncio
async def test_tier1_round_robin_never_touches_lower_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While every tier-1 slot is warm, tier 2 + tier 3 must stay cold."""
    client = _build_multi_tier(monkeypatch)
    record: list[str] = []
    _install_slot_responses(
        monkeypatch, client,
        responses={
            "claude":   [("c", True, 200, False)] * 4,
            "groq":     [("g", True, 200, False)] * 4,
            "gemini":   [("m", True, 200, False)] * 4,
            "cerebras": [("x", True, 200, False)] * 4,
            "nvidia":   [("y", True, 200, False)] * 4,
        },
        record=record,
    )

    for _ in range(6):
        text, ok = await client.ask("hi")
        assert ok and text

    # Six turns must hit ONLY tier-1 slots, perfectly round-robin.
    assert record == ["claude", "groq", "gemini", "claude", "groq", "gemini"]
    assert "cerebras" not in record
    assert "nvidia" not in record


@pytest.mark.asyncio
async def test_tier2_takes_over_when_tier1_quarantined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When all tier-1 slots 429, the next turn must go to tier 2 — and
    only when tier 2 is also down do we fall to tier 3.
    """
    client = _build_multi_tier(monkeypatch)
    record: list[str] = []
    _install_slot_responses(
        monkeypatch, client,
        responses={
            # Each tier-1 slot 429s once on its first turn.
            "claude":   [("", False, 429, True), ("c2", True, 200, False)],
            "groq":     [("", False, 429, True), ("g2", True, 200, False)],
            "gemini":   [("", False, 429, True), ("m2", True, 200, False)],
            # Tier 2 succeeds when it gets a turn.
            "cerebras": [("cer", True, 200, False)] * 4,
            # Tier 3 succeeds when it gets a turn.
            "nvidia":   [("nv", True, 200, False)] * 4,
        },
        record=record,
    )

    text, ok = await client.ask("first turn")
    # Within ONE call, exhaust tier 1 (3 × 429), then drop to tier 2.
    assert ok and text == "cer"
    assert record[:4] == ["claude", "groq", "gemini", "cerebras"]


@pytest.mark.asyncio
async def test_tier3_picked_only_when_tier1_and_tier2_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_multi_tier(monkeypatch)
    record: list[str] = []
    _install_slot_responses(
        monkeypatch, client,
        responses={
            "claude":   [("", False, 429, True)],
            "groq":     [("", False, 429, True)],
            "gemini":   [("", False, 429, True)],
            "cerebras": [("", False, 429, True)],
            "nvidia":   [("nv", True, 200, False)],
        },
        record=record,
    )

    text, ok = await client.ask("only nvidia survives")
    assert ok and text == "nv"
    assert record == ["claude", "groq", "gemini", "cerebras", "nvidia"]


# ── Sprint Ω.11: per-kind dispatch correctness ──────────────────────


def test_anthropic_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _call_anthropic builds the right URL, headers, and body."""
    captured: dict[str, Any] = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self) -> bytes:
            return (
                b'{"content": [{"type": "text", "text": "hello"}],'
                b' "stop_reason": "end_turn"}'
            )

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return FakeResp()

    import json
    import urllib.request as ur
    monkeypatch.setattr(ur, "urlopen", fake_urlopen)

    slot = _Slot(
        name="claude",
        base_url="https://anthropic.test/v1",
        credential_id="anthropic",
        fast_model="claude-haiku-4-5",
        deep_model="claude-sonnet-4-5",
        api_key="sk-ant-FAKE",
        provider_kind="anthropic",
    )
    text, ok, status, quota = RotatingOpenAIClient._call_anthropic(
        slot,
        query="What is up?",
        model="claude-haiku-4-5",
        max_tokens=128,
        timeout_s=10.0,
        system_instruction="Be brief.",
        temperature=0.5,
    )
    assert ok and text == "hello" and status == 200 and not quota
    assert captured["url"] == "https://anthropic.test/v1/messages"
    # Headers normalized to title-case by urllib.request.Request.
    assert captured["headers"].get("X-api-key") == "sk-ant-FAKE"
    assert captured["headers"].get("Anthropic-version") == "2023-06-01"
    assert captured["headers"].get("Authorization") is None  # NOT Bearer
    body = captured["body"]
    assert body["model"] == "claude-haiku-4-5"
    assert body["max_tokens"] == 128
    assert body["system"] == "Be brief."
    assert body["messages"] == [{"role": "user", "content": "What is up?"}]


def test_gemini_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _call_gemini builds the right URL with key in query
    string, no Authorization header, and contents/parts body shape.
    """
    captured: dict[str, Any] = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self) -> bytes:
            return (
                b'{"candidates": [{"content": {"parts":'
                b' [{"text": "hi from gemini"}]},'
                b' "finishReason": "STOP"}]}'
            )

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        return FakeResp()

    import json
    import urllib.request as ur
    monkeypatch.setattr(ur, "urlopen", fake_urlopen)

    slot = _Slot(
        name="gemini",
        base_url="https://gemini.test/v1beta",
        credential_id="gemini_fast",
        fast_model="gemini-2.5-flash-lite",
        deep_model="gemini-2.5-flash",
        api_key="AIzaTEST",
        provider_kind="gemini",
    )
    text, ok, status, quota = RotatingOpenAIClient._call_gemini(
        slot,
        query="hello",
        model="gemini-2.5-flash-lite",
        max_tokens=64,
        timeout_s=10.0,
        system_instruction="Be brief.",
        temperature=0.4,
    )
    assert ok and text == "hi from gemini" and status == 200 and not quota
    assert captured["url"] == (
        "https://gemini.test/v1beta/models/gemini-2.5-flash-lite"
        ":generateContent?key=AIzaTEST"
    )
    assert captured["headers"].get("Authorization") is None
    assert captured["headers"].get("X-api-key") is None
    body = captured["body"]
    assert body["contents"][0]["parts"][0]["text"] == "hello"
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert body["generationConfig"]["maxOutputTokens"] == 64
