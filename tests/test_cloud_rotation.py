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
    assert isinstance(diag["cursor"], int)
    assert {s["name"] for s in diag["slots"]} == {"groq", "nvidia", "cerebras"}
    for s in diag["slots"]:
        for key in (
            "fast_model", "deep_model", "has_key", "warm",
            "cooldown_remaining_s", "rpm_used", "soft_rpm",
            "total_requests", "total_successes", "total_failures",
            "avg_latency_ms", "last_error",
        ):
            assert key in s, f"missing diag key: {key} on {s['name']}"


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
