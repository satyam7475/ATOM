"""ATOM — Rotating multi-provider OpenAI-compatible cloud client.

Sprint Ω.9 (Apr 26 2026). Boss asked for cycle-based key rotation across
three free OpenAI-compatible providers — Groq, NVIDIA NIM, Cerebras —
so a single per-key 429 cap never blocks the cloud lane. All three speak
the same ``POST /v1/chat/completions`` body and serve the same Llama 3.x
weights, so a slot rotation is invisible to ATOM's prompt builder, router,
local-brain controller, and TTS pipeline.

This client duck-types the public surface of
:class:`core.cloud.gemini_client.GeminiClient` and
:class:`core.cloud.groq_client.GroqClient` so the rest of ATOM
(``CloudBrainRouter``, ``Router``, ``LocalBrainController``,
``SearchTool``, ``CognitiveKernel``) can use it without any conditional
branches:

    is_available           — property
    configure_api_key(k)   — distributes to all slots that share a key id
    ask(query, ...)        — (text, ok) — round-robin with quarantine
    ask_buddy(query, ...)  — (text, ok)
    ask_reasoning(...)     — (text, ok)
    cancel_streaming()
    ask_streaming(...)     — falls through to ask() for v1

Rotation policy
---------------
1. **Round-robin every turn.** The cursor advances on every successful
   call too (not just on failure). This spreads RPM evenly across slots
   so no single provider hits its rate-limit window first.
2. **Quarantine on 429.** A 429 (or body containing ``rate_limit`` /
   ``quota``) opens the slot's circuit for ``cooldown_429_s`` (default
   60 s). 5xx opens for ``cooldown_5xx_s`` (default 30 s). Three
   consecutive failures opens for ``cooldown_hard_s`` (default 300 s).
3. **Soft RPM gate.** Each slot tracks request timestamps in a deque;
   when the count in the last 60 s reaches ``soft_rpm_per_slot`` the
   slot is pre-emptively skipped. Avoids the 500-800 ms tax of
   discovering a 429 reactively.
4. **All-down → ("", False).** If every slot is cold the call returns
   the same ``("", False)`` shape Gemini/Groq return; ``CloudBrainRouter``
   then falls through to local MLX as designed.

Privacy posture matches Gemini/Groq: every outbound query passes through
``SecurityGateway.allow_cloud`` + ``sanitize_outbound`` and every
response is tagged via ``tag_cloud_response``.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("atom.cloud.rotating")

_USER_AGENT = "ATOM-CognitiveOS/2.0 (+rotating)"

_DEFAULT_BUDDY_SYSTEM = (
    "You are ATOM, a personal AI assistant created by Satyam Yadav. "
    "You call him 'Boss'. Friendly, witty, concise. Keep voice replies "
    "to 1-2 sentences unless asked for detail. Never invent actions the "
    "user did not request. If the input looks like noisy STT, ask one "
    "short clarifying question."
)

_DEFAULT_REASONING_SYSTEM = (
    "You are ATOM, an advanced AI created by Satyam Yadav. "
    "Methodical, thorough, precise. Think step by step. "
    "Provide structured, well-grounded answers. State assumptions "
    "explicitly when the question is ambiguous; never fabricate "
    "specific numbers, dates, or quotes."
)


@dataclass
class _Slot:
    """One OpenAI-compatible provider behind the rotating client."""

    name: str
    base_url: str
    credential_id: str
    fast_model: str
    deep_model: str
    api_key: str = ""
    timeout_s: float = 8.0
    timeout_reasoning_s: float = 30.0
    max_tokens: int = 1024
    max_tokens_reasoning: int = 4096
    soft_rpm: int = 25

    # Runtime state
    circuit_open_until: float = 0.0
    consecutive_failures: int = 0
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_latency_ms: float = 0.0
    last_error: str = ""
    request_window: deque[float] = field(default_factory=deque)

    def is_warm(self, now: float) -> bool:
        """Slot can take a request right now?"""
        if not self.api_key:
            return False
        if now < self.circuit_open_until:
            return False
        # Soft RPM: drop entries older than 60s, then check the budget.
        while self.request_window and (now - self.request_window[0]) > 60.0:
            self.request_window.popleft()
        if self.soft_rpm > 0 and len(self.request_window) >= self.soft_rpm:
            return False
        return True


class RotatingOpenAIClient:
    """Round-robin OpenAI-compatible client across N free-tier providers.

    Surface is intentionally identical to :class:`GroqClient` so ATOM's
    cloud pipeline doesn't know it's talking to a pool.
    """

    def __init__(
        self,
        config: dict | None = None,
        security_gateway: Any = None,
    ) -> None:
        cloud_cfg = (config or {}).get("cloud", {}) or {}
        rot_cfg = cloud_cfg.get("rotation", {}) or {}

        self._enabled: bool = bool(cloud_cfg.get("enabled", True)) and bool(
            rot_cfg.get("enabled", True)
        )
        self._gateway = security_gateway

        self._cooldown_429_s: float = float(rot_cfg.get("cooldown_429_s", 60.0))
        self._cooldown_5xx_s: float = float(rot_cfg.get("cooldown_5xx_s", 30.0))
        self._cooldown_hard_s: float = float(rot_cfg.get("cooldown_hard_s", 300.0))
        self._hard_failure_threshold: int = int(
            rot_cfg.get("hard_failure_threshold", 3)
        )

        # Default per-slot timeouts/limits inherit from cloud.* so existing
        # tunables (cloud.timeout_seconds etc.) keep working.
        default_timeout = float(cloud_cfg.get("timeout_seconds", 8.0))
        default_timeout_deep = float(
            cloud_cfg.get("timeout_reasoning_seconds", 30.0)
        )
        default_max_tokens = int(cloud_cfg.get("max_tokens", 1024))
        default_max_tokens_deep = int(
            cloud_cfg.get("max_tokens_reasoning", 4096)
        )
        default_soft_rpm = int(rot_cfg.get("soft_rpm_per_slot", 25))

        self._slots: list[_Slot] = []
        for entry in rot_cfg.get("providers", []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            base_url = str(entry.get("base_url", "")).strip().rstrip("/")
            credential_id = str(entry.get("credential_id", "")).strip()
            fast_model = str(entry.get("fast_model", "")).strip()
            deep_model = str(entry.get("deep_model", fast_model)).strip()
            if not (name and base_url and credential_id and fast_model):
                logger.warning(
                    "RotatingOpenAIClient: skipping malformed provider entry %r",
                    entry,
                )
                continue
            self._slots.append(
                _Slot(
                    name=name,
                    base_url=base_url,
                    credential_id=credential_id,
                    fast_model=fast_model,
                    deep_model=deep_model or fast_model,
                    timeout_s=float(entry.get("timeout_seconds", default_timeout)),
                    timeout_reasoning_s=float(
                        entry.get("timeout_reasoning_seconds", default_timeout_deep)
                    ),
                    max_tokens=int(entry.get("max_tokens", default_max_tokens)),
                    max_tokens_reasoning=int(
                        entry.get("max_tokens_reasoning", default_max_tokens_deep)
                    ),
                    soft_rpm=int(entry.get("soft_rpm", default_soft_rpm)),
                )
            )

        self._cursor = 0
        self._streaming_cancelled = False
        self._temperature: float = float(cloud_cfg.get("temperature", 0.7))
        self._lock = asyncio.Lock()

        # Eagerly pull keys from secrets_manager so __init__ leaves the
        # client either fully ready or with a clear "no key for X" log.
        self._hydrate_keys_from_vault()

        if self._slots:
            ready = [s.name for s in self._slots if s.api_key]
            cold = [s.name for s in self._slots if not s.api_key]
            logger.info(
                "RotatingOpenAIClient: slots=%d ready=%s missing_key=%s "
                "cooldown(429/5xx/hard)=%.0f/%.0f/%.0fs soft_rpm=%d",
                len(self._slots),
                ready or ["<none>"],
                cold or ["<none>"],
                self._cooldown_429_s,
                self._cooldown_5xx_s,
                self._cooldown_hard_s,
                default_soft_rpm,
            )

    # ── public surface (duck-typed against GroqClient/GeminiClient) ─

    @property
    def is_available(self) -> bool:
        if not self._enabled or not self._slots:
            return False
        now = time.monotonic()
        return any(s.is_warm(now) for s in self._slots)

    @property
    def requests_remaining_estimate(self) -> int:
        if self._gateway is not None:
            try:
                return int(self._gateway._rate_limiter.tokens)
            except Exception:
                logger.debug("rotating client gateway rate probe failed", exc_info=True)
        # Sum of remaining soft-RPM budget across warm slots.
        now = time.monotonic()
        total = 0
        for s in self._slots:
            if not s.is_warm(now):
                continue
            total += max(0, s.soft_rpm - len(s.request_window))
        return total

    def configure_api_key(self, key: str) -> None:
        """Compatibility shim. RotatingOpenAIClient sources keys from
        ``secrets_manager`` per slot, but a single-key handoff is still
        useful for tests / one-shot scripts: it populates every slot
        that doesn't already have one.
        """
        clean = (key or "").strip()
        if not clean:
            return
        for slot in self._slots:
            if not slot.api_key:
                slot.api_key = clean
        logger.info("RotatingOpenAIClient: configure_api_key applied to bare slots")

    def configure_slot_key(self, name: str, key: str) -> bool:
        """Set the API key for one named slot (used by main.py wiring)."""
        clean = (key or "").strip()
        if not clean:
            return False
        for slot in self._slots:
            if slot.name == name:
                slot.api_key = clean
                slot.circuit_open_until = 0.0
                slot.consecutive_failures = 0
                logger.info(
                    "RotatingOpenAIClient: %s key configured", slot.name,
                )
                return True
        return False

    def cancel_streaming(self) -> None:
        self._streaming_cancelled = True

    # ── core call surface ────────────────────────────────────────

    async def ask(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> tuple[str, bool]:
        """Round-robin call. ``model='deep'`` selects each slot's deep
        model; anything else (or None) selects the fast model. Caller
        can also pass an explicit slot-specific model id and we'll
        forward it verbatim, but the typical path is fast/deep tier.
        """
        if not self._enabled or not self._slots:
            return "", False

        if self._gateway is not None:
            try:
                allowed, reason = self._gateway.allow_cloud(query)
                if not allowed:
                    logger.info(
                        "RotatingOpenAIClient blocked by SecurityGateway: %s",
                        reason,
                    )
                    return "", False
                query = self._gateway.sanitize_outbound(query)
            except Exception:
                logger.debug(
                    "SecurityGateway error on RotatingOpenAIClient.ask",
                    exc_info=True,
                )

        if not (query or "").strip():
            return "", False

        is_deep = (model == "deep") or (model == "reasoning")
        explicit_model = model if model and not is_deep else None

        # Try every slot once per call before giving up. Cursor advances
        # on every attempt (success or failure) so consecutive turns hit
        # different providers.
        attempts = 0
        n = len(self._slots)
        while attempts < n:
            slot, picked = self._pick_next_warm()
            attempts += 1
            if slot is None:
                break
            chosen_model = explicit_model or (
                slot.deep_model if is_deep else slot.fast_model
            )
            chosen_max_tokens = max_tokens or (
                slot.max_tokens_reasoning if is_deep else slot.max_tokens
            )
            chosen_timeout = slot.timeout_reasoning_s if is_deep else slot.timeout_s

            text, ok, http_status, quota = await self._call_slot(
                slot,
                query=query,
                model=chosen_model,
                max_tokens=chosen_max_tokens,
                timeout_s=chosen_timeout,
                system_instruction=system_instruction,
                temperature=(
                    self._temperature if temperature is None else float(temperature)
                ),
            )
            if ok:
                if self._gateway is not None:
                    try:
                        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
                        self._gateway.record_cloud_call(
                            query_hash=query_hash,
                            sanitized_length=len(query),
                            response_length=len(text),
                            latency_ms=slot.total_latency_ms,
                            provider=f"rotating:{slot.name}",
                        )
                        tagged = self._gateway.tag_cloud_response(
                            text, provider=f"rotating:{slot.name}",
                        )
                        return tagged["text"], True
                    except Exception:
                        logger.debug(
                            "SecurityGateway audit on rotating slot raised",
                            exc_info=True,
                        )
                return text, True

            # Failure: open the circuit appropriately and try the next slot.
            self._punish_slot(slot, http_status=http_status, quota=quota)
            logger.info(
                "RotatingOpenAIClient: slot=%s failed (http=%s quota=%s) "
                "— rotating to next",
                slot.name, http_status, quota,
            )

        return "", False

    async def ask_buddy(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> tuple[str, bool]:
        return await self.ask(
            query,
            max_tokens=max_tokens,
            model=None,  # fast tier
            system_instruction=system_instruction or _DEFAULT_BUDDY_SYSTEM,
            temperature=0.7,
        )

    async def ask_reasoning(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> tuple[str, bool]:
        return await self.ask(
            query,
            max_tokens=max_tokens,
            model="deep",
            system_instruction=system_instruction or _DEFAULT_REASONING_SYSTEM,
            temperature=0.4,
        )

    async def ask_streaming(
        self,
        query: str,
        on_token: Callable[[str, bool], None] | None = None,
        *,
        max_tokens: int | None = None,
        model: str | None = None,
        system_instruction: str | None = None,
    ) -> tuple[str, bool]:
        """Streaming compatibility shim: emit the whole response as one
        token to ``on_token``. Real SSE rotation can ride a follow-up.
        """
        text, ok = await self.ask(
            query,
            max_tokens=max_tokens,
            model=model,
            system_instruction=system_instruction,
        )
        if on_token is not None and text:
            try:
                on_token(text, True)
            except Exception:
                logger.debug("on_token callback raised", exc_info=True)
        return text, ok

    # ── internals ────────────────────────────────────────────────

    def _hydrate_keys_from_vault(self) -> None:
        """Best-effort vault lookup. Missing keys are non-fatal;
        ``main.py`` may still call ``configure_slot_key`` later.
        """
        try:
            from core.secrets_manager import get_api_key
        except Exception:
            logger.debug("secrets_manager unavailable on init", exc_info=True)
            return
        for slot in self._slots:
            try:
                key = get_api_key(slot.credential_id) or ""
                if key:
                    slot.api_key = key
            except Exception:
                logger.debug(
                    "secrets lookup for %s failed", slot.credential_id,
                    exc_info=True,
                )

    def _pick_next_warm(self) -> tuple[_Slot | None, int]:
        """Round-robin scan; advance cursor every call (success/fail
        spread). Returns (slot, original_index) or (None, -1).
        """
        if not self._slots:
            return None, -1
        now = time.monotonic()
        n = len(self._slots)
        for offset in range(n):
            idx = (self._cursor + offset) % n
            slot = self._slots[idx]
            if slot.is_warm(now):
                self._cursor = (idx + 1) % n
                return slot, idx
        return None, -1

    async def _call_slot(
        self,
        slot: _Slot,
        *,
        query: str,
        model: str,
        max_tokens: int,
        timeout_s: float,
        system_instruction: str | None,
        temperature: float,
    ) -> tuple[str, bool, int, bool]:
        """Single-slot call. Returns (text, ok, http_status, quota_signal)."""
        # Reserve our slot in the soft-RPM window before the call so
        # a stampede doesn't overrun a single provider.
        slot.request_window.append(time.monotonic())

        call_fn = functools.partial(
            self._call_sync,
            slot,
            query=query,
            model=model,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, call_fn),
                timeout=timeout_s + 2.0,
            )
        except asyncio.TimeoutError:
            slot.last_error = "async_timeout"
            slot.total_failures += 1
            slot.consecutive_failures += 1
            logger.warning(
                "Rotating slot=%s async timeout (%.0fs, model=%s)",
                slot.name, timeout_s, model,
            )
            return "", False, 0, False

    @staticmethod
    def _call_sync(
        slot: _Slot,
        *,
        query: str,
        model: str,
        max_tokens: int,
        timeout_s: float,
        system_instruction: str | None,
        temperature: float,
    ) -> tuple[str, bool, int, bool]:
        url = f"{slot.base_url}/chat/completions"
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": query})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": 0.9,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {slot.api_key}",
            "User-Agent": _USER_AGENT,
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                slot.last_error = "no_choices"
                slot.total_failures += 1
                slot.consecutive_failures += 1
                return "", False, 0, False
            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            if not text:
                slot.last_error = "empty_content"
                slot.total_failures += 1
                slot.consecutive_failures += 1
                return "", False, 0, False
            latency_ms = (time.perf_counter() - t0) * 1000.0
            slot.total_requests += 1
            slot.total_successes += 1
            slot.total_latency_ms += latency_ms
            slot.consecutive_failures = 0
            slot.last_error = ""
            logger.info(
                "Rotating[%s] %s: %.0fms in=%d out=%d",
                slot.name, model, latency_ms, len(query), len(text),
            )
            return text, True, 200, False

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                pass
            quota = (
                e.code == 429
                or "rate_limit" in error_body.lower()
                or "quota" in error_body.lower()
                or "resource_exhausted" in error_body.lower()
            )
            slot.last_error = f"HTTP {e.code}: {error_body[:120]}"
            slot.total_requests += 1
            slot.total_failures += 1
            slot.consecutive_failures += 1
            logger.warning(
                "Rotating[%s] HTTP %d: %s",
                slot.name, e.code, error_body[:120],
            )
            return "", False, int(e.code), bool(quota)

        except Exception as e:
            slot.last_error = str(e)[:200]
            slot.total_requests += 1
            slot.total_failures += 1
            slot.consecutive_failures += 1
            logger.warning("Rotating[%s] call failed: %s", slot.name, e)
            return "", False, 0, False

    def _punish_slot(
        self,
        slot: _Slot,
        *,
        http_status: int,
        quota: bool,
    ) -> None:
        now = time.monotonic()
        if quota or http_status == 429:
            slot.circuit_open_until = max(
                slot.circuit_open_until, now + self._cooldown_429_s,
            )
            return
        if 500 <= http_status < 600:
            slot.circuit_open_until = max(
                slot.circuit_open_until, now + self._cooldown_5xx_s,
            )
            return
        if slot.consecutive_failures >= self._hard_failure_threshold:
            slot.circuit_open_until = max(
                slot.circuit_open_until, now + self._cooldown_hard_s,
            )

    # ── diagnostics ──────────────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "provider": "rotating",
            "enabled": self._enabled,
            "available": self.is_available,
            "cursor": self._cursor,
            "slots": [
                {
                    "name": s.name,
                    "base_url": s.base_url,
                    "fast_model": s.fast_model,
                    "deep_model": s.deep_model,
                    "has_key": bool(s.api_key),
                    "warm": s.is_warm(now),
                    "cooldown_remaining_s": max(
                        0.0, s.circuit_open_until - now,
                    ),
                    "rpm_used": len(s.request_window),
                    "soft_rpm": s.soft_rpm,
                    "total_requests": s.total_requests,
                    "total_successes": s.total_successes,
                    "total_failures": s.total_failures,
                    "avg_latency_ms": (
                        s.total_latency_ms / s.total_successes
                        if s.total_successes
                        else 0.0
                    ),
                    "last_error": s.last_error,
                }
                for s in self._slots
            ],
        }


__all__ = ["RotatingOpenAIClient"]
