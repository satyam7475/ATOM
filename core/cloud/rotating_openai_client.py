"""ATOM — Rotating multi-vendor cloud client.

Sprint Ω.9 (Apr 26 2026): cycle-based key rotation across three free
OpenAI-compatible providers — Groq, NVIDIA NIM, Cerebras — so a single
per-key 429 cap never blocks the cloud lane.

Sprint Ω.11 (Apr 26 2026): generalised to also speak Anthropic Messages
and Google Gemini ``generateContent`` natively, in the same rotation,
behind the same duck-typed surface. Each slot declares a
``provider_kind`` (``openai`` / ``anthropic`` / ``gemini``) and the
client dispatches to the right request shape, auth header, and response
parser. Rotation, tiering, circuit breakers, soft-RPM gates, and
streaming-collect all stay shared across kinds.

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
1. **Tiered round-robin every turn.** Each slot has a ``tier`` (default
   ``1``). The picker scans tiers in ascending order: tier-1 slots are
   rotated round-robin first; tier-2+ slots are last-resort fallbacks
   that only get a turn when **every** lower-tier slot is cold
   (quarantined, RPM-saturated, or missing a key). NVIDIA NIM rides
   tier 2 because its synchronous endpoint stalls on this network —
   we never pay its tail latency unless Groq + Cerebras are both down.
2. **Cursor advances on every attempt** within the chosen tier — both
   on success and on failure. This spreads RPM evenly across the
   active pool so no single provider hits its rate-limit window first.
3. **Quarantine on 429.** A 429 (or body containing ``rate_limit`` /
   ``quota``) opens the slot's circuit for ``cooldown_429_s`` (default
   60 s). 5xx opens for ``cooldown_5xx_s`` (default 30 s). Three
   consecutive failures opens for ``cooldown_hard_s`` (default 300 s).
4. **Soft RPM gate.** Each slot tracks request timestamps in a deque;
   when the count in the last 60 s reaches ``soft_rpm_per_slot`` the
   slot is pre-emptively skipped. Avoids the 500-800 ms tax of
   discovering a 429 reactively.
5. **All-down → ("", False).** If every slot in every tier is cold the
   call returns the same ``("", False)`` shape Gemini/Groq return;
   ``CloudBrainRouter`` then falls through to local MLX as designed.

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
import socket
import ssl as _ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("atom.cloud.rotating")

_USER_AGENT = "ATOM-CognitiveOS/2.0 (+rotating)"

# Supported provider wire kinds. Each maps to one ``_call_<kind>`` method.
_KIND_OPENAI = "openai"        # Groq, Cerebras, NVIDIA NIM, generic
_KIND_ANTHROPIC = "anthropic"  # Claude — /v1/messages, x-api-key header
_KIND_GEMINI = "gemini"        # Google — /v1beta/.../generateContent?key=

_VALID_KINDS = (_KIND_OPENAI, _KIND_ANTHROPIC, _KIND_GEMINI)
_ANTHROPIC_VERSION = "2023-06-01"

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
    # Tier 1 = active rotation pool. Tier 2+ = last-resort fallbacks that
    # are only consulted when every lower-tier slot is cold. Lower number
    # is preferred. Slots within the same tier round-robin among
    # themselves; tiers fall through in ascending order.
    tier: int = 1
    # Wire kind for this slot. ``openai`` covers Groq/Cerebras/NVIDIA
    # (POST /chat/completions, Bearer auth). ``anthropic`` covers Claude
    # (POST /v1/messages, x-api-key header, system as top-level field).
    # ``gemini`` covers Google (POST /v1beta/.../generateContent, key in
    # query string, contents/parts shape). The picker is kind-agnostic;
    # only ``_call_sync`` dispatches on this field.
    provider_kind: str = _KIND_OPENAI
    # Some providers (NVIDIA NIM observed Apr 26 2026) buffer the entire
    # response on the sync ``stream=false`` path and intermittently stall
    # for >30s before delivering a single byte. Switching to
    # ``stream=true`` makes the same key return first-token in ~1s. We
    # still collect the full response and return it as a single string,
    # so the caller surface is unchanged.
    prefer_streaming: bool = False

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
            tier_raw = entry.get("tier", 1)
            try:
                tier_val = int(tier_raw) if tier_raw is not None else 1
            except (TypeError, ValueError):
                tier_val = 1
            if tier_val < 1:
                tier_val = 1
            kind_val = str(entry.get("provider_kind", _KIND_OPENAI)).strip().lower()
            if kind_val not in _VALID_KINDS:
                logger.warning(
                    "RotatingCloudClient: unknown provider_kind %r on slot %s — "
                    "defaulting to %r",
                    kind_val, name, _KIND_OPENAI,
                )
                kind_val = _KIND_OPENAI
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
                    tier=tier_val,
                    provider_kind=kind_val,
                    prefer_streaming=bool(entry.get("prefer_streaming", False)),
                )
            )

        # Tiered cursor map: per tier we keep our own round-robin cursor
        # so a tier-2 fallback never disturbs the spread within tier 1.
        self._tier_cursors: dict[int, int] = {}
        self._streaming_cancelled = False
        self._temperature: float = float(cloud_cfg.get("temperature", 0.7))
        self._lock = asyncio.Lock()

        # Eagerly pull keys from secrets_manager so __init__ leaves the
        # client either fully ready or with a clear "no key for X" log.
        self._hydrate_keys_from_vault()

        if self._slots:
            tiers_summary = ", ".join(
                f"T{t}={[f'{s.name}:{s.provider_kind}' for s in self._slots if s.tier == t and s.api_key] or ['<none>']}"
                for t in sorted({s.tier for s in self._slots})
            )
            cold = [s.name for s in self._slots if not s.api_key]
            logger.info(
                "RotatingCloudClient: slots=%d %s missing_key=%s "
                "cooldown(429/5xx/hard)=%.0f/%.0f/%.0fs soft_rpm=%d",
                len(self._slots),
                tiers_summary,
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
        """Tiered round-robin scan. Tier 1 is the active pool; tier 2+
        are last-resort fallbacks that only see traffic when **every**
        slot in lower tiers is cold (quarantined or RPM-saturated).

        Within a tier we round-robin via a per-tier cursor that
        advances on every pick (success or failure). Returns
        ``(slot, original_index)`` or ``(None, -1)`` when no warm slot
        exists in any tier.
        """
        if not self._slots:
            return None, -1
        now = time.monotonic()
        # Bucket slot indices by tier in a stable, ascending order.
        by_tier: dict[int, list[int]] = {}
        for idx, slot in enumerate(self._slots):
            by_tier.setdefault(slot.tier, []).append(idx)
        for tier in sorted(by_tier.keys()):
            pool = by_tier[tier]
            if not pool:
                continue
            cursor = self._tier_cursors.get(tier, 0)
            m = len(pool)
            for offset in range(m):
                pool_idx = (cursor + offset) % m
                slot_idx = pool[pool_idx]
                slot = self._slots[slot_idx]
                if slot.is_warm(now):
                    self._tier_cursors[tier] = (pool_idx + 1) % m
                    return slot, slot_idx
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
        """Per-kind dispatcher. The picker, circuit breaker, and
        soft-RPM gate don't care which vendor we're hitting; only this
        method does. Returns ``(text, ok, http_status, quota_signal)``.
        """
        kind = slot.provider_kind
        if kind == _KIND_ANTHROPIC:
            return RotatingOpenAIClient._call_anthropic(
                slot, query=query, model=model,
                max_tokens=max_tokens, timeout_s=timeout_s,
                system_instruction=system_instruction,
                temperature=temperature,
            )
        if kind == _KIND_GEMINI:
            return RotatingOpenAIClient._call_gemini(
                slot, query=query, model=model,
                max_tokens=max_tokens, timeout_s=timeout_s,
                system_instruction=system_instruction,
                temperature=temperature,
            )
        return RotatingOpenAIClient._call_openai(
            slot, query=query, model=model,
            max_tokens=max_tokens, timeout_s=timeout_s,
            system_instruction=system_instruction,
            temperature=temperature,
        )

    @staticmethod
    def _record_success(
        slot: _Slot,
        *,
        text: str,
        model: str,
        t0: float,
        label: str = "",
    ) -> tuple[str, bool, int, bool]:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        slot.total_requests += 1
        slot.total_successes += 1
        slot.total_latency_ms += latency_ms
        slot.consecutive_failures = 0
        slot.last_error = ""
        logger.info(
            "Rotating[%s] %s%s: %.0fms out=%d",
            slot.name, model, f" ({label})" if label else "",
            latency_ms, len(text),
        )
        return text, True, 200, False

    @staticmethod
    def _record_empty(slot: _Slot, marker: str) -> tuple[str, bool, int, bool]:
        slot.last_error = marker
        slot.total_requests += 1
        slot.total_failures += 1
        slot.consecutive_failures += 1
        return "", False, 0, False

    @staticmethod
    def _record_http_error(
        slot: _Slot,
        e: urllib.error.HTTPError,
    ) -> tuple[str, bool, int, bool]:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="ignore")[:200]
        except Exception:
            pass
        low = error_body.lower()
        quota = (
            e.code == 429
            or "rate_limit" in low
            or "quota" in low
            or "resource_exhausted" in low
            or "overloaded" in low
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

    @staticmethod
    def _record_exception(
        slot: _Slot,
        e: BaseException,
    ) -> tuple[str, bool, int, bool]:
        slot.last_error = str(e)[:200]
        slot.total_requests += 1
        slot.total_failures += 1
        slot.consecutive_failures += 1
        logger.warning("Rotating[%s] call failed: %s", slot.name, e)
        return "", False, 0, False

    # ── per-kind callers ────────────────────────────────────────

    @staticmethod
    def _call_openai(
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
        use_streaming = bool(slot.prefer_streaming)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": 0.9,
            "stream": use_streaming,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {slot.api_key}",
            "User-Agent": _USER_AGENT,
        }
        if use_streaming:
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        t0 = time.perf_counter()
        try:
            if use_streaming:
                text = RotatingOpenAIClient._read_sse_collect(req, timeout_s)
                if not text:
                    return RotatingOpenAIClient._record_empty(slot, "empty_stream")
                return RotatingOpenAIClient._record_success(
                    slot, text=text, model=model, t0=t0, label="stream",
                )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                return RotatingOpenAIClient._record_empty(slot, "no_choices")
            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            if not text:
                return RotatingOpenAIClient._record_empty(slot, "empty_content")
            return RotatingOpenAIClient._record_success(
                slot, text=text, model=model, t0=t0,
            )
        except urllib.error.HTTPError as e:
            return RotatingOpenAIClient._record_http_error(slot, e)
        except Exception as e:
            return RotatingOpenAIClient._record_exception(slot, e)

    @staticmethod
    def _call_anthropic(
        slot: _Slot,
        *,
        query: str,
        model: str,
        max_tokens: int,
        timeout_s: float,
        system_instruction: str | None,
        temperature: float,
    ) -> tuple[str, bool, int, bool]:
        """Anthropic Messages API. Differences from OpenAI shape:
          * Endpoint: ``POST {base_url}/messages``
          * Auth: ``x-api-key`` header (not Bearer)
          * Required header: ``anthropic-version``
          * ``system`` is top-level, not a chat message
          * Response: ``content`` is a list of typed blocks; we join
            every ``{"type": "text"}`` block.
          * Quota / overload: returns 529 ``overloaded_error`` in
            addition to 429 ``rate_limit``; we treat both as quota.
        """
        url = f"{slot.base_url}/messages"
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "messages": [{"role": "user", "content": query}],
        }
        if system_instruction:
            payload["system"] = system_instruction
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": slot.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "User-Agent": _USER_AGENT,
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            blocks = data.get("content") or []
            if not blocks:
                return RotatingOpenAIClient._record_empty(slot, "no_content_blocks")
            text = "".join(
                (b.get("text") or "")
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if not text:
                return RotatingOpenAIClient._record_empty(slot, "empty_content")
            return RotatingOpenAIClient._record_success(
                slot, text=text, model=model, t0=t0,
            )
        except urllib.error.HTTPError as e:
            return RotatingOpenAIClient._record_http_error(slot, e)
        except Exception as e:
            return RotatingOpenAIClient._record_exception(slot, e)

    @staticmethod
    def _call_gemini(
        slot: _Slot,
        *,
        query: str,
        model: str,
        max_tokens: int,
        timeout_s: float,
        system_instruction: str | None,
        temperature: float,
    ) -> tuple[str, bool, int, bool]:
        """Google Gemini ``generateContent``. Differences from OpenAI:
          * Endpoint: ``POST {base_url}/models/{model}:generateContent?key=...``
          * Auth: API key in query string (no Authorization header)
          * Body: ``contents`` array of parts; ``systemInstruction``
            is its own top-level object.
          * Response: ``candidates[0].content.parts[*].text``
          * 429 ``RESOURCE_EXHAUSTED`` is the quota signal.
        """
        # Slot.base_url for Gemini is the API root (".../v1beta"); the
        # model id is appended into the path.
        key = slot.api_key
        url = (
            f"{slot.base_url}/models/{model}:generateContent"
            f"?key={urllib.parse.quote(key, safe='')}"
        )
        payload: dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": query}]},
            ],
            "generationConfig": {
                "maxOutputTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": 0.9,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates") or []
            if not candidates:
                # Gemini sometimes returns an empty candidates array on
                # safety blocks; promptFeedback carries the reason.
                pf = data.get("promptFeedback") or {}
                marker = "blocked_" + str(pf.get("blockReason", "no_candidates")).lower()
                return RotatingOpenAIClient._record_empty(slot, marker[:60])
            cand = candidates[0]
            parts = ((cand.get("content") or {}).get("parts")) or []
            text = "".join(
                (p.get("text") or "")
                for p in parts
                if isinstance(p, dict)
            ).strip()
            if not text:
                # MAX_TOKENS with thinking budget eaten still counts as a
                # logical empty for our caller; it's better to rotate to
                # the next slot than to surface "" upstream.
                finish = cand.get("finishReason", "no_text")
                return RotatingOpenAIClient._record_empty(
                    slot, f"empty_text_{str(finish).lower()}"[:60],
                )
            return RotatingOpenAIClient._record_success(
                slot, text=text, model=model, t0=t0,
            )
        except urllib.error.HTTPError as e:
            return RotatingOpenAIClient._record_http_error(slot, e)
        except Exception as e:
            return RotatingOpenAIClient._record_exception(slot, e)

    @staticmethod
    def _read_sse_collect(
        req: urllib.request.Request,
        timeout_s: float,
    ) -> str:
        """Open the chat-completions endpoint in SSE mode and collect
        every ``delta.content`` chunk into a single string. Used for
        providers whose sync ``stream=false`` path stalls (e.g. NVIDIA
        NIM observed Apr 26 2026). The caller still gets a plain
        ``(text, ok)`` shape — streaming is internal.
        """
        full = []
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]" or not payload_str:
                    continue
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                # Tolerate both ``delta`` (mid-stream) and ``message``
                # (some servers send the final aggregated message).
                piece = (
                    (choices[0].get("delta") or {}).get("content")
                    or (choices[0].get("message") or {}).get("content")
                    or ""
                )
                if piece:
                    full.append(piece)
        return "".join(full).strip()

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

    # ── TLS warmer + circuit auto-recovery (Sprint Ω.10) ─────────

    # Default cooldown (s) applied to a slot when the auto-recovery
    # ping confirms the endpoint is unreachable. We push the circuit
    # forward another window so we don't immediately flap back into a
    # broken slot. Mirrors ``cooldown_5xx_s`` semantics.
    _RECOVERY_COOLDOWN_S: float = 30.0
    # How many seconds AFTER ``circuit_open_until`` we'll still treat
    # as the "freshly-recovered" window. Used to decide whether the
    # auto-recovery loop should fire a TLS ping on a given slot.
    _RECOVERY_GRACE_S: float = 30.0

    @staticmethod
    def _slot_warm_skip(slot: _Slot, skip_kinds: set[str]) -> str:
        """Return a short reason if ``slot`` should be skipped by the
        TLS warmer, or empty string if the warmer should hit it.

        Reasons are surfaced in the warmer's diagnostics so an
        operator reading boot logs can tell why a particular slot
        wasn't pinged (no key, Anthropic-safe skip, malformed URL).
        """
        if slot.provider_kind in skip_kinds:
            return f"skip_kind={slot.provider_kind}"
        if not slot.api_key:
            return "no_key"
        if not slot.base_url:
            return "no_base_url"
        return ""

    @staticmethod
    def _warm_tls_one(slot: _Slot, *, timeout_s: float) -> dict[str, Any]:
        """Open a TLS connection to ``slot.base_url``'s host:port.

        Sprint Ω.10 (Apr 27 2026): the goal is to pay the TCP
        handshake + TLS negotiation cost *now* (during boot or
        post-cooldown) so the first user-driven request to this slot
        skips the 200-400 ms one-time penalty. We deliberately
        DO NOT send any HTTP request — opening the TLS socket alone
        is enough for OpenSSL to cache the session, and crucially it
        cannot trip a billable request on any provider.

        Returns a small dict with ``ok``, ``elapsed_ms``, ``reason``
        so callers can log per-slot results without re-inspecting
        the slot.
        """
        t0 = time.perf_counter()
        try:
            parsed = urllib.parse.urlsplit(slot.base_url)
        except Exception as exc:
            return {
                "ok": False,
                "elapsed_ms": 0.0,
                "reason": f"parse_error:{exc!s}"[:80],
            }
        host = parsed.hostname or ""
        if not host:
            return {"ok": False, "elapsed_ms": 0.0, "reason": "no_host"}
        port = parsed.port or (443 if parsed.scheme in ("https", "") else 80)
        scheme_is_tls = parsed.scheme in ("", "https")
        try:
            sock = socket.create_connection((host, port), timeout=timeout_s)
        except Exception as exc:
            return {
                "ok": False,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                "reason": f"tcp:{type(exc).__name__}",
            }
        try:
            if scheme_is_tls:
                ctx = _ssl.create_default_context()
                with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                    ssock.settimeout(timeout_s)
                    return {
                        "ok": True,
                        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                        "reason": "",
                        "tls": True,
                    }
            return {
                "ok": True,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                "reason": "",
                "tls": False,
            }
        except Exception as exc:
            return {
                "ok": False,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                "reason": f"tls:{type(exc).__name__}",
            }
        finally:
            try:
                sock.close()
            except Exception:
                logger.debug("warm_tls socket close raised", exc_info=True)

    async def warm_tls(
        self,
        *,
        skip_kinds: set[str] | None = None,
        timeout_s: float = 4.0,
    ) -> dict[str, Any]:
        """Pre-warm TLS to every reachable cloud slot in parallel.

        Sprint Ω.10 (Apr 27 2026): the live audit of ATOM's first
        cloud reply showed a 200-400 ms one-time penalty per
        provider on cold TLS handshakes that the user perceived as
        "ATOM thinks for half a second on the first cloud question".
        Calling this once at boot collapses that penalty for every
        non-Anthropic slot. Anthropic is skipped by default because
        Boss explicitly opted out (cost-conscious — Anthropic bills
        per request even on a 4xx, while every other vendor in the
        rotation returns the TLS handshake for free).

        Returns a dict with a per-slot result list and aggregate
        timing so the caller can ``logger.info`` a concise summary.
        """
        skip = skip_kinds if skip_kinds is not None else {_KIND_ANTHROPIC}
        if not self._enabled or not self._slots:
            return {"ok": True, "warmed": 0, "skipped": 0, "slots": []}

        loop = asyncio.get_running_loop()
        t_total = time.perf_counter()
        tasks: list[asyncio.Future[dict[str, Any]]] = []
        meta: list[tuple[_Slot, str]] = []
        for slot in self._slots:
            skip_reason = self._slot_warm_skip(slot, skip)
            meta.append((slot, skip_reason))
            if skip_reason:
                tasks.append(loop.create_future())
                tasks[-1].set_result({
                    "ok": False, "elapsed_ms": 0.0, "reason": skip_reason,
                })
                continue
            tasks.append(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        self._warm_tls_one, slot, timeout_s=timeout_s,
                    ),
                ),
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        warmed = 0
        skipped = 0
        slot_payload: list[dict[str, Any]] = []
        for (slot, skip_reason), result in zip(meta, results):
            if isinstance(result, BaseException):
                payload = {
                    "ok": False,
                    "elapsed_ms": 0.0,
                    "reason": f"raised:{type(result).__name__}",
                }
            else:
                payload = dict(result)
            if skip_reason:
                skipped += 1
            elif payload.get("ok"):
                warmed += 1
            slot_payload.append({
                "name": slot.name,
                "kind": slot.provider_kind,
                "elapsed_ms": float(payload.get("elapsed_ms", 0.0) or 0.0),
                "ok": bool(payload.get("ok", False)),
                "reason": str(payload.get("reason", "") or ""),
            })

        elapsed_total_ms = (time.perf_counter() - t_total) * 1000.0
        logger.info(
            "RotatingCloudClient: TLS warm pass — warmed=%d skipped=%d "
            "in %.0fms (per-slot=%s)",
            warmed,
            skipped,
            elapsed_total_ms,
            ", ".join(
                f"{p['name']}:{int(p['elapsed_ms'])}ms{'!' if not p['ok'] and not p.get('reason', '').startswith('skip_kind=') else ''}"
                for p in slot_payload
            ) or "<none>",
        )
        return {
            "ok": True,
            "warmed": warmed,
            "skipped": skipped,
            "elapsed_ms": elapsed_total_ms,
            "slots": slot_payload,
        }

    def __init_recovery_state(self) -> None:
        """Lazy attribute init so existing tests that monkeypatch the
        client don't blow up on missing recovery fields."""
        if not hasattr(self, "_recovery_task"):
            self._recovery_task: asyncio.Task | None = None
        if not hasattr(self, "_recovery_shutdown"):
            self._recovery_shutdown: asyncio.Event | None = None
        if not hasattr(self, "_recovery_last_check"):
            self._recovery_last_check: float = 0.0
        if not hasattr(self, "_recovery_validated_until"):
            self._recovery_validated_until: dict[str, float] = {}

    def start_circuit_auto_recovery(
        self,
        *,
        interval_s: float = 30.0,
    ) -> bool:
        """Kick off a background loop that re-validates a slot the
        moment its circuit cooldown expires.

        Sprint Ω.10 (Apr 27 2026): without this, a slot whose
        circuit just timed out is treated as ``warm`` immediately
        and the very next user-driven request slams into it. If the
        underlying issue (rate limit, regional outage, key
        revocation) is still in effect, the user pays the failure
        latency and the slot circuit re-opens — sometimes flapping
        for tens of minutes. The recovery loop fires a TLS ping
        within ``interval_s`` of cooldown expiry; on failure it
        bumps the cooldown forward by ``_RECOVERY_COOLDOWN_S`` so
        Boss never lands on a still-broken slot.

        Idempotent: returns ``True`` on the first call after every
        stop, ``False`` if a recovery loop is already running. The
        loop holds NO references back into the running event loop
        outside the task itself, so its lifecycle is bounded by
        the caller's ``stop_circuit_auto_recovery``.
        """
        self.__init_recovery_state()
        if self._recovery_task is not None and not self._recovery_task.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                "start_circuit_auto_recovery: no running loop, skipping",
            )
            return False
        self._recovery_shutdown = asyncio.Event()
        self._recovery_task = loop.create_task(
            self._recovery_loop(interval_s=interval_s),
        )
        logger.info(
            "RotatingCloudClient: circuit auto-recovery started "
            "(interval=%.0fs)", interval_s,
        )
        return True

    def stop_circuit_auto_recovery(self) -> None:
        self.__init_recovery_state()
        evt = self._recovery_shutdown
        if evt is not None:
            evt.set()
        task = self._recovery_task
        if task is not None and not task.done():
            task.cancel()
        self._recovery_task = None
        self._recovery_shutdown = None

    async def _recovery_loop(self, *, interval_s: float) -> None:
        """Body of ``start_circuit_auto_recovery``.

        Wakes every ``interval_s``. For every slot whose cooldown
        elapsed within the last ``_RECOVERY_GRACE_S`` AND has an
        api_key AND isn't an Anthropic slot, fires a TLS ping. If
        the ping fails, push the circuit forward by
        ``_RECOVERY_COOLDOWN_S`` so the next user-driven request
        doesn't slam into a still-broken endpoint.
        """
        self.__init_recovery_state()
        evt = self._recovery_shutdown
        try:
            while True:
                if evt is not None:
                    try:
                        await asyncio.wait_for(evt.wait(), timeout=interval_s)
                        return
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(interval_s)

                now = time.monotonic()
                self._recovery_last_check = now
                candidates: list[_Slot] = []
                for slot in self._slots:
                    if not slot.api_key:
                        continue
                    if slot.provider_kind == _KIND_ANTHROPIC:
                        continue
                    if slot.circuit_open_until <= 0:
                        continue
                    if slot.circuit_open_until > now:
                        continue
                    elapsed = now - slot.circuit_open_until
                    if elapsed > self._RECOVERY_GRACE_S:
                        continue
                    last_validated = self._recovery_validated_until.get(
                        slot.name, 0.0,
                    )
                    if last_validated >= slot.circuit_open_until:
                        continue
                    candidates.append(slot)

                if not candidates:
                    continue

                tasks: list[asyncio.Future[dict[str, Any]]] = []
                for slot in candidates:
                    tasks.append(
                        asyncio.get_running_loop().run_in_executor(
                            None,
                            functools.partial(
                                self._warm_tls_one, slot, timeout_s=4.0,
                            ),
                        ),
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)
                healed = 0
                punished = 0
                for slot, raw in zip(candidates, results):
                    if isinstance(raw, BaseException):
                        ok = False
                        reason = f"raised:{type(raw).__name__}"
                    else:
                        ok = bool(raw.get("ok"))
                        reason = str(raw.get("reason", ""))
                    if ok:
                        self._recovery_validated_until[slot.name] = (
                            slot.circuit_open_until or now
                        )
                        healed += 1
                    else:
                        slot.circuit_open_until = max(
                            slot.circuit_open_until,
                            now + self._RECOVERY_COOLDOWN_S,
                        )
                        slot.last_error = (
                            f"auto_recovery:{reason}"
                        )[:200]
                        punished += 1
                if healed or punished:
                    logger.info(
                        "RotatingCloudClient: circuit auto-recovery — "
                        "healed=%d punished=%d (%s)",
                        healed,
                        punished,
                        ", ".join(
                            f"{s.name}:"
                            f"{'ok' if self._recovery_validated_until.get(s.name, 0) >= s.circuit_open_until else 'fail'}"
                            for s in candidates
                        ),
                    )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception(
                "RotatingCloudClient: auto-recovery loop crashed; restart on next start_circuit_auto_recovery()",
            )

    # ── diagnostics ──────────────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "provider": "rotating",
            "enabled": self._enabled,
            "available": self.is_available,
            "tier_cursors": dict(self._tier_cursors),
            "slots": [
                {
                    "name": s.name,
                    "base_url": s.base_url,
                    "fast_model": s.fast_model,
                    "deep_model": s.deep_model,
                    "tier": s.tier,
                    "provider_kind": s.provider_kind,
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


# Sprint Ω.11 (Apr 26 2026): the class is now a generic multi-vendor
# rotating cloud client (OpenAI-compatible + Anthropic Messages + Gemini
# generateContent). The historical name ``RotatingOpenAIClient`` stays
# as the canonical export so ``main.py`` and existing tests don't have
# to change. ``RotatingCloudClient`` is the forward-looking alias and
# is preferred for new code.
RotatingCloudClient = RotatingOpenAIClient

__all__ = ["RotatingOpenAIClient", "RotatingCloudClient"]
