"""ATOM — Groq cloud client (OpenAI-compatible).

Sprint Ω.8 (Apr 26 2026). Boss flagged the Gemini free tier as too
quota-restrictive and asked for a faster, free, model-rich alternative
that runs cleanly alongside ATOM's local MLX brain on a 16 GB M-series
Mac. Groq fits: their LPU stack ships ``llama-3.1-8b-instant`` at
500-800 tokens/s for buddy turns and ``llama-3.3-70b-versatile`` at
~250 tokens/s for deep reasoning, both behind a 100% OpenAI-compatible
HTTP API on a generous free tier.

This client duck-types the public surface of
:class:`core.cloud.gemini_client.GeminiClient` so the rest of ATOM
(``CloudBrainRouter``, ``Router``, ``LocalBrainController``,
``SearchTool``, ``DesktopAgent``) can use it without conditional
branches:

    is_available           — property
    configure_api_key(k)
    ask(query, ...)        — (text, ok)
    ask_buddy(query, ...)  — (text, ok)
    ask_reasoning(...)     — (text, ok)
    cancel_streaming()
    ask_streaming(...)     — falls through to ask() for v1

Network calls go through the same SecurityGateway (sanitize_outbound,
rate limit, audit log) as Gemini, so privacy posture is identical.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

logger = logging.getLogger("atom.cloud.groq")

_DEFAULT_API_BASE = "https://api.groq.com/openai/v1"
_DEFAULT_BUDDY_MODEL = "llama-3.1-8b-instant"
_DEFAULT_REASONING_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_TIMEOUT_S = 8.0
_DEFAULT_TIMEOUT_REASONING_S = 30.0
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_MAX_TOKENS_REASONING = 4096
_USER_AGENT = "ATOM-CognitiveOS/2.0 (+groq)"


class GroqClient:
    """OpenAI-compatible Groq REST client with security gating."""

    def __init__(
        self,
        config: dict | None = None,
        security_gateway: Any = None,
    ) -> None:
        cfg = (config or {}).get("cloud", {})
        self._api_base: str = str(
            cfg.get("api_base", _DEFAULT_API_BASE)
        ).rstrip("/")
        self._api_key_env: str = str(
            cfg.get("api_key_env", "GROQ_API_KEY")
        )
        # Try config first, then env var. Vault/secret loader can call
        # ``configure_api_key`` later — see main.py.
        self._api_key: str = (
            cfg.get("groq_api_key", "")
            or os.environ.get(self._api_key_env, "")
            or ""
        )
        self._buddy_model: str = cfg.get("buddy_model", _DEFAULT_BUDDY_MODEL)
        self._reasoning_model: str = cfg.get(
            "reasoning_model", _DEFAULT_REASONING_MODEL,
        )
        self._timeout: float = float(cfg.get("timeout_seconds", _DEFAULT_TIMEOUT_S))
        self._timeout_reasoning: float = float(
            cfg.get("timeout_reasoning_seconds", _DEFAULT_TIMEOUT_REASONING_S),
        )
        self._max_tokens: int = int(cfg.get("max_tokens", _DEFAULT_MAX_TOKENS))
        self._max_tokens_reasoning: int = int(
            cfg.get("max_tokens_reasoning", _DEFAULT_MAX_TOKENS_REASONING),
        )
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._temperature: float = float(cfg.get("temperature", 0.7))

        self._gateway = security_gateway

        # Stats (mirrors GeminiClient surface so dashboards work)
        self._total_requests = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_latency_ms = 0.0
        self._last_error: str = ""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_threshold = 4
        self._circuit_cooldown_s = 45.0
        # Groq surfaces 429 with X-RateLimit headers; if we hit a hard
        # quota cap they return ``rate_limit_exceeded`` with a retry-
        # after that's usually 30-90s. Pessimistic 5-minute open is
        # safe for a personal AI.
        self._quota_cooldown_s = 300.0

        self._streaming_cancelled = False

        if self._api_key:
            logger.info(
                "GroqClient: buddy=%s, reasoning=%s, timeout=%.0f/%.0fs",
                self._buddy_model, self._reasoning_model,
                self._timeout, self._timeout_reasoning,
            )

    # ── public surface (duck-typed against GeminiClient) ─────────

    def configure_api_key(self, key: str) -> None:
        self._api_key = (key or "").strip()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        if self._api_key:
            logger.info("GroqClient: API key configured")

    @property
    def is_available(self) -> bool:
        if not self._enabled or not self._api_key:
            return False
        if time.monotonic() < self._circuit_open_until:
            return False
        return True

    @property
    def requests_remaining_estimate(self) -> int:
        if self._gateway is not None:
            try:
                return int(self._gateway._rate_limiter.tokens)
            except Exception:
                pass
        return 25

    def cancel_streaming(self) -> None:
        """No-op for v1 — streaming falls through to ``ask`` for now."""
        self._streaming_cancelled = True

    # ── core ──────────────────────────────────────────────────────

    def _build_payload(
        self,
        query: str,
        *,
        model: str,
        max_tokens: int,
        system_instruction: str | None,
        temperature: float,
    ) -> tuple[str, bytes, dict[str, str]]:
        url = f"{self._api_base}/chat/completions"
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": query})
        payload: dict[str, Any] = {
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
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
        }
        return url, body, headers

    def _call_sync(
        self,
        query: str,
        *,
        model_override: str | None = None,
        max_tokens_override: int | None = None,
        system_instruction: str | None = None,
        temperature_override: float | None = None,
    ) -> tuple[str, bool]:
        t0 = time.perf_counter()
        model = model_override or self._buddy_model
        is_reasoning = model == self._reasoning_model
        max_tokens = max_tokens_override or (
            self._max_tokens_reasoning if is_reasoning else self._max_tokens
        )
        timeout = self._timeout_reasoning if is_reasoning else self._timeout
        temperature = (
            self._temperature if temperature_override is None
            else float(temperature_override)
        )

        try:
            url, body, headers = self._build_payload(
                query,
                model=model,
                max_tokens=max_tokens,
                system_instruction=system_instruction,
                temperature=temperature,
            )
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            choices = data.get("choices") or []
            if not choices:
                return "", False
            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            if not text:
                return "", False

            latency_ms = (time.perf_counter() - t0) * 1000
            self._record_success(latency_ms, len(query), len(text))
            logger.info(
                "Groq [%s] response: %.0fms, %d chars in, %d chars out",
                model, latency_ms, len(query), len(text),
            )
            return text, True

        except urllib.error.HTTPError as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                pass
            quota = (
                e.code == 429
                or "rate_limit" in error_body.lower()
                or "quota" in error_body.lower()
            )
            self._record_failure(
                f"HTTP {e.code}: {error_body}",
                http_status=int(e.code),
                quota_hit=quota,
            )
            logger.warning(
                "Groq HTTP error %d (%.0fms): %s",
                e.code, latency_ms, error_body[:120],
            )
            return "", False

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            self._record_failure(str(e))
            logger.warning("Groq call failed (%.0fms): %s", latency_ms, e)
            return "", False

    async def ask(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> tuple[str, bool]:
        if not self.is_available:
            return "", False
        if self._gateway is not None:
            try:
                allowed, reason = self._gateway.allow_cloud(query)
                if not allowed:
                    logger.info("Groq blocked by SecurityGateway: %s", reason)
                    return "", False
                query = self._gateway.sanitize_outbound(query)
            except Exception:
                logger.debug("SecurityGateway error on Groq.ask", exc_info=True)
        if not (query or "").strip():
            return "", False

        effective_model = model or self._buddy_model
        is_reasoning = effective_model == self._reasoning_model
        effective_timeout = (
            self._timeout_reasoning if is_reasoning else self._timeout
        )
        effective_max_tokens = max_tokens or (
            self._max_tokens_reasoning if is_reasoning else self._max_tokens
        )

        call_fn = functools.partial(
            self._call_sync,
            query,
            model_override=model,
            max_tokens_override=effective_max_tokens,
            system_instruction=system_instruction,
            temperature_override=temperature,
        )

        loop = asyncio.get_running_loop()
        try:
            text, ok = await asyncio.wait_for(
                loop.run_in_executor(None, call_fn),
                timeout=effective_timeout + 2.0,
            )
        except asyncio.TimeoutError:
            self._record_failure("async_timeout")
            logger.warning(
                "Groq async timeout (%.0fs, model=%s)",
                effective_timeout, effective_model,
            )
            return "", False

        if ok and self._gateway is not None:
            try:
                query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
                self._gateway.record_cloud_call(
                    query_hash=query_hash,
                    sanitized_length=len(query),
                    response_length=len(text),
                    latency_ms=self._total_latency_ms,
                    provider="groq",
                )
                tagged = self._gateway.tag_cloud_response(text, provider="groq")
                return tagged["text"], True
            except Exception:
                logger.debug("SecurityGateway audit on Groq raised", exc_info=True)
        return text, ok

    async def ask_buddy(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> tuple[str, bool]:
        default_system = (
            "You are ATOM, a personal AI assistant created by Satyam Yadav. "
            "You call him 'Boss'. Friendly, witty, concise. "
            "Keep voice replies to 1-2 sentences unless asked for detail. "
            "Never invent actions the user did not request. If the input "
            "looks like noisy STT, ask one short clarifying question."
        )
        return await self.ask(
            query,
            max_tokens=max_tokens or self._max_tokens,
            model=self._buddy_model,
            system_instruction=system_instruction or default_system,
            temperature=0.7,
        )

    async def ask_reasoning(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> tuple[str, bool]:
        default_system = (
            "You are ATOM, an advanced AI created by Satyam Yadav. "
            "Methodical, thorough, precise. Think step by step. "
            "Provide structured, well-grounded answers. State assumptions "
            "explicitly when the question is ambiguous; never fabricate "
            "specific numbers, dates, or quotes."
        )
        return await self.ask(
            query,
            max_tokens=max_tokens or self._max_tokens_reasoning,
            model=self._reasoning_model,
            system_instruction=system_instruction or default_system,
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
        """Streaming compatibility shim.

        Sprint Ω.8 v1 keeps the implementation synchronous: we issue a
        single non-streaming call and emit the whole response as one
        token to ``on_token`` so callers that expect a streaming surface
        (``router.py`` line 3046) keep working. Real SSE streaming can
        ride a follow-up sprint without touching call sites.
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
                logger.debug("Groq.ask_streaming on_token raised", exc_info=True)
        return text, ok

    # ── stats / circuit breaker ──────────────────────────────────

    def _record_success(self, latency_ms: float, qlen: int, rlen: int) -> None:
        self._total_requests += 1
        self._total_successes += 1
        self._total_latency_ms += latency_ms
        self._consecutive_failures = 0

    def _record_failure(
        self,
        reason: str,
        *,
        http_status: int = 0,
        quota_hit: bool = False,
    ) -> None:
        self._total_requests += 1
        self._total_failures += 1
        self._consecutive_failures += 1
        self._last_error = reason[:200]
        if quota_hit:
            self._circuit_open_until = time.monotonic() + self._quota_cooldown_s
        elif self._consecutive_failures >= self._circuit_threshold:
            self._circuit_open_until = (
                time.monotonic() + self._circuit_cooldown_s
            )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "provider": "groq",
            "buddy_model": self._buddy_model,
            "reasoning_model": self._reasoning_model,
            "enabled": self._enabled,
            "available": self.is_available,
            "total_requests": self._total_requests,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "avg_latency_ms": (
                self._total_latency_ms / self._total_successes
                if self._total_successes
                else 0.0
            ),
            "consecutive_failures": self._consecutive_failures,
            "circuit_open": time.monotonic() < self._circuit_open_until,
            "last_error": self._last_error,
        }
