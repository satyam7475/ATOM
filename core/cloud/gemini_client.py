"""
ATOM — Gemini Free-Tier Cloud Client.

Provides cloud-augmented reasoning via Google Gemini API (free tier).
Every request is gated through SecurityGateway — no raw user data ever
leaves the system.

Architecture:
  - stdlib urllib.request (zero external dependencies)
  - Async wrappers via asyncio.run_in_executor
  - SecurityGateway sanitizes ALL outbound queries
  - No conversation history, no memory context sent to cloud
  - 4-second hard timeout (fast-fail → local fallback)
  - Response tagged as untrusted via SecurityGateway

Free tier limits (as of 2026):
  - 15 RPM / 1M TPM / 1500 RPD
  - We self-limit to 10 RPM for safety margin

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

logger = logging.getLogger("atom.cloud.gemini")

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_MODEL = "gemini-2.0-flash"
_BUDDY_MODEL = "gemini-2.0-flash"       # Fast, conversational, buddy-like
_REASONING_MODEL = "gemini-2.5-flash"   # Deep reasoning, thinking, complex
_TIMEOUT_S = 8
_TIMEOUT_REASONING_S = 30
_MAX_TOKENS = 1024
_MAX_TOKENS_REASONING = 8192
_USER_AGENT = "ATOM-CognitiveOS/2.0"


class GeminiClient:
    """Stateless Gemini REST client with security gating.

    Usage:
        client = GeminiClient(config, security_gateway)
        if client.is_available:
            text, ok = await client.ask("Explain quantum computing briefly")
    """

    def __init__(
        self,
        config: dict | None = None,
        security_gateway: Any = None,
    ) -> None:
        cfg = (config or {}).get("cloud", {})
        self._api_key: str = cfg.get("gemini_api_key", "")
        self._model = cfg.get("model", _DEFAULT_MODEL)
        self._buddy_model = cfg.get("buddy_model", _BUDDY_MODEL)
        self._reasoning_model = cfg.get("reasoning_model", _REASONING_MODEL)
        self._timeout = float(cfg.get("timeout_seconds", _TIMEOUT_S))
        self._timeout_reasoning = float(cfg.get("timeout_reasoning_seconds", _TIMEOUT_REASONING_S))
        self._max_tokens = int(cfg.get("max_tokens", _MAX_TOKENS))
        self._max_tokens_reasoning = int(cfg.get("max_tokens_reasoning", _MAX_TOKENS_REASONING))
        self._enabled = bool(cfg.get("enabled", True))
        self._temperature = float(cfg.get("temperature", 0.7))

        self._gateway = security_gateway

        # Stats
        self._total_requests = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_latency_ms = 0.0
        self._last_error: str = ""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

        # Circuit breaker settings
        self._circuit_threshold = 3
        self._circuit_cooldown_s = 60.0
        # When Gemini replies with 429 RESOURCE_EXHAUSTED (daily free-tier
        # quota hit), there is no point retrying for 60s — the quota
        # typically only resets on a rolling window of minutes-to-hours.
        # Open the circuit immediately with a longer cooldown so every
        # buddy turn stops paying the 500-800ms 429-round-trip tax.
        self._quota_cooldown_s = 900.0  # 15 minutes

        # Thread-safe cancellation flag for streaming requests.
        # Set by cancel_streaming(); checked by _call_streaming_sync().
        self._streaming_cancelled = False

        if self._api_key:
            logger.info(
                "GeminiClient: buddy=%s, reasoning=%s, timeout=%.0f/%.0fs",
                self._buddy_model, self._reasoning_model,
                self._timeout, self._timeout_reasoning,
            )
        # Sprint Ω.1: deliberately do NOT log "no API key" here. The
        # vault probe in main.py runs *after* GeminiClient.__init__,
        # so the old log line was cosmetically misleading -- it
        # claimed cloud reasoning was disabled and then 200ms later
        # logged "API key configured". main.py owns the final
        # "key missing" warning when both settings.json AND the vault
        # come up empty, so the right call here is silence.

    def configure_api_key(self, key: str) -> None:
        """Set or update the API key at runtime (e.g., from vault)."""
        self._api_key = key
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        logger.info("GeminiClient: API key configured")

    @property
    def is_available(self) -> bool:
        """Whether the client can make requests right now."""
        if not self._enabled or not self._api_key:
            return False
        if time.monotonic() < self._circuit_open_until:
            return False
        return True

    @property
    def requests_remaining_estimate(self) -> int:
        """Rough estimate of remaining requests (based on self-imposed RPM)."""
        if self._gateway:
            return int(self._gateway._rate_limiter.tokens)
        return 10

    # ── Core API ─────────────────────────────────────────────────────

    def _build_request(
        self,
        query: str,
        *,
        model_override: str | None = None,
        max_tokens_override: int | None = None,
        system_instruction: str | None = None,
        temperature_override: float | None = None,
    ) -> tuple[str, bytes, dict[str, str]]:
        """Build the Gemini API request with optional model override."""
        model = model_override or self._model
        url = (
            f"{_GEMINI_API_URL}/{model}:generateContent"
            f"?key={self._api_key}"
        )

        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [{"text": query}],
                }
            ],
            "generationConfig": {
                "temperature": temperature_override if temperature_override is not None else self._temperature,
                "maxOutputTokens": max_tokens_override or self._max_tokens,
                "topP": 0.9,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
            ],
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
        """Synchronous API call — runs in executor for async."""
        t0 = time.perf_counter()
        effective_model = model_override or self._model

        try:
            url, body, headers = self._build_request(
                query,
                model_override=model_override,
                max_tokens_override=max_tokens_override,
                system_instruction=system_instruction,
                temperature_override=temperature_override,
            )
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST",
            )

            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                return "", False

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return "", False

            text = parts[0].get("text", "").strip()
            if not text:
                return "", False

            latency_ms = (time.perf_counter() - t0) * 1000
            self._record_success(latency_ms, len(query), len(text))

            logger.info(
                "Gemini [%s] response: %.0fms, %d chars in, %d chars out",
                effective_model, latency_ms, len(query), len(text),
            )
            return text, True

        except urllib.error.HTTPError as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                logger.debug('core cloud gemini client optional step failed', exc_info=True)
            self._record_failure(
                f"HTTP {e.code}: {error_body}",
                http_status=int(e.code),
                error_body=error_body,
            )
            logger.warning(
                "Gemini HTTP error %d (%.0fms): %s",
                e.code, latency_ms, error_body[:100],
            )
            return "", False

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            self._record_failure(str(e))
            logger.warning("Gemini call failed (%.0fms): %s", latency_ms, e)
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
        """Ask Gemini a question (async, security-gated).

        Args:
            query: The question to ask.
            max_tokens: Override max output tokens.
            model: Override model (e.g. 'gemini-2.5-flash' for reasoning).
            system_instruction: Optional system prompt for persona/behavior.
            temperature: Override temperature.

        Returns (response_text, success).
        """
        if not self.is_available:
            return "", False

        # Security gate
        if self._gateway:
            allowed, reason = self._gateway.allow_cloud(query)
            if not allowed:
                logger.info("Gemini blocked by SecurityGateway: %s", reason)
                return "", False
            query = self._gateway.sanitize_outbound(query)

        if not query.strip():
            return "", False

        # Determine timeout based on model type
        effective_model = model or self._model
        is_reasoning = effective_model == self._reasoning_model
        effective_timeout = self._timeout_reasoning if is_reasoning else self._timeout
        effective_max_tokens = max_tokens or (self._max_tokens_reasoning if is_reasoning else self._max_tokens)

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

            # Record to gateway audit
            if ok and self._gateway:
                query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
                self._gateway.record_cloud_call(
                    query_hash=query_hash,
                    sanitized_length=len(query),
                    response_length=len(text),
                    latency_ms=self._total_latency_ms,  # approximate
                    provider="gemini",
                )

            # Tag response as untrusted
            if ok and self._gateway:
                tagged = self._gateway.tag_cloud_response(
                    text, provider="gemini",
                )
                return tagged["text"], True

            return text, ok

        except asyncio.TimeoutError:
            self._record_failure("async_timeout")
            logger.warning("Gemini async timeout (%.0fs, model=%s)",
                          effective_timeout, effective_model)
            return "", False

    async def ask_buddy(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> tuple[str, bool]:
        """Ask using the fast conversational (buddy) model."""
        default_system = (
            "You are ATOM, a personal AI assistant (JARVIS-style) created by Satyam Yadav. "
            "You call him 'Boss'. You are friendly, witty, concise, and helpful. "
            "Keep responses short and conversational unless asked for detail. "
            "Never invent or promise actions the user did not explicitly request. "
            "If the query is unclear or looks like noisy transcription, ask ONE short "
            "clarifying question instead of guessing. Ground factual claims in supplied "
            "context; if you don't have the information, say so."
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
        """Ask using the deep reasoning model for complex tasks."""
        default_system = (
            "You are ATOM, an advanced AI system created by Satyam Yadav. "
            "You are methodical, thorough, and precise. "
            "Think step by step. Provide detailed, well-structured answers. "
            "For code, always include explanations. "
            "Ground every factual claim in supplied context or tool outputs; never "
            "fabricate specifics. If the question is ambiguous, state your assumption "
            "briefly before answering, or ask one clarifying question."
        )
        return await self.ask(
            query,
            max_tokens=max_tokens or self._max_tokens_reasoning,
            model=self._reasoning_model,
            system_instruction=system_instruction or default_system,
            temperature=0.4,
        )

    # ── Streaming API ─────────────────────────────────────────────────

    def cancel_streaming(self) -> None:
        """Signal the executor thread to abort the current streaming read.

        Safe to call from any thread or coroutine.  The next iteration of
        ``_call_streaming_sync``'s read loop will see the flag and exit.
        """
        self._streaming_cancelled = True

    def _call_streaming_sync(
        self,
        query: str,
        on_token: Callable[[str, bool], None],
        *,
        model_override: str | None = None,
        max_tokens_override: int | None = None,
        system_instruction: str | None = None,
        temperature_override: float | None = None,
    ) -> tuple[str, bool]:
        """Synchronous streaming call — reads SSE chunks line by line.

        Calls ``on_token(chunk_text, is_last)`` for every text fragment
        received from the Gemini ``streamGenerateContent`` endpoint.
        Runs in an executor; the callback must be thread-safe (e.g. use
        ``loop.call_soon_threadsafe``).
        """
        t0 = time.perf_counter()
        effective_model = model_override or self._model
        self._streaming_cancelled = False

        try:
            _, body, headers = self._build_request(
                query,
                model_override=model_override,
                max_tokens_override=max_tokens_override,
                system_instruction=system_instruction,
                temperature_override=temperature_override,
            )
            url = (
                f"{_GEMINI_API_URL}/{effective_model}"
                f":streamGenerateContent?alt=sse&key={self._api_key}"
            )
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST",
            )

            effective_timeout = (
                self._timeout_reasoning
                if effective_model == self._reasoning_model
                else self._timeout
            )
            full_text = ""
            cancelled = False

            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                # Cap per-read blocking to 100ms so the cancel flag is
                # checked promptly regardless of network buffer state.
                try:
                    resp.fp.raw._sock.settimeout(0.1)
                except Exception:
                    logger.debug('core cloud gemini client optional step failed', exc_info=True)
                resp_iter = iter(resp)
                while True:
                    if self._streaming_cancelled:
                        cancelled = True
                        logger.info("Gemini streaming cancelled by caller")
                        break
                    try:
                        raw_line = next(resp_iter)
                    except TimeoutError:
                        continue
                    except StopIteration:
                        break

                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        data = json.loads(payload_str)
                        candidates = data.get("candidates", [])
                        if not candidates:
                            continue
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if self._streaming_cancelled:
                                cancelled = True
                                break
                            chunk = part.get("text", "")
                            if chunk:
                                full_text += chunk
                                on_token(chunk, False)
                        if cancelled:
                            break
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            if not cancelled:
                on_token("", True)

            latency_ms = (time.perf_counter() - t0) * 1000
            if full_text:
                self._record_success(latency_ms, len(query), len(full_text))
                logger.info(
                    "Gemini streaming [%s]: %.0fms, %d chars",
                    effective_model, latency_ms, len(full_text),
                )
            return full_text, bool(full_text)

        except urllib.error.HTTPError as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                logger.debug('Stream end callback failed', exc_info=True)
            self._record_failure(
                f"HTTP {e.code}: {error_body}",
                http_status=int(e.code),
                error_body=error_body,
            )
            logger.warning(
                "Gemini streaming HTTP error %d (%.0fms): %s",
                e.code, latency_ms, error_body[:100],
            )
            return "", False

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            self._record_failure(str(e))
            logger.warning("Gemini streaming failed (%.0fms): %s", latency_ms, e)
            return "", False

    async def ask_streaming(
        self,
        query: str,
        *,
        on_token: Callable[[str, bool], None] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> tuple[str, bool]:
        """Ask Gemini with streaming response (async, security-gated).

        ``on_token(chunk, is_last)`` is called from the executor thread for
        each received token chunk.  The caller is responsible for making
        that callback thread-safe (e.g. ``loop.call_soon_threadsafe``).

        Falls back to the non-streaming ``ask()`` when no ``on_token``
        callback is provided.
        """
        if on_token is None:
            return await self.ask(
                query, max_tokens=max_tokens, model=model,
                system_instruction=system_instruction, temperature=temperature,
            )

        if not self.is_available:
            return "", False

        if self._gateway:
            allowed, reason = self._gateway.allow_cloud(query)
            if not allowed:
                logger.info("Gemini streaming blocked by SecurityGateway: %s", reason)
                return "", False
            query = self._gateway.sanitize_outbound(query)

        if not query.strip():
            return "", False

        effective_model = model or self._model
        is_reasoning = effective_model == self._reasoning_model
        effective_timeout = self._timeout_reasoning if is_reasoning else self._timeout
        effective_max_tokens = max_tokens or (
            self._max_tokens_reasoning if is_reasoning else self._max_tokens
        )

        call_fn = functools.partial(
            self._call_streaming_sync,
            query,
            on_token,
            model_override=model,
            max_tokens_override=effective_max_tokens,
            system_instruction=system_instruction,
            temperature_override=temperature,
        )

        loop = asyncio.get_running_loop()
        try:
            text, ok = await asyncio.wait_for(
                loop.run_in_executor(None, call_fn),
                timeout=effective_timeout + 5.0,
            )

            if ok and self._gateway:
                query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
                self._gateway.record_cloud_call(
                    query_hash=query_hash,
                    sanitized_length=len(query),
                    response_length=len(text),
                    latency_ms=0,
                    provider="gemini",
                )
                tagged = self._gateway.tag_cloud_response(text, provider="gemini")
                return tagged["text"], True

            return text, ok

        except asyncio.TimeoutError:
            self._record_failure("streaming_timeout")
            logger.warning(
                "Gemini streaming timeout (%.0fs, model=%s)",
                effective_timeout, effective_model,
            )
            return "", False

    # ── Circuit breaker ──────────────────────────────────────────────

    def _record_success(
        self, latency_ms: float, query_len: int, response_len: int,
    ) -> None:
        self._total_requests += 1
        self._total_successes += 1
        self._total_latency_ms += latency_ms
        self._consecutive_failures = 0
        self._last_error = ""

    def _record_failure(
        self,
        error: str,
        *,
        http_status: int | None = None,
        error_body: str = "",
    ) -> None:
        self._total_requests += 1
        self._total_failures += 1
        self._consecutive_failures += 1
        self._last_error = error

        # Fast-fail on 429 RESOURCE_EXHAUSTED: daily/project quota is
        # exhausted and will not recover within the normal 60s
        # cooldown. Open the circuit right away with an extended
        # cooldown so routing stops wasting every buddy turn on a
        # round-trip that's guaranteed to fail.
        quota_signal = False
        if http_status == 429:
            quota_signal = True
        elif error_body and (
            "RESOURCE_EXHAUSTED" in error_body
            or "exceeded your current quota" in error_body.lower()
            or "quota" in error_body.lower() and "exceed" in error_body.lower()
        ):
            quota_signal = True

        if quota_signal:
            now = time.monotonic()
            new_until = now + self._quota_cooldown_s
            if new_until > self._circuit_open_until:
                self._circuit_open_until = new_until
                logger.warning(
                    "GeminiClient circuit OPEN on quota exhaustion "
                    "(HTTP %s, cooldown=%.0fs) — falling back to local brain",
                    http_status or "?", self._quota_cooldown_s,
                )
            return

        if self._consecutive_failures >= self._circuit_threshold:
            self._circuit_open_until = (
                time.monotonic() + self._circuit_cooldown_s
            )
            logger.warning(
                "GeminiClient circuit OPEN (failures=%d, cooldown=%.0fs)",
                self._consecutive_failures, self._circuit_cooldown_s,
            )

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        avg_latency = (
            self._total_latency_ms / self._total_successes
            if self._total_successes > 0 else 0.0
        )
        return {
            "available": self.is_available,
            "model": self._model,
            "total_requests": self._total_requests,
            "successes": self._total_successes,
            "failures": self._total_failures,
            "avg_latency_ms": round(avg_latency, 1),
            "last_error": self._last_error,
            "circuit_open": time.monotonic() < self._circuit_open_until,
            "consecutive_failures": self._consecutive_failures,
        }


__all__ = ["GeminiClient"]
