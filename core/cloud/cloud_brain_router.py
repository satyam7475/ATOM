"""Cloud Brain Router -- confidence-gated escalation to a hosted LLM.

ATOM's local MLX brain handles every voice turn by default (sub-second
intents, 1-2 s LLM responses, complete privacy). The cloud router is
the *thinking-cap* layer Boss pulls on when the question really needs
it -- multi-step reasoning, source synthesis, code reviews, planning.

Decision matrix
---------------
::

    explicit "deep" cue ("think hard", "reason: ...", "deep:")
        -> cloud reasoning model (Gemini 2.5 Flash / Claude Opus)
    local failure / empty response (after one retry)
        -> cloud reasoning model
    long, multi-clause, "explain / compare / why / how would" question
        -> cloud reasoning model (subject to quota)
    everything else
        -> local MLX (no cloud call)

The router is intentionally cheap to call from the hot path: when the
gating question is a fast no-go, ``maybe_escalate`` returns ``None`` in
microseconds without touching the network.

It is also optional. If no :class:`GeminiClient` (or future Claude
client) is wired in, the router silently disables itself and the
gating heuristics still produce useful classification metadata for
Boss-facing telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.cloud.brain_router")


# ── Query classification regexes ────────────────────────────────────


_DEEP_PREFIXES = re.compile(
    r"^\s*(?:deep\s*:|think\s+(?:hard|deeply|carefully)|reason\s*:"
    r"|use\s+(?:the\s+)?cloud|switch\s+to\s+(?:gpt|gemini|claude|cloud))",
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r"\b(why\s+(?:is|does|do|did|would|should|could)|how\s+(?:does|do|would|could|should)"
    r"|compare\s+\w+\s+(?:and|vs|versus)|what\s+if|in\s+detail|step[-\s]?by[-\s]?step"
    r"|walk\s+me\s+through|explain\s+(?:me\s+)?(?:in\s+detail|how|why)|deep\s+dive"
    r"|trade[-\s]?offs?|pros\s+(?:and|vs)\s+cons|design\s+(?:a|the)|architect\s+(?:a|the)"
    r"|debug\s+this|optimize\s+(?:this|the)|review\s+(?:this|the))",
    re.IGNORECASE,
)

_LIGHT_INTENT_BLOCK = re.compile(
    r"^\s*(?:hi|hello|hey|yo|thanks?|thank\s+you|bye|goodnight|stop|cancel|wake|sleep|"
    r"play|pause|next|previous|skip|open|close|launch|quit|set\s+volume|"
    r"increase\s+volume|decrease\s+volume|mute|unmute|"
    r"who\s+(?:are\s+you|am\s+i)|what(?:\s+is|s|'s)?\s+(?:your|the)\s+name|"
    r"yes|yeah|yep|yup|sure|okay|ok|no|nope|nah)",
    re.IGNORECASE,
)


_EMPTY_RESPONSE_TOKENS = ("", "...", "…", "(", ")")


# ── Data classes ────────────────────────────────────────────────────


@dataclass(slots=True)
class CloudBrainConfig:
    """Tuning knobs for the router."""

    enabled: bool = True
    daily_quota: int = 60
    min_query_chars_for_auto_escalate: int = 80
    cooldown_after_failure_s: float = 30.0
    fallback_only: bool = False  # if True, ONLY escalate on local failure
    log_decisions: bool = False


@dataclass(slots=True)
class CloudDecision:
    """Result of :meth:`CloudBrainRouter.classify`."""

    use_cloud: bool
    reason: str
    profile: str  # "fast" | "deep" -- which cloud profile to pick


@dataclass(slots=True)
class CloudResult:
    """Outcome of :meth:`CloudBrainRouter.maybe_escalate`."""

    text: str
    provider: str
    profile: str
    latency_ms: float
    fallback_to_local: bool = False
    error: str | None = None


@dataclass(slots=True)
class _Stats:
    requests_today: int = 0
    successes: int = 0
    failures: int = 0
    last_request_at: float = 0.0
    last_failure_at: float = 0.0
    decisions: dict[str, int] = field(default_factory=dict)


# ── Router ──────────────────────────────────────────────────────────


class CloudBrainRouter:
    """Decide whether a query should escalate to the cloud LLM.

    Designed to be drop-in callable from the LocalBrainController or
    Router. The hot path (``classify``) is regex-only -- no network,
    no executor -- so it can run on every voice turn.
    """

    def __init__(
        self,
        gemini_client: Any = None,
        *,
        claude_client: Any = None,
        config: CloudBrainConfig | None = None,
    ) -> None:
        self.gemini = gemini_client
        self.claude = claude_client
        self.config = config or CloudBrainConfig()
        self._stats = _Stats()
        self._stat_lock = asyncio.Lock()
        self._enabled_runtime = bool(self.config.enabled and self._has_any_provider())

    # ── public surface ────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._enabled_runtime and self._under_quota()

    def reset_quota(self) -> None:
        """Manual reset (we use a wall-clock day boundary by default)."""
        self._stats.requests_today = 0
        self._stats.last_request_at = 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled_runtime,
            "available": self.is_available,
            "providers": [name for name in ("gemini", "claude") if getattr(self, name, None) is not None],
            "requests_today": self._stats.requests_today,
            "daily_quota": self.config.daily_quota,
            "successes": self._stats.successes,
            "failures": self._stats.failures,
            "last_request_at": self._stats.last_request_at,
            "decisions": dict(self._stats.decisions),
        }

    def classify(
        self,
        query: str,
        *,
        local_response: str | None = None,
        local_failed: bool = False,
        deep_hint: bool = False,
    ) -> CloudDecision:
        """Pure-CPU decision: should this query go to the cloud?

        ``local_response`` and ``local_failed`` are the post-local
        signals; pass them on the *retry* call after a failed local
        generation. On the *first* call only ``query`` and
        ``deep_hint`` matter.
        """

        clean = (query or "").strip()
        if not clean:
            return self._record_decision(False, "empty_query", "fast")

        if not self._enabled_runtime:
            return self._record_decision(False, "router_disabled", "fast")

        if not self._under_quota():
            return self._record_decision(False, "daily_quota_exhausted", "fast")

        if self.config.cooldown_after_failure_s > 0:
            cool = self.config.cooldown_after_failure_s
            since = time.time() - self._stats.last_failure_at
            if self._stats.last_failure_at > 0 and since < cool:
                return self._record_decision(False, "post_failure_cooldown", "fast")

        if deep_hint:
            return self._record_decision(True, "deep_hint", "deep")

        if _DEEP_PREFIXES.search(clean):
            return self._record_decision(True, "deep_prefix", "deep")

        if local_failed:
            return self._record_decision(True, "local_failed", "deep")

        if local_response is not None:
            stripped = local_response.strip()
            if stripped in _EMPTY_RESPONSE_TOKENS or len(stripped) < 6:
                return self._record_decision(True, "local_empty_after_retry", "deep")

        if self.config.fallback_only:
            # Skip the proactive heuristics in fallback-only mode.
            return self._record_decision(False, "fallback_only_mode", "fast")

        if _LIGHT_INTENT_BLOCK.match(clean):
            return self._record_decision(False, "light_intent", "fast")

        looks_reasoning = (
            _REASONING_PATTERNS.search(clean) is not None
            and len(clean) >= self.config.min_query_chars_for_auto_escalate
        )
        if looks_reasoning:
            return self._record_decision(True, "reasoning_pattern", "deep")

        return self._record_decision(False, "default_local", "fast")

    async def maybe_escalate(
        self,
        query: str,
        *,
        local_response: str | None = None,
        local_failed: bool = False,
        deep_hint: bool = False,
        system_instruction: str | None = None,
    ) -> CloudResult | None:
        """Run :meth:`classify` and, if it says go, call the cloud."""

        decision = self.classify(
            query,
            local_response=local_response,
            local_failed=local_failed,
            deep_hint=deep_hint,
        )
        if not decision.use_cloud:
            return None

        if self.config.log_decisions:
            logger.info(
                "Cloud router: escalating query=%r reason=%s profile=%s",
                query[:80], decision.reason, decision.profile,
            )

        provider, profile = self._pick_provider(decision.profile)
        if provider is None:
            return None

        t0 = time.perf_counter()
        text, ok = "", False
        try:
            if provider == "gemini":
                if profile == "deep" and hasattr(self.gemini, "ask_reasoning"):
                    text, ok = await self.gemini.ask_reasoning(
                        query, system_instruction=system_instruction,
                    )
                elif hasattr(self.gemini, "ask_buddy"):
                    text, ok = await self.gemini.ask_buddy(
                        query, system_instruction=system_instruction,
                    )
                else:
                    text, ok = await self.gemini.ask(query, system_instruction=system_instruction)
            elif provider == "claude" and self.claude is not None:
                if hasattr(self.claude, "ask"):
                    text, ok = await self.claude.ask(
                        query, system_instruction=system_instruction,
                    )
        except Exception as exc:
            await self._on_failure(reason=str(exc))
            logger.warning("Cloud router %s failed: %s", provider, exc)
            return CloudResult(
                text="",
                provider=provider,
                profile=profile,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                fallback_to_local=True,
                error=str(exc),
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        if ok and text.strip():
            await self._on_success()
            return CloudResult(text=text, provider=provider, profile=profile, latency_ms=latency_ms)

        await self._on_failure(reason="empty_or_unsuccessful")
        return CloudResult(
            text="",
            provider=provider,
            profile=profile,
            latency_ms=latency_ms,
            fallback_to_local=True,
            error="empty",
        )

    # ── internals ──────────────────────────────────────────

    def _has_any_provider(self) -> bool:
        if self.gemini is not None and getattr(self.gemini, "is_available", True):
            return True
        if self.claude is not None and getattr(self.claude, "is_available", True):
            return True
        return False

    def _under_quota(self) -> bool:
        if self.config.daily_quota <= 0:
            return True
        # Reset on day boundary.
        last_day = time.strftime("%Y-%m-%d", time.localtime(self._stats.last_request_at)) if self._stats.last_request_at else ""
        today = time.strftime("%Y-%m-%d")
        if last_day and last_day != today:
            self._stats.requests_today = 0
        return self._stats.requests_today < self.config.daily_quota

    def _pick_provider(self, profile: str) -> tuple[str | None, str]:
        if profile == "deep" and self.claude is not None and getattr(self.claude, "is_available", True):
            return "claude", profile
        if self.gemini is not None and getattr(self.gemini, "is_available", True):
            return "gemini", profile
        if self.claude is not None and getattr(self.claude, "is_available", True):
            return "claude", profile
        return None, profile

    def _record_decision(self, use_cloud: bool, reason: str, profile: str) -> CloudDecision:
        key = ("cloud:" if use_cloud else "local:") + reason
        self._stats.decisions[key] = self._stats.decisions.get(key, 0) + 1
        return CloudDecision(use_cloud=use_cloud, reason=reason, profile=profile)

    async def _on_success(self) -> None:
        async with self._stat_lock:
            self._stats.requests_today += 1
            self._stats.successes += 1
            self._stats.last_request_at = time.time()

    async def _on_failure(self, *, reason: str) -> None:
        async with self._stat_lock:
            self._stats.failures += 1
            self._stats.last_failure_at = time.time()
            self._stats.requests_today += 1  # still counts against quota
        logger.debug("Cloud router failure recorded: %s", reason)
