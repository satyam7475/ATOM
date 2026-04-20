"""
ATOM — Runtime watchdog + per-module execution budgets.

Two layers of protection:
  1. State dwell supervision:
     - THINKING / SPEAKING held too long -> recovery burst
  2. Active budget helpers for hot paths:
     - Intent classification
     - Cache lookup
     - RAG retrieval budget cap
     - LLM inference
     - TTS synthesis
     - Tool execution

This keeps ATOM responsive even when individual subsystems degrade.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

logger = logging.getLogger("atom.watchdog")


@dataclass
class BudgetResult:
    """Result envelope for a budgeted execution."""

    value: Any
    elapsed_ms: float
    timed_out: bool = False


class RuntimeWatchdog:
    """State watchdog + active execution-budget enforcer."""

    __slots__ = (
        "_bus", "_state", "_config", "_state_entered", "_task",
        "_shutdown", "_cooldown_s", "_last_recovery", "_think_s", "_speak_s",
        "_poll_interval",
        "_intent_s", "_cache_s", "_rag_s", "_llm_s", "_tts_s", "_tool_s",
        "_tts_started_at", "_local_brain", "_tts",
        "_consecutive_llm_timeouts", "_consecutive_tts_timeouts",
        "_timeout_demote_threshold", "_profile_demoted",
    )

    def __init__(
        self,
        bus: "AsyncEventBus",
        state: "StateManager",
        config: dict,
    ) -> None:
        self._bus = bus
        self._state = state
        self._config = config
        perf = config.get("performance", {}) or {}
        self._think_s = float(perf.get("watchdog_thinking_timeout_s", 120))
        self._speak_s = float(perf.get("watchdog_speaking_timeout_s", 300))
        self._intent_s = float(perf.get("watchdog_intent_timeout_ms", 50)) / 1000.0
        self._cache_s = float(perf.get("watchdog_cache_timeout_ms", 100)) / 1000.0
        self._rag_s = float(perf.get("watchdog_rag_timeout_ms", 500)) / 1000.0
        self._llm_s = float(perf.get("watchdog_llm_timeout_s", 30))
        self._tts_s = float(perf.get("watchdog_tts_timeout_s", 15))
        self._tool_s = float(perf.get("watchdog_tool_timeout_s", 10))
        self._cooldown_s = float(perf.get("supervisor_restart_cooldown_s", 8))
        self._poll_interval = float(perf.get("watchdog_poll_interval_s", 2.0))
        self._state_entered = time.monotonic()
        self._task: asyncio.Task | None = None
        self._shutdown = False
        self._last_recovery = time.monotonic() - self._cooldown_s
        self._tts_started_at = 0.0
        self._local_brain: Any = None
        self._tts: Any = None
        # Stability: track *consecutive* budget timeouts per stage so we can
        # escalate to a profile demotion (instead of silently looping). Reset
        # on the first successful response/turn.
        self._consecutive_llm_timeouts = 0
        self._consecutive_tts_timeouts = 0
        self._timeout_demote_threshold = int(
            perf.get("watchdog_timeout_demote_threshold", 2),
        )
        self._profile_demoted = False

        self._bus.on("response_ready", self._on_tts_started)
        self._bus.on("partial_response", self._on_tts_started)
        self._bus.on("tts_complete", self._on_tts_complete)
        self._bus.on("response_complete", self._on_successful_turn)
        self._bus.on("response_emitted", self._on_successful_turn)

    def attach_local_brain(self, brain: Any) -> None:
        """Attach LocalBrainController so LLM timeouts can preempt/unload it."""
        self._local_brain = brain

    def attach_tts(self, tts: Any) -> None:
        """Attach TTS engine so a stuck synth can be force-stopped on timeout."""
        self._tts = tts

    def timeout_s(self, stage: str) -> float:
        mapping = {
            "intent_engine": self._intent_s,
            "cache_lookup": self._cache_s,
            "rag_retrieval": self._rag_s,
            "llm_inference": self._llm_s,
            "tts_synthesis": self._tts_s,
            "tool_execution": self._tool_s,
        }
        return float(mapping.get(stage, 0.0))

    def cap_budget_ms(self, stage: str, budget_ms: float) -> float:
        timeout_s = self.timeout_s(stage)
        if timeout_s <= 0:
            return float(budget_ms)
        return min(float(budget_ms), timeout_s * 1000.0)

    async def run_sync(
        self,
        stage: str,
        func: Callable[..., Any],
        *args: Any,
        default: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetResult:
        """Run sync work in the executor with a stage-specific timeout."""
        timeout_s = self.timeout_s(stage)
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            if timeout_s > 0:
                value = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args),
                    timeout=timeout_s,
                )
            else:
                value = await loop.run_in_executor(None, func, *args)
            return BudgetResult(
                value=value,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                timed_out=False,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._handle_budget_timeout(stage, timeout_s, metadata=metadata)
            return BudgetResult(value=default, elapsed_ms=elapsed_ms, timed_out=True)

    async def run_async(
        self,
        stage: str,
        awaitable: Awaitable[Any],
        *,
        default: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetResult:
        """Run async work with a stage-specific timeout."""
        timeout_s = self.timeout_s(stage)
        t0 = time.perf_counter()
        try:
            if timeout_s > 0:
                value = await asyncio.wait_for(awaitable, timeout=timeout_s)
            else:
                value = await awaitable
            return BudgetResult(
                value=value,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                timed_out=False,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._handle_budget_timeout(stage, timeout_s, metadata=metadata)
            return BudgetResult(value=default, elapsed_ms=elapsed_ms, timed_out=True)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._shutdown = False
            self._task = asyncio.create_task(
                self._loop(), name="atom_runtime_watchdog"
            )
            logger.info(
                "RuntimeWatchdog started (think=%.0fs speak=%.0fs intent=%.0fms cache=%.0fms rag=%.0fms llm=%.0fs tts=%.0fs tool=%.0fs cooldown=%.0fs)",
                self._think_s,
                self._speak_s,
                self._intent_s * 1000.0,
                self._cache_s * 1000.0,
                self._rag_s * 1000.0,
                self._llm_s,
                self._tts_s,
                self._tool_s,
                self._cooldown_s,
            )

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def on_state_changed(
        self,
        old: Any = None,
        new: Any = None,
        **_kw: Any,
    ) -> None:
        self._state_entered = time.monotonic()
        if getattr(new, "value", "") != "speaking":
            self._tts_started_at = 0.0

    async def _on_tts_started(
        self,
        text: str = "",
        is_first: bool = False,
        **_kw: Any,
    ) -> None:
        if not text.strip():
            return
        if self._tts_started_at > 0 and not is_first:
            return
        self._tts_started_at = time.monotonic()

    async def _on_tts_complete(self, **_kw: Any) -> None:
        self._tts_started_at = 0.0

    async def _on_successful_turn(self, **_kw: Any) -> None:
        """Clear consecutive-timeout counters when a turn actually lands."""
        if self._consecutive_llm_timeouts or self._consecutive_tts_timeouts:
            logger.debug(
                "Watchdog: clearing consecutive timeouts after successful turn "
                "(llm=%d tts=%d)",
                self._consecutive_llm_timeouts,
                self._consecutive_tts_timeouts,
            )
        self._consecutive_llm_timeouts = 0
        self._consecutive_tts_timeouts = 0

    def _maybe_demote_profile(self, trigger: str) -> None:
        """Ask local-brain to drop to a lighter profile after repeated timeouts.

        Safe to call multiple times; demotes only once per session unless
        someone explicitly reverts profile via router/CLI.
        """
        if self._profile_demoted:
            return
        brain = self._local_brain
        if brain is None:
            return
        demote_fn = getattr(brain, "request_profile_demote", None)
        if not callable(demote_fn):
            # Fall back to the legacy unload path; better than nothing when
            # the brain doesn't expose an explicit demote hook.
            try:
                brain.unload_llm_for_power()
            except Exception:
                logger.debug("Watchdog fallback demote failed", exc_info=True)
            self._profile_demoted = True
            return
        try:
            demote_fn(reason=trigger)
            self._profile_demoted = True
            logger.warning(
                "Watchdog: demoted brain profile after %d consecutive %s "
                "timeouts (trigger=%s). ATOM is in lite mode; say "
                "\"go fast mode\" to keep it, or \"go smart mode\" to retry.",
                self._timeout_demote_threshold,
                "LLM" if "llm" in trigger else "TTS",
                trigger,
            )
            try:
                self._bus.emit(
                    "text_display",
                    text=(
                        "I've dropped to a lighter model because responses "
                        "were timing out. Ready when you are, Boss."
                    ),
                )
            except Exception:
                logger.debug("Demote notification emit failed", exc_info=True)
        except Exception:
            logger.warning(
                "Watchdog: profile demote request failed",
                exc_info=True,
            )

    def _handle_budget_timeout(
        self,
        stage: str,
        timeout_s: float,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        details = ""
        if metadata:
            detail_parts = [f"{k}={v}" for k, v in metadata.items() if v not in (None, "", [])]
            if detail_parts:
                details = " | " + ", ".join(detail_parts[:4])

        logger.warning(
            "Runtime budget exceeded: %s (limit=%.3fs)%s",
            stage,
            timeout_s,
            details,
        )
        try:
            self._bus.emit_fast("metrics_event", counter="errors_total")
        except Exception:
            logger.debug('Fast bus emit failed', exc_info=True)

        if stage == "llm_inference":
            self._consecutive_llm_timeouts += 1
            if self._local_brain is not None:
                try:
                    self._local_brain.request_preempt()
                except Exception:
                    logger.debug("Watchdog could not preempt local brain", exc_info=True)
                try:
                    self._local_brain.unload_llm_for_power()
                except Exception:
                    logger.debug("Watchdog could not unload local brain", exc_info=True)
            try:
                self._bus.emit("llm_error", source="watchdog", error="llm_timeout")
            except Exception:
                logger.debug('Metrics counter emit failed', exc_info=True)
            if self._consecutive_llm_timeouts >= self._timeout_demote_threshold:
                self._maybe_demote_profile("llm_timeout_streak")
            self._maybe_recover(f"LLM inference timed out after {timeout_s:.1f}s")
            return

        if stage == "tts_synthesis":
            self._consecutive_tts_timeouts += 1
            self._tts_started_at = 0.0
            try:
                self._bus.emit("text_display", text="[Watchdog] TTS timed out; audio skipped.")
            except Exception:
                logger.debug('Metrics counter emit failed', exc_info=True)
            # Force-stop the synth so we don't sit in SPEAKING state spinning
            # the run-loop on a stuck utterance. Without this the audio engine
            # stays bound and the next reply piles up behind it.
            tts = self._tts
            if tts is not None:
                try:
                    stop_attr = getattr(tts, "stop", None)
                    if stop_attr is not None:
                        result = stop_attr()
                        if asyncio.iscoroutine(result):
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                loop = None
                            if loop is not None:
                                loop.create_task(result)
                            else:
                                asyncio.run(result)
                except Exception:
                    logger.debug("Watchdog TTS stop failed", exc_info=True)
            try:
                self._bus.emit("tts_complete", reason="watchdog_timeout")
            except Exception:
                logger.debug("tts_complete emit failed", exc_info=True)
            if self._consecutive_tts_timeouts >= self._timeout_demote_threshold:
                self._maybe_demote_profile("tts_timeout_streak")
            self._maybe_recover(
                f"TTS synthesis timed out after {timeout_s:.1f}s",
                schedule_restart=False,
            )
            return

    def _maybe_recover(self, reason: str, *, schedule_restart: bool = True) -> None:
        now = time.monotonic()
        if now - self._last_recovery < self._cooldown_s:
            logger.debug("Watchdog skip %s (cooldown)", reason)
            return
        self._last_recovery = now
        logger.warning("Watchdog recovery: %s", reason)
        self._bus.emit("metrics_event", counter="watchdog_recoveries")
        self._bus.emit("resume_listening")
        if schedule_restart:
            asyncio.get_running_loop().call_later(
                0.05, lambda: self._bus.emit("restart_listening")
            )

    async def _loop(self) -> None:
        from core.state_manager import AtomState

        while not self._shutdown:
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            if self._shutdown:
                break
            st = self._state.current
            elapsed = time.monotonic() - self._state_entered
            if self._tts_started_at > 0:
                tts_elapsed = time.monotonic() - self._tts_started_at
                if tts_elapsed > self._tts_s:
                    self._handle_budget_timeout(
                        "tts_synthesis",
                        self._tts_s,
                        metadata={"elapsed_s": round(tts_elapsed, 2)},
                    )
            if st is AtomState.THINKING and elapsed > self._think_s:
                self._maybe_recover(f"THINKING stuck {elapsed:.0f}s")
            elif st is AtomState.SPEAKING and elapsed > self._speak_s:
                self._maybe_recover(f"SPEAKING stuck {elapsed:.0f}s")
