"""
ATOM -- STT Self-Healing Watchdog.

Monitors the STT engine for stuck/silent states and automatically
restarts it. Detects:

  1. No partials for > N seconds while mic should be active
  2. Recognition chain stuck (task completed but not restarted)
  3. Audio engine stopped unexpectedly
  4. Repeated errors without recovery

Runs as a lightweight async task alongside the STT loop.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

logger = logging.getLogger("atom.stt_watchdog")

_DIAGNOSTIC_PARTIAL_RE = re.compile(
    r"\b(rag|rack|memory|embedding|snippet|boot|diagnostic|watchdog|pressure)\b"
    r".*\b(pressure mode|snippet budget|engine shut ?down|boot diagnostic|budget reduced)\b",
    re.I,
)
_SELF_SPEECH_PARTIAL_RE = re.compile(
    r"^\s*(?:what\s+do\s+you\s+need|one\s+moment|working\s+on\s+it|"
    r"right\s+away|on\s+it(?:\s+boss)?|let\s+me\s+check|"
    r"give\s+me\s+a\s+sec)\s*[.?!]?\s*$",
    re.I,
)

_SILENT_TIMEOUT_S = 8.0
_STUCK_TIMEOUT_S = 15.0
_MAX_RESTARTS_PER_WINDOW = 3
# Sprint A5 → K1: shrink the breaker cooldown from 300s → 60s → 30s.
# Five minutes of forced deafness in the middle of a session is
# unacceptable. The Apple framework typically clears whatever made it
# choke within ~30 seconds, and the audible "STT recovering"
# announcement (`_speak_breaker_open`) means Boss is never surprised.
_RESTART_WINDOW_S = 30.0
_CHECK_INTERVAL_S = 2.0
# After this many consecutive chain restarts (with SFSpeechRecognizer
# recreate) that still produce zero partials, escalate to a full engine
# restart — stop_listening + recreate_recognizer + start_listening, which
# also rebuilds the sounddevice stream and CoreAudio binding. Keeping this
# low is important because if chain-level recreate cannot recover (e.g.
# stale CoreAudio session), spending more attempts there just delays the
# real fix.
_CHAIN_RESTARTS_BEFORE_FULL = 2
# If a full-restart sequence produces zero productive partials within this
# window, we count it as a failed recovery. Accumulating the configured
# threshold of failed recoveries triggers engine failover (to Whisper).
_FULL_RESTART_PRODUCTIVITY_WINDOW_S = 10.0
_FULL_RESTART_FAILURES_BEFORE_SWAP = 3


class STTWatchdog:
    """Monitors STT health and triggers auto-recovery."""

    def __init__(
        self,
        bus: Any,
        *,
        silent_timeout_s: float = _SILENT_TIMEOUT_S,
        stuck_timeout_s: float = _STUCK_TIMEOUT_S,
    ) -> None:
        self._bus = bus
        self._silent_timeout = max(3.0, float(silent_timeout_s))
        self._stuck_timeout = max(5.0, float(stuck_timeout_s))

        self._last_partial_time: float = time.monotonic()
        self._last_final_time: float = time.monotonic()
        self._last_tap_count: int = 0
        self._stt_ref: Any = None
        # Tighter starvation threshold once we've had at least one
        # successful turn — the recognizer is warmed up so a silent
        # stretch is more likely a real stall than a slow cold start.
        self._had_successful_turn: bool = False

        self._restart_times: list[float] = []
        self._total_restarts: int = 0
        self._total_silent_detections: int = 0
        self._consecutive_chain_restarts: int = 0
        self._was_listening: bool = False
        # Full-restart failure tracking for engine failover to Whisper.
        self._full_restart_attempts: int = 0
        self._full_restart_failures: int = 0
        self._last_full_restart_time: float = 0.0
        self._failover_pending: bool = False
        self._degraded_announced: bool = False

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    def attach_stt(self, stt: Any) -> None:
        self._stt_ref = stt

    def reset_timers(self) -> None:
        """Reset liveness timers -- call when STT (re)starts listening."""
        now = time.monotonic()
        self._last_partial_time = now
        self._last_final_time = now
        self._last_tap_count = 0

    async def on_external_chain_restart(self, reason: str = "", **_kw: Any) -> None:
        """Called when the STT engine self-heals (e.g. reactive
        kLSRErrorDomain 301 chain restart in stt_macos.py).

        Without this, the watchdog's ``_last_final_time`` keeps
        growing during long silences and we falsely log
        ``STT Watchdog: stuck for 28.8s`` even though the engine
        already recovered. Resetting both timers tells the watchdog
        "we're listening fresh, stop counting from the previous
        final".
        """
        if not reason or "reactive_klsr" not in str(reason):
            return
        now = time.monotonic()
        self._last_partial_time = now
        self._last_final_time = now
        self._consecutive_chain_restarts = 0

    async def on_speech_partial(self, text: str = "", **_kw: Any) -> None:
        """Called from bus on every STT partial -- updates liveness."""
        now = time.monotonic()
        self._last_partial_time = now
        self._consecutive_chain_restarts = 0
        # A productive partial proves the most recent full restart worked;
        # reset the failover failure counter so noise doesn't accumulate.
        if (self._last_full_restart_time > 0
                and (now - self._last_full_restart_time)
                < _FULL_RESTART_PRODUCTIVITY_WINDOW_S
                and text and text.strip()):
            if self._full_restart_failures > 0:
                logger.info(
                    "STT Watchdog: productive partial after full restart — "
                    "resetting failure counter (%d -> 0)",
                    self._full_restart_failures,
                )
            self._full_restart_failures = 0
            self._last_full_restart_time = 0.0

    async def on_speech_final(self, text: str = "", **_kw: Any) -> None:
        """Called from bus on every STT final -- updates liveness."""
        self._last_final_time = time.monotonic()
        self._last_partial_time = time.monotonic()
        if text and text.strip():
            self._had_successful_turn = True
        if (self._last_full_restart_time > 0
                and text and text.strip()):
            self._full_restart_failures = 0
            self._last_full_restart_time = 0.0

    async def on_needs_full_restart(self, reason: str = "", **_kw: Any) -> None:
        """Bus handler for hardened-STT escalation signals (empty-final
        cascade, recreate storm). Forces a full engine restart."""
        if reason == "native_permanently_degraded":
            self._failover_pending = True
            self._emit_degraded_once(reason=reason)
            logger.error(
                "STT Watchdog: native STT permanently degraded; "
                "suppressing further restart attempts",
            )
            return
        stt = self._stt_ref
        if stt is None:
            return
        if not self._can_restart():
            return
        logger.warning(
            "STT Watchdog: escalation from engine (%s) — forcing full restart",
            reason or "unspecified",
        )
        # Jump straight to full restart — the engine already decided chain
        # restarts won't help.
        self._consecutive_chain_restarts = _CHAIN_RESTARTS_BEFORE_FULL
        await self._restart_stt(stt, f"engine_escalation:{reason}")

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        # Subscribe to the bus event the STT engine emits when it
        # self-heals from a benign 301 idle timeout. Done here rather
        # than __init__ so we still construct cleanly in unit tests
        # that pass a mock bus without ``on``.
        try:
            self._bus.on("stt_watchdog_restart", self.on_external_chain_restart)
        except Exception:
            logger.debug(
                "STT Watchdog: bus.on(stt_watchdog_restart) not available "
                "(test mock?); external 301 resets will not propagate.",
                exc_info=True,
            )
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("STT Watchdog started (silent=%.0fs stuck=%.0fs)",
                     self._silent_timeout, self._stuck_timeout)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _monitor_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=_CHECK_INTERVAL_S,
                )
                break
            except asyncio.TimeoutError:
                pass

            try:
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("STT Watchdog check error", exc_info=True)

    async def _check_health(self) -> None:
        stt = self._stt_ref
        if stt is None:
            return
        if self._failover_pending:
            return

        now = time.monotonic()
        is_listening = getattr(stt, "_listening", False)
        is_running = getattr(stt, "_running_async", False)

        if not is_running:
            self._was_listening = False
            return

        # Auto-reset timers on listening transition (fixes premature restart after TTS)
        if is_listening and not self._was_listening:
            self.reset_timers()
            self._was_listening = True
            return
        if not is_listening:
            self._was_listening = False
            return

        state_mgr = getattr(stt, "_state", None)
        if state_mgr is not None:
            try:
                from core.state_manager import AtomState
                cur = state_mgr.current
                if cur is AtomState.SPEAKING:
                    return
            except Exception:
                logger.debug('voice stt watchdog optional step failed', exc_info=True)

        since_partial = now - self._last_partial_time
        since_final = now - self._last_final_time

        tap_count = getattr(stt, "_tap_buffer_count", 0)
        audio_flowing = tap_count > self._last_tap_count
        self._last_tap_count = tap_count
        last_audio_rms_db = float(getattr(stt, "_last_audio_rms_db", -96.0))
        last_speech_candidate = float(getattr(stt, "_last_speech_candidate_time", 0.0) or 0.0)
        speech_likely = (now - last_speech_candidate) < 3.0 if last_speech_candidate > 0 else False

        # Guard: don't restart if we received a partial very recently
        last_partial_text = str(getattr(stt, "_last_partial", "") or "")
        recent_partial = since_partial < 3.0

        def _should_salvage_partial(text: str) -> bool:
            text = (text or "").strip()
            if not text:
                return False
            if (
                _DIAGNOSTIC_PARTIAL_RE.search(text)
                or _SELF_SPEECH_PARTIAL_RE.search(text)
            ):
                return False
            echo_guard = getattr(stt, "_echo_guard", None)
            if callable(echo_guard):
                try:
                    if echo_guard(text):
                        return False
                except Exception:
                    logger.debug("STT Watchdog: echo guard check failed", exc_info=True)
            return True

        # After the first successful turn the recognizer is warmed up — a
        # silent stretch is a much stronger signal of a real stall, so we
        # drop the threshold from 8s to 5s. On the very first turn we keep
        # the more generous budget to avoid restarting a recognizer that's
        # simply slow to emit its first partial.
        effective_silent_timeout = (
            max(4.0, self._silent_timeout - 3.0)
            if self._had_successful_turn
            else self._silent_timeout
        )

        if is_listening and since_partial > effective_silent_timeout and audio_flowing and speech_likely:
            if recent_partial:
                return
            self._total_silent_detections += 1
            logger.warning(
                "STT Watchdog: no partials for %.1fs despite speech-like audio "
                "(tap_count=%d, rms=%.1f dB) -- recognizer likely starved",
                since_partial, tap_count, last_audio_rms_db,
            )
            starvation_fn = getattr(stt, "_on_recognition_starvation", None)
            if callable(starvation_fn):
                try:
                    starvation_fn()
                except Exception:
                    logger.debug("STT Watchdog: recognition starvation hook failed", exc_info=True)
            if self._can_restart():
                # Preserve accumulated partial before restart only when it
                # is not ATOM's own delayed TTS captured by the mic.
                if _should_salvage_partial(last_partial_text):
                    logger.info(
                        "STT Watchdog: salvaging partial '%s' before restart",
                        last_partial_text[:120],
                    )
                    self._bus.emit_fast("speech_partial", text=last_partial_text)
                await self._restart_stt(stt, "recognition_starved")
        elif is_listening and since_partial > effective_silent_timeout and audio_flowing:
            logger.debug(
                "STT Watchdog: audio is flowing but no speech-like activity yet "
                "(tap_count=%d, rms=%.1f dB) -- suppressing restart",
                tap_count, last_audio_rms_db,
            )

        if is_listening and since_final > self._stuck_timeout and since_partial > self._stuck_timeout:
            last_error = getattr(stt, "_last_error", None)
            if last_error:
                err_str = str(last_error)
                # kLSRErrorDomain code=301 is Apple's "recognition session
                # cancelled/expired" signal — it fires whenever the session
                # sits idle for a while. It is NOT fatal; escalating it to
                # a full engine rebuild causes an avoidable race with
                # AudioIntelligenceEngine's device switch and produces the
                # `PaMacCore err=-50` cascade seen in the field. Prefer a
                # lightweight chain-restart here.
                is_soft_timeout = "kLSRErrorDomain" in err_str and "code=301" in err_str

                # 301 is benign idle expiry — log at INFO so the user-
                # visible boot log doesn't shout "stuck for 28s" every
                # time they pause for half a minute.
                log = logger.info if is_soft_timeout else logger.warning
                log(
                    "STT Watchdog: %s for %.1fs with error '%s' -- %s",
                    "idle" if is_soft_timeout else "stuck",
                    since_final,
                    err_str[:80],
                    "soft chain-restart (kLSRErrorDomain 301)" if is_soft_timeout else "restarting",
                )
                if self._can_restart():
                    if _should_salvage_partial(last_partial_text):
                        logger.info(
                            "STT Watchdog: salvaging partial '%s' before restart",
                            last_partial_text[:120],
                        )
                        self._bus.emit_fast("speech_partial", text=last_partial_text)
                    if is_soft_timeout:
                        await self._soft_chain_restart(stt, reason="klsr_301_timeout")
                    else:
                        await self._restart_stt(stt, "stuck_with_error")

    def _can_restart(self) -> bool:
        now = time.monotonic()
        before = len(self._restart_times)
        self._restart_times = [
            t for t in self._restart_times
            if now - t < _RESTART_WINDOW_S
        ]
        # Detect the "breaker just closed" edge: previously OPEN (logged)
        # and now back below the threshold. Tell the user so they know
        # ATOM can hear them again -- the A5 user-experience fix.
        breaker_was_open = getattr(self, "_circuit_open_logged", False)
        if len(self._restart_times) >= _MAX_RESTARTS_PER_WINDOW:
            if not breaker_was_open:
                logger.warning(
                    "STT Watchdog: circuit breaker OPEN — %d restarts in %.0fs window, "
                    "suppressing further restarts until window clears",
                    len(self._restart_times), _RESTART_WINDOW_S,
                )
                self._circuit_open_logged = True
                self._speak_breaker_open()
                self._emit_degraded_once(
                    reason=(
                        f"{len(self._restart_times)} restarts in "
                        f"{_RESTART_WINDOW_S:.0f}s"
                    ),
                )
            return False
        if breaker_was_open and len(self._restart_times) < _MAX_RESTARTS_PER_WINDOW:
            logger.info(
                "STT Watchdog: circuit breaker CLOSED — %d/%d restarts in window, "
                "STT recovery permitted again",
                len(self._restart_times), _MAX_RESTARTS_PER_WINDOW,
            )
            self._speak_breaker_recovered()
        if before > 0 and len(self._restart_times) == 0:
            # Window cleared completely; suppress any stale flag.
            pass
        self._circuit_open_logged = False
        return True

    # ── audible breaker UX ──────────────────────────────────────

    def _speak_breaker_open(self) -> None:
        """Announce to the user that ATOM has stopped listening so the
        long silence isn't mysterious. Best-effort -- never raises."""
        try:
            self._bus.emit_fast(
                "tts_say",
                text="STT recovering, give me a moment, Boss.",
                source="stt_watchdog",
            )
        except Exception:
            logger.debug("breaker_open speak failed", exc_info=True)

    def _emit_degraded_once(self, *, reason: str) -> None:
        """One-shot hard degradation event for UI/voice-loop fallbacks."""
        if self._degraded_announced:
            return
        self._degraded_announced = True
        try:
            self._bus.emit_fast(
                "stt_degraded",
                reason=reason,
                text=(
                    "Speech recognition is unstable, Boss. "
                    "I'll stop restarting it for a moment."
                ),
            )
        except Exception:
            logger.debug("stt_degraded emit failed", exc_info=True)

    def _speak_breaker_recovered(self) -> None:
        try:
            self._bus.emit_fast(
                "tts_say",
                text="Listening again, Boss.",
                source="stt_watchdog",
            )
        except Exception:
            logger.debug("breaker_recovered speak failed", exc_info=True)

    async def _restart_stt(self, stt: Any, reason: str) -> None:
        from voice.recovery_lock import voice_recovery_lock

        async with voice_recovery_lock(
            f"stt_watchdog:{reason}",
            max_wait_s=2.0,
        ) as got_lock:
            if not got_lock:
                logger.info(
                    "STT Watchdog: skipping '%s' restart — another voice "
                    "recovery path is in flight",
                    reason,
                )
                return
            await self._restart_stt_locked(stt, reason)

    async def _soft_chain_restart(self, stt: Any, *, reason: str) -> None:
        """Lightweight chain-level restart for benign Apple-framework
        timeouts (kLSRErrorDomain 301). Only touches the SFSpeechRecognition
        chain — NOT the audio engine / CoreAudio tap — so it can't race
        with AudioIntelligenceEngine.seamless_switch and produce the
        ``PaMacCore err=-50`` cascade.

        Still serialised through ``voice_recovery_lock`` so overlapping
        soft + hard restarts inside the same watchdog cycle are impossible.
        """
        from voice.recovery_lock import voice_recovery_lock

        async with voice_recovery_lock(
            f"stt_watchdog_soft:{reason}",
            max_wait_s=1.0,
        ) as got_lock:
            if not got_lock:
                logger.info(
                    "STT Watchdog: skipping soft chain restart (%s) — another "
                    "voice recovery path is in flight",
                    reason,
                )
                return

            self._total_restarts += 1
            self._consecutive_chain_restarts += 1
            self._restart_times.append(time.monotonic())

            restart_fn = getattr(stt, "_restart_recognition_chain", None)
            if not callable(restart_fn):
                logger.info(
                    "STT Watchdog: soft chain restart requested but backend "
                    "'%s' exposes no _restart_recognition_chain — falling back "
                    "to full restart",
                    type(stt).__name__,
                )
                await self._restart_stt_locked(stt, f"soft_fallback:{reason}")
                return

            try:
                # Ω.10 step-6: heavy pool for the soft chain restart so a
                # wedged STT recovery cannot block the default 3-worker
                # pool that boot warm-up shares.
                from core.async_event_bus import get_heavy_executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(get_heavy_executor(), restart_fn)
                # Reset BOTH liveness timers — without resetting
                # ``_last_final_time`` the watchdog's next health
                # check will still see the previous (now-irrelevant)
                # silence as "stuck" and re-trigger.
                now_t = time.monotonic()
                self._last_partial_time = now_t
                self._last_final_time = now_t
                logger.info(
                    "STT Watchdog: soft chain-restart completed (%s, #%d)",
                    reason, self._total_restarts,
                )
                self._bus.emit_fast(
                    "stt_watchdog_restart",
                    reason=f"soft:{reason}",
                    restart_count=self._total_restarts,
                )
            except Exception:
                logger.warning(
                    "STT Watchdog: soft chain-restart failed (%s) — escalating",
                    reason,
                    exc_info=True,
                )
                await self._restart_stt_locked(stt, f"soft_fallback:{reason}")

    async def _restart_stt_locked(self, stt: Any, reason: str) -> None:
        self._total_restarts += 1
        self._consecutive_chain_restarts += 1
        self._restart_times.append(time.monotonic())

        needs_full_restart = self._consecutive_chain_restarts >= _CHAIN_RESTARTS_BEFORE_FULL

        if needs_full_restart:
            logger.warning(
                "STT Watchdog: %d consecutive chain restarts with no partials — "
                "escalating to full engine restart with device rebinding (#%d)",
                self._consecutive_chain_restarts, self._total_restarts,
            )
            self._consecutive_chain_restarts = 0
            # Before running the restart, check whether the PREVIOUS full
            # restart produced any productive output. If not, bump the
            # failure counter; once we hit the swap threshold, ask the
            # pipeline to swap engines (one-shot) and skip further native
            # recovery in this session.
            now = time.monotonic()
            prev = self._last_full_restart_time
            if prev > 0:
                since_prev = now - prev
                if since_prev >= _FULL_RESTART_PRODUCTIVITY_WINDOW_S:
                    # We only reach here if no productive partial landed in
                    # on_speech_partial within the window; that resets the
                    # counter. Therefore this is a failure.
                    self._full_restart_failures += 1
                    logger.warning(
                        "STT Watchdog: previous full restart produced no partials "
                        "within %.0fs — failure %d/%d",
                        _FULL_RESTART_PRODUCTIVITY_WINDOW_S,
                        self._full_restart_failures,
                        _FULL_RESTART_FAILURES_BEFORE_SWAP,
                    )
                    if (self._full_restart_failures
                            >= _FULL_RESTART_FAILURES_BEFORE_SWAP
                            and not self._failover_pending):
                        self._failover_pending = True
                        logger.error(
                            "STT Watchdog: %d full-restart failures — asking "
                            "pipeline to swap to Whisper",
                            self._full_restart_failures,
                        )
                        try:
                            self._bus.emit_fast(
                                "stt_swap_to_whisper",
                                reason=f"{self._full_restart_failures}_full_restart_failures",
                            )
                        except Exception:
                            logger.debug(
                                "STT Watchdog: emit stt_swap_to_whisper failed",
                                exc_info=True,
                            )
                        # Skip the native restart — pipeline will own the swap.
                        return
            self._full_restart_attempts += 1
            self._last_full_restart_time = now
        else:
            logger.info("STT Watchdog: restarting STT (%s, restart #%d, chain #%d/%d)",
                        reason, self._total_restarts,
                        self._consecutive_chain_restarts, _CHAIN_RESTARTS_BEFORE_FULL)

        if not needs_full_restart:
            restart_fn = getattr(stt, "_restart_recognition_chain", None)
            if callable(restart_fn):
                try:
                    # Ω.10 step-6: see soft-restart comment above.
                    from core.async_event_bus import get_heavy_executor
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(get_heavy_executor(), restart_fn)
                    self._last_partial_time = time.monotonic()
                    logger.info("STT Watchdog: recognition chain restarted successfully")
                    self._bus.emit_fast("stt_watchdog_restart", reason=reason, restart_count=self._total_restarts)
                    return
                except Exception:
                    logger.warning("STT Watchdog: chain restart failed, trying full stop/start", exc_info=True)

        rebind_fn = getattr(stt, "_rebind_audio_device", None)
        if callable(rebind_fn):
            try:
                rebind_fn()
                logger.info("STT Watchdog: re-asserted CoreAudio device before full restart")
            except Exception:
                logger.debug("STT Watchdog: device rebind failed", exc_info=True)

        # Raise the auto-start suppression gate BEFORE stop_listening so the
        # STT's own _run_async loop can't race us and call start_listening
        # during our await — which would bind a fresh recognition task to
        # the stale recognizer and orphan the subsequent recreate.
        begin_fn = getattr(stt, "begin_full_restart", None)
        end_fn = getattr(stt, "end_full_restart", None)
        try:
            if callable(begin_fn):
                begin_fn()
            else:
                # Defensive path for older builds; may race with _run_async.
                stt.stop_listening()
            await asyncio.sleep(0.8)
            recreate_fn = getattr(stt, "_recreate_recognizer", None)
            if callable(recreate_fn):
                try:
                    recreate_fn()
                except Exception:
                    logger.debug("STT Watchdog: recognizer recreate failed", exc_info=True)
            start_fn = getattr(stt, "start_listening", None)
            if callable(start_fn):
                loop = getattr(stt, "_loop", None)
                on_final = getattr(stt, "_on_final", None)
                on_partial = getattr(stt, "_on_partial", None)
                ok = start_fn(loop=loop, on_final=on_final, on_partial=on_partial)
                if ok:
                    self._last_partial_time = time.monotonic()
                    logger.info(
                        "STT Watchdog: full STT restart completed (engine + recognizer rebuilt)",
                    )
                else:
                    logger.warning(
                        "STT Watchdog: start_listening returned False after full restart",
                    )
        except Exception:
            logger.exception("STT Watchdog: full restart failed")
        finally:
            if callable(end_fn):
                try:
                    end_fn()
                except Exception:
                    logger.debug("STT Watchdog: end_full_restart failed", exc_info=True)

        self._bus.emit_fast("stt_watchdog_restart", reason=reason, restart_count=self._total_restarts)

    def get_diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "total_restarts": self._total_restarts,
            "total_silent_detections": self._total_silent_detections,
            "since_last_partial_s": round(now - self._last_partial_time, 1),
            "since_last_final_s": round(now - self._last_final_time, 1),
            "recent_restarts": len(self._restart_times),
            "full_restart_attempts": self._full_restart_attempts,
            "full_restart_failures": self._full_restart_failures,
            "failover_pending": self._failover_pending,
        }


__all__ = ["STTWatchdog"]
