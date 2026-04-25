"""Voice round-trip latency probe — Sprint Ω.2.

Measures the single most important number for "Friday-class" voice
UX: how long between the user finishing a sentence and ATOM's first
spoken token? The probe sits on the event bus, anchors every
``speech_final`` with a monotonic timestamp, and on the next
``partial_response`` (``is_first=True``) emits / records a
``voice_round_trip_ms`` sample.

Design notes
------------
- Pure subscriber. Adds no work to the hot path beyond a dict write
  and a subtraction. The probe never blocks any handler chain.
- Anchors are keyed by ``utterance_id`` when supplied (multiple
  concurrent transcripts during long-form recording), otherwise by
  the singleton key ``"_default"``. Anchors expire after 30 s so a
  failed reply doesn't wedge the probe forever.
- Surfaces the sample through three channels for maximum
  observability uptake without mandating any one of them:
    1. ``logger.info`` — visible in atomLogs.txt for boot-day audits.
    2. The shared ``ObservabilityLatencyBoard`` under
       ``module="tts"`` so the existing dashboard / health snapshot
       picks the data up automatically.
    3. ``bus.emit("voice_round_trip", ms=..., utterance_id=...)``
       so any downstream coach (cognitive loop, LiveKit room,
       browser dashboard) can react in real time.
- A rolling 50-sample deque powers a ``stats()`` accessor for
  on-demand inspection from REPL or admin endpoints.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Any

logger = logging.getLogger("atom.observability.voice")

_ANCHOR_TTL_S = 30.0
_SAMPLE_WINDOW = 50


class VoiceLatencyProbe:
    """Subscriber that turns ``speech_final``/``partial_response`` pairs
    into ``voice_round_trip_ms`` samples."""

    __slots__ = (
        "_bus",
        "_anchors",
        "_samples",
        "_lock",
        "_count",
        "_min_ms",
        "_max_ms",
        "_sum_ms",
        "_attached",
    )

    def __init__(self, bus: Any) -> None:
        self._bus = bus
        self._anchors: dict[str, float] = {}
        self._samples: deque = deque(maxlen=_SAMPLE_WINDOW)
        self._lock = Lock()
        self._count = 0
        self._min_ms = float("inf")
        self._max_ms = 0.0
        self._sum_ms = 0.0
        self._attached = False

    def attach(self) -> None:
        """Wire the probe into the event bus. Idempotent."""
        if self._attached or self._bus is None:
            return
        try:
            self._bus.on("speech_final", self._on_speech_final)
            self._bus.on("partial_response", self._on_partial_response)
            self._attached = True
            logger.info("Voice latency probe attached")
        except Exception:
            logger.debug("VoiceLatencyProbe attach failed", exc_info=True)

    # ── handlers ─────────────────────────────────────────────────

    async def _on_speech_final(self, **payload: Any) -> None:
        utterance_id = str(payload.get("utterance_id") or "_default")
        now = time.monotonic()
        with self._lock:
            self._anchors[utterance_id] = now
            self._sweep_expired_locked(now)

    async def _on_partial_response(self, **payload: Any) -> None:
        if not payload.get("is_first", False):
            return
        utterance_id = str(payload.get("utterance_id") or "_default")
        now = time.monotonic()
        with self._lock:
            anchor = self._anchors.pop(utterance_id, None)
        if anchor is None:
            return
        delta_ms = (now - anchor) * 1000.0
        self._record(delta_ms, utterance_id, payload)

    # ── sample sink ──────────────────────────────────────────────

    def _record(self, delta_ms: float, utterance_id: str, payload: dict) -> None:
        with self._lock:
            self._samples.append(delta_ms)
            self._count += 1
            self._sum_ms += delta_ms
            if delta_ms < self._min_ms:
                self._min_ms = delta_ms
            if delta_ms > self._max_ms:
                self._max_ms = delta_ms

        # Logger surface — single line so atomLogs.txt audits stay
        # easy to scan.
        source = payload.get("source", "")
        suffix = f" source={source}" if source else ""
        logger.info(
            "voice_round_trip_ms=%.0f utterance=%s%s",
            delta_ms, utterance_id, suffix,
        )

        try:
            from core.observability.per_module_latency import get_latency_board
            board = get_latency_board()
            if board is not None:
                # Fold into the "tts" lane because the round-trip is
                # ultimately a TTS-first-byte measurement; keeps the
                # dashboard layout stable.
                board.record_module_call("tts", delta_ms, error=False)
                board.log_event(
                    "voice_round_trip", f"{delta_ms:.0f}ms ({utterance_id})",
                )
        except Exception:
            logger.debug("voice probe board sink failed", exc_info=True)

        try:
            self._bus.emit(
                "voice_round_trip",
                ms=delta_ms,
                utterance_id=utterance_id,
                source=source,
            )
        except Exception:
            logger.debug("voice probe bus emit failed", exc_info=True)

    # ── helpers ──────────────────────────────────────────────────

    def _sweep_expired_locked(self, now: float) -> None:
        # Caller holds self._lock.
        if not self._anchors:
            return
        cutoff = now - _ANCHOR_TTL_S
        stale = [k for k, ts in self._anchors.items() if ts < cutoff]
        for k in stale:
            self._anchors.pop(k, None)

    def stats(self) -> dict[str, Any]:
        """Snapshot of the rolling round-trip stats."""
        with self._lock:
            samples = list(self._samples)
            count = self._count
            min_ms = self._min_ms if self._min_ms != float("inf") else 0.0
            max_ms = self._max_ms
            sum_ms = self._sum_ms
        if not samples:
            return {
                "count": 0,
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "lifetime_count": count,
                "lifetime_avg_ms": (sum_ms / count) if count else 0.0,
            }
        ordered = sorted(samples)
        n = len(ordered)
        p50 = ordered[n // 2]
        p95 = ordered[min(n - 1, int(n * 0.95))]
        return {
            "count": n,
            "avg_ms": sum(samples) / n,
            "p50_ms": p50,
            "p95_ms": p95,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "lifetime_count": count,
            "lifetime_avg_ms": (sum_ms / count) if count else 0.0,
        }


_PROBE: VoiceLatencyProbe | None = None
_PROBE_LOCK = Lock()


def get_voice_latency_probe(bus: Any | None = None) -> VoiceLatencyProbe | None:
    """Return the process-wide probe. Constructs one on first call if
    a bus is supplied; later calls without a bus just return whatever
    has been wired (or None)."""
    global _PROBE
    with _PROBE_LOCK:
        if _PROBE is None and bus is not None:
            _PROBE = VoiceLatencyProbe(bus)
        return _PROBE
