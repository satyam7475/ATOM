"""
ATOM -- Mood Inference (Phase G4).

Fuses *cheap* signals -- face presence, conversation sentiment,
prosody (TTS rate / mic loudness), session length and time-of-day --
into a single :class:`MoodState`. Downstream consumers (the Jarvis
suggester, the adaptive personality, the focus advisor) read this
state to decide *whether* and *how* to nudge the user.

Why not run a real classifier? On-device ML for affect would force
a third inference path (alongside MLX brain + VLM + Apple Vision)
and the value is marginal. The signals we actually have are coarse
enough that a tiny rules-based blender produces a stable, debuggable
mood label without burning latency or memory.

Design contract:

* Pure function over a fused :class:`MoodSignals` snapshot.
* Mood updates are emitted only when the *category* changes (no
  same-state spam).
* All inputs are optional. Missing data degrades to ``"unknown"``
  rather than guessing.
* Hysteresis: must see at least ``min_consecutive`` consistent
  signals before flipping (default 2). Prevents one bad sample from
  swinging the suggester.

Owner: Satyam
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.cognitive.mood")


# ── enums (lightweight strings -- easy to log + JSON-serialise) ────


VALID_MOODS = (
    "unknown",
    "focused",        # face present, low session length, neutral sentiment
    "engaged",        # face present + positive sentiment
    "frustrated",     # negative sentiment + repeats
    "tired",          # late hour OR long session OR low presence quality
    "idle",           # no face, no recent voice
    "distracted",     # face present but quality low / multiple people
)


@dataclass(slots=True)
class MoodSignals:
    """All inputs to the inference. Every field optional."""

    presence_present: bool | None = None
    presence_face_count: int | None = None
    presence_quality: str | None = None     # "good"|"low_light"|"no_face"|...
    sentiment: str | None = None             # "positive"|"neutral"|"negative"
    repeat_count: int = 0                    # consecutive repeated/clarification turns
    session_minutes: float | None = None     # how long the user has been active
    hour_of_day: int | None = None           # 0..23 (24h)
    voice_rms_db: float | None = None        # mic loudness (dBFS)
    last_user_chars: int | None = None       # length of last utterance


@dataclass(slots=True)
class MoodResult:
    mood: str
    confidence: float
    rationale: list[str] = field(default_factory=list)
    ts: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "confidence": round(self.confidence, 2),
            "rationale": list(self.rationale),
            "ts": self.ts,
        }


# ── pure inference ────────────────────────────────────────────────


def infer_mood(signals: MoodSignals) -> MoodResult:
    """Score and return the most likely mood from current signals."""
    rationale: list[str] = []
    scores: dict[str, float] = {m: 0.0 for m in VALID_MOODS if m != "unknown"}

    # Presence drives the floor: no-face -> idle.
    if signals.presence_present is False or signals.presence_face_count == 0:
        scores["idle"] += 0.6
        rationale.append("no face in front of camera")
    elif signals.presence_present is True:
        if (signals.presence_quality or "").lower() == "good":
            scores["focused"] += 0.4
            scores["engaged"] += 0.1
            rationale.append("face detected with good lighting")
        if signals.presence_face_count and signals.presence_face_count > 1:
            scores["distracted"] += 0.6
            rationale.append("multiple people in frame")
        if (signals.presence_quality or "").lower() in ("low_light", "blurry"):
            scores["distracted"] += 0.4
            rationale.append("poor face quality")

    # Sentiment & repetition.
    sentiment = (signals.sentiment or "").lower()
    if sentiment == "negative":
        scores["frustrated"] += 0.4
        rationale.append("negative sentiment last turn")
    elif sentiment == "positive":
        scores["engaged"] += 0.4
        rationale.append("positive sentiment last turn")
    if signals.repeat_count >= 2:
        scores["frustrated"] += 0.3
        rationale.append(f"{signals.repeat_count} repeated turns")

    # Session length + time-of-day -> tiredness.
    if signals.session_minutes is not None and signals.session_minutes >= 90:
        scores["tired"] += 0.3
        rationale.append("long active session")
    if signals.hour_of_day is not None:
        if signals.hour_of_day >= 23 or signals.hour_of_day < 5:
            scores["tired"] += 0.4
            rationale.append("late-night hour")
        elif 9 <= signals.hour_of_day < 12:
            scores["focused"] += 0.2
            rationale.append("morning peak hours")

    # Prosody hint (low RMS = whispering / leaning back).
    if signals.voice_rms_db is not None and signals.voice_rms_db < -45:
        scores["tired"] += 0.1
        rationale.append("quiet voice")

    # Long utterances suggest engagement; short ones + repeats suggest
    # frustration (already scored).
    if signals.last_user_chars is not None and signals.last_user_chars > 80:
        scores["engaged"] += 0.1

    # Pick the winner. Require a minimum threshold, otherwise unknown.
    if not scores or max(scores.values(), default=0.0) <= 0.0:
        return MoodResult(mood="unknown", confidence=0.0, rationale=rationale,
                          ts=time.time())

    mood, score = max(scores.items(), key=lambda kv: kv[1])
    confidence = max(0.0, min(1.0, score))
    if confidence < 0.3:
        return MoodResult(mood="unknown", confidence=confidence,
                          rationale=rationale, ts=time.time())
    return MoodResult(mood=mood, confidence=confidence,
                      rationale=rationale, ts=time.time())


# ── stateful engine ───────────────────────────────────────────────


class MoodInferenceEngine:
    """Subscribes to upstream signal events and emits ``mood.state``.

    Inputs we listen for (any subset; missing signals degrade
    gracefully):

    * ``presence.snapshot``     -> presence + face_count + quality
    * ``user_emotion_detected`` -> sentiment string
    * ``command_loop_trace``    -> last_user_chars / repeats
    * ``voice_metrics``         -> voice_rms_db

    The engine **does not** fire on every signal; instead it stores
    the latest snapshot and re-runs ``infer_mood`` whenever any input
    changes. A new ``mood.state`` event fires only when the mood
    *category* changes after passing the hysteresis gate.
    """

    __slots__ = (
        "_bus", "_signals", "_attached",
        "_min_consecutive", "_streak_mood", "_streak_count",
        "_last_emit_at", "_last_mood",
        "_total_emits", "_total_updates",
    )

    def __init__(
        self,
        bus: "AsyncEventBus",
        *,
        min_consecutive: int = 2,
    ) -> None:
        self._bus = bus
        self._signals = MoodSignals()
        self._attached = False
        self._min_consecutive = max(1, int(min_consecutive))
        self._streak_mood: str = "unknown"
        self._streak_count: int = 0
        self._last_emit_at = 0.0
        self._last_mood: str = "unknown"
        self._total_emits = 0
        self._total_updates = 0

    # ── public API ───────────────────────────────────────────────

    def attach(self) -> None:
        if self._attached:
            return
        self._bus.on("presence.snapshot", self._on_presence)
        self._bus.on("user_emotion_detected", self._on_emotion)
        self._bus.on("command_loop_trace", self._on_command_trace)
        self._bus.on("voice_metrics", self._on_voice_metrics)
        self._attached = True
        logger.info(
            "MoodInferenceEngine attached (min_consecutive=%d)",
            self._min_consecutive,
        )

    def detach(self) -> None:
        if not self._attached:
            return
        self._bus.off("presence.snapshot", self._on_presence)
        self._bus.off("user_emotion_detected", self._on_emotion)
        self._bus.off("command_loop_trace", self._on_command_trace)
        self._bus.off("voice_metrics", self._on_voice_metrics)
        self._attached = False

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "attached": self._attached,
            "current_mood": self._last_mood,
            "streak_mood": self._streak_mood,
            "streak_count": self._streak_count,
            "updates": self._total_updates,
            "emits": self._total_emits,
            "last_emit_age_s": (
                round(time.monotonic() - self._last_emit_at, 1)
                if self._last_emit_at else None
            ),
        }

    @property
    def current_mood(self) -> str:
        return self._last_mood

    def update_signals(self, **fields: Any) -> None:
        """Manually patch one or more signals and re-infer.

        Useful for tests + the future ``session_length`` background
        ticker (we update ``session_minutes`` here, not on the bus).
        """
        for name, value in fields.items():
            if hasattr(self._signals, name):
                setattr(self._signals, name, value)
        self._reinfer()

    # ── handlers ────────────────────────────────────────────────

    async def _on_presence(self, **payload: Any) -> None:
        self._signals.presence_present = bool(payload.get("present", False))
        self._signals.presence_face_count = int(payload.get("face_count", 0) or 0)
        self._signals.presence_quality = str(payload.get("quality", "") or "")
        self._reinfer()

    async def _on_emotion(self, **payload: Any) -> None:
        sentiment = str(payload.get("emotion") or payload.get("sentiment") or "")
        if sentiment:
            self._signals.sentiment = sentiment.lower()
            self._reinfer()

    async def _on_command_trace(
        self, *,
        stage: str = "", text: str = "", **_kw: Any,
    ) -> None:
        if stage == "start" and text:
            self._signals.last_user_chars = len(text)
            self._reinfer()

    async def _on_voice_metrics(
        self, *, rms_dbfs: float | None = None, **_kw: Any,
    ) -> None:
        if rms_dbfs is not None:
            self._signals.voice_rms_db = float(rms_dbfs)
            self._reinfer()

    # ── inference + hysteresis ─────────────────────────────────

    def _reinfer(self) -> None:
        self._total_updates += 1
        result = infer_mood(self._signals)

        if result.mood == self._streak_mood:
            self._streak_count += 1
        else:
            self._streak_mood = result.mood
            self._streak_count = 1

        if (
            self._streak_mood != self._last_mood
            and self._streak_count >= self._min_consecutive
            and result.mood != "unknown"
        ):
            self._emit(result)

    def _emit(self, result: MoodResult) -> None:
        self._last_mood = result.mood
        self._last_emit_at = time.monotonic()
        self._total_emits += 1
        try:
            self._bus.emit_long("mood.state", **result.as_dict())
        except Exception:
            logger.exception("mood.state emit failed")
        logger.info(
            "mood -> %s (conf=%.2f, %s)",
            result.mood, result.confidence, "; ".join(result.rationale)[:120],
        )


__all__ = [
    "VALID_MOODS",
    "MoodSignals",
    "MoodResult",
    "MoodInferenceEngine",
    "infer_mood",
]
