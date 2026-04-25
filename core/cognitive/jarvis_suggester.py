"""
ATOM -- Jarvis-style Suggester (Phase G5).

The reflective loop (G1) decides what to say *right after a turn*.
This module is the *between-turn* suggester -- the part that makes
ATOM feel like Jarvis: occasionally noticing something useful and
piping up unprompted ("you have been heads-down for two hours,
Boss; want me to dim the screen and put on focus music?").

Design constraints (the user was very clear: "don't make everything
like a reminder"):

* Hard cadence gates first, evidence second:
    - **At most 1 suggestion every ``cooldown_s``** (default 12 min).
    - **At most ``daily_cap`` suggestions per UTC day** (default 4).
* **Mood-aware suppression**: never nudge while ``mood == frustrated``
  or ``focused`` (the user does not want a friendly tap on the
  shoulder mid-flow). When ``mood == idle`` we also skip -- they're
  not at the laptop.
* **Quiet-hours suppression**: configurable bedtime range; default
  23:00 -> 06:00 local hour.
* **Idempotent**: once a suggestion fires for a category, the same
  category is locked out for ``category_cooldown_s`` (default 45 min).
* **Relevance threshold**: every candidate carries a 0..1 score; we
  only emit when ``score >= relevance_threshold`` (default 0.7).
* **Bus-friendly**: emits ``response_ready`` with ``proactive=True``
  and ``source="jarvis_suggester"`` so the speech controller can
  apply a softer prosody and the analytics can filter these out.

Owner: Satyam
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.cognitive.suggester")


# ── candidate ──────────────────────────────────────────────────────


@dataclass(slots=True)
class SuggestionCandidate:
    text: str
    category: str
    score: float = 0.0
    rationale: str = ""

    def is_actionable(self) -> bool:
        return bool(self.text.strip()) and 0.0 <= self.score <= 1.0


# ── default candidate generator ────────────────────────────────────


def default_candidates(
    *,
    mood: str,
    session_minutes: float | None,
    hour_of_day: int | None,
    presence_present: bool | None,
    last_user_chars: int | None = None,
) -> list[SuggestionCandidate]:
    """A small library of mood/context-aware nudges.

    These are deliberately stateless and pure -- the suggester engine
    decides which ones are *allowed* to fire."""
    out: list[SuggestionCandidate] = []

    if mood == "tired" and (hour_of_day is not None and hour_of_day >= 22):
        out.append(SuggestionCandidate(
            text="Boss, it's getting late -- want me to switch to wind-down mode?",
            category="wind_down",
            score=0.78,
            rationale="late hour + tired",
        ))

    if mood == "tired" and session_minutes and session_minutes >= 90:
        out.append(SuggestionCandidate(
            text="You've been at it a while -- shall I dim the screen and pause music for a five-minute break?",
            category="break_suggest",
            score=0.74,
            rationale="long session + tired",
        ))

    if mood == "engaged" and session_minutes and session_minutes < 30:
        out.append(SuggestionCandidate(
            text="On a roll, Boss. Want me to silence notifications so you don't break the flow?",
            category="focus_offer",
            score=0.72,
            rationale="engaged + short session",
        ))

    if mood == "distracted" and presence_present:
        out.append(SuggestionCandidate(
            text="Looks busy on your side -- want me to summarise what we were on?",
            category="recap_offer",
            score=0.70,
            rationale="distracted + present",
        ))

    return out


# ── engine ─────────────────────────────────────────────────────────


class JarvisSuggester:
    """Cadence-gated suggestion publisher.

    The engine listens to ``mood.state`` (and optionally
    ``presence.snapshot`` / ``scene.context``) and considers firing a
    suggestion. All gates must pass:

      1. Cooldown (global + per-category)
      2. Daily cap
      3. Quiet hours
      4. Mood is not in the suppression set
      5. Best candidate score >= ``relevance_threshold``
    """

    __slots__ = (
        "_bus", "_candidate_provider",
        "_cooldown_s", "_category_cooldown_s",
        "_daily_cap", "_quiet_hours",
        "_relevance_threshold", "_suppress_moods",
        "_attached", "_mood",
        "_session_started_at", "_presence_present",
        "_last_user_chars", "_last_emit_at",
        "_last_emit_per_category", "_emits_today",
        "_today_key", "_total_attempts",
        "_total_emits", "_total_blocked",
        "_response_emitter",
    )

    def __init__(
        self,
        bus: "AsyncEventBus",
        *,
        candidate_provider: Callable[..., list[SuggestionCandidate]] | None = None,
        cooldown_s: float = 720.0,            # 12 min
        category_cooldown_s: float = 2700.0,  # 45 min
        daily_cap: int = 4,
        quiet_hours: tuple[int, int] = (23, 6),
        relevance_threshold: float = 0.7,
        suppress_moods: tuple[str, ...] = ("frustrated", "focused", "idle"),
        response_emitter: Callable[[str], None] | None = None,
    ) -> None:
        self._bus = bus
        self._candidate_provider = candidate_provider or default_candidates
        self._cooldown_s = float(cooldown_s)
        self._category_cooldown_s = float(category_cooldown_s)
        self._daily_cap = int(daily_cap)
        self._quiet_hours = quiet_hours
        self._relevance_threshold = float(relevance_threshold)
        self._suppress_moods = tuple(suppress_moods)
        self._attached = False
        self._mood = "unknown"
        self._session_started_at = time.monotonic()
        self._presence_present: bool | None = None
        self._last_user_chars: int | None = None
        self._last_emit_at = 0.0
        self._last_emit_per_category: dict[str, float] = {}
        self._emits_today = 0
        self._today_key = self._utc_day_key()
        self._total_attempts = 0
        self._total_emits = 0
        self._total_blocked = 0
        self._response_emitter = response_emitter

    # ── lifecycle ───────────────────────────────────────────────

    def attach(self) -> None:
        if self._attached:
            return
        self._bus.on("mood.state", self._on_mood)
        self._bus.on("presence.snapshot", self._on_presence)
        self._bus.on("command_loop_trace", self._on_trace)
        self._attached = True
        logger.info(
            "JarvisSuggester attached (cooldown=%.0fs, cap=%d/day, "
            "threshold=%.2f, quiet=%s)",
            self._cooldown_s, self._daily_cap,
            self._relevance_threshold, self._quiet_hours,
        )

    def detach(self) -> None:
        if not self._attached:
            return
        self._bus.off("mood.state", self._on_mood)
        self._bus.off("presence.snapshot", self._on_presence)
        self._bus.off("command_loop_trace", self._on_trace)
        self._attached = False

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "attached": self._attached,
            "current_mood": self._mood,
            "attempts": self._total_attempts,
            "emits": self._total_emits,
            "blocked": self._total_blocked,
            "emits_today": self._emits_today,
            "session_minutes": round(self._session_minutes(), 1),
            "last_emit_age_s": (
                round(time.monotonic() - self._last_emit_at, 1)
                if self._last_emit_at else None
            ),
        }

    # ── handlers ────────────────────────────────────────────────

    async def _on_mood(self, **payload: Any) -> None:
        new_mood = str(payload.get("mood", "") or "unknown").lower()
        self._mood = new_mood
        await self._maybe_emit()

    async def _on_presence(self, **payload: Any) -> None:
        self._presence_present = bool(payload.get("present", False))

    async def _on_trace(
        self, *, stage: str = "", text: str = "", **_kw: Any,
    ) -> None:
        if stage == "start" and text:
            self._last_user_chars = len(text)

    # ── core gate ───────────────────────────────────────────────

    async def _maybe_emit(self) -> None:
        self._roll_day_if_needed()
        if not self._gate_passes():
            self._total_blocked += 1
            return

        candidates = self._gather_candidates()
        if not candidates:
            self._total_blocked += 1
            return

        candidate = max(candidates, key=lambda c: c.score)
        self._total_attempts += 1

        if candidate.score < self._relevance_threshold:
            self._total_blocked += 1
            return

        if not self._category_allowed(candidate.category):
            self._total_blocked += 1
            return

        self._emit(candidate)

    async def consider_candidates(
        self,
        candidates: list[SuggestionCandidate],
        *,
        reason: str = "",
    ) -> bool:
        """Externally pushed candidates (Sprint F1: AwarenessLoop).

        Honours every gate the regular ``_maybe_emit`` honours
        (cooldown, daily cap, quiet hours, mood suppression,
        per-category lockout, relevance threshold). Returns ``True``
        iff a candidate actually fired.
        """
        if not candidates:
            return False
        self._roll_day_if_needed()
        if not self._gate_passes():
            self._total_blocked += 1
            return False
        candidate = max(candidates, key=lambda c: c.score)
        self._total_attempts += 1
        if candidate.score < self._relevance_threshold:
            self._total_blocked += 1
            return False
        if not self._category_allowed(candidate.category):
            self._total_blocked += 1
            return False
        if reason:
            candidate.rationale = (candidate.rationale + " " + reason).strip()
        self._emit(candidate)
        return True

    def _gather_candidates(self) -> list[SuggestionCandidate]:
        try:
            return list(self._candidate_provider(
                mood=self._mood,
                session_minutes=self._session_minutes(),
                hour_of_day=_dt.datetime.now().hour,
                presence_present=self._presence_present,
                last_user_chars=self._last_user_chars,
            ))
        except Exception:
            logger.exception("candidate provider raised")
            return []

    def _gate_passes(self) -> bool:
        if self._mood in self._suppress_moods:
            return False
        if self._is_quiet_hour():
            return False
        if self._emits_today >= self._daily_cap:
            return False
        if self._last_emit_at and (
            time.monotonic() - self._last_emit_at < self._cooldown_s
        ):
            return False
        return True

    def _category_allowed(self, category: str) -> bool:
        last = self._last_emit_per_category.get(category)
        if last is None:
            return True
        return (time.monotonic() - last) >= self._category_cooldown_s

    def _is_quiet_hour(self) -> bool:
        start, end = self._quiet_hours
        hour = _dt.datetime.now().hour
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        # wraps midnight
        return hour >= start or hour < end

    def _session_minutes(self) -> float:
        return max(0.0, (time.monotonic() - self._session_started_at) / 60.0)

    def _roll_day_if_needed(self) -> None:
        today = self._utc_day_key()
        if today != self._today_key:
            self._today_key = today
            self._emits_today = 0
            self._last_emit_per_category.clear()

    @staticmethod
    def _utc_day_key() -> str:
        return _dt.datetime.utcnow().strftime("%Y%m%d")

    # ── emission ───────────────────────────────────────────────

    def _emit(self, candidate: SuggestionCandidate) -> None:
        self._last_emit_at = time.monotonic()
        self._last_emit_per_category[candidate.category] = self._last_emit_at
        self._emits_today += 1
        self._total_emits += 1
        if self._response_emitter is not None:
            try:
                self._response_emitter(candidate.text)
                return
            except Exception:
                logger.exception("response_emitter raised; falling back to bus")
        try:
            self._bus.emit_long(
                "response_ready",
                text=candidate.text,
                source="jarvis_suggester",
                proactive=True,
                category=candidate.category,
                score=round(candidate.score, 2),
                mood=self._mood,
            )
        except Exception:
            logger.exception("jarvis suggester emit failed")
        logger.info(
            "jarvis suggestion '%s' (mood=%s, score=%.2f, %s)",
            candidate.category, self._mood,
            candidate.score, candidate.rationale,
        )

    # ── test/manual hook ───────────────────────────────────────

    def reset_session(self) -> None:
        self._session_started_at = time.monotonic()


__all__ = [
    "JarvisSuggester",
    "SuggestionCandidate",
    "default_candidates",
]
