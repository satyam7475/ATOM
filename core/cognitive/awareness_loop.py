"""Continuous awareness loop -- the bit that makes ATOM feel like Friday.

The existing :class:`core.cognitive.jarvis_suggester.JarvisSuggester`
already does cadence + mood-gated emission. What it does *not* do is
react to context *transitions*: Boss walked away and came back, Boss
just switched apps from VS Code to Slack, Boss has been silent for an
unusually long stretch despite being present. That kind of fused
awareness is what separates "rules-driven nudges" from a colleague
who actually noticed.

This module owns the fusion. It subscribes to the high-signal events
already on the bus -- ``presence.snapshot``, ``mood.state``,
``scene.context``, ``speech_final``, ``state_changed``,
``tts_complete`` -- and maintains a small in-memory snapshot of
"what's happening with Boss right now". On a meaningful transition it
generates a high-score :class:`SuggestionCandidate` and either:

1. Pushes it through the suggester (preferred, because the suggester
   already enforces cooldowns / quiet hours / daily cap), or
2. Emits ``response_ready`` directly when the user explicitly
   configured ``allow_direct_emit=True`` (e.g. for the greeting on
   return from a long absence).

Design rules (from Boss's review):

* "Don't make everything a reminder." -- reuse the suggester's gates.
* "Friday-feeling, not chatbox." -- the loop only speaks when the
  fusion produces *new* information; same-state ticks are silent.
* "Local privacy" -- nothing in this module touches the network.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.cognitive.jarvis_suggester import JarvisSuggester, SuggestionCandidate

logger = logging.getLogger("atom.cognitive.awareness")


# ── internal state ─────────────────────────────────────────────────


@dataclass(slots=True)
class _Snapshot:
    """Live picture of Boss + ATOM right now."""

    mood: str = "unknown"
    presence_present: bool | None = None
    last_presence_change_at: float = 0.0
    last_seen_present_at: float = 0.0
    last_seen_absent_at: float = 0.0
    scene_label: str = ""
    scene_changed_at: float = 0.0
    last_user_speech_at: float = 0.0
    last_atom_speech_at: float = 0.0
    last_state: str = ""
    state_changed_at: float = 0.0


# ── awareness loop ─────────────────────────────────────────────────


@dataclass(slots=True)
class AwarenessConfig:
    """Tuning knobs for the fusion engine."""

    welcome_back_after_absent_s: float = 240.0
    silent_present_warn_s: float = 1800.0   # 30 min present, no speech
    scene_dwell_warn_s: float = 2400.0      # 40 min in same app
    welcome_back_score: float = 0.95
    silent_present_score: float = 0.78
    scene_dwell_score: float = 0.72
    min_emit_gap_s: float = 90.0            # local floor between bursts
    enable_direct_welcome_emit: bool = True


class AwarenessLoop:
    """Fuse presence + mood + scene + voice into proactive nudges.

    Hooks into :class:`JarvisSuggester` for cadence-gated emission and
    optionally writes a friendly welcome-back line directly through the
    bus when Boss returns after a long absence (the welcome-back path
    bypasses the suggester's cooldown so it always fires).
    """

    __slots__ = (
        "_bus", "_state_manager", "_suggester", "_config",
        "_snapshot", "_last_emit_at", "_attached",
        "_emit_count", "_welcome_count", "_silent_count", "_scene_count",
    )

    def __init__(
        self,
        bus: Any,
        *,
        suggester: JarvisSuggester | None = None,
        state_manager: Any = None,
        config: AwarenessConfig | None = None,
    ) -> None:
        self._bus = bus
        self._suggester = suggester
        self._state_manager = state_manager
        self._config = config or AwarenessConfig()
        self._snapshot = _Snapshot()
        self._last_emit_at = 0.0
        self._attached = False
        self._emit_count = 0
        self._welcome_count = 0
        self._silent_count = 0
        self._scene_count = 0

    # ── lifecycle ─────────────────────────────────────────────

    def attach(self) -> None:
        if self._attached:
            return
        on = getattr(self._bus, "on", None)
        if not callable(on):
            logger.warning("AwarenessLoop: bus has no .on() -- skipping")
            return
        on("presence.snapshot", self._on_presence)
        on("mood.state", self._on_mood)
        on("scene.context", self._on_scene)
        on("speech_final", self._on_user_speech)
        on("response_ready", self._on_atom_response)
        on("state_changed", self._on_state_changed)
        self._attached = True
        logger.info(
            "AwarenessLoop attached (welcome=%.0fs, silent=%.0fs, dwell=%.0fs)",
            self._config.welcome_back_after_absent_s,
            self._config.silent_present_warn_s,
            self._config.scene_dwell_warn_s,
        )

    def detach(self) -> None:
        if not self._attached:
            return
        off = getattr(self._bus, "off", None)
        if callable(off):
            off("presence.snapshot", self._on_presence)
            off("mood.state", self._on_mood)
            off("scene.context", self._on_scene)
            off("speech_final", self._on_user_speech)
            off("response_ready", self._on_atom_response)
            off("state_changed", self._on_state_changed)
        self._attached = False

    # ── public introspection ─────────────────────────────────

    @property
    def snapshot(self) -> dict[str, Any]:
        return {
            "mood": self._snapshot.mood,
            "presence_present": self._snapshot.presence_present,
            "scene_label": self._snapshot.scene_label,
            "last_user_speech_age_s": self._age(self._snapshot.last_user_speech_at),
            "last_atom_speech_age_s": self._age(self._snapshot.last_atom_speech_at),
            "state": self._snapshot.last_state,
            "metrics": {
                "emits": self._emit_count,
                "welcomes": self._welcome_count,
                "silent_warnings": self._silent_count,
                "scene_dwell_warnings": self._scene_count,
            },
        }

    # ── event handlers ───────────────────────────────────────

    async def _on_presence(self, *, present: bool | None = None, **_kw: Any) -> None:
        prev = self._snapshot.presence_present
        now = time.time()
        self._snapshot.presence_present = present
        if present is True:
            self._snapshot.last_seen_present_at = now
            if prev is False:
                # Welcome-back transition: only fire if the absence was
                # long enough to feel like a real return.
                gap = now - (self._snapshot.last_seen_absent_at or now)
                if gap >= self._config.welcome_back_after_absent_s:
                    self._snapshot.last_presence_change_at = now
                    await self._maybe_welcome_back(gap_s=gap)
        elif present is False:
            self._snapshot.last_seen_absent_at = now
            if prev is True:
                self._snapshot.last_presence_change_at = now

    async def _on_mood(self, *, mood: str | None = None, **_kw: Any) -> None:
        if mood:
            self._snapshot.mood = str(mood)
        await self._maybe_long_silence()

    async def _on_scene(self, *, label: str | None = None, **_kw: Any) -> None:
        new_label = (label or "").strip()
        if not new_label:
            return
        if new_label != self._snapshot.scene_label:
            self._snapshot.scene_label = new_label
            self._snapshot.scene_changed_at = time.time()
        await self._maybe_scene_dwell()

    async def _on_user_speech(self, *, text: str | None = None, **_kw: Any) -> None:
        self._snapshot.last_user_speech_at = time.time()

    async def _on_atom_response(self, **_kw: Any) -> None:
        self._snapshot.last_atom_speech_at = time.time()

    async def _on_state_changed(self, *, old: object | None = None, new: object | None = None, **_kw: Any) -> None:
        new_state = str(getattr(new, "value", new) or "")
        if new_state != self._snapshot.last_state:
            self._snapshot.last_state = new_state
            self._snapshot.state_changed_at = time.time()

    # ── fusion -> candidate ──────────────────────────────────

    async def _maybe_welcome_back(self, *, gap_s: float) -> None:
        """High-priority: Boss just returned to the desk."""

        if not self._gate(force=True):
            return
        minutes = int(gap_s // 60)
        text = (
            f"Welcome back, Boss. You were away for about {minutes} "
            f"minute{'s' if minutes != 1 else ''}. Want a quick rundown of what changed?"
        )
        candidate = SuggestionCandidate(
            text=text,
            category="awareness.welcome_back",
            score=self._config.welcome_back_score,
            rationale=f"absence_gap_s={int(gap_s)}",
        )
        # Prefer the suggester so cadence remains polite, but allow a
        # direct emit if the user opted in.
        if self._suggester is not None:
            await self._suggester.consider_candidates([candidate], reason="welcome_back")
        elif self._config.enable_direct_welcome_emit:
            self._emit_direct(text)
        self._welcome_count += 1
        self._record_emit()

    async def _maybe_long_silence(self) -> None:
        """Boss is present and engaged-mood but hasn't spoken in a while."""
        snap = self._snapshot
        if snap.presence_present is not True:
            return
        if snap.mood not in {"engaged", "happy", "neutral"}:
            return
        last = snap.last_user_speech_at
        if last <= 0:
            return
        silent_for = time.time() - last
        if silent_for < self._config.silent_present_warn_s:
            return
        if not self._gate():
            return

        candidate = SuggestionCandidate(
            text=(
                "You've been quiet for a while, Boss. "
                "Want me to summarise what's on screen or take any notes?"
            ),
            category="awareness.silent_present",
            score=self._config.silent_present_score,
            rationale=f"silent_for_s={int(silent_for)}",
        )
        if self._suggester is not None:
            await self._suggester.consider_candidates([candidate], reason="silent_present")
        self._silent_count += 1
        self._record_emit()

    async def _maybe_scene_dwell(self) -> None:
        """Boss has been in the same app forever -- offer a break."""
        snap = self._snapshot
        if snap.presence_present is False:
            return
        if not snap.scene_label or not snap.scene_changed_at:
            return
        dwell = time.time() - snap.scene_changed_at
        if dwell < self._config.scene_dwell_warn_s:
            return
        if not self._gate():
            return

        scene = snap.scene_label
        candidate = SuggestionCandidate(
            text=(
                f"You've been heads-down in {scene} for a long stretch, Boss. "
                f"Want me to pause music for a five-minute breather?"
            ),
            category=f"awareness.scene_dwell:{scene[:32]}",
            score=self._config.scene_dwell_score,
            rationale=f"dwell_s={int(dwell)} scene={scene[:48]}",
        )
        if self._suggester is not None:
            await self._suggester.consider_candidates([candidate], reason="scene_dwell")
        self._scene_count += 1
        self._record_emit()

    # ── helpers ──────────────────────────────────────────────

    def _gate(self, *, force: bool = False) -> bool:
        """Local rate-limit so a flapping signal can't spam the bus."""
        if force:
            return True
        gap = time.time() - self._last_emit_at
        return gap >= self._config.min_emit_gap_s

    def _record_emit(self) -> None:
        self._last_emit_at = time.time()
        self._emit_count += 1

    def _emit_direct(self, text: str) -> None:
        emit = getattr(self._bus, "emit_fast", None)
        if not callable(emit):
            return
        try:
            emit(
                "response_ready",
                text=text,
                source="awareness_loop",
                proactive=True,
            )
        except Exception:
            logger.exception("AwarenessLoop direct emit failed")

    @staticmethod
    def _age(ts: float) -> float | None:
        if ts <= 0:
            return None
        return round(time.time() - ts, 1)


# ── compatibility shim for JarvisSuggester ─────────────────────────
# The current JarvisSuggester only exposes ``attach``/``detach`` and
# its event-driven internal loop. Boss may run an older build where
# ``consider_candidates`` is private. This shim adapts both API
# generations so AwarenessLoop works against either.


def _ensure_consider_candidates(suggester: JarvisSuggester) -> None:
    """Backfill a public ``consider_candidates`` if missing."""
    if hasattr(suggester, "consider_candidates"):
        return
    inner = getattr(suggester, "_consider", None) or getattr(suggester, "_evaluate_candidates", None)
    if not callable(inner):
        return

    async def consider_candidates(candidates: list[SuggestionCandidate], *, reason: str = "") -> bool:  # noqa: ARG001
        try:
            return bool(await inner(candidates))
        except Exception:
            logger.exception("JarvisSuggester adapter call failed")
            return False

    suggester.consider_candidates = consider_candidates  # type: ignore[attr-defined]


__all__ = [
    "AwarenessConfig",
    "AwarenessLoop",
    "_ensure_consider_candidates",
]
