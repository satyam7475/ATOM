"""
ATOM -- Morning briefing service (Sprint D1).

First-wake-of-the-day greeting that blends battery, weather, today's
calendar, and top news headlines into one short spoken line. Designed
to fire **exactly once per calendar day**, on either:

    * startup, if we boot after the wake window started and have not
      yet briefed today, OR
    * the first user utterance after midnight inside the wake window.

Whichever happens first wins; subsequent triggers inside the same
calendar day are no-ops.

Design notes:

    * The service pulls live signals from ``RealWorldIntelligence`` for
      weather/news (cached there, so this is cheap) and ``psutil`` for
      battery — we intentionally avoid adding a dedicated battery
      dependency to keep the hot path light.
    * Calendar is optional. macOS ``osascript`` is invoked with a short
      timeout; if it fails for any reason (permission denied, timeout,
      non-mac host) we skip the calendar line and still deliver the rest.
    * ``response_ready`` is the canonical bus event for proactive speech
      in ATOM — emitting there guarantees the TTS, state machine, and
      indicator all stay in sync.
    * State is persisted to ``data/morning_briefing.json`` with just the
      last-briefed date + hour so a restart inside the same day does
      not cause a second greeting.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.proactive.morning_briefing")


_DEFAULT_STATE_PATH = "data/morning_briefing.json"
_DEFAULT_WAKE_START = 6
_DEFAULT_WAKE_END = 11
_DEFAULT_NEWS_COUNT = 3
_DEFAULT_CALENDAR_TIMEOUT = 3.0


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _hour_now() -> int:
    return datetime.now().hour


def _time_of_day_greeting() -> str:
    hour = _hour_now()
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


class MorningBriefingService:
    """Single-shot daily briefing orchestrator."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        bus: Any = None,
        real_world_intel: Any = None,
        state_path: str | Path | None = None,
    ) -> None:
        cfg = (config or {}).get("morning_briefing", {}) or {}
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._wake_start: int = int(cfg.get("wake_hour_start", _DEFAULT_WAKE_START))
        self._wake_end: int = int(cfg.get("wake_hour_end", _DEFAULT_WAKE_END))
        self._include_battery: bool = bool(cfg.get("include_battery", True))
        self._include_weather: bool = bool(cfg.get("include_weather", True))
        self._include_calendar: bool = bool(cfg.get("include_calendar", True))
        self._include_news: bool = bool(cfg.get("include_news", True))
        self._news_count: int = max(1, int(cfg.get("news_count", _DEFAULT_NEWS_COUNT)))
        self._calendar_timeout_s: float = float(
            cfg.get("calendar_timeout_s", _DEFAULT_CALENDAR_TIMEOUT)
        )

        self._state_path = Path(
            state_path or cfg.get("state_path", _DEFAULT_STATE_PATH)
        )

        self._bus = bus
        self._real_world_intel = real_world_intel

        self._last_briefed_date: str = ""
        self._last_briefed_t: float = 0.0
        self._in_flight = asyncio.Lock()

        self._restore_state()

    # ── Public API ────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_briefed_date(self) -> str:
        return self._last_briefed_date

    def is_window_open(self, hour: int | None = None) -> bool:
        h = _hour_now() if hour is None else int(hour)
        return self._wake_start <= h <= self._wake_end

    def should_fire(self, *, now_date: str | None = None, hour: int | None = None) -> bool:
        if not self._enabled:
            return False
        today = now_date or _today_str()
        if self._last_briefed_date == today:
            return False
        return self.is_window_open(hour)

    async def maybe_trigger(self, reason: str = "speech_final") -> str | None:
        """Compose + emit briefing if conditions are right. Returns text or None."""
        if not self.should_fire():
            return None
        if self._in_flight.locked():
            return None
        async with self._in_flight:
            if not self.should_fire():
                return None
            try:
                text = await self._compose_briefing()
            except Exception:
                logger.info("Morning briefing compose failed", exc_info=True)
                return None
            if not text:
                return None
            self._mark_fired()
            self._emit(text, reason=reason)
            return text

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "last_briefed_date": self._last_briefed_date,
            "last_briefed_age_s": (
                time.time() - self._last_briefed_t if self._last_briefed_t else None
            ),
            "wake_window": [self._wake_start, self._wake_end],
            "state_path": str(self._state_path),
        }

    # ── Composition ───────────────────────────────────────────────

    async def _compose_briefing(self) -> str:
        loop = asyncio.get_running_loop()
        parts: list[str] = []

        greeting = _time_of_day_greeting()
        honorific = self._get_honorific()
        now = datetime.now()
        opener = f"{greeting}, {honorific}. It's {now.strftime('%A, %B %-d')}."
        parts.append(opener)

        if self._include_battery:
            battery_line = await loop.run_in_executor(None, self._fetch_battery_line)
            if battery_line:
                parts.append(battery_line)

        if self._include_weather and self._real_world_intel is not None:
            weather_line = self._fetch_weather_line()
            if weather_line:
                parts.append(weather_line)

        if self._include_calendar:
            cal_line = await self._fetch_calendar_line()
            if cal_line:
                parts.append(cal_line)

        if self._include_news and self._real_world_intel is not None:
            news_line = self._fetch_news_line()
            if news_line:
                parts.append(news_line)

        return " ".join(p.strip() for p in parts if p and p.strip())

    def _get_honorific(self) -> str:
        try:
            from core.adaptive_personality import get_identity_snapshot
            snap = get_identity_snapshot() or {}
            title = str(snap.get("owner_title") or "").strip()
            if title:
                return title
        except Exception:
            pass
        return "Boss"

    def _fetch_battery_line(self) -> str:
        try:
            import psutil
            bat = psutil.sensors_battery()
            if bat is None:
                return ""
            pct = int(round(float(bat.percent or 0.0)))
            plugged = bool(bat.power_plugged)
            if plugged and pct >= 95:
                return "Battery is fully charged and plugged in."
            if plugged:
                return f"Battery is at {pct} percent and charging."
            if pct <= 20:
                return f"Battery is at {pct} percent — you'll want a cable soon."
            return f"Battery is at {pct} percent."
        except Exception:
            return ""

    def _fetch_weather_line(self) -> str:
        try:
            summary = self._real_world_intel.get_weather_summary()  # type: ignore[union-attr]
            summary = str(summary or "").strip()
            if not summary:
                return ""
            lowered = summary.lower()
            if "not available" in lowered or "unavailable" in lowered:
                return ""
            if not summary.endswith((".", "!", "?")):
                summary += "."
            return summary
        except Exception:
            logger.debug("weather fetch failed", exc_info=True)
            return ""

    def _fetch_news_line(self) -> str:
        try:
            summary = self._real_world_intel.get_news_summary(count=self._news_count)  # type: ignore[union-attr]
            summary = str(summary or "").strip()
            if not summary:
                return ""
            lowered = summary.lower()
            if "no news" in lowered or "unavailable" in lowered:
                return ""
            # Trim numbered-list preamble into a natural spoken line.
            header, _, body = summary.partition(":")
            if body:
                lines = [
                    ln.lstrip("0123456789. ").strip()
                    for ln in body.strip().splitlines()
                    if ln.strip()
                ]
                lines = [ln for ln in lines if ln]
                if not lines:
                    return ""
                if len(lines) == 1:
                    return f"Top headline: {lines[0]}."
                joined = "; ".join(lines[: self._news_count])
                return f"Top headlines — {joined}."
            return summary
        except Exception:
            logger.debug("news fetch failed", exc_info=True)
            return ""

    async def _fetch_calendar_line(self) -> str:
        events = await self._fetch_calendar_events_today()
        if not events:
            return ""
        if len(events) == 1:
            return f"Your calendar has one event today: {events[0]}."
        head = ", ".join(events[:3])
        if len(events) <= 3:
            return f"Your calendar has {len(events)} events today: {head}."
        return f"Your calendar has {len(events)} events today, starting with {events[0]}."

    async def _fetch_calendar_events_today(self) -> list[str]:
        """Query macOS Calendar.app via AppleScript. Best-effort."""
        try:
            from core.macos.calendar_today import fetch_today_events
            return await fetch_today_events(self._calendar_timeout_s)
        except Exception:
            logger.debug("calendar_today fetch failed", exc_info=True)
            return []

    # ── Bus emission ──────────────────────────────────────────────

    def _emit(self, text: str, *, reason: str) -> None:
        bus = self._bus
        if bus is None:
            return
        try:
            payload = {"text": text, "source": "morning_briefing", "reason": reason}
            emit_long = getattr(bus, "emit_long", None)
            if callable(emit_long):
                emit_long("response_ready", **payload)
                return
            emit = getattr(bus, "emit", None)
            if callable(emit):
                emit("response_ready", **payload)
        except Exception:
            logger.info("morning briefing emit failed", exc_info=True)

    # ── Persistence ───────────────────────────────────────────────

    def _mark_fired(self) -> None:
        self._last_briefed_date = _today_str()
        self._last_briefed_t = time.time()
        self._persist_state()

    def _persist_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            payload = {
                "last_briefed_date": self._last_briefed_date,
                "last_briefed_ts": self._last_briefed_t,
                "version": 1,
            }
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception:
            logger.debug("morning briefing persist failed", exc_info=True)

    def _restore_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            self._last_briefed_date = str(raw.get("last_briefed_date") or "")
            self._last_briefed_t = float(raw.get("last_briefed_ts") or 0.0)
        except Exception:
            logger.debug("morning briefing restore failed", exc_info=True)


__all__ = ["MorningBriefingService"]
