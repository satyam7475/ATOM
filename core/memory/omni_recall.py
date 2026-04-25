"""ATOM Sprint N3 -- omni-memory recall.

Friday-class memory means Boss can ask things like:

    "What was I doing on screen around 5 pm yesterday?"
    "Did I ask you anything about gemini last Tuesday?"
    "What did we talk about this morning?"

A single ``OmniRecall`` query fans out across every persistent memory
source ATOM owns:

    * :class:`TimelineMemory`     -- spoken queries / actions / files
    * :class:`ConversationMemory` -- recent dialogue turns (RAM)
    * :class:`ScreenPerceptionLoop` -- desktop OCR observations

It uses :func:`core.memory.temporal_resolver.resolve` to turn the
free-form phrase ("last Tuesday", "5 pm yesterday", "since lunch")
into a (start, end) window, then merges hits sorted by recency.

Returns a :class:`RecallReport` that's both machine-friendly (for the
multi-step planner / awareness loop) *and* human-friendly (`speak()`
hands a one-sentence summary to TTS).

Owner: Boss (Satyam).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from core.memory.temporal_resolver import TemporalRange, resolve

logger = logging.getLogger("atom.memory.omni_recall")


# ── data ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RecallHit:
    source: str             # "timeline" | "conversation" | "screen"
    ts: float
    text: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def age_hint(self) -> str:
        delta = max(0.0, time.time() - self.ts)
        if delta < 60:
            return "just now"
        if delta < 3600:
            mins = int(delta // 60)
            return f"{mins} min ago"
        if delta < 86400:
            hrs = int(delta // 3600)
            return f"{hrs} h ago"
        days = int(delta // 86400)
        return f"{days} d ago"


@dataclass(slots=True)
class RecallReport:
    query: str
    window: TemporalRange | None
    hits: list[RecallHit] = field(default_factory=list)
    used_default_window: bool = False
    error: str = ""

    @property
    def empty(self) -> bool:
        return not self.hits

    def by_source(self) -> dict[str, list[RecallHit]]:
        out: dict[str, list[RecallHit]] = {}
        for h in self.hits:
            out.setdefault(h.source, []).append(h)
        return out

    def speak(self, *, max_examples: int = 2) -> str:
        if self.error:
            return f"I couldn't recall that, Boss -- {self.error}."
        if self.empty:
            scope = (self.window.label
                     if self.window is not None else "recent memory")
            return (
                f"Nothing in my memory matches that, Boss -- "
                f"checked {scope}."
            )

        scope = (self.window.label
                 if self.window is not None else "recent memory")
        bys = self.by_source()
        bits: list[str] = []
        if "timeline" in bys:
            bits.append(
                f"{len(bys['timeline'])} thing"
                f"{'s' if len(bys['timeline']) != 1 else ''} you said"
            )
        if "conversation" in bys:
            bits.append(
                f"{len(bys['conversation'])} dialogue turn"
                f"{'s' if len(bys['conversation']) != 1 else ''}"
            )
        if "screen" in bys:
            bits.append(
                f"{len(bys['screen'])} screen snapshot"
                f"{'s' if len(bys['screen']) != 1 else ''}"
            )
        body = " and ".join(bits) if bits else f"{len(self.hits)} items"

        sample_lines: list[str] = []
        for hit in self.hits[:max_examples]:
            text = hit.text.strip().replace("\n", " ")
            if len(text) > 110:
                text = text[:107].rstrip() + "..."
            sample_lines.append(f'"{text}" ({hit.age_hint})')
        sample = "; ".join(sample_lines)
        return (
            f"Found {body} from {scope}, Boss. "
            + (f"Most relevant: {sample}." if sample else "")
        )


# ── recall engine ─────────────────────────────────────────────────────


@dataclass(slots=True)
class OmniRecallConfig:
    default_lookback_hours: float = 48.0
    max_hits_per_source: int = 5
    max_total_hits: int = 12
    enable_timeline: bool = True
    enable_conversation: bool = True
    enable_screen: bool = True


class OmniRecall:
    """Fan-out memory recall across timeline + conversation + screen."""

    def __init__(
        self,
        *,
        timeline_memory: Any | None = None,
        conversation_memory: Any | None = None,
        screen_perception_loop: Any | None = None,
        config: OmniRecallConfig | None = None,
    ) -> None:
        self.timeline = timeline_memory
        self.conversation = conversation_memory
        self.screen = screen_perception_loop
        self.config = config or OmniRecallConfig()

    # ── public API ────────────────────────────────────────────────

    def recall(
        self,
        *,
        query: str = "",
        when: str = "",
        sources: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> RecallReport:
        """Fan-out recall.

        :param query: keyword to match in stored text. Empty = "any".
        :param when: free-form temporal phrase ("yesterday", "5pm", ...).
                     Empty = default lookback window.
        :param sources: override which sources to query
                       ({"timeline", "conversation", "screen"}).
        :param limit: cap on total hits (defaults to config).
        """
        report = RecallReport(query=query.strip(), window=None)
        try:
            window = resolve(when) if when.strip() else None
            if window is None:
                end = datetime.now()
                start = end - timedelta(hours=self.config.default_lookback_hours)
                window = TemporalRange(
                    start, end,
                    f"the past {int(self.config.default_lookback_hours)}h",
                )
                report.used_default_window = True
            report.window = window

            allowed = {s.lower() for s in (sources or [])} or None

            hits: list[RecallHit] = []
            if (
                self.config.enable_timeline
                and (allowed is None or "timeline" in allowed)
            ):
                hits.extend(self._from_timeline(query, window))
            if (
                self.config.enable_conversation
                and (allowed is None or "conversation" in allowed)
            ):
                hits.extend(self._from_conversation(query, window))
            if (
                self.config.enable_screen
                and (allowed is None or "screen" in allowed)
            ):
                hits.extend(self._from_screen(query, window))

            hits.sort(key=lambda h: h.ts, reverse=True)
            cap = int(limit or self.config.max_total_hits)
            report.hits = hits[:cap]
        except Exception as exc:
            logger.exception("OmniRecall failed")
            report.error = str(exc)[:200]
        return report

    # ── per-source fans ───────────────────────────────────────────

    def _from_timeline(
        self, query: str, window: TemporalRange,
    ) -> list[RecallHit]:
        if self.timeline is None:
            return []
        try:
            events = self.timeline.search_user_queries(
                query,
                since_ts=window.start_ts,
                until_ts=window.end_ts,
                limit=self.config.max_hits_per_source,
            )
        except Exception:
            logger.debug("timeline search failed", exc_info=True)
            return []
        out: list[RecallHit] = []
        for ev in events:
            text = ""
            try:
                text = (ev.data.get("text") if isinstance(
                    ev.data, dict,
                ) else "") or ""
            except Exception:
                text = ""
            if not text.strip():
                continue
            out.append(
                RecallHit(
                    source="timeline",
                    ts=float(getattr(ev, "timestamp", 0.0)),
                    text=text.strip(),
                    extra={"event_type": getattr(ev, "type", "")},
                ),
            )
        return out

    def _from_conversation(
        self, query: str, window: TemporalRange,
    ) -> list[RecallHit]:
        if self.conversation is None:
            return []
        turns: list[Any] = []
        try:
            getter = (
                getattr(self.conversation, "iter_turns", None)
                or getattr(self.conversation, "all_turns", None)
                or getattr(self.conversation, "turns", None)
            )
            if callable(getter):
                turns = list(getter()) or []
            elif getter is not None:
                turns = list(getter) or []
        except Exception:
            logger.debug("conversation memory iter failed", exc_info=True)
            return []
        kw = query.strip().lower()
        out: list[RecallHit] = []
        for turn in turns:
            ts = float(
                getattr(turn, "timestamp", None)
                or getattr(turn, "ts", 0.0)
                or 0.0
            )
            if not (window.start_ts <= ts <= window.end_ts):
                continue
            text = (
                getattr(turn, "user_text", None)
                or getattr(turn, "text", None)
                or getattr(turn, "user", None)
                or ""
            )
            text = (str(text) or "").strip()
            if not text:
                continue
            if kw and kw not in text.lower():
                continue
            out.append(RecallHit(source="conversation", ts=ts, text=text))
            if len(out) >= self.config.max_hits_per_source:
                break
        return out

    def _from_screen(
        self, query: str, window: TemporalRange,
    ) -> list[RecallHit]:
        if self.screen is None:
            return []
        try:
            rows = self.screen.query(
                since_ts=window.start_ts,
                until_ts=window.end_ts,
                text_contains=query or None,
                limit=self.config.max_hits_per_source,
            )
        except Exception:
            logger.debug("screen recall failed", exc_info=True)
            return []
        out: list[RecallHit] = []
        for r in rows:
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            out.append(
                RecallHit(
                    source="screen",
                    ts=float(r.get("ts") or 0.0),
                    text=txt[:600],
                    extra={
                        "app": r.get("app", ""),
                        "tokens": int(r.get("tokens") or 0),
                    },
                ),
            )
        return out


__all__ = ["OmniRecall", "OmniRecallConfig", "RecallHit", "RecallReport"]
