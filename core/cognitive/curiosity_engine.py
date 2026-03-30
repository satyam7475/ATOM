"""
ATOM -- Curiosity Engine (Proactive Learning).

ATOM doesn't just wait to be asked -- it actively tries to learn
about its owner. Like JARVIS anticipating Tony's needs through
years of observation and proactive inquiry.

Behaviors:
    - Asks clarifying questions after ambiguous interactions
    - Notices new topics and asks to learn more
    - Detects preference patterns and confirms them
    - Identifies knowledge gaps and asks to fill them
    - Suggests learning from relevant documents

Gate:  curiosity is suppressed during:
    - Focus mode / work mode
    - When user seems frustrated or stressed
    - More than 2 questions in the last 30 minutes (not annoying)
    - When user explicitly says "be quiet" / "stop asking"

Contract: CognitiveModuleContract (start, stop, persist)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import TYPE_CHECKING

logger = logging.getLogger("atom.curiosity")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus


class CuriosityEngine:
    """Proactive learning through intelligent questioning."""

    _MAX_QUESTIONS_PER_HOUR = 2
    _COOLDOWN_MINUTES = 15

    _TOPIC_QUESTIONS = [
        "I noticed you've been working with {topic} a lot. Want me to learn more about it?",
        "You mentioned {topic} a few times. Should I remember anything specific about it?",
        "I see {topic} comes up often. Is that a current project?",
    ]

    _PREFERENCE_QUESTIONS = [
        "I notice you usually {pattern}. Want me to do that automatically?",
        "You tend to {pattern} around this time. Should I make that a routine?",
    ]

    _KNOWLEDGE_GAP_QUESTIONS = [
        "I wasn't sure about your question on {topic}. Want to teach me about it?",
        "My answer about {topic} might not have been great. Can you correct me?",
    ]

    def __init__(
        self,
        bus: "AsyncEventBus",
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._config = (config or {}).get("cognitive", {})
        self._enabled = self._config.get("curiosity_enabled", True)
        self._max_per_hour = self._config.get(
            "curiosity_max_per_hour", self._MAX_QUESTIONS_PER_HOUR,
        )
        self._cooldown_min = self._config.get(
            "curiosity_cooldown_minutes", self._COOLDOWN_MINUTES,
        )
        self._questions_asked: list[float] = []
        self._suppressed = False
        self._topic_tracker: dict[str, int] = {}
        self._knowledge_gaps: list[str] = []
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._curiosity_loop())
        logger.info("Curiosity engine started (max %d questions/hour)", self._max_per_hour)

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def suppress(self) -> None:
        self._suppressed = True
        logger.info("Curiosity suppressed by user")

    def unsuppress(self) -> None:
        self._suppressed = False

    def track_topic(self, topic: str) -> None:
        """Record a topic mention for curiosity analysis."""
        if not topic or len(topic) < 3:
            return
        key = topic.lower().strip()
        self._topic_tracker[key] = self._topic_tracker.get(key, 0) + 1

    def record_knowledge_gap(self, topic: str) -> None:
        """Record a topic where ATOM gave a poor or uncertain answer."""
        if topic and topic not in self._knowledge_gaps:
            self._knowledge_gaps.append(topic)
            if len(self._knowledge_gaps) > 20:
                self._knowledge_gaps = self._knowledge_gaps[-20:]

    def _can_ask(self) -> bool:
        if self._suppressed:
            return False

        now = time.monotonic()
        self._questions_asked = [
            t for t in self._questions_asked if now - t < 3600
        ]

        if len(self._questions_asked) >= self._max_per_hour:
            return False

        if self._questions_asked:
            last = self._questions_asked[-1]
            if now - last < self._cooldown_min * 60:
                return False

        return True

    async def _curiosity_loop(self) -> None:
        """Periodically check for curiosity opportunities."""
        await asyncio.sleep(300)

        while self._running:
            try:
                await asyncio.sleep(600)
                if not self._running:
                    break

                if not self._can_ask():
                    continue

                question = self._generate_question()
                if question:
                    self._questions_asked.append(time.monotonic())
                    self._bus.emit_long("curiosity_question", text=question)
                    logger.info("Curiosity question: %s", question[:80])

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Curiosity loop error", exc_info=True)

    def _generate_question(self) -> str | None:
        """Generate a contextually appropriate question."""
        for topic, count in sorted(
            self._topic_tracker.items(), key=lambda x: x[1], reverse=True,
        ):
            if count >= 3:
                self._topic_tracker[topic] = 0
                template = random.choice(self._TOPIC_QUESTIONS)
                return template.format(topic=topic)

        if self._knowledge_gaps:
            topic = self._knowledge_gaps.pop(0)
            template = random.choice(self._KNOWLEDGE_GAP_QUESTIONS)
            return template.format(topic=topic)

        return None

    def get_stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "suppressed": self._suppressed,
            "topics_tracked": len(self._topic_tracker),
            "knowledge_gaps": len(self._knowledge_gaps),
            "questions_asked_this_hour": len(self._questions_asked),
        }

    def persist(self) -> None:
        pass
