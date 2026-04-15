"""
ATOM -- Intent Continuity Layer.

Bridges GoalEngine, SessionMemory, and ConversationMemory to give the
LLM a coherent picture of *what the user is trying to achieve* across
multiple turns -- not just what was said last.

Tracked state:
  - current_intent: the active multi-step goal (from GoalEngine)
  - conversation_thread: topic continuity from recent turns
  - pending_follow_ups: actions offered but not yet acted on

Wired into StructuredPromptBuilder so the LLM sees:
  "User is currently working on: X, last step was: Y"
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.cognitive.goal_engine import GoalEngine
    from core.memory.session_memory import SessionMemory
    from core.conversation_memory import ConversationMemory

logger = logging.getLogger("atom.intent_continuity")

_MAX_THREAD_DEPTH = 10
_MAX_FOLLOW_UPS = 5
_FOLLOW_UP_EXPIRY_S = 300.0


@dataclass
class FollowUp:
    """An action offered to the user that hasn't been acted on yet."""
    action: str
    description: str
    offered_at: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        return (time.time() - self.offered_at) > _FOLLOW_UP_EXPIRY_S


class IntentContinuity:
    """Tracks the user's ongoing intent across conversation turns.

    Provides a prompt-ready context block that tells the LLM what
    the user is *trying to accomplish*, not just what they said.
    """

    def __init__(
        self,
        bus: AsyncEventBus | None = None,
        goals: GoalEngine | None = None,
        session: SessionMemory | None = None,
        conv_memory: ConversationMemory | None = None,
    ) -> None:
        self._bus = bus
        self._goals = goals
        self._session = session
        self._conv_memory = conv_memory
        self._lock = threading.Lock()

        self._conversation_thread: deque[str] = deque(maxlen=_MAX_THREAD_DEPTH)
        self._pending_follow_ups: deque[FollowUp] = deque(maxlen=_MAX_FOLLOW_UPS)
        self._last_intent_label: str = ""
        self._last_command_text: str = ""
        self._last_response_snippet: str = ""

    def wire(
        self,
        goals: GoalEngine | None = None,
        session: SessionMemory | None = None,
        conv_memory: ConversationMemory | None = None,
    ) -> None:
        """Late-bind dependencies after construction."""
        if goals is not None:
            self._goals = goals
        if session is not None:
            self._session = session
        if conv_memory is not None:
            self._conv_memory = conv_memory

    def on_command_complete(
        self,
        text: str,
        *,
        intent: str = "",
        response_snippet: str = "",
    ) -> None:
        """Called after each command completes to update thread state."""
        with self._lock:
            self._last_command_text = text
            self._last_intent_label = intent
            self._last_response_snippet = response_snippet

            topic = self._extract_topic(text)
            if topic and (
                not self._conversation_thread
                or self._conversation_thread[-1] != topic
            ):
                self._conversation_thread.append(topic)

            self._prune_follow_ups()

    def add_follow_up(self, action: str, description: str) -> None:
        """Register an action offered to the user (e.g. 'book that too?')."""
        with self._lock:
            self._pending_follow_ups.append(
                FollowUp(action=action, description=description),
            )

    def consume_follow_up(self, action: str) -> FollowUp | None:
        """If the user acts on a follow-up, remove and return it."""
        with self._lock:
            for i, fu in enumerate(self._pending_follow_ups):
                if fu.action == action:
                    del self._pending_follow_ups[i]
                    return fu
        return None

    def get_current_goal(self) -> dict[str, Any] | None:
        """Return the top active goal from GoalEngine, if any."""
        if not self._goals:
            return None
        try:
            active = self._goals.get_active_goals()
            return active[0] if active else None
        except Exception:
            return None

    def get_current_step(self, goal: dict[str, Any]) -> dict[str, Any] | None:
        """Return the first incomplete step of a goal."""
        steps = goal.get("steps", [])
        for step in steps:
            if step.get("status") != "done":
                return step
        return None

    def context_for_prompt(self) -> str:
        """Build a concise prompt block for LLM injection.

        Example output:
          [INTENT CONTINUITY]
          Current goal: Plan trip to Japan (step 2/5: Research flights)
          Thread: trip planning → flights → budget
          Pending: "Want me to book that too?"
        """
        lines: list[str] = []

        goal = self.get_current_goal()
        if goal:
            title = goal.get("title", "")
            steps = goal.get("steps", [])
            done = sum(1 for s in steps if s.get("status") == "done")
            total = len(steps)
            current_step = self.get_current_step(goal)
            step_label = current_step.get("title", "") if current_step else ""

            goal_line = f"Current goal: {title}"
            if total > 0:
                goal_line += f" (step {done + 1}/{total}"
                if step_label:
                    goal_line += f": {step_label}"
                goal_line += ")"
            lines.append(goal_line)

        with self._lock:
            if self._conversation_thread:
                thread = " → ".join(self._conversation_thread)
                lines.append(f"Thread: {thread}")

            self._prune_follow_ups()
            if self._pending_follow_ups:
                descs = [fu.description for fu in self._pending_follow_ups]
                lines.append(f"Pending offers: {'; '.join(descs)}")

            if self._session and self._session.last_command:
                is_fu = self._session.is_follow_up(self._last_command_text)
                if is_fu:
                    lines.append("(User is continuing a thread — maintain context)")

        if not lines:
            return ""
        return "[INTENT CONTINUITY]\n" + "\n".join(f"  {l}" for l in lines)

    def _extract_topic(self, text: str) -> str:
        """Simple topic extraction from command text."""
        text = text.strip()
        if len(text) < 3:
            return ""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be",
            "to", "of", "and", "in", "that", "it", "for", "on",
            "with", "as", "at", "by", "this", "from", "or", "do",
            "my", "me", "i", "you", "can", "will", "please",
            "what", "how", "who", "where", "when", "hey", "atom",
        }
        words = [
            w for w in text.lower().split()
            if w not in stop_words and len(w) > 2
        ]
        return " ".join(words[:4]) if words else ""

    def _prune_follow_ups(self) -> None:
        """Remove expired follow-ups (must hold lock)."""
        while self._pending_follow_ups and self._pending_follow_ups[0].expired:
            self._pending_follow_ups.popleft()

    def get_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "thread_depth": len(self._conversation_thread),
                "pending_follow_ups": len(self._pending_follow_ups),
                "last_intent": self._last_intent_label,
                "current_goal": bool(self.get_current_goal()),
            }


__all__ = ["IntentContinuity", "FollowUp"]
