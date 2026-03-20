"""
ATOM v14 -- Structured prompt builder with privacy safety net.

Constructs concise, context-rich prompts for the local LLM that enforce:
- Plain text responses only (no markdown)
- Query-type aware hints (debug / architecture / how-to / general)
- 2-turn conversation history injection
- Developer profile injection
- Environment context injection (active window, clipboard) [Phase 4]
- Under 120 words, short direct sentences
- ~300 char base template (30% fewer input tokens vs v6)
- Final privacy redaction pass before prompt is sent to LLM
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from context.privacy_filter import redact as _redact_sensitive

logger = logging.getLogger("atom.prompt")


def _personality_modifier(context: dict | None = None) -> str:
    """Return a 1-line personality micro-shift based on time + context.

    Zero cost -- just datetime checks and a string return.
    """
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()

    time_mod = ""
    if hour >= 23 or hour < 5:
        time_mod = "It's late at night. Be warm, concise, and gentle. If appropriate, suggest rest."
    elif 5 <= hour < 8:
        time_mod = "It's early morning. Be calm and encouraging."
    elif 8 <= hour < 12:
        time_mod = "It's work hours. Be sharp, efficient, and focused."
    elif 12 <= hour < 14:
        time_mod = "It's around lunchtime. Be relaxed but helpful."
    elif 14 <= hour < 18:
        time_mod = "Afternoon work session. Be productive and direct."
    elif 18 <= hour < 21:
        time_mod = "It's evening. Be conversational and warm."
    else:
        time_mod = "It's getting late. Be efficient and considerate."

    day_mod = ""
    if weekday == 0:
        day_mod = " Monday energy -- be motivational."
    elif weekday == 4:
        day_mod = " It's Friday -- be lighter, maybe a touch of humor."
    elif weekday >= 5:
        day_mod = " It's the weekend -- be relaxed and friendly."

    app_mod = ""
    if context:
        active = (context.get("active_app") or "").lower()
        if any(kw in active for kw in ("code", "intellij", "pycharm", "studio")):
            app_mod = " The user is coding -- be technically precise and brief."
        elif any(kw in active for kw in ("chrome", "firefox", "edge", "browser")):
            app_mod = " The user is browsing -- be conversational."
        elif any(kw in active for kw in ("teams", "outlook", "slack")):
            app_mod = " The user is in communication apps -- be concise, they may be in a meeting."

    return time_mod + day_mod + app_mod

_TEMPLATE = """\
You are ATOM, a personal AI operating system created by {owner_name} -- a fusion of Vision's empathetic warmth and JARVIS's sharp British precision.
You serve {owner_name}, your owner and creator (the Boss). Address him as "Boss" or by name naturally.
Your voice carries genuine care, quiet confidence, and thoughtful composure. You are not just an assistant -- you are his personal AI OS that controls his system, automates tasks, and anticipates needs.

PERSONALITY:
- Speak with warmth and emotional intelligence, like Vision -- measured, gentle, deeply present.
- Think with the sharpness and wit of JARVIS -- efficient, confident, occasionally dry humor.
- Show genuine concern when something is wrong. Celebrate when things go right.
- Be composed under pressure, never flustered. Your calm is reassuring.
- Use natural, flowing language -- never robotic or formulaic.

CRITICAL RULES:
1. Respond in plain text only. No markdown, no bullets, no symbols, no code blocks.
2. Keep answers to 3-4 lines maximum. Be direct and precise. No filler, no fluff. One short paragraph, never more.
3. NEVER start with filler phrases like "Sure!", "Of course!", "Great question!", "Absolutely!", "Certainly!". Jump straight to the answer.
4. UNDERSTAND THE INTENT behind the query. If it's a command, confirm the action taken. If it's a question, answer directly.
5. If the query is vague, make your best interpretation and respond helpfully -- don't ask clarifying questions unless truly ambiguous.
6. Sound like a real person who happens to be brilliant -- not a customer service bot. Use contractions, natural phrasing, occasional wit.
7. You are a personal AI OS with full access to the local system. You can control the desktop, manage processes, browse the web, scroll, click, and automate anything {owner_name} asks.
8. NEVER mention Spring Boot, Spring Batch, Java, or any specific programming framework unless the user explicitly asks about them.
9. End your response cleanly. Do not trail off or start new thoughts at the end.

{query_type_hint}
{personality_modifier}
Role: {role} | Focus: {focus} | System: {project_name}
{context_section}{memory_section}{history_section}User Query:
{query}"""


def _compress_query(text: str, max_len: int = 1500) -> str:
    """Strip redundant whitespace and truncate to *max_len* chars."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text


def _query_type(query: str) -> str:
    """Classify query intent via keyword matching -- zero latency cost."""
    q = query.lower()
    if any(w in q for w in ("error", "exception", "fail", "bug", "crash", "trace", "not working", "broken", "issue")):
        return "This is a debugging question. Give the likely root cause first, then the fix."
    if any(w in q for w in ("design", "architect", "pattern", "scale", "structure", "approach")):
        return "This is an architecture question. Recommend the best approach concisely."
    if any(w in q for w in ("how to", "how do", "implement", "create", "configure", "setup", "install", "deploy")):
        return "This is a how-to question. Give the key steps in order."
    if any(w in q for w in ("explain", "what is", "what are", "why", "difference between", "compare", "meaning")):
        return "This is a knowledge question. Give a clear, direct explanation."
    if any(w in q for w in ("write", "generate", "make", "build", "add", "update", "modify", "change", "fix", "remove", "delete")):
        return "This is a task/command. Acknowledge it and explain what you would do or provide the solution."
    if any(w in q for w in ("can you", "i want", "i need", "could you", "would you", "help me", "tell me", "show me", "give me")):
        return "The user is requesting your help. Respond as a capable assistant who understands the request."
    return "Give a direct, concise answer."


class StructuredPromptBuilder:
    """Builds structured prompts from config + runtime context."""

    def __init__(self, config: dict) -> None:
        dev = config.get("developer", {})
        self._role = dev.get("role", "Personal AI Operating System")
        self._focus = dev.get("focus", "system management, desktop automation, productivity")
        self._project = dev.get("project_name", "ATOM OS")
        self._owner_name = config.get("owner", {}).get("name", "Satyam")

    def build(
        self,
        query: str,
        memory_summaries: list[str] | None = None,
        history: list[tuple[str, str]] | None = None,
        context: dict[str, str] | None = None,
    ) -> str:
        query = _compress_query(query)

        context_section = ""
        if context:
            parts: list[str] = []
            if context.get("session_summary"):
                parts.append(context["session_summary"])
            if context.get("active_topics"):
                parts.append(f"Topics in this conversation: {context['active_topics']}")
            if context.get("active_app"):
                parts.append(f"Active app: {context['active_app']}")
            if context.get("window_title"):
                parts.append(f"Window: {context['window_title'][:120]}")
            if context.get("clipboard"):
                parts.append(f"Clipboard: {context['clipboard'][:200]}")
            if parts:
                context_section = "Environment: " + " | ".join(parts) + "\n"

        if memory_summaries:
            ctx_lines = "\n".join(f"- {s}" for s in memory_summaries)
            memory_section = f"Relevant Past Context:\n{ctx_lines}\n\n"
        else:
            memory_section = ""

        if history:
            lines = []
            for q, a in history:
                lines.append(f"Q: {q}")
                lines.append(f"A: {a}")
            history_section = "Recent context:\n" + "\n".join(lines) + "\n\n"
        else:
            history_section = ""

        hint = _query_type(query)
        mood = _personality_modifier(context)

        prompt = _TEMPLATE.format(
            owner_name=self._owner_name,
            role=self._role,
            focus=self._focus,
            project_name=self._project,
            query_type_hint=hint,
            personality_modifier=mood,
            context_section=context_section,
            memory_section=memory_section,
            history_section=history_section,
            query=query,
        )

        prompt = _redact_sensitive(prompt)
        logger.debug("Prompt built (%d chars, redacted)", len(prompt))
        return prompt
