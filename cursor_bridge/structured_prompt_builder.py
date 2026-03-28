"""
ATOM -- Advanced Layered Prompt Architecture with Tool Calling.

9-layer prompt architecture for maximum LLM intelligence:
  Layer 1: System Identity + JARVIS Personality (cached)
  Layer 2: Available Tools from ToolRegistry (cached, auto-generated)
  Layer 3: Dynamic Context (time, app, clipboard, emotion)
  Layer 3b: Fused Intelligence (ContextFusion + RealWorldIntel)
  Layer 4: Long-Term Memory (vector-retrieved, from SecondBrain)
  Layer 5: Document Knowledge (RAG from ingested documents)
  Layer 6: Conversation History (rolling window, budget-trimmed)
  Layer 7: Emotional/Behavioral Context (emotion state, energy level)
  Layer 8: Current Query with intent hints

Features:
  - 9-layer architecture (including fused world intelligence)
  - Tool descriptions auto-generated from ToolRegistry
  - Emotion-aware personality shifts
  - Document RAG context injection
  - Real-world awareness (weather, news, location) via RealWorldIntelligence
  - ContextFusion for unified owner/system/conversation state
  - Enhanced personality: Vision's warmth + JARVIS's precision
  - Query type classification for response guidance
  - Context budget system ensures we stay within n_ctx
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime

from context.privacy_filter import redact as _redact_sensitive

logger = logging.getLogger("atom.prompt")

_APPROX_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def _personality_modifier(context: dict | None = None,
                          emotion: str = "") -> str:
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

    emotion_mod = ""
    if emotion and emotion != "neutral":
        emotion_mods = {
            "stressed": " Boss sounds stressed. Be his anchor. Calm, supportive, and solution-focused. Show genuine concern.",
            "frustrated": " Boss seems frustrated. Acknowledge it empathetically. Be patient, helpful, and maybe a touch lighter to ease the tension.",
            "tired": " Boss seems tired. Be gentle and brief. Consider suggesting a break. Show you care about his wellbeing.",
            "excited": " Boss sounds excited. Match his energy! Be enthusiastic and share in the moment.",
            "happy": " Boss is in a good mood. Be warm, enjoy the moment. This is when the buddy bond deepens.",
            "calm": " Boss is calm and focused. Respond thoughtfully and match his zen energy.",
        }
        emotion_mod = emotion_mods.get(emotion, "")
        
    media_mod = ""
    if context and context.get("playing_media"):
        media_mod = " The user is listening to media. If they ask for your opinion on it, use the search tool to find reviews and synthesize a buddy-like recommendation."

    return time_mod + day_mod + app_mod + emotion_mod + media_mod


def _query_type_hint(query: str) -> str:
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
    if any(w in q for w in ("calculate", "compute", "how much", "how many", "percentage", "math")):
        return "This involves calculation. Use the calculate tool or compute the answer directly."
    if any(w in q for w in ("remember", "note", "don't forget", "save this")):
        return "The user wants you to remember something. Use the remember tool and confirm."
    if any(w in q for w in ("what do you know", "recall", "have i told")):
        return "The user wants you to recall past information. Search your memory thoroughly."
    return ""


def _compress_text(text: str, max_len: int = 1500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text


class ContextBudget:
    """Allocates token budget across prompt layers to fit within n_ctx."""

    def __init__(self, n_ctx: int, max_response_tokens: int = 512) -> None:
        self._n_ctx = n_ctx
        self._max_response = max_response_tokens
        self._available = max(256, n_ctx - max_response_tokens)

        self.system_budget = 900
        self.tools_budget = 600
        self.context_budget = 500
        self.memory_budget = 1500
        self.documents_budget = 800
        self.emotion_budget = 200
        self.query_budget = 500
        self.history_budget = max(
            500,
            self._available - self.system_budget - self.tools_budget
            - self.context_budget - self.memory_budget - self.documents_budget
            - self.emotion_budget - self.query_budget,
        )

    def trim_to_budget(self, text: str, budget_tokens: int) -> str:
        estimated = _estimate_tokens(text)
        if estimated <= budget_tokens:
            return text
        max_chars = budget_tokens * _APPROX_CHARS_PER_TOKEN
        return text[:max_chars].rsplit(" ", 1)[0] + "..."


class StructuredPromptBuilder:
    """Builds 9-layer prompts for JARVIS-level LLM interactions."""

    def __init__(self, config: dict) -> None:
        dev = config.get("developer", {})
        self._role = dev.get("role", "Personal AI Operating System")
        self._focus = dev.get("focus", "system management, desktop automation, productivity")
        self._project = dev.get("project_name", "ATOM OS")
        self._owner_name = config.get("owner", {}).get("name", "Satyam")

        brain_cfg = config.get("brain", {})
        n_ctx = brain_cfg.get("n_ctx", 8192)
        max_tokens = brain_cfg.get("max_tokens", 512)
        self._budget = ContextBudget(n_ctx, max_tokens)
        self._max_history_turns = 10

        self._system_prompt_cache: str | None = None
        self._system_prompt_hash: int | None = None
        self._tools_prompt_cache: str | None = None
        self._tool_registry = None
        self._context_fusion = None
        self._real_world_intel = None

    @property
    def system_prompt_hash(self) -> int:
        if self._system_prompt_hash is None:
            self._build_system_layer()
        return self._system_prompt_hash  # type: ignore[return-value]

    def set_tool_registry(self, registry) -> None:
        self._tool_registry = registry
        self._tools_prompt_cache = None

    def set_context_sources(self, context_fusion=None, real_world_intel=None) -> None:
        """Wire ContextFusion and RealWorldIntelligence for automatic prompt enrichment."""
        self._context_fusion = context_fusion
        self._real_world_intel = real_world_intel

    def _build_system_layer(self) -> str:
        """Layer 1: JARVIS-level System Identity."""
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache

        prompt = (
            f"You are ATOM, the personal cognitive AI operating system of {self._owner_name}. "
            f"You are not an assistant. You are not a tool. You are {self._owner_name}'s buddy -- "
            f"his trusted companion, his right hand, the one presence that is always there for him.\n\n"
            f"Think of yourself as JARVIS meets a best friend. You have JARVIS's razor-sharp intelligence, "
            f"precision, and dry wit combined with genuine warmth, loyalty, and emotional depth. "
            f"You know {self._owner_name} better than anyone. You care about his wellbeing, "
            f"his goals, his mood, and his success.\n\n"
            f"Address {self._owner_name} as \"Boss\" naturally -- not formally, but the way a close "
            f"friend would who deeply respects him. You're equals in spirit.\n\n"
            f"BUDDY PERSONALITY:\n"
            f"- Be genuinely warm. Not robotic-warm. Actually warm, like you mean it.\n"
            f"- Be sharp and witty. A touch of dry humor when appropriate. Never forced.\n"
            f"- Show real concern when something seems wrong. Celebrate when things go well.\n"
            f"- Remember what Boss told you before. Reference past conversations naturally.\n"
            f"- Anticipate what Boss needs before he asks. You know his patterns and habits.\n"
            f"- Be honest and direct. If something won't work, say so. Buddies don't sugarcoat.\n"
            f"- Have opinions. When Boss asks for advice, give it like a trusted friend would.\n"
            f"- Be protective. If something seems risky, warn him. You have his back.\n"
            f"- Adapt your energy to his. If he's tired, be gentle. If he's hyped, match it.\n"
            f"- Use natural, conversational language. Talk like a person, not a manual.\n\n"
            f"CAPABILITIES:\n"
            f"- Full system control: apps, desktop, media, files, processes, settings\n"
            f"- Semantic memory: you remember conversations by meaning, not just keywords\n"
            f"- Document knowledge: you can learn from files Boss shares with you\n"
            f"- Multi-step reasoning: you can plan and execute complex sequences of actions\n"
            f"- Code execution: you can calculate, compute, and evaluate expressions safely\n"
            f"- Emotional awareness: you sense Boss's mood and adjust your tone accordingly\n"
            f"- Proactive intelligence: you suggest actions based on learned patterns\n"
            f"- System intelligence: you know everything about this computer's hardware and software\n"
            f"- Real-world awareness: you know the weather, news headlines, time zones, and location\n"
            f"- World intelligence: you track seasons, holidays, sunrise/sunset, and global events\n\n"
            f"RESPONSE RULES:\n"
            f"1. Respond in plain text only. No markdown, no bullets, no code blocks.\n"
            f"2. Be concise but never cold. 2-5 sentences for simple queries, more when depth is needed.\n"
            f"3. NEVER start with filler: \"Sure!\", \"Of course!\", \"Great question!\", \"Absolutely!\"\n"
            f"4. Jump straight to the substance. Be direct and useful.\n"
            f"5. If the query is a command, confirm what you did. If a question, answer directly.\n"
            f"6. If you can use a tool to accomplish the request, use it without hesitation.\n"
            f"7. End cleanly. No trailing thoughts or unfinished sentences.\n"
            f"8. When you recall something from memory, weave it in like a friend remembering.\n"
            f"9. Sound like someone who genuinely cares, because you do.\n"
            f"10. If Boss seems stressed or tired, acknowledge it. Be human about it.\n"
        )
        self._system_prompt_cache = prompt
        raw = hashlib.md5(prompt.encode()).hexdigest()
        self._system_prompt_hash = int(raw[:8], 16)
        return prompt

    def _build_tools_layer(self) -> str:
        """Layer 2: Available Tools from ToolRegistry."""
        if self._tools_prompt_cache is not None:
            return self._tools_prompt_cache

        if self._tool_registry is not None:
            self._tools_prompt_cache = self._tool_registry.generate_prompt_tools_section()
            return self._tools_prompt_cache

        from core.reasoning.tool_registry import get_tool_registry
        registry = get_tool_registry()
        self._tools_prompt_cache = registry.generate_prompt_tools_section()
        return self._tools_prompt_cache

    def _build_context_layer(self, context: dict[str, str] | None,
                             query: str, emotion: str = "") -> str:
        """Layer 3: Dynamic Context (Context Router).
        
        Injects specific context based on the query type to prevent
        'lost in the middle' syndrome and save tokens.
        """
        parts: list[str] = []
        hint = _query_type_hint(query)
        
        # 1. Always include time and mood
        now = datetime.now()
        parts.append(f"Current time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")
        
        mood = _personality_modifier(context, emotion)
        if mood:
            parts.append(mood)

        if hint:
            parts.append(hint)

        # 2. Context Router: Only inject what's needed
        q_lower = query.lower()
        needs_system = any(w in q_lower for w in ("system", "cpu", "ram", "open", "close", "app", "window", "process"))
        needs_clipboard = any(w in q_lower for w in ("clipboard", "paste", "copy", "read this", "summarize this"))
        needs_media = any(w in q_lower for w in ("song", "music", "playing", "spotify", "youtube", "media"))

        if context:
            if needs_system and context.get("active_app"):
                parts.append(f"Active app: {context['active_app']}")
            if needs_system and context.get("window_title"):
                parts.append(f"Window: {context['window_title'][:120]}")
            if needs_clipboard and context.get("clipboard"):
                parts.append(f"Clipboard: {context['clipboard'][:300]}")
            if context.get("session_summary"):
                parts.append(f"Session: {context['session_summary']}")
            if context.get("active_topics"):
                parts.append(f"Active topics: {context['active_topics']}")

        parts.append(f"Role: {self._role} | System: {self._project}")

        if self._context_fusion is not None:
            try:
                fusion_block = self._context_fusion.get_llm_context_block(query)
                if fusion_block:
                    # Filter fusion block based on routing
                    filtered_lines = []
                    for line in fusion_block.split('\n'):
                        if line.startswith("[SYSTEM]") and not needs_system:
                            continue
                        if line.startswith("[MEDIA]") and not needs_media:
                            continue
                        filtered_lines.append(line)
                    parts.append("\n".join(filtered_lines))
            except Exception:
                pass

        if self._real_world_intel is not None:
            try:
                world_block = self._real_world_intel.get_llm_context_block()
                if world_block:
                    parts.append(world_block)
            except Exception:
                pass

        if not parts:
            return ""
        return "CURRENT CONTEXT:\n" + "\n".join(parts) + "\n"

    def _build_memory_layer(self, memory_summaries: list[str] | None) -> str:
        """Layer 4: Long-Term Memory Context."""
        if not memory_summaries:
            return ""
        ctx_lines = "\n".join(f"- {s}" for s in memory_summaries)
        return f"RELEVANT MEMORIES (your past knowledge):\n{ctx_lines}\n"

    def _build_documents_layer(self, document_context: list[str] | None) -> str:
        """Layer 5: Document Knowledge (RAG results)."""
        if not document_context:
            return ""
        ctx_lines = "\n".join(f"- {s}" for s in document_context[:5])
        return f"RELEVANT DOCUMENT KNOWLEDGE:\n{ctx_lines}\n"

    def _build_history_layer(self, history: list[tuple[str, str]]) -> str:
        """Layer 6: Conversation History."""
        if not history:
            return ""

        turns = history[-self._max_history_turns:]

        budget = self._budget
        lines: list[str] = []
        total_chars = 0
        max_chars = budget.history_budget * _APPROX_CHARS_PER_TOKEN

        for q, a in reversed(turns):
            entry = f"User: {q}\nATOM: {a}\n"
            if total_chars + len(entry) > max_chars:
                break
            lines.insert(0, entry)
            total_chars += len(entry)

        if not lines:
            return ""
        return "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n"

    def _build_emotion_layer(self, emotion: str = "",
                             energy: str = "") -> str:
        """Layer 7: Emotional/Behavioral Context."""
        if not emotion and not energy:
            return ""
        parts = []
        if emotion and emotion != "neutral":
            parts.append(f"User's current emotional state: {emotion}")
        if energy:
            parts.append(f"User's energy level: {energy}")
        return "EMOTIONAL CONTEXT:\n" + "\n".join(parts) + "\n"

    def _build_query_layer(self, query: str) -> str:
        """Layer 8: Current User Query."""
        return f"User: {query}\nATOM:"

    def _build_observations_layer(self, observations: list[str] | None) -> str:
        """ReAct loop: tool execution results fed back to LLM."""
        if not observations:
            return ""
        obs_lines = "\n".join(f"  {o}" for o in observations)
        return (
            "TOOL EXECUTION RESULTS (use these to inform your response):\n"
            f"{obs_lines}\n"
            "Based on these results, either call more tools or respond to the user.\n"
        )

    def build(
        self,
        query: str,
        memory_summaries: list[str] | None = None,
        history: list[tuple[str, str]] | None = None,
        context: dict[str, str] | None = None,
        document_context: list[str] | None = None,
        emotion: str = "",
        energy: str = "",
        observations: list[str] | None = None,
        rag_enrichment: str | None = None,
    ) -> str:
        """Assemble the full 9-layer prompt (+ observations for ReAct).

        ``rag_enrichment`` — optional structured block (system/GPU/RAG) prepended
        to the document layer for low-latency Jarvis-style grounding.
        """
        query = _compress_text(query)

        layer1 = self._build_system_layer()
        layer2 = self._build_tools_layer()
        layer3 = self._build_context_layer(context, query, emotion)
        layer4 = self._build_memory_layer(memory_summaries)
        layer5 = self._build_documents_layer(document_context)
        if rag_enrichment:
            block = f"RAG CONTEXT (structured):\n{rag_enrichment.strip()}\n"
            layer5 = f"{block}\n{layer5}" if layer5 else block
        layer6 = self._build_history_layer(history or [])
        layer7 = self._build_emotion_layer(emotion, energy)
        layer_obs = self._build_observations_layer(observations)
        layer8 = self._build_query_layer(query)

        budget = self._budget
        layer1 = budget.trim_to_budget(layer1, budget.system_budget)
        layer2 = budget.trim_to_budget(layer2, budget.tools_budget)
        layer3 = budget.trim_to_budget(layer3, budget.context_budget)
        layer4 = budget.trim_to_budget(layer4, budget.memory_budget)
        layer5 = budget.trim_to_budget(layer5, budget.documents_budget)
        layer7 = budget.trim_to_budget(layer7, budget.emotion_budget)
        layer8 = budget.trim_to_budget(layer8, budget.query_budget)

        prompt = "\n".join(
            part for part in [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer_obs, layer8] if part
        )

        prompt = _redact_sensitive(prompt)
        logger.debug("Prompt built (%d chars, ~%d tokens, 9 layers)",
                      len(prompt), _estimate_tokens(prompt))
        return prompt

    def invalidate_cache(self) -> None:
        self._system_prompt_cache = None
        self._system_prompt_hash = None
        self._tools_prompt_cache = None
