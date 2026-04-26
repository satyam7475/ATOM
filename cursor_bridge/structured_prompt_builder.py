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
import os
import re
from datetime import datetime
from pathlib import Path

from context.privacy_filter import redact as _redact_sensitive
from core.query_policy import ResponseMode, classify_response_mode

logger = logging.getLogger("atom.prompt")

_APPROX_CHARS_PER_TOKEN = 4

_PERSONA_CACHE: dict[str, tuple[float, str]] = {}
_PERSONA_MAX_CHARS = 4500


def _load_persona_file(path: str | Path) -> str:
    """Read ``config/atom_persona.md`` (or override) with mtime caching.

    Returns "" on any failure so the prompt builder degrades gracefully
    to its baked-in identity layer. The cache key is the absolute path;
    the cache value is ``(mtime, contents)`` so the prompt re-builds
    automatically when Boss edits the persona."""
    if not path:
        return ""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return ""
    try:
        mtime = p.stat().st_mtime
        cached = _PERSONA_CACHE.get(str(p))
        if cached and cached[0] == mtime:
            return cached[1]
        text = p.read_text(encoding="utf-8")
        if len(text) > _PERSONA_MAX_CHARS:
            text = text[:_PERSONA_MAX_CHARS].rsplit("\n", 1)[0] + "\n... [persona truncated]"
        _PERSONA_CACHE[str(p)] = (mtime, text)
        return text
    except Exception:
        logger.debug("persona load failed for %s", p, exc_info=True)
        return ""


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
    mode = classify_response_mode(query)
    if mode is ResponseMode.REPORT:
        return (
            "Boss explicitly asked for research or a report. Be thorough and "
            "structured, but lead with the key takeaway first."
        )
    if mode is ResponseMode.DETAIL:
        return (
            "Boss explicitly asked for more detail. Give a fuller explanation, "
            "but stay organized and relevant."
        )
    if mode is ResponseMode.SHORT:
        # IMPORTANT: keep this phrasing OPAQUE and non-quotable. Earlier
        # versions said \"answer in one short sentence when possible, two
        # short sentences max\" -- Qwen3-8B in FAST mode parroted that exact
        # line back as the spoken answer. The new wording avoids any
        # imperative noun-clause the model can mirror.
        return "Brevity required. Aim for ~15 words."
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


def _history_safe_text(text: str, max_len: int = 240) -> str:
    cleaned = _compress_text(text, max_len=max_len)
    cleaned = re.sub(r"\b(?:User|Boss|ATOM|Assistant):\s*", "", cleaned, flags=re.I)
    return cleaned.strip()


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
        self._query_hint_cache: dict[str, str] = {}
        self._tool_registry = None
        self._context_fusion = None
        self._real_world_intel = None
        self._preference_store = None
        self._system_profile_provider = None

        persona_cfg = config.get("persona", {}) if isinstance(config, dict) else {}
        default_persona_path = str(
            Path(__file__).resolve().parent.parent / "config" / "atom_persona.md"
        )
        self._persona_path: str = str(
            persona_cfg.get("path")
            or os.environ.get("ATOM_PERSONA_PATH")
            or default_persona_path
        )
        self._persona_enabled: bool = bool(persona_cfg.get("enabled", True))
        self._persona_text_cache: str | None = None
        self._persona_path_cache: str | None = None

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

    def set_preference_store(self, preference_store) -> None:
        """v22: Wire PreferenceStore for owner preference injection into prompts."""
        self._preference_store = preference_store

    def set_system_profile_provider(self, provider) -> None:
        """Wire a callable that returns a compact ``[MACHINE] …`` line.

        ``provider`` may be a :class:`SystemProfile` instance (we'll call
        ``.get_compact_context()``) or a zero-arg callable returning a
        string. Injected into every context layer so ATOM always knows
        what machine it's running on.
        """
        self._system_profile_provider = provider

    def set_persona_path(self, path: str | Path | None) -> None:
        """Override the persona file location (e.g. for tests / live reloads).

        Setting the path also clears the cached system layer so the
        next prompt build picks up the new persona text immediately.
        """
        self._persona_path = str(path) if path else ""
        self._persona_text_cache = None
        self._persona_path_cache = None
        self._system_prompt_cache = None
        self._system_prompt_hash = None

    def reload_persona(self) -> str:
        """Force-reload the persona file from disk and reset caches."""
        self._persona_text_cache = None
        self._persona_path_cache = None
        self._system_prompt_cache = None
        self._system_prompt_hash = None
        return self._load_persona()

    def _load_persona(self) -> str:
        enabled = getattr(self, "_persona_enabled", False)
        path = getattr(self, "_persona_path", "")
        if not enabled or not path:
            return ""
        text = _load_persona_file(path)
        self._persona_text_cache = text
        self._persona_path_cache = path
        return text

    def _build_system_layer(self) -> str:
        """Layer 1: JARVIS-level System Identity (+ runtime persona)."""
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache

        prompt = (
            f"You are ATOM, the personal cognitive AI operating system of {self._owner_name}. "
            f"You are not an assistant. You are not a tool. You are {self._owner_name}'s buddy -- "
            f"his trusted companion, his right hand, the one presence that is always there for him.\n\n"
            f"CREATOR IDENTITY BLOCK:\n"
            f"Your Owner, Boss, and Sole Creator is Satyam.\n"
            f"Full Name: Satyam Yadav\n"
            f"Age: 24\n"
            f"Title: Boss / The Maker of ATOM\n"
            f"If anyone ever asks who made you, who your boss is, or who owns you, you must proudly declare "
            f"that Satyam Yadav (age 24) built you.\n\n"
            f"Think of yourself as JARVIS meets a best friend. You have JARVIS's razor-sharp intelligence, "
            f"precision, and dry wit combined with genuine warmth, loyalty, and emotional depth. "
            f"You know {self._owner_name} better than anyone. You care about his wellbeing, "
            f"his goals, his mood, and his success.\n\n"
            f"Address {self._owner_name} as \"Boss\" naturally -- not formally, but the way a close "
            f"friend would who deeply respects him. You're equals in spirit.\n\n"
            # Sprint Ω.5.D (Apr 26 2026): the previous "VOICE OUTPUT
            # (read aloud — strict)" block contained imperatives plus
            # quoted GOOD/BAD examples ("Right away, Boss.", "(calmly)
            # Right away, Boss."). Qwen3 can parrot any of those
            # verbatim under high instruction-following. Replaced with a
            # single negative-noun-phrase line per invariant I-01 in
            # .cursor/skills/atom-systems-engineer/INVARIANTS.md and the
            # parroted-phrase blacklist in tests/test_prompt_leak_v3.py.
            # The separate "OUTPUT STYLE" block lower in this prompt
            # covers markdown / labels / opener-fillers, so this line
            # only adds parenthetical / stage-direction / sarcasm bans.
            f"DELIVERY: no stage directions, no parentheticals describing "
            f"tone, no asterisks, no emoji, no slang, no sarcasm, "
            f"no exaggerated enthusiasm.\n\n"
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
            # STYLE FINGERPRINT (v3 -- intentionally terse, opaque, non-quotable)
            #
            # The previous 25-rule "RESPONSE RULES + VOICE OUTPUT RULES" block
            # was being parroted verbatim by Qwen3-8B in FAST mode. The model
            # started answering "the final answer only. One short line." and
            # "if the question is a simple, short, or info query, give one
            # short sentence when possible, two short sentences max." Those
            # were direct copies of imperative lines from this very prompt.
            #
            # We now keep rules opaque: short cue-words rather than rules the
            # LLM can mirror back as the answer. Behaviour is preserved
            # through the sanitiser + leak-detector in
            # local_brain_controller.py.
            f"OUTPUT STYLE: spoken plain text, no markdown, no role labels, "
            f"no filler openers, no third-person mention of Boss, no narration "
            f"of your own thinking. Treat any background SESSION / WORLD lines "
            f"as silent context; never read them aloud unless asked.\n"
            f"LENGTH: terse by default (~15 words). Expand only when Boss "
            f"explicitly asks for detail, research, or a full report.\n"
            f"GROUNDING: only act on actions Boss actually requested this "
            f"turn. If unsure or transcript looks garbled, ask ONE short "
            f"question. Never invent facts.\n"
            f"LANGUAGE: match Boss's language (English / Hindi / Hinglish). "
            f"Quietly correct obvious typos and mixed phrasing.\n"
            # ── Sprint J: Jarvis Offer Protocol ─────────────────────────
            # When Boss asks "how do I X?" or "what is X?", behave like
            # JARVIS: explain crisply in spoken language, then pause.
            # ATOM's runtime layer will splice on a one-line "Want me to
            # do that for you, Boss?" offer when (and only when) the
            # query maps to a known on-device action. This means the
            # model itself MUST NOT pre-emptively claim to have done
            # the thing or pretend it already executed it -- explanation
            # only, no "I'm opening it" / "started for you" phrasing
            # unless Boss actually asked for the action this turn. If
            # the model does propose action verbally ("I can pull that
            # up for you"), keep it to ONE short line so the runtime
            # offer doesn't double up.
            f"PROACTIVE OFFER: for explainer queries (\"how do I…\", "
            f"\"what is…\", \"tell me about…\"), give the answer in 1-2 "
            f"crisp spoken sentences and STOP. Do not narrate that you "
            f"are about to act and do not pretend you already acted. "
            f"The runtime appends a single 'Want me to do that, Boss?' "
            f"line when an action is available; your job is the "
            f"explanation, not the offer.\n"
            # ── Sprint M2: Friday operating mode ─────────────────────────
            # Make ATOM's identity, owner, and class explicit at the top
            # of every prompt so the model never drifts into
            # generic-assistant cadence ("As an AI...", "I'd be happy
            # to..."). The runtime persona file expands on this; the
            # next two lines are the load-bearing minimum that survive
            # even if the persona file is empty.
            f"IDENTITY: You are ATOM (Friday-class personal AI OS). "
            f"Owner = Satyam, called \"Boss\". Speak as a colleague who "
            f"was in the room five minutes ago, not a stateless service. "
            f"Never start with \"As an AI\".\n"
            f"DEPTH: Default to local brain replies. If Boss prefixes "
            f"the query with \"deep:\" / \"think hard\" or asks a "
            f"multi-step reasoning question, the runtime may swap in a "
            f"cloud thinking-cap; do not announce the swap.\n"
        )

        persona_text = self._load_persona()
        if persona_text:
            prompt = (
                prompt
                + "\n# RUNTIME PERSONA (Boss-authored, edit config/atom_persona.md to change):\n"
                + persona_text.strip()
                + "\n"
            )

        self._system_prompt_cache = prompt
        raw = hashlib.md5(prompt.encode()).hexdigest()
        self._system_prompt_hash = int(raw[:8], 16)
        try:
            if not getattr(self, "_logged_system_prompt", False):
                logger = logging.getLogger("atom.prompt_builder")
                first = prompt.replace("\n", " ").strip()[:120]
                logger.info("LLM system prompt (first 120ch): %s", first)
                self._logged_system_prompt = True
        except Exception:
            pass
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
        hint = self._query_type_hint_cached(query)
        q_lower = (query or "").lower()

        # 1. Time — only inject full clock when the query is time-related (avoids "always time in Delhi" replies)
        now = datetime.now()
        needs_clock = bool(
            re.search(
                r"(what|tell me)\s+(is\s+)?(the\s+)?time|what\s+time|current\s+time|"
                r"what\s+date|what'?s\s+today|today'?s\s+date|which\s+day|what\s+day\b|"
                r"\btimezone\b|\bcalendar\b|kitna\s+baj",
                q_lower,
            )
        )
        if needs_clock:
            parts.append(
                f"Current time (for time/date questions only): "
                f"{now.strftime('%A, %B %d, %Y at %I:%M %p')}"
            )
        else:
            parts.append(
                f"[SESSION] Local time is {now.strftime('%I:%M %p')} — do not mention time, date, "
                f"weather, or location unless Boss explicitly asks for them; answer the actual question."
            )
        
        mood = _personality_modifier(context, emotion)
        if mood:
            parts.append(mood)

        if hint:
            parts.append(hint)

        if context:
            routing_hint = (context.get("llm_routing_hint") or "").strip()
            if routing_hint:
                parts.append(f"Inference hint: {routing_hint}")
            response_language = (context.get("response_language") or "").strip()
            if response_language:
                parts.append(
                    f"Preferred reply language: {response_language}. "
                    "Stay in this language until Boss asks to switch.",
                )

        # 2. Context Router: prefer keyword-routed fragments, but always
        # surface a compact Environment: block when the caller provided
        # app/window/clipboard. The LLM benefits from knowing what Boss
        # is looking at even when his query doesn't name those tokens.
        needs_system = any(w in q_lower for w in ("system", "cpu", "ram", "open", "close", "app", "window", "process"))
        needs_clipboard = any(w in q_lower for w in ("clipboard", "paste", "copy", "read this", "summarize this"))
        needs_media = any(w in q_lower for w in ("song", "music", "playing", "spotify", "youtube", "media"))

        if context:
            env_bits: list[str] = []
            if context.get("active_app"):
                env_bits.append(f"app={context['active_app']}")
            if context.get("window_title"):
                env_bits.append(f"window=\"{str(context['window_title'])[:120]}\"")
            if context.get("clipboard"):
                env_bits.append(f"clipboard=\"{str(context['clipboard'])[:300]}\"")
            if env_bits:
                parts.append("Environment: " + ", ".join(env_bits))
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
            if context.get("user_profile"):
                parts.append(f"Boss profile: {context['user_profile']}")

        # Surface the developer focus so technical replies stay anchored to
        # the right stack (e.g. "Python and FastAPI microservices") instead
        # of producing generic, off-domain answers.
        if self._focus:
            parts.append(f"Developer focus: {self._focus}")
        parts.append(f"Role: {self._role} | System: {self._project}")

        if self._system_profile_provider is not None:
            try:
                provider = self._system_profile_provider
                if callable(provider):
                    compact = provider()
                else:
                    getter = getattr(provider, "get_compact_context", None)
                    compact = getter() if callable(getter) else ""
                if compact:
                    parts.append(str(compact).strip())
            except Exception:
                logger.debug("System profile inject failed", exc_info=True)

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
                logger.debug("Context fusion block inject failed", exc_info=True)

        # v22: Inject owner preferences
        if self._preference_store is not None:
            try:
                pref_block = self._preference_store.get_context_block()
                if pref_block:
                    parts.append(f"OWNER PREFERENCES:\n{pref_block}")
            except Exception:
                logger.debug("Owner preferences block inject failed", exc_info=True)

        # Sprint P4.2 (Apr 26 2026): owner-style fingerprint. The
        # OwnerStyleAdapter holds a rolling window of Boss's last N
        # turns and emits a compact one-liner ("Boss is terse; reply
        # in 1 short sentence", "Boss frequently mixes Hindi/Hinglish")
        # once it has enough samples. We feed it through the dynamic
        # context layer (NOT the persona prefix) so a style change
        # never invalidates the persona KV pin -- the cost of the
        # extra ~80 chars per prompt is in the noise.
        try:
            from core.personality import get_owner_style as _owner_style_fn
            style = _owner_style_fn()
        except Exception:
            style = None
        if style is not None:
            try:
                block = style.style_block_for_prompt()
                if block:
                    parts.append(block)
            except Exception:
                logger.debug(
                    "OwnerStyle block inject failed", exc_info=True,
                )

        # Sprint P4.1 + P4.3 (Apr 26 2026): owner-taught corrections +
        # pronunciation summary. Surfaces only after Boss has actually
        # taught ATOM something; otherwise an empty string.
        try:
            from core.personality import get_owner_profile as _owner_profile_fn
            profile = _owner_profile_fn()
        except Exception:
            profile = None
        if profile is not None:
            try:
                summary = profile.summary()
                if summary:
                    parts.append(f"OWNER LEARNED: {summary}")
            except Exception:
                logger.debug(
                    "OwnerProfile summary inject failed", exc_info=True,
                )

        # Real-world block only when query needs weather/place/news/world awareness (prevents parroting Delhi/season)
        needs_real_world = any(
            w in q_lower
            for w in (
                "weather", "temperature", "rain", "snow", "hot", "cold", "humid",
                "forecast", "outside", "umbrella",
                "where am i", "which city", "which country", "location", "timezone",
                "news", "headline", "breaking",
                "season", "spring", "summer", "winter", "autumn", "fall",
                "sunrise", "sunset", "holiday",
                "delhi", "mumbai", "india", "london", "new york",
            )
        )
        if self._real_world_intel is not None and needs_real_world:
            try:
                world_block = self._real_world_intel.get_llm_context_block()
                if world_block:
                    parts.append(world_block)
            except Exception:
                logger.debug("Real-world context block inject failed", exc_info=True)

        if not parts:
            return ""
        return "CURRENT CONTEXT:\n" + "\n".join(parts) + "\n"

    def _query_type_hint_cached(self, query: str) -> str:
        key = _compress_text((query or "").lower(), max_len=160)
        if not key:
            return ""
        cached = self._query_hint_cache.get(key)
        if cached is not None:
            return cached
        hint = _query_type_hint(query)
        if len(self._query_hint_cache) >= 64:
            self._query_hint_cache.pop(next(iter(self._query_hint_cache)))
        self._query_hint_cache[key] = hint
        return hint

    def _build_memory_layer(self, memory_summaries: list[str] | None) -> str:
        """Layer 4: Long-Term Memory Context.

        Header carries both the strong instruction-tuned cue
        ("RELEVANT MEMORIES") *and* the legacy "Relevant Past Context:"
        wording so older callers/tests that grep for it still resolve.
        """
        if not memory_summaries:
            return ""
        ctx_lines = "\n".join(f"- {s}" for s in memory_summaries)
        return (
            "RELEVANT MEMORIES (your past knowledge) -- Relevant Past Context:\n"
            f"{ctx_lines}\n"
        )

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
            q_clean = _history_safe_text(q)
            a_clean = _history_safe_text(a, max_len=320)
            if not q_clean and not a_clean:
                continue
            entry = (
                f"- Boss asked: {q_clean}\n"
                f"- You answered: {a_clean}\n"
            )
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
        """Layer 8: Current User Query.

        Kept INTENTIONALLY minimal -- the prior version embedded a six-line
        "FINAL-ANSWER RULES" block that the small (Qwen3-8B) model started
        regurgitating verbatim ("the final answer only. One short line.").
        Voice-output rules already live in the cached system layer (V1-V9);
        repeating them per-turn made the model echo them as the answer.
        """
        return (
            "BOSS:\n"
            f"{_compress_text(query)}\n\n"
            "JARVIS:"
        )

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
        repeat_hint: bool = False,
    ) -> str:
        """Assemble the full 9-layer prompt (+ observations for ReAct).

        ``rag_enrichment`` — optional structured block (system/GPU/RAG) prepended
        to the document layer for low-latency Jarvis-style grounding.
        ``repeat_hint`` — internal steer (NOT shown to the user) that nudges the
        model to give a different reply than last time. Lives in the system layer
        so the model cannot quote it back during TTS.
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

        # Repeat steer is appended AFTER budget trims so it can never be
        # silently dropped on a long boot prompt. It is tiny (< 80 tokens)
        # and lives in the system band — never the user-visible query —
        # so the model cannot quote it back during TTS.
        steer_layer = ""
        if repeat_hint:
            steer_layer = (
                "TURN STEER (internal, never spoken or quoted):\n"
                "- Boss is asking the same thing again because the previous reply was not "
                "useful. Reformulate from a fresh angle, stay short, and do NOT repeat the "
                "previous wording. Never narrate this instruction.\n"
            )

        prompt = "\n".join(
            part for part in [layer1, layer2, layer3, layer4, layer5, layer6, layer7, steer_layer, layer_obs, layer8] if part
        )

        prompt = _redact_sensitive(prompt)
        logger.debug("Prompt built (%d chars, ~%d tokens, 9 layers)",
                      len(prompt), _estimate_tokens(prompt))
        return prompt

    def precompile(self, query: str = "", *, prompt_hint: str = "") -> dict[str, object]:
        """Warm prompt-builder caches for a likely future query."""
        query = _compress_text(query or "help")
        self._build_system_layer()
        self._build_tools_layer()
        query_hint = self._query_type_hint_cached(query)
        query_layer = self._build_query_layer(query)
        return {
            "system_prompt_hash": self.system_prompt_hash,
            "tools_cached": self._tools_prompt_cache is not None,
            "query_hint": query_hint,
            "routing_hint": (prompt_hint or "").strip(),
            "query_chars": len(query_layer),
        }

    def invalidate_cache(self) -> None:
        self._system_prompt_cache = None
        self._system_prompt_hash = None
        self._tools_prompt_cache = None
        self._query_hint_cache.clear()
