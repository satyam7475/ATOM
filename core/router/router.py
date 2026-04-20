"""
ATOM -- Intelligent Router (Agentic 3-Layer Architecture).

Refactoring:
    - ConfirmationManager extracted to confirmation_manager.py
    - DiagnosticsHandler extracted to diagnostics_handler.py
    - Uses adaptive_personality instead of static personality
    - Integrates ContextFusionEngine for action logging

Architecture:
    Layer 1: Intent Engine (<5ms, regex fast-path for obvious commands)
    Layer 2: Cache (LRU + Jaccard) + Memory (keyword overlap)
    Layer 3: LLM Reasoning Agent (tool-use, ReAct loop, multi-step plans)

Fully offline. Security-gated. Every action goes through SecurityPolicy.
Action execution is delegated to focused sub-modules:
    system_actions, app_actions, media_actions, network_actions,
    utility_actions, file_actions
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import time
from typing import TYPE_CHECKING, Any

from core import adaptive_personality as personality
from core.command_cache import get_command_cache
from core.process_manager import ProcessManager
from core.query_policy import ResponseMode, classify_response_mode, normalize_query
from core.router.confirmation_manager import ConfirmationManager
from core.router.diagnostics_handler import DiagnosticsHandler
from core.security_policy import SecurityPolicy
from . import (
    app_actions,
    file_actions,
    media_actions,
    network_actions,
    system_actions,
    utility_actions,
)

if TYPE_CHECKING:
    from context.context_engine import ContextEngine
    from core.async_event_bus import AsyncEventBus
    from core.cache_engine import CacheEngine
    from core.cognitive_kernel import CognitiveKernel
    from core.intent_engine import IntentEngine
    from core.memory_engine import MemoryEngine
    from core.runtime_watchdog import RuntimeWatchdog
    from core.self_evolution import SelfEvolutionEngine
    from core.state_manager import StateManager
    from core.task_scheduler import TaskScheduler

logger = logging.getLogger("atom.router")

from core.router.conversation_manager import ConversationManager, compress_query

_CLIPBOARD_REF = re.compile(
    r"\b(that\s+error|this\s+error|that\s+code|this\s+code|"
    r"that\s+message|this\s+message|what\s+i\s+copied|"
    r"the\s+clipboard|from\s+clipboard|clipboard\s+content|"
    r"that\s+exception|this\s+exception|that\s+bug|this\s+bug|"
    r"that\s+text|this\s+text)\b", re.I)


class Router:
    """Intelligent Router with agentic 3-layer architecture.

    Priority:
        1. Intent Engine (<5ms, regex fast-path for obvious commands)
        2. Cache + Memory (instant, repeated/similar queries)
        3. LLM Reasoning Agent (tool-use, ReAct loop, 1-4s)

    The LLM is the true brain. IntentEngine is a speed optimization.
    """

    _INFO_INTENTS = frozenset({
        "time", "date", "cpu", "ram", "battery", "disk",
        "system_info", "ip", "wifi", "uptime", "top_processes",
        "resource_report", "resource_trend", "app_history",
        "show_reminders", "self_diagnostic", "system_analyze",
        "self_check", "audio_diagnostics", "behavior_report", "mode_status", "detailed_status",
    })

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        cache: CacheEngine,
        memory: MemoryEngine,
        intent_engine: IntentEngine,
        context_engine: ContextEngine | None = None,
        config: dict | None = None,
        scheduler: TaskScheduler | None = None,
        process_mgr: ProcessManager | None = None,
        evolution: SelfEvolutionEngine | None = None,
        behavior_tracker: Any | None = None,
        brain_mode_manager: Any | None = None,
        assistant_mode_manager: Any | None = None,
        skills_registry: Any | None = None,
        conversation_memory: Any | None = None,
        timeline_memory: Any | None = None,
        security_policy: SecurityPolicy | None = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._cache = cache
        self._memory = memory
        self._intent = intent_engine
        self._context = context_engine
        self._config = config or {}
        self._security = security_policy or SecurityPolicy(self._config)
        self._brain_enabled = bool(self._config.get("brain", {}).get("enabled", False))
        self._brain_mode_mgr = brain_mode_manager
        self._assistant_mode_mgr = assistant_mode_manager
        self._processing_lock = asyncio.Lock()
        self._local_queries = 0
        self._llm_queries = 0
        self._scheduler = scheduler
        self._process_mgr = process_mgr or ProcessManager()
        self._evolution = evolution
        self._behavior_tracker = behavior_tracker
        self._skills = skills_registry
        self._conv_memory = conversation_memory
        self._timeline = timeline_memory

        self._cognitive_kernel: CognitiveKernel | None = None
        self._runtime_watchdog: RuntimeWatchdog | None = None
        self._task_manager: Any = None

        self._system_state_engine: Any = None
        self._session_memory: Any = None
        self._user_memory: Any = None

        # Stream generation counter — incremented on every new streaming call
        # so stale tokens from a cancelled stream are discarded.
        self._cloud_stream_generation: int = 0

        # Adaptive perception profile — updated by perception_adaptive events.
        self._perception_concise: bool = False

        # Phase 2 Adaptive Intelligence Engine (wired via attach_adaptive_engine)
        self._adaptive: Any = None

        # Routine engine (wired via attach_routine_engine, Sprint D4)
        self._routine_engine: Any = None

        # v22: Cloud intelligence components (wired via attach_cloud_intelligence)
        self._gemini_client: Any = None
        self._search_tool: Any = None
        self._confidence_engine: Any = None
        self._decision_engine: Any = None
        self._semantic_cache: Any = None
        self._preference_store: Any = None
        self._security_gateway: Any = None

        # Extracted modules
        self._conv_mgr = ConversationManager(self._conv_memory)
        self._confirmation = ConfirmationManager(self._security)
        self._diagnostics = DiagnosticsHandler(self._config)

        from core.reasoning.action_executor import ActionExecutor
        self._action_executor = ActionExecutor(
            dispatch_fn=self._dispatch_action,
            async_dispatch_fn=self._dispatch_action_async,
            timeline=timeline_memory,
            security=self._security,
        )
        self._code_sandbox = None
        logger.info("ActionExecutor initialized with %d registered tools",
                     self._action_executor.get_stats()["registered_tools"])

    @property
    def action_executor(self):
        """Expose the ActionExecutor for wiring to LocalBrainController."""
        return self._action_executor

    def attach_cognitive_kernel(self, kernel: "CognitiveKernel") -> None:
        """Wire the Cognitive Kernel for intelligent LLM routing."""
        self._cognitive_kernel = kernel
        logger.info("Cognitive Kernel attached to Router")

    def attach_runtime_watchdog(self, watchdog: "RuntimeWatchdog") -> None:
        """Wire RuntimeWatchdog so hot router stages use active budgets."""
        self._runtime_watchdog = watchdog
        logger.info("RuntimeWatchdog attached to Router")

    def attach_task_manager(self, task_manager: Any) -> None:
        """Wire the centralized background task manager."""
        self._task_manager = task_manager
        logger.info("TaskManager attached to Router")

    def attach_adaptive_engine(self, adaptive: Any) -> None:
        """Wire the Phase 2 Adaptive Intelligence Engine."""
        self._adaptive = adaptive
        logger.info("AdaptiveEngine attached to Router")

    def attach_routine_engine(self, routine_engine: Any) -> None:
        """Wire the user-defined routine engine (Sprint D4)."""
        self._routine_engine = routine_engine
        try:
            routine_engine.set_dispatcher(self._routine_step_dispatch)
        except Exception:
            logger.info("routine engine set_dispatcher failed", exc_info=True)
        logger.info(
            "RoutineEngine attached to Router (routines=%d)",
            len(routine_engine.list_routines()) if hasattr(routine_engine, "list_routines") else 0,
        )

    def attach_context_layer(
        self,
        *,
        system_state_engine: Any = None,
        session_memory: Any = None,
        user_memory: Any = None,
    ) -> None:
        """Wire the real-time context layer (Phase 5+6)."""
        self._system_state_engine = system_state_engine
        self._session_memory = session_memory
        self._user_memory = user_memory
        logger.info("Context layer attached to Router (state=%s session=%s user=%s)",
                     system_state_engine is not None,
                     session_memory is not None,
                     user_memory is not None)

    def attach_cloud_intelligence(
        self,
        *,
        gemini_client: Any = None,
        search_tool: Any = None,
        confidence_engine: Any = None,
        decision_engine: Any = None,
        semantic_cache: Any = None,
        preference_store: Any = None,
        security_gateway: Any = None,
    ) -> None:
        """v22: Wire all cloud intelligence components into the Router."""
        self._gemini_client = gemini_client
        self._search_tool = search_tool
        self._confidence_engine = confidence_engine
        self._decision_engine = decision_engine
        self._semantic_cache = semantic_cache
        self._preference_store = preference_store
        self._security_gateway = security_gateway
        logger.info(
            "Router v22 cloud intelligence wired: gemini=%s, search=%s, "
            "confidence=%s, decision=%s, sem_cache=%s, prefs=%s, gateway=%s",
            gemini_client is not None, search_tool is not None,
            confidence_engine is not None, decision_engine is not None,
            semantic_cache is not None, preference_store is not None,
            security_gateway is not None,
        )

    async def _handle_component_failure(
        self,
        source: str,
        exc: Exception,
        *,
        user_message: str,
    ) -> None:
        logger.exception("Router %s failed: %s", source, exc)
        try:
            self._bus.emit_fast("metrics_event", counter="errors_total")
        except Exception:
            logger.debug('Fast bus emit failed', exc_info=True)
        if self._timeline is not None:
            try:
                self._timeline.append_event(
                    "error",
                    {
                        "source": source,
                        "message": str(exc)[:300],
                    },
                )
            except Exception:
                logger.debug('Fast bus emit failed', exc_info=True)
        try:
            self._emit_response(user_message)
        except Exception:
            logger.debug("Router fallback response failed", exc_info=True)
        try:
            await self._state.on_error(source=source)
        except Exception:
            logger.debug("Router recovery hook failed", exc_info=True)

    def _emit_response(self, text: str, **kw) -> None:
        """Emit response through adaptive shaping + output polishing."""
        out = text or ""
        if self._adaptive is not None:
            out, _speech = self._adaptive.process_response(out)
        polished = personality.polish_response(out, source="router")
        self._bus.emit_long("response_ready", text=polished, **kw)

    # Short, warm acks for bare wake calls. JARVIS doesn't say "Yes, Boss?"
    # every single time -- he varies between a calm acknowledgement and a
    # quiet "I'm here" so it sounds present without feeling scripted.
    _BARE_WAKE_ACKS: tuple[str, ...] = (
        "Yes, Boss?",
        "Right here.",
        "I'm here, Boss.",
        "Listening.",
        "Go ahead, Boss.",
        "Yes?",
    )
    _BARE_WAKE_ACK_INDEX: int = 0

    @classmethod
    def _pick_bare_wake_ack(cls) -> str:
        """Return the next short ack in the rotation. We rotate
        deterministically (rather than ``random``) so consecutive wakes
        never repeat the same phrase yet the sequence stays predictable
        for log-trace debugging.
        """
        idx = cls._BARE_WAKE_ACK_INDEX % len(cls._BARE_WAKE_ACKS)
        cls._BARE_WAKE_ACK_INDEX = (cls._BARE_WAKE_ACK_INDEX + 1) % len(cls._BARE_WAKE_ACKS)
        return cls._BARE_WAKE_ACKS[idx]

    @staticmethod
    def _is_bare_wake_utterance(text: str) -> bool:
        """True when ``text`` consists ONLY of wake/direct-address tokens.

        Used to short-circuit the LLM on standalone calls like ``atom``,
        ``hey atom``, ``boss``, ``dear boss`` (the recurring SFSpeech
        mishearing of ``hey boss``), so the user gets a one-word ack
        instead of a 5-second LLM narration.
        """
        if not text:
            return False
        try:
            from voice.listening_modes import WakeWordFilter
        except Exception:
            return False
        normalized = " ".join(str(text).strip().lower().split())
        if not normalized:
            return False
        # Strip a single trailing punctuation char so "atom?" / "boss." count.
        if normalized[-1] in "?.!,":
            normalized = normalized[:-1].strip()
        if not normalized:
            return False
        if normalized in WakeWordFilter.WAKE_PHRASES:
            return True
        if normalized in {p.lower() for p in WakeWordFilter.DIRECT_ADDRESS}:
            return True
        # Word-bounded match: the utterance is JUST the wake/address phrase
        # plus optional "please" / "there" filler. Anything longer is a real
        # command and must keep going through the LLM.
        words = normalized.split()
        if len(words) <= 4:
            for phrase in WakeWordFilter.DIRECT_ADDRESS:
                p = phrase.lower()
                if normalized == p or normalized == f"{p} please":
                    return True
        return False

    # ── LLM output guardrail (hallucinated-action + low-confidence) ─
    _ACTION_PROMISE_PATTERNS = (
        "i'll ", "i will ", "i am going to ", "i'm going to ", "let me ",
        "okay, i'll", "okay, i will", "sure, i'll", "sure, i will",
        "playing ", "opening ", "setting ", "starting ", "launching ",
        "turning on ", "turning off ", "enabling ", "disabling ",
        "sending ", "creating ", "adding ", "deleting ",
    )
    # Verb roots matched against the user query. If the reply promises an
    # action but none of these roots appears (or a close synonym), we treat
    # it as a fabricated action and replace with a clarifier.
    _ACTION_VERBS = {
        "play": ("play", "song", "music", "track", "video"),
        "open": ("open", "launch", "show", "bring up"),
        "start": ("start", "begin", "run", "launch"),
        "set": ("set", "schedule", "remind", "timer", "alarm"),
        "send": ("send", "email", "message", "text", "ping"),
        "create": ("create", "make", "new"),
        "add": ("add", "append", "insert"),
        "delete": ("delete", "remove", "clear", "trash"),
        "turn": ("turn", "toggle", "enable", "disable"),
        "close": ("close", "quit", "kill", "stop"),
    }
    # Strip these leading formatting wrappers before vetting. Small local
    # models sometimes emit `The answer is "..."` which hides a fabricated
    # action promise inside the quoted content.
    _REPLY_WRAPPER_PREFIXES = (
        "the answer is ", "my answer is ", "the answer: ",
        "answer: ", "response: ", "reply: ", "final answer: ",
        "here is the answer: ", "here's the answer: ",
    )
    # WH / definition style queries that should never legitimately produce
    # an action promise. If the query starts with one of these AND the reply
    # promises an action, it's a hallucination regardless of verb-match.
    _WH_QUERY_PREFIXES = (
        "what ", "what's ", "whats ", "who ", "who's ", "whos ",
        "when ", "where ", "why ", "how ", "how's ", "hows ",
        "which ", "whose ", "define ", "explain ", "tell me about ",
        "meaning of ", "describe ",
    )
    _LOW_CONFIDENCE_THRESHOLD: float = 0.5
    _CLARIFIER_TEMPLATES = (
        "I didn't quite catch that, Boss — what did you mean exactly?",
        "I'm not sure I heard you right, Boss. Could you rephrase that?",
        "That didn't come through clearly, Boss. Mind repeating it?",
    )

    @classmethod
    def _unwrap_reply(cls, reply: str) -> str:
        """Strip `The answer is "..."` / `Answer: ...` style wrappers so the
        inner content can be action-vetted. Returns the unwrapped text with
        surrounding quotes trimmed; falls back to the original string when
        no wrapper is found."""
        stripped = (reply or "").strip()
        lowered = stripped.lower()
        for prefix in cls._REPLY_WRAPPER_PREFIXES:
            if lowered.startswith(prefix):
                inner = stripped[len(prefix):].strip()
                # Trim matching wrapping quotes (straight, curly, or back).
                for quote_pair in (('"', '"'), ("'", "'"),
                                   ("\u201c", "\u201d"), ("\u2018", "\u2019"),
                                   ("`", "`")):
                    lq, rq = quote_pair
                    if inner.startswith(lq) and inner.endswith(rq) and len(inner) > 1:
                        inner = inner[1:-1].strip()
                        break
                # Strip a trailing "." / ".." dangling from the wrapper.
                while inner.endswith(".."):
                    inner = inner[:-1]
                return inner
        return stripped

    @classmethod
    def _reply_contains_action_promise(cls, reply_lower: str) -> str:
        """Return the first action-promise pattern found anywhere in the
        first 160 characters of the (lowercased) reply, or empty string.

        Using a substring scan over an inspection window (vs ``startswith``)
        catches wrappers like `The answer is "Okay, I'll play..."` and
        padded prefaces like `Sure thing, I'll ...`.
        """
        window = reply_lower[:160]
        for pattern in cls._ACTION_PROMISE_PATTERNS:
            if pattern in window:
                return pattern
        return ""

    @classmethod
    def _query_has_matching_verb(cls, query_lower: str) -> bool:
        for _verb, synonyms in cls._ACTION_VERBS.items():
            if any(syn in query_lower for syn in synonyms):
                return True
        return False

    @classmethod
    def _query_is_wh(cls, query_lower: str) -> bool:
        return any(query_lower.startswith(p) for p in cls._WH_QUERY_PREFIXES)

    def vet_llm_response(self, query: str, reply: str,
                         confidence: float = 0.6) -> str:
        """Guardrail over LLM output before TTS.

        Replaces the reply with a short clarification question when:
          1. The reply promises an action ("I'll play...", "Opening...")
             AND either (a) the query is a WH/definition question (no action
             should ever be legitimate) or (b) no matching verb/noun root
             appears in the query. Wrappers such as `The answer is "..."`
             are unwrapped before the scan so quoted action-promises cannot
             slip through.
          2. Confidence is below _LOW_CONFIDENCE_THRESHOLD and the reply
             looks like a confident factual / action statement.

        Returns the possibly-rewritten reply. Safe to call from any thread;
        purely synchronous text transform.
        """
        q = (query or "").lower().strip()
        r_raw = (reply or "").strip()
        if not r_raw or not q:
            return reply

        r_unwrapped = self._unwrap_reply(r_raw)
        lower_reply = r_unwrapped.lower()

        pattern_hit = self._reply_contains_action_promise(lower_reply)
        if pattern_hit:
            is_wh_query = self._query_is_wh(q)
            matched_verb = self._query_has_matching_verb(q)
            if is_wh_query or not matched_verb:
                import random
                clarifier = random.choice(self._CLARIFIER_TEMPLATES)
                logger.warning(
                    "Router guardrail: action-promise '%s' in reply without "
                    "matching %s (query='%s', reply='%s') — emitting clarifier",
                    pattern_hit.strip(),
                    "WH-query context" if is_wh_query else "verb",
                    q[:60], r_raw[:120],
                )
                return clarifier

        if confidence < self._LOW_CONFIDENCE_THRESHOLD and len(r_raw) > 15:
            ack = ""
            try:
                if self._conv_mgr is not None:
                    ack = self._conv_mgr.smart_ack(query) or ""
            except Exception:
                ack = ""
            clarifier = (
                (ack + " " if ack else "")
                + "I'm not fully confident I got that right — what did you mean exactly, Boss?"
            ).strip()
            logger.info(
                "Router guardrail: low LLM confidence (%.2f) — asking clarification",
                confidence,
            )
            return clarifier

        return reply

    def _emit_thinking_ack(self, text: str) -> None:
        polished = personality.polish_response(text or "", source="thinking_ack")
        self._bus.emit_long("thinking_ack", text=polished)

    @staticmethod
    def _should_emit_thinking_ack(clean_text: str, query_plan: Any | None) -> bool:
        if classify_response_mode(clean_text) is ResponseMode.SHORT:
            return False
        if query_plan is None:
            return True
        path = getattr(getattr(query_plan, "path", None), "value", getattr(query_plan, "path", ""))
        path_name = str(path or "").strip().lower()
        if path_name in {"direct", "cache", "quick"}:
            return False
        budget_tier = str(getattr(query_plan, "budget_tier", "") or "").strip().lower()
        requested_tier = str(getattr(query_plan, "requested_tier", "") or "").strip().lower()
        if budget_tier in {"command", "info", "simple"} or requested_tier in {"command", "info", "simple"}:
            return False
        return True

    async def on_speech(self, text: str, **_kw) -> None:
        async with self._processing_lock:
            try:
                await self._route(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._handle_component_failure(
                    "router.on_speech",
                    exc,
                    user_message=(
                        "Something went wrong while routing that, Boss. "
                        "Try again in a moment."
                    ),
                )

    async def _route(self, text: str) -> None:
        from core.state_manager import AtomState
        from core.fast_path import LatencyBudget

        t0 = time.perf_counter()
        _budget = LatencyBudget(label=text[:40])
        text = text.strip()
        if not text:
            return

        text, was_sanitized = self._security.sanitize_input(text)
        if was_sanitized:
            logger.info("Input sanitized (injection chars or length capped)")
        if not text:
            return

        raw_text = compress_query(text)
        if not raw_text:
            return
        normalized_text = normalize_query(raw_text)
        if normalized_text and normalized_text != raw_text.lower().strip():
            logger.info("Input normalized: '%s' -> '%s'", raw_text[:80], normalized_text[:80])
        clean_text = compress_query(normalized_text or raw_text)
        if not clean_text:
            return

        if self._timeline is not None:
            try:
                self._timeline.append_event(
                    "user_query",
                    {"text": raw_text[:2000], "source": "router"},
                )
            except Exception:
                logger.debug('Fast path step failed', exc_info=True)

        # ── Bare wake / direct-address short-circuit ──────────────────
        # Treat utterances that contain ONLY a wake phrase or direct-address
        # token (e.g. "atom", "hey atom", "boss", "dear boss" -- the latter
        # being SFSpeech's recurring mishearing of "hey boss") as a soft
        # "are you there?" ping. Without this they fall to the LLM where
        # "dear boss" becomes a 7-second narration about how the user is
        # greeting you. We answer with a one-word ack and stay LISTENING
        # so the user's real follow-up is captured cleanly.
        if self._is_bare_wake_utterance(clean_text):
            logger.info(
                "Bare wake/direct-address — short-circuiting LLM ('%s')",
                clean_text[:60],
            )
            self._emit_response(self._pick_bare_wake_ack())
            return

        if self._conv_memory is not None:
            self._conv_memory.on_new_user_query(raw_text)

        # ── 1. Pronoun resolution (conversational continuity) ────────
        resolved = self._conv_mgr.resolve_pronouns(clean_text)
        if resolved != clean_text:
            logger.info("Pronoun resolved: '%s' -> '%s'", clean_text, resolved)
            clean_text = resolved
            raw_text = resolved

        # ── 2b. System context + reference resolution ─────────────────
        _system_ctx: dict = {}
        if self._system_state_engine is not None:
            try:
                _system_ctx = self._system_state_engine.get_context()
                refs = self._system_state_engine.resolve_reference(clean_text)
                if refs:
                    _this_app = refs.get("this_app", "")
                    _prev_app = refs.get("previous_app", "")
                    _media_app = refs.get("media_app", "")
                    _lower = clean_text.lower()
                    if "this" in _lower or "close this" in _lower or "this app" in _lower:
                        if _this_app:
                            clean_text = clean_text.replace("this app", _this_app).replace("this", _this_app)
                            raw_text = clean_text
                            logger.info("Reference resolved: 'this' -> '%s'", _this_app)
                    if "switch back" in _lower and _prev_app:
                        clean_text = f"switch to {_prev_app}"
                        raw_text = clean_text
                        logger.info("Reference resolved: 'switch back' -> '%s'", _prev_app)
                    if ("pause" in _lower or "stop music" in _lower) and _media_app:
                        if _media_app not in clean_text:
                            clean_text = f"{clean_text} ({_media_app})"
                            raw_text = clean_text
            except Exception:
                logger.debug("System context injection failed", exc_info=True)

        if self._user_memory is not None:
            try:
                self._user_memory.track_app_usage(_system_ctx.get("active_app", ""))
            except Exception:
                logger.debug('User memory inject failed', exc_info=True)

        # ── 3. Clipboard injection (implicit context) ────────────────
        clipboard_injected = False
        if _CLIPBOARD_REF.search(clean_text) and self._context is not None:
            try:
                bundle = self._context.get_bundle()
                clip = (bundle or {}).get("clipboard", "")
                if clip and len(clip) < 1000:
                    clip, _ = self._security.sanitize_input(clip)
                    raw_text = f"{raw_text}\n\nReferenced content: {clip}"
                    clipboard_injected = True
                    logger.info("Clipboard injected (%d chars)", len(clip))
            except Exception:
                logger.debug("Clipboard injection failed", exc_info=True)

        cmd_cache = get_command_cache()
        cached = cmd_cache.get(clean_text)
        intent_timed_out = False
        if cached is not None:
            result = cached
            classify_ms = 0.0
            logger.info("Router: '%s' -> %s (CACHED, 0ms)",
                         clean_text[:60], result.intent)
        else:
            if self._runtime_watchdog is not None:
                from core.intent_engine import IntentResult

                classify_result = await self._runtime_watchdog.run_sync(
                    "intent_engine",
                    self._intent.classify,
                    clean_text,
                    default=IntentResult("fallback"),
                    metadata={"query": clean_text[:80]},
                )
                result = classify_result.value
                classify_ms = classify_result.elapsed_ms
                intent_timed_out = classify_result.timed_out
            else:
                result = self._intent.classify(clean_text)
                classify_ms = (time.perf_counter() - t0) * 1000
                intent_timed_out = False
            if not intent_timed_out:
                cmd_cache.put(clean_text, result)
            used_intent_cache = False
            if result.intent in self._INFO_INTENTS:
                intent_cached = cmd_cache.get("info:" + result.intent)
                if intent_cached is not None:
                    result = intent_cached
                    used_intent_cache = True
                    logger.info("Router: '%s' -> %s (INTENT CACHED, 0ms)",
                                 clean_text[:60], result.intent)
                else:
                    cmd_cache.put_intent_key("info:" + result.intent, result)
            if classify_ms > 0 and not used_intent_cache:
                logger.info("Router: '%s' -> %s (%.1fms)",
                             clean_text[:60], result.intent, classify_ms)

        _skill_chain: list[str] = []
        if result.intent == "fallback" and self._skills is not None:
            match = self._skills.try_expand_full(clean_text)
            if match is not None:
                logger.info(
                    "Skill '%s': '%s' -> '%s'%s",
                    match.skill_id, clean_text[:80], match.primary[:80],
                    f" +{len(match.chain)} chain" if match.chain else "",
                )
                clean_text = match.primary
                raw_text = match.primary
                _skill_chain = list(match.chain)
                cached = cmd_cache.get(clean_text)
                if cached is not None:
                    result = cached
                    classify_ms = 0.0
                    logger.info("Router: '%s' -> %s (CACHED, 0ms)",
                                clean_text[:60], result.intent)
                else:
                    if self._runtime_watchdog is not None:
                        from core.intent_engine import IntentResult

                        classify_result = await self._runtime_watchdog.run_sync(
                            "intent_engine",
                            self._intent.classify,
                            clean_text,
                            default=IntentResult("fallback"),
                            metadata={"query": clean_text[:80], "source": "skill_expand"},
                        )
                        result = classify_result.value
                        classify_ms = classify_result.elapsed_ms
                        intent_timed_out = classify_result.timed_out
                    else:
                        result = self._intent.classify(clean_text)
                        classify_ms = (time.perf_counter() - t0) * 1000
                        intent_timed_out = False
                    if not intent_timed_out:
                        cmd_cache.put(clean_text, result)
                    logger.info("Router: '%s' -> %s (skill expanded, %.1fms)",
                                clean_text[:60], result.intent, classify_ms)

        self._bus.emit_fast("intent_classified",
                            intent=result.intent, ms=classify_ms,
                            text=clean_text,
                            action_args=result.action_args)

        _budget.warn_if_slow("intent_classify")

        try:
            from core.observability.per_module_latency import get_latency_board

            b = get_latency_board()
            route_ms = (time.perf_counter() - t0) * 1000
            if classify_ms > 0:
                b.record_module_call(
                    "intent_engine",
                    float(classify_ms),
                    error=bool(intent_timed_out),
                )
            extra_router = max(0.0, route_ms - float(classify_ms or 0.0))
            if extra_router > 0.05:
                b.record_module_call("router", extra_router, error=False)
        except Exception:
            logger.debug('Fast bus emit failed', exc_info=True)

        if self._conv_memory is not None:
            self._conv_memory.set_classified(result.intent, result.action)

        _COGNITIVE_INTENTS = frozenset({
            "goal_create", "goal_show", "goal_progress", "goal_decompose",
            "goal_log_progress", "goal_complete_step", "goal_pause",
            "goal_resume", "goal_abandon",
            "prediction", "mode_switch",
            "cognitive_behavior_report", "scheduling_advice",
            "brain_remember", "brain_recall", "brain_preferences",
            "self_optimize",
        })
        if result.intent in _COGNITIVE_INTENTS:
            self._local_queries += 1
            self._bus.emit_fast("metrics_event", counter="local_routed_queries")
            return

        # ── Entity tracking (conversational continuity) ──────────────
        self._conv_mgr.track_entity(clean_text)

        is_local = result.intent not in ("fallback", "screen_analyze")

        if is_local:
            if result.intent in ("confirm", "deny"):
                await self._handle_confirmation(result.intent)
                return

            if result.intent == "go_silent":
                self._emit_response(
                    result.response or personality.silent_response(),
                    is_sleep=True,
                )
                return

            if result.intent == "exit":
                self._emit_response(
                    result.response or personality.exit_response(),
                    is_exit=True,
                )
                return

            if result.action:
                self._local_queries += 1
                self._bus.emit_fast("metrics_event", counter="local_routed_queries")
                if self._confirmation.requires_confirmation(result):
                    prompt = self._confirmation.set_pending_action(result)
                    self._emit_response(prompt)
                    return
                confidence = getattr(result, "confidence", 1.0)
                if confidence < 0.7 and result.action not in self._INFO_INTENTS:
                    logger.info(
                        "Low confidence (%.2f) for '%s' — asking confirmation",
                        confidence, result.action,
                    )
                    prompt = self._confirmation.set_pending_action(result)
                    action_label = result.action.replace("_", " ")
                    self._emit_response(
                        f"Did you mean {action_label}, Boss? Say yes to confirm.",
                    )
                    return
                await self._execute_action(result)
                if _skill_chain:
                    await self._run_skill_chain(_skill_chain)
                return

            if result.response:
                self._local_queries += 1
                self._bus.emit_fast("metrics_event", counter="local_routed_queries")
                if result.intent == "status":
                    status_text = self._status_with_usage(
                        result.response, query=clean_text,
                    )
                    # Defense-in-depth: run canned status text through the
                    # same vetter the LLM output uses. A future edit that
                    # adds an action phrase to status replies would still
                    # be caught.
                    try:
                        status_text = self.vet_llm_response(
                            clean_text, status_text, confidence=0.95,
                        )
                    except Exception:
                        logger.debug("status vet failed", exc_info=True)
                    self._emit_response(status_text)
                    return
                self._emit_response(result.response)
                return
        else:
            await self._state.transition(AtomState.THINKING)

            if result.intent == "screen_analyze":
                self._llm_queries += 1
                self._bus.emit_fast("metrics_event", counter="screen_analyze_queries")
                await self._handle_screen_analyze(result.action_args or {})
                return

            self._llm_queries += 1
            self._bus.emit_fast("metrics_event", counter="llm_routed_queries")
            await self._handle_llm_fallback(raw_text, clean_text,
                                            clipboard_injected=clipboard_injected)
            return

    # ── Confirmation flow (delegated to ConfirmationManager) ───────────

    async def _handle_confirmation(self, confirm_intent: str) -> None:
        """Resolve pending confirmations via the extracted ConfirmationManager."""
        outcome = self._confirmation.handle(confirm_intent)

        if outcome.response:
            self._emit_response(outcome.response)

        if outcome.action_result is not None:
            await self._execute_action(outcome.action_result)
        elif outcome.tool_call is not None:
            tool_call = outcome.tool_call
            allowed, reason = self._security.allow_action(
                tool_call.name, dict(tool_call.arguments),
            )
            if not allowed:
                if reason.startswith("ESCALATABLE|"):
                    human_reason = reason.split("|", 1)[1]
                    logger.info("Security ESCALATABLE for tool '%s': %s", tool_call.name, human_reason)
                    prompt = self._confirmation.set_pending_escalation(tool_call.name, tool_call)
                    self._bus.emit_long("response_ready", text=prompt)
                    return
                logger.warning("Security BLOCKED confirmed tool '%s': %s",
                               tool_call.name, reason)
                self._bus.emit_long(
                    "response_ready",
                    text=personality.polish_response(
                        f"Sorry Boss, that action is blocked. {reason}",
                        source="security_block",
                    ),
                )
                return
            try:
                result = self._dispatch_action(
                    tool_call.name, dict(tool_call.arguments),
                )
                response = result or personality.action_done(tool_call.name)
                self._emit_response(response)
            except Exception as exc:
                logger.error("Confirmed tool execution failed: %s", exc)
                self._emit_response(personality.error_response(tool_call.name))



    # ── Action dispatcher ───────────────────────────────────────────────

    async def _execute_action(self, result) -> None:
        from core.execution.behavior_monitor import strip_signing_keys
        from core.security.action_signing import merge_signed_args

        args = result.action_args or {}
        args = merge_signed_args(self._security, result.action, args)

        allowed, reason = self._security.allow_action(result.action, args)
        if not allowed:
            if reason.startswith("ESCALATABLE|"):
                human_reason = reason.split("|", 1)[1]
                logger.info("Security ESCALATABLE for '%s': %s", result.action, human_reason)
                prompt = self._confirmation.set_pending_escalation(result.action, result)
                self._emit_response(prompt)
                return
            logger.warning("Security BLOCKED action '%s': %s", result.action, reason)
            self._emit_response(f"Sorry Boss, that action is blocked. {reason}")
            return

        self._security.audit_log(
            result.action,
            f"args={strip_signing_keys(args)}" if args else "",
        )

        response_text = result.response or personality.action_done(result.action)
        dispatch_args = strip_signing_keys(args)

        if result.action in self._FIRE_AND_FORGET_ACTIONS:
            self._emit_response(response_text)
            try:
                self._dispatch_action(result.action, dispatch_args)
            except Exception as exc:
                logger.error("Background action failed: %s", exc)
            self._emit_chain_suggestion(result.action, dispatch_args)
            return

        if result.action in self._SLOW_ACTIONS:
            self._emit_thinking_ack("On it, Boss.")

        try:
            response = self._dispatch_action(result.action, dispatch_args)
            if response is not None:
                self._emit_response(response)
                self._emit_chain_suggestion(result.action, dispatch_args)
                return
        except Exception as exc:
            logger.error("Action failed: %s", exc)
            self._emit_response(personality.error_response(result.action))
            return

        self._emit_response(response_text)
        self._emit_chain_suggestion(result.action, dispatch_args)

    async def _run_skill_chain(self, chain: list[str]) -> None:
        """Execute remaining steps of a multi-step skill sequentially."""
        import asyncio
        from core.execution.behavior_monitor import strip_signing_keys
        from core.security.action_signing import merge_signed_args

        for step_text in chain:
            await asyncio.sleep(0.8)
            step_result = self._intent.classify(step_text)
            if step_result.action:
                sargs = merge_signed_args(
                    self._security,
                    step_result.action,
                    step_result.action_args or {},
                )
                allowed, _ = self._security.allow_action(step_result.action, sargs)
                if allowed:
                    logger.info("Skill chain step: '%s' -> %s", step_text[:60], step_result.intent)
                    try:
                        self._dispatch_action(
                            step_result.action,
                            strip_signing_keys(sargs),
                        )
                    except Exception as exc:
                        logger.warning("Skill chain step failed: %s", exc)

    def _emit_chain_suggestion(self, action: str, args: dict) -> None:
        suggestion = self._conv_mgr.get_chain_suggestion(action, args)
        if suggestion:
            self._bus.emit_fast("intent_chain_suggestion",
                                suggestion=suggestion, action=action)

    _FIRE_AND_FORGET_ACTIONS = frozenset({
        "open_app", "play_youtube", "search", "lock_screen", "screenshot",
        "minimize_window", "maximize_window", "switch_window",
        "flush_dns", "open_url",
    })

    _SLOW_ACTIONS = frozenset({
        "list_apps", "resource_report", "resource_trend",
        "system_analyze", "research_topic", "self_check",
        "self_diagnostic", "behavior_report",
        "spotlight_search", "smart_find_file",
        "whats_on_my_plate",
    })

    def _dispatch_action(self, action: str, args: dict) -> str | None:
        """Route action to the appropriate handler module.

        Uses a dispatch table for O(1) lookup instead of long if/elif chains.
        Returns the response text, or None to use default response.
        """
        if hasattr(self, '_security') and hasattr(self._security, 'fortress_gate'):
            fg_ok, fg_reason = self._security.fortress_gate(action)
            if not fg_ok:
                logger.warning("fortress_gate denied action: %s — %s", action, fg_reason)
                return fg_reason

        handler = self._ACTION_DISPATCH.get(action)
        if handler is not None:
            return handler(self, action, args)
        method_name = self._LATE_DISPATCH.get(action)
        if method_name is not None:
            return getattr(self, method_name)(action, args)
        return None

    async def _dispatch_action_async(self, action: str, args: dict) -> str | None:
        """Async wrapper that offloads slow/blocking actions to a thread.

        Fast actions run directly in the event loop.
        Slow actions (file listing, system analysis, etc.) run in executor.
        """
        if action in self._SLOW_ACTIONS:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._dispatch_action, action, args
            )
        return self._dispatch_action(action, args)

    # ── Action handlers (called from dispatch table) ─────────────────

    def _do_open_app(self, _action: str, args: dict) -> str:
        app_actions.open_app(args.get("exe", ""), args.get("args", []))
        return personality.action_done("open_app", args.get("name", "app"))

    def _do_close_app(self, _action: str, args: dict) -> str:
        proc_name = args.get("process", "")
        app_actions.close_app(proc_name)
        return personality.action_done("close_app", args.get("name", proc_name))

    def _do_list_apps(self, _action: str, _args: dict) -> str:
        return app_actions.list_installed_apps_cached()

    def _do_search(self, _action: str, args: dict) -> str:
        network_actions.web_search(args.get("url", ""))
        return personality.action_done("search")

    def _do_spotlight_search(self, _action: str, args: dict) -> str:
        import sys

        if sys.platform != "darwin":
            return "Spotlight search only runs on macOS, Boss."
        from core.macos.spotlight_engine import SpotlightEngine

        q = (args.get("query") or "").strip()
        if not q:
            return "What should I search for on your Mac, Boss?"
        try:
            limit = int(args.get("limit", 10))
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(limit, 50))

        hits = SpotlightEngine().search(q, limit=limit, timeout=15.0)
        if not hits:
            return f"No Spotlight results for «{q[:120]}», Boss."

        paths = [h.get("path", "") for h in hits if h.get("path")]
        body = "\n".join(paths)
        if len(body) > 4000:
            body = body[:4000] + "\n…(truncated)"
        intro = personality.action_done("spotlight_search", q[:80])
        return f"{intro}\n{body}"

    def _do_smart_find_file(self, _action: str, args: dict) -> str:
        import sys

        if sys.platform != "darwin":
            return "File search only runs on macOS, Boss."
        q = (args.get("query") or "").strip()
        if not q:
            return "What should I look for, Boss?"
        try:
            limit = int(args.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 20))
        try:
            from core.proactive.file_finder import find_files_for_text
            summary, paths = find_files_for_text(q, limit=limit)
        except Exception as exc:
            logger.info("smart_find_file failed", exc_info=True)
            return f"File search hit an error, Boss. ({exc.__class__.__name__})"
        if paths:
            extras = paths[1:4]
            if extras:
                nice = "\n".join(f"  • {p}" for p in extras)
                return f"{summary}\n{nice}"
        return summary

    def _do_play_youtube(self, _action: str, args: dict) -> str:
        query = args.get("query", "music")
        media_actions.play_youtube(query)
        self._bus.emit_fast("media_started")
        return personality.action_done("play_youtube", query)

    def _do_stop_music(self, _action: str, _args: dict) -> str:
        media_actions.send_media_play_pause()
        return personality.action_done("stop_music")

    def _do_set_volume(self, _action: str, args: dict) -> str:
        pct = int(args.get("percent", 50))
        media_actions.set_system_volume_percent(pct)
        return personality.action_done("set_volume", str(pct))

    def _do_mute(self, _action: str, _args: dict) -> str:
        media_actions.send_mute_toggle()
        return personality.action_done("mute")

    def _do_unmute(self, _action: str, _args: dict) -> str:
        media_actions.send_mute_toggle()
        return personality.action_done("unmute")

    def _do_create_folder(self, _action: str, args: dict) -> str:
        created = file_actions.create_folder(args.get("name", "").strip(),
                                             args.get("path", "").strip())
        return personality.action_done("create_folder", f"Folder created at {created}")

    def _do_move_path(self, _action: str, args: dict) -> str:
        moved = file_actions.move_path(args.get("src", "").strip(),
                                       args.get("dst", "").strip())
        return personality.action_done("move_path", f"Moved to {moved}")

    def _do_copy_path(self, _action: str, args: dict) -> str:
        copied = file_actions.copy_path(args.get("src", "").strip(),
                                        args.get("dst", "").strip())
        return personality.action_done("copy_path", f"Copied to {copied}")

    def _do_lock_screen(self, _action: str, _args: dict) -> str:
        system_actions.lock_screen()
        return personality.action_done("lock_screen")

    def _do_screenshot(self, _action: str, _args: dict) -> str:
        system_actions.take_screenshot()
        return personality.action_done("screenshot")

    def _do_set_brightness(self, _action: str, args: dict) -> str:
        actual = system_actions.set_brightness(args.get("percent"), args.get("delta"))
        return personality.action_done("set_brightness", str(actual))

    def _do_shutdown_pc(self, _action: str, _args: dict) -> str:
        system_actions.shutdown_pc()
        return personality.action_done("shutdown_pc", "shutdown in 30 seconds")

    def _do_restart_pc(self, _action: str, _args: dict) -> str:
        system_actions.restart_pc()
        return personality.action_done("restart_pc", "restart in 30 seconds")

    def _do_logoff(self, _action: str, _args: dict) -> str:
        system_actions.logoff()
        return personality.action_done("logoff", "logging off")

    def _do_sleep_pc(self, _action: str, _args: dict) -> str:
        system_actions.sleep_pc()
        return personality.action_done("sleep_pc", "sleep")

    def _do_empty_recycle_bin(self, _action: str, _args: dict) -> str:
        system_actions.empty_recycle_bin()
        return personality.action_done("empty_recycle_bin")

    def _do_flush_dns(self, _action: str, _args: dict) -> str:
        system_actions.flush_dns()
        return personality.action_done("flush_dns")

    def _do_minimize_window(self, _action: str, _args: dict) -> str:
        utility_actions.minimize_active_window()
        return personality.action_done("minimize_window")

    def _do_maximize_window(self, _action: str, _args: dict) -> str:
        utility_actions.maximize_active_window()
        return personality.action_done("maximize_window")

    def _do_switch_window(self, _action: str, _args: dict) -> str:
        utility_actions.switch_active_window()
        return personality.action_done("switch_window")

    def _do_timer(self, _action: str, args: dict) -> str:
        seconds = int(args.get("seconds", 30))
        label = args.get("label", f"{seconds}s")
        asyncio.create_task(utility_actions.run_timer(seconds, label, self._bus))
        return personality.action_done("timer", label)

    def _do_read_clipboard(self, _action: str, _args: dict) -> str:
        clip_text = utility_actions.read_clipboard_text()
        if clip_text:
            from context.privacy_filter import redact as _redact
            return f"{personality.action_done('read_clipboard')} {_redact(clip_text)}"
        return "Your clipboard is empty, boss."

    def _do_open_url(self, _action: str, args: dict) -> str:
        network_actions.open_url(args.get("url", ""))
        return personality.action_done("open_url")

    def _do_weather(self, _action: str, _args: dict) -> str:
        feats = self._config.get("features") or {}
        if not feats.get("online_weather", False):
            return (
                "Online weather is disabled for offline ATOM, Boss. "
                "Set features.online_weather to true if you want wttr.in."
            )
        weather_data = network_actions.get_weather()
        if weather_data:
            return f"Current weather: {weather_data}"
        network_actions.open_weather_fallback()
        return "Opening weather info in browser, boss."

    def _do_wifi_status(self, _action: str, _args: dict) -> str:
        return network_actions.get_wifi_status()

    # ── AI OS action handlers ─────────────────────────────────────────

    def _do_set_reminder(self, _action: str, args: dict) -> str:
        if self._scheduler is None:
            return "Reminder system is not active right now, Boss."
        label = args.get("label", "something")
        delay = int(args.get("delay_seconds", 300))
        task = self._scheduler.add_reminder(label, delay)
        return f"Got it, Boss. I'll remind you to {label} in {task.human_due()}."

    def _do_show_reminders(self, _action: str, _args: dict) -> str:
        if self._scheduler is None:
            return "Reminder system is not active."
        return self._scheduler.format_pending()

    def _do_whats_on_my_plate(self, _action: str, _args: dict) -> str:
        try:
            from core.proactive.whats_on_plate import generate_plate_summary_sync
            feats = (self._config or {}).get("morning_briefing") or {}
            timeout_s = float(feats.get("calendar_timeout_s", 3.0))
            return generate_plate_summary_sync(
                task_scheduler=self._scheduler,
                calendar_timeout_s=timeout_s,
            )
        except Exception as exc:
            logger.info("whats_on_my_plate failed", exc_info=True)
            return (
                "I couldn't pull your schedule right now, Boss. "
                f"({exc.__class__.__name__})"
            )

    def _do_cancel_reminders(self, _action: str, _args: dict) -> str:
        if self._scheduler is None:
            return "Reminder system is not active."
        count = self._scheduler.cancel_all()
        if count > 0:
            return f"Cancelled {count} reminder{'s' if count > 1 else ''}, Boss."
        return "No pending reminders to cancel."

    def _do_kill_process(self, _action: str, args: dict) -> str:
        name = args.get("name", "")
        success, msg = self._process_mgr.kill_process(name)
        return msg

    def _do_resource_report(self, _action: str, _args: dict) -> str:
        return self._process_mgr.format_resource_summary()

    def _do_resource_trend(self, _action: str, _args: dict) -> str:
        return self._process_mgr.get_resource_trend()

    def _do_app_history(self, _action: str, _args: dict) -> str:
        return self._process_mgr.format_app_history()

    def _do_research_topic(self, _action: str, args: dict) -> str:
        feats = self._config.get("features") or {}
        if not feats.get("web_research", False):
            return (
                "Web research is off for offline ATOM, Boss. "
                "Set features.web_research to true in settings if you want DuckDuckGo lookup."
            )
        topic = args.get("topic", "")
        if not topic:
            return "What would you like me to research, Boss?"
        from core.web_researcher import research_topic
        return research_topic(topic)

    def _do_behavior_report(self, _action: str, _args: dict) -> str:
        return self._diagnostics.behavior_report()

    def _do_self_diagnostic(self, _action: str, _args: dict) -> str:
        return self._diagnostics.self_diagnostic()

    # ── Desktop control actions ────────────────────────────────────

    def _do_scroll_down(self, _action: str, args: dict) -> str:
        from core.desktop_control import scroll_down
        return scroll_down(args.get("clicks", 5))

    def _do_scroll_up(self, _action: str, args: dict) -> str:
        from core.desktop_control import scroll_up
        return scroll_up(args.get("clicks", 5))

    def _do_click_screen(self, _action: str, args: dict) -> str:
        from core.desktop_control import click_center, double_click_center
        if args.get("double"):
            return double_click_center()
        return click_center()

    def _do_press_key(self, _action: str, args: dict) -> str:
        from core.desktop_control import press_key
        return press_key(args.get("key", "enter"))

    def _do_go_back(self, _action: str, _args: dict) -> str:
        from core.desktop_control import hotkey_combo
        return hotkey_combo("alt+left")

    def _do_hotkey_combo(self, _action: str, args: dict) -> str:
        from core.desktop_control import hotkey_combo
        return hotkey_combo(args.get("combo", ""))

    def _do_type_text(self, _action: str, args: dict) -> str:
        from core.desktop_control import type_text
        return type_text(args.get("text", ""))

    def _do_system_analyze(self, _action: str, _args: dict) -> str:
        return self._process_mgr.get_full_system_report()

    # ── ATOM self-check diagnostics (delegated) ─────────────────────

    def configure_diagnostics(
        self,
        *,
        stt=None,
        tts=None,
        metrics=None,
        local_brain=None,
        health_monitor=None,
        state_snapshot_provider=None,
        report_publisher=None,
        audio_intel=None,
    ) -> None:
        self._diagnostics.configure(
            stt=stt, tts=tts, metrics=metrics,
            local_brain=local_brain, health_monitor=health_monitor,
            evolution=self._evolution,
            behavior_tracker=self._behavior_tracker,
            state_snapshot_provider=state_snapshot_provider,
            report_publisher=report_publisher,
            audio_intel=audio_intel,
        )

    def _do_self_check(self, _action: str, _args: dict) -> str:
        return self._diagnostics.self_check()

    def _do_audio_diagnostics(self, _action: str, _args: dict) -> str:
        return self._diagnostics.audio_diagnostics()

    def _do_mode_status(self, _action: str, _args: dict) -> str:
        return self._diagnostics.mode_status()

    def _do_detailed_status(self, _action: str, _args: dict) -> str:
        return self._diagnostics.detailed_status()

    def _do_set_performance_mode(self, _action: str, args: dict) -> str:
        mode = (args.get("mode") or "optimal").strip().lower().replace("-", "_")
        aliases = {
            "full": "full_performance",
            "brain": "full_performance",
            "lite": "optimal",
            "ultra_lite": "optimal",
            "optimal": "optimal",
            "full_performance": "full_performance",
            "auto": "auto",
        }
        canonical = aliases.get(mode)
        if canonical is None:
            return (
                f"Unknown mode '{mode}'. Available: optimal, full performance, or auto."
            )
        self._bus.emit_long("set_performance_mode", mode=canonical)
        return ""

    def _do_set_brain_profile(self, _action: str, args: dict) -> str:
        mgr = getattr(self, "_brain_mode_mgr", None)
        if mgr is None:
            return "Brain profiles are not active, Boss."
        profile = (args.get("profile") or "").strip().lower()
        ok, msg = mgr.set_profile(profile)
        if ok:
            self._bus.emit_fast(
                "runtime_settings_changed",
                brain_profile=mgr.active_profile,
            )
        return msg

    def _do_set_assistant_mode(self, _action: str, args: dict) -> str:
        mgr = getattr(self, "_assistant_mode_mgr", None)
        if mgr is None:
            return "Assistant mode manager is not active."
        mode = (args.get("mode") or "").strip().lower().replace(" ", "_")
        ok, msg = mgr.set_mode(mode)
        if ok:
            self._bus.emit_fast(
                "runtime_settings_changed",
                assistant_mode=mgr.active,
            )
        return msg

    def _do_run_routine(self, _action: str, args: dict) -> str:
        name = (args.get("name") or "").strip().lower()
        phase = (args.get("phase") or "enter").strip().lower()
        engine = getattr(self, "_routine_engine", None)
        if engine is None:
            return "Routines aren't active, Boss."
        if not name:
            return "Which routine should I run?"
        try:
            return engine.execute(name, phase)
        except Exception as exc:
            logger.info("run_routine failed", exc_info=True)
            return f"Routine {name.replace('_', ' ')} hit an error. ({exc.__class__.__name__})"

    def _routine_step_dispatch(self, kind: str, args: dict) -> str:
        """Map routine steps to existing router handlers."""
        try:
            if kind == "volume":
                return self._do_set_volume("set_volume", args)
            if kind == "mute":
                return self._do_mute("mute", {})
            if kind == "unmute":
                return self._do_unmute("unmute", {})
            if kind == "brain_profile":
                return self._do_set_brain_profile("set_brain_profile", args)
            if kind == "assistant_mode":
                return self._do_set_assistant_mode("set_assistant_mode", args)
        except Exception as exc:
            logger.info("routine step '%s' failed", kind, exc_info=True)
            return f"step '{kind}' errored: {exc.__class__.__name__}"
        return f"unknown step '{kind}'"

    # ── v22: Advanced System & UI Tools ───────────────────────────────

    def _get_system_control(self) -> Any:
        from core.system_control import SystemControl
        if not hasattr(self, "_sys_ctl_cache") or self._sys_ctl_cache is None:
            self._sys_ctl_cache = SystemControl(self._config)
        return self._sys_ctl_cache

    def _do_describe_focused_element(self, _action: str, _args: dict) -> str:
        from core.desktop_control import describe_focused_element
        return describe_focused_element()

    def _do_read_focused_text(self, _action: str, _args: dict) -> str:
        from core.desktop_control import read_focused_text
        return read_focused_text()

    def _do_set_focused_text(self, _action: str, args: dict) -> str:
        from core.desktop_control import set_focused_text
        text = args.get("text", "")
        return set_focused_text(text)

    def _do_click_ui_element(self, _action: str, args: dict) -> str:
        from core.desktop_control import click_ui_element
        label = args.get("label", "")
        role = args.get("role")
        return click_ui_element(label, role)

    def _do_get_process_details(self, _action: str, args: dict) -> str:
        pid = args.get("pid")
        if pid is None:
            return "Please provide a valid process ID (pid)."
        res = self._get_system_control().get_process_details(int(pid))
        if not res.success:
            return res.message
        return f"{res.message}: " + ", ".join(f"{k}={v}" for k, v in res.data.items() if str(v))

    def _do_find_process_by_name(self, _action: str, args: dict) -> str:
        name = args.get("name")
        if not name:
            return "Please provide a process name to find."
        res = self._get_system_control().find_process_by_name(name)
        if not res.success:
            return res.message
        matches = res.data.get("matches", [])
        if not matches:
            return f"No processes found matching '{name}'."
        summary = "\n".join(f"[{m['pid']}] {m['name']} (CPU: {m['cpu']}%, RAM: {m['mem_mb']}MB)" for m in matches[:10])
        return res.message + ":\n" + summary

    def _do_set_process_priority(self, _action: str, args: dict) -> str:
        pid = args.get("pid")
        priority = args.get("priority", "normal")
        if pid is None:
            return "Please provide a process ID (pid)."
        res = self._get_system_control().set_process_priority(int(pid), priority)
        return res.message

    def _do_optimize_for_atom(self, _action: str, _args: dict) -> str:
        res = self._get_system_control().optimize_for_atom()
        return res.message

    def _do_get_open_ports(self, _action: str, _args: dict) -> str:
        res = self._get_system_control().get_open_ports()
        if not res.success:
            return res.message
        ports = res.data.get("ports", [])
        summary = "\n".join(f"Port {p['port']} ({p['process'][:20] if p['process'] else 'Unknown'})" for p in ports[:20])
        return f"{res.message}. Sample:\n{summary}"

    def _do_get_wifi_networks(self, _action: str, _args: dict) -> str:
        res = self._get_system_control().get_wifi_networks()
        if not res.success:
            return res.message
        nets = res.data.get("networks", [])
        summary = "\n".join(f"SSID: {n['ssid']}" for n in nets[:10])
        return f"{res.message}. Sample:\n{summary}"

    def _do_analyze_temp_files(self, _action: str, _args: dict) -> str:
        res = self._get_system_control().analyze_temp_files()
        if not res.success:
            return res.message
        return (
            f"Temp Files Analysis: {res.data.get('total_size_mb')} MB total "
            f"in {res.data.get('file_count')} files."
        )

    def _do_find_large_files(self, _action: str, args: dict) -> str:
        path = args.get("path", "")
        min_size = args.get("min_size_mb", 100)
        res = self._get_system_control().find_large_files(path, min_size)
        if not res.success:
            return res.message
        files = res.data.get("files", [])
        summary = "\n".join(f"{f['size_mb']}MB: {f['path']}" for f in files[:10])
        return f"{res.message}:\n{summary}"

    def _do_execute_desktop_macro(self, _action: str, args: dict) -> str:
        goal = args.get("goal")
        if not goal:
            return "Please specify a goal for the macro."
            
        import asyncio
        from core.desktop_agent import DesktopAgent
        
        # Desktop Agent relies on Gemini for recursive reasoning
        if hasattr(self, "_gemini_client") and self._gemini_client:
            gemini = self._gemini_client
        else:
            return "Macro execution failed: Gemini Cloud intelligence must be connected."
            
        agent = DesktopAgent(gemini_client=gemini, router=self, config=self._config)
        
        # Execute it async and block on it internally
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(agent.execute_macro(goal))
            # We want this to block the router essentially untill the macro finishes
            import concurrent.futures
            future = concurrent.futures.Future()
            task.add_done_callback(lambda t: future.set_result(t.result()) if not t.exception() else future.set_exception(t.exception()))
            return "Started Macro. Check logs." # For true blocking we'd need async action handlers
            # Currently ATOM action handlers are synchronous. Let's just create an event loop for it if needed:
        except RuntimeError:
            return asyncio.run(agent.execute_macro(goal))

    _ACTION_DISPATCH: dict[str, Any] = {
        "open_app": _do_open_app,
        "close_app": _do_close_app,
        "list_apps": _do_list_apps,
        "search": _do_search,
        "spotlight_search": _do_spotlight_search,
        "smart_find_file": _do_smart_find_file,
        "play_youtube": _do_play_youtube,
        "stop_music": _do_stop_music,
        "set_volume": _do_set_volume,
        "mute": _do_mute,
        "unmute": _do_unmute,
        "create_folder": _do_create_folder,
        "move_path": _do_move_path,
        "copy_path": _do_copy_path,
        "lock_screen": _do_lock_screen,
        "screenshot": _do_screenshot,
        "set_brightness": _do_set_brightness,
        "shutdown_pc": _do_shutdown_pc,
        "restart_pc": _do_restart_pc,
        "logoff": _do_logoff,
        "sleep_pc": _do_sleep_pc,
        "empty_recycle_bin": _do_empty_recycle_bin,
        "flush_dns": _do_flush_dns,
        "minimize_window": _do_minimize_window,
        "maximize_window": _do_maximize_window,
        "switch_window": _do_switch_window,
        "timer": _do_timer,
        "read_clipboard": _do_read_clipboard,
        "open_url": _do_open_url,
        "weather": _do_weather,
        "wifi_status": _do_wifi_status,
        "set_reminder": _do_set_reminder,
        "show_reminders": _do_show_reminders,
        "whats_on_my_plate": _do_whats_on_my_plate,
        "cancel_reminders": _do_cancel_reminders,
        "kill_process": _do_kill_process,
        "resource_report": _do_resource_report,
        "resource_trend": _do_resource_trend,
        "app_history": _do_app_history,
        "research_topic": _do_research_topic,
        "self_diagnostic": _do_self_diagnostic,
        "behavior_report": _do_behavior_report,
        "scroll_down": _do_scroll_down,
        "scroll_up": _do_scroll_up,
        "click_screen": _do_click_screen,
        "press_key": _do_press_key,
        "go_back": _do_go_back,
        "hotkey_combo": _do_hotkey_combo,
        "type_text": _do_type_text,
        "system_analyze": _do_system_analyze,
        "self_check": _do_self_check,
        "audio_diagnostics": _do_audio_diagnostics,
        "mode_status": _do_mode_status,
        "detailed_status": _do_detailed_status,
        "set_performance_mode": _do_set_performance_mode,
        "set_brain_profile": _do_set_brain_profile,
        "set_assistant_mode": _do_set_assistant_mode,
        "run_routine": _do_run_routine,
        
        # v22: Advanced System Control
        "describe_focused_element": _do_describe_focused_element,
        "read_focused_text": _do_read_focused_text,
        "set_focused_text": _do_set_focused_text,
        "click_ui_element": _do_click_ui_element,
        "get_process_details": _do_get_process_details,
        "find_process_by_name": _do_find_process_by_name,
        "set_process_priority": _do_set_process_priority,
        "optimize_for_atom": _do_optimize_for_atom,
        "get_open_ports": _do_get_open_ports,
        "get_wifi_networks": _do_get_wifi_networks,
        "analyze_temp_files": _do_analyze_temp_files,
        "find_large_files": _do_find_large_files,
        "execute_desktop_macro": _do_execute_desktop_macro,
    }

    _LATE_DISPATCH = {
        "remember": "_do_remember",
        "recall": "_do_recall",
        "learn_document": "_do_learn_document",
        "run_code": "_do_run_code",
        "calculate": "_do_calculate",
        "record_workflow": "_do_record_workflow",
        "stop_recording": "_do_stop_recording",
        "run_workflow": "_do_run_workflow",
        "list_workflows": "_do_list_workflows",
        "screen_read": "_do_screen_read",
        "show_dream_summary": "_do_show_dream_summary",
        "run_terminal_command": "_do_run_terminal_command",
        "set_goal": "_do_set_goal",
        "show_goals": "_do_show_goals",
    }

    # ── Reasoning Engine actions ───────────────────────────────────────

    def _do_remember(self, _action: str, args: dict) -> str:
        fact = args.get("fact", "")
        if not fact:
            return "What should I remember, Boss?"
        self._bus.emit_fast("brain_remember_request", fact=fact)
        return f"Got it, Boss. I'll remember that: {fact[:100]}"

    def _do_recall(self, _action: str, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return "What should I recall, Boss?"
        self._bus.emit_fast("brain_recall_request", query=query)
        return ""

    def _do_learn_document(self, _action: str, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return "Which document should I learn from, Boss?"
        self._bus.emit_fast("document_ingest_request", path=path)
        return f"I'll start learning from that document, Boss."

    def _get_sandbox(self):
        if self._code_sandbox is None:
            from core.reasoning.code_sandbox import CodeSandbox
            self._code_sandbox = CodeSandbox(self._config)
        return self._code_sandbox

    def _do_run_code(self, _action: str, args: dict) -> str:
        code = args.get("code", args.get("expression", ""))
        if not code:
            return "What should I calculate or run, Boss?"
        try:
            result = self._get_sandbox().execute(code)
            if result["success"]:
                return f"Result: {result['result']}"
            return f"Couldn't compute that: {result['error']}"
        except Exception as e:
            return f"Calculation error: {str(e)[:100]}"

    def _do_calculate(self, _action: str, args: dict) -> str:
        expr = args.get("expression", "")
        if not expr:
            return "What should I calculate, Boss?"
        try:
            return self._get_sandbox().evaluate_math(expr)
        except Exception as e:
            return f"Calculation error: {str(e)[:100]}"

    def _do_record_workflow(self, _action: str, args: dict) -> str:
        name = args.get("name", "")
        self._bus.emit_fast("workflow_start_recording", name=name)
        return f"Recording workflow '{name or 'unnamed'}'. I'll capture your actions."

    def _do_stop_recording(self, _action: str, _args: dict) -> str:
        self._bus.emit_fast("workflow_stop_recording")
        return "Workflow recording stopped."

    def _do_run_workflow(self, _action: str, args: dict) -> str:
        name = args.get("name", "")
        if not name:
            return "Which workflow should I run, Boss?"
        self._bus.emit_fast("workflow_replay_request", name=name)
        return f"Running workflow '{name}'."

    def _do_list_workflows(self, _action: str, _args: dict) -> str:
        self._bus.emit_fast("workflow_list_request")
        return ""

    def _do_screen_read(self, _action: str, _args: dict) -> str:
        self._bus.emit_fast("screen_read_request")
        return "Reading your screen, Boss."

    def _do_show_dream_summary(self, _action: str, _args: dict) -> str:
        self._bus.emit_fast("dream_summary_request")
        return ""

    def _do_set_goal(self, _action: str, args: dict) -> str:
        title = args.get("title", "")
        if not title:
            return "What goal should I set, Boss?"
        self._bus.emit_fast("intent_classified", intent="goal_create",
                            action_args={"title": title})
        return ""

    def _do_show_goals(self, _action: str, _args: dict) -> str:
        self._bus.emit_fast("intent_classified", intent="goal_show",
                            action_args={})
        return ""

    def _do_run_terminal_command(self, _action: str, args: dict) -> str:
        import subprocess as _sp
        cmd = args.get("command", "").strip()
        if not cmd:
            return "No command provided, Boss."
        cmd_ok, cmd_reason = self._security.is_safe_command(cmd)
        if not cmd_ok:
            return f"Blocked for safety: {cmd_reason}"
        try:
            argv = shlex.split(cmd, posix=(os.name != "nt"))
        except ValueError:
            return "Couldn't parse that command, Boss."
        if not argv:
            return "No command provided, Boss."
        try:
            result = _sp.run(
                argv, shell=False, capture_output=True, text=True, timeout=30,
            )
            output = (result.stdout or result.stderr or "").strip()[:500]
            return output or "Command completed with no output."
        except _sp.TimeoutExpired:
            return "Command timed out after 30 seconds."
        except FileNotFoundError:
            return f"Command not found: {argv[0]}"
        except Exception as e:
            return f"Command failed: {str(e)[:100]}"

    # ── Screen analysis (with OCR) ─────────────────────────────────────

    async def _handle_screen_analyze(self, args: dict) -> None:
        """Screen analysis via local OCR."""
        _q = args.get("question", "")
        try:
            from context.screen_reader import ScreenReader
            reader = ScreenReader(self._config)
            summary = reader.get_screen_summary()
            if _q:
                full_text = f"You asked: {_q}. {summary}"
            else:
                full_text = summary
            self._emit_response(full_text)
        except Exception:
            self._emit_response(
                "Screen reading isn't fully available yet, Boss. "
                "Say 'take a screenshot' or paste text to the clipboard."
                + (f" You asked: {_q[:80]}" if _q else "")
            )

    # ── v22: Cloud execution handlers ────────────────────────────────

    async def _handle_cloud_reason(
        self,
        original_text: str,
        clean_text: str,
        query_plan: Any,
        t0: float,
    ) -> bool:
        """Execute CLOUD_REASON path: Gemini for abstract reasoning.

        Routes to Buddy vs Reasoning model based on cognitive planning.
        """
        if self._gemini_client is None or not self._gemini_client.is_available:
            logger.info("CLOUD_REASON: Gemini unavailable, falling back to local")
            return False

        model_role = getattr(query_plan, "model_role", "buddy").lower()
        prompt_hint = getattr(query_plan, "prompt_hint", None)

        ack = "On it, Boss." if model_role == "reasoning" else "Let's chat."
        self._emit_thinking_ack(ack)

        try:
            if model_role == "reasoning":
                response, ok = await self._gemini_client.ask_reasoning(
                    clean_text, system_instruction=prompt_hint
                )
            else:
                response, ok = await self._gemini_client.ask_buddy(
                    clean_text, system_instruction=prompt_hint
                )

            latency_ms = (time.perf_counter() - t0) * 1000

            if not ok or not response:
                logger.info("CLOUD_REASON: Gemini [%s] failed (%.0fms), falling back", model_role, latency_ms)
                return False

            # Tag as cloud-sourced and enrich
            if self._decision_engine is not None:
                try:
                    enriched = self._decision_engine.enrich(clean_text, response)
                    response = enriched.enriched or response
                except Exception:
                    logger.debug("Decision engine enrichment failed", exc_info=True)

            # Cache the response
            if self._semantic_cache is not None:
                try:
                    self._semantic_cache.put(clean_text, response, source=f"cloud:{model_role}")
                except Exception:
                    logger.debug('Cloud reason gate failed', exc_info=True)

            # Update metrics
            self._llm_queries += 1
            self._bus.emit_fast("metrics_event", counter=f"cloud_{model_role}_queries")
            logger.info(
                "CLOUD_REASON [%s] served in %.0fms (%d chars)",
                model_role, latency_ms, len(response),
            )
            self._emit_response(response)
            return True

        except Exception as exc:
            logger.warning("CLOUD_REASON failed: %s", exc)
            return False

    async def _handle_cloud_search(
        self,
        original_text: str,
        clean_text: str,
        query_plan: Any,
        t0: float,
    ) -> bool:
        """Execute CLOUD_SEARCH path: DuckDuckGo search + local summarization.

        Returns True if handled, False to fall through to local LLM.
        """
        if self._search_tool is None:
            logger.info("CLOUD_SEARCH: SearchTool unavailable, falling back to local")
            return False

        self._emit_thinking_ack("Searching for that, Boss.")

        try:
            result = await self._search_tool.search(clean_text)
            latency_ms = (time.perf_counter() - t0) * 1000

            if not result.get("success") or not result.get("text"):
                logger.info("CLOUD_SEARCH: no results (%.0fms), falling back", latency_ms)
                return False

            search_text = result["text"]
            sources = result.get("sources", [])

            # Summarize via Gemini if available, else serve raw results to local LLM
            summary = ""
            if self._gemini_client and self._gemini_client.is_available:
                try:
                    summary = await self._search_tool.search_and_summarize(
                        clean_text, use_cloud_summarizer=True,
                    )
                except Exception:
                    logger.debug('Cloud reason gate failed', exc_info=True)

            if not summary:
                summary = search_text

            # Enrich
            if self._decision_engine is not None:
                try:
                    enriched = self._decision_engine.enrich(clean_text, summary)
                    summary = enriched.enriched or summary
                except Exception:
                    logger.debug('Cloud search gate failed', exc_info=True)

            # Cache
            if self._semantic_cache is not None:
                try:
                    self._semantic_cache.put(clean_text, summary, source="search")
                except Exception:
                    logger.debug('Cloud search gate failed', exc_info=True)

            self._llm_queries += 1
            self._bus.emit_fast("metrics_event", counter="cloud_search_queries")
            logger.info(
                "CLOUD_SEARCH served in %.0fms (%d sources)",
                latency_ms, len(sources),
            )
            self._emit_response(summary)
            return True

        except Exception as exc:
            logger.warning("CLOUD_SEARCH failed: %s", exc)
            return False

    # ── LLM fallback ────────────────────────────────────────────────

    async def _handle_llm_fallback(self, original_text: str,
                                   clean_text: str,
                                   clipboard_injected: bool = False) -> None:
        ack = self._conv_mgr.smart_ack(clean_text)
        if clipboard_injected:
            ack = "I see what's on your clipboard. " + ack

        cache_key = clean_text.lower()
        t_lookup = time.perf_counter()

        is_repeat = self._conv_mgr.check_repeat(cache_key)
        if is_repeat:
            logger.info("Repeat query detected -- bypassing cache")

        query_plan = None
        if self._cognitive_kernel is not None:
            try:
                query_plan = self._cognitive_kernel.route(
                    clean_text,
                    allow_cache=not is_repeat,
                )
            except Exception:
                logger.debug("Cognitive Kernel routing failed", exc_info=True)

        if (
            query_plan is not None
            and getattr(query_plan, "skip_llm", False)
            and getattr(query_plan, "direct_response", None)
        ):
            logger.info(
                "Cognitive Kernel served %s in fallback path (%.1fms)",
                getattr(query_plan, "reason", "direct"),
                (time.perf_counter() - t_lookup) * 1000,
            )
            self._emit_response(query_plan.direct_response)
            return

        # ── v22: Intercept CLOUD_REASON / CLOUD_SEARCH before LLM fallback ──
        if query_plan is not None:
            from core.cognitive_kernel import ExecPath
            plan_path = getattr(query_plan, "path", None)

            if plan_path is ExecPath.CLOUD_SEARCH:
                handled = await self._handle_cloud_search(
                    original_text, clean_text, query_plan, t_lookup,
                )
                if handled:
                    return

            if plan_path is ExecPath.CLOUD_REASON:
                handled = await self._handle_cloud_reason(
                    original_text, clean_text, query_plan, t_lookup,
                )
                if handled:
                    return

        if query_plan is None:
            if self._runtime_watchdog is not None:
                cache_task = asyncio.create_task(
                    self._runtime_watchdog.run_sync(
                        "cache_lookup",
                        self._cache.get,
                        cache_key,
                        default=None,
                        metadata={"query": cache_key[:80]},
                    ),
                )
            else:
                cache_task = asyncio.create_task(
                    asyncio.get_running_loop().run_in_executor(
                        None, self._cache.get, cache_key)
                )

            from core.quick_replies import try_quick_reply

            memory_task = asyncio.create_task(self._memory.retrieve(clean_text, k=2))
            cached_result, memory_ctx = await asyncio.gather(cache_task, memory_task)
            cached = (
                cached_result.value
                if self._runtime_watchdog is not None
                else cached_result
            )

            if cached and not is_repeat:
                logger.info("Serving from cache (%.1fms)",
                            (time.perf_counter() - t_lookup) * 1000)
                self._emit_response(cached)
                return

            qr = try_quick_reply(clean_text, self._config)
            if qr:
                logger.info("Quick reply served (no LLM)")
                self._emit_response(qr)
                return
        else:
            memory_ctx = None
            if getattr(query_plan, "use_memory", False):
                memory_k = max(1, int(getattr(query_plan, "memory_limit", 2) or 2))
                memory_ctx = await self._memory.retrieve(clean_text, k=memory_k)

        # NOTE: do NOT splice a "[SYSTEM NOTE: ...]" string into the user
        # query. Small models (Qwen3-8B-4bit) sometimes echo the bracketed
        # instruction back during TTS as quoted analysis like
        # `"Dear Boss" — the user is greeting you, so respond politely…`.
        # Instead, signal the repeat through bus metadata so the prompt
        # builder injects a clean steer in the SYSTEM layer.

        if self._assistant_mode_mgr is not None:
            if not self._assistant_mode_mgr.allows_llm_fallback():
                logger.info("Assistant mode command_only — skipping LLM")
                self._emit_response(self._assistant_mode_mgr.command_only_message())
                return

        context_bundle = None
        if self._context is not None:
            try:
                context_bundle = self._context.get_bundle()
            except Exception:
                logger.debug("Context bundle retrieval failed", exc_info=True)

        # Inject real-time context from SystemStateEngine, SessionMemory, UserMemory
        if self._system_state_engine is not None:
            try:
                ctx_str = self._system_state_engine.get_context_string()
                if ctx_str:
                    context_bundle = dict(context_bundle or {})
                    context_bundle["system_state"] = ctx_str
            except Exception:
                logger.debug("System state context injection failed", exc_info=True)

        if self._session_memory is not None:
            try:
                sess_str = self._session_memory.context_for_prompt()
                if sess_str:
                    context_bundle = dict(context_bundle or {})
                    context_bundle["recent_commands"] = sess_str
            except Exception:
                logger.debug("Session memory context injection failed", exc_info=True)

        if self._user_memory is not None:
            try:
                user_str = self._user_memory.context_for_prompt()
                if user_str:
                    context_bundle = dict(context_bundle or {})
                    context_bundle["user_profile"] = user_str
            except Exception:
                logger.debug("User memory context injection failed", exc_info=True)

        if self._conv_memory is not None:
            summ = self._conv_memory.summary_for_prompt()
            if summ:
                context_bundle = dict(context_bundle or {})
                context_bundle["session_summary"] = summ
            if self._conv_memory.turn_count > 0:
                topics = self._conv_memory.active_topics
                if topics:
                    context_bundle = dict(context_bundle or {})
                    context_bundle["active_topics"] = ", ".join(topics)

        if not self._security.is_feature_enabled("llm"):
            logger.info("LLM feature disabled by policy")
            self._emit_response(personality.offline_fallback())
            return

        if not self._brain_enabled:
            if self._gemini_client is not None and self._gemini_client.is_available:
                await self._handle_cloud_streaming(
                    original_text, clean_text, ack, query_plan,
                )
                return
            logger.info("No local brain and no cloud — offline fallback")
            self._emit_response(personality.offline_fallback())
            return

        history = (
            self._conv_memory.get_pairs()
            if self._conv_memory is not None and self._conv_memory.turn_count > 0
            else self.get_conversation_history()
        )

        if self._should_emit_thinking_ack(clean_text, query_plan):
            self._emit_thinking_ack(ack)

        self._bus.emit_long(
            "cursor_query",
            text=original_text,
            policy_query=clean_text,
            memory_context=memory_ctx,
            context=context_bundle,
            history=history,
            query_plan=query_plan,
            repeat_hint=bool(is_repeat),
        )

    # ── Cloud streaming (token-by-token → TTS) ─────────────────────

    async def _handle_cloud_streaming(
        self,
        original_text: str,
        clean_text: str,
        ack: str,
        query_plan: Any = None,
    ) -> None:
        """Stream Gemini response into partial_response events.

        Applies two safeguards:
        - **Micro-batching**: tokens are accumulated and flushed every 5
          tokens or 50ms, whichever comes first, to avoid flooding the TTS
          queue with single-token events.
        - **Stream generation guard**: each streaming call gets a generation
          number.  If a new call starts (e.g. after barge-in), the old
          callback silently discards remaining tokens.
        """
        import uuid as _uuid

        if self._should_emit_thinking_ack(clean_text, query_plan):
            self._emit_thinking_ack(ack)

        # Cancel any in-flight streaming request in the executor thread.
        if self._gemini_client is not None:
            self._gemini_client.cancel_streaming()

        self._cloud_stream_generation += 1
        my_generation = self._cloud_stream_generation

        stream_id = _uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()
        first_token = [True]
        batch: list[str] = []
        last_flush = [time.perf_counter()]
        _BATCH_SIZE = 5
        _BATCH_INTERVAL = 0.05

        def _flush_batch(is_last: bool) -> None:
            if my_generation != self._cloud_stream_generation:
                return
            text = "".join(batch)
            batch.clear()
            if not text and not is_last:
                return
            is_first = first_token[0]
            first_token[0] = False
            last_flush[0] = time.perf_counter()

            def _emit() -> None:
                if my_generation != self._cloud_stream_generation:
                    return
                self._bus.emit(
                    "partial_response",
                    text=text,
                    is_first=is_first,
                    is_last=is_last,
                    source="gemini_stream",
                    stream_id=stream_id,
                )

            loop.call_soon_threadsafe(_emit)

        def _on_token(chunk: str, is_last: bool) -> None:
            if my_generation != self._cloud_stream_generation:
                return

            if chunk:
                batch.append(chunk)

            now = time.perf_counter()
            elapsed = now - last_flush[0]

            if is_last:
                _flush_batch(True)
            elif len(batch) >= _BATCH_SIZE or elapsed >= _BATCH_INTERVAL:
                _flush_batch(False)

        default_system = (
            "You are ATOM, a personal AI assistant (JARVIS-style) created by Satyam Yadav. "
            "You call him 'Boss'. You are friendly, witty, concise, and helpful. "
            "Keep responses short and conversational unless asked for detail. "
            "Never invent or promise actions the user did not explicitly request — "
            "do not say things like 'I'll play the song', 'opening YouTube', or "
            "'setting a reminder' unless Boss asked for that exact action in this turn. "
            "If the transcribed query is unclear or nonsensical, ask ONE short clarifying "
            "question instead of guessing. Ground every factual claim in supplied context; "
            "if the context doesn't contain the answer, say you don't have that yet."
        )

        _adaptive_concise = (
            self._adaptive is not None and self._adaptive.should_be_concise()
        )
        if self._perception_concise or _adaptive_concise:
            default_system += (
                " The user has been interrupting or seems impatient—"
                "keep this response very brief (1-2 sentences max)."
            )

        try:
            full_text, ok = await self._gemini_client.ask_streaming(
                original_text,
                on_token=_on_token,
                system_instruction=default_system,
            )
            if ok and full_text and my_generation == self._cloud_stream_generation:
                self.record_turn(clean_text, full_text)
        except asyncio.CancelledError:
            self._gemini_client.cancel_streaming()
            raise
        except Exception:
            logger.exception("Cloud streaming failed — emitting error response")
            if my_generation == self._cloud_stream_generation:
                self._emit_response("Cloud request failed, Boss. Try again.")

    # ── Perception adaptive profile ─────────────────────────────────

    def apply_perception_profile(
        self,
        concise: bool = False,
        rate_boost: float = 0.0,
        **_kw: Any,
    ) -> None:
        """Called from the bus when the perception layer updates its profile."""
        self._perception_concise = concise

    # ── Contextual follow-up ────────────────────────────────────────

    def _suggest_follow_up(self, query: str, response: str) -> str | None:
        return self._conv_mgr.suggest_follow_up(query, response)

    # ── Conversation window ─────────────────────────────────────────

    def record_turn(self, query: str, response: str) -> None:
        return self._conv_mgr.record_turn(query, response)

    def get_conversation_history(self) -> list[tuple[str, str]]:
        return self._conv_mgr.get_conversation_history()

    # ── Helpers ─────────────────────────────────────────────────────

    # Presence-check queries — short social "are you there" pings that
    # should NEVER get a diagnostic percent readout. Matched against the
    # lowercased query in ``_status_with_usage``.
    _PRESENCE_CHECK_PATTERNS: tuple[str, ...] = (
        "can you hear me", "can u hear me",
        "are you there", "you there", "u there",
        "are you alive", "you alive",
        "are you awake", "you awake",
        "are you ready", "you ready",
        "are you listening", "you listening",
        "hello there", "hey there", "hi there",
        "are you up", "you up",
        "are you online",
    )

    def _is_presence_check(self, query: str) -> bool:
        q = (query or "").lower().strip()
        if not q:
            return False
        return any(p in q for p in self._PRESENCE_CHECK_PATTERNS)

    def _status_with_usage(self, base: str, *, query: str = "") -> str:
        """Return ATOM's status reply. For casual presence checks (e.g.
        "can you hear me", "you there?") we drop the percent readout and
        answer like a person — anything else keeps the diagnostic tail.
        """
        if self._is_presence_check(query):
            return base
        total = self._local_queries + self._llm_queries
        if total <= 0:
            return base
        llm_pct = (self._llm_queries / total) * 100
        return (f"{base} LLM handled {llm_pct:.0f} percent "
                f"of routed queries.")
