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
import re
import time
from typing import TYPE_CHECKING, Any

from core import adaptive_personality as personality
from core.command_cache import get_command_cache
from core.process_manager import ProcessManager
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
    from core.intent_engine import IntentEngine
    from core.memory_engine import MemoryEngine
    from core.self_evolution import SelfEvolutionEngine
    from core.state_manager import StateManager
    from core.task_scheduler import TaskScheduler

logger = logging.getLogger("atom.router")

_FILLER = re.compile(
    r"\b(um+|uh+|hmm+|ah+|oh+|like|actually|basically|"
    r"you know|i mean|so+|well|okay so|right so|"
    r"please|kindly)\b",
    re.I,
)
_MULTI_SPACE = re.compile(r"\s+")

# ── Conversational continuity ────────────────────────────────────────
_DANGLING_PRONOUN = re.compile(
    r"\b(it|that|this|there|those|these|them)\b", re.I)
_STOP_VERBS = frozenset({
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "has", "have", "had", "be", "been",
    "tell", "show", "give", "get", "make", "let", "know", "say",
    "explain", "search", "find", "open", "close", "check", "set",
})
_STOP_PREPS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "about", "from", "by", "as", "into", "through", "my", "your",
    "me", "i", "and", "or", "but", "so", "if", "what", "why", "how",
    "when", "where", "who", "which", "please",
})
_CLIPBOARD_REF = re.compile(
    r"\b(that\s+error|this\s+error|that\s+code|this\s+code|"
    r"that\s+message|this\s+message|what\s+i\s+copied|"
    r"the\s+clipboard|from\s+clipboard|clipboard\s+content|"
    r"that\s+exception|this\s+exception|that\s+bug|this\s+bug|"
    r"that\s+text|this\s+text)\b", re.I)


def _extract_entity(text: str) -> str:
    """Extract the likely topic/entity from a query -- zero-cost heuristic.

    Strips verbs, prepositions, and stop words from the end, then takes
    the remaining significant tail as the entity.
    """
    words = text.lower().split()
    significant = [
        w for w in words
        if w not in _STOP_VERBS and w not in _STOP_PREPS
        and len(w) > 2 and not w.isdigit()
    ]
    if not significant:
        return ""
    return " ".join(significant[-3:])


def _resolve_pronouns(query: str, last_entity: str) -> str:
    """Replace dangling pronouns with the last known entity."""
    if not last_entity:
        return query
    words = query.split()
    if len(words) > 8:
        return query
    if not _DANGLING_PRONOUN.search(query):
        return query
    has_noun = any(
        w.lower() not in _STOP_VERBS and w.lower() not in _STOP_PREPS
        and len(w) > 2 and not w.isdigit()
        and w.lower() not in ("it", "that", "this", "there", "those",
                              "these", "them", "he", "she", "they")
        for w in words
    )
    if has_noun:
        return query
    resolved = _DANGLING_PRONOUN.sub(last_entity, query, count=1)
    return resolved


def compress_query(text: str) -> str:
    cleaned = _FILLER.sub("", text)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned[:1500]


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
        "self_check", "behavior_report",
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
    ) -> None:
        self._bus = bus
        self._state = state
        self._cache = cache
        self._memory = memory
        self._intent = intent_engine
        self._context = context_engine
        self._config = config or {}
        self._security = SecurityPolicy(self._config)
        self._brain_enabled = bool(self._config.get("brain", {}).get("enabled", False))
        self._brain_mode_mgr = brain_mode_manager
        self._assistant_mode_mgr = assistant_mode_manager
        self._processing_lock = asyncio.Lock()
        self._local_queries = 0
        self._llm_queries = 0
        self._last_entity: str = ""
        self._recent_queries: list[tuple[str, float]] = []
        self._conversation_window: list[tuple[str, str]] = []
        self._conv_window_max = 20
        self._scheduler = scheduler
        self._process_mgr = process_mgr or ProcessManager()
        self._evolution = evolution
        self._behavior_tracker = behavior_tracker
        self._skills = skills_registry
        self._conv_memory = conversation_memory
        self._timeline = timeline_memory

        # Extracted modules
        self._confirmation = ConfirmationManager(self._security)
        self._diagnostics = DiagnosticsHandler(self._config)

        from core.reasoning.action_executor import ActionExecutor
        self._action_executor = ActionExecutor(
            dispatch_fn=self._dispatch_action,
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

    async def on_speech(self, text: str, **_kw) -> None:
        if self._processing_lock.locked():
            logger.warning("Dropped speech '%s' -- already processing",
                           text[:40])
            return
        async with self._processing_lock:
            await self._route(text)

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

        clean_text = compress_query(text)
        if not clean_text:
            return

        if self._timeline is not None:
            try:
                self._timeline.append_event(
                    "user_query",
                    {"text": clean_text[:2000], "source": "router"},
                )
            except Exception:
                pass

        if self._conv_memory is not None:
            self._conv_memory.on_new_user_query(clean_text)

        _skill_chain: list[str] = []
        if self._skills is not None:
            match = self._skills.try_expand_full(clean_text)
            if match is not None:
                logger.info(
                    "Skill '%s': '%s' -> '%s'%s",
                    match.skill_id, clean_text[:80], match.primary[:80],
                    f" +{len(match.chain)} chain" if match.chain else "",
                )
                clean_text = match.primary
                text = match.primary
                _skill_chain = list(match.chain)

        # ── 1. Pronoun resolution (conversational continuity) ────────
        resolved = _resolve_pronouns(clean_text, self._last_entity)
        if resolved != clean_text:
            logger.info("Pronoun resolved: '%s' -> '%s'", clean_text, resolved)
            clean_text = resolved
            text = resolved

        # ── 3. Clipboard injection (implicit context) ────────────────
        clipboard_injected = False
        if _CLIPBOARD_REF.search(clean_text) and self._context is not None:
            try:
                bundle = self._context.get_bundle()
                clip = (bundle or {}).get("clipboard", "")
                if clip and len(clip) < 1000:
                    clip, _ = self._security.sanitize_input(clip)
                    text = f"{text}\n\nReferenced content: {clip}"
                    clipboard_injected = True
                    logger.info("Clipboard injected (%d chars)", len(clip))
            except Exception:
                logger.debug("Clipboard injection failed", exc_info=True)

        cmd_cache = get_command_cache()
        cached = cmd_cache.get(clean_text)
        if cached is not None:
            result = cached
            classify_ms = 0.0
            logger.info("Router: '%s' -> %s (CACHED, 0ms)",
                         clean_text[:60], result.intent)
        else:
            result = self._intent.classify(clean_text)
            classify_ms = (time.perf_counter() - t0) * 1000
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

        self._bus.emit_fast("intent_classified",
                            intent=result.intent, ms=classify_ms,
                            text=clean_text,
                            action_args=result.action_args)

        _budget.warn_if_slow("intent_classify")

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
        entity = _extract_entity(clean_text)
        if entity:
            self._last_entity = entity

        is_local = result.intent not in ("fallback", "screen_analyze")

        if is_local:
            if result.intent in ("confirm", "deny"):
                await self._handle_confirmation(result.intent)
                return

            if result.intent == "go_silent":
                self._bus.emit_long("response_ready",
                                    text=result.response or personality.silent_response(),
                                    is_sleep=True)
                return

            if result.intent == "exit":
                self._bus.emit_long("response_ready",
                                    text=result.response or personality.exit_response(),
                                    is_exit=True)
                return

            if result.action:
                self._local_queries += 1
                self._bus.emit_fast("metrics_event", counter="local_routed_queries")
                if self._confirmation.requires_confirmation(result):
                    prompt = self._confirmation.set_pending_action(result)
                    self._bus.emit_long("response_ready", text=prompt)
                    return
                await self._execute_action(result)
                if _skill_chain:
                    await self._run_skill_chain(_skill_chain)
                return

            if result.response:
                self._local_queries += 1
                self._bus.emit_fast("metrics_event", counter="local_routed_queries")
                if result.intent == "status":
                    self._bus.emit_long("response_ready",
                                   text=self._status_with_usage(result.response))
                    return
                self._bus.emit_long("response_ready", text=result.response)
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
            await self._handle_llm_fallback(text, clean_text,
                                            clipboard_injected=clipboard_injected)
            return

    # ── Confirmation flow (delegated to ConfirmationManager) ───────────

    async def _handle_confirmation(self, confirm_intent: str) -> None:
        """Resolve pending confirmations via the extracted ConfirmationManager."""
        outcome = self._confirmation.handle(confirm_intent)

        if outcome.response:
            self._bus.emit_long("response_ready", text=outcome.response)

        if outcome.action_result is not None:
            await self._execute_action(outcome.action_result)
        elif outcome.tool_call is not None:
            tool_call = outcome.tool_call
            try:
                result = self._dispatch_action(
                    tool_call.name, dict(tool_call.arguments),
                )
                response = result or personality.action_done(tool_call.name)
                self._bus.emit_long("response_ready", text=response)
            except Exception as exc:
                logger.error("Confirmed tool execution failed: %s", exc)
                self._bus.emit_long(
                    "response_ready",
                    text=personality.error_response(tool_call.name),
                )

    # ── Intent chaining (post-action follow-ups) ───────────────────────
    _CHAIN_MAP: dict[str, str | dict[str, str]] = {
        "open_app": {
            "code": "Want me to check your git status?",
            "vscode": "Want me to check your git status?",
            "teams": "Should I check your calendar?",
            "outlook": "Want me to read your latest emails?",
            "chrome": "Need me to search for something?",
            "firefox": "Need me to search for something?",
        },
        "search": "Want me to analyze the results on your screen?",
        "screenshot": "Want me to analyze what's on screen?",
        "set_volume": "Should I also pause media?",
        "weather": "Want me to check traffic for your commute?",
        "lock_screen": "I'll keep watch. Say 'Atom' when you're back.",
    }

    def _get_chain_suggestion(self, action: str, args: dict) -> str | None:
        chain = self._CHAIN_MAP.get(action)
        if chain is None:
            return None
        if isinstance(chain, dict):
            target = (args.get("name", "") or args.get("exe", "")).lower()
            for key, suggestion in chain.items():
                if key in target:
                    return suggestion
            return None
        if action == "set_volume":
            pct = int(args.get("percent", 50))
            if pct > 20:
                return None
        return chain

    # ── Action dispatcher ───────────────────────────────────────────────

    async def _execute_action(self, result) -> None:
        from core.execution.behavior_monitor import strip_signing_keys
        from core.security.action_signing import merge_signed_args

        args = result.action_args or {}
        args = merge_signed_args(self._security, result.action, args)

        allowed, reason = self._security.allow_action(result.action, args)
        if not allowed:
            logger.warning("Security BLOCKED action '%s': %s", result.action, reason)
            self._bus.emit_long("response_ready",
                                text=f"Sorry Boss, that action is blocked. {reason}")
            return

        self._security.audit_log(
            result.action,
            f"args={strip_signing_keys(args)}" if args else "",
        )

        response_text = result.response or personality.action_done(result.action)
        dispatch_args = strip_signing_keys(args)

        if result.action in self._FIRE_AND_FORGET_ACTIONS:
            self._bus.emit_long("response_ready", text=response_text)
            try:
                self._dispatch_action(result.action, dispatch_args)
            except Exception as exc:
                logger.error("Background action failed: %s", exc)
            self._emit_chain_suggestion(result.action, dispatch_args)
            return

        if result.action in self._SLOW_ACTIONS:
            self._bus.emit_long("thinking_ack", text="On it, Boss.")

        try:
            response = self._dispatch_action(result.action, dispatch_args)
            if response is not None:
                self._bus.emit_long("response_ready", text=response)
                self._emit_chain_suggestion(result.action, dispatch_args)
                return
        except Exception as exc:
            logger.error("Action failed: %s", exc)
            self._bus.emit_long("response_ready",
                           text=personality.error_response(result.action))
            return

        self._bus.emit_long("response_ready", text=response_text)
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
        suggestion = self._get_chain_suggestion(action, args)
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
    })

    def _dispatch_action(self, action: str, args: dict) -> str | None:
        """Route action to the appropriate handler module.

        Uses a dispatch table for O(1) lookup instead of long if/elif chains.
        Returns the response text, or None to use default response.
        """
        handler = self._ACTION_DISPATCH.get(action)
        if handler is None:
            return None
        return handler(self, action, args)

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
        asyncio.ensure_future(utility_actions.run_timer(seconds, label, self._bus))
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

    def configure_diagnostics(self, *, stt=None, tts=None,
                               metrics=None,
                               local_brain=None,
                               health_monitor=None) -> None:
        self._diagnostics.configure(
            stt=stt, tts=tts, metrics=metrics,
            local_brain=local_brain, health_monitor=health_monitor,
            evolution=self._evolution,
            behavior_tracker=self._behavior_tracker,
        )

    def _do_self_check(self, _action: str, _args: dict) -> str:
        return self._diagnostics.self_check()

    def _do_set_performance_mode(self, _action: str, args: dict) -> str:
        mode = args.get("mode", "lite")
        if mode not in ("full", "lite", "ultra_lite"):
            return f"Unknown mode '{mode}'. Available: full, lite, ultra lite."
        self._bus.emit_long("set_performance_mode", mode=mode)
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

    _ACTION_DISPATCH: dict[str, Any] = {
        "open_app": _do_open_app,
        "close_app": _do_close_app,
        "list_apps": _do_list_apps,
        "search": _do_search,
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
        "set_performance_mode": _do_set_performance_mode,
        "set_brain_profile": _do_set_brain_profile,
        "set_assistant_mode": _do_set_assistant_mode,
        "remember": _do_remember,
        "recall": _do_recall,
        "learn_document": _do_learn_document,
        "run_code": _do_run_code,
        "calculate": _do_calculate,
        "record_workflow": _do_record_workflow,
        "stop_recording": _do_stop_recording,
        "run_workflow": _do_run_workflow,
        "list_workflows": _do_list_workflows,
        "screen_read": _do_screen_read,
        "show_dream_summary": _do_show_dream_summary,
        "set_goal": _do_set_goal,
        "show_goals": _do_show_goals,
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
            self._bus.emit_long("response_ready", text=full_text)
        except Exception:
            self._bus.emit_long(
                "response_ready",
                text=(
                    "Screen reading isn't fully available yet, Boss. "
                    "Say 'take a screenshot' or paste text to the clipboard."
                    + (f" You asked: {_q[:80]}" if _q else "")
                ),
            )

    # ── LLM fallback ────────────────────────────────────────────────

    # ── Smart acknowledgments ───────────────────────────────────────
    _ACK_MAP: list[tuple[list[str], str]] = [
        (["error", "bug", "crash", "exception", "fail", "broken", "not working"],
         "Let me look into that issue."),
        (["weather", "temperature", "forecast", "rain", "humid"],
         "Checking the forecast."),
        (["code", "function", "class", "method", "variable", "syntax"],
         "Looking at the code."),
        (["explain", "what is", "what are", "meaning", "define", "difference"],
         "Let me think about that."),
        (["search", "find", "look up", "google"],
         "Let me find that for you."),
        (["how to", "how do", "steps", "guide", "tutorial"],
         "Let me work through that."),
        (["compare", "versus", "vs", "better", "which one"],
         "Weighing the options."),
        (["history", "when did", "who was", "origin"],
         "Let me recall that."),
    ]
    _GENERIC_ACKS = [
        "On it.",
        "Working on it.",
        "One moment, Boss.",
        "Let me think...",
        "Give me a sec.",
    ]

    def _smart_ack(self, query: str) -> str:
        q = query.lower()
        for keywords, ack in self._ACK_MAP:
            if any(kw in q for kw in keywords):
                return ack
        idx = hash(query) % len(self._GENERIC_ACKS)
        return self._GENERIC_ACKS[idx]

    # ── Repeat query detection ───────────────────────────────────────
    def _check_repeat(self, cache_key: str) -> bool:
        now = time.monotonic()
        self._recent_queries = [
            (q, t) for q, t in self._recent_queries if now - t < 60
        ]
        for prev_q, _ in self._recent_queries:
            if prev_q == cache_key:
                return True
        self._recent_queries.append((cache_key, now))
        if len(self._recent_queries) > 5:
            self._recent_queries = self._recent_queries[-5:]
        return False

    async def _handle_llm_fallback(self, original_text: str,
                                   clean_text: str,
                                   clipboard_injected: bool = False) -> None:
        ack = self._smart_ack(clean_text)
        if clipboard_injected:
            ack = "I see what's on your clipboard. " + ack
        self._bus.emit_long("thinking_ack", text=ack)

        cache_key = clean_text.lower()
        t_lookup = time.perf_counter()

        is_repeat = self._check_repeat(cache_key)
        if is_repeat:
            logger.info("Repeat query detected -- bypassing cache")

        cache_task = asyncio.ensure_future(
            asyncio.get_running_loop().run_in_executor(
                None, self._cache.get, cache_key)
        )
        memory_task = asyncio.ensure_future(
            self._memory.retrieve(clean_text, k=2))
        cached, memory_ctx = await asyncio.gather(cache_task, memory_task)

        if cached and not is_repeat:
            logger.info("Serving from cache (%.1fms)",
                        (time.perf_counter() - t_lookup) * 1000)
            self._bus.emit_long("response_ready", text=cached)
            return

        if is_repeat:
            repeat_hint = (
                "\n\n[SYSTEM NOTE: The user asked this before and wasn't "
                "satisfied with the previous answer. Provide a different, "
                "more thorough response.]"
            )
            original_text = original_text + repeat_hint

        from core.quick_replies import try_quick_reply

        qr = try_quick_reply(clean_text, self._config)
        if qr:
            logger.info("Quick reply served (no LLM)")
            self._bus.emit_long("response_ready", text=qr)
            return

        if self._assistant_mode_mgr is not None:
            if not self._assistant_mode_mgr.allows_llm_fallback():
                logger.info("Assistant mode command_only — skipping LLM")
                self._bus.emit_long(
                    "response_ready",
                    text=self._assistant_mode_mgr.command_only_message(),
                )
                return

        context_bundle = None
        if self._context is not None:
            try:
                context_bundle = self._context.get_bundle()
            except Exception:
                logger.debug("Context bundle retrieval failed", exc_info=True)

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
            self._bus.emit_long("response_ready",
                           text=personality.offline_fallback())
            return

        if not self._brain_enabled:
            logger.info("No local brain (brain.enabled is false)")
            self._bus.emit_long("response_ready",
                           text=personality.offline_fallback())
            return

        history = (
            self._conv_memory.get_pairs()
            if self._conv_memory is not None and self._conv_memory.turn_count > 0
            else self.get_conversation_history()
        )

        self._bus.emit_long(
            "cursor_query",
            text=original_text,
            memory_context=memory_ctx,
            context=context_bundle,
            history=history,
        )

    # ── Contextual follow-up ────────────────────────────────────────

    _FOLLOW_UP_HINTS: list[tuple[list[str], str]] = [
        (["error", "exception", "traceback", "stack trace", "bug"],
         "Want me to read the clipboard or run a screenshot?"),
        (["install", "download", "pip install", "npm install", "setup"],
         "Should I open the terminal for you?"),
        (["documentation", "docs", "reference", "guide", "manual"],
         "Want me to search for the docs?"),
        (["reminder", "later", "tomorrow", "don't forget"],
         "Want me to set a timer for that?"),
    ]

    def _suggest_follow_up(self, query: str, response: str) -> str | None:
        """Return a short follow-up suggestion if the response context warrants one."""
        lower_resp = response.lower()
        lower_query = query.lower()
        combined = lower_query + " " + lower_resp
        for keywords, suggestion in self._FOLLOW_UP_HINTS:
            if any(kw in combined for kw in keywords):
                return suggestion
        return None

    # ── Conversation window ─────────────────────────────────────────

    def record_turn(self, query: str, response: str) -> None:
        """Append a Q&A turn to the rolling conversation window + ConversationMemory."""
        snippet = " ".join(response.split()[:60])
        self._conversation_window.append((query[:100], snippet))
        if len(self._conversation_window) > self._conv_window_max:
            self._conversation_window = self._conversation_window[-self._conv_window_max:]
        if self._conv_memory is not None:
            self._conv_memory.record(query, "llm_response", response)

    def get_conversation_history(self) -> list[tuple[str, str]]:
        return list(self._conversation_window)

    # ── Helpers ─────────────────────────────────────────────────────

    def _status_with_usage(self, base: str) -> str:
        total = self._local_queries + self._llm_queries
        if total <= 0:
            return base
        llm_pct = (self._llm_queries / total) * 100
        return (f"{base} LLM handled {llm_pct:.0f} percent "
                f"of routed queries.")
