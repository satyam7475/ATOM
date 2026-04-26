"""
ATOM -- Cold-start optimizer for faster first response.

Preloads the fast MLX role, warms embeddings, restores a small slice of the
previous conversation, seeds the hot command cache, and replays the most
recent system context so the next boot feels alive immediately.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.command_cache import get_command_cache
from core.persistence_manager import persistence_manager

logger = logging.getLogger("atom.boot.cold_start")

_SNAPSHOT_PATH = Path("logs/cold_start_snapshot.json")
_SNAPSHOT_KEY = "cold_start_snapshot"
_DEFAULT_TOP_COMMANDS = 24
_DEFAULT_SESSION_TURNS = 8
# Keep restored context useful across a long weekend / multi-day break.
# A 6h window meant Monday mornings always booted blind even though the
# user's last Friday session was still the freshest signal we had.
# 72h is long enough to cover a typical time-off gap while still expiring
# stale snapshots after a real break from the machine.
_MAX_RESTORED_CONTEXT_AGE_S = 72 * 3600
_INFO_INTENTS = frozenset({
    "time", "date", "cpu", "ram", "battery", "disk",
    "system_info", "ip", "wifi", "uptime", "top_processes",
    "resource_report", "resource_trend", "app_history",
    "show_reminders", "self_diagnostic", "system_analyze",
    "self_check", "behavior_report",
})


@dataclass
class ColdStartReport:
    elapsed_ms: float
    fast_model_ready: bool
    embeddings_ready: bool
    restored_turns: int
    cached_commands: int
    restored_context_available: bool
    vlm_ready: bool = False
    vlm_warmup_ms: float = 0.0


class ColdStartOptimizer:
    """Warm the startup path and persist a lightweight next-boot snapshot."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None,
        state_manager: Any,
        memory_store: Any,
        intent_engine: Any,
        bus: Any = None,
        local_brain: Any = None,
        conversation_memory: Any = None,
        system_monitor: Any = None,
        vlm_captioner: Any = None,
        skills_registry: Any = None,
        snapshot_path: str | Path | None = None,
    ) -> None:
        self._config = config or {}
        self._state = state_manager
        self._memory = memory_store
        self._intent = intent_engine
        self._bus = bus
        self._local_brain = local_brain
        self._conversation_memory = conversation_memory
        self._system_monitor = system_monitor
        # Optional VLM captioner. When wired, ``warm_up`` triggers
        # ``captioner._load()`` in the executor in parallel with the
        # other warmup tasks so the first wake-word fire doesn't pay
        # the ~2s mlx-vlm cold-load latency on its critical path.
        self._vlm_captioner = vlm_captioner
        # Optional SkillsRegistry. When wired, ``warm_up`` pre-classifies
        # every skill expansion target into the command cache so the
        # second intent pass after a skill match (atom_log.txt L597-599
        # — "self check" classify cost ~150 ms) lands on a hot cache.
        self._skills_registry = skills_registry
        self._snapshot_path = Path(snapshot_path or _SNAPSHOT_PATH)
        self._boot_time = 0.0
        self._restored_snapshot: dict[str, Any] = {}
        self._restored_context_emitted = False
        # Sprint Ω.2 — flipped to True by ``_load_snapshot`` when the
        # persisted snapshot was found but expired. ``warm_up`` then
        # schedules a fresh persist in the background so the *next*
        # boot lands on a current snapshot instead of the same stale
        # one we just dropped.
        self._snapshot_was_stale = False

        persistence_manager.register(_SNAPSHOT_KEY, self._snapshot_path)

    async def warm_up(self) -> ColdStartReport:
        """Preload the hot path pieces needed for the first real query.

        Concurrency model
        -----------------
        Boot has three Metal/GPU consumers that all attach
        ``addCompletedHandler:`` callbacks to the same Metal device queue:
          1. ``_preload_fast_model`` -- MLX (Qwen3-4B-Instruct-2507-4bit)
          2. ``_preload_embeddings`` -- torch.mps (SentenceTransformer)
          3. ``_preload_vlm``        -- mlx-vlm   (SmolVLM-Instruct-4bit)

        Running them concurrently triggers the Apple-internal assertion
        ``-[_MTLCommandBuffer addCompletedHandler:]:1011: failed
        assertion 'Completed handler provided after commit call'`` which
        aborts the process with SIGABRT (exit 134). The first user-driven
        inference after the race is the typical crash site, even when the
        boot greeting itself appears to succeed.

        We therefore split the warmup into two phases:
          * **Phase A (Metal-serial):** fast model -> embeddings -> VLM,
            awaited one at a time so each subsystem owns the Metal queue
            exclusively while it builds its first command buffer.
          * **Phase B (CPU-parallel):** session restore, command cache,
            and intent regex priming are all CPU-bound and gather safely.

        Phase B runs concurrently *with* Phase A so we don't pay for the
        serialization on the wall clock; we only delay the Metal stages
        relative to one another. The total cold-start budget is unchanged
        in the typical case (CPU work finishes during the LLM warmup).
        """
        self._boot_time = time.monotonic()
        self._restored_snapshot = self._load_snapshot()

        async def _metal_serial_warmup() -> tuple[bool, bool, tuple[bool, float]]:
            fast_ok = False
            try:
                fast_ok = bool(await self._preload_fast_model())
            except Exception as exc:
                logger.debug("Cold start fast-model warmup raised: %s", exc)
            emb_ok = False
            try:
                emb_ok = bool(await self._preload_embeddings())
            except Exception as exc:
                logger.debug("Cold start embeddings warmup raised: %s", exc)
            vlm_payload: tuple[bool, float] = (False, 0.0)
            try:
                vlm_payload = await self._preload_vlm()
            except Exception as exc:
                logger.debug("Cold start vlm warmup raised: %s", exc)
            return fast_ok, emb_ok, vlm_payload

        metal_task = asyncio.create_task(_metal_serial_warmup())
        cpu_results = await asyncio.gather(
            self._restore_session(),
            self._cache_top_commands(),
            self._prime_intent_engine(),
            self._cache_skill_expansions(),
            return_exceptions=True,
        )
        try:
            fast_model_ready, embeddings_ready, vlm_payload = await metal_task
        except Exception as exc:
            logger.debug("Cold start metal warmup chain raised: %s", exc)
            fast_model_ready, embeddings_ready, vlm_payload = False, False, (False, 0.0)

        restored_turns = self._coerce_int(cpu_results[0], "session_restore")
        cached_commands = self._coerce_int(cpu_results[1], "command_cache")
        primed_intents = self._coerce_int(cpu_results[2], "intent_warmup")
        cached_skills = self._coerce_int(cpu_results[3], "skill_expansions")
        vlm_ready = bool(vlm_payload[0])
        vlm_ms = float(vlm_payload[1])
        if primed_intents:
            logger.info(
                "Cold start: primed %d intent-engine regex paths", primed_intents,
            )
        if cached_skills:
            logger.info(
                "Cold start: pre-classified %d skill expansion target(s)",
                cached_skills,
            )
        cached_commands += cached_skills
        elapsed_ms = (time.monotonic() - self._boot_time) * 1000
        restored_context_available = bool(
            (self._restored_snapshot or {}).get("system_state"),
        )

        logger.info(
            "Cold start ready in %.0fms (fast=%s embeddings=%s session=%d "
            "cache=%d context=%s vlm=%s vlm_ms=%.0f)",
            elapsed_ms,
            fast_model_ready,
            embeddings_ready,
            restored_turns,
            cached_commands,
            restored_context_available,
            vlm_ready,
            vlm_ms,
        )

        if self._snapshot_was_stale:
            # Background-refresh the snapshot so the *next* boot has a
            # current context payload to land on. Done as a fire-and-
            # forget task with a small initial sleep so the system is
            # quiescent (CPU/RAM steady-state) when we sample it.
            async def _background_refresh_snapshot() -> None:
                try:
                    await asyncio.sleep(2.5)
                    self.persist_snapshot()
                    logger.debug(
                        "Cold start snapshot refreshed in background "
                        "(stale on this boot, fresh for the next)",
                    )
                except Exception:
                    logger.debug(
                        "Background snapshot refresh failed", exc_info=True,
                    )
            try:
                asyncio.create_task(_background_refresh_snapshot())
            except RuntimeError:
                # No running loop (test harness); skip silently.
                pass

        return ColdStartReport(
            elapsed_ms=elapsed_ms,
            fast_model_ready=fast_model_ready,
            embeddings_ready=embeddings_ready,
            restored_turns=restored_turns,
            cached_commands=cached_commands,
            restored_context_available=restored_context_available,
            vlm_ready=vlm_ready,
            vlm_warmup_ms=vlm_ms,
        )

    async def emit_restored_context(self) -> bool:
        """Replay the last lightweight system snapshot after handlers are wired."""
        if self._restored_context_emitted or self._bus is None:
            return False
        snapshot = self._restored_snapshot or self._load_snapshot()
        payload = self._build_context_payload(snapshot)
        if not payload:
            return False

        self._bus.emit_fast("context_snapshot", **payload)
        self._restored_context_emitted = True
        logger.info(
            "Cold start restored context: cpu=%.1f ram=%.1f active_app=%s",
            float(payload.get("cpu", 0.0)),
            float(payload.get("ram", 0.0)),
            payload.get("active_app", ""),
        )
        return True

    def persist_snapshot(self) -> bool:
        """Store a compact boot snapshot for the next launch."""
        snapshot = {
            "saved_at": time.time(),
            "atom_state": getattr(getattr(self._state, "current", None), "value", "idle"),
            "conversation_pairs": self._capture_conversation_pairs(),
            "system_state": self._capture_system_state(),
        }
        try:
            persistence_manager.save_now(_SNAPSHOT_KEY, snapshot)
            self._restored_snapshot = snapshot
            return True
        except Exception:
            logger.debug("Cold start snapshot persist failed", exc_info=True)
            return False

    def _load_snapshot(self) -> dict[str, Any]:
        try:
            loaded = persistence_manager.load(_SNAPSHOT_KEY)
        except Exception:
            logger.debug("Cold start snapshot load failed", exc_info=True)
            return {}

        if not isinstance(loaded, dict):
            return {}

        saved_at = float(loaded.get("saved_at", 0.0) or 0.0)
        if saved_at > 0:
            age_s = max(0.0, time.time() - saved_at)
            # Drop snapshots older than the configured TTL right at load
            # time. Doing it here keeps the rest of the boot pipeline (and
            # the "Cold start ready" report) honest -- previously the
            # staleness check only ran inside ``_build_context_payload``,
            # so the snapshot still showed up as ``context=True`` for the
            # whole boot even when it was actually skipped.
            if age_s > _MAX_RESTORED_CONTEXT_AGE_S:
                # Sprint Ω.2 — log at DEBUG (was INFO) so the boot log
                # doesn't carry a misleading "discarding" line every
                # time Boss takes a long break. We schedule a fresh
                # snapshot in :py:meth:`warm_up` so the next cold
                # start lands on a current snapshot instead of the
                # already-stale-on-arrival one we just dropped.
                logger.debug(
                    "Cold start snapshot stale (age %.0fs > %.0fs); will refresh",
                    age_s,
                    _MAX_RESTORED_CONTEXT_AGE_S,
                )
                self._snapshot_was_stale = True
                return {}
            logger.info("Cold start snapshot found (age %.0fs)", age_s)
        return loaded

    async def _preload_fast_model(self) -> bool:
        if self._local_brain is None:
            return False
        warm_up = getattr(self._local_brain, "warm_up", None)
        if not callable(warm_up):
            return False

        try:
            # ``load_all=True`` aliases the ``primary`` role off the
            # same loaded weights after ``fast`` is materialised (zero
            # extra memory, ~0 ms). Previously ``primary`` was loaded
            # lazily on the first non-fast request, adding ~2 s to the
            # first streaming turn's first-token latency.
            try:
                result = await warm_up(load_all=True)
            except TypeError:
                try:
                    result = await warm_up(model_role="fast")
                except TypeError:
                    result = await warm_up()
            return bool(result)
        except Exception:
            logger.debug("Cold start fast-model preload failed", exc_info=True)
            return False

    async def _preload_embeddings(self) -> bool:
        warm_up = getattr(self._memory, "warm_up_embeddings", None)
        if not callable(warm_up):
            return False
        try:
            return bool(await warm_up())
        except Exception:
            logger.debug("Cold start embedding preload failed", exc_info=True)
            return False

    async def _preload_vlm(self) -> tuple[bool, float]:
        """Load the VLM weights into memory in parallel with the other
        warmup tasks.

        Returning ``(True, ms)`` means ``mlx_vlm.load`` succeeded and the
        captioner is ready to caption. Returning ``(False, ms)`` is the
        fail-open path — captioner stays in its disabled state, and the
        on-wake describe handler will quietly skip until the user fixes
        the underlying issue (missing weights, wrong path, etc.).

        Runs ``captioner._load`` in the default executor because mlx-vlm
        does heavy CPU/GPU work synchronously and we don't want to block
        the asyncio loop for ~2s during the ``asyncio.gather`` above.

        When ``vision.vlm.warm_at_boot`` is ``false`` (the default after
        the JARVIS-grade rewrite), we deliberately skip the boot warmup
        and let the captioner's own lazy ``_load()`` fire on the first
        ``describe()`` call. This frees ~1.6 GB of RAM at idle on
        machines that go a whole session without needing vision —
        exactly the common case for a desktop assistant. The first
        actual vision call pays an extra ~1.5 s, which the user will
        notice once and then never again (subsequent describes hit the
        hot path).
        """
        captioner = self._vlm_captioner
        if captioner is None:
            return False, 0.0
        vlm_cfg = (
            (self._config.get("vision") or {}).get("vlm") or {}
            if isinstance(self._config, dict) else {}
        )
        # Sprint P2.7 (Apr 26 2026): align the code default with the
        # shipping config (settings.json:76 = false). VLM warm-at-boot
        # buys ~1.5s on the *first* describe() at the cost of ~1.6 GB
        # RAM at idle on a 16 GB machine. Default off; users on bigger
        # boxes can opt in via vision.vlm.warm_at_boot=true.
        warm_at_boot = bool(vlm_cfg.get("warm_at_boot", False))
        if not warm_at_boot:
            logger.info(
                "Cold start: VLM warm-at-boot disabled by config "
                "(vision.vlm.warm_at_boot=false); will load lazily on "
                "first describe() call",
            )
            return False, 0.0
        loader = getattr(captioner, "_load", None)
        if not callable(loader):
            return False, 0.0
        is_loaded = getattr(captioner, "is_loaded", False)
        if is_loaded:
            return True, 0.0

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        def _do_load() -> bool:
            try:
                return bool(loader())
            except Exception:
                logger.debug("VLM cold-load raised", exc_info=True)
                return False

        try:
            ok = await loop.run_in_executor(None, _do_load)
        except Exception:
            logger.debug("VLM cold-load dispatch failed", exc_info=True)
            ok = False
        dt_ms = (time.monotonic() - t0) * 1000.0
        if ok:
            logger.info(
                "Cold start: VLM warmed in %.0fms (next wake-word "
                "describe will hit the hot path)", dt_ms,
            )
        else:
            reason_fn = getattr(captioner, "disabled_reason", None)
            reason = ""
            if callable(reason_fn):
                try:
                    reason = str(reason_fn() or "")
                except Exception:
                    reason = ""
            logger.info(
                "Cold start: VLM warmup skipped (%s)",
                reason or "captioner unavailable",
            )
        return ok, dt_ms

    async def _restore_session(self) -> int:
        if self._conversation_memory is None:
            return 0
        if int(getattr(self._conversation_memory, "turn_count", 0) or 0) > 0:
            return 0

        snapshot = self._restored_snapshot or {}
        raw_pairs = snapshot.get("conversation_pairs") or []
        if not isinstance(raw_pairs, list):
            return 0

        restored = 0
        for pair in raw_pairs[-self._session_turn_limit() :]:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            query = str(pair[0] or "").strip()
            response = str(pair[1] or "").strip()
            if not query or not response:
                continue
            try:
                self._conversation_memory.record(query, "restored_session", response)
                restored += 1
            except Exception:
                logger.debug("Cold start session turn restore failed", exc_info=True)
        return restored

    async def _prime_intent_engine(self) -> int:
        """Force every intent-engine regex / heavy import to be compiled
        and resolved before the first real user query.

        The first call to ``IntentEngine.classify`` on a cold process can
        take 80-200ms because each sub-module compiles its module-level
        regex tables and resolves lazy imports on demand. The watchdog
        budgets ``intent_engine`` at 50ms, so the first 5-10 user queries
        repeatedly trip the budget and trigger the recovery path.

        We synchronously dispatch a small set of priming queries chosen
        to exercise every category (info, system, app, media, network,
        memory, productivity, routine, world, file, OS, cognitive, and
        the runtime-mode/meta paths). All run via ``classify_silent`` so
        the priming doesn't pollute the boot log.
        """
        if self._intent is None:
            return 0
        silent = getattr(self._intent, "classify_silent", None)
        if not callable(silent):
            return 0
        priming_queries: tuple[str, ...] = (
            # Meta / runtime-mode / OS self-check
            "are you there",
            "switch to fast mode",
            "self check",
            # Routine triggers (D4)
            "enter focus mode",
            "exit deep work",
            # Productivity / memory recall / world / info
            "what's on my plate",
            "what did I ask earlier",
            "what is the weather in mumbai",
            "what is the time",
            # System / media / desktop / app
            "lower the volume",
            "play some music",
            "show desktop",
            "open safari",
            # File / network / OS
            "find my notes about meeting",
            "check my internet speed",
            "battery status",
            # Cognitive fallback
            "explain quantum tunneling in two lines",
        )
        loop = asyncio.get_running_loop()
        primed = 0

        def _do_one(text: str) -> bool:
            try:
                silent(text)
                return True
            except Exception:
                logger.debug(
                    "Intent priming failed for '%s'", text[:60], exc_info=True,
                )
                return False

        for query in priming_queries:
            try:
                ok = await loop.run_in_executor(None, _do_one, query)
                if ok:
                    primed += 1
            except Exception:
                logger.debug(
                    "Intent priming dispatch failed for '%s'", query[:60],
                    exc_info=True,
                )
        return primed

    async def _cache_top_commands(self) -> int:
        get_top_commands = getattr(self._memory, "get_top_commands", None)
        if not callable(get_top_commands):
            return 0

        try:
            commands = list(get_top_commands(limit=self._top_command_limit()))
        except Exception:
            logger.debug("Cold start top-command lookup failed", exc_info=True)
            return 0

        if not commands:
            return 0

        cmd_cache = get_command_cache()
        cached = 0
        seen: set[str] = set()
        for command in commands:
            text = str(command or "").strip()
            if not text:
                continue
            norm = text.lower()
            if norm in seen:
                continue
            seen.add(norm)
            try:
                silent = getattr(self._intent, "classify_silent", None)
                if callable(silent):
                    result = silent(text)
                else:
                    result = self._intent.classify(text)
            except Exception:
                logger.debug("Cold start classify failed for '%s'", text[:80], exc_info=True)
                continue

            intent = str(getattr(result, "intent", "") or "")
            if intent in {"", "fallback", "confirm", "deny"}:
                continue
            if intent in _INFO_INTENTS:
                continue
            cmd_cache.put(text, result)
            cached += 1
        return cached

    async def _cache_skill_expansions(self) -> int:
        """Pre-classify every distinct skill expansion target.

        Live evidence (atom_log.txt L597-599): when a skill expands
        ("yeah give me a summary" -> "self check"), the router runs a
        SECOND ``intent_engine.classify`` over the expansion text and
        on a cold cache that pass costs ~150 ms — pushing the
        ``intent_classify`` budget past 250 ms and missing the
        fast-path SLA. Skill expansion targets are bounded (~30 today)
        and identical across boots, so we classify them all up front
        and store them in the same command cache the router checks
        before re-classifying. Result: every future skill expansion
        lands on a hot cache (sub-ms).
        """
        registry = self._skills_registry
        if registry is None:
            return 0
        targets_fn = getattr(registry, "expansion_targets", None)
        if not callable(targets_fn):
            return 0
        try:
            targets = list(targets_fn())
        except Exception:
            logger.debug("Cold start: skill targets lookup failed", exc_info=True)
            return 0
        if not targets:
            return 0

        cmd_cache = get_command_cache()
        loop = asyncio.get_running_loop()
        silent = getattr(self._intent, "classify_silent", None)

        def _classify_one(text: str) -> Any:
            try:
                if callable(silent):
                    return silent(text)
                return self._intent.classify(text)
            except Exception:
                logger.debug(
                    "Cold start skill classify failed for '%s'",
                    text[:80],
                    exc_info=True,
                )
                return None

        cached = 0
        for text in targets:
            try:
                result = await loop.run_in_executor(None, _classify_one, text)
            except Exception:
                logger.debug(
                    "Cold start skill classify dispatch failed for '%s'",
                    text[:80],
                    exc_info=True,
                )
                continue
            if result is None:
                continue
            intent = str(getattr(result, "intent", "") or "")
            if intent in {"", "fallback"}:
                continue
            try:
                cmd_cache.put(text, result, force=True)
                cached += 1
            except Exception:
                logger.debug(
                    "Cold start skill cache put failed for '%s'",
                    text[:80],
                    exc_info=True,
                )
        return cached

    def _capture_conversation_pairs(self) -> list[list[str]]:
        if self._conversation_memory is None:
            return []

        getter = getattr(self._conversation_memory, "get_pairs", None)
        if not callable(getter):
            return []

        try:
            pairs = list(getter())
        except Exception:
            logger.debug("Cold start conversation capture failed", exc_info=True)
            return []

        out: list[list[str]] = []
        for query, response in pairs[-self._session_turn_limit() :]:
            q = str(query or "").strip()
            r = str(response or "").strip()
            if q and r:
                out.append([q, r])
        return out

    def _capture_system_state(self) -> dict[str, Any]:
        if self._system_monitor is None:
            return {}
        getter = getattr(self._system_monitor, "get_system_state", None)
        if not callable(getter):
            return {}
        try:
            state = getter()
            return state if isinstance(state, dict) else {}
        except Exception:
            logger.debug("Cold start system snapshot failed", exc_info=True)
            return {}

    def _build_context_payload(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        system_state = snapshot.get("system_state") or {}
        if not isinstance(system_state, dict) or not system_state:
            return {}

        ts = float(
            system_state.get("ts")
            or snapshot.get("saved_at")
            or 0.0,
        )
        if ts > 0:
            age_s = max(0.0, time.time() - ts)
            if age_s > _MAX_RESTORED_CONTEXT_AGE_S:
                # Should already have been filtered in _load_snapshot;
                # defensive in case a snapshot was injected via another
                # path. Log at DEBUG to avoid duplicating the load-time
                # warning seen in production boot logs.
                logger.debug(
                    "Cold start context snapshot too old (%.0fs); skipping restore",
                    age_s,
                )
                return {}
            dt = datetime.fromtimestamp(ts)
        else:
            dt = datetime.now()

        cpu = float(system_state.get("cpu_percent", system_state.get("cpu", 0.0)) or 0.0)
        ram = float(system_state.get("ram_percent", system_state.get("ram", 0.0)) or 0.0)
        active_app = str(
            system_state.get("foreground_window_title")
            or system_state.get("active_app")
            or "",
        )[:120]
        weekday = dt.weekday()

        return {
            "time_of_day": self._time_of_day(dt.hour),
            "hour": dt.hour,
            "cpu": cpu,
            "ram": ram,
            "idle_minutes": 0.0,
            "active_app": active_app,
            "is_weekday": weekday < 5,
            "weekday": weekday,
        }

    def _top_command_limit(self) -> int:
        return _DEFAULT_TOP_COMMANDS

    def _session_turn_limit(self) -> int:
        return _DEFAULT_SESSION_TURNS

    @staticmethod
    def _time_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"

    @staticmethod
    def _coerce_bool(value: Any, stage: str) -> bool:
        if isinstance(value, Exception):
            logger.debug("Cold start stage failed: %s (%s)", stage, value)
            return False
        return bool(value)

    @staticmethod
    def _coerce_int(value: Any, stage: str) -> int:
        if isinstance(value, Exception):
            logger.debug("Cold start stage failed: %s (%s)", stage, value)
            return 0
        try:
            return int(value)
        except Exception:
            return 0


__all__ = ["ColdStartOptimizer", "ColdStartReport"]
