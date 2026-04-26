"""ATOM — Idle Maintenance (Sprint Ω.12, Apr 27 2026).

Production-grade memory hygiene that complements
:class:`core.memory_governor.MemoryGovernor`. The governor reacts to
unified-memory *pressure* by evicting roles. This module proactively
keeps steady-state RSS low **before** pressure ever builds, by doing
three cheap things during true idle windows:

1. **One-shot ``gc.freeze()`` after boot.** Once the heavy modules
   (MLX runtime, tokenizers, system_profile, config dicts, the loaded
   LLM weights' Python wrappers) have settled, we move them to the
   permanent generation. The Python cycle collector then skips them on
   every subsequent gen-0/gen-1 sweep — saving 3-8 ms of jitter per
   sweep that occasionally trips first-token latency. Refcount-based
   freeing is unaffected; this is purely a "stop scanning what will
   never become garbage" optimisation.

2. **Periodic ``gc.collect(1)`` on idle.** When ATOM has been idle for
   ``idle_threshold_s`` and no turn is in flight, we run a generation-1
   sweep. Cheap (sub-millisecond on the post-freeze working set) and
   reclaims any short-lived cycles created during the previous turn
   (httpx pools, asyncio tasks, etc.).

3. **Periodic ``mx.clear_cache()`` on idle.** The MLX Metal allocator
   keeps a high-water-mark buffer pool. On a sustained idle window we
   release it, returning ~200-600 MB of unified memory to macOS without
   unloading the model. The next turn pays a sub-50 ms re-mmap; the
   user-perceived latency cost is zero because the next turn was going
   to allocate fresh tensors anyway.

Plus a one-time ``gc.set_threshold()`` tweak at construction:
``(gen0=2000, gen1=25, gen2=25)`` (defaults are ``(700, 10, 10)``).
ATOM allocates a lot of short-lived dicts during a turn; bumping gen-0
cuts cycle-collector frequency in half during active turns and gives
back ~1-2 ms at p99.

All knobs live under ``idle_maintenance`` in ``config/settings.json``.
The module is event-driven (subscribes to ``speech_final`` /
``response_ready`` / ``turn_complete``) so it always knows when ATOM
is busy. ``maybe_tick()`` is the single entry point the periodic-
maintenance loop in ``main.py`` calls.

Public surface::

    idle = IdleMaintenance(
        config=config,
        bus=bus,
        clear_metal_cache=lambda: local_brain.clear_metal_cache(),
        is_busy=lambda: state.current.value not in {"idle", "listening"},
    )
    idle.start()                      # registers gc.set_threshold + bus subs
    idle.schedule_freeze_after_boot() # one-shot, runs on event-loop later
    idle.maybe_tick()                 # called every ~30 s by the maintenance loop

The module is idempotent and safe — calling ``maybe_tick()`` while busy
or before the cooldown window is a no-op.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger("atom.idle_maintenance")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus


# ── defaults (tuned for 16 GB MacBook Air M5) ────────────────────────

_DEFAULT_FREEZE_AFTER_BOOT_S = 60.0
_DEFAULT_IDLE_THRESHOLD_S = 120.0
_DEFAULT_TICK_INTERVAL_S = 30.0
_DEFAULT_MIN_ACTION_INTERVAL_S = 60.0
_DEFAULT_GC_THRESHOLDS: tuple[int, int, int] = (2000, 25, 25)
_DEFAULT_CLEAR_MLX = True
_DEFAULT_LOG_ACTIONS = True

# Events that count as "ATOM is doing work right now". Subscribing to
# these lets us track wall-clock idle without polling state.
_ACTIVITY_EVENTS = (
    "speech_final",
    "partial_response",
    "response_ready",
    "turn_started",
    "turn_complete",
)


class IdleMaintenance:
    """Periodic Python + MLX cache hygiene driven by idle windows.

    Threading: ``maybe_tick`` is called from the asyncio maintenance
    loop on the main thread. ``_on_activity`` runs on whatever thread
    the event bus uses; we only mutate ``_last_activity_ts`` (a float)
    under a lock. ``gc.collect`` and ``clear_metal_cache`` run inline
    on the calling thread; both are cheap (<5 ms) on the post-freeze
    object set.
    """

    def __init__(
        self,
        *,
        config: dict | None = None,
        bus: "AsyncEventBus | None" = None,
        clear_metal_cache: Callable[[], None] | None = None,
        is_busy: Callable[[], bool] | None = None,
    ) -> None:
        cfg = (config or {}).get("idle_maintenance", {}) or {}
        self._enabled = bool(cfg.get("enabled", True))
        self._freeze_after_boot_s = float(
            cfg.get("freeze_after_boot_s", _DEFAULT_FREEZE_AFTER_BOOT_S),
        )
        self._idle_threshold_s = float(
            cfg.get("idle_threshold_s", _DEFAULT_IDLE_THRESHOLD_S),
        )
        self._tick_interval_s = float(
            cfg.get("tick_interval_s", _DEFAULT_TICK_INTERVAL_S),
        )
        self._min_action_interval_s = float(
            cfg.get("min_action_interval_s", _DEFAULT_MIN_ACTION_INTERVAL_S),
        )
        gc_t = (
            int(cfg.get("gc_threshold_gen0", _DEFAULT_GC_THRESHOLDS[0])),
            int(cfg.get("gc_threshold_gen1", _DEFAULT_GC_THRESHOLDS[1])),
            int(cfg.get("gc_threshold_gen2", _DEFAULT_GC_THRESHOLDS[2])),
        )
        # Sanity-check thresholds; bad config falls back to defaults so
        # boot never aborts because of a typo here.
        if any(v <= 0 for v in gc_t):
            logger.warning(
                "IdleMaintenance: invalid gc thresholds %s; using defaults %s",
                gc_t, _DEFAULT_GC_THRESHOLDS,
            )
            gc_t = _DEFAULT_GC_THRESHOLDS
        self._gc_thresholds = gc_t
        self._clear_mlx_on_idle = bool(
            cfg.get("clear_mlx_cache_on_idle", _DEFAULT_CLEAR_MLX),
        )
        self._log_actions = bool(cfg.get("log_actions", _DEFAULT_LOG_ACTIONS))

        self._bus = bus
        self._clear_metal_cache = clear_metal_cache
        self._is_busy = is_busy

        self._lock = threading.Lock()
        now = time.monotonic()
        self._last_activity_ts: float = now
        self._last_action_ts: float = 0.0
        self._frozen: bool = False
        self._started: bool = False

        # Lifetime stats — exposed via ``diagnostics()`` for the dashboard.
        self._gen1_collections: int = 0
        self._mlx_clears: int = 0
        self._frozen_object_count: int = 0
        self._last_action_kind: str = ""

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Apply gc threshold tweak and subscribe to activity events.

        Idempotent. Safe to call before the bus exists (the threshold
        tweak still applies; bus subscription is skipped). The freeze
        is **not** triggered here — call :meth:`schedule_freeze_after_boot`
        from the asyncio loop after boot warmup completes.
        """
        if self._started:
            return
        if not self._enabled:
            logger.info("IdleMaintenance: disabled by config")
            self._started = True
            return

        try:
            gc.set_threshold(*self._gc_thresholds)
            logger.info(
                "IdleMaintenance: gc thresholds set to %s "
                "(default was (700, 10, 10))",
                self._gc_thresholds,
            )
        except Exception:
            logger.debug("IdleMaintenance: gc.set_threshold failed", exc_info=True)

        if self._bus is not None:
            for event in _ACTIVITY_EVENTS:
                try:
                    self._bus.on(event, self._on_activity)
                except Exception:
                    logger.debug(
                        "IdleMaintenance: bus.on(%s) failed", event, exc_info=True,
                    )

        self._started = True
        logger.info(
            "IdleMaintenance: armed "
            "(idle>=%.0fs, tick=%.0fs, min_gap=%.0fs, "
            "clear_mlx=%s, freeze_after_boot=%.0fs)",
            self._idle_threshold_s, self._tick_interval_s,
            self._min_action_interval_s, self._clear_mlx_on_idle,
            self._freeze_after_boot_s,
        )

    def stop(self) -> None:
        """Unsubscribe from the bus. ``gc`` thresholds are not reset
        because reverting them mid-runtime would just churn the GC."""
        if self._bus is None or not self._started:
            return
        for event in _ACTIVITY_EVENTS:
            try:
                off = getattr(self._bus, "off", None)
                if callable(off):
                    off(event, self._on_activity)
            except Exception:
                logger.debug(
                    "IdleMaintenance: bus.off(%s) failed", event, exc_info=True,
                )

    # ── activity tracking ───────────────────────────────────────────

    def _on_activity(self, **_payload: Any) -> None:
        """Bus subscriber. Records the current monotonic time so the
        next tick can compute idle duration. The payload is irrelevant
        — any mention on these channels means ATOM is alive."""
        with self._lock:
            self._last_activity_ts = time.monotonic()

    def mark_active(self) -> None:
        """Public hook for callers without a bus (tests, scripts)."""
        self._on_activity()

    # ── one-shot freeze ─────────────────────────────────────────────

    def schedule_freeze_after_boot(self, *, delay_s: float | None = None) -> None:
        """Run :meth:`freeze_now` after a delay using a daemon timer.

        Designed to be called from the boot path. The delay defaults
        to ``freeze_after_boot_s`` from config; tests can override it
        with ``delay_s=0.05``. Uses a daemon ``threading.Timer`` so it
        survives a sync boot path that hasn't reached the asyncio loop
        yet, and so it never blocks shutdown.
        """
        if not self._enabled or self._frozen:
            return
        delay = float(delay_s) if delay_s is not None else self._freeze_after_boot_s
        if delay <= 0:
            self.freeze_now()
            return
        timer = threading.Timer(delay, self.freeze_now)
        timer.daemon = True
        timer.name = "idle-maintenance-freeze"
        timer.start()

    def freeze_now(self) -> int:
        """Move all currently-tracked objects to the permanent
        generation. Returns the number of objects frozen.

        ``gc.freeze()`` is safe even at runtime: refcount-based freeing
        keeps working for frozen objects (so anything that loses its
        last reference still releases immediately), the cycle collector
        just stops scanning them. We freeze exactly once per process —
        repeated freezes would re-promote post-boot objects and defeat
        the purpose of normal cycle collection on transient state.
        """
        if not self._enabled or self._frozen:
            return 0
        try:
            gc.collect()
            gc.freeze()
            count = gc.get_freeze_count()
        except Exception:
            logger.exception("IdleMaintenance: gc.freeze failed")
            return 0
        self._frozen = True
        self._frozen_object_count = count
        if self._log_actions:
            logger.info(
                "IdleMaintenance: gc.freeze complete (%d objects in "
                "permanent generation; cycle collector will skip them)",
                count,
            )
        return count

    # ── periodic tick ───────────────────────────────────────────────

    def maybe_tick(self) -> dict[str, Any]:
        """Single entry point for the periodic maintenance loop.

        Returns a small status dict so the caller can log or expose it.
        Decisions:

        - If disabled, busy, or cooldown not met → no-op.
        - If idle ≥ ``idle_threshold_s`` and the last action is ≥
          ``min_action_interval_s`` ago → run gen-1 collect. Optionally
          run ``mx.clear_cache()`` too. Update ``last_action_ts``.
        """
        result: dict[str, Any] = {"action": "skip", "reason": "disabled"}
        if not self._enabled or not self._started:
            return result

        now = time.monotonic()
        with self._lock:
            idle_s = now - self._last_activity_ts
            since_action = now - self._last_action_ts
        result["idle_s"] = round(idle_s, 1)
        result["since_action_s"] = round(since_action, 1)

        if idle_s < self._idle_threshold_s:
            result["action"] = "skip"
            result["reason"] = "not_idle"
            return result
        if since_action < self._min_action_interval_s:
            result["action"] = "skip"
            result["reason"] = "cooldown"
            return result
        if self._is_busy is not None:
            try:
                if self._is_busy():
                    result["action"] = "skip"
                    result["reason"] = "busy"
                    return result
            except Exception:
                logger.debug(
                    "IdleMaintenance: is_busy probe raised, treating as busy",
                    exc_info=True,
                )
                result["action"] = "skip"
                result["reason"] = "busy_probe_error"
                return result

        # Run the actions. Each is wrapped so a failure in one does
        # not block the other.
        actions: list[str] = []
        gen1_freed = self._run_gen1_collect()
        if gen1_freed >= 0:
            actions.append(f"gc1={gen1_freed}")
            self._gen1_collections += 1

        mlx_cleared = False
        if self._clear_mlx_on_idle and self._clear_metal_cache is not None:
            mlx_cleared = self._run_clear_metal_cache()
            if mlx_cleared:
                actions.append("mlx_clear")
                self._mlx_clears += 1

        with self._lock:
            self._last_action_ts = now
        self._last_action_kind = ",".join(actions) if actions else "noop"

        if self._log_actions and actions:
            logger.info(
                "IdleMaintenance: idle=%.0fs -> ran %s",
                idle_s, ", ".join(actions),
            )

        result["action"] = "ran"
        result["actions"] = actions
        return result

    # ── action helpers ──────────────────────────────────────────────

    def _run_gen1_collect(self) -> int:
        """Run a generation-1 sweep. Returns the number of objects
        collected (negative on error)."""
        try:
            return gc.collect(1)
        except Exception:
            logger.debug("IdleMaintenance: gc.collect(1) failed", exc_info=True)
            return -1

    def _run_clear_metal_cache(self) -> bool:
        """Invoke the injected MLX cache clearer. Errors are swallowed
        so a flaky MLX runtime never crashes the maintenance loop."""
        fn = self._clear_metal_cache
        if fn is None:
            return False
        try:
            fn()
            return True
        except Exception:
            logger.debug(
                "IdleMaintenance: clear_metal_cache callback raised",
                exc_info=True,
            )
            return False

    # ── diagnostics ─────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            idle_s = time.monotonic() - self._last_activity_ts
            since_action = time.monotonic() - self._last_action_ts
        return {
            "enabled": self._enabled,
            "started": self._started,
            "frozen": self._frozen,
            "frozen_object_count": self._frozen_object_count,
            "idle_s": round(idle_s, 1),
            "since_last_action_s": round(since_action, 1),
            "gen1_collections_total": self._gen1_collections,
            "mlx_clears_total": self._mlx_clears,
            "last_action_kind": self._last_action_kind,
            "thresholds": {
                "idle_threshold_s": self._idle_threshold_s,
                "tick_interval_s": self._tick_interval_s,
                "min_action_interval_s": self._min_action_interval_s,
                "freeze_after_boot_s": self._freeze_after_boot_s,
                "gc_thresholds": list(self._gc_thresholds),
            },
        }


__all__ = ["IdleMaintenance"]
