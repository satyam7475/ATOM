"""ATOM -- Memory Governor (Sprint Ω.4.C, Apr 26 2026).

Per-role tunable eviction order on top of ``SiliconGovernor``.

ATOM on a 16 GB MacBook Air M5 routinely pushes pressure tier 2 (82 % unified
memory) when the 8 B primary, 4 B speculative draft, persona KV cache,
embedding warm cache, and SmolVLM are all warm at the same time. The silicon
governor will *warn* in that situation, but the order in which subsystems
release memory was previously implicit (whoever happened to idle out first).

This module makes the eviction order **explicit and tunable per-role**.
Roles register an ``evict()`` callback at startup; the governor watches
``silicon_stats_update`` events and walks the configured ``eviction_order``
on tier escalation. The last role in the list is treated as *sacred* —
it is dropped last (only at tier 3, the "we are truly out of memory"
lever), so cheaper-to-rebuild roles get pushed out first.

Tier policy (all percentages tunable in ``config/settings.json``):

    tier 0 — pressure < tier1_threshold_pct        → no action
    tier 1 — tier1..tier2                          → evict first ~⅓ of order
    tier 2 — tier2..tier3                          → evict first ~⅔ of order
    tier 3 — pressure ≥ tier3_threshold_pct        → evict everything,
                                                      sacred role included

Hysteresis: once a tier triggers, the governor remembers it and only resets
the tier when memory drops below ``threshold - rewarm_hysteresis_pct``.
This prevents thrashing when pressure dances around a threshold.

Public surface::

    governor = MemoryGovernor(bus, config)
    governor.register("draft_model", evict=brain.unload_draft)
    governor.register("smolvlm", evict=vlm.unload)
    governor.register("persona_kv_cache", evict=brain.clear_persona_cache)
    governor.start()                  # subscribes to silicon_stats_update

Eviction is best-effort; an exception inside an ``evict`` callback is logged
but does not stop the governor walking down the order.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger("atom.memory_governor")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus


# ── defaults ─────────────────────────────────────────────────────────

# Conservative defaults sized for a 16 GB MacBook Air M5. Tier 1 lands a
# little above the SiliconGovernor's ``memory_threshold_pct=78`` warning so
# the human-visible "memory pressure" log line still fires *before* we evict
# anything. Tier 3 is the floor at which we are willing to drop the persona
# KV cache; below it the cache stays warm.
_DEFAULT_TIER1_PCT = 80.0
_DEFAULT_TIER2_PCT = 86.0
_DEFAULT_TIER3_PCT = 92.0
_DEFAULT_HYSTERESIS_PCT = 6.0
_DEFAULT_EVICTION_ORDER: tuple[str, ...] = (
    "smolvlm",
    "whisper_confirmer",
    "draft_model",
    "embeddings_warm_cache",
    "persona_kv_cache",
)


# ── role registry ────────────────────────────────────────────────────


@dataclass(slots=True)
class _Role:
    """A single evictable subsystem registered with the governor."""

    name: str
    evict: Callable[[], None]
    rewarm: Callable[[], None] | None = None
    last_evicted_at: float = 0.0
    is_evicted: bool = False


# ── public class ─────────────────────────────────────────────────────


class MemoryGovernor:
    """Per-role tunable eviction in response to unified-memory pressure.

    Subscribes to ``silicon_stats_update`` events from
    :class:`core.silicon_governor.SiliconGovernor` and walks the configured
    eviction order on tier escalation. Stateful: tracks which roles are
    currently evicted so a steady-state high-pressure regime does not
    re-fire ``evict()`` every poll.
    """

    def __init__(
        self,
        bus: "AsyncEventBus | None" = None,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        cfg = (config or {}).get("memory_governor", {}) or {}
        self._enabled = bool(cfg.get("enabled", True))
        self._tier1 = float(cfg.get("tier1_threshold_pct", _DEFAULT_TIER1_PCT))
        self._tier2 = float(cfg.get("tier2_threshold_pct", _DEFAULT_TIER2_PCT))
        self._tier3 = float(cfg.get("tier3_threshold_pct", _DEFAULT_TIER3_PCT))
        self._hysteresis = float(
            cfg.get("rewarm_hysteresis_pct", _DEFAULT_HYSTERESIS_PCT),
        )
        order = cfg.get("eviction_order") or _DEFAULT_EVICTION_ORDER
        self._order: tuple[str, ...] = tuple(str(name) for name in order)

        # Validate threshold ordering. Bad config falls back to defaults
        # rather than aborting boot.
        if not (
            0.0 < self._tier1 <= self._tier2 <= self._tier3 < 100.0
        ):
            logger.warning(
                "MemoryGovernor: invalid tier thresholds "
                "(tier1=%.1f, tier2=%.1f, tier3=%.1f); "
                "falling back to defaults",
                self._tier1, self._tier2, self._tier3,
            )
            self._tier1 = _DEFAULT_TIER1_PCT
            self._tier2 = _DEFAULT_TIER2_PCT
            self._tier3 = _DEFAULT_TIER3_PCT

        self._roles: dict[str, _Role] = {}
        self._lock = threading.Lock()
        self._current_tier: int = 0
        self._evictions_total: int = 0
        self._last_event_at: float = 0.0
        self._subscribed: bool = False

    # ── properties ──────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def current_tier(self) -> int:
        return self._current_tier

    @property
    def eviction_order(self) -> tuple[str, ...]:
        return self._order

    @property
    def evictions_total(self) -> int:
        return self._evictions_total

    # ── role registration ───────────────────────────────────────────

    def register(
        self,
        name: str,
        evict: Callable[[], None],
        *,
        rewarm: Callable[[], None] | None = None,
    ) -> None:
        """Register an evictable subsystem.

        ``name`` MUST appear in the configured ``eviction_order`` for the
        governor to ever call ``evict``. Roles registered outside the order
        are kept (so the registry stays a complete map of who can be
        evicted) but are never walked. We log a warning so a typo in the
        config or a forgotten config update is visible at boot.
        """
        if not callable(evict):
            raise TypeError(
                f"MemoryGovernor.register({name!r}): evict must be callable",
            )
        with self._lock:
            self._roles[name] = _Role(name=name, evict=evict, rewarm=rewarm)
        if name not in self._order:
            logger.warning(
                "MemoryGovernor: registered role %r is not in "
                "eviction_order=%s; it will never be evicted automatically",
                name, list(self._order),
            )
        else:
            logger.info(
                "MemoryGovernor: registered evictable role %r (position %d)",
                name, self._order.index(name),
            )

    def unregister(self, name: str) -> None:
        with self._lock:
            self._roles.pop(name, None)

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Subscribe to silicon stats events.

        Idempotent. The governor keeps working without a bus (e.g. in
        tests) -- callers can drive it directly via :meth:`on_stats`.
        """
        if not self._enabled:
            logger.info("MemoryGovernor: disabled by config")
            return
        if self._bus is None or self._subscribed:
            return
        try:
            self._bus.on("silicon_stats_update", self._handle_event)
            self._subscribed = True
            logger.info(
                "MemoryGovernor: watching silicon_stats_update "
                "(thresholds: tier1=%.0f%%, tier2=%.0f%%, tier3=%.0f%%, "
                "hysteresis=%.0f%%; order=%s)",
                self._tier1, self._tier2, self._tier3, self._hysteresis,
                list(self._order),
            )
        except Exception:
            logger.exception("MemoryGovernor: failed to subscribe to bus")

    def stop(self) -> None:
        if self._bus is None or not self._subscribed:
            return
        try:
            off = getattr(self._bus, "off", None)
            if callable(off):
                off("silicon_stats_update", self._handle_event)
        except Exception:
            logger.debug("MemoryGovernor: failed to unsubscribe", exc_info=True)
        self._subscribed = False

    def shutdown(self) -> None:
        self.stop()

    # ── event entry points ─────────────────────────────────────────

    def _handle_event(self, **payload: Any) -> None:
        """Async-event-bus subscriber. Extracts memory_pct and forwards."""
        stats = payload.get("stats") or {}
        if not isinstance(stats, dict):
            return
        mem_pct = stats.get("memory_pct")
        if mem_pct is None:
            return
        try:
            self.on_stats(float(mem_pct))
        except Exception:
            logger.exception("MemoryGovernor: on_stats failed")

    def on_stats(self, memory_pct: float) -> int:
        """Public driver: feed a memory-pressure reading.

        Returns the tier the governor is now in (0-3). Visible for testing
        without a real ``SiliconGovernor``.
        """
        if not self._enabled:
            return 0
        self._last_event_at = time.monotonic()
        new_tier = self._classify_tier(memory_pct)
        if new_tier > self._current_tier:
            self._escalate(new_tier, memory_pct)
        elif new_tier < self._current_tier:
            self._maybe_relax(new_tier, memory_pct)
        return new_tier

    # ── internal: tier classification ──────────────────────────────

    def _classify_tier(self, memory_pct: float) -> int:
        # Hysteresis: once we're in tier N, only step down when pressure
        # drops below ``threshold - hysteresis``. This prevents flapping
        # when memory dances around a threshold during steady state.
        if self._current_tier >= 3:
            if memory_pct < self._tier3 - self._hysteresis:
                return self._classify_fresh(memory_pct)
            return 3
        if self._current_tier >= 2:
            if memory_pct < self._tier2 - self._hysteresis:
                return self._classify_fresh(memory_pct)
            if memory_pct >= self._tier3:
                return 3
            return 2
        if self._current_tier >= 1:
            if memory_pct < self._tier1 - self._hysteresis:
                return 0
            if memory_pct >= self._tier3:
                return 3
            if memory_pct >= self._tier2:
                return 2
            return 1
        return self._classify_fresh(memory_pct)

    def _classify_fresh(self, memory_pct: float) -> int:
        if memory_pct >= self._tier3:
            return 3
        if memory_pct >= self._tier2:
            return 2
        if memory_pct >= self._tier1:
            return 1
        return 0

    # ── internal: action ────────────────────────────────────────────

    def _escalate(self, tier: int, memory_pct: float) -> None:
        target = self._eviction_count_for(tier)
        if target <= 0:
            self._current_tier = tier
            return
        logger.warning(
            "MemoryGovernor: pressure %.1f%% -> tier %d, evicting up to %d role(s)",
            memory_pct, tier, target,
        )
        evicted_now: list[str] = []
        # Walk in declared order, evicting the first ``target`` roles that
        # are not already evicted.
        for name in self._order[:target]:
            with self._lock:
                role = self._roles.get(name)
            if role is None:
                continue
            if role.is_evicted:
                continue
            if self._safe_evict(role):
                evicted_now.append(name)
        self._current_tier = tier
        if evicted_now and self._bus is not None:
            try:
                self._bus.emit_fast(
                    "memory_governor_evicted",
                    tier=tier,
                    memory_pct=memory_pct,
                    roles=evicted_now,
                )
            except Exception:
                logger.debug(
                    "MemoryGovernor: failed to emit eviction event",
                    exc_info=True,
                )

    def _maybe_relax(self, tier: int, memory_pct: float) -> None:
        # When pressure relaxes, mark roles below the current tier as
        # eligible for re-warm. We don't auto-rewarm here -- subsystems
        # warm themselves on their next access path. The governor just
        # clears the ``is_evicted`` flag so a future spike can evict
        # them again. If a role registered an explicit ``rewarm`` callback,
        # call it on relax for the inverse symmetry.
        target = self._eviction_count_for(tier)
        rewarmed_now: list[str] = []
        for idx, name in enumerate(self._order):
            if idx < target:
                continue
            with self._lock:
                role = self._roles.get(name)
            if role is None:
                continue
            if not role.is_evicted:
                continue
            role.is_evicted = False
            if role.rewarm is not None:
                try:
                    role.rewarm()
                    rewarmed_now.append(name)
                except Exception:
                    logger.debug(
                        "MemoryGovernor: rewarm of %r raised",
                        name, exc_info=True,
                    )
        if tier < self._current_tier:
            logger.info(
                "MemoryGovernor: pressure %.1f%% -> tier %d "
                "(relaxed from %d, %d role(s) marked re-warmable)",
                memory_pct, tier, self._current_tier, len(rewarmed_now),
            )
        self._current_tier = tier

    def _eviction_count_for(self, tier: int) -> int:
        """Number of leading entries in ``eviction_order`` to evict at ``tier``.

        - Tier 0: 0
        - Tier 1: ceil(N/3), at least 1 (when N >= 1)
        - Tier 2: ceil(2N/3), at least 1
        - Tier 3: N (everything, sacred role included -- tier 3 is the
                  "we are out of memory" lever; even the persona KV cache
                  goes, accepting an ~8 s re-prefill on the next turn)

        ``eviction_order`` semantics: the *order* matters (sacred role is
        last because we want it dropped last); at tier 3 we still drop it.
        """
        n = len(self._order)
        if n == 0 or tier <= 0:
            return 0
        if tier >= 3:
            return n
        if tier == 2:
            return max(1, math.ceil((2 * n) / 3))
        return max(1, math.ceil(n / 3))

    def _safe_evict(self, role: _Role) -> bool:
        try:
            t0 = time.perf_counter()
            role.evict()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            role.is_evicted = True
            role.last_evicted_at = time.monotonic()
            self._evictions_total += 1
            logger.info(
                "MemoryGovernor: evicted role=%r in %.0f ms",
                role.name, elapsed_ms,
            )
            return True
        except Exception:
            logger.exception(
                "MemoryGovernor: evict callback for %r raised",
                role.name,
            )
            return False

    # ── diagnostics ────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            registered = sorted(self._roles.keys())
            evicted = sorted(
                name for name, r in self._roles.items() if r.is_evicted
            )
        return {
            "enabled": self._enabled,
            "current_tier": self._current_tier,
            "thresholds": {
                "tier1": self._tier1,
                "tier2": self._tier2,
                "tier3": self._tier3,
                "hysteresis": self._hysteresis,
            },
            "eviction_order": list(self._order),
            "registered_roles": registered,
            "evicted_roles": evicted,
            "evictions_total": self._evictions_total,
            "last_event_at": self._last_event_at,
        }


__all__ = ["MemoryGovernor"]
