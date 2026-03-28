"""Sliding-window action frequency and repetition checks."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger("atom.behavior_monitor")


class BehaviorMonitor:
    """Track recent actions; enforce per-window caps."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        sec = cfg.get("security", {})
        bm = sec.get("behavior_monitor") or {}
        self._enabled = bool(bm.get("enabled", True))
        self._window_s = float(bm.get("window_s", 60.0))
        self._max_per_window = int(bm.get("max_actions_per_window", 40))
        self._max_repeat = int(bm.get("max_same_action_repeats", 15))
        paranoid = bm.get("paranoid") or {}
        self._p_window_s = float(paranoid.get("window_s", self._window_s))
        self._p_max_per_window = int(paranoid.get("max_actions_per_window", 20))
        self._p_max_repeat = int(paranoid.get("max_same_action_repeats", 8))
        self._per_action: dict[str, deque[float]] = defaultdict(deque)
        self._global_ts: deque[float] = deque()
        self._lock_mode = "open"

    def set_lock_mode(self, mode: str) -> None:
        self._lock_mode = mode

    def check_action_allowed(
        self,
        action: str,
        args: dict | None,
        *,
        policy_context: str = "execute",
    ) -> tuple[bool, str]:
        if not self._enabled or policy_context == "plan_validate":
            return True, ""
        window = self._p_window_s if self._lock_mode == "paranoid" else self._window_s
        g_cap = self._p_max_per_window if self._lock_mode == "paranoid" else self._max_per_window
        rep_cap = self._p_max_repeat if self._lock_mode == "paranoid" else self._max_repeat

        now = time.monotonic()

        while self._global_ts and now - self._global_ts[0] > window:
            self._global_ts.popleft()
        if len(self._global_ts) >= g_cap:
            reason = (
                f"behavior_monitor: global rate limit ({g_cap} actions in {window:.0f}s)"
            )
            logger.warning("paranoid:behavior_rate %s", reason)
            return False, reason

        dq = self._per_action[action]
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= rep_cap:
            reason = (
                f"behavior_monitor: action '{action}' repeated too often "
                f"({rep_cap} in {window:.0f}s)"
            )
            logger.warning("paranoid:behavior_burst %s", action)
            return False, reason

        dq.append(now)
        self._global_ts.append(now)
        return True, ""


def strip_signing_keys(args: dict | None) -> dict:
    if not args:
        return {}
    return {k: v for k, v in args.items() if not k.startswith("_atom_")}
