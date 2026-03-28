"""
ATOM V6.5 -- Performance profiler (decorators + structured latency metrics).

Measures planning, simulation, memory query, and execution phases for
optimization and health scoring. Thread-safe rolling aggregates.

Does not require running the full stack; safe to import anywhere.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, TypeVar

logger = logging.getLogger("atom.profiler")

F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_MAX_SAMPLES = 256
_LOCK = threading.Lock()
_SAMPLES: dict[str, deque[float]] = {}
_TOTAL_MS: dict[str, float] = {}
_COUNTS: dict[str, int] = {}


def _record(phase: str, elapsed_ms: float) -> None:
    with _LOCK:
        if phase not in _SAMPLES:
            _SAMPLES[phase] = deque(maxlen=_DEFAULT_MAX_SAMPLES)
        _SAMPLES[phase].append(elapsed_ms)
        _TOTAL_MS[phase] = _TOTAL_MS.get(phase, 0.0) + elapsed_ms
        _COUNTS[phase] = _COUNTS.get(phase, 0) + 1
    try:
        from core.metrics import get_metrics
        get_metrics().record_latency(f"profiler_{phase}", elapsed_ms)
    except Exception:
        pass


def profile(phase: str) -> Callable[[F], F]:
    """Decorator: record wall-clock time for *phase* (sync functions)."""

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _record(phase, (time.perf_counter() - t0) * 1000.0)

        return wrapper  # type: ignore[return-value]

    return deco


def profile_async(phase: str) -> Callable[[F], F]:
    """Decorator: record wall-clock time for async functions."""

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                _record(phase, (time.perf_counter() - t0) * 1000.0)

        return wrapper  # type: ignore[return-value]

    return deco


def measure(phase: str) -> Any:
    """Context manager for ad-hoc timing."""

    class _CM:
        def __enter__(self) -> None:
            self._t0 = time.perf_counter()

        def __exit__(self, *exc: Any) -> None:
            _record(phase, (time.perf_counter() - self._t0) * 1000.0)

    return _CM()


def get_latency_snapshot() -> dict[str, Any]:
    """Rolling means and last samples per phase (for health / CLI)."""
    out: dict[str, Any] = {}
    with _LOCK:
        for phase, samples in _SAMPLES.items():
            if not samples:
                continue
            arr = list(samples)
            out[phase] = {
                "mean_ms": sum(arr) / len(arr),
                "max_ms": max(arr),
                "min_ms": min(arr),
                "n": len(arr),
                "total_ms": _TOTAL_MS.get(phase, 0.0),
                "total_calls": _COUNTS.get(phase, 0),
            }
    return out


def reset_metrics() -> None:
    """Clear profiler aggregates (e.g. between validation runs)."""
    with _LOCK:
        _SAMPLES.clear()
        _TOTAL_MS.clear()
        _COUNTS.clear()


def log_summary(prefix: str = "ATOM profiler") -> None:
    snap = get_latency_snapshot()
    if not snap:
        logger.info("%s: (no samples yet)", prefix)
        return
    parts = [f"{k}: {v['mean_ms']:.1f}ms" for k, v in sorted(snap.items())]
    logger.info("%s — %s", prefix, " | ".join(parts))


__all__ = [
    "profile",
    "profile_async",
    "measure",
    "get_latency_snapshot",
    "reset_metrics",
    "log_summary",
]
