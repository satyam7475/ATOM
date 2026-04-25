"""Sprint C6 -- Intent engine 150 ms budget regression suite.

The live log (``atomLogs.txt`` L419 + L453) showed the intent engine
breaching its configured 150 ms budget. Profiling proved the regex
work itself takes ~0.03 ms p50 / 0.12 ms p99 -- i.e. *the engine is
not slow*. The breaches were caused by ``loop.run_in_executor``
dispatch overhead under MLX/vision GIL contention turning a 30 µs
synchronous call into a 150-300 ms wall-time.

This file pins three invariants that, taken together, guarantee the
breaches don't come back:

  1. Every regex used by an intent module is compiled at *module load
     time* (not on every call) -- the implicit ``re._cache`` is irrelevant
     because we never go through the dynamic-string compile path.
  2. End-to-end ``IntentEngine.classify_silent`` p99 over 18 representative
     phrases is < 5 ms -- 30x under the configured 150 ms budget.
  3. ``RuntimeWatchdog.run_inline`` correctly bypasses the executor:
     the work runs synchronously in the caller thread, the elapsed
     time is measured, and a budget breach surfaces a warning *without*
     a thread-pool round trip.

If any of these break, the watchdog log will start screaming again and
the user-facing latency will jump 100-300 ms per turn.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import time
from pathlib import Path

import pytest

from core.intent_engine import IntentEngine
from core.runtime_watchdog import BudgetResult, RuntimeWatchdog


_REPRESENTATIVE_QUERIES = [
    "play some music",
    "open chrome",
    "what time is it",
    "describe my screen",
    "can you see me",
    "pause music",
    "what is the weather",
    "show me my battery",
    "thank you",
    "good morning",
    "play bohemian rhapsody on spotify",
    "next song",
    "what is the meaning of life",
    "open my downloads folder",
    "set volume to 50",
    "tell me a joke",
    "how are you doing today my friend",
    "launch the terminal application now",
]


# ── Invariant 1: regex compile-once ────────────────────────────────


def _intent_module_paths() -> list[Path]:
    """All ``core.intent_engine.*_intents`` source files."""
    import core.intent_engine as pkg

    pkg_dir = Path(pkg.__path__[0])
    return sorted(p for p in pkg_dir.glob("*.py") if "intent" in p.stem)


def _calls_re_compile_inside_function(source: str) -> list[str]:
    """Return names of functions whose body contains ``re.compile(...)``.

    A function-level ``re.compile`` is the bug we're guarding against:
    every call would JIT-compile (or hit the small global ``re._cache``,
    which is bounded at 512 patterns and may evict under churn).
    """
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if isinstance(func, ast.Attribute) and func.attr == "compile":
                value = func.value
                if isinstance(value, ast.Name) and value.id == "re":
                    offenders.append(node.name)
                    break
    return offenders


@pytest.mark.parametrize("path", _intent_module_paths(), ids=lambda p: p.stem)
def test_intent_module_compiles_regex_at_module_level(path: Path) -> None:
    """No intent module may compile regex inside a function body.

    Module-level ``re.compile`` runs once at import time and is then
    a frozen pattern object reused by every ``check`` / ``quick_match``
    call. Function-level ``re.compile`` would defeat the cache and re-run
    the regex compiler every turn, blowing the 150 ms budget under load.
    """
    source = path.read_text()
    offenders = _calls_re_compile_inside_function(source)
    assert not offenders, (
        f"{path.name}: re.compile() called inside function(s) {offenders}; "
        f"move the pattern to module scope so it compiles once at import."
    )


# ── Invariant 2: p99 latency ───────────────────────────────────────


def test_intent_engine_p99_under_5ms() -> None:
    """End-to-end classify p99 must stay 30x under the 150 ms budget."""
    engine = IntentEngine()

    for q in _REPRESENTATIVE_QUERIES:
        engine.classify_silent(q)

    samples_ms: list[float] = []
    for q in _REPRESENTATIVE_QUERIES:
        for _ in range(100):
            t0 = time.perf_counter()
            engine.classify_silent(q)
            samples_ms.append((time.perf_counter() - t0) * 1000)

    samples_ms.sort()
    n = len(samples_ms)
    p50 = samples_ms[n // 2]
    p99 = samples_ms[int(n * 0.99) - 1]

    assert p99 < 5.0, (
        f"intent engine p99={p99:.3f} ms, p50={p50:.3f} ms (n={n}); "
        f"this is the regex work itself -- something added a backtracking "
        f"pattern or restored per-call ``re.compile`` somewhere."
    )


# ── Invariant 3: run_inline correctness ────────────────────────────


def _make_watchdog(budget_ms: float = 150.0) -> RuntimeWatchdog:
    """Build a stripped-down watchdog with the production intent budget."""

    class _StubBus:
        def on(self, *a, **kw): pass
        def emit(self, *a, **kw): pass
        def emit_fast(self, *a, **kw): pass

    class _StubState:
        current = "idle"

    config = {"performance": {
        "watchdog_intent_timeout_ms": float(budget_ms),
        "watchdog_intent_boot_grace_s": 0.0,
    }}
    return RuntimeWatchdog(_StubBus(), _StubState(), config)


def test_run_inline_returns_result_without_executor() -> None:
    """``run_inline`` must not require an asyncio loop.

    This is the whole point of the API: callers can invoke it from
    inside an async function without paying the executor dispatch tax.
    """
    wd = _make_watchdog(budget_ms=150.0)

    calls: list[str] = []

    def _classify(text: str) -> str:
        calls.append(text)
        return f"intent::{text}"

    res: BudgetResult = wd.run_inline(
        "intent_engine", _classify, "play music",
        default="fallback",
    )
    assert isinstance(res, BudgetResult)
    assert res.value == "intent::play music"
    assert res.timed_out is False
    assert res.elapsed_ms >= 0.0
    assert calls == ["play music"]


def test_run_inline_marks_breach_when_func_exceeds_budget() -> None:
    """Slow synchronous work must be flagged ``timed_out=True``.

    Note we cannot interrupt the synchronous call -- by design -- so
    the value is still returned. The breach is recorded so the
    watchdog logs the regression.
    """
    wd = _make_watchdog(budget_ms=5.0)

    def _slow() -> str:
        time.sleep(0.020)
        return "ok"

    res = wd.run_inline("intent_engine", _slow, default="fallback")
    assert res.value == "ok"
    assert res.timed_out is True
    assert res.elapsed_ms >= 15.0


def test_run_inline_swallows_exception_and_returns_default() -> None:
    """A buggy intent module must not take down the router."""
    wd = _make_watchdog(budget_ms=150.0)

    def _boom() -> str:
        raise RuntimeError("regex backtrack exploded")

    res = wd.run_inline("intent_engine", _boom, default="fallback")
    assert res.value == "fallback"
    assert res.timed_out is False
    assert res.elapsed_ms >= 0.0


def test_run_inline_passes_args_through() -> None:
    """Positional args must reach the wrapped callable unchanged."""
    wd = _make_watchdog(budget_ms=150.0)

    def _join(a: str, b: str) -> str:
        return f"{a}|{b}"

    res = wd.run_inline("intent_engine", _join, "foo", "bar", default="x")
    assert res.value == "foo|bar"


# ── Invariant 4: every intent module imports cleanly ───────────────


@pytest.mark.parametrize("path", _intent_module_paths(), ids=lambda p: p.stem)
def test_intent_module_imports_cheaply(path: Path) -> None:
    """Importing an intent module must not exceed 200 ms.

    A heavy import-time side effect (e.g. loading a model) would
    blow the boot budget. All compile work happens here, once.
    """
    module_name = f"core.intent_engine.{path.stem}"
    if module_name in importlib.sys.modules:
        del importlib.sys.modules[module_name]
    t0 = time.perf_counter()
    importlib.import_module(module_name)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 200.0, (
        f"{module_name} import took {elapsed_ms:.1f} ms -- intent modules "
        f"must be near-zero-cost imports (regex compile only)."
    )


# ── Invariant 5: every intent module exposes a stable surface ──────


def test_intent_engine_classify_returns_intent_result() -> None:
    """The router relies on every classify call returning an
    ``IntentResult`` with at least a ``.intent`` attribute. A wrong
    return type would trip downstream attribute access *before* the
    budget check ever fired."""
    from core.intent_engine import IntentResult

    engine = IntentEngine()
    for q in _REPRESENTATIVE_QUERIES:
        result = engine.classify_silent(q)
        assert isinstance(result, IntentResult)
        assert isinstance(result.intent, str)
        assert result.intent
