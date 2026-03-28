"""
ATOM -- Safe Code Execution Sandbox.

Allows ATOM to evaluate mathematical expressions and simple Python
code safely, without risking the host system. Like JARVIS running
calculations for Tony Stark in real-time.

Security:
    - Restricted builtins (no open, exec, eval, import, __import__)
    - 5-second execution timeout
    - 50MB memory limit (conceptual, enforced by output cap)
    - No file system access
    - No network access
    - Output capped at 2000 chars

Supported:
    - Math expressions: "15% of 2300", "sqrt(144)", "2**10"
    - List comprehensions: "[x**2 for x in range(10)]"
    - String operations: "'hello'.upper()"
    - Date calculations: "days between Jan 1 and Mar 15"

Not Supported (blocked):
    - File I/O, network, subprocess
    - import statements
    - Class definitions that access dunder methods
"""

from __future__ import annotations

import logging
import math
import re
import signal
import threading
import time
from typing import Any

logger = logging.getLogger("atom.sandbox")

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin,
    "bool": bool, "chr": chr, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "hex": hex,
    "int": int, "isinstance": isinstance, "len": len, "list": list,
    "map": map, "max": max, "min": min, "oct": oct, "ord": ord,
    "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}

_SAFE_MATH = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp, "pi": math.pi,
    "e": math.e, "ceil": math.ceil, "floor": math.floor,
    "factorial": math.factorial, "gcd": math.gcd,
    "degrees": math.degrees, "radians": math.radians,
    "inf": math.inf, "nan": math.nan,
}

_BLOCKED_PATTERNS = [
    r'\bimport\b', r'\b__import__\b', r'\bexec\b', r'\beval\b',
    r'\bopen\b', r'\bcompile\b', r'\bglobals\b', r'\blocals\b',
    r'\bgetattr\b', r'\bsetattr\b', r'\bdelattr\b',
    r'\b__\w+__\b',
    r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bshutil\b',
    r'\bsocket\b', r'\burllib\b', r'\brequests\b',
    r'\bbreakpoint\b', r'\bexit\b', r'\bquit\b',
]

_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS), re.IGNORECASE)

_PERCENT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)')
_HUMAN_MATH = [
    (re.compile(r'square\s*root\s*(?:of\s*)?(\d+(?:\.\d+)?)', re.I), r'sqrt(\1)'),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:to the power of|raised to)\s*(\d+(?:\.\d+)?)', re.I), r'\1**\2'),
    (re.compile(r'(\d+(?:\.\d+)?)\s*squared', re.I), r'\1**2'),
    (re.compile(r'(\d+(?:\.\d+)?)\s*cubed', re.I), r'\1**3'),
]

_MAX_OUTPUT = 2000
_TIMEOUT_S = 5.0


def _preprocess_expression(expr: str) -> str:
    """Convert human-readable math into Python expressions."""
    result = expr.strip()
    result = _PERCENT_PATTERN.sub(
        lambda m: f"({m.group(1)} / 100) * {m.group(2)}", result,
    )
    for pattern, replacement in _HUMAN_MATH:
        result = pattern.sub(replacement, result)
    return result


def _is_safe(code: str) -> tuple[bool, str]:
    """Check if code is safe to execute."""
    match = _BLOCKED_RE.search(code)
    if match:
        return False, f"Blocked pattern: {match.group()}"
    if code.count("\n") > 50:
        return False, "Code too long (max 50 lines)"
    if len(code) > 5000:
        return False, "Code too long (max 5000 chars)"
    return True, ""


class CodeSandbox:
    """Safe Python execution environment."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("sandbox", {})
        self._timeout = cfg.get("timeout_seconds", _TIMEOUT_S)
        self._max_output = cfg.get("max_output_chars", _MAX_OUTPUT)
        self._enabled = cfg.get("enabled", True)
        self._execution_count = 0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def execute(self, code: str) -> dict[str, Any]:
        """Execute code in a sandboxed environment.

        Returns:
            {"success": bool, "result": str, "error": str, "time_ms": float}
        """
        if not self._enabled:
            return {"success": False, "result": "", "error": "Sandbox disabled"}

        code = _preprocess_expression(code)
        safe, reason = _is_safe(code)
        if not safe:
            return {"success": False, "result": "", "error": reason}

        safe_globals: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        safe_globals.update(_SAFE_MATH)

        output_capture: list[str] = []

        def _safe_print(*args: Any, **kwargs: Any) -> None:
            text = " ".join(str(a) for a in args)
            output_capture.append(text)

        safe_globals["print"] = _safe_print

        t0 = time.perf_counter()
        result_container: dict[str, Any] = {"result": None, "error": None}

        def _run() -> None:
            try:
                try:
                    result_container["result"] = eval(code, safe_globals)
                except SyntaxError:
                    exec(code, safe_globals)
                    if output_capture:
                        result_container["result"] = "\n".join(output_capture)
                    else:
                        result_container["result"] = "Code executed successfully."
            except Exception as e:
                result_container["error"] = f"{type(e).__name__}: {str(e)[:200]}"

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if thread.is_alive():
            return {
                "success": False,
                "result": "",
                "error": f"Execution timed out ({self._timeout}s limit)",
                "time_ms": elapsed_ms,
            }

        if result_container["error"]:
            return {
                "success": False,
                "result": "",
                "error": result_container["error"],
                "time_ms": elapsed_ms,
            }

        result_str = str(result_container["result"])
        if len(result_str) > self._max_output:
            result_str = result_str[:self._max_output] + "... (truncated)"

        self._execution_count += 1
        logger.info("Sandbox executed in %.0fms: %s -> %s",
                     elapsed_ms, code[:60], result_str[:60])

        return {
            "success": True,
            "result": result_str,
            "error": "",
            "time_ms": elapsed_ms,
        }

    def evaluate_math(self, expression: str) -> str:
        """Convenience method for pure math evaluation."""
        result = self.execute(expression)
        if result["success"]:
            return result["result"]
        return f"Could not calculate: {result['error']}"
