"""
ATOM -- Self-Evolution Engine (AI OS Meta-Intelligence).

ATOM's self-awareness and continuous improvement system:
  - Analyzes own performance metrics in real-time
  - Identifies bottlenecks and patterns
  - Generates actionable improvement suggestions
  - Tracks evolution history over sessions
  - Searches the web for improvement techniques

This is what makes ATOM a self-improving AI OS --
not a static assistant but one that learns and evolves.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.metrics import MetricsCollector

logger = logging.getLogger("atom.evolution")

_EVOLUTION_FILE = Path("logs/evolution.json")
_MAX_ENTRIES = 200


class SelfEvolutionEngine:
    """Analyzes ATOM's performance and generates improvement suggestions.

    Periodically runs diagnostics, identifies issues, and proposes
    concrete actions to improve speed, accuracy, and capability.
    """

    def __init__(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics
        self._history: list[dict] = []
        self._session_start = time.time()
        self._load()

    def _load(self) -> None:
        try:
            if _EVOLUTION_FILE.exists():
                data = json.loads(_EVOLUTION_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._history = data[-_MAX_ENTRIES:]
        except Exception:
            self._history = []

    def persist(self) -> None:
        try:
            _EVOLUTION_FILE.parent.mkdir(parents=True, exist_ok=True)
            _EVOLUTION_FILE.write_text(
                json.dumps(self._history[-_MAX_ENTRIES:], indent=2),
                encoding="utf-8",
            )
            try:
                import os
                os.chmod(_EVOLUTION_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.debug("Failed to save evolution history", exc_info=True)

    def run_diagnostic(self) -> dict:
        """Run a comprehensive self-diagnostic and return findings."""
        snap = self._metrics.snapshot()

        findings: dict = {
            "timestamp": datetime.now().isoformat(),
            "session_uptime_min": round((time.time() - self._session_start) / 60, 1),
            "metrics": snap,
            "issues": [],
            "suggestions": [],
            "health_score": 10.0,
        }

        total_queries = snap.get("queries_total", 0)
        cache_hit_pct = snap.get("cache_hit_rate_pct", 0)
        perceived_avg = snap.get("perceived_avg_ms", 0)

        local_pct = 0.0
        if total_queries > 0:
            local_routed = snap.get("local_routed_queries", 0)
            local_pct = (local_routed / total_queries) * 100

        # -- Latency analysis --
        if perceived_avg and float(perceived_avg) > 3000:
            findings["issues"].append(
                "High perceived latency (>3s). LLM responses are slow.")
            findings["suggestions"].append(
                "Switch to a faster LLM model or increase cache TTL.")
            findings["health_score"] -= 1.5
        elif perceived_avg and float(perceived_avg) > 1500:
            findings["issues"].append("Moderate latency (1.5-3s).")
            findings["suggestions"].append(
                "Improve cache hit rate for faster repeated responses.")
            findings["health_score"] -= 0.5

        # -- Cache efficiency --
        if total_queries > 20 and cache_hit_pct < 15:
            findings["issues"].append(
                f"Low cache hit rate ({cache_hit_pct:.0f}%).")
            findings["suggestions"].append(
                "Add more intent patterns to handle common queries locally.")
            findings["health_score"] -= 1.0
        elif cache_hit_pct > 70:
            findings["suggestions"].append(
                "Excellent cache performance. Consider reducing TTL for freshness.")

        # -- LLM dependency --
        if total_queries > 10 and local_pct < 50:
            findings["issues"].append(
                f"Only {local_pct:.0f}% handled locally. Heavy LLM dependency.")
            findings["suggestions"].append(
                "Teach ATOM new regex patterns for your frequent questions.")
            findings["health_score"] -= 0.5

        # -- Session insights --
        if total_queries == 0:
            findings["suggestions"].append("No queries yet. Ready and waiting.")
        elif total_queries > 50:
            findings["suggestions"].append(
                f"Active session ({total_queries} queries). "
                "ATOM is learning your patterns.")

        # -- Uptime-based --
        uptime_min = findings["session_uptime_min"]
        if uptime_min > 120:
            findings["suggestions"].append(
                f"Running for {uptime_min:.0f}min. All systems stable.")

        # -- Web improvement ideas --
        findings["suggestions"].append(
            "Ask me to 'research voice assistant techniques' "
            "for web-sourced improvement ideas.")

        findings["health_score"] = max(1.0, min(10.0, findings["health_score"]))

        self._history.append(findings)
        self.persist()
        return findings

    def format_diagnostic(self, diag: dict | None = None) -> str:
        """Format diagnostic results for voice/text output."""
        if diag is None:
            diag = self.run_diagnostic()

        score = diag.get("health_score", 0)
        issues = diag.get("issues", [])
        suggestions = diag.get("suggestions", [])
        metrics = diag.get("metrics", {})
        uptime = diag.get("session_uptime_min", 0)

        parts = [f"ATOM Self-Diagnostic. Health score: {score:.1f} out of 10."]

        total = metrics.get("queries_total", 0)
        if total > 0:
            parts.append(f"Processed {total} queries in {uptime:.0f} minutes.")

        if issues:
            parts.append(
                f"Found {len(issues)} issue{'s' if len(issues) > 1 else ''}:")
            for issue in issues[:3]:
                parts.append(f"  {issue}")

        if suggestions:
            parts.append("Suggestions:")
            for s in suggestions[:3]:
                parts.append(f"  {s}")

        return " ".join(parts)

    def get_evolution_summary(self) -> str:
        """Summarize the evolution history across sessions."""
        if not self._history:
            return "No evolution history yet. Say 'self diagnostic' to start."

        total_diagnostics = len(self._history)
        scores = [h.get("health_score", 0) for h in self._history]
        avg_score = sum(scores) / len(scores)
        latest_score = scores[-1]

        all_issues: list[str] = []
        for h in self._history[-10:]:
            all_issues.extend(h.get("issues", []))

        if len(scores) > 1:
            trend = "improving" if scores[-1] >= scores[0] else "needs attention"
        else:
            trend = "just started"

        return (
            f"Evolution report: {total_diagnostics} diagnostics run. "
            f"Average health: {avg_score:.1f}/10. Latest: {latest_score:.1f}/10. "
            f"Trend: {trend}. Recent issues: {len(all_issues)}."
        )

    def decide_action(self) -> str | None:
        """Self-decision: recommend an action based on current state.

        Returns a suggestion string or None if everything is optimal.
        This is ATOM's autonomous decision-making capability.
        """
        snap = self._metrics.snapshot()
        total = snap.get("queries_total", 0)

        if total < 5:
            return None

        cache_pct = snap.get("cache_hit_rate_pct", 0)
        if cache_pct < 10 and total > 20:
            return ("Boss, I notice most queries go to the full LLM pipeline. "
                    "I could handle more via fast-path if you teach me "
                    "new patterns. Just say what you ask most often.")

        perceived = snap.get("perceived_avg_ms", 0)
        if perceived and float(perceived) > 4000:
            return ("Boss, my response times are getting slow. "
                    "Might be network latency. Want me to run a "
                    "network diagnostic?")

        return None
