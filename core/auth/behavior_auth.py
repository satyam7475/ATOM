"""
ATOM -- Behavioral Authentication (Passive Continuous Verification).

While VoicePrintAuth verifies identity at specific moments, BehavioralAuth
continuously monitors usage patterns to detect if someone other than
the owner is using ATOM. This is the "something you do" factor.

Tracked Behavioral Signals:
    1. COMMAND VOCABULARY -- which commands are used and how often
    2. TEMPORAL PATTERNS -- normal working hours, session durations
    3. QUERY STYLE -- average query length, question vs command ratio
    4. APP PREFERENCES -- which apps are opened, in what order
    5. INTERACTION RHYTHM -- time between commands, burst patterns
    6. TOPIC FINGERPRINT -- what topics come up frequently

Anomaly Detection:
    - Builds a baseline profile over the first N sessions
    - Computes an anomaly score (0.0 = normal, 1.0 = completely foreign)
    - When anomaly exceeds threshold -> triggers re-verification event
    - Uses exponential moving average to adapt to gradual changes

Trust Score:
    - Starts at 1.0 after successful authentication
    - Decays slowly over time (session expiry)
    - Boosted by matching behavioral patterns
    - Drops sharply on anomalous behavior
    - Below 0.3 -> trigger re-verification

Contract:
    observe(action, context) -> None       # feed behavioral data
    get_trust_score() -> float             # current trust level [0, 1]
    is_anomalous() -> bool                 # above anomaly threshold
    get_anomaly_report() -> str            # human-readable report

Owner: Satyam
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.auth.behavior")

_BEHAVIOR_PROFILE_FILE = Path("data/security/behavior_profile.json")

_BASELINE_MIN_OBSERVATIONS = 50
_ANOMALY_THRESHOLD = 0.65
_TRUST_DECAY_PER_HOUR = 0.02
_TRUST_BOOST_PER_MATCH = 0.01
_TRUST_DROP_PER_ANOMALY = 0.15
_TRUST_MIN_FOR_REAUTH = 0.30
_EMA_ALPHA = 0.05
_MAX_HISTORY = 500


class BehavioralAuth:
    """Passive continuous authentication through usage pattern analysis.

    Observes every interaction and maintains a trust score that reflects
    how closely current behavior matches the owner's established patterns.
    """

    __slots__ = (
        "_config", "_trust_score", "_last_activity_time",
        "_baseline", "_current_session", "_anomaly_score",
        "_is_baselined", "_observation_count",
        "_reauth_callback", "_last_anomaly_time",
    )

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("behavior_auth", {})
        self._config = cfg
        self._trust_score = 1.0
        self._last_activity_time = time.monotonic()
        self._anomaly_score = 0.0
        self._is_baselined = False
        self._observation_count = 0
        self._reauth_callback: Any = None
        self._last_anomaly_time = 0.0

        self._baseline = BehaviorBaseline()
        self._current_session = SessionBehavior()

        self._load_baseline()

    def set_reauth_callback(self, callback: Any) -> None:
        """Set callback to trigger when re-authentication is needed."""
        self._reauth_callback = callback

    # ── Observation ──────────────────────────────────────────────────

    def observe(
        self,
        action: str,
        detail: str = "",
        query_text: str = "",
        active_app: str = "",
    ) -> None:
        """Feed a behavioral observation. Called on every user interaction."""
        now = time.monotonic()
        hour = datetime.now().hour

        gap = now - self._last_activity_time if self._last_activity_time > 0 else 0
        self._last_activity_time = now
        self._observation_count += 1

        self._current_session.record(
            action=action,
            detail=detail,
            query_text=query_text,
            active_app=active_app,
            hour=hour,
            gap_s=gap,
        )

        if self._is_baselined:
            self._update_anomaly_score()
            self._update_trust_score()

            if self._anomaly_score > _ANOMALY_THRESHOLD:
                if now - self._last_anomaly_time > 300:
                    self._last_anomaly_time = now
                    logger.warning(
                        "Behavioral anomaly detected: score=%.2f, trust=%.2f",
                        self._anomaly_score, self._trust_score,
                    )
                    if self._trust_score < _TRUST_MIN_FOR_REAUTH:
                        self._trigger_reauth()
        else:
            self._baseline.absorb(self._current_session)
            if self._observation_count >= _BASELINE_MIN_OBSERVATIONS:
                self._is_baselined = True
                self._save_baseline()
                logger.info(
                    "Behavioral baseline established (%d observations)",
                    self._observation_count,
                )

    def on_authenticated(self) -> None:
        """Called when owner successfully authenticates (voice or passphrase).

        Resets trust score and absorbs current behavior into baseline
        (since we now know it's the real owner).
        """
        self._trust_score = 1.0
        self._anomaly_score = 0.0
        if self._current_session.action_count > 5:
            self._baseline.absorb(self._current_session)
            self._save_baseline()

    # ── Anomaly Detection ────────────────────────────────────────────

    def _update_anomaly_score(self) -> None:
        """Compute how anomalous current session behavior is vs baseline."""
        scores: list[float] = []

        vocab_score = self._vocabulary_anomaly()
        scores.append(vocab_score)

        temporal_score = self._temporal_anomaly()
        scores.append(temporal_score)

        rhythm_score = self._rhythm_anomaly()
        scores.append(rhythm_score)

        style_score = self._style_anomaly()
        scores.append(style_score)

        if scores:
            raw = sum(scores) / len(scores)
            self._anomaly_score = (
                _EMA_ALPHA * raw
                + (1 - _EMA_ALPHA) * self._anomaly_score
            )

    def _vocabulary_anomaly(self) -> float:
        """How different is the command vocabulary from baseline?"""
        if not self._baseline.command_freq or not self._current_session.commands:
            return 0.0

        baseline_cmds = set(self._baseline.command_freq.keys())
        session_cmds = set(self._current_session.commands.keys())

        if not baseline_cmds:
            return 0.0

        known = session_cmds & baseline_cmds
        unknown = session_cmds - baseline_cmds

        if not session_cmds:
            return 0.0

        unknown_ratio = len(unknown) / len(session_cmds)

        baseline_dist = self._baseline.normalized_command_freq()
        session_dist = self._current_session.normalized_commands()
        kl_div = self._kl_divergence(session_dist, baseline_dist)

        return min(1.0, unknown_ratio * 0.4 + min(kl_div / 5.0, 1.0) * 0.6)

    def _temporal_anomaly(self) -> float:
        """Is the user active outside normal hours?"""
        hour = datetime.now().hour
        if not self._baseline.hour_freq:
            return 0.0

        total = sum(self._baseline.hour_freq.values())
        if total == 0:
            return 0.0

        hour_pct = self._baseline.hour_freq.get(hour, 0) / total

        if hour_pct > 0.05:
            return 0.0
        if hour_pct > 0.01:
            return 0.3
        if hour_pct > 0.0:
            return 0.5
        return 0.8

    def _rhythm_anomaly(self) -> float:
        """Is the interaction rhythm (gaps between commands) unusual?"""
        if not self._current_session.gaps or not self._baseline.avg_gap_s:
            return 0.0

        session_avg = (
            sum(self._current_session.gaps) / len(self._current_session.gaps)
        )
        baseline_avg = self._baseline.avg_gap_s

        if baseline_avg < 0.1:
            return 0.0

        ratio = session_avg / baseline_avg
        deviation = abs(math.log(max(ratio, 0.01)))
        return min(1.0, deviation / 3.0)

    def _style_anomaly(self) -> float:
        """Is the query style (length, question ratio) different?"""
        if not self._current_session.query_lengths:
            return 0.0

        session_avg_len = (
            sum(self._current_session.query_lengths)
            / len(self._current_session.query_lengths)
        )
        baseline_avg_len = self._baseline.avg_query_length

        if baseline_avg_len < 1:
            return 0.0

        len_ratio = session_avg_len / baseline_avg_len
        len_deviation = abs(math.log(max(len_ratio, 0.01)))

        return min(1.0, len_deviation / 2.0)

    @staticmethod
    def _kl_divergence(
        p: dict[str, float], q: dict[str, float],
    ) -> float:
        """Compute KL divergence D(P || Q) with smoothing."""
        all_keys = set(p.keys()) | set(q.keys())
        if not all_keys:
            return 0.0
        epsilon = 1e-6
        total = 0.0
        for key in all_keys:
            p_val = p.get(key, epsilon)
            q_val = q.get(key, epsilon)
            if p_val > epsilon:
                total += p_val * math.log(p_val / q_val)
        return max(0.0, total)

    # ── Trust Score ──────────────────────────────────────────────────

    def _update_trust_score(self) -> None:
        """Update trust score based on behavioral match."""
        if self._anomaly_score < 0.2:
            self._trust_score = min(1.0, self._trust_score + _TRUST_BOOST_PER_MATCH)
        elif self._anomaly_score > _ANOMALY_THRESHOLD:
            self._trust_score = max(0.0, self._trust_score - _TRUST_DROP_PER_ANOMALY)

    def apply_time_decay(self) -> None:
        """Apply time-based trust decay. Call periodically (e.g., every minute)."""
        elapsed_hours = (
            (time.monotonic() - self._last_activity_time) / 3600
        )
        decay = _TRUST_DECAY_PER_HOUR * elapsed_hours
        self._trust_score = max(0.0, self._trust_score - decay)

    def _trigger_reauth(self) -> None:
        """Trigger re-authentication when trust drops too low."""
        logger.warning(
            "Trust score below threshold (%.2f < %.2f). Re-authentication required.",
            self._trust_score, _TRUST_MIN_FOR_REAUTH,
        )
        if self._reauth_callback:
            try:
                self._reauth_callback()
            except Exception:
                logger.debug("Re-auth callback failed", exc_info=True)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def trust_score(self) -> float:
        return self._trust_score

    @property
    def anomaly_score(self) -> float:
        return self._anomaly_score

    @property
    def is_anomalous(self) -> bool:
        return self._anomaly_score > _ANOMALY_THRESHOLD

    @property
    def is_baselined(self) -> bool:
        return self._is_baselined

    @property
    def needs_reauth(self) -> bool:
        return self._trust_score < _TRUST_MIN_FOR_REAUTH

    def get_trust_level(self) -> str:
        if self._trust_score >= 0.8:
            return "high"
        if self._trust_score >= 0.5:
            return "medium"
        if self._trust_score >= 0.3:
            return "low"
        return "critical"

    def get_anomaly_report(self) -> str:
        """Human-readable anomaly report."""
        if not self._is_baselined:
            remaining = _BASELINE_MIN_OBSERVATIONS - self._observation_count
            return (
                f"Building behavioral baseline. {self._observation_count} "
                f"observations collected, {remaining} more needed."
            )

        parts = [
            f"Trust score: {self._trust_score:.0%} ({self.get_trust_level()}).",
            f"Anomaly score: {self._anomaly_score:.0%}.",
        ]

        if self._anomaly_score > _ANOMALY_THRESHOLD:
            parts.append("WARNING: Behavioral patterns do not match owner profile.")
        elif self._anomaly_score > 0.3:
            parts.append("Minor deviations from normal patterns detected.")
        else:
            parts.append("Behavior matches owner profile.")

        parts.append(
            f"Observations: {self._observation_count}. "
            f"Session commands: {self._current_session.action_count}."
        )

        return " ".join(parts)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "trust_score": round(self._trust_score, 3),
            "anomaly_score": round(self._anomaly_score, 3),
            "trust_level": self.get_trust_level(),
            "is_baselined": self._is_baselined,
            "observations": self._observation_count,
            "baseline_commands": len(self._baseline.command_freq),
            "baseline_hours": len(self._baseline.hour_freq),
            "session_commands": self._current_session.action_count,
        }

    # ── Persistence ──────────────────────────────────────────────────

    def _save_baseline(self) -> None:
        try:
            _BEHAVIOR_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "command_freq": dict(self._baseline.command_freq),
                "hour_freq": {
                    str(k): v for k, v in self._baseline.hour_freq.items()
                },
                "avg_gap_s": self._baseline.avg_gap_s,
                "avg_query_length": self._baseline.avg_query_length,
                "total_observations": self._observation_count,
                "app_freq": dict(self._baseline.app_freq),
                "is_baselined": self._is_baselined,
            }
            _BEHAVIOR_PROFILE_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except Exception:
            logger.debug("Behavior baseline save failed", exc_info=True)

    def _load_baseline(self) -> None:
        if not _BEHAVIOR_PROFILE_FILE.exists():
            return
        try:
            data = json.loads(
                _BEHAVIOR_PROFILE_FILE.read_text(encoding="utf-8"),
            )
            self._baseline.command_freq = Counter(data.get("command_freq", {}))
            self._baseline.hour_freq = Counter({
                int(k): v for k, v in data.get("hour_freq", {}).items()
            })
            self._baseline.avg_gap_s = data.get("avg_gap_s", 0.0)
            self._baseline.avg_query_length = data.get("avg_query_length", 0.0)
            self._baseline.app_freq = Counter(data.get("app_freq", {}))
            self._observation_count = data.get("total_observations", 0)
            self._is_baselined = data.get("is_baselined", False)

            if self._is_baselined:
                logger.info(
                    "Behavioral baseline loaded: %d commands, %d hours, %d obs",
                    len(self._baseline.command_freq),
                    len(self._baseline.hour_freq),
                    self._observation_count,
                )
        except Exception:
            logger.debug("Behavior baseline load failed", exc_info=True)

    def persist(self) -> None:
        if self._is_baselined or self._observation_count > 10:
            self._save_baseline()

    def shutdown(self) -> None:
        self.persist()
        logger.info("Behavioral auth shut down")


class BehaviorBaseline:
    """Accumulated behavioral baseline for the owner."""

    __slots__ = (
        "command_freq", "hour_freq", "app_freq",
        "avg_gap_s", "avg_query_length",
        "_gap_sum", "_gap_count", "_len_sum", "_len_count",
    )

    def __init__(self) -> None:
        self.command_freq: Counter = Counter()
        self.hour_freq: Counter = Counter()
        self.app_freq: Counter = Counter()
        self.avg_gap_s: float = 0.0
        self.avg_query_length: float = 0.0
        self._gap_sum: float = 0.0
        self._gap_count: int = 0
        self._len_sum: float = 0.0
        self._len_count: int = 0

    def absorb(self, session: SessionBehavior) -> None:
        """Merge session behavior into the baseline using EMA."""
        self.command_freq.update(session.commands)
        self.hour_freq.update(session.hours)
        self.app_freq.update(session.apps)

        for gap in session.gaps:
            self._gap_sum += gap
            self._gap_count += 1

        for length in session.query_lengths:
            self._len_sum += length
            self._len_count += 1

        if self._gap_count > 0:
            self.avg_gap_s = self._gap_sum / self._gap_count
        if self._len_count > 0:
            self.avg_query_length = self._len_sum / self._len_count

    def normalized_command_freq(self) -> dict[str, float]:
        total = sum(self.command_freq.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.command_freq.items()}


class SessionBehavior:
    """Behavioral signals from the current session."""

    __slots__ = (
        "commands", "hours", "apps", "gaps", "query_lengths",
        "action_count",
    )

    def __init__(self) -> None:
        self.commands: Counter = Counter()
        self.hours: Counter = Counter()
        self.apps: Counter = Counter()
        self.gaps: list[float] = []
        self.query_lengths: list[float] = []
        self.action_count: int = 0

    def record(
        self,
        action: str,
        detail: str = "",
        query_text: str = "",
        active_app: str = "",
        hour: int = 0,
        gap_s: float = 0.0,
    ) -> None:
        if action:
            self.commands[action] += 1
        self.hours[hour] += 1
        if active_app:
            self.apps[active_app] += 1
        if gap_s > 0:
            self.gaps.append(gap_s)
            if len(self.gaps) > _MAX_HISTORY:
                self.gaps = self.gaps[-_MAX_HISTORY:]
        if query_text:
            self.query_lengths.append(float(len(query_text.split())))
            if len(self.query_lengths) > _MAX_HISTORY:
                self.query_lengths = self.query_lengths[-_MAX_HISTORY:]
        self.action_count += 1

    def normalized_commands(self) -> dict[str, float]:
        total = sum(self.commands.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.commands.items()}
