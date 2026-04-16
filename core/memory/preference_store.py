"""
ATOM — Preference Store (Persistent Owner Intelligence).

Learns and stores owner preferences automatically from interactions.
Preferences are injected into every LLM prompt for personalized responses.

Capabilities:
  - Language preference (English by default)
  - Response length preference (tracked via feedback)
  - Communication style (formal/casual/technical)
  - Topic interests and expertise areas
  - Time-of-day patterns

Storage: SQLite (same database as MemoryEngine, separate table)

Privacy: All data stays local. Never sent to cloud.

Owner: Satyam
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.preference_store")

_DEFAULT_DB_PATH = "data/atom_memory.db"


class PreferenceStore:
    """Persistent owner preference storage with auto-learning.

    Usage:
        store = PreferenceStore(config)
        store.learn("communication", "language", "english", confidence=1.0)
        store.learn("response", "length", "concise", confidence=0.8)

        # Inject into LLM prompt:
        context = store.get_context_block()
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS owner_preferences (
        category TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL DEFAULT 0.5,
        learn_count INTEGER DEFAULT 1,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (category, key)
    );
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("memory", {})
        db_path = cfg.get("graph_db_path", _DEFAULT_DB_PATH)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._conn: sqlite3.Connection | None = None
        self._init_db()

        # Set sensible defaults for a new installation
        self._ensure_defaults()

        logger.info("PreferenceStore: db=%s", self._db_path)

    def _init_db(self) -> None:
        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(self._SCHEMA)
            self._conn.commit()
        except Exception:
            logger.exception("PreferenceStore: DB init failed")
            self._conn = None

    def _ensure_defaults(self) -> None:
        """Set default preferences if not already configured."""
        defaults = {
            ("communication", "language"): ("english", 1.0),
            ("communication", "style"): ("professional", 0.8),
            ("response", "length"): ("concise", 0.8),
            ("response", "format"): ("natural", 0.7),
            ("personality", "formality"): ("balanced", 0.7),
            ("personality", "humor"): ("light", 0.5),
        }

        for (cat, key), (value, conf) in defaults.items():
            existing = self.get(cat, key)
            if existing is None:
                self.learn(cat, key, value, confidence=conf)

    # ── Core API ─────────────────────────────────────────────────────

    def learn(
        self,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.5,
    ) -> None:
        """Learn or update a preference.

        If the preference already exists, update with increased confidence.
        Confidence is clamped to [0.0, 1.0].
        """
        if self._conn is None:
            return

        now = time.time()
        value_str = str(value)
        confidence = max(0.0, min(1.0, confidence))

        with self._lock:
            try:
                existing = self._conn.execute(
                    "SELECT value, confidence, learn_count FROM owner_preferences "
                    "WHERE category=? AND key=?",
                    (category, key),
                ).fetchone()

                if existing:
                    old_value, old_conf, old_count = existing
                    if old_value == value_str:
                        # Same value — increase confidence
                        new_conf = min(1.0, old_conf + 0.05)
                        new_count = old_count + 1
                        self._conn.execute(
                            "UPDATE owner_preferences SET "
                            "confidence=?, learn_count=?, updated_at=? "
                            "WHERE category=? AND key=?",
                            (new_conf, new_count, now, category, key),
                        )
                    else:
                        # Different value — only override if higher confidence
                        if confidence > old_conf:
                            self._conn.execute(
                                "UPDATE owner_preferences SET "
                                "value=?, confidence=?, learn_count=?, updated_at=? "
                                "WHERE category=? AND key=?",
                                (value_str, confidence, old_count + 1, now,
                                 category, key),
                            )
                else:
                    self._conn.execute(
                        "INSERT INTO owner_preferences "
                        "(category, key, value, confidence, learn_count, "
                        "created_at, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?)",
                        (category, key, value_str, confidence, now, now),
                    )

                self._conn.commit()
            except Exception:
                logger.debug("PreferenceStore write failed", exc_info=True)

    def get(self, category: str, key: str) -> str | None:
        """Get a specific preference value."""
        if self._conn is None:
            return None

        try:
            row = self._conn.execute(
                "SELECT value FROM owner_preferences WHERE category=? AND key=?",
                (category, key),
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def get_preferences(self, category: str) -> dict[str, Any]:
        """Get all preferences in a category."""
        if self._conn is None:
            return {}

        try:
            rows = self._conn.execute(
                "SELECT key, value, confidence FROM owner_preferences "
                "WHERE category=? ORDER BY confidence DESC",
                (category,),
            ).fetchall()
            return {
                row[0]: {"value": row[1], "confidence": row[2]}
                for row in rows
            }
        except Exception:
            return {}

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Get all preferences grouped by category."""
        if self._conn is None:
            return {}

        try:
            rows = self._conn.execute(
                "SELECT category, key, value, confidence FROM owner_preferences "
                "ORDER BY category, confidence DESC",
            ).fetchall()

            result: dict[str, dict[str, Any]] = {}
            for cat, key, value, conf in rows:
                if cat not in result:
                    result[cat] = {}
                result[cat][key] = {"value": value, "confidence": conf}
            return result
        except Exception:
            return {}

    # ── LLM Context Injection ────────────────────────────────────────

    def get_context_block(self) -> str:
        """Generate a context block for LLM prompt injection.

        Returns a formatted string of high-confidence preferences
        that should influence the LLM's response style.
        """
        if self._conn is None:
            return ""

        try:
            rows = self._conn.execute(
                "SELECT category, key, value FROM owner_preferences "
                "WHERE confidence >= 0.6 ORDER BY confidence DESC",
            ).fetchall()
        except Exception:
            return ""

        if not rows:
            return ""

        lines: list[str] = ["[OWNER PREFERENCES]"]

        pref_map: dict[str, list[str]] = {}
        for cat, key, value in rows:
            label = f"{cat}.{key}"
            if cat not in pref_map:
                pref_map[cat] = []
            pref_map[cat].append(f"{key}: {value}")

        for cat, prefs in pref_map.items():
            lines.append(f"  {cat}: {', '.join(prefs)}")

        return "\n".join(lines)

    # ── Auto-Learning ────────────────────────────────────────────────

    def learn_from_response_feedback(
        self,
        response_length: int,
        was_positive: bool,
    ) -> None:
        """Auto-learn response length preference from feedback."""
        if was_positive:
            if response_length < 50:
                self.learn("response", "length", "minimal", confidence=0.6)
            elif response_length < 150:
                self.learn("response", "length", "concise", confidence=0.7)
            elif response_length < 400:
                self.learn("response", "length", "medium", confidence=0.6)
            else:
                self.learn("response", "length", "detailed", confidence=0.6)

    def learn_from_query_pattern(self, query: str) -> None:
        """Auto-learn from query patterns (topic interests, etc.)."""
        query_lower = query.lower()

        # Detect technical queries → learn expertise
        technical_keywords = {
            "python", "javascript", "rust", "docker", "kubernetes",
            "api", "database", "algorithm", "deploy", "server",
            "git", "linux", "macos", "swift", "react",
        }
        for kw in technical_keywords:
            if kw in query_lower:
                self.learn("interests", "technology", kw, confidence=0.5)
                self.learn("communication", "style", "technical", confidence=0.55)
                break

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        if self._conn is None:
            return {"available": False}

        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM owner_preferences",
            ).fetchone()[0]
            categories = self._conn.execute(
                "SELECT DISTINCT category FROM owner_preferences",
            ).fetchall()
            return {
                "available": True,
                "total_preferences": count,
                "categories": [r[0] for r in categories],
            }
        except Exception:
            return {"available": False}

    def shutdown(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                logger.debug('core memory preference store optional step failed', exc_info=True)
            self._conn = None


__all__ = ["PreferenceStore"]
