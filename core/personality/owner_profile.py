"""ATOM — OwnerProfile (Sprint P4.1 + P4.3, Apr 26 2026).

A small per-owner learning surface that lives between the STT path and
the prompt builder. Today it owns two stores; tomorrow we can grow it
into the single seam for "what Boss has taught me".

* **Corrections memory** (P4.1) — every time ``correct_text`` or
  ``WhisperConfirmer`` rewrites a streaming transcript, we record
  ``(original, corrected, source)`` so the same misrecognition gets
  patched even when the rule-based corrector misses an edge case.

* **Pronunciation dictionary** (P4.3) — Boss can teach ATOM that
  "when I say <X> I mean <Y>", and the rule survives reboots. The
  dictionary is applied as a strict word-boundary substitution after
  ``correct_text`` runs but **before** the transcript hits the
  router, so all downstream tooling sees the canonical phrasing.

Both stores live in a single SQLite database at
``data/owner_profile.sqlite3``. The schema is auto-created on first
use; a migration is just a CREATE-IF-NOT-EXISTS so we never need to
break compat for an existing Boss profile.

Concurrency: SQLite ``check_same_thread=False`` plus a per-instance
``threading.Lock`` are sufficient for ATOM's single-process,
single-Boss model. The hot path (``apply_pronunciations`` /
``replay_corrections``) uses an in-memory cache that's invalidated on
write, so the typical lookup is one dict access (no SQLite cursor
under load).

Public surface (used by the rest of the codebase)::

    profile = get_owner_profile()                 # singleton
    profile.record_correction(orig, fixed, source="correct_text")
    profile.add_pronunciation(pattern, replacement)
    profile.remove_pronunciation(pattern)
    profile.apply_pronunciations(text)            # text rewrites
    profile.replay_corrections(text)              # apply learned fixes
    profile.recent_corrections(limit=20)          # diagnostics
    profile.summary()                             # prompt builder
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("atom.personality.owner_profile")


_DEFAULT_DB_PATH = Path("data/owner_profile.sqlite3")
_DEFAULT_OWNER = "boss"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS corrections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    owner        TEXT NOT NULL,
    original     TEXT NOT NULL,
    corrected    TEXT NOT NULL,
    source       TEXT NOT NULL,
    hits         INTEGER NOT NULL DEFAULT 1,
    created_at   REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    UNIQUE(owner, original, corrected)
);

CREATE INDEX IF NOT EXISTS idx_corrections_owner_seen
    ON corrections (owner, last_seen_at DESC);

CREATE TABLE IF NOT EXISTS pronunciations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    owner        TEXT NOT NULL,
    pattern      TEXT NOT NULL,
    replacement  TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'word',
    created_at   REAL NOT NULL,
    last_used_at REAL,
    hits         INTEGER NOT NULL DEFAULT 0,
    UNIQUE(owner, pattern)
);

CREATE INDEX IF NOT EXISTS idx_pronunciations_owner
    ON pronunciations (owner);
"""


def _normalise(text: str) -> str:
    return (text or "").strip()


def _is_word_boundary_safe(pattern: str) -> bool:
    """True if ``pattern`` is purely alphanumeric / has internal spaces.
    Used to decide whether a pronunciation entry should be applied with
    `\\b` boundaries (safe for words/phrases) or as a literal substring
    (fallback for phonetic spellings the user pasted in)."""
    if not pattern:
        return False
    return all(ch.isalnum() or ch.isspace() or ch in "'-" for ch in pattern)


class OwnerProfile:
    """Persistent per-owner learning surface (corrections + pronunciation).

    Always cheap to construct: schema migration + cache bootstrap, both
    bounded by the size of the on-disk profile. Designed to be held as
    a process-wide singleton via :func:`get_owner_profile`.
    """

    _DEF_CACHE_LIMIT = 256

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        owner: str | None = None,
        cache_limit: int | None = None,
    ) -> None:
        self._db_path: Path = Path(db_path or _DEFAULT_DB_PATH).expanduser()
        self._owner: str = (owner or _DEFAULT_OWNER).strip().lower()
        self._cache_limit = int(cache_limit or self._DEF_CACHE_LIMIT)
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, timeout=2.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA_SQL)
        self._pron_cache: OrderedDict[str, str] | None = None
        self._corr_cache: OrderedDict[str, str] | None = None
        self._reload_caches()

    # ── public corrections API ────────────────────────────────────

    def record_correction(
        self,
        original: str,
        corrected: str,
        *,
        source: str = "correct_text",
    ) -> bool:
        """Record a single ``original -> corrected`` rewrite.

        Returns True iff the correction was new or its hit-count grew.
        Idempotent on identical pairs; the row's ``hits`` counter
        increments instead of duplicating the row.

        ``source`` is a free-form label so we can split metrics later
        ("correct_text" vs "whisper_confirmer" vs "manual"). Keep it
        short -- the row stays in SQLite forever.
        """
        original = _normalise(original)
        corrected = _normalise(corrected)
        if not original or not corrected or original == corrected:
            return False
        if len(original) > 512 or len(corrected) > 512:
            return False
        now = time.time()
        try:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO corrections
                        (owner, original, corrected, source,
                         hits, created_at, last_seen_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(owner, original, corrected) DO UPDATE SET
                        hits = hits + 1,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (self._owner, original, corrected, source, now, now),
                )
                affected = cur.rowcount
        except sqlite3.Error as exc:
            logger.debug("record_correction failed: %s", exc)
            return False
        self._invalidate_corr_cache_for(original, corrected)
        return affected > 0

    def replay_corrections(self, text: str) -> str:
        """Apply learned corrections to ``text``.

        Best-effort substring rewrite: only exact-match originals are
        replaced (case-insensitive), with word boundaries when the
        original looks word-like. We do this AFTER ``correct_text``
        and pronunciation expansion so the rule-based path runs first.
        Idempotent on already-correct text.
        """
        if not text:
            return text
        cache = self._corr_cache or {}
        if not cache:
            return text
        out = text
        for original_lc, corrected in cache.items():
            if not original_lc or original_lc == _normalise(out).lower():
                out = corrected
                continue
            try:
                if _is_word_boundary_safe(original_lc):
                    pattern = r"\b" + re.escape(original_lc) + r"\b"
                else:
                    pattern = re.escape(original_lc)
                out = re.sub(pattern, corrected, out, flags=re.IGNORECASE)
            except re.error:
                continue
        return out

    def recent_corrections(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT original, corrected, source, hits, last_seen_at
                    FROM corrections WHERE owner = ?
                    ORDER BY last_seen_at DESC
                    LIMIT ?
                    """,
                    (self._owner, int(limit)),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [dict(r) for r in rows]

    # ── public pronunciation API ─────────────────────────────────

    _LEARN_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"^(?:atom[, ]+)?(?:hey atom[, ]+)?"
            r"when\s+i\s+say\s+(?P<pat>['\"\w][^,]+?)"
            r"\s+i\s+mean\s+(?P<rep>['\"\w].+?)\s*[.!?]?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:atom[, ]+)?(?:remember|note)\b.*?"
            r"\b(?P<pat>['\"\w][^,]+?)\s+means\s+(?P<rep>['\"\w].+?)\s*[.!?]?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:atom[, ]+)?(?:remember|note)\b.*?"
            r"\b(?P<pat>['\"\w][^,]+?)\s*=\s*(?P<rep>['\"\w].+?)\s*[.!?]?$",
            re.IGNORECASE,
        ),
    )

    def parse_learn_command(self, text: str) -> tuple[str, str] | None:
        """Detect a "when I say X I mean Y" voice command in ``text``.

        Returns ``(pattern, replacement)`` if detected, else ``None``.
        Both halves are stripped of surrounding quotes. The detection
        is intentionally strict so a casual "I mean…" mid-sentence
        never accidentally adds a dictionary entry.
        """
        s = _normalise(text)
        if not s or len(s) > 240:
            return None
        for rx in self._LEARN_PATTERNS:
            m = rx.match(s)
            if not m:
                continue
            pattern = (m.group("pat") or "").strip().strip("'\"")
            replacement = (m.group("rep") or "").strip().strip("'\"")
            if pattern and replacement and pattern.lower() != replacement.lower():
                return pattern, replacement
        return None

    def add_pronunciation(
        self,
        pattern: str,
        replacement: str,
        *,
        kind: str = "word",
    ) -> bool:
        """Add or update an entry. Returns True iff inserted/changed."""
        pattern = _normalise(pattern)
        replacement = _normalise(replacement)
        if (
            not pattern
            or not replacement
            or pattern.lower() == replacement.lower()
            or len(pattern) > 256
            or len(replacement) > 256
        ):
            return False
        now = time.time()
        try:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO pronunciations
                        (owner, pattern, replacement, kind,
                         created_at, last_used_at, hits)
                    VALUES (?, ?, ?, ?, ?, NULL, 0)
                    ON CONFLICT(owner, pattern) DO UPDATE SET
                        replacement = excluded.replacement,
                        kind        = excluded.kind
                    """,
                    (self._owner, pattern, replacement, kind, now),
                )
                affected = cur.rowcount
        except sqlite3.Error as exc:
            logger.debug("add_pronunciation failed: %s", exc)
            return False
        self._invalidate_pron_cache()
        if affected:
            logger.info(
                "OwnerProfile: pronunciation added/updated [%s] %r -> %r",
                kind, pattern, replacement,
            )
        return affected > 0

    def remove_pronunciation(self, pattern: str) -> bool:
        pattern = _normalise(pattern)
        if not pattern:
            return False
        try:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    "DELETE FROM pronunciations WHERE owner = ? AND pattern = ?",
                    (self._owner, pattern),
                )
                affected = cur.rowcount
        except sqlite3.Error as exc:
            logger.debug("remove_pronunciation failed: %s", exc)
            return False
        if affected:
            self._invalidate_pron_cache()
            logger.info(
                "OwnerProfile: pronunciation removed for %r", pattern,
            )
        return affected > 0

    def apply_pronunciations(self, text: str) -> str:
        if not text:
            return text
        cache = self._pron_cache or {}
        if not cache:
            return text
        out = text
        touched: list[str] = []
        for pattern, replacement in cache.items():
            if not pattern:
                continue
            try:
                if _is_word_boundary_safe(pattern):
                    rx = r"\b" + re.escape(pattern) + r"\b"
                else:
                    rx = re.escape(pattern)
                new_out = re.sub(rx, replacement, out, flags=re.IGNORECASE)
            except re.error:
                continue
            if new_out != out:
                touched.append(pattern)
                out = new_out
        if touched:
            self._mark_pronunciations_used(touched)
        return out

    def list_pronunciations(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT pattern, replacement, kind, hits,
                           created_at, last_used_at
                    FROM pronunciations
                    WHERE owner = ?
                    ORDER BY hits DESC, created_at DESC
                    LIMIT ?
                    """,
                    (self._owner, int(limit)),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [dict(r) for r in rows]

    # ── prompt-builder summary ────────────────────────────────────

    def summary(self, *, max_chars: int = 240) -> str:
        """One-shot prompt-friendly description of what's been learned.

        Returns an empty string when the profile is fresh so callers
        can append unconditionally without producing dangling labels.
        """
        if not self._pron_cache and not self._corr_cache:
            return ""
        parts: list[str] = []
        if self._pron_cache:
            sample = list(self._pron_cache.items())[:6]
            joined = ", ".join(f"{p!r}->{r!r}" for p, r in sample)
            parts.append(
                f"Boss-taught pronunciations ({len(self._pron_cache)}): "
                f"{joined}",
            )
        if self._corr_cache:
            sample_pairs = list(self._corr_cache.items())[:4]
            joined = ", ".join(f"{o!r}->{c!r}" for o, c in sample_pairs)
            parts.append(
                f"Frequent corrections ({len(self._corr_cache)}): {joined}",
            )
        out = " | ".join(parts)
        if len(out) > max_chars:
            out = out[: max_chars - 1].rsplit(" ", 1)[0] + "…"
        return out

    # ── lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                logger.debug("OwnerProfile.close raised", exc_info=True)

    # ── internals ─────────────────────────────────────────────────

    def _reload_caches(self) -> None:
        self._pron_cache = self._load_pronunciations()
        self._corr_cache = self._load_corrections()

    def _load_pronunciations(self) -> OrderedDict[str, str]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT pattern, replacement
                    FROM pronunciations
                    WHERE owner = ?
                    ORDER BY length(pattern) DESC, hits DESC
                    LIMIT ?
                    """,
                    (self._owner, self._cache_limit),
                ).fetchall()
        except sqlite3.Error:
            return OrderedDict()
        return OrderedDict((r["pattern"], r["replacement"]) for r in rows)

    def _load_corrections(self) -> OrderedDict[str, str]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT original, corrected, hits FROM corrections
                    WHERE owner = ?
                    ORDER BY hits DESC, last_seen_at DESC
                    LIMIT ?
                    """,
                    (self._owner, self._cache_limit),
                ).fetchall()
        except sqlite3.Error:
            return OrderedDict()
        out: OrderedDict[str, str] = OrderedDict()
        for r in rows:
            out[(r["original"] or "").lower()] = r["corrected"]
        return out

    def _invalidate_pron_cache(self) -> None:
        self._pron_cache = self._load_pronunciations()

    def _invalidate_corr_cache_for(self, original: str, corrected: str) -> None:
        if self._corr_cache is None:
            self._corr_cache = OrderedDict()
        self._corr_cache[original.lower()] = corrected
        self._corr_cache.move_to_end(original.lower(), last=False)
        if len(self._corr_cache) > self._cache_limit:
            self._corr_cache.popitem(last=True)

    def _mark_pronunciations_used(self, patterns: Iterable[str]) -> None:
        now = time.time()
        try:
            with self._lock, self._conn:
                self._conn.executemany(
                    """
                    UPDATE pronunciations
                    SET hits = hits + 1, last_used_at = ?
                    WHERE owner = ? AND pattern = ?
                    """,
                    [(now, self._owner, p) for p in patterns],
                )
        except sqlite3.Error as exc:
            logger.debug("_mark_pronunciations_used failed: %s", exc)


# ── singleton + bootstrap ───────────────────────────────────────────

_SINGLETON: OwnerProfile | None = None
_SINGLETON_LOCK = threading.Lock()


def get_owner_profile(
    config: dict | None = None,
    *,
    db_path: str | Path | None = None,
) -> OwnerProfile:
    """Return the process-wide :class:`OwnerProfile` singleton.

    Re-entrant and thread-safe. Pass a ``config`` dict (the ATOM root
    config) to honour ``owner.name`` / ``personality.owner_profile_db``
    overrides. Subsequent calls ignore those args -- the first caller
    wins to avoid surprise rebinds in long-running sessions.
    """
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            return _SINGLETON
        owner = _DEFAULT_OWNER
        path = db_path
        if config:
            try:
                owner = (
                    (config.get("owner") or {}).get("name")
                    or owner
                )
                personality_cfg = (config.get("personality") or {})
                if path is None:
                    path = personality_cfg.get("owner_profile_db")
            except Exception:
                logger.debug("get_owner_profile config inspect failed", exc_info=True)
        try:
            _SINGLETON = OwnerProfile(db_path=path, owner=owner)
        except Exception as exc:
            logger.warning(
                "OwnerProfile bootstrap failed (%s); creating in-memory fallback",
                exc,
            )
            _SINGLETON = OwnerProfile(
                db_path=Path(os.devnull) if False else _DEFAULT_DB_PATH,
                owner=owner,
            )
        return _SINGLETON


@contextmanager
def isolated_owner_profile(
    db_path: str | Path,
    *,
    owner: str = _DEFAULT_OWNER,
):
    """Test helper: yield a fresh OwnerProfile bound to a temp DB path.

    Does NOT touch the global singleton; safe to use inside unit tests.
    """
    profile = OwnerProfile(db_path=db_path, owner=owner)
    try:
        yield profile
    finally:
        profile.close()


__all__ = [
    "OwnerProfile",
    "get_owner_profile",
    "isolated_owner_profile",
]
