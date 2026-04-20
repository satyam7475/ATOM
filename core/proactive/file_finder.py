"""
ATOM -- Natural-language file finder (Sprint D3).

Parses utterances like::

    "find the invoice PDF from last week"
    "find my tax document from this month"
    "find PDFs I downloaded yesterday"
    "locate the receipt file from two days ago"

into a structured :class:`FileQuery` and runs it through Spotlight
(``mdfind``) using metadata predicates so we get far better precision
than the keyword-only ``spotlight_search`` path.

The public surface is small on purpose:

* :func:`parse_file_query` -- NL → :class:`FileQuery` (testable, no I/O)
* :func:`build_mdfind_query` -- :class:`FileQuery` → raw query string
* :func:`find_files_for_text` -- orchestration, returns a tuple of
  ``(summary_text, paths)``

Everything is pure unless ``SpotlightEngine`` is actually invoked.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger("atom.proactive.file_finder")


# ── Kind / type mapping ─────────────────────────────────────────────

_KIND_MAP: dict[str, tuple[str, ...]] = {
    "pdf": ("com.adobe.pdf",),
    "photo": ("public.image",),
    "image": ("public.image",),
    "picture": ("public.image",),
    "screenshot": ("public.image",),
    "video": ("public.movie", "public.video"),
    "movie": ("public.movie",),
    "doc": ("com.microsoft.word.doc", "org.openxmlformats.wordprocessingml.document", "public.rtf"),
    "document": ("com.microsoft.word.doc", "org.openxmlformats.wordprocessingml.document", "public.rtf"),
    "word": ("com.microsoft.word.doc", "org.openxmlformats.wordprocessingml.document"),
    "excel": ("com.microsoft.excel.xls", "org.openxmlformats.spreadsheetml.sheet"),
    "spreadsheet": ("com.microsoft.excel.xls", "org.openxmlformats.spreadsheetml.sheet"),
    "powerpoint": ("com.microsoft.powerpoint.ppt", "org.openxmlformats.presentationml.presentation"),
    "slides": ("com.microsoft.powerpoint.ppt", "org.openxmlformats.presentationml.presentation"),
    "presentation": ("com.microsoft.powerpoint.ppt", "org.openxmlformats.presentationml.presentation"),
    "audio": ("public.audio",),
    "music": ("public.audio",),
    "mp3": ("public.mp3",),
    "zip": ("public.zip-archive",),
    "archive": ("public.archive",),
}

_KIND_ALIASES: dict[str, str] = {
    "pdfs": "pdf",
    "photos": "photo",
    "images": "image",
    "pictures": "picture",
    "screenshots": "screenshot",
    "videos": "video",
    "movies": "movie",
    "documents": "document",
    "docs": "doc",
    "spreadsheets": "spreadsheet",
    "slides": "slides",
    "presentations": "presentation",
    "mp3s": "mp3",
    "audios": "audio",
}

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "my", "some", "any", "all", "that", "this",
    "please", "on", "in", "at", "for", "about", "around",
    "find", "locate", "search", "look", "lookup", "get",
    "show", "pull", "up", "dig", "tell",
    "files", "file", "stuff",
    "want", "need", "can", "you", "me", "i", "to",
    "from", "with", "of", "it",
    "today", "yesterday",
    "week", "month", "year",
    "days", "weeks", "months", "years",
    "ago", "past", "last", "previous",
})

_TIME_UNIT_TO_DAYS: dict[str, int] = {
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30,
    "months": 30,
    "year": 365,
    "years": 365,
}


# ── Data class ──────────────────────────────────────────────────────


@dataclass
class FileQuery:
    keywords: List[str] = field(default_factory=list)
    kind: str = ""
    since_days: int | None = None
    until_days: int | None = None
    only_in: str | None = None
    raw: str = ""

    def describe_scope(self) -> str:
        parts: list[str] = []
        if self.kind:
            parts.append(self.kind)
        if self.since_days is not None:
            parts.append(self._scope_phrase())
        return " ".join(parts).strip()

    def _scope_phrase(self) -> str:
        n = int(self.since_days or 0)
        if n <= 0:
            return "today"
        if n == 1:
            return "yesterday or today"
        if n <= 3:
            return f"in the last {n} days"
        if n <= 8:
            return "last week"
        if n <= 35:
            return "last month"
        return f"in the last {n} days"


# ── Parsing ────────────────────────────────────────────────────────


def _find_kind(text: str) -> tuple[str, str]:
    """Return (canonical_kind_key, matched_word) or ("", "")."""
    lowered = text.lower()
    best: tuple[str, str] = ("", "")
    for word in re.findall(r"[a-z]+", lowered):
        canon = _KIND_ALIASES.get(word, word)
        if canon in _KIND_MAP:
            best = (canon, word)
            if canon in ("pdf", "image", "video", "audio"):
                return best
    return best


def _find_time_scope(text: str) -> int | None:
    """Return ``since_days`` or ``None`` if no scope detected."""
    lowered = text.lower()

    if re.search(r"\btoday\b", lowered):
        return 1
    if re.search(r"\byesterday\b", lowered):
        return 2
    if re.search(r"\bthis\s+week\b", lowered):
        return 7
    if re.search(r"\blast\s+week\b", lowered):
        return 14
    if re.search(r"\bthis\s+month\b", lowered):
        return 30
    if re.search(r"\blast\s+month\b", lowered):
        return 60
    if re.search(r"\bthis\s+year\b", lowered):
        return 365

    m = re.search(
        r"\b(?:past|last|previous)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(day|days|week|weeks|month|months|year|years)\b",
        lowered,
    )
    if m:
        num = _word_to_int(m.group(1))
        unit = m.group(2)
        return num * _TIME_UNIT_TO_DAYS.get(unit, 1)

    m = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b",
        lowered,
    )
    if m:
        num = _word_to_int(m.group(1))
        unit = m.group(2)
        return max(1, num * _TIME_UNIT_TO_DAYS.get(unit, 1) + 1)

    return None


def _word_to_int(w: str) -> int:
    try:
        return int(w)
    except ValueError:
        table = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        return table.get(w.lower(), 1)


def _extract_keywords(text: str, kind_word: str) -> list[str]:
    lowered = re.sub(r"[^a-z0-9\s']", " ", text.lower())
    tokens = [t for t in lowered.split() if t]
    skip_next = 0
    out: list[str] = []
    multi_phrase_stops = {
        ("this", "week"), ("last", "week"), ("this", "month"),
        ("last", "month"), ("this", "year"), ("past", "week"),
        ("past", "month"),
    }
    for i, tok in enumerate(tokens):
        if skip_next > 0:
            skip_next -= 1
            continue
        if tok in _STOPWORDS:
            continue
        if tok == kind_word or _KIND_ALIASES.get(tok, tok) in _KIND_MAP:
            continue
        if tok in {"today", "yesterday"}:
            continue
        if i + 1 < len(tokens):
            pair = (tok, tokens[i + 1])
            if pair in multi_phrase_stops:
                skip_next = 1
                continue
        if tok in {"ago"} and out:
            out.pop()
            continue
        if tok.isdigit():
            continue
        if len(tok) < 2:
            continue
        out.append(tok)
    deduped: list[str] = []
    seen: set[str] = set()
    for tok in out:
        if tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)
    return deduped[:8]


def parse_file_query(text: str) -> FileQuery:
    """Turn a natural-language phrase into a :class:`FileQuery`."""
    raw = (text or "").strip()
    if not raw:
        return FileQuery(raw="")
    kind_key, kind_word = _find_kind(raw)
    since_days = _find_time_scope(raw)
    keywords = _extract_keywords(raw, kind_word)
    return FileQuery(
        keywords=keywords,
        kind=kind_key,
        since_days=since_days,
        raw=raw,
    )


# ── Query compilation ──────────────────────────────────────────────


def build_mdfind_query(q: FileQuery) -> str:
    """Compose an ``mdfind`` expression from a structured :class:`FileQuery`."""
    parts: list[str] = []

    if q.kind and q.kind in _KIND_MAP:
        types = _KIND_MAP[q.kind]
        type_expr = " || ".join(
            f"kMDItemContentTypeTree == '{t}'" for t in types
        )
        parts.append(f"({type_expr})")

    if q.since_days:
        parts.append(f"(kMDItemFSContentChangeDate >= $time.iso(-{int(q.since_days)}d))")

    for kw in q.keywords:
        safe = kw.replace("'", "").replace('"', "")
        if not safe:
            continue
        parts.append(
            f"(kMDItemFSName == '*{safe}*'cd || kMDItemDisplayName == '*{safe}*'cd "
            f"|| kMDItemTextContent == '*{safe}*'cd)"
        )

    if not parts:
        return ""
    return " && ".join(parts)


# ── Orchestration ──────────────────────────────────────────────────


def _pretty_path(path: str) -> str:
    home = os.path.expanduser("~")
    if home and path.startswith(home):
        return "~" + path[len(home):]
    return path


def find_files_for_text(
    text: str,
    *,
    limit: int = 5,
    timeout_s: float = 8.0,
    only_in: str | None = None,
) -> Tuple[str, List[str]]:
    """Main entry point. Returns ``(spoken_summary, list_of_paths)``."""
    query = parse_file_query(text)
    if not (query.keywords or query.kind or query.since_days):
        return (
            "Tell me a bit more — what kind of file, or what to look for?",
            [],
        )

    md_query = build_mdfind_query(query)
    if not md_query:
        return (
            "I couldn't build a search for that, Boss. Try naming the file type or a keyword.",
            [],
        )

    try:
        from core.macos.spotlight_engine import SpotlightEngine
        engine = SpotlightEngine()
        raw_hits = engine.search(md_query, limit=limit, timeout=timeout_s)
    except Exception:
        logger.info("spotlight search failed", exc_info=True)
        return ("I couldn't reach Spotlight just now, Boss.", [])

    paths = [h.get("path", "") for h in raw_hits if h.get("path")]
    paths = [p for p in paths if p]
    if only_in:
        onlyin = os.path.expanduser(only_in)
        paths = [p for p in paths if p.startswith(onlyin)]

    if not paths:
        scope = query.describe_scope()
        if scope:
            return (
                f"I couldn't find any {scope} matching that, Boss.",
                [],
            )
        return ("I couldn't find any files matching that, Boss.", [])

    top = paths[0]
    pretty_top = _pretty_path(top)
    name = os.path.basename(top)
    count = len(paths)
    scope = query.describe_scope()

    if count == 1:
        summary = (
            f"Found one match: {name} at {pretty_top}."
            if not scope
            else f"Found one {scope}: {name} at {pretty_top}."
        )
    else:
        extras = max(0, count - 1)
        summary = (
            f"Found {count} matches. Top result: {name} at {pretty_top}."
            if not scope
            else f"Found {count} {scope}. Top result: {name} at {pretty_top}."
        )
        if extras:
            extras_word = "other" if extras == 1 else "others"
            summary += f" {extras} {extras_word} queued up."

    return summary, paths


__all__ = [
    "FileQuery",
    "parse_file_query",
    "build_mdfind_query",
    "find_files_for_text",
]
