"""ATOM — OwnerStyleAdapter (Sprint P4.2, Apr 26 2026).

A tiny rolling fingerprint of how Boss talks, distilled into a single
compact "style block" the prompt builder can splice in. The
StructuredPromptBuilder stays the source of truth; we just inform it.

Signals tracked
---------------

* **Hinglish ratio** — the share of recent turns that mix Hindi+English
  characters or Devanagari script. Drives the model's language pull.
* **Mean tokens / turn** — Boss's verbosity. Short utterances bias the
  reply toward terse single-sentence answers; long turns let the LLM
  expand without feeling chatty.
* **Tone** — formal vs casual via a tiny lexicon (please/kindly/sir vs
  bro/dude/yaar/ki/please-already). This is intentionally rough: it's
  a hint, not a classifier.
* **Imperative ratio** — fraction of turns starting with a command verb
  (open, run, kill, read, show…). High ratio = Boss prefers ATOM as a
  tool; ATOM should drop "Would you like me to…" framing.

The adapter holds an in-memory deque (last 200 user turns) and
recomputes the fingerprint lazily on read. It is feature-flagged via
``config["personality"]["owner_style"]["enabled"]`` (defaults true) so
operators can disable the prompt insertion without ripping wiring.

Public surface::

    style = get_owner_style(config)        # singleton
    style.observe(user_text)               # call from final transcript
    block = style.style_block_for_prompt() # one-line hint or ""
    fingerprint = style.fingerprint()      # dict for diagnostics
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

logger = logging.getLogger("atom.personality.owner_style")


_DEFAULT_WINDOW = 200
_DEFAULT_MIN_OBSERVATIONS = 12

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_HINGLISH_LATIN_TOKENS = frozenset({
    "kya", "kyun", "nahi", "nahin", "haan", "thik", "theek", "acha", "accha",
    "aur", "kaise", "kab", "kahan", "yaar", "bhai", "bhaiya", "bhaisaab",
    "abhi", "matlab", "samajh", "samjha", "lekin", "magar", "kuch", "phir",
    "chal", "chalo", "ruk", "kar", "kya", "kaam", "kar", "do", "dega", "deko",
    "batao", "bhej", "le", "leke", "raha", "rahi", "ho", "hai", "hain",
    "mai", "main", "mujhe", "tu", "tum", "tumhe", "aap", "aapko", "ji",
    "kuchh", "thoda", "zyada", "kam", "wala", "waali", "wali", "ka", "ki",
    "ke", "se", "tak", "par", "kuchh", "ho gaya", "ho jaye",
})
_FORMAL_TOKENS = frozenset({
    "please", "kindly", "could", "would", "may", "sir", "ma'am", "thank",
})
_CASUAL_TOKENS = frozenset({
    "bro", "dude", "yo", "yaar", "buddy", "mate", "lol", "nah", "yep",
    "yup", "lmao", "haha", "ki", "kar", "abhi",
})
_IMPERATIVE_LEAD = frozenset({
    "open", "close", "run", "kill", "list", "show", "play", "pause",
    "stop", "start", "read", "search", "find", "summarize", "summarise",
    "tell", "remind", "set", "create", "add", "delete", "remove", "update",
    "fix", "build", "git", "commit", "push", "pull", "merge", "make",
    "shutdown", "lock", "unlock", "mute", "unmute", "increase", "decrease",
    "switch", "go", "click", "type", "schedule", "ping", "ssh",
    # Hinglish imperatives
    "kar", "do", "kardo", "rok", "band",
})


@dataclass
class _Observation:
    text: str
    tokens: int
    has_devanagari: bool
    has_hinglish_latin: bool
    has_formal: bool
    has_casual: bool
    starts_imperative: bool


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    nf = unicodedata.normalize("NFKC", text).lower()
    return [
        tok for tok in re.split(r"[^a-z\u0900-\u097F0-9']+", nf) if tok
    ]


def _classify(text: str) -> _Observation | None:
    if not text or len(text) > 1024:
        return None
    s = text.strip()
    if not s:
        return None
    toks = _tokenise(s)
    if not toks:
        return None
    has_dev = bool(_DEVANAGARI_RE.search(s))
    tok_set = set(toks)
    has_latin_hin = bool(tok_set & _HINGLISH_LATIN_TOKENS)
    has_formal = bool(tok_set & _FORMAL_TOKENS)
    has_casual = bool(tok_set & _CASUAL_TOKENS)
    leading = toks[0]
    starts_imp = leading in _IMPERATIVE_LEAD
    return _Observation(
        text=s,
        tokens=len(toks),
        has_devanagari=has_dev,
        has_hinglish_latin=has_latin_hin,
        has_formal=has_formal,
        has_casual=has_casual,
        starts_imperative=starts_imp,
    )


class OwnerStyleAdapter:
    """Rolling fingerprint of Boss's recent style.

    Observations are appended on :meth:`observe`; the fingerprint is
    recomputed lazily (and cached) on :meth:`fingerprint`. The lazy
    recompute is dirt-cheap relative to the surrounding LLM call but
    we still avoid doing it on the audio-tap thread.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_size: int = _DEFAULT_WINDOW,
        min_observations: int = _DEFAULT_MIN_OBSERVATIONS,
    ) -> None:
        self._enabled = bool(enabled)
        self._window: Deque[_Observation] = deque(maxlen=int(window_size))
        self._min_observations = int(min_observations)
        self._lock = threading.Lock()
        self._cache: dict[str, Any] | None = None

    def observe(self, text: str) -> None:
        if not self._enabled:
            return
        obs = _classify(text)
        if obs is None:
            return
        with self._lock:
            self._window.append(obs)
            self._cache = None

    def fingerprint(self) -> dict[str, Any]:
        if not self._enabled:
            return {"enabled": False}
        with self._lock:
            if self._cache is not None:
                return dict(self._cache)
            n = len(self._window)
            if n == 0:
                self._cache = {
                    "enabled": True,
                    "samples": 0,
                    "ready": False,
                }
                return dict(self._cache)
            hin_dev = sum(1 for o in self._window if o.has_devanagari)
            hin_latin = sum(1 for o in self._window if o.has_hinglish_latin)
            formal = sum(1 for o in self._window if o.has_formal)
            casual = sum(1 for o in self._window if o.has_casual)
            imps = sum(1 for o in self._window if o.starts_imperative)
            mean_tokens = sum(o.tokens for o in self._window) / n
            hinglish_ratio = (hin_dev + hin_latin) / n
            formality = (formal - casual) / max(1, n)
            self._cache = {
                "enabled": True,
                "samples": n,
                "ready": n >= self._min_observations,
                "hinglish_ratio": round(hinglish_ratio, 2),
                "mean_tokens": round(mean_tokens, 1),
                "tone": (
                    "formal" if formality >= 0.2
                    else "casual" if formality <= -0.2
                    else "neutral"
                ),
                "imperative_ratio": round(imps / n, 2),
            }
            return dict(self._cache)

    def style_block_for_prompt(self) -> str:
        """One-line hint suitable for the dynamic context layer.

        Returns ``""`` until the rolling window has at least
        ``min_observations`` so we don't bias the model on a thin
        sample. The wording is intentionally directive ("prefer", "lean
        towards") so the LLM treats it as guidance, not policy.
        """
        fp = self.fingerprint()
        if not fp.get("ready"):
            return ""
        bits: list[str] = []
        hin = float(fp.get("hinglish_ratio") or 0.0)
        if hin >= 0.4:
            bits.append("Boss frequently mixes Hindi/Hinglish; match it")
        elif hin >= 0.15:
            bits.append("Boss sometimes uses Hinglish; mirror only when he does")
        mean_tokens = float(fp.get("mean_tokens") or 0.0)
        if mean_tokens <= 6:
            bits.append("Boss is terse; reply in 1 short sentence by default")
        elif mean_tokens >= 18:
            bits.append("Boss writes long; you may answer in 2-3 sentences")
        tone = fp.get("tone")
        if tone == "casual":
            bits.append("tone: casual, no 'kindly'")
        elif tone == "formal":
            bits.append("tone: respectful, slightly formal")
        if float(fp.get("imperative_ratio") or 0.0) >= 0.5:
            bits.append("Boss issues commands; skip 'Would you like me to'")
        if not bits:
            return ""
        return "OWNER STYLE (learned, last %d turns): %s." % (
            int(fp.get("samples") or 0),
            "; ".join(bits),
        )

    def reset(self) -> None:
        with self._lock:
            self._window.clear()
            self._cache = None

    @property
    def enabled(self) -> bool:
        return self._enabled


# ── singleton ────────────────────────────────────────────────────────

_SINGLETON: OwnerStyleAdapter | None = None
_SINGLETON_LOCK = threading.Lock()


def get_owner_style(config: dict | None = None) -> OwnerStyleAdapter:
    """Return the process-wide :class:`OwnerStyleAdapter` singleton."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            return _SINGLETON
        cfg = ((config or {}).get("personality") or {}).get("owner_style") or {}
        _SINGLETON = OwnerStyleAdapter(
            enabled=bool(cfg.get("enabled", True)),
            window_size=int(cfg.get("window_size", _DEFAULT_WINDOW)),
            min_observations=int(
                cfg.get("min_observations", _DEFAULT_MIN_OBSERVATIONS),
            ),
        )
        return _SINGLETON


__all__ = ["OwnerStyleAdapter", "get_owner_style"]
