"""
ATOM -- Native macOS TTS (NSSpeechSynthesizer + say fallback).

Uses the same **Apple on-device speech synthesis** APIs: ``NSSpeechSynthesizer`` with
resolution order ``defaultVoice`` → **Accessibility Spoken Content prefs**
(``com.apple.speech.voice.prefs``) so **Siri (Voice 2)** and similar picks match System Settings.
Recognition uses **SFSpeechRecognizer** (see ``voice/stt_macos.py``).

Two backends in priority order:
  1. NSSpeechSynthesizer (pyobjc) — no subprocess, direct AppKit API,
     premium/neural voice support, instant barge-in. ~0ms spawn overhead.
  2. `say` subprocess — fallback if pyobjc not installed. ~5ms spawn.

Both are fully offline and run on the Neural Engine for premium voices.

Same public interface as EdgeTTSAsync / TTSAsync for drop-in replacement.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import collections
import logging
import plistlib
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brain._speech_sanitizer import StreamingLeakBuffer

logger = logging.getLogger("atom.tts_macos")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager


# v3 prompt-text-leak fingerprint (mirror of
# cursor_bridge.local_brain_controller._PROMPT_LEAK_FINGERPRINT_RE).
#
# This is the FINAL audio guard. The controller already drops these but
# the network of code paths that reach TTS (cloud streaming, ack engine,
# direct text_display fallback, recovery prompts) is wide enough that a
# defensive check here is cheap insurance. If we ever speak our own
# system-prompt rules out loud again, this is the last fence.
_PROMPT_LEAK_FINGERPRINT_RE = re.compile(
    r"""
    ^\s*
    (?:
        the\s+final\s+answer\s+only\b |
        reply\s+with\s+the\s+final\s+answer\b |
        one\s+short\s+(?:jarvis-style\s+)?line\b |
        plain\s+text\s+only\b |
        if\s+the\s+question\s+is\s+a\s+simple,?\s+short,?\s+or\s+info\s+query\b |
        give\s+one\s+short\s+sentence\s+when\s+possible\b |
        two\s+short\s+sentences\s+max\b |
        output\s+only\s+the\s+final\s+answer\b |
        spoken\s*=\s*final\s+answer\b |
        if\s+the\s+thought\s+feels\s+like\s+planning\b |
        boss\s+only\s+hears\s+what'?s\s+spoken\b |
        respond\s+in\s+plain\s+text\b |
        no\s+markdown,?\s+no\s+bullets\b |
        output\s+style\s*:\s+spoken\s+plain\s+text\b |
        brevity\s+required\.\s+aim\s+for\s+~?\s*15\s+words\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_prompt_leak(text: str) -> bool:
    if not text:
        return False
    head = text.strip()
    if not head:
        return False
    return bool(_PROMPT_LEAK_FINGERPRINT_RE.match(head))

# ── Try to import pyobjc for native synthesis ─────────────────────────
_HAS_NATIVE = False
_AppKit: Any = None
_Foundation: Any = None
try:
    import AppKit as _AppKit      # type: ignore[import-untyped]
    import Foundation as _Foundation  # type: ignore[import-untyped]
    _HAS_NATIVE = True
except ImportError:
    pass

# ── Markdown cleanup ─────────────────────────────────────────────────
_RE_CODE_BLOCK = re.compile(r'```.*?```', re.DOTALL)
_RE_INLINE_CODE = re.compile(r'`([^`]*)`')
_RE_BOLD = re.compile(r'\*\*([^*]+)\*\*')
_RE_ITALIC_STAR = re.compile(r'\*([^*]+)\*')
_RE_ITALIC_UNDER = re.compile(r'_([^_]+)_')
_RE_HEADER = re.compile(r'^#+\s*', re.MULTILINE)
_RE_BULLET = re.compile(r'^\s*[-*\u2022]\s+', re.MULTILINE)
_RE_NUMBERED = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r'^\s*>\s+', re.MULTILINE)
_RE_TRANSCRIPT_LABEL = re.compile(r"\b(?:User|Boss|ATOM|Assistant):\s*", re.I)
_RE_TRANSCRIPT_ONLY = re.compile(r"^(?:(?:User|Boss|ATOM|Assistant)\s*:?\s*)+$", re.I)
_RE_INTERNAL_TTS_LINE = re.compile(
    r"^\s*(?:\[?\s*system\s*\]?\s+initiating\s+system\s+diagnostics|"
    r"atom\.?\s*local\s*brain|atom\.?\s*localbrain|"
    r"system\s+is\s+degraded\s*,?\s+boss\.\s+issues?:\s*"
    r".*readiness\s+checks?)\s*\.?\s*$",
    re.I,
)

_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
# First grammatically complete sentence at buffer start (non-greedy up to first . ! ?)
_RE_FIRST_SENTENCE = re.compile(r"^(.+?[.!?])\s*", re.DOTALL)

ACK_PHRASES = [
    "Yes, Boss?", "I'm here.", "Go ahead.", "I'm listening.", "How can I help?",
    "What do you need?", "Standing by.", "Ready when you are.",
    "One moment.", "Certainly.", "Working on that.",
    "Give me a moment.", "Let me check.", "Right away, Boss.",
    "One second.", "Pulling that up now.", "Checking now.", "Just a moment.",
    "I didn't quite catch that — could you repeat?",
    "That didn't go through. Shall I try again?",
    "Done.", "Complete.", "All set.",
    "Searching now.", "Here are the results.",
    "On it.", "Processing now.",
]


def _clean_for_tts(text: str) -> str:
    """Strip markdown so the synthesizer speaks clean prose."""
    if _RE_INTERNAL_TTS_LINE.search(text or ""):
        return ""
    text = "\n".join(
        line for line in str(text or "").splitlines()
        if not _RE_INTERNAL_TTS_LINE.search(line)
    )
    text = _RE_CODE_BLOCK.sub('', text)
    text = _RE_INLINE_CODE.sub(r'\1', text)
    text = _RE_BOLD.sub(r'\1', text)
    text = _RE_ITALIC_STAR.sub(r'\1', text)
    text = _RE_ITALIC_UNDER.sub(r'\1', text)
    text = _RE_HEADER.sub('', text)
    text = _RE_BULLET.sub('', text)
    text = _RE_NUMBERED.sub('', text)
    text = _RE_BLOCKQUOTE.sub('', text)
    return text.strip()


def _truncate(text: str, max_lines: int = 4) -> str:
    text = _clean_for_tts(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return " ".join(lines)
    return " ".join(lines[:max_lines])


def _split_sentences(text: str) -> list:
    parts = _RE_SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


# ── Voice discovery ──────────────────────────────────────────────────

# Aliases: use the same default voice as System Settings → Accessibility → Spoken Content
# (NSSpeechSynthesizer defaultVoice — Apple’s on-device neural TTS family as Siri uses).
_SYSTEM_VOICE_ALIASES = frozenset({
    "system", "default", "siri", "apple", "apple_siri", "match_system", "",
})

# Preset voice requests with a stable quality order. ``jarvis`` intentionally
# prefers a British voice family, falling back to Daniel compact on a stock macOS
# install and auto-upgrading to enhanced/premium voices if the user installs them.
_VOICE_PRESETS: dict[str, tuple[str, ...]] = {
    "jarvis": (
        "com.apple.voice.premium.en-GB.Daniel",
        "com.apple.voice.enhanced.en-GB.Daniel",
        "com.apple.voice.compact.en-GB.Daniel",
        "com.apple.voice.premium.en-GB.Serena",
        "com.apple.voice.enhanced.en-GB.Serena",
        "com.apple.voice.premium.en-GB.Kate",
        "com.apple.voice.enhanced.en-GB.Kate",
        "com.apple.voice.premium.en-GB.Martha",
        "com.apple.voice.enhanced.en-GB.Martha",
    ),
}

# When no explicit voice matches: premium neural first, then compact natural
# voices, then Eloquence as last resort. Warm feminine voices are preferred
# for the "Friday" assistant personality.
_PREFERRED_VOICES = [
    # Premium / enhanced neural (Siri-class, require download from Spoken Content settings)
    "com.apple.voice.premium.en-GB.Daniel",
    "com.apple.voice.enhanced.en-GB.Daniel",
    "com.apple.voice.premium.en-US.Samantha",
    "com.apple.voice.enhanced.en-US.Samantha",
    "com.apple.voice.premium.en-GB.Serena",
    "com.apple.voice.enhanced.en-GB.Serena",
    "com.apple.voice.premium.en-IN.Rani",
    "com.apple.voice.enhanced.en-IN.Rani",
    "com.apple.voice.premium.en-AU.Karen",
    "com.apple.voice.enhanced.en-AU.Karen",
    "com.apple.voice.premium.en-GB.Kate",
    "com.apple.voice.enhanced.en-GB.Kate",
    "com.apple.voice.premium.en-GB.Martha",
    "com.apple.voice.enhanced.en-GB.Martha",
    "com.apple.voice.premium.en-US.Zoe",
    "com.apple.voice.enhanced.en-US.Zoe",
    # Compact natural voices (pre-installed, decent quality)
    "com.apple.voice.compact.en-US.Samantha",
    "com.apple.voice.Tara",
    "com.apple.voice.compact.en-AU.Karen",
    "com.apple.voice.compact.en-IE.Moira",
    "com.apple.voice.compact.en-ZA.Tessa",
    "com.apple.voice.compact.en-GB.Daniel",
    # Eloquence (robotic, avoid unless nothing else is available)
    "com.apple.eloquence.en-GB.Shelley",
    "com.apple.eloquence.en-US.Eddy",
]


def _voice_quality_rank(voice_id: str) -> tuple[int, int]:
    lower = (voice_id or "").lower()
    if "premium" in lower or "enhanced" in lower:
        quality = 0
    elif "compact" in lower:
        quality = 1
    elif "eloquence" in lower:
        quality = 3
    else:
        quality = 2
    return (quality, len(voice_id or ""))


def _preferred_pitch_shift(voice_id: str) -> float:
    lower = (voice_id or "").lower()
    if not lower or "eloquence" in lower:
        return 0.0
    if any(token in lower for token in ("daniel", "eddy", "reed", "rocko", "grandpa")):
        return 0.0
    return 2.0


def list_voices_native() -> list[dict]:
    """List available macOS voices via NSSpeechSynthesizer. Returns dicts with
    id, name, locale, is_premium keys."""
    if not _HAS_NATIVE:
        return []
    voices = []
    for vid in _AppKit.NSSpeechSynthesizer.availableVoices():
        attrs = _AppKit.NSSpeechSynthesizer.attributesForVoice_(vid)
        name = attrs.get(_AppKit.NSVoiceName, "")
        locale = attrs.get(_AppKit.NSVoiceLocaleIdentifier, "")
        is_premium = "premium" in vid.lower() or "enhanced" in vid.lower()
        voices.append({
            "id": str(vid), "name": str(name),
            "locale": str(locale), "is_premium": is_premium,
        })
    return voices


def list_voices() -> list:
    """Return available macOS voices as [(name, locale, sample), ...]."""
    if _HAS_NATIVE:
        return [
            (v["name"], v["locale"], "")
            for v in list_voices_native()
        ]
    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True, text=True, timeout=5,
        )
        voices = []
        for line in result.stdout.strip().splitlines():
            match = re.match(
                r'^(.+?)\s{2,}(\w{2}_\w{2})\s+#\s*(.*)$', line,
            )
            if match:
                voices.append((
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                ))
        return voices
    except Exception as exc:
        logger.warning("Could not list macOS voices: %s", exc)
        return []


def _match_voice_by_display_name(available: set[str], display: str) -> str:
    """Match Accessibility label (e.g. 'Siri (Voice 2)') to an NSSpeechSynthesizer voice id."""
    dlow = (display or "").strip().lower()
    if not dlow:
        return ""
    for vid in available:
        attrs = _AppKit.NSSpeechSynthesizer.attributesForVoice_(vid)
        name = str(attrs.get(_AppKit.NSVoiceName, "") or "")
        if not name:
            continue
        nl = name.lower()
        if dlow == nl or dlow in nl or nl in dlow:
            return str(vid)
    return ""


def _spoken_content_voice_from_prefs(available: set[str]) -> str:
    """Read System Settings → Accessibility → Spoken Content voice from Apple prefs.

    ``NSSpeechSynthesizer.defaultVoice()`` is often empty in CLI/headless processes; the
    user-selected voice is still stored under ``com.apple.speech.voice.prefs`` (see
    ``~/Library/Preferences/com.apple.speech.voice.prefs``).
    """
    id_keys = (
        "SelectedVoiceID",
        "SelectedSystemVoiceID",
        "SystemVoiceID",
        "TTSSelectedVoiceIdentifier",
        "NSSpeechSelectedVoice",
    )

    def try_dict(d: Any) -> str:
        if not isinstance(d, dict):
            return ""
        for key in id_keys:
            val = d.get(key)
            if val is None:
                continue
            s = str(val).strip()
            if s and s in available:
                return s
        for val in d.values():
            if isinstance(val, str) and "com.apple." in val and val in available:
                return val
        name = d.get("SelectedVoiceName")
        if name:
            hit = _match_voice_by_display_name(available, str(name))
            if hit:
                return hit
        return ""

    if _Foundation is None:
        return ""

    try:
        ud = _Foundation.NSUserDefaults.standardUserDefaults()
        dom = ud.persistentDomainForName_("com.apple.speech.voice.prefs")
        if dom:
            hit = try_dict(dict(dom))
            if hit:
                logger.info(
                    "TTS: using Spoken Content voice from UserDefaults (Accessibility): %s",
                    hit,
                )
                return hit
    except Exception:
        logger.debug("speech.voice.prefs UserDefaults failed", exc_info=True)

    plist_path = Path.home() / "Library/Preferences/com.apple.speech.voice.prefs.plist"
    try:
        if plist_path.is_file():
            with plist_path.open("rb") as fp:
                data = plistlib.load(fp)
            hit = try_dict(data)
            if hit:
                logger.info(
                    "TTS: using Spoken Content voice from prefs plist: %s",
                    hit,
                )
                return hit
    except Exception:
        logger.debug("speech.voice.prefs plist read failed", exc_info=True)

    return ""


def _pick_best_voice(requested: str) -> str:
    """Find the best available voice. Returns voice identifier string.

    For ``system`` / ``siri`` / ``default``: try ``NSSpeechSynthesizer.defaultVoice()``,
    then ``com.apple.speech.voice.prefs`` (Accessibility → Spoken Content), then a
    neural fallback list. For preset aliases like ``jarvis``, use a deterministic
    quality-ordered list rather than whichever matching identifier happens to come
    first from ``availableVoices()``.
    """
    if not _HAS_NATIVE:
        return requested

    available = {str(v) for v in _AppKit.NSSpeechSynthesizer.availableVoices()}
    req_raw = (requested or "").strip()
    req_lower = req_raw.lower()

    if req_lower in _SYSTEM_VOICE_ALIASES:
        try:
            dv = _AppKit.NSSpeechSynthesizer.defaultVoice()
            ds = str(dv) if dv is not None else ""
            if ds and ds in available:
                logger.info(
                    "TTS: using NSSpeechSynthesizer.defaultVoice (system neural): %s",
                    ds,
                )
                return ds
            if not ds:
                logger.debug("NSSpeechSynthesizer.defaultVoice empty; trying Spoken Content prefs")
            else:
                logger.warning(
                    "TTS: defaultVoice %s not in availableVoices — trying Spoken Content prefs",
                    ds,
                )
        except Exception:
            logger.debug("NSSpeechSynthesizer.defaultVoice failed", exc_info=True)

        spoken = _spoken_content_voice_from_prefs(available)
        if spoken:
            return spoken

        logger.info(
            "TTS: no Accessibility/Spoken Content voice resolved — using preferred on-device neural voice",
        )

    if req_raw and req_lower not in _SYSTEM_VOICE_ALIASES:
        preset = _VOICE_PRESETS.get(req_lower)
        if preset is not None:
            for vid in preset:
                if vid in available:
                    logger.info("TTS: using preset voice '%s' -> %s", req_raw, vid)
                    return vid

        matches = sorted(
            (vid for vid in available if req_lower in vid.lower()),
            key=lambda vid: (
                0 if vid.rsplit(".", 1)[-1].lower() == req_lower else 1,
                _voice_quality_rank(vid),
            ),
        )
        if matches:
            chosen = matches[0]
            logger.info(
                "TTS: matched requested voice substring '%s' -> %s",
                req_raw, chosen,
            )
            return chosen

    for vid in _PREFERRED_VOICES:
        if vid in available:
            is_compact = "compact" in vid or "eloquence" in vid
            logger.info("TTS: using preferred bundled voice: %s", vid)
            if is_compact:
                logger.warning(
                    "TTS UPGRADE TIP: You are using a compact voice (%s). "
                    "For Jarvis-quality neural TTS, open System Settings → "
                    "Accessibility → Spoken Content → System Voice → Manage Voices "
                    "and download 'Daniel', 'Samantha (Premium)', or 'Zoe (Premium)'. "
                    "ATOM will auto-detect and use it on next launch.",
                    vid.split(".")[-1],
                )
            return vid

    return ""


# ── NSSpeechSynthesizer backend ──────────────────────────────────────

class _NativeSynth:
    """Thread-safe wrapper around NSSpeechSynthesizer.

    Speech runs in a dedicated thread with its own NSRunLoop so we
    don't block asyncio. The asyncio layer awaits via run_in_executor.

    Performance note (Sprint live-fix Apr 2026)
    -------------------------------------------
    Earlier revisions allocated a fresh ``NSSpeechSynthesizer`` and
    re-ran ``setVoice_`` on every utterance. Under sustained memory
    pressure (>80 % on 16 GB M5 Air with the 7B MLX model loaded),
    re-loading the voice file from disk could block ``startSpeakingString_``
    long enough for the 6 s TTS deadman watchdog to fire mid-phrase
    ("I'm listening, Boss." force-stopped after 6 s in production logs).

    The synth is now created **once** and reused across utterances.
    The voice is loaded eagerly in :meth:`prewarm` so the first call
    after boot doesn't pay the cold-load tax. We also detect and report
    a stuck startup (``isSpeaking() == False`` for >1.5 s after
    ``startSpeakingString_``) so the caller can fall back to ``say``.
    """

    def __init__(self, voice_id: str, rate: float, pitch_shift: float = 0.0) -> None:
        self._voice_id = voice_id
        self._rate = rate
        self._pitch_shift = pitch_shift
        # Single reused synthesizer instance. Lazily created on first
        # use (or eagerly via :meth:`prewarm`) so we don't pay the
        # voice-load cost in the executor thread of the first speak
        # call, which is what was blowing the 6 s deadman.
        self._synth: Any = None
        self._stop_flag = threading.Event()
        self._synth_lock = threading.Lock()
        # Counts how often the synth refused to start (``isSpeaking()``
        # never went True). Surfaced so the higher-level TTS can decide
        # to fall back to the ``say`` subprocess on a flapping synth.
        self._stuck_starts: int = 0
        # ── First-word warmup / tail drain ────────────────────────────
        # Bluetooth + USB-C dongle outputs latch the audio device on the
        # first sample after a silent gap. Without a pre-roll silence
        # NSSpeechSynthesizer ships the first ~80ms of audio while the
        # device is still ramping up, which truncates the first word
        # ("Boss" -> "oss"). The tail drain mirrors this on shutdown so
        # the last sample isn't cut by the buffer flush. Both can be
        # tuned via _set_warmup_drain or disabled by setting them to 0.
        self._first_word_warmup_s: float = 0.140
        self._tail_drain_s: float = 0.120
        self._tail_drain_bluetooth_s: float = 0.200
        # Anchor so consecutive sentences in the same speech stream
        # don't pay the warmup tax. monotonic timestamp of the last
        # observed speak completion (0.0 means never spoken).
        self._last_speak_finished_at: float = 0.0
        self._warmup_skip_window_s: float = 0.800
        # Set by audio_intelligence when output device is bluetooth
        # (longer post-buffer flush). Defaults to False -> 120ms.
        self._output_is_bluetooth: bool = False

    def set_warmup_drain(
        self,
        *,
        first_word_warmup_s: float | None = None,
        tail_drain_s: float | None = None,
        tail_drain_bluetooth_s: float | None = None,
        warmup_skip_window_s: float | None = None,
    ) -> None:
        """Tune pre-roll and tail-drain durations from config."""
        if first_word_warmup_s is not None:
            self._first_word_warmup_s = max(0.0, float(first_word_warmup_s))
        if tail_drain_s is not None:
            self._tail_drain_s = max(0.0, float(tail_drain_s))
        if tail_drain_bluetooth_s is not None:
            self._tail_drain_bluetooth_s = max(0.0, float(tail_drain_bluetooth_s))
        if warmup_skip_window_s is not None:
            self._warmup_skip_window_s = max(0.0, float(warmup_skip_window_s))

    def set_output_is_bluetooth(self, is_bt: bool) -> None:
        """Update tail drain length for Bluetooth outputs.

        Called by ``voice.audio_intelligence`` whenever the active output
        device changes so the longer Bluetooth flush is applied
        automatically without restarting TTS.
        """
        self._output_is_bluetooth = bool(is_bt)

    def _effective_tail_drain_s(self) -> float:
        return self._tail_drain_bluetooth_s if self._output_is_bluetooth \
            else self._tail_drain_s

    def _needs_warmup(self) -> bool:
        if self._first_word_warmup_s <= 0:
            return False
        last = self._last_speak_finished_at
        if last <= 0:
            return True
        return (time.monotonic() - last) >= self._warmup_skip_window_s

    def _ensure_synth(self) -> Any:
        """Allocate and configure the persistent NSSpeechSynthesizer.

        Idempotent. Holds ``_synth_lock`` so concurrent prewarm + first
        speak don't race on initialization.
        """
        with self._synth_lock:
            if self._synth is not None:
                return self._synth
            synth = _AppKit.NSSpeechSynthesizer.alloc().init()
            if self._voice_id:
                try:
                    synth.setVoice_(self._voice_id)
                except Exception:
                    logger.debug("setVoice_ failed; continuing with default", exc_info=True)
            try:
                synth.setRate_(self._rate)
            except Exception:
                logger.debug("setRate_ failed", exc_info=True)
            if self._pitch_shift != 0.0:
                try:
                    pitch_prop = getattr(_AppKit, "NSSpeechPitchBaseProperty", "pbas")
                    result = synth.objectForProperty_error_(pitch_prop, None)
                    base_pitch = result[0] if isinstance(result, tuple) else result
                    if base_pitch is not None:
                        new_pitch = float(base_pitch) + self._pitch_shift
                        synth.setObject_forProperty_error_(new_pitch, pitch_prop, None)
                except Exception:
                    logger.debug('macOS speech synth pitch step failed', exc_info=True)
            self._synth = synth
            return synth

    def prewarm(self) -> None:
        """Eagerly load the voice so the first speak call is fast.

        Safe to call from any thread (e.g. main thread during TTS init).
        Silently swallows any AVFoundation/AppKit hiccups — the synth
        will be lazily allocated on first speak if prewarm fails.
        """
        try:
            self._ensure_synth()
            logger.debug("NSSpeechSynthesizer prewarmed (voice=%s)", self._voice_id)
        except Exception:
            logger.debug("NSSpeechSynthesizer prewarm failed", exc_info=True)

    def speak_blocking(self, text: str) -> None:
        """Speak text synchronously (called from executor thread).

        Blocks the calling thread until the utterance ends or
        :meth:`stop` is invoked. Pumps an NSRunLoop on this thread so
        ``NSSpeechSynthesizer`` callbacks are processed.
        """
        self._stop_flag.clear()
        synth = self._ensure_synth()

        # Update the rate live each call. Cheap (no I/O) and lets the
        # adaptive layer change pacing without needing to recreate the
        # synth.
        try:
            synth.setRate_(self._rate)
        except Exception:
            logger.debug("setRate_ failed in speak_blocking", exc_info=True)

        # First-word warmup: hold the audio device active for ~140ms
        # before the first sample so Bluetooth / USB-C dongles don't
        # latch onto the first phoneme. Skipped when the previous
        # utterance ended within _warmup_skip_window_s -- continuous
        # speech keeps the device hot already.
        if self._needs_warmup():
            try:
                time.sleep(self._first_word_warmup_s)
            except Exception:
                logger.debug("first-word warmup sleep raised", exc_info=True)

        try:
            synth.startSpeakingString_(text)
        except Exception:
            logger.warning("startSpeakingString_ raised; aborting speak", exc_info=True)
            self._last_speak_finished_at = time.monotonic()
            return

        rl = _Foundation.NSRunLoop.currentRunLoop()
        # Detect a stuck synth: if isSpeaking() doesn't flip to True
        # within ~1.5 s of startSpeakingString_, the voice file failed
        # to load (most often under memory pressure) or the audio
        # session is wedged. Bail so the caller can fall back to `say`.
        startup_grace_s = 1.5
        startup_deadline = time.monotonic() + startup_grace_s

        # Progress watchdog for the *already-speaking* case. If
        # ``isSpeaking()`` stays True much longer than the text could
        # realistically need, the audio output path is wedged (seen when
        # CoreAudio rescan fires during TTS on macOS 15+). We bail
        # around 3× the estimated finish time so the outer deadman
        # doesn't need the full 12s floor to unblock the caller. The
        # rate is in words/minute per NSSpeechSynthesizer convention
        # (~180 wpm default), so words/(rate/60) seconds per word.
        word_count = max(1, len(text.split()))
        effective_rate = max(60.0, float(self._rate or 180.0))
        est_finish_s = word_count / (effective_rate / 60.0)
        progress_deadline_slack_s = 1.5
        progress_budget_s = max(3.5, est_finish_s * 2.0 + progress_deadline_slack_s)
        # Sprint Ω.8 (Apr 26 2026) R9: tighten the wedge cap from 6s
        # → 4s for short utterances. atomCurrentLogs.txt L400 showed
        # the synth pinned at "isSpeaking=True for 6.0s" on a 10-word
        # 2-sentence reply that should have finished in ~3s. Waiting
        # the full 6s blocked Boss from getting a fallback ``say``
        # voice — by the time we bailed and forked ``say``, ATOM had
        # been silent for more than 9s. 4s is still well above the
        # estimated finish time for any utterance ≤ 12 words at
        # default 193 wpm, so we won't false-trigger on healthy synth.
        if word_count <= 12:
            progress_budget_s = min(progress_budget_s, 4.0)
        elif word_count <= 24:
            progress_budget_s = min(progress_budget_s, 6.5)
        speaking_since: float = 0.0

        ever_speaking = False
        while not self._stop_flag.is_set():
            try:
                speaking = bool(synth.isSpeaking())
            except Exception:
                logger.debug("isSpeaking() raised", exc_info=True)
                speaking = False
            now = time.monotonic()
            if speaking:
                if not ever_speaking:
                    speaking_since = now
                ever_speaking = True
                if (now - speaking_since) > progress_budget_s:
                    self._stuck_starts += 1
                    logger.warning(
                        "NSSpeechSynthesizer wedged mid-utterance "
                        "(isSpeaking=True for %.1fs, expected <%.1fs, %d words, rate=%.0f);"
                        " aborting. stuck_starts=%d",
                        now - speaking_since, progress_budget_s,
                        word_count, effective_rate, self._stuck_starts,
                    )
                    try:
                        synth.stopSpeaking()
                    except Exception:
                        logger.debug(
                            "stopSpeaking on wedged synth failed", exc_info=True,
                        )
                    self._last_speak_finished_at = time.monotonic()
                    return
            elif ever_speaking:
                # Tail drain: hold the audio device open after
                # isSpeaking() flips to False so the last sample fully
                # flushes through CoreAudio's render buffer. Without
                # this the final word/syllable gets clipped on
                # Bluetooth headsets (extra ~80ms hardware latency).
                tail = self._effective_tail_drain_s()
                if tail > 0:
                    try:
                        time.sleep(tail)
                    except Exception:
                        logger.debug("tail drain sleep raised", exc_info=True)
                self._last_speak_finished_at = time.monotonic()
                return
            elif now >= startup_deadline:
                self._stuck_starts += 1
                logger.warning(
                    "NSSpeechSynthesizer never started (isSpeaking=False after %.1fs);"
                    " aborting blocking wait. stuck_starts=%d",
                    startup_grace_s, self._stuck_starts,
                )
                try:
                    synth.stopSpeaking()
                except Exception:
                    logger.debug("stopSpeaking on stuck synth failed", exc_info=True)
                self._last_speak_finished_at = time.monotonic()
                return
            try:
                rl.runMode_beforeDate_(
                    _Foundation.NSDefaultRunLoopMode,
                    _Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.03),
                )
            except Exception:
                logger.debug("NSRunLoop runMode_ raised", exc_info=True)
                return

    @property
    def stuck_starts(self) -> int:
        """How often :meth:`speak_blocking` saw the synth refuse to start."""
        return self._stuck_starts

    def stop(self) -> None:
        """Immediately stop speech from any thread."""
        self._stop_flag.set()
        synth = self._synth
        if synth is not None:
            try:
                synth.stopSpeaking()
            except Exception:
                logger.debug('macOS speech synth step failed', exc_info=True)


# ── Main TTS class ───────────────────────────────────────────────────

class MacOSTTSAsync:
    """Native macOS TTS with premium voice support.

    Public API matches EdgeTTSAsync / TTSAsync for drop-in replacement.
    """

    # Sprint Ω.5.C (Apr 26 2026): the cap WAS 80 words to defend against
    # runaway LLMs. With ``brain.max_tokens=320`` (I-06) the LLM cannot
    # produce more than ~250 words anyway, so an 80-word cap silently
    # truncated 1/3 of a substantive reply to ``screen_buffer`` and never
    # spoke it. Raised to 250 so ATOM speaks the FULL response while
    # still honouring the deadman / watchdog budgets above.
    _SPEAK_WORD_LIMIT: int = 250
    # Coalesce tiny stream fragments so NSSpeechSynthesizer does not speak word-by-word.
    # Lowered from 8/14 for faster first-audio on the streaming LLM path.
    _STREAM_UNPUNCT_MIN_WORDS: int = 5
    _STREAM_UNPUNCT_BATCH: int = 10

    @staticmethod
    def _should_skip_chunking(text: str) -> bool:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        if not clean:
            return False
        return len(clean) < 60 or len(clean.split()) < 12

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        max_lines: int = 4,
        voice: str = "system",
        rate: int = 165,
        *,
        first_word_warmup_ms: int = 140,
        tail_drain_ms: int = 120,
        tail_drain_bluetooth_ms: int = 200,
        warmup_skip_window_ms: int = 800,
    ) -> None:
        self._bus = bus
        self._state = state
        self._max_lines = max_lines
        self._voice_request = voice
        self._rate = rate
        self._voice_id: str = ""
        self._backend: str = "none"
        self._first_word_warmup_s = max(0.0, first_word_warmup_ms / 1000.0)
        self._tail_drain_s = max(0.0, tail_drain_ms / 1000.0)
        self._tail_drain_bluetooth_s = max(0.0, tail_drain_bluetooth_ms / 1000.0)
        self._warmup_skip_window_s = max(0.0, warmup_skip_window_ms / 1000.0)

        self._native_synth: _NativeSynth | None = None
        self._say_proc: asyncio.subprocess.Process | None = None
        self._playing = False
        self._cancel_requested = False
        self._speak_lock = asyncio.Lock()
        self._ack_idx = 0
        self._available = sys.platform == "darwin"

        # ── Deadman timer (Sprint C2) ──────────────────────────────
        # Every speak start records a budget derived from text length;
        # a background task kills the current utterance if the budget
        # is exceeded so the SPEAKING state can never pin ATOM forever
        # (stuck NSSpeechSynthesizer / wedged audio queue / runaway
        # streaming stall). When force-stop triggers, a bus event
        # ``tts_deadman_fired`` is emitted so the state machine + UI
        # can react (e.g. transition back to LISTENING).
        self._speak_start_t: float = 0.0
        self._speak_budget_s: float = 0.0
        self._speak_text_preview: str = ""
        self._deadman_task: asyncio.Task | None = None
        self._deadman_shutdown: asyncio.Event | None = None
        self._deadman_fired_count: int = 0
        # Absolute upper bound for a single utterance regardless of
        # text length. 120s is generous — even a 300-word reply at the
        # slowest configurable rate finishes inside this window.
        self._speak_max_s: float = 120.0

        self._active_source: str | None = None
        self._active_stream_id: str | None = None
        self._chunk_buffer: list[str] = []
        self._screen_buffer: list[str] = []
        self._spoken_word_count: int = 0
        self._recent_spoken_chunks: list[str] = []
        self._stream_queue: asyncio.Queue[tuple[str, bool]] | None = None
        self._stream_task: asyncio.Task | None = None
        self._stream_generation: int = 0
        self._stream_speak_buffer: str = ""
        self._stream_start_t: float = 0.0
        # Holds the first ~60 chars of every fresh stream so a leading
        # stage-direction parenthetical leaked by the LLM never reaches
        # the speaker. Cleared on every is_first chunk.
        self._stream_leak_buffer: StreamingLeakBuffer = StreamingLeakBuffer()
        self._tts_interrupt_count: int = 0
        # Echo guard ring: lowercased, alpha-only word lists from the most
        # recent spoken slices. Lets ``is_echo()`` answer whether an STT
        # partial is just the mic catching ATOM's own speakers — so we
        # don't trigger a barge-in on our own voice.
        self._spoken_echo_window: collections.deque[set[str]] = collections.deque(maxlen=6)
        self._last_spoke_t: float = 0.0
        # True when the last spoken slice was a yes/no confirmation prompt
        # ("Confirm?", "Proceed?", "Okay?"). The echo guard stops flagging
        # reply tokens like "yes", "no", "confirm yes" as self-echo while
        # this is set so a legitimate user reply isn't dropped.
        self._last_spoken_was_confirmation: bool = False

        from voice.speech_enhancer import SpeechEnhancer
        self._enhancer = SpeechEnhancer(base_rate=rate)
        self._current_emotion: str = "neutral"

    # ── Initialization ─────────────────────────────────────────────

    async def init_voice(self) -> None:
        """Select the best voice and backend."""
        if not self._available:
            logger.error("macOS TTS: not on macOS (platform=%s)", sys.platform)
            return

        if _HAS_NATIVE:
            self._voice_id = _pick_best_voice(self._voice_request)
            is_premium = "premium" in self._voice_id or "enhanced" in self._voice_id
            is_eloquence = "eloquence" in self._voice_id
            pitch_shift = _preferred_pitch_shift(self._voice_id)
            self._native_synth = _NativeSynth(self._voice_id, float(self._rate), pitch_shift)
            self._native_synth.set_warmup_drain(
                first_word_warmup_s=self._first_word_warmup_s,
                tail_drain_s=self._tail_drain_s,
                tail_drain_bluetooth_s=self._tail_drain_bluetooth_s,
                warmup_skip_window_s=self._warmup_skip_window_s,
            )
            self._backend = "NSSpeechSynthesizer"

            # Prewarm: load the voice file NOW (during boot, on the
            # main thread, before audio is even needed) so the first
            # speak call doesn't pay the cold-load cost in its
            # executor thread — the source of the 6 s deadman fires
            # observed in the live boot logs.
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._native_synth.prewarm)
            except RuntimeError:
                self._native_synth.prewarm()

            voice_name = self._voice_id.rsplit(".", 1)[-1] if self._voice_id else "default"
            if is_premium:
                quality = "premium neural"
            elif is_eloquence:
                quality = "eloquence"
            else:
                quality = "compact"
            logger.info(
                "macOS TTS ready — %s (%s voice '%s', rate=%d)",
                self._backend, quality, voice_name, self._rate,
            )
        else:
            self._backend = "say"
            logger.info(
                "macOS TTS ready — say command (voice=%s, rate=%d). "
                "Install pyobjc-framework-Cocoa for premium neural voices.",
                self._voice_request, self._rate,
            )

    # ── Core speech dispatch ───────────────────────────────────────

    async def _speak_one(self, text: str) -> None:
        """Speak a single utterance via the active backend.

        If the persistent ``NSSpeechSynthesizer`` reports a stuck start
        (voice file failed to load / audio session wedged), we fall
        back to the ``say`` subprocess on the very next utterance so
        the user keeps hearing ATOM even if the in-process synth dies.
        """
        if self._cancel_requested or not text:
            return

        # Echo-guard bookkeeping must happen BEFORE the audio plays so a
        # racing STT partial caught by the mic still finds the words in
        # the recent-spoken window.
        self._record_spoken(text)

        if self._backend == "NSSpeechSynthesizer" and self._native_synth:
            stuck_before = self._native_synth.stuck_starts
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self._native_synth.speak_blocking, text,
            )
            stuck_after = self._native_synth.stuck_starts
            if stuck_after > stuck_before:
                # The native synth refused to start — most likely the
                # audio session is held by another process or memory
                # pressure paged out the voice. Re-speak via ``say`` so
                # the user actually hears this utterance instead of
                # silently losing it. Logged at WARNING so the live
                # boot logs make it obvious when this fallback fires.
                logger.warning(
                    "TTS: native synth stuck (#%d) — falling back to `say` for: '%s'",
                    stuck_after, text[:60],
                )
                await self._say_subprocess(text)
        else:
            await self._say_subprocess(text)

    async def _say_subprocess(self, text: str) -> None:
        """Fallback: spawn `say` subprocess with dynamic rate."""
        if self._cancel_requested or not text:
            return
        emo = self._current_emotion
        enhanced = self._enhancer.enhance(text, emotion=emo)
        rate = enhanced.rate
        speak_text = enhanced.say_silence_text if enhanced.pause_points else text
        cmd = ["say"]
        if self._voice_request:
            cmd.extend(["-v", self._voice_request])
        cmd.extend(["-r", str(rate), "--", speak_text])
        try:
            self._say_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(self._say_proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("say process timed out after 10s, killing")
                self._say_proc.kill()
                await self._say_proc.wait()
        except asyncio.CancelledError:
            await self._kill_procs()
            raise
        except Exception as exc:
            logger.warning("say error: %s", exc)
        finally:
            self._say_proc = None

    async def _kill_procs(self) -> None:
        """Terminate all speech immediately."""
        if self._native_synth:
            self._native_synth.stop()
        proc = self._say_proc
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    logger.debug('voice tts macos optional step failed', exc_info=True)

    def _normalize_stream_text(self, text: str) -> str:
        cleaned = _clean_for_tts(text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        label_hits = len(_RE_TRANSCRIPT_LABEL.findall(cleaned))
        if label_hits >= 2:
            # Sprint Ω.5.C (Apr 26 2026): two or more "User:" / "Assistant:"
            # labels in one chunk is almost always a transcript-style
            # CoT leak; speaking it would parrot the prompt rules. Log
            # so we can audit silent drops if the pattern triggers on
            # legit content.
            logger.debug(
                "TTS stream chunk dropped (transcript-label leak, %d hits): '%s'",
                label_hits, cleaned[:80],
            )
            return ""
        cleaned = _RE_TRANSCRIPT_LABEL.sub("", cleaned).strip(" -:>")
        if not cleaned or _RE_TRANSCRIPT_ONLY.match(cleaned):
            return ""
        words = cleaned.lower().replace(":", " ").split()
        if len(words) >= 3 and len(set(words)) == 1 and words[0] in {"atom", "user", "assistant", "boss"}:
            logger.debug(
                "TTS stream chunk dropped (single-token repeat): '%s'",
                cleaned[:80],
            )
            return ""
        return cleaned

    def _split_stream_chunk(self, text: str) -> tuple[str, str]:
        text = self._normalize_stream_text(text)
        if not text:
            return "", ""

        words = text.split()
        remaining = max(0, self._SPEAK_WORD_LIMIT - self._spoken_word_count)
        if remaining <= 0:
            return "", text
        if len(words) <= remaining:
            return text, ""

        speak_text = " ".join(words[:remaining]).strip()
        overflow_text = " ".join(words[remaining:]).strip()

        last_period = speak_text.rfind(".")
        last_question = speak_text.rfind("?")
        last_exclaim = speak_text.rfind("!")
        cut_pos = max(last_period, last_question, last_exclaim)
        if cut_pos > len(speak_text) // 3:
            tail = speak_text[cut_pos + 1:].strip()
            speak_text = speak_text[:cut_pos + 1].strip()
            overflow_text = " ".join(part for part in (tail, overflow_text) if part).strip()

        if not speak_text:
            return "", text
        return speak_text, overflow_text

    @staticmethod
    def _chunk_key(text: str) -> str:
        lowered = re.sub(r"[^a-z0-9\s]", "", (text or "").lower())
        return re.sub(r"\s+", " ", lowered).strip()

    # ── Self-voice / echo guard ────────────────────────────────────
    def _record_spoken(self, text: str) -> None:
        """Remember the bag-of-words ATOM just spoke so ``is_echo`` can
        match noisy partials caught by the mic from our own speakers.
        """
        key = self._chunk_key(text)
        if not key:
            return
        words = {w for w in key.split() if len(w) >= 3}
        # Track whether the most recent TTS chunk was a confirmation
        # prompt. Used by is_echo() to stop flagging "yes"/"no"/"confirm"
        # as self-echo when Boss is actually replying to our question.
        try:
            head = (text or "").strip()
            raw_lower = head.lower()
            word_count = len([w for w in head.split() if w])
            is_question = head.endswith("?") or "?" in head
            has_confirm_cue = any(
                cue in raw_lower
                for cue in (
                    "confirm?", "confirm ", "proceed?", "proceed ",
                    "shall i", "should i", "go ahead?", "okay?", "ok?",
                    "ready?", "sure?", "continue?", "yes or no",
                )
            )
            short_prompt = word_count <= 5 and is_question
            self._last_spoken_was_confirmation = bool(short_prompt or has_confirm_cue)
        except Exception:
            self._last_spoken_was_confirmation = False
        if not words:
            return
        self._spoken_echo_window.append(words)
        self._last_spoke_t = time.monotonic()

    def is_echo(self, partial_text: str, *, window_s: float = 6.0) -> bool:
        """Best-effort echo guard for STT partials.

        Returns ``True`` when the partial looks like the mic catching our
        own voice during a SPEAKING state — specifically, when every
        meaningful word in the partial appears in the last few spoken
        slices and we spoke recently. The interrupt handler uses this to
        suppress self-feedback barge-ins.

        Sprint Ω.5.B (Apr 26 2026): tightened the temporal gate. The
        previous router-side guard ran with ``window_s=12.0`` which
        meant a fresh user turn 10 s after ATOM finished could be
        false-flagged as echo if it shared 80 % bag-of-words. The
        floor is now bounded to ``min(window_s, 4.0)`` once the state
        machine has left SPEAKING -- after that, it's almost certainly
        the user, not the speaker tail.
        """
        if not partial_text:
            return False
        if not self._spoken_echo_window:
            return False
        elapsed_since_speech = time.monotonic() - self._last_spoke_t
        try:
            from core.state_manager import AtomState
            currently_speaking = (
                self._state.current is AtomState.SPEAKING
            )
        except Exception:
            currently_speaking = False
        effective_window = float(window_s)
        if not currently_speaking:
            # WhisperKit can return delayed finals several seconds after ATOM
            # finishes speaking. Keep the caller's window so finalization can
            # still reject our own previous sentence instead of treating it as
            # Boss input.
            effective_window = max(4.0, effective_window)
        if elapsed_since_speech > max(0.5, effective_window):
            return False
        key = self._chunk_key(partial_text)
        if not key:
            return False
        # Confirmation-reply exception: if ATOM's most recent speech was a
        # short yes/no prompt (e.g. "Confirm?", "Proceed?"), Boss is now
        # answering us — DO NOT flag his reply as an echo. We check this
        # FIRST so even partials that fuzzy-match pass through.
        if self._last_spoken_was_confirmation:
            reply_norm = key.strip()
            reply_tokens = [w for w in reply_norm.split() if w]
            # Short reply (<= 4 words) containing a yes/no/confirm marker
            # is trusted as a user answer, not echo.
            if 0 < len(reply_tokens) <= 4:
                reply_corpus = set(reply_tokens)
                affirmations = {
                    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
                    "confirm", "confirmed", "proceed", "go", "do",
                    "no", "nope", "nah", "cancel", "stop", "abort",
                    "hold",
                }
                if reply_corpus & affirmations:
                    return False
        partial_words = [w for w in key.split() if len(w) >= 3]
        if not partial_words:
            # Single short token (e.g. "the") - treat as echo when speaking
            # because such tokens do not carry user intent and we are
            # confidently talking right now.
            return True
        recent_corpus: set[str] = set()
        for slice_words in self._spoken_echo_window:
            recent_corpus.update(slice_words)
        if not recent_corpus:
            return False
        hits = sum(1 for w in partial_words if w in recent_corpus)
        # Require near-total overlap (>= 80%) before declaring echo so a
        # genuine interrupt that *also* happens to share a stop word
        # ("ok", "and") still gets through.
        return hits >= max(1, int(0.8 * len(partial_words)))

    def _is_duplicate_chunk(self, text: str) -> bool:
        """Skip only immediate repeats of the same spoken chunk (streaming overlap).

        Uses a sliding window of the last 2 full normalized chunks.
        Short chunks (< 4 words) are never considered duplicates since
        common phrases like "Sure, Boss" can legitimately repeat.
        """
        key = self._chunk_key(text)
        if not key:
            return True
        if len(key.split()) < 4:
            self._recent_spoken_chunks.append(key)
            if len(self._recent_spoken_chunks) > 3:
                self._recent_spoken_chunks = self._recent_spoken_chunks[-3:]
            return False
        if self._recent_spoken_chunks and key == self._recent_spoken_chunks[-1]:
            return True
        self._recent_spoken_chunks.append(key)
        if len(self._recent_spoken_chunks) > 3:
            self._recent_spoken_chunks = self._recent_spoken_chunks[-3:]
        return False

    # Hold off the eager first-audio flush this long after the very first
    # streamed chunk arrived so a tight burst of single-token chunks
    # coalesces into one utterance instead of being chopped at 3 words.
    # 60ms is well below human-perceptible "first sound" latency yet long
    # enough to absorb token-at-a-time bursts from a fast LLM stream.
    _STREAM_FIRST_FLUSH_DEBOUNCE_S: float = 0.06

    def _pop_next_stream_segment(self, force: bool, more_pending: bool = False) -> str:
        """Take the next speakable slice from _stream_speak_buffer.

        Complete sentences (ending in . ! ?) flush immediately so latency stays
        low. Unpunctuated fragments wait until we have enough words to avoid
        NSSpeechSynthesizer speaking one token at a time.

        ``more_pending`` is True when the producer queue still has unread
        items. We also apply a short time-based debounce on the very first
        flush so a token-at-a-time burst (e.g. an 8-word reply arriving as
        8 separate chunks) lands as one utterance.
        """
        buf = self._stream_speak_buffer.strip()
        if not buf:
            return ""
        if force:
            self._stream_speak_buffer = ""
            return buf

        m = _RE_FIRST_SENTENCE.match(buf)
        if m:
            sentence = m.group(1).strip()
            self._stream_speak_buffer = buf[m.end() :].strip()
            return sentence

        words = buf.split()
        n = len(words)

        # End-of-speech alignment: when the buffer ends with sentence
        # punctuation, flush immediately so speech aligns with natural
        # pauses instead of arbitrary word-count boundaries.
        if n >= 2 and buf.rstrip()[-1:] in ".!?":
            self._stream_speak_buffer = ""
            return buf

        # First-flush debounce: while the very first chunk is still fresh
        # and nothing has been spoken yet, hold ANY unpunctuated flush so a
        # token-burst coalesces. The is_last force=True path bypasses this.
        in_first_flush_window = (
            self._spoken_word_count == 0
            and self._stream_start_t > 0.0
            and (time.perf_counter() - self._stream_start_t)
            < self._STREAM_FIRST_FLUSH_DEBOUNCE_S
        )

        # Fast first-audio: flush with as few as 3 words when nothing has
        # been spoken yet so the user hears something immediately — but
        # only when the producer is briefly quiet AND we're past the
        # initial debounce window.
        if (
            self._spoken_word_count == 0
            and n >= 3
            and not more_pending
            and not in_first_flush_window
        ):
            self._stream_speak_buffer = ""
            return buf

        if (
            n >= self._STREAM_UNPUNCT_BATCH
            and not more_pending
            and not in_first_flush_window
        ):
            seg = " ".join(words[: self._STREAM_UNPUNCT_BATCH])
            self._stream_speak_buffer = " ".join(words[self._STREAM_UNPUNCT_BATCH :])
            return seg
        if (
            n >= self._STREAM_UNPUNCT_MIN_WORDS
            and not more_pending
            and not in_first_flush_window
        ):
            take = min(n, self._STREAM_UNPUNCT_BATCH)
            seg = " ".join(words[:take])
            self._stream_speak_buffer = " ".join(words[take:])
            return seg
        return ""

    async def _speak_stream_slice(self, raw_segment: str) -> None:
        """Apply word-cap + duplicate filtering, then speak one slice."""
        speak_text, overflow_text = self._split_stream_chunk(raw_segment)
        if speak_text and not self._cancel_requested:
            if _RE_INTERNAL_TTS_LINE.search(speak_text):
                logger.warning(
                    "TTS stream slice suppressed (internal-status guard): '%s'",
                    speak_text[:100],
                )
                return
            if _is_prompt_leak(speak_text):
                logger.warning(
                    "TTS stream slice suppressed (prompt-leak guard): '%s'",
                    speak_text[:100],
                )
                if overflow_text:
                    self._screen_buffer.append(overflow_text)
                return
            if self._is_duplicate_chunk(speak_text):
                logger.info("TTS stream duplicate chunk skipped: '%s'", speak_text[:100])
                if overflow_text:
                    self._screen_buffer.append(overflow_text)
                return
            self._spoken_word_count += len(speak_text.split())
            logger.info(
                "TTS stream slice (%d/%d words): '%s'",
                self._spoken_word_count,
                self._SPEAK_WORD_LIMIT,
                speak_text[:100],
            )
            await self._speak_internal(speak_text)
        if overflow_text:
            self._screen_buffer.append(overflow_text)

    async def _play_stream_chunks(self, generation: int) -> None:
        queue = self._stream_queue
        if queue is None:
            return

        try:
            while True:
                text, is_last = await queue.get()
                if generation != self._stream_generation:
                    return

                if text:
                    self._chunk_buffer.append(text)
                    merged = (
                        f"{self._stream_speak_buffer} {text}".strip()
                        if self._stream_speak_buffer
                        else text.strip()
                    )
                    self._stream_speak_buffer = re.sub(r"\s+", " ", merged).strip()

                # When more chunks are already buffered, hold flushes so
                # tiny per-token streams coalesce into one utterance and
                # NSSpeechSynthesizer doesn't speak one word at a time.
                more_pending = queue.qsize() > 0 and not is_last
                while True:
                    seg = self._pop_next_stream_segment(
                        force=False, more_pending=more_pending,
                    )
                    if not seg:
                        break
                    await self._speak_stream_slice(seg)

                if is_last:
                    tail = self._pop_next_stream_segment(force=True)
                    if tail:
                        await self._speak_stream_slice(tail)
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("TTS stream error")
        finally:
            if self._stream_task is asyncio.current_task():
                self._stream_task = None
            if generation != self._stream_generation:
                # Sprint Ω.5.C (Apr 26 2026): the streaming worker bailed
                # because barge-in / stop() bumped the generation. Without
                # a terminal event, downstream listeners (voice pipeline
                # passive-revert timer, interrupt handler, command loop,
                # iPhone bridge SSE proxies) sit waiting forever for a
                # ``tts_complete`` that never arrives. Emit one bounded
                # interrupted-completion event so the state machine
                # reliably returns to LISTENING and the next user turn
                # is never silently swallowed. We deliberately keep the
                # delivery-metrics emission so the latency dashboard
                # accounts for partial-speech duration.
                stream_duration_ms = (
                    (time.perf_counter() - self._stream_start_t) * 1000
                    if self._stream_start_t > 0 else 0.0
                )
                try:
                    self._bus.emit(
                        "tts_delivery_metrics",
                        words_spoken=self._spoken_word_count,
                        duration_ms=round(stream_duration_ms, 1),
                        backend=self._backend,
                        interrupt_count=self._tts_interrupt_count,
                        interrupted=True,
                    )
                except Exception:
                    logger.debug(
                        "tts_delivery_metrics (interrupt) emit failed",
                        exc_info=True,
                    )
                try:
                    self._bus.emit("tts_complete", interrupted=True)
                except Exception:
                    logger.debug(
                        "tts_complete (interrupt) emit failed",
                        exc_info=True,
                    )
                return

            self._stop_deadman()

            overflow_text = " ".join(self._screen_buffer).strip()
            self._chunk_buffer.clear()
            self._screen_buffer.clear()
            self._stream_speak_buffer = ""
            self._stream_queue = None
            self._active_source = None
            self._active_stream_id = None

            if overflow_text:
                # Sprint Ω.5.C (Apr 26 2026): with the spoken cap raised
                # to 250 words this branch should be cold for any
                # well-formed turn. If we still hit it (e.g. report mode
                # generated a 600-word block), surface the rest on screen
                # AND emit a short audible cue so Boss isn't left
                # wondering whether the answer ended -- silent screen
                # writes feel like the response was lost.
                logger.info(
                    "Screen-only overflow (%d words after spoken cap %d): '%s'",
                    len(overflow_text.split()),
                    self._SPEAK_WORD_LIMIT,
                    overflow_text[:100],
                )
                self._bus.emit("text_display", text=overflow_text)
                try:
                    self._bus.emit(
                        "tts_overflow_screen",
                        words=len(overflow_text.split()),
                        preview=overflow_text[:120],
                    )
                except Exception:
                    logger.debug(
                        "tts_overflow_screen emit failed", exc_info=True,
                    )

            stream_duration_ms = (
                (time.perf_counter() - self._stream_start_t) * 1000
                if self._stream_start_t > 0 else 0.0
            )
            logger.info(
                "TTS stream done: %d words spoken in %.0fms",
                self._spoken_word_count,
                stream_duration_ms,
            )
            self._bus.emit(
                "tts_delivery_metrics",
                words_spoken=self._spoken_word_count,
                duration_ms=round(stream_duration_ms, 1),
                backend=self._backend,
                interrupt_count=self._tts_interrupt_count,
            )
            self._tts_interrupt_count = 0
            self._bus.emit("tts_complete")

    # ── Internal speak (no tts_complete emission) ──────────────────

    async def _speak_internal(self, text: str,
                              emotion: str | None = None) -> None:
        text = _truncate(text, self._max_lines)
        if not text or not self._available:
            return
        if self._cancel_requested:
            return

        emo = emotion or self._current_emotion
        enhanced = self._enhancer.enhance(text, emotion=emo)
        dynamic_rate = enhanced.rate

        async with self._speak_lock:
            if self._cancel_requested:
                return
            self._cancel_requested = False
            self._playing = True
            self._start_deadman(text)

            if self._native_synth:
                self._native_synth._rate = float(dynamic_rate)

            try:
                sentences = _split_sentences(text)
                if len(sentences) <= 1:
                    logger.info("TTS [%s rate=%d]: '%s'", self._backend, dynamic_rate, text[:80])
                    await self._speak_one(text)
                else:
                    logger.info(
                        "TTS [%s, %d sentences, rate=%d]: '%s'",
                        self._backend, len(sentences), dynamic_rate, text[:80],
                    )
                    for i, sentence in enumerate(sentences):
                        if self._cancel_requested:
                            break
                        await self._speak_one(sentence)
                        if i < len(sentences) - 1 and not self._cancel_requested:
                            pause = self._enhancer.compute_inter_sentence_pause(
                                sentence, emotion=emo,
                            )
                            if pause > 0:
                                await asyncio.sleep(pause)
            except asyncio.CancelledError:
                await self.stop()
                raise
            except Exception:
                logger.exception("TTS speak error")
                self._bus.emit(
                    "text_display", text=f"[Response on screen] {text}",
                )
            finally:
                self._stop_deadman()
                self._playing = False
                if self._native_synth:
                    self._native_synth._rate = float(self._enhancer._base_rate)

    # ── Public API ─────────────────────────────────────────────────

    async def speak(self, text: str, emotion: str | None = None) -> None:
        """Speak text. Emits tts_complete when done."""
        if _is_prompt_leak(text):
            logger.warning(
                "TTS suppressed prompt-leak text (final guard): '%s'",
                (text or "")[:80],
            )
            self._bus.emit("tts_complete")
            return
        await self._speak_internal(text, emotion)
        self._bus.emit("tts_complete")

    async def speak_ack(self, phrase: str) -> None:
        """Speak a short acknowledgement phrase."""
        if not phrase or not self._available:
            return
        if _is_prompt_leak(phrase):
            logger.warning(
                "TTS suppressed prompt-leak ack (final guard): '%s'",
                phrase[:80],
            )
            return
        # Pre-register before the speak lock so racing STT partials see the
        # acknowledgement as ATOM audio, even if another utterance is winding down.
        self._record_spoken(phrase)
        logger.info("TTS ack [%s]: '%s'", self._backend, phrase)
        async with self._speak_lock:
            if self._cancel_requested:
                return
            self._playing = True
            self._start_deadman(phrase)
            try:
                self._record_spoken(phrase)
                await self._speak_one(phrase)
            finally:
                self._stop_deadman()
                self._playing = False

    def set_emotion(self, emotion: str) -> None:
        """Update the current emotional context for dynamic rate control."""
        if emotion and emotion != self._current_emotion:
            self._current_emotion = emotion
            logger.debug("TTS emotion updated: %s", emotion)

    def apply_perception_style(
        self,
        rate_multiplier: float = 1.0,
        pause_multiplier: float = 1.0,
    ) -> None:
        """Adapt speech pacing from merged perception + adaptive params.

        Adjusts the SpeechEnhancer base rate and pause multiplier so
        urgency, emotion, and learned preferences flow through the
        existing enhancement pipeline.
        """
        new_rate = int(self._rate * max(0.7, min(1.4, rate_multiplier)))
        self._enhancer._base_rate = new_rate
        self._enhancer._pause_multiplier = max(0.3, min(2.0, pause_multiplier))
        if self._native_synth is not None:
            self._native_synth._rate = float(new_rate)
        logger.debug(
            "TTS perception style: rate_mult=%.2f pause_mult=%.1f -> base_rate=%d",
            rate_multiplier, pause_multiplier, new_rate,
        )

    def next_ack_phrase(self) -> str:
        phrase = ACK_PHRASES[self._ack_idx % len(ACK_PHRASES)]
        self._ack_idx += 1
        return phrase

    # ── Deadman timer helpers (Sprint C2) ──────────────────────────

    @staticmethod
    def _estimate_speak_budget_s(text: str) -> float:
        """Estimate a safe time budget for speaking ``text``.

        Conservative model so the deadman is a *true* safety net, not
        a UX bottleneck. Live-fix Apr 2026: short streaming slices
        like "I'm listening, Boss." were force-stopped at exactly 6 s
        because the per-slice budget plus first-call NSSpeechSynthesizer
        warmup blew through the prior 6 s floor. We scale at ~2.5 wps
        with a 5 s setup grace to absorb the worst-case voice paging
        seen on memory-pressured M5 Air (>80 % unified memory
        utilization with the 7B MLX model loaded).

        Apr 22 2026 — the 12 s floor is fine for streamed phrases
        (word limit per slice is large), but it made the deadman wait
        a full 12 s on two-word acks like ``"I'm here."`` when the
        audio output path was wedged by a CoreAudio rescan. Short
        utterances now get a tighter 6 s floor; phrases of ~10+ words
        still scale up to the original budget.
        """
        if not text:
            return 6.0
        words = max(1, len(text.split()))
        base = words / 2.5
        computed = base * 2.5 + 5.0
        floor = 6.0 if words <= 5 else 12.0
        return max(floor, computed)

    def _start_deadman(self, text: str) -> None:
        """Record the expected finish time and spawn the watchdog task."""
        self._speak_start_t = time.monotonic()
        budget = self._estimate_speak_budget_s(text)
        # Clamp to the absolute max to protect against runaway budgets
        # (e.g. very long RAG answers). The deadman is safety-net, not
        # pacing — ATOM's UX layer already limits answer length.
        self._speak_budget_s = min(budget, self._speak_max_s)
        self._speak_text_preview = (text or "")[:80]
        existing = self._deadman_task
        if existing is not None and not existing.done():
            existing.cancel()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._deadman_shutdown = asyncio.Event()
        self._deadman_task = loop.create_task(self._deadman_loop())

    def _stop_deadman(self) -> None:
        """Cancel any pending deadman check (speak finished cleanly)."""
        evt = self._deadman_shutdown
        if evt is not None:
            evt.set()
        task = self._deadman_task
        if task is not None and not task.done():
            task.cancel()
        self._deadman_task = None
        self._deadman_shutdown = None
        self._speak_start_t = 0.0
        self._speak_budget_s = 0.0
        self._speak_text_preview = ""

    async def _deadman_loop(self) -> None:
        """Force-stop any utterance that outruns its budget."""
        evt = self._deadman_shutdown
        # 1s floor is a safety net against misconfigured budgets — any
        # realistic utterance budget is much larger (see
        # ``_estimate_speak_budget_s``). The floor also stops us from
        # flapping when a zero budget is passed during tests.
        budget = max(1.0, self._speak_budget_s)
        try:
            try:
                if evt is not None:
                    await asyncio.wait_for(evt.wait(), timeout=budget)
                else:
                    await asyncio.sleep(budget)
                return  # clean finish
            except asyncio.TimeoutError:
                pass

            if not self._playing:
                return
            elapsed = time.monotonic() - self._speak_start_t
            self._deadman_fired_count += 1
            logger.error(
                "TTS deadman: utterance exceeded budget (%.1fs > %.1fs) — "
                "force-stopping '%s...' [fire #%d]",
                elapsed, budget, self._speak_text_preview,
                self._deadman_fired_count,
            )
            try:
                await self.force_stop()
            except Exception:
                logger.exception("TTS deadman: force_stop failed")
            try:
                self._bus.emit_fast(
                    "tts_deadman_fired",
                    elapsed_s=round(elapsed, 1),
                    budget_s=round(budget, 1),
                    preview=self._speak_text_preview,
                    total_fires=self._deadman_fired_count,
                )
                # ``tts_complete`` is the standard protocol for "speaking
                # ended"; emit it so the state machine + listener return
                # to LISTENING even when the normal finally-path was
                # blocked.
                self._bus.emit("tts_complete")
            except Exception:
                logger.debug("TTS deadman: emit failed", exc_info=True)
        except asyncio.CancelledError:
            pass

    def tts_deadman_stats(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = (now - self._speak_start_t) if self._speak_start_t > 0 else 0.0
        return {
            "enabled": True,
            "playing": self._playing,
            "fired_count": self._deadman_fired_count,
            "current_elapsed_s": round(elapsed, 1),
            "current_budget_s": round(self._speak_budget_s, 1),
            "preview": self._speak_text_preview,
        }

    async def force_stop(self) -> None:
        """Hard stop: immediate silence, cancel stream, kill processes.

        Called by ``VoiceInterruptHandler`` -- the single public API for
        external interrupt callers so they never touch private fields.

        Order matters: clear state first so no in-flight coroutine can
        re-queue audio, then kill processes, then run the full ``stop()``
        for any remaining cleanup.
        """
        self._cancel_requested = True
        self._playing = False
        self._stream_generation += 1
        self._tts_interrupt_count += 1
        self._stream_speak_buffer = ""
        self._chunk_buffer.clear()
        self._screen_buffer.clear()
        self._stop_deadman()
        queue = self._stream_queue
        if queue is not None:
            try:
                while not queue.empty():
                    queue.get_nowait()
            except Exception:
                logger.debug('Audio stream stop failed', exc_info=True)
        await self._kill_procs()
        await self.stop()

    async def stop(self) -> None:
        """Barge-in: immediately stop all speech."""
        self._cancel_requested = True
        self._playing = False
        self._stream_generation += 1
        self._active_source = None
        self._active_stream_id = None
        self._chunk_buffer.clear()
        self._screen_buffer.clear()
        self._stream_speak_buffer = ""
        self._stop_deadman()
        queue = self._stream_queue
        self._stream_queue = None
        if queue is not None:
            try:
                queue.put_nowait(("", True))
            except Exception:
                logger.debug('Audio stream stop failed', exc_info=True)
        await self._kill_procs()
        await asyncio.sleep(0.02)

    # ── Governor hooks (no-ops) ────────────────────────────────────

    def set_postprocess(self, enabled: bool) -> None:
        pass

    def restore_postprocess(self) -> None:
        pass

    def refresh_output_device(self) -> bool:
        return False

    # ── Event handlers ─────────────────────────────────────────────

    async def on_speech_partial(self, text: str, **_kw) -> None:
        from core.state_manager import AtomState

        t = (text or "").strip()
        if not t:
            return
        has_audio = bool(
            self._playing or self._stream_task is not None or self._stream_queue is not None
        )
        if not has_audio:
            return
        if t in ("Listening...", "Processing..."):
            logger.info(
                "Barge-in: confirmer status indicator '%s' "
                "(playing=%s, stream_active=%s, buffered_chunks=%d) -- stopping TTS",
                t,
                self._playing,
                self._stream_task is not None or self._stream_queue is not None,
                len(self._chunk_buffer),
            )
            await self.stop()
            return
        if self._state.current is AtomState.SPEAKING and len(t) >= 2:
            logger.info(
                "Barge-in: user speech during TTS "
                "(text='%s', playing=%s, buffered_chunks=%d) -- stopping",
                t[:56],
                self._playing,
                len(self._chunk_buffer),
            )
            await self.stop()

    async def on_response(self, text: str, is_exit: bool = False,
                          is_sleep: bool = False, **_kw) -> None:
        from core.state_manager import AtomState

        self._active_source = None
        self._active_stream_id = None
        if self._playing or self._stream_queue is not None or self._stream_task is not None:
            await self.stop()
            self._cancel_requested = False
        if self._state.current is AtomState.SPEAKING:
            return
        self._cancel_requested = False
        await self._state.transition(AtomState.SPEAKING)

        async def _speak_bg() -> None:
            try:
                if is_sleep:
                    await self._speak_internal(text)
                    self._bus.emit("enter_sleep_mode")
                    return
                await self.speak(text)
            except Exception:
                logger.exception("TTS background speak error")
                self._bus.emit(
                    "text_display", text=f"[Response on screen] {text}",
                )
                self._bus.emit("tts_complete")
            if is_exit:
                self._bus.emit("shutdown_requested")

        asyncio.create_task(_speak_bg())

    async def on_partial_response(
        self, text: str, is_first: bool = False,
        is_last: bool = False, source: str = "", stream_id: str = "",
        bypass_chunking: bool = False, **_kw,
    ) -> None:
        from core.state_manager import AtomState

        normalized_text = self._normalize_stream_text(text) if text else ""

        if (
            is_first
            and is_last
            and normalized_text
            and (bypass_chunking or self._should_skip_chunking(normalized_text))
        ):
            leak_guard = StreamingLeakBuffer()
            cleaned_slices = leak_guard.feed(normalized_text) or leak_guard.flush()
            direct_text = " ".join(s for s in cleaned_slices if s).strip()
            if not direct_text:
                direct_text = normalized_text
            logger.info(
                "TTS short reply bypassing stream chunker (%d words): '%s'",
                len(direct_text.split()),
                direct_text[:100],
            )
            if self._playing or self._stream_queue is not None or self._stream_task is not None:
                await self.stop()
                self._cancel_requested = False
            await self._state.transition(AtomState.SPEAKING)
            await self.speak(direct_text)
            return

        if is_first:
            self._active_source = source or "unknown"
            self._chunk_buffer.clear()
            self._screen_buffer.clear()
            self._stream_speak_buffer = ""
            self._spoken_word_count = 0
            self._recent_spoken_chunks.clear()
            # Reset the leading-leak buffer for the new utterance. Until
            # this buffer "releases", we hold every slice off the queue
            # so the LLM cannot speak its own stage direction.
            self._stream_leak_buffer.reset()
            self._active_stream_id = stream_id or None
            self._stream_start_t = time.perf_counter()
            logger.info(
                "TTS stream: source='%s' stream_id=%s'",
                self._active_source,
                stream_id or "none",
            )
            await self.stop()
            self._cancel_requested = False
            await self._state.transition(AtomState.SPEAKING)
            self._active_source = source or "unknown"
            self._active_stream_id = stream_id or None
            self._stream_queue = asyncio.Queue()
            self._stream_task = asyncio.create_task(
                self._play_stream_chunks(self._stream_generation)
            )
            # Streaming replies can be longer than the budget we'd
            # compute from a single chunk, so seed the deadman with the
            # absolute cap. Chunks keep extending the playing window;
            # the deadman only kicks in if the WHOLE stream is stuck.
            self._start_deadman("streaming utterance")
            self._speak_budget_s = self._speak_max_s
        elif source and self._active_source and source != self._active_source:
            return
        elif stream_id and self._active_stream_id and stream_id != self._active_stream_id:
            return

        if not normalized_text and not is_last:
            return
        if self._state.current is not AtomState.SPEAKING and not is_first:
            self._cancel_requested = False
            await self._state.transition(AtomState.SPEAKING)
        queue = self._stream_queue
        if queue is None:
            self._stream_queue = asyncio.Queue()
            queue = self._stream_queue
            self._stream_task = asyncio.create_task(
                self._play_stream_chunks(self._stream_generation)
            )

        # ── Sprint A3: leading stage-direction guard ──────────────
        # Run the slice through the StreamingLeakBuffer first. While
        # the buffer is still accumulating, *no* audio is queued -- it
        # waits for either a sentence-boundary or 60 chars before
        # releasing the cleaned head. After release every subsequent
        # slice passes straight through.
        if not self._stream_leak_buffer.released and normalized_text:
            cleaned_slices = self._stream_leak_buffer.feed(normalized_text)
            if not cleaned_slices:
                if is_last:
                    cleaned_slices = self._stream_leak_buffer.flush()
                    for cleaned in cleaned_slices:
                        if cleaned:
                            queue.put_nowait((cleaned, False))
                    queue.put_nowait(("", True))
                return
            # Replace the raw slice with the sanitised head; the rest
            # of the stream (after release) bypasses the buffer.
            normalized_text = " ".join(s for s in cleaned_slices if s).strip()
            if not normalized_text and not is_last:
                return

        # Backpressure: when TTS is falling behind (queue depth > 5),
        # merge text directly into the speak buffer instead of queueing
        # to prevent unbounded growth and silent latency creep.
        _BACKPRESSURE_DEPTH = 5
        if queue.qsize() > _BACKPRESSURE_DEPTH and normalized_text and not is_last:
            merged = (
                f"{self._stream_speak_buffer} {normalized_text}".strip()
                if self._stream_speak_buffer
                else normalized_text
            )
            self._stream_speak_buffer = re.sub(r"\s+", " ", merged)
            logger.debug(
                "TTS backpressure: merged into buffer (queue depth %d)",
                queue.qsize(),
            )
            return

        queue.put_nowait((normalized_text, is_last))

    # ── Shutdown ───────────────────────────────────────────────────

    async def shutdown(self) -> None:
        await self.stop()
        self._native_synth = None
        logger.info("macOS TTS shut down (%s)", self._backend)
