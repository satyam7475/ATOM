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
import logging
import plistlib
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.tts_macos")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

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

# When no explicit voice matches: premium neural first (Siri-class quality on Apple Silicon).
_PREFERRED_VOICES = [
    "com.apple.voice.premium.en-GB.Martha",
    "com.apple.voice.enhanced.en-GB.Martha",
    "com.apple.eloquence.en-GB.Flo",
    "com.apple.eloquence.en-GB.Shelley",
    "com.apple.eloquence.en-GB.Sandy",
    "com.apple.voice.premium.en-US.Samantha",
    "com.apple.voice.enhanced.en-US.Samantha",
    "com.apple.voice.compact.en-US.Samantha",
    "com.apple.voice.premium.en-GB.Malcolm",
    "com.apple.voice.premium.en-GB.Daniel",
    "com.apple.voice.enhanced.en-GB.Daniel",
    "com.apple.voice.premium.en-US.Evan",
    "com.apple.voice.enhanced.en-US.Evan",
    "com.apple.voice.premium.en-AU.Lee",
    "com.apple.speech.synthesis.voice.daniel.premium",
    "com.apple.eloquence.en-US.Eddy",
]


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
    neural fallback list.
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
        for vid in available:
            if req_lower in vid.lower():
                logger.info("TTS: matched requested voice substring '%s' -> %s", req_raw, vid)
                return vid

    for vid in _PREFERRED_VOICES:
        if vid in available:
            logger.info("TTS: using preferred bundled voice: %s", vid)
            return vid

    return ""


# ── NSSpeechSynthesizer backend ──────────────────────────────────────

class _NativeSynth:
    """Thread-safe wrapper around NSSpeechSynthesizer.

    Speech runs in a dedicated thread with its own NSRunLoop so we
    don't block asyncio. The asyncio layer awaits via run_in_executor.
    """

    def __init__(self, voice_id: str, rate: float) -> None:
        self._voice_id = voice_id
        self._rate = rate
        self._synth: Any = None
        self._stop_flag = threading.Event()

    def speak_blocking(self, text: str) -> None:
        """Speak text synchronously (called from executor thread)."""
        self._stop_flag.clear()
        synth = _AppKit.NSSpeechSynthesizer.alloc().init()
        if self._voice_id:
            synth.setVoice_(self._voice_id)
        synth.setRate_(self._rate)
        self._synth = synth

        synth.startSpeakingString_(text)

        rl = _Foundation.NSRunLoop.currentRunLoop()
        while synth.isSpeaking() and not self._stop_flag.is_set():
            rl.runMode_beforeDate_(
                _Foundation.NSDefaultRunLoopMode,
                _Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.03),
            )
        self._synth = None

    def stop(self) -> None:
        """Immediately stop speech from any thread."""
        self._stop_flag.set()
        synth = self._synth
        if synth is not None:
            try:
                synth.stopSpeaking()
            except Exception:
                pass


# ── Main TTS class ───────────────────────────────────────────────────

class MacOSTTSAsync:
    """Native macOS TTS with premium voice support.

    Public API matches EdgeTTSAsync / TTSAsync for drop-in replacement.
    """

    _SPEAK_WORD_LIMIT: int = 18
    # Coalesce tiny stream fragments so NSSpeechSynthesizer does not speak word-by-word.
    _STREAM_UNPUNCT_MIN_WORDS: int = 8
    _STREAM_UNPUNCT_BATCH: int = 14

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        max_lines: int = 4,
        voice: str = "system",
        rate: int = 200,
    ) -> None:
        self._bus = bus
        self._state = state
        self._max_lines = max_lines
        self._voice_request = voice
        self._rate = rate
        self._voice_id: str = ""
        self._backend: str = "none"

        self._native_synth: _NativeSynth | None = None
        self._say_proc: asyncio.subprocess.Process | None = None
        self._playing = False
        self._cancel_requested = False
        self._speak_lock = asyncio.Lock()
        self._ack_idx = 0
        self._available = sys.platform == "darwin"

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

    # ── Initialization ─────────────────────────────────────────────

    async def init_voice(self) -> None:
        """Select the best voice and backend."""
        if not self._available:
            logger.error("macOS TTS: not on macOS (platform=%s)", sys.platform)
            return

        if _HAS_NATIVE:
            self._voice_id = _pick_best_voice(self._voice_request)
            self._native_synth = _NativeSynth(self._voice_id, float(self._rate))
            self._backend = "NSSpeechSynthesizer"

            voice_name = self._voice_id.rsplit(".", 1)[-1] if self._voice_id else "default"
            is_premium = "premium" in self._voice_id or "enhanced" in self._voice_id
            quality = "premium neural" if is_premium else "standard"
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
        """Speak a single utterance via the active backend."""
        if self._cancel_requested or not text:
            return

        if self._backend == "NSSpeechSynthesizer" and self._native_synth:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self._native_synth.speak_blocking, text,
            )
        else:
            await self._say_subprocess(text)

    async def _say_subprocess(self, text: str) -> None:
        """Fallback: spawn `say` subprocess."""
        if self._cancel_requested or not text:
            return
        cmd = ["say"]
        if self._voice_request:
            cmd.extend(["-v", self._voice_request])
        cmd.extend(["-r", str(self._rate), "--", text])
        try:
            self._say_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
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
                    pass

    def _normalize_stream_text(self, text: str) -> str:
        cleaned = _clean_for_tts(text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        label_hits = len(_RE_TRANSCRIPT_LABEL.findall(cleaned))
        if label_hits >= 2:
            return ""
        cleaned = _RE_TRANSCRIPT_LABEL.sub("", cleaned).strip(" -:>")
        if not cleaned or _RE_TRANSCRIPT_ONLY.match(cleaned):
            return ""
        words = cleaned.lower().replace(":", " ").split()
        if len(words) >= 3 and len(set(words)) == 1 and words[0] in {"atom", "user", "assistant", "boss"}:
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

    def _is_duplicate_chunk(self, text: str) -> bool:
        """Skip only immediate repeats of the same spoken chunk (streaming overlap).

        A short ring buffer used to catch duplicates caused the model to repeat
        a phrase later in the stream to be mistaken for a duplicate — skipping
        speech and sounding like dropped words.
        """
        key = self._chunk_key(text)
        if not key:
            return True
        if self._recent_spoken_chunks and key == self._recent_spoken_chunks[-1]:
            return True
        self._recent_spoken_chunks.append(key)
        if len(self._recent_spoken_chunks) > 2:
            self._recent_spoken_chunks = self._recent_spoken_chunks[-2:]
        return False

    def _pop_next_stream_segment(self, force: bool) -> str:
        """Take the next speakable slice from _stream_speak_buffer.

        Complete sentences (ending in . ! ?) flush immediately so latency stays
        low. Unpunctuated fragments wait until we have enough words to avoid
        NSSpeechSynthesizer speaking one token at a time.
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
        if n >= self._STREAM_UNPUNCT_BATCH:
            seg = " ".join(words[: self._STREAM_UNPUNCT_BATCH])
            self._stream_speak_buffer = " ".join(words[self._STREAM_UNPUNCT_BATCH :])
            return seg
        if n >= self._STREAM_UNPUNCT_MIN_WORDS:
            take = min(n, self._STREAM_UNPUNCT_BATCH)
            seg = " ".join(words[:take])
            self._stream_speak_buffer = " ".join(words[take:])
            return seg
        return ""

    async def _speak_stream_slice(self, raw_segment: str) -> None:
        """Apply word-cap + duplicate filtering, then speak one slice."""
        speak_text, overflow_text = self._split_stream_chunk(raw_segment)
        if speak_text and not self._cancel_requested:
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

                while True:
                    seg = self._pop_next_stream_segment(force=False)
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
                return

            overflow_text = " ".join(self._screen_buffer).strip()
            self._chunk_buffer.clear()
            self._screen_buffer.clear()
            self._stream_speak_buffer = ""
            self._stream_queue = None
            self._active_source = None
            self._active_stream_id = None

            if overflow_text:
                logger.info(
                    "Screen-only (%d words): '%s'",
                    len(overflow_text.split()),
                    overflow_text[:100],
                )
                self._bus.emit("text_display", text=overflow_text)

            logger.info(
                "TTS stream done: %d words spoken",
                self._spoken_word_count,
            )
            self._bus.emit("tts_complete")

    # ── Internal speak (no tts_complete emission) ──────────────────

    async def _speak_internal(self, text: str,
                              emotion: str | None = None) -> None:
        text = _truncate(text, self._max_lines)
        if not text or not self._available:
            return
        if self._cancel_requested:
            return

        async with self._speak_lock:
            if self._cancel_requested:
                return
            self._cancel_requested = False
            self._playing = True
            try:
                sentences = _split_sentences(text)
                if len(sentences) <= 1:
                    logger.info("TTS [%s]: '%s'", self._backend, text[:80])
                    await self._speak_one(text)
                else:
                    logger.info(
                        "TTS [%s, %d sentences]: '%s'",
                        self._backend, len(sentences), text[:80],
                    )
                    for sentence in sentences:
                        if self._cancel_requested:
                            break
                        await self._speak_one(sentence)
            except asyncio.CancelledError:
                await self.stop()
                raise
            except Exception:
                logger.exception("TTS speak error")
                self._bus.emit(
                    "text_display", text=f"[Response on screen] {text}",
                )
            finally:
                self._playing = False

    # ── Public API ─────────────────────────────────────────────────

    async def speak(self, text: str, emotion: str | None = None) -> None:
        """Speak text. Emits tts_complete when done."""
        await self._speak_internal(text, emotion)
        self._bus.emit("tts_complete")

    async def speak_ack(self, phrase: str) -> None:
        """Speak a short acknowledgement phrase."""
        if not phrase or not self._available:
            return
        logger.info("TTS ack [%s]: '%s'", self._backend, phrase)
        async with self._speak_lock:
            if self._cancel_requested:
                return
            self._playing = True
            try:
                await self._speak_one(phrase)
            finally:
                self._playing = False

    def next_ack_phrase(self) -> str:
        phrase = ACK_PHRASES[self._ack_idx % len(ACK_PHRASES)]
        self._ack_idx += 1
        return phrase

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
        queue = self._stream_queue
        self._stream_queue = None
        if queue is not None:
            try:
                queue.put_nowait(("", True))
            except Exception:
                pass
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
            logger.info("Barge-in detected (status placeholder), stopping TTS")
            await self.stop()
            return
        if self._state.current is AtomState.SPEAKING and len(t) >= 2:
            logger.info("Barge-in: user speech during TTS, stopping (%s)", t[:56])
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
        is_last: bool = False, source: str = "", stream_id: str = "", **_kw,
    ) -> None:
        from core.state_manager import AtomState

        normalized_text = self._normalize_stream_text(text) if text else ""

        if is_first:
            self._active_source = source or "unknown"
            self._chunk_buffer.clear()
            self._screen_buffer.clear()
            self._stream_speak_buffer = ""
            self._spoken_word_count = 0
            self._recent_spoken_chunks.clear()
            self._active_stream_id = stream_id or None
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
        queue.put_nowait((normalized_text, is_last))

    # ── Shutdown ───────────────────────────────────────────────────

    async def shutdown(self) -> None:
        await self.stop()
        self._native_synth = None
        logger.info("macOS TTS shut down (%s)", self._backend)
