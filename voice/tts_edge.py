"""
ATOM -- Edge Neural TTS with JARVIS-level speech quality.

Features:
    - Microsoft Neural voices (en-GB-RyanNeural default -- warm British male)
    - Sentence-level streaming (plays sentence N while generating N+1)
    - SSML micro-pauses at sentence/comma boundaries
    - Emotion profiles (rate/pitch/volume per context)
    - Audio post-processing (RMS normalization + tanh soft limiter)
    - Pre-cached acknowledgement phrases for instant short replies
    - Barge-in support (instant stop on hotkey / resume_listening)
    - Orphan temp file cleanup on startup

Same public interface as TTSAsync (SAPI) for drop-in replacement.

Requires: edge-tts, pygame
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from voice.voice_profiles import (
    VoiceProfile, detect_emotion, get_profile, get_time_aware_profile,
)

logger = logging.getLogger("atom.tts_edge")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

# ── Markdown cleanup ────────────────────────────────────────────────
_RE_CODE_BLOCK = re.compile(r'```.*?```', re.DOTALL)
_RE_INLINE_CODE = re.compile(r'`([^`]*)`')
_RE_BOLD = re.compile(r'\*\*([^*]+)\*\*')
_RE_ITALIC_STAR = re.compile(r'\*([^*]+)\*')
_RE_ITALIC_UNDER = re.compile(r'_([^_]+)_')
_RE_HEADER = re.compile(r'^#+\s*', re.MULTILINE)
_RE_BULLET = re.compile(r'^\s*[-*\u2022]\s+', re.MULTILINE)
_RE_NUMBERED = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r'^\s*>\s+', re.MULTILINE)

_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# ── Post-processing constants ──────────────────────────────────────
TARGET_OUTPUT_RMS = 3500.0
MAX_OUTPUT_GAIN = 1.8
LIMITER_THRESHOLD = 18000.0

# ── Pre-cached acknowledgement phrases ─────────────────────────────
ACK_PHRASES = [
    "Yes, Boss?", "I'm here.", "Go ahead.", "I'm listening.", "Tell me.",
    "What do you need?", "Right here.", "Present.",
    "Hmm, let me think...", "One sec, Boss.", "Working on it.",
    "Give me a moment.", "Let me check.", "On it, Boss.",
    "One second.", "Let me pull that up.", "Checking now.", "Hang on, Boss.",
    "I didn't catch that, Boss. Try again?",
    "That didn't work. Want me to try again?",
    "Done.", "Handled.", "All done.",
    "Searching now.", "Pulling up results.",
    "Let me look into that issue.",
    "Checking the forecast.",
    "Looking at the code.",
    "Let me think about that.",
    "Let me find that for you.",
    "Let me work through that.",
    "Weighing the options.",
    "Let me recall that.",
    "On it.",
    "Working on it now.",
    "Let me look at your screen, Boss.",
]


def _clean_for_tts(text: str) -> str:
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


def _split_sentences(text: str) -> list[str]:
    parts = _RE_SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts
            if p.strip() and len(p.strip()) > 1]


def _normalize_and_limit(raw_pcm: bytes) -> bytes:
    """RMS normalization + tanh soft limiter for consistent output volume.

    Normalizes to TARGET_OUTPUT_RMS (comfortable listening level).
    The soft limiter prevents harsh clipping on peaks while preserving
    speech dynamics -- much smoother than hard clipping.
    """
    samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return raw_pcm

    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms < 1.0:
        return raw_pcm

    gain = min(TARGET_OUTPUT_RMS / rms, MAX_OUTPUT_GAIN)
    samples *= gain

    mask = np.abs(samples) > LIMITER_THRESHOLD
    if mask.any():
        over = samples[mask]
        headroom = 32767.0 - LIMITER_THRESHOLD
        samples[mask] = np.sign(over) * (
            LIMITER_THRESHOLD
            + headroom * np.tanh(
                (np.abs(over) - LIMITER_THRESHOLD) / max(headroom, 1.0)
            )
        )

    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


class EdgeTTSAsync:
    """Edge Neural TTS with streaming, emotion profiles, and post-processing.

    Public API matches TTSAsync (SAPI) so they're interchangeable.
    """

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        max_lines: int = 4,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+15%",
        enable_postprocess: bool = True,
        enable_ack_cache: bool = True,
    ) -> None:
        self._bus = bus
        self._state = state
        self._max_lines = max_lines
        self._voice = voice
        self._default_rate = rate
        self._enable_postprocess = enable_postprocess
        self._config_postprocess = enable_postprocess  # Restored on governor_normal
        self._enable_ack_cache = enable_ack_cache
        self._mixer_ready = False
        self._playing = False
        self._cancel_requested = False
        self._tmp_files: list[str] = []
        self._ack_cache: dict[str, str] = {}
        self._ack_idx = 0
        self._current_output_device: str | None = None
        self._speak_lock = asyncio.Lock()
        self._consecutive_failures: int = 0
        self._active_source: str | None = None

        # Smart partial buffering -- batch chunks, cap audio at ~45 words
        self._chunk_buffer: list[str] = []
        self._spoken_word_count: int = 0
        self._SPEAK_WORD_LIMIT: int = 45
        self._screen_overflow: list[str] = []

        self._cleanup_orphan_temps()

    # ── Orphan temp file cleanup ──────────────────────────────────

    @staticmethod
    def _cleanup_orphan_temps() -> None:
        """Remove leftover atom_tts_*.mp3 and atom_ack_*.mp3 from temp dir.

        Previous ATOM sessions may have crashed before cleaning up.
        """
        import glob
        tmp_dir = tempfile.gettempdir()
        count = 0
        for pattern in ("atom_tts_*.mp3", "atom_ack_*.mp3"):
            for path in glob.glob(os.path.join(tmp_dir, pattern)):
                try:
                    os.unlink(path)
                    count += 1
                except OSError:
                    pass
        if count:
            logger.info("Cleaned %d orphan TTS temp files from previous session", count)

    # ── Initialization ─────────────────────────────────────────────

    _BT_KEYWORDS = ("headset", "hands-free", "bluetooth", "bt", "buds",
                     "airpods", "earbuds", "jbl", "bose", "sony", "mivi",
                     "oneplus", "realme", "yealink", "blaupunkt", "jabra")

    # Smaller buffer reduces latency (Apple CoreAudio style)
    # 1024 frames at 24kHz is ~42ms of latency (very fast, no dropouts)
    _MIXER_BUFFER = 1024

    async def init_voice(self) -> None:
        """Initialize pygame mixer, preferring Bluetooth output if available."""
        try:
            import pygame
            bt_device = self._find_bluetooth_output()
            if bt_device:
                logger.info("Bluetooth output detected -- routing TTS to: '%s'",
                            bt_device)
                try:
                    pygame.mixer.init(
                        frequency=24000, size=-16, channels=1,
                        buffer=self._MIXER_BUFFER,
                        devicename=bt_device,
                    )
                    self._current_output_device = bt_device
                except Exception as bt_exc:
                    logger.warning(
                        "Bluetooth mixer init failed (%s), using system default",
                        bt_exc,
                    )
                    pygame.mixer.init(
                        frequency=24000, size=-16, channels=1,
                        buffer=self._MIXER_BUFFER,
                    )
                    self._current_output_device = "system_default"
            else:
                pygame.mixer.init(
                    frequency=24000, size=-16, channels=1,
                    buffer=self._MIXER_BUFFER,
                )
                self._current_output_device = "system_default"
            self._mixer_ready = True
            logger.info(
                "Edge-TTS ready (voice=%s, rate=%s, postprocess=%s)",
                self._voice, self._default_rate, self._enable_postprocess,
            )
        except ImportError:
            logger.error("pygame not installed (pip install pygame)")
            return
        except Exception as exc:
            logger.error("pygame mixer init failed: %s", exc)
            return

        if self._enable_ack_cache:
            asyncio.create_task(self._pregenerate_acks())

    def _find_bluetooth_output(self) -> str | None:
        """Find a connected Bluetooth output device name for pygame."""
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            from pygame._sdl2.audio import get_audio_device_names
            output_devices = get_audio_device_names(False)
            for dev in output_devices:
                if any(kw in dev.lower() for kw in self._BT_KEYWORDS):
                    return dev
        except Exception:
            pass
        return None

    def set_postprocess(self, enabled: bool) -> None:
        """Enable or disable TTS audio post-processing (normalize + limit).

        Used by the CPU governor: when throttled, post-processing is disabled
        to reduce CPU; when normal, restore to config-driven value.
        """
        self._enable_postprocess = enabled
        logger.debug("TTS postprocess set to %s", enabled)

    def restore_postprocess(self) -> None:
        """Restore post-processing to the config-driven value (e.g. after governor_normal)."""
        self._enable_postprocess = self._config_postprocess
        logger.debug("TTS postprocess restored to config: %s", self._config_postprocess)

    def refresh_output_device(self) -> bool:
        """Re-init pygame mixer if Bluetooth output appears/disappears.

        Returns True if output device was changed. Synchronous because
        pygame.mixer operations are blocking calls, not coroutines.
        Skips re-init if already on the same device to avoid audio glitches.
        """
        try:
            import pygame
            bt_device = self._find_bluetooth_output()
            if bt_device:
                if self._current_output_device == bt_device:
                    return False
                if self._playing:
                    return False
                pygame.mixer.quit()
                pygame.mixer.init(
                    frequency=24000, size=-16, channels=1,
                    buffer=self._MIXER_BUFFER,
                    devicename=bt_device,
                )
                self._current_output_device = bt_device
                logger.info("TTS output switched to Bluetooth: '%s'", bt_device)
                return True
            elif self._current_output_device and \
                    self._current_output_device != "system_default":
                if self._playing:
                    return False
                pygame.mixer.quit()
                pygame.mixer.init(
                    frequency=24000, size=-16, channels=1,
                    buffer=self._MIXER_BUFFER,
                )
                self._current_output_device = "system_default"
                logger.info("BT disconnected -- TTS output switched to system default")
                return True
        except Exception as exc:
            logger.debug("Bluetooth output refresh error: %s", exc)
        return False

    async def _pregenerate_acks(self) -> None:
        """Pre-generate acknowledgement audio at startup for instant playback."""
        try:
            import edge_tts

            profile = get_profile("ack")
            for phrase in ACK_PHRASES:
                fd, tmp = tempfile.mkstemp(suffix=".mp3", prefix="atom_ack_")
                os.close(fd)
                comm = edge_tts.Communicate(
                    phrase, self._voice,
                    rate=profile.rate, pitch=profile.pitch,
                    volume=profile.volume,
                )
                await comm.save(tmp)
                self._ack_cache[phrase.lower()] = tmp
            logger.info("Pre-cached %d ack phrases for instant playback",
                        len(self._ack_cache))
        except Exception as exc:
            logger.warning("Ack pre-cache failed (non-fatal): %s", exc)

    # ── Audio generation ──────────────────────────────────────────

    async def _generate_audio(self, text: str,
                              profile: VoiceProfile,
                              timeout_s: float = 10.0) -> str:
        """Generate MP3 via Edge-TTS websocket with a hard timeout.

        If Edge-TTS doesn't respond within timeout_s, raises TimeoutError
        so the caller can skip the sentence instead of hanging forever.
        """
        import edge_tts

        comm = edge_tts.Communicate(
            text, self._voice,
            rate=profile.rate, pitch=profile.pitch,
            volume=profile.volume,
        )
        fd, tmp = tempfile.mkstemp(suffix=".mp3", prefix="atom_tts_")
        os.close(fd)
        self._tmp_files.append(tmp)

        await asyncio.wait_for(comm.save(tmp), timeout=timeout_s)
        return tmp

    # ── Audio playback with post-processing ──────────────────────

    async def _play_audio(self, path: str, cleanup: bool = True) -> None:
        """Play audio with optional normalization + soft limiter.

        Uses mixer.Sound for postprocessed playback; falls back to
        mixer.music if Sound fails (e.g. some Bluetooth drivers).
        Reinitializes mixer if it becomes stale (BT disconnect etc).
        """
        import pygame

        if self._cancel_requested:
            return

        if not pygame.mixer.get_init():
            logger.warning("Mixer went stale -- reinitializing")
            try:
                await self.init_voice()
            except Exception:
                logger.exception("Mixer reinit failed")
                return

        try:
            if self._enable_postprocess:
                sound = pygame.mixer.Sound(path)
                raw = sound.get_raw()
                processed = _normalize_and_limit(raw)
                processed_sound = pygame.mixer.Sound(buffer=processed)
                channel = processed_sound.play()
                while (channel and channel.get_busy()
                       and not self._cancel_requested):
                    await asyncio.sleep(0.02)
            else:
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(0.85)
                pygame.mixer.music.play()
                while (pygame.mixer.music.get_busy()
                       and not self._cancel_requested):
                    await asyncio.sleep(0.02)
        except Exception as exc:
            if self._cancel_requested:
                return
            logger.warning(
                "Sound playback error (%s), trying music fallback",
                exc,
            )
            try:
                if not os.path.exists(path):
                    return
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(0.85)
                pygame.mixer.music.play()
                while (pygame.mixer.music.get_busy()
                       and not self._cancel_requested):
                    await asyncio.sleep(0.02)
            except Exception as fallback_exc:
                logger.exception(
                    "Fallback playback failed: %s",
                    fallback_exc,
                )

        if cleanup and path in self._tmp_files:
            self._tmp_files.remove(path)
            try:
                os.unlink(path)
            except OSError:
                pass

    # ── Sentence-level streaming ─────────────────────────────────

    async def _stream_sentences(self, sentences: list[str],
                                profile: VoiceProfile) -> None:
        """Producer-consumer streaming: generate sentence N+1 while playing N.

        This overlaps synthesis and playback so multi-sentence responses
        have minimal gaps between sentences.
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)

        async def produce() -> None:
            try:
                for sentence in sentences:
                    if self._cancel_requested:
                        break
                    if self._consecutive_failures >= 2:
                        logger.warning("Edge-TTS unstable -- skipping remaining sentences")
                        break
                    try:
                        tmp = await self._generate_audio(sentence, profile)
                        await queue.put(tmp)
                        self._consecutive_failures = 0
                    except asyncio.TimeoutError:
                        logger.warning("Edge-TTS timeout for: '%s' -- retrying",
                                       sentence[:40])
                        try:
                            tmp = await self._generate_audio(sentence, profile,
                                                             timeout_s=8.0)
                            await queue.put(tmp)
                            self._consecutive_failures = 0
                        except Exception:
                            self._consecutive_failures += 1
                            logger.warning("Edge-TTS retry failed (%d) -- skipping: '%s'",
                                           self._consecutive_failures, sentence[:40])
                    except Exception:
                        self._consecutive_failures += 1
                        logger.exception("Sentence generation failed (%d): '%s'",
                                         self._consecutive_failures, sentence[:40])
            finally:
                await queue.put(None)

        async def consume() -> None:
            while not self._cancel_requested:
                tmp = await queue.get()
                if tmp is None:
                    break
                await self._play_audio(tmp)

        await asyncio.gather(produce(), consume())

    # ── Core speak (internal, no tts_complete emission) ──────────

    async def _speak_internal(self, text: str,
                              emotion: str | None = None) -> None:
        """Core speak logic used by all public methods.

        Acquires _speak_lock to guarantee only one piece of audio
        plays at a time -- prevents overlap between ack / partial / response.

        Checks the ack cache first for instant playback of common phrases.
        """
        text = _truncate(text, self._max_lines)
        if not text:
            return

        if not self._mixer_ready:
            logger.warning("Edge-TTS not initialized, skipping")
            return

        if self._cancel_requested:
            return

        async with self._speak_lock:
            if self._cancel_requested:
                return

            cached = self._ack_cache.get(text.lower().strip())
            if cached is None:
                cached = self._ack_cache.get(text.lower().strip().rstrip("."))
            if cached and os.path.exists(cached):
                self._playing = True
                self._cancel_requested = False
                logger.info("Edge-TTS [cached]: '%s'", text[:80])
                try:
                    await self._play_audio(cached, cleanup=False)
                finally:
                    self._playing = False
                return

            if emotion is None:
                emotion = detect_emotion(text)
            profile = get_time_aware_profile(emotion)

            logger.info("Edge-TTS [%s]: '%s'", profile.name, text[:80])

            self._cancel_requested = False
            self._playing = True

            try:
                sentences = _split_sentences(text)
                if len(sentences) <= 1:
                    tmp = await self._generate_audio(text, profile)
                    await self._play_audio(tmp)
                else:
                    await self._stream_sentences(sentences, profile)
                self._consecutive_failures = 0
            except asyncio.CancelledError:
                await self.stop()
                raise
            except Exception:
                self._consecutive_failures += 1
                logger.exception("Edge-TTS speak error (failure %d)",
                                 self._consecutive_failures)
                self._bus.emit("text_display",
                               text=f"[Response on screen] {text}")
            finally:
                self._playing = False
                self._cleanup_temps()

    # ── Public API ───────────────────────────────────────────────

    async def speak(self, text: str, emotion: str | None = None) -> None:
        """Speak text with emotion profile. Emits tts_complete when done."""
        await self._speak_internal(text, emotion)
        self._bus.emit("tts_complete")

    async def speak_ack(self, phrase: str) -> None:
        """Play a short acknowledgement -- instant from pre-cached audio.

        Uses _speak_lock so the ack is cleanly interrupted when a
        real response arrives (stop() sets _cancel_requested, the lock
        is released, and the response acquires the lock next).
        """
        if not phrase or not self._mixer_ready:
            return

        logger.info("Edge-TTS ack: '%s'", phrase)

        async with self._speak_lock:
            if self._cancel_requested:
                return

            key = phrase.lower().strip()
            cached = self._ack_cache.get(key)
            if cached is None:
                stripped = key.rstrip(".")
                cached = self._ack_cache.get(stripped)

            if cached and os.path.exists(cached):
                self._playing = True
                self._cancel_requested = False
                try:
                    await self._play_audio(cached, cleanup=False)
                finally:
                    self._playing = False
                return

            try:
                profile = get_profile("ack")
                tmp = await self._generate_audio(phrase, profile)
                self._playing = True
                self._cancel_requested = False
                await self._play_audio(tmp)
            except Exception:
                logger.exception("Edge-TTS ack generation error")
            finally:
                self._playing = False

    def next_ack_phrase(self) -> str:
        """Cycle through acknowledgement phrases for variety."""
        phrase = ACK_PHRASES[self._ack_idx % len(ACK_PHRASES)]
        self._ack_idx += 1
        return phrase

    async def stop(self) -> None:
        """Barge-in: immediately stop all audio playback."""
        self._cancel_requested = True
        self._playing = False
        if self._mixer_ready:
            try:
                import pygame
                pygame.mixer.stop()
                pygame.mixer.music.stop()
            except Exception:
                pass
            await asyncio.sleep(0.05)

    def _cleanup_temps(self) -> None:
        for path in list(self._tmp_files):
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()

    # ── Event handlers (same interface as TTSAsync) ──────────────

    async def on_speech_partial(self, text: str, **_kw) -> None:
        """Barge-in: stop speaking immediately when user starts talking."""
        if self._playing and text in ("Listening...", "Processing..."):
            logger.info("Barge-in detected, stopping TTS")
            await self.stop()

    async def on_response(self, text: str, is_exit: bool = False,
                          is_sleep: bool = False, **_kw) -> None:
        from core.state_manager import AtomState

        self._active_source = None

        if self._playing:
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
                logger.exception("Edge-TTS background speak error")
                self._bus.emit("text_display",
                               text=f"[Response on screen] {text}")
                self._bus.emit("tts_complete")
            if is_exit:
                self._bus.emit("shutdown_requested")

        asyncio.create_task(_speak_bg())

    async def on_partial_response(
        self,
        text: str,
        is_first: bool = False,
        is_last: bool = False,
        source: str = "",
        **_kw,
    ) -> None:
        from core.state_manager import AtomState

        if is_first:
            self._active_source = source or "unknown"
            self._chunk_buffer.clear()
            self._screen_overflow.clear()
            self._spoken_word_count = 0
            logger.info("TTS stream: source='%s'", self._active_source)
            await self.stop()
            self._cancel_requested = False
            await self._state.transition(AtomState.SPEAKING)
        elif source and self._active_source and source != self._active_source:
            return

        if not text.strip() and not is_last:
            return

        if (self._state.current is not AtomState.SPEAKING
                and not is_first):
            self._cancel_requested = False
            await self._state.transition(AtomState.SPEAKING)

        if text:
            self._chunk_buffer.append(text)

        if is_last:
            asyncio.create_task(self._flush_and_finish())

    async def _flush_and_finish(self) -> None:
        """Assemble full response, speak first ~45 words, show rest on screen."""
        try:
            full_text = " ".join(self._chunk_buffer).strip()
            self._chunk_buffer.clear()

            full_text = _clean_for_tts(full_text).strip()
            import re as _re
            full_text = _re.sub(r'\s+', ' ', full_text)
            if not full_text:
                logger.info("TTS: empty response, nothing to speak")
                self._active_source = None
                self._bus.emit("tts_complete")
                return

            words = full_text.split()
            total_words = len(words)

            if total_words <= self._SPEAK_WORD_LIMIT:
                speak_text = full_text
                overflow_text = ""
            else:
                speak_words = words[:self._SPEAK_WORD_LIMIT]
                speak_text = " ".join(speak_words)
                last_period = speak_text.rfind(".")
                last_question = speak_text.rfind("?")
                last_exclaim = speak_text.rfind("!")
                cut_pos = max(last_period, last_question, last_exclaim)
                if cut_pos > len(speak_text) // 3:
                    speak_text = speak_text[:cut_pos + 1]
                overflow_text = full_text[len(speak_text):].strip()

            if speak_text:
                self._spoken_word_count = len(speak_text.split())
                logger.info("TTS speak (%d/%d words): '%s'",
                            self._spoken_word_count, total_words, speak_text[:100])
                await self._speak_internal(speak_text)

            if overflow_text:
                overflow_words = len(overflow_text.split())
                logger.info("Screen-only (%d words): '%s'",
                            overflow_words, overflow_text[:100])
                self._bus.emit("text_display", text=overflow_text)

            logger.info("TTS done: %d/%d words spoken via audio",
                        self._spoken_word_count, total_words)
        except Exception:
            logger.exception("Edge-TTS flush error")
        self._active_source = None
        self._bus.emit("tts_complete")

    # ── Shutdown ─────────────────────────────────────────────────

    async def shutdown(self) -> None:
        await self.stop()
        for path in self._ack_cache.values():
            try:
                os.unlink(path)
            except OSError:
                pass
        self._ack_cache.clear()
        if self._mixer_ready:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass
        logger.info("Edge-TTS shut down (ack cache cleared)")
