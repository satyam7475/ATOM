"""
ATOM -- voice.tts_async (compat shim).

The asynchronous TTS engine moved into ``voice.tts_macos`` (and
``voice.tts_edge`` / ``voice.tts_kokoro`` for the optional cloud paths).

This module exists so older callers and tests that still import
``voice.tts_async`` keep working: it re-exports the canonical macOS
TTS class and text sanitizers used before any synthesizer reads text.
"""

from __future__ import annotations

from voice.tts_macos import (
    MacOSTTSAsync as TTSAsync,
    _clean_for_tts as clean_for_tts,
    _truncate,
)

__all__ = ["TTSAsync", "clean_for_tts", "_truncate"]
