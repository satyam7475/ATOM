"""
ATOM -- voice.tts_async (compat shim).

The asynchronous TTS engine moved into ``voice.tts_macos`` (and
``voice.tts_edge`` / ``voice.tts_kokoro`` for the optional cloud paths).

This module exists so older callers and tests that still import
``voice.tts_async`` keep working: it re-exports the canonical
``clean_for_tts`` markdown stripper from the macOS engine, which is
the production sanitizer used before any synthesizer reads text.
"""

from __future__ import annotations

from voice.tts_macos import _clean_for_tts as clean_for_tts

__all__ = ["clean_for_tts"]
