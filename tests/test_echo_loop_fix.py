"""
ATOM -- Echo-loop regression fix focused tests.

Covers the three root causes of the "ATOM is talking to itself" bug:

  1. ``brain.mlx_llm._guard_visible_text`` strips a leading roleplay-quote
     wrapper (``"Boss, I'm showing ..."``) so TTS never speaks that shape.
  2. ``MacOSTTSAsync.is_echo`` flags an STT partial that mirrors the
     just-spoken content -- the STT promoter uses this to refuse final.
  3. ``NativeSTT._is_self_echo`` delegates to the injected guard and
     returns False when the guard is unwired (safe default).
"""
from __future__ import annotations

import sys
from pathlib import Path


def _root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def test_brain_strips_leading_quoted_boss_opener() -> None:
    sys.path.insert(0, str(_root()))
    from brain.mlx_llm import MLXBrain

    raw = '"Boss, I\'m showing you your active goals. Here they are:"'
    cleaned, _reason, _stop = MLXBrain._guard_visible_text(raw, ())
    # Normalize: at minimum, the leading quote must be gone.
    assert not cleaned.lstrip().startswith('"'), (
        f"leading quote survived: {cleaned!r}"
    )
    # And the body must still exist (no over-strip).
    assert "active goals" in cleaned.lower()


def test_brain_preserves_plain_sentence() -> None:
    from brain.mlx_llm import MLXBrain

    raw = "The weather today is sunny, 24 degrees."
    cleaned, _reason, _stop = MLXBrain._guard_visible_text(raw, ())
    assert cleaned.strip() == raw


def test_brain_strips_trailing_unclosed_quote() -> None:
    from brain.mlx_llm import MLXBrain

    raw = '"Boss, your 2 active goals are: one and two."'
    cleaned, _reason, _stop = MLXBrain._guard_visible_text(raw, ())
    assert not cleaned.rstrip().endswith('"'), f"trailing quote kept: {cleaned!r}"


def test_tts_is_echo_flags_own_voice() -> None:
    from voice.tts_macos import MacOSTTSAsync

    class _Stub(MacOSTTSAsync):
        def __init__(self) -> None:  # skip heavy init
            import collections, threading, time
            self._spoken_echo_window = collections.deque(maxlen=6)
            self._last_spoke_t = time.monotonic()
            self._recent_spoken_chunks = collections.deque(maxlen=3)
            self._last_spoken_was_confirmation = False
            self._echo_lock = threading.Lock()

    tts = _Stub()
    tts._record_spoken("Boss, I'm showing you your active goals.")
    tts._record_spoken("Here they are:")

    assert tts.is_echo("Boss I'm showing you your active goals") is True
    assert tts.is_echo("Boss I'm showing you your active goals here they are") is True
    assert tts.is_echo("what is the weather today") is False


def test_tts_is_echo_expires_after_window() -> None:
    import collections, threading, time as _time
    from voice.tts_macos import MacOSTTSAsync

    class _Stub(MacOSTTSAsync):
        def __init__(self) -> None:
            self._spoken_echo_window = collections.deque(maxlen=6)
            self._last_spoke_t = 0.0  # far in the past
            self._recent_spoken_chunks = collections.deque(maxlen=3)
            self._last_spoken_was_confirmation = False
            self._echo_lock = threading.Lock()

    tts = _Stub()
    tts._spoken_echo_window.append({"boss", "showing", "active", "goals"})
    assert tts.is_echo("Boss I'm showing you your active goals", window_s=0.1) is False


def test_stt_self_echo_without_guard_is_false() -> None:
    import importlib, sys, types

    class _FakeBus:
        def on(self, *a, **k): pass
        def emit(self, *a, **k): pass
        def emit_fast(self, *a, **k): pass

    class _FakeState:
        class AtomState:
            LISTENING = "listening"
        current = "listening"
        def on(self, *a, **k): pass

    stt_mod = importlib.import_module("voice.stt_macos")
    NativeSTT = stt_mod.NativeSTT

    class _StubSTT(NativeSTT):
        def __init__(self) -> None:
            self._echo_guard = None

    s = _StubSTT()
    assert s._is_self_echo("hello world") is False


def test_stt_self_echo_delegates_to_guard() -> None:
    from voice.stt_macos import NativeSTT

    class _StubSTT(NativeSTT):
        def __init__(self) -> None:
            self._echo_guard = lambda t: "echo" in t.lower()

    s = _StubSTT()
    assert s._is_self_echo("This is an ECHO of ATOM") is True
    assert s._is_self_echo("normal user query") is False


if __name__ == "__main__":
    test_brain_strips_leading_quoted_boss_opener()
    test_brain_preserves_plain_sentence()
    test_brain_strips_trailing_unclosed_quote()
    test_tts_is_echo_flags_own_voice()
    test_tts_is_echo_expires_after_window()
    test_stt_self_echo_without_guard_is_false()
    test_stt_self_echo_delegates_to_guard()
    print("[echo-loop] All echo-loop fix tests passed.")
