"""Sprint K recovery regressions.

These tests pin the fixes from the 2026-04-25 bad-demo log:
missing whisper model fallback, FAST stop-sequence empty replies,
camera/screen routing, false "Done" on music failure, short TTS
chunking, native-STT echo mute, and mood no-face debounce.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest


def test_fast_path_stop_sequences_do_not_include_open_paren() -> None:
    from brain.mlx_llm import _FAST_PATH_STOP_SEQUENCES

    assert "(" not in _FAST_PATH_STOP_SEQUENCES
    assert "\n\n" in _FAST_PATH_STOP_SEQUENCES


def test_vision_intent_tolerates_trailing_stt_fragment() -> None:
    from core.intent_engine import vision_intents

    result = vision_intents.check("Can you see me at")
    assert result is not None
    assert result.action == "vision_describe"


def test_music_actions_not_fire_and_forget() -> None:
    from core.router.router import Router

    assert "music_play" not in Router._FIRE_AND_FORGET_ACTIONS
    assert "music_pause" not in Router._FIRE_AND_FORGET_ACTIONS
    assert "music_next" not in Router._FIRE_AND_FORGET_ACTIONS
    assert "music_prev" not in Router._FIRE_AND_FORGET_ACTIONS


def test_tts_short_reply_bypasses_chunking() -> None:
    from voice.tts_macos import MacOSTTSAsync

    assert MacOSTTSAsync._should_skip_chunking("Here, Boss. 2 active goals.")
    assert not MacOSTTSAsync._should_skip_chunking(
        " ".join(f"word{i}" for i in range(20)),
    )


def test_whisper_install_skips_existing_model(tmp_path: Path) -> None:
    from voice.whisper_install import ensure_model

    model = tmp_path / "ggml-tiny.en-q5_1.bin"
    # Larger than half of the tiny model's advertised 32 MB size.
    model.write_bytes(b"0" * (17 * 1024 * 1024))
    seen: list[str] = []

    out = ensure_model(model_path=model, model_key="tiny.en-q5_1", progress_cb=seen.append)

    assert out == model.resolve()
    assert any("already present" in msg for msg in seen)


def test_whisper_install_rejects_unknown_model(tmp_path: Path) -> None:
    from voice.whisper_install import ensure_model

    with pytest.raises(ValueError):
        ensure_model(model_path=tmp_path / "x.bin", model_key="unknown")


def test_llm_streaming_supports_repair_stop_override() -> None:
    import inspect

    from cursor_bridge.local_brain_controller import LocalBrainController

    sig = inspect.signature(LocalBrainController._run_llm_streaming)
    assert "extra_stop_sequences_override" in sig.parameters


def test_router_music_play_stashes_apple_music_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.router import router as router_mod
    from core.router.offer_registry import get_offer_registry, reset_offer_registry
    from core.router.router import Router

    reset_offer_registry()
    monkeypatch.setattr(router_mod.spotify_actions, "play", lambda: False)
    stub = Router.__new__(Router)
    stub._bus = type("Bus", (), {"emit_fast": lambda *_a, **_kw: None})()

    text = Router._do_music_play(stub, "music_play", {})
    offer = get_offer_registry().peek()

    assert "Spotify isn't installed" in text
    assert offer is not None
    assert offer.action == "open_app"
    assert "Music" in str(offer.args)


@pytest.mark.asyncio
async def test_mood_no_face_requires_three_samples() -> None:
    from core.cognitive.mood_inference import MoodInferenceEngine

    class Bus:
        def __init__(self) -> None:
            self.handlers: dict[str, list[Any]] = {}
            self.emitted: list[tuple[str, dict[str, Any]]] = []

        def on(self, event: str, handler: Any) -> None:
            self.handlers.setdefault(event, []).append(handler)

        def emit_long(self, event: str, **payload: Any) -> None:
            self.emitted.append((event, payload))

        async def fire(self, event: str, **payload: Any) -> None:
            for handler in list(self.handlers.get(event, ())):
                await handler(**payload)

    bus = Bus()
    engine = MoodInferenceEngine(bus, min_consecutive=1)
    engine.attach()

    await bus.fire("presence.snapshot", present=False, face_count=0, quality="no_face")
    await bus.fire("presence.snapshot", present=False, face_count=0, quality="no_face")
    assert not [p for evt, p in bus.emitted if p.get("mood") == "idle"]

    await bus.fire("presence.snapshot", present=False, face_count=0, quality="no_face")
    assert any(p.get("mood") == "idle" for evt, p in bus.emitted)


def test_native_stt_tail_mute_blocks_recognizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.state_manager import AtomState
    from voice.stt_macos import NativeSTT

    stt = NativeSTT.__new__(NativeSTT)
    stt._state = type("State", (), {"current": AtomState.LISTENING})()
    stt._barge_in_during_speak = False
    stt._callback_hard_muted = False
    stt._callback_muted_until = 10**12

    assert not NativeSTT._should_feed_recognizer(stt)
