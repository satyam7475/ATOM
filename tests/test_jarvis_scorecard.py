from __future__ import annotations

from pathlib import Path

from scripts.jarvis_scorecard import parse_scorecard


def test_scorecard_parses_voice_boot_and_memory_metrics(tmp_path: Path) -> None:
    log = tmp_path / "atom.log"
    log.write_text(
        "\n".join([
            "2026-04-26 | atom.main | INFO | TTS ready (1084ms: voice select + prewarm)",
            "2026-04-26 | atom.local_brain | INFO | Local brain ready in 4275ms (MLX)",
            "2026-04-26 | atom.main | INFO | Cold-start bootstrap: 6022ms (fast=True)",
            "2026-04-26 | atom.voice_pipeline | INFO | VOICE_LOOP_READY: 6ms | stt=WhisperKit",
            "2026-04-26 | atom.stt_whisperkit | INFO | WhisperKitSTT preloaded (model=x) in 11263 ms",
            "2026-04-26 | atom.main | INFO | STT pipeline ready (22708ms: devices + model)",
            "2026-04-26 | atom.boot.timeline | INFO | Boot timeline: total=24762ms | ∥tts_init=1084ms cold_start=6022ms persona_pin=0ms ∥stt_preload=22708ms",
            "2026-04-26 | atom.stt_whisperkit | INFO | WhisperKitSTT listening (16000 Hz)",
            "2026-04-26 | atom.main | WARNING | Memory pressure tier 0 -> 1 (memory_pct=81.8%)",
            "2026-04-26 | atom.voice_interrupt | INFO | Voice interrupt partial detected: 'Thank you.'",
            "2026-04-26 | atom.voice_interrupt | INFO | Echo suppressed (TTS self-feedback): 'Thank you.'",
        ]),
        encoding="utf-8",
    )

    card = parse_scorecard(log)

    assert card.boot_total_ms == 24762
    assert card.stt_pipeline_ready_ms == 22708
    assert card.whisperkit_preload_ms == 11263
    assert card.max_memory_pct == 81.8
    assert card.voice_pipeline_active is True
    assert card.stt_listening_active is True
    assert card.echo_suppressions == 1
    assert card.polite_interrupt_candidates == 1
    assert card.score < 80


def test_scorecard_rewards_clean_fast_boot(tmp_path: Path) -> None:
    log = tmp_path / "atom.log"
    log.write_text(
        "\n".join([
            "2026-04-26 | atom.voice_pipeline | INFO | VOICE_LOOP_READY: 4ms | stt=WhisperKit",
            "2026-04-26 | atom.stt_whisperkit | INFO | WhisperKitSTT listening (16000 Hz)",
            "2026-04-26 | atom.boot.timeline | INFO | Boot timeline: total=14500ms | ∥tts_init=800ms cold_start=4300ms persona_pin=0ms ∥stt_preload=9000ms",
            "2026-04-26 | atom.observability.snapshot | INFO | system_state: {'memory_percent': 71.5}",
        ]),
        encoding="utf-8",
    )

    card = parse_scorecard(log)

    assert card.score >= 90
    assert card.grade == "A"
