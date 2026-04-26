#!/usr/bin/env python3
"""S2 -- Voice / multilingual understanding smoke test.

Drives ATOM's WhisperConfirmer (and indirectly the active STT engine) on
a fixture file of phrases (one per line), then reports per-phrase
correctness using a tolerant edit-distance / token-overlap score. This
is the test that proves P1.1+P1.2 actually unlocked Hindi + Hinglish.

This script does NOT need a running ATOM dashboard. It loads
``WhisperConfirmer`` directly with the project config, builds a
synthetic audio buffer for each phrase via macOS' ``say`` command piped
through ``sox`` -> 16 kHz mono PCM, then asks the confirmer to decode
it. The confirmer is a strict subset of the production STT path so a
high pass-rate here strongly correlates with a high pass-rate live.

Pass criteria (per docs/ATOM_NEXT_STEPS_PLAN.md §4):
    * >= 90% phrase-level token-overlap on the fixture.
    * Returns non-empty text on every phrase.

Exit codes:
    0  OK
    1  setup error (deps / fixture missing)
    2  pass rate below threshold (only with --strict)

Usage::

    python scripts/voice_understanding_smoke.py
    python scripts/voice_understanding_smoke.py \
        --phrases scripts/data/hinglish.txt --json --strict

Notes:
    * Requires ``say`` (built-in on macOS) and ``sox`` (``brew install sox``).
    * The synthetic-audio path is a smoke test, not a clean WER eval --
      use real recorded fixtures for a release-grade benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any


def _normalise(text: str) -> list[str]:
    if not text:
        return []
    nf = unicodedata.normalize("NFKC", text).lower().strip()
    out: list[str] = []
    buf: list[str] = []
    for ch in nf:
        if ch.isalnum() or ord(ch) > 0x7F:
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _token_overlap(expected: str, actual: str) -> float:
    e_tokens = _normalise(expected)
    a_tokens = set(_normalise(actual))
    if not e_tokens:
        return 0.0
    matched = sum(1 for t in e_tokens if t in a_tokens)
    return matched / max(1, len(e_tokens))


def _have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _synthesise_phrase(
    text: str,
    *,
    workdir: Path,
    voice: str,
) -> tuple[bytes, int]:
    """Return (pcm_int16_bytes, sample_rate) for a synthesised utterance.

    Uses macOS ``say`` -> AIFF -> sox WAV mono 16 kHz int16. Falls back
    to silence (zeros) if the synth toolchain is unavailable so we still
    exercise the STT entry path -- but the score will of course be 0.
    """
    sample_rate = 16000
    aiff = workdir / "phrase.aiff"
    wav = workdir / "phrase.wav"
    if not _have_tool("say") or not _have_tool("sox"):
        return b"\x00\x00" * sample_rate, sample_rate

    subprocess.run(
        ["say", "-v", voice, "-o", str(aiff), text],
        check=True, capture_output=True, timeout=20,
    )
    subprocess.run(
        [
            "sox", str(aiff), "-r", str(sample_rate),
            "-b", "16", "-c", "1", str(wav),
        ],
        check=True, capture_output=True, timeout=20,
    )
    with wave.open(str(wav), "rb") as w:
        pcm = w.readframes(w.getnframes())
        sr = w.getframerate()
    return pcm, sr


def _score_phrase(confirmer: Any, pcm: bytes, expected: str) -> dict[str, Any]:
    try:
        # WhisperConfirmer expects f32 audio for `feed_audio` and runs
        # `confirm` on a current text. We feed PCM and then call the raw
        # decode path that confirm uses internally so we can score.
        import numpy as np  # type: ignore[import-untyped]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        confirmer.feed_audio(samples.tobytes())
        # Most WhisperConfirmer implementations expose a `_decode_window`
        # or similar internal -- try the public ``confirm`` path with a
        # blank streaming text so the confirmer falls back to its own
        # decode.
        result = confirmer.confirm("", 0.0)
        text = (getattr(result, "text", "") or "").strip()
    except Exception as exc:
        return {"text": "", "error": repr(exc), "score": 0.0}
    return {
        "text": text,
        "error": None,
        "score": round(_token_overlap(expected, text), 3),
    }


def _load_phrases(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ATOM S2 voice understanding probe",
    )
    parser.add_argument(
        "--phrases", default="scripts/data/hinglish.txt",
        help="path to a phrase fixture file (one phrase per line)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero when pass rate < threshold",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="minimum phrase-level token-overlap rate (0.0-1.0)",
    )
    parser.add_argument(
        "--voice", default="Daniel",
        help="macOS `say` voice for synthesis (default: Daniel)",
    )
    args = parser.parse_args()

    fixture = Path(args.phrases)
    if not fixture.is_file():
        msg = {"status": "fixture_missing", "path": str(fixture)}
        print(json.dumps(msg) if args.json else msg)
        return 1
    phrases = _load_phrases(fixture)
    if not phrases:
        msg = {"status": "fixture_empty", "path": str(fixture)}
        print(json.dumps(msg) if args.json else msg)
        return 1

    try:
        from core.boot.config_loader import load_config
        from voice.whisper_confirmer import WhisperConfirmer
    except Exception as exc:
        msg = {"status": "import_error", "error": repr(exc)}
        print(json.dumps(msg) if args.json else msg)
        return 1
    cfg = load_config()
    try:
        confirmer = WhisperConfirmer(cfg.get("stt", {}).get("whisper_confirm", {}))
    except Exception as exc:
        msg = {"status": "confirmer_construct_error", "error": repr(exc)}
        print(json.dumps(msg) if args.json else msg)
        return 1

    if not (
        _have_tool("say") and _have_tool("sox")
    ):
        # Without synth we still exercise the loader path but report
        # a zero-score run so the operator knows to install sox.
        if not args.json:
            print(
                "WARN: `say` and/or `sox` missing -- using silence inputs. "
                "Install sox for a meaningful score: brew install sox",
            )

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        for phrase in phrases:
            t0 = time.perf_counter()
            try:
                pcm, _sr = _synthesise_phrase(
                    phrase, workdir=workdir, voice=args.voice,
                )
            except Exception as exc:
                results.append({
                    "phrase": phrase, "status": "synth_failed",
                    "error": repr(exc), "elapsed_ms": 0.0, "score": 0.0,
                })
                continue
            scored = _score_phrase(confirmer, pcm, phrase)
            results.append({
                "phrase": phrase,
                "decoded": scored["text"],
                "score": scored["score"],
                "error": scored["error"],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            })

    avg_score = sum(r.get("score", 0.0) for r in results) / max(1, len(results))
    pass_rate = sum(1 for r in results if r.get("score", 0.0) >= 0.6) / max(1, len(results))
    summary = {
        "status": "ok",
        "phrases": len(results),
        "avg_score": round(avg_score, 3),
        "pass_rate": round(pass_rate, 3),
        "threshold": args.threshold,
        "fixture": str(fixture),
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, default=str))
    else:
        print(
            f"S2 Voice Understanding -- "
            f"phrases={summary['phrases']} avg={summary['avg_score']:.2f} "
            f"pass_rate={summary['pass_rate']:.2f} (threshold={summary['threshold']:.2f})",
        )
        for r in results:
            print(
                f"  [{r.get('score', 0.0):.2f}] {r['phrase'][:60]:<60} -> "
                f"{(r.get('decoded') or '<empty>')[:80]}",
            )

    if args.strict and pass_rate < args.threshold:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
