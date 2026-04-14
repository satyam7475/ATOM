#!/usr/bin/env python3
"""
ATOM — Golden path E2E (CI-safe by default).

Default: no mic, no full boot — loads config, checks STT resolution, cognitive route.
Optional --live-mic: macOS + native STT only; skips with exit 2 when unsupported.

Exit codes: 0 success, 1 failure, 2 skipped (unsupported environment).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)


def _ci_golden() -> int:
    from core.boot.config_loader import load_config
    from core.cognitive_kernel import CognitiveKernel

    config = load_config()
    if not config:
        print("golden_path_e2e: FAIL — empty config")
        return 1

    kernel = CognitiveKernel(config=config)
    plan = kernel.route("What time is it?")
    if plan is None or not getattr(plan, "path", None):
        print("golden_path_e2e: FAIL — cognitive kernel returned empty plan")
        return 1

    cloud = bool(config.get("cloud", {}).get("enabled", True))
    tier = str((config.get("deployment") or {}).get("product_tier", "") or "balanced")
    print(
        f"golden_path_e2e: OK — route={plan.path!s} cloud.enabled={cloud} "
        f"deployment.product_tier={tier}"
    )
    return 0


async def _live_mic_golden(timeout_s: float) -> int:
    if sys.platform != "darwin":
        print("golden_path_e2e: SKIP — --live-mic requires macOS")
        return 2

    from voice.stt_macos import native_stt_launch_supported

    ok, reason = native_stt_launch_supported()
    if not ok or os.environ.get("ATOM_LAUNCH_MODE") == "venv":
        print(
            "golden_path_e2e: SKIP — native STT not available for this process:",
            reason,
            "(use ATOM.app bundle launcher or unset ATOM_LAUNCH_MODE=venv)",
        )
        return 2

    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager, AtomState

    bus = AsyncEventBus()
    bus.start()
    state = StateManager(bus)
    await state.transition(AtomState.LISTENING)

    from voice.stt_macos import NativeSTT
    from core.boot.config_loader import load_config

    config = load_config()
    stt = NativeSTT(bus, state, config)

    if not stt.preload():
        print("golden_path_e2e: FAIL — STT preload:", getattr(stt, "_last_error", ""))
        return 1

    heard: list[str] = []

    async def _on_speech_final(text: str = "", **_kw: object) -> None:
        if text:
            heard.append(str(text))

    bus.on("speech_final", _on_speech_final)  # type: ignore[arg-type]

    task = asyncio.create_task(stt.async_start_listening())
    try:
        await asyncio.wait_for(
            asyncio.sleep(timeout_s),
            timeout=timeout_s + 1.0,
        )
    finally:
        stt.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if not heard:
        print(
            "golden_path_e2e: WARN — no speech_final within window "
            f"({timeout_s}s). Speak toward the mic and ensure permissions."
        )
        return 2

    print("golden_path_e2e: OK — live mic heard:", heard[-1][:120])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ATOM golden path checks")
    ap.add_argument(
        "--live-mic",
        action="store_true",
        help="Optional macOS native STT window (noisy; may SKIP)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Seconds to listen when --live-mic (default 12)",
    )
    args = ap.parse_args()

    if args.live_mic:
        return asyncio.run(_live_mic_golden(args.timeout))
    return _ci_golden()


if __name__ == "__main__":
    raise SystemExit(main())
