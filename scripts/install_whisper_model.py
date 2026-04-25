"""Install the whisper.cpp model used by ATOM's primary STT.

This script is now a thin CLI wrapper around ``voice.whisper_install`` so
the same installer can also run from ATOM's boot path without calling
``sys.exit`` or depending on argparse.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.whisper_install import (  # noqa: E402
    DEFAULT_MODEL,
    KNOWN_MODELS,
    ensure_model,
    human_mb,
    verify_runtime_can_load,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the whisper.cpp GGML model used by ATOM STT.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(KNOWN_MODELS),
        help=(
            "Which Whisper variant to install. "
            f"Default {DEFAULT_MODEL!r} -- the speed/quality sweet spot "
            "on M-series."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file is already present.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-download pywhispercpp load check.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _progress(message: str) -> None:
        print(f"[install_whisper_model] {message}", flush=True)

    model_path = ensure_model(
        model_key=args.model,
        force=bool(args.force),
        progress_cb=_progress,
    )
    if not args.no_verify:
        try:
            verify_runtime_can_load(model_path)
            print(
                f"[install_whisper_model] verified pywhispercpp can load "
                f"{model_path.name}",
                flush=True,
            )
        except RuntimeError as exc:
            raise SystemExit(f"[install_whisper_model] FAILED: {exc}") from exc

    print(
        "\n[install_whisper_model] All set, Boss.\n"
        f"  model file : {model_path}\n"
        f"  size       : {human_mb(model_path.stat().st_size)}\n"
        f"  config key : voice.stt_engine = \"whisper\" "
        f"(see config/settings.json)\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
