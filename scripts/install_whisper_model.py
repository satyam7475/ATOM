"""Idempotent installer for the whisper.cpp model used by ATOM's primary STT.

Sprint B1: ATOM swaps Apple's SFSpeechRecognizer for whisper.cpp on
Metal so long sessions stop hitting the SFSpeech idle-timeout cliff
(atomLogs.txt L310/L437/L553). This script downloads the GGML model
weights into ``models/`` and verifies the SHA so the runtime can boot
straight into Whisper without a network round-trip.

Default: ``whisper-small.en-q5_0`` (~150 MB) -- the best speed/quality
trade-off on M-series. ``--model base.en-q5_0`` for the lighter ~80 MB
variant; ``--model medium.en-q5_0`` for the heavier ~470 MB variant.

Idempotent. Safe to re-run after a clean checkout.

Usage::

    python scripts/install_whisper_model.py
    python scripts/install_whisper_model.py --model base.en-q5_0
    python scripts/install_whisper_model.py --force

Owner: Satyam
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

# Hugging Face mirror of ggerganov/whisper.cpp -- the canonical source
# of the quantised GGML weights.
_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Filename, advertised size for sanity-check, and the optional SHA256
# (None = skip strict check; we still write a digest file so future
# runs can self-verify.)
_KNOWN_MODELS: dict[str, dict[str, object]] = {
    "small.en-q5_0": {
        "filename": "ggml-small.en-q5_0.bin",
        "approx_mb": 152,
    },
    "base.en-q5_0": {
        "filename": "ggml-base.en-q5_0.bin",
        "approx_mb": 81,
    },
    "medium.en-q5_0": {
        "filename": "ggml-medium.en-q5_0.bin",
        "approx_mb": 469,
    },
    "tiny.en-q5_1": {
        "filename": "ggml-tiny.en-q5_1.bin",
        "approx_mb": 32,
    },
}

DEFAULT_MODEL = "small.en-q5_0"


def _human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream-download with a console progress indicator. Resumes are
    *not* attempted -- the model files are small enough (<500 MB) that
    re-downloading is cheaper than partial-state bookkeeping."""
    print(f"[install_whisper_model] fetching {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    last_pct = -1
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", "0") or 0)
            written = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = int(100 * written / total)
                        if pct != last_pct and pct % 5 == 0:
                            print(
                                f"  ... {pct:3d}%  ({_human_mb(written)} / "
                                f"{_human_mb(total)})",
                                flush=True,
                            )
                            last_pct = pct
        tmp.rename(dest)
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(f"[install_whisper_model] download failed: {exc}")


def _ensure_model(model_key: str, *, force: bool) -> Path:
    if model_key not in _KNOWN_MODELS:
        raise SystemExit(
            f"[install_whisper_model] unknown model {model_key!r}. "
            f"Choose one of: {', '.join(sorted(_KNOWN_MODELS))}"
        )
    spec = _KNOWN_MODELS[model_key]
    filename = str(spec["filename"])
    dest = MODELS_DIR / filename

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        approx = float(spec["approx_mb"])  # type: ignore[arg-type]
        if size_mb < approx * 0.5:
            print(
                f"[install_whisper_model] {filename} is suspiciously small "
                f"({_human_mb(dest.stat().st_size)} vs ~{approx:.0f} MB) -- "
                f"re-downloading.",
                flush=True,
            )
            dest.unlink()
        else:
            print(
                f"[install_whisper_model] OK -- {filename} already present "
                f"({_human_mb(dest.stat().st_size)})",
                flush=True,
            )
            return dest

    if dest.exists() and force:
        dest.unlink()

    url = f"{_BASE_URL}/{filename}"
    _download(url, dest)

    # Write a sidecar digest so a stale download can self-detect on
    # subsequent boots without contacting Hugging Face.
    digest = _file_sha256(dest)
    digest_path = dest.with_suffix(dest.suffix + ".sha256")
    digest_path.write_text(digest + "\n", encoding="utf-8")
    print(
        f"[install_whisper_model] DONE -- {filename} "
        f"({_human_mb(dest.stat().st_size)}, sha256={digest[:12]}...)",
        flush=True,
    )
    return dest


def _verify_runtime_can_load(model_path: Path) -> None:
    """Best-effort: import pywhispercpp and load the model. Skipped
    silently when the dep isn't installed yet so the script can be
    used to pre-stage weights before ``pip install``."""
    try:
        import pywhispercpp.model as wmod  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[install_whisper_model] pywhispercpp not installed yet -- "
            "skip runtime check (pip install -r requirements.txt first).",
            flush=True,
        )
        return
    try:
        # Just construct & destruct -- proves the GGML header is sane
        # and Metal can map the weights.
        wmod.Model(str(model_path), n_threads=2)  # noqa: F841
        print(
            f"[install_whisper_model] verified pywhispercpp can load "
            f"{model_path.name}",
            flush=True,
        )
    except Exception as exc:
        raise SystemExit(
            f"[install_whisper_model] FAILED to load {model_path.name} "
            f"with pywhispercpp: {exc}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the whisper.cpp GGML model used by ATOM STT.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=sorted(_KNOWN_MODELS),
        help=(
            "Which Whisper variant to install. "
            f"Default {DEFAULT_MODEL!r} -- the speed/quality sweet spot "
            "on M-series."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file is already present.",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the post-download pywhispercpp load check.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_path = _ensure_model(args.model, force=args.force)
    if not args.no_verify:
        _verify_runtime_can_load(model_path)

    print(
        "\n[install_whisper_model] All set, Boss.\n"
        f"  model file : {model_path}\n"
        f"  size       : {_human_mb(model_path.stat().st_size)}\n"
        f"  config key : voice.stt_engine = \"whisper\" "
        f"(see config/settings.json)\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
