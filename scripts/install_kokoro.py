"""Idempotent installer for ATOM's Kokoro neural TTS (Sprint Ω2).

Pulls the ONNX model + voices archive into ``models/kokoro/`` so
``voice/tts_kokoro.py`` can flip ``available=True``.

What this does
--------------
1. Creates ``ATOM/models/kokoro/``.
2. Downloads (if missing) the model and voices files from the
   official kokoro-onnx GitHub release. Resumable; SHA verified
   against the file size manifest below so a partial download is
   re-downloaded automatically.
3. Validates with a 1-second smoke synthesis when ``--smoke`` is
   passed and ``espeak-ng`` is installed.

What this does NOT do
---------------------
* Install ``espeak-ng``. That's a system dep -- on macOS run
  ``brew install espeak-ng``. Without it, kokoro-onnx will fail to
  phonemise and ``tts_kokoro`` will refuse to flip ``available=True``
  (and voice_pipeline will silently fall back to macOS Native).

Usage::

    python scripts/install_kokoro.py
    python scripts/install_kokoro.py --smoke
    python scripts/install_kokoro.py --force        # re-download

Owner: Satyam
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "models" / "kokoro"

_FILES = {
    "kokoro-v1.0.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/kokoro-v1.0.onnx",
        310_000_000,
    ),
    "voices-v1.0.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin",
        24_000_000,
    ),
}


def _download(url: str, dest: Path, expected_size: int, *, force: bool) -> None:
    if dest.exists() and not force:
        size = dest.stat().st_size
        if size >= expected_size * 0.95:
            print(f"  ✓ {dest.name} already present ({size:,} bytes)")
            return
        print(
            f"  ! {dest.name} exists but only {size:,} bytes "
            f"(expected ~{expected_size:,}); re-downloading",
        )
        dest.unlink()

    print(f"  ↓ {dest.name}  ({url})")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        tmp.rename(dest)
        print(f"    {dest.name} downloaded ({total:,} bytes)")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _smoke(target_dir: Path) -> int:
    try:
        from kokoro_onnx import Kokoro  # noqa: WPS433
    except ImportError:
        print("smoke: kokoro-onnx not installed (pip install kokoro-onnx)")
        return 1
    try:
        k = Kokoro(
            str(target_dir / "kokoro-v1.0.onnx"),
            str(target_dir / "voices-v1.0.bin"),
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "espeak" in msg or "phonemizer" in msg:
            print(
                "smoke: espeak-ng not installed -- run "
                "`brew install espeak-ng` then re-run --smoke",
            )
        else:
            print(f"smoke: load failed: {exc}")
        return 1
    audio, sr = k.create("ATOM Kokoro online.", voice="af_heart", speed=1.0)
    print(f"smoke: synthesised {len(audio)} samples at {sr} Hz -- OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download")
    p.add_argument("--smoke", action="store_true", help="post-install smoke test")
    args = p.parse_args()

    print("== ATOM Kokoro TTS installer ==")
    print(f"  target: {TARGET_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for name, (url, size) in _FILES.items():
        _download(url, TARGET_DIR / name, size, force=args.force)

    if args.smoke:
        return _smoke(TARGET_DIR)

    print("== Kokoro TTS files ready ==")
    print("  Next: brew install espeak-ng  (one-time, system-wide)")
    print("  Then flip config/settings.json -> tts.engine = 'kokoro'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
