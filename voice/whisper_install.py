"""Runtime-safe installer for ATOM's whisper.cpp STT model.

Unlike ``scripts/install_whisper_model.py``, this module is safe to call
from the app boot path: it raises normal exceptions, supports progress
callbacks, and never calls ``sys.exit``. Sprint K uses it to block the
first launch until the whisper.cpp model is present instead of falling
back to Apple's unstable SFSpeechRecognizer.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger("atom.whisper_install")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# Apr 25 2026 hot-fix: ggerganov/whisper.cpp dropped every q5_0 quant
# of the *.en family from the HF repo (404), so the previous default
# (small.en-q5_0) can no longer be downloaded. We're switching to the
# q5_1 variant of the same model -- same architecture, same disk
# footprint, identical pywhispercpp load path -- and aliasing the old
# key so any pinned config keeps working.
DEFAULT_MODEL = "small.en-q5_1"

KNOWN_MODELS: dict[str, dict[str, object]] = {
    "small.en-q5_1": {
        "filename": "ggml-small.en-q5_1.bin",
        "approx_mb": 181,
    },
    "small.en-q8_0": {
        "filename": "ggml-small.en-q8_0.bin",
        "approx_mb": 264,
    },
    "small.en": {
        "filename": "ggml-small.en.bin",
        "approx_mb": 466,
    },
    "base.en-q5_1": {
        "filename": "ggml-base.en-q5_1.bin",
        "approx_mb": 57,
    },
    "base.en-q8_0": {
        "filename": "ggml-base.en-q8_0.bin",
        "approx_mb": 81,
    },
    "base.en": {
        "filename": "ggml-base.en.bin",
        "approx_mb": 142,
    },
    "medium.en-q5_0": {
        "filename": "ggml-medium.en-q5_0.bin",
        "approx_mb": 514,
    },
    "tiny.en-q5_1": {
        "filename": "ggml-tiny.en-q5_1.bin",
        "approx_mb": 32,
    },
    "tiny.en": {
        "filename": "ggml-tiny.en.bin",
        "approx_mb": 75,
    },
}

# Legacy → current redirects so callers / configs that still reference
# the removed q5_0 file names automatically use the q5_1 replacement
# without surprise.
_DEPRECATED_KEY_REDIRECTS: dict[str, str] = {
    "small.en-q5_0": "small.en-q5_1",
    "base.en-q5_0": "base.en-q5_1",
}
_DEPRECATED_FILENAME_REDIRECTS: dict[str, str] = {
    "ggml-small.en-q5_0.bin": "ggml-small.en-q5_1.bin",
    "ggml-base.en-q5_0.bin": "ggml-base.en-q5_1.bin",
}

ProgressCallback = Callable[[str], None]


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_key_for_path(path: Path) -> str:
    """Return the known model key that matches ``path.name``.

    Falls back to ``DEFAULT_MODEL`` when the path is custom/empty; callers
    can still pass ``model_key`` explicitly to override this. Filenames
    pointing at removed upstream quants are auto-redirected so old
    config files keep booting.
    """
    filename = path.name
    if filename in _DEPRECATED_FILENAME_REDIRECTS:
        new_name = _DEPRECATED_FILENAME_REDIRECTS[filename]
        for key, spec in KNOWN_MODELS.items():
            if spec.get("filename") == new_name:
                logger.warning(
                    "whisper model %s no longer exists upstream; "
                    "redirecting to %s",
                    filename,
                    new_name,
                )
                return key
    for key, spec in KNOWN_MODELS.items():
        if spec.get("filename") == filename:
            return key
    return DEFAULT_MODEL


def _emit(progress_cb: ProgressCallback | None, message: str) -> None:
    if progress_cb is not None:
        try:
            progress_cb(message)
        except Exception:
            logger.debug("whisper install progress callback failed", exc_info=True)
    logger.info("%s", message)


def _download(url: str, dest: Path, *, progress_cb: ProgressCallback | None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    last_pct = -1
    _emit(progress_cb, f"Downloading whisper.cpp speech model: {dest.name}")
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
                        if pct != last_pct and pct % 10 == 0:
                            _emit(
                                progress_cb,
                                (
                                    f"Whisper model download {pct}% "
                                    f"({human_mb(written)} / {human_mb(total)})"
                                ),
                            )
                            last_pct = pct
        tmp.rename(dest)
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"whisper model download failed: {exc}") from exc


def ensure_model(
    *,
    model_path: Path | str | None = None,
    model_key: str | None = None,
    force: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Ensure the configured GGML model exists locally and return its path.

    This is idempotent. A suspiciously tiny existing file is treated as a
    failed partial download and replaced. The function never exits the
    process; callers decide whether to fail boot or fall back.
    """
    key = model_key or DEFAULT_MODEL
    if model_path is not None:
        path = Path(model_path)
        if path.name:
            key = model_key or model_key_for_path(path)
    if key in _DEPRECATED_KEY_REDIRECTS:
        new_key = _DEPRECATED_KEY_REDIRECTS[key]
        logger.warning(
            "whisper model key %r is deprecated (removed upstream); "
            "using %r instead",
            key,
            new_key,
        )
        key = new_key
    if key not in KNOWN_MODELS:
        raise ValueError(
            f"unknown whisper model {key!r}; choose one of: "
            f"{', '.join(sorted(KNOWN_MODELS))}",
        )

    spec = KNOWN_MODELS[key]
    filename = str(spec["filename"])
    if model_path is not None:
        # Honour an explicit caller path, but if the basename points at a
        # removed quant rewrite it to the live filename so we don't try
        # to download a 404.
        explicit_path = Path(model_path).expanduser()
        if explicit_path.name in _DEPRECATED_FILENAME_REDIRECTS:
            explicit_path = explicit_path.with_name(filename)
        dest = explicit_path
    else:
        dest = MODELS_DIR / filename
    if not dest.is_absolute():
        dest = ROOT / "models" / dest.name
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    approx = float(spec["approx_mb"])  # type: ignore[arg-type]
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        if size_mb >= approx * 0.5:
            _emit(
                progress_cb,
                f"Whisper model already present: {dest.name} ({human_mb(dest.stat().st_size)})",
            )
            return dest
        _emit(
            progress_cb,
            (
                f"Whisper model file is incomplete ({human_mb(dest.stat().st_size)}); "
                "re-downloading."
            ),
        )
        dest.unlink()

    if dest.exists() and force:
        dest.unlink()

    _download(f"{BASE_URL}/{filename}", dest, progress_cb=progress_cb)
    digest = file_sha256(dest)
    dest.with_suffix(dest.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    _emit(
        progress_cb,
        f"Whisper model ready: {dest.name} ({human_mb(dest.stat().st_size)}, sha256={digest[:12]}...)",
    )
    return dest


def verify_runtime_can_load(model_path: Path, *, strict: bool = True) -> None:
    """Verify ``pywhispercpp`` can actually load the model.

    Apr 25 2026: this used to silently no-op when ``pywhispercpp`` was
    missing, which masked an empty venv -- ATOM booted, claimed STT was
    "verified", then crashed at runtime. ``strict=True`` (the default,
    used by the CLI installer) now raises so we can never ship a model
    file without the binding to load it. ``strict=False`` keeps the old
    soft behaviour for callers that explicitly tolerate a missing
    binding (e.g. dev tooling on non-mac platforms).
    """
    try:
        import pywhispercpp.model as wmod  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = (
            "pywhispercpp not installed -- run "
            "`pip install pywhispercpp>=1.2.0` (Apple Silicon ships a "
            "Metal-built wheel)."
        )
        if strict:
            raise RuntimeError(msg) from exc
        logger.warning("%s (continuing because strict=False)", msg)
        return
    try:
        wmod.Model(str(model_path), n_threads=2)  # noqa: F841
    except Exception as exc:
        raise RuntimeError(
            f"pywhispercpp failed to load {model_path.name}: {exc}",
        ) from exc
