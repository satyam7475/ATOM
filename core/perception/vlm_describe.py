"""ATOM -- VLM captioner (SmolVLM-Instruct-4bit via mlx-vlm).

Purpose
-------
Add a *one-sentence* caption to any frame the camera captured. This is
the "always-watching" step that separates a speak-and-hear assistant
from a see-speak-and-hear one. Deliberately tiny surface area:

* :class:`VLMCaptioner` — lazy-loads the configured VLM the first time
  ``describe(jpeg_path)`` is called. Everything after that is a hot
  inference path.
* Fails open. If ``mlx-vlm`` isn't installed, the model directory is
  missing, or inference errors, ``describe`` returns ``""`` and logs
  the reason once. ATOM continues to boot and run exactly as it did
  before the module existed — the only regression vs. a working VLM
  is an empty caption.
* Thread-safe. The AVCapture path can call us from whichever worker
  the event bus picked, and the ``vision_describe`` tool handler runs
  under the router's executor. A single process-wide lock serialises
  model calls so we never hand two prompts to mlx-vlm in flight.

Why SmolVLM-Instruct-4bit
-------------------------
* ~1.2 GB on disk, ~1.4 GB resident after load — comfortably fits
  alongside Qwen3-4B-Instruct-2507-4bit (~2.4 GB) on a 16 GB Apple
  Silicon machine with multiple GB of slack for the KV cache,
  Cursor, and Chrome (the post-v3.3 brain swap freed ~1.9 GB of
  unified memory at idle).
* Idefics3 architecture, officially supported by mlx-vlm. First-token
  latency ~150-400 ms on M-series for a 512x512 input — short enough
  to fire on wake-word without the user noticing a stall.
* moondream2 has no first-class mlx-vlm build today (the only "MLX"
  moondream is moondream3-preview at ~7 GB, which would push 16 GB
  unified memory too hard alongside Qwen). SmolVLM is the smallest
  proven-good replacement; the captioner is model-agnostic and will
  pick up moondream2 the moment an mlx-vlm port lands.

Failure modes we deliberately tolerate
--------------------------------------
* ``mlx-vlm`` not installed → lazy load returns ``None``, describe
  returns ``""``, warning logged once.
* ``model_path`` missing on disk → same behaviour. The user gets a
  single-line instruction in the log telling them the ``huggingface-
  cli`` command they'd run to fetch the model.
* An individual inference errors → described as ``""``; error logged
  at ``debug`` the first N times, then suppressed. We never let a
  VLM failure take down the vision engine or the voice loop.

Owner: Satyam
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.perception.vlm")


_DEFAULT_PROMPT = "Describe this image in one short sentence."
_DEFAULT_MAX_TOKENS = 48
_DEFAULT_TEMPERATURE = 0.0
_LOAD_WARNING_LIMIT = 1
_INFERENCE_WARNING_LIMIT = 3


class VLMCaptioner:
    """Lazy-loaded VLM captioner. Fails open if unavailable.

    ``model_path`` is the *primary* source — usually a local directory
    holding the mlx-vlm-converted weights. ``model_repo`` is an optional
    Hugging Face repo id used as a fallback when ``model_path`` is
    missing on disk: we hand it directly to ``mlx_vlm.load`` which will
    fetch + cache via huggingface_hub. This keeps ATOM offline-by-
    default while still allowing a one-shot bootstrap.
    """

    __slots__ = (
        "_model_path",
        "_model_repo",
        "_prompt",
        "_max_tokens",
        "_temperature",
        "_load_lock",
        "_inference_lock",
        "_model",
        "_processor",
        "_config",
        "_load_failed",
        "_load_error",
        "_load_warn_count",
        "_inference_error_count",
        # ── public-facing metrics (read by status snapshot / tests) ──
        "_load_ms",
        "_inference_count",
        "_last_inference_ms",
        "_last_inference_at",
    )

    def __init__(
        self,
        *,
        model_path: str | Path,
        model_repo: str | None = None,
        prompt: str = _DEFAULT_PROMPT,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self._model_path = str(model_path)
        self._model_repo = (model_repo or "").strip() or None
        self._prompt = prompt or _DEFAULT_PROMPT
        self._max_tokens = max(4, int(max_tokens))
        self._temperature = max(0.0, float(temperature))
        self._load_lock = threading.Lock()
        # ``mlx-vlm`` is not re-entrant on the same model; one inference
        # at a time process-wide. Outside of pathological tool loops
        # this is never the critical path.
        self._inference_lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        # Loaded alongside the model; used by mlx-vlm's apply_chat_template
        # to insert the right <image> placeholder for each architecture.
        self._config: Any = None
        self._load_failed = False
        self._load_error: str = ""
        self._load_warn_count = 0
        self._inference_error_count = 0
        # Metrics. ``_load_ms`` is set once when ``_load`` succeeds;
        # the inference counters tick on every ``describe`` call so the
        # status snapshot can report cold-load + steady-state cost
        # without needing a separate observability hook.
        self._load_ms: float = 0.0
        self._inference_count: int = 0
        self._last_inference_ms: float = 0.0
        self._last_inference_at: float = 0.0

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def is_available(self) -> bool:
        """True if a call to ``describe`` is worth attempting."""
        if self._load_failed:
            return False
        if self.is_loaded:
            return True
        if Path(self._model_path).exists():
            return True
        return self._model_repo is not None

    def disabled_reason(self) -> str:
        """Short string explaining why the captioner is offline, empty
        when ready.

        Called by the boot banner and by the router when the VLM tool
        is invoked and the captioner is idle.
        """
        if self._load_failed and self._load_error:
            return self._load_error
        if not Path(self._model_path).exists() and self._model_repo is None:
            return (
                f"VLM weights not found at {self._model_path} and no "
                f"model_repo configured. Fetch with: hf download "
                f"mlx-community/SmolVLM-Instruct-4bit --local-dir "
                f"{self._model_path}"
            )
        return ""

    def _warn_load_once(self, message: str) -> None:
        if self._load_warn_count < _LOAD_WARNING_LIMIT:
            logger.warning("VLMCaptioner: %s", message)
        else:
            logger.debug("VLMCaptioner: %s", message)
        self._load_warn_count += 1

    @staticmethod
    def _manual_idefics3_load(load_source: str | Path) -> tuple[object | None, object | None]:
        """Defensive fallback when ``mlx_vlm.load`` blows up on the
        AutoProcessor path (e.g. transformers' Idefics3 image-processor
        auto-route fails because torchvision is missing or the
        preprocessor_config.json is shaped unexpectedly).

        Returns (model, processor) on success or (None, None) on failure.
        Best-effort only: never raises — caller treats None as load
        failure and falls back through its own degraded path.
        """
        try:
            from mlx_vlm.utils import load_model as _mlx_load_model  # type: ignore
            from transformers import (
                Idefics3ImageProcessor,
                Idefics3Processor,
                AutoTokenizer,
            )
            img = Idefics3ImageProcessor.from_pretrained(str(load_source))
            tok = AutoTokenizer.from_pretrained(str(load_source))
            processor = Idefics3Processor(image_processor=img, tokenizer=tok)
            model = _mlx_load_model(str(load_source))
            return model, processor
        except Exception as exc:
            logger.warning(
                "VLMCaptioner manual Idefics3 fallback failed: %s", exc,
            )
            return None, None

    def _load(self) -> bool:
        if self.is_loaded:
            return True
        if self._load_failed:
            return False

        with self._load_lock:
            if self.is_loaded:
                return True
            if self._load_failed:
                return False

            path = Path(self._model_path)
            # Pick the source mlx-vlm should load from. Local dir wins
            # (offline-by-default); fall back to the HF repo id only if
            # the local dir is missing AND the operator opted in.
            if path.exists():
                load_source = self._model_path
                source_kind = "local"
            elif self._model_repo:
                load_source = self._model_repo
                source_kind = "hf-repo"
                logger.info(
                    "VLMCaptioner: %s missing locally; will fetch %s "
                    "via mlx-vlm + huggingface_hub on first use",
                    self._model_path, self._model_repo,
                )
            else:
                self._load_failed = True
                self._load_error = (
                    f"model_path does not exist and no model_repo set: "
                    f"{self._model_path}"
                )
                self._warn_load_once(self._load_error)
                return False

            t0 = time.perf_counter()
            try:
                from mlx_vlm import load as mlx_vlm_load  # type: ignore
            except ImportError:
                self._load_failed = True
                self._load_error = (
                    "mlx-vlm is not installed; add mlx-vlm to "
                    "requirements.txt and pip install"
                )
                self._warn_load_once(self._load_error)
                return False
            except Exception as exc:
                self._load_failed = True
                self._load_error = f"mlx-vlm import failed: {exc}"
                self._warn_load_once(self._load_error)
                return False

            try:
                model, processor = mlx_vlm_load(load_source)
            except Exception as exc:
                # mlx-vlm 0.4.x sometimes routes SmolVLM through an
                # AutoProcessor path that fails when transformers can't
                # find an Idefics3 image processor (typically when
                # torchvision is missing). Fall back to a manual load
                # so vision stays available even on a thin venv.
                logger.warning(
                    "VLMCaptioner: mlx_vlm.load failed for %s (%s); "
                    "attempting manual Idefics3 fallback",
                    load_source, exc,
                )
                model, processor = self._manual_idefics3_load(load_source)
                if model is None or processor is None:
                    self._load_failed = True
                    self._load_error = (
                        f"mlx_vlm.load failed for {load_source}: {exc}"
                    )
                    logger.warning(
                        "VLMCaptioner load failed for %s: %s",
                        load_source, exc,
                    )
                    return False

            # ``load_config`` is what ``apply_chat_template`` consults
            # to figure out which architecture-specific template to use
            # (Idefics3 wants a different <image> placeholder than
            # Qwen2-VL etc.). Failing here is non-fatal — we fall back
            # to the raw prompt and most processors will still work.
            try:
                from mlx_vlm.utils import load_config as _mlx_vlm_load_config  # type: ignore
                self._config = _mlx_vlm_load_config(load_source)
            except Exception as exc:
                logger.debug(
                    "VLMCaptioner: load_config soft-failed (%s); "
                    "falling back to raw prompt", exc,
                )
                self._config = None

            self._model = model
            self._processor = processor
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._load_ms = dt_ms
            logger.info(
                "VLMCaptioner ready: %s via %s (%.0fms load)",
                load_source, source_kind, dt_ms,
            )
            return True

    def describe(
        self,
        jpeg_path: str | Path,
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run one caption pass. Always returns; never raises.

        Returns an empty string when the captioner is unavailable or
        inference fails. Callers should treat ``""`` as "no visual
        context available" rather than "no visual content".
        """
        path = Path(jpeg_path)
        if not path.exists():
            logger.debug("VLMCaptioner.describe: jpeg_path missing %s", path)
            return ""

        if not self._load():
            return ""

        model = self._model
        processor = self._processor
        if model is None or processor is None:
            return ""

        effective_prompt = prompt or self._prompt
        effective_max_tokens = (
            self._max_tokens if max_tokens is None else max(4, int(max_tokens))
        )

        with self._inference_lock:
            t0 = time.perf_counter()
            try:
                caption = _mlx_vlm_generate_caption(
                    model=model,
                    processor=processor,
                    config=self._config,
                    image_path=str(path),
                    prompt=effective_prompt,
                    max_tokens=effective_max_tokens,
                    temperature=self._temperature,
                )
            except Exception as exc:
                caption = ""
                if self._inference_error_count < _INFERENCE_WARNING_LIMIT:
                    logger.warning(
                        "VLMCaptioner inference failed (%d/%d): %s",
                        self._inference_error_count + 1,
                        _INFERENCE_WARNING_LIMIT, exc,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "VLMCaptioner inference failed again: %s", exc,
                    )
                self._inference_error_count += 1

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._inference_count += 1
            self._last_inference_ms = dt_ms
            self._last_inference_at = time.monotonic()

        caption = _tidy_caption(caption)
        if caption:
            logger.debug(
                "VLMCaptioner: caption=%r (%.0fms)", caption[:80], dt_ms,
            )
        return caption

    def metrics(self) -> dict[str, Any]:
        """Snapshot of captioner runtime state.

        Plain-dict shape so it can be serialised straight into the
        status JSON / web dashboard / log lines without a separate
        adapter. All fields are always present (zero-valued when the
        corresponding event hasn't happened yet) so downstream
        consumers don't need defensive ``get(...)`` calls.
        """
        avg_ms = (
            (self._last_inference_ms if self._inference_count else 0.0)
        )
        return {
            "model_path": self._model_path,
            "model_repo": self._model_repo or "",
            "is_loaded": self.is_loaded,
            "is_available": self.is_available,
            "load_failed": self._load_failed,
            "load_error": self._load_error,
            "load_ms": round(self._load_ms, 1),
            "inference_count": self._inference_count,
            "inference_failed_count": self._inference_error_count,
            "last_inference_ms": round(self._last_inference_ms, 1),
            "last_inference_age_s": (
                round(time.monotonic() - self._last_inference_at, 1)
                if self._last_inference_at > 0 else 0.0
            ),
            "avg_inference_ms": round(avg_ms, 1),
        }


def _mlx_vlm_generate_caption(
    *,
    model: Any,
    processor: Any,
    config: Any,
    image_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Thin adapter around mlx_vlm.generate.

    Kept as a module-level function (not a method) so tests can
    monkey-patch it without instantiating the heavy model stack.

    Uses ``apply_chat_template`` when available so architecture-specific
    ``<image>`` placeholders are inserted correctly (Idefics3 / SmolVLM
    needs them; Qwen2-VL has its own format; etc.). Falls back to the
    raw prompt on any template error.
    """
    try:
        from mlx_vlm import generate as mlx_vlm_generate  # type: ignore
    except ImportError:
        return ""

    formatted_prompt = prompt
    if config is not None:
        try:
            from mlx_vlm.prompt_utils import (  # type: ignore
                apply_chat_template as _mlx_vlm_apply_template,
            )
            formatted_prompt = _mlx_vlm_apply_template(
                processor, config, prompt, num_images=1,
            )
        except Exception:
            # Any template error is silent — caller still gets a
            # caption attempt with the raw prompt. mlx-vlm itself
            # will raise a clearer "missing <image> token" if the
            # raw prompt also fails, which we surface upstream.
            formatted_prompt = prompt

    # mlx-vlm generate signatures have shifted across versions; try the
    # most-recent shape first, then fall back. ``temperature`` moved
    # under ``sampler`` in 0.4.x but the kwarg is still tolerated as a
    # passthrough on older builds — keeping both attempts means we work
    # against any 0.1+ install without pinning.
    def _call(image_arg: Any) -> Any:
        try:
            return mlx_vlm_generate(
                model,
                processor,
                prompt=formatted_prompt,
                image=image_arg,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
        except TypeError:
            return mlx_vlm_generate(
                model, processor, formatted_prompt, image_arg,
                max_tokens=max_tokens, temperature=temperature,
                verbose=False,
            )

    try:
        result = _call([image_path])
    except Exception:
        # Some processors expect a single string, others a list — try
        # the other shape before giving up to the outer fail-open.
        result = _call(image_path)

    if isinstance(result, str):
        return result
    text_attr = getattr(result, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        if isinstance(first, str):
            return first
        text_attr = getattr(first, "text", None)
        if isinstance(text_attr, str):
            return text_attr
    return ""


def _tidy_caption(raw: str) -> str:
    """Clean VLM output into a single spoken-friendly sentence.

    Small VLMs (moondream2, SmolVLM, etc.) occasionally prefix output
    with prompt tokens or with ``"Caption:"`` / ``"Answer:"``. We strip
    those, collapse whitespace, and cap length so the result reads well
    when echoed to TTS or shown in a log line.
    """
    if not raw:
        return ""
    text = raw.strip()
    for marker in ("Caption:", "Answer:", "Description:", "Assistant:"):
        if text.lower().startswith(marker.lower()):
            text = text[len(marker):].strip()
    text = " ".join(text.split())
    if len(text) > 280:
        text = text[:277].rstrip() + "..."
    return text


__all__ = ["VLMCaptioner"]
