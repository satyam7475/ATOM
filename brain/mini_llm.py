"""
ATOM v16 -- Offline LLM Brain (Local GGUF Inference).

Wraps llama-cpp-python. Supports dual-model routing:
  - 1B model (~750MB): fast casual queries, greetings, simple facts
  - 3B model (~1.9GB): complex reasoning, explanations, technical queries

Loads lazily; supports BrainModeManager profiles (ATOM / balanced / brain)
with hot-reload when model path or n_ctx changes.
Inference runs in a ThreadPoolExecutor; load/unload guarded by a lock.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.brain")

if TYPE_CHECKING:
    from core.brain_mode_manager import BrainModeManager

_HAS_LLAMA = True
try:
    from llama_cpp import Llama
except ImportError:
    _HAS_LLAMA = False
    Llama = None  # type: ignore[assignment,misc]


class MiniLLM:
    """Lazy-loading offline LLM wrapper using llama.cpp."""

    def __init__(self, config: dict) -> None:
        self._config = config
        brain_cfg = config.get("brain", {})
        self._model_path = str(Path(brain_cfg.get(
            "model_path",
            "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        )))
        self._model_path_1b = str(Path(brain_cfg.get(
            "model_path_1b",
            "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        )))
        self._dual_model = brain_cfg.get("dual_model", False)
        self._n_ctx = brain_cfg.get("n_ctx", 2048)
        self._n_threads = brain_cfg.get(
            "n_threads", max(2, (os.cpu_count() or 4) // 2),
        )
        self._n_gpu_layers = brain_cfg.get("n_gpu_layers", 0)
        self._max_tokens = brain_cfg.get("max_tokens", 150)
        self._temperature = brain_cfg.get("temperature", 0.4)
        self._timeout = brain_cfg.get("timeout_seconds", 20)

        self._llm: Llama | None = None
        self._llm_1b: Llama | None = None
        self._load_failed = False
        self._load_failed_1b = False
        self._loaded = False
        self._loaded_1b = False
        self._fingerprint: tuple[str, int, int] | None = None
        self._fingerprint_1b: tuple[str, int, int] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        self._load_lock = threading.RLock()
        self._brain_mode_mgr: BrainModeManager | None = None
        self._abort_generation: int = 0
        self._active_model: str = "3b"

    def set_brain_mode_manager(self, mgr: "BrainModeManager | None") -> None:
        self._brain_mode_mgr = mgr

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        if not _HAS_LLAMA:
            return False
        if self._load_failed:
            return False
        p = self._current_model_path()
        return Path(p).is_file()

    def _current_model_path(self) -> str:
        if self._brain_mode_mgr is not None:
            return self._brain_mode_mgr.effective_params()["model_path"]
        return self._model_path

    def _effective_inference(self) -> dict[str, Any]:
        if self._brain_mode_mgr is not None:
            return self._brain_mode_mgr.effective_params()
        return {
            "model_path": self._model_path,
            "n_ctx": self._n_ctx,
            "n_threads": self._n_threads,
            "n_gpu_layers": self._n_gpu_layers,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "timeout_seconds": self._timeout,
            "extra_stop_sequences": [],
            "profile": "default",
        }

    def _unload_unlocked(self) -> None:
        if self._llm is not None:
            logger.info("Unloading local LLM (profile or model change)")
            del self._llm
            self._llm = None
            self._loaded = False
            self._fingerprint = None

    def _ensure_loaded(self) -> bool:
        if not _HAS_LLAMA:
            return False
        eff = self._effective_inference()
        fp = (eff["model_path"], eff["n_ctx"], eff["n_threads"])
        model_file = Path(eff["model_path"])
        if not model_file.is_file():
            logger.warning("Model file not found: %s", eff["model_path"])
            self._load_failed = True
            return False

        with self._load_lock:
            if self._llm is not None and self._fingerprint == fp:
                return True
            self._unload_unlocked()
            if self._load_failed and not model_file.is_file():
                return False
            self._load_failed = False
            try:
                t0 = time.monotonic()
                logger.info(
                    "Loading local LLM profile=%s path=%s ctx=%d threads=%d",
                    eff.get("profile"), model_file.name, eff["n_ctx"], eff["n_threads"],
                )
                self._llm = Llama(
                    model_path=str(model_file),
                    n_ctx=eff["n_ctx"],
                    n_threads=eff["n_threads"],
                    n_gpu_layers=eff["n_gpu_layers"],
                    n_batch=256,
                    verbose=False,
                )
                self._fingerprint = fp
                self._loaded = True
                elapsed = (time.monotonic() - t0) * 1000
                logger.info("Local LLM loaded in %.0fms", elapsed)
                return True
            except Exception:
                logger.exception("Failed to load local LLM")
                self._load_failed = True
                return False

    def preload(self) -> bool:
        ok = self._ensure_loaded()
        if self._dual_model:
            self._ensure_loaded_1b()
        return ok

    def _ensure_loaded_1b(self) -> bool:
        """Load the 1B model for fast casual queries."""
        if not _HAS_LLAMA:
            return False
        model_file = Path(self._model_path_1b)
        if not model_file.is_file():
            logger.info("1B model not found at '%s' -- using 3B only", self._model_path_1b)
            self._load_failed_1b = True
            return False

        n_ctx_1b = 1024
        n_threads_1b = max(2, self._n_threads // 2)
        fp = (str(model_file), n_ctx_1b, n_threads_1b)

        with self._load_lock:
            if self._llm_1b is not None and self._fingerprint_1b == fp:
                return True
            if self._load_failed_1b:
                return False
            try:
                t0 = time.monotonic()
                logger.info("Loading 1B model path=%s ctx=%d threads=%d",
                            model_file.name, n_ctx_1b, n_threads_1b)
                self._llm_1b = Llama(
                    model_path=str(model_file),
                    n_ctx=n_ctx_1b,
                    n_threads=n_threads_1b,
                    n_gpu_layers=0,
                    n_batch=256,
                    verbose=False,
                )
                self._fingerprint_1b = fp
                self._loaded_1b = True
                elapsed = (time.monotonic() - t0) * 1000
                logger.info("1B model loaded in %.0fms", elapsed)
                return True
            except Exception:
                logger.exception("Failed to load 1B model")
                self._load_failed_1b = True
                return False

    @property
    def is_1b_available(self) -> bool:
        return self._loaded_1b and self._llm_1b is not None

    def request_abort_preempt(self) -> None:
        """Invalidate the current streaming generation (user spoke again or barge-in)."""
        self._abort_generation += 1

    def _generate_sync(self, prompt: str) -> tuple[str, bool]:
        """Returns (text, preempted). Preempted generations discard partial output."""
        eff = self._effective_inference()
        self._timeout = eff["timeout_seconds"]
        if not self._ensure_loaded() or self._llm is None:
            return "", False
        self._active_model = "3b"
        my_gen = self._abort_generation
        base_stops = ["</s>", "\n\n\n", "<|eot_id|>", "<|end|>"]
        extra = eff.get("extra_stop_sequences") or []
        stop = base_stops + [s for s in extra if s and len(s) < 80][:16]
        t0 = time.perf_counter()
        try:
            stream = self._llm(
                prompt,
                max_tokens=eff["max_tokens"],
                temperature=eff["temperature"],
                top_p=0.9,
                stop=stop,
                echo=False,
                stream=True,
            )
            parts: list[str] = []
            for chunk in stream:
                if self._abort_generation != my_gen:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.info(
                        "Local LLM [%s]: preempted after %.0fms",
                        eff.get("profile"),
                        elapsed_ms,
                    )
                    return "", True
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not choices:
                    continue
                delta = choices[0].get("text") or ""
                if delta:
                    parts.append(delta)
            text = "".join(parts).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tokens_generated = max(0, len(text.split()))
            tps = tokens_generated / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            logger.info(
                "Local LLM [%s]: %.0fms, ~%d words, %.1f tok/s",
                eff.get("profile"), elapsed_ms, tokens_generated, tps,
            )
            return text, False
        except TypeError:
            # Older llama-cpp without stream=True on __call__
            try:
                output = self._llm(
                    prompt,
                    max_tokens=eff["max_tokens"],
                    temperature=eff["temperature"],
                    top_p=0.9,
                    stop=stop,
                    echo=False,
                )
                text = (output.get("choices", [{}])[0]
                        .get("text", "").strip())
                return text, False
            except Exception:
                logger.exception("Local LLM inference error")
                return "", False
        except Exception:
            logger.exception("Local LLM inference error")
            return "", False

    def _generate_sync_1b(self, prompt: str) -> tuple[str, bool]:
        """Generate with the 1B model — faster for simple queries."""
        if self._llm_1b is None:
            return "", False
        my_gen = self._abort_generation
        self._active_model = "1b"
        stop = ["</s>", "\n\n\n", "<|eot_id|>", "<|end|>"]
        t0 = time.perf_counter()
        try:
            stream = self._llm_1b(
                prompt,
                max_tokens=60,
                temperature=0.35,
                top_p=0.9,
                stop=stop,
                echo=False,
                stream=True,
            )
            parts: list[str] = []
            for chunk in stream:
                if self._abort_generation != my_gen:
                    logger.info("1B model: preempted after %.0fms",
                                (time.perf_counter() - t0) * 1000)
                    return "", True
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not choices:
                    continue
                delta = choices[0].get("text") or ""
                if delta:
                    parts.append(delta)
            text = "".join(parts).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("1B model: %.0fms, ~%d words", elapsed_ms, len(text.split()))
            return text, False
        except Exception:
            logger.exception("1B model inference error")
            return "", False

    async def generate_1b(self, prompt: str) -> tuple[str, bool]:
        """Async wrapper for 1B model generation."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._generate_sync_1b, prompt),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("1B model timed out")
            return "", False
        except Exception:
            logger.exception("1B model generate error")
            return "", False

    async def generate(self, prompt: str) -> tuple[str, bool]:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._generate_sync, prompt),
                timeout=float(self._effective_inference()["timeout_seconds"]),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Local LLM timed out after %ss",
                self._effective_inference()["timeout_seconds"],
            )
            return "", False
        except Exception:
            logger.exception("Local LLM generate error")
            return "", False

    def shutdown(self) -> None:
        with self._load_lock:
            self._unload_unlocked()
            if self._llm_1b is not None:
                logger.info("Unloading 1B model")
                del self._llm_1b
                self._llm_1b = None
                self._loaded_1b = False
        self._executor.shutdown(wait=False)
