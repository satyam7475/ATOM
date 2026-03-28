"""
ATOM -- Offline LLM Brain (GPU-Accelerated Single-Model Inference).

Wraps llama-cpp-python with a single GGUF model fully offloaded to GPU.

Recommended model: Qwen3-8B-Q4_K_M (~5GB, 32K context, native tool calling)

Architecture:
  - n_gpu_layers=-1 (full GPU offload) -- inference 1-4s
  - n_batch=512 (GPU handles larger batches efficiently)
  - n_ctx=8192 (or up to 32768 for Qwen3) -- deep conversation memory
  - KV cache persistence for system prompt (saves ~200-400ms per query)
  - True token streaming callback support for real-time sentence emission
  - BrainModeManager profiles (ATOM / balanced / brain) with hot-reload

Inference runs in a ThreadPoolExecutor; load/unload guarded by a lock.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
    """Lazy-loading offline LLM wrapper using llama.cpp with full GPU offload."""

    def __init__(self, config: dict) -> None:
        self._config = config
        brain_cfg = config.get("brain", {})
        self._model_path = str(Path(brain_cfg.get(
            "model_path",
            "models/qwen3-8b-q4_k_m.gguf",
        )))
        self._n_ctx = brain_cfg.get("n_ctx", 8192)
        self._n_threads = brain_cfg.get(
            "n_threads", max(2, (os.cpu_count() or 4) // 2),
        )
        self._n_gpu_layers = brain_cfg.get("n_gpu_layers", -1)
        self._n_batch = brain_cfg.get("n_batch", 512)
        self._max_tokens = brain_cfg.get("max_tokens", 512)
        self._temperature = brain_cfg.get("temperature", 0.7)
        self._top_p = brain_cfg.get("top_p", 0.9)
        self._repeat_penalty = brain_cfg.get("repeat_penalty", 1.1)
        self._timeout = brain_cfg.get("timeout_seconds", 30)

        self._llm: Llama | None = None
        self._load_failed = False
        self._loaded = False
        self._fingerprint: tuple[str, int, int] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        self._load_lock = threading.RLock()
        self._brain_mode_mgr: BrainModeManager | None = None
        self._abort_generation: int = 0

        self._kv_cache_state: Any | None = None
        self._kv_cache_prompt_hash: int | None = None

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
            "n_batch": self._n_batch,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "repeat_penalty": self._repeat_penalty,
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
            self._invalidate_kv_cache()

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
                gpu_layers = eff.get("n_gpu_layers", -1)
                batch_size = eff.get("n_batch", 512)
                logger.info(
                    "Loading LLM profile=%s path=%s ctx=%d threads=%d gpu_layers=%s batch=%d",
                    eff.get("profile"), model_file.name, eff["n_ctx"],
                    eff["n_threads"], gpu_layers, batch_size,
                )
                self._llm = Llama(
                    model_path=str(model_file),
                    n_ctx=eff["n_ctx"],
                    n_threads=eff["n_threads"],
                    n_gpu_layers=gpu_layers,
                    n_batch=batch_size,
                    verbose=False,
                )
                self._fingerprint = fp
                self._loaded = True
                elapsed = (time.monotonic() - t0) * 1000
                logger.info("LLM loaded in %.0fms (GPU layers=%s)", elapsed, gpu_layers)
                return True
            except Exception:
                logger.exception("Failed to load LLM")
                self._load_failed = True
                return False

    def preload(self) -> bool:
        return self._ensure_loaded()

    def request_abort_preempt(self) -> None:
        """Invalidate the current streaming generation (user spoke again or barge-in)."""
        self._abort_generation += 1

    # ── KV Cache Persistence ──────────────────────────────────────────

    def save_kv_cache(self, system_prompt_hash: int) -> None:
        """Save the KV cache state after processing the system prompt.

        On subsequent queries, we can restore this state and skip
        re-processing the system prompt tokens (~200-400ms savings).
        """
        if self._llm is None:
            return
        try:
            self._kv_cache_state = self._llm.save_state()
            self._kv_cache_prompt_hash = system_prompt_hash
            logger.debug("KV cache saved (prompt_hash=%d)", system_prompt_hash)
        except Exception:
            logger.debug("KV cache save failed", exc_info=True)
            self._kv_cache_state = None
            self._kv_cache_prompt_hash = None

    def restore_kv_cache(self, system_prompt_hash: int) -> bool:
        """Restore cached KV state if the system prompt hash matches.

        Returns True if cache was restored, False if a full re-process is needed.
        """
        if (
            self._llm is None
            or self._kv_cache_state is None
            or self._kv_cache_prompt_hash != system_prompt_hash
        ):
            return False
        try:
            self._llm.load_state(self._kv_cache_state)
            logger.debug("KV cache restored (prompt_hash=%d)", system_prompt_hash)
            return True
        except Exception:
            logger.debug("KV cache restore failed", exc_info=True)
            self._invalidate_kv_cache()
            return False

    def _invalidate_kv_cache(self) -> None:
        """Discard cached KV state (model change, prompt change, etc.)."""
        self._kv_cache_state = None
        self._kv_cache_prompt_hash = None

    # ── Streaming Inference ───────────────────────────────────────────

    def _generate_sync_streaming(
        self,
        prompt: str,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> tuple[str, bool]:
        """Generate with true token-by-token streaming.

        Args:
            prompt: The full prompt string.
            on_token: Callback fired for each token. Signature: (token_text, is_done).
                      Called from the inference thread. The caller is responsible for
                      thread-safe emission to the event bus.

        Returns:
            (full_text, preempted) — preempted is True if generation was aborted.
        """
        eff = self._effective_inference()
        self._timeout = eff["timeout_seconds"]
        if not self._ensure_loaded() or self._llm is None:
            return "", False

        my_gen = self._abort_generation
        base_stops = ["</s>", "\n\n\n", "<|eot_id|>", "<|end|>",
                       "<|im_end|>", "<|endoftext|>"]
        extra = eff.get("extra_stop_sequences") or []
        stop = base_stops + [s for s in extra if s and len(s) < 80][:16]

        t0 = time.perf_counter()
        try:
            stream = self._llm(
                prompt,
                max_tokens=eff["max_tokens"],
                temperature=eff["temperature"],
                top_p=eff.get("top_p", 0.9),
                repeat_penalty=eff.get("repeat_penalty", 1.1),
                stop=stop,
                echo=False,
                stream=True,
            )
            parts: list[str] = []
            for chunk in stream:
                if self._abort_generation != my_gen:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.info("LLM [%s]: preempted after %.0fms",
                                eff.get("profile"), elapsed_ms)
                    if on_token:
                        on_token("", True)
                    return "", True

                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not choices:
                    continue
                delta = choices[0].get("text") or ""
                if delta:
                    parts.append(delta)
                    if on_token:
                        on_token(delta, False)

            if on_token:
                on_token("", True)

            text = "".join(parts).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tokens_generated = len(parts)
            tps = tokens_generated / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            logger.info(
                "LLM [%s]: %.0fms, ~%d tokens, ~%d words, %.1f tok/s",
                eff.get("profile"), elapsed_ms, tokens_generated,
                len(text.split()), tps,
            )
            return text, False

        except TypeError:
            try:
                output = self._llm(
                    prompt,
                    max_tokens=eff["max_tokens"],
                    temperature=eff["temperature"],
                    top_p=eff.get("top_p", 0.9),
                    repeat_penalty=eff.get("repeat_penalty", 1.1),
                    stop=stop,
                    echo=False,
                )
                text = (output.get("choices", [{}])[0]
                        .get("text", "").strip())
                if on_token and text:
                    on_token(text, True)
                return text, False
            except Exception:
                logger.exception("LLM inference error (non-streaming fallback)")
                return "", False
        except Exception:
            logger.exception("LLM inference error")
            return "", False

    def _generate_sync(self, prompt: str) -> tuple[str, bool]:
        """Non-streaming generation (backward compat). Collects all tokens then returns."""
        return self._generate_sync_streaming(prompt, on_token=None)

    async def generate(self, prompt: str) -> tuple[str, bool]:
        """Async generation without streaming callback."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._generate_sync, prompt),
                timeout=float(self._effective_inference()["timeout_seconds"]),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM timed out after %ss",
                           self._effective_inference()["timeout_seconds"])
            return "", False
        except Exception:
            logger.exception("LLM generate error")
            return "", False

    async def generate_streaming(
        self,
        prompt: str,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> tuple[str, bool]:
        """Async generation with true token streaming callback.

        The on_token callback fires from the executor thread for each token.
        Use a thread-safe mechanism (e.g., queue) to bridge to the async event bus.
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._generate_sync_streaming,
                    prompt,
                    on_token,
                ),
                timeout=float(self._effective_inference()["timeout_seconds"]),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM streaming timed out after %ss",
                           self._effective_inference()["timeout_seconds"])
            return "", False
        except Exception:
            logger.exception("LLM streaming generate error")
            return "", False

    def shutdown(self) -> None:
        with self._load_lock:
            self._unload_unlocked()
        self._executor.shutdown(wait=False)
