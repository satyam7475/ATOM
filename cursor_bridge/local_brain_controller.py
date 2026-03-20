"""
ATOM v16 -- Local Brain Controller (Offline LLM via llama.cpp).

Runs local GGUF models through brain/mini_llm.py with the standard
ATOM event bus interface (partial_response, cursor_response, metrics).

Dual-model routing:
  - 1B model: casual greetings, simple factual queries, short definitions
  - 3B model: complex reasoning, multi-step explanations, technical queries
  Complexity is determined by query length and keyword heuristics.

Key features:
  - Fake streaming: splits output into sentences and emits
    partial_response per sentence for responsive UX (feels 2x faster)
  - Warm-up: pre-loads both models at startup in background thread
  - Fully offline: zero network calls, zero API keys
  - Thread-safe: inference runs in a dedicated ThreadPoolExecutor

Event contract:
  Emits: partial_response, cursor_response, metrics_latency, metrics_event
  On error: response_ready, llm_error
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.brain_mode_manager import BrainModeManager
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

logger = logging.getLogger("atom.local_brain")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_COMPLEX_KEYWORDS = re.compile(
    r"\b(explain|how does|how do|why does|why do|compare|difference|"
    r"analyze|write.*code|implement|create.*function|debug|architecture|"
    r"what happens when|step by step|in detail|elaborate|pros? and cons?|"
    r"trade.?offs?|design|algorithm|optimize|refactor)\b", re.I)

_SIMPLE_KEYWORDS = re.compile(
    r"^(what is|what.?s|define|meaning of|who is|when was|where is|"
    r"translate|convert|spell|tell me|do you know)\b", re.I)


def _is_complex_query(text: str) -> bool:
    """Heuristic: classify query as needing the 3B model."""
    words = text.split()
    if len(words) > 20:
        return True
    if _COMPLEX_KEYWORDS.search(text):
        return True
    if len(words) <= 8 and _SIMPLE_KEYWORDS.search(text):
        return False
    if len(words) <= 6:
        return False
    return len(words) > 12


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for fake streaming."""
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text.strip())
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > 120:
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip() if current else s
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text.strip()]


class LocalBrainController:
    """Offline LLM brain controller with event bus interface.

    Wraps MiniLLM for the sole offline inference path in ATOM.
    """

    def __init__(
        self,
        bus: "AsyncEventBus",
        prompt_builder: "StructuredPromptBuilder",
        config: dict,
        brain_mode_manager: "BrainModeManager | None" = None,
    ) -> None:
        self._bus = bus
        self._prompt_builder = prompt_builder
        self._config = config

        from brain.mini_llm import MiniLLM
        self._llm = MiniLLM(config)
        if brain_mode_manager is not None:
            self._llm.set_brain_mode_manager(brain_mode_manager)

        self._total_calls = 0
        self._total_tokens_approx = 0

    @property
    def available(self) -> bool:
        """True if the local model file exists and llama_cpp is installed."""
        return self._llm.is_available()

    @property
    def is_loaded(self) -> bool:
        return self._llm.is_loaded

    def request_preempt(self) -> None:
        """Stop current local generation so new speech can take over."""
        self._llm.request_abort_preempt()

    async def warm_up(self) -> None:
        """Pre-load the model in a background thread at startup."""
        if not self._llm.is_available():
            logger.warning("Local brain not available (model missing or llama_cpp not installed)")
            return
        logger.info("Local brain: warming up (loading model)...")
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        loaded = await loop.run_in_executor(None, self._llm.preload)
        elapsed = (time.monotonic() - t0) * 1000
        if loaded:
            logger.info("Local brain ready in %.0fms", elapsed)
        else:
            logger.warning("Local brain warm-up failed")

    async def on_query(
        self,
        text: str,
        memory_context: list[str] | None = None,
        context: dict[str, str] | None = None,
        history: list[tuple[str, str]] | None = None,
        **_kw: Any,
    ) -> None:
        """Process a query through the local LLM.

        Builds a prompt using StructuredPromptBuilder (same as cloud brains),
        runs inference, then emits partial_response chunks (fake streaming)
        and cursor_response.
        """
        if not self._llm.is_available():
            self._bus.emit_long(
                "response_ready",
                text="Local brain is not available. Check that the model file exists and llama-cpp-python is installed.",
            )
            return

        prompt = self._prompt_builder.build(
            text,
            memory_summaries=memory_context,
            history=history or [],
            context=context,
        )

        from context.privacy_filter import redact as _redact

        use_1b = (self._llm.is_1b_available
                  and not _is_complex_query(text))

        model_label = "1B" if use_1b else "3B"
        logger.info("Local brain query [%s]: '%s'", model_label, _redact(text[:80]))

        t0 = time.perf_counter()
        if use_1b:
            answer, preempted = await self._llm.generate_1b(prompt)
        else:
            answer, preempted = await self._llm.generate(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if preempted:
            logger.info("Local brain preempted — skipping response (%.0fms)", elapsed_ms)
            self._bus.emit("metrics_event", counter="llm_preempted")
            return

        if not answer:
            logger.warning("Local brain returned empty response (%.0fms)", elapsed_ms)
            self._bus.emit_long(
                "response_ready",
                text="My local brain couldn't process that, Boss. Try rephrasing.",
            )
            self._bus.emit("llm_error", source="local", error="empty_response")
            self._bus.emit("metrics_event", counter="llm_errors")
            return

        self._total_calls += 1
        word_count = len(answer.split())
        self._total_tokens_approx += word_count

        self._bus.emit("metrics_latency", name="llm", ms=elapsed_ms)
        self._bus.emit("metrics_latency", name="llm_first_token", ms=elapsed_ms * 0.3)

        chunks = _split_into_sentences(answer)
        for i, chunk in enumerate(chunks):
            is_first = (i == 0)
            is_last = (i == len(chunks) - 1)
            self._bus.emit_long(
                "partial_response",
                text=chunk,
                is_first=is_first,
                is_last=is_last,
                source="local",
            )
            if not is_last:
                await asyncio.sleep(0.05)

        logger.info("Local brain: %.0fms, %d words, %d chunks",
                     elapsed_ms, word_count, len(chunks))

        self._bus.emit(
            "cursor_response",
            query=text.lower().strip(),
            response=answer,
        )

    def close(self) -> None:
        """Release the model and free memory."""
        logger.info("Local brain stats: %d calls, ~%d tokens generated",
                     self._total_calls, self._total_tokens_approx)
        self._llm.shutdown()

    def get_stats(self) -> dict:
        return {
            "available": self.available,
            "loaded": self.is_loaded,
            "total_calls": self._total_calls,
            "total_tokens_approx": self._total_tokens_approx,
        }
