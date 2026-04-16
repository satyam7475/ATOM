"""
ATOM -- MLX-native LLM wrapper for Apple Silicon.

Compatibility goals:
  - Mirrors the current MiniLLM async contract used by LocalBrainController
  - Supports streaming callbacks with preemption
  - Keeps primary + fast model roles ready for Phase 3 dual-model routing

This wrapper intentionally stays close to the existing MiniLLM interface so
the migration step can swap implementations with minimal controller changes.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger("atom.brain.mlx")

if TYPE_CHECKING:
    from core.brain_mode_manager import BrainModeManager

_HAS_MLX = True
try:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
except ImportError:
    _HAS_MLX = False
    mx = None  # type: ignore[assignment]
    load = None  # type: ignore[assignment]
    stream_generate = None  # type: ignore[assignment]
    make_sampler = None  # type: ignore[assignment]
    make_logits_processors = None  # type: ignore[assignment]


_DEFAULT_STOP_SEQUENCES: tuple[str, ...] = (
    "\nUser:",
    "\nBoss:",
    "\nAssistant:",
    "\nATOM:",
    "User:",
    "Boss:",
    "Assistant:",
    "ATOM:",
)
_LEADING_ASSISTANT_LABEL_RE = re.compile(
    r"^\s*(?:(?:ATOM|Assistant)\s*:\s*)+",
    re.I,
)
_ASSISTANT_LABEL_ONLY_RE = re.compile(
    r"^\s*(?:(?:ATOM|Assistant)\s*:\s*){2,}\s*$",
    re.I,
)
_TRAILING_ASSISTANT_LOOP_RE = re.compile(
    r"(?:\s*(?:ATOM|Assistant)\s*:\s*){2,}\s*$",
    re.I,
)
# Wrapper-only prefixes emitted by small models when the generation collapses
# into a speaker-label loop. When the visible text after trimming is ONLY one
# of these + junk, we treat it as unusable so the caller can fall back instead
# of shipping half-a-sentence + a hallucinated quoted completion.
_WRAPPER_ONLY_PREFIXES: tuple[str, ...] = (
    "the answer is",
    "my answer is",
    "the answer:",
    "answer:",
    "response:",
    "final answer:",
    "here is the answer",
    "here's the answer",
)

# Chain-of-thought prefaces that small instruction-tuned models (Qwen3-4B,
# Llama-3-8B, etc.) still leak even after system-prompt rules. We strip any
# leading run of these from the visible output before TTS so Boss never hears
# ATOM narrating its own reasoning aloud.
#
# The pattern peels off one preface sentence at a time; we keep applying it
# until the output starts with an answer token. The final ". " / "? " boundary
# is included in the match so the following real sentence stays clean.
_COT_PREFACE_RE = re.compile(
    r"""
    ^\s*                                  # leading whitespace
    (?:
        # \"Okay(,)? let's/lets see\"  /  \"let me think\"  /  \"alright so\"
        (?:okay|ok|alright|well|so|hmm+|um+|uh+)\b[,.!]?\s*
        (?:let(?:'|\u2019)?s?\s+(?:see|think|break|try|start|go)\b[^.?!]*[.?!]\s*)?
      |
        let(?:'|\u2019)?s?\s+(?:see|think|break|try|start|go)\b[^.?!]*[.?!]\s*
      |
        let\s+me\s+think\b[^.?!]*[.?!]\s*
      |
        # Third-person narration about the user.
        (?:the\s+user|boss|the\s+speaker)\s+(?:is\s+)?(?:asking|wants|says|said|needs|wondering)[^.?!]*[.?!]\s*
      |
        # Meta narration \"The question is ...\" / \"So, the question is ...\"
        (?:so\s+)?(?:the\s+)?(?:question|query|request)\s+is\b[^.?!]*[.?!]\s*
      |
        # \"I should / I need to ...\" internal-monologue stems.
        i\s+(?:should|need\s+to|have\s+to|must)\s+(?:think|consider|figure|reason|recall)\b[^.?!]*[.?!]\s*
      |
        # Fill / stall particles at the very front.
        (?:hmm+|um+|uh+|er+|ah+)[,.!]?\s+
    )+
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _strip_cot_prefaces(text: str) -> str:
    """Remove chain-of-thought / stall preface sentences from the head of a
    reply. Idempotent and safe on empty strings — returns the trimmed tail
    which is the actual spoken answer.
    """
    if not text:
        return text
    prev = None
    out = text
    # Loop until fixed point (each run peels at most one preface sentence
    # thanks to the outer `+`, so two passes are usually enough).
    for _ in range(3):
        if out == prev:
            break
        prev = out
        out = _COT_PREFACE_RE.sub("", out, count=1).lstrip()
    return out


def _looks_like_wrapper_preface(text: str) -> bool:
    """Return True when *text* begins with a preface like 'The answer is ...'
    AND the content after the wrapper looks like a stalled-model artefact
    (short remainder OR a quoted one-liner). Examples::

        "The answer is \\"Okay, I'll play the song for you.\\""   -> True
        "The answer is 42."                                    -> True
        "The answer is Newton's first law: every object..."     -> False

    The second form (short remainder, no quotes) catches raw collapses;
    the first form (any remainder, but wrapped in quotes) catches the
    classic Qwen-small hallucination regardless of inner length because
    quoting is itself a symptom of the model pretending to answer.
    Empty input returns False — caller already handles emptiness.
    """
    stripped = (text or "").strip().lower()
    if not stripped:
        return False
    for prefix in _WRAPPER_ONLY_PREFIXES:
        if stripped.startswith(prefix):
            remainder = stripped[len(prefix):].strip(" :.,;")
            # Quoted inner text — always treat as wrapper hallucination.
            for lq, rq in (('"', '"'), ("'", "'"),
                           ("\u201c", "\u201d"), ("\u2018", "\u2019"),
                           ("`", "`")):
                if remainder.startswith(lq) and rq in remainder[1:]:
                    return True
            # Unquoted but short — stall pattern.
            if len(remainder.split()) <= 6:
                return True
    return False


class MLXBrain:
    """Lazy-loading MLX wrapper with MiniLLM-compatible behavior."""

    _VALID_ROLES = frozenset({"primary", "fast"})

    def __init__(self, config: dict) -> None:
        self._config = config
        brain_cfg = config.get("brain", {})

        self._primary_path = str(
            Path(brain_cfg.get("mlx_primary_model", "models/qwen3-4b-mlx")).expanduser(),
        )
        self._fast_path = str(
            Path(brain_cfg.get("mlx_fast_model", "models/qwen3-1.7b-mlx")).expanduser(),
        )
        default_role = str(brain_cfg.get("mlx_default_role", "primary")).strip().lower()
        self._active_role = default_role if default_role in self._VALID_ROLES else "primary"

        self._max_tokens = int(brain_cfg.get("max_tokens", 512))
        self._temperature = float(brain_cfg.get("temperature", 0.7))
        self._top_p = float(brain_cfg.get("top_p", 0.9))
        self._timeout = float(brain_cfg.get("timeout_seconds", 30))

        self._models: dict[str, Any | None] = {"primary": None, "fast": None}
        self._tokenizers: dict[str, Any | None] = {"primary": None, "fast": None}
        self._fingerprints: dict[str, str | None] = {"primary": None, "fast": None}
        self._loaded_roles: dict[str, bool] = {"primary": False, "fast": False}
        self._load_failed: dict[str, bool] = {"primary": False, "fast": False}

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._load_lock = threading.RLock()
        self._brain_mode_mgr: BrainModeManager | None = None
        self._abort_generation = 0
        self._gen_lock = threading.Lock()
        self._streaming_depth = 0

    def set_brain_mode_manager(self, mgr: "BrainModeManager | None") -> None:
        self._brain_mode_mgr = mgr

    @property
    def is_loaded(self) -> bool:
        return any(self._loaded_roles.values())

    @property
    def active_role(self) -> str:
        return self._active_role

    def set_model_role(self, role: str) -> None:
        key = self._normalize_role(role)
        if key != self._active_role:
            logger.info("MLX role switch: %s -> %s", self._active_role, key)
            self._active_role = key

    def is_available(self) -> bool:
        if not _HAS_MLX:
            return False
        return Path(self._primary_path).is_dir() or Path(self._fast_path).is_dir()

    def _normalize_role(self, role: str | None) -> str:
        key = (role or self._active_role or "primary").strip().lower()
        if key in self._VALID_ROLES:
            return key
        return "primary"

    def _path_for_role(self, role: str) -> str:
        key = self._normalize_role(role)
        if key == "fast":
            return self._fast_path
        return self._primary_path

    def _effective_inference(self, model_role: str | None = None) -> dict[str, Any]:
        role = self._normalize_role(model_role)
        eff = self._brain_mode_mgr.effective_params() if self._brain_mode_mgr is not None else {}
        return {
            "profile": eff.get("profile", "default"),
            "model_role": role,
            "model_path": self._path_for_role(role),
            "max_tokens": int(eff.get("max_tokens", self._max_tokens)),
            "temperature": float(eff.get("temperature", self._temperature)),
            "top_p": float(eff.get("top_p", self._top_p)),
            "repeat_penalty": float(eff.get("repeat_penalty", 1.1)),
            "timeout_seconds": float(eff.get("timeout_seconds", self._timeout)),
            "extra_stop_sequences": [
                str(s).strip()
                for s in eff.get("extra_stop_sequences", [])
                if str(s).strip()
            ][:16],
        }

    def _unload_role_unlocked(self, role: str) -> None:
        key = self._normalize_role(role)
        if self._models[key] is not None:
            logger.info("Unloading MLX model role=%s", key)
        self._models[key] = None
        self._tokenizers[key] = None
        self._fingerprints[key] = None
        self._loaded_roles[key] = False

    def _clear_mlx_cache(self) -> None:
        if not _HAS_MLX or mx is None:
            return
        try:
            mx.clear_cache()
        except Exception:
            logger.debug("MLX cache clear failed", exc_info=True)

    def _ensure_loaded(self, model_role: str | None = None) -> bool:
        if not _HAS_MLX or load is None:
            return False

        eff = self._effective_inference(model_role)
        role = eff["model_role"]
        model_path = Path(eff["model_path"])
        if not model_path.is_dir():
            logger.warning("MLX model directory not found for role=%s: %s", role, model_path)
            self._load_failed[role] = True
            return False

        with self._load_lock:
            if self._loaded_roles[role] and self._fingerprints[role] == str(model_path):
                return True

            # When primary and fast point to the same directory, reuse one load (RAM + stability).
            other = "fast" if role == "primary" else "primary"
            other_path = Path(self._path_for_role(other))
            if (
                other_path.resolve() == model_path.resolve()
                and self._loaded_roles[other]
                and self._fingerprints[other] == str(model_path)
                and self._models[other] is not None
            ):
                self._models[role] = self._models[other]
                self._tokenizers[role] = self._tokenizers[other]
                self._fingerprints[role] = str(model_path)
                self._loaded_roles[role] = True
                self._load_failed[role] = False
                logger.info(
                    "MLX: sharing loaded weights for role=%s with %s (%s)",
                    role, other, model_path.name,
                )
                return True

            self._load_failed[role] = False
            self._unload_role_unlocked(role)
            try:
                t0 = time.monotonic()
                logger.info(
                    "Loading MLX model role=%s profile=%s path=%s",
                    role,
                    eff["profile"],
                    model_path.name,
                )
                model, tokenizer = load(str(model_path))
                self._models[role] = model
                self._tokenizers[role] = tokenizer
                self._fingerprints[role] = str(model_path)
                self._loaded_roles[role] = True
                elapsed = (time.monotonic() - t0) * 1000
                logger.info("MLX model role=%s loaded in %.0fms", role, elapsed)
                return True
            except Exception:
                logger.exception("Failed to load MLX model role=%s", role)
                self._load_failed[role] = True
                self._unload_role_unlocked(role)
                self._clear_mlx_cache()
                return False

    def preload(self, *, model_role: str | None = None, load_all: bool = False) -> bool:
        if load_all:
            ok = True
            for role in ("primary", "fast"):
                ok = self._ensure_loaded(role) and ok
            return ok
        return self._ensure_loaded(model_role)

    def request_abort_preempt(self) -> None:
        """Invalidate the current streaming generation."""
        self._abort_generation += 1

    def is_generating(self) -> bool:
        """True while MLX inference is active inside the worker thread."""
        with self._gen_lock:
            return self._streaming_depth > 0

    def save_kv_cache(self, system_prompt_hash: int) -> None:
        """Compatibility no-op: MLX wrapper does not persist KV cache yet."""
        del system_prompt_hash

    def restore_kv_cache(self, system_prompt_hash: int) -> bool:
        """Compatibility no-op: MLX wrapper does not persist KV cache yet."""
        del system_prompt_hash
        return False

    def _invalidate_kv_cache(self) -> None:
        """Compatibility no-op to match MiniLLM surface."""
        return

    def _make_sampler(self, temperature: float, top_p: float):
        temp = max(0.0, float(temperature))
        nucleus = max(0.0, min(1.0, float(top_p)))
        if make_sampler is None:
            return None
        return make_sampler(temp=temp, top_p=nucleus)

    def _make_logits_processors(self, repeat_penalty: float):
        penalty = float(repeat_penalty or 1.0)
        if make_logits_processors is None or penalty <= 1.0:
            return None
        try:
            return make_logits_processors(
                repetition_penalty=penalty,
                repetition_context_size=48,
            )
        except Exception:
            logger.debug("MLX logits processor setup failed", exc_info=True)
            return None

    @staticmethod
    def _stop_sequences(extra: list[str] | None = None) -> tuple[str, ...]:
        merged: list[str] = list(_DEFAULT_STOP_SEQUENCES)
        for seq in extra or []:
            candidate = str(seq or "").strip()
            if candidate and candidate not in merged:
                merged.append(candidate)
        return tuple(sorted(merged, key=len, reverse=True))

    @staticmethod
    def _find_stop_hit(text: str, stop_sequences: tuple[str, ...]) -> tuple[int, str] | None:
        best_idx = -1
        best_seq = ""
        for seq in stop_sequences:
            idx = text.find(seq)
            if idx == -1:
                continue
            if best_idx == -1 or idx < best_idx or (idx == best_idx and len(seq) > len(best_seq)):
                best_idx = idx
                best_seq = seq
        if best_idx == -1:
            return None
        return best_idx, best_seq

    @staticmethod
    def _partial_stop_suffix_len(text: str, stop_sequences: tuple[str, ...]) -> int:
        best = 0
        for seq in stop_sequences:
            limit = min(len(seq) - 1, len(text))
            for prefix_len in range(limit, 0, -1):
                if text.endswith(seq[:prefix_len]):
                    best = max(best, prefix_len)
                    break
        return best

    @classmethod
    def _guard_visible_text(
        cls,
        text: str,
        stop_sequences: tuple[str, ...],
    ) -> tuple[str, str | None, bool]:
        if not text:
            return "", None, False

        if _ASSISTANT_LABEL_ONLY_RE.fullmatch(text):
            return "", "speaker_label_loop", True

        guarded = _LEADING_ASSISTANT_LABEL_RE.sub("", text)
        if not guarded:
            return "", None, False

        # Peel off any chain-of-thought / meta prefaces leaked by the model
        # BEFORE the speaker-loop / stop-sequence checks. This way a reply
        # that starts with "Okay, let's see. Newton's first law is …" gets
        # trimmed to the useful sentence instead of being shipped whole or
        # declared unusable.
        stripped_cot = _strip_cot_prefaces(guarded)
        if stripped_cot != guarded:
            guarded = stripped_cot
        if not guarded:
            # Entire response was narration. Force caller fallback.
            return "", "cot_only", True

        trimmed = _TRAILING_ASSISTANT_LOOP_RE.sub("", guarded).rstrip()
        if trimmed != guarded:
            # A speaker-label loop terminated generation early. If the pre-loop
            # buffer is only a wrapper preface like `The answer is "..."` with
            # almost no real content, treat it as unusable — small models
            # produce exactly this pattern when they stall, and emitting it
            # causes "ATOM invented an action" hallucinations downstream.
            if _looks_like_wrapper_preface(trimmed):
                return "", "speaker_label_loop_wrapper", True
            return trimmed, "speaker_label_loop", True

        stop_hit = cls._find_stop_hit(guarded, stop_sequences)
        if stop_hit is not None:
            idx, seq = stop_hit
            return guarded[:idx].rstrip(), f"stop_sequence:{seq}", True

        partial_len = cls._partial_stop_suffix_len(guarded, stop_sequences)
        if partial_len > 0:
            return guarded[:-partial_len], None, False

        return guarded, None, False

    def _generate_sync_streaming(
        self,
        prompt: str,
        on_token: Callable[[str, bool], None] | None = None,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
    ) -> tuple[str, bool]:
        eff = self._effective_inference(model_role)
        role = eff["model_role"]
        if not self._ensure_loaded(role):
            return "", False

        model = self._models[role]
        tokenizer = self._tokenizers[role]
        if model is None or tokenizer is None or stream_generate is None:
            return "", False

        with self._gen_lock:
            self._streaming_depth += 1
        try:
            return self._generate_sync_streaming_inner(
                role, eff, model, tokenizer, prompt, on_token,
                max_tokens_override=max_tokens_override,
            )
        finally:
            with self._gen_lock:
                self._streaming_depth -= 1

    def _generate_sync_streaming_inner(
        self,
        role: str,
        eff: dict[str, Any],
        model: Any,
        tokenizer: Any,
        prompt: str,
        on_token: Callable[[str, bool], None] | None = None,
        *,
        max_tokens_override: int | None = None,
    ) -> tuple[str, bool]:
        """Core stream loop (wrapped for active-generation accounting)."""
        my_gen = self._abort_generation
        sampler = self._make_sampler(eff["temperature"], eff["top_p"])
        logits_processors = self._make_logits_processors(eff["repeat_penalty"])
        stop_sequences = self._stop_sequences(eff["extra_stop_sequences"])

        visible_text = ""
        raw_text = ""
        last_resp: Any = None
        t0 = time.perf_counter()
        stop_reason: str | None = None

        if _HAS_MLX and mx is not None:
            try:
                mx.reset_peak_memory()
            except Exception:
                logger.debug('MLX peak memory reset failed', exc_info=True)

        try:
            for resp in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=int(max_tokens_override or eff["max_tokens"]),
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                if self._abort_generation != my_gen:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.info(
                        "MLX [%s/%s]: preempted after %.0fms",
                        eff["profile"],
                        role,
                        elapsed_ms,
                    )
                    if on_token:
                        on_token("", True)
                    return "", True

                last_resp = resp
                segment = getattr(resp, "text", "") or ""
                if segment:
                    raw_text += segment
                    next_visible, reason, should_stop = self._guard_visible_text(
                        raw_text,
                        stop_sequences,
                    )
                    if len(next_visible) > len(visible_text):
                        delta = next_visible[len(visible_text):]
                        visible_text = next_visible
                        if delta and on_token:
                            on_token(delta, False)
                    if should_stop:
                        stop_reason = reason or "guard"
                        break

            if on_token:
                on_token("", True)

            text = visible_text.strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            generation_tokens = int(
                getattr(last_resp, "generation_tokens", max(1, len(raw_text.split()))),
            )
            generation_tps = float(getattr(last_resp, "generation_tps", 0.0))
            peak_memory = float(getattr(last_resp, "peak_memory", 0.0))
            logger.info(
                "MLX [%s/%s]: %.0fms, %d tokens, ~%d words, %.1f tok/s, peak %.2fGB",
                eff["profile"],
                role,
                elapsed_ms,
                generation_tokens,
                len(text.split()),
                generation_tps,
                peak_memory,
            )
            if stop_reason:
                logger.info("MLX [%s/%s]: stopped early on %s", eff["profile"], role, stop_reason)
            return text, False
        except Exception:
            logger.exception("MLX inference error role=%s", role)
            if on_token:
                try:
                    on_token("", True)
                except Exception:
                    logger.debug('Stream end callback failed', exc_info=True)
            return "", False

    def _generate_sync(
        self,
        prompt: str,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
    ) -> tuple[str, bool]:
        return self._generate_sync_streaming(
            prompt,
            on_token=None,
            model_role=model_role,
            max_tokens_override=max_tokens_override,
        )

    async def generate(
        self,
        prompt: str,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
    ) -> tuple[str, bool]:
        loop = asyncio.get_running_loop()
        timeout_s = float(self._effective_inference(model_role)["timeout_seconds"])
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    partial(
                        self._generate_sync,
                        prompt,
                        model_role=model_role,
                        max_tokens_override=max_tokens_override,
                    ),
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            self.request_abort_preempt()
            logger.warning("MLX generation timed out after %.1fs", timeout_s)
            return "", False
        except Exception:
            logger.exception("MLX generate error")
            return "", False

    async def generate_streaming(
        self,
        prompt: str,
        on_token: Callable[[str, bool], None] | None = None,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
    ) -> tuple[str, bool]:
        loop = asyncio.get_running_loop()
        timeout_s = float(self._effective_inference(model_role)["timeout_seconds"])
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    partial(
                        self._generate_sync_streaming,
                        prompt,
                        on_token,
                        model_role=model_role,
                        max_tokens_override=max_tokens_override,
                    ),
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            self.request_abort_preempt()
            logger.warning("MLX streaming timed out after %.1fs", timeout_s)
            return "", False
        except Exception:
            logger.exception("MLX streaming generate error")
            return "", False

    def shutdown(self) -> None:
        with self._load_lock:
            self._unload_role_unlocked("primary")
            self._unload_role_unlocked("fast")
        self._clear_mlx_cache()

    def close(self) -> None:
        self.shutdown()
        self._executor.shutdown(wait=False)


__all__ = ["MLXBrain"]
