"""
ATOM -- MLX-native LLM wrapper for Apple Silicon.

Compatibility goals:
  - Mirrors the current MiniLLM async contract used by LocalBrainController
  - Supports streaming callbacks with preemption
  - Supports either shared single-model or split-role MLX routing

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

# Sprint P2.6 (Apr 26 2026): defer the MLX import until the first
# ``_ensure_loaded`` call. The eager ``import mlx.core`` paid ~600 ms on
# every Python startup -- including subagents and unit tests that never
# touch the brain. We replace it with a cheap ``find_spec`` probe so
# ``is_available()`` stays honest, and only the actual load path pays
# the real import cost. Once paid, subsequent inference calls hit the
# fast path because the imports populate module-level globals.
import importlib.util as _importlib_util  # cheap stdlib probe

_HAS_MLX: bool = bool(
    _importlib_util.find_spec("mlx") is not None
    and _importlib_util.find_spec("mlx_lm") is not None,
)
mx: Any = None
load: Any = None
stream_generate: Any = None
make_sampler: Any = None
make_logits_processors: Any = None

# Prompt-prefix KV cache. The LRU cache stores full (prompt+response) KV
# states keyed by the token list; on the next turn it finds the longest
# common prefix and hands back a trimmed cache + the remaining tokens, so
# MLX only runs forward on the delta. For ATOM this lets us skip
# reprocessing the ~500-token system identity block on every single turn,
# which accounts for ~30-40% of first-token latency.
_HAS_PROMPT_CACHE: bool = False
_LRUPromptCache: Any = None
_make_prompt_cache: Any = None
_trim_prompt_cache: Any = None
_can_trim_prompt_cache: Any = None
_save_prompt_cache: Any = None
_load_prompt_cache: Any = None

# Set once the heavy imports actually run. Idempotent so concurrent
# `_ensure_loaded` calls converge.
_MLX_IMPORTED: bool = False
_MLX_IMPORT_LOCK = threading.Lock()


def _lazy_import_mlx() -> bool:
    """Heavy MLX imports, deferred to first model-load.

    Populates the module-level globals (``mx``, ``load``,
    ``stream_generate``, ``make_sampler``, ``make_logits_processors``,
    plus the prompt-cache symbols) so the rest of the file can keep
    referencing them as before. Returns ``False`` if MLX is unavailable
    or the import fails -- callers must respect that and fall back.
    """
    global _MLX_IMPORTED, _HAS_MLX, mx, load, stream_generate
    global make_sampler, make_logits_processors
    global _LRUPromptCache, _make_prompt_cache, _trim_prompt_cache
    global _can_trim_prompt_cache, _save_prompt_cache, _load_prompt_cache
    global _HAS_PROMPT_CACHE
    if _MLX_IMPORTED:
        return _HAS_MLX
    if not _HAS_MLX:
        _MLX_IMPORTED = True
        return False
    with _MLX_IMPORT_LOCK:
        if _MLX_IMPORTED:
            return _HAS_MLX
        try:
            t0 = time.perf_counter()
            import mlx.core as _mx
            from mlx_lm import load as _load, stream_generate as _sg
            from mlx_lm.sample_utils import (
                make_logits_processors as _mlp,
                make_sampler as _ms,
            )
            mx = _mx
            load = _load
            stream_generate = _sg
            make_sampler = _ms
            make_logits_processors = _mlp
            try:
                from mlx_lm.models.cache import (  # type: ignore[import-not-found]
                    LRUPromptCache as _LRU,
                    can_trim_prompt_cache as _can_trim,
                    make_prompt_cache as _mpc,
                    trim_prompt_cache as _tpc,
                )
                _LRUPromptCache = _LRU
                _make_prompt_cache = _mpc
                _trim_prompt_cache = _tpc
                _can_trim_prompt_cache = _can_trim
                _HAS_PROMPT_CACHE = True
            except Exception:
                _HAS_PROMPT_CACHE = False
            try:
                from mlx_lm.models.cache import (  # type: ignore[import-not-found]
                    save_prompt_cache as _spc,
                    load_prompt_cache as _lpc,
                )
                _save_prompt_cache = _spc
                _load_prompt_cache = _lpc
            except Exception:
                _save_prompt_cache = None
                _load_prompt_cache = None
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            try:
                import platform
                logger.info(
                    "MLX deferred-import done (mlx=%s default_device=%s "
                    "macOS=%s prompt_cache=%s) in %.0f ms",
                    getattr(__import__("mlx"), "__version__", "?"),
                    getattr(mx, "default_device", lambda: "?")(),
                    platform.mac_ver()[0] or "n/a",
                    "yes" if _HAS_PROMPT_CACHE else "no",
                    elapsed_ms,
                )
            except Exception:
                logger.debug(
                    "MLX deferred-import boot log raised", exc_info=True,
                )
            _HAS_MLX = True
        except Exception:
            logger.exception("MLX deferred-import failed")
            _HAS_MLX = False
        finally:
            _MLX_IMPORTED = True
    return _HAS_MLX


# Sprint C4: extra stop sequences applied ONLY when the caller is the
# FAST/QUICK voice path. We hard-stop on a leading parenthesis so any
# stage-direction leak (atomLogs.txt L301 ``'(in a.'``) is killed at
# the token layer -- not after the speech sanitiser has tried to scrub
# it. ``\n\n`` keeps a FAST reply to a single paragraph.
_FAST_PATH_STOP_SEQUENCES: tuple[str, ...] = (
    "\n\n",
)

_DEFAULT_STOP_SEQUENCES: tuple[str, ...] = (
    "\nUser:",
    "\nBoss:",
    "\nAssistant:",
    "\nATOM:",
    "User:",
    "Boss:",
    "Assistant:",
    "ATOM:",
    # Special / control tokens leaked when the model collapses out of the
    # ATOM persona. Treating them as hard stop sequences forces generation
    # to halt before any of the post-leak text reaches TTS.
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|eot_id|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_0|>",
    "Human:",
    "\nHuman:",
)
_LEADING_ASSISTANT_LABEL_RE = re.compile(
    r"^\s*(?:(?:ATOM|Assistant)\s*:\s*)+",
    re.I,
)
# Special / control tokens anywhere in the output. The stop-sequence list
# halts new generation, but partial token boundaries can still let a few
# characters of these slip in before the stop fires; strip them defensively.
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:endoftext|im_(?:end|start)|user|assistant|system|eot_id|"
    r"begin_of_text|end_of_text|start_header_id|end_header_id|"
    r"reserved_special_token_\d+)\|>",
    re.I,
)
# Leading quote-wrapped roleplay openers. Small instruction-tuned models
# often break from the system rule and emit things like:
#     "Boss, I'm showing you your active goals. Here they are:"
# Those quotes leak into TTS verbatim, get heard by the mic, and — because
# they start with "Boss," — the model treats the echo as a fresh user turn.
# Strip the leading quote (both straight and curly) plus any matching
# trailing quote so TTS speaks the sentence naturally.
#
# We also catch a *lone* leading quote that the controller's earlier passes
# left half-stripped (e.g. ``" haven't set an alarm`` or ``"'m sorry``). Those
# fragments were responsible for production leaks where TTS spoke the literal
# quote character followed by a contraction.
_LEADING_QUOTE_WRAP_RE = re.compile(
    r"""^\s*[\"\u201c\u201d\u2018\u2019]+\s*"""
    r"""(?="""
    r"""(?:Boss|Satyam|Sir|Ma'am|Madam|Hey|OK|Okay|Alright)\b"""
    r"""|[A-Z]"""               # Any normal sentence start
    r"""|'[a-z]"""              # Lone apostrophe + lowercase: "'m sorry`
    r"""|[a-z]"""               # Lowercase letter:           " haven't`
    r""")""",
    re.U,
)
# Trailing unclosed quote (with optional trailing punctuation/space).
_TRAILING_UNCLOSED_QUOTE_RE = re.compile(r"""[\"\u201c\u201d\u2018\u2019]+\s*$""")
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
    # Optional leading quoted user-text + dash/colon separator. Catches the
    # 'Dear Boss' — the user is greeting you, so respond politely... pattern
    # that small instruction-tuned models still leak even with strict V-rules.
    (?:
        ["'\u201c\u2018`]
        [^"'\u201c\u201d\u2018\u2019`\n]{1,80}
        ["'\u201d\u2019`]
        \s*[\u2013\u2014:,\-]+\s*
    )?
    (?:
        # \"Okay(,)? let's/lets see\"  /  \"let me think\"  /  \"alright so\"
        (?:okay|ok|alright|well|so|hmm+|um+|uh+)\b[,.!]?\s*
        (?:let(?:'|\u2019)?s?\s+(?:see|think|break|try|start|go|check|verify|look|reason|figure)\b[^.?!]*[.?!]\s*)?
      |
        let(?:'|\u2019)?s?\s+(?:see|think|break|try|start|go|check|verify|look|reason|figure)\b[^.?!]*[.?!]\s*
      |
        # \"Let me check my memory.\" / \"Let me check that.\" / \"Let me think about that.\"
        let\s+me\s+(?:think|see|try|consider|check|verify|look|reason|figure)\b[^.?!]*[.?!]\s*
      |
        # Third-person narration about the user (greeting/asking/wanting...).
        (?:the\s+user|boss|the\s+speaker)\s+(?:is\s+|has\s+|was\s+)?
        (?:greeting|asking|wants|says|said|needs|wondering|requesting|trying|having|expressing)
        [^.?!]*[.?!]\s*
      |
        # Meta narration \"The question is ...\" / \"So, the question is ...\"
        (?:so\s+)?(?:the\s+)?(?:question|query|request|issue|problem)\s+is\b[^.?!]*[.?!]\s*
      |
        # Direct-instruction echoes seen in production ("so respond politely...")
        so\s+respond\s+(?:politely|warmly|briefly|kindly|directly|gently|empathetically)[^.?!]*[.?!]\s*
      |
        # \"Keep it concise and friendly.\" / \"Keep the answer brief.\"
        keep\s+(?:it|the\s+(?:answer|response|reply|tone))\s+(?:concise|brief|short|simple|friendly|warm|professional|polite|casual|natural)
        [^.?!]*[.?!]\s*
      |
        # \"My role is to respond as ATOM ...\" / \"My job is to ...\"
        my\s+(?:role|job|task|goal)\s+(?:is|here\s+is)\s+(?:to\s+)?[^.?!]*[.?!]\s*
      |
        # \"In the current context, there's no mention of an alarm being set.\"
        # \"In my memory, ...\"  / \"From the conversation history, ...\"
        (?:in|from|within)\s+
        (?:the\s+)?
        (?:current\s+)?
        (?:context|conversation(?:\s+history)?|memory|chat\s+history|transcript|history)
        [,]?\s+[^.?!]*[.?!]\s*
      |
        # \"I need to acknowledge their difficulty, show empathy, and offer help.\"
        # \"I should think about ...\" — generic internal-monologue stem.
        i\s+(?:should|need\s+to|have\s+to|must|will|am\s+going\s+to)\s+
        (?:think|consider|figure|reason|recall|acknowledge|show|offer|respond|respond\s+with|focus|make\s+sure|remember|note|check|verify|look|process|prepare)
        \b[^.?!]*[.?!]\s*
      |
        # Mid-stream stall fragment: \"Okay, let me process this.\"
        (?:okay|ok|alright)[,.!]?\s+let\s+me\s+process\b[^.?!]*[.?!]\s*
      |
        # Fill / stall particles at the very front.
        (?:hmm+|um+|uh+|er+|ah+)[,.!]?\s+
    )+
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


# Stage-direction parenthetical sanitiser. Definition lives in the
# shared brain._speech_sanitizer module so the streaming TTS path,
# the batch LLM path and the LocalBrainController hot-text path all
# use the same regex. Sprint A3 unified the three duplicate copies.
from brain._speech_sanitizer import (  # noqa: E402  -- intentional rebinding
    strip_stage_direction_leak as _strip_stage_direction_leak,
)


def _strip_cot_prefaces(text: str) -> str:
    """Remove chain-of-thought / stall preface sentences from the head of a
    reply. Idempotent and safe on empty strings — returns the trimmed tail
    which is the actual spoken answer.
    """
    if not text:
        return text
    out = _strip_stage_direction_leak(text)
    prev = None
    # Loop until fixed point (each run peels at most one preface sentence
    # thanks to the outer `+`, so two passes are usually enough).
    for _ in range(3):
        if out == prev:
            break
        prev = out
        out = _COT_PREFACE_RE.sub("", out, count=1).lstrip()
        out = _strip_stage_direction_leak(out)
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
    """Single-model MLX wrapper with MiniLLM-compatible behavior.

    ATOM runs ONE local MLX model (Qwen3-4B-Instruct-2507-4bit by default).
    The cognitive kernel still tags each ``QueryPlan`` with a role label
    (``primary`` | ``fast``) for logs + telemetry so we can see which
    routing path picked the brain; both labels resolve to the same
    weights here, so the state dicts are keyed by role purely for
    observability -- not for separate loads.
    """

    # Role labels emitted by the kernel. Kept as a tuple (not frozenset)
    # to make the enumeration order explicit for the per-role state dicts.
    _ROLES: tuple[str, ...] = ("primary", "fast")

    _DEFAULT_MODEL_PATH = "models/qwen3-4b-instruct-4bit"

    @classmethod
    def _resolve_model_path(cls, brain_cfg: dict) -> str:
        """Resolve the *primary* MLX model path with back-compat for pre-v3.2 keys.

        Preferred key: ``brain.mlx_model``. For older settings.json
        files we also accept ``mlx_primary_model`` / ``mlx_fast_model`` /
        ``model_path`` (in that order) so an upgrade doesn't crash on a
        stale config.

        The "fast" role resolves separately via :py:meth:`_resolve_fast_model_path`
        so dual-tier (small fast + large primary) is configurable; see
        Sprint P3.1 (Apr 26 2026).
        """
        for key in ("mlx_model", "mlx_primary_model", "mlx_fast_model", "model_path"):
            val = brain_cfg.get(key)
            if val:
                return str(Path(str(val)).expanduser())
        return cls._DEFAULT_MODEL_PATH

    @classmethod
    def _resolve_fast_model_path(
        cls, brain_cfg: dict, primary_path: str,
    ) -> str:
        """Resolve the optional "fast" role's model path (Sprint P3.1).

        If ``brain.mlx_fast_model`` is set AND that directory exists on
        disk, dual-tier mode is unblocked: the "fast" role loads the
        smaller model while "primary" loads the full one. Otherwise we
        fall back to the primary path so a single-model profile keeps
        working unchanged.

        We deliberately do **not** auto-pick ``mlx_model_fallback`` for
        "fast"; that key is a *cold-boot fallback* (used when the
        primary model directory is missing), not a dual-tier hint. Mixing
        the two would silently double RAM on the operator the moment
        they install the smaller model. Dual-tier must be opt-in.
        """
        val = brain_cfg.get("mlx_fast_model")
        if not val:
            return primary_path
        path = Path(str(val)).expanduser()
        if not path.is_dir():
            logger.info(
                "MLX dual-tier: fast model path %s not on disk; "
                "falling back to primary for role=fast.",
                path,
            )
            return primary_path
        return str(path)

    def __init__(self, config: dict) -> None:
        self._config = config
        brain_cfg = config.get("brain", {})

        self._model_path = self._resolve_model_path(brain_cfg)
        # Sprint P3.1 (Apr 26 2026): role-aware model paths. The "fast"
        # role can be pointed at a separate, smaller model directory via
        # ``brain.mlx_fast_model`` so the kernel actually runs dual-tier
        # (e.g. Qwen3-4B for fast intent, Qwen3-8B for full reasoning)
        # instead of aliasing both roles to the same weights. When the
        # fast model directory is missing or unset, we fall back to the
        # primary path -- callers see no behavior change.
        fast_model_path = self._resolve_fast_model_path(
            brain_cfg, self._model_path,
        )
        self._role_model_paths: dict[str, str] = {
            "primary": self._model_path,
            "fast": fast_model_path,
        }
        self._dual_tier_enabled: bool = (fast_model_path != self._model_path)
        self._active_role = "primary"

        self._max_tokens = int(brain_cfg.get("max_tokens", 512))
        self._temperature = float(brain_cfg.get("temperature", 0.7))
        self._top_p = float(brain_cfg.get("top_p", 0.9))
        self._timeout = float(brain_cfg.get("timeout_seconds", 30))

        # Sprint P3.2 (Apr 26 2026): Speculative decoding (4B drafts 8B).
        # mlx-lm's stream_generate accepts `draft_model=...` and the kwarg
        # `num_draft_tokens` for the verification window. With Qwen3-4B
        # drafting Qwen3-8B we expect 1.5-2x tokens/s on warm runs. Off
        # by default; enable with `brain.speculative_decoding.enabled=true`.
        spec_cfg = brain_cfg.get("speculative_decoding", {}) or {}
        self._speculative_enabled: bool = bool(spec_cfg.get("enabled", False))
        self._speculative_draft_path: str | None = spec_cfg.get("draft_model_path")
        self._speculative_num_draft_tokens: int = int(
            spec_cfg.get("num_draft_tokens", 3),
        )
        self._draft_model: Any = None
        self._draft_tokenizer: Any = None
        self._draft_loaded: bool = False
        self._draft_load_failed: bool = False

        # Sprint P3.5 (Apr 26 2026): mx.compile on the hot sampler path.
        # The token sampler runs once per generated token, so a compiled
        # version saves ~10-25% steady-state time on M-series with
        # macOS 26.2+. Off if the user is on older macOS or wants to
        # compare. Compiled samplers are cached per (temp, top_p) pair.
        self._mx_compile_enabled: bool = bool(
            brain_cfg.get("mx_compile_enabled", True),
        )
        self._compiled_sampler_cache: dict[tuple[float, float], Any] = {}
        self._compiled_sampler_lock = threading.Lock()

        # Sprint C5: KV cache quantisation. mlx-lm 0.22+ accepts
        # ``kv_bits`` on ``stream_generate`` -- 8 halves KV memory and
        # frees ~10-15% generation throughput on long prompts (mlx-lm
        # release notes). 0/None disables quantisation. We start
        # quantising after ``kv_quant_warmup_tokens`` so the first
        # tokens (where quality matters most) keep full precision.
        self._kv_bits: int = int(brain_cfg.get("kv_bits", 8))
        self._kv_group_size: int = int(brain_cfg.get("kv_group_size", 64))
        self._kv_quant_warmup: int = int(
            brain_cfg.get("kv_quant_warmup_tokens", 512)
        )
        if self._kv_bits not in (0, 4, 8):
            logger.warning(
                "Unsupported kv_bits=%d (must be 0, 4, or 8) -- "
                "falling back to 0 (disabled)", self._kv_bits,
            )
            self._kv_bits = 0

        self._models: dict[str, Any | None] = {role: None for role in self._ROLES}
        self._tokenizers: dict[str, Any | None] = {role: None for role in self._ROLES}
        self._fingerprints: dict[str, str | None] = {role: None for role in self._ROLES}
        self._loaded_roles: dict[str, bool] = {role: False for role in self._ROLES}
        self._load_failed: dict[str, bool] = {role: False for role in self._ROLES}
        self._role_last_used: dict[str, float] = {role: 0.0 for role in self._ROLES}

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._load_lock = threading.RLock()
        self._brain_mode_mgr: BrainModeManager | None = None
        self._abort_generation = 0
        self._gen_lock = threading.Lock()
        self._streaming_depth = 0

        # Prompt-prefix KV cache (opt-in via config). Keyed per role.
        # Sprint P2.6 (Apr 26 2026): the runtime availability check
        # (_HAS_PROMPT_CACHE / _save_prompt_cache) is deferred until the
        # MLX import has actually run. We persist the user's *intent*
        # here and gate the runtime *capability* via helpers below so
        # config doesn't get silently False'd at __init__ time before
        # the lazy import has populated the symbols.
        self._prompt_cache_user_pref: bool = bool(
            brain_cfg.get("prompt_cache_enabled", True),
        )
        self._prompt_cache_max_size: int = int(
            brain_cfg.get("prompt_cache_max_size", 4),
        )
        self._prompt_cache_max_bytes: int = int(
            brain_cfg.get("prompt_cache_max_mb", 512),
        ) * 1024 * 1024
        self._prompt_caches: dict[str, Any] = {}
        self._prompt_cache_lock = threading.Lock()
        self._prompt_cache_hits: int = 0
        self._prompt_cache_misses: int = 0

        # Cross-boot prompt-cache persistence (B7). On first turn we
        # snapshot the (system_prompt + first_response) KV state to disk;
        # next boot we mmap-load it back into the LRU so the second-boot
        # first-token latency drops from ~7s (cold prefill) to <1s
        # (warm cache reuse). Keyed by role + model_path md5 so a model
        # swap or prompt rewrite invalidates safely.
        self._prompt_cache_persist_user_pref: bool = bool(
            brain_cfg.get("prompt_cache_persist", True),
        )
        persist_path = brain_cfg.get(
            "prompt_cache_persist_path", "data/prompt_cache_v33.safetensors",
        )
        self._prompt_cache_persist_path = Path(str(persist_path)).expanduser()
        self._prompt_cache_persist_min_tokens: int = int(
            brain_cfg.get("prompt_cache_persist_min_tokens", 256),
        )
        self._prompt_cache_persisted_role: dict[str, bool] = {
            role: False for role in self._ROLES
        }
        self._prompt_cache_restore_attempted: dict[str, bool] = {
            role: False for role in self._ROLES
        }
        # Lazy GC: when the persistence symbols load post __init__, the
        # first ``_ensure_loaded`` call will run ``_gc_stale_prompt_caches``
        # via ``_maybe_init_prompt_cache_runtime``. Doing it here would
        # require the heavy MLX import, which P2.6 explicitly defers.

        # Lifetime perf counters used by the periodic perf snapshot (logged
        # every ~60s by the main loop). We track totals rather than a
        # rolling window because the boot-time interest is "is the cache
        # paying off so far?" not "what's my last 30s look like?". A
        # second pass can add a windowed view if needed.
        self._perf_total_turns: int = 0
        self._perf_total_tokens: int = 0
        self._perf_total_ms: float = 0.0
        self._perf_peak_memory_gb: float = 0.0
        self._perf_lock = threading.Lock()

        # Thermal derate (Sprint C4): an external thermal-aware orchestrator
        # (Silicon Governor) can ask us to cap ``max_tokens`` so the laptop
        # doesn't thrash under sustained heat. 1.0 = normal, 0.7 = mild
        # throttle, 0.5 = hot, 0.35 = critical. The active profile's
        # ``max_tokens`` is multiplied by this ratio at request time.
        self._thermal_clamp_ratio: float = 1.0
        self._thermal_clamp_reason: str = ""

        # Sprint C1 — runtime persona is pinned as a stable KV prefix.
        # The atomLogs.txt audit showed KV reuse stuck at 67-75% because
        # the persona block (~600 tokens) was being prefilled cold on
        # most turns. We hold the persona file path + the mtime we
        # pinned with so a single ``repin_persona_if_changed`` call at
        # the start of every turn is O(1) when nothing changed and
        # automatically re-runs the prefill if Boss has edited
        # ``config/atom_persona.md``. The pin itself is implemented by
        # warming the existing trie via a 1-token generation; the trie
        # then holds the prefix KV state and ``_prepare_prompt_cache``
        # finds it as the longest matching prefix on every later turn.
        self._pinned_persona_path: Path | None = None
        self._pinned_persona_mtime: float = 0.0
        self._pinned_persona_role: str = "fast"
        self._pinned_persona_token_count: int = 0
        self._pinned_persona_lock = threading.Lock()

        # Sprint P3.1 (Apr 26 2026): one-line summary of the model wiring
        # so a fresh boot trace shows whether dual-tier actually took
        # effect. Operators previously had to grep for fingerprints across
        # multiple log lines to answer "is fast=4B or 8B right now?"
        if self._dual_tier_enabled:
            logger.info(
                "MLXBrain dual-tier wired: primary=%s fast=%s%s",
                Path(self._role_model_paths["primary"]).name,
                Path(self._role_model_paths["fast"]).name,
                (
                    f" speculative=on(draft={Path(self._speculative_draft_path).name})"
                    if self._speculative_enabled and self._speculative_draft_path
                    else ""
                ),
            )
        else:
            logger.info(
                "MLXBrain single-tier: model=%s (set brain.mlx_fast_model "
                "to a separate small model dir to enable dual-tier)",
                Path(self._model_path).name,
            )

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
        return Path(self._model_path).is_dir()

    @property
    def _prompt_cache_enabled(self) -> bool:
        """User-config preference AND runtime symbol availability.

        Sprint P2.6 (Apr 26 2026): the runtime side of this flips True
        only after ``_lazy_import_mlx()`` has populated
        ``_LRUPromptCache``. Before that, _make_prompt_cache may be
        None and ``_get_prompt_lru`` would crash; this gate keeps the
        rest of the code unchanged.
        """
        return bool(
            self._prompt_cache_user_pref
            and _HAS_PROMPT_CACHE
            and _LRUPromptCache is not None,
        )

    @property
    def _prompt_cache_persist_enabled(self) -> bool:
        """User-config preference AND runtime save/load availability."""
        return bool(
            self._prompt_cache_persist_user_pref
            and _save_prompt_cache is not None
            and _load_prompt_cache is not None,
        )

    def _maybe_init_prompt_cache_runtime(self) -> None:
        """Run the one-time prompt-cache GC after the MLX import lands.

        Was previously done eagerly in ``__init__`` -- moved out as part
        of P2.6 so the heavy MLX symbol import doesn't run until first
        model-load. Idempotent.
        """
        if getattr(self, "_prompt_cache_runtime_inited", False):
            return
        if not self._prompt_cache_persist_enabled:
            return
        try:
            self._gc_stale_prompt_caches()
        except Exception:
            logger.debug(
                "MLX prompt-cache GC raised post-lazy-import", exc_info=True,
            )
        self._prompt_cache_runtime_inited = True

    def _normalize_role(self, role: str | None) -> str:
        """Coerce any role label to one of the two tracked tags.

        Historical values like "deep" now collapse to "primary" so old
        callers keep working without a KeyError on the state dicts.
        """
        key = (role or self._active_role or "primary").strip().lower()
        if key in self._ROLES:
            return key
        return "primary"

    def _path_for_role(self, role: str) -> str:
        """Resolve the model path for ``role``.

        Sprint P3.1 (Apr 26 2026): when ``brain.mlx_fast_model`` is set
        and the fast model directory exists, "fast" and "primary" can
        resolve to *different* on-disk weights -- this is the dual-tier
        path. Otherwise the role tag is just observability metadata and
        every role aliases the same primary weights (the prior single-
        model behavior, preserved verbatim for low-RAM setups).
        """
        norm = self._normalize_role(role)
        path = self._role_model_paths.get(norm)
        if path:
            return path
        return self._model_path

    def _effective_inference(self, model_role: str | None = None) -> dict[str, Any]:
        role = self._normalize_role(model_role)
        eff = self._brain_mode_mgr.effective_params() if self._brain_mode_mgr is not None else {}
        base_max = int(eff.get("max_tokens", self._max_tokens))
        clamped_max = base_max
        ratio = float(self._thermal_clamp_ratio)
        if ratio < 0.999:
            # Always leave a sensible floor so short answers still fit.
            clamped_max = max(64, int(round(base_max * max(0.2, ratio))))
        return {
            "profile": eff.get("profile", "default"),
            "model_role": role,
            "model_path": self._path_for_role(role),
            "max_tokens": clamped_max,
            "max_tokens_base": base_max,
            "thermal_clamp_ratio": ratio,
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

    def set_thermal_clamp(self, ratio: float, *, reason: str = "") -> None:
        """Adjust the thermal ``max_tokens`` multiplier.

        Called by the main orchestrator when sustained thermal pressure
        is detected. ``ratio`` is clamped to ``[0.25, 1.0]`` so a buggy
        feed can't silently disable generation.
        """
        try:
            r = max(0.25, min(1.0, float(ratio)))
        except (TypeError, ValueError):
            return
        prev = self._thermal_clamp_ratio
        if abs(r - prev) < 1e-3:
            self._thermal_clamp_reason = reason or self._thermal_clamp_reason
            return
        self._thermal_clamp_ratio = r
        self._thermal_clamp_reason = reason or self._thermal_clamp_reason
        logger.info(
            "MLX thermal clamp: %.2f -> %.2f (%s)",
            prev, r, reason or "unspecified",
        )

    @property
    def thermal_clamp_ratio(self) -> float:
        return self._thermal_clamp_ratio

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
        # Sprint P2.6 (Apr 26 2026): pay the heavy MLX import cost only on
        # first model-load; subsequent calls hit the populated globals.
        if not _lazy_import_mlx() or load is None:
            return False
        self._maybe_init_prompt_cache_runtime()

        eff = self._effective_inference(model_role)
        role = eff["model_role"]
        model_path = Path(eff["model_path"])
        if not model_path.is_dir():
            logger.warning("MLX model directory not found for role=%s: %s", role, model_path)
            self._load_failed[role] = True
            return False

        with self._load_lock:
            if self._loaded_roles[role] and self._fingerprints[role] == str(model_path):
                self._role_last_used[role] = time.monotonic()
                return True

            # Single-model profile: if any role already holds this weight
            # file, alias the tensors into the requested role instead of
            # loading twice. Saves ~4.5 GB RAM + ~6 s of reload latency
            # when the kernel toggles between primary/fast labels.
            for other in self._ROLES:
                if other == role:
                    continue
                if (
                    self._loaded_roles[other]
                    and self._fingerprints[other] == str(model_path)
                    and self._models[other] is not None
                ):
                    self._models[role] = self._models[other]
                    self._tokenizers[role] = self._tokenizers[other]
                    self._fingerprints[role] = str(model_path)
                    self._loaded_roles[role] = True
                    self._load_failed[role] = False
                    self._role_last_used[role] = time.monotonic()
                    logger.info(
                        "MLX: aliasing loaded weights for role=%s from %s",
                        role, other,
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
                self._role_last_used[role] = time.monotonic()
                elapsed = (time.monotonic() - t0) * 1000
                logger.info("MLX model role=%s loaded in %.0fms", role, elapsed)
                self._restore_persisted_prompt_cache(role)
                return True
            except Exception:
                logger.exception("Failed to load MLX model role=%s", role)
                self._load_failed[role] = True
                self._unload_role_unlocked(role)
                self._clear_mlx_cache()
                return False

    def preload(self, *, model_role: str | None = None, load_all: bool = False) -> bool:
        if load_all:
            # Single model: loading primary is sufficient, fast aliases
            # the same tensors on first request. Kept as a loop so a
            # future multi-model profile only needs _ROLES expanded.
            ok = True
            for role in self._ROLES:
                ok = self._ensure_loaded(role) and ok
            return ok
        return self._ensure_loaded(model_role)

    def _ensure_draft_loaded(self) -> bool:
        """Lazy-load the speculative draft model (Sprint P3.2).

        Returns True only when speculative decoding is enabled, the draft
        model path is configured AND on disk, and load() succeeded. Any
        failure flips ``_draft_load_failed`` so we don't retry every
        call. mlx-lm's speculative path requires the draft and target
        models to share a tokenizer; we log a warning if vocabs disagree
        and disable speculation for safety.
        """
        if not self._speculative_enabled:
            return False
        if self._draft_loaded:
            return True
        if self._draft_load_failed:
            return False
        if not _lazy_import_mlx() or load is None:
            return False
        path_str = self._speculative_draft_path
        if not path_str:
            logger.warning(
                "MLX speculative decoding enabled but "
                "brain.speculative_decoding.draft_model_path is empty",
            )
            self._draft_load_failed = True
            return False
        path = Path(str(path_str)).expanduser()
        if not path.is_dir():
            logger.warning(
                "MLX draft model path missing: %s -- "
                "disabling speculative decoding",
                path,
            )
            self._draft_load_failed = True
            return False
        # If the user pointed the draft at the same dir as the target the
        # speculation is a no-op (draft == target). Skip and warn.
        if str(path) == str(Path(self._model_path).expanduser()):
            logger.warning(
                "MLX speculative decoding draft path == target path (%s); "
                "skipping (would not yield speedup)",
                path,
            )
            self._draft_load_failed = True
            return False

        try:
            t0 = time.monotonic()
            draft_model, draft_tokenizer = load(str(path))
            self._draft_model = draft_model
            self._draft_tokenizer = draft_tokenizer
            self._draft_loaded = True
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.info(
                "MLX draft model loaded for speculative decoding: %s "
                "(num_draft_tokens=%d) in %.0f ms",
                path.name,
                self._speculative_num_draft_tokens,
                elapsed_ms,
            )
            return True
        except Exception:
            logger.exception(
                "MLX draft model load failed for %s; "
                "disabling speculative decoding",
                path,
            )
            self._draft_load_failed = True
            return False

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

    # ── Prompt-prefix KV cache (opt-in) ─────────────────────────────

    def _get_prompt_lru(self, role: str) -> Any | None:
        """Return the LRU prompt cache for ``role``, lazy-creating one.

        Returns None when prompt caching is disabled or the MLX runtime
        doesn't expose the LRU machinery (older mlx_lm versions).
        """
        if not self._prompt_cache_enabled or _LRUPromptCache is None:
            return None
        key = self._normalize_role(role)
        with self._prompt_cache_lock:
            existing = self._prompt_caches.get(key)
            if existing is not None:
                return existing
            try:
                lru = _LRUPromptCache(
                    max_size=self._prompt_cache_max_size,
                    max_bytes=self._prompt_cache_max_bytes,
                )
            except Exception:
                logger.debug("LRUPromptCache init failed", exc_info=True)
                return None
            self._prompt_caches[key] = lru
            return lru

    def drop_prompt_caches(self, reason: str = "pressure") -> None:
        """Release all cached KV prefixes. Called from memory-pressure
        hooks so ATOM can reclaim RAM without unloading the model itself.
        """
        with self._prompt_cache_lock:
            if not self._prompt_caches:
                return
            self._prompt_caches.clear()
        logger.info("MLX prompt-cache dropped (%s)", reason)

    @property
    def prompt_cache_stats(self) -> dict[str, Any]:
        hits = self._prompt_cache_hits
        misses = self._prompt_cache_misses
        total = hits + misses
        hit_rate = (hits / total) if total else 0.0
        return {
            "enabled": self._prompt_cache_enabled,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 3),
            "roles_cached": sorted(self._prompt_caches.keys()),
        }

    def get_perf_snapshot(self) -> dict[str, Any]:
        """Return a lifetime performance summary suitable for periodic INFO logs.

        Returned fields:
            - ``turns``: number of completed generations since boot
            - ``tokens``: total decoded tokens
            - ``avg_tok_s``: lifetime average decode rate (tokens / total ms)
            - ``avg_ms``: lifetime average per-turn wall time
            - ``peak_memory_gb``: largest peak GPU memory observed
            - ``cache``: ``prompt_cache_stats`` dict
            - ``thermal_clamp_ratio``: current Silicon Governor clamp
        """
        with self._perf_lock:
            turns = self._perf_total_turns
            tokens = self._perf_total_tokens
            total_ms = self._perf_total_ms
            peak_gb = self._perf_peak_memory_gb
        avg_tok_s = (tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        avg_ms = (total_ms / turns) if turns > 0 else 0.0
        return {
            "turns": turns,
            "tokens": tokens,
            "avg_tok_s": round(avg_tok_s, 1),
            "avg_ms": round(avg_ms, 0),
            "peak_memory_gb": round(peak_gb, 2),
            "cache": self.prompt_cache_stats,
            "thermal_clamp_ratio": round(self._thermal_clamp_ratio, 2),
        }

    @staticmethod
    def _encode_prompt_tokens(tokenizer: Any, prompt: str) -> list[int] | None:
        """Tokenise ``prompt`` the SAME way ``stream_generate`` does.

        We must match the library's special-token handling or the token
        sequence we feed back to ``generate_step`` via ``prompt_cache``
        will drift from the model's expectations.
        """
        try:
            bos = getattr(tokenizer, "bos_token", None)
            add_special = bos is None or not prompt.startswith(bos)
            toks = tokenizer.encode(prompt, add_special_tokens=add_special)
        except TypeError:
            try:
                toks = tokenizer.encode(prompt)
            except Exception:
                return None
        except Exception:
            return None
        if toks is None:
            return None
        if hasattr(toks, "tolist"):
            toks = toks.tolist()
        else:
            toks = list(toks)
        return [int(t) for t in toks] if toks else None

    def _prompt_cache_model_key(self, role: str) -> tuple[str, str]:
        """Stable hashable key for the trie.

        ``LRUPromptCache`` uses the model object as a dict key, but raw
        ``mlx.nn.Module`` instances aren't hashable. mlx_lm's own server
        sidesteps this by using a ``(path, adapter, draft)`` tuple; we do
        the same, keyed on the configured model path so reloads reuse the
        same trie slot.
        """
        normalized = self._normalize_role(role)
        return (normalized, str(self._model_path or normalized))

    def _prepare_prompt_cache(
        self,
        role: str,
        tokenizer: Any,
        model: Any,
        prompt: Any,
    ) -> tuple[Any | None, Any, list[int] | None]:
        """Fetch the nearest cached prefix for ``prompt``.

        Returns ``(cache, run_prompt, full_tokens)`` where:
          - ``cache`` is the KV prompt cache to pass to ``stream_generate``
            (``None`` means no caching this turn — caller should run the
            full prompt through vanilla MLX)
          - ``run_prompt`` is the prompt to actually pass to
            ``stream_generate`` — the non-cached delta tokens as an
            ``mx.array`` (or the original string when caching is off)
          - ``full_tokens`` is the complete token list used for insertion
            after generation completes (``None`` when we can't insert)
        """
        lru = self._get_prompt_lru(role)
        if lru is None or _make_prompt_cache is None or not isinstance(prompt, str):
            return None, prompt, None

        full_tokens = self._encode_prompt_tokens(tokenizer, prompt)
        if not full_tokens:
            return None, prompt, None

        model_key = self._prompt_cache_model_key(role)
        try:
            fetched, remaining = lru.fetch_nearest_cache(model_key, full_tokens)
        except Exception:
            logger.debug("prompt_cache fetch failed", exc_info=True)
            return None, prompt, full_tokens

        hit_tokens = len(full_tokens) - (len(remaining) if remaining else 0)
        if fetched is None:
            try:
                fetched = _make_prompt_cache(model)
            except Exception:
                logger.debug("make_prompt_cache failed", exc_info=True)
                return None, prompt, full_tokens
            remaining = list(full_tokens)
            self._prompt_cache_misses += 1
        else:
            # Any non-trivial prefix match is a win for first-token
            # latency. Log at INFO only when the hit is substantial so
            # the logs stay readable on uncached turns.
            self._prompt_cache_hits += 1
            if hit_tokens > 32:
                logger.info(
                    "MLX prompt-cache: %d/%d tokens reused (%.0f%%)",
                    hit_tokens,
                    len(full_tokens),
                    100.0 * hit_tokens / max(1, len(full_tokens)),
                )

        if not remaining:
            # Everything is cached. ``generate_step`` needs at least
            # one input token to emit a logprob, so feed the final
            # token back in. The cache already holds its KV so this
            # is effectively free.
            remaining = [full_tokens[-1]]

        if mx is None:
            return None, prompt, full_tokens

        try:
            run_prompt = mx.array(remaining)
        except Exception:
            logger.debug("mx.array(remaining) failed", exc_info=True)
            return None, prompt, full_tokens

        return fetched, run_prompt, full_tokens

    def _commit_prompt_cache(
        self,
        role: str,
        cache: Any,
        full_prompt_tokens: list[int],
        response_tokens: list[int],
    ) -> None:
        """Store the post-generation cache under (prompt + response) so the
        next turn can reuse the longest shared prefix.

        Safe against any failure in the LRU / trim path — a failed
        insertion just means we skip caching this turn.
        """
        if cache is None or not full_prompt_tokens:
            return
        lru = self._get_prompt_lru(role)
        if lru is None:
            return
        try:
            model_key = self._prompt_cache_model_key(role)
            tokens_for_trie = list(full_prompt_tokens)
            if response_tokens:
                tokens_for_trie.extend(int(t) for t in response_tokens)
            lru.insert_cache(model_key, tokens_for_trie, cache)
        except Exception:
            logger.debug("prompt_cache insert failed", exc_info=True)
            return
        if (
            self._prompt_cache_persist_enabled
            and not self._prompt_cache_persisted_role.get(role)
            and len(full_prompt_tokens) >= self._prompt_cache_persist_min_tokens
        ):
            try:
                self._persist_prompt_cache(role, cache, tokens_for_trie)
                self._prompt_cache_persisted_role[role] = True
            except Exception:
                logger.debug("prompt_cache persist failed", exc_info=True)

    # ── Cross-boot prompt-cache persistence (B7) ─────────────────────

    def _persist_path_for_role(self, role: str) -> Path:
        """Per-role + per-model snapshot path.

        Sprint Ω.2: previously the file name was role-only
        (``prompt_cache_v33-fast.safetensors``). When the brain swapped
        from Qwen3-4B → Qwen3-8B, the metadata's ``model_path``
        no longer matched, so ``_restore_persisted_prompt_cache``
        silently bailed and every cold boot paid the full ~6.3 s
        persona prefill again. Worse, the stale 42 MB file kept
        squatting on disk forever.

        Now we suffix the model directory's basename so the 4B and 8B
        caches live in distinct files. The first cold boot on a new
        model still pays the prefill once, then every subsequent boot
        reuses its own cache. Old role-only files become orphaned and
        get garbage-collected by ``_gc_stale_prompt_caches`` below.
        """
        base = self._prompt_cache_persist_path
        suffix = base.suffix or ".safetensors"
        model_tag = Path(self._model_path).name.lower().replace("/", "_") or "default"
        return base.with_name(f"{base.stem}-{role}-{model_tag}{suffix}")

    def _gc_stale_prompt_caches(self) -> None:
        """Remove role-only / wrong-model snapshot files left over from
        an earlier brain. Keeps the disk tidy and prevents the legacy
        v33 layout from confusing operators inspecting ``data/``.

        Best-effort: a missing file or a permissions issue is fine; we
        log at DEBUG and move on. Runs once at MLXBrain construction so
        the cleanup never re-races with a write.
        """
        try:
            base = self._prompt_cache_persist_path
            if not base.parent.exists():
                return
            valid_paths = {
                self._persist_path_for_role(role).name for role in self._ROLES
            }
            stem = base.stem
            for path in base.parent.glob(f"{stem}*{base.suffix or '.safetensors'}"):
                if path.name in valid_paths:
                    continue
                try:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    path.unlink()
                    logger.info(
                        "MLX prompt-cache GC: removed stale snapshot %s (%.1f MB) -- "
                        "wrong model fingerprint",
                        path.name, size_mb,
                    )
                except Exception:
                    logger.debug("prompt_cache GC unlink failed", exc_info=True)
        except Exception:
            logger.debug("prompt_cache GC scan failed", exc_info=True)

    def _persist_prompt_cache(
        self,
        role: str,
        cache: Any,
        tokens_for_trie: list[int],
    ) -> None:
        """Write the current cache to disk + a sidecar JSON of metadata
        (tokens, model_path, prompt_md5, version) so we can re-insert it
        into the LRU on the next boot under the same trie key."""
        if _save_prompt_cache is None:
            return
        path = self._persist_path_for_role(role)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_path": str(self._model_path),
            "tokens": ",".join(str(t) for t in tokens_for_trie),
            "n_tokens": str(len(tokens_for_trie)),
            "version": "v33",
        }
        try:
            _save_prompt_cache(str(path), cache, meta)
            logger.info(
                "MLX prompt-cache persisted: role=%s tokens=%d -> %s",
                role, len(tokens_for_trie), path.name,
            )
        except Exception as exc:
            logger.debug("save_prompt_cache failed: %s", exc)

    def _restore_persisted_prompt_cache(self, role: str) -> None:
        """On model load, try to load the persisted cache from disk and
        inject it into the LRU. Skips silently if the file is missing,
        the model_path doesn't match, or anything goes wrong — never
        raises (failure just means cold prefill on first turn)."""
        if (
            not self._prompt_cache_persist_enabled
            or _load_prompt_cache is None
            or self._prompt_cache_restore_attempted.get(role)
        ):
            return
        self._prompt_cache_restore_attempted[role] = True
        path = self._persist_path_for_role(role)
        if not path.is_file():
            return
        try:
            cache, meta = _load_prompt_cache(str(path), return_metadata=True)
        except Exception as exc:
            logger.debug("load_prompt_cache failed: %s", exc)
            return
        if not meta or meta.get("model_path") != str(self._model_path):
            logger.debug(
                "MLX prompt-cache: model_path mismatch (have=%s, persisted=%s); ignoring",
                self._model_path, (meta or {}).get("model_path"),
            )
            return
        try:
            tokens = [int(t) for t in (meta.get("tokens") or "").split(",") if t]
        except Exception:
            tokens = []
        if not tokens:
            return
        lru = self._get_prompt_lru(role)
        if lru is None:
            return
        try:
            model_key = self._prompt_cache_model_key(role)
            lru.insert_cache(model_key, tokens, cache, cache_type="system")
            self._prompt_cache_persisted_role[role] = True
            logger.info(
                "MLX prompt-cache restored: role=%s tokens=%d (warm prefill on next turn)",
                role, len(tokens),
            )
        except Exception as exc:
            logger.debug("prompt_cache restore insert failed: %s", exc)

    def _make_sampler(self, temperature: float, top_p: float):
        temp = max(0.0, float(temperature))
        nucleus = max(0.0, min(1.0, float(top_p)))
        if make_sampler is None:
            return None
        # Sprint P3.5 (Apr 26 2026): wrap the sampler in mx.compile so the
        # per-token graph is JIT-fused once and reused for every later
        # token. Cache by (temp, top_p) so multiple profiles each get a
        # warm compiled function. If compile fails (older mlx, weird
        # backend) we fall back transparently to the eager sampler.
        base = make_sampler(temp=temp, top_p=nucleus)
        if not self._mx_compile_enabled or mx is None:
            return base
        compile_fn = getattr(mx, "compile", None)
        if compile_fn is None:
            return base
        key = (round(temp, 6), round(nucleus, 6))
        with self._compiled_sampler_lock:
            cached = self._compiled_sampler_cache.get(key)
            if cached is not None:
                return cached
            try:
                compiled = compile_fn(base)
            except Exception:
                logger.debug(
                    "mx.compile failed for sampler "
                    "(temp=%.3f top_p=%.3f); using eager",
                    temp, nucleus, exc_info=True,
                )
                return base
            self._compiled_sampler_cache[key] = compiled
            logger.info(
                "MLX sampler compiled via mx.compile (temp=%.2f top_p=%.2f)",
                temp, nucleus,
            )
            return compiled

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
            # Sprint C4: do NOT ``.strip()`` here -- "\n\n" is a
            # legitimate FAST-path stop sequence and stripping would
            # discard it. Only filter out genuinely empty / None values.
            candidate = str(seq) if seq is not None else ""
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

        # Defense-in-depth: cut at first leaked ChatML / HF control token
        # (the model occasionally emits these even after we add them to the
        # stop-sequence list because the tokenizer can split them across
        # streamed chunks). Anything after such a token is a hallucinated
        # next-turn prompt and must never reach TTS.
        m = _SPECIAL_TOKEN_RE.search(guarded)
        if m is not None:
            guarded = guarded[: m.start()].rstrip()
        # Also strip any stray copies that the search above missed (e.g.
        # tokens inside a longer body). Do this after the truncation so we
        # don't leave half-tokens in place.
        guarded = _SPECIAL_TOKEN_RE.sub("", guarded)
        if not guarded:
            return "", "control_token_only", True

        leading_stripped = _LEADING_QUOTE_WRAP_RE.sub("", guarded)
        if leading_stripped != guarded:
            guarded = leading_stripped
            trailing_stripped = _TRAILING_UNCLOSED_QUOTE_RE.sub("", guarded).rstrip()
            if trailing_stripped:
                guarded = trailing_stripped
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
            # P0c (test_mlx_speaker_label_loop_on_wrapper_returns_empty):
            # *first* check whether the pre-loop buffer is only a wrapper
            # preface like `The answer is "..."` with no real content. If
            # so, treat as unusable speaker-label loop regardless of
            # length -- shipping it produces "ATOM invented an action"
            # hallucinations downstream.
            if _looks_like_wrapper_preface(trimmed):
                return "", "speaker_label_loop_wrapper", True
            # Sprint K: long replies that end with a duplicated speaker
            # label have already produced usable content; we still flag
            # the loop so the caller can stop streaming, but we hand the
            # trimmed body back as visible text.
            if len(trimmed.split()) >= 30:
                return trimmed, "speaker_label_loop", True
            # Sprint K: short FAST replies (<30 words) frequently end in
            # a harmless duplicated `Assistant:` label after already
            # delivering the answer. Treating those as a hard
            # speaker-label loop caused "empty response" recoveries in
            # atomlogs.txt. We *do* still report the loop (so streaming
            # halts), but we keep the trimmed body.
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
        extra_stop_sequences: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[str, bool]:
        eff = self._effective_inference(model_role)
        role = eff["model_role"]
        if not self._ensure_loaded(role):
            return "", False

        model = self._models[role]
        tokenizer = self._tokenizers[role]
        if model is None or tokenizer is None or stream_generate is None:
            return "", False

        # Stamp last-used so the deep-model idle-unloader knows when this
        # role was actually busy.
        self._role_last_used[role] = time.monotonic()

        with self._gen_lock:
            self._streaming_depth += 1
        try:
            return self._generate_sync_streaming_inner(
                role, eff, model, tokenizer, prompt, on_token,
                max_tokens_override=max_tokens_override,
                extra_stop_sequences=extra_stop_sequences,
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
        extra_stop_sequences: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[str, bool]:
        """Core stream loop (wrapped for active-generation accounting)."""
        my_gen = self._abort_generation
        sampler = self._make_sampler(eff["temperature"], eff["top_p"])
        logits_processors = self._make_logits_processors(eff["repeat_penalty"])
        # Sprint C4: merge per-call extra stop sequences (e.g. FAST
        # path adds "(" + "\n\n") on top of the role-level ones from
        # ``eff``. Per-call stops are NOT cached on the role -- they
        # only apply to this generation.
        merged_extra: list[str] = list(eff["extra_stop_sequences"])
        if extra_stop_sequences:
            for seq in extra_stop_sequences:
                if seq and seq not in merged_extra:
                    merged_extra.append(seq)
        stop_sequences = self._stop_sequences(merged_extra)

        visible_text = ""
        raw_text = ""
        last_resp: Any = None
        t0 = time.perf_counter()
        stop_reason: str | None = None
        response_tokens: list[int] = []

        if _HAS_MLX and mx is not None:
            try:
                mx.reset_peak_memory()
            except Exception:
                logger.debug('MLX peak memory reset failed', exc_info=True)

        # Prompt-prefix KV cache handshake. When there's a usable match,
        # we only feed the non-cached delta tokens to ``stream_generate``
        # while passing the matching cache through as ``prompt_cache``.
        # This is the single biggest first-token-latency win for ATOM,
        # because the ~500-token system identity block is identical on
        # every single turn.
        prompt_cache_obj, run_prompt, full_tokens = self._prepare_prompt_cache(
            role, tokenizer, model, prompt,
        )
        stream_kwargs: dict[str, Any] = {}
        if prompt_cache_obj is not None:
            stream_kwargs["prompt_cache"] = prompt_cache_obj

        # Sprint C5: pass KV-cache quantisation params so generation
        # uses ``QuantizedKVCache`` once we cross the warmup threshold.
        # ``kv_bits=0`` (or absence) keeps the legacy unquantised path.
        if self._kv_bits in (4, 8):
            stream_kwargs["kv_bits"] = int(self._kv_bits)
            stream_kwargs["kv_group_size"] = int(self._kv_group_size)
            stream_kwargs["quantized_kv_start"] = int(self._kv_quant_warmup)

        # Sprint P3.2 (Apr 26 2026): plumb the draft model + draft window
        # into stream_generate when speculative decoding is enabled. The
        # mlx-lm path drops `max_kv_size` and `prompt_progress_callback`
        # automatically when `draft_model is not None`, so we just pass
        # what we know. Speculation is silently disabled if the draft
        # model never loaded.
        speculative_active = False
        # Tracks where the main-model portion of `prompt_cache_obj` ends
        # when we have to extend the list with a fresh draft segment for
        # `mlx_lm.speculative_generate_step`. Sliced back off after
        # generation so `_commit_prompt_cache` only stores the main cache
        # in the LRU. ``None`` means no extension was performed.
        main_cache_len: int | None = None
        if self._speculative_enabled and self._ensure_draft_loaded():
            stream_kwargs["draft_model"] = self._draft_model
            stream_kwargs["num_draft_tokens"] = int(
                self._speculative_num_draft_tokens,
            )
            speculative_active = True
            # mlx_lm's speculative_generate_step expects ``prompt_cache``
            # to be sized for BOTH models concatenated:
            #     prompt_cache = [main_layer_0..N, draft_layer_0..M]
            # When ATOM passes only the main-model cache (which it does
            # on every prompt-cache hit), the draft segment slices to []
            # and the very first ``model(y, cache=[])`` raises
            # ``IndexError: list index out of range`` from inside
            # ``create_attention_mask(h, cache[0])``. We extend the cache
            # list in-place with a fresh draft prompt cache so the shape
            # is correct. The draft starts with no cached prefix — its
            # speculation may be lower quality on the first turn but it
            # is correct AND won't crash; the main model still benefits
            # from its cached prefix unchanged.
            if (
                prompt_cache_obj is not None
                and isinstance(prompt_cache_obj, list)
                and _make_prompt_cache is not None
                and self._draft_model is not None
            ):
                try:
                    main_cache_len = len(prompt_cache_obj)
                    draft_cache_segment = _make_prompt_cache(
                        self._draft_model,
                    )
                    prompt_cache_obj.extend(draft_cache_segment)
                except Exception:
                    logger.debug(
                        "MLX speculative: failed to build draft cache "
                        "segment; disabling speculation for this turn",
                        exc_info=True,
                    )
                    stream_kwargs.pop("draft_model", None)
                    stream_kwargs.pop("num_draft_tokens", None)
                    speculative_active = False
                    main_cache_len = None

        try:
            for resp in stream_generate(
                model,
                tokenizer,
                run_prompt,
                max_tokens=int(max_tokens_override or eff["max_tokens"]),
                sampler=sampler,
                logits_processors=logits_processors,
                **stream_kwargs,
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
                tok_id = getattr(resp, "token", None)
                if tok_id is not None:
                    try:
                        response_tokens.append(int(tok_id))
                    except Exception:
                        logger.debug("Could not capture response token id", exc_info=True)
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

            # Strip the draft-cache segment we appended for speculative
            # decoding (if any) so ``_commit_prompt_cache`` stores ONLY
            # the main-model cache in the LRU. The LRU is shape-keyed by
            # the main model and any draft-cache leakage would corrupt
            # subsequent ``fetch_nearest_cache`` slicing.
            if (
                main_cache_len is not None
                and isinstance(prompt_cache_obj, list)
                and len(prompt_cache_obj) > main_cache_len
            ):
                del prompt_cache_obj[main_cache_len:]

            # Commit the post-generation cache back into the LRU trie.
            # This keys the cache under (prompt + response) tokens so the
            # next turn's ``fetch_nearest_cache`` can trim down to the
            # longest shared prefix (typically the system block).
            #
            # We commit even when ``stop_reason`` is set (e.g. guard
            # stripped a ``<think>`` block) because the KV state for the
            # user/system prefix is still valuable — the guard stops the
            # *visible output*, not the token processing, and the next
            # turn can still trim down to that prefix. We only skip the
            # commit when the whole generation was aborted (another
            # request preempted this one).
            if (
                prompt_cache_obj is not None
                and full_tokens is not None
                and self._abort_generation == my_gen
            ):
                self._commit_prompt_cache(
                    role, prompt_cache_obj, full_tokens, response_tokens,
                )

            text = visible_text.strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            generation_tokens = int(
                getattr(last_resp, "generation_tokens", max(1, len(raw_text.split()))),
            )
            generation_tps = float(getattr(last_resp, "generation_tps", 0.0))
            peak_memory = float(getattr(last_resp, "peak_memory", 0.0))
            with self._perf_lock:
                self._perf_total_turns += 1
                self._perf_total_tokens += generation_tokens
                self._perf_total_ms += elapsed_ms
                if peak_memory > self._perf_peak_memory_gb:
                    self._perf_peak_memory_gb = peak_memory
            logger.info(
                "MLX [%s/%s]: %.0fms, %d tokens, ~%d words, %.1f tok/s, "
                "peak %.2fGB%s",
                eff["profile"],
                role,
                elapsed_ms,
                generation_tokens,
                len(text.split()),
                generation_tps,
                peak_memory,
                " spec=on" if speculative_active else "",
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
        finally:
            # Defensive: even if generation crashed mid-stream, ensure
            # ``prompt_cache_obj`` is restored to main-model shape. This
            # makes the function side-effect-safe — the LRU never sees a
            # combined main+draft cache list, and the next turn can
            # reuse the (partial) main prefix without surprises.
            if (
                main_cache_len is not None
                and isinstance(prompt_cache_obj, list)
                and len(prompt_cache_obj) > main_cache_len
            ):
                del prompt_cache_obj[main_cache_len:]

    def _generate_sync(
        self,
        prompt: str,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
        extra_stop_sequences: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[str, bool]:
        return self._generate_sync_streaming(
            prompt,
            on_token=None,
            model_role=model_role,
            max_tokens_override=max_tokens_override,
            extra_stop_sequences=extra_stop_sequences,
        )

    async def generate(
        self,
        prompt: str,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
        extra_stop_sequences: tuple[str, ...] | list[str] | None = None,
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
                        extra_stop_sequences=extra_stop_sequences,
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

    def render_chat_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        model_role: str | None = None,
    ) -> str:
        """Render OpenAI-style chat messages to a prompt string.

        Sprint P4.4 (Apr 26 2026): consumed by the OpenAI-compat bridge
        shim so iPhone clients (Enchanted, etc.) can hit ``/v1/chat/
        completions`` and have the messages converted using the model's
        actual chat_template instead of a hand-rolled string concat.

        Falls back to a plain ``role: content`` join if the tokenizer
        can't apply its template (e.g. on a backend without one).
        """
        norm_role = self._normalize_role(model_role)
        try:
            self._ensure_loaded(norm_role)
        except Exception:
            logger.debug(
                "render_chat_prompt: ensure_loaded raised", exc_info=True,
            )
        tokenizer = self._tokenizers.get(norm_role)
        cleaned: list[dict[str, str]] = []
        for m in messages or []:
            role = str(m.get("role") or "").strip().lower()
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            content = str(content or "")
            if role and content:
                cleaned.append({"role": role, "content": content})
        if not cleaned:
            return ""
        if tokenizer is not None:
            try:
                rendered = tokenizer.apply_chat_template(
                    cleaned,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                logger.debug(
                    "apply_chat_template fell back to plain join",
                    exc_info=True,
                )
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in cleaned
        )

    async def chat_streaming(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str, bool], None] | None = None,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
    ) -> tuple[str, bool]:
        """OpenAI-compatible chat shim. Renders messages then delegates
        to :py:meth:`generate_streaming`. Sprint P4.4 (Apr 26 2026).
        """
        prompt = self.render_chat_prompt(messages, model_role=model_role)
        if not prompt:
            return "", False
        return await self.generate_streaming(
            prompt,
            on_token=on_token,
            model_role=model_role,
            max_tokens_override=max_tokens_override,
        )

    async def generate_streaming(
        self,
        prompt: str,
        on_token: Callable[[str, bool], None] | None = None,
        *,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
        extra_stop_sequences: tuple[str, ...] | list[str] | None = None,
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
                        extra_stop_sequences=extra_stop_sequences,
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

    # ── Sprint C1 — persona-as-pinned-KV-prefix ──────────────────

    def pin_prompt_prefix(
        self,
        prefix_text: str,
        *,
        model_role: str = "fast",
        source_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Warm the prompt-cache trie with ``prefix_text`` as a stable
        prefix that all later ``generate`` calls can reuse for free.

        Returns a dict with keys ``ok``, ``tokens``, ``elapsed_ms`` and
        (on failure) ``reason``. Safe to call multiple times --
        successive calls with identical text are very cheap because the
        trie's nearest-prefix lookup finds 100% of the prefix already
        cached.

        ``source_path`` (optional) registers a file whose mtime can be
        watched via :py:meth:`repin_persona_if_changed`. The intended
        use is ``source_path="config/atom_persona.md"`` so a Boss-
        authored edit to the persona file invalidates and re-pins
        without a process restart.
        """
        if not prefix_text or not isinstance(prefix_text, str):
            return {"ok": False, "reason": "empty prefix"}
        if not _HAS_MLX or not self.is_available():
            return {"ok": False, "reason": "mlx unavailable"}

        role = self._normalize_role(model_role)
        try:
            loaded = self._ensure_loaded(role)
        except Exception:
            logger.debug("pin_prompt_prefix: ensure_loaded raised", exc_info=True)
            return {"ok": False, "reason": "ensure_loaded raised"}
        if not loaded:
            return {"ok": False, "reason": f"role {role!r} not loaded"}

        token_count = 0
        try:
            tokenizer = self._tokenizers.get(role)
            if tokenizer is not None:
                tokens = self._encode_prompt_tokens(tokenizer, prefix_text) or []
                token_count = len(tokens)
        except Exception:
            logger.debug("pin_prompt_prefix: tokenize raised", exc_info=True)

        t0 = time.perf_counter()
        try:
            self._generate_sync(
                prefix_text,
                model_role=role,
                max_tokens_override=1,
            )
        except Exception:
            logger.exception("pin_prompt_prefix: prefill raised")
            return {"ok": False, "reason": "prefill raised"}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        with self._pinned_persona_lock:
            self._pinned_persona_role = role
            self._pinned_persona_token_count = token_count
            if source_path is not None:
                try:
                    p = Path(str(source_path)).expanduser()
                    self._pinned_persona_path = p
                    self._pinned_persona_mtime = (
                        p.stat().st_mtime if p.exists() else 0.0
                    )
                except Exception:
                    logger.debug(
                        "pin_prompt_prefix: source_path stat raised",
                        exc_info=True,
                    )

        logger.info(
            "MLX prompt prefix pinned: role=%s tokens=%d elapsed=%.0fms"
            "%s",
            role,
            token_count,
            elapsed_ms,
            f" source={Path(str(source_path)).name}" if source_path else "",
        )
        return {
            "ok": True,
            "tokens": token_count,
            "elapsed_ms": elapsed_ms,
            "role": role,
        }

    def repin_persona_if_changed(self) -> bool:
        """Re-pin the persona prefix if its source file's mtime moved.

        Returns ``True`` iff a re-pin actually ran. The 99% case where
        nothing has changed is a cheap ``stat()`` + a float compare and
        bails immediately. Designed to be called at the start of every
        turn from the cognitive kernel without measurable overhead.
        """
        with self._pinned_persona_lock:
            path = self._pinned_persona_path
            pinned_mtime = self._pinned_persona_mtime
            role = self._pinned_persona_role
        if path is None:
            return False
        try:
            current_mtime = path.stat().st_mtime
        except FileNotFoundError:
            return False
        except Exception:
            logger.debug("repin_persona_if_changed: stat raised", exc_info=True)
            return False
        if current_mtime <= pinned_mtime:
            return False
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            logger.debug(
                "repin_persona_if_changed: read raised", exc_info=True,
            )
            return False
        logger.info(
            "MLX persona file mtime moved (%.0f -> %.0f) -- re-pinning",
            pinned_mtime, current_mtime,
        )
        result = self.pin_prompt_prefix(
            text,
            model_role=role,
            source_path=str(path),
        )
        return bool(result.get("ok"))

    @property
    def pinned_persona_info(self) -> dict[str, Any]:
        """Diagnostics surface for the cognitive_kernel UI / logs."""
        with self._pinned_persona_lock:
            return {
                "path": str(self._pinned_persona_path) if self._pinned_persona_path else None,
                "mtime": self._pinned_persona_mtime,
                "role": self._pinned_persona_role,
                "tokens": self._pinned_persona_token_count,
            }

    @property
    def dual_tier_info(self) -> dict[str, Any]:
        """Diagnostics surface for Sprint P3.1 dual-tier brain.

        Returns the per-role model paths and whether dual-tier is
        actually live. Surfaced by the unified status badge so an
        operator can tell at a glance whether ``mlx_fast_model``
        actually took effect.
        """
        return {
            "enabled": bool(self._dual_tier_enabled),
            "primary_path": self._role_model_paths.get("primary"),
            "fast_path": self._role_model_paths.get("fast"),
            "draft_path": self._speculative_draft_path,
            "speculative_decoding_enabled": bool(self._speculative_enabled),
        }

    def shutdown(self) -> None:
        with self._load_lock:
            self._unload_role_unlocked("primary")
            self._unload_role_unlocked("fast")
        self.drop_prompt_caches(reason="shutdown")
        self._clear_mlx_cache()

    def close(self) -> None:
        self.shutdown()
        self._executor.shutdown(wait=False)


__all__ = ["MLXBrain"]
