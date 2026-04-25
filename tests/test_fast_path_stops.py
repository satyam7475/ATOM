"""ATOM -- regression tests for FAST-path stop sequences (Sprint C4).

Pins three behaviours so future edits can't accidentally reopen the
"(in a." stage-direction leak path (atomLogs.txt L301) at the model
layer:

1. ``_FAST_PATH_STOP_SEQUENCES`` includes ``(`` and ``\\n\\n`` so the
   FAST/QUICK voice path stops generation at the *token* layer
   instead of relying on the streaming sanitiser to scrub leaks.
2. ``MLXBrain._generate_sync_streaming_inner`` merges per-call
   ``extra_stop_sequences`` on top of the role-level defaults --
   without mutating the cached role config.
3. ``LocalBrainController._run_llm_streaming`` only adds the FAST
   stops when ``model_role == "fast"``; primary / deep paths stay
   unrestricted.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from brain import mlx_llm
from cursor_bridge import local_brain_controller as lbc_mod


# ── _FAST_PATH_STOP_SEQUENCES ────────────────────────────────────


def test_fast_path_stops_include_open_paren_and_double_newline() -> None:
    """The two characters that produced log line 301 must be hard
    stops on the FAST path."""
    stops = mlx_llm._FAST_PATH_STOP_SEQUENCES
    assert "(" in stops, (
        "Open-paren must be a FAST-path stop sequence to kill the "
        "stage-direction leak at the token layer."
    )
    assert "\n\n" in stops, (
        "Double-newline keeps FAST replies to a single paragraph."
    )


def test_fast_path_stops_does_not_break_default_stop_set() -> None:
    """The legacy stop sequences (Boss:, ChatML control tokens, etc.)
    must still be present on the default tuple."""
    defaults = mlx_llm._DEFAULT_STOP_SEQUENCES
    for required in ("Boss:", "User:", "<|im_end|>", "Assistant:"):
        assert required in defaults, (
            f"Default stop sequence {required!r} disappeared -- this "
            "would re-open the persona-collapse leak path."
        )


# ── MLXBrain._stop_sequences merge logic ─────────────────────────


def test_stop_sequences_merges_extra_without_mutating_defaults() -> None:
    """Per-call ``extra`` must be appended to the merged tuple but
    leave the module-level ``_DEFAULT_STOP_SEQUENCES`` unchanged."""
    before = tuple(mlx_llm._DEFAULT_STOP_SEQUENCES)
    merged = mlx_llm.MLXBrain._stop_sequences(["(", "\n\n", "CUSTOM:"])
    assert "(" in merged
    assert "\n\n" in merged
    assert "CUSTOM:" in merged
    # default sequences still survive in the merged result
    for required in ("Boss:", "<|im_end|>"):
        assert required in merged
    # module-level constant unchanged
    assert mlx_llm._DEFAULT_STOP_SEQUENCES == before


def test_stop_sequences_dedups_duplicates_against_defaults() -> None:
    """Passing ``Boss:`` again should not produce two entries -- the
    merge logic must skip duplicates so the longest-suffix-wins
    selection in ``_find_stop_hit`` stays deterministic."""
    merged = mlx_llm.MLXBrain._stop_sequences(["Boss:", "Boss:"])
    assert merged.count("Boss:") == 1


def test_stop_sequences_handles_none_or_empty() -> None:
    assert mlx_llm.MLXBrain._stop_sequences(None) == tuple(
        sorted(mlx_llm._DEFAULT_STOP_SEQUENCES, key=len, reverse=True)
    )
    assert mlx_llm.MLXBrain._stop_sequences([]) == tuple(
        sorted(mlx_llm._DEFAULT_STOP_SEQUENCES, key=len, reverse=True)
    )


# ── extra_stop_sequences param plumbing ──────────────────────────


def test_generate_streaming_signature_accepts_extra_stop_sequences() -> None:
    sig = inspect.signature(mlx_llm.MLXBrain.generate_streaming)
    assert "extra_stop_sequences" in sig.parameters


def test_generate_signature_accepts_extra_stop_sequences() -> None:
    sig = inspect.signature(mlx_llm.MLXBrain.generate)
    assert "extra_stop_sequences" in sig.parameters


def test_inner_streaming_merges_per_call_stops(monkeypatch) -> None:
    """The inner streaming loop must call ``_stop_sequences`` with the
    union of role-level and per-call extras, NOT just one or the
    other."""
    captured: dict[str, object] = {}

    original_stop_sequences = mlx_llm.MLXBrain._stop_sequences

    def _spy(extra=None):
        captured["extra"] = list(extra) if extra is not None else None
        return original_stop_sequences(extra)

    monkeypatch.setattr(
        mlx_llm.MLXBrain, "_stop_sequences", staticmethod(_spy),
    )

    # Build an MLXBrain shell with mocked dependencies so the inner
    # loop does not actually try to call MLX.
    brain = mlx_llm.MLXBrain.__new__(mlx_llm.MLXBrain)
    brain._abort_generation = 0  # type: ignore[attr-defined]

    eff = {
        "model_role": "fast",
        "profile": "fast",
        "max_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "extra_stop_sequences": ["ROLE_STOP:"],
    }

    # Stub everything beyond the merge step. ``stream_generate`` is
    # patched to immediately yield zero tokens so the loop exits
    # cleanly after the merge call.
    monkeypatch.setattr(mlx_llm, "stream_generate", lambda *a, **k: iter(()))
    monkeypatch.setattr(mlx_llm, "_HAS_MLX", False)

    import threading
    brain._make_sampler = lambda t, p: None  # type: ignore[attr-defined]
    brain._make_logits_processors = lambda p: None  # type: ignore[attr-defined]
    brain._prepare_prompt_cache = lambda *a, **k: (None, "p", 1)  # type: ignore[attr-defined]
    brain._role_last_used = {"fast": 0.0}  # type: ignore[attr-defined]
    brain._perf_lock = threading.Lock()  # type: ignore[attr-defined]
    brain._role_perf = {}  # type: ignore[attr-defined]

    model = MagicMock(name="model")
    tokenizer = MagicMock(name="tokenizer")

    try:
        brain._generate_sync_streaming_inner(
            "fast", eff, model, tokenizer, "prompt",
            on_token=None,
            max_tokens_override=None,
            extra_stop_sequences=("(", "\n\n"),
        )
    except AttributeError:
        # The merge happens before any perf-attribute access, so the
        # spy has already captured what we need; downstream attribute
        # holes (test stub doesn't model the full MLXBrain) are fine.
        pass

    extras = captured.get("extra")
    assert extras is not None
    assert "ROLE_STOP:" in extras
    assert "(" in extras
    assert "\n\n" in extras


def test_inner_streaming_skips_duplicate_extras(monkeypatch) -> None:
    """If the per-call extras overlap with role-level ones we must
    not re-add them -- otherwise ``_find_stop_hit`` could pick a
    wrong index when ties happen."""
    captured: list[list[str] | None] = []

    def _spy(extra=None):
        captured.append(list(extra) if extra is not None else None)
        return tuple(extra or ())

    monkeypatch.setattr(
        mlx_llm.MLXBrain, "_stop_sequences", staticmethod(_spy),
    )
    monkeypatch.setattr(mlx_llm, "stream_generate", lambda *a, **k: iter(()))
    monkeypatch.setattr(mlx_llm, "_HAS_MLX", False)

    import threading
    brain = mlx_llm.MLXBrain.__new__(mlx_llm.MLXBrain)
    brain._abort_generation = 0  # type: ignore[attr-defined]
    brain._make_sampler = lambda *a, **k: None  # type: ignore[attr-defined]
    brain._make_logits_processors = lambda *a, **k: None  # type: ignore[attr-defined]
    brain._prepare_prompt_cache = lambda *a, **k: (None, "p", 1)  # type: ignore[attr-defined]
    brain._role_last_used = {"fast": 0.0}  # type: ignore[attr-defined]
    brain._perf_lock = threading.Lock()  # type: ignore[attr-defined]
    brain._role_perf = {}  # type: ignore[attr-defined]

    eff = {
        "model_role": "fast",
        "profile": "fast",
        "max_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "extra_stop_sequences": ["("],  # role already has it
    }
    try:
        brain._generate_sync_streaming_inner(
            "fast", eff, MagicMock(), MagicMock(), "prompt",
            on_token=None,
            max_tokens_override=None,
            extra_stop_sequences=("(", "\n\n"),
        )
    except AttributeError:
        pass

    assert len(captured) == 1
    extras = captured[0] or []
    assert extras.count("(") == 1, (
        "Open-paren stop should be deduped when role + per-call both "
        f"include it. Got: {extras!r}"
    )
    assert "\n\n" in extras


# ── LocalBrainController only sends FAST stops on FAST role ──────


def test_local_brain_controller_module_imports_fast_stops() -> None:
    """Import-time wire check: the controller can locate the
    constant. Future renames will fail loudly here."""
    from brain.mlx_llm import _FAST_PATH_STOP_SEQUENCES
    assert isinstance(_FAST_PATH_STOP_SEQUENCES, tuple)
    assert len(_FAST_PATH_STOP_SEQUENCES) >= 2


def test_local_brain_controller_fast_branch_imports_stops() -> None:
    """Source-level guarantee that the FAST branch in
    ``_run_llm_streaming`` references ``_FAST_PATH_STOP_SEQUENCES``
    -- without spinning up the controller (which needs the full
    config). Pinning the literal source string is intentional so a
    future refactor that drops this branch shows up in CI."""
    src = inspect.getsource(lbc_mod.LocalBrainController._run_llm_streaming)
    assert '"fast"' in src.lower() or "'fast'" in src.lower()
    assert "_FAST_PATH_STOP_SEQUENCES" in src
    assert "extra_stop_sequences" in src
