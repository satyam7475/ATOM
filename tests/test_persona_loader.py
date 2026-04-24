"""Regression tests for the Phase H2 persona loader.

The runtime persona lives in ``config/atom_persona.md`` and is folded
into Layer 1 of the structured prompt. The contract:

  * ``StructuredPromptBuilder`` reads the file once on first build and
    caches the result keyed by mtime;
  * the persona text is appended to the system prompt under a clear
    "# RUNTIME PERSONA" header;
  * the system-prompt hash changes when the persona file changes
    (so the LLM warm-cache invalidates correctly);
  * a missing persona file degrades silently to the baked-in identity;
  * ``set_persona_path`` / ``reload_persona`` give callers a knob.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from cursor_bridge.structured_prompt_builder import (
    StructuredPromptBuilder,
    _load_persona_file,
    _PERSONA_CACHE,
)


_BASE_CFG = {
    "developer": {"role": "OS", "focus": "x", "project_name": "ATOM"},
    "owner": {"name": "Boss"},
    "brain": {"n_ctx": 4096, "max_tokens": 256},
}


def _builder(tmp_path: Path, persona: str | None) -> StructuredPromptBuilder:
    cfg = dict(_BASE_CFG)
    if persona is not None:
        path = tmp_path / "atom_persona.md"
        path.write_text(persona, encoding="utf-8")
        cfg["persona"] = {"path": str(path), "enabled": True}
    else:
        cfg["persona"] = {"enabled": False, "path": ""}
    return StructuredPromptBuilder(cfg)


# ── pure loader ───────────────────────────────────────────────────


def test_load_persona_file_returns_text(tmp_path: Path) -> None:
    p = tmp_path / "p.md"
    p.write_text("hello boss", encoding="utf-8")
    assert _load_persona_file(p) == "hello boss"


def test_load_persona_file_returns_empty_when_missing(tmp_path: Path) -> None:
    assert _load_persona_file(tmp_path / "missing.md") == ""


def test_load_persona_file_truncates_when_too_long(tmp_path: Path) -> None:
    p = tmp_path / "p.md"
    p.write_text("x" * 9000, encoding="utf-8")
    out = _load_persona_file(p)
    assert "[persona truncated]" in out
    assert len(out) <= 9000


def test_load_persona_file_returns_empty_for_blank_path() -> None:
    assert _load_persona_file("") == ""


def test_load_persona_file_uses_mtime_cache(tmp_path: Path) -> None:
    p = tmp_path / "p.md"
    p.write_text("v1", encoding="utf-8")
    original_mtime = p.stat().st_mtime
    assert _load_persona_file(p) == "v1"
    # Force the file to have a different body but the same mtime --
    # the cache must hold (we use mtime-keyed memoization).
    p.write_text("v2", encoding="utf-8")
    os.utime(p, (p.stat().st_atime, original_mtime))
    assert _load_persona_file(p) == "v1"
    # Now bump mtime forward and the loader re-reads from disk.
    os.utime(p, (time.time(), original_mtime + 5))
    assert _load_persona_file(p) == "v2"


# ── system layer wiring ───────────────────────────────────────────


def test_system_layer_includes_persona_when_enabled(tmp_path: Path) -> None:
    builder = _builder(tmp_path, persona="Persona block: speak Hindi.")
    layer = builder._build_system_layer()
    assert "RUNTIME PERSONA" in layer
    assert "Persona block: speak Hindi." in layer


def test_system_layer_excludes_persona_when_disabled(tmp_path: Path) -> None:
    builder = _builder(tmp_path, persona=None)
    layer = builder._build_system_layer()
    assert "RUNTIME PERSONA" not in layer


def test_system_prompt_hash_changes_when_persona_changes(tmp_path: Path) -> None:
    persona_path = tmp_path / "atom_persona.md"
    persona_path.write_text("first persona", encoding="utf-8")
    cfg = dict(_BASE_CFG, persona={"path": str(persona_path), "enabled": True})
    b1 = StructuredPromptBuilder(cfg)
    h1 = b1.system_prompt_hash

    persona_path.write_text("second persona, different cadence", encoding="utf-8")
    os.utime(persona_path, (time.time(), time.time() + 60))

    b2 = StructuredPromptBuilder(cfg)
    h2 = b2.system_prompt_hash
    assert h1 != h2


def test_set_persona_path_swaps_persona_in(tmp_path: Path) -> None:
    p1 = tmp_path / "a.md"
    p1.write_text("alpha persona", encoding="utf-8")
    p2 = tmp_path / "b.md"
    p2.write_text("beta persona", encoding="utf-8")
    cfg = dict(_BASE_CFG, persona={"path": str(p1), "enabled": True})
    builder = StructuredPromptBuilder(cfg)
    assert "alpha persona" in builder._build_system_layer()
    builder.set_persona_path(p2)
    assert "beta persona" in builder._build_system_layer()


def test_reload_persona_rereads_disk(tmp_path: Path) -> None:
    p = tmp_path / "p.md"
    p.write_text("v1", encoding="utf-8")
    cfg = dict(_BASE_CFG, persona={"path": str(p), "enabled": True})
    builder = StructuredPromptBuilder(cfg)
    assert "v1" in builder._build_system_layer()
    p.write_text("v2 persona content", encoding="utf-8")
    os.utime(p, (time.time(), time.time() + 60))
    builder.reload_persona()
    assert "v2 persona content" in builder._build_system_layer()


def test_missing_persona_file_does_not_break_builder(tmp_path: Path) -> None:
    cfg = dict(_BASE_CFG, persona={"path": str(tmp_path / "no.md"), "enabled": True})
    builder = StructuredPromptBuilder(cfg)
    layer = builder._build_system_layer()
    assert layer  # non-empty
    assert "RUNTIME PERSONA" not in layer


def test_default_persona_file_loaded_when_no_config(tmp_path: Path) -> None:
    builder = StructuredPromptBuilder(dict(_BASE_CFG))
    layer = builder._build_system_layer()
    # Repo ships a default persona; it should be present unless missing.
    persona_path = (
        Path(__file__).resolve().parent.parent / "config" / "atom_persona.md"
    )
    if persona_path.exists():
        assert "RUNTIME PERSONA" in layer
        assert "Boss" in layer
    else:
        pytest.skip("default persona not present in repo")


def test_env_override_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "env.md"
    p.write_text("env persona wins", encoding="utf-8")
    monkeypatch.setenv("ATOM_PERSONA_PATH", str(p))
    builder = StructuredPromptBuilder(dict(_BASE_CFG))
    assert "env persona wins" in builder._build_system_layer()
