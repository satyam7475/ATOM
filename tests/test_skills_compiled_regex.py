"""Regression tests for the boot-cached skill trigger regex.

Live log evidence (atom_log.txt L597-599):
    ``yeah give me a summary`` -> skill expand -> ``self check`` ->
    second intent pass costs ~150 ms cold, pushing the
    ``intent_classify`` budget past 250 ms.

The fix has two layers:

1. ``SkillsRegistry`` compiles every trigger into a single union regex
   at boot, so per-call expansion is one regex search instead of an
   O(n) substring loop over every trigger.
2. ``SkillsRegistry.expansion_targets()`` exposes every distinct
   ``expand_to`` plus chain step so the cold-start optimizer can
   pre-classify them and store the results in the command cache.

These tests pin both layers without touching the live registry file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.skills_registry import SkillMatch, SkillsRegistry


# ---------------------------------------------------------------------------
# Compiled regex
# ---------------------------------------------------------------------------


def _write_skills(tmp_path: Path, skills: list[dict]) -> Path:
    path = tmp_path / "skills.json"
    path.write_text(json.dumps({"skills": skills}), encoding="utf-8")
    return path


@pytest.fixture
def registry(tmp_path: Path) -> SkillsRegistry:
    skills = [
        {
            "id": "whats_up",
            "triggers": ["whats up atom", "status report"],
            "expand_to": "self check",
            "chain": [],
        },
        {
            "id": "morning_routine",
            "triggers": ["good morning", "start my day"],
            "expand_to": "open calendar",
            "chain": ["read today's news", "weather report"],
        },
        {
            "id": "play_youtube",
            "triggers": ["play music on youtube"],
            "expand_to": "open youtube",
            "chain": [],
        },
    ]
    cfg = {
        "skills": {
            "enabled": True,
            "path": str(_write_skills(tmp_path, skills)),
        },
    }
    return SkillsRegistry(cfg)


def test_registry_loads_entries(registry: SkillsRegistry) -> None:
    """Sanity: the registry actually parsed the test fixture."""
    assert registry._enabled is True
    assert len(registry._entries) == 5  # 2 + 2 + 1 triggers
    # Compiled regex must exist after a successful load.
    assert registry._compiled_re is not None


def test_compiled_regex_matches_known_trigger(
    registry: SkillsRegistry,
) -> None:
    match = registry.try_expand_full("whats up atom")
    assert isinstance(match, SkillMatch)
    assert match.primary == "self check"
    assert match.skill_id == "whats_up"


def test_compiled_regex_matches_inside_a_sentence(
    registry: SkillsRegistry,
) -> None:
    """The regex anchors with ``\\W`` so the trigger only fires when
    it stands alone — but mid-sentence is still allowed when the
    boundary is whitespace / punctuation."""
    match = registry.try_expand_full("hey atom, status report please")
    assert isinstance(match, SkillMatch)
    assert match.skill_id == "whats_up"


def test_compiled_regex_prefers_longer_trigger(tmp_path: Path) -> None:
    """When two triggers overlap, the longer (more specific) one wins
    because we sort triggers longest-first when building the regex."""
    skills = [
        {
            "id": "short_one",
            "triggers": ["status"],
            "expand_to": "ping",
        },
        {
            "id": "long_one",
            "triggers": ["status report"],
            "expand_to": "self check",
        },
    ]
    cfg = {
        "skills": {
            "enabled": True,
            "path": str(_write_skills(tmp_path, skills)),
        },
    }
    reg = SkillsRegistry(cfg)
    match = reg.try_expand_full("status report")
    assert match is not None
    assert match.skill_id == "long_one"
    assert match.primary == "self check"


def test_compiled_regex_returns_none_on_miss(
    registry: SkillsRegistry,
) -> None:
    assert registry.try_expand_full("tell me a joke about quantum physics") is None


def test_compiled_regex_skips_self_loop(
    registry: SkillsRegistry,
) -> None:
    """If the user types the expansion itself ("self check"), the
    registry must NOT return a match — otherwise we'd loop forever."""
    assert registry.try_expand_full("self check") is None


def test_disabled_registry_returns_none(tmp_path: Path) -> None:
    skills = [{
        "id": "noop",
        "triggers": ["status"],
        "expand_to": "self check",
    }]
    cfg = {
        "skills": {
            "enabled": False,
            "path": str(_write_skills(tmp_path, skills)),
        },
    }
    reg = SkillsRegistry(cfg)
    assert reg.try_expand_full("status") is None


def test_empty_text_returns_none(registry: SkillsRegistry) -> None:
    assert registry.try_expand_full("") is None
    assert registry.try_expand_full("   ") is None


def test_try_expand_legacy_signature(registry: SkillsRegistry) -> None:
    """The 2-tuple legacy signature must remain backward compatible."""
    out = registry.try_expand("status report")
    assert out is not None
    expand, sid = out
    assert expand == "self check"
    assert sid == "whats_up"


# ---------------------------------------------------------------------------
# expansion_targets()
# ---------------------------------------------------------------------------


def test_expansion_targets_includes_expand_and_chain(
    registry: SkillsRegistry,
) -> None:
    targets = registry.expansion_targets()
    assert "self check" in targets
    assert "open calendar" in targets
    assert "read today's news" in targets
    assert "weather report" in targets
    assert "open youtube" in targets


def test_expansion_targets_deduplicates(tmp_path: Path) -> None:
    """Two skills sharing the same expand_to or chain step must not
    pollute the cold-start pre-classify pass with duplicates."""
    skills = [
        {
            "id": "a",
            "triggers": ["alpha"],
            "expand_to": "open calendar",
        },
        {
            "id": "b",
            "triggers": ["beta"],
            "expand_to": "self check",
            "chain": ["open calendar"],
        },
    ]
    cfg = {
        "skills": {
            "enabled": True,
            "path": str(_write_skills(tmp_path, skills)),
        },
    }
    reg = SkillsRegistry(cfg)
    targets = reg.expansion_targets()
    assert targets.count("open calendar") == 1


def test_expansion_targets_empty_when_no_skills(tmp_path: Path) -> None:
    cfg = {
        "skills": {
            "enabled": True,
            "path": str(_write_skills(tmp_path, [])),
        },
    }
    reg = SkillsRegistry(cfg)
    assert reg.expansion_targets() == []


# ---------------------------------------------------------------------------
# Cold-start integration (smoke)
# ---------------------------------------------------------------------------


def test_cold_start_accepts_skills_registry_kwarg() -> None:
    """The constructor must accept ``skills_registry`` so main.py can
    wire it without a TypeError."""
    from core.boot.cold_start import ColdStartOptimizer

    class _StubIntent:
        def classify(self, text: str) -> object:
            return type("R", (), {"intent": "self_check"})()

        def classify_silent(self, text: str) -> object:
            return self.classify(text)

    class _StubMemory:
        def get_top_commands(self, limit: int) -> list[str]:
            return []

    class _StubRegistry:
        def expansion_targets(self) -> list[str]:
            return ["self check", "open calendar"]

    opt = ColdStartOptimizer(
        config={},
        state_manager=None,
        memory_store=_StubMemory(),
        intent_engine=_StubIntent(),
        skills_registry=_StubRegistry(),
    )
    assert opt._skills_registry is not None


@pytest.mark.asyncio
async def test_cold_start_caches_skill_expansions() -> None:
    """``_cache_skill_expansions`` must classify each registry target
    and store the result in the command cache."""
    from core.boot.cold_start import ColdStartOptimizer
    from core.command_cache import get_command_cache

    classified: list[str] = []

    class _SpyIntent:
        def classify_silent(self, text: str) -> object:
            classified.append(text)
            return type("R", (), {"intent": "self_check"})()

        def classify(self, text: str) -> object:
            return self.classify_silent(text)

    class _StubMemory:
        def get_top_commands(self, limit: int) -> list[str]:
            return []

    class _StubRegistry:
        def expansion_targets(self) -> list[str]:
            return ["self check", "open calendar", "weather report"]

    opt = ColdStartOptimizer(
        config={},
        state_manager=None,
        memory_store=_StubMemory(),
        intent_engine=_SpyIntent(),
        skills_registry=_StubRegistry(),
    )
    cached = await opt._cache_skill_expansions()
    assert cached == 3
    # Each target was classified exactly once.
    assert sorted(classified) == sorted(
        ["self check", "open calendar", "weather report"],
    )
    # And every target now lives in the command cache.
    cmd_cache = get_command_cache()
    for target in ["self check", "open calendar", "weather report"]:
        assert cmd_cache.get(target) is not None


@pytest.mark.asyncio
async def test_cold_start_skill_expansions_no_op_when_unwired() -> None:
    """Without a registry, the warmer must return 0 and not error."""
    from core.boot.cold_start import ColdStartOptimizer

    class _StubIntent:
        def classify(self, text: str) -> object:
            raise AssertionError("intent must not be called when registry is None")

    class _StubMemory:
        def get_top_commands(self, limit: int) -> list[str]:
            return []

    opt = ColdStartOptimizer(
        config={},
        state_manager=None,
        memory_store=_StubMemory(),
        intent_engine=_StubIntent(),
        skills_registry=None,
    )
    cached = await opt._cache_skill_expansions()
    assert cached == 0
