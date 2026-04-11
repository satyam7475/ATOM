"""
Focused tests for the identity engine and adaptive personality wiring.

Run: python3 -m tests.test_identity_engine
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Emotion:
    def __init__(self, primary: str = "neutral") -> None:
        self.primary = primary


class _Anticipation:
    def __init__(self, energy: str = "normal", should_suggest_break: bool = False) -> None:
        self.current_energy_level = energy
        self.should_suggest_break = should_suggest_break


class _Communication:
    def __init__(
        self,
        preferred_response_length: str = "medium",
        formality_level: float = 0.3,
        uses_humor: bool = True,
    ) -> None:
        self.preferred_response_length = preferred_response_length
        self.formality_level = formality_level
        self.uses_humor = uses_humor


class _Topics:
    def __init__(self) -> None:
        self.expertise_areas = {"python": 8.0, "systems": 7.2}


class _Context:
    def __init__(self, active_project: str = "ATOM") -> None:
        self.active_projects = [{"name": active_project}]


class _Owner:
    def __init__(
        self,
        emotion: str = "neutral",
        energy: str = "normal",
        preferred_response_length: str = "medium",
        formality_level: float = 0.3,
        uses_humor: bool = True,
        interactions: int = 42,
    ) -> None:
        self.emotion = _Emotion(emotion)
        self.anticipation = _Anticipation(energy)
        self.communication = _Communication(
            preferred_response_length=preferred_response_length,
            formality_level=formality_level,
            uses_humor=uses_humor,
        )
        self.topics = _Topics()
        self.context = _Context()
        self._total_interactions = interactions


class _ModeConfig:
    def __init__(self, proactive_alerts: bool = True) -> None:
        self.proactive_alerts = proactive_alerts


class _Modes:
    def __init__(
        self,
        current_mode: str = "work",
        verbosity: str = "full",
        proactive_alerts: bool = True,
    ) -> None:
        self.current_mode = current_mode
        self.verbosity = verbosity
        self.current_config = _ModeConfig(proactive_alerts=proactive_alerts)


def test_frustrated_voice_profile() -> None:
    from core.identity_engine import IdentityEngine

    engine = IdentityEngine(
        config={"owner": {"name": "Satyam", "title": "Boss"}},
        owner=_Owner(emotion="frustrated"),
        modes=_Modes(),
    )
    profile = engine.get_voice_profile({"hour": 14})
    assert profile["tone"] == "calm"
    assert profile["verbosity"] == "minimal"
    assert profile["proactive"] is False
    print("  PASS: frustrated state yields calm minimal profile")


def test_focus_mode_voice_profile() -> None:
    from core.identity_engine import IdentityEngine

    engine = IdentityEngine(
        config={"owner": {"name": "Satyam", "title": "Boss"}},
        owner=_Owner(emotion="focused"),
        modes=_Modes(current_mode="focus", verbosity="minimal", proactive_alerts=False),
    )
    profile = engine.get_voice_profile({"hour": 11})
    assert profile["tone"] == "efficient"
    assert profile["verbosity"] == "terse"
    assert profile["proactive"] is False
    print("  PASS: focus mode yields terse non-proactive profile")


def test_chill_mode_profile() -> None:
    from core.identity_engine import IdentityEngine

    engine = IdentityEngine(
        config={"owner": {"name": "Satyam", "title": "Boss"}},
        owner=_Owner(emotion="happy", preferred_response_length="long"),
        modes=_Modes(current_mode="chill", verbosity="full"),
    )
    profile = engine.get_voice_profile({"hour": 18})
    assert profile["tone"] == "warm"
    assert profile["verbosity"] == "normal"
    assert profile["proactive"] is True
    print("  PASS: chill mode keeps warm normal profile")


def test_personality_adjustment_maps_prompt_controls() -> None:
    from core.identity_engine import IdentityEngine

    engine = IdentityEngine(
        config={"owner": {"name": "Satyam", "title": "Boss"}},
        owner=_Owner(emotion="stressed", preferred_response_length="long", uses_humor=True),
        modes=_Modes(),
    )
    adjustment = engine.get_personality_adjustment({"hour": 15})
    assert adjustment["tone"] == "calm"
    assert adjustment["verbosity"] == "short"
    assert adjustment["proactivity"] == "high"
    assert adjustment["humor"] is False
    assert "ATOM, Satyam's AI OS" in adjustment["identity"]
    print("  PASS: personality adjustment maps identity to prompt controls")


def test_adaptive_personality_wiring() -> None:
    from core import adaptive_personality as personality

    personality.set_owner("Satyam", "Boss")
    personality.attach_owner(_Owner(emotion="frustrated"))
    personality.attach_modes(_Modes())
    profile = personality.get_voice_profile({"hour": 13})
    assert profile["tone"] == "calm"
    assert profile["owner_title"] == "Boss"
    print("  PASS: adaptive personality uses identity engine profile")


def run_all() -> None:
    test_frustrated_voice_profile()
    test_focus_mode_voice_profile()
    test_chill_mode_profile()
    test_personality_adjustment_maps_prompt_controls()
    test_adaptive_personality_wiring()
    print("\n=== IDENTITY ENGINE TESTS PASSED ===\n")


if __name__ == "__main__":
    run_all()
