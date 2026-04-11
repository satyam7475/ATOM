"""
Identity engine for ATOM's self-model and owner relationship.

This sits between raw owner state and response generation so multiple
subsystems can adapt tone and proactivity in a consistent way.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.owner_understanding import OwnerUnderstanding
    from core.personality_modes import PersonalityModes


class IdentityEngine:
    """ATOM's self-identity and owner relationship model."""

    __slots__ = (
        "_config",
        "_owner",
        "_modes",
        "_system_name",
        "_system_role",
        "_owner_name",
        "_owner_title",
    )

    def __init__(
        self,
        config: dict | None = None,
        owner: OwnerUnderstanding | None = None,
        modes: PersonalityModes | None = None,
    ) -> None:
        self._config = config or {}
        self._owner = owner
        self._modes = modes
        self._system_name = "ATOM"
        self._system_role = "AI OS"
        owner_cfg = self._config.get("owner") or {}
        self._owner_name = str(owner_cfg.get("name", "Satyam") or "Satyam")
        self._owner_title = str(owner_cfg.get("title", "Boss") or "Boss")

    def configure_owner(self, name: str = "Satyam", title: str = "Boss") -> None:
        self._owner_name = str(name or "Satyam")
        self._owner_title = str(title or "Boss")

    def attach_owner(self, owner: OwnerUnderstanding | None) -> None:
        self._owner = owner

    def attach_modes(self, modes: PersonalityModes | None) -> None:
        self._modes = modes

    @staticmethod
    def _time_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    @staticmethod
    def _prompt_verbosity(
        voice_verbosity: str,
        preferred_verbosity: str,
    ) -> str:
        vv = (voice_verbosity or "").strip().lower()
        pv = (preferred_verbosity or "medium").strip().lower()
        if vv in {"silent", "minimal", "terse", "brief"}:
            return "short"
        if pv in {"short", "medium", "long"}:
            return pv
        return "medium"

    def describe_identity(self, context: dict[str, Any] | None = None) -> str:
        ctx = self.get_identity_snapshot(context)
        return (
            f"{ctx['system_name']}, {ctx['owner_name']}'s {ctx['system_role']}. "
            f"Address the owner as {ctx['owner_title']}."
        )

    def get_identity_snapshot(
        self,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = dict(context or {})
        now = datetime.now()
        hour = int(ctx.get("hour", now.hour))

        owner = self._owner
        modes = self._modes

        expertise_hint = ""
        if owner and owner.topics.expertise_areas:
            top = sorted(
                owner.topics.expertise_areas.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            expertise_hint = ", ".join(name for name, _score in top)

        active_project = ""
        if owner and owner.context.active_projects:
            active_project = str(owner.context.active_projects[0].get("name", ""))

        mode_name = str(
            ctx.get(
                "personality_mode",
                modes.current_mode if modes is not None else "work",
            ),
        )
        mode_verbosity = str(
            ctx.get(
                "mode_verbosity",
                modes.verbosity if modes is not None else "full",
            ),
        )
        mode_allows_proactive = bool(
            ctx.get(
                "mode_allows_proactive",
                getattr(getattr(modes, "current_config", None), "proactive_alerts", True)
                if modes is not None
                else True,
            ),
        )

        owner_emotion = str(
            ctx.get(
                "owner_emotion",
                owner.emotion.primary if owner is not None else "neutral",
            ),
        )
        owner_energy = str(
            ctx.get(
                "owner_energy",
                owner.anticipation.current_energy_level if owner is not None else "normal",
            ),
        )
        preferred_verbosity = str(
            ctx.get(
                "preferred_verbosity",
                owner.communication.preferred_response_length if owner is not None else "medium",
            ),
        )
        communication_style = str(
            ctx.get(
                "communication_style",
                (
                    "formal"
                    if owner is not None and owner.communication.formality_level > 0.6
                    else "casual"
                ),
            ),
        )
        uses_humor = bool(
            ctx.get(
                "uses_humor",
                owner.communication.uses_humor if owner is not None else True,
            ),
        )
        relationship_depth = int(
            ctx.get(
                "relationship_depth",
                getattr(owner, "_total_interactions", 0) if owner is not None else 0,
            ),
        )
        should_suggest_break = bool(
            ctx.get(
                "should_suggest_break",
                owner.anticipation.should_suggest_break if owner is not None else False,
            ),
        )
        in_flow_state = bool(
            ctx.get(
                "in_flow_state",
                owner_emotion == "focused" or mode_name == "focus",
            ),
        )

        return {
            "system_name": self._system_name,
            "system_role": self._system_role,
            "owner_name": str(ctx.get("owner_name", self._owner_name)),
            "owner_title": str(ctx.get("owner_title", self._owner_title)),
            "hour": hour,
            "time_of_day": str(ctx.get("time_of_day", self._time_of_day(hour))),
            "owner_emotion": owner_emotion,
            "owner_energy": owner_energy,
            "preferred_verbosity": preferred_verbosity,
            "communication_style": communication_style,
            "uses_humor": uses_humor,
            "active_project": str(ctx.get("active_project", active_project)),
            "expertise_hint": str(ctx.get("expertise_hint", expertise_hint)),
            "relationship_depth": relationship_depth,
            "personality_mode": mode_name,
            "mode_verbosity": mode_verbosity,
            "mode_allows_proactive": mode_allows_proactive,
            "should_suggest_break": should_suggest_break,
            "in_flow_state": in_flow_state,
        }

    def get_voice_profile(
        self,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self.get_identity_snapshot(context)
        tone = "confident"
        verbosity = "normal"
        proactive = True

        emotion = ctx["owner_emotion"]
        mode_name = ctx["personality_mode"]
        hour = int(ctx["hour"])

        if ctx["mode_verbosity"] == "silent" or mode_name == "sleep":
            tone = "gentle"
            verbosity = "silent"
            proactive = False
        elif emotion == "frustrated":
            tone = "calm"
            verbosity = "minimal"
            proactive = False
        elif emotion == "stressed":
            tone = "calm"
            verbosity = "brief"
            proactive = True
        elif emotion == "tired" or hour < 8 or hour >= 23:
            tone = "gentle"
            verbosity = "brief"
            proactive = False
        elif ctx["in_flow_state"]:
            tone = "efficient"
            verbosity = "terse"
            proactive = False
        elif mode_name == "chill":
            tone = "warm"
            verbosity = "normal"
            proactive = True

        if ctx["mode_verbosity"] == "minimal" and verbosity == "normal":
            verbosity = "brief"
        if ctx["preferred_verbosity"] == "short" and verbosity == "normal":
            verbosity = "brief"
        if (
            ctx["preferred_verbosity"] == "long"
            and verbosity == "brief"
            and mode_name == "work"
            and emotion not in {"frustrated", "stressed", "tired"}
            and not ctx["in_flow_state"]
        ):
            verbosity = "normal"

        if not ctx["mode_allows_proactive"] or ctx["in_flow_state"]:
            proactive = False

        return {
            "tone": tone,
            "verbosity": verbosity,
            "proactive": proactive,
            "address": ctx["owner_title"],
            "system_name": ctx["system_name"],
            "owner_name": ctx["owner_name"],
            "owner_title": ctx["owner_title"],
            "mode": mode_name,
            "active_project": ctx["active_project"],
        }

    def get_personality_adjustment(
        self,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self.get_identity_snapshot(context)
        voice = self.get_voice_profile(ctx)
        emotion = ctx["owner_emotion"]
        tone = str(voice.get("tone", "normal"))

        if emotion == "frustrated":
            tone = "supportive"
        elif emotion in ("happy", "excited") and tone not in {"efficient", "calm"}:
            tone = "enthusiastic"

        if not voice.get("proactive", True):
            proactivity = "low"
        elif emotion in ("stressed", "excited"):
            proactivity = "high"
        else:
            proactivity = "normal"

        humor = bool(ctx["uses_humor"]) and emotion not in {"frustrated", "stressed"}
        if tone in {"calm", "gentle", "supportive"}:
            humor = False

        return {
            "tone": tone,
            "verbosity": self._prompt_verbosity(
                str(voice.get("verbosity", "normal")),
                str(ctx.get("preferred_verbosity", "medium")),
            ),
            "formality": ctx["communication_style"],
            "humor": humor,
            "proactivity": proactivity,
            "voice_profile": voice,
            "identity": self.describe_identity(ctx),
        }
