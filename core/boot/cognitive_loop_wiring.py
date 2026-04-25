"""
ATOM -- Phase G cognitive-loop wiring.

The Phase G modules (reflective_loop, presence_sampler, scene_context,
mood_inference, jarvis_suggester) are pure components -- they only run
when something explicitly subscribes them to the live bus. This module
is that "something". It is called once from ``main.py`` *after* the
event bus, command loop, vision engine and local brain are all live,
and it returns the assembled handles so callers can hold a reference
(otherwise the modules would be garbage-collected and silently stop
firing).

Each subsystem is independently togglable from
``config["cognitive_loop"]`` so a demo run can disable, say, the
suggester or the presence sampler without touching code.

Owner: Satyam
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.command_loop import CommandLoop
    from core.state_manager import StateManager

logger = logging.getLogger("atom.boot.cognitive")


@dataclass
class CognitiveLoopHandles:
    """Live references to the wired Phase G subsystems."""

    reflective: Any = None
    presence: Any = None
    scene: Any = None
    mood: Any = None
    suggester: Any = None
    awareness: Any = None
    enabled_summary: dict[str, bool] | None = None

    def stop(self) -> None:
        """Best-effort detach for clean shutdown."""
        for handle, name in (
            (self.awareness, "awareness"),
            (self.suggester, "suggester"),
            (self.mood, "mood"),
            (self.scene, "scene"),
            (self.reflective, "reflective"),
        ):
            if handle is None:
                continue
            try:
                detach = getattr(handle, "detach", None)
                if callable(detach):
                    detach()
            except Exception:
                logger.debug("cognitive_loop: detach failed for %s", name, exc_info=True)
        if self.presence is not None:
            try:
                stopper = getattr(self.presence, "stop", None)
                if callable(stopper):
                    import asyncio
                    coro = stopper()
                    if asyncio.iscoroutine(coro):
                        try:
                            asyncio.get_event_loop().create_task(coro)
                        except Exception:
                            pass
            except Exception:
                logger.debug("cognitive_loop: presence stop failed", exc_info=True)


def wire_cognitive_loop(
    *,
    bus: "AsyncEventBus",
    state: "StateManager",
    command_loop: "CommandLoop",
    config: dict,
    local_brain: Any = None,
    vision_engine: Any = None,
    captioner: Any = None,
) -> CognitiveLoopHandles:
    """Assemble + attach the Phase G subsystems. Idempotent on a fresh boot.

    Returns :class:`CognitiveLoopHandles` so the caller (``main.py``)
    can keep references alive and inspect what was actually wired vs
    silently disabled (e.g. presence/scene without a real camera)."""

    handles = CognitiveLoopHandles()
    handles.enabled_summary = {}

    cfg = (config or {}).get("cognitive_loop", {}) or {}
    if not cfg.get("enabled", True):
        logger.info("Cognitive loop disabled in config -- Phase G inactive")
        handles.enabled_summary = {"loop": False}
        return handles

    # ── turn_complete emitter (required by reflective + suggester) ─
    try:
        command_loop.attach_turn_emitter()
        logger.info("Cognitive loop: turn_complete emitter attached")
    except Exception:
        logger.exception("Cognitive loop: failed to attach turn emitter")

    # ── G4: mood inference (cheap, always-on) ──────────────────────
    mood_cfg = cfg.get("mood", {}) or {}
    if mood_cfg.get("enabled", True):
        try:
            from core.cognitive.mood_inference import MoodInferenceEngine
            handles.mood = MoodInferenceEngine(
                bus, min_consecutive=int(mood_cfg.get("min_consecutive", 2)),
            )
            handles.mood.attach()
            handles.enabled_summary["mood"] = True
            logger.info("Cognitive loop: MoodInferenceEngine attached")
        except Exception:
            logger.exception("Cognitive loop: MoodInferenceEngine failed to attach")
            handles.enabled_summary["mood"] = False
    else:
        handles.enabled_summary["mood"] = False

    # ── G5: jarvis suggester (cadence-gated, always-on) ────────────
    sug_cfg = cfg.get("suggester", {}) or {}
    if sug_cfg.get("enabled", True):
        try:
            from core.cognitive.jarvis_suggester import JarvisSuggester
            quiet = sug_cfg.get("quiet_hours", [23, 6])
            quiet_tuple = (int(quiet[0]), int(quiet[1])) if len(quiet) >= 2 else (23, 6)
            handles.suggester = JarvisSuggester(
                bus,
                cooldown_s=float(sug_cfg.get("cooldown_s", 720.0)),
                category_cooldown_s=float(sug_cfg.get("category_cooldown_s", 2700.0)),
                daily_cap=int(sug_cfg.get("daily_cap", 4)),
                relevance_threshold=float(sug_cfg.get("relevance_threshold", 0.7)),
                quiet_hours=quiet_tuple,
                suppress_moods=tuple(sug_cfg.get("suppress_moods", ("frustrated", "focused", "idle"))),
            )
            handles.suggester.attach()
            handles.enabled_summary["suggester"] = True
            logger.info(
                "Cognitive loop: JarvisSuggester attached (cooldown=%.0fs, cap=%d)",
                handles.suggester.metrics.get("cap_today_left", 0)
                if hasattr(handles.suggester, "metrics") else 0,
                int(sug_cfg.get("daily_cap", 4)),
            )
        except Exception:
            logger.exception("Cognitive loop: JarvisSuggester failed to attach")
            handles.enabled_summary["suggester"] = False
    else:
        handles.enabled_summary["suggester"] = False

    # ── G1: reflective loop (needs the local brain) ────────────────
    refl_cfg = cfg.get("reflective", {}) or {}
    if refl_cfg.get("enabled", True) and local_brain is not None:
        try:
            from core.cognitive.reflective_loop import (
                ReflectiveLoop,
                make_default_llm_provider,
            )

            def _state_provider() -> str:
                try:
                    return str(getattr(state.current, "value", state.current) or "").lower()
                except Exception:
                    return ""

            def _execute_emitter(text: str) -> None:
                if not text:
                    return
                try:
                    bus.emit_long("speech_final", text=text, language="en", source="reflective_loop")
                except Exception:
                    logger.debug("reflective execute_emitter failed", exc_info=True)

            # Sprint A1: pass the underlying MLXBrain (which is what
            # ``make_default_llm_provider`` calls ``.generate()`` on),
            # not the wrapping ``LocalBrainController``. The controller
            # exposes the brain at ``_llm``; older test doubles pass the
            # MLX-shaped object directly (have ``.generate`` themselves)
            # so we honour both shapes.
            mlx_brain = getattr(local_brain, "_llm", None)
            if mlx_brain is None or not hasattr(mlx_brain, "generate"):
                if hasattr(local_brain, "generate"):
                    mlx_brain = local_brain
                else:
                    raise RuntimeError(
                        "ReflectiveLoop needs the underlying MLXBrain "
                        "(local_brain._llm). The wrapping "
                        "LocalBrainController doesn't expose .generate()."
                    )
            llm_provider = make_default_llm_provider(
                mlx_brain,
                model_role="fast",
                max_tokens=int(refl_cfg.get("max_tokens", 220)),
            )
            handles.reflective = ReflectiveLoop(
                bus, llm_provider,
                cooldown_s=float(refl_cfg.get("cooldown_s", 60.0)),
                min_user_chars=int(refl_cfg.get("min_user_chars", 5)),
                state_provider=_state_provider,
                execute_emitter=_execute_emitter,
            )
            handles.reflective.attach()
            handles.enabled_summary["reflective"] = True
            logger.info("Cognitive loop: ReflectiveLoop attached")
        except Exception:
            logger.exception("Cognitive loop: ReflectiveLoop failed to attach")
            handles.enabled_summary["reflective"] = False
    else:
        if local_brain is None and refl_cfg.get("enabled", True):
            logger.warning(
                "Cognitive loop: ReflectiveLoop skipped -- local_brain not available",
            )
        handles.enabled_summary["reflective"] = False

    # ── G2: presence sampler (gated by vision availability) ────────
    pres_cfg = cfg.get("presence", {}) or {}
    vision_ready = (
        vision_engine is not None
        and bool(getattr(vision_engine, "enabled", False))
        and not (vision_engine.disabled_reason() if hasattr(vision_engine, "disabled_reason") else "")
    )
    if pres_cfg.get("enabled", True) and vision_ready:
        try:
            from core.perception.presence_sampler import PresenceSampler

            def _state_provider_p() -> str:
                try:
                    return str(getattr(state.current, "value", state.current) or "").lower()
                except Exception:
                    return ""

            def _busy_provider() -> bool:
                try:
                    if command_loop.is_busy:
                        return True
                except Exception:
                    pass
                try:
                    is_capturing = getattr(vision_engine, "is_capturing", None)
                    if callable(is_capturing) and is_capturing():
                        return True
                except Exception:
                    pass
                return False

            handles.presence = PresenceSampler(
                bus,
                interval_s=float(pres_cfg.get("interval_s", 30.0)),
                min_interval_s=float(pres_cfg.get("min_interval_s", 8.0)),
                state_provider=_state_provider_p,
                busy_provider=_busy_provider,
            )
            handles.presence.start()
            handles.enabled_summary["presence"] = True
            logger.info("Cognitive loop: PresenceSampler started")
        except Exception:
            logger.exception("Cognitive loop: PresenceSampler failed to start")
            handles.enabled_summary["presence"] = False
    else:
        if pres_cfg.get("enabled", True) and not vision_ready:
            logger.info("Cognitive loop: PresenceSampler skipped -- vision not ready")
        handles.enabled_summary["presence"] = False

    # ── G3: scene context (needs presence + a captioner) ──────────
    scene_cfg = cfg.get("scene", {}) or {}
    scene_ready = (
        scene_cfg.get("enabled", True)
        and handles.presence is not None
        and captioner is not None
    )
    if scene_ready:
        try:
            from core.perception.scene_context import SceneContextEngine

            def _busy_provider_s() -> bool:
                try:
                    if command_loop.is_busy:
                        return True
                except Exception:
                    pass
                try:
                    is_capturing = getattr(vision_engine, "is_capturing", None)
                    if callable(is_capturing) and is_capturing():
                        return True
                except Exception:
                    pass
                return False

            handles.scene = SceneContextEngine(
                bus, captioner,
                cooldown_s=float(scene_cfg.get("cooldown_s", 300.0)),
                significance_min_seconds=float(scene_cfg.get("significance_min_seconds", 30.0)),
                busy_provider=_busy_provider_s,
            )
            handles.scene.attach()
            handles.enabled_summary["scene"] = True
            logger.info("Cognitive loop: SceneContextEngine attached")
        except Exception:
            logger.exception("Cognitive loop: SceneContextEngine failed to attach")
            handles.enabled_summary["scene"] = False
    else:
        if scene_cfg.get("enabled", True) and captioner is None:
            logger.info("Cognitive loop: SceneContextEngine skipped -- no VLM captioner")
        handles.enabled_summary["scene"] = False

    # ── F1: continuous awareness loop (mood + presence + scene + voice) ──
    aw_cfg = cfg.get("awareness", {}) or {}
    if aw_cfg.get("enabled", True):
        try:
            from core.cognitive.awareness_loop import (
                AwarenessConfig,
                AwarenessLoop,
            )
            handles.awareness = AwarenessLoop(
                bus,
                suggester=handles.suggester,
                state_manager=state,
                config=AwarenessConfig(
                    welcome_back_after_absent_s=float(
                        aw_cfg.get("welcome_back_after_absent_s", 240.0),
                    ),
                    silent_present_warn_s=float(
                        aw_cfg.get("silent_present_warn_s", 1800.0),
                    ),
                    scene_dwell_warn_s=float(
                        aw_cfg.get("scene_dwell_warn_s", 2400.0),
                    ),
                    welcome_back_score=float(
                        aw_cfg.get("welcome_back_score", 0.95),
                    ),
                    silent_present_score=float(
                        aw_cfg.get("silent_present_score", 0.78),
                    ),
                    scene_dwell_score=float(
                        aw_cfg.get("scene_dwell_score", 0.72),
                    ),
                    min_emit_gap_s=float(aw_cfg.get("min_emit_gap_s", 90.0)),
                    enable_direct_welcome_emit=bool(
                        aw_cfg.get("enable_direct_welcome_emit", True),
                    ),
                ),
            )
            handles.awareness.attach()
            handles.enabled_summary["awareness"] = True
            logger.info("Cognitive loop: AwarenessLoop attached")
        except Exception:
            logger.exception("Cognitive loop: AwarenessLoop failed to attach")
            handles.enabled_summary["awareness"] = False
    else:
        handles.enabled_summary["awareness"] = False

    logger.info("Cognitive loop wiring complete: %s", handles.enabled_summary)
    return handles


__all__ = ["CognitiveLoopHandles", "wire_cognitive_loop"]
