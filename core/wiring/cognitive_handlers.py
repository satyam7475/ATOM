"""ATOM -- Cognitive layer event handler wiring.

Handles all Ring 6 cognitive intents: goals, predictions, mode switching,
behavior reports, brain memory, and self-optimization.

Extracted from main.py for testability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.cognitive.goal_engine import GoalEngine
    from core.cognitive.prediction_engine import PredictionEngine
    from core.cognitive.behavior_model import BehaviorModel
    from core.cognitive.self_optimizer import SelfOptimizer
    from core.cognitive.second_brain import SecondBrain
    from core.personality_modes import PersonalityModes

logger = logging.getLogger("atom.wiring.cognitive")


def wire(
    *,
    bus: AsyncEventBus,
    goal_engine: GoalEngine,
    prediction_engine: PredictionEngine,
    behavior_model: BehaviorModel,
    self_optimizer: SelfOptimizer,
    second_brain: SecondBrain,
    personality_modes: PersonalityModes,
    indicator: Any,
    tts: Any,
    web_dashboard: Any | None = None,
) -> None:
    """Register all cognitive layer event handlers on the bus."""

    async def _on_cognitive_intent(intent: str = "", **kw) -> None:
        args = kw.get("action_args", {}) or {}

        if intent == "goal_create":
            title = args.get("title", "")
            if title:
                result = goal_engine.create_goal(title)
                if "error" in result:
                    bus.emit_long("response_ready", text=result["error"])
                else:
                    msg = f"Goal set: '{title}'. Say 'break down this goal' for a step plan."
                    bus.emit_long("response_ready", text=msg)
            return

        if intent == "goal_show":
            summary = goal_engine.format_goals_summary()
            bus.emit_long("response_ready", text=summary)
            return

        if intent == "goal_progress":
            target = args.get("target", "")
            if target:
                goal = goal_engine.find_goal(target)
                if goal:
                    ev = goal.get("evaluation", {})
                    pct = int(goal.get("progress", 0) * 100)
                    msg = (
                        f"Goal '{goal['title']}': {pct}% done. "
                        f"Trajectory: {ev.get('trajectory', 'unknown')}. "
                        f"Streak: {goal.get('streak_days', 0)} days."
                    )
                    bus.emit_long("response_ready", text=msg)
                else:
                    bus.emit_long("response_ready", text="I couldn't find that goal, Boss.")
            else:
                briefing = goal_engine.get_daily_briefing()
                bus.emit_long("response_ready", text=briefing or "No active goals to report on.")
            return

        if intent == "goal_decompose":
            active = goal_engine.get_active_goals()
            if active:
                bus.emit_long("response_ready", text="Breaking down your latest goal with AI...")
                result = await goal_engine.decompose_with_llm(active[-1]["id"])
                bus.emit_long("response_ready", text=result)
            else:
                bus.emit_long("response_ready", text="No active goals to decompose, Boss.")
            return

        if intent == "goal_log_progress":
            topic = args.get("topic", "")
            minutes = args.get("minutes", 30)
            active = goal_engine.get_active_goals()
            if active:
                goal = active[0]
                steps = goal.get("steps", [])
                matched_step = None
                for s in steps:
                    if topic.lower() in s["title"].lower():
                        matched_step = s
                        break
                if matched_step:
                    result = goal_engine.log_progress(goal["id"], matched_step["id"], minutes)
                    bus.emit_long("response_ready", text=result)
                elif steps:
                    result = goal_engine.log_progress(goal["id"], steps[0]["id"], minutes)
                    bus.emit_long("response_ready", text=result)
                else:
                    bus.emit_long("response_ready",
                                  text=f"Logged {minutes} minutes. Add steps to track properly.")
            else:
                bus.emit_long("response_ready", text="No active goals to log progress on.")
            return

        if intent == "goal_complete_step":
            step_name = args.get("step_name", "")
            active = goal_engine.get_active_goals()
            for goal in active:
                for step in goal.get("steps", []):
                    if step_name.lower() in step["title"].lower():
                        result = goal_engine.complete_step(goal["id"], step["id"])
                        bus.emit_long("response_ready", text=result)
                        return
            bus.emit_long("response_ready", text="Couldn't find that step, Boss.")
            return

        if intent == "goal_pause":
            target = args.get("target", "")
            goal = goal_engine.find_goal(target) if target else None
            if goal:
                result = goal_engine.pause_goal(goal["id"])
                bus.emit_long("response_ready", text=result)
            else:
                bus.emit_long("response_ready", text="Goal not found, Boss.")
            return

        if intent == "goal_resume":
            target = args.get("target", "")
            goal = goal_engine.find_goal(target) if target else None
            if goal:
                result = goal_engine.resume_goal(goal["id"])
                bus.emit_long("response_ready", text=result)
            else:
                bus.emit_long("response_ready", text="Goal not found, Boss.")
            return

        if intent == "goal_abandon":
            target = args.get("target", "")
            goal = goal_engine.find_goal(target) if target else None
            if goal:
                result = goal_engine.abandon_goal(goal["id"])
                bus.emit_long("response_ready", text=result)
            else:
                bus.emit_long("response_ready", text="Goal not found, Boss.")
            return

        if intent == "prediction":
            summary = prediction_engine.format_predictions()
            bus.emit_long("response_ready", text=summary)
            return

        if intent == "mode_switch":
            mode = args.get("mode", "work")
            result = personality_modes.switch_mode(mode)
            bus.emit_long("response_ready", text=result)
            return

        if intent == "cognitive_behavior_report":
            report = behavior_model.get_profile_summary()
            bus.emit_long("response_ready", text=report)
            return

        if intent == "scheduling_advice":
            advice = behavior_model.get_scheduling_advice()
            bus.emit_long("response_ready", text=advice)
            return

        if intent == "brain_remember":
            fact = args.get("fact", "")
            if fact:
                second_brain.learn_fact(fact, source="voice")
                bus.emit_long("response_ready",
                              text=f"Got it, Boss. I'll remember: {fact[:80]}")
            return

        if intent == "brain_recall":
            query = args.get("query", "")
            if query:
                results = second_brain.retrieve(query, k=3)
                if results:
                    formatted = ". ".join(r for r in results)
                    bus.emit_long("response_ready",
                                  text=f"Here's what I know: {formatted}")
                else:
                    bus.emit_long("response_ready",
                                  text="I don't have anything stored on that yet, Boss.")
            return

        if intent == "brain_preferences":
            prefs = second_brain.preferences
            if prefs:
                items = [f"{k}: {v}" for k, v in list(prefs.items())[:8]]
                bus.emit_long("response_ready",
                              text=f"Your preferences: {', '.join(items)}")
            else:
                bus.emit_long("response_ready",
                              text="No preferences stored yet. I'm still learning, Boss.")
            return

        if intent == "self_optimize":
            report = self_optimizer.format_optimization_report()
            bus.emit_long("response_ready", text=report)
            return

    bus.on("intent_classified", _on_cognitive_intent)

    async def _on_habit_suggestion_mode_gate(text: str = "", **kw) -> None:
        if personality_modes and not personality_modes.should_allow_suggestion():
            personality_modes.queue_suggestion({"text": text, **kw})
            return
    bus.on("habit_suggestion", _on_habit_suggestion_mode_gate)

    async def _on_cursor_response_for_brain(query: str = "", response: str = "", **_kw) -> None:
        if "goal_decompose:" not in query and len(response) > 20:
            second_brain.learn_fact(
                f"Q: {query[:100]} A: {response[:200]}",
                source="llm_conversation",
                tags=["conversation"],
            )
    bus.on("cursor_response", _on_cursor_response_for_brain)

    async def _on_goal_briefing(text: str = "", **_kw) -> None:
        if text:
            indicator.add_log("info", f"[briefing] {text}")
            bus.emit_long("response_ready", text=text)
    bus.on("goal_briefing", _on_goal_briefing)

    async def _on_goal_update(
        goal_id: str = "",
        action: str = "",
        title: str = "",
        **_kw,
    ) -> None:
        """Reflect GoalEngine changes in indicator + dashboard (Ring 6 → 5)."""
        label = title or goal_id or "goal"
        indicator.add_log("info", f"[goal {action}] {label}")
        if web_dashboard is not None:
            try:
                web_dashboard.broadcast_goals(goal_engine.get_goals_for_dashboard())
            except Exception:
                logger.debug("Dashboard goals broadcast failed", exc_info=True)

    bus.on("goal_update", _on_goal_update)

    async def _on_mode_changed(mode: str = "", **_kw) -> None:
        indicator.add_log("action", f"Mode: {mode.upper()}")
        if hasattr(tts, "_rate_override"):
            rate_adj = _kw.get("voice_rate_adj", 0)
            tts._rate_override = f"{rate_adj:+d}%" if rate_adj else None
        if web_dashboard is not None:
            web_dashboard.broadcast_mode(personality_modes.get_mode_for_dashboard())
    bus.on("mode_changed", _on_mode_changed)

    async def _on_prediction_ready(predictions: list = None, **_kw) -> None:
        if web_dashboard is not None and predictions:
            web_dashboard.broadcast_predictions(predictions)
    bus.on("prediction_ready", _on_prediction_ready)

    async def _on_optimization_suggestions(suggestions: list = None, **_kw) -> None:
        if suggestions:
            logger.info("Self-optimizer: %d suggestions generated", len(suggestions))
            for s in suggestions[:2]:
                indicator.add_log("info", f"[optimize] {s.get('message', '')}")
    bus.on("optimization_suggestions", _on_optimization_suggestions)

    logger.info("Cognitive event handlers wired (goals, predictions, brain, optimizer)")
