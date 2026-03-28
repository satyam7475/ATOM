# ATOM Module 06: Cognition Layer (Ring 6)

> Read this before changing: `core/cognitive/`

## Modules

| Module | File | Purpose | Data |
|--------|------|---------|------|
| **SecondBrain** | `core/cognitive/second_brain.py` | Unified knowledge store | `logs/second_brain.json` |
| **GoalEngine** | `core/cognitive/goal_engine.py` | Goal → Plan → Execute → Evaluate | `logs/goals.json` |
| **PredictionEngine** | `core/cognitive/prediction_engine.py` | Predict next user action | In-memory (rebuilt from interactions) |
| **BehaviorModel** | `core/cognitive/behavior_model.py` | Personal profile + energy inference | `logs/user_profile.json` |
| **SelfOptimizer** | `core/cognitive/self_optimizer.py` | Suggest ATOM improvements | `logs/optimizer.json` |

## Cognitive Module Contract (ALL modules follow this)

```python
class CognitiveModuleContract:
    def start() -> None     # Begin background asyncio task
    def stop() -> None      # Cancel task + persist state
    def persist() -> None   # Save to disk
```

## SecondBrain — Unified Knowledge Store

Three knowledge sources:
1. **Facts** — learned from conversations, max 500
2. **Preferences** — inferred from behavior (key-value)
3. **Corrections** — voice/text correction patterns

Retrieval: keyword overlap + recency bonus + tag matching. No ML.

## GoalEngine — Goal-Based Intelligence

Lifecycle: `Create → Decompose (LLM) → Track Steps → Evaluate → Briefing`

- **Create:** voice "set a goal to learn Rust"
- **Decompose:** LLM breaks goal into 5-8 actionable steps
- **Track:** log minutes per step, mark steps complete
- **Evaluate:** trajectory = ahead / on_track / behind / stalled
- **Briefing:** daily 7-10 AM TTS summary of active goals
- **Streaks:** consecutive days of progress tracked

Max 20 goals, max 30 steps per goal.

## PredictionEngine — Predict Before You Speak

Two prediction models (no ML, pure frequency):
1. **Time-slot frequency:** action counts per hour × weekday/weekend
2. **Transition probability:** action A → action B within 5 minutes

Predictions emitted as `prediction_ready` event for dashboard display.
Rebuilt from full interaction history every 100 interactions.

## BehaviorModel — Energy Awareness

Energy states inferred from activity (no ML):
- **high:** ≥5 actions/10min, switch rate ≤0.3
- **medium:** ≥2 actions/10min, switch rate ≤0.6
- **low:** less than above
- **resting:** idle >5min or late night (23:00-05:00)

App categorization: deep_work / communication / browsing / productivity / media
Focus session tracking: records duration per app category.

## SelfOptimizer — ATOM Improves Itself

Analyzes every 30 minutes:
- **Unused features** — not used in 7+ days → suggest disabling
- **High fallback rate** — >30% queries hit LLM → suggest new patterns
- **High latency** — >2500ms avg → suggest cache/pattern improvements

SAFETY: Never auto-disables. Only suggests.

## Cognitive Intents (handled via bus)

```
goal_create, goal_show, goal_progress, goal_decompose,
goal_log_progress, goal_complete_step, goal_pause, goal_resume, goal_abandon,
prediction, mode_switch,
cognitive_behavior_report, scheduling_advice,
brain_remember, brain_recall, brain_preferences,
self_optimize
```

## Events

| Event | Emitter | Consumer |
|-------|---------|----------|
| `goal_briefing` | GoalEngine | main.py → TTS |
| `prediction_ready` | PredictionEngine | Dashboard |
| `user_energy_state` | BehaviorModel | PersonalityModes |
| `optimization_suggestions` | SelfOptimizer | Dashboard |
| `mode_changed` | PersonalityModes | TTS, Dashboard |

## Configuration

```json
{
  "cognitive": {
    "enabled": true,
    "goals_enabled": true,
    "predictions_enabled": true,
    "behavior_model_enabled": true,
    "self_optimizer_enabled": true,
    "prediction_check_interval_s": 120,
    "behavior_update_interval_s": 900,
    "goal_evaluation_interval_s": 3600,
    "optimizer_check_interval_s": 1800,
    "default_mode": "work",
    "max_goals": 20,
    "max_predictions": 5
  }
}
```
