# ATOM Module 02: Understanding Layer (Ring 2)

> Read this before changing: `core/intent_engine/`, `core/command_cache.py`, `core/skills_registry.py`

## Modules

| Module | File | Latency |
|--------|------|---------|
| **IntentEngine** | `core/intent_engine/__init__.py` | <5ms total |
| **IntentResult** | `core/intent_engine/base.py` | — (DTO) |
| **CommandCache** | `core/command_cache.py` | O(1) |
| **SkillsRegistry** | `core/skills_registry.py` | O(n) scan |

## IntentEngine Contract

```python
class IntentEngineContract:
    def classify(text: str) -> IntentResult   # <5ms, MUST be synchronous
    def quick_match(text: str) -> str | None  # Used by STT for early exit
```

## IntentResult DTO

```python
class IntentResult:
    intent: str           # e.g. "open_app", "time", "fallback"
    response: str | None  # Pre-built response (for info intents)
    action: str | None    # Action to dispatch (for action intents)
    action_args: dict | None  # Arguments for the action
```

## Intent Sub-Module Map (classification order)

| Priority | Module | File | Handles |
|----------|--------|------|---------|
| 1 | `meta_intents` | `meta_intents.py` | greeting, thanks, exit, go_silent, confirm/deny |
| 2 | `runtime_mode_intents` | `runtime_mode_intents.py` | perf mode, brain profile, assistant mode |
| 3 | `os_intents` (self_check) | `os_intents.py` | self-check shortcut |
| 4 | `info_intents` | `info_intents.py` | time, date, CPU, RAM, battery, disk, IP, uptime |
| 5 | `system_intents` | `system_intents.py` | lock, screenshot, shutdown, brightness |
| 6 | `media_intents` | `media_intents.py` | volume, mute, play, stop |
| 7 | `desktop_intents` | `desktop_intents.py` | scroll, click, press key, type |
| 8 | `file_intents` | `file_intents.py` | create folder, move, copy |
| 9 | `network_intents` | `network_intents.py` | search, URL, weather, research |
| 10 | `os_intents` (full) | `os_intents.py` | diagnostics, kill process, reminders |
| 11 | `cognitive_intents` | `cognitive_intents.py` | goals, predictions, brain recall |
| 12 | `app_intents` | `app_intents.py` | open/close/list apps (last = catches wide) |

**Order matters.** Meta intents (greeting/exit) are checked first to prevent false matches. App intents are last because they use broad patterns.

## How to Add a New Intent

1. Choose the right sub-module based on category
2. Add a regex pattern with named groups for arguments
3. Return `IntentResult(intent="your_intent", action="your_action", action_args={...})`
4. Add matching action handler to the Router (see Module 04)
5. Test: `IntentEngine().classify("your test phrase")`

## Skills Registry

Skills expand phrases before intent classification:

```json
{
  "id": "morning_stack",
  "triggers": ["start my day", "morning routine"],
  "expand_to": "open chrome",
  "chain": ["open teams", "open outlook"]
}
```

The `expand_to` is classified first, then `chain` steps execute sequentially with 800ms delay.

## Grammar Words

`base.py` contains `GRAMMAR_WORDS` — command vocabulary for intent matching. When adding new commands, add relevant words here too.

## Configuration

```json
{
  "skills": { "enabled": true, "path": "config/skills.json" }
}
```
