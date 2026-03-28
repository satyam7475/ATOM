# ATOM Module 04: Execution Layer (Ring 4)

> Read this before changing: `core/router/*_actions.py`, `core/desktop_control.py`

## Action Modules

| Module | File | Actions |
|--------|------|---------|
| **system_actions** | `core/router/system_actions.py` | lock, screenshot, brightness, shutdown, restart, sleep, logoff, flush DNS, empty recycle bin |
| **app_actions** | `core/router/app_actions.py` | open, close, list apps |
| **media_actions** | `core/router/media_actions.py` | volume, mute/unmute, play YouTube, stop music |
| **network_actions** | `core/router/network_actions.py` | search, open URL, weather, WiFi status |
| **file_actions** | `core/router/file_actions.py` | create folder, move path, copy path |
| **utility_actions** | `core/router/utility_actions.py` | minimize/maximize/switch window, timer, read clipboard |
| **desktop_control** | `core/desktop_control.py` | scroll, click, press key, type text, hotkey combos |

## Router Dispatch Table

Actions are dispatched via `Router._ACTION_DISPATCH` — an O(1) dict mapping action names to handler methods. Currently 37 actions registered.

## How to Add a New Action

1. **Add intent pattern** in `core/intent_engine/<category>_intents.py`
2. **Add handler function** in `core/router/<category>_actions.py`
3. **Add dispatch method** in `core/router/router.py`:
   ```python
   def _do_my_action(self, _action: str, args: dict) -> str:
       my_actions.do_thing(args.get("param", ""))
       return personality.action_done("my_action")
   ```
4. **Add to dispatch table**: `"my_action": _do_my_action,` in `_ACTION_DISPATCH`
5. **Add to commands.json** if confirmation is needed
6. **Add to SecurityPolicy** if the action is sensitive

## Action Categories

### Fire-and-Forget (response sent BEFORE action)
`open_app, play_youtube, search, lock_screen, screenshot, minimize_window, maximize_window, switch_window, flush_dns, open_url`

### Slow Actions (thinking ack sent first)
`list_apps, resource_report, resource_trend, system_analyze, research_topic, self_check, self_diagnostic, behavior_report`

### Standard (action executed, then response)
All others — action runs, result is the response.

## Confirmation Flow

1. Router checks `_requires_confirmation(result)`
2. If yes → store `_pending_action`, speak confirmation prompt
3. User says "yes"/"no" → `_handle_confirmation()`
4. Pending action expires after 25 seconds

## Security Gate

Every action passes through `SecurityPolicy.allow_action()` BEFORE dispatch:
- Executable allowlist (open_app)
- Close target allowlist (close_app)
- Feature flags (desktop_control, file_ops)
- Lock mode enforcement
- Path safety (file operations)

## Configuration

```json
{
  "features": {
    "desktop_control": true,
    "file_ops": true,
    "system_analyze": true,
    "web_research": false
  },
  "security": {
    "require_confirmation_for": ["shutdown_pc", "close_app", ...]
  }
}
```
