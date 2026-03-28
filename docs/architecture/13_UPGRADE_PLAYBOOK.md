# ATOM Module 13: Module Upgrade Playbook

> Read this before replacing, upgrading, or adding any module.

## How to Upgrade ANY Module

### Step 1: Identify the Contract
Find the module's architecture doc (`docs/architecture/`). Check its contract section — your new implementation MUST satisfy every method and event.

### Step 2: Check Event Dependencies
Read `docs/architecture/08_EVENT_BUS.md`. Your module MUST:
- Emit the same events with the same payload shapes
- Handle the same incoming events

### Step 3: Implement the Replacement
Create your new module. Contract checklist:
```
[ ] All contract methods implemented
[ ] All required events emitted
[ ] All consumed events handled
[ ] Shutdown/cleanup works properly
[ ] Persistence backward-compatible
[ ] Config section backward-compatible
```

### Step 4: Wire It In
Modify `main.py` to import your new module. The wiring in `_wire_events()` should work unchanged if your contract is satisfied.

### Step 5: Test
```bash
python -m pytest tests/ -v
```

## Common Upgrade Patterns

### Replacing STT Engine
```python
# In main.py, change:
from voice.stt_async import STTAsync
# to:
from voice.stt_new import NewSTT as STTAsync
# Everything else stays the same if STT Contract is satisfied
```

### Adding a New Action
1. Add regex in `core/intent_engine/<category>_intents.py`
2. Add handler in `core/router/<category>_actions.py`
3. Add dispatch entry in `Router._ACTION_DISPATCH`
4. Add to `config/commands.json` if confirmation needed
5. Add to `SecurityPolicy` if sensitive

### Adding a New Cognitive Module
1. Create `core/cognitive/new_module.py`
2. Implement: `start()`, `stop()`, `persist()`
3. Subscribe to events via `bus.on()` in `start()`
4. In `main.py`, instantiate in cognitive section
5. Add intent patterns to `cognitive_intents.py`
6. Add event handler in cognitive handler block

### Adding a New Event
1. Choose tier: fast/normal/long
2. Document in `docs/architecture/08_EVENT_BUS.md`
3. Use `**_kw` in all handlers for forward compatibility
4. Never block in `emit_fast` handlers

### Adding a New Config Section
1. Add to `config/settings.json`
2. Add validation to `core/config_schema.py`
3. Document in the relevant architecture module doc
4. Pass through to module constructor via `config` dict

## What NOT to Do

- Never bypass SecurityPolicy for action execution
- Never emit events with different payload shapes than documented
- Never add direct module-to-module imports (use the event bus)
- Never store secrets in config/settings.json
- Never auto-execute destructive actions regardless of confidence
