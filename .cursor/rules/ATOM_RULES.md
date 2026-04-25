---
description: ATOM Evolution — Read memory bank before any work
globs: ["**/*.py", "**/*.json", "**/*.md", "**/*.html"]
alwaysApply: true
---

# ATOM Project Rules

## CRITICAL: Read Before Any Work

1. **Use the `atom-systems-engineer` skill** at `.cursor/skills/atom-systems-engineer/SKILL.md` — it is the authoritative engineering methodology for this repo (diagnostic workflow, subsystem map, playbooks, invariants). Read it before any non-trivial change.

2. Before blaming code, check the **runtime · commit · log triangle**: the log's boot timestamp must be AFTER the latest relevant commit. A log from a pre-fix runtime is a validation TODO, not a bug.

3. For hard-won "never break these" rules, see `.cursor/skills/atom-systems-engineer/INVARIANTS.md`. Every violation re-introduces a bug we have already paid for.

4. For symptom → root-cause → fix recipes, see `.cursor/skills/atom-systems-engineer/PLAYBOOKS.md` before freestyling.

5. Useful tools (read-only):
   - `python3 .cursor/skills/atom-systems-engineer/scripts/triage_log.py atomlogs.txt` — structured log summary
   - `bash .cursor/skills/atom-systems-engineer/scripts/validate_boot.sh` — post-boot validator

## Project Identity

- **ATOM** = Satyam's personal cognitive AI OS (JARVIS-level, local-first, voice-driven)
- **Owner** = Satyam ("Boss")
- **Hardware** = MacBook Air M5 (Apple Silicon, Unified Memory, Neural Engine, Metal)
- **Language** = Python 3.11+
- **Codebase** = ~51,400 lines across 158 Python files

## Key Conventions

- All modules communicate via `AsyncEventBus` (pub/sub)
- State machine: SLEEP → IDLE → LISTENING → THINKING → SPEAKING → ERROR_RECOVERY
- Security: every action passes through `SecurityPolicy.allow_action()`
- Config: `config/settings.json` — all tunable parameters
- Logging: `logging.getLogger("atom.<module_name>")`
- Type checking: use `TYPE_CHECKING` guard for circular import prevention
- Error handling: wrap module entry points, never let one module crash the bus

## Code Style

- Docstrings: module-level with purpose, architecture notes, owner attribution
- Imports: `from __future__ import annotations` at top of every file
- Slots: use `__slots__` on hot-path classes
- Async: use `asyncio` for I/O, `ThreadPoolExecutor` for CPU-bound work
- No comments that narrate what code does — only non-obvious intent
