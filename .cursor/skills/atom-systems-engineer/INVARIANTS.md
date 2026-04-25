# ATOM Invariants — Hard Rules That Must Not Break

Each of these has a playbook behind it (see [PLAYBOOKS.md](PLAYBOOKS.md)). Violating one **always** re-introduces a bug we have already paid for.

If you're about to change a file listed here, read the invariant first. If your change conflicts, stop and discuss with the user.

---

## I-01 · Prompt builder must not emit quotable rule text

**File:** `cursor_bridge/structured_prompt_builder.py`

**Rule:** Rules in the system prompt must be expressed as **negative noun phrases** the model cannot plausibly parrot. No imperatives. No numbered lists of "Do X." Every quotable sentence is a TTS leak waiting to happen.

**Bad (leaks):**
```
Give the final answer only. One short line. If the question is simple, give one short sentence.
```

**Good (doesn't leak):**
```
Final answer only — no preface, no rules, no speaker labels, no "Okay, let me…", no "the user is asking".
```

---

## I-02 · Every TTS utterance hits the echo ring buffer before leaving the synthesizer

**Files:** `voice/tts_macos.py` (primary), any other TTS backend added later

**Rule:** Before `NSSpeechSynthesizer.startSpeakingString_` (or equivalent) is called, the utterance must be pushed into the echo ring buffer via `self._record_spoken(text, now)`. The buffer is consulted by STT to suppress self-echo.

**If you add a new TTS engine**, it must call `_record_spoken()` identically.

---

## I-03 · STT finalization must consult `tts.is_echo()` on every path

**File:** `voice/stt_macos.py`

**Three paths** all promote partials to finals. **All three** must check echo first:

1. `on_stable_partial_promote()`
2. `on_final()` (delegated from SFSpeechRecognizer)
3. `on_interrupt_predicted()` (barge-in prediction path)

Miss any one and ATOM will self-talk under the right timing (see PB-03).

---

## I-04 · Guardrail rewrite ≠ quality rejection

**Files:** `core/router/router.py`, `cursor_bridge/local_brain_controller.py`

**Rule:** When the router rewrites a response (e.g. strips a refused action, swaps in a clarifier), the controller's quality gate must treat the rewrite as **accepted** — not re-gate it through length / profanity / emptiness checks.

A rewritten response is valid by definition; rejecting it cascades to "I lost that answer, Boss" (PB-05).

---

## I-05 · Intent engine cold-start priming is mandatory

**File:** `core/boot/cold_start.py` → `_prime_intent_engine_regexes()`

**Rule:** Every `core/intent_engine/*_intents.py` module must be imported and have its pattern list materialized during boot. A lazy first-match regex compile takes 60–120 ms per class and trips the 50 ms watchdog on the first real query (PB-06).

If you add a new intent module, add it to the priming list.

---

## I-06 · `max_tokens` ≤ 320 for voice turns

**File:** `config/settings.json` → `brain.max_tokens`

**Rule:** Longer outputs almost always contain CoT, prompt leaks, or rambling. Voice UX prefers a second short follow-up turn over one long verbose turn.

- Voice default: **≤ 320 tokens** (currently 320).
- Document-summary / long-task mode: bump to 800 only inside a scoped call, never the global default.

---

## I-07 · `max_action_tier` ≤ 3 unless biometrically verified

**Files:** `core/security_policy.py`, `core/security_tiers.py`

**Rule:** Destructive actions (tier 4+: force-kill by PID, change process priority, delete files, modify system settings) require **active biometric verification within the last 60 seconds**. Without enrollment (voice + face), they fall back to passphrase — never auto-allow.

If you add a new tier-4 action, it must escalate to `SecurityPolicy.require_biometric()` before executing.

---

## I-08 · `config/settings.json` is the single source of truth for tunables

**Rule:** No module may hardcode a model name, voice name, timeout, or threshold that also exists in `settings.json`. If you find a hardcoded value, either (a) route it through config, or (b) delete the config key.

**Especially:**
- Model paths — grep `rg "qwen3-|phi-3|llama" --type py` before any model swap.
- TTS voice names — only from `settings.tts.macos_voice`.
- STT locale — only from `settings.stt.locale`.
- Timeouts — only from `settings.watchdog.*`.

---

## I-09 · Legacy keys must stay consistent or be explicitly disabled

**Files:** `config/settings.json`, loaders under `brain/` and `core/`

**Rule:** Old keys like `brain.model_path` (llama.cpp GGUF path) can still be read by legacy loaders. If present, they must either (a) point to the same model as `mlx_primary_model` in spirit, or (b) be set to an empty string / absent so the legacy loader stays dormant.

Silent divergence → "the model in the log isn't the model in config" (PB-12).

---

## I-10 · Memory pressure tier is cooperative, not advisory

**File:** `core/main.py` (pressure watcher), downstream consumers

**Rule:** When `memory_pct > 80%`, the pressure tier goes from 0 → 1. Consumers **must** respond:
- RAG: reduce `top_k` from 8 to 4.
- Prompt builder: drop the system profile detail block.
- MLX: run a `mlx.metal.clear_cache()` on the next idle.
- Vector store: stop background embedding writes.

Ignoring the signal silently degrades to OOM / swap thrashing (the MacBook Air M5 has unified memory, no dedicated GPU RAM — every byte the LLM wants comes from the same 16 GB).

---

## I-11 · Every shipped voice/LLM change requires a live validation log

**Workflow rule** (not a code rule):

After pushing a commit that touches `voice/`, `brain/`, `cursor_bridge/`, `core/router/`, or `cursor_bridge/structured_prompt_builder.py`, you **must** ask the user for a fresh `atomlogs.txt` from a post-commit boot before declaring the sprint done.

Pre-commit logs prove nothing — they are from an older runtime (see the runtime-commit-log triangle in SKILL.md).

---

## I-12 · No network calls unless `cloud.enabled: true` AND user consented

**File:** `config/settings.json` → `cloud.enabled`

**Rule:** No module may call out to Gemini, OpenAI, HuggingFace inference, news APIs, or any other network endpoint unless:
1. `cloud.enabled` is `true`, AND
2. The specific capability (`cloud.reasoning`, `cloud.search`, `cloud.realtime`) is also `true`, AND
3. The privacy redactor has run on the outbound payload.

Default is local-only. Cloud is opt-in per capability.

---

## I-13 · AsyncEventBus pub/sub — no direct cross-module calls for hot-path events

**Files:** All modules subscribing to `tts_start`, `tts_done`, `user_query`, `speech_final`, `state_changed`

**Rule:** Modules communicate by events. Don't reach into another module and call a method directly on the hot path. This keeps the state machine sane and testable.

Exception: attach-style wiring at boot (`router.attach_tts_echo_guard(tts)`) is allowed because it's one-time setup, not per-turn traffic.

---

## I-14 · No narration comments

**All Python files.**

**Rule:** Comments explain *why* or *constraint* — never *what*. Reviewers will flag and remove narration comments in PRs.

**Bad:**
```python
# Import the module
import foo
# Increment counter
counter += 1
```

**Good:**
```python
# Must import after logging is configured — foo.__init__ logs at module load
import foo
```

---

## I-15 · `from __future__ import annotations` at the top of every new file

**Rule:** All new Python files start with `from __future__ import annotations` to avoid circular-import pain and defer annotation evaluation. Also enables `TYPE_CHECKING` pattern for optional type hints.

---

## When an invariant stops being true

If the system evolves and an invariant is no longer correct, **do not silently remove it**. Update this file with the reason, date, and commit hash, so future agents can trace why the invariant changed.
