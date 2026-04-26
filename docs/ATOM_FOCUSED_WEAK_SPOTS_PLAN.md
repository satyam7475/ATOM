# ATOM Focus Plan: Jarvis/Friday Owner Experience

Updated: 2026-04-26

## Current Rating

ATOM is now a strong personal AI OS prototype: roughly **7.2/10 for a practical local Jarvis-style assistant** after the cleanup sprint, assuming the next fresh boot confirms the old memory and STT bottlenecks improved.

Against movie Jarvis/Friday, ATOM is still closer to **30%**: the architecture is real, local, and useful, but the remaining gap is latency, seamless perception, trusted autonomy, and always-correct conversational grounding.

## Baseline From Old Boot Log

Measured from `atomCurrentLogs.txt` before the current config/latency cleanup:

- Boot readiness: **24.8s**
- STT pipeline ready: **22.7s**
- WhisperKit preload: **11.3s**
- TTS ready: **1.1s**
- Local brain ready: **4.3s**
- Cold start bootstrap: **6.0s**
- Memory pressure: **81.8-82%**, tier 1 triggered
- Voice pipeline: active, WhisperKit listening
- Known UX issue: polite partial `"Thank you."` entered interrupt handling

Use `scripts/jarvis_scorecard.py atomCurrentLogs.txt` after every fresh boot to replace this baseline with current numbers.

## North-Star Targets

- Boss speaks, ATOM answers simple requests naturally in **1.5-2.5s** after the voice loop is ready.
- Fresh boot reaches useful voice readiness in **under 18s**, then trends toward **under 15s**.
- STT pipeline readiness stays **under 12s** after WhisperKit cache is warm.
- Idle memory settles **below 75%** on the 16 GB MacBook Air.
- Prompt leaks, CoT leaks, and false polite interrupts stay at **0**.
- Autonomy feels aware but quiet: useful suggestions only, no noisy background chatter.

## Sprint 1: Voice + Latency Stability

Status: in progress.

Changes already made:

- Overlapped WhisperKit model preload with audio intelligence in `main.py`.
- Kept prompt KV cache during tier-1 boot pressure; eviction now waits for higher pressure.
- Prevented duplicate WhisperKit `speech_*` events.
- Registered short TTS acknowledgements before the speak lock so echo suppression can catch racing STT partials.
- Blocked non-explicit THINKING partials like `"Thank you."` from cancelling an answer.
- Added query-scoped tool prompt slimming so simple questions do not carry the full tool catalogue.

Next validation:

- Restart ATOM and capture a fresh `atomCurrentLogs.txt`.
- Run `python3 scripts/jarvis_scorecard.py atomCurrentLogs.txt`.
- Pass bar: score >= 80, voice active, STT listening, no prompt/CoT leak candidates, no polite interrupt candidates, memory below 80% on first fresh boot.

## Sprint 2: Memory Pressure + Idle Calm

Status: prepared.

Changes already made:

- Disabled `cross_device.enabled` and `realtime.enabled` by default.
- Kept one resident local LLM via `brain.single_resident=true`.
- Kept SmolVLM installed but lazy with `vision.vlm.warm_at_boot=false`.
- Slowed screen perception and cognitive loop intervals for a quieter balanced profile.

Next work:

- Confirm fresh idle memory after 5 minutes.
- If memory is still above 75%, reduce or gate `screen_perception_loop`, `SystemStateEngine` polling, and embedding warm-cache residency before touching the LLM.
- Only then consider deleting old Whisper fallback models or regenerable caches with owner approval.

## Sprint 3: Jarvis Scorecard Discipline

Status: implemented initial version.

Track every sprint:

- Boot total
- STT ready time
- WhisperKit preload time
- TTS ready time
- Local brain warm time
- Memory high-water mark
- Prompt/CoT leak candidates
- Echo suppression count
- Polite interrupt candidates
- Optional network/background loops that started

Goal: ATOM should not gain features unless the scorecard stays stable or improves.

## Sprint 4: Trusted Autonomy

Status: next after voice/memory validation.

Focus:

- Raise proactive thresholds instead of adding more autonomous engines.
- Prefer owner-confirmed actions for anything that changes files, apps, system state, or schedules.
- Evaluate suggestions with acceptance rate and false-nudge rate.
- Keep daily suggestion count low until usefulness is proven.

Pass bar:

- ATOM gives at most a few high-signal proactive suggestions per day.
- No autonomous action surprises Boss.
- Every proactive action has a visible reason and a clean dismissal path.

## Sprint 5: Seamless Perception

Status: later, after latency is stable.

Focus:

- Keep camera and VLM on demand.
- Use screen summaries only when useful, not as a constant background drain.
- Prefer fast Apple Vision OCR for screen facts; load SmolVLM only for real visual understanding.

Pass bar:

- Asking “what do you see?” or “what’s on my screen?” works reliably.
- Vision does not trigger memory pressure by default.
- Perception improves responses without making ATOM feel watched or sluggish.

## Operating Rule

Voice and responsiveness beat feature count. If a change makes ATOM more “intelligent” but slower, noisier, or more memory hungry, it waits until the scorecard proves there is headroom.
