#!/usr/bin/env bash
# ATOM post-boot validator — read-only sanity checks.
# Run from the ATOM repo root:
#   bash .cursor/skills/atom-systems-engineer/scripts/validate_boot.sh

set -u

cd "$(dirname "$0")/../../../.." || {
    echo "validate_boot: could not cd to repo root" >&2
    exit 2
}

PASS=0
FAIL=0
WARN=0

ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL+1)); }
warn() { printf "  \033[33mWARN\033[0m  %s\n" "$1"; WARN=$((WARN+1)); }
hdr()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

hdr "========================  ATOM BOOT VALIDATOR  ========================"

# --- git + runtime triangle ----------------------------------------------
hdr "Runtime · commit · log triangle"
last_commit_time="$(git log -1 --format='%ad' --date=iso 2>/dev/null || echo '')"
if [ -n "$last_commit_time" ]; then
    ok "last commit: $last_commit_time"
else
    warn "not a git repo or no commits"
fi

if [ -f "atomlogs.txt" ]; then
    boot_line="$(head -10 atomlogs.txt | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1 || true)"
    if [ -n "$boot_line" ]; then
        boot_ts="${boot_line:0:19}"
        ok "atomlogs boot: $boot_ts"
        if [ -n "$last_commit_time" ]; then
            commit_epoch="$(date -j -f '%Y-%m-%d %H:%M:%S %z' "$last_commit_time" +%s 2>/dev/null || echo 0)"
            boot_epoch="$(date -j -f '%Y-%m-%d %H:%M:%S' "$boot_ts" +%s 2>/dev/null || echo 0)"
            if [ "$boot_epoch" -gt 0 ] && [ "$commit_epoch" -gt 0 ]; then
                if [ "$boot_epoch" -lt "$commit_epoch" ]; then
                    warn "log is from BEFORE last commit — needs fresh boot to validate"
                else
                    ok "log is from AFTER last commit — valid validation artifact"
                fi
            fi
        fi
    else
        warn "atomlogs.txt has no parseable timestamp"
    fi
else
    warn "no atomlogs.txt yet — fine if ATOM hasn't booted since cleaning logs"
fi

# --- models on disk ------------------------------------------------------
hdr "Models on disk"
model_dirs=(models/*/)
if [ -d "models" ]; then
    count=0
    for d in "${model_dirs[@]}"; do
        [ -d "$d" ] || continue
        size="$(du -sh "$d" 2>/dev/null | awk '{print $1}')"
        ok "$(basename "$d")  ($size)"
        count=$((count+1))
    done
    if [ $count -eq 0 ]; then
        bad "no models found — LLM will not load"
    elif [ $count -gt 2 ]; then
        warn "$count models present — owner prefers lightweight single-LLM stack"
    fi
else
    bad "models/ directory missing"
fi

# --- config sanity -------------------------------------------------------
hdr "config/settings.json sanity"
if [ ! -f "config/settings.json" ]; then
    bad "config/settings.json missing"
else
    python3 - <<'PY'
import json, sys
try:
    c = json.load(open('config/settings.json'))
except Exception as e:
    print(f"  FAIL  cannot parse settings.json: {e}")
    sys.exit(1)

brain = c.get('brain', {})
tts = c.get('tts', {})
stt = c.get('stt', {})
voice = c.get('voice', {})
cloud = c.get('cloud', {})

def ok(msg):   print(f"  \033[32mPASS\033[0m  {msg}")
def bad(msg):  print(f"  \033[31mFAIL\033[0m  {msg}")
def warn(msg): print(f"  \033[33mWARN\033[0m  {msg}")

primary = brain.get('mlx_model') or brain.get('mlx_primary_model', '')
fast = brain.get('mlx_fast_model', '')
fallback = brain.get('mlx_model_fallback', '')
legacy = brain.get('model_path', '')
single_resident = brain.get('single_resident')
if primary: ok(f"brain.mlx_model        = {primary}")
else:       bad("brain.mlx_model not set (also no legacy mlx_primary_model)")
if fast and fast != primary:
    if single_resident:
        ok(f"brain.mlx_fast_model   = {fast} (single_resident on -> evicted on swap)")
    else:
        warn(f"brain.mlx_fast_model = {fast} (single_resident off -> 4B+8B can co-reside)")
elif not fast:
    ok("brain.mlx_fast_model not set (single-model profile, fast aliases primary)")
if fallback and fallback != primary:
    warn(f"brain.mlx_model_fallback = {fallback} -- only used if primary fails to load")
if single_resident is True:
    ok("brain.single_resident = true (one chat model in RAM at a time)")
elif single_resident is False:
    warn("brain.single_resident = false -- siblings can co-reside; verify RAM headroom")
spec = brain.get('speculative_decoding', {}) or {}
if spec.get('enabled') and single_resident:
    bad("speculative_decoding.enabled=true with single_resident=true -- "
        "draft load will be refused, no speedup")
elif spec.get('enabled'):
    warn("speculative_decoding.enabled = true -- target+draft will co-reside")
if legacy and legacy != primary:
    warn(f"brain.model_path (legacy GGUF) = {legacy} -- diverges from mlx_model")

locale = stt.get('locale', '')
if locale == 'en-US': ok(f"stt.locale = {locale}")
elif locale:          warn(f"stt.locale = {locale} (en-US recommended for wake-word)")
else:                 bad("stt.locale not set")

voice_name = tts.get('macos_voice', '')
if voice_name: ok(f"tts.macos_voice = {voice_name}")
else:          warn("tts.macos_voice not set")

mode = voice.get('activation_mode', '')
if mode == 'always_on': ok(f"voice.activation_mode = {mode}")
else:                   warn(f"voice.activation_mode = {mode or '<unset>'} (always_on recommended)")

if brain.get('max_tokens', 0) > 320:
    warn(f"brain.max_tokens = {brain.get('max_tokens')} (≤ 320 recommended for voice)")
else:
    ok(f"brain.max_tokens = {brain.get('max_tokens')}")

if cloud.get('enabled', False):
    warn("cloud.enabled = true — outbound network calls possible")
else:
    ok("cloud.enabled = false (local-only)")
PY
fi

# --- triage latest log ---------------------------------------------------
hdr "Log triage"
triage="$(dirname "$0")/triage_log.py"
if [ -f "atomlogs.txt" ] && [ -f "$triage" ]; then
    python3 "$triage" atomlogs.txt
    rc=$?
    if [ $rc -eq 0 ]; then
        ok "triage: no P0 findings"
    elif [ $rc -eq 1 ]; then
        bad "triage: P0 findings detected — see output above"
    else
        warn "triage: exit=$rc (unexpected)"
    fi
else
    warn "skipping triage (no atomlogs.txt or triage script)"
fi

# --- focused smoke test (quick) ------------------------------------------
hdr "Focused smoke tests (read-only)"
if command -v pytest >/dev/null 2>&1; then
    if [ -f "tests/test_atom_smoke.py" ]; then
        if pytest tests/test_atom_smoke.py -x --tb=no -q 2>&1 | tail -5; then
            ok "tests/test_atom_smoke.py"
        else
            bad "tests/test_atom_smoke.py failed"
        fi
    else
        warn "tests/test_atom_smoke.py not found"
    fi
else
    warn "pytest not in PATH — activate venv first"
fi

# --- summary -------------------------------------------------------------
hdr "Summary"
printf "  PASS=%d  WARN=%d  FAIL=%d\n" "$PASS" "$WARN" "$FAIL"
if [ $FAIL -gt 0 ]; then
    exit 1
elif [ $WARN -gt 0 ]; then
    exit 0
fi
exit 0
