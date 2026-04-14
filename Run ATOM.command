#!/usr/bin/env bash
# Double-click in Finder or run from Terminal: starts ATOM via venv (reliable), or bundle launcher when it passes self-test.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER="${HERE}/ATOM.app/Contents/MacOS/atom_python"
VENV_PY="${HERE}/.venv/bin/python"
MAIN_PY="${HERE}/main.py"
LOG="${HERE}/logs/atom_run_command.log"
mkdir -p "${HERE}/logs"

export PYTHONUNBUFFERED=1
export ATOM_APP_BUNDLE="${HERE}/ATOM.app"

if [[ ! -f "${MAIN_PY}" ]]; then
  echo "Missing main.py at ${MAIN_PY}" | tee -a "${LOG}" >&2
  if [[ -t 0 ]]; then read -r -p "Press Enter to close..."; fi
  exit 1
fi

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing venv interpreter: ${VENV_PY}" | tee -a "${LOG}" >&2
  echo "Create it with: python3 -m venv .venv && pip install -r requirements.txt (from ${HERE})" | tee -a "${LOG}" >&2
  if [[ -t 0 ]]; then read -r -p "Press Enter to close..."; fi
  exit 1
fi

export VIRTUAL_ENV="${HERE}/.venv"

USE_BUNDLE=0
if [[ -x "${LAUNCHER}" ]]; then
  echo "=== ATOM bundle launcher self-test (optional) ===" | tee -a "${LOG}"
  SELF=0
  BUNDLE_TEST_OUT="$("${LAUNCHER}" -c "import sys; print('ok', sys.executable)" 2>&1)" || SELF=$?
  echo "${BUNDLE_TEST_OUT}" | tee -a "${LOG}"
  if [[ "${SELF}" -eq 0 ]]; then
    USE_BUNDLE=1
  else
    echo "Bundle launcher self-test failed (exit ${SELF}); using venv Python. Native macOS Speech (SFSpeechRecognizer) needs a working ATOM.app process; use dashboard/browser voice or rebuild/sign the bundle." | tee -a "${LOG}" >&2
  fi
else
  echo "No bundle launcher at ${LAUNCHER}; using venv Python." | tee -a "${LOG}"
fi

if [[ "${USE_BUNDLE}" -eq 1 ]]; then
  export ATOM_LAUNCH_MODE=bundle
  export ATOM_LAUNCHED_FROM_APP=1
  echo "=== Starting ATOM via bundle launcher — leave this window open ===" | tee -a "${LOG}"
  set +e
  "${LAUNCHER}" 2>&1 | tee -a "${LOG}"
  EXIT=${PIPESTATUS[0]}
  set -e
else
  export ATOM_LAUNCH_MODE=venv
  export ATOM_LAUNCHED_FROM_APP=0
  echo "=== Starting ATOM via venv (${VENV_PY}) — leave this window open ===" | tee -a "${LOG}"
  set +e
  "${VENV_PY}" "${MAIN_PY}" 2>&1 | tee -a "${LOG}"
  EXIT=${PIPESTATUS[0]}
  set -e
fi

echo "" | tee -a "${LOG}"
echo "ATOM process ended with exit code: ${EXIT}" | tee -a "${LOG}"
if [[ "${EXIT}" -eq 137 ]] || [[ "${EXIT}" -eq 9 ]]; then
  echo "Note: 137/9 often means the OS sent SIGKILL (memory pressure, or a security kill)." | tee -a "${LOG}" >&2
  echo "Try: close other heavy apps, rebuild the launcher (scripts/build_atom_app_launcher.sh), or keep a copy of ATOM outside iCloud Desktop for signing." | tee -a "${LOG}" >&2
fi

if [[ -t 0 ]]; then
  read -r -p "Press Enter to close this window..."
fi
exit "${EXIT}"
