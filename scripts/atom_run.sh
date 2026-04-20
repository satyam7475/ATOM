#!/usr/bin/env bash
# Used by launchd (com.atom.agent.plist): prefer the bundle launcher when it
# self-tests cleanly so native macOS STT keeps bundle usage strings; otherwise
# fall back to the plain venv entrypoint.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHER="${HERE}/ATOM.app/Contents/MacOS/atom_python"
export PYTHONUNBUFFERED=1
export ATOM_APP_BUNDLE="${HERE}/ATOM.app"
export VIRTUAL_ENV="${HERE}/.venv"

if [[ -x "${LAUNCHER}" ]]; then
  if "${LAUNCHER}" -c "import sys; print('ok', sys.executable)" >/dev/null 2>&1; then
    export ATOM_LAUNCH_MODE=bundle
    export ATOM_LAUNCHED_FROM_APP=1
    exec "${LAUNCHER}" "$@"
  fi
fi

export ATOM_LAUNCH_MODE=venv
export ATOM_LAUNCHED_FROM_APP=0
exec "${HERE}/.venv/bin/python" "${HERE}/main.py" "$@"
