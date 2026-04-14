#!/usr/bin/env bash
# Used by launchd (com.atom.agent.plist): stable venv entry; no dependency on atom_python.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1
export ATOM_APP_BUNDLE="${HERE}/ATOM.app"
export VIRTUAL_ENV="${HERE}/.venv"
export ATOM_LAUNCH_MODE=venv
export ATOM_LAUNCHED_FROM_APP=0
exec "${HERE}/.venv/bin/python" "${HERE}/main.py" "$@"
