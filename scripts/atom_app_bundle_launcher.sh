#!/usr/bin/env bash
# ATOM.app entry: exec the repo venv Python so stdlib/site-packages match (no embedded Py_Main mismatch).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTENTS="$(cd "$HERE/.." && pwd)"
APP="$(cd "$CONTENTS/.." && pwd)"
REPO="$(cd "$APP/.." && pwd)"
MAIN="$REPO/main.py"
VENV_PY="$REPO/.venv/bin/python"

if [[ ! -f "$MAIN" ]]; then
  echo "ATOM: could not find main.py at $MAIN (move ATOM.app next to the ATOM repo folder)." >&2
  exit 6
fi
if [[ ! -x "$VENV_PY" ]]; then
  echo "ATOM: missing venv interpreter at $VENV_PY — create .venv in $REPO" >&2
  exit 6
fi

filtered=()
for a in "$@"; do
  [[ "$a" == -psn_* ]] && continue
  filtered+=("$a")
done

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export VIRTUAL_ENV="$REPO/.venv"
export ATOM_APP_BUNDLE="$APP"
export ATOM_LAUNCHED_FROM_APP=1
export ATOM_LAUNCH_MODE=bundle
# Repo root must precede deps (same as C launcher intent).
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO:$PYTHONPATH"
else
  export PYTHONPATH="$REPO"
fi

cd "$REPO"

if [[ ${#filtered[@]} -eq 0 ]]; then
  exec "$VENV_PY" -u "$MAIN"
else
  exec "$VENV_PY" -u "${filtered[@]}"
fi
