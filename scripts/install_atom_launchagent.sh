#!/usr/bin/env bash
# Install ATOM as a per-user launchd agent (macOS).
# Uses scripts/atom_run.sh + .venv/bin/python (stable; does not require atom_python).
# Usage: from repo root — bash scripts/install_atom_launchagent.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="${ROOT}/scripts/com.atom.agent.plist"
DEST="${HOME}/Library/LaunchAgents/com.atom.agent.plist"
RUN_SCRIPT="${ROOT}/scripts/atom_run.sh"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
	echo "This installer is for macOS only." >&2
	exit 1
fi
if [[ ! -f "${TEMPLATE}" ]]; then
	echo "Missing template: ${TEMPLATE}" >&2
	exit 1
fi
if [[ ! -x "${VENV_PY}" ]]; then
	echo "Missing venv Python at ${VENV_PY}. Create with: python3 -m venv .venv && pip install -r requirements.txt" >&2
	exit 1
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
	echo "Missing runner script: ${RUN_SCRIPT}" >&2
	exit 1
fi
chmod +x "${RUN_SCRIPT}"
if ! command -v python3 >/dev/null 2>&1; then
	echo "python3 not found on PATH." >&2
	exit 1
fi

mkdir -p "${ROOT}/logs"
mkdir -p "${HOME}/Library/LaunchAgents"

export ATOM_LA_ROOT="${ROOT}"
export ATOM_LA_TEMPLATE="${TEMPLATE}"
export ATOM_LA_DEST="${DEST}"

python3 <<'PY'
import io
import os
import pathlib
import plistlib
import subprocess
import sys

root = pathlib.Path(os.environ["ATOM_LA_ROOT"])
template = pathlib.Path(os.environ["ATOM_LA_TEMPLATE"])
dest = pathlib.Path(os.environ["ATOM_LA_DEST"])

text = template.read_text(encoding="utf-8")
text = text.replace("@@@ATOM_REPO@@@", str(root))

try:
    plistlib.load(io.BytesIO(text.encode("utf-8")))
except Exception as exc:
    print("Generated plist failed validation:", exc, file=sys.stderr)
    sys.exit(1)

dest.write_text(text, encoding="utf-8")
dest.chmod(0o644)
print("Wrote", dest)

uid = os.getuid()
# Best-effort unload (ignore errors if not loaded)
subprocess.run(
    ["launchctl", "bootout", f"gui/{uid}/com.atom.agent"],
    capture_output=True,
    text=True,
)
r = subprocess.run(
    ["launchctl", "bootstrap", f"gui/{uid}", str(dest)],
    capture_output=True,
    text=True,
)
if r.returncode != 0:
    print("launchctl bootstrap failed:", r.stderr or r.stdout, file=sys.stderr)
    print("Try: launchctl bootout gui/{}/com.atom.agent".format(uid), file=sys.stderr)
    print("Then re-run this script, or use legacy: launchctl load", str(dest), file=sys.stderr)
    sys.exit(r.returncode)
print("Loaded agent com.atom.agent for gui/{}".format(uid))
PY
