#!/usr/bin/env bash
# Install ATOM as a per-user launchd agent (macOS).
# Usage: from repo root — bash scripts/install_atom_launchagent.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="${ROOT}/scripts/com.atom.agent.plist"
DEST="${HOME}/Library/LaunchAgents/com.atom.agent.plist"
PYTHON3="${PYTHON3:-$(command -v python3)}"

if [[ "$(uname -s)" != "Darwin" ]]; then
	echo "This installer is for macOS only." >&2
	exit 1
fi
if [[ ! -f "${TEMPLATE}" ]]; then
	echo "Missing template: ${TEMPLATE}" >&2
	exit 1
fi
if [[ -z "${PYTHON3}" ]]; then
	echo "python3 not found on PATH. Set PYTHON3=/path/to/python3" >&2
	exit 1
fi

mkdir -p "${ROOT}/logs"
mkdir -p "${HOME}/Library/LaunchAgents"

export ATOM_LA_ROOT="${ROOT}"
export ATOM_LA_PYTHON="${PYTHON3}"
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
py = pathlib.Path(os.environ["ATOM_LA_PYTHON"])
template = pathlib.Path(os.environ["ATOM_LA_TEMPLATE"])
dest = pathlib.Path(os.environ["ATOM_LA_DEST"])

text = template.read_text(encoding="utf-8")
text = text.replace("@@@ATOM_REPO@@@", str(root))
text = text.replace("@@@PYTHON3@@@", str(py))

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
