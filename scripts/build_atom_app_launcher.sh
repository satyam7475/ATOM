#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT}/ATOM.app"
SRC="${ROOT}/scripts/atom_app_launcher.c"
OUT="${APP}/Contents/MacOS/atom_python"
FRAMEWORK_DIR="/Library/Developer/CommandLineTools/Library/Frameworks"
PY_HEADERS="${FRAMEWORK_DIR}/Python3.framework/Headers"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This builder is for macOS only." >&2
    exit 1
fi
if [[ ! -d "${APP}/Contents/MacOS" ]]; then
    echo "Missing app bundle at ${APP}" >&2
    exit 1
fi
if [[ ! -f "${SRC}" ]]; then
    echo "Missing launcher source: ${SRC}" >&2
    exit 1
fi
if [[ ! -f "${PY_HEADERS}/Python.h" ]]; then
    echo "Python framework headers not found at ${PY_HEADERS}" >&2
    exit 1
fi

clang \
    -O2 \
    -Wall \
    -Wextra \
    -F "${FRAMEWORK_DIR}" \
    -I "${PY_HEADERS}" \
    -framework Python3 \
    -Wl,-rpath,"${FRAMEWORK_DIR}" \
    "${SRC}" \
    -o "${OUT}"

chmod +x "${OUT}"

rm -rf "${APP}/Contents/_CodeSignature"
# Strip extended attributes (iCloud Desktop provenance breaks codesign).
xattr -cr "${OUT}"
xattr -cr "${APP}"

if codesign --force --deep -s - "${APP}" 2>/tmp/atom_codesign.err; then
    echo "Ad-hoc signed ${APP}"
else
    echo "Warning: ad-hoc codesign failed; ${APP} may remain unsigned." >&2
    if grep -q "resource fork\|detritus\|not allowed" /tmp/atom_codesign.err 2>/dev/null; then
        echo "  Common cause: iCloud Desktop/file-provider metadata on the repo. Copy ATOM to a non-iCloud folder (e.g. ~/Developer), run xattr -cr ATOM.app, then rebuild." >&2
    else
        sed 's/^/  codesign: /' /tmp/atom_codesign.err >&2 || cat /tmp/atom_codesign.err >&2
    fi
    rm -f /tmp/atom_codesign.err
fi
rm -f /tmp/atom_codesign.err

echo "Built ${OUT}"
