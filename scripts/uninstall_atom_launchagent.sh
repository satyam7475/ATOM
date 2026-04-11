#!/usr/bin/env bash
# Remove ATOM launchd user agent (macOS).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
	echo "This script is for macOS only." >&2
	exit 1
fi

DEST="${HOME}/Library/LaunchAgents/com.atom.agent.plist"
UIDN="$(id -u)"

launchctl bootout "gui/${UIDN}/com.atom.agent" 2>/dev/null || true
launchctl unload "${DEST}" 2>/dev/null || true

if [[ -f "${DEST}" ]]; then
	rm -f "${DEST}"
	echo "Removed ${DEST}"
else
	echo "No plist at ${DEST}"
fi
