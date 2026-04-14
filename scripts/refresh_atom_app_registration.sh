#!/usr/bin/env bash
# Re-register ATOM.app with Launch Services (use after Info.plist or executable changes).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT}/ATOM.app"
if [[ ! -d "${APP}" ]]; then
  echo "Missing ${APP}" >&2
  exit 1
fi
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "${APP}"
echo "Registered: ${APP}"
