"""
Validate launchd plist template for ATOM (step 5.5).

Run: python3 -m tests.test_launch_agent_plist
"""

from __future__ import annotations

import io
import os
import plistlib
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_plist_template_substitutes_and_loads() -> None:
    root = Path(__file__).resolve().parent.parent
    template = root / "scripts" / "com.atom.agent.plist"
    assert template.is_file(), f"missing {template}"
    repo = "/tmp/atom_test_repo"
    text = template.read_text(encoding="utf-8")
    filled = text.replace("@@@ATOM_REPO@@@", repo)
    data = plistlib.load(io.BytesIO(filled.encode("utf-8")))
    assert data["Label"] == "com.atom.agent"
    args = data["ProgramArguments"]
    assert args == ["/bin/bash", f"{repo}/scripts/atom_run.sh"]
    assert data["WorkingDirectory"] == repo
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["ThrottleInterval"] == 15
    assert "launchagent.stdout.log" in data["StandardOutPath"]
    print("  PASS: plist template parses after substitution")


if __name__ == "__main__":
    test_plist_template_substitutes_and_loads()
    print("Launch agent plist tests passed.")
