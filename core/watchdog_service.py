"""
Lightweight process watchdog: polls child PIDs and restarts if exited unexpectedly.
Use from a small launcher script or integrate with run_v4 (optional).
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("atom.watchdog")


class ServiceWatchdog:
    def __init__(self, atom_root: Optional[Path] = None, poll_interval_s: float = 5.0):
        self.atom_root = atom_root or Path(__file__).resolve().parent.parent
        self.poll_interval_s = poll_interval_s
        self._processes: Dict[str, subprocess.Popen] = {}

    def start_service(self, name: str, rel_script: str, args: Optional[List[str]] = None) -> None:
        script = self.atom_root / rel_script
        cmd = [sys.executable, str(script)]
        if args:
            cmd.extend(args)
        logger.info("Watchdog starting %s: %s", name, cmd)
        self._processes[name] = subprocess.Popen(cmd, cwd=str(self.atom_root))

    def poll(self) -> List[Tuple[str, bool]]:
        """Return list of (name, restarted)."""
        restarted: List[Tuple[str, bool]] = []
        for name, proc in list(self._processes.items()):
            code = proc.poll()
            if code is not None:
                logger.warning("Service %s exited with code %s — restarting", name, code)
                restarted.append((name, True))
        return restarted

    def restart_service(
        self,
        name: str,
        rel_script: str,
        args: Optional[List[str]] = None,
    ) -> None:
        """Stop existing process for name (if any) and start a new one."""
        old = self._processes.pop(name, None)
        if old is not None and old.poll() is None:
            try:
                old.terminate()
                old.wait(timeout=5)
            except Exception:
                logger.debug("terminate %s", name, exc_info=True)
        self.start_service(name, rel_script, args=args)

    def run_forever(self) -> None:
        while True:
            self.poll()
            time.sleep(self.poll_interval_s)
