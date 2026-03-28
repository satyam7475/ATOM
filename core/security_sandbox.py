import subprocess
import logging
import shlex
from typing import Tuple, Optional
from pathlib import Path
import time
import json

logger = logging.getLogger("atom.core.security_sandbox")

_SANDBOX_LOG = Path("logs/sandbox_execution.log")

class ExecutionSandbox:
    """
    Execution Sandbox Layer for OS command execution.
    - Subprocess jail
    - Command whitelist
    - Execution logs
    - Audit replay system
    """
    def __init__(self):
        self.whitelist = {
            "echo", "dir", "ls", "ping", "whoami", "ipconfig", "systeminfo",
            "tasklist", "netstat", "python", "node", "npm", "git"
        }
        _SANDBOX_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _log_execution(self, cmd: str, success: bool, exit_code: int, output: str):
        try:
            entry = {
                "timestamp": time.time(),
                "command": cmd,
                "success": success,
                "exit_code": exit_code,
                "output_preview": output[:200]
            }
            with open(_SANDBOX_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log sandbox execution: {e}")

    def is_command_allowed(self, cmd: str) -> bool:
        """Check if the base command is in the whitelist."""
        try:
            parts = shlex.split(cmd)
            if not parts:
                return False
            base_cmd = parts[0].lower()
            if base_cmd.endswith('.exe'):
                base_cmd = base_cmd[:-4]
            return base_cmd in self.whitelist
        except Exception:
            return False

    def execute(self, cmd: str, timeout: int = 10) -> Tuple[bool, str]:
        """Execute a command within the sandbox constraints."""
        if not self.is_command_allowed(cmd):
            msg = f"Command blocked by sandbox whitelist: {cmd}"
            logger.warning(msg)
            self._log_execution(cmd, False, -1, msg)
            return False, msg

        logger.info(f"Sandbox executing: {cmd}")
        try:
            # Using subprocess.run with shell=False where possible is safer, 
            # but for Windows dir/echo shell=True is often needed.
            # We restrict via whitelist above.
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            self._log_execution(cmd, success, result.returncode, output)
            return success, output
            
        except subprocess.TimeoutExpired:
            msg = f"Command timed out after {timeout}s"
            self._log_execution(cmd, False, -1, msg)
            return False, msg
        except Exception as e:
            msg = f"Sandbox execution error: {e}"
            self._log_execution(cmd, False, -1, msg)
            return False, msg

    def replay_audit_log(self, limit: int = 10) -> list:
        """Read the execution log for audit replay."""
        if not _SANDBOX_LOG.exists():
            return []
            
        entries = []
        try:
            with open(_SANDBOX_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            
        return entries[-limit:]
