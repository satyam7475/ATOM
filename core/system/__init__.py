"""System awareness (CPU, RAM, processes, foreground window)."""

from core.system.system_monitor import SystemMonitor, get_system_state

__all__ = ["SystemMonitor", "get_system_state"]
