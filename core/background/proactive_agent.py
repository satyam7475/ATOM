"""
ATOM OS -- Background Proactive Loop
A silent, non-blocking asynchronous daemon that runs in the background. It reads the StateGraph,
checks hardware loads, and optionally interrupts with notifications or automated tool dispatches.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("atom.background.proactive")

class ProactiveDaemon:
    """The silent watcher daemon."""
    
    def __init__(self, state_graph, tts_engine=None):
        self.state_graph = state_graph
        self.tts = tts_engine
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def _daemon_loop(self):
        logger.info("Proactive background daemon engaged. ATOM is now watching.")
        while self._running:
            try:
                # In production, check battery state before executing
                # Check System Load
                if hasattr(self.state_graph, "system_load") and self.state_graph.system_load > 90.0:
                    logger.warning("Spike in system load detected autonomously.")
                    if self.tts:
                        self.tts.speak("Boss, I'm detecting a critical load spike on your CPU. Shall I investigate?")
                
                # Check for stale tasks or forgotten reminders
                
                # Sleep heavily to prevent draining background battery
                await asyncio.sleep(120)  # Runs every 2 minutes
            except asyncio.CancelledError:
                logger.info("Proactive daemon cancelled.")
                break
            except Exception as e:
                logger.error(f"Daemon exception: {e}")
                await asyncio.sleep(60)

    def start(self):
        if not self._running:
            self._running = True
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._daemon_loop())

    def stop(self):
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
