import logging
import random
from typing import Optional

from core.runtime_config import SystemMode, get_system_mode

logger = logging.getLogger("atom.brain.exploration_engine")


class ExplorationEngine:
    """Epsilon-greedy exploration; disabled in CRITICAL system mode."""

    def __init__(self, base_rate: float = 0.05):
        self.base_rate = base_rate

    def should_explore(
        self,
        uncertainty: float = 0.5,
        system_mode: Optional[SystemMode] = None,
    ) -> bool:
        mode = system_mode if system_mode is not None else get_system_mode()
        if mode == SystemMode.CRITICAL:
            return False
        rate = min(0.5, self.base_rate + (uncertainty * 0.5))
        r = random.random()
        do = r < rate
        logger.debug("Exploration check: r=%.3f rate=%.3f -> %s", r, rate, do)
        return do
