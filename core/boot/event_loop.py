"""Event-loop policy selection for ATOM boot."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("atom.boot.event_loop")

_INSTALLED = False


def install_fast_event_loop() -> bool:
    """Install uvloop when available on supported platforms.

    Returns True when uvloop is active for subsequently-created loops.
    The helper is deliberately fail-open so optional acceleration never
    blocks boot.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    if sys.platform.startswith("win"):
        return False
    try:
        import uvloop
    except ImportError:
        logger.debug("uvloop unavailable; using default asyncio event loop")
        return False
    try:
        uvloop.install()
    except Exception:
        logger.warning("uvloop install failed; using default asyncio event loop", exc_info=True)
        return False
    _INSTALLED = True
    logger.info("uvloop event loop policy installed")
    return True
