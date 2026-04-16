"""
ATOM -- Earcons (subtle audio cues).

Plays short audio cues to reinforce voice-pipeline events so ATOM feels
alive without chatter:

    * wake            - mic just went ACTIVE (wake word detected)
    * done            - TTS finished a reply
    * error           - watchdog failover or recognition crash
    * heartbeat       - optional "still here" tick while LISTENING

On macOS we prefer system sounds (``/System/Library/Sounds/*.aiff``) via
``afplay`` so the UX is consistent with the rest of the OS and we don't
need to ship binary assets. If a custom file is supplied (``assets/earcons/
wake.aiff`` etc.) we use that instead.

Everything is best-effort and fire-and-forget. A missing ``afplay`` or a
non-macOS host silently disables the cue without raising.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.earcons")


_SYSTEM_SOUND_DIR = Path("/System/Library/Sounds")
_CUSTOM_SOUND_DIR = Path("assets/earcons")


_DEFAULT_MAPPING: dict[str, str] = {
    "wake": "Tink.aiff",
    "done": "Pop.aiff",
    "error": "Basso.aiff",
    "heartbeat": "Tink.aiff",
    # "thinking" is a soft single tick played after ~1.2s of silence while
    # the LLM is still composing the first sentence. Communicates "I heard
    # you, I'm on it" without verbose acks. Uses Pop.aiff at lower volume.
    "thinking": "Pop.aiff",
}


class Earcons:
    """Plays tiny audio cues tied to voice-pipeline lifecycle events.

    Thread-safe (the ``afplay`` subprocess is started detached). Calls
    are cheap no-ops when disabled or when the underlying player is
    unavailable so callers don't need to guard them.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        volume: float = 0.45,
        heartbeat_enabled: bool = False,
        heartbeat_interval_s: float = 600.0,
    ) -> None:
        self._enabled = bool(enabled) and sys.platform == "darwin"
        self._volume = max(0.0, min(1.0, float(volume)))
        self._heartbeat_enabled = bool(heartbeat_enabled)
        self._heartbeat_interval_s = max(60.0, float(heartbeat_interval_s))
        self._afplay = shutil.which("afplay") if self._enabled else None
        self._last_play_at: dict[str, float] = {}
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

        if self._enabled and self._afplay is None:
            logger.info("Earcons: afplay not found — cues disabled")
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _resolve_path(self, event: str) -> Path | None:
        filename = _DEFAULT_MAPPING.get(event)
        if not filename:
            return None
        custom = _CUSTOM_SOUND_DIR / filename
        if custom.exists():
            return custom
        system = _SYSTEM_SOUND_DIR / filename
        if system.exists():
            return system
        return None

    def play(self, event: str, *, min_interval_s: float = 0.0) -> None:
        """Play the earcon associated with ``event``. Non-blocking.

        ``min_interval_s`` rate-limits per-event so rapid-fire triggers
        (e.g. 10 wake-word false positives) don't spam the speaker.
        """
        if not self._enabled:
            return
        path = self._resolve_path(event)
        if path is None:
            return
        now = time.monotonic()
        if min_interval_s > 0:
            last = self._last_play_at.get(event, 0.0)
            if now - last < min_interval_s:
                return
        self._last_play_at[event] = now
        try:
            subprocess.Popen(
                [self._afplay, "-v", f"{self._volume:.2f}", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            logger.debug("Earcon play failed for '%s'", event, exc_info=True)

    def start_heartbeat(self, is_listening_fn: Any) -> None:
        """Optional: play a very quiet tick every N seconds while
        ``is_listening_fn()`` returns True. Off by default (can feel
        naggy); callers opt in via config."""
        if not self._enabled or not self._heartbeat_enabled:
            return
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()

        def _loop() -> None:
            while not self._heartbeat_stop.is_set():
                try:
                    alive = bool(is_listening_fn())
                except Exception:
                    alive = False
                if alive:
                    self.play("heartbeat", min_interval_s=self._heartbeat_interval_s - 1)
                self._heartbeat_stop.wait(self._heartbeat_interval_s)

        t = threading.Thread(target=_loop, name="atom.earcons.heartbeat", daemon=True)
        t.start()
        self._heartbeat_thread = t

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()

    def shutdown(self) -> None:
        self.stop_heartbeat()


__all__ = ["Earcons"]
