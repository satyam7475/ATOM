"""
ATOM V3 -- Standalone TTS Worker Process.

This script runs the TTS engine (Kokoro/Edge/SAPI) in its own process.
It connects to the ZMQ Broker, listens for `tts_request` events,
synthesizes audio, plays it, and emits `tts_done` when finished.

Owner: Satyam
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import load_config
from core.ipc.zmq_bus import ZmqEventBus
from core.state_manager import StateManager

from core.ipc.interrupt_manager import SystemInterruptManager

# Import TTS engines
try:
    from voice.tts_kokoro import KokoroTTSAsync
except ImportError:
    KokoroTTSAsync = None

try:
    from voice.tts_edge import EdgeTTSAsync
except ImportError:
    EdgeTTSAsync = None

from voice.tts_sapi import SapiTTSAsync

logger = logging.getLogger("atom.services.tts")


class TTSWorker:
    def __init__(self) -> None:
        self.config = load_config()
        self.bus = ZmqEventBus(worker_name="tts_worker")
        self.state = StateManager(self.bus) # Dummy state for TTS engine init
        
        self.tts_engine = self._init_tts_engine()
        
        self.interrupt_mgr = SystemInterruptManager(self.bus, "tts_worker")
        self.interrupt_mgr.register_cancel_callback(self.handle_interrupt)
        
        # Register handlers
        self.bus.on("tts_request", self.handle_tts_request)

    def _init_tts_engine(self):
        """Initialize the configured TTS engine."""
        tts_cfg = self.config.get("tts", {})
        engine_name = (tts_cfg.get("engine") or "sapi").lower()
        
        if engine_name == "kokoro" and KokoroTTSAsync:
            logger.info(f"Initializing Kokoro TTS ({tts_cfg.get('kokoro_voice', 'af_heart')})")
            return KokoroTTSAsync(
                self.bus, self.state,
                max_lines=tts_cfg.get("max_lines", 4),
                voice=tts_cfg.get("kokoro_voice", "af_heart")
            )
            
        if engine_name == "edge" and EdgeTTSAsync:
            logger.info("Initializing Edge TTS")
            return EdgeTTSAsync(self.bus, self.state)
            
        logger.info("Initializing SAPI TTS (fallback)")
        return SapiTTSAsync(self.bus, self.state)

    async def handle_tts_request(self, event: str, **data) -> None:
        """Handle incoming request to speak text."""
        req_id = data.get("req_id", "unknown")
        text = data.get("text", "")
        interruptible = data.get("interruptible", True)
        
        logger.info(f"TTS Worker received request {req_id}: {text[:20]}...")
        
        try:
            # The actual TTS engine handles playing the audio
            await self.tts_engine.speak(text, interruptible=interruptible)
        except Exception as e:
            logger.error(f"TTS Engine error: {e}")
        finally:
            # Notify main process that we're done
            self.bus.emit("tts_done", req_id=req_id)
            logger.debug(f"TTS Worker finished request {req_id}")

    async def handle_interrupt(self) -> None:
        """Handle global interrupt signal."""
        logger.info("TTS Worker received interrupt signal. Stopping playback.")
        result = self.tts_engine.stop()
        if asyncio.iscoroutine(result):
            await result

    async def run(self) -> None:
        """Main worker loop."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logger.info("Starting TTS Worker Process...")
        
        self.bus.start()
        
        try:
            # Keep process alive
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            await self.tts_engine.shutdown()
            self.bus.stop()
            logger.info("TTS Worker Process stopped.")


if __name__ == "__main__":
    worker = TTSWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
