"""
ATOM V3 -- Standalone STT Worker Process.

This script runs the `STTAsync` (faster-whisper) engine in its own process.
It connects to the ZMQ Broker, listens for `stt_preload_request` events,
and continuously listens to the microphone. When speech is detected and
transcribed, it emits `speech_final` events over ZMQ.

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
from core.gpu_resource_manager import EVENT_GPU_ACK, EVENT_GPU_UNLOAD
from voice.stt_async import STTAsync

logger = logging.getLogger("atom.services.stt")


class STTWorker:
    def __init__(self) -> None:
        self.config = load_config()
        self.bus = ZmqEventBus(worker_name="stt_worker")
        
        logger.info("Initializing STTAsync in worker process...")
        
        # We need a dummy state manager for STTAsync
        from core.state_manager import StateManager
        self.state = StateManager(self.bus)
        
        # We need MicManager
        from voice.mic_manager import MicManager
        self.mic_manager = MicManager()
        self.mic_manager.profile_devices()
        
        # STTAsync internally uses the bus to emit `speech_final` and `speech_partial`.
        # Since we pass it the ZmqEventBus, it will automatically broadcast these
        # to the main process!
        self.stt = STTAsync(
            bus=self.bus, 
            state=self.state, 
            config=self.config,
            mic_manager=self.mic_manager
        )
        
        # Register handlers
        self.bus.on("stt_preload_request", self.handle_preload_request)
        self.bus.on("stt_start_listening", self.handle_start_listening)
        self.bus.on("interrupt", self.handle_interrupt)
        self.bus.on(EVENT_GPU_UNLOAD, self.handle_v7_gpu_unload)

    async def handle_preload_request(self, event: str, **data) -> None:
        """Handle request to preload faster-whisper models."""
        logger.info("STT Worker received preload request.")
        try:
            await self.stt.preload()
            logger.info("STT Worker preload complete.")
            self.bus.emit("stt_preload_done")
        except Exception as e:
            logger.error(f"STT Preload error: {e}")

    async def handle_start_listening(self, event: str, **data) -> None:
        """Handle request to start listening."""
        logger.info("STT Worker received start listening request.")
        # Start the continuous listening loop in a task so we don't block the event loop
        asyncio.create_task(self.stt.start_listening())

    async def handle_v7_gpu_unload(self, event: str, **data) -> None:
        if data.get("slot") != "stt":
            return
        logger.info("V7 GPU: STT unload requested (ack; model stays until STTAsync supports unload)")
        self.bus.emit(EVENT_GPU_ACK, slot="stt", unloaded=True)

    async def handle_interrupt(self, event: str, **data) -> None:
        """Handle global interrupt signal."""
        logger.info("STT Worker received interrupt signal. (STT usually keeps listening)")
        # If we had active transcribing we wanted to cancel, we'd do it here.

    async def run(self) -> None:
        """Main worker loop."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logger.info("Starting STT Worker Process...")
        
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
            await self.stt.shutdown()
            self.bus.stop()
            logger.info("STT Worker Process stopped.")


if __name__ == "__main__":
    worker = STTWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
