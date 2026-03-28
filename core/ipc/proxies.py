"""
ATOM V3 -- RPC Proxies for Distributed Services.

These classes replace the local instances of STT, TTS, and the Brain
in `main.py` when running in multi-process mode. They simply serialize
method calls into ZMQ events and wait for responses over the bus.

Owner: Satyam
"""

import asyncio
import logging
import uuid
from typing import Any, Optional

from core.state_manager import AtomState, StateManager
from core.ipc.zmq_bus import ZmqEventBus

logger = logging.getLogger("atom.ipc.proxies")


class TTSProxy:
    """
    Proxy for the Text-to-Speech engine.
    Instead of synthesizing audio locally, it emits a `tts_request` event
    to the ZMQ broker. The standalone TTS worker picks it up, synthesizes,
    and plays the audio.
    """
    def __init__(self, bus: ZmqEventBus, state: StateManager) -> None:
        self.bus = bus
        self.state = state
        logger.info("TTSProxy initialized.")

    async def init_voice(self) -> None:
        """Initialize remote voice (no-op locally)."""
        pass

    async def speak(self, text: str, interruptible: bool = True) -> None:
        """Send text to the remote TTS worker."""
        if not text.strip():
            return
            
        req_id = str(uuid.uuid4())
        logger.debug(f"TTSProxy sending request {req_id}: {text[:20]}...")
        
        # Transition state locally so UI updates
        await self.state.transition(AtomState.SPEAKING)
        
        # Emit over ZMQ
        self.bus.emit("tts_request", req_id=req_id, text=text, interruptible=interruptible)
        
        # We don't block here; the TTS worker will emit `tts_done` when finished,
        # which `main.py` handles to transition back to LISTENING.

    async def on_response(self, text: str, **kw) -> None:
        """Handle response_ready event."""
        await self.speak(text, interruptible=kw.get("interruptible", True))

    async def on_partial_response(self, text: str, **kw) -> None:
        """Handle partial_response event."""
        await self.speak(text, interruptible=kw.get("interruptible", True))

    async def on_speech_partial(self, text: str, **kw) -> None:
        """Handle speech_partial event (usually stop TTS)."""
        pass

    async def speak_ack(self) -> None:
        """Send a short acknowledgment sound/phrase."""
        await self.speak("Yes, Boss.", interruptible=False)

    def stop(self) -> None:
        """Send an interrupt signal to the remote TTS worker."""
        logger.info("TTSProxy sending stop signal.")
        self.bus.emit_fast("interrupt")

    async def shutdown(self) -> None:
        """Cleanup proxy."""
        self.stop()
        logger.info("TTSProxy shutdown.")


class STTProxy:
    """
    Proxy for the Speech-to-Text engine.
    The actual STT processing happens in a separate worker process.
    This proxy just provides the interface `main.py` expects.
    """
    def __init__(self, bus: ZmqEventBus) -> None:
        self.bus = bus
        self.mic_name = "Remote Mic"
        logger.info("STTProxy initialized.")

    async def preload(self) -> None:
        """Tell the remote STT worker to load its models."""
        logger.info("STTProxy requesting model preload.")
        self.bus.emit("stt_preload_request")

    async def start_listening(self) -> None:
        """Tell the remote STT worker to start listening."""
        logger.info("STTProxy requesting start listening.")
        self.bus.emit("stt_start_listening")

    async def on_state_changed(self, old_state: AtomState, new_state: AtomState, **kw) -> None:
        """Handle state changes."""
        pass

    async def on_media_started(self, **kw) -> None:
        """Handle media started."""
        pass

    def stop(self) -> None:
        self.bus.emit("stt_stop_listening")

    async def shutdown(self) -> None:
        """Cleanup proxy."""
        logger.info("STTProxy shutdown.")


class BrainProxy:
    """
    Proxy for the LocalBrainController (LLM).
    Sends the user's query to the remote LLM worker and waits for chunks.
    """
    def __init__(self, bus: ZmqEventBus) -> None:
        self.bus = bus
        logger.info("BrainProxy initialized.")

    @property
    def available(self) -> bool:
        return True

    async def warm_up(self) -> None:
        pass

    def close(self) -> None:
        pass

    def request_preempt(self) -> None:
        self.bus.emit_fast("interrupt")

    async def on_query(
        self,
        text: str,
        memory_context: Optional[list[str]] = None,
        context: Optional[dict[str, str]] = None,
        history: Optional[list[tuple[str, str]]] = None,
        **_kw: Any,
    ) -> None:
        """Send query to the remote LLM worker."""
        req_id = str(uuid.uuid4())
        logger.info(f"BrainProxy sending query {req_id}: {text[:20]}...")
        
        # Emit over ZMQ
        self.bus.emit(
            "llm_query_request",
            req_id=req_id,
            text=text,
            memory_context=memory_context,
            context=context,
            history=history,
            **_kw
        )
        
        # The LLM worker will stream back `llm_chunk` events,
        # and finally an `llm_done` event. `main.py` handles these.
