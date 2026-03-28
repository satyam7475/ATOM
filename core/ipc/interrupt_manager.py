"""
ATOM V4 -- Global Interrupt System.

Handles broadcasting and acknowledging global interrupts across all workers.
Ensures ATOM feels alive and responsive when the user barges in.

Owner: Satyam
"""

import asyncio
import logging
from typing import Callable, List
from .zmq_bus import ZmqEventBus

logger = logging.getLogger("atom.ipc.interrupt")

class SystemInterruptManager:
    def __init__(self, bus: ZmqEventBus, worker_name: str):
        self.bus = bus
        self.worker_name = worker_name
        self._cancel_callbacks: List[Callable] = []
        
        # Listen for global interrupts
        self.bus.on("INTERRUPT_ALL", self._handle_interrupt)

    def register_cancel_callback(self, callback: Callable):
        """Register a callback to be executed when an interrupt is received."""
        self._cancel_callbacks.append(callback)

    async def broadcast_interrupt(self):
        """Broadcast a global interrupt to all workers."""
        logger.warning(f"[{self.worker_name}] Broadcasting global INTERRUPT_ALL")
        # We don't wait for ACKs here to ensure it's fast, 
        # but the bus will handle high priority routing.
        self.bus.emit("INTERRUPT_ALL", source_worker=self.worker_name)

    async def _handle_interrupt(self, event: str, **data):
        """Handle incoming global interrupt."""
        source = data.get("source_worker", "unknown")
        if source == self.worker_name:
            return # Ignore our own broadcast
            
        logger.warning(f"[{self.worker_name}] Received INTERRUPT_ALL from {source}")
        
        # Execute all registered cancel callbacks
        for cb in self._cancel_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb()
                else:
                    cb()
            except Exception as e:
                logger.error(f"Error in interrupt callback: {e}")
                
        # Send ACK back
        self.bus.emit(f"INTERRUPT_ACK_{source}", worker=self.worker_name)
