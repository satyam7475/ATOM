"""
ATOM V4 -- ZeroMQ Distributed Event Bus.

A hardened distributed event bus supporting a strict message contract,
ACKs, retries with exponential backoff, and timeout handling.

Owner: Satyam
"""

import asyncio
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, Optional

try:
    import zmq
    import zmq.asyncio
except ImportError:
    pass

logger = logging.getLogger("atom.ipc.bus")

class ZmqEventBus:
    """
    Distributed Event Bus using ZeroMQ PUB/SUB.
    Connects to the ZmqBroker.
    """

    def __init__(
        self,
        pub_url: str = "tcp://127.0.0.1:5555",
        sub_url: str = "tcp://127.0.0.1:5556",
        req_url: str = "tcp://127.0.0.1:5557",
        rep_url: str = "tcp://127.0.0.1:5558",
        dealer_url: str = "tcp://127.0.0.1:5559",
        worker_name: str = "unknown"
    ) -> None:
        self._pub_url = pub_url
        self._sub_url = sub_url
        self._req_url = req_url
        self._rep_url = rep_url
        self.worker_name = worker_name
        
        self.ctx = zmq.asyncio.Context()
        
        # PUB/SUB for streaming/broadcast
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.connect(self._pub_url)
        
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.sub_socket.connect(self._sub_url)
        
        # REQ/REP for critical synchronous calls
        self.req_socket = self.ctx.socket(zmq.REQ)
        self.req_socket.connect(self._req_url)
        
        # DEALER for async tasks (port now configurable)
        self.dealer_socket = self.ctx.socket(zmq.DEALER)
        self.dealer_socket.setsockopt_string(zmq.IDENTITY, self.worker_name)
        self.dealer_socket.connect(dealer_url)
        
        # Local handlers: event_name -> list of async callables
        self._handlers: dict[str, list[Callable[..., Coroutine[Any, Any, None]]]] = defaultdict(list)
        self._req_handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        
        self._running = False
        self._listen_task: asyncio.Task | None = None
        self._dealer_task: asyncio.Task | None = None
        
        # Backpressure control
        self._max_concurrent_tasks = 50
        self._semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        
        # ACK tracking
        self._pending_acks: Dict[str, asyncio.Future] = {}
        
        # Subscribe to ACKs for this worker
        self.on(f"ACK_{self.worker_name}", self._handle_ack)

    def start(self) -> None:
        """Start the background listener task."""
        if self._running:
            return
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._dealer_task = asyncio.create_task(self._dealer_loop())
        logger.info(f"ZmqHybridBus started ({self.worker_name}). PUB->{self._pub_url}, SUB<-{self._sub_url}")

    def stop(self) -> None:
        """Stop the background listener and close sockets."""
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
        if self._dealer_task:
            self._dealer_task.cancel()
        
        self.pub_socket.close()
        self.sub_socket.close()
        self.req_socket.close()
        self.dealer_socket.close()
        self.ctx.term()
        logger.info("ZmqHybridBus stopped.")

    def on(self, event: str, handler: Callable[..., Coroutine[Any, Any, None]]) -> None:
        """Register an async handler for a specific event (PUB/SUB)."""
        if not self._handlers[event]:
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, event)
            logger.debug(f"Subscribed to ZMQ topic: {event}")
        self._handlers[event].append(handler)

    def register_request_handler(self, command: str, handler: Callable[..., Coroutine[Any, Any, Any]]) -> None:
        """Register a handler for critical REQ/REP calls."""
        self._req_handlers[command] = handler

    async def request(self, command: str, timeout: float = 5.0, **data: Any) -> Any:
        """Send a critical request and wait for a reply (REQ/REP)."""
        msg = self._build_message(command, payload=data)
        try:
            await self.req_socket.send_json(msg)
            # Use asyncio.wait_for to implement timeout
            reply = await asyncio.wait_for(self.req_socket.recv_json(), timeout=timeout)
            return reply.get("result")
        except asyncio.TimeoutError:
            logger.error(f"REQ timeout for command: {command}")
            # Fix: set LINGER=0 before close to prevent socket leak
            self.req_socket.setsockopt(zmq.LINGER, 0)
            self.req_socket.close()
            self.req_socket = self.ctx.socket(zmq.REQ)
            self.req_socket.connect(self._req_url)
            return None
        except Exception as e:
            logger.error(f"REQ error for {command}: {e}")
            return None

    def send_task(self, target: str, task_type: str, **data: Any) -> None:
        """Send an async task via DEALER/ROUTER."""
        msg = self._build_message(task_type, target=target, payload=data)
        try:
            self.dealer_socket.send_multipart([target.encode("utf-8"), json.dumps(msg).encode("utf-8")])
        except Exception as e:
            logger.error(f"Failed to send task {task_type} to {target}: {e}")

    async def _dealer_loop(self) -> None:
        """Background loop to receive DEALER messages."""
        try:
            while self._running:
                parts = await self.dealer_socket.recv_multipart()
                if len(parts) < 2:
                    continue
                # For dealer, we get [sender_identity, payload]
                payload_bytes = parts[-1]
                try:
                    msg = json.loads(payload_bytes.decode("utf-8"))
                    event = msg.get("type", "unknown")
                    data = msg.get("payload", {})
                    
                    handlers = self._handlers.get(event, [])
                    for handler in handlers:
                        asyncio.create_task(self._safe_invoke(handler, event, data))
                except Exception as e:
                    logger.error(f"Error processing dealer message: {e}")
        except asyncio.CancelledError:
            pass

    def _build_message(self, event: str, target: str = "all", priority: str = "normal", payload: dict = None) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "trace_id": payload.get("trace_id", str(uuid.uuid4())) if payload else str(uuid.uuid4()),
            "source": self.worker_name,
            "target": target,
            "type": event,
            "priority": priority,
            "timestamp": time.time(),
            "payload": payload or {}
        }

    def emit(self, event: str, **data: Any) -> None:
        """Publish an event to the ZMQ broker."""
        try:
            msg = self._build_message(event, payload=data)
            payload_str = json.dumps(msg).encode("utf-8")
            self.pub_socket.send_multipart([event.encode("utf-8"), payload_str])
            logger.debug(f"ZMQ Emit: {event} (id: {msg['id']})")
        except Exception as e:
            logger.error(f"Failed to emit ZMQ event {event}: {e}")

    async def emit_with_ack(self, event: str, target: str, timeout: float = 2.0, retries: int = 3, **data: Any) -> bool:
        """Emit an event and wait for an ACK from the target worker.

        Uses exponential backoff with jitter to prevent thundering herd
        when multiple workers time out simultaneously.
        """
        msg = self._build_message(event, target=target, priority="high", payload=data)
        msg_id = msg["id"]
        base_timeout = timeout
        
        for attempt in range(retries):
            try:
                payload_str = json.dumps(msg).encode("utf-8")
                
                # Create future for ACK
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                self._pending_acks[msg_id] = fut
                
                self.pub_socket.send_multipart([event.encode("utf-8"), payload_str])
                logger.debug(f"ZMQ Emit with ACK: {event} (id: {msg_id}, attempt: {attempt+1})")
                
                # Exponential backoff: timeout doubles each retry + random jitter
                current_timeout = base_timeout * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.wait_for(fut, timeout=current_timeout)
                return True
                
            except asyncio.TimeoutError:
                logger.warning(
                    f"ACK timeout for {event} (id: {msg_id}, attempt {attempt+1}/{retries}, "
                    f"waited {current_timeout:.1f}s). Retrying..."
                )
            except Exception as e:
                logger.error(f"Failed to emit ZMQ event {event}: {e}")
            finally:
                self._pending_acks.pop(msg_id, None)
                
        logger.error(f"Failed to receive ACK for {event} after {retries} retries.")
        return False

    async def _handle_ack(self, event: str, **data: Any) -> None:
        """Handle incoming ACKs."""
        msg_id = data.get("ack_id")
        if msg_id and msg_id in self._pending_acks:
            if not self._pending_acks[msg_id].done():
                self._pending_acks[msg_id].set_result(True)

    def _send_ack(self, source: str, msg_id: str) -> None:
        """Send an ACK back to the source worker."""
        if source == "unknown":
            return
        ack_topic = f"ACK_{source}"
        msg = self._build_message(ack_topic, target=source, payload={"ack_id": msg_id})
        try:
            self.pub_socket.send_multipart([ack_topic.encode("utf-8"), json.dumps(msg).encode("utf-8")])
        except Exception as e:
            logger.error(f"Failed to send ACK to {source}: {e}")

    def emit_fast(self, event: str, **data: Any) -> None:
        self.emit(event, **data)

    def emit_long(self, event: str, **data: Any) -> None:
        self.emit(event, **data)

    async def _listen_loop(self) -> None:
        """Background loop to receive messages from the broker and dispatch locally."""
        try:
            while self._running:
                parts = await self.sub_socket.recv_multipart()
                if len(parts) != 2:
                    continue
                    
                topic_bytes, payload_bytes = parts
                event = topic_bytes.decode("utf-8")
                
                try:
                    msg = json.loads(payload_bytes.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON payload for event {event}")
                    continue
                
                # Check if message is in contract format
                if "id" in msg and "payload" in msg:
                    # It's a contract message
                    data = msg["payload"]
                    source = msg.get("source", "unknown")
                    msg_id = msg.get("id")
                    target = msg.get("target", "all")
                    
                    # Filter by target
                    if target != "all" and target != self.worker_name:
                        continue
                        
                    # Auto-ACK if priority is high or target is specific
                    if (msg.get("priority") == "high" or target == self.worker_name) and not event.startswith("ACK_"):
                        self._send_ack(source, msg_id)
                else:
                    # Legacy fallback
                    data = msg
                
                # Dispatch to all registered local handlers
                handlers = self._handlers.get(event, [])
                for handler in handlers:
                    asyncio.create_task(self._safe_invoke(handler, event, data))
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ZmqEventBus listener crashed: {e}")

    async def _safe_invoke(self, handler: Callable, event: str, data: dict) -> None:
        """Safely invoke a handler with backpressure control and catch exceptions."""
        try:
            async with self._semaphore:
                await handler(event=event, **data)
        except Exception as e:
            logger.error(f"Error in handler for {event}: {e}", exc_info=True)
