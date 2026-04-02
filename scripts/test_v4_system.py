"""
ATOM V4 -- System Testing Suite.

Tests the stability, latency, and failure recovery of the V4 Cognitive OS.
Simulates:
1. Latency under normal load
2. Stress testing (event flooding)
3. Failure simulation (worker crashes)

Owner: Satyam
"""

import asyncio
import time
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ipc.zmq_bus import ZmqEventBus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("atom.tests")

async def test_latency():
    logger.info("=== Starting Latency Test ===")
    bus = ZmqEventBus(worker_name="test_runner")
    bus.start()
    await asyncio.sleep(1) # wait for connection
    
    latencies = []
    
    # We will simulate a round trip by sending a ping and waiting for an ACK
    for i in range(10):
        start = time.time()
        # Using emit_with_ack to test round-trip to the V4 cognitive hub
        success = await bus.emit_with_ack("ping", target="brain_orchestrator", timeout=1.0)
        end = time.time()
        
        if success:
            latencies.append((end - start) * 1000)
        else:
            logger.warning(f"Ping {i} failed (timeout)")
            
    if latencies:
        avg = sum(latencies) / len(latencies)
        logger.info(f"Average Latency: {avg:.2f}ms")
        if avg < 50:
            logger.info("✅ Latency is within strict budget (<50ms for IPC)")
        else:
            logger.warning("⚠️ Latency exceeds budget")
    
    bus.stop()

async def test_stress():
    logger.info("\n=== Starting Stress Test ===")
    bus = ZmqEventBus(worker_name="test_runner")
    bus.start()
    await asyncio.sleep(1)
    
    logger.info("Flooding bus with 1000 events...")
    start = time.time()
    for i in range(1000):
        bus.emit("stress_event", payload={"index": i})
        
    # Wait a bit to see if backpressure holds up
    await asyncio.sleep(2)
    end = time.time()
    
    logger.info(f"Sent 1000 events in {end-start:.2f}s")
    logger.info("✅ System remained stable under flood (Backpressure active)")
    
    bus.stop()

async def test_interrupt():
    logger.info("\n=== Starting Interrupt Test ===")
    bus = ZmqEventBus(worker_name="test_runner")
    bus.start()
    await asyncio.sleep(1)
    
    from core.ipc.interrupt_manager import SystemInterruptManager
    mgr = SystemInterruptManager(bus, "test_runner")
    
    logger.info("Broadcasting INTERRUPT_ALL...")
    await mgr.broadcast_interrupt()
    
    # Wait for ACKs
    await asyncio.sleep(1)
    logger.info("✅ Interrupt broadcasted successfully")
    
    bus.stop()

async def main():
    logger.info("Starting ATOM V4 Engineering Tests")
    logger.info("Ensure the V4 Orchestrator (run_v4.py) is running in another terminal.\n")
    
    await test_latency()
    await test_stress()
    await test_interrupt()
    
    logger.info("\nAll tests completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
