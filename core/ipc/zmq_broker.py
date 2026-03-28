"""
ATOM V3 -- ZeroMQ Message Broker.

This is the central nervous system of the multi-process architecture.
It acts as a PUB/SUB proxy.
- Publishers (emitters) connect to the XSUB socket (port 5555).
- Subscribers (listeners) connect to the XPUB socket (port 5556).

The broker simply forwards messages from publishers to subscribers,
allowing decoupled micro-services (STT, TTS, LLM, Core) to communicate
without knowing about each other.

Owner: Satyam
"""

import logging
import sys

try:
    import zmq
except ImportError:
    print("pyzmq not installed. Run: pip install pyzmq")
    sys.exit(1)

logger = logging.getLogger("atom.ipc.broker")


def run_pubsub_proxy(context, frontend_port=5555, backend_port=5556):
    frontend = context.socket(zmq.XSUB)
    frontend.bind(f"tcp://127.0.0.1:{frontend_port}")
    backend = context.socket(zmq.XPUB)
    backend.bind(f"tcp://127.0.0.1:{backend_port}")
    logger.info(f"ZMQ PUB/SUB Broker started. IN: {frontend_port} | OUT: {backend_port}")
    try:
        zmq.proxy(frontend, backend)
    except Exception as e:
        logger.error(f"PUB/SUB Proxy error: {e}")
    finally:
        frontend.close()
        backend.close()

def run_reqrep_proxy(context, frontend_port=5557, backend_port=5558):
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(f"tcp://127.0.0.1:{frontend_port}")
    backend = context.socket(zmq.DEALER)
    backend.bind(f"tcp://127.0.0.1:{backend_port}")
    logger.info(f"ZMQ REQ/REP Broker started. IN: {frontend_port} | OUT: {backend_port}")
    try:
        zmq.proxy(frontend, backend)
    except Exception as e:
        logger.error(f"REQ/REP Proxy error: {e}")
    finally:
        frontend.close()
        backend.close()

def run_dealer_router_proxy(context, port=5559):
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(f"tcp://127.0.0.1:{port}")
    backend = context.socket(zmq.DEALER)
    backend.bind(f"tcp://127.0.0.1:{port+1}") # Internal routing if needed, but for simple async tasks, a single ROUTER might be enough.
    # Actually, for DEALER-to-DEALER via ROUTER, we can just use a ROUTER socket that routes based on identity.
    logger.info(f"ZMQ DEALER/ROUTER Broker started on port {port}")
    try:
        zmq.proxy(frontend, backend)
    except Exception as e:
        logger.error(f"DEALER/ROUTER Proxy error: {e}")
    finally:
        frontend.close()
        backend.close()

def run_broker() -> None:
    """Run the ZMQ proxy brokers."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    context = zmq.Context()
    
    import threading
    t1 = threading.Thread(target=run_pubsub_proxy, args=(context,), daemon=True)
    t2 = threading.Thread(target=run_reqrep_proxy, args=(context,), daemon=True)
    t3 = threading.Thread(target=run_dealer_router_proxy, args=(context,), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("ZMQ Broker shutting down (KeyboardInterrupt)")
    finally:
        context.term()
        logger.info("ZMQ Broker terminated.")


if __name__ == "__main__":
    run_broker()
