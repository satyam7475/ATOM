"""
ATOM V4 Orchestrator.

This script launches the multi-process architecture of ATOM V4 (Cognitive OS).
It starts the ZMQ Broker, then spawns the STT, TTS, LLM, and the new Brain workers,
and finally launches the main ATOM core process with the --v4 flag.

Owner: Satyam
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("atom.v4.orchestrator")

def main():
    logger.info("Starting ATOM V4 Cognitive OS Architecture...")
    
    try:
        import zmq
    except ImportError:
        logger.error("pyzmq is not installed. Please run: pip install pyzmq")
        sys.exit(1)

    processes = []
    
    try:
        # 1. Start ZMQ Broker
        logger.info("Starting ZMQ Broker...")
        broker_proc = subprocess.Popen([sys.executable, "core/ipc/zmq_broker.py"])
        processes.append(("Broker", broker_proc))
        time.sleep(1) # Give broker time to bind sockets
        
        # 2. Start Workers
        logger.info("Starting STT Worker...")
        stt_proc = subprocess.Popen([sys.executable, "services/stt_worker.py"])
        processes.append(("STT", stt_proc))
        
        logger.info("Starting TTS Worker...")
        tts_proc = subprocess.Popen([sys.executable, "services/tts_worker.py"])
        processes.append(("TTS", tts_proc))
        
        logger.info("Starting LLM Worker...")
        llm_proc = subprocess.Popen([sys.executable, "services/llm_worker.py"])
        processes.append(("LLM", llm_proc))
        
        logger.info("Starting Intent Worker...")
        intent_proc = subprocess.Popen([sys.executable, "services/intent_worker.py"])
        processes.append(("Intent", intent_proc))
        
        logger.info("Starting Context Worker...")
        context_proc = subprocess.Popen([sys.executable, "services/context_worker.py"])
        processes.append(("Context", context_proc))
        
        logger.info("Starting Decision Worker...")
        decision_proc = subprocess.Popen([sys.executable, "services/decision_worker.py"])
        processes.append(("Decision", decision_proc))
        
        logger.info("Starting Memory Worker...")
        memory_proc = subprocess.Popen([sys.executable, "services/memory_worker.py"])
        processes.append(("Memory", memory_proc))
        
        logger.info("Starting V5 Brain Orchestrator (Cognitive Hub)...")
        brain_proc = subprocess.Popen([sys.executable, "services/brain_orchestrator.py"])
        processes.append(("BrainOrchestrator", brain_proc))
        
        time.sleep(2) # Give workers time to connect
        
        # 3. Start Main Core Process
        logger.info("Starting ATOM Core (main.py --v4)...")
        main_proc = subprocess.Popen([sys.executable, "main.py", "--v4"])
        processes.append(("Core", main_proc))
        
        # Wait for main process to exit
        main_proc.wait()
        
    except KeyboardInterrupt:
        logger.info("Shutting down ATOM V4...")
    finally:
        # Terminate all child processes
        for name, proc in processes:
            if proc.poll() is None:
                logger.info(f"Terminating {name} process...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name} process...")
                    proc.kill()
        
        logger.info("ATOM V4 shutdown complete.")

if __name__ == "__main__":
    main()
