#!/usr/bin/env python3
"""
Quick smoke test for the native macOS ATOM UI.
Launches the NativeATOMWindow, cycles through states,
adds some log entries. Close the window to exit.

Run: .venv/bin/python ui/test_native_ui.py
"""
import sys
import time
import threading
import signal

# Add project root to path
sys.path.insert(0, ".")

from ui.native_ui import NativeATOMWindow


def main():
    print("🚀 Launching ATOM Native UI smoke test...")
    print("   Close the window or press Ctrl+C to exit.\n")

    ui = NativeATOMWindow(mic_name="MacBook Air Microphone")

    def on_shutdown():
        print("\n⏹ Shutdown requested via UI close button.")
        os._exit(0)

    import os
    ui.set_shutdown_callback(on_shutdown)

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    # Start the UI (blocks until ready)
    ui.start()

    print("✅ Window launched. Running state cycle demo...\n")

    def test_cycle():
        """Cycle through all states with demo data."""
        time.sleep(1.5)

        # ── IDLE ──
        print("  → IDLE")
        ui.update_state("idle")
        ui.add_log("info", "ATOM Native UI initialized. All systems nominal.")
        ui.add_log("info", "Security: authenticated, integrity clean.")
        ui.set_status("Ready — speak anytime")
        time.sleep(2.5)

        # ── LISTENING ──
        print("  → LISTENING")
        ui.update_state("listening")
        time.sleep(1)
        ui.show_hearing("Hey Atom, what's the weather")
        time.sleep(0.5)
        ui.show_hearing("Hey Atom, what's the weather like today")
        time.sleep(1)
        ui.add_log("heard", "Hey Atom, what's the weather like today")
        ui.clear_hearing()

        # ── THINKING ──
        print("  → THINKING")
        ui.update_state("thinking")
        ui.add_log("action", "Thinking with local brain...")
        ui.add_log("intent", "weather_query → RealWorldIntel")
        time.sleep(3)

        # ── SPEAKING ──
        print("  → SPEAKING")
        ui.update_state("speaking")
        ui.add_log("action", "Currently it's 28°C and partly cloudy in your area, Boss. Humidity is around 45%. Perfect weather for a walk outside.")
        time.sleep(3)

        # ── Back to LISTENING ──
        print("  → LISTENING")
        ui.update_state("listening")
        ui.add_log("info", "Ready for next command.")
        time.sleep(1)

        # Mic level demo
        for level in [0.1, 0.3, 0.5, 0.8, 0.95, 0.6, 0.3, 0.1]:
            ui.update_mic_level(level)
            time.sleep(0.15)

        time.sleep(1)

        # More conversation
        ui.show_hearing("Open Safari for me")
        time.sleep(1)
        ui.add_log("heard", "Open Safari for me")
        ui.clear_hearing()

        ui.update_state("thinking")
        time.sleep(1)
        ui.update_state("speaking")
        ui.add_log("action", "Opening Safari for you, Boss.")
        ui.add_log("action", "Running: open -a Safari")
        time.sleep(2)

        # ── ERROR_RECOVERY ──
        print("  → ERROR_RECOVERY")
        ui.update_state("error_recovery")
        ui.add_log("error", "TTS engine timeout — auto-recovering...")
        time.sleep(2)

        # ── Back to IDLE ──
        print("  → IDLE (recovered)")
        ui.update_state("idle")
        ui.add_log("info", "Self-healing complete. All systems nominal.")
        ui.set_status("Systems nominal")

        print("\n✅ Demo complete! Window stays open — close to exit.\n")

    t = threading.Thread(target=test_cycle, daemon=True)
    t.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n⏹ Interrupted.")
        ui.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
