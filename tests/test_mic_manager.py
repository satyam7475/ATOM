"""
ATOM v9 -- Phase 3 MicManager Tests.

Tests the threading.Condition-based microphone ownership lock
for correctness under single-threaded, multi-owner, and concurrent
handoff scenarios.

Run: python -m tests.test_mic_manager
"""

from __future__ import annotations

import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.mic_manager import MicManager


def test_single_owner() -> None:
    """First acquire succeeds, second with short timeout fails."""
    mic = MicManager()

    assert mic.acquire("holder_a") is True
    assert mic.owner == "holder_a"
    assert not mic.is_free

    assert mic.acquire("stt", timeout=0.1) is False
    assert mic.owner == "holder_a"

    print("  PASS: Single owner lock works")


def test_release_and_reacquire() -> None:
    """After release, a new owner can acquire."""
    mic = MicManager()

    assert mic.acquire("holder_a") is True
    mic.release("holder_a")
    assert mic.is_free
    assert mic.owner is None

    assert mic.acquire("stt") is True
    assert mic.owner == "stt"

    mic.release("stt")
    assert mic.is_free

    print("  PASS: Release and reacquire works")


def test_release_wrong_owner() -> None:
    """Release by non-owner is a no-op -- does not free the mic."""
    mic = MicManager()

    assert mic.acquire("holder_a") is True

    mic.release("stt")
    assert mic.owner == "holder_a", "Wrong owner release must not free the mic"

    mic.release("holder_a")
    assert mic.is_free

    print("  PASS: Wrong owner release is a no-op")


def test_release_when_free() -> None:
    """Release on already-free mic is a no-op (no crash)."""
    mic = MicManager()

    mic.release("holder_a")
    assert mic.is_free

    mic.release("nobody")
    assert mic.is_free

    print("  PASS: Release when free is safe")


def test_owner_property() -> None:
    """Owner property reflects current holder or None."""
    mic = MicManager()

    assert mic.owner is None
    assert mic.is_free is True

    mic.acquire("holder_a")
    assert mic.owner == "holder_a"
    assert mic.is_free is False

    mic.release("holder_a")
    assert mic.owner is None
    assert mic.is_free is True

    print("  PASS: Owner property correct")


def test_concurrent_handoff() -> None:
    """Simulate one holder releasing and STT acquiring across threads."""
    mic = MicManager()
    mic.acquire("holder_a")

    results: dict[str, bool] = {}

    def stt_thread():
        acquired = mic.acquire("stt", timeout=3.0)
        results["stt_acquired"] = acquired
        if acquired:
            results["stt_owner"] = (mic.owner == "stt")

    t = threading.Thread(target=stt_thread)
    t.start()

    time.sleep(0.1)
    assert mic.owner == "holder_a", "holder_a should still own mic"

    mic.release("holder_a")

    t.join(timeout=3.0)
    assert not t.is_alive(), "STT thread should have completed"

    assert results.get("stt_acquired") is True, "STT should acquire after release"
    assert results.get("stt_owner") is True, "STT should be the owner"

    mic.release("stt")
    assert mic.is_free

    print("  PASS: Concurrent handoff (holder_a -> stt)")


def test_acquire_timeout() -> None:
    """Acquire with zero timeout fails immediately if mic is held."""
    mic = MicManager()
    mic.acquire("holder_a")

    assert mic.acquire("stt", timeout=0.0) is False
    assert mic.owner == "holder_a"

    mic.release("holder_a")

    print("  PASS: Zero-timeout acquire fails immediately")


def test_reentrant_acquire_same_owner() -> None:
    """Same owner acquiring twice should block (not deadlock with timeout)."""
    mic = MicManager()
    mic.acquire("holder_a")

    assert mic.acquire("holder_a", timeout=0.1) is False, \
        "Same owner re-acquire should timeout (not deadlock)"

    mic.release("holder_a")

    print("  PASS: Re-entrant acquire times out (no deadlock)")


def run_all() -> None:
    print("\n=== ATOM v9 Phase 3 -- MicManager Tests ===\n")

    test_single_owner()
    test_release_and_reacquire()
    test_release_wrong_owner()
    test_release_when_free()
    test_owner_property()
    test_concurrent_handoff()
    test_acquire_timeout()
    test_reentrant_acquire_same_owner()

    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    run_all()
