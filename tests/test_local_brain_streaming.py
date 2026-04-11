"""Focused tests for local-brain streaming edge cases."""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cursor_bridge.local_brain_controller import LocalBrainController


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit_long(self, event: str, **data) -> None:
        self.events.append((event, data))


class FakePreemptedLLM:
    async def generate_streaming(self, prompt: str, **kwargs):
        on_token = kwargs.get("on_token")
        if callable(on_token):
            on_token("Hello there.", False)
            on_token("", True)
        return "", True


class FakeCompletedLLM:
    async def generate_streaming(self, prompt: str, **kwargs):
        on_token = kwargs.get("on_token")
        if callable(on_token):
            on_token("Hello there.", False)
            on_token("", True)
        return "Hello there.", False


class DummyController:
    def __init__(self, llm) -> None:
        self._runtime_watchdog = None
        self._bus = FakeBus()
        self._llm = llm

    def _extract_complete_sentence(self, text: str):
        return LocalBrainController._extract_complete_sentence(text)


async def test_preempted_stream_skips_final_partial() -> None:
    controller = DummyController(FakePreemptedLLM())

    full_text, _first_token_ms, preempted = await LocalBrainController._run_llm_streaming(
        controller,
        "say hello",
        time.perf_counter(),
        emit_partial=True,
    )

    assert preempted is True
    assert full_text == ""
    assert not any(event == "partial_response" for event, _ in controller._bus.events)
    print("  PASS: Preempted stream does not replay a stale final chunk")


async def test_completed_stream_flushes_final_partial() -> None:
    controller = DummyController(FakeCompletedLLM())

    full_text, _first_token_ms, preempted = await LocalBrainController._run_llm_streaming(
        controller,
        "say hello",
        time.perf_counter(),
        emit_partial=True,
    )

    matches = [
        data
        for event, data in controller._bus.events
        if event == "partial_response"
    ]
    assert preempted is False
    assert full_text == "Hello there."
    assert len(matches) == 1
    assert matches[0]["text"] == "Hello there."
    assert matches[0]["is_last"] is True
    assert matches[0]["stream_id"]
    print("  PASS: Completed stream still flushes the final chunk")


async def run_all() -> None:
    print("\n=== Local Brain Streaming Tests ===\n")
    await test_preempted_stream_skips_final_partial()
    await test_completed_stream_flushes_final_partial()
    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    asyncio.run(run_all())
