from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cursor_bridge.local_brain_controller import LocalBrainController  # noqa: E402


def test_strip_model_artifacts_extracts_assistant_reply() -> None:
    controller = object.__new__(LocalBrainController)
    text = (
        "User: what is docker "
        "ATOM: Docker packages apps and dependencies into isolated containers. "
        "User: thanks"
    )
    assert (
        controller._strip_model_artifacts("what is docker", text)
        == "Docker packages apps and dependencies into isolated containers."
    )


def test_strip_model_artifacts_drops_transcript_noise() -> None:
    controller = object.__new__(LocalBrainController)
    text = "ATOM: ATOM: ATOM: User: what is docker ATOM:"
    assert controller._strip_model_artifacts("what is docker", text) == ""


if __name__ == "__main__":
    test_strip_model_artifacts_extracts_assistant_reply()
    test_strip_model_artifacts_drops_transcript_noise()
    print("test_local_brain_controller: ALL PASSED")
