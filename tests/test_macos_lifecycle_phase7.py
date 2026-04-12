"""Phase 7.2 macOS lifecycle helpers + MemoryGraph pressure (importable tests).

Run: python3 -m tests.test_macos_lifecycle_phase7
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_SAMPLE_SPAUDIO = """
Audio:

    Devices:

        MacBook Air Microphone:

          Default Input Device: Yes
          Input Channels: 1

        MacBook Air Speakers:

          Default Output Device: Yes
          Default System Output Device: Yes
          Output Channels: 2

        Satyam’s AirPods Pro:

          Default Output Device: No
          Output Channels: 2
"""


def test_parse_default_system_output_builtin() -> None:
    from core.macos.phase7_lifecycle import parse_default_system_output_device

    name = parse_default_system_output_device(_SAMPLE_SPAUDIO)
    assert name == "MacBook Air Speakers"


def test_parse_default_system_output_minimal_airpods() -> None:
    from core.macos.phase7_lifecycle import parse_default_system_output_device

    text = """
        Boss AirPods Pro:

          Default System Output Device: Yes
          Output Channels: 2
    """
    assert parse_default_system_output_device(text) == "Boss AirPods Pro"


def test_read_kern_boottime_dict_parses() -> None:
    from core.macos.phase7_lifecycle import read_kern_boottime_dict
    from unittest import mock

    fake = mock.Mock(returncode=0, stdout="{ sec = 1700000000, usec = 0 }\n")
    with mock.patch("subprocess.run", return_value=fake):
        d = read_kern_boottime_dict()
    assert d == {"sec": 1700000000, "usec": 0}


def test_memory_graph_pressure_cycle() -> None:
    from brain.memory_graph import MemoryGraph

    fd, path = tempfile.mkstemp(suffix="_mg_p7test.db")
    os.close(fd)
    try:
        g = MemoryGraph(
            path,
            {"memory": {"pressure_threshold_pct": 85.0, "pressure_relief_pct": 75.0}},
        )
        assert g.apply_memory_pressure(90.0)["active"] is True
        assert g.apply_memory_pressure(74.0)["active"] is False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> None:
    test_parse_default_system_output_builtin()
    test_parse_default_system_output_minimal_airpods()
    test_read_kern_boottime_dict_parses()
    test_memory_graph_pressure_cycle()
    print("test_macos_lifecycle_phase7: OK")


if __name__ == "__main__":
    main()
