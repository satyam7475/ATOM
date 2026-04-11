"""
Tests for macOS Spotlight engine (mdfind wrapper).

Run: python3 -m tests.test_spotlight_engine
"""

from __future__ import annotations

import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeRunner:
    def __init__(self, response: tuple[int, str, str]) -> None:
        self.response = response
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, command: list[str], timeout: float) -> tuple[int, str, str]:
        self.calls.append((command, timeout))
        return self.response


def test_search_parses_paths() -> None:
    from core.macos.spotlight_engine import SpotlightEngine

    out = "/a/one.txt\n/b/two.pdf\n"
    fake = _FakeRunner((0, out, ""))
    eng = SpotlightEngine(runner=fake)
    hits = eng.search("kMDItemFSName == '*.pdf'c", limit=10, timeout=5.0)
    assert hits == [{"path": "/a/one.txt"}, {"path": "/b/two.pdf"}]
    assert fake.calls[0][0][:3] == ["mdfind", "-limit", "10"]
    assert "kMDItemFSName" in fake.calls[0][0][-1]
    print("  PASS: search parses paths")


def test_find_first_path() -> None:
    from core.macos.spotlight_engine import SpotlightEngine

    eng = SpotlightEngine(runner=_FakeRunner((0, "/Applications/Foo.app\n", "")))
    assert eng.find_first_path("query", timeout=1.0) == "/Applications/Foo.app"
    print("  PASS: find_first_path")


def test_non_darwin_returns_empty() -> None:
    from core.macos.spotlight_engine import SpotlightEngine

    with mock.patch("sys.platform", "linux"):
        eng = SpotlightEngine(runner=_FakeRunner((0, "/nope\n", "")))
        assert eng.search("anything") == []
    print("  PASS: non-darwin returns empty")


def test_spotlight_search_top_level() -> None:
    from core.macos import spotlight_engine as se
    from core.macos.spotlight_engine import SpotlightEngine as RealSpotlightEngine
    from core.macos.spotlight_engine import spotlight_search

    fake = _FakeRunner((0, "/only\n", ""))

    class _Shim:
        def __init__(self, runner: _FakeRunner) -> None:
            self._e = RealSpotlightEngine(runner=runner)

        def search(self, query: str, limit: int = 10, timeout: float = 10.0):
            return self._e.search(query, limit=limit, timeout=timeout)

    with mock.patch("sys.platform", "darwin"):
        with mock.patch.object(se, "SpotlightEngine", lambda r=fake: _Shim(r)):
            hits = spotlight_search("q", 5)
    assert hits == [{"path": "/only"}]
    assert fake.calls[0][0] == ["mdfind", "-limit", "5", "q"]
    print("  PASS: spotlight_search top-level")


def test_file_intent_spotlight_phrases() -> None:
    try:
        from core.intent_engine.file_intents import check
    except ModuleNotFoundError as exc:
        print(f"  SKIP: file intents ({exc.name})")
        return

    r = check("search my mac for ATOM readme")
    assert r is not None
    assert r.action == "spotlight_search"
    assert r.action_args["query"] == "ATOM readme"

    r2 = check("spotlight for budget.xlsx")
    assert r2 and r2.action_args["query"] == "budget.xlsx"

    r3 = check("spotlight notes draft")
    assert r3 and r3.action_args["query"] == "notes draft"
    print("  PASS: file intents spotlight phrases")


if __name__ == "__main__":
    test_search_parses_paths()
    test_find_first_path()
    test_non_darwin_returns_empty()
    test_spotlight_search_top_level()
    test_file_intent_spotlight_phrases()
    print("All spotlight tests passed.")
