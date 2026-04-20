"""
ATOM -- Sprint D4 focused tests.

Covers:
    1. Default routines load with sensible aliases.
    2. ``match()`` returns the correct (name, phase) for common phrasings.
    3. ``execute("deep_work", "enter")`` dispatches all steps via injected
       callable and sets the engine's ``active`` routine.
    4. Exit rolls the active back to ``None``.
    5. Custom routines can be supplied via ``config["routines"]``.
    6. Intent matcher routes "enter focus mode" to ``run_routine``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.intent_engine import routine_intents
from core.proactive.routine_engine import RoutineEngine


def test_default_routines_loaded() -> None:
    eng = RoutineEngine({})
    names = {r.name for r in eng.list_routines()}
    assert {"deep_work", "bedtime", "meeting"}.issubset(names)


def test_match_enter_deep_work_aliases() -> None:
    eng = RoutineEngine({})
    for phrase in [
        "enter deep work mode",
        "start focus mode",
        "activate deep work",
        "begin focus",
        "turn on deep work",
    ]:
        match = eng.match(phrase)
        assert match is not None, f"no match for {phrase!r}"
        name, phase = match
        assert name == "deep_work"
        assert phase == "enter"


def test_match_exit_bedtime() -> None:
    eng = RoutineEngine({})
    match = eng.match("exit bedtime mode")
    assert match is not None
    assert match == ("bedtime", "exit")


def test_match_unknown_returns_none() -> None:
    eng = RoutineEngine({})
    assert eng.match("what is the weather") is None
    assert eng.match("random chatter") is None


def test_execute_enter_dispatches_all_steps() -> None:
    calls: list[tuple[str, dict]] = []

    def dispatcher(kind: str, args: dict) -> str:
        calls.append((kind, dict(args)))
        return "ok"

    eng = RoutineEngine({}, dispatcher=dispatcher)
    spoken = eng.execute("deep_work", "enter")

    assert "deep" in spoken.lower() or "quiet" in spoken.lower()
    kinds = [c[0] for c in calls]
    assert "volume" in kinds
    assert "assistant_mode" in kinds
    assert "brain_profile" in kinds
    assert eng.active == "deep_work"


def test_execute_exit_clears_active() -> None:
    def dispatcher(kind: str, args: dict) -> str:
        return "ok"

    eng = RoutineEngine({}, dispatcher=dispatcher)
    eng.execute("deep_work", "enter")
    assert eng.active == "deep_work"
    eng.execute("deep_work", "exit")
    assert eng.active is None


def test_custom_routine_from_config() -> None:
    cfg = {
        "routines": [
            {
                "name": "writing_sprint",
                "aliases": ["writing sprint", "writing mode"],
                "enter_say": "Starting writing sprint.",
                "enter_steps": [
                    {"kind": "assistant_mode", "mode": "silent_mode"},
                    {"kind": "volume", "percent": 0},
                ],
            }
        ]
    }
    calls: list[tuple[str, dict]] = []

    def dispatcher(kind: str, args: dict) -> str:
        calls.append((kind, args))
        return "ok"

    eng = RoutineEngine(cfg, dispatcher=dispatcher)
    match = eng.match("enter writing sprint")
    assert match == ("writing_sprint", "enter")
    spoken = eng.execute(*match)
    assert "Starting writing sprint" in spoken
    assert [c[0] for c in calls] == ["assistant_mode", "volume"]


def test_routine_from_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "routines.json"
        rp.write_text(
            '{"routines": [{"name": "morning_routine", "aliases": ["morning routine"],'
            ' "enter_say": "Ready for morning.", "enter_steps": [{"kind": "volume", "percent": 40}]}]}'
        )
        eng = RoutineEngine({"routine_path": str(rp)})
        names = {r.name for r in eng.list_routines()}
        assert "morning_routine" in names


def test_unknown_step_reports_issue_in_spoken() -> None:
    cfg = {
        "routines": [
            {
                "name": "broken",
                "aliases": ["broken mode"],
                "enter_steps": [{"kind": "not_real"}],
                "enter_say": "Trying broken mode.",
            }
        ]
    }

    def dispatcher(kind: str, args: dict) -> str:
        return "ok"

    eng = RoutineEngine(cfg, dispatcher=dispatcher)
    spoken = eng.execute("broken", "enter")
    assert "Trying broken mode" in spoken
    assert "issues" in spoken.lower() or "not_real" in spoken.lower()


def test_intent_matcher_routes_to_run_routine() -> None:
    eng = RoutineEngine({})
    routine_intents.set_routine_engine(eng)
    try:
        res = routine_intents.check("enter deep work mode")
        assert res is not None
        assert res.intent == "run_routine"
        assert res.action == "run_routine"
        assert (res.action_args or {}).get("name") == "deep_work"
        assert (res.action_args or {}).get("phase") == "enter"

        res2 = routine_intents.check("exit bedtime mode")
        assert res2 is not None
        assert (res2.action_args or {}).get("name") == "bedtime"
        assert (res2.action_args or {}).get("phase") == "exit"

        res3 = routine_intents.check("tell me the time")
        assert res3 is None
    finally:
        routine_intents.set_routine_engine(None)


if __name__ == "__main__":
    test_default_routines_loaded()
    test_match_enter_deep_work_aliases()
    test_match_exit_bedtime()
    test_match_unknown_returns_none()
    test_execute_enter_dispatches_all_steps()
    test_execute_exit_clears_active()
    test_custom_routine_from_config()
    test_routine_from_file()
    test_unknown_step_reports_issue_in_spoken()
    test_intent_matcher_routes_to_run_routine()
    print("[D4] All routine engine tests passed.")
