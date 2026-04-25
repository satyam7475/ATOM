from __future__ import annotations

from core import json_fast


def test_json_fast_round_trip_preserves_shape() -> None:
    payload = {"event": "hello", "values": [1, 2, 3], "ok": True}

    encoded = json_fast.dumps(payload)

    assert isinstance(encoded, str)
    assert json_fast.loads(encoded) == payload


def test_json_fast_default_handler_matches_hot_path_contract() -> None:
    class Custom:
        def __repr__(self) -> str:
            return "custom-value"

    encoded = json_fast.dumps({"x": Custom()}, default=repr)

    assert json_fast.loads(encoded) == {"x": "custom-value"}
