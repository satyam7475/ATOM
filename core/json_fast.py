"""Shared JSON helpers with an orjson fast path.

Callers should use these wrappers on hot paths that need a stable str-based
API. When orjson is not installed, behavior falls back to the stdlib json
module without changing the caller contract.
"""

from __future__ import annotations

import json as _json
from typing import Any, Callable

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - depends on optional wheel availability
    _orjson = None

JSONDecodeError = _json.JSONDecodeError
DefaultFn = Callable[[Any], Any]


def dumps(
    obj: Any,
    *,
    default: DefaultFn | None = None,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize to a JSON str.

    ``separators`` is accepted for stdlib compatibility; orjson already emits
    compact JSON by default, matching the hot-path use cases in ATOM.
    """
    if _orjson is not None:
        option = _orjson.OPT_SORT_KEYS if sort_keys else 0
        return _orjson.dumps(obj, default=default, option=option).decode("utf-8")
    kwargs: dict[str, Any] = {}
    if default is not None:
        kwargs["default"] = default
    if sort_keys:
        kwargs["sort_keys"] = True
    if separators is not None:
        kwargs["separators"] = separators
    return _json.dumps(obj, **kwargs)


def loads(data: str | bytes | bytearray) -> Any:
    """Deserialize JSON from str or bytes."""
    if _orjson is not None:
        return _orjson.loads(data)
    return _json.loads(data)


def is_fast_path_enabled() -> bool:
    """Expose whether orjson is active for diagnostics/tests."""
    return _orjson is not None
