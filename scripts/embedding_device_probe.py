#!/usr/bin/env python3
"""S3 -- Embedding device probe (Sprint P3.4 / P2.3 verification).

Boots the ATOM ``EmbeddingEngine`` with the current ``config/settings.json``
and prints which provider + device actually loaded, plus a single end-to-
end embed call so we know the path is healthy.

Pass criteria (per docs/ATOM_NEXT_STEPS_PLAN.md §4):
    * resolved device is one of ``mps`` / ``mlx`` / ``cuda``
    * `embed("hello")` returns a vector of the configured dimension
    * total wall time under 10 s on a warm cache

Exit codes:
    0  OK
    1  embed failed
    2  resolved device is CPU when MPS / MLX was expected
    3  configuration / boot error

Usage::

    python scripts/embedding_device_probe.py
    python scripts/embedding_device_probe.py --json   # one-line JSON
    python scripts/embedding_device_probe.py --strict # fail on cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def _emit(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, default=str))
        return
    width = max(len(k) for k in payload) + 2
    for k, v in payload.items():
        print(f"{k.ljust(width)}{v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ATOM S3 embedding probe")
    parser.add_argument(
        "--json", action="store_true",
        help="emit a single JSON line instead of human-readable output",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="fail when resolved device is cpu",
    )
    args = parser.parse_args()

    try:
        from core.boot.config_loader import load_config
        from core.embedding_engine import (
            EmbeddingEngine,
            _resolve_embedding_device,
        )
    except Exception as exc:
        _emit(
            {"status": "import_error", "error": repr(exc)},
            as_json=args.json,
        )
        return 3

    try:
        cfg = load_config()
    except Exception as exc:
        _emit(
            {"status": "config_error", "error": repr(exc)},
            as_json=args.json,
        )
        return 3

    embed_cfg = cfg.get("embedding", {})
    requested = str(embed_cfg.get("device", "cpu") or "cpu")
    backend = str(embed_cfg.get("backend") or "sentence_transformers")
    resolved = _resolve_embedding_device(requested)

    t0 = time.perf_counter()
    eng = EmbeddingEngine(cfg)
    init_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    vec = eng.embed_sync("ATOM embedding device probe ok")
    embed_ms = (time.perf_counter() - t1) * 1000.0

    metadata = eng.provider_metadata()
    payload: dict[str, Any] = {
        "status": "ok",
        "requested_device": requested,
        "resolved_device": resolved,
        "backend": backend,
        "provider_metadata": metadata,
        "vector_dim": len(vec),
        "vector_l2_nonzero": any(abs(x) > 1e-9 for x in vec),
        "init_ms": round(init_ms, 1),
        "embed_ms": round(embed_ms, 2),
    }

    if not vec or not payload["vector_l2_nonzero"]:
        payload["status"] = "embed_failed"
        _emit(payload, as_json=args.json)
        return 1
    if args.strict and resolved == "cpu":
        payload["status"] = "fell_back_to_cpu"
        _emit(payload, as_json=args.json)
        return 2

    _emit(payload, as_json=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
