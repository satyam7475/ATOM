#!/usr/bin/env python3
"""Explicit embedding-provider shadow comparison.

Runs the active configured provider against the legacy SentenceTransformer
provider for a fixed phrase set. This is intentionally opt-in because it may
load both embedding runtimes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.boot.config_loader import load_config
from core.embedding_engine import EmbeddingEngine


DEFAULT_PHRASES = [
    "what time is it",
    "open spotify",
    "what did we talk about earlier",
    "describe what's on screen",
    "plan my day",
    "summarize this and email it",
]


def _phrases_from_args(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PHRASES)
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare active embedding backend with legacy ST.")
    parser.add_argument("--phrases", help="Optional newline-delimited phrase file")
    parser.add_argument("--backend", help="Override embedding.backend for this comparison")
    parser.add_argument("--model", help="Override embedding.model for this comparison")
    args = parser.parse_args()

    config = load_config()
    embedding_cfg = dict((config.get("embedding") or {}))
    if args.backend:
        embedding_cfg["backend"] = args.backend
    if args.model:
        embedding_cfg["model"] = args.model
    embedding_cfg["warm_file"] = {"enabled": False}
    config["embedding"] = embedding_cfg

    engine = EmbeddingEngine(config)
    report = engine.shadow_compare_phrases(_phrases_from_args(args.phrases))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("error"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
