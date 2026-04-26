#!/usr/bin/env python3
"""ATOM — One-shot setup for the rotating cloud lane keys.

Sprint Ω.9 (Apr 26 2026): three free OpenAI-compatible cloud providers
(Groq, NVIDIA NIM, Cerebras) in a round-robin.

Sprint Ω.11 (Apr 26 2026): rotation extended to multi-vendor — Gemini
(Google generateContent) joined the tier-1 pool alongside Groq. The
Anthropic/Claude slot was retired on 2026-04-27 (owner does not hold an
``sk-ant-`` key); add it back here if a key is ever provisioned.

This helper accepts the supported vendors via env vars and encrypts
each one via :class:`core.secure_credentials.CredentialManager` so they
never sit in plaintext on disk.

Usage:
    GROQ_API_KEY=gsk_...           \\
    GOOGLE_API_KEY=AIza...         \\
    NVIDIA_API_KEY=nvapi-...       \\
    CEREBRAS_API_KEY=csk-...       \\
        python3 scripts/configure_rotating_keys.py

Any subset is fine — slots without a key stay cold and the picker skips
them transparently.

Requires:
    ATOM_MASTER_PASSWORD set in the environment (same vault used by
    Gemini keys). Run scripts/setup_api_keys.py once first if you've
    never set up the vault.

Behaviour:
    - Reads each key from its env variable.
    - Validates the prefix on each (gsk_ / AIza / nvapi- / csk-).
    - Stores them under canonical credential ids.
    - Re-running is idempotent and safe — overwrites existing entries.
    - Never prints the key value; only the prefix and length.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.secure_credentials import CredentialManager  # noqa: E402
from core.secrets_manager import (  # noqa: E402
    GROQ,
    NVIDIA_NIM,
    CEREBRAS,
    GEMINI_FAST,
    GEMINI_PRO,
)

# (label, env var, credential id, allowed prefixes). Tier-1 vendors come
# first so the on-screen list reflects rotation priority.
_PROVIDERS = (
    ("Groq",     "GROQ_API_KEY",      GROQ,       ("gsk_",)),
    ("Gemini",   "GOOGLE_API_KEY",    GEMINI_FAST, ("AIza",)),
    ("Cerebras", "CEREBRAS_API_KEY",  CEREBRAS,   ("csk-",)),
    ("NVIDIA",   "NVIDIA_API_KEY",    NVIDIA_NIM, ("nvapi-",)),
)

# Gemini key gets mirrored to gemini_pro too so the standalone
# GeminiClient lane (cloud.provider="gemini") works without a second
# setup step.
_KEY_MIRRORS = {
    GEMINI_FAST: (GEMINI_PRO,),
}


def _mask(key: str) -> str:
    if len(key) <= 12:
        return f"<{len(key)} chars>"
    return f"{key[:6]}…{key[-4:]} ({len(key)} chars)"


def main() -> int:
    master = os.environ.get("ATOM_MASTER_PASSWORD", "").strip()
    if not master:
        print(
            "ERROR: ATOM_MASTER_PASSWORD is not set. Run "
            "scripts/setup_api_keys.py once to initialise the vault.",
            file=sys.stderr,
        )
        return 2

    try:
        cm = CredentialManager(master)
    except Exception as exc:  # corrupt vault, wrong password, etc.
        print(f"ERROR: cannot open credential vault: {exc}", file=sys.stderr)
        return 3

    stored = 0
    skipped = 0
    for label, env_var, cred_id, prefixes in _PROVIDERS:
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            print(f"  {label:<8} | {env_var:<18} | not provided — skipped")
            skipped += 1
            continue
        if not any(raw.startswith(p) for p in prefixes):
            print(
                f"  {label:<8} | {env_var:<18} | unexpected prefix "
                f"(expected one of {prefixes}) — refusing to store",
                file=sys.stderr,
            )
            skipped += 1
            continue
        cm.set_credential(cred_id, raw)
        print(f"  {label:<8} | {env_var:<18} | stored {_mask(raw)}")
        stored += 1
        for mirror_id in _KEY_MIRRORS.get(cred_id, ()):
            cm.set_credential(mirror_id, raw)
            print(f"  {'':<8} | {'':<18} |   mirrored → {mirror_id}")

    if stored == 0:
        print("\nNo keys stored. Set the env vars and try again.", file=sys.stderr)
        return 1

    available = cm.list_credentials()
    print(f"\nVault now contains: {sorted(available)}")
    print(
        f"Stored: {stored}  |  skipped: {skipped}  |  "
        "rotation lane will be active on next ATOM boot."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
