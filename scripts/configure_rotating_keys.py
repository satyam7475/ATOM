#!/usr/bin/env python3
"""ATOM — One-shot setup for the rotating cloud lane keys.

Sprint Ω.9 (Apr 26 2026). Boss runs three free OpenAI-compatible cloud
providers in a round-robin: Groq, NVIDIA NIM, Cerebras. This helper
encrypts the three keys via :class:`core.secure_credentials.CredentialManager`
so they never sit in plaintext on disk.

Usage:
    GROQ_API_KEY=gsk_...        \\
    NVIDIA_API_KEY=nvapi-...    \\
    CEREBRAS_API_KEY=csk-...    \\
        python3 scripts/configure_rotating_keys.py

Requires:
    ATOM_MASTER_PASSWORD set in the environment (same vault used by
    Gemini keys). Run scripts/setup_api_keys.py once first if you've
    never set up the vault.

Behaviour:
    - Reads the three keys from environment variables.
    - Validates the prefix on each (gsk_ / nvapi- / csk-).
    - Stores them under credential ids ``groq``, ``nvidia``, ``cerebras``.
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
)

_PROVIDERS = (
    ("Groq",     "GROQ_API_KEY",     GROQ,       ("gsk_",)),
    ("NVIDIA",   "NVIDIA_API_KEY",   NVIDIA_NIM, ("nvapi-",)),
    ("Cerebras", "CEREBRAS_API_KEY", CEREBRAS,   ("csk-",)),
)


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
