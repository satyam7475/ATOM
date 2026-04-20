"""ATOM -- one-shot environment-secret scrub.

Why: ATOM reads third-party API keys (GEMINI_API_KEY, HF_TOKEN, OpenAI, ...)
from ``os.environ`` at boot so users can point ATOM at their own providers
without editing ``settings.json``. Once those values are captured in the
in-memory ``SecurityGateway`` / cloud clients, keeping them in ``os.environ``
is pure attack surface:

* any subprocess ATOM spawns inherits them by default;
* any crash dump / traceback handler that dumps the environment leaks them;
* any plugin that logs ``os.environ`` for diagnostics exfiltrates them.

This module snapshots the sensitive vars, then blanks them. The snapshot is
returned to the caller so ``SecurityGateway`` / gemini_client can still pull
the value once at configure time. After scrub, ``os.environ`` contains only
empty strings for those keys; the live secret lives inside the gateway closure.

Idempotent: calling :func:`scrub_sensitive_env` twice is a no-op the second
time (the snapshot is captured the first call only).
"""

from __future__ import annotations

import logging
import os
from typing import Mapping

logger = logging.getLogger("atom.security.secret_scrub")

# Env vars we know carry secrets. Kept explicit (not regex) so an accidental
# rename doesn't silently start sniffing unrelated vars.
_SENSITIVE_ENV_VARS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "REPLICATE_API_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "ATOM_DASHBOARD_TOKEN",
    "ATOM_ADMIN_TOKEN",
    "ATOM_CLOUD_KEY",
    "NOTION_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_USER_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)

_already_scrubbed: bool = False
_snapshot: dict[str, str] = {}


def sensitive_env_vars() -> tuple[str, ...]:
    """Return the list of env vars ATOM treats as secrets (for tests / docs)."""
    return _SENSITIVE_ENV_VARS


def snapshot_sensitive_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return {name: value} for every secret env var that is set and non-empty.

    Does not mutate the environment. Safe to call multiple times.
    """
    source = os.environ if env is None else env
    out: dict[str, str] = {}
    for name in _SENSITIVE_ENV_VARS:
        val = source.get(name)
        if val and val.strip():
            out[name] = val.strip()
    return out


def scrub_sensitive_env(
    *,
    env: dict[str, str] | None = None,
    preserve: tuple[str, ...] = (),
) -> dict[str, str]:
    """Snapshot secrets into an in-process dict, then blank them in ``env``.

    Returns the snapshot so callers can hand-off to ``SecurityGateway`` /
    cloud clients. After this call the env (default ``os.environ``) no longer
    carries the secret -- subprocesses inherit an empty string instead.

    ``preserve`` lets callers keep specific vars untouched (useful for the
    dashboard token when someone deliberately exports it for a single run).
    """
    global _already_scrubbed, _snapshot
    target = os.environ if env is None else env
    if _already_scrubbed:
        return dict(_snapshot)

    snap = snapshot_sensitive_env(target)
    preserved = {p for p in preserve if p}
    cleared: list[str] = []
    for name, _val in snap.items():
        if name in preserved:
            continue
        try:
            target[name] = ""
            cleared.append(name)
        except Exception:
            logger.debug("Could not clear env var %s", name, exc_info=True)

    _snapshot = snap
    _already_scrubbed = True
    if cleared:
        logger.info(
            "SecretScrub: cleared %d secret env vars from process environment "
            "(%s). Values retained in-process.",
            len(cleared),
            ", ".join(sorted(cleared)),
        )
    else:
        logger.debug("SecretScrub: no sensitive env vars were set.")
    return dict(snap)


def reset_for_tests() -> None:
    """Clear the module-level latches so unit tests can re-run the scrub."""
    global _already_scrubbed, _snapshot
    _already_scrubbed = False
    _snapshot = {}


__all__ = [
    "scrub_sensitive_env",
    "snapshot_sensitive_env",
    "sensitive_env_vars",
    "reset_for_tests",
]
