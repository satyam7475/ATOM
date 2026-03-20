"""
ATOM v14 -- Privacy filter for corporate data protection.

Redacts sensitive patterns from text before sending to external LLM.
All patterns are compiled at import time for near-zero runtime cost.

Catches:
    - API keys, tokens, secrets, passwords in key=value format
    - PEM private keys
    - Bearer/Basic auth headers
    - GitHub PATs (ghp_, gho_, github_pat_)
    - Google API keys (AIzaSy...)
    - AWS access keys (AKIA...)
    - JDBC / database connection strings
    - .env variable assignments with secrets
    - MongoDB connection URIs
    - Email addresses
"""

from __future__ import annotations

import re

_REDACTED = "[REDACTED]"

_PATTERNS: list[re.Pattern[str]] = [
    # Key=value secrets (API keys, tokens, passwords, etc.)
    re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|token|password|passwd|pwd|"
        r"access[_-]?key|private[_-]?key|client[_-]?secret|"
        r"auth[_-]?token|refresh[_-]?token|session[_-]?id|"
        r"db[_-]?password|database[_-]?password|"
        r"encryption[_-]?key|signing[_-]?key|master[_-]?key)"
        r"\s*[:=]\s*\S+"
    ),
    # PEM private keys
    re.compile(
        r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|ENCRYPTED\s+)?PRIVATE\s+KEY-----"
        r"[\s\S]*?"
        r"-----END\s+(?:RSA\s+|EC\s+|DSA\s+|ENCRYPTED\s+)?PRIVATE\s+KEY-----"
    ),
    # Auth headers
    re.compile(r"(?i)(bearer|basic)\s+[A-Za-z0-9+/=_\-]{20,}"),
    # GitHub PATs
    re.compile(r"(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]{20,}"),
    # Google API keys
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{33}"),
    # AWS access keys
    re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    # JDBC connection strings (PostgreSQL, MySQL, Oracle, SQL Server, etc.)
    re.compile(
        r"jdbc:[a-z]+://[^\s\"']+",
        re.I,
    ),
    # MongoDB connection URIs
    re.compile(
        r"mongodb(?:\+srv)?://[^\s\"']+",
        re.I,
    ),
    # Generic database connection strings (user:password@host pattern)
    re.compile(
        r"(?:postgres|mysql|redis|amqp|ftp)s?://[^\s\"']+",
        re.I,
    ),
    # .env-style secret assignments (SECRET_KEY=..., DB_PASS=..., etc.)
    re.compile(
        r"(?i)^(?:export\s+)?(?:[A-Z_]*(?:SECRET|PASSWORD|PASSWD|PWD|TOKEN|KEY|CREDENTIAL|AUTH)[A-Z_]*)"
        r"\s*=\s*\S+",
        re.MULTILINE,
    ),
    # Email addresses
    re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
]


def redact(text: str) -> str:
    """Replace all sensitive patterns in *text* with [REDACTED].

    Safe to call on any string -- returns the original if nothing matches.
    """
    if not text:
        return text
    for pattern in _PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text
