"""ATOM personality package -- per-owner profile + style adaptation.

This package was added in Sprint P4 (Apr 26 2026). It owns the Boss-
specific learning surfaces:

* :mod:`owner_profile` — persistent corrections memory + pronunciation
  dictionary (P4.1 + P4.3).
* :mod:`owner_style` — rolling style fingerprint (Hinglish ratio,
  verbosity, tone) used by the prompt builder to bias the LLM (P4.2).

All state is stored under ``data/owner_profile.sqlite3`` keyed by
owner name so a single Mac with a single ATOM install never crosses
profiles between users.
"""

from __future__ import annotations

from core.personality.owner_profile import OwnerProfile, get_owner_profile
from core.personality.owner_style import OwnerStyleAdapter, get_owner_style

__all__ = [
    "OwnerProfile",
    "OwnerStyleAdapter",
    "get_owner_profile",
    "get_owner_style",
]
