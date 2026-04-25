"""ATOM -- Pending-Offer Registry (Sprint J: Jarvis Offer Protocol).

The Jarvis loop ATOM is supposed to deliver:

    User:  "How do I check the weather in Mumbai?"
    ATOM:  "Open the Weather app, type Mumbai, hit return, Boss.
            Want me to pull it up for you?"   <- THE OFFER
    User:  "Yes please."                       <- THE CONFIRM
    ATOM:  *executes weather action*           <- THE PAYOFF

Today the second turn falls through to the LLM because there is no
durable "the assistant just offered to do something" state. This module
provides exactly that: a single-slot, TTL-bounded ``PendingOffer``
record that the router checks BEFORE the intent engine on every new
utterance. If the user confirms within the window, the staged action
runs immediately (sub-100 ms, no LLM round-trip). If they deny or
switch topics, the offer is silently cleared.

Why a *single slot* and not a queue? Because Jarvis never says "you
have three pending things; pick one" -- the most recent offer is the
only relevant one. Any new offer evicts the previous one, and any
unrelated user turn evicts it too. This matches the conversational
contract a human partner expects and avoids "pending action zombies"
that surprise the user 10 minutes later.

Why separate from ``ConfirmationManager``? ``ConfirmationManager``
handles *dangerous* action staging ("you said `delete file foo` --
shall I?") with a stricter prompt, security gate, and 25 s timeout.
That flow STARTS with an explicit user request. The offer flow STARTS
with an explanation -- the user did NOT ask us to do it; we proposed.
Mixing the two would either weaken the security guarantees of
``ConfirmationManager`` or weaken the conversational naturalness of
``OfferRegistry``. So they live side by side and the router checks
``OfferRegistry`` first (cheap, no security implication) before
falling through to ``ConfirmationManager``.

Threading note: the registry is touched from a single asyncio thread
(the router) so we don't need a lock. If ever called from a
background worker, wrap the calls in a ``threading.Lock`` at the call
site.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.router.offer")


_DEFAULT_TTL_S = 60.0


@dataclass
class PendingOffer:
    """A single staged Jarvis-style offer awaiting user confirmation.

    Attributes
    ----------
    action:
        Router action name (must exist in ``Router._ACTION_DISPATCH``
        or be a tool name the LLM path can dispatch). When the user
        confirms, the router builds an ``IntentResult`` from this and
        passes it through ``_execute_action`` -- the exact same path
        a normal local intent would take, so security gates, late
        dispatch, and result emission all behave identically.
    args:
        Action arguments. Must be JSON-serialisable so the registry
        can be inspected and logged safely.
    offer_text:
        The human-readable one-liner ATOM appended to its reply, e.g.
        "Want me to pull up the weather for Mumbai, Boss?". Stored so
        the registry can echo "Got it, doing that now" with the same
        framing on confirmation.
    source_query:
        The user query that triggered the offer (truncated). Used for
        log breadcrumbs and to detect topic switches.
    source_response:
        The first ~200 chars of the assistant reply that contained
        the offer. Used for log breadcrumbs only.
    expires_at:
        ``time.monotonic()`` deadline. After this, ``OfferRegistry``
        treats the offer as already expired on the next access.
    metadata:
        Free-form dict for downstream consumers (e.g. ``"category":
        "weather"`` so analytics can group offers).
    """

    action: str
    args: dict[str, Any] = field(default_factory=dict)
    offer_text: str = ""
    source_query: str = ""
    source_response: str = ""
    expires_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def remaining_s(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())


class OfferRegistry:
    """Single-slot store for the most recent Jarvis offer.

    Lifecycle:
        stash(offer)   -> overwrites any previous slot
        peek()         -> returns the offer if live, else None (auto-evicts)
        consume()      -> returns the offer AND clears the slot
        clear(reason)  -> manual eviction (logged with reason)

    Every access checks expiry; a stale offer is dropped silently so
    callers never have to remember the TTL. The cleanup is O(1).
    """

    __slots__ = ("_offer", "_default_ttl_s", "_stashed_count", "_consumed_count")

    def __init__(self, default_ttl_s: float = _DEFAULT_TTL_S) -> None:
        self._offer: PendingOffer | None = None
        self._default_ttl_s = float(default_ttl_s)
        self._stashed_count = 0
        self._consumed_count = 0

    @property
    def has_pending(self) -> bool:
        """True iff a non-expired offer is staged."""
        return self.peek() is not None

    def peek(self) -> PendingOffer | None:
        """Return the live offer (if any) without consuming it.

        Auto-evicts an expired offer on the way out so callers always
        see a coherent view -- no half-stale records.
        """
        offer = self._offer
        if offer is None:
            return None
        if offer.is_expired:
            logger.debug(
                "OfferRegistry: dropping expired offer (action=%s, source='%s')",
                offer.action, offer.source_query[:60],
            )
            self._offer = None
            return None
        return offer

    def stash(
        self,
        action: str,
        args: dict[str, Any] | None = None,
        offer_text: str = "",
        source_query: str = "",
        source_response: str = "",
        ttl_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PendingOffer:
        """Stage a new offer, evicting any previous one.

        We deliberately do NOT keep a queue: a second offer arriving
        before the first was answered means the conversation moved
        on, and the user almost certainly wants the latest proposal.
        """
        ttl = float(ttl_s if ttl_s is not None else self._default_ttl_s)
        if ttl <= 0:
            raise ValueError(f"TTL must be positive, got {ttl}")
        if not action:
            raise ValueError("Cannot stash an offer with no action name")

        if self._offer is not None:
            logger.debug(
                "OfferRegistry: evicting previous offer (action=%s) for new (action=%s)",
                self._offer.action, action,
            )

        offer = PendingOffer(
            action=action,
            args=dict(args or {}),
            offer_text=offer_text or "",
            source_query=(source_query or "")[:200],
            source_response=(source_response or "")[:200],
            expires_at=time.monotonic() + ttl,
            metadata=dict(metadata or {}),
        )
        self._offer = offer
        self._stashed_count += 1
        logger.info(
            "OfferRegistry: staged '%s' (ttl=%.0fs, query='%s')",
            action, ttl, offer.source_query[:60],
        )
        return offer

    def consume(self) -> PendingOffer | None:
        """Pop and return the live offer, or None if expired/empty.

        After this returns the offer, the slot is empty -- a second
        confirmation would not re-fire the action.
        """
        offer = self.peek()
        if offer is None:
            return None
        self._offer = None
        self._consumed_count += 1
        logger.info(
            "OfferRegistry: consumed '%s' (age=%.1fs)",
            offer.action,
            float(self._default_ttl_s) - offer.remaining_s(),
        )
        return offer

    def clear(self, reason: str = "") -> None:
        """Manually drop the staged offer.

        Used on topic switches and explicit denials so the next user
        turn doesn't accidentally trigger an old action just because
        the user said "yes" to an unrelated question.
        """
        if self._offer is None:
            return
        offer = self._offer
        self._offer = None
        logger.info(
            "OfferRegistry: cleared '%s' (reason=%s)",
            offer.action, reason or "unspecified",
        )

    def stats(self) -> dict[str, Any]:
        """Snapshot for observability dashboards / health endpoints."""
        offer = self._offer
        return {
            "has_pending": offer is not None and not offer.is_expired,
            "stashed_total": self._stashed_count,
            "consumed_total": self._consumed_count,
            "current_action": offer.action if offer else None,
            "current_remaining_s": offer.remaining_s() if offer else 0.0,
            "default_ttl_s": self._default_ttl_s,
        }


# ── Process-wide singleton (mirrors get_command_cache, get_tool_registry) ──


_singleton: OfferRegistry | None = None


def get_offer_registry() -> OfferRegistry:
    """Lazy singleton accessor.

    The router stashes from one call site and consumes from another;
    a singleton avoids threading the registry through every helper.
    Tests can call ``reset_offer_registry()`` to start clean.
    """
    global _singleton
    if _singleton is None:
        _singleton = OfferRegistry()
    return _singleton


def reset_offer_registry(default_ttl_s: float = _DEFAULT_TTL_S) -> OfferRegistry:
    """Replace the singleton with a fresh instance.

    Useful for tests that want a known TTL or a guaranteed-empty slot.
    Production code should prefer ``get_offer_registry().clear()`` to
    drop a single offer without touching the counters.
    """
    global _singleton
    _singleton = OfferRegistry(default_ttl_s=default_ttl_s)
    return _singleton
