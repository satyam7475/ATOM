"""
ATOM V7 — Single gate for tool/actions: trust tier + SecurityPolicy + sanitization.

OS calls should go only through Router/ActionExecutor; this class is for
callers that want an explicit sandbox entry (e.g. future plugins).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.trust_model import TrustLevel, trust_allows_action

if TYPE_CHECKING:
    from core.security_policy import SecurityPolicy

logger = logging.getLogger("atom.action_sandbox")


class ActionSandbox:
    def __init__(self, policy: "SecurityPolicy") -> None:
        self._policy = policy

    def prepare_text_for_llm(self, text: str) -> tuple[str, bool]:
        """Delegate to SecurityPolicy sanitization before model input."""
        from core.security_policy import SecurityPolicy
        return SecurityPolicy.sanitize_input(text)

    def gate(
        self,
        action: str,
        args: dict[str, Any] | None,
        trust: TrustLevel = TrustLevel.USER,
    ) -> tuple[bool, str]:
        if not trust_allows_action(trust, action):
            reason = f"Trust {trust.value} insufficient for action '{action}'"
            logger.warning("ActionSandbox: %s", reason)
            return False, reason
        from core.security.action_signing import merge_signed_args

        merged = merge_signed_args(self._policy, action, args)
        return self._policy.allow_action(action, merged)


__all__ = ["ActionSandbox"]
