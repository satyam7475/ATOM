"""
ATOM — Security Gateway (Cloud Isolation Wall).

The trust boundary between ATOM's local intelligence and untrusted cloud
services. Every outbound query passes through this gateway before reaching
any external API.

Responsibilities:
  1. Sanitize outbound queries — strip file paths, IPs, tokens, emails,
     system references, and any personally identifiable data.
  2. Gate cloud access — reject queries containing system commands,
     local data references, or exceeding length limits.
  3. Tag cloud responses — mark all cloud-sourced data as untrusted
     so it never feeds directly into tool execution.
  4. Rate limiting — enforce requests-per-minute caps (Gemini free tier).
  5. Audit logging — log all cloud interactions to SecurityAuditTrail.

NON-NEGOTIABLE: Cloud NEVER executes tools, sees system data, or
accesses memory directly. Cloud is a brain extender, not a brain controller.

Owner: Satyam
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.security_gateway")


# ── Sensitive pattern definitions ────────────────────────────────────

_PATH_PATTERN = re.compile(
    r"(?:/(?:Users|home|var|etc|tmp|opt|usr|private|Applications|Library)"
    r"[/\\][^\s\"'`,;)}\]]+)",
    re.I,
)

_IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

_TOKEN_PATTERN = re.compile(
    r"(?:(?:token|key|secret|password|passwd|api_key|apikey|auth|bearer"
    r"|credential|access_key|private_key)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{8,}['\"]?)",
    re.I,
)

_ENV_VAR_PATTERN = re.compile(
    r"\$\{?[A-Z_][A-Z0-9_]*\}?"
)

_SYSTEM_CMD_PATTERN = re.compile(
    r"\b(?:sudo|rm\s+-rf|chmod|chown|kill\s+-9|pkill|launchctl"
    r"|defaults\s+write|diskutil|csrutil|nvram|shutdown"
    r"|reboot|halt|systemctl)\b",
    re.I,
)

_CONFIG_REF_PATTERN = re.compile(
    r"\b(?:config/|settings\.json|\.env|\.plist|keychain|vault\.enc"
    r"|\.ssh/|\.gnupg/|\.aws/)\b",
    re.I,
)

# Blocked words that should never leave the system
_BLOCKED_LITERALS = frozenset({
    "password", "passwd", "secret_key", "private_key",
    "access_token", "refresh_token", "session_token",
    "vault.enc", ".vault_key", "lockout.json",
})


@dataclass
class CloudAuditEntry:
    """Record of a cloud interaction for audit trail."""
    timestamp: float
    query_hash: str  # SHA-256 of original query (not the query itself)
    sanitized_length: int
    allowed: bool
    block_reason: str = ""
    provider: str = "gemini"
    response_length: int = 0
    latency_ms: float = 0.0


@dataclass
class _RateLimitState:
    """Token bucket rate limiter state."""
    tokens: float = 10.0
    max_tokens: float = 10.0
    refill_rate: float = 0.167  # ~10 per minute
    last_refill: float = field(default_factory=time.monotonic)

    def try_consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class SecurityGateway:
    """Cloud isolation wall — all outbound traffic passes through here.

    Usage:
        gateway = SecurityGateway(config)

        # Before sending to cloud:
        allowed, reason = gateway.allow_cloud(query, intent="knowledge")
        if allowed:
            safe_query = gateway.sanitize_outbound(query)
            response = await gemini.ask(safe_query)
            tagged = gateway.tag_cloud_response(response)
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("security_gateway", {})
        self._max_outbound_length = int(cfg.get("max_outbound_length", 500))
        self._block_system_paths = bool(cfg.get("block_system_paths", True))
        self._audit_cloud_calls = bool(cfg.get("audit_cloud_calls", True))
        self._max_rpm = int(cfg.get("max_requests_per_minute", 10))

        self._rate_limiter = _RateLimitState(
            max_tokens=float(self._max_rpm),
            refill_rate=self._max_rpm / 60.0,
        )

        self._audit_trail: list[CloudAuditEntry] = []
        self._total_blocked = 0
        self._total_allowed = 0
        self._total_sanitized_chars = 0

        # Reference to SecurityAuditTrail (wired later)
        self._security_audit: Any = None

        logger.info(
            "SecurityGateway: max_outbound=%d, max_rpm=%d, audit=%s",
            self._max_outbound_length, self._max_rpm, self._audit_cloud_calls,
        )

    def attach_audit_trail(self, audit: Any) -> None:
        """Wire the SecurityAuditTrail for persistent logging."""
        self._security_audit = audit

    # ── Core API ─────────────────────────────────────────────────────

    def sanitize_outbound(self, query: str) -> str:
        """Remove all sensitive data from an outbound query.

        This is the LAST line of defense before data leaves the system.
        Applies multiple sanitization passes in order of severity.
        """
        if not query:
            return ""

        sanitized = query

        # Pass 1: File system paths
        sanitized = _PATH_PATTERN.sub("[PATH_REDACTED]", sanitized)

        # Pass 2: IP addresses
        sanitized = _IP_PATTERN.sub("[IP_REDACTED]", sanitized)

        # Pass 3: Email addresses
        sanitized = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", sanitized)

        # Pass 4: Tokens, keys, secrets
        sanitized = _TOKEN_PATTERN.sub("[CREDENTIAL_REDACTED]", sanitized)

        # Pass 5: Environment variables
        sanitized = _ENV_VAR_PATTERN.sub("[ENV_REDACTED]", sanitized)

        # Pass 6: Config file references
        sanitized = _CONFIG_REF_PATTERN.sub("[CONFIG_REDACTED]", sanitized)

        # Pass 7: Blocked literal words
        for blocked in _BLOCKED_LITERALS:
            if blocked.lower() in sanitized.lower():
                sanitized = re.sub(
                    re.escape(blocked), "[REDACTED]", sanitized, flags=re.I,
                )

        # Pass 8: Truncate to max length
        if len(sanitized) > self._max_outbound_length:
            sanitized = sanitized[:self._max_outbound_length] + "..."

        chars_removed = len(query) - len(sanitized)
        if chars_removed > 0:
            self._total_sanitized_chars += abs(chars_removed)
            logger.debug(
                "SecurityGateway: sanitized query (delta=%+d chars)",
                -chars_removed,
            )

        return sanitized.strip()

    def allow_cloud(
        self,
        query: str,
        intent: str = "",
    ) -> tuple[bool, str]:
        """Decide whether a query is safe to send to cloud.

        Returns (allowed, reason).
        """
        if not query or not query.strip():
            return False, "empty_query"

        # Rule 1: System commands are NEVER sent to cloud
        if _SYSTEM_CMD_PATTERN.search(query):
            self._record_block("system_command")
            return False, "system_command_detected"

        # Rule 2: Direct config/security references blocked
        if _CONFIG_REF_PATTERN.search(query):
            self._record_block("config_reference")
            return False, "config_reference_detected"

        # Rule 3: Intents that imply local-only execution
        local_only_intents = frozenset({
            "shutdown", "restart", "kill", "open_app", "close_app",
            "file_op", "system_control", "lock", "sleep",
            "screenshot", "clipboard", "terminal",
        })
        if intent.lower() in local_only_intents:
            self._record_block("local_only_intent")
            return False, f"local_only_intent:{intent}"

        # Rule 4: Query length check
        if len(query) > self._max_outbound_length * 2:
            self._record_block("too_long")
            return False, "query_too_long"

        # Rule 5: Rate limiting
        if not self._rate_limiter.try_consume():
            self._record_block("rate_limited")
            return False, "rate_limited"

        # Rule 6: Heavy path/token content (even after future sanitization)
        path_count = len(_PATH_PATTERN.findall(query))
        token_count = len(_TOKEN_PATTERN.findall(query))
        if path_count > 3 or token_count > 0:
            self._record_block("sensitive_data_density")
            return False, "sensitive_data_density"

        self._total_allowed += 1
        return True, "allowed"

    def tag_cloud_response(
        self,
        response: str,
        provider: str = "gemini",
        latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Tag a cloud response as untrusted.

        The tagged response MUST NOT be used for:
          - Tool execution decisions
          - System command generation
          - Memory storage without local validation
        """
        return {
            "text": response,
            "source": "cloud_untrusted",
            "provider": provider,
            "trust_level": "advisory",
            "latency_ms": latency_ms,
            "can_execute_tools": False,
            "can_access_system": False,
            "can_store_to_memory": False,
            "timestamp": time.time(),
        }

    def rate_check(self) -> bool:
        """Check if a cloud request is allowed without consuming a token."""
        return self._rate_limiter.tokens >= 1.0

    # ── Audit ────────────────────────────────────────────────────────

    def _record_block(self, reason: str) -> None:
        self._total_blocked += 1
        logger.info("SecurityGateway BLOCKED: %s (total=%d)", reason, self._total_blocked)

        if self._audit_cloud_calls and self._security_audit:
            try:
                self._security_audit.log(
                    event_type="cloud_blocked",
                    details=reason,
                    severity="INFO",
                    source="security_gateway",
                )
            except Exception:
                logger.debug("Audit log failed", exc_info=True)

    def record_cloud_call(
        self,
        query_hash: str,
        sanitized_length: int,
        response_length: int,
        latency_ms: float,
        provider: str = "gemini",
    ) -> None:
        """Record a completed cloud call for audit."""
        entry = CloudAuditEntry(
            timestamp=time.time(),
            query_hash=query_hash,
            sanitized_length=sanitized_length,
            allowed=True,
            provider=provider,
            response_length=response_length,
            latency_ms=latency_ms,
        )
        self._audit_trail.append(entry)

        # Keep audit trail bounded
        if len(self._audit_trail) > 1000:
            self._audit_trail = self._audit_trail[-500:]

        if self._audit_cloud_calls and self._security_audit:
            try:
                self._security_audit.log(
                    event_type="cloud_call",
                    details=(
                        f"provider={provider} "
                        f"query_len={sanitized_length} "
                        f"response_len={response_length} "
                        f"latency={latency_ms:.0f}ms"
                    ),
                    severity="INFO",
                    source="security_gateway",
                )
            except Exception:
                logger.debug("Audit log failed", exc_info=True)

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        """Return gateway diagnostics for health monitoring."""
        return {
            "total_allowed": self._total_allowed,
            "total_blocked": self._total_blocked,
            "total_sanitized_chars": self._total_sanitized_chars,
            "rate_tokens_available": round(self._rate_limiter.tokens, 1),
            "audit_trail_size": len(self._audit_trail),
            "recent_calls": [
                {
                    "provider": e.provider,
                    "latency_ms": round(e.latency_ms, 1),
                    "sanitized_length": e.sanitized_length,
                }
                for e in self._audit_trail[-5:]
            ],
        }


__all__ = ["SecurityGateway"]
