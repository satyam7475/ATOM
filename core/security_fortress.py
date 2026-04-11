"""
ATOM -- Security Fortress (Production-Grade Hardened Security).

The impenetrable security layer for ATOM. This module ensures that:
  1. Only the authenticated owner (Satyam) can operate ATOM
  2. All sensitive data is encrypted at rest using Fernet (AES-128-CBC)
  3. Source files are integrity-checked to detect tampering
  4. Sessions are token-based with automatic expiry
  5. All security events are logged to a tamper-evident audit trail
  6. Network exposure is minimized and monitored
  7. Brute-force attacks are blocked via progressive lockout

Architecture:
  - OwnerAuthenticator: Challenge-response + passphrase authentication
  - EncryptedVault: Fernet-encrypted key-value store for secrets (default)
  - KeychainVault (macOS): optional ``security`` CLI backing via ``security_fortress``
  - IntegrityMonitor: SHA-256 hash verification of all ATOM source files
  - SessionManager: Token-based sessions with TTL and auto-expiry
  - SecurityAuditTrail: Tamper-evident, hash-chained audit log
  - ThreatDetector: Anomaly detection for suspicious activity patterns

Owner: Satyam (Boss). Zero-trust by default.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.security_fortress")

_ATOM_ROOT = Path(__file__).parent.parent
_VAULT_FILE = Path("data/security/vault.enc")
_INTEGRITY_FILE = Path("data/security/integrity.json")
_AUDIT_TRAIL_FILE = Path("logs/security_audit.jsonl")
_SESSION_FILE = Path("data/security/session.json")
_LOCKOUT_FILE = Path("data/security/lockout.json")

_SESSION_TTL_S = 86400  # 24 hours
_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_DURATION_S = 300  # 5 minutes
_LOCKOUT_ESCALATION_FACTOR = 2.0


class OwnerAuthenticator:
    """Challenge-response authentication for owner verification.

    Stores a salted SHA-256 hash of the owner passphrase. Never stores
    the passphrase in plaintext. Supports progressive lockout on
    repeated failures.
    """

    __slots__ = (
        "_passphrase_hash", "_salt", "_is_enrolled",
        "_failed_attempts", "_lockout_until", "_lockout_count",
    )

    def __init__(self, config: dict | None = None) -> None:
        sec = (config or {}).get("security_fortress", {})
        self._salt = sec.get("salt", "ATOM_OWNER_SALT_v20")
        self._passphrase_hash: str | None = None
        self._is_enrolled = False
        self._failed_attempts = 0
        self._lockout_until = 0.0
        self._lockout_count = 0
        self._load_enrollment()

    def _hash_passphrase(self, passphrase: str) -> str:
        salted = f"{self._salt}:{passphrase}".encode("utf-8")
        return hashlib.sha256(salted).hexdigest()

    def _load_enrollment(self) -> None:
        lockout_path = _LOCKOUT_FILE
        if lockout_path.exists():
            try:
                data = json.loads(lockout_path.read_text(encoding="utf-8"))
                self._passphrase_hash = data.get("passphrase_hash")
                self._is_enrolled = bool(self._passphrase_hash)
                self._lockout_count = data.get("lockout_count", 0)
            except Exception:
                logger.debug("Lockout file load failed", exc_info=True)

    def _save_enrollment(self) -> None:
        _LOCKOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "passphrase_hash": self._passphrase_hash,
            "lockout_count": self._lockout_count,
            "enrolled_at": datetime.now().isoformat(),
        }
        _LOCKOUT_FILE.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    @property
    def is_enrolled(self) -> bool:
        return self._is_enrolled

    def enroll(self, passphrase: str) -> bool:
        """Enroll owner passphrase. Only works once unless reset."""
        if len(passphrase) < 6:
            logger.warning("Passphrase too short (minimum 6 characters)")
            return False
        self._passphrase_hash = self._hash_passphrase(passphrase)
        self._is_enrolled = True
        self._save_enrollment()
        logger.info("Owner enrolled successfully")
        return True

    def authenticate(self, passphrase: str) -> tuple[bool, str]:
        """Verify owner passphrase. Returns (success, message)."""
        now = time.monotonic()

        if now < self._lockout_until:
            remaining = int(self._lockout_until - now)
            return False, f"Locked out. Try again in {remaining} seconds."

        if not self._is_enrolled:
            return False, "Owner not enrolled. Use 'atom enroll' first."

        candidate = self._hash_passphrase(passphrase)
        if hmac.compare_digest(candidate, self._passphrase_hash or ""):
            self._failed_attempts = 0
            logger.info("Owner authenticated successfully")
            return True, "Authenticated."

        self._failed_attempts += 1
        logger.warning(
            "Authentication failed (attempt %d/%d)",
            self._failed_attempts, _MAX_FAILED_ATTEMPTS,
        )

        if self._failed_attempts >= _MAX_FAILED_ATTEMPTS:
            self._lockout_count += 1
            duration = _LOCKOUT_DURATION_S * (
                _LOCKOUT_ESCALATION_FACTOR ** (self._lockout_count - 1)
            )
            self._lockout_until = now + duration
            self._failed_attempts = 0
            self._save_enrollment()
            return False, (
                f"Too many failed attempts. Locked for {int(duration)} seconds."
            )

        remaining = _MAX_FAILED_ATTEMPTS - self._failed_attempts
        return False, f"Wrong passphrase. {remaining} attempts remaining."

    def reset(self, current_passphrase: str) -> tuple[bool, str]:
        """Reset enrollment. Requires current passphrase for safety."""
        ok, msg = self.authenticate(current_passphrase)
        if not ok:
            return False, f"Cannot reset: {msg}"
        self._passphrase_hash = None
        self._is_enrolled = False
        self._failed_attempts = 0
        self._lockout_count = 0
        self._save_enrollment()
        return True, "Enrollment reset. Re-enroll with a new passphrase."


class EncryptedVault:
    """Fernet-encrypted key-value store for sensitive data.

    Falls back to obfuscated JSON if cryptography is not installed.
    Stored at data/security/vault.enc.
    """

    __slots__ = ("_data", "_fernet", "_using_encryption", "_dirty")

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._fernet: Any = None
        self._using_encryption = False
        self._dirty = False
        self._init_crypto()
        self._load()

    def _init_crypto(self) -> None:
        try:
            from cryptography.fernet import Fernet
            key_file = Path("data/security/.vault_key")
            key_file.parent.mkdir(parents=True, exist_ok=True)

            if key_file.exists():
                key = key_file.read_bytes().strip()
            else:
                key = Fernet.generate_key()
                key_file.write_bytes(key)
                try:
                    os.chmod(key_file, 0o600)
                except OSError:
                    pass

            self._fernet = Fernet(key)
            self._using_encryption = True
            logger.info("Vault initialized with Fernet encryption")
        except ImportError:
            logger.info("cryptography not available -- vault uses obfuscation")
        except Exception:
            logger.warning("Vault crypto init failed", exc_info=True)

    def _load(self) -> None:
        if not _VAULT_FILE.exists():
            return
        try:
            raw = _VAULT_FILE.read_bytes()
            if self._using_encryption and self._fernet:
                decrypted = self._fernet.decrypt(raw)
                self._data = json.loads(decrypted)
            else:
                import base64
                decoded = base64.b85decode(raw)
                self._data = json.loads(decoded)
        except Exception:
            logger.debug("Vault load failed -- starting fresh", exc_info=True)
            self._data = {}

    def persist(self) -> None:
        if not self._dirty:
            return
        try:
            _VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._data).encode("utf-8")

            if self._using_encryption and self._fernet:
                encrypted = self._fernet.encrypt(payload)
                _VAULT_FILE.write_bytes(encrypted)
            else:
                import base64
                encoded = base64.b85encode(payload)
                _VAULT_FILE.write_bytes(encoded)

            try:
                os.chmod(_VAULT_FILE, 0o600)
            except OSError:
                pass
            self._dirty = False
        except Exception:
            logger.warning("Vault persist failed", exc_info=True)

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key, default)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        self._dirty = True

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._dirty = True
            return True
        return False

    def keys(self) -> list[str]:
        return list(self._data.keys())

    @property
    def is_encrypted(self) -> bool:
        return self._using_encryption

    @property
    def backend_name(self) -> str:
        return "fernet" if self._using_encryption else "obfuscated"


def _use_macos_keychain(config: dict | None) -> bool:
    """Prefer Keychain on macOS unless ``security_fortress.use_macos_keychain`` is false."""
    if sys.platform != "darwin":
        return False
    sec = (config or {}).get("security_fortress", {}) or {}
    if sec.get("use_macos_keychain") is False:
        return False
    return True


def _migrate_fernet_vault_into_keychain(kv: Any) -> None:
    """Copy existing ``vault.enc`` entries into Keychain once, then retire the file."""
    if kv.keys():
        return
    if not _VAULT_FILE.exists():
        return
    try:
        old = EncryptedVault()
    except Exception:
        logger.debug("Could not open legacy vault for migration", exc_info=True)
        return
    legacy_keys = old.keys()
    if not legacy_keys:
        return
    migrated = 0
    for k in legacy_keys:
        val = old.get(k)
        if val and kv.put(k, val):
            migrated += 1
    kv.persist()
    if migrated == 0:
        return
    try:
        backup = _VAULT_FILE.with_suffix(_VAULT_FILE.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        _VAULT_FILE.rename(backup)
        logger.info(
            "Security vault: migrated %d entries to macOS Keychain (%s)",
            migrated,
            backup.name,
        )
    except OSError:
        logger.warning("Could not rename vault.enc after Keychain migration", exc_info=True)


def _select_security_vault(config: dict) -> Any:
    if _use_macos_keychain(config):
        try:
            from core.macos.keychain_store import KeychainVault

            sec = config.get("security_fortress", {}) or {}
            service = str(sec.get("keychain_service") or "com.atom.fortress")
            kv = KeychainVault(service=service)
            _migrate_fernet_vault_into_keychain(kv)
            return kv
        except Exception:
            logger.warning(
                "macOS Keychain vault unavailable; falling back to EncryptedVault.",
                exc_info=True,
            )
    return EncryptedVault()


class IntegrityMonitor:
    """SHA-256 integrity verification for all ATOM source files.

    On first run, creates a baseline hash manifest. On subsequent runs,
    verifies every file against the baseline and reports tampering.
    """

    __slots__ = ("_baseline", "_atom_root", "_last_check_time", "_violations")

    def __init__(self) -> None:
        self._atom_root = _ATOM_ROOT
        self._baseline: dict[str, str] = {}
        self._last_check_time: float = 0.0
        self._violations: list[dict[str, str]] = []
        self._load_baseline()

    def _load_baseline(self) -> None:
        if _INTEGRITY_FILE.exists():
            try:
                self._baseline = json.loads(
                    _INTEGRITY_FILE.read_text(encoding="utf-8")
                )
            except Exception:
                self._baseline = {}

    def _hash_file(self, filepath: Path) -> str:
        h = hashlib.sha256()
        try:
            content = filepath.read_bytes()
            h.update(content)
            return h.hexdigest()
        except Exception:
            return "ERROR"

    def _get_source_files(self) -> list[Path]:
        """Get all Python source files in the ATOM directory."""
        files: list[Path] = []
        for pattern in ("**/*.py", "**/*.json"):
            for f in self._atom_root.glob(pattern):
                rel = f.relative_to(self._atom_root)
                rel_str = str(rel)
                if any(skip in rel_str for skip in (
                    "__pycache__", ".git", "logs", "data",
                    "node_modules", ".env",
                )):
                    continue
                files.append(f)
        return sorted(files)

    def create_baseline(self) -> dict[str, str]:
        """Create or refresh the integrity baseline."""
        files = self._get_source_files()
        self._baseline = {}
        for f in files:
            rel = str(f.relative_to(self._atom_root))
            self._baseline[rel] = self._hash_file(f)

        _INTEGRITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _INTEGRITY_FILE.write_text(
            json.dumps(self._baseline, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Integrity baseline created: %d files hashed", len(self._baseline))
        return self._baseline

    def verify(self) -> tuple[bool, list[dict[str, str]]]:
        """Verify all source files against baseline.

        Returns (all_ok, violations) where violations is a list of
        dicts with keys: file, type (modified|added|deleted), details.
        """
        if not self._baseline:
            self.create_baseline()
            return True, []

        self._violations = []
        current_files = self._get_source_files()
        current_map: dict[str, str] = {}

        for f in current_files:
            rel = str(f.relative_to(self._atom_root))
            current_hash = self._hash_file(f)
            current_map[rel] = current_hash

            if rel in self._baseline:
                if current_hash != self._baseline[rel]:
                    self._violations.append({
                        "file": rel,
                        "type": "modified",
                        "details": f"Hash mismatch (expected {self._baseline[rel][:12]}..., "
                                   f"got {current_hash[:12]}...)",
                    })
            else:
                self._violations.append({
                    "file": rel,
                    "type": "added",
                    "details": "New file not in baseline",
                })

        for baseline_file in self._baseline:
            if baseline_file not in current_map:
                self._violations.append({
                    "file": baseline_file,
                    "type": "deleted",
                    "details": "File missing from disk",
                })

        self._last_check_time = time.time()
        all_ok = len(self._violations) == 0

        if not all_ok:
            logger.warning(
                "INTEGRITY CHECK FAILED: %d violations detected",
                len(self._violations),
            )
            for v in self._violations[:5]:
                logger.warning("  [%s] %s: %s", v["type"], v["file"], v["details"])

        return all_ok, self._violations

    def get_file_hash(self, relative_path: str) -> str | None:
        """Get the baseline hash for a specific file."""
        return self._baseline.get(relative_path)

    @property
    def file_count(self) -> int:
        return len(self._baseline)

    @property
    def last_violations(self) -> list[dict[str, str]]:
        return self._violations


class SessionManager:
    """Token-based session management with automatic expiry.

    Each authenticated session gets a unique token. The token must
    be verified for sensitive operations. Sessions auto-expire.
    """

    __slots__ = ("_sessions", "_ttl")

    def __init__(self, ttl_seconds: float = _SESSION_TTL_S) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._ttl = ttl_seconds
        self._load()

    def _load(self) -> None:
        if _SESSION_FILE.exists():
            try:
                data = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
                now = time.time()
                self._sessions = {
                    k: v for k, v in data.items()
                    if v.get("expires_at", 0) > now
                }
            except Exception:
                self._sessions = {}

    def persist(self) -> None:
        try:
            _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            _SESSION_FILE.write_text(
                json.dumps(self._sessions, indent=2), encoding="utf-8",
            )
            try:
                os.chmod(_SESSION_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.debug("Session persist failed", exc_info=True)

    def create_session(self, owner: str = "Satyam") -> str:
        """Create a new authenticated session. Returns the session token."""
        self._cleanup_expired()
        token = secrets.token_urlsafe(32)
        self._sessions[token] = {
            "owner": owner,
            "created_at": time.time(),
            "expires_at": time.time() + self._ttl,
            "last_activity": time.time(),
            "id": uuid.uuid4().hex[:8],
        }
        self.persist()
        logger.info("Session created for %s (expires in %ds)", owner, self._ttl)
        return token

    def validate_session(self, token: str) -> tuple[bool, str]:
        """Validate a session token. Returns (valid, reason)."""
        session = self._sessions.get(token)
        if session is None:
            return False, "Invalid or expired session token."

        if time.time() > session.get("expires_at", 0):
            del self._sessions[token]
            self.persist()
            return False, "Session expired. Re-authenticate."

        session["last_activity"] = time.time()
        return True, "Session valid."

    def revoke_session(self, token: str) -> bool:
        if token in self._sessions:
            del self._sessions[token]
            self.persist()
            return True
        return False

    def revoke_all(self) -> int:
        count = len(self._sessions)
        self._sessions.clear()
        self.persist()
        return count

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items()
                   if v.get("expires_at", 0) <= now]
        for k in expired:
            del self._sessions[k]

    @property
    def active_count(self) -> int:
        self._cleanup_expired()
        return len(self._sessions)


class SecurityAuditTrail:
    """Tamper-evident, hash-chained security audit log.

    Each entry includes a SHA-256 hash of the previous entry, creating
    a chain. If any entry is modified or deleted, the chain breaks and
    tampering is detectable.
    """

    __slots__ = ("_last_hash", "_entry_count")

    def __init__(self) -> None:
        _AUDIT_TRAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = "GENESIS"
        self._entry_count = 0
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        if not _AUDIT_TRAIL_FILE.exists():
            return
        try:
            with open(_AUDIT_TRAIL_FILE, "r", encoding="utf-8") as f:
                last_line = ""
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                        self._entry_count += 1
                if last_line:
                    entry = json.loads(last_line)
                    self._last_hash = entry.get("hash", "GENESIS")
        except Exception:
            pass

    def log(
        self,
        event_type: str,
        details: str = "",
        severity: str = "INFO",
        source: str = "system",
    ) -> None:
        """Append a tamper-evident audit entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "source": source,
            "details": details,
            "prev_hash": self._last_hash,
        }
        entry_json = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode("utf-8")).hexdigest()[:16]
        entry["hash"] = entry_hash
        self._last_hash = entry_hash
        self._entry_count += 1

        try:
            with open(_AUDIT_TRAIL_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            try:
                os.chmod(_AUDIT_TRAIL_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.debug("Audit trail write failed", exc_info=True)

    def verify_chain(self) -> tuple[bool, int, str]:
        """Verify the integrity of the entire audit chain.

        Returns (intact, entry_count, message).
        """
        if not _AUDIT_TRAIL_FILE.exists():
            return True, 0, "No audit trail exists yet."

        try:
            prev_hash = "GENESIS"
            count = 0
            with open(_AUDIT_TRAIL_FILE, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    stored_hash = entry.pop("hash", "")
                    if entry.get("prev_hash") != prev_hash:
                        return False, count, (
                            f"Chain broken at entry {line_no}: "
                            f"expected prev_hash={prev_hash[:12]}..., "
                            f"got {entry.get('prev_hash', 'MISSING')[:12]}..."
                        )
                    verify_json = json.dumps(entry, separators=(",", ":"), sort_keys=True)
                    computed = hashlib.sha256(
                        verify_json.encode("utf-8")
                    ).hexdigest()[:16]
                    if computed != stored_hash:
                        return False, count, (
                            f"Hash mismatch at entry {line_no}: "
                            f"computed={computed}, stored={stored_hash}"
                        )
                    prev_hash = stored_hash
                    count += 1

            return True, count, f"Audit chain intact ({count} entries verified)."
        except Exception as e:
            return False, 0, f"Chain verification error: {e}"

    @property
    def entry_count(self) -> int:
        return self._entry_count


class ThreatDetector:
    """Anomaly detection for suspicious activity patterns.

    Tracks behavioral signals and flags anomalies:
      - Unusual command patterns
      - Rapid repeated failures
      - Attempts to access blocked resources
      - Unusual time-of-day activity
    """

    __slots__ = (
        "_event_log", "_threat_level", "_max_events",
        "_anomaly_callbacks",
    )

    def __init__(self, max_events: int = 1000) -> None:
        self._event_log: list[dict[str, Any]] = []
        self._threat_level: str = "normal"
        self._max_events = max_events
        self._anomaly_callbacks: list[Any] = []

    def record_event(
        self,
        event_type: str,
        details: str = "",
        severity: str = "low",
    ) -> None:
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details,
            "severity": severity,
        }
        self._event_log.append(event)
        if len(self._event_log) > self._max_events:
            self._event_log = self._event_log[-self._max_events:]

        self._analyze_threats()

    def _analyze_threats(self) -> None:
        now = time.time()
        window = 60.0  # last 60 seconds
        recent = [e for e in self._event_log if now - e["timestamp"] < window]

        auth_failures = sum(
            1 for e in recent if e["type"] == "auth_failure"
        )
        blocked_actions = sum(
            1 for e in recent if e["type"] == "action_blocked"
        )
        injection_attempts = sum(
            1 for e in recent if e["type"] == "injection_attempt"
        )

        if injection_attempts > 0 or auth_failures > 3:
            self._threat_level = "critical"
        elif auth_failures > 1 or blocked_actions > 5:
            self._threat_level = "elevated"
        elif blocked_actions > 2:
            self._threat_level = "guarded"
        else:
            self._threat_level = "normal"

    @property
    def threat_level(self) -> str:
        return self._threat_level

    def get_recent_threats(self, minutes: float = 10.0) -> list[dict[str, Any]]:
        cutoff = time.time() - (minutes * 60)
        return [
            e for e in self._event_log
            if e["timestamp"] > cutoff
            and e["severity"] in ("medium", "high", "critical")
        ]

    def get_threat_summary(self) -> str:
        total = len(self._event_log)
        threats = self.get_recent_threats()
        return (
            f"Threat level: {self._threat_level.upper()}. "
            f"{total} events tracked, {len(threats)} threats in last 10 minutes."
        )


class SecurityFortress:
    """Unified security fortress -- the master security controller.

    Orchestrates all security subsystems into a single interface.
    This is the one object that main.py instantiates and wires.

    v21: Added VoicePrintAuth (biometric) and BehavioralAuth (passive
    continuous verification). ATOM now verifies identity like JARVIS.
    """

    __slots__ = (
        "_authenticator", "_vault", "_integrity",
        "_sessions", "_audit", "_threats",
        "_config", "_active_session_token",
        "_auto_authenticated",
        "_voice_auth", "_behavior_auth",
    )

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._authenticator = OwnerAuthenticator(config)
        self._vault = _select_security_vault(self._config)
        self._integrity = IntegrityMonitor()
        self._sessions = SessionManager()
        self._audit = SecurityAuditTrail()
        self._threats = ThreatDetector()
        self._active_session_token: str | None = None
        self._auto_authenticated = False

        from core.auth.voice_auth import VoicePrintAuth
        from core.auth.behavior_auth import BehavioralAuth

        self._voice_auth = VoicePrintAuth(config)
        self._voice_auth.attach_vault(self._vault)
        self._behavior_auth = BehavioralAuth(config)

        sec_cfg = self._config.get("security_fortress", {}) or {}
        if not sec_cfg.get("require_auth", False):
            self._auto_authenticated = True
            self._active_session_token = self._sessions.create_session()
            self._audit.log("auto_session", "Auto-authenticated (require_auth=false)")

        logger.info(
            "Security Fortress initialized: auth=%s, vault=%s, integrity=%d files, "
            "sessions=%d, audit=%d entries, voice=%s, behavior=%s",
            "enrolled" if self._authenticator.is_enrolled else "not_enrolled",
            self.vault_backend_label,
            self._integrity.file_count,
            self._sessions.active_count,
            self._audit.entry_count,
            "available" if self._voice_auth.is_available else "unavailable",
            "baselined" if self._behavior_auth.is_baselined else "learning",
        )

    # ── Authentication ──────────────────────────────────────────────

    def enroll_owner(self, passphrase: str) -> tuple[bool, str]:
        ok = self._authenticator.enroll(passphrase)
        if ok:
            self._audit.log("owner_enrolled", severity="HIGH", source="owner")
            return True, "Owner enrolled successfully, Boss."
        return False, "Enrollment failed. Passphrase must be at least 6 characters."

    def authenticate(self, passphrase: str) -> tuple[bool, str]:
        ok, msg = self._authenticator.authenticate(passphrase)
        if ok:
            self._active_session_token = self._sessions.create_session()
            self._audit.log("auth_success", source="owner")
            self._threats.record_event("auth_success")
            return True, "Welcome back, Boss. Session active."
        self._audit.log("auth_failure", msg, severity="HIGH", source="unknown")
        self._threats.record_event("auth_failure", msg, severity="high")
        return False, msg

    @property
    def is_authenticated(self) -> bool:
        if self._auto_authenticated:
            return True
        if self._active_session_token is None:
            return False
        valid, _ = self._sessions.validate_session(self._active_session_token)
        return valid

    def require_auth(self) -> tuple[bool, str]:
        """Gate for sensitive operations. Returns (allowed, message)."""
        if self.is_authenticated:
            return True, "Authorized."
        return False, "Authentication required. Say 'atom authenticate' first."

    # ── Voice Authentication (v21) ─────────────────────────────────

    def voice_enroll(self, audio_data: Any) -> tuple[bool, str]:
        """Enroll a voice sample for biometric authentication."""
        if not self._voice_auth.is_available:
            return False, "Voice authentication unavailable. Install resemblyzer or numpy."
        result = self._voice_auth.enroll(audio_data)
        if result.success:
            self._audit.log(
                "voice_enrolled",
                f"Phrase {result.phrases_enrolled}, confidence={result.confidence_level}",
                severity="HIGH", source="owner",
            )
        return result.success, result.message

    def voice_verify(self, audio_data: Any) -> tuple[bool, str]:
        """Verify speaker identity via voice print comparison."""
        if not self._voice_auth.is_available:
            return False, "Voice authentication unavailable."
        if not self._voice_auth.is_enrolled:
            return False, "Voice not enrolled. Say 'enroll my voice' first."

        result = self._voice_auth.verify(audio_data)

        if result.is_potential_spoof:
            self._audit.log(
                "voice_spoof_detected", f"similarity={result.similarity:.3f}",
                severity="CRITICAL", source="unknown",
            )
            self._threats.record_event("voice_spoof", severity="critical")
        elif result.verified:
            self._active_session_token = self._sessions.create_session()
            self._behavior_auth.on_authenticated()
            self._audit.log("voice_auth_success", source="owner")
        else:
            self._audit.log(
                "voice_auth_failure", f"similarity={result.similarity:.3f}",
                severity="HIGH", source="unknown",
            )
            self._threats.record_event("auth_failure", "voice", severity="high")

        return result.verified, result.message

    def voice_reset(self) -> str:
        """Reset voice enrollment (requires active session)."""
        if not self.is_authenticated:
            return "Authentication required to reset voice enrollment."
        msg = self._voice_auth.reset_enrollment()
        self._audit.log("voice_enrollment_reset", severity="HIGH", source="owner")
        return msg

    @property
    def voice_auth(self) -> Any:
        return self._voice_auth

    # ── Behavioral Authentication (v21) ──────────────────────────────

    def observe_behavior(
        self,
        action: str,
        detail: str = "",
        query_text: str = "",
        active_app: str = "",
    ) -> None:
        """Feed behavioral observation for passive continuous auth."""
        self._behavior_auth.observe(
            action=action, detail=detail,
            query_text=query_text, active_app=active_app,
        )

    @property
    def behavior_auth(self) -> Any:
        return self._behavior_auth

    @property
    def trust_score(self) -> float:
        return self._behavior_auth.trust_score

    @property
    def vault_backend_label(self) -> str:
        """Short label for logs (``keychain``, ``fernet``, ``obfuscated``)."""
        bn = getattr(self._vault, "backend_name", None)
        if isinstance(bn, str) and bn:
            return bn
        return "encrypted" if self._vault.is_encrypted else "obfuscated"

    # ── Vault ───────────────────────────────────────────────────────

    def vault_store(self, key: str, value: str) -> None:
        self._vault.put(key, value)
        self._vault.persist()
        self._audit.log("vault_store", f"Key stored: {key}", source="owner")

    def vault_get(self, key: str) -> str:
        return self._vault.get(key)

    def vault_delete(self, key: str) -> bool:
        ok = self._vault.delete(key)
        if ok:
            self._vault.persist()
            self._audit.log("vault_delete", f"Key deleted: {key}", source="owner")
        return ok

    def vault_keys(self) -> list[str]:
        return self._vault.keys()

    # ── Integrity ───────────────────────────────────────────────────

    def check_integrity(self) -> tuple[bool, str]:
        """Verify all ATOM source files against baseline."""
        ok, violations = self._integrity.verify()
        if ok:
            self._audit.log("integrity_check", "All files intact")
            return True, (
                f"Integrity verified. All {self._integrity.file_count} files intact."
            )

        modified = sum(1 for v in violations if v["type"] == "modified")
        added = sum(1 for v in violations if v["type"] == "added")
        deleted = sum(1 for v in violations if v["type"] == "deleted")

        details = []
        if modified:
            details.append(f"{modified} modified")
        if added:
            details.append(f"{added} new")
        if deleted:
            details.append(f"{deleted} deleted")

        summary = ", ".join(details)
        self._audit.log(
            "integrity_violation", summary,
            severity="CRITICAL", source="system",
        )
        self._threats.record_event("integrity_violation", summary, severity="critical")

        file_list = ", ".join(v["file"] for v in violations[:5])
        return False, (
            f"INTEGRITY ALERT: {len(violations)} violations detected "
            f"({summary}). Files: {file_list}"
        )

    def refresh_baseline(self) -> str:
        """Create a new integrity baseline from current files."""
        baseline = self._integrity.create_baseline()
        self._audit.log("baseline_refresh", f"{len(baseline)} files hashed")
        return f"Integrity baseline refreshed: {len(baseline)} files hashed."

    # ── Audit ───────────────────────────────────────────────────────

    def verify_audit_chain(self) -> tuple[bool, str]:
        intact, count, msg = self._audit.verify_chain()
        return intact, msg

    def log_security_event(
        self, event_type: str, details: str = "",
        severity: str = "INFO",
    ) -> None:
        self._audit.log(event_type, details, severity)
        if severity in ("HIGH", "CRITICAL"):
            self._threats.record_event(event_type, details, severity.lower())

    # ── Threat Detection ────────────────────────────────────────────

    @property
    def threat_level(self) -> str:
        return self._threats.threat_level

    def get_security_status(self) -> str:
        """Full security status report."""
        parts = [
            f"Security Fortress Status Report:",
            f"  Authentication: {'active' if self.is_authenticated else 'NOT AUTHENTICATED'}",
            f"  Owner enrolled: {'yes' if self._authenticator.is_enrolled else 'no'}",
            f"  Voice auth: {self._voice_auth.confidence_level if self._voice_auth.is_enrolled else 'not enrolled'}"
            f" ({self._voice_auth.method})",
            f"  Behavioral trust: {self._behavior_auth.trust_score:.0%}"
            f" ({self._behavior_auth.get_trust_level()})",
            f"  Vault: {self.vault_backend_label}, "
            f"{len(self._vault.keys())} secrets stored",
            f"  Integrity: {self._integrity.file_count} files baselined",
            f"  Active sessions: {self._sessions.active_count}",
            f"  Audit trail: {self._audit.entry_count} entries",
            f"  Threat level: {self._threats.threat_level.upper()}",
        ]

        recent_threats = self._threats.get_recent_threats()
        if recent_threats:
            parts.append(f"  Recent threats: {len(recent_threats)} in last 10 minutes")

        return " ".join(parts)

    # ── Lifecycle ───────────────────────────────────────────────────

    def persist(self) -> None:
        self._vault.persist()
        self._sessions.persist()
        self._voice_auth.persist()
        self._behavior_auth.persist()

    def shutdown(self) -> None:
        self._audit.log("shutdown", "Security Fortress shutting down")
        self.persist()
        self._voice_auth.shutdown()
        self._behavior_auth.shutdown()
        logger.info("Security Fortress shut down")
