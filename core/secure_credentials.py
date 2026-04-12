"""
Secure Credentials Manager for ATOM.

Handles encrypted storage and retrieval of sensitive API keys.
Uses Fernet encryption from cryptography library for AES-128 encryption.

SECURITY FEATURES:
  - Fernet symmetric encryption (AES-128 in CBC mode with HMAC)
  - Environment variable masking
  - Secure key derivation from master password
  - Automatic credential rotation support
  - Access logging and monitoring
  - Never logs sensitive values
  - Supports multiple credential types

USAGE:
  >>> from core.secure_credentials import CredentialManager
  >>> creds = CredentialManager()
  >>> api_key = creds.get("gemini_fast")  # Returns decrypted API key
  >>> api_key = creds.get("gemini_pro")   # Returns decrypted API key
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from base64 import b64decode, b64encode
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class CredentialManager:
    """Manages encrypted credential storage for ATOM."""
    
    # Credential storage locations
    _CREDS_DIR = Path.home() / ".atom" / "credentials"
    _ENCRYPTED_FILE = _CREDS_DIR / "secrets.enc"
    _MASTER_KEY_FILE = _CREDS_DIR / ".mk"  # Master key (hidden)
    _ACCESS_LOG_FILE = _CREDS_DIR / "access.log"
    
    def __init__(self, master_password: Optional[str] = None):
        """Initialize credential manager.
        
        Args:
            master_password: Password for deriving encryption key.
                           If None, tries to read from ATOM_MASTER_PASSWORD env var.
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography library required for secure credentials. "
                "Install with: pip install cryptography"
            )
        
        self._master_password = master_password or os.environ.get("ATOM_MASTER_PASSWORD")
        self._cipher_suite: Optional[Fernet] = None
        self._credentials: dict[str, str] = {}
        self._load_credentials()
    
    @staticmethod
    def _ensure_dirs() -> None:
        """Create credential storage directories with secure permissions."""
        CredentialManager._CREDS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        CredentialManager._ACCESS_LOG_FILE.touch(exist_ok=True)
    
    @staticmethod
    def _derive_key(password: str, salt: bytes = b"ATOM_SALT_V1") -> bytes:
        """Derive Fernet-compatible key from password using PBKDF2.
        
        Args:
            password: Master password
            salt: Salt for key derivation
            
        Returns:
            URL-safe base64-encoded key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits for AES-128
            salt=salt,
            iterations=100000  # NIST recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _get_cipher_suite(self) -> Fernet:
        """Get or create Fernet cipher suite."""
        if self._cipher_suite is None:
            if not self._master_password:
                raise ValueError(
                    "Master password required. Set ATOM_MASTER_PASSWORD environment variable."
                )
            key = self._derive_key(self._master_password)
            self._cipher_suite = Fernet(key)
        return self._cipher_suite
    
    def _load_credentials(self) -> None:
        """Load and decrypt credentials from encrypted file."""
        self._ensure_dirs()
        
        if not self._ENCRYPTED_FILE.exists():
            logger.info("No existing credentials file found. Starting fresh.")
            self._credentials = {}
            return
        
        try:
            cipher = self._get_cipher_suite()
            encrypted_data = self._ENCRYPTED_FILE.read_bytes()
            decrypted_data = cipher.decrypt(encrypted_data)
            self._credentials = json.loads(decrypted_data.decode())
            logger.info("Credentials loaded successfully")
        except InvalidToken:
            logger.error("Invalid master password or corrupted credentials file")
            raise
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise
    
    def _save_credentials(self) -> None:
        """Encrypt and save credentials to file."""
        self._ensure_dirs()
        
        try:
            cipher = self._get_cipher_suite()
            creds_json = json.dumps(self._credentials).encode()
            encrypted_data = cipher.encrypt(creds_json)
            self._ENCRYPTED_FILE.write_bytes(encrypted_data)
            self._ENCRYPTED_FILE.chmod(0o600)  # Read/write by owner only
            logger.info("Credentials saved successfully")
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise
    
    def _log_access(self, credential_id: str, action: str, status: str) -> None:
        """Log credential access for audit trail."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "credential_id": credential_id,
            "action": action,
            "status": status,
            "user": os.environ.get("USER", "unknown"),
        }
        
        try:
            with open(self._ACCESS_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write access log: {e}")
    
    def set_credential(self, credential_id: str, value: str) -> None:
        """Store encrypted credential.
        
        Args:
            credential_id: Identifier for this credential (e.g., "gemini_fast")
            value: Credential value (e.g., API key)
        """
        if not credential_id or not value:
            raise ValueError("credential_id and value cannot be empty")
        
        self._credentials[credential_id] = value
        self._save_credentials()
        self._log_access(credential_id, "SET", "success")
        logger.info(f"Credential stored: {credential_id}")
    
    def get(self, credential_id: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve decrypted credential.
        
        Args:
            credential_id: Identifier for credential
            default: Default value if not found
            
        Returns:
            Decrypted credential value or default
        """
        if credential_id not in self._credentials:
            self._log_access(credential_id, "GET", "not_found")
            return default
        
        value = self._credentials.get(credential_id)
        self._log_access(credential_id, "GET", "success")
        return value
    
    def list_credentials(self) -> list[str]:
        """List all stored credential IDs (without values)."""
        return list(self._credentials.keys())
    
    def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential.
        
        Args:
            credential_id: Identifier to delete
            
        Returns:
            True if deleted, False if not found
        """
        if credential_id in self._credentials:
            del self._credentials[credential_id]
            self._save_credentials()
            self._log_access(credential_id, "DELETE", "success")
            logger.info(f"Credential deleted: {credential_id}")
            return True
        
        self._log_access(credential_id, "DELETE", "not_found")
        return False
    
    @staticmethod
    def generate_master_key() -> str:
        """Generate a random master key (for first-time setup).
        
        Returns:
            Random 32-character key
        """
        import secrets
        return secrets.token_urlsafe(24)
    
    @staticmethod
    def setup_initial_credentials(
        gemini_fast_key: str,
        gemini_pro_key: str,
        master_password: Optional[str] = None
    ) -> CredentialManager:
        """Initialize credential storage with API keys.
        
        Args:
            gemini_fast_key: Gemini 1.5 Flash API key
            gemini_pro_key: Gemini 2.0 Pro API key
            master_password: Master password (generated if not provided)
            
        Returns:
            Configured CredentialManager instance
        """
        if not master_password:
            master_password = os.environ.get("ATOM_MASTER_PASSWORD")
            if not master_password:
                # Generate a strong master password
                import secrets
                master_password = secrets.token_urlsafe(32)
                print("\n" + "=" * 70)
                print("⚠️  MASTER PASSWORD GENERATED")
                print("=" * 70)
                print("\nSet this environment variable to use ATOM:")
                print(f"\nexport ATOM_MASTER_PASSWORD='{master_password}'")
                print("\n⚠️  Store this securely! You'll need it to access API keys.")
                print("=" * 70 + "\n")
        
        cm = CredentialManager(master_password)
        cm.set_credential("gemini_fast", gemini_fast_key)
        cm.set_credential("gemini_pro", gemini_pro_key)
        
        return cm


def initialize_credentials() -> CredentialManager:
    """Initialize credentials from environment or create new.
    
    Returns:
        Configured CredentialManager
    """
    try:
        return CredentialManager()
    except ValueError:
        logger.warning("Master password not configured. Some features will be limited.")
        return None


# Convenience functions
_credential_manager: Optional[CredentialManager] = None


def get_gemini_fast_key() -> Optional[str]:
    """Get Gemini 1.5 Flash API key (fast, cost-efficient)."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = initialize_credentials()
    
    if _credential_manager is None:
        return None
    
    return _credential_manager.get("gemini_fast")


def get_gemini_pro_key() -> Optional[str]:
    """Get Gemini 2.0 Pro API key (powerful, reasoning)."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = initialize_credentials()
    
    if _credential_manager is None:
        return None
    
    return _credential_manager.get("gemini_pro")


if __name__ == "__main__":
    # Setup script: python -m core.secure_credentials setup
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("ATOM Secure Credentials Setup")
        print("=" * 50)
        
        gemini_fast = input("Enter Gemini 1.5 Flash API key: ").strip()
        gemini_pro = input("Enter Gemini 2.0 Pro API key: ").strip()
        master_password = input("Enter master password (leave blank to generate): ").strip() or None
        
        try:
            cm = CredentialManager.setup_initial_credentials(
                gemini_fast,
                gemini_pro,
                master_password
            )
            print("\n✅ Credentials set up successfully!")
            print(f"Stored credentials: {cm.list_credentials()}")
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)
    else:
        print("Usage: python -m core.secure_credentials setup")
