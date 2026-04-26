"""
ATOM Secrets Manager - Central Credential Access Point

Use this module to access all stored credentials securely.

USAGE:
    from core.secrets_manager import get_api_key, list_available_keys
    
    # Get Gemini API keys
    fast_key = get_api_key("gemini_fast")
    pro_key = get_api_key("gemini_pro")
    
    # List all available keys
    available = list_available_keys()
"""

import logging
from typing import Optional
from core.secure_credentials import CredentialManager

logger = logging.getLogger(__name__)

# Credential identifiers
GEMINI_FAST = "gemini_fast"
GEMINI_PRO = "gemini_pro"

# Sprint Ω.9 (Apr 26 2026): rotating OpenAI-compatible cloud lane.
# Each provider stores its raw API key under one credential id; the
# RotatingCloudClient pulls them on init via ``get_api_key(<id>)``.
GROQ = "groq"
NVIDIA_NIM = "nvidia"
CEREBRAS = "cerebras"

# Sprint Ω.11 (Apr 26 2026): multi-vendor rotation. The rotating client
# can carry non-OpenAI-compatible slots (Gemini) alongside the
# OpenAI-compatible ones; Gemini reuses ``gemini_fast`` for the
# rotating-lane key. The Anthropic/Claude slot was retired on
# 2026-04-27 — owner does not hold an ``sk-ant-`` key, so the slot
# was removed from the rotation to keep the boot log clean.

# Backward-compat: older callsites and ``main.py`` use ``groq_api_key``
# for the single-provider Groq lane. We mirror it to the canonical id.
GROQ_LEGACY_ID = "groq_api_key"

# All available credential IDs
AVAILABLE_CREDENTIALS = {
    "GEMINI_FAST": GEMINI_FAST,
    "GEMINI_PRO": GEMINI_PRO,
    "GROQ": GROQ,
    "NVIDIA_NIM": NVIDIA_NIM,
    "CEREBRAS": CEREBRAS,
}

# Global credential manager instance. `_init_failed` short-circuits subsequent
# calls so a missing vault logs once per process, not once per credential lookup.
_manager: Optional[CredentialManager] = None
_init_failed: bool = False


def _get_manager() -> Optional[CredentialManager]:
    """Get or initialize credential manager.

    Returns None (never raises) when the vault cannot be opened. The boot
    path treats a missing vault as "no cloud keys available" and continues
    on local MLX only; a raised exception here would loop crash_guard.
    """
    global _manager, _init_failed
    if _manager is not None:
        return _manager
    if _init_failed:
        return None
    try:
        _manager = CredentialManager()
    except (ValueError, ImportError) as e:
        _init_failed = True
        logger.warning("Credential manager disabled: %s", e)
        return None
    except Exception as e:  # corrupt vault, permission denied, etc.
        _init_failed = True
        logger.error("Credential manager failed to initialise: %s", e)
        return None
    return _manager


def get_api_key(credential_id: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an API key by identifier.
    
    Args:
        credential_id: Credential identifier (e.g., "gemini_fast")
        default: Default value if not found
        
    Returns:
        API key or default value
        
    Example:
        >>> key = get_api_key("gemini_fast")
        >>> if key:
        ...     print("Key found")
    """
    manager = _get_manager()
    if manager is None:
        logger.warning(f"Credential manager unavailable, cannot retrieve {credential_id}")
        return default
    
    return manager.get(credential_id, default)


def get_gemini_fast_key() -> Optional[str]:
    """
    Get Gemini 1.5 Flash API key (fast, cost-efficient).
    
    Returns:
        API key or None
    """
    return get_api_key(GEMINI_FAST)


def get_gemini_pro_key() -> Optional[str]:
    """
    Get Gemini 2.0 Pro API key (powerful, reasoning).
    
    Returns:
        API key or None
    """
    return get_api_key(GEMINI_PRO)


def get_groq_key() -> Optional[str]:
    """Get Groq API key (Llama 3.x via LPU, fast OpenAI-compatible).

    Falls back to the legacy ``groq_api_key`` id used by the
    single-provider Groq client wired in ``main.py`` before Sprint Ω.9.
    """
    key = get_api_key(GROQ)
    if key:
        return key
    return get_api_key(GROQ_LEGACY_ID)


def get_nvidia_key() -> Optional[str]:
    """Get NVIDIA NIM (build.nvidia.com) API key (``nvapi-...``)."""
    return get_api_key(NVIDIA_NIM)


def get_cerebras_key() -> Optional[str]:
    """Get Cerebras Cloud API key (``csk-...``)."""
    return get_api_key(CEREBRAS)


def set_api_key(credential_id: str, value: str) -> bool:
    """
    Store an API key.
    
    WARNING: This is typically called only during setup.
    Use `scripts/setup_api_keys.py` for initial setup.
    
    Args:
        credential_id: Credential identifier
        value: API key value
        
    Returns:
        True if successful
    """
    manager = _get_manager()
    if manager is None:
        logger.error("Credential manager unavailable")
        return False
    
    try:
        manager.set_credential(credential_id, value)
        return True
    except Exception as e:
        logger.error(f"Failed to set credential {credential_id}: {e}")
        return False


def list_available_keys() -> list[str]:
    """
    List all stored credential identifiers (without values).
    
    Returns:
        List of credential IDs
        
    Example:
        >>> keys = list_available_keys()
        >>> print(keys)
        ['gemini_fast', 'gemini_pro']
    """
    manager = _get_manager()
    if manager is None:
        logger.warning("Credential manager unavailable")
        return []
    
    return manager.list_credentials()


def has_credential(credential_id: str) -> bool:
    """
    Check if a credential is available.
    
    Args:
        credential_id: Credential identifier
        
    Returns:
        True if credential exists
    """
    return credential_id in list_available_keys()


def delete_credential(credential_id: str) -> bool:
    """
    Delete a stored credential.
    
    WARNING: This is permanent!
    
    Args:
        credential_id: Credential identifier
        
    Returns:
        True if successful
    """
    manager = _get_manager()
    if manager is None:
        logger.error("Credential manager unavailable")
        return False
    
    try:
        return manager.delete_credential(credential_id)
    except Exception as e:
        logger.error(f"Failed to delete credential {credential_id}: {e}")
        return False


# Convenience exports
__all__ = [
    "get_api_key",
    "get_gemini_fast_key",
    "get_gemini_pro_key",
    "get_groq_key",
    "get_nvidia_key",
    "get_cerebras_key",
    "set_api_key",
    "list_available_keys",
    "has_credential",
    "delete_credential",
    "GEMINI_FAST",
    "GEMINI_PRO",
    "GROQ",
    "NVIDIA_NIM",
    "CEREBRAS",
    "AVAILABLE_CREDENTIALS",
]
