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

# All available credential IDs
AVAILABLE_CREDENTIALS = {
    "GEMINI_FAST": GEMINI_FAST,
    "GEMINI_PRO": GEMINI_PRO,
    # Add more as needed:
    # "ANTHROPIC_CLAUDE": "anthropic_claude",
    # "OPENAI_GPT": "openai_gpt",
}

# Global credential manager instance
_manager: Optional[CredentialManager] = None


def _get_manager() -> Optional[CredentialManager]:
    """Get or initialize credential manager."""
    global _manager
    if _manager is None:
        try:
            _manager = CredentialManager()
        except ValueError as e:
            logger.warning(f"Credential manager not available: {e}")
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
    "set_api_key",
    "list_available_keys",
    "has_credential",
    "delete_credential",
    "GEMINI_FAST",
    "GEMINI_PRO",
    "AVAILABLE_CREDENTIALS",
]
