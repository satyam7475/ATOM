"""ATOM macOS-native modules (pyobjc wrappers for Apple frameworks)."""

from .accessibility_api import AccessibilityAPI
from .applescript_engine import AppleScriptEngine
from .fs_watcher import FSWatcher
from .fs_watcher_config import fs_watcher_settings, notable_file_hint
from .keychain_store import (
    KeychainVault,
    keychain_delete,
    keychain_get,
    keychain_set,
)
from .spotlight_engine import SpotlightEngine, spotlight_search

__all__ = [
    "AccessibilityAPI",
    "AppleScriptEngine",
    "FSWatcher",
    "KeychainVault",
    "SpotlightEngine",
    "fs_watcher_settings",
    "notable_file_hint",
    "keychain_delete",
    "keychain_get",
    "keychain_set",
    "spotlight_search",
]
