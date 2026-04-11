"""
macOS Accessibility API for deep app control.

Provides a small, safe wrapper around AXUIElement so ATOM can inspect the
focused UI element, search the visible accessibility tree, click controls by
name, and read or set text fields when macOS Accessibility permission is
granted.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Callable

logger = logging.getLogger("atom.macos.accessibility")

_AX: Any = None
_APPKIT: Any = None
_HAS_ACCESSIBILITY = False

try:
    import ApplicationServices as _AX  # type: ignore[import-untyped]
    import AppKit as _APPKIT  # type: ignore[import-untyped]
    _HAS_ACCESSIBILITY = True
except ImportError:
    pass

_TEXT_ATTRS = (
    "AXValue",
    "AXSelectedText",
    "AXDescription",
    "AXTitle",
    "AXHelp",
)
_PRESS_ACTIONS = ("AXPress", "Press", "AXConfirm", "Confirm")


class AccessibilityAPI:
    """Thin wrapper around macOS AXUIElement APIs."""

    __slots__ = (
        "_ax",
        "_appkit",
        "_available",
        "_watch_thread",
        "_watch_stop",
    )

    def __init__(
        self,
        ax_module: Any | None = None,
        appkit_module: Any | None = None,
    ) -> None:
        self._ax = ax_module if ax_module is not None else _AX
        self._appkit = appkit_module if appkit_module is not None else _APPKIT
        self._available = bool(
            self._ax is not None and (sys.platform == "darwin" or ax_module is not None)
        )
        self._watch_thread: threading.Thread | None = None
        self._watch_stop: threading.Event | None = None

    @property
    def is_available(self) -> bool:
        return self._available

    def is_trusted(self, prompt: bool = False) -> bool:
        if not self._available or self._ax is None:
            return False
        try:
            trusted_with_options = getattr(self._ax, "AXIsProcessTrustedWithOptions", None)
            if trusted_with_options is not None:
                return bool(
                    trusted_with_options({"AXTrustedCheckOptionPrompt": bool(prompt)}),
                )
            trusted = getattr(self._ax, "AXIsProcessTrusted", None)
            if trusted is not None:
                return bool(trusted())
        except Exception:
            logger.debug("Accessibility trust check failed", exc_info=True)
        return False

    def get_frontmost_application(self) -> dict[str, Any]:
        if not self._available:
            return {"name": "", "bundle_id": "", "pid": 0}

        if self._appkit is not None:
            try:
                workspace = self._appkit.NSWorkspace.sharedWorkspace()
                app = workspace.frontmostApplication()
                if app is not None:
                    return {
                        "name": str(app.localizedName() or ""),
                        "bundle_id": str(app.bundleIdentifier() or ""),
                        "pid": int(app.processIdentifier() or 0),
                    }
            except Exception:
                logger.debug("Frontmost app lookup via AppKit failed", exc_info=True)

        app_ref = self._focused_application_ref()
        if app_ref is None:
            return {"name": "", "bundle_id": "", "pid": 0}
        title = self._text_from_element(app_ref)
        return {"name": title, "bundle_id": "", "pid": 0}

    def get_focused_element(self) -> dict[str, Any]:
        if not self._available:
            return {"available": False, "trusted": False}
        if not self.is_trusted(prompt=False):
            return {"available": True, "trusted": False}

        element = self._focused_element_ref()
        if element is None:
            return {
                "available": True,
                "trusted": True,
                "frontmost_app": self.get_frontmost_application(),
            }
        snapshot = self._element_snapshot(element)
        snapshot["available"] = True
        snapshot["trusted"] = True
        snapshot["frontmost_app"] = self.get_frontmost_application()
        return snapshot

    def get_element_tree(
        self,
        max_depth: int = 2,
        max_children: int = 20,
    ) -> dict[str, Any]:
        if not self._available:
            return {"available": False, "trusted": False}
        if not self.is_trusted(prompt=False):
            return {"available": True, "trusted": False}

        root = self._focused_window_ref() or self._focused_application_ref()
        if root is None:
            return {
                "available": True,
                "trusted": True,
                "frontmost_app": self.get_frontmost_application(),
            }
        tree = self._element_snapshot(
            root,
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
        )
        tree["available"] = True
        tree["trusted"] = True
        tree["frontmost_app"] = self.get_frontmost_application()
        return tree

    def read_focused_text(self, max_chars: int = 4000) -> str:
        if not self._available or not self.is_trusted(prompt=False):
            return ""
        element = self._focused_element_ref()
        if element is None:
            return ""
        return self._text_from_element(element)[:max_chars]

    def set_focused_text(self, text: str, *, append: bool = False) -> bool:
        if not self._available or not self.is_trusted(prompt=False):
            return False
        element = self._focused_element_ref()
        if element is None:
            return False
        if not self._is_attribute_settable(element, "AXValue"):
            return False

        final_text = str(text or "")
        if append:
            current = self.read_focused_text(max_chars=100000)
            final_text = current + final_text
        return self._set_attribute(element, "AXValue", final_text)

    def find_element_by_title(
        self,
        title: str,
        *,
        role: str | None = None,
        max_depth: int = 6,
        max_children: int = 40,
    ) -> dict[str, Any] | None:
        found = self._find_element_ref(title, role=role, max_depth=max_depth, max_children=max_children)
        if found is None:
            return None
        _element, snapshot = found
        snapshot["frontmost_app"] = self.get_frontmost_application()
        return snapshot

    def click_element_by_title(
        self,
        title: str,
        *,
        role: str | None = None,
        max_depth: int = 6,
        max_children: int = 40,
    ) -> bool:
        found = self._find_element_ref(title, role=role, max_depth=max_depth, max_children=max_children)
        if found is None:
            return False
        element, snapshot = found
        actions = [str(a) for a in snapshot.get("actions", []) if a]
        for action in _PRESS_ACTIONS:
            if action in actions:
                return self._perform_action(element, action)
        return False

    def start_focus_watcher(
        self,
        callback: Callable[[dict[str, Any]], None],
        *,
        interval_s: float = 0.5,
    ) -> bool:
        if not self._available or not self.is_trusted(prompt=False):
            return False
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return True

        self._watch_stop = threading.Event()

        def _watch() -> None:
            last_signature = ""
            while self._watch_stop is not None and not self._watch_stop.wait(interval_s):
                snapshot = self.get_focused_element()
                signature = self._signature(snapshot)
                if signature and signature != last_signature:
                    last_signature = signature
                    try:
                        callback(snapshot)
                    except Exception:
                        logger.debug("Accessibility watcher callback failed", exc_info=True)

        self._watch_thread = threading.Thread(
            target=_watch,
            name="accessibility_focus_watcher",
            daemon=True,
        )
        self._watch_thread.start()
        return True

    def stop_focus_watcher(self) -> None:
        if self._watch_stop is not None:
            self._watch_stop.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=2.0)
            self._watch_thread = None
        self._watch_stop = None

    def _signature(self, snapshot: dict[str, Any]) -> str:
        app = snapshot.get("frontmost_app") or {}
        return "|".join(
            [
                str((app or {}).get("name", "")),
                str(snapshot.get("role", "")),
                str(snapshot.get("title", "")),
                str(snapshot.get("value", ""))[:120],
            ],
        )

    def _focused_application_ref(self) -> Any | None:
        system = self._system_wide_element()
        if system is None:
            return None
        return self._copy_attribute(system, "AXFocusedApplication")

    def _focused_window_ref(self) -> Any | None:
        app = self._focused_application_ref()
        if app is None:
            return None
        return self._copy_attribute(app, "AXFocusedWindow")

    def _focused_element_ref(self) -> Any | None:
        system = self._system_wide_element()
        if system is None:
            return None
        element = self._copy_attribute(system, "AXFocusedUIElement")
        if element is not None:
            return element
        return self._focused_window_ref()

    def _system_wide_element(self) -> Any | None:
        if not self._available or self._ax is None:
            return None
        try:
            return self._ax.AXUIElementCreateSystemWide()
        except Exception:
            logger.debug("AXUIElementCreateSystemWide failed", exc_info=True)
            return None

    def _success(self, code: Any) -> bool:
        ok_code = getattr(self._ax, "kAXErrorSuccess", 0) if self._ax is not None else 0
        return code in (None, ok_code, 0)

    def _unwrap_result(self, result: Any, default: Any = None) -> Any:
        if isinstance(result, tuple):
            if not result:
                return default
            if len(result) == 1:
                return result[0]
            code = result[0]
            value = result[1]
            if self._success(code):
                return value
            return default
        return result

    def _copy_attribute(self, element: Any, name: str, default: Any = None) -> Any:
        if self._ax is None or element is None:
            return default
        try:
            result = self._ax.AXUIElementCopyAttributeValue(element, name, None)
            value = self._unwrap_result(result, default)
            return default if value is None else value
        except Exception:
            logger.debug("AX attribute read failed: %s", name, exc_info=True)
            return default

    def _copy_action_names(self, element: Any) -> list[str]:
        if self._ax is None or element is None:
            return []
        try:
            result = self._ax.AXUIElementCopyActionNames(element, None)
            raw = self._unwrap_result(result, [])
            return [str(item) for item in (raw or [])]
        except Exception:
            logger.debug("AX action read failed", exc_info=True)
            return []

    def _copy_children(self, element: Any) -> list[Any]:
        children = self._copy_attribute(element, "AXChildren", [])
        return list(children or [])

    def _is_attribute_settable(self, element: Any, name: str) -> bool:
        if self._ax is None or element is None:
            return False
        try:
            result = self._ax.AXUIElementIsAttributeSettable(element, name, None)
            value = self._unwrap_result(result, False)
            return bool(value)
        except Exception:
            logger.debug("AX settable check failed: %s", name, exc_info=True)
            return False

    def _set_attribute(self, element: Any, name: str, value: Any) -> bool:
        if self._ax is None or element is None:
            return False
        try:
            code = self._ax.AXUIElementSetAttributeValue(element, name, value)
            return self._success(code)
        except Exception:
            logger.debug("AX attribute set failed: %s", name, exc_info=True)
            return False

    def _perform_action(self, element: Any, action: str) -> bool:
        if self._ax is None or element is None:
            return False
        try:
            code = self._ax.AXUIElementPerformAction(element, action)
            return self._success(code)
        except Exception:
            logger.debug("AX action failed: %s", action, exc_info=True)
            return False

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _coerce_geometry(self, value: Any, kind: str) -> dict[str, float] | str | None:
        if value is None:
            return None
        try:
            if hasattr(value, "pointValue"):
                point = value.pointValue()
                return {"x": float(point.x), "y": float(point.y)}
            if hasattr(value, "sizeValue"):
                size = value.sizeValue()
                return {"width": float(size.width), "height": float(size.height)}
            if kind == "position" and hasattr(value, "x") and hasattr(value, "y"):
                return {"x": float(value.x), "y": float(value.y)}
            if kind == "size" and hasattr(value, "width") and hasattr(value, "height"):
                return {"width": float(value.width), "height": float(value.height)}
        except Exception:
            logger.debug("AX geometry conversion failed", exc_info=True)
        return self._normalize_text(value) or None

    def _element_snapshot(
        self,
        element: Any,
        *,
        depth: int = 0,
        max_depth: int = 0,
        max_children: int = 20,
    ) -> dict[str, Any]:
        snapshot = {
            "role": self._normalize_text(self._copy_attribute(element, "AXRole", "")),
            "subrole": self._normalize_text(self._copy_attribute(element, "AXSubrole", "")),
            "title": self._normalize_text(self._copy_attribute(element, "AXTitle", "")),
            "description": self._normalize_text(self._copy_attribute(element, "AXDescription", "")),
            "value": self._normalize_text(self._copy_attribute(element, "AXValue", "")),
            "identifier": self._normalize_text(self._copy_attribute(element, "AXIdentifier", "")),
            "enabled": bool(self._copy_attribute(element, "AXEnabled", False)),
            "focused": bool(self._copy_attribute(element, "AXFocused", False)),
            "position": self._coerce_geometry(self._copy_attribute(element, "AXPosition"), "position"),
            "size": self._coerce_geometry(self._copy_attribute(element, "AXSize"), "size"),
            "actions": self._copy_action_names(element),
        }
        if depth < max_depth:
            children = self._copy_children(element)[:max_children]
            snapshot["children"] = [
                self._element_snapshot(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_children=max_children,
                )
                for child in children
            ]
        return snapshot

    def _text_from_element(self, element: Any) -> str:
        for attr in _TEXT_ATTRS:
            value = self._copy_attribute(element, attr)
            text = self._normalize_text(value).strip()
            if text:
                return text
        return ""

    def _find_element_ref(
        self,
        title: str,
        *,
        role: str | None = None,
        max_depth: int,
        max_children: int,
    ) -> tuple[Any, dict[str, Any]] | None:
        if not self._available or not self.is_trusted(prompt=False):
            return None
        needle = " ".join(str(title or "").strip().lower().split())
        if not needle:
            return None
        role_filter = " ".join(str(role or "").strip().lower().split())
        root = self._focused_window_ref() or self._focused_application_ref()
        if root is None:
            return None

        queue: list[tuple[Any, int]] = [(root, 0)]
        seen: set[int] = set()
        while queue:
            element, depth = queue.pop(0)
            marker = id(element)
            if marker in seen:
                continue
            seen.add(marker)

            snapshot = self._element_snapshot(element)
            haystack_parts = [
                snapshot.get("title", ""),
                snapshot.get("description", ""),
                snapshot.get("value", ""),
                snapshot.get("identifier", ""),
            ]
            haystack = " ".join(str(part).lower() for part in haystack_parts if part)
            role_match = True
            if role_filter:
                role_text = " ".join(
                    [
                        str(snapshot.get("role", "")).lower(),
                        str(snapshot.get("subrole", "")).lower(),
                    ],
                )
                role_match = role_filter in role_text

            if needle in haystack and role_match:
                return element, snapshot

            if depth >= max_depth:
                continue
            for child in self._copy_children(element)[:max_children]:
                queue.append((child, depth + 1))
        return None
