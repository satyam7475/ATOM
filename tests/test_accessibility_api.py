"""
Focused tests for the macOS Accessibility API wrapper.

Run: python3 -m tests.test_accessibility_api
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeRunningApp:
    def localizedName(self) -> str:
        return "Mail"

    def bundleIdentifier(self) -> str:
        return "com.apple.mail"

    def processIdentifier(self) -> int:
        return 4321


class _FakeWorkspace:
    @staticmethod
    def sharedWorkspace() -> "_FakeWorkspace":
        return _FakeWorkspace()

    def frontmostApplication(self) -> _FakeRunningApp:
        return _FakeRunningApp()


class _FakeAppKit:
    NSWorkspace = _FakeWorkspace


class _FakeAX:
    kAXErrorSuccess = 0

    def __init__(self, trusted: bool = True) -> None:
        self.trusted = trusted
        self.prompt_requested = False
        self.system = object()
        self.app = object()
        self.window = object()
        self.field = object()
        self.button = object()
        self.performed: list[tuple[object, str]] = []
        self.set_calls: list[tuple[object, str, str]] = []
        self.attrs = {
            (self.system, "AXFocusedApplication"): self.app,
            (self.system, "AXFocusedUIElement"): self.field,
            (self.app, "AXFocusedWindow"): self.window,
            (self.window, "AXRole"): "AXWindow",
            (self.window, "AXTitle"): "Compose",
            (self.window, "AXChildren"): [self.button, self.field],
            (self.button, "AXRole"): "AXButton",
            (self.button, "AXTitle"): "Send",
            (self.button, "AXEnabled"): True,
            (self.field, "AXRole"): "AXTextField",
            (self.field, "AXTitle"): "To",
            (self.field, "AXValue"): "alice@example.com",
            (self.field, "AXEnabled"): True,
            (self.field, "AXFocused"): True,
        }
        self.actions = {self.button: ["AXPress"]}
        self.settable = {(self.field, "AXValue"): True}

    def AXIsProcessTrustedWithOptions(self, options: dict) -> bool:
        self.prompt_requested = bool(options.get("AXTrustedCheckOptionPrompt"))
        return self.trusted

    def AXUIElementCreateSystemWide(self) -> object:
        return self.system

    def AXUIElementCopyAttributeValue(self, element: object, name: str, _unused: object) -> tuple[int, object]:
        return (0, self.attrs.get((element, name)))

    def AXUIElementCopyActionNames(self, element: object, _unused: object) -> tuple[int, list[str]]:
        return (0, self.actions.get(element, []))

    def AXUIElementIsAttributeSettable(self, element: object, name: str, _unused: object) -> tuple[int, bool]:
        return (0, self.settable.get((element, name), False))

    def AXUIElementSetAttributeValue(self, element: object, name: str, value: str) -> int:
        self.set_calls.append((element, name, value))
        self.attrs[(element, name)] = value
        return 0

    def AXUIElementPerformAction(self, element: object, action: str) -> int:
        self.performed.append((element, action))
        return 0


def _build_api(trusted: bool = True):
    from core.macos.accessibility_api import AccessibilityAPI

    fake_ax = _FakeAX(trusted=trusted)
    api = AccessibilityAPI(ax_module=fake_ax, appkit_module=_FakeAppKit())
    return api, fake_ax


def test_focused_element_snapshot() -> None:
    api, _fake_ax = _build_api()
    data = api.get_focused_element()
    assert data["role"] == "AXTextField"
    assert data["title"] == "To"
    assert data["value"] == "alice@example.com"
    assert data["frontmost_app"]["name"] == "Mail"
    print("  PASS: focused element snapshot")


def test_element_tree_contains_children() -> None:
    api, _fake_ax = _build_api()
    tree = api.get_element_tree(max_depth=2, max_children=5)
    titles = [child.get("title", "") for child in tree.get("children", [])]
    assert "Send" in titles
    assert "To" in titles
    print("  PASS: accessibility tree includes visible children")


def test_click_element_by_title() -> None:
    api, fake_ax = _build_api()
    ok = api.click_element_by_title("send", role="button")
    assert ok is True
    assert fake_ax.performed == [(fake_ax.button, "AXPress")]
    print("  PASS: click by accessible title performs press action")


def test_set_focused_text() -> None:
    api, fake_ax = _build_api()
    ok = api.set_focused_text("bob@example.com")
    assert ok is True
    assert fake_ax.set_calls[-1][1:] == ("AXValue", "bob@example.com")
    assert api.read_focused_text() == "bob@example.com"
    print("  PASS: focused text can be updated")


def test_desktop_control_wrapper_uses_accessibility() -> None:
    import core.desktop_control as desktop_control

    class _FakeAccessibilityWrapper:
        is_available = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def is_trusted(self, prompt: bool = False) -> bool:
            self.calls.append(("trust", prompt))
            return True

        def set_focused_text(self, text: str, append: bool = False) -> bool:
            self.calls.append((text, append))
            return True

    fake = _FakeAccessibilityWrapper()
    old_acc = desktop_control._ACCESSIBILITY  # noqa: SLF001
    old_macos = desktop_control._IS_MACOS  # noqa: SLF001
    try:
        desktop_control._ACCESSIBILITY = fake  # noqa: SLF001
        desktop_control._IS_MACOS = True  # noqa: SLF001
        msg = desktop_control.set_focused_text("hello")
        assert ("hello", False) in fake.calls
        assert "updated" in msg.lower()
    finally:
        desktop_control._ACCESSIBILITY = old_acc  # noqa: SLF001
        desktop_control._IS_MACOS = old_macos  # noqa: SLF001
    print("  PASS: desktop control routes focused text writes through accessibility")


def run_all() -> None:
    test_focused_element_snapshot()
    test_element_tree_contains_children()
    test_click_element_by_title()
    test_set_focused_text()
    test_desktop_control_wrapper_uses_accessibility()
    print("\n=== ACCESSIBILITY API TESTS PASSED ===\n")


if __name__ == "__main__":
    run_all()
