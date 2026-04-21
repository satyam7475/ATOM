"""
ATOM Intent Engine -- Desktop control intents (scroll, click, press_key, go_back, tab_nav,
clipboard_shortcut, type_text, minimize, maximize, switch_window).
"""

from __future__ import annotations

import re

from .base import IntentResult

_SCROLL_DOWN = re.compile(
    r"\b(?:scroll\s+(?:down|below)|go\s+(?:down|below)|page\s+down|"
    r"show\s+(?:me\s+)?more(?:\s+options?)?|neeche(?:\s+jao)?|"
    r"move\s+down|swipe\s+down)\b", re.I)

_SCROLL_UP = re.compile(
    r"\b(?:scroll\s+(?:up|above)|go\s+(?:up|above)|page\s+up|"
    r"upar(?:\s+jao)?|move\s+up|swipe\s+up|go\s+back\s+up)\b", re.I)

_CLICK_SCREEN = re.compile(
    r"\b(?:click(?:\s+(?:here|on\s+(?:that|this|it)))?|"
    r"press\s+(?:that|this|it)|tap(?:\s+(?:here|that|this))?|"
    r"select\s+(?:that|this|it)|double\s+click)\b", re.I)

_PRESS_ENTER = re.compile(
    r"\b(?:press\s+enter|hit\s+enter|enter\s+key|submit|"
    r"press\s+return|hit\s+return)\b", re.I)

_PRESS_ESCAPE = re.compile(
    r"\b(?:press\s+escape|hit\s+escape|escape\s+key|"
    r"close\s+(?:this|that)|dismiss|press\s+esc)\b", re.I)

_GO_BACK = re.compile(
    r"\b(?:go\s+back|press\s+back|back\s+button|"
    r"previous\s+page|navigate\s+back)\b", re.I)

_TAB_NAV = re.compile(
    r"\b(?:next\s+tab|previous\s+tab|switch\s+tab|"
    r"new\s+tab|close\s+tab)\b", re.I)

_CLIPBOARD_SHORTCUT = re.compile(
    r"\b(?:copy(?:\s+(?:this|that|it))?|paste(?:\s+(?:it|here))?|"
    r"select\s+all|undo(?:\s+(?:that|it))?|cut(?:\s+(?:this|that|it))?)\b", re.I)

_TYPE_TEXT = re.compile(
    r"\b(?:type|write|enter\s+text)\s+(?P<text>.+)", re.I)

_MINIMIZE_WINDOW = re.compile(
    r"\b(minimize|minimize\s+(this\s+)?window|minimize\s+all|"
    r"chhota\s+karo|neeche\s+karo)\b", re.I)

_MAXIMIZE_WINDOW = re.compile(
    r"\b(maximize|maximize\s+(this\s+)?window|full\s+screen|fullscreen|"
    r"bada\s+karo)\b", re.I)

_SWITCH_WINDOW = re.compile(
    r"\b(switch\s+window|alt\s+tab|next\s+window|toggle\s+window|"
    r"dusri\s+window)\b", re.I)

# ── v22 Accessibility / UI element control ───────────────────────────

_DESCRIBE_FOCUSED = re.compile(
    r"\b(?:(?:what'?s|what\s+is|describe|tell\s+me\s+about)\s+"
    r"(?:the\s+|this\s+|that\s+)?focused\s*(?:element|field|control|item)?|"
    r"what\s+(?:am\s+i|is)\s+(?:currently\s+)?focused\s*(?:on|right\s+now|now)?|"
    r"describe\s+(?:the\s+)?(?:focus|focused)|what\s+has\s+focus)\b",
    re.I,
)

_READ_FOCUSED_TEXT = re.compile(
    r"\b(?:read\s+(?:the\s+|this\s+|that\s+)?focused\s+(?:field|text|input|control)|"
    r"(?:read|speak)\s+(?:what'?s|what\s+is)\s+in\s+(?:the\s+|this\s+|that\s+)?"
    r"(?:focused\s+)?(?:field|text\s+box|input|textbox)|"
    r"what'?s\s+(?:written|typed)\s+in\s+(?:the\s+)?(?:focused\s+)?(?:field|text\s+box|input))\b",
    re.I,
)

_SET_FOCUSED_TEXT = re.compile(
    r"\b(?:(?:type|write|fill|enter|put)\s+"
    r"(?P<text>.+?)\s+(?:in(?:to)?|to)\s+(?:the\s+|this\s+|that\s+)?"
    r"(?:focused|current|active)\s+(?:field|text\s+box|input|textbox|control)|"
    r"fill\s+(?:the\s+|this\s+|that\s+)?(?:focused\s+)?(?:field|text\s+box|input|textbox)"
    r"\s+with\s+(?P<text2>.+))"
    r"[\s.!]*$",
    re.I,
)

_CLICK_UI_ELEMENT = re.compile(
    r"\b(?:click\s+(?:on\s+)?(?:the\s+)?(?P<label>[^.?!]+?)\s+"
    r"(?P<role>button|link|checkbox|menu\s*item|toolbar\s*button|tab|field)|"
    r"press\s+(?:the\s+)?(?P<label2>[^.?!]+?)\s+(?P<role2>button|link)|"
    r"tap\s+(?:on\s+)?(?:the\s+)?(?P<label3>[^.?!]+?)\s+(?P<role3>button|link))"
    r"[\s.!]*$",
    re.I,
)


def check(text: str) -> IntentResult | None:
    if _MINIMIZE_WINDOW.search(text):
        return IntentResult("minimize_window", action="minimize_window", action_args={})
    if _MAXIMIZE_WINDOW.search(text):
        return IntentResult("maximize_window", action="maximize_window", action_args={})
    if _SWITCH_WINDOW.search(text):
        return IntentResult("switch_window", action="switch_window", action_args={})

    m = _CLICK_UI_ELEMENT.search(text)
    if m:
        label = (m.group("label") or m.group("label2") or m.group("label3") or "").strip()
        role = (m.group("role") or m.group("role2") or m.group("role3") or "button").strip()
        role = role.replace(" ", "")
        if label:
            return IntentResult(
                "click_ui_element", action="click_ui_element",
                action_args={"label": label, "role": role},
            )

    m = _SET_FOCUSED_TEXT.search(text)
    if m:
        typed = (m.group("text") or m.group("text2") or "").strip()
        if typed:
            return IntentResult(
                "set_focused_text", action="set_focused_text",
                action_args={"text": typed},
            )

    if _READ_FOCUSED_TEXT.search(text):
        return IntentResult(
            "read_focused_text", action="read_focused_text", action_args={},
        )

    if _DESCRIBE_FOCUSED.search(text):
        return IntentResult(
            "describe_focused_element",
            action="describe_focused_element", action_args={},
        )

    if _SCROLL_DOWN.search(text):
        amount = 5
        nums = re.findall(r"\d+", text)
        if nums:
            amount = min(int(nums[0]), 30)
        return IntentResult("scroll_down", action="scroll_down",
                            action_args={"clicks": amount})
    if _SCROLL_UP.search(text):
        amount = 5
        nums = re.findall(r"\d+", text)
        if nums:
            amount = min(int(nums[0]), 30)
        return IntentResult("scroll_up", action="scroll_up",
                            action_args={"clicks": amount})
    if _CLICK_SCREEN.search(text):
        is_double = "double" in text.lower()
        return IntentResult("click_screen", action="click_screen",
                            action_args={"double": is_double})
    if _PRESS_ENTER.search(text):
        return IntentResult("press_key", action="press_key",
                            action_args={"key": "enter"})
    if _PRESS_ESCAPE.search(text):
        return IntentResult("press_key", action="press_key",
                            action_args={"key": "escape"})
    if _GO_BACK.search(text):
        return IntentResult("go_back", action="go_back", action_args={})

    m = _TAB_NAV.search(text)
    if m:
        matched = m.group().lower()
        if "close" in matched:
            return IntentResult("tab_nav", action="hotkey_combo",
                                action_args={"combo": "ctrl+w"})
        elif "new" in matched:
            return IntentResult("tab_nav", action="hotkey_combo",
                                action_args={"combo": "ctrl+t"})
        elif "previous" in matched:
            return IntentResult("tab_nav", action="hotkey_combo",
                                action_args={"combo": "ctrl+shift+tab"})
        else:
            return IntentResult("tab_nav", action="hotkey_combo",
                                action_args={"combo": "ctrl+tab"})

    m = _CLIPBOARD_SHORTCUT.search(text)
    if m:
        matched = m.group().lower()
        combos = {
            "copy": "ctrl+c", "paste": "ctrl+v", "cut": "ctrl+x",
            "undo": "ctrl+z", "select all": "ctrl+a",
        }
        for keyword, combo in combos.items():
            if keyword in matched:
                return IntentResult("clipboard_shortcut", action="hotkey_combo",
                                    action_args={"combo": combo})

    m = _TYPE_TEXT.search(text)
    if m:
        typed = m.group("text").strip()
        if typed:
            return IntentResult("type_text", action="type_text",
                                action_args={"text": typed})
    return None


def quick_match(text: str) -> str | None:
    if _MINIMIZE_WINDOW.search(text):
        return "minimize_window"
    if _MAXIMIZE_WINDOW.search(text):
        return "maximize_window"
    if _SWITCH_WINDOW.search(text):
        return "switch_window"
    return None
