"""
ATOM Intent Engine -- File intents (create_folder, move_path, copy_path).
"""

from __future__ import annotations

import re

from .base import IntentResult, clean_slot

_CREATE_FOLDER = re.compile(
    r"\b(create|make)\s+(a\s+)?(new\s+)?folder\s+(named\s+)?(?P<name>.+?)(\s+(in|under|inside)\s+(?P<path>.+))?$",
    re.I,
)

_MOVE_PATH = re.compile(
    r"\b(move|shift)\s+(file|folder)?\s*(?P<src>.+?)\s+(to|into)\s+(?P<dst>.+)$",
    re.I,
)

_COPY_PATH = re.compile(
    r"\b(copy|duplicate)\s+(file|folder)?\s*(?P<src>.+?)\s+(to|into)\s+(?P<dst>.+)$",
    re.I,
)

_SPOTLIGHT_FOR = re.compile(
    r"\bspotlight(?:\s+search)?\s+for\s+(?P<q>.+)$",
    re.I,
)
_SPOTLIGHT = re.compile(r"\bspotlight\s+(?P<q>.+)$", re.I)
_SEARCH_MY_MAC = re.compile(r"\bsearch\s+my\s+mac\s+for\s+(?P<q>.+)$", re.I)
_FIND_ON_MAC = re.compile(
    r"\bfind\s+(?P<q>.+?)\s+on\s+my\s+mac\b",
    re.I,
)

_SMART_FIND = re.compile(
    r"\b(?:find|locate|pull\s+up|dig\s+up|show\s+me)\b"
    r"[^?.]{2,120}?\b"
    r"(?:pdf|pdfs|photo|photos|image|images|picture|pictures|"
    r"screenshot|screenshots|video|videos|movie|movies|"
    r"document|documents|doc|docs|word|excel|spreadsheet|"
    r"spreadsheets|powerpoint|slides|presentation|presentations|"
    r"audio|music|mp3|mp3s|zip|archive)\b",
    re.I,
)

_SMART_FIND_TIME = re.compile(
    r"\b(?:find|locate|pull\s+up|dig\s+up|show\s+me)\b"
    r"[^?.]{2,120}?\b"
    r"(?:today|yesterday|this\s+week|last\s+week|this\s+month|last\s+month|"
    r"this\s+year|\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago|"
    r"(?:past|last|previous)\s+\d+\s+(?:day|days|week|weeks|month|months))\b",
    re.I,
)


def check(text: str) -> IntentResult | None:
    if _SMART_FIND.search(text) or _SMART_FIND_TIME.search(text):
        return IntentResult(
            "smart_find_file",
            action="smart_find_file",
            action_args={"query": text.strip()},
        )

    m = _SEARCH_MY_MAC.search(text) or _SPOTLIGHT_FOR.search(text)
    if not m:
        m = _FIND_ON_MAC.search(text)
    if not m:
        m = _SPOTLIGHT.search(text)
    if m:
        q = clean_slot(m.group("q"))
        if q:
            return IntentResult(
                "spotlight_search",
                response=f"Searching your Mac for {q}.",
                action="spotlight_search",
                action_args={"query": q},
            )

    m = _CREATE_FOLDER.search(text)
    if m:
        name = clean_slot(m.group("name"))
        base_path = clean_slot(m.group("path"))
        if name:
            return IntentResult("create_folder", response=f"Creating folder {name}.",
                                action="create_folder",
                                action_args={"name": name, "path": base_path})

    m = _MOVE_PATH.search(text)
    if m:
        src = clean_slot(m.group("src"))
        dst = clean_slot(m.group("dst"))
        if src and dst:
            return IntentResult("move_path", response="Moving it now.",
                                action="move_path", action_args={"src": src, "dst": dst})

    m = _COPY_PATH.search(text)
    if m:
        src = clean_slot(m.group("src"))
        dst = clean_slot(m.group("dst"))
        if src and dst:
            return IntentResult("copy_path", response="Copying it now.",
                                action="copy_path", action_args={"src": src, "dst": dst})
    return None
