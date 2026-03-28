"""
ATOM -- Web Research Agent (AI OS Intelligence Service).

Enables ATOM to search the public web for information:
  - DuckDuckGo Instant Answer API (no API key needed)
  - URL-based search via DuckDuckGo Lite
  - Results feed into LLM for summarization
  - Self-improvement: searches for AI assistant techniques

Security: Only searches public web via HTTPS.
Corporate-safe: No dark/deep web access, respects proxy settings.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger("atom.researcher")

_DDG_API = "https://api.duckduckgo.com/"
_TIMEOUT = 8
_USER_AGENT = "ATOM-AI-OS/14 (Personal Assistant)"


def search_instant(query: str) -> dict:
    """Query DuckDuckGo Instant Answer API for quick facts.

    No API key required. Returns dict with abstract, answer,
    definition, and related_topics.
    """
    try:
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"{_DDG_API}?{params}"
        req = urllib.request.Request(
            url, headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        result = {
            "abstract": data.get("Abstract", ""),
            "abstract_source": data.get("AbstractSource", ""),
            "abstract_url": data.get("AbstractURL", ""),
            "answer": data.get("Answer", ""),
            "definition": data.get("Definition", ""),
            "related_topics": [],
        }
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and "Text" in topic:
                result["related_topics"].append({
                    "text": topic["Text"][:200],
                    "url": topic.get("FirstURL", ""),
                })
        return result
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return {"abstract": "", "answer": "", "error": str(e)}


def search_web_urls(query: str, num_results: int = 5) -> list[dict]:
    """Search DuckDuckGo Lite for web result URLs.

    Returns list of {title, url} dicts. Falls back gracefully.
    """
    try:
        params = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{params}"
        req = urllib.request.Request(
            url, headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        links = re.findall(
            r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
            html,
        )
        results = []
        seen: set[str] = set()
        for href, title in links:
            if (href.startswith("http") and href not in seen
                    and "duckduckgo" not in href):
                seen.add(href)
                results.append({"title": title.strip(), "url": href})
                if len(results) >= num_results:
                    break
        return results
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return []


def research_topic(topic: str) -> str:
    """Full research pipeline: instant answer + web results.

    Returns a formatted summary string suitable for TTS output.
    """
    instant = search_instant(topic)
    urls = search_web_urls(topic, num_results=5)

    parts: list[str] = []

    if instant.get("answer"):
        parts.append(f"Quick answer: {instant['answer']}")
    if instant.get("abstract"):
        parts.append(instant["abstract"][:500])
    if instant.get("definition"):
        parts.append(f"Definition: {instant['definition'][:200]}")

    if not parts and urls:
        parts.append(f"I found {len(urls)} web results for '{topic}'.")
        for i, u in enumerate(urls[:3], 1):
            parts.append(f"{i}. {u['title']}")

    if instant.get("related_topics"):
        related = [t["text"] for t in instant["related_topics"][:3]]
        if related:
            parts.append("Related: " + " | ".join(related))

    if not parts:
        return (f"I couldn't find immediate results for '{topic}'. "
                "Let me send this to AI for a deeper answer.")

    return " ".join(parts)


def get_self_improvement_ideas() -> list[str]:
    """Search for AI assistant improvement techniques.

    Returns actionable suggestions from public web sources.
    """
    queries = [
        "voice assistant performance optimization 2025",
        "AI assistant speech recognition accuracy tips",
        "personal AI assistant best features",
    ]
    ideas: list[str] = []
    for q in queries[:1]:
        instant = search_instant(q)
        if instant.get("abstract"):
            ideas.append(instant["abstract"][:300])
        for topic in instant.get("related_topics", [])[:2]:
            ideas.append(topic["text"][:200])

    return ideas if ideas else [
        "Consider adding more voice command patterns for your daily workflow.",
        "Optimize cache TTL based on your most frequent queries.",
        "Add keyboard shortcuts for your most used ATOM commands.",
    ]
