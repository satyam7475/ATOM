"""
ATOM — Search Tool (Real-Time Knowledge Gateway).

Integrates DuckDuckGo web search into the Cognitive Kernel routing
pipeline. When a query requires real-time information (news, prices,
weather, scores, current events), the kernel routes to CLOUD_SEARCH
which flows through this tool.

Pipeline:
  1. SecurityGateway sanitizes the query
  2. DuckDuckGo search (via existing web_researcher.py)
  3. Results summarized by local LLM (or Gemini fallback)
  4. Response tagged with source attribution

All search queries go through SecurityGateway — no raw system data
ever reaches the search engine.

Owner: Satyam
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("atom.tools.search")


class SearchTool:
    """Real-time knowledge search tool for the Cognitive Kernel.

    Wraps the existing web_researcher module and adds:
      - SecurityGateway integration
      - Result formatting for LLM summarization
      - Source attribution in responses
      - Latency tracking
    """

    def __init__(
        self,
        config: dict | None = None,
        security_gateway: Any = None,
        gemini_client: Any = None,
    ) -> None:
        self._config = config or {}
        self._gateway = security_gateway
        self._gemini = gemini_client

        cfg = self._config.get("search", {})
        self._max_results = int(cfg.get("max_results", 5))
        self._timeout = float(cfg.get("timeout_seconds", 8))

        self._total_searches = 0
        self._total_latency_ms = 0.0

        logger.info("SearchTool: max_results=%d, timeout=%.0fs", self._max_results, self._timeout)

    async def search(self, query: str) -> dict[str, Any]:
        """Search the web for real-time information.

        Returns a dict with:
          - text: formatted search results
          - sources: list of source URLs
          - latency_ms: search latency
          - success: whether search returned results
        """
        t0 = time.perf_counter()

        # Sanitize query through SecurityGateway
        safe_query = query
        if self._gateway:
            allowed, reason = self._gateway.allow_cloud(query, intent="search")
            if not allowed:
                return {
                    "text": "",
                    "sources": [],
                    "latency_ms": 0,
                    "success": False,
                    "error": f"blocked: {reason}",
                }
            safe_query = self._gateway.sanitize_outbound(query)

        if not safe_query.strip():
            return {
                "text": "",
                "sources": [],
                "latency_ms": 0,
                "success": False,
                "error": "empty_after_sanitization",
            }

        try:
            # Use existing web_researcher module
            import asyncio
            from core.web_researcher import search_instant, search_web_urls

            loop = asyncio.get_running_loop()

            # Run both searches in parallel
            instant_future = loop.run_in_executor(
                None, search_instant, safe_query,
            )
            urls_future = loop.run_in_executor(
                None, search_web_urls, safe_query, self._max_results,
            )

            instant_result, url_results = await asyncio.gather(
                instant_future, urls_future,
            )

            latency_ms = (time.perf_counter() - t0) * 1000
            self._total_searches += 1
            self._total_latency_ms += latency_ms

            # Format results
            text = self._format_results(safe_query, instant_result, url_results)
            sources = [u["url"] for u in url_results if u.get("url")]

            logger.info(
                "Search: %.0fms, %d sources, instant=%s",
                latency_ms, len(sources), bool(instant_result.get("abstract")),
            )

            return {
                "text": text,
                "sources": sources,
                "latency_ms": latency_ms,
                "success": bool(text),
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning("Search failed (%.0fms): %s", latency_ms, e)
            return {
                "text": "",
                "sources": [],
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e),
            }

    async def search_and_summarize(
        self,
        query: str,
        *,
        use_cloud_summarizer: bool = False,
    ) -> str:
        """Search and return a summarized response.

        By default, returns formatted results for local LLM to summarize.
        If use_cloud_summarizer=True and Gemini is available, uses Gemini
        to create a concise summary.
        """
        result = await self.search(query)

        if not result["success"] or not result["text"]:
            return ""

        # If cloud summarizer requested and available
        if use_cloud_summarizer and self._gemini and self._gemini.is_available:
            summarize_prompt = (
                f"Based on these search results, give a concise, accurate "
                f"answer to: {query}\n\n"
                f"Search results:\n{result['text'][:1500]}\n\n"
                f"Answer concisely:"
            )
            summary, ok = await self._gemini.ask(summarize_prompt, max_tokens=256)
            if ok and summary:
                return summary

        return result["text"]

    def _format_results(
        self,
        query: str,
        instant: dict[str, Any],
        urls: list[dict[str, str]],
    ) -> str:
        """Format search results into a readable summary."""
        parts: list[str] = []

        # Instant answer (DuckDuckGo Knowledge Graph)
        if instant.get("answer"):
            parts.append(f"Quick answer: {instant['answer']}")

        if instant.get("abstract"):
            abstract = instant["abstract"][:500]
            source = instant.get("abstract_source", "")
            if source:
                parts.append(f"{abstract} (Source: {source})")
            else:
                parts.append(abstract)

        if instant.get("definition"):
            parts.append(f"Definition: {instant['definition'][:200]}")

        # Web results
        if urls:
            parts.append(f"\nWeb results for '{query}':")
            for i, u in enumerate(urls[:self._max_results], 1):
                parts.append(f"  {i}. {u.get('title', 'Untitled')}")

        # Related topics
        if instant.get("related_topics"):
            related = [t["text"] for t in instant["related_topics"][:3]]
            if related:
                parts.append("Related: " + " | ".join(related))

        if not parts:
            return ""

        return "\n".join(parts)

    # ── Real-time info detection ─────────────────────────────────────

    @staticmethod
    def needs_realtime_info(query: str) -> bool:
        """Quick check if a query needs real-time search.

        Used by CognitiveKernel for pre-routing decisions.
        """
        import re
        pattern = re.compile(
            r"\b("
            r"latest|current|today|tonight|yesterday|this week|this month|"
            r"news|price|stock|weather|forecast|score|results?|"
            r"happening|right now|live|trending|update|"
            r"202[5-9]|who won|who is winning|"
            r"breaking|recent|how much (?:is|does|are)|"
            r"where (?:is|can i)|nearby|closest|"
            r"release date|when (?:is|does|will)"
            r")\b",
            re.I,
        )
        return bool(pattern.search(query))

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        avg = (
            self._total_latency_ms / self._total_searches
            if self._total_searches > 0 else 0.0
        )
        return {
            "total_searches": self._total_searches,
            "avg_latency_ms": round(avg, 1),
        }


__all__ = ["SearchTool"]
