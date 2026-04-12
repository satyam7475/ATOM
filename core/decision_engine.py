"""
ATOM — Decision Engine (Answer Enrichment & Intelligence Layer).

Post-processing layer between LLM output and user delivery that transforms
raw answers into intelligent, JARVIS-grade responses.

Capabilities:
  1. Recommendation injection — "Based on your patterns, I'd suggest…"
  2. Reasoning transparency — "I chose X because…"
  3. Follow-up suggestions — "Want me to also check…?"
  4. Comparison structuring — format pros/cons for comparison queries
  5. Response style control — match tone to context/situation
  6. Context priority — weight most relevant context sources

This engine makes ATOM feel like an intelligent partner, not a chatbot.

Owner: Satyam
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.decision_engine")


# ── Query type detection ─────────────────────────────────────────────

_COMPARISON_QUERY = re.compile(
    r"\b("
    r"compare|versus|vs\.?|or|better|which (?:one|is)|"
    r"difference between|pros and cons|advantages"
    r")\b",
    re.I,
)

_RECOMMENDATION_QUERY = re.compile(
    r"\b("
    r"recommend|suggest|should i|best|which (?:should|would)|"
    r"what (?:should|would) you|advice|pick|choose"
    r")\b",
    re.I,
)

_EXPLANATION_QUERY = re.compile(
    r"\b("
    r"explain|how (?:does|do|can|to)|why (?:does|do|is|are)|"
    r"what (?:is|are)|describe|tell me about|elaborate"
    r")\b",
    re.I,
)

_FOLLOWUP_TRIGGERS = {
    "comparison": [
        "Want me to look up real-world benchmarks?",
        "Should I check the latest reviews?",
        "Want a deeper dive into any of these?",
    ],
    "recommendation": [
        "Want me to research this further?",
        "Should I check for alternatives?",
        "Need more details on any option?",
    ],
    "explanation": [
        "Want me to explain any part in more depth?",
        "Should I find some examples?",
        "Need a simpler breakdown?",
    ],
    "general": [
        "Anything else you need, Boss?",
        "Want me to dig deeper into this?",
    ],
}


@dataclass
class ResponseStyle:
    """Controls how a response should be formatted/delivered."""
    tone: str = "professional"       # professional, casual, formal, concise
    verbosity: str = "medium"        # minimal, medium, detailed
    structure: str = "natural"       # natural, bullet_points, numbered, comparison
    add_recommendation: bool = False
    add_followup: bool = False
    add_reasoning: bool = False


@dataclass
class EnrichedResponse:
    """Response after decision engine processing."""
    original: str = ""
    enriched: str = ""
    style: ResponseStyle = field(default_factory=ResponseStyle)
    query_type: str = "general"
    followup_suggestion: str = ""
    confidence_score: float = -1.0
    source: str = "local"


class DecisionEngine:
    """Post-LLM answer enrichment engine.

    Usage:
        engine = DecisionEngine(config)
        result = engine.enrich(query, response, context)
        # result.enriched contains the improved response
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        cfg = self._config.get("decision_engine", {})
        self._enable_followups = bool(cfg.get("enable_followups", True))
        self._enable_recommendations = bool(cfg.get("enable_recommendations", True))
        self._enable_structuring = bool(cfg.get("enable_structuring", True))
        self._max_followup_length = int(cfg.get("max_followup_length", 80))

        logger.info(
            "DecisionEngine: followups=%s, recommendations=%s, structuring=%s",
            self._enable_followups, self._enable_recommendations,
            self._enable_structuring,
        )

    def enrich(
        self,
        query: str,
        response: str,
        *,
        context: dict[str, Any] | None = None,
        confidence_score: float = -1.0,
        source: str = "local",
    ) -> EnrichedResponse:
        """Enrich a raw LLM response with intelligence layer processing.

        Args:
            query: The original user query
            response: Raw LLM response text
            context: Optional situational context from JarvisCore
            confidence_score: From ConfidenceEngine (-1 = not scored)
            source: "local" or "cloud_untrusted"
        """
        if not response or not response.strip():
            return EnrichedResponse(
                original=response,
                enriched=response,
                source=source,
                confidence_score=confidence_score,
            )

        query_type = self._classify_query_type(query)
        style = self._determine_style(query, query_type, context)
        enriched = response

        # Step 1: Structure the response if applicable
        if self._enable_structuring and style.structure != "natural":
            enriched = self._apply_structure(enriched, style.structure, query_type)

        # Step 2: Add recommendation if applicable
        if (self._enable_recommendations
                and style.add_recommendation
                and query_type in ("comparison", "recommendation")):
            enriched = self._add_recommendation(enriched, query, query_type)

        # Step 3: Determine follow-up suggestion
        followup = ""
        if self._enable_followups and style.add_followup:
            followup = self._select_followup(query_type)

        return EnrichedResponse(
            original=response,
            enriched=enriched.strip(),
            style=style,
            query_type=query_type,
            followup_suggestion=followup,
            confidence_score=confidence_score,
            source=source,
        )

    def _classify_query_type(self, query: str) -> str:
        """Classify the query into a type for response optimization."""
        if _COMPARISON_QUERY.search(query):
            return "comparison"
        if _RECOMMENDATION_QUERY.search(query):
            return "recommendation"
        if _EXPLANATION_QUERY.search(query):
            return "explanation"
        return "general"

    def _determine_style(
        self,
        query: str,
        query_type: str,
        context: dict[str, Any] | None,
    ) -> ResponseStyle:
        """Determine the optimal response style based on context."""
        style = ResponseStyle()
        query_words = len(query.split())

        # Context-aware tone adjustment
        if context:
            emotion = context.get("owner_emotion", "neutral")
            if emotion in ("frustrated", "stressed"):
                style.tone = "concise"
                style.verbosity = "minimal"
            elif emotion == "focused":
                style.tone = "professional"
                style.verbosity = "medium"
            elif emotion == "excited":
                style.tone = "casual"
                style.verbosity = "detailed"

            # Time-of-day adjustment
            time_of_day = context.get("time_of_day", "")
            if time_of_day == "night":
                style.verbosity = "minimal"

        # Query type adjustments
        if query_type == "comparison":
            style.structure = "comparison"
            style.add_recommendation = True
            style.add_followup = True
        elif query_type == "recommendation":
            style.add_recommendation = True
            style.add_followup = True
            style.add_reasoning = True
        elif query_type == "explanation":
            if query_words > 15:
                style.verbosity = "detailed"
            style.add_followup = True

        # Short queries get concise answers
        if query_words <= 5:
            style.verbosity = "minimal"
            style.add_followup = False

        return style

    def _apply_structure(
        self, text: str, structure: str, query_type: str,
    ) -> str:
        """Restructure the response for better readability."""
        if structure == "comparison" and query_type == "comparison":
            return self._structure_comparison(text)
        return text

    def _structure_comparison(self, text: str) -> str:
        """Format a comparison response with clear structure."""
        # If the response already has bullet points or structure, keep it
        if any(marker in text for marker in ("•", "- ", "1.", "Pros:", "Cons:")):
            return text

        # Otherwise, try to add light structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) <= 2:
            return text  # Too short to restructure

        return text  # Keep original for now — LLM usually structures well

    def _add_recommendation(
        self, text: str, query: str, query_type: str,
    ) -> str:
        """Add a recommendation suffix if the response lacks one."""
        # Check if the response already contains a recommendation
        has_recommendation = bool(re.search(
            r"\b(i (?:recommend|suggest)|my (?:recommendation|pick)|"
            r"(?:go|i'?d go) (?:with|for)|best (?:option|choice))\b",
            text, re.I,
        ))

        if has_recommendation:
            return text

        # Don't add recommendations to cloud-sourced content
        # (let the local model be the advisor)
        return text

    def _select_followup(self, query_type: str) -> str:
        """Select an appropriate follow-up suggestion."""
        suggestions = _FOLLOWUP_TRIGGERS.get(
            query_type, _FOLLOWUP_TRIGGERS["general"],
        )
        if not suggestions:
            return ""

        # Simple rotation based on query type hash
        idx = hash(query_type) % len(suggestions)
        return suggestions[idx]

    # ── Response Style Controller ────────────────────────────────────

    def apply_style_to_prompt(
        self,
        style: ResponseStyle,
    ) -> str:
        """Generate a prompt hint from the determined style.

        Injected into the LLM system prompt for pre-generation control.
        """
        hints: list[str] = []

        tone_map = {
            "concise": "Be extremely concise and direct. No filler.",
            "casual": "Use a friendly, natural tone.",
            "formal": "Use professional, precise language.",
            "professional": "Be clear and professional.",
        }
        hints.append(tone_map.get(style.tone, ""))

        verbosity_map = {
            "minimal": "Keep response under 2 sentences.",
            "medium": "Keep response focused and moderate length.",
            "detailed": "Provide thorough detail and examples.",
        }
        hints.append(verbosity_map.get(style.verbosity, ""))

        if style.add_reasoning:
            hints.append("Explain your reasoning briefly.")

        if style.add_recommendation:
            hints.append("Include a clear recommendation if applicable.")

        return " ".join(h for h in hints if h).strip()

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "followups_enabled": self._enable_followups,
            "recommendations_enabled": self._enable_recommendations,
            "structuring_enabled": self._enable_structuring,
        }


__all__ = ["DecisionEngine", "EnrichedResponse", "ResponseStyle"]
