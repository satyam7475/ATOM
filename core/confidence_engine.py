"""
ATOM — Confidence Engine (Response Quality Scoring).

Post-LLM quality gate that analyzes generated responses and produces a
confidence score (0.0 → 1.0). When confidence drops below threshold,
the query is escalated to cloud intelligence for a better answer.

Scoring signals (weighted):
  - Repetition detection (n-gram overlap)        0.20
  - Vague language detection                     0.15
  - Length sanity check                          0.15
  - Coherence scoring                            0.15
  - Factual hedging detection                    0.10
  - Question avoidance detection                 0.10
  - Self-contradiction detection                 0.15

Integration:
  Called after local LLM generates a response. If should_escalate()
  returns True, the Cognitive Kernel reroutes to CLOUD_REASON with
  the ORIGINAL query (not the local response — no data leak).

Owner: Satyam
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger("atom.confidence")


# ── Detection patterns ───────────────────────────────────────────────

_VAGUE_PHRASES = re.compile(
    r"\b("
    r"i think|i believe|maybe|perhaps|not sure|i'?m not certain|"
    r"it depends|hard to say|it'?s complicated|basically|"
    r"sort of|kind of|more or less|to some extent|"
    r"i guess|i suppose|possibly|potentially|arguably|"
    r"it could be|that might|there might be"
    r")\b",
    re.I,
)

_HEDGING_PHRASES = re.compile(
    r"\b("
    r"as of my (?:training|knowledge|last update)|"
    r"i (?:cannot|can'?t) verify|"
    r"i don'?t have (?:access|information|data)|"
    r"i'?m (?:just )?an? (?:ai|language model|assistant)|"
    r"my (?:training|knowledge) (?:data|cutoff)|"
    r"i (?:cannot|can'?t) (?:browse|search|access)|"
    r"i (?:recommend|suggest) (?:checking|verifying)"
    r")\b",
    re.I,
)

_AVOIDANCE_PHRASES = re.compile(
    r"\b("
    r"i (?:cannot|can'?t) (?:help|assist) with that|"
    r"that'?s (?:outside|beyond) my|"
    r"i (?:don'?t|do not) (?:have|know)|"
    r"you (?:should|might want to) (?:ask|check|consult)|"
    r"i'?m not (?:able|qualified|equipped)"
    r")\b",
    re.I,
)

_CONTRADICTION_MARKERS = re.compile(
    r"\b("
    r"however|but actually|on the other hand|"
    r"wait|actually|i take that back|"
    r"that said|although|nevertheless|"
    r"contrary to what i (?:just )?said"
    r")\b",
    re.I,
)

# Question types for length sanity
_COMPLEX_QUESTION = re.compile(
    r"\b("
    r"explain|describe|compare|analyze|discuss|"
    r"what (?:are|is) the difference|how (?:does|do|can)|"
    r"why (?:does|do|is|are)|elaborate|detail"
    r")\b",
    re.I,
)

_SIMPLE_QUESTION = re.compile(
    r"\b("
    r"what (?:time|day|date)|who (?:is|are|was)|"
    r"when (?:is|was|did)|where (?:is|are)|"
    r"how (?:many|much|old|long|tall|far)|"
    r"yes or no|true or false"
    r")\b",
    re.I,
)


class ConfidenceEngine:
    """Post-LLM response quality scorer.

    Usage:
        engine = ConfidenceEngine(config)
        score = engine.score(query, response, plan)
        if engine.should_escalate(score, plan):
            # reroute to cloud
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("confidence", {})
        self._escalation_threshold = float(
            cfg.get("escalation_threshold", 0.45),
        )
        self._cloud_only_for_abstract = bool(
            cfg.get("cloud_only_for_abstract", True),
        )

        # Signal weights (must sum to 1.0)
        self._weights = {
            "repetition": 0.20,
            "vagueness": 0.15,
            "length_sanity": 0.15,
            "coherence": 0.15,
            "hedging": 0.10,
            "avoidance": 0.10,
            "contradiction": 0.15,
        }

        logger.info(
            "ConfidenceEngine: threshold=%.2f, cloud_abstract_only=%s",
            self._escalation_threshold, self._cloud_only_for_abstract,
        )

    def score(
        self,
        query: str,
        response: str,
        plan: Any = None,
    ) -> float:
        """Score a response's quality from 0.0 (terrible) to 1.0 (excellent).

        Higher is better. Each signal contributes a partial score weighted
        by its importance.
        """
        if not response or not response.strip():
            return 0.0

        signals = {
            "repetition": self._score_repetition(response),
            "vagueness": self._score_vagueness(response),
            "length_sanity": self._score_length_sanity(query, response),
            "coherence": self._score_coherence(response),
            "hedging": self._score_hedging(response),
            "avoidance": self._score_avoidance(response),
            "contradiction": self._score_contradiction(response),
        }

        # Weighted sum
        total = sum(
            signals[key] * self._weights[key]
            for key in self._weights
        )

        # Clamp to [0, 1]
        total = max(0.0, min(1.0, total))

        logger.debug(
            "Confidence: %.2f (rep=%.2f vag=%.2f len=%.2f coh=%.2f "
            "hed=%.2f avo=%.2f con=%.2f)",
            total,
            signals["repetition"], signals["vagueness"],
            signals["length_sanity"], signals["coherence"],
            signals["hedging"], signals["avoidance"],
            signals["contradiction"],
        )

        return round(total, 3)

    def should_escalate(
        self,
        score: float,
        plan: Any = None,
    ) -> bool:
        """Decide whether the response should be escalated to cloud.

        Returns True if:
          1. Score is below escalation threshold, AND
          2. The query is suitable for cloud (abstract/knowledge, not system)
        """
        if score >= self._escalation_threshold:
            return False

        # If plan indicates a local-only action (intent match, tool use),
        # never escalate regardless of confidence
        if plan is not None:
            path = getattr(plan, "path", None)
            if path is not None:
                path_val = path.value if hasattr(path, "value") else str(path)
                if path_val == "direct":
                    return False

            direct_action = getattr(plan, "direct_action", None)
            if direct_action:
                return False

        logger.info(
            "Confidence %.2f < threshold %.2f → escalation recommended",
            score, self._escalation_threshold,
        )
        return True

    # ── Signal scorers (each returns 0.0 = bad, 1.0 = good) ─────────

    def _score_repetition(self, text: str) -> float:
        """Detect repetitive content via n-gram overlap analysis."""
        words = text.lower().split()
        if len(words) < 10:
            return 0.9  # Short responses can't be judged for repetition

        # Trigram repetition detection
        trigrams = [
            tuple(words[i : i + 3])
            for i in range(len(words) - 2)
        ]
        if not trigrams:
            return 0.9

        counts = Counter(trigrams)
        total_trigrams = len(trigrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        repetition_ratio = repeated / total_trigrams

        # Also check sentence-level repetition
        sentences = [s.strip().lower() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 2:
            unique_sentences = len(set(sentences))
            sentence_ratio = 1.0 - (unique_sentences / len(sentences))
            repetition_ratio = max(repetition_ratio, sentence_ratio * 0.8)

        # Map: 0% repetition → 1.0, >30% → 0.0
        return max(0.0, 1.0 - (repetition_ratio / 0.30))

    def _score_vagueness(self, text: str) -> float:
        """Detect vague, non-committal language."""
        matches = _VAGUE_PHRASES.findall(text)
        words = text.split()
        if not words:
            return 0.5

        vague_density = len(matches) / max(len(words), 1)

        # Map: 0% vague → 1.0, >10% → 0.0
        return max(0.0, 1.0 - (vague_density / 0.10))

    def _score_length_sanity(self, query: str, response: str) -> float:
        """Check if response length is appropriate for the question type."""
        response_words = len(response.split())
        query_words = len(query.split())

        is_complex = bool(_COMPLEX_QUESTION.search(query))
        is_simple = bool(_SIMPLE_QUESTION.search(query))

        if is_simple:
            # Simple questions: 5-50 words is ideal
            if response_words < 2:
                return 0.1
            if response_words > 100:
                return 0.5  # Too verbose for a simple question
            return 1.0

        if is_complex:
            # Complex questions: 20-300 words is ideal
            if response_words < 10:
                return 0.2  # Too short for a complex question
            if response_words < 20:
                return 0.5
            if response_words > 500:
                return 0.7  # Verbose but at least attempting depth
            return 1.0

        # Unknown question type: medium expectations
        if response_words < 3:
            return 0.2
        if response_words > 400:
            return 0.7
        return 0.9

    def _score_coherence(self, text: str) -> float:
        """Basic coherence check via sentence transition analysis."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) <= 1:
            return 0.8  # Single sentence is coherent by default

        # Check if sentences share topic continuity (word overlap)
        total_transitions = 0
        smooth_transitions = 0

        for i in range(len(sentences) - 1):
            total_transitions += 1
            words_a = set(sentences[i].lower().split())
            words_b = set(sentences[i + 1].lower().split())
            # Remove common function words
            stop_words = {"the", "a", "an", "is", "are", "was", "were",
                         "in", "on", "at", "to", "for", "of", "and",
                         "but", "or", "it", "this", "that", "with"}
            words_a -= stop_words
            words_b -= stop_words

            if words_a and words_b:
                overlap = len(words_a & words_b) / max(
                    min(len(words_a), len(words_b)), 1,
                )
                if overlap > 0.05:
                    smooth_transitions += 1

        if total_transitions == 0:
            return 0.8

        return 0.4 + 0.6 * (smooth_transitions / total_transitions)

    def _score_hedging(self, text: str) -> float:
        """Detect factual hedging and knowledge limitation disclaimers."""
        matches = _HEDGING_PHRASES.findall(text)
        if not matches:
            return 1.0

        # Moderate penalty per hedging phrase
        penalty = min(len(matches) * 0.25, 0.8)
        return max(0.2, 1.0 - penalty)

    def _score_avoidance(self, text: str) -> float:
        """Detect question avoidance / refusal to answer."""
        matches = _AVOIDANCE_PHRASES.findall(text)
        if not matches:
            return 1.0

        # Heavy penalty for avoidance
        penalty = min(len(matches) * 0.35, 0.9)
        return max(0.1, 1.0 - penalty)

    def _score_contradiction(self, text: str) -> float:
        """Detect self-contradictions in the response."""
        matches = _CONTRADICTION_MARKERS.findall(text)
        if not matches:
            return 1.0

        # Light penalty — "however" is normal discourse
        if len(matches) <= 1:
            return 0.85
        if len(matches) <= 2:
            return 0.7

        penalty = min(len(matches) * 0.15, 0.6)
        return max(0.4, 1.0 - penalty)

    # ── Pre-confidence heuristic (fast, no LLM needed) ───────────────

    def pre_confidence_heuristic(self, query: str) -> float:
        """Quick estimate of how likely local LLM will handle this well.

        Used by CognitiveKernel to decide routing BEFORE generating.
        Returns 0.0 (local will struggle) to 1.0 (local will nail it).
        """
        query_lower = query.lower().strip()
        query_words = len(query_lower.split())

        # Factual real-time queries → low local confidence
        realtime_hints = re.compile(
            r"\b(latest|current|today|news|price|stock|weather|score|"
            r"result|happening|right now|202[5-9]|this week|yesterday)\b",
            re.I,
        )
        if realtime_hints.search(query_lower):
            return 0.2

        # Deep knowledge queries → medium confidence
        deep_hints = re.compile(
            r"\b(explain|compare|analyze|history of|difference between|"
            r"pros and cons|advantages|disadvantages|research)\b",
            re.I,
        )
        if deep_hints.search(query_lower) and query_words > 15:
            return 0.4

        # Simple conversational → high confidence
        if query_words < 8:
            return 0.85

        # System/tool queries → very high (local handles these)
        system_hints = re.compile(
            r"\b(open|close|kill|battery|cpu|ram|disk|time|date|"
            r"reminder|timer|screenshot|volume|brightness)\b",
            re.I,
        )
        if system_hints.search(query_lower):
            return 0.95

        return 0.6  # Default: moderate confidence

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "escalation_threshold": self._escalation_threshold,
            "weights": dict(self._weights),
            "cloud_abstract_only": self._cloud_only_for_abstract,
        }


__all__ = ["ConfidenceEngine"]
