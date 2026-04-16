"""
ATOM -- Native macOS NaturalLanguage Framework Bridge.

Provides sentiment analysis, language detection, and named-entity recognition
using Apple's on-device NaturalLanguage framework via PyObjC. Runs entirely on
the Neural Engine — zero network, zero external models.

Used by:
  - Emotion detector (sentiment → emotion mapping)
  - STT language routing (auto Hindi/English detection)
  - Intent engine enrichment (NER extracts person names, places, dates)

Requires: pyobjc-framework-NaturalLanguage (already in requirements.txt)

Owner: Satyam
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atom.macos.native_nlp")

_HAS_NL = False
_NL: Any = None

try:
    import NaturalLanguage as _NL  # type: ignore[import-untyped]
    _HAS_NL = True
except ImportError:
    pass


@dataclass
class SentimentResult:
    score: float
    label: str
    intensity: float


@dataclass
class LanguageResult:
    language: str
    confidence: float
    is_hindi: bool
    is_english: bool


@dataclass
class Entity:
    text: str
    entity_type: str
    range_start: int
    range_length: int


class NativeNLP:
    """Apple NaturalLanguage framework bridge for on-device NLP.

    All operations run on the Neural Engine via Apple's optimised CoreML
    models — no network, no additional model downloads.
    """

    def __init__(self) -> None:
        self._available = sys.platform == "darwin" and _HAS_NL
        if not self._available:
            logger.info(
                "NativeNLP unavailable (platform=%s, NaturalLanguage=%s)",
                sys.platform,
                _HAS_NL,
            )

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Sentiment Analysis ────────────────────────────────────────

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Return sentiment score in [-1.0, +1.0] with a human label.

        Negative = frustrated/angry, Positive = happy/satisfied, ~0 = neutral.
        """
        if not self._available or not text or not text.strip():
            return SentimentResult(score=0.0, label="neutral", intensity=0.0)

        try:
            tagger = _NL.NLTagger.alloc().initWithTagSchemes_(
                [_NL.NLTagSchemeSentimentScore]
            )
            tagger.setString_(text)
            tag_result = tagger.tagAtIndex_unit_scheme_tokenRange_(
                0,
                _NL.NLTokenUnitDocument,
                _NL.NLTagSchemeSentimentScore,
                None,
            )
            tag = tag_result[0] if isinstance(tag_result, tuple) else tag_result
            score = float(tag) if tag is not None else 0.0
        except Exception:
            logger.debug("Sentiment analysis failed", exc_info=True)
            return SentimentResult(score=0.0, label="neutral", intensity=0.0)

        abs_score = abs(score)
        if score > 0.3:
            label = "positive"
        elif score < -0.3:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            score=round(score, 3),
            label=label,
            intensity=round(abs_score, 3),
        )

    # ── Language Detection ────────────────────────────────────────

    def detect_language(self, text: str) -> LanguageResult:
        """Detect dominant language. Optimised for English/Hindi bilingual use."""
        if not self._available or not text or not text.strip():
            return LanguageResult(
                language="en", confidence=0.0,
                is_hindi=False, is_english=True,
            )

        try:
            recognizer = _NL.NLLanguageRecognizer.alloc().init()
            recognizer.processString_(text)
            dominant = recognizer.dominantLanguage()
            lang_code = str(dominant) if dominant else "en"

            hypotheses = recognizer.languageHypothesesWithMaximum_(3)
            confidence = 0.0
            if hypotheses and dominant in hypotheses:
                confidence = float(hypotheses[dominant])
        except Exception:
            logger.debug("Language detection failed", exc_info=True)
            return LanguageResult(
                language="en", confidence=0.0,
                is_hindi=False, is_english=True,
            )

        return LanguageResult(
            language=lang_code,
            confidence=round(confidence, 3),
            is_hindi=lang_code == "hi",
            is_english=lang_code in ("en", "en-US", "en-GB", "en-IN"),
        )

    # ── Named Entity Recognition ──────────────────────────────────

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities (people, places, organizations)."""
        if not self._available or not text or not text.strip():
            return []

        try:
            tagger = _NL.NLTagger.alloc().initWithTagSchemes_(
                [_NL.NLTagSchemeNameType]
            )
            tagger.setString_(text)

            entities: list[Entity] = []
            full_range = _NL.NSRange(0, len(text))

            tag_map = {
                _NL.NLTagPersonalName: "person",
                _NL.NLTagPlaceName: "place",
                _NL.NLTagOrganizationName: "organization",
            }

            def _callback(tag: Any, token_range: Any, stop: Any) -> None:
                if tag is None:
                    return
                entity_type = tag_map.get(tag)
                if entity_type is None:
                    return
                try:
                    start = token_range.location
                    length = token_range.length
                    entity_text = text[start : start + length]
                    entities.append(Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        range_start=start,
                        range_length=length,
                    ))
                except Exception:
                    logger.debug("NER entity span extract failed", exc_info=True)

            tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
                full_range,
                _NL.NLTokenUnitWord,
                _NL.NLTagSchemeNameType,
                _NL.NLTaggerOmitPunctuation | _NL.NLTaggerOmitWhitespace,
                _callback,
            )
            return entities
        except Exception:
            logger.debug("NER extraction failed", exc_info=True)
            return []

    # ── Tokenization ──────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        """Split text into linguistic tokens (words)."""
        if not self._available or not text:
            return text.split() if text else []

        try:
            tokenizer = _NL.NLTokenizer.alloc().initWithUnit_(
                _NL.NLTokenUnitWord
            )
            tokenizer.setString_(text)
            full_range = _NL.NSRange(0, len(text))
            tokens: list[str] = []

            def _callback(token_range: Any, stop: Any) -> None:
                try:
                    start = token_range.location
                    length = token_range.length
                    tokens.append(text[start : start + length])
                except Exception:
                    logger.debug("Tokenizer slice append failed", exc_info=True)

            tokenizer.enumerateTokensInRange_usingBlock_(full_range, _callback)
            return tokens
        except Exception:
            logger.debug("Tokenization failed", exc_info=True)
            return text.split()

    # ── Convenience: sentiment-to-emotion mapping ─────────────────

    def sentiment_to_emotion(self, text: str) -> str:
        """Map NL sentiment to ATOM emotion labels used by the TTS pipeline."""
        result = self.analyze_sentiment(text)
        if result.intensity < 0.15:
            return "neutral"
        if result.score > 0.5:
            return "happy"
        if result.score > 0.2:
            return "calm"
        if result.score < -0.5:
            return "frustrated"
        if result.score < -0.2:
            return "concerned"
        return "neutral"
