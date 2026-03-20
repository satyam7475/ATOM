"""
ATOM v9 -- Comprehensive Component Tests.

Tests all components that are safe to test without hardware:
  - AsyncEventBus (pub/sub, error isolation)
  - CacheEngine (LRU, TTL, normalisation)
  - MemoryEngine (keyword matching, persistence)
  - Router (intent classification, filler removal)
  - StructuredPromptBuilder (template, context injection)
  - TTS markdown cleaner (clean_for_tts)
  - STTAsync (adaptive timeout logic)

Run: python -m tests.test_all_components
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ────────────────────────────────────────────
# 1. AsyncEventBus
# ────────────────────────────────────────────

async def test_event_bus_basic() -> None:
    """Emit fires all registered handlers with correct kwargs."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    results: list[str] = []

    async def handler_a(text: str, **_kw) -> None:
        results.append(f"a:{text}")

    async def handler_b(text: str, **_kw) -> None:
        results.append(f"b:{text}")

    bus.on("test_event", handler_a)
    bus.on("test_event", handler_b)
    bus.emit("test_event", text="hello")

    await asyncio.sleep(0.05)
    assert "a:hello" in results
    assert "b:hello" in results
    print("  PASS: EventBus basic emit + multi-handler")


async def test_event_bus_error_isolation() -> None:
    """A failing handler does not crash other handlers."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    results: list[str] = []

    async def failing_handler(**_kw) -> None:
        raise ValueError("boom")

    async def good_handler(text: str, **_kw) -> None:
        results.append(text)

    bus.on("test", failing_handler)
    bus.on("test", good_handler)
    bus.emit("test", text="ok")

    await asyncio.sleep(0.05)
    assert "ok" in results
    print("  PASS: EventBus error isolation")


async def test_event_bus_off() -> None:
    """Unregistered handler is not called."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    results: list[str] = []

    async def handler(text: str, **_kw) -> None:
        results.append(text)

    bus.on("ev", handler)
    bus.off("ev", handler)
    bus.emit("ev", text="nope")

    await asyncio.sleep(0.05)
    assert len(results) == 0
    print("  PASS: EventBus off() removes handler")


async def test_event_bus_no_duplicates() -> None:
    """Same handler registered twice is only called once."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    count = {"n": 0}

    async def handler(**_kw) -> None:
        count["n"] += 1

    bus.on("ev", handler)
    bus.on("ev", handler)
    bus.emit("ev")

    await asyncio.sleep(0.05)
    assert count["n"] == 1
    print("  PASS: EventBus no duplicate handlers")


# ────────────────────────────────────────────
# 2. CacheEngine
# ────────────────────────────────────────────

def test_cache_put_get() -> None:
    """Basic put/get works."""
    from core.cache_engine import CacheEngine

    cache = CacheEngine(max_size=10, ttl=60)
    cache.put("explain java generics", "Java generics provide type safety")
    assert cache.get("explain java generics") == "Java generics provide type safety"
    assert cache.size == 1
    print("  PASS: Cache put/get")


def test_cache_normalisation() -> None:
    """Keys are normalised (lowercase, collapsed whitespace, stemmed)."""
    from core.cache_engine import CacheEngine

    cache = CacheEngine()
    cache.put("Explain  JAVA  Generics", "answer")
    assert cache.get("explain java generics") == "answer"
    assert cache.get("EXPLAIN JAVA GENERICS") == "answer"
    print("  PASS: Cache key normalisation")


def test_cache_fuzzy_plurals() -> None:
    """Plural/singular variants hit the same cache entry."""
    from core.cache_engine import CacheEngine

    cache = CacheEngine()
    cache.put("java streams explained", "Streams are functional pipelines")
    assert cache.get("java stream explained") == "Streams are functional pipelines"
    assert cache.get("java streams explained") == "Streams are functional pipelines"
    print("  PASS: Cache fuzzy plural matching")


def test_cache_fuzzy_stop_words() -> None:
    """Queries differing only by stop words hit the same entry.

    "what is dependency injection" strips "what" + "is" -> "dependency injection"
    "how does dependency injection" strips "how" + "does" -> "dependency injection"
    Both normalise to the same key.
    """
    from core.cache_engine import CacheEngine

    cache = CacheEngine()
    cache.put("what is dependency injection", "DI is a design pattern")
    assert cache.get("how does dependency injection") is not None
    assert cache.get("the dependency injection") is not None
    assert cache.get("which dependency injection") is not None
    # "explain" is a content word, so this is a DIFFERENT key (correct behaviour)
    assert cache.get("explain dependency injection") is None
    print("  PASS: Cache fuzzy stop-word removal")


def test_cache_fuzzy_verb_forms() -> None:
    """'-ing' suffix stripped so different verb forms match."""
    from core.cache_engine import CacheEngine

    cache = CacheEngine()
    cache.put("configuring spring boot", "Use application.yml")
    assert cache.get("configure spring boot") == "Use application.yml"
    print("  PASS: Cache fuzzy verb stemming")


def test_cache_ttl_expiry() -> None:
    """Entries expire after TTL."""
    from core.cache_engine import CacheEngine

    cache = CacheEngine(max_size=10, ttl=0.05)
    cache.put("test query alpha", "a")
    assert cache.get("test query alpha") == "a"

    time.sleep(0.1)
    assert cache.get("test query alpha") is None
    print("  PASS: Cache TTL expiry")


def test_cache_lru_eviction() -> None:
    """Oldest entry evicted when max_size exceeded."""
    from core.cache_engine import CacheEngine

    cache = CacheEngine(max_size=3, ttl=300)
    cache.put("alpha query one", "1")
    cache.put("beta query two", "2")
    cache.put("gamma query three", "3")
    cache.put("delta query four", "4")

    assert cache.get("alpha query one") is None
    assert cache.get("delta query four") == "4"
    assert cache.size == 3
    print("  PASS: Cache LRU eviction")


def test_cache_invalidate() -> None:
    """Invalidate removes specific entry."""
    from core.cache_engine import CacheEngine

    cache = CacheEngine()
    cache.put("test query gamma", "a")
    cache.invalidate("test query gamma")
    assert cache.get("test query gamma") is None
    print("  PASS: Cache invalidate")


# ────────────────────────────────────────────
# 3. MemoryEngine
# ────────────────────────────────────────────

async def test_memory_should_store() -> None:
    """Only stores technically relevant or long queries."""
    from core.memory_engine import MemoryEngine

    assert MemoryEngine.should_store("how to configure spring boot for kafka integration") is True
    assert MemoryEngine.should_store("what is java") is True
    assert MemoryEngine.should_store("hello") is False
    assert MemoryEngine.should_store("hi there") is False
    print("  PASS: Memory should_store filter")


async def test_memory_add_retrieve() -> None:
    """Add and retrieve by keyword overlap."""
    from core.memory_engine import MemoryEngine

    mem = MemoryEngine.__new__(MemoryEngine)
    mem._entries = []
    mem._max_entries = 500
    mem._default_top_k = 3

    await mem.add("spring boot kafka configuration", "Use @KafkaListener annotation")
    assert len(mem._entries) == 1

    results = await mem.retrieve("kafka configuration help", k=1)
    assert len(results) == 1
    assert "KafkaListener" in results[0]
    print("  PASS: Memory add + retrieve")


async def test_memory_empty_retrieve() -> None:
    """Retrieve from empty memory returns empty list."""
    from core.memory_engine import MemoryEngine

    mem = MemoryEngine.__new__(MemoryEngine)
    mem._entries = []
    mem._max_entries = 500
    mem._default_top_k = 3

    results = await mem.retrieve("anything", k=5)
    assert results == []
    print("  PASS: Memory empty retrieve")


async def test_memory_max_entries() -> None:
    """Memory caps at _max_entries."""
    from core.memory_engine import MemoryEngine

    cap = 100
    mem = MemoryEngine.__new__(MemoryEngine)
    mem._entries = []
    mem._max_entries = cap
    mem._default_top_k = 3

    for i in range(250):
        await mem.add(
            f"spring boot configuration item number {i} with details",
            f"Answer {i}",
        )

    assert len(mem._entries) <= cap
    print(f"  PASS: Memory max entries capped at {cap}")


# ────────────────────────────────────────────
# 4. Intent engine (was Router._classify in v9)
# ────────────────────────────────────────────

def test_intent_classify_exit() -> None:
    """Hard exit phrases (shutdown / quit) map to intent ``exit``."""
    from core.intent_engine import IntentEngine

    ie = IntentEngine()
    for text in ["shutdown", "exit", "quit", "stop atom"]:
        assert ie.classify(text).intent == "exit", f"'{text}' should be exit"
    print("  PASS: Intent exit classification")


def test_intent_classify_go_silent() -> None:
    """Casual bye phrases map to ``go_silent`` (checked before exit in meta_intents)."""
    from core.intent_engine import IntentEngine

    ie = IntentEngine()
    for text in ["bye", "goodbye", "chup karo"]:
        assert ie.classify(text).intent == "go_silent", f"'{text}' should be go_silent"
    print("  PASS: Intent go_silent classification")


def test_intent_classify_greeting() -> None:
    """Greeting intents are detected."""
    from core.intent_engine import IntentEngine

    ie = IntentEngine()
    for text in ["hello", "hi", "hey", "namaste", "good morning", "howdy"]:
        assert ie.classify(text).intent == "greeting", f"'{text}' should be greeting"
    print("  PASS: Intent greeting classification")


def test_intent_classify_system_style() -> None:
    """Open-app, screenshot, and lock map to specific intents (not a single system_cmd)."""
    from core.intent_engine import IntentEngine

    ie = IntentEngine()
    expected = {
        "open notepad": "open_app",
        "open chrome": "open_app",
        "take screenshot": "screenshot",
        "lock screen": "lock_screen",
    }
    for text, intent in expected.items():
        assert ie.classify(text).intent == intent, f"'{text}' should be {intent}"
    print("  PASS: Intent system-style classification")


def test_intent_classify_fallback_llm() -> None:
    """Generic queries fall through to ``fallback`` (local LLM / router)."""
    from core.intent_engine import IntentEngine

    ie = IntentEngine()
    # Avoid phrases that match other intents (e.g. "write" -> type_text).
    for text in [
        "explain dependency injection",
        "what is polymorphism in object oriented design",
        "what is the best framework for REST APIs",
    ]:
        assert ie.classify(text).intent == "fallback", f"'{text}' should be fallback"
    print("  PASS: Intent fallback (LLM) classification")


def test_router_compress_query() -> None:
    """Filler words are removed, whitespace collapsed."""
    from core.router import compress_query

    assert compress_query("um so basically how do I configure spring") == "how do I configure spring"
    assert compress_query("uh like you know what is java") == "what is java"
    assert compress_query("   extra   spaces   ") == "extra spaces"
    assert len(compress_query("x" * 2000)) == 1500
    print("  PASS: Router compress_query")


# ────────────────────────────────────────────
# 5. StructuredPromptBuilder
# ────────────────────────────────────────────

def test_prompt_builder_basic() -> None:
    """Basic prompt generation (v14+ uses role/focus/project_name, not language/framework)."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    builder = StructuredPromptBuilder({
        "developer": {
            "role": "Backend engineer",
            "focus": "Python and FastAPI microservices",
            "project_name": "TestProj",
        },
    })
    prompt = builder.build("how to add middleware")
    assert "Python" in prompt
    assert "FastAPI" in prompt
    assert "TestProj" in prompt
    assert "how to add middleware" in prompt
    assert "ATOM" in prompt
    print("  PASS: PromptBuilder basic generation")


def test_prompt_builder_with_context() -> None:
    """Context bundle is injected into prompt."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    builder = StructuredPromptBuilder({
        "developer": {"focus": "Java services", "project_name": "DemoSvc"},
    })
    prompt = builder.build(
        "explain this code",
        context={"active_app": "VS Code", "window_title": "main.py - VS Code", "clipboard": "def foo(): pass"},
    )
    assert "VS Code" in prompt
    assert "def foo(): pass" in prompt
    assert "Environment:" in prompt
    print("  PASS: PromptBuilder context injection")


def test_prompt_builder_without_context() -> None:
    """No context does not break prompt generation."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    builder = StructuredPromptBuilder({})
    prompt = builder.build("test query", context=None)
    assert "test query" in prompt
    assert "Environment:" not in prompt
    print("  PASS: PromptBuilder without context")


def test_prompt_builder_with_memory() -> None:
    """Memory summaries are injected."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    builder = StructuredPromptBuilder({})
    prompt = builder.build("test", memory_summaries=["Used @KafkaListener before"])
    assert "KafkaListener" in prompt
    assert "Relevant Past Context:" in prompt
    print("  PASS: PromptBuilder memory injection")


# ────────────────────────────────────────────
# 6. TTS markdown cleaner
# ────────────────────────────────────────────

def test_clean_for_tts() -> None:
    """Markdown is stripped for TTS consumption."""
    from voice.tts_async import clean_for_tts

    assert clean_for_tts("**bold text**") == "bold text"
    assert clean_for_tts("*italic*") == "italic"
    assert clean_for_tts("`inline code`") == "inline code"
    assert clean_for_tts("```python\nprint('hi')\n```") == ""
    assert clean_for_tts("# Header") == "Header"
    assert clean_for_tts("- bullet point") == "bullet point"
    assert clean_for_tts("> blockquote") == "blockquote"
    assert clean_for_tts("1. numbered item") == "numbered item"
    print("  PASS: TTS clean_for_tts markdown stripping")


# ────────────────────────────────────────────
# 7. STT / speech pipeline (v14: no _effective_timeout; Recognizer handles listen)
# ────────────────────────────────────────────

def test_stt_pipeline_helpers() -> None:
    """Public timing bounds + text correction helpers stay sane."""
    from voice.stt_async import MAX_RECORD_S, MIN_AUDIO_DURATION_S
    from voice.speech_detector import MAX_IDLE_LISTEN_S, correct_text

    assert MAX_IDLE_LISTEN_S >= 5
    assert MAX_RECORD_S >= MAX_IDLE_LISTEN_S
    assert 0.3 <= MIN_AUDIO_DURATION_S <= 2.0
    assert correct_text("hey adam") == "hey atom"
    print("  PASS: STT timing constants + speech_detector correct_text")


# ────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────

async def run_all() -> None:
    print("\n=== ATOM v9 -- Comprehensive Component Tests ===\n")

    # EventBus (async)
    await test_event_bus_basic()
    await test_event_bus_error_isolation()
    await test_event_bus_off()
    await test_event_bus_no_duplicates()

    # Cache (sync)
    test_cache_put_get()
    test_cache_normalisation()
    test_cache_fuzzy_plurals()
    test_cache_fuzzy_stop_words()
    test_cache_fuzzy_verb_forms()
    test_cache_ttl_expiry()
    test_cache_lru_eviction()
    test_cache_invalidate()

    # Memory (async)
    await test_memory_should_store()
    await test_memory_add_retrieve()
    await test_memory_empty_retrieve()
    await test_memory_max_entries()

    # Intent engine + router helpers (sync)
    test_intent_classify_exit()
    test_intent_classify_go_silent()
    test_intent_classify_greeting()
    test_intent_classify_system_style()
    test_intent_classify_fallback_llm()
    test_router_compress_query()

    # PromptBuilder (sync)
    test_prompt_builder_basic()
    test_prompt_builder_with_context()
    test_prompt_builder_without_context()
    test_prompt_builder_with_memory()

    # TTS cleaner (sync)
    test_clean_for_tts()

    # STT timeout (sync)
    test_stt_pipeline_helpers()

    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_all())
