"""
ATOM v22 — Hybrid Intelligence Verification Script.

Tests all new modules individually to verify they import correctly
and their core logic works. Run from the ATOM root directory:

    python tests/test_v22_hybrid.py
"""

import sys
import os
import logging

logger = logging.getLogger('atom.tests.test_v22_hybrid')

# Ensure ATOM root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = []


def _record_test(name, func):
    """Run a verification step and record pass/fail. Renamed from ``test``
    so pytest doesn't try to collect this dispatcher as a test case."""
    try:
        result = func()
        if result:
            results.append((PASS, name))
            print(f"  {PASS} {name}")
        else:
            results.append((FAIL, name))
            print(f"  {FAIL} {name}")
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))
        print(f"  {FAIL} {name}: {e}")


# Back-compat alias for the pre-rename callers below. Underscore prefix
# keeps it out of pytest's auto-collection net.
_test = _record_test


print("\n" + "=" * 60)
print("ATOM v22 — Hybrid Intelligence Verification")
print("=" * 60)

# ── Phase 1: Security Gateway ────────────────────────────────────
print("\n🔐 Phase 1: Security Gateway")


def test_gateway_import():
    from core.security_gateway import SecurityGateway
    return SecurityGateway is not None


def test_gateway_sanitize():
    from core.security_gateway import SecurityGateway
    gw = SecurityGateway()

    # Should redact file paths
    sanitized = gw.sanitize_outbound("Check /Users/satyam/Documents/secret.txt")
    assert "/Users/" not in sanitized, f"Path not redacted: {sanitized}"

    # Should redact IPs
    sanitized = gw.sanitize_outbound("Connect to 192.168.1.100")
    assert "192.168.1.100" not in sanitized, f"IP not redacted: {sanitized}"

    # Should redact emails
    sanitized = gw.sanitize_outbound("Send to satyam@gmail.com")
    assert "@gmail.com" not in sanitized, f"Email not redacted: {sanitized}"

    # Should redact tokens
    sanitized = gw.sanitize_outbound("token: abc123def456ghi789")
    assert "abc123def456ghi789" not in sanitized, f"Token not redacted: {sanitized}"

    return True


def test_gateway_block():
    from core.security_gateway import SecurityGateway
    gw = SecurityGateway()

    # System commands should be blocked
    allowed, reason = gw.allow_cloud("sudo rm -rf /tmp", intent="")
    assert not allowed, "System command should be blocked"

    # Config references should be blocked
    allowed, reason = gw.allow_cloud("read settings.json", intent="")
    assert not allowed, "Config reference should be blocked"

    # Normal knowledge query should be allowed
    allowed, reason = gw.allow_cloud("What is quantum computing?", intent="")
    assert allowed, f"Normal query should be allowed, got: {reason}"

    return True


def test_gateway_rate_limit():
    from core.security_gateway import SecurityGateway
    gw = SecurityGateway({"security_gateway": {"max_requests_per_minute": 2}})

    # Should allow first 2
    assert gw.allow_cloud("test query 1")[0], "First query should pass"
    assert gw.allow_cloud("test query 2")[0], "Second query should pass"

    # Third should be rate limited
    allowed, reason = gw.allow_cloud("test query 3")
    assert not allowed, "Third query should be rate limited"

    return True


_record_test("Import SecurityGateway", test_gateway_import)
_record_test("Sanitize sensitive data", test_gateway_sanitize)
_record_test("Block system commands", test_gateway_block)
_record_test("Rate limiting", test_gateway_rate_limit)

# ── Phase 2: Gemini Client ───────────────────────────────────────
print("\n☁️  Phase 2: Gemini Client")


def test_gemini_import():
    from core.cloud.gemini_client import GeminiClient
    return GeminiClient is not None


def test_gemini_unavailable_without_key():
    from core.cloud.gemini_client import GeminiClient
    client = GeminiClient()
    assert not client.is_available, "Should be unavailable without key"
    return True


def test_gemini_circuit_breaker():
    from core.cloud.gemini_client import GeminiClient
    client = GeminiClient()
    # Simulate failures
    for _ in range(3):
        client._record_failure("test")
    assert not client.is_available, "Circuit should be open after 3 failures"
    return True


_record_test("Import GeminiClient", test_gemini_import)
_record_test("Unavailable without API key", test_gemini_unavailable_without_key)
_record_test("Circuit breaker opens after failures", test_gemini_circuit_breaker)

# ── Phase 3: Cognitive Kernel ────────────────────────────────────
print("\n🧠 Phase 3: Cognitive Kernel Upgrade")


def test_kernel_cloud_paths():
    from core.cognitive_kernel import ExecPath
    assert hasattr(ExecPath, "CLOUD_REASON"), "Missing CLOUD_REASON path"
    assert hasattr(ExecPath, "CLOUD_SEARCH"), "Missing CLOUD_SEARCH path"
    assert ExecPath.CLOUD_REASON.value == "cloud_reason"
    assert ExecPath.CLOUD_SEARCH.value == "cloud_search"
    return True


def test_kernel_queryplan_fields():
    from core.cognitive_kernel import QueryPlan, ExecPath
    plan = QueryPlan(path=ExecPath.DIRECT)
    assert hasattr(plan, "cloud_augmented"), "Missing cloud_augmented field"
    assert hasattr(plan, "confidence_score"), "Missing confidence_score field"
    assert plan.cloud_augmented is False
    assert plan.confidence_score == -1.0
    return True


_record_test("CLOUD_REASON + CLOUD_SEARCH paths", test_kernel_cloud_paths)
_record_test("QueryPlan cloud fields", test_kernel_queryplan_fields)

# ── Phase 4: Confidence Engine ───────────────────────────────────
print("\n📊 Phase 4: Confidence Engine")


def test_confidence_import():
    from core.confidence_engine import ConfidenceEngine
    return ConfidenceEngine is not None


def test_confidence_scoring():
    from core.confidence_engine import ConfidenceEngine
    engine = ConfidenceEngine()

    # Good response should score high
    score = engine.score(
        "What is Python?",
        "Python is a high-level, general-purpose programming language. "
        "It emphasizes code readability with its notable use of significant "
        "indentation. Python supports multiple programming paradigms including "
        "structured, object-oriented and functional programming.",
    )
    assert score > 0.5, f"Good response scored too low: {score}"

    # Empty response should score 0
    score = engine.score("What is Python?", "")
    assert score == 0.0, f"Empty response should score 0: {score}"

    # Vague response should score lower
    score_vague = engine.score(
        "What is the best framework?",
        "I think maybe perhaps it depends on what you're doing, "
        "I'm not sure but I believe it could be sort of kind of "
        "hard to say, I suppose possibly arguably it's complicated.",
    )
    score_good = engine.score(
        "What is the best framework?",
        "For web development, React and Next.js are excellent choices. "
        "React provides a component-based architecture with a large ecosystem. "
        "Next.js adds server-side rendering and static generation on top.",
    )
    assert score_good > score_vague, (
        f"Good response ({score_good}) should score higher than vague ({score_vague})"
    )

    return True


def test_confidence_pre_heuristic():
    from core.confidence_engine import ConfidenceEngine
    engine = ConfidenceEngine()

    # System query → high confidence (local handles it)
    score = engine.pre_confidence_heuristic("open VS Code")
    assert score > 0.8, f"System query should be high confidence: {score}"

    # Real-time query → low confidence
    score = engine.pre_confidence_heuristic("What is the latest news today?")
    assert score < 0.4, f"Real-time query should be low confidence: {score}"

    return True


def test_confidence_escalation():
    from core.confidence_engine import ConfidenceEngine
    engine = ConfidenceEngine({"confidence": {"escalation_threshold": 0.5}})

    assert engine.should_escalate(0.3), "Low score should escalate"
    assert not engine.should_escalate(0.8), "High score should not escalate"
    return True


_record_test("Import ConfidenceEngine", test_confidence_import)
_record_test("Response quality scoring", test_confidence_scoring)
_record_test("Pre-confidence heuristic", test_confidence_pre_heuristic)
_record_test("Escalation decision", test_confidence_escalation)

# ── Phase 5: Decision Engine ────────────────────────────────────
print("\n🎯 Phase 5: Decision Engine")


def test_decision_import():
    from core.decision_engine import DecisionEngine
    return DecisionEngine is not None


def test_decision_enrichment():
    from core.decision_engine import DecisionEngine
    engine = DecisionEngine()

    result = engine.enrich(
        "Compare React vs Vue",
        "React has a larger ecosystem. Vue is easier to learn.",
    )
    assert result.query_type == "comparison", f"Should be comparison: {result.query_type}"
    assert result.enriched, "Should have enriched response"
    return True


def test_decision_style():
    from core.decision_engine import DecisionEngine, ResponseStyle
    engine = DecisionEngine()

    style = ResponseStyle(tone="concise", verbosity="minimal")
    hint = engine.apply_style_to_prompt(style)
    assert "concise" in hint.lower(), f"Style hint should mention concise: {hint}"
    return True


_record_test("Import DecisionEngine", test_decision_import)
_record_test("Query enrichment", test_decision_enrichment)
_record_test("Response style control", test_decision_style)

# ── Phase 6: Search Tool ────────────────────────────────────────
print("\n🔍 Phase 6: Search Tool")


def test_search_import():
    from core.tools.search_tool import SearchTool
    return SearchTool is not None


def test_search_realtime_detection():
    from core.tools.search_tool import SearchTool
    assert SearchTool.needs_realtime_info("What is the latest news?")
    assert SearchTool.needs_realtime_info("Current Bitcoin price")
    assert SearchTool.needs_realtime_info("Weather today in Delhi")
    assert not SearchTool.needs_realtime_info("What is a binary tree?")
    assert not SearchTool.needs_realtime_info("Explain recursion")
    return True


_record_test("Import SearchTool", test_search_import)
_record_test("Real-time info detection", test_search_realtime_detection)

# ── Phase 7: Preference Store ───────────────────────────────────
print("\n💾 Phase 7: Preference Store")


def test_preference_import():
    from core.memory.preference_store import PreferenceStore
    return PreferenceStore is not None


def test_preference_crud():
    from core.memory.preference_store import PreferenceStore
    import tempfile
    store = PreferenceStore({
        "memory": {"graph_db_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_prefs.db"
        )},
    })

    # Write
    store.learn("test", "language", "english", confidence=0.9)

    # Read
    value = store.get("test", "language")
    assert value == "english", f"Expected 'english', got '{value}'"

    # Context block
    block = store.get_context_block()
    assert "english" in block.lower() or "PREFERENCES" in block, (
        f"Context block should contain preferences: {block}"
    )

    store.shutdown()

    # Cleanup
    try:
        os.remove(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_prefs.db"
        ))
    except Exception:
        logger.debug('File remove failed', exc_info=True)

    return True


_record_test("Import PreferenceStore", test_preference_import)
_record_test("Preference CRUD + context block", test_preference_crud)

# ── Phase 8: Semantic Cache ─────────────────────────────────────
print("\n⚡ Phase 8: Semantic Cache")


def test_semantic_cache_import():
    from core.semantic_cache import SemanticCache
    return SemanticCache is not None


def test_semantic_cache_exact_match():
    import tempfile
    from pathlib import Path

    from core.semantic_cache import SemanticCache

    # Sprint A1: the semantic cache now persists to SQLite. Use a unique
    # temp DB per test invocation so the "miss" assertion doesn't hit
    # entries from a prior session (or another test run).
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "semantic_cache_test.sqlite")
        cfg = {
            "semantic_cache": {
                "enabled": True,
                "ttl_seconds": 60,
                "persistent": True,
                "db_path": db_path,
            },
        }
        cache = SemanticCache(cfg)

        assert cache.get("hello world") is None

        cache.put("hello world", "Hi there!")
        result = cache.get("hello world")
        assert result == "Hi there!", f"Expected 'Hi there!', got '{result}'"

    return True


_record_test("Import SemanticCache", test_semantic_cache_import)
_record_test("Exact match cache", test_semantic_cache_exact_match)

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total = len(results)
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print(f"\n{PASS} ALL TESTS PASSED — v22 Hybrid Intelligence is GO")
else:
    print(f"\n{FAIL} {failed} test(s) failed — review above")
print("=" * 60 + "\n")

# Only call sys.exit when this file is run as a standalone script.
# Pytest collects modules by importing them; calling sys.exit() at import
# time would crash the entire collection phase and skip every other test.
if __name__ == "__main__":
    sys.exit(0 if failed == 0 else 1)
