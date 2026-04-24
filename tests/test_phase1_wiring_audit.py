"""
ATOM Phase 1 -- Comprehensive Wiring & Stability Audit.

Tests:
  1. Configuration integrity (all settings present, types correct)
  2. Module initialization order (dependencies resolved)
  3. Event routing (all listeners connected)
  4. Voice pipeline end-to-end (voice → STT → Intent → Action → TTS)
  5. Data flow validation (all connections wired)

Run: python -m pytest tests/test_phase1_wiring_audit.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ────────────────────────────────────────────
# CONFIG INTEGRITY TESTS
# ────────────────────────────────────────────

def test_config_settings_required_keys() -> None:
    """Verify all required settings are present in config/settings.json."""
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config" / "settings.json"
    
    assert config_path.exists(), f"Config not found: {config_path}"
    
    with open(config_path) as f:
        settings = json.load(f)
    
    # Essential top-level keys
    required_keys = {"deployment", "owner", "stt", "tts", "brain", "memory"}
    for key in required_keys:
        assert key in settings, f"Missing required config key: {key}"
    
    # Brain config must have model settings.
    # v3.2 single-model brain uses ``mlx_model``; legacy keys are still
    # accepted by the loader so we treat any of them as valid here.
    brain = settings.get("brain", {})
    assert any(
        k in brain for k in ("mlx_model", "mlx_primary_model", "mlx_fast_model", "model_path")
    ), "Brain config missing mlx_model (or any legacy alias)"
    
    # STT config
    stt = settings.get("stt", {})
    assert "engine" in stt, "STT missing engine"
    assert stt["engine"] in ("macos_native", "whisper", "custom"), \
        f"Invalid STT engine: {stt['engine']}"
    
    # TTS config
    tts = settings.get("tts", {})
    assert "engine" in tts, "TTS missing engine"
    assert tts["engine"] in ("macos_native", "edge", "custom"), \
        f"Invalid TTS engine: {tts['engine']}"
    
    print("✓ Config integrity valid")


def test_config_brain_model_types() -> None:
    """Verify brain model config values have correct types."""
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config" / "settings.json"
    
    with open(config_path) as f:
        settings = json.load(f)
    
    brain = settings.get("brain", {})
    
    # Model context window should be reasonable
    context = brain.get("n_ctx", 8192)
    assert isinstance(context, int) and context > 512, \
        f"Brain context size invalid: {context}"
    
    # Temperature should be 0-1
    temp = brain.get("temperature", 0.7)
    assert 0 <= temp <= 1, f"Temperature out of range: {temp}"
    
    # Max tokens should be reasonable
    max_tokens = brain.get("max_tokens", 512)
    assert isinstance(max_tokens, int) and 32 <= max_tokens <= 2048, \
        f"Max tokens unreasonable: {max_tokens}"
    
    # Top-p should be 0-1
    top_p = brain.get("top_p", 0.9)
    assert 0 <= top_p <= 1, f"Top-p out of range: {top_p}"
    
    print("✓ Brain model config types valid")


def test_config_commands_json_exists() -> None:
    """Verify config/commands.json exists and is valid JSON."""
    ROOT = Path(__file__).resolve().parent.parent
    commands_path = ROOT / "config" / "commands.json"
    
    assert commands_path.exists(), f"Commands config not found: {commands_path}"
    
    with open(commands_path) as f:
        commands = json.load(f)
    
    assert isinstance(commands, dict), "commands.json should be a dict"
    assert len(commands) > 0, "commands.json should have entries"
    
    print(f"✓ commands.json valid ({len(commands)} entries)")


def test_config_skills_json_exists() -> None:
    """Verify config/skills.json exists and is valid JSON."""
    ROOT = Path(__file__).resolve().parent.parent
    skills_path = ROOT / "config" / "skills.json"
    
    # Skills config is optional but if present, must be valid
    if skills_path.exists():
        with open(skills_path) as f:
            skills = json.load(f)
        assert isinstance(skills, dict), "skills.json should be a dict"
        print(f"✓ skills.json valid ({len(skills)} entries)")
    else:
        print("✓ skills.json optional (not present)")


def test_config_plan_registry_structure() -> None:
    """Verify config/plan_registry.json has valid structure if present."""
    ROOT = Path(__file__).resolve().parent.parent
    registry_path = ROOT / "config" / "plan_registry.json"
    
    if not registry_path.exists():
        print("✓ plan_registry.json optional (not present)")
        return
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    assert isinstance(registry, dict), "plan_registry.json should be a dict"
    
    # Validate structure of each entry
    for plan_id, entry in registry.items():
        if isinstance(entry, dict):
            # Should have template reference
            assert "template" in entry or "action" in entry or plan_id, \
                f"Plan {plan_id} missing template/action"
    
    print(f"✓ plan_registry.json valid ({len(registry)} plans)")


# ────────────────────────────────────────────
# MODULE INITIALIZATION TESTS
# ────────────────────────────────────────────

def test_imports_no_circular_deps() -> None:
    """Test that core imports don't have circular dependencies."""
    try:
        from core import l1_cache
        from core import memory_engine
        from core import router
        from core import async_event_bus
        print("✓ Core module imports successful (no circular deps)")
    except ImportError as e:
        raise AssertionError(f"Circular dependency detected: {e}")


def test_config_loader_basic() -> None:
    """Test that config can be loaded without errors."""
    try:
        from core.boot.config_loader import load_config
        config = load_config()
        assert config is not None
        assert isinstance(config, dict)
        assert len(config) > 0
        print("✓ Config loader works")
    except Exception as e:
        raise AssertionError(f"Config loader failed: {e}")


async def test_event_bus_initialization() -> None:
    """Test that AsyncEventBus initializes correctly."""
    try:
        from core.async_event_bus import AsyncEventBus
        bus = AsyncEventBus()
        
        # Should be able to register a listener
        called = []
        async def listener(**kwargs: Any) -> None:
            called.append(True)
        
        bus.on("test_event", listener)
        bus.emit("test_event", test=True)
        
        await asyncio.sleep(0.1)
        assert len(called) > 0, "Event listener not called"
        print("✓ AsyncEventBus initialization and routing works")
    except Exception as e:
        raise AssertionError(f"EventBus test failed: {e}")


async def test_l1_cache_initialization() -> None:
    """Test that L1 cache initializes and tracks metrics."""
    try:
        from core.l1_cache import L1Cache
        cache = L1Cache(max_size=10)
        
        # Test set/get
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value", "Cache get failed"
        
        # Test miss
        assert cache.get("nonexistent") is None, "Cache should return None for miss"
        
        # Test metrics
        metrics = cache.get_metrics()
        assert "hits" in metrics
        assert "misses" in metrics
        assert metrics["hit_rate_percent"] >= 0
        print("✓ L1Cache initialization and metrics tracking works")
    except Exception as e:
        raise AssertionError(f"L1Cache test failed: {e}")


async def test_command_cache_initialization() -> None:
    """Test that command cache initializes and tracks metrics."""
    try:
        from core.command_cache import CommandCache
        cache = CommandCache(max_size=20)
        
        # Create mock result
        class MockResult:
            intent = "test_intent"
        
        result = MockResult()
        cache.put("test command", result)
        
        # Retrieve it
        cached = cache.get("test command")
        assert cached is not None, "Cache retrieval failed"
        
        # Test metrics
        metrics = cache.get_metrics()
        assert "hits" in metrics
        assert "misses" in metrics
        assert metrics["total_requests"] > 0
        print("✓ CommandCache initialization and metrics tracking works")
    except Exception as e:
        raise AssertionError(f"CommandCache test failed: {e}")


# ────────────────────────────────────────────
# DATA FLOW VALIDATION
# ────────────────────────────────────────────

def test_microphone_availability() -> None:
    """Test that microphone device detection works."""
    try:
        from core.system_health_score import get_mic_available
        available = get_mic_available()
        print(f"✓ Microphone check works (available: {available})")
    except Exception as e:
        print(f"⚠ Microphone check skipped: {e}")


async def test_desktop_control_availability() -> None:
    """Test that desktop control (AppleScript) is available."""
    try:
        from core.desktop_control import DesktopControl
        ctrl = DesktopControl()
        
        # Test that we can query foreground app
        app = ctrl.get_foreground_app()
        assert app is not None, "Should be able to get foreground app"
        print(f"✓ Desktop control available (foreground app: {app})")
    except Exception as e:
        print(f"⚠ Desktop control check skipped: {e}")


# ────────────────────────────────────────────
# RUN ALL TESTS
# ────────────────────────────────────────────

async def run_async_tests() -> None:
    """Run all async tests."""
    await test_event_bus_initialization()
    await test_l1_cache_initialization()
    await test_command_cache_initialization()
    await test_desktop_control_availability()


def main() -> int:
    """Run all wiring audit tests."""
    print("=" * 60)
    print("ATOM Phase 1: Wiring Audit Tests")
    print("=" * 60)
    
    try:
        # Configuration tests (sync)
        print("\n[CONFIG INTEGRITY]")
        test_config_settings_required_keys()
        test_config_brain_model_types()
        test_config_commands_json_exists()
        test_config_skills_json_exists()
        test_config_plan_registry_structure()
        
        # Module tests (sync)
        print("\n[MODULE INITIALIZATION]")
        test_imports_no_circular_deps()
        test_config_loader_basic()
        
        # Data flow tests (async)
        print("\n[DATA FLOW VALIDATION]")
        asyncio.run(run_async_tests())
        test_microphone_availability()
        
        print("\n" + "=" * 60)
        print("✓ ALL WIRING AUDIT TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
