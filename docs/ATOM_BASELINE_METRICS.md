## STEP 0.3 — BASELINE PERFORMANCE METRICS

> **Date:** 2026-04-27
> **Platform:** darwin / Python 3.11.15
> **CPU cores:** 10 | **RAM:** 7.0GB free / 16.0GB

### Module Initialization Latency

| Module | Time (ms) | Memory After (MB) | Memory Delta (MB) | Status |
|--------|-----------|-------------------|-------------------|--------|
| Config load + parse | 0.3 | 23.3 | +0.1 | OK |
| setup_logging() | 0.2 | 23.8 | +0.0 | OK |
| validate_and_log(config) | 44.4 | 30.4 | +6.4 | OK |
| Import 8 core modules | 53.9 | 35.0 | +4.6 | OK |
| AsyncEventBus() | 0.0 | 35.1 | +0.0 | OK |
| StateManager(bus) | 0.1 | 35.1 | +0.0 | OK |
| CacheEngine() | 0.0 | 35.1 | +0.0 | OK |
| MemoryEngine(config) | 1018.1 | 254.3 | +219.3 | OK |
| IntentEngine() | 0.0 | 254.3 | +0.0 | OK |
| CommandRegistry (get_registry) | 0.7 | 254.3 | +0.0 | OK |
| ContextEngine(config) | 0.0 | 254.3 | +0.0 | OK |
| IntentEngine.match() avg (10 queries) | 0.2 | 254.3 | +0.0 | OK |
| ContextEngine.get_bundle() | 158.6 | 276.8 | +0.0 | OK |
| SecurityPolicy(config) | 1.2 | 276.8 | +0.0 | OK |
| SecurityFortress(config) | 62.3 | 278.8 | +2.0 | OK |
| CodeIntrospector.scan() | 1793.8 | 300.8 | +22.0 | OK |
| SystemScanner(bus, config) | 2.2 | 298.7 | +0.0 | OK |
| SystemIndexer.start() | 31.0 | 298.9 | +0.2 | FAIL: RuntimeError: no running event loop |
| ToolRegistry (get_tool_registry) | 1.6 | 298.9 | +0.0 | OK |
| ReasoningPlanner(config) | 1.1 | 298.9 | +0.0 | OK |
| Cognitive layer (7 modules) | 7.0 | 298.9 | +0.0 | OK |
| speech_recognition import | 0.1 | 298.9 | +0.0 | FAIL: ImportError: speech_recognition not installed |
| faster_whisper import | 816.3 | 322.0 | +23.1 | OK |
| edge_tts import | 0.1 | 322.0 | +0.0 | FAIL: ImportError: edge-tts not installed |
| pygame import | 0.1 | 322.0 | +0.0 | FAIL: ImportError: pygame not installed |
| llama_cpp import | 0.1 | 322.0 | +0.0 | FAIL: ImportError: llama-cpp-python not installed |

### Intent Engine Latency (per query)

| Query | Time (ms) | Match |
|-------|-----------|-------|
| open chrome | 0.449 | open_app |
| what time is it | 0.091 | time |
| set volume to 50 | 0.158 | fallback |
| play some music | 0.100 | music_play |
| take a screenshot | 0.066 | screenshot |
| tell me a joke | 0.132 | fallback |
| what's the weather like | 0.078 | weather_report |
| search for python tutorials | 0.451 | search |
| how does quantum computing work | 0.219 | fallback |
| remind me to call mom at 5pm | 0.220 | fallback |

**Intent avg:** 0.196ms | **min:** 0.066ms | **max:** 0.451ms

### Memory Footprint

| Metric | Value |
|--------|-------|
| Baseline RSS (Python startup) | 21.0 MB |
| Final RSS (all modules loaded) | 322.0 MB |
| RSS delta | +301.0 MB |
| System RAM total | 16.0 GB |
| System RAM available | 7.0 GB |

### Voice Pipeline Status

| Component | Status |
|-----------|--------|
| speech_recognition import | MISSING |
| faster_whisper import | OK |
| edge_tts import | MISSING |
| pygame import | MISSING |
| llama_cpp import | MISSING |

### Performance Targets vs Current

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Intent match latency | 0.196ms | <100ms | MET |
| Module load (all) | 3993ms | <5000ms | MET |
| Memory (steady state) | 322MB | <3072MB | MET |
| STT latency | N/A (not installed) | <300ms | BLOCKED |
| LLM latency | N/A (not installed) | <2000ms | BLOCKED |
| TTS latency | N/A (not installed) | <100ms | BLOCKED |
| E2E voice-to-voice | N/A | <3000ms | BLOCKED |
