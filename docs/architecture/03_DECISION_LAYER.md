# ATOM Module 03: Decision Layer (Ring 3)

> Read this before changing: `core/router/router.py`, `core/cache_engine.py`, `core/memory_engine.py`, `core/conversation_memory.py`

## The Router — 3-Layer Intelligence

```
Layer 1: Intent Engine (<5ms, handles ~85%)
  → If match: direct action dispatch or pre-built response
  → If "fallback": drop to Layer 2

Layer 2: Cache + Memory (instant, parallel lookup)
  → CacheEngine: O(1) exact + O(32) Jaccard similarity
  → MemoryEngine: keyword overlap retrieval
  → If cache hit: serve instantly (no LLM)
  → If miss: drop to Layer 3

Layer 3: Local LLM (offline, 5-25s)
  → PromptBuilder → MiniLLM (1B or 3B based on complexity)
  → Fake streaming (sentence-by-sentence)
  → Result cached for next time
```

## Router Pipeline (inside `_route()`)

1. **Security sanitize** — strip injection chars, cap length
2. **Filler removal** — strip "um", "uh", "like", "basically"
3. **Skill expansion** — check SkillsRegistry for phrase match
4. **Pronoun resolution** — replace "it"/"that" with last entity
5. **Clipboard injection** — if "that error"/"this code" detected
6. **CommandCache check** — O(1) intent result lookup
7. **IntentEngine.classify()** — regex classification (<5ms)
8. **Cognitive check** — goal/prediction intents → dispatch via bus
9. **Local action** — confirm → execute → TTS response
10. **LLM fallback** — cache/memory lookup → quick reply → LLM query

## CacheEngine Design

- **Data structure:** OrderedDict-based LRU
- **TTL:** configurable (default 300s), self-tuned at runtime
- **Normalization:** stop-word removal + suffix stemming
- **Tier 1:** O(1) exact match on normalized key
- **Tier 2:** Jaccard similarity scan (top 32 entries, threshold 0.75)
- **Thread-safe:** threading.Lock guards all mutations

## MemoryEngine Design

- **Storage:** JSON file (`logs/memory.json`)
- **Retrieval:** keyword overlap scoring (set intersection)
- **Storage filter:** only stores queries >10 words OR containing tech keywords
- **Privacy:** all entries pass through privacy filter before storage
- **Interaction log:** every classified command logged to `logs/interactions.json`

## Conversation Continuity

- **Pronoun resolution:** "it"/"that"/"this" → last known entity
- **Entity extraction:** strips verbs/prepositions, takes significant tail words
- **Conversation window:** 5-turn rolling Q&A pairs for LLM context
- **Repeat detection:** same query within 60s triggers different LLM response

## Events Emitted by Router

| Event | When |
|-------|------|
| `intent_classified` | After every classification |
| `response_ready` | Local response generated |
| `thinking_ack` | Acknowledgment before LLM query |
| `cursor_query` | LLM query dispatched |
| `intent_chain_suggestion` | Follow-up suggestion after action |
| `set_performance_mode` | Performance mode change requested |

## Configuration

```json
{
  "cache": { "max_size": 128, "ttl_seconds": 300 },
  "memory": { "max_entries": 500, "top_k": 3 }
}
```

## Self-Tuning

Cache TTL auto-adjusts every 10 minutes:
- Hit rate > 65% → TTL × 1.2 (keep hits longer)
- Hit rate < 15% → TTL × 0.8 (reduce stale entries)
- Cooldown: 3 cycles between adjustments
