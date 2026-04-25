from __future__ import annotations

from core.embedding_engine import EmbeddingEngine
from core.rag.embedding_disk_cache import PersistentEmbeddingCache
from core.vector_store import VectorStore


def test_embedding_metadata_signature_tracks_provider() -> None:
    engine = EmbeddingEngine(
        {
            "embedding": {
                "backend": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "dimension": 384,
                "provider_version": "1",
                "warm_file": {"enabled": False},
            },
        },
    )

    meta = engine.provider_metadata()

    assert meta["provider"] == "fastembed"
    assert meta["model_name"] == "BAAI/bge-small-en-v1.5"
    assert meta["dimension"] == 384
    assert meta["signature"] == "fastembed:BAAI/bge-small-en-v1.5:384:1"


def test_vector_store_tags_fallback_entries_with_embedding_signature(tmp_path) -> None:
    store = VectorStore(
        {
            "vector_store": {"backend": "json", "path": str(tmp_path / "vectors")},
            "embedding": {
                "backend": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "dimension": 384,
                "provider_version": "1",
            },
        },
    )
    store._init_fallback()

    store.add("conversations", "hello", [0.1, 0.2], metadata={"source": "test"}, doc_id="x")

    entry = store._fallback_data["conversations"][0]
    assert entry["metadata"]["source"] == "test"
    assert entry["metadata"]["_embedding_provider"] == "fastembed"
    assert entry["metadata"]["_embedding_signature"] == "fastembed:BAAI/bge-small-en-v1.5:384:1"


def test_rag_embedding_disk_cache_namespaces_vectors(tmp_path) -> None:
    cache_a = PersistentEmbeddingCache(str(tmp_path / "emb.sqlite"), namespace="a")
    cache_b = PersistentEmbeddingCache(str(tmp_path / "emb.sqlite"), namespace="b")

    cache_a.put("same query", [1.0, 0.0])
    cache_b.put("same query", [0.0, 1.0])

    assert cache_a.get("same query") == [1.0, 0.0]
    assert cache_b.get("same query") == [0.0, 1.0]


def test_shadow_compare_is_explicit_for_legacy_backend() -> None:
    engine = EmbeddingEngine(
        {
            "embedding": {
                "backend": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "warm_file": {"enabled": False},
            },
        },
    )

    report = engine.shadow_compare_phrases(["what time is it"])

    assert report["enabled"] is False
    assert report["avg_similarity"] == 1.0
