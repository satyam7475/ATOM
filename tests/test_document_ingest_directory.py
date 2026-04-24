"""Tests for ``DocumentIngestionEngine.ingest_directory`` (Phase 2 corpus).

Covers the bulk walker added in v3.7:
  * directory walk filters to supported extensions
  * hidden dirs / common build dirs (``.git``, ``node_modules``...) are skipped
  * files larger than ``max_file_bytes`` are counted in ``skipped_size``
  * the ``max_files`` cap stops the walk early
  * dedupe cache is honored across calls (already_ingested branch)
  * dedupe cache is rehydrated from disk on a fresh engine
  * mtime-aware hashing causes edited files to re-ingest
  * non-existent path / file-instead-of-dir return structured errors

Run: python3 -m pytest tests/test_document_ingest_directory.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from core.document_ingestion import (
    _DEFAULT_MAX_FILE_BYTES,
    DocumentIngestionEngine,
)


class _FakeEmbedder:
    async def embed_batch(self, chunks):
        return [[0.0] * 4 for _ in chunks]

    async def embed(self, query):
        return [0.0] * 4


class _FakeVectorStore:
    def __init__(self):
        self.added: list[tuple[str, list, list, list, list]] = []

    def add_batch(self, collection, chunks, embeddings, metadatas=None, doc_ids=None):
        self.added.append((collection, list(chunks), list(embeddings),
                           list(metadatas or []), list(doc_ids or [])))

    def search(self, collection, query_emb, k=5, min_score=0.3):
        return []

    def persist(self):
        pass


def _make_engine(tmp_cache: Path) -> DocumentIngestionEngine:
    """Build an engine with fakes wired in -- bypasses heavy deps."""
    eng = DocumentIngestionEngine.__new__(DocumentIngestionEngine)
    eng._config = {}
    eng._ingested = {}
    eng._embedding_engine = _FakeEmbedder()
    eng._vector_store = _FakeVectorStore()
    eng._ready = True
    # Redirect persistence to a temp file so tests don't trample the
    # real data/ingested_documents.json cache.
    return eng


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path_factory, monkeypatch):
    """Each test gets its own ingested-cache file.

    Important: the cache must live OUTSIDE the directory we ingest,
    or the walker will pick it up as a ``.json`` document and try
    to embed it on every pass -- defeating the dedupe entirely.
    """
    cache_dir = tmp_path_factory.mktemp("ingest_cache")
    fake = cache_dir / "ingested_documents.json"
    monkeypatch.setattr(
        "core.document_ingestion._INGESTED_CACHE_PATH", fake,
    )
    yield fake


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Sandbox dir for the corpus under test (separate from cache)."""
    d = tmp_path / "corpus"
    d.mkdir()
    return d


def test_ingest_directory_filters_extensions_and_hidden(docs_dir: Path):
    (docs_dir / "ok.md").write_text("# hello world from ATOM corpus")
    (docs_dir / "ok.txt").write_text("plain text body for ingestion test")
    (docs_dir / "skip.bin").write_bytes(b"\x00\x01\x02\x03")
    (docs_dir / ".hidden.txt").write_text("should be skipped (hidden file)")
    sub = docs_dir / ".cache"
    sub.mkdir()
    (sub / "nope.md").write_text("hidden dir contents")

    eng = _make_engine(docs_dir)
    out = asyncio.run(eng.ingest_directory(str(docs_dir)))

    assert out["status"] == "success"
    assert out["files_total"] == 2  # only ok.md + ok.txt
    assert out["ingested"] == 2
    assert out["skipped_already"] == 0
    assert out["errors"] == 0
    assert out["chunks"] >= 2


def test_ingest_directory_skips_excluded_dirs(docs_dir: Path):
    (docs_dir / "keep.md").write_text("this is a kept document with enough body")
    for excluded in ("node_modules", ".git", "__pycache__", "dist"):
        d = docs_dir / excluded
        d.mkdir()
        (d / "noise.md").write_text("should be excluded from ingestion entirely")

    eng = _make_engine(docs_dir)
    out = asyncio.run(eng.ingest_directory(str(docs_dir)))

    assert out["status"] == "success"
    assert out["files_total"] == 1
    assert out["ingested"] == 1


def test_ingest_directory_size_cap(docs_dir: Path):
    (docs_dir / "small.md").write_text("tiny payload but long enough body")
    (docs_dir / "big.txt").write_text("X" * 5000)

    eng = _make_engine(docs_dir)
    out = asyncio.run(eng.ingest_directory(
        str(docs_dir), max_file_bytes=1000,
    ))

    assert out["files_total"] == 1
    assert out["ingested"] == 1
    assert out["skipped_size"] == 1


def test_ingest_directory_max_files_cap(docs_dir: Path):
    for i in range(8):
        (docs_dir / f"doc{i}.md").write_text(f"chunk {i} body content with more text")

    eng = _make_engine(docs_dir)
    out = asyncio.run(eng.ingest_directory(str(docs_dir), max_files=3))

    assert out["files_total"] == 3
    assert out["ingested"] == 3


def test_ingest_directory_dedupe_second_pass_is_no_op(docs_dir: Path):
    (docs_dir / "one.md").write_text("alpha bravo charlie delta echo foxtrot")
    (docs_dir / "two.md").write_text("lima mike november oscar papa quebec")

    eng = _make_engine(docs_dir)
    first = asyncio.run(eng.ingest_directory(str(docs_dir)))
    assert first["ingested"] == 2

    second = asyncio.run(eng.ingest_directory(str(docs_dir)))
    assert second["ingested"] == 0
    assert second["skipped_already"] == 2


def test_ingest_directory_persists_and_reloads_cache(
    docs_dir: Path, _isolate_cache: Path,
):
    (docs_dir / "memo.md").write_text("important memo body with sufficient text")

    eng_a = _make_engine(docs_dir)
    out = asyncio.run(eng_a.ingest_directory(str(docs_dir)))
    assert out["ingested"] == 1
    eng_a.persist()

    assert _isolate_cache.exists(), "ingested cache was not written to disk"
    cache = json.loads(_isolate_cache.read_text())
    assert isinstance(cache, dict) and len(cache) == 1

    # Fresh engine should rehydrate the dedupe cache from disk.
    eng_b = _make_engine(docs_dir)
    eng_b._load_ingested_cache()
    out2 = asyncio.run(eng_b.ingest_directory(str(docs_dir)))
    assert out2["ingested"] == 0
    assert out2["skipped_already"] == 1


def test_file_hash_changes_on_mtime_change(docs_dir: Path):
    p = docs_dir / "edit.md"
    p.write_text("v1 content with enough text to pass length filter")
    eng = _make_engine(docs_dir)
    h1 = eng._compute_file_hash(p)

    # Bump mtime explicitly so the test isn't filesystem-timing-flaky.
    new_mtime = os.path.getmtime(p) + 100
    os.utime(p, (new_mtime, new_mtime))
    h2 = eng._compute_file_hash(p)
    assert h1 != h2, "mtime change must invalidate dedupe hash"


def test_ingest_directory_missing_path(docs_dir: Path):
    eng = _make_engine(docs_dir)
    out = asyncio.run(eng.ingest_directory(str(docs_dir / "nope")))
    assert "error" in out and "not found" in out["error"].lower()


def test_ingest_directory_rejects_file_argument(docs_dir: Path):
    f = docs_dir / "single.md"
    f.write_text("a file, not a directory but with enough body text")

    eng = _make_engine(docs_dir)
    out = asyncio.run(eng.ingest_directory(str(f)))
    assert "error" in out and "not a directory" in out["error"].lower()


def test_default_max_file_bytes_constant_sane():
    # 2MB is the documented default; if it ever changes accidentally
    # the corpus walker could stall on huge logs / generated docs.
    assert _DEFAULT_MAX_FILE_BYTES == 2 * 1024 * 1024


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
