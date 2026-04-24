"""
ATOM -- Document Ingestion Pipeline.

Enables ATOM to learn from documents, just like JARVIS learns from
Tony Stark's research. Supports: .txt, .md, .pdf, .docx, .py, .json, .csv

Pipeline: File -> Extract Text -> Chunk (~500 tokens, 50 overlap)
          -> Embed Chunks -> Store in Vector DB -> Available for RAG

Voice commands:
    "learn this document [path]"
    "what does [file] say about [topic]?"
    "forget document [name]"

Contract: CognitiveModuleContract (start, stop, persist)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.document")

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".pdf", ".docx", ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html"}

# Directories we never want to recursively ingest. ``.git`` alone can be
# millions of small files; ``node_modules`` and friends bloat the vector
# store with code we already have indexed structurally elsewhere. Owner
# can override via ``documents.excluded_dirs`` in settings.json.
_DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    ".venv", "venv", "env", ".env", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target", "out",
    ".idea", ".vscode", ".cursor",
    ".DS_Store",
})

_DEFAULT_MAX_FILES_PER_DIR = 200
_DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB -- bigger than this is
                                            # almost always logs / generated
                                            # output that pollute RAG.

_INGESTED_CACHE_PATH = Path("data/ingested_documents.json")


def _extract_text(path: Path) -> str:
    """Extract text content from a file based on its extension."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in {".txt", ".md", ".py", ".json", ".csv", ".log",
               ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return path.read_text(encoding="latin-1", errors="replace")
    return ""


def _extract_pdf(path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        logger.warning("pymupdf not installed -- cannot read PDF files")
        return ""
    except Exception:
        logger.debug("PDF extraction failed: %s", path, exc_info=True)
        return ""


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.warning("python-docx not installed -- cannot read DOCX files")
        return ""
    except Exception:
        logger.debug("DOCX extraction failed: %s", path, exc_info=True)
        return ""


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE,
                overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of approximately chunk_size tokens."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap

    return chunks


class DocumentIngestionEngine:
    """Ingest documents into ATOM's knowledge base for RAG retrieval."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = (config or {}).get("documents", {})
        self._ingested: dict[str, dict] = {}
        self._vector_store: Any = None
        self._embedding_engine: Any = None
        self._ready = False
        self._init()
        self._load_ingested_cache()

    def _init(self) -> None:
        try:
            from core.embedding_engine import get_embedding_engine
            from core.vector_store import VectorStore
            self._embedding_engine = get_embedding_engine()
            self._vector_store = VectorStore()
            self._ready = True
        except Exception:
            logger.info("Document ingestion: vectors unavailable")

    def _load_ingested_cache(self) -> None:
        """Restore the dedupe cache so repeat boots don't re-embed.

        Without this, every restart re-walks the same docs and pays the
        full embed cost again -- and bloats the vector store with the
        same chunks under fresh ids. We persist a tiny JSON of
        {file_hash: {name, path, chunks, chars, ingested_at}} alongside
        the vector DB.
        """
        try:
            if _INGESTED_CACHE_PATH.exists():
                data = json.loads(_INGESTED_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._ingested = {str(k): dict(v) for k, v in data.items()
                                      if isinstance(v, dict)}
                    logger.info(
                        "Document ingestion: %d previously-ingested files restored",
                        len(self._ingested),
                    )
        except Exception:
            logger.debug("Failed to restore ingested cache", exc_info=True)
            self._ingested = {}

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Hash of (path, size, mtime) -- changes when the file changes.

        We keep mtime in the key so editing a doc invalidates the dedupe
        cache and triggers re-ingestion next time. Path-only hashing
        (the original behaviour) silently skipped updated files.
        """
        try:
            st = path.stat()
            payload = f"{path}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            payload = str(path)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def ingest(self, file_path: str) -> dict:
        """Ingest a document: extract, chunk, embed, store."""
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return {"error": f"Unsupported file type: {path.suffix}"}

        if not self._ready:
            return {"error": "Document ingestion not available (vectors not ready)"}

        file_hash = self._compute_file_hash(path)
        if file_hash in self._ingested:
            return {
                "status": "already_ingested",
                "name": path.name,
                "chunks": self._ingested[file_hash]["chunks"],
            }

        text = _extract_text(path)
        if not text or len(text.strip()) < 10:
            return {"error": f"No readable content in {path.name}"}

        chunks = _chunk_text(text)
        if not chunks:
            return {"error": "Document produced no chunks after processing"}

        logger.info("Ingesting '%s': %d chars, %d chunks", path.name, len(text), len(chunks))

        try:
            embeddings = await self._embedding_engine.embed_batch(chunks)

            n = len(chunks)
            doc_ids = [f"doc_{file_hash}_{i}" for i in range(n)]
            metas = [
                {
                    "file_name": path.name,
                    "file_path": str(path),
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": n,
                }
                for i in range(n)
            ]
            self._vector_store.add_batch(
                "documents", chunks, embeddings,
                metadatas=metas, doc_ids=doc_ids,
            )

            self._ingested[file_hash] = {
                "name": path.name,
                "path": str(path),
                "chunks": len(chunks),
                "chars": len(text),
                "ingested_at": time.time(),
            }

            logger.info(
                "Document ingested: '%s' (%d chunks, %d chars)",
                path.name, len(chunks), len(text),
            )

            return {
                "status": "success",
                "name": path.name,
                "chunks": len(chunks),
                "chars": len(text),
            }

        except Exception as e:
            logger.exception("Document ingestion failed: %s", path.name)
            return {"error": f"Ingestion failed: {str(e)[:100]}"}

    async def ingest_directory(
        self,
        dir_path: str,
        *,
        recursive: bool = True,
        max_files: int | None = None,
        max_file_bytes: int | None = None,
    ) -> dict:
        """Bulk-ingest every supported file under a directory.

        Walks ``dir_path`` (recursively by default), filters to files
        we know how to extract, skips hidden directories, common build
        / cache dirs (``.git``, ``node_modules``, ``__pycache__`` etc.),
        and oversized files. Each candidate goes through the regular
        :meth:`ingest` pipeline so the per-file dedupe / persistence /
        vector-store path stays single-source-of-truth.

        Returns aggregate stats:

            {
                "status": "success",
                "directory": "/abs/path",
                "files_total": 42,            # candidates after filters
                "ingested": 38,               # newly embedded this call
                "skipped_already": 3,         # dedupe hits
                "skipped_size": 1,            # > max_file_bytes
                "errors": 0,                  # extraction / embed failures
                "chunks": 612,                # total chunks added
                "elapsed_s": 12.4,
            }

        On directory-level errors (missing dir, vectors not ready) a
        flat ``{"error": "..."}`` is returned instead.
        """
        base = Path(dir_path).expanduser().resolve()
        if not base.exists():
            return {"error": f"Directory not found: {dir_path}"}
        if not base.is_dir():
            return {"error": f"Not a directory: {dir_path}"}
        if not self._ready:
            return {"error": "Document ingestion not available (vectors not ready)"}

        cfg = self._config or {}
        excluded_dirs = set(cfg.get("excluded_dirs") or _DEFAULT_EXCLUDED_DIRS)
        if max_files is None:
            max_files = int(cfg.get("max_files_per_directory", _DEFAULT_MAX_FILES_PER_DIR))
        if max_file_bytes is None:
            max_file_bytes = int(cfg.get("max_file_bytes", _DEFAULT_MAX_FILE_BYTES))

        t0 = time.monotonic()
        candidates: list[Path] = []
        skipped_size = 0

        iterator = base.rglob("*") if recursive else base.iterdir()
        for p in iterator:
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue

            try:
                rel_parts = p.relative_to(base).parts
            except ValueError:
                rel_parts = p.parts

            # Hidden anywhere in the relative path -> skip (don't index
            # editor swap files, OS metadata, dotfiles, etc.).
            if any(part.startswith(".") for part in rel_parts):
                continue
            if any(part in excluded_dirs for part in rel_parts):
                continue
            if p.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue

            try:
                if p.stat().st_size > max_file_bytes:
                    skipped_size += 1
                    continue
            except OSError:
                continue

            candidates.append(p)
            if len(candidates) >= max_files:
                logger.info(
                    "ingest_directory: cap of %d files hit at %s; "
                    "remaining files in '%s' will be picked up later",
                    max_files, p, base,
                )
                break

        results = {
            "status": "success",
            "directory": str(base),
            "files_total": len(candidates),
            "ingested": 0,
            "skipped_already": 0,
            "skipped_size": skipped_size,
            "errors": 0,
            "chunks": 0,
        }

        for p in candidates:
            try:
                res = await self.ingest(str(p))
            except Exception:
                logger.exception("ingest_directory: ingest raised on %s", p)
                results["errors"] += 1
                continue

            status = res.get("status") if isinstance(res, dict) else None
            if status == "success":
                results["ingested"] += 1
                results["chunks"] += int(res.get("chunks", 0))
            elif status == "already_ingested":
                results["skipped_already"] += 1
            else:
                results["errors"] += 1

        results["elapsed_s"] = round(time.monotonic() - t0, 2)
        if results["ingested"] > 0:
            # Persist after a bulk run so the dedupe cache survives a
            # crash mid-corpus on the next boot.
            self.persist()
        logger.info(
            "ingest_directory '%s': %d ingested / %d already / %d skipped_size "
            "/ %d errors in %.1fs (%d total chunks)",
            base, results["ingested"], results["skipped_already"],
            results["skipped_size"], results["errors"],
            results["elapsed_s"], results["chunks"],
        )
        return results

    async def query_documents(self, query: str, k: int = 5) -> list[dict]:
        """Search ingested documents for relevant chunks."""
        if not self._ready:
            return []

        try:
            query_emb = await self._embedding_engine.embed(query)
            results = self._vector_store.search(
                "documents", query_emb, k=k, min_score=0.3,
            )
            return [
                {
                    "text": r.text,
                    "score": r.score,
                    "file": r.metadata.get("file_name", "unknown"),
                    "chunk": r.metadata.get("chunk_index", 0),
                }
                for r in results
            ]
        except Exception:
            logger.debug("Document query failed", exc_info=True)
            return []

    def get_ingested_list(self) -> list[dict]:
        return list(self._ingested.values())

    def forget_document(self, name: str) -> bool:
        for file_hash, info in list(self._ingested.items()):
            if info["name"].lower() == name.lower():
                del self._ingested[file_hash]
                logger.info("Forgot document: %s", name)
                return True
        return False

    def persist(self) -> None:
        if self._vector_store is not None:
            self._vector_store.persist()
        try:
            _INGESTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _INGESTED_CACHE_PATH.write_text(
                json.dumps(self._ingested, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to persist ingested cache", exc_info=True)
