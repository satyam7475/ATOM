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
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.document")

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".pdf", ".docx", ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html"}


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

    def _init(self) -> None:
        try:
            from core.embedding_engine import get_embedding_engine
            from core.vector_store import VectorStore
            self._embedding_engine = get_embedding_engine()
            self._vector_store = VectorStore()
            self._ready = True
        except Exception:
            logger.info("Document ingestion: vectors unavailable")

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def ingest(self, file_path: str) -> dict:
        """Ingest a document: extract, chunk, embed, store."""
        path = Path(file_path).resolve()

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return {"error": f"Unsupported file type: {path.suffix}"}

        if not self._ready:
            return {"error": "Document ingestion not available (vectors not ready)"}

        file_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
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
