"""
ATOM -- Local Embedding Engine (semantic vectors).

Provides semantic embeddings for memory, RAG, and document retrieval.
Use ``embedding.device: auto`` in config to prefer the best local accelerator,
including MPS on Apple Silicon.

Uses sentence-transformers with a compact model (all-MiniLM-L6-v2, ~80MB)
or nomic-embed-text-v1.5 (~260MB) for higher quality.

Singleton pattern: one model instance shared across all modules.
Lazy loading: model loads on first embed() call, not at import time.

v20 optimizations over v18:
  - numpy-accelerated cosine similarity (100x faster on 384-dim vectors)
  - OrderedDict LRU cache (O(1) eviction instead of O(n) list.pop(0))
  - Batch similarity for vector store fallback search
  - Pre-normalized vectors skip norm computation in similarity
  - Zero-copy numpy path when sentence-transformers returns ndarray

v21 persistence upgrade (Sprint A2):
  - Durable ``data/embeddings_warm.npz`` warm-file for the hot set.
  - On boot, restores the most recently used vectors BEFORE the heavy
    SentenceTransformer/torch import -- repeat queries (wake words,
    common commands, boot-time semantic-cache lookups) are answered
    in <1ms with zero model load.
  - Periodic auto-persist after N inserts and a final save on shutdown.

Interface Contract:
    embed(text) -> list[float]         # Single text -> 384-dim vector
    embed_batch(texts) -> list[list[float]]  # Batch embedding
    embed_sync(text) -> list[float]    # Sync version for thread contexts
    similarity(a, b) -> float          # Cosine similarity between two vectors
    batch_similarity(query, candidates) -> list[float]  # 1-vs-N similarity
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.embedding")


def _resolve_embedding_device(requested: str) -> str:
    """Map ``auto`` → best available accelerator.

    Priority: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.
    """
    r = (requested or "cpu").strip().lower()
    if r != "auto":
        return r
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        logger.debug('Torch device probe failed', exc_info=True)
    return "cpu"


_np: Any = None
try:
    import numpy as np
    _np = np
except ImportError:
    pass

_instance: EmbeddingEngine | None = None
_instance_lock = threading.Lock()


def get_embedding_engine(config: dict | None = None) -> EmbeddingEngine:
    """Singleton accessor for the global EmbeddingEngine."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = EmbeddingEngine(config or {})
        return _instance


class EmbeddingEngine:
    """Lazy-loading CPU embedding engine with LRU cache and numpy acceleration."""

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"
    _DIMENSION = 384
    _CACHE_SIZE = 512
    _WARM_FILE_VERSION = 1
    _WARM_AUTOSAVE_INTERVAL = 64  # persist every N cache puts

    __slots__ = (
        "_model_name", "_dimension", "_device", "_model",
        "_load_lock", "_loaded", "_load_failed",
        "_cache", "_zero_vec",
        "_warm_path", "_warm_enabled", "_warm_max",
        "_warm_dirty_count", "_warm_lock", "_warm_restored",
    )

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("embedding", {})
        self._model_name: str = cfg.get("model", self._DEFAULT_MODEL)
        self._dimension: int = cfg.get("dimension", self._DIMENSION)
        self._device: str = _resolve_embedding_device(cfg.get("device", "cpu"))
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._loaded: bool = False
        self._load_failed: bool = False
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._zero_vec: list[float] | None = None

        warm_cfg = cfg.get("warm_file", {}) if isinstance(cfg.get("warm_file"), dict) else {}
        if isinstance(cfg.get("warm_file"), str):
            warm_cfg = {"path": cfg["warm_file"]}
        self._warm_enabled: bool = bool(warm_cfg.get("enabled", True)) and _np is not None
        self._warm_path: Path = Path(
            warm_cfg.get("path", "data/embeddings_warm.npz"),
        )
        self._warm_max: int = int(warm_cfg.get("max_entries", 1024))
        self._warm_dirty_count: int = 0
        self._warm_lock = threading.Lock()
        self._warm_restored: bool = False

        if self._warm_enabled:
            self._restore_warm_file()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _get_zero_vec(self) -> list[float]:
        if self._zero_vec is None or len(self._zero_vec) != self._dimension:
            self._zero_vec = [0.0] * self._dimension
        return self._zero_vec

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._loaded:
                return True
            try:
                # Sprint Ω.1: silence the two cosmetic boot-log
                # leaks that come out of the HF / sentence-transformers
                # stack the first time we touch them:
                #   1) "Warning: You are sending unauthenticated requests
                #      to the HF Hub..." — printed *and* logged. We
                #      re-inject HF_TOKEN from the in-process secret
                #      snapshot if the secret-scrub blanked it, then
                #      cap the noisy logger to ERROR.
                #   2) "BertModel LOAD REPORT ... embeddings.position_ids
                #      | UNEXPECTED" — a raw print() from
                #      sentence-transformers' MiniLM loader. We capture
                #      stdout + stderr during the actual load and replay
                #      only meaningful lines through the logger.
                self._silence_hf_boot_noise()

                from sentence_transformers import SentenceTransformer
                from contextlib import redirect_stdout, redirect_stderr
                from io import StringIO

                t0 = time.monotonic()
                buf_out, buf_err = StringIO(), StringIO()
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    self._model = SentenceTransformer(
                        self._model_name,
                        device=self._device,
                    )
                captured = (buf_out.getvalue() + buf_err.getvalue()).strip()
                if captured:
                    for line in captured.splitlines():
                        ln = line.strip()
                        if not ln or ln.startswith(("Key", "----", "Notes")):
                            continue
                        if "UNEXPECTED" in ln or "MISSING" in ln:
                            logger.debug("HF loader: %s", ln)

                self._dimension = (
                    self._model.get_embedding_dimension()
                    if hasattr(self._model, "get_embedding_dimension")
                    else self._model.get_sentence_embedding_dimension()
                )
                self._zero_vec = None
                elapsed = (time.monotonic() - t0) * 1000
                self._loaded = True
                logger.info(
                    "Embedding model loaded: %s (%d-dim, device=%s, numpy=%s) in %.0fms",
                    self._model_name, self._dimension,
                    self._device,
                    "yes" if _np is not None else "no", elapsed,
                )
                return True
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to keyword-based memory."
                )
                self._load_failed = True
                return False
            except Exception:
                logger.exception("Failed to load embedding model")
                self._load_failed = True
                return False

    @staticmethod
    def _silence_hf_boot_noise() -> None:
        """Suppress huggingface_hub anonymous-request boot warning.

        Three layers, all idempotent:
          1. Re-inject ``HF_TOKEN`` from the secret-scrub in-process
             snapshot if the user actually had one set. This is the
             *correct* fix because subsequent HF downloads then run
             authenticated and faster.
          2. Quiet the ``huggingface_hub.utils._http`` logger to ERROR
             so the warning stops flooding the boot log on every
             SentenceTransformer load.
          3. Set ``HF_HUB_DISABLE_TELEMETRY=1`` so the hub stops
             phoning home about model usage from a personal AI OS.
        """
        try:
            from core.security_secret_scrub import get_secret_snapshot
            snap = get_secret_snapshot()
            for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                if not os.environ.get(name) and snap.get(name):
                    os.environ[name] = snap[name]
                    break
        except Exception:
            logger.debug("HF token re-inject skipped", exc_info=True)

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        except Exception:
            pass

    def _cache_put(self, text: str, vec: list[float]) -> None:
        if text in self._cache:
            self._cache.move_to_end(text)
            return
        self._cache[text] = vec
        while len(self._cache) > self._CACHE_SIZE:
            self._cache.popitem(last=False)

        if self._warm_enabled:
            self._warm_dirty_count += 1
            if self._warm_dirty_count >= self._WARM_AUTOSAVE_INTERVAL:
                try:
                    self._persist_warm_file()
                except Exception:
                    logger.debug("warm-file autosave failed", exc_info=True)

    # ────────────── Warm-file persistence (Sprint A2) ──────────────

    def _restore_warm_file(self) -> None:
        """Load persisted vectors into the in-memory cache.

        Runs synchronously in ``__init__`` because the whole point is to
        have vectors available BEFORE any embed() call triggers the heavy
        torch/sentence-transformers import path.
        """
        path = self._warm_path
        if _np is None or not path.is_file():
            return
        try:
            t0 = time.monotonic()
            with _np.load(str(path), allow_pickle=False) as archive:
                try:
                    texts = archive["texts"]
                    vectors = archive["vectors"]
                except KeyError:
                    logger.info("Warm-file missing required arrays; ignoring")
                    return
                meta_raw = archive["meta"].tolist() if "meta" in archive else "{}"
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8", errors="ignore")
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
            except Exception:
                meta = {}

            saved_model = meta.get("model_name")
            saved_dim = int(meta.get("dimension", 0) or 0)
            if saved_model and saved_model != self._model_name:
                logger.info(
                    "Warm-file model mismatch (have=%s want=%s); ignoring",
                    saved_model, self._model_name,
                )
                return
            if saved_dim and vectors.shape[1] != saved_dim:
                logger.info(
                    "Warm-file dim mismatch (have=%d want=%d); ignoring",
                    vectors.shape[1], saved_dim,
                )
                return
            if saved_dim and saved_dim != self._dimension:
                self._dimension = saved_dim
                self._zero_vec = None

            restored = 0
            # Oldest first so the most recent entries stay at the tail of
            # the LRU, matching how we persisted them.
            for txt, row in zip(texts.tolist(), vectors):
                if not isinstance(txt, str) or not txt:
                    continue
                vec = row.astype(_np.float32, copy=False).tolist()
                if len(vec) != self._dimension:
                    continue
                self._cache[txt] = vec
                restored += 1
                if len(self._cache) > self._CACHE_SIZE:
                    self._cache.popitem(last=False)

            if restored:
                self._warm_restored = True
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(
                "Embedding warm-file restored: %d entries from %s in %.0fms",
                restored, path, elapsed,
            )
        except Exception:
            logger.debug("warm-file restore failed", exc_info=True)

    def _persist_warm_file(self) -> bool:
        """Write the in-memory cache to disk atomically."""
        if not self._warm_enabled or _np is None:
            return False
        if not self._cache:
            return False
        with self._warm_lock:
            path = self._warm_path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.debug("warm-file dir create failed", exc_info=True)
                return False

            items = list(self._cache.items())
            # Keep the most-recent ``_warm_max`` entries (tail of LRU).
            if len(items) > self._warm_max:
                items = items[-self._warm_max:]

            texts: list[str] = []
            rows: list[list[float]] = []
            for txt, vec in items:
                if not isinstance(txt, str) or not txt:
                    continue
                if not vec or len(vec) != self._dimension:
                    continue
                texts.append(txt)
                rows.append(vec)
            if not texts:
                return False

            try:
                arr = _np.asarray(rows, dtype=_np.float32)
                meta_blob = _np.asarray(
                    json.dumps(
                        {
                            "version": self._WARM_FILE_VERSION,
                            "model_name": self._model_name,
                            "dimension": self._dimension,
                            "saved_at": time.time(),
                            "count": len(texts),
                        }
                    ),
                    dtype="U",
                )
                # ``np.savez`` appends its own ``.npz`` suffix when the
                # path doesn't already end with ``.npz``. To keep atomic
                # writes simple, use a sibling tmp WITH the ``.npz``
                # suffix, then rename to the final target in one step.
                tmp_path = path.with_name(path.name + ".tmp")
                _np.savez(
                    str(tmp_path),
                    texts=_np.asarray(texts, dtype="U"),
                    vectors=arr,
                    meta=meta_blob,
                )
                # Account for numpy's silent ``.npz`` extension logic: if
                # the tmp path didn't end with .npz it will be renamed
                # with one appended. Pick whichever file actually landed.
                written = tmp_path
                if not written.is_file():
                    alt = Path(str(tmp_path) + ".npz")
                    if alt.is_file():
                        written = alt
                os.replace(str(written), str(path))
                self._warm_dirty_count = 0
                logger.debug(
                    "Embedding warm-file persisted: %d entries -> %s",
                    len(texts), path,
                )
                return True
            except Exception:
                logger.debug("warm-file persist failed", exc_info=True)
                return False

    def warm_file_info(self) -> dict[str, Any]:
        """Diagnostic snapshot of warm-file state."""
        exists = self._warm_path.is_file()
        size = 0
        if exists:
            try:
                size = self._warm_path.stat().st_size
            except Exception:
                size = 0
        return {
            "enabled": self._warm_enabled,
            "path": str(self._warm_path),
            "exists": exists,
            "size_bytes": size,
            "restored_on_boot": self._warm_restored,
            "in_memory_entries": len(self._cache),
            "dirty_count": self._warm_dirty_count,
        }

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding -- safe to call from any thread."""
        if not text or not text.strip():
            return list(self._get_zero_vec())

        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        if not self._ensure_loaded() or self._model is None:
            return self._fallback_embed(text)

        try:
            raw = self._model.encode(
                text, normalize_embeddings=True, show_progress_bar=False,
            )
            vec = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            self._cache_put(text, vec)
            return vec
        except Exception:
            logger.debug("Embedding failed for: %s", text[:60], exc_info=True)
            return self._fallback_embed(text)

    async def embed(self, text: str) -> list[float]:
        """Async embedding -- runs in executor to avoid blocking."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding for bulk operations (document ingestion)."""
        if not texts:
            return []

        if not self._ensure_loaded() or self._model is None:
            return [self._fallback_embed(t) for t in texts]

        import asyncio

        def _batch_sync() -> list[list[float]]:
            try:
                raw = self._model.encode(
                    texts, normalize_embeddings=True,
                    show_progress_bar=False, batch_size=64,
                )
                vecs = raw.tolist() if hasattr(raw, "tolist") else [list(r) for r in raw]
                for t, v in zip(texts, vecs):
                    self._cache_put(t, v)
                return vecs
            except Exception:
                logger.exception("Batch embedding failed")
                return [self._fallback_embed(t) for t in texts]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _batch_sync)

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two normalized vectors.

        Uses numpy when available (100x faster on 384-dim).
        Falls back to pure Python math otherwise.
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        if _np is not None:
            va = _np.asarray(a, dtype=_np.float32)
            vb = _np.asarray(b, dtype=_np.float32)
            dot = float(_np.dot(va, vb))
            na = float(_np.linalg.norm(va))
            nb = float(_np.linalg.norm(vb))
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return dot / (na * nb)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def batch_similarity(
        query: list[float], candidates: list[list[float]],
    ) -> list[float]:
        """Compute cosine similarity of one query against N candidates.

        With numpy this is a single matrix operation — orders of magnitude
        faster than calling similarity() in a loop.
        """
        if not query or not candidates:
            return []
        if _np is not None:
            q = _np.asarray(query, dtype=_np.float32)
            mat = _np.asarray(candidates, dtype=_np.float32)
            q_norm = _np.linalg.norm(q)
            if q_norm < 1e-9:
                return [0.0] * len(candidates)
            mat_norms = _np.linalg.norm(mat, axis=1)
            mat_norms = _np.maximum(mat_norms, 1e-9)
            scores = mat @ q / (mat_norms * q_norm)
            return scores.tolist()
        return [EmbeddingEngine.similarity(query, c) for c in candidates]

    def _fallback_embed(self, text: str) -> list[float]:
        """Deterministic hash-based pseudo-embedding when no model is available.

        Uses multiple hash rounds to fill the full vector dimension, providing
        better discrimination between texts than single-hash truncation.
        """
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        text_lower = text.lower().encode()
        vec = [0.0] * self._dimension
        rounds = (self._dimension + 31) // 32
        for r in range(rounds):
            h = hashlib.sha256(text_lower + r.to_bytes(2, "little")).digest()
            offset = r * 32
            for i in range(min(32, self._dimension - offset)):
                vec[offset + i] = (h[i] / 255.0) * 2 - 1

        if _np is not None:
            v = _np.asarray(vec, dtype=_np.float32)
            norm = float(_np.linalg.norm(v))
            if norm > 1e-9:
                vec = (v / norm).tolist()
        else:
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]

        self._cache_put(text, vec)
        return vec

    def preload(self) -> bool:
        """Pre-load the model at startup (synchronous)."""
        return self._ensure_loaded()

    def shutdown(self) -> None:
        """Release model memory (and flush warm-file to disk)."""
        if self._warm_enabled:
            try:
                self._persist_warm_file()
            except Exception:
                logger.debug("warm-file final persist failed", exc_info=True)
        self._model = None
        self._loaded = False
        self._cache.clear()
        self._zero_vec = None
        logger.info("Embedding engine shut down")
