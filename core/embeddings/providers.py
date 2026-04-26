"""Pluggable embedding providers.

The public ATOM contract stays in ``core.embedding_engine``. Provider
classes here keep heavyweight runtime choices isolated so we can move
production away from PyTorch without changing memory/vector callers.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover - optional acceleration
    _np = None  # type: ignore[assignment]


class ProviderLoadError(RuntimeError):
    """Raised when an optional embedding provider cannot be loaded."""


class EmbeddingProvider:
    """Small sync provider contract used by EmbeddingEngine."""

    name = "base"
    version = "1"

    def __init__(self, *, model_name: str, dimension: int, **_: Any) -> None:
        self.model_name = model_name
        self.dimension = int(dimension)

    def load(self) -> bool:
        raise NotImplementedError

    def encode(self, text: str) -> list[float]:
        batch = self.encode_batch([text])
        return batch[0] if batch else []

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "provider_version": self.version,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }


class FastEmbedProvider(EmbeddingProvider):
    """ONNX-backed FastEmbed provider.

    FastEmbed's common small English models are 384-dimensional, matching
    ATOM's current MiniLM vector width while avoiding the PyTorch/MPS boot
    stack. Callers still validate the provider signature before reusing
    persisted vectors.
    """

    name = "fastembed"
    version = "1"

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        cache_dir: str | None = None,
        threads: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Any = None

    def load(self) -> bool:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ProviderLoadError("fastembed is not installed") from exc

        kwargs: dict[str, Any] = {"model_name": self.model_name}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        if self._threads:
            kwargs["threads"] = int(self._threads)
        self._model = TextEmbedding(**kwargs)
        return True

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise ProviderLoadError("fastembed provider is not loaded")
        vectors = list(self._model.embed(texts))
        out: list[list[float]] = []
        for vec in vectors:
            values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            out.append(_normalise(values))
        if out:
            self.dimension = len(out[0])
        return out

    def shutdown(self) -> None:
        self._model = None


class MLXEmbeddingsProvider(EmbeddingProvider):
    """``mlx-embeddings``-backed provider (Sprint P3.4, Apr 26 2026).

    Uses Apple's `mlx-embeddings` package to run sentence-embedding models
    natively on the Apple Neural Engine / unified memory. On M-series Macs
    this is materially faster than torch-MPS for 384-dim models and avoids
    the PyTorch boot stack entirely. We keep `sentence_transformers` and
    `fastembed` providers around as fallbacks; the engine picks via
    ``embedding.backend`` in config.

    The wire format is normalised float32 vectors -- identical to what
    SentenceTransformer / FastEmbed return -- so the provider is a drop-in
    on the EmbeddingEngine side and the warm-file format does not change.
    """

    name = "mlx_embeddings"
    version = "1"

    _DEFAULT_MODEL = "mlx-community/all-MiniLM-L6-v2-mlx"

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        max_length: int = 512,
        **_: Any,
    ) -> None:
        super().__init__(
            model_name=model_name or self._DEFAULT_MODEL,
            dimension=dimension,
        )
        self._max_length = int(max_length)
        self._model: Any = None
        self._tokenizer: Any = None
        self._mx: Any = None  # mlx.core, lazy-imported

    def load(self) -> bool:
        try:
            import mlx.core as _mx  # type: ignore[import-untyped]
            from mlx_embeddings.utils import load as _ml_load  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ProviderLoadError(
                "mlx-embeddings is not installed (pip install mlx-embeddings)",
            ) from exc

        # `mlx_embeddings.utils.load` returns (model, tokenizer). The model
        # exposes `__call__(input_ids=..., attention_mask=...)` and a
        # `.text_embeds` field that is the pooled, L2-normalised vector.
        try:
            model, tokenizer = _ml_load(self.model_name)
        except Exception as exc:  # pragma: no cover -- env-specific
            raise ProviderLoadError(
                f"mlx-embeddings load failed for {self.model_name!r}: {exc}",
            ) from exc

        self._model = model
        self._tokenizer = tokenizer
        self._mx = _mx
        # First inference materialises the model on-device and lets us
        # discover the actual embedding width (mlx-embeddings does not
        # expose `.dim` consistently across model families).
        try:
            sample = self._encode_one("dimension probe")
            if sample:
                self.dimension = len(sample)
        except Exception:
            # Probe failure is non-fatal -- caller will retry on real
            # input and surface the real exception there.
            pass
        return True

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise ProviderLoadError("mlx-embeddings provider is not loaded")
        if not texts:
            return []
        out: list[list[float]] = []
        # mlx-embeddings supports batched encode but falling back to a
        # per-text loop is robust against models that don't pad-to-max
        # uniformly. The expected ATOM batch sizes are tiny (<32).
        for text in texts:
            vec = self._encode_one(text)
            out.append(_normalise(vec))
        if out:
            self.dimension = len(out[0])
        return out

    def _encode_one(self, text: str) -> list[float]:
        if self._model is None or self._tokenizer is None or self._mx is None:
            return []
        enc = self._tokenizer.encode(
            text,
            return_tensors="mlx",
            max_length=self._max_length,
            truncation=True,
        )
        if isinstance(enc, dict):
            input_ids = enc.get("input_ids")
            attention_mask = enc.get("attention_mask")
        else:
            input_ids = enc
            attention_mask = None
        result = self._model(
            input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = getattr(result, "text_embeds", None)
        if text_embeds is None and isinstance(result, (tuple, list)):
            text_embeds = result[0]
        if text_embeds is None:
            return []
        # text_embeds is shape (batch, dim). For a single input we want
        # the first row as a Python list.
        try:
            arr = text_embeds[0]
        except Exception:
            arr = text_embeds
        try:
            return [float(x) for x in arr.tolist()]
        except Exception:
            try:
                return [float(x) for x in list(arr)]
            except Exception:
                return []

    def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        self._mx = None


def _normalise(vec: list[float]) -> list[float]:
    if not vec:
        return []
    if _np is not None:
        arr = _np.asarray(vec, dtype=_np.float32)
        norm = float(_np.linalg.norm(arr))
        if norm > 1e-9:
            return (arr / norm).tolist()
        return arr.tolist()
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-9:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]
