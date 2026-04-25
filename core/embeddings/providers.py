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
