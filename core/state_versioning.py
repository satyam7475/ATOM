"""
ATOM V6.5 -- Versioned context snapshots with async critical sections.

Pairs monotonic state_version with immutable deep copies for deterministic
read paths. Use ``async with versioned.update_lock()`` for writes that must
not race with snapshot readers.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, TypeVar

logger = logging.getLogger("atom.state_versioning")

T = TypeVar("T", bound=Dict[str, Any])


@dataclass
class VersionedSnapshot(Generic[T]):
    """Immutable point-in-time view of versioned state."""

    version: int
    data: T
    created_monotonic: float


class VersionedState(Generic[T]):
    """
    Holds mutable state with a monotonic version counter.

    - ``bump()`` increments version after logical commits.
    - ``snapshot()`` returns deepcopy(data) + version (no lock required for
      single-threaded read of CPython dict refs; use lock for strict consistency).
    - ``update_lock`` guards concurrent async writers.
    """

    __slots__ = ("_data", "_version", "_lock")

    def __init__(self, initial: T) -> None:
        self._data: T = initial
        self._version: int = 1
        self._lock = asyncio.Lock()

    @property
    def version(self) -> int:
        return self._version

    def get_data(self) -> T:
        return self._data

    @property
    def update_lock(self) -> asyncio.Lock:
        return self._lock

    def snapshot(self) -> VersionedSnapshot[T]:
        """Deep copy of current data for read-only consumers."""
        import time

        return VersionedSnapshot(
            version=self._version,
            data=copy.deepcopy(self._data),
            created_monotonic=time.monotonic(),
        )

    async def apply(self, mutator) -> VersionedSnapshot[T]:
        """Run *mutator(data)* under the async lock; bump version after."""

        async with self._lock:
            mutator(self._data)
            self._version += 1
            logger.debug("state_version -> %s", self._version)
            return self.snapshot()

    def bump(self) -> int:
        """Increment version without copying (caller holds lock)."""
        self._version += 1
        return self._version


__all__ = ["VersionedSnapshot", "VersionedState"]
