"""
ATOM V6.5 -- Bounded thread pool and async background worker for heavy tasks.

Limits CPU worker threads (default 4) and batches embedding-friendly work.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("atom.resource_pool")

T = TypeVar("T")

_DEFAULT_MAX_WORKERS = 4


def max_worker_threads(config: Optional[dict] = None) -> int:
    perf = (config or {}).get("performance", {}) if config else {}
    raw = perf.get("max_worker_threads", os.environ.get("ATOM_MAX_WORKERS"))
    if raw is None:
        return _DEFAULT_MAX_WORKERS
    try:
        return max(1, min(32, int(raw)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_WORKERS


_executor: ThreadPoolExecutor | None = None
_executor_n: int = 0


def get_io_executor(config: Optional[dict] = None) -> ThreadPoolExecutor:
    """Process-wide bounded executor for blocking / CPU-heavy chunks."""
    global _executor, _executor_n
    n = max_worker_threads(config)
    if _executor is None or n != _executor_n:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="atom_v65")
        _executor_n = n
        logger.info("ThreadPoolExecutor max_workers=%d", n)
    return _executor


async def run_in_worker(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    ex = get_io_executor()
    return await loop.run_in_executor(ex, lambda: fn(*args, **kwargs))


class BackgroundTaskQueue:
    """Single-worker asyncio queue for sequential background jobs (e.g. indexing)."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[Callable[[], Any]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="atom_bg_worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = await asyncio.wait_for(self._q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                job()
            except Exception:
                logger.debug("background job failed", exc_info=True)
            finally:
                self._q.task_done()

    async def submit(self, job: Callable[[], Any]) -> None:
        await self._q.put(job)


__all__ = [
    "max_worker_threads",
    "get_io_executor",
    "run_in_worker",
    "BackgroundTaskQueue",
]
