"""
ATOM -- Background Task Manager & Proactive Agent.

Manages long-running background tasks (downloads, scans, research)
separately from the foreground voice pipeline. Reports completion
via bus events and announces results when the user is idle.

Also acts as a proactive daemon: monitors system state and emits
notifications for critical conditions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.background")

_MAX_CONCURRENT_TASKS = 5
_DAEMON_INTERVAL_S = 60.0
_TASK_TIMEOUT_S = 300.0


@dataclass
class BackgroundJob:
    """A tracked background task."""
    id: str
    name: str
    task: asyncio.Task | None = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    result: str = ""
    status: str = "pending"
    announce_on_complete: bool = True


class BackgroundTaskManager:
    """Manages async background jobs outside the foreground voice lock."""

    def __init__(self, bus: AsyncEventBus | None = None) -> None:
        self._bus = bus
        self._jobs: dict[str, BackgroundJob] = {}
        self._completed: list[BackgroundJob] = []
        self._job_counter = 0
        self._running = False
        self._daemon_task: asyncio.Task | None = None
        self._state_graph: Any = None
        self._tts: Any = None

    def wire(
        self,
        state_graph: Any = None,
        tts: Any = None,
    ) -> None:
        self._state_graph = state_graph
        self._tts = tts

    def submit(
        self,
        name: str,
        coro_factory: Callable[[], Coroutine],
        *,
        announce: bool = True,
    ) -> str:
        """Submit a background task. Returns the job ID."""
        self._job_counter += 1
        job_id = f"bg_{self._job_counter}"

        async def _run_job() -> None:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "running"
            try:
                result = await asyncio.wait_for(
                    coro_factory(), timeout=_TASK_TIMEOUT_S,
                )
                job.result = str(result) if result else "Done"
                job.status = "completed"
            except asyncio.TimeoutError:
                job.result = "Timed out"
                job.status = "timeout"
            except asyncio.CancelledError:
                job.status = "cancelled"
                return
            except Exception as exc:
                job.result = str(exc)[:200]
                job.status = "failed"
                logger.warning("Background job '%s' failed: %s", name, exc)
            finally:
                job.completed_at = time.time()
                self._completed.append(job)
                if len(self._completed) > 50:
                    self._completed = self._completed[-50:]

                if self._bus is not None:
                    self._bus.emit_fast(
                        "background_task_complete",
                        job_id=job_id,
                        name=name,
                        status=job.status,
                        result=job.result[:200],
                    )

                if job.announce_on_complete and job.status == "completed":
                    self._announce_completion(job)

        if len(self._jobs) >= _MAX_CONCURRENT_TASKS:
            oldest = min(self._jobs.values(), key=lambda j: j.submitted_at)
            if oldest.task and not oldest.task.done():
                oldest.task.cancel()
            del self._jobs[oldest.id]

        job = BackgroundJob(
            id=job_id, name=name, announce_on_complete=announce,
        )
        job.task = asyncio.create_task(_run_job(), name=f"bg_{name}")
        self._jobs[job_id] = job

        logger.info("Background task submitted: %s (id=%s)", name, job_id)
        return job_id

    def _announce_completion(self, job: BackgroundJob) -> None:
        """Announce task completion via TTS if available."""
        if self._bus is not None:
            msg = f"Boss, {job.name} is done."
            if job.result and job.result != "Done":
                msg += f" {job.result[:100]}"
            self._bus.emit_fast("response_ready", text=msg)

    def cancel(self, job_id: str) -> bool:
        """Cancel a running background task."""
        job = self._jobs.get(job_id)
        if not job or not job.task:
            return False
        job.task.cancel()
        job.status = "cancelled"
        return True

    def get_active(self) -> list[dict[str, Any]]:
        """List active background tasks."""
        return [
            {
                "id": j.id,
                "name": j.name,
                "status": j.status,
                "elapsed_s": time.time() - j.submitted_at,
            }
            for j in self._jobs.values()
            if j.status in ("pending", "running")
        ]

    def get_completed(self, n: int = 5) -> list[dict[str, Any]]:
        """List recently completed tasks."""
        return [
            {
                "id": j.id,
                "name": j.name,
                "status": j.status,
                "result": j.result[:100],
                "elapsed_s": j.completed_at - j.submitted_at,
            }
            for j in self._completed[-n:]
        ]

    # ── Proactive Daemon ─────────────────────────────────────────────

    def start(self) -> None:
        """Start the background daemon loop."""
        if self._running:
            return
        self._running = True
        self._daemon_task = asyncio.create_task(
            self._daemon_loop(), name="atom_background_daemon",
        )
        logger.info("BackgroundTaskManager started")

    def stop(self) -> None:
        self._running = False
        if self._daemon_task and not self._daemon_task.done():
            self._daemon_task.cancel()
        for job in self._jobs.values():
            if job.task and not job.task.done():
                job.task.cancel()

    async def _daemon_loop(self) -> None:
        """Periodic background checks for system health."""
        await asyncio.sleep(30.0)
        while self._running:
            try:
                self._cleanup_finished_jobs()
                self._check_system_health()
                await asyncio.sleep(_DAEMON_INTERVAL_S)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Background daemon error", exc_info=True)
                await asyncio.sleep(_DAEMON_INTERVAL_S)

    def _cleanup_finished_jobs(self) -> None:
        """Remove completed/failed jobs from the active map."""
        done_ids = [
            jid for jid, j in self._jobs.items()
            if j.status not in ("pending", "running")
        ]
        for jid in done_ids:
            del self._jobs[jid]

    def _check_system_health(self) -> None:
        """Monitor for critical system conditions."""
        if self._state_graph is None:
            return
        load = getattr(self._state_graph, "system_load", 0)
        if load > 90.0 and self._bus is not None:
            logger.warning("High system load detected: %.0f%%", load)
            self._bus.emit_fast(
                "system_alert",
                alert_type="high_load",
                load=load,
            )

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "active_jobs": len(self.get_active()),
            "total_completed": len(self._completed),
            "running": self._running,
        }


# Backward-compatible alias
ProactiveDaemon = BackgroundTaskManager

__all__ = ["BackgroundTaskManager", "ProactiveDaemon"]
