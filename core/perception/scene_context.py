"""
ATOM -- Scene Context (Phase G3).

The presence sampler tells us *whether* a face is in front of the
camera. The scene context tells us *what is going on around it* --
"the user is looking at a code editor", "the iPhone is on the desk
next to a notebook", etc.

Running a vision-language model on every frame is too expensive (and
too noisy for the suggester). This module gates the VLM call behind
two protections:

1. **Significance**: a snapshot only triggers a fresh caption when
   it's meaningfully different from the last captioned snapshot
   (face-count delta, presence flip, quality jump).
2. **Rate-limit**: at most one VLM call every ``cooldown_s`` (default
   300s = 5 min) regardless of significance.

When both gates open, we capture one fresh JPEG and hand it to a
``VLMCaptioner``-shaped object via DI. The result is emitted as
``scene.context``.

Both gates fail-open: if the captioner is missing, if the camera
won't capture, or if any step raises, we log and stay silent. The
suggester treats absence of ``scene.context`` as "no scene info" and
still works fine.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.perception.scene_context")


# ── data class ─────────────────────────────────────────────────────


@dataclass(slots=True)
class SceneContext:
    ts: float
    caption: str
    trigger: str  # "presence_change" | "first" | "manual"
    face_count: int
    quality: str
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "caption": self.caption,
            "trigger": self.trigger,
            "face_count": self.face_count,
            "quality": self.quality,
            "elapsed_ms": self.elapsed_ms,
        }


# ── presence-snapshot DTO (loose duck typing) ──────────────────────


@dataclass(slots=True)
class _Snapshot:
    present: bool = False
    face_count: int = 0
    quality: str = ""
    ts: float = 0.0


def _coerce_snapshot(payload: dict[str, Any]) -> _Snapshot:
    return _Snapshot(
        present=bool(payload.get("present", False)),
        face_count=int(payload.get("face_count", 0) or 0),
        quality=str(payload.get("quality", "") or ""),
        ts=float(payload.get("ts", time.time())),
    )


# ── scene context engine ───────────────────────────────────────────


class SceneContextEngine:
    """Listens to ``presence.snapshot`` and emits ``scene.context``.

    Parameters mirror :class:`PresenceSampler` for symmetry. The VLM
    captioner is injected so we can stub it in tests without dragging
    in mlx-vlm.
    """

    __slots__ = (
        "_bus", "_cooldown_s", "_significance_min_seconds",
        "_captioner", "_capture_fn", "_camera_chooser",
        "_busy_provider", "_last_caption_at",
        "_last_snapshot", "_last_caption_snapshot",
        "_last_caption_text", "_attached", "_in_flight",
        "_total_attempts", "_total_emits", "_total_skips_cooldown",
        "_total_skips_no_change", "_total_errors",
    )

    def __init__(
        self,
        bus: "AsyncEventBus",
        captioner: Any,
        *,
        cooldown_s: float = 300.0,
        significance_min_seconds: float = 30.0,
        capture_fn: Callable[..., Any] | None = None,
        camera_chooser: Callable[[], Any] | None = None,
        busy_provider: Callable[[], bool] | None = None,
    ) -> None:
        self._bus = bus
        self._captioner = captioner
        self._cooldown_s = float(cooldown_s)
        self._significance_min_seconds = float(significance_min_seconds)
        self._capture_fn = capture_fn
        self._camera_chooser = camera_chooser
        self._busy_provider = busy_provider
        self._last_caption_at = 0.0
        self._last_snapshot: _Snapshot | None = None
        self._last_caption_snapshot: _Snapshot | None = None
        self._last_caption_text: str = ""
        self._attached = False
        self._in_flight = False
        self._total_attempts = 0
        self._total_emits = 0
        self._total_skips_cooldown = 0
        self._total_skips_no_change = 0
        self._total_errors = 0

    # ── lifecycle ───────────────────────────────────────────────

    def attach(self) -> None:
        if self._attached:
            return
        self._bus.on("presence.snapshot", self._on_presence)
        self._attached = True
        logger.info(
            "SceneContextEngine attached (cooldown=%.0fs, change_min=%.0fs)",
            self._cooldown_s, self._significance_min_seconds,
        )

    def detach(self) -> None:
        if not self._attached:
            return
        self._bus.off("presence.snapshot", self._on_presence)
        self._attached = False

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "attached": self._attached,
            "cooldown_s": self._cooldown_s,
            "attempts": self._total_attempts,
            "emits": self._total_emits,
            "skips_cooldown": self._total_skips_cooldown,
            "skips_no_change": self._total_skips_no_change,
            "errors": self._total_errors,
            "last_caption": self._last_caption_text,
            "last_caption_age_s": (
                round(time.monotonic() - self._last_caption_at, 1)
                if self._last_caption_at else None
            ),
        }

    # ── presence handler ────────────────────────────────────────

    async def _on_presence(self, **payload: Any) -> None:
        snapshot = _coerce_snapshot(payload)
        prior = self._last_snapshot
        self._last_snapshot = snapshot

        trigger = self._eval_trigger(prior, snapshot)
        if trigger is None:
            self._total_skips_no_change += 1
            return

        if not self._cooldown_passed():
            self._total_skips_cooldown += 1
            return

        if self._busy_provider is not None:
            try:
                if bool(self._busy_provider()):
                    return
            except Exception:
                pass

        if self._in_flight:
            return

        self._in_flight = True
        try:
            await self._caption_now(trigger, snapshot)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:
            logger.exception("scene context: unexpected error")
            self._total_errors += 1
        finally:
            self._in_flight = False

    # ── significance ────────────────────────────────────────────

    def _eval_trigger(
        self, prior: _Snapshot | None, current: _Snapshot,
    ) -> str | None:
        """Return a string trigger name when this snapshot warrants
        a fresh caption, or ``None`` to skip silently."""
        if not current.present and current.face_count == 0:
            # Nothing in front of the camera -- skip silently.
            return None
        if self._last_caption_text == "" and self._last_caption_at == 0.0:
            return "first"
        if prior is None:
            return "first"
        if prior.present != current.present:
            return "presence_change"
        if prior.face_count != current.face_count:
            return "face_count_change"
        if prior.quality != current.quality:
            return "quality_change"
        # Same presence + count + quality. Only re-caption if we have
        # not done so for a *long* time.
        seconds_since_caption = (
            time.monotonic() - self._last_caption_at
        ) if self._last_caption_at else float("inf")
        if seconds_since_caption >= self._cooldown_s:
            return "stale"
        return None

    def _cooldown_passed(self) -> bool:
        if self._last_caption_at == 0.0:
            return True
        return (time.monotonic() - self._last_caption_at) >= max(
            self._cooldown_s, self._significance_min_seconds,
        )

    # ── caption pass ────────────────────────────────────────────

    async def _caption_now(self, trigger: str, snapshot: _Snapshot) -> None:
        self._total_attempts += 1
        loop = asyncio.get_running_loop()
        scene = await loop.run_in_executor(
            None, self._caption_blocking, trigger, snapshot,
        )
        if scene is None:
            return
        self._last_caption_text = scene.caption
        self._last_caption_at = time.monotonic()
        self._last_caption_snapshot = snapshot
        self._total_emits += 1
        try:
            self._bus.emit_long("scene.context", **scene.as_dict())
        except Exception:
            logger.exception("scene.context emit failed")

    def _caption_blocking(
        self, trigger: str, snapshot: _Snapshot,
    ) -> SceneContext | None:
        start = time.monotonic()
        cap_module = self._capture_module()
        if cap_module is None:
            self._total_errors += 1
            return None

        camera = self._select_camera(cap_module)
        if camera is None:
            self._total_errors += 1
            return None

        out_path = Path(tempfile.gettempdir()) / "atom_scene.jpg"
        result = cap_module.capture_jpeg(camera, out_path=out_path, timeout_s=2.5)
        if not getattr(result, "ok", False):
            self._total_errors += 1
            return None

        jpeg_path = getattr(result, "saved_path", str(out_path))
        try:
            caption = self._captioner.describe(jpeg_path) if self._captioner else ""
        except Exception:
            logger.exception("VLM describe raised")
            self._total_errors += 1
            return None

        caption = (caption or "").strip()
        if not caption:
            return None

        elapsed_ms = (time.monotonic() - start) * 1000.0
        return SceneContext(
            ts=time.time(),
            caption=caption[:240],
            trigger=trigger,
            face_count=snapshot.face_count,
            quality=snapshot.quality,
            elapsed_ms=round(elapsed_ms, 1),
        )

    # ── helpers ─────────────────────────────────────────────────

    def _capture_module(self) -> Any:
        if self._capture_fn is not None:
            return self._capture_fn
        try:
            from core.perception import camera_capture
            return camera_capture
        except Exception:
            logger.warning("scene context: camera_capture unavailable")
            return None

    def _select_camera(self, cap_module: Any) -> Any:
        chooser = self._camera_chooser
        if chooser is not None:
            try:
                return chooser()
            except Exception:
                logger.exception("camera_chooser failed")
        # Mirror :class:`PresenceSampler._select_camera` -- accept
        # either ``list_cameras`` (canonical) or ``discover_cameras``
        # (legacy alias) so a stub harness doesn't silently break the
        # scene-context capture path.
        discover = getattr(cap_module, "list_cameras", None)
        if discover is None:
            discover = getattr(cap_module, "discover_cameras", None)
        if discover is None:
            return None
        try:
            cams = discover()
        except Exception:
            return None
        try:
            return cap_module.choose_preferred(cams, preferred="builtin")
        except Exception:
            return None

    # ── public hooks ───────────────────────────────────────────

    async def caption_now(self, trigger: str = "manual") -> SceneContext | None:
        """Force one caption pass (for tests + voice command)."""
        snapshot = self._last_snapshot or _Snapshot(present=True, face_count=1)
        loop = asyncio.get_running_loop()
        scene = await loop.run_in_executor(
            None, self._caption_blocking, trigger, snapshot,
        )
        if scene is not None:
            self._last_caption_text = scene.caption
            self._last_caption_at = time.monotonic()
            self._total_emits += 1
            try:
                self._bus.emit_long("scene.context", **scene.as_dict())
            except Exception:
                logger.exception("scene.context emit failed")
        return scene


__all__ = ["SceneContext", "SceneContextEngine"]
