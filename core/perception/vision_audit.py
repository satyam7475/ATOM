"""ATOM -- Append-only camera-capture audit log.

Every time ATOM opens the camera (boot face check or on-demand
``vision_look``), we drop a single JSONL record so the owner can answer
the basic privacy question "when did ATOM look, why, and what did it
see?" without having to grep through a verbose runtime log.

Records are write-only from the runtime's perspective; nothing in
ATOM ever reads this file at decision time. Inspect with::

    tail -f logs/atom_vision_audit.jsonl

Schema (all keys present in every record)::

    ts            UTC ISO-8601 timestamp
    reason        free-form short string ("boot_face_check",
                  "vision_look", "manual_test", …)
    source        camera identifier (uniqueID / localizedName)
    source_kind   "builtin" | "continuity" | "external" | "unknown"
    capture_ms    wall time spent inside the AVFoundation capture
    detection_ms  wall time spent inside Apple Vision
    faces         number of face rectangles detected
    summary       short description string emitted by the engine
    saved_path    file path the JPEG was written to (or "")
    error         error string, "" on success

The file is created on first write with mode 0o600 so only the owner
can read it. Each line is a self-contained JSON object — easy to grep,
easy to truncate.

Owner: Satyam
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.vision.audit")


_DEFAULT_PATH = "logs/atom_vision_audit.jsonl"


class VisionAuditLog:
    """Append-only JSONL audit sink for camera captures."""

    __slots__ = ("_path", "_lock")

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or _DEFAULT_PATH)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        *,
        reason: str,
        source: str = "",
        source_kind: str = "unknown",
        capture_ms: float = 0.0,
        detection_ms: float = 0.0,
        faces: int = 0,
        summary: str = "",
        saved_path: str = "",
        error: str = "",
    ) -> None:
        """Append a single capture record. Never raises."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": str(reason or ""),
            "source": str(source or ""),
            "source_kind": str(source_kind or "unknown"),
            "capture_ms": round(float(capture_ms), 1),
            "detection_ms": round(float(detection_ms), 1),
            "faces": int(faces),
            "summary": str(summary or "")[:240],
            "saved_path": str(saved_path or ""),
            "error": str(error or "")[:240],
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                first_write = not self._path.exists()
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                if first_write:
                    try:
                        os.chmod(self._path, 0o600)
                    except OSError:
                        logger.debug("chmod 0o600 failed for %s", self._path)
        except Exception:
            logger.debug("vision audit write failed", exc_info=True)
