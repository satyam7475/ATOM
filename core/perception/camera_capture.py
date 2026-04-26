"""ATOM -- AVFoundation single-frame camera capture.

Wraps the *minimum* AVFoundation surface needed to ask "give me one
JPEG from a camera". Intentionally avoids becoming a video pipeline:
a single still frame is enough for the boot face check and the
``vision_look`` tool, and a one-shot capture is much friendlier than
a long-lived ``AVCaptureSession`` we'd have to babysit.

Camera policy (Apr 26 2026)
---------------------------
ATOM is **MacBook-camera only**. Apple's "Continuity Camera" feature
(iPhone-as-webcam) was removed by owner request — vision frames must
come from the laptop's built-in FaceTime HD camera, never from a
paired iPhone. The discovery session therefore asks AVFoundation for:

* ``AVCaptureDeviceTypeBuiltInWideAngleCamera`` — the laptop webcam
  (i.e. the FaceTime HD on the MacBook Air M5). **Canonical source.**
* ``AVCaptureDeviceTypeExternal`` / ``…ExternalUnknown`` — USB / DSLR
  rigs if the user has them attached. iPhone-shaped externals
  (``modelID`` starting with ``iPhone…`` or ``localizedName`` containing
  ``iPhone``) are filtered out by :func:`list_cameras` so they can
  never become the active source.

We deliberately do NOT ask AVFoundation for
``AVCaptureDeviceTypeContinuityCamera`` and we do NOT classify any
device as the ``continuity`` kind. macOS may still surface a paired
iPhone under the ``External`` device type — those rows are dropped
during discovery.

Output choice (live-fix Apr 2026)
---------------------------------
We use ``AVCaptureVideoDataOutput`` rather than ``AVCapturePhotoOutput``.
The latter raises ``NSKVONotifying_AVCapturePhotoOutput' not linked``
+ ``AVFoundationErrorDomain Code=-11800`` from pyobjc on macOS 15+
because the dynamic KVO subclass isn't loaded into the embedded Python
runtime. ``AVCaptureVideoDataOutput`` doesn't rely on KVO and works
for built-in webcams (and would for Continuity, which we don't use).
We grab exactly one ``CMSampleBuffer``, JPEG-encode it via
``CIContext``, and tear down the session.

The "preferred" device is chosen by :func:`choose_preferred` based on
the ``vision.preferred_camera`` config knob:

* ``"builtin"`` (default) — the first built-in webcam (FaceTime HD).
* ``"auto"`` — alias of ``"builtin"`` after the iPhone-camera removal.
* ``"continuity"`` — **deprecated**; treated as an alias of
  ``"builtin"`` and logged once at startup so old configs don't break.
* anything else — fall back to ``"builtin"``.

Failure posture
---------------
Every public function returns ``CaptureResult`` and never raises.
Missing pyobjc bindings, denied TCC, or zero cameras all degrade
silently with ``ok=False`` plus a short ``error`` string. The
runtime is expected to log this and continue without vision rather
than crash-loop.

Owner: Satyam
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("atom.perception.camera")


# ── Optional pyobjc imports ───────────────────────────────────────────
# We deliberately catch ``ImportError`` instead of ``BaseException`` so
# a broken AVFoundation install (e.g. someone vendored an arm64-only
# wheel onto an x86 box) still raises and gets debugged. The runtime
# checks ``HAS_AVFOUNDATION`` before calling anything below.
HAS_AVFOUNDATION = False
_AVF: Any = None
_NF: Any = None
_objc: Any = None
_CoreMedia: Any = None
_Quartz: Any = None
# ``CGColorSpaceCreateDeviceRGB`` lives in CoreGraphics. PyObjC's
# ``Quartz`` umbrella *usually* re-exports it, but on some installs (and
# on the embedded Python interpreter inside ATOM.app) the symbol is
# missing on the Quartz module and only resolvable via
# ``Quartz.CoreGraphics`` or the dedicated ``CoreGraphics`` package. We
# resolve it once at import time and fail-soft to ``None`` so the video
# delegate can pick a sensible default colour space (or skip JPEG
# encoding entirely) instead of crashing the boot face check.
_CGColorSpaceCreateDeviceRGB: Any = None
try:
    import AVFoundation as _AVF  # type: ignore[import-untyped]
    import CoreMedia as _CoreMedia  # type: ignore[import-untyped]
    import Foundation as _NF  # type: ignore[import-untyped]
    import objc as _objc  # type: ignore[import-untyped]
    import Quartz as _Quartz  # type: ignore[import-untyped]
    HAS_AVFOUNDATION = True
except ImportError:
    pass


def _resolve_cg_color_space_create_device_rgb() -> Any:
    """Find ``CGColorSpaceCreateDeviceRGB`` across PyObjC layouts.

    Returns the callable or ``None`` if no source resolves it. Tries (in
    order): the Quartz umbrella, ``Quartz.CoreGraphics``, and the
    standalone ``CoreGraphics`` module. Each step is wrapped in a
    try/except so a partial PyObjC install (e.g. CoreGraphics shipped
    without the Quartz umbrella, or vice versa) still produces a
    working symbol when at least one source has it.
    """
    candidates: list[Any] = []
    if _Quartz is not None:
        candidates.append(_Quartz)
        sub = getattr(_Quartz, "CoreGraphics", None)
        if sub is not None:
            candidates.append(sub)
    try:
        import CoreGraphics as _CG  # type: ignore[import-untyped]
        candidates.append(_CG)
    except ImportError:
        pass
    for src in candidates:
        fn = getattr(src, "CGColorSpaceCreateDeviceRGB", None)
        if callable(fn):
            return fn
    return None


if HAS_AVFOUNDATION:
    _CGColorSpaceCreateDeviceRGB = _resolve_cg_color_space_create_device_rgb()
    if _CGColorSpaceCreateDeviceRGB is None:
        logger.debug(
            "camera_capture: CGColorSpaceCreateDeviceRGB unresolved; "
            "JPEG encoding will request the CIContext-default colour space"
        )


# ── libdispatch via ctypes ────────────────────────────────────────────
# pyobjc on macOS 15+ does not expose ``dispatch_queue_create`` /
# ``dispatch_get_global_queue`` as Python callables. We pull them
# straight out of the system library with ctypes and wrap the
# returned queue pointer back into an ObjC object so AVFoundation
# accepts it as the delegate-callback queue.
_libdispatch: Any = None
if HAS_AVFOUNDATION:
    try:
        import ctypes
        import ctypes.util as _ctypes_util

        _libpath = _ctypes_util.find_library("System")
        if _libpath:
            _libdispatch = ctypes.cdll.LoadLibrary(_libpath)
            _libdispatch.dispatch_get_global_queue.restype = ctypes.c_void_p
            _libdispatch.dispatch_get_global_queue.argtypes = [
                ctypes.c_long, ctypes.c_ulong,
            ]
    except Exception:
        logger.debug("libdispatch ctypes load failed", exc_info=True)
        _libdispatch = None


def _global_dispatch_queue() -> Any:
    """Return a libdispatch global queue wrapped as an ObjC object.

    AVCaptureVideoDataOutput refuses ``None`` for its callback queue.
    Returns ``None`` if libdispatch isn't loadable — capture_jpeg will
    fall through with a clear error message.
    """
    if _libdispatch is None or _objc is None:
        return None
    try:
        # Priority 0 = DEFAULT, flags 0. Documented as never returning
        # NULL on supported macOS versions.
        ptr = _libdispatch.dispatch_get_global_queue(0, 0)
        if not ptr:
            return None
        return _objc.objc_object(c_void_p=ptr)
    except Exception:
        logger.debug("dispatch_get_global_queue wrap failed", exc_info=True)
        return None


# ── Public dataclasses ────────────────────────────────────────────────


# NOTE: ``AVCaptureDeviceTypeContinuityCamera`` is intentionally NOT
# listed here. ATOM's policy (Apr 26 2026, owner request) is
# MacBook-camera only — iPhone Continuity Camera is excluded both at
# discovery and at the picker layer.
_BUILTIN_TYPE = "AVCaptureDeviceTypeBuiltInWideAngleCamera"
_EXTERNAL_TYPES = (
    "AVCaptureDeviceTypeExternal",
    "AVCaptureDeviceTypeExternalUnknown",
)


def _looks_like_iphone(*, name: str = "", model_id: str = "") -> bool:
    """True when an AVCaptureDevice's metadata reveals a paired iPhone.

    Empirically (macOS 15+ on Apple Silicon), a Continuity-Camera-paired
    iPhone reports ``device_type=AVCaptureDeviceTypeExternal`` with a
    ``modelID`` like ``"iPhone15,4"`` and a ``localizedName`` containing
    ``"iPhone"``. We also see the dedicated
    ``AVCaptureDeviceTypeContinuityCamera`` type on some macOS builds.
    Either signal is sufficient to ban the device from the active list.
    """
    name_lower = (name or "").lower()
    model_lower = (model_id or "").lower()
    return (
        model_lower.startswith("iphone")
        or "iphone" in name_lower
    )


def _classify_kind(device_type: str, *, name: str = "", model_id: str = "") -> str:
    """Map an AVCaptureDevice to one of our diagnostic kinds.

    Returns one of:

    * ``"builtin"`` — the laptop's built-in FaceTime HD webcam.
    * ``"iphone"`` — a paired iPhone (Continuity Camera or
      iPhone-shaped External). **Diagnostic only** — :func:`list_cameras`
      filters these out before they reach the picker, so they can
      never become the active source.
    * ``"external"`` — a non-iPhone External device (USB webcam, DSLR
      capture, etc.). Eligible if the user has no built-in available.
    * ``"unknown"`` — anything else AVFoundation surfaces.
    """
    if device_type == _BUILTIN_TYPE:
        return "builtin"
    if _looks_like_iphone(name=name, model_id=model_id):
        return "iphone"
    if device_type in _EXTERNAL_TYPES:
        return "external"
    return "unknown"


@dataclass(frozen=True)
class CameraInfo:
    """Lightweight metadata about a discovered camera."""

    name: str
    unique_id: str
    device_type: str
    kind: str  # "builtin" | "iphone" | "external" | "unknown"

    def __str__(self) -> str:
        return f"{self.name} ({self.kind})"


@dataclass
class CaptureResult:
    """Outcome of a single-frame capture attempt."""

    ok: bool = False
    saved_path: str = ""
    capture_ms: float = 0.0
    camera: CameraInfo | None = None
    error: str = ""


# ── Device discovery ──────────────────────────────────────────────────


def list_cameras() -> list[CameraInfo]:
    """Return every NON-iPhone camera AVFoundation can see right now.

    Empty list when AVFoundation is unavailable or no eligible cameras
    are attached. Per the Apr 26 2026 policy, any device classified
    as ``"iphone"`` is dropped from the returned list — it is logged
    once at DEBUG level so a curious operator can confirm the iPhone
    was *seen* but intentionally ignored.

    The discovery session asks AVFoundation only for the built-in
    webcam type and the External device types. We do NOT enumerate
    ``AVCaptureDeviceTypeContinuityCamera`` so macOS won't even pop
    the Continuity reconnect handshake on ATOM's behalf.
    """
    if not HAS_AVFOUNDATION:
        return []

    types: list[Any] = []
    for name in (_BUILTIN_TYPE, *_EXTERNAL_TYPES):
        const = getattr(_AVF, name, None)
        if const is not None:
            types.append(const)
    if not types:
        return []

    try:
        sess = _AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            types,
            _AVF.AVMediaTypeVideo,
            _AVF.AVCaptureDevicePositionUnspecified,
        )
        devices = list(sess.devices() or [])
    except Exception:
        logger.debug("AVCaptureDeviceDiscoverySession failed", exc_info=True)
        return []

    out: list[CameraInfo] = []
    iphones_filtered: list[str] = []
    for d in devices:
        try:
            name = str(d.localizedName())
            uid = str(d.uniqueID())
            dt = str(d.deviceType())
        except Exception:
            continue
        # ``modelID`` is the AVCaptureDevice's hardware identifier
        # (e.g. ``"iPhone15,4"``). Apple doesn't always set it, so we
        # tolerate failure and fall back to the localized name when
        # classifying.
        model_id = ""
        try:
            model_id = str(d.modelID() or "")
        except Exception:
            pass
        kind = _classify_kind(dt, name=name, model_id=model_id)
        if kind == "iphone":
            iphones_filtered.append(name or model_id or uid)
            continue
        out.append(CameraInfo(
            name=name, unique_id=uid, device_type=dt, kind=kind,
        ))
    if iphones_filtered:
        logger.debug(
            "list_cameras: filtered %d iPhone-shaped device(s) "
            "(MacBook-camera-only policy): %s",
            len(iphones_filtered),
            ", ".join(iphones_filtered),
        )
    return out


# Logged once per process so legacy ``preferred_camera="continuity"``
# config doesn't spam the boot log on every camera selection.
_continuity_deprecation_logged = False


def choose_preferred(
    cameras: Iterable[CameraInfo] | None = None,
    *,
    preferred: str = "builtin",
    explicit_uid: str | None = None,
) -> CameraInfo | None:
    """Pick the camera matching the configured preference.

    The ATOM policy as of Apr 26 2026 is **MacBook-camera only**, so
    this function will never return an iPhone-shaped camera regardless
    of the ``preferred`` argument. iPhone devices are already filtered
    out by :func:`list_cameras`; the second-line defence here is to
    rewrite legacy preferences (``"continuity"`` / ``"auto"``) to
    ``"builtin"``.

    Parameters
    ----------
    cameras
        Iterable from :func:`list_cameras`. ``None`` triggers a fresh
        discovery — use the explicit form when callers want to log the
        full menu.
    preferred
        ``"builtin"`` (default) — first built-in webcam, falling back
        to a non-iPhone external rig if no built-in is present.
        ``"auto"`` — alias of ``"builtin"`` after the iPhone-camera
        removal.
        ``"continuity"`` — **deprecated**; aliased to ``"builtin"`` and
        logged once at startup so old configs keep booting.
        Anything else also falls back to ``"builtin"``.
    explicit_uid
        If non-empty, look for this exact ``uniqueID`` first. Wins
        over ``preferred``. The UID is still rejected if it points to
        an iPhone-shaped device (defence in depth — those should
        already be filtered out upstream).

    Returns ``None`` when no eligible camera is available.
    """
    cams = list(cameras) if cameras is not None else list_cameras()
    if not cams:
        return None

    # Defence in depth: even if a caller hands us a pre-built list that
    # still contains an iPhone (e.g. an older test harness), we drop
    # those rows here so they cannot win the picker.
    cams = [c for c in cams if c.kind != "iphone"]
    if not cams:
        return None

    if explicit_uid:
        for c in cams:
            if c.unique_id == explicit_uid:
                return c

    p = (preferred or "builtin").strip().lower()
    if p == "continuity":
        global _continuity_deprecation_logged
        if not _continuity_deprecation_logged:
            logger.warning(
                "vision.preferred_camera='continuity' is deprecated as of "
                "Apr 26 2026 — ATOM is MacBook-camera only and will use "
                "the built-in webcam instead. Update your config to "
                "'builtin' to silence this warning.",
            )
            _continuity_deprecation_logged = True
        p = "builtin"
    elif p == "auto":
        p = "builtin"
    elif p != "builtin":
        # Unknown preference — fail safe to built-in. We don't log
        # because the schema validator already rejects bad values at
        # config-load time; this path only fires on programmatic
        # callers that bypass the schema.
        p = "builtin"

    def _first(kind: str) -> CameraInfo | None:
        return next((c for c in cams if c.kind == kind), None)

    return _first("builtin") or _first("external") or cams[0]


def _resolve_avdevice(camera: CameraInfo) -> Any | None:
    if not HAS_AVFOUNDATION:
        return None
    try:
        return _AVF.AVCaptureDevice.deviceWithUniqueID_(camera.unique_id)
    except Exception:
        logger.debug("deviceWithUniqueID_ failed for %s", camera.unique_id, exc_info=True)
        return None


# ── Video data output delegate ────────────────────────────────────────
#
# We use AVCaptureVideoDataOutput rather than AVCapturePhotoOutput
# because the latter triggers
# ``NSKVONotifying_AVCapturePhotoOutput' not linked into application``
# under pyobjc on macOS 15+ — the embedded Python runtime can't
# resolve AVFoundation's dynamic KVO subclasses, and the resulting
# ``-11800`` AVErrorUnknown silently broke our boot face check in
# production. Video data output bypasses KVO entirely and is the
# AVFoundation pattern Apple recommends for "snap one frame from the
# camera" use-cases that don't need exposure / RAW / mirror control.


def _build_video_delegate_class() -> Any:
    """Lazily build the ObjC video sample buffer delegate.

    Built lazily so the module remains importable on systems without
    pyobjc (we expose ``HAS_AVFOUNDATION`` for the runtime to gate on).
    """
    base = _NF.NSObject

    class _VideoSampleDelegate(base):  # type: ignore[misc, valid-type]
        """Receives ``CMSampleBuffer`` callbacks and JPEG-encodes one frame.

        We capture the *second* sample buffer rather than the first:
        the very first frame off the FaceTime HD sensor is sometimes a
        partially-initialised black frame while the camera ramps
        exposure. Skipping one buffer adds <50 ms latency and
        eliminates a class of "all-black image" bug reports.
        """

        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self, output: Any, sample_buffer: Any, connection: Any,
        ) -> None:
            if self._event.is_set():
                return
            self._frames_seen += 1
            # Skip the first frame on Continuity Camera / built-in
            # both: the first sample buffer often comes out before
            # the sensor finishes ramping. We deliberately do NOT
            # signal the event here — caller keeps waiting for a
            # subsequent frame.
            if self._frames_seen < 2:
                return
            try:
                pixel_buffer = _CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
                if pixel_buffer is None:
                    self._error = "CMSampleBufferGetImageBuffer returned nil"
                    self._event.set()
                    return
                ci_image = _Quartz.CIImage.imageWithCVPixelBuffer_(pixel_buffer)
                if ci_image is None:
                    self._error = "CIImage.imageWithCVPixelBuffer_ returned nil"
                    self._event.set()
                    return
                # Force a software-backed CIContext: passing
                # ``contextWithOptions_(None)`` defaults to a GPU-backed
                # context that issues its own Metal command buffers,
                # which races MLX/mlx-vlm for the same device queue and
                # has been observed to trigger
                # ``-[_MTLCommandBuffer addCompletedHandler:]:1011``
                # SIGABRTs on cold start. Software encoding adds <30ms
                # for one 1080p JPEG -- inconsequential vs. a process
                # crash, and matches the comment that already lived on
                # this site.
                opts: dict[Any, Any] = {}
                use_sw_key = getattr(_Quartz, "kCIContextUseSoftwareRenderer", None)
                if use_sw_key is not None:
                    opts[use_sw_key] = True
                ctx = _Quartz.CIContext.contextWithOptions_(opts or None)
                # Resolved at import time across (Quartz |
                # Quartz.CoreGraphics | CoreGraphics) so we don't
                # crash on the partial PyObjC layouts shipped inside
                # ATOM.app.
                color_space = (
                    _CGColorSpaceCreateDeviceRGB()
                    if _CGColorSpaceCreateDeviceRGB is not None
                    else None
                )
                jpeg_options = _NF.NSDictionary.dictionary()
                jpeg_data = ctx.JPEGRepresentationOfImage_colorSpace_options_(
                    ci_image, color_space, jpeg_options,
                )
                if jpeg_data is None:
                    self._error = (
                        "JPEGRepresentationOfImage_ returned nil"
                        if color_space is not None
                        else "JPEGRepresentationOfImage_ returned nil "
                             "(no colour space — CGColorSpaceCreateDeviceRGB "
                             "missing in PyObjC)"
                    )
                    self._event.set()
                    return
                self._jpeg_bytes = bytes(jpeg_data)
                self._event.set()
            except Exception as exc:
                self._error = f"video delegate exception: {exc}"
                self._event.set()

        def init(self) -> Any:
            self = _objc.super(_VideoSampleDelegate, self).init()
            if self is None:
                return None
            self._event = threading.Event()
            self._jpeg_bytes = None
            self._error = ""
            self._frames_seen = 0
            return self

    return _VideoSampleDelegate


_video_delegate_cls: Any = None


def _get_video_delegate_cls() -> Any:
    global _video_delegate_cls
    if _video_delegate_cls is None:
        _video_delegate_cls = _build_video_delegate_class()
    return _video_delegate_cls


# ── Capture entrypoint ────────────────────────────────────────────────


def capture_jpeg(
    camera: CameraInfo,
    *,
    out_path: str | Path | None = None,
    timeout_s: float = 3.5,
) -> CaptureResult:
    """Capture exactly one JPEG from *camera* and return its file path.

    Synchronous: blocks the calling thread until the photo arrives or
    *timeout_s* elapses. The capture session is created and torn down
    inside this call — no long-lived state is left behind, so you can
    safely call this from any worker thread.

    macOS will prompt for camera TCC permission on the first call
    after the bundle is installed. If the user denies, the prompt
    won't reappear; future captures will return ``ok=False`` with
    ``error="permission denied"``.
    """
    if not HAS_AVFOUNDATION:
        return CaptureResult(error="AVFoundation not available")

    device = _resolve_avdevice(camera)
    if device is None:
        return CaptureResult(camera=camera, error="device not resolvable by uniqueID")

    out_p: Path
    if out_path:
        out_p = Path(out_path)
    else:
        # Use a per-process temp dir so multiple concurrent ATOM
        # instances don't clobber each other.
        tmpdir = Path(tempfile.gettempdir()) / "atom_vision"
        tmpdir.mkdir(parents=True, exist_ok=True)
        out_p = tmpdir / f"capture_{int(time.time() * 1000)}.jpg"

    t0 = time.perf_counter()
    session = None
    runner_started = False
    try:
        session = _AVF.AVCaptureSession.alloc().init()

        # Set the session preset BEFORE adding inputs/outputs. Some
        # devices refuse to honour outputs added before the preset is
        # locked in, so we set it first as a defensive default.
        # ``Photo`` works for the built-in FaceTime HD; ``High`` is a
        # safe fallback if the device can't do photo-quality.
        try:
            preset = getattr(_AVF, "AVCaptureSessionPresetPhoto", None) or \
                getattr(_AVF, "AVCaptureSessionPresetHigh", None)
            if preset is not None and session.canSetSessionPreset_(preset):
                session.setSessionPreset_(preset)
        except Exception:
            logger.debug("setSessionPreset_ failed; using default", exc_info=True)

        try:
            session.beginConfiguration()
        except Exception:
            return CaptureResult(camera=camera, error="beginConfiguration failed")

        input_pair = _AVF.AVCaptureDeviceInput.alloc().initWithDevice_error_(
            device, None,
        )
        # pyobjc returns either ``(input, None)`` or a single object
        # depending on the binding version; normalise.
        if isinstance(input_pair, tuple):
            input_, err = input_pair
        else:
            input_, err = input_pair, None
        if err is not None:
            return CaptureResult(
                camera=camera, error=f"AVCaptureDeviceInput error: {err}",
            )
        if not input_ or not session.canAddInput_(input_):
            return CaptureResult(
                camera=camera,
                error="session refused device input (likely TCC denied)",
            )
        session.addInput_(input_)

        video_out = _AVF.AVCaptureVideoDataOutput.alloc().init()
        # Drop late video frames so we never queue up frames the user
        # won't see. We only need one anyway.
        try:
            video_out.setAlwaysDiscardsLateVideoFrames_(True)
        except Exception:
            logger.debug("setAlwaysDiscardsLateVideoFrames_ unavailable", exc_info=True)
        if not session.canAddOutput_(video_out):
            return CaptureResult(
                camera=camera, error="session refused video data output",
            )
        session.addOutput_(video_out)

        # Run the delegate callback on a libdispatch global queue
        # (low-priority) so our pyobjc delegate is invoked off the
        # main thread. The main thread is owned by the asyncio loop
        # in a typical ATOM run, and the MLX brain owns the GPU.
        # AVCaptureVideoDataOutput refuses a NULL queue, so we must
        # supply one.
        delegate = _get_video_delegate_cls().alloc().init()
        queue = _global_dispatch_queue()
        if queue is None:
            return CaptureResult(
                camera=camera,
                error="libdispatch unavailable; cannot deliver sample buffers",
            )
        try:
            video_out.setSampleBufferDelegate_queue_(delegate, queue)
        except Exception as exc:
            return CaptureResult(
                camera=camera,
                error=f"setSampleBufferDelegate_queue_ failed: {exc}",
            )

        try:
            session.commitConfiguration()
        except Exception:
            return CaptureResult(camera=camera, error="commitConfiguration failed")

        session.startRunning()
        runner_started = True

        # Wait for the delegate to JPEG-encode one frame. We skip the
        # first sample buffer (see _VideoSampleDelegate docstring) so
        # the worst-case wall time is roughly 2 × frame interval (~66 ms
        # at 30 fps) plus session warmup. Built-in FaceTime HD wakes
        # in <500 ms typically; the 3.5 s default timeout is generous
        # for first-call cold starts after lid-open.
        if not delegate._event.wait(timeout_s):
            return CaptureResult(
                camera=camera,
                error=f"capture timed out after {timeout_s:.1f}s",
            )

        if delegate._error:
            return CaptureResult(camera=camera, error=delegate._error)

        if not delegate._jpeg_bytes:
            return CaptureResult(camera=camera, error="empty frame")

        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically: write to .tmp then rename so a partial
        # write can never be observed by the Vision pipeline.
        tmp_path = out_p.with_suffix(out_p.suffix + ".tmp")
        tmp_path.write_bytes(delegate._jpeg_bytes)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            logger.debug("chmod 0o600 failed for %s", tmp_path)
        os.replace(tmp_path, out_p)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return CaptureResult(
            ok=True,
            saved_path=str(out_p),
            capture_ms=round(elapsed_ms, 1),
            camera=camera,
        )
    except Exception as exc:  # pragma: no cover -- defensive
        return CaptureResult(camera=camera, error=f"unexpected: {exc}")
    finally:
        if runner_started and session is not None:
            try:
                session.stopRunning()
            except Exception:
                logger.debug("session.stopRunning failed", exc_info=True)
