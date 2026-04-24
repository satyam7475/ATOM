"""iPhone Shortcuts HTTP bridge.

Exposes a tiny aiohttp listener on localhost so Apple Shortcuts can
POST Face ID verifications, presence changes, and named triggers to
ATOM. No Xcode, no SwiftUI, no Apple developer account -- only the
stock iOS Shortcuts app is required on the iPhone.

Endpoints
---------
* ``GET  /health`` -- round-trip sanity check (no auth; returns
  ``{"ok": true, "version": "1"}``). Safe to expose because it leaks
  nothing.
* ``POST /faceid`` -- ``{device_id, verified, timestamp?, label?}``
  records a Face ID verification.
* ``POST /presence`` -- ``{device_id, state, timestamp?}``
  publishes ``iphone.presence.changed`` onto the event bus.
* ``POST /trigger`` -- ``{device_id, name, args?}`` publishes
  ``iphone.trigger.fired``.

Auth
----
Every non-``/health`` endpoint requires ``X-ATOM-Token: <token>``.
Mismatch -> 401 + audit line. Missing -> 401 + audit line.

Single-device lock
------------------
First successful ``/faceid`` (or any authenticated POST with a
``device_id``) claims the trusted-iPhone slot. A different device_id
thereafter returns 409 Conflict until the owner resets the slot.

Port fallback
-------------
Tries ``port``, ``port+1``, ``port+2`` (defaults 8787/8788/8789). The
chosen port is written to ``logs/atom_bridge.port`` so Shortcuts can
read it on next run.

Rate limiting
-------------
Each endpoint is limited to 1 successful request / second / device
(token-bucket). Bursts beyond that return 429.

Failure posture
---------------
Every class-level failure (port bound, aiohttp missing, network
unreachable) logs CRITICAL and returns without raising -- ATOM keeps
booting without cross-device, never crashes because a bridge couldn't
start.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from core.cross_device.bridge_auth import (
    AuthAuditLog,
    generate_or_load_token,
    verify_token,
)
from core.cross_device.trusted_device import TrustedIPhoneRegistry

logger = logging.getLogger("atom.bridge")


_DEFAULT_PORT = 8787
_PORT_FALLBACK_COUNT = 3
_AUTH_HEADER = "X-ATOM-Token"
_MAX_BODY_BYTES = 16 * 1024
_PRESENCE_STATES = frozenset({"at_desk", "leaving", "home", "away", "busy"})
_TRIGGER_NAME_MAX = 64
_RATE_WINDOW_S = 1.0


EmitFn = Callable[..., None]
"""Shape of the event-bus emit. We accept any callable so tests can
inject a capturing fake without pulling the real bus in."""


class IPhoneBridge:
    """aiohttp HTTP listener on localhost for iPhone Shortcuts.

    Designed to be instantiated once during boot, wrapped in an
    ``asyncio.Task`` via ``start()``, and torn down via ``stop()``.
    """

    __slots__ = (
        "_config",
        "_emit",
        "_token",
        "_audit",
        "_trusted",
        "_port",
        "_bound_port",
        "_host",
        "_port_file",
        "_app",
        "_runner",
        "_site",
        "_rate_last_accept",
        "_started",
    )

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        emit: Optional[EmitFn] = None,
        atom_root: str | Path | None = None,
    ) -> None:
        cfg = dict((config or {}).get("cross_device") or {})
        self._config = cfg
        self._emit = emit or _noop_emit

        root = Path(atom_root) if atom_root else _infer_atom_root()
        token_path = (
            cfg.get("token_path")
            or str(root / "config" / "bridge_token")
        )
        audit_path = (
            cfg.get("audit_log_path")
            or str(root / "logs" / "atom_bridge_audit.jsonl")
        )
        trusted_path = (
            cfg.get("trusted_device_path")
            or str(root / "data" / "trusted_iphone.json")
        )
        port_file = (
            cfg.get("port_file_path")
            or str(root / "logs" / "atom_bridge.port")
        )

        self._token = generate_or_load_token(token_path)
        self._audit = AuthAuditLog(audit_path)
        self._trusted = TrustedIPhoneRegistry(trusted_path)

        self._host = str(cfg.get("bind_host") or "127.0.0.1")
        self._port = int(cfg.get("bridge_port") or _DEFAULT_PORT)
        self._bound_port: int | None = None
        self._port_file = Path(port_file)

        self._app = None
        self._runner = None
        self._site = None
        self._rate_last_accept: dict[tuple[str, str], float] = {}
        self._started = False

    @property
    def token(self) -> str:
        return self._token

    @property
    def actual_port(self) -> int | None:
        """The port the listener actually bound to (after fallback).

        Remains set after :py:meth:`stop` so callers (e.g. tests, or a
        graceful-shutdown handler that wants to log the final URL) can
        still read the value post-stop. ``None`` until :py:meth:`start`
        succeeds at least once.
        """
        return self._bound_port

    async def start(self) -> bool:
        """Bind to a port (with fallback) and start serving.

        Returns True on success, False on any failure (port exhaustion,
        aiohttp missing, etc.). Never raises so a degraded bridge does
        not prevent ATOM from booting.
        """
        if self._started:
            return True

        try:
            from aiohttp import web
        except ImportError:
            logger.critical("aiohttp missing; iPhone bridge cannot start")
            return False

        app = web.Application(client_max_size=_MAX_BODY_BYTES)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/faceid", self._handle_faceid)
        app.router.add_post("/presence", self._handle_presence)
        app.router.add_post("/trigger", self._handle_trigger)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()

        bound_port: int | None = None
        last_err: Exception | None = None
        for offset in range(_PORT_FALLBACK_COUNT):
            candidate = self._port + offset
            try:
                site = web.TCPSite(runner, host=self._host, port=candidate)
                await site.start()
                bound_port = candidate
                self._site = site
                break
            except OSError as exc:
                last_err = exc
                continue

        if bound_port is None:
            logger.critical(
                "iPhone bridge could not bind any port in %d..%d: %s",
                self._port,
                self._port + _PORT_FALLBACK_COUNT - 1,
                last_err,
            )
            await runner.cleanup()
            return False

        self._app = app
        self._runner = runner
        self._started = True
        self._bound_port = bound_port
        self._persist_port(bound_port)
        logger.info(
            "iPhone bridge listening on %s:%d (token=%s…)",
            self._host, bound_port, self._token[:6],
        )
        return True

    async def stop(self) -> None:
        """Graceful shutdown. Safe to call multiple times."""
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                logger.debug("site.stop failed", exc_info=True)
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                logger.debug("runner.cleanup failed", exc_info=True)
            self._runner = None
        self._app = None
        self._started = False

    def _persist_port(self, port: int) -> None:
        try:
            self._port_file.parent.mkdir(parents=True, exist_ok=True)
            self._port_file.write_text(str(port), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not persist bridge port: %s", exc)

    def _client_ip(self, request: Any) -> str:
        try:
            peer = request.transport.get_extra_info("peername")
            if peer:
                return str(peer[0])
        except (AttributeError, IndexError):
            pass
        return "unknown"

    def _check_rate(self, device_id: str, endpoint: str) -> bool:
        """Token-bucket-ish rate limit: one successful accept per second
        per (device, endpoint). Returns True if accepted, False if
        throttled."""
        now = time.monotonic()
        key = (device_id or "unknown", endpoint)
        last = self._rate_last_accept.get(key, 0.0)
        if now - last < _RATE_WINDOW_S:
            return False
        self._rate_last_accept[key] = now
        return True

    async def _require_auth(self, request: Any, endpoint: str) -> Any:
        """Return None on success, or a 401 Response on failure."""
        from aiohttp import web
        supplied = request.headers.get(_AUTH_HEADER)
        if not verify_token(self._token, supplied):
            reason = "missing_token" if not supplied else "bad_token"
            should_warn = self._audit.record_failure(
                source_ip=self._client_ip(request),
                endpoint=endpoint,
                reason=reason,
            )
            if should_warn:
                try:
                    self._emit("iphone.bridge.flood_warning", reason=reason)
                except Exception:
                    logger.debug("emit flood_warning failed", exc_info=True)
            return web.json_response(
                {"ok": False, "error": "unauthorized"},
                status=401,
            )
        return None

    async def _read_json(self, request: Any) -> tuple[dict[str, Any] | None, Any]:
        """Parse the JSON body. Returns ``(payload, error_response)``;
        one of the two is always None."""
        from aiohttp import web
        try:
            raw = await request.read()
        except Exception as exc:
            return None, web.json_response(
                {"ok": False, "error": f"read_failed:{type(exc).__name__}"},
                status=400,
            )
        if not raw:
            return {}, None
        if len(raw) > _MAX_BODY_BYTES:
            return None, web.json_response(
                {"ok": False, "error": "body_too_large"},
                status=413,
            )
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            return None, web.json_response(
                {"ok": False, "error": "bad_json"},
                status=400,
            )
        if not isinstance(data, dict):
            return None, web.json_response(
                {"ok": False, "error": "not_an_object"},
                status=400,
            )
        return data, None

    def _check_device(
        self,
        request: Any,
        payload: dict[str, Any],
        endpoint: str,
    ) -> tuple[str | None, Any]:
        """Enforce single-device lock. Returns ``(device_id, err_response)``;
        one of the two is always None."""
        from aiohttp import web
        device_id = str(payload.get("device_id") or "").strip()
        if not device_id:
            return None, web.json_response(
                {"ok": False, "error": "missing_device_id"},
                status=400,
            )
        label = str(payload.get("label") or "")[:64]
        ok, reason = self._trusted.register_or_verify(
            device_id, label=label,
        )
        if not ok:
            self._audit.record_reject(
                source_ip=self._client_ip(request),
                endpoint=endpoint,
                reason=f"device_conflict:{reason}",
            )
            return None, web.json_response(
                {
                    "ok": False,
                    "error": "device_conflict",
                    "hint": (
                        "A different iPhone already holds the trusted-device "
                        "slot. Run `python -m core.cross_device.trusted_device "
                        "reset` on the Mac if you intend to switch phones."
                    ),
                },
                status=409,
            )
        return device_id, None

    async def _handle_health(self, request: Any) -> Any:
        from aiohttp import web
        return web.json_response({"ok": True, "version": 1})

    async def _handle_faceid(self, request: Any) -> Any:
        from aiohttp import web
        err = await self._require_auth(request, "/faceid")
        if err is not None:
            return err
        payload, perr = await self._read_json(request)
        if perr is not None:
            return perr

        device_id, derr = self._check_device(request, payload or {}, "/faceid")
        if derr is not None:
            return derr

        verified = bool((payload or {}).get("verified"))
        ts = _coerce_ts((payload or {}).get("timestamp"))
        label = str((payload or {}).get("label") or "")[:64]

        # Rate-limit is applied last so malformed payloads get their
        # real 400 back on the very next retry instead of being silently
        # swallowed as 429.
        if not self._check_rate(device_id, "/faceid"):
            return web.json_response(
                {"ok": False, "error": "rate_limited"}, status=429,
            )

        self._emit(
            "iphone.faceid.verified",
            device_id=device_id,
            verified=verified,
            timestamp=ts,
            label=label,
        )
        return web.json_response({"ok": True, "accepted_at": ts})

    async def _handle_presence(self, request: Any) -> Any:
        from aiohttp import web
        err = await self._require_auth(request, "/presence")
        if err is not None:
            return err
        payload, perr = await self._read_json(request)
        if perr is not None:
            return perr

        device_id, derr = self._check_device(request, payload or {}, "/presence")
        if derr is not None:
            return derr

        state = str((payload or {}).get("state") or "").strip().lower()
        if state not in _PRESENCE_STATES:
            return web.json_response(
                {
                    "ok": False,
                    "error": "bad_state",
                    "allowed": sorted(_PRESENCE_STATES),
                },
                status=400,
            )
        ts = _coerce_ts((payload or {}).get("timestamp"))

        if not self._check_rate(device_id, "/presence"):
            return web.json_response(
                {"ok": False, "error": "rate_limited"}, status=429,
            )

        self._emit(
            "iphone.presence.changed",
            device_id=device_id,
            state=state,
            timestamp=ts,
        )
        return web.json_response({"ok": True, "state": state})

    async def _handle_trigger(self, request: Any) -> Any:
        from aiohttp import web
        err = await self._require_auth(request, "/trigger")
        if err is not None:
            return err
        payload, perr = await self._read_json(request)
        if perr is not None:
            return perr

        device_id, derr = self._check_device(request, payload or {}, "/trigger")
        if derr is not None:
            return derr

        name = str((payload or {}).get("name") or "").strip()
        if not name or len(name) > _TRIGGER_NAME_MAX:
            return web.json_response(
                {"ok": False, "error": "bad_trigger_name"},
                status=400,
            )
        args_raw = (payload or {}).get("args")
        args = args_raw if isinstance(args_raw, dict) else {}

        if not self._check_rate(device_id, "/trigger"):
            return web.json_response(
                {"ok": False, "error": "rate_limited"}, status=429,
            )

        self._emit(
            "iphone.trigger.fired",
            device_id=device_id,
            name=name,
            args=args,
        )
        return web.json_response({"ok": True, "name": name})


def _coerce_ts(raw: Any) -> float:
    """Return the supplied timestamp if it parses as a positive float,
    else the local epoch. Shortcuts sometimes sends ISO-8601; we keep
    parsing lenient so the first try on iPhone doesn't silently drop
    the payload."""
    if raw is None:
        return time.time()
    try:
        val = float(raw)
        if val > 0:
            return val
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return time.time()


def _noop_emit(*_args: Any, **_kwargs: Any) -> None:
    pass


def _infer_atom_root() -> Path:
    """Walk up from this file to ATOM's root (the folder that contains
    ``config/settings.json``). Falls back to ``cwd`` if the walk fails."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "settings.json").exists():
            return candidate
    return Path.cwd()


__all__ = ["IPhoneBridge"]
