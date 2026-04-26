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
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

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
_OPENAI_BODY_BYTES = 1024 * 1024  # 1 MiB; chat history can be long
_PRESENCE_STATES = frozenset({"at_desk", "leaving", "home", "away", "busy"})
_TRIGGER_NAME_MAX = 64
_RATE_WINDOW_S = 1.0
_OPENAI_RATE_WINDOW_S = 0.5  # gentler limit for chat (long-running)
_OPENAI_DEFAULT_MAX_TOKENS = 512


EmitFn = Callable[..., None]
"""Shape of the event-bus emit. We accept any callable so tests can
inject a capturing fake without pulling the real bus in."""


# Sprint P4.4 (Apr 26 2026): OpenAI-compatible chat handler contract.
# The handler takes a list of OpenAI-style messages and yields text
# chunks. A final empty string with done=True closes the stream. We
# keep the contract dependency-free so the bridge module never has to
# import the brain.
ChatStreamFn = Callable[
    [list[dict[str, str]]],
    AsyncIterator[tuple[str, bool]],
]
"""Async iterator yielding ``(token_text, is_done)`` pairs."""


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
        "_chat_stream",
        "_openai_model_id",
        "_openai_default_max_tokens",
        "_openai_enabled",
        "_status_provider",
    )

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        emit: Optional[EmitFn] = None,
        atom_root: str | Path | None = None,
        chat_stream: ChatStreamFn | None = None,
        status_provider: Callable[[], dict[str, Any]] | None = None,
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

        # Sprint P4.4 (Apr 26 2026): OpenAI-compat /v1/* shim. When a
        # chat-stream callable is provided, GET /v1/models and
        # POST /v1/chat/completions are exposed so the iPhone Enchanted
        # app (and any other openai-compatible client over Tailscale)
        # can talk to ATOM's local brain. Without a handler the routes
        # are unregistered -- callers see 404, not 500.
        self._chat_stream = chat_stream

        # Sprint P4.6 (Apr 26 2026): unified status badge surface. When
        # set, GET /badge returns the same rollup the dashboard menubar
        # uses. No auth (the badge has no PII; same posture as /health).
        self._status_provider = status_provider
        openai_cfg = dict(cfg.get("openai_compat") or {})
        self._openai_enabled = bool(openai_cfg.get("enabled", True)) and (
            chat_stream is not None
        )
        self._openai_model_id = str(
            openai_cfg.get("model_id") or "atom-local",
        )
        self._openai_default_max_tokens = int(
            openai_cfg.get("default_max_tokens")
            or _OPENAI_DEFAULT_MAX_TOKENS,
        )

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

        # Larger client_max_size lets long chat histories arrive intact;
        # individual handlers still enforce stricter caps. The original
        # (16 KiB) cap is preserved for /faceid /presence /trigger via
        # explicit body-size checks in ``_read_json``.
        app_max_size = (
            _OPENAI_BODY_BYTES if self._openai_enabled else _MAX_BODY_BYTES
        )
        app = web.Application(client_max_size=app_max_size)
        app.router.add_get("/health", self._handle_health)
        # Sprint P4.6 (Apr 26 2026): /badge is always registered so it
        # can be wired late at boot via :py:attr:`_status_provider`. The
        # handler returns a grey "unknown" badge when no provider is
        # attached -- never 404 -- so a polling menubar app gets a
        # stable shape from t=0.
        app.router.add_get("/badge", self._handle_badge)
        app.router.add_post("/faceid", self._handle_faceid)
        app.router.add_post("/presence", self._handle_presence)
        app.router.add_post("/trigger", self._handle_trigger)
        if self._openai_enabled:
            app.router.add_get("/v1/models", self._handle_v1_models)
            app.router.add_post(
                "/v1/chat/completions", self._handle_v1_chat,
            )
            logger.info(
                "iPhone bridge: OpenAI-compat /v1/* shim enabled "
                "(model_id=%s)",
                self._openai_model_id,
            )

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

    async def _handle_badge(self, request: Any) -> Any:
        """Sprint P4.6 (Apr 26 2026): unified status badge.

        Returns the same rollup the dashboard menubar uses, in a
        no-auth, low-PII shape. Body looks like::

            {
                "ok": true,
                "level": "ok" | "warn" | "critical",
                "color": "green" | "amber" | "red",
                "text":  "ATOM is OK",
                "headline": "stt: degraded",
                "subsystems_total": 10,
                "uptime_s": 1234.5
            }

        Designed to be polled by a Mac menubar app at ~5 s, an iPhone
        widget at ~30 s, and the smoke scripts at boot. Fully tolerant
        of the provider raising -- on any failure we return a grey
        ``unknown`` badge with HTTP 200 so the menubar never flashes
        red just because the provider hiccuped.
        """
        from aiohttp import web
        if self._status_provider is None:
            return web.json_response(
                {
                    "ok": False,
                    "level": "unknown",
                    "color": "grey",
                    "text": "ATOM status unknown",
                    "headline": "no status provider",
                    "subsystems_total": 0,
                },
                status=200,
            )
        try:
            snapshot = self._status_provider() or {}
        except Exception as exc:
            logger.debug("status_provider raised: %s", exc, exc_info=True)
            return web.json_response(
                {
                    "ok": False,
                    "level": "unknown",
                    "color": "grey",
                    "text": "ATOM status unknown",
                    "headline": f"provider_error:{type(exc).__name__}",
                    "subsystems_total": 0,
                },
                status=200,
            )
        badge = (
            snapshot.get("badge")
            if isinstance(snapshot, dict) else None
        ) or {}
        body = {
            "ok": bool(snapshot.get("ok")) if isinstance(snapshot, dict) else False,
            "level": badge.get("level", "unknown"),
            "color": badge.get("color", "grey"),
            "text": badge.get("text", "ATOM status unknown"),
            "headline": badge.get("headline", ""),
            "warnings": badge.get("warnings", []),
            "criticals": badge.get("criticals", []),
            "subsystems_total": badge.get("subsystems_total", 0),
            "uptime_s": (
                snapshot.get("uptime_s")
                if isinstance(snapshot, dict) else None
            ),
        }
        return web.json_response(body)

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

    # ── Sprint P4.4 (Apr 26 2026) -- OpenAI-compatible /v1/* shim ────
    #
    # Goal: let the Enchanted iPhone app (and any other openai-style
    # client over Tailscale) talk to ATOM's local brain without us
    # having to ship a SwiftUI native app. We accept *either* the
    # native ATOM ``X-ATOM-Token`` header OR an ``Authorization:
    # Bearer <token>`` header so off-the-shelf clients can paste the
    # bridge token into the standard "API key" field.
    #
    # We deliberately do NOT enforce the trusted-device single-slot
    # lock here -- a chat session does not provide a stable
    # ``device_id`` and rejecting Enchanted on first contact would
    # be hostile. Auth is via the pre-shared bridge token, same as
    # the rest of the bridge. The audit log still records every
    # bad-token attempt.

    def _check_openai_auth(self, request: Any) -> str | None:
        """Return the supplied token if it matches, else None.

        Accepts ``X-ATOM-Token`` (preferred) or ``Authorization: Bearer
        <token>``. Records a single audit-log entry on rejection.
        """
        from aiohttp import web  # noqa: F401  (kept consistent with peers)
        supplied = request.headers.get(_AUTH_HEADER)
        if not supplied:
            authz = request.headers.get("Authorization", "") or ""
            if authz.lower().startswith("bearer "):
                supplied = authz[len("Bearer "):].strip() or None
        if verify_token(self._token, supplied):
            return supplied or ""
        reason = "missing_token" if not supplied else "bad_token"
        try:
            self._audit.record_failure(
                source_ip=self._client_ip(request),
                endpoint=str(request.path),
                reason=reason,
            )
        except Exception:
            logger.debug("openai_auth audit failed", exc_info=True)
        return None

    async def _handle_v1_models(self, request: Any) -> Any:
        from aiohttp import web
        if self._check_openai_auth(request) is None:
            return web.json_response(
                {"error": {"message": "unauthorized", "type": "auth"}},
                status=401,
            )
        now = int(time.time())
        body = {
            "object": "list",
            "data": [
                {
                    "id": self._openai_model_id,
                    "object": "model",
                    "owned_by": "atom",
                    "created": now,
                },
            ],
        }
        return web.json_response(body)

    async def _handle_v1_chat(self, request: Any) -> Any:
        from aiohttp import web
        if self._chat_stream is None:
            return web.json_response(
                {
                    "error": {
                        "message": "chat handler not registered",
                        "type": "not_configured",
                    },
                },
                status=503,
            )
        if self._check_openai_auth(request) is None:
            return web.json_response(
                {"error": {"message": "unauthorized", "type": "auth"}},
                status=401,
            )

        try:
            raw = await request.read()
        except Exception as exc:
            return web.json_response(
                {
                    "error": {
                        "message": f"read_failed:{type(exc).__name__}",
                        "type": "bad_request",
                    },
                },
                status=400,
            )
        if not raw or len(raw) > _OPENAI_BODY_BYTES:
            return web.json_response(
                {"error": {"message": "body_too_large", "type": "bad_request"}},
                status=413 if raw else 400,
            )
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(payload, dict):
                raise ValueError("not_an_object")
        except Exception:
            return web.json_response(
                {"error": {"message": "bad_json", "type": "bad_request"}},
                status=400,
            )

        messages_raw = payload.get("messages") or []
        if not isinstance(messages_raw, list) or not messages_raw:
            return web.json_response(
                {"error": {"message": "messages required", "type": "bad_request"}},
                status=400,
            )
        messages: list[dict[str, str]] = []
        for m in messages_raw:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip().lower()
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            content = str(content or "")
            if role and content:
                messages.append({"role": role, "content": content})
        if not messages:
            return web.json_response(
                {
                    "error": {
                        "message": "no usable messages",
                        "type": "bad_request",
                    },
                },
                status=400,
            )

        max_tokens = payload.get("max_tokens")
        try:
            max_tokens_int = int(max_tokens) if max_tokens else None
        except (TypeError, ValueError):
            max_tokens_int = None

        stream_requested = bool(payload.get("stream"))
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Soft rate-limit for chat: per-source-IP, half-second window.
        source_ip = self._client_ip(request)
        rate_key = f"openai_chat:{source_ip}"
        if not self._check_rate(rate_key, "/v1/chat/completions"):
            return web.json_response(
                {
                    "error": {
                        "message": "rate_limited",
                        "type": "rate_limit",
                    },
                },
                status=429,
            )

        try:
            stream = self._chat_stream(
                messages,
                model=self._openai_model_id,
                max_tokens=max_tokens_int or self._openai_default_max_tokens,
            )
        except TypeError:
            try:
                stream = self._chat_stream(messages)
            except Exception:
                logger.exception("chat_stream invocation failed")
                return web.json_response(
                    {
                        "error": {
                            "message": "chat_stream_failed",
                            "type": "server_error",
                        },
                    },
                    status=500,
                )
        except Exception:
            logger.exception("chat_stream invocation failed")
            return web.json_response(
                {
                    "error": {
                        "message": "chat_stream_failed",
                        "type": "server_error",
                    },
                },
                status=500,
            )

        if stream_requested:
            return await self._stream_v1_chat(
                request,
                stream,
                completion_id=completion_id,
                created=created,
            )
        return await self._collect_v1_chat(
            stream,
            completion_id=completion_id,
            created=created,
        )

    async def _collect_v1_chat(
        self,
        stream: AsyncIterator[tuple[str, bool]],
        *,
        completion_id: str,
        created: int,
    ) -> Any:
        from aiohttp import web
        full = []
        try:
            async for token, done in stream:
                if token:
                    full.append(token)
                if done:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("openai_compat: collect stream failed")
            return web.json_response(
                {
                    "error": {
                        "message": "stream_failed",
                        "type": "server_error",
                    },
                },
                status=500,
            )
        text = "".join(full)
        body = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": self._openai_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(text.split()),
                "total_tokens": len(text.split()),
            },
        }
        return web.json_response(body)

    async def _stream_v1_chat(
        self,
        request: Any,
        stream: AsyncIterator[tuple[str, bool]],
        *,
        completion_id: str,
        created: int,
    ) -> Any:
        from aiohttp import web
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        try:
            await resp.prepare(request)
        except Exception:
            logger.debug("openai_compat: SSE prepare failed", exc_info=True)
            return resp

        async def _send(frame: dict[str, Any]) -> None:
            data = "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
            try:
                await resp.write(data.encode("utf-8"))
            except (ConnectionResetError, BrokenPipeError):
                raise
            except Exception:
                logger.debug("SSE write failed", exc_info=True)
                raise

        opener = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self._openai_model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                },
            ],
        }
        try:
            await _send(opener)
            async for token, done in stream:
                if token:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self._openai_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            },
                        ],
                    }
                    await _send(chunk)
                if done:
                    break
            done_frame = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self._openai_model_id,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                ],
            }
            await _send(done_frame)
            try:
                await resp.write(b"data: [DONE]\n\n")
            except Exception:
                logger.debug("SSE [DONE] flush failed", exc_info=True)
        except (ConnectionResetError, BrokenPipeError):
            logger.debug("openai_compat: client disconnected mid-stream")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("openai_compat: stream loop crashed")
        try:
            await resp.write_eof()
        except Exception:
            logger.debug("openai_compat: write_eof raised", exc_info=True)
        return resp


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
