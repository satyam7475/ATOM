"""
ATOM v14 -- JARVIS-style Web Dashboard.

Replaces the Tkinter UI with a browser-based dashboard served over
aiohttp (HTTP + WebSocket).  The dashboard features a Three.js animated
orb, real-time system status panels, activity monitor, conversation
log, and live performance mode switcher -- all pushed via WebSocket.

Public API mirrors FloatingIndicator so main.py wiring is a drop-in swap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from aiohttp import web

if TYPE_CHECKING:
    from core.state_manager import AtomState

logger = logging.getLogger("atom.ui")

STATE_META: dict[str, dict[str, str]] = {
    "sleep":          {"label": "SLEEP",     "status": "Ctrl+Alt+A or UNSTICK to resume"},
    "idle":           {"label": "IDLE",      "status": "Ready — speak when listening"},
    "listening":      {"label": "LISTENING", "status": "Listening..."},
    "thinking":       {"label": "THINKING",  "status": "Processing..."},
    "speaking":       {"label": "SPEAKING",  "status": "Speaking..."},
    "error_recovery": {"label": "RECOVERY",  "status": "Recovering..."},
}

_DASHBOARD_DIR = Path(__file__).parent / "dashboard"


class WebDashboard:
    """JARVIS-style web dashboard served over aiohttp with live WebSocket."""

    def __init__(
        self,
        mic_name: str = "Detecting...",
        port: int = 8765,
        auto_open: bool = True,
    ) -> None:
        self._mic_name = mic_name
        self._port = port
        self._auto_open = auto_open
        self._current_state = "sleep"
        self._clients: set[web.WebSocketResponse] = set()
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._system_task: asyncio.Task | None = None
        self._shutdown_callback: Any = None
        self._unstick_callback: Callable[[], Awaitable[None]] | None = None
        self._mode_change_callback: Callable[[str], None] | None = None
        self._personality_mode_callback: Callable[[str], None] | None = None
        self._text_input_callback: Callable[[str], Awaitable[None]] | None = None
        self._brain_mode_mgr: Any = None
        self._assistant_mode_mgr: Any = None
        self._security_policy: Any = None
        self._init_info: dict[str, str] = {}
        self._owner_detected = False
        self._owner_status = "disabled"
        self._last_query = ""
        self._last_intent = ""
        self._last_latency_ms: float | None = None
        self._activity_log: list[dict] = []
        self._conv_log: list[dict] = []

    # ── Security ──────────────────────────────────────────────────────

    @web.middleware
    async def _security_headers(self, request: web.Request,
                                handler) -> web.StreamResponse:
        resp = await handler(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://esm.sh https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' ws://127.0.0.1:*; "
            "img-src 'self' data:; "
        )
        return resp

    # ── Startup ─────────────────────────────────────────────────────

    async def start(self) -> None:
        self._app = web.Application(middlewares=[self._security_headers])
        self._app.router.add_get("/ws", self._ws_handler)
        self._app.router.add_get("/", self._serve_dashboard)
        self._app.router.add_static("/static", _DASHBOARD_DIR, show_index=False)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await self._site.start()

        url = f"http://127.0.0.1:{self._port}/"
        logger.info("Web dashboard running at %s", url)

        if self._auto_open:
            opened = False
            if sys.platform == "win32":
                try:
                    subprocess.Popen(
                        ["cmd", "/c", "start", "", url],
                        shell=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                    )
                    opened = True
                except OSError as e:
                    logger.warning("Could not launch default browser via cmd /c start: %s", e)
            if not opened:
                try:
                    webbrowser.open(url)
                    opened = True
                except Exception as e:
                    logger.warning("webbrowser.open failed: %s", e)
            if not opened:
                logger.warning(
                    "Browser did not open automatically. In Cursor: "
                    "Ctrl+Shift+P → 'Simple Browser: Show' → paste %s",
                    url,
                )

        self._system_task = asyncio.create_task(self._push_system_info_loop())

    async def _serve_dashboard(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(_DASHBOARD_DIR / "index.html")

    # ── WebSocket ───────────────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        origin = request.headers.get("Origin", "")
        if origin:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            host = parsed.hostname or ""
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise web.HTTPForbidden(text="WebSocket connections from external origins are blocked")
        ws = web.WebSocketResponse(heartbeat=15.0)
        await ws.prepare(request)
        self._clients.add(ws)
        logger.info("Dashboard client connected (%d total)", len(self._clients))

        await self._send_one(ws, {
            "type": "init",
            **self._init_info,
            "mic": self._mic_name,
            "state": self._current_state,
            "owner_detected": self._owner_detected,
            "owner_status": self._owner_status,
            "last_query": self._last_query,
            "last_intent": self._last_intent,
            "last_latency_ms": self._last_latency_ms,
            **STATE_META.get(self._current_state, {}),
        })

        for entry in self._activity_log[-50:]:
            await self._send_one(ws, {**entry, "type": "activity"})
        for entry in self._conv_log[-100:]:
            await self._send_one(ws, {**entry, "type": "log"})

        if hasattr(self, "_last_habits") and self._last_habits:
            await self._send_one(ws, {
                "type": "habits_update",
                "habits": self._last_habits,
            })

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    action = data.get("action")
                    msg_type = data.get("type", data.get("action", ""))
                    if msg_type == "shutdown" and self._shutdown_callback:
                        self._shutdown_callback()
                    elif msg_type == "unstick" and self._unstick_callback:
                        try:
                            await self._unstick_callback()
                        except Exception:
                            logger.exception("Dashboard unstick callback failed")
                    elif msg_type == "change_mode":
                        new_mode = data.get("mode", "")
                        if new_mode in ("full", "lite", "ultra_lite"):
                            logger.info("UI requested mode change -> %s", new_mode)
                            await self._broadcast({
                                "type": "restarting",
                                "mode": new_mode,
                            })
                            if self._mode_change_callback:
                                self._mode_change_callback(new_mode)
                    elif msg_type == "switch_mode":
                        personality_mode = data.get("mode", "")
                        if personality_mode and self._personality_mode_callback:
                            self._personality_mode_callback(personality_mode)
                    elif msg_type == "set_brain_profile":
                        profile = (data.get("profile") or "").strip().lower()
                        sec = self._security_policy
                        if sec is not None and not sec.can_switch_runtime_modes():
                            await self._send_one(ws, {
                                "type": "runtime_error",
                                "message": "Brain profile changes are disabled (lock or config).",
                            })
                        elif profile in ("atom", "balanced", "brain") and self._brain_mode_mgr:
                            ok, _msg = self._brain_mode_mgr.set_profile(profile)
                            if ok:
                                am = self._assistant_mode_mgr.active if self._assistant_mode_mgr else ""
                                await self.broadcast_runtime_settings(
                                    self._brain_mode_mgr.active_profile, am,
                                )
                    elif msg_type == "set_assistant_mode":
                        mode = (data.get("mode") or "").strip().lower().replace(" ", "_")
                        sec = self._security_policy
                        if sec is not None and not sec.can_switch_runtime_modes():
                            await self._send_one(ws, {
                                "type": "runtime_error",
                                "message": "Assistant mode changes are disabled (lock or config).",
                            })
                        elif mode in ("command_only", "hybrid", "conversational"):
                            if self._assistant_mode_mgr:
                                ok, _msg = self._assistant_mode_mgr.set_mode(mode)
                                if ok:
                                    bp = self._brain_mode_mgr.active_profile if self._brain_mode_mgr else ""
                                    await self.broadcast_runtime_settings(
                                        bp, self._assistant_mode_mgr.active,
                                    )
                    elif msg_type == "text_input":
                        text = (data.get("text") or "").strip()
                        if text and self._text_input_callback:
                            try:
                                await self._text_input_callback(text)
                            except Exception:
                                logger.exception("Text input callback failed")
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            logger.info("Dashboard client disconnected (%d remaining)",
                        len(self._clients))
        return ws

    async def _send_one(self, ws: web.WebSocketResponse, data: dict) -> None:
        try:
            if not ws.closed:
                await ws.send_json(data)
        except Exception:
            self._clients.discard(ws)

    async def _broadcast(self, data: dict) -> None:
        if not self._clients:
            return
        dead: list[web.WebSocketResponse] = []

        async def _send(ws: web.WebSocketResponse) -> None:
            try:
                if ws.closed:
                    dead.append(ws)
                else:
                    await ws.send_json(data)
            except Exception:
                dead.append(ws)

        await asyncio.gather(*(_send(ws) for ws in self._clients))
        for ws in dead:
            self._clients.discard(ws)

    # ── Periodic system info push ───────────────────────────────────

    async def _push_system_info_loop(self) -> None:
        try:
            import psutil
        except ImportError:
            return

        while True:
            try:
                cpu = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage("C:\\")
                battery = psutil.sensors_battery()

                payload: dict[str, Any] = {
                    "type": "system",
                    "cpu": round(cpu, 1),
                    "ram": round(mem.percent, 1),
                    "ram_used_gb": round(mem.used / (1024 ** 3), 1),
                    "ram_total_gb": round(mem.total / (1024 ** 3), 1),
                    "disk_free_gb": round(disk.free / (1024 ** 3), 0),
                    "disk_total_gb": round(disk.total / (1024 ** 3), 0),
                }
                if battery:
                    payload["battery"] = round(battery.percent)
                    payload["charging"] = battery.power_plugged

                await self._broadcast(payload)
            except Exception as exc:
                logger.debug("System info push error: %s", exc)

            await asyncio.sleep(5.0)

    # ── Public API (mirrors FloatingIndicator) ──────────────────────

    def set_init_info(self, **kwargs: Any) -> None:
        """Init payload for WebSocket clients (strings + optional runtime fields)."""
        self._init_info.update({k: str(v) if v is not None else "" for k, v in kwargs.items()})

    def update_state(self, state_value: str) -> None:
        self._current_state = state_value
        meta = STATE_META.get(state_value, STATE_META["sleep"])

        entry = {
            "state": state_value,
            "label": meta["label"],
            "status": meta["status"],
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self._activity_log.append(entry)
        if len(self._activity_log) > 200:
            self._activity_log = self._activity_log[-100:]

        asyncio.ensure_future(self._broadcast({
            "type": "state",
            **entry,
        }))

    def add_log(self, tag: str, message: str) -> None:
        entry = {
            "tag": tag,
            "message": message,
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self._conv_log.append(entry)
        if len(self._conv_log) > 500:
            self._conv_log = self._conv_log[-300:]

        asyncio.ensure_future(self._broadcast({
            "type": "log",
            **entry,
        }))

    def show_hearing(self, text: str) -> None:
        asyncio.ensure_future(self._broadcast({
            "type": "hearing",
            "text": text[:80],
        }))

    def clear_hearing(self) -> None:
        asyncio.ensure_future(self._broadcast({"type": "hearing", "text": ""}))

    def set_mic_name(self, name: str) -> None:
        self._mic_name = name
        asyncio.ensure_future(self._broadcast({
            "type": "mic",
            "name": name,
        }))

    def set_owner_status(self, detected: bool = False, status: str = "disabled") -> None:
        """Push owner (face) recognition status to dashboard clients."""
        self._owner_detected = detected
        self._owner_status = status
        asyncio.ensure_future(self._broadcast({
            "type": "owner_status",
            "detected": detected,
            "status": status,
        }))

    def set_last_query(self, text: str) -> None:
        self._last_query = (text or "")[:80]
        asyncio.ensure_future(self._broadcast({
            "type": "last_query",
            "query": self._last_query,
            "intent": self._last_intent,
            "latency_ms": self._last_latency_ms,
        }))

    def set_last_intent(self, intent: str) -> None:
        self._last_intent = intent or ""
        asyncio.ensure_future(self._broadcast({
            "type": "last_query",
            "query": self._last_query,
            "intent": self._last_intent,
            "latency_ms": self._last_latency_ms,
        }))

    def set_last_latency_ms(self, ms: float) -> None:
        self._last_latency_ms = ms
        asyncio.ensure_future(self._broadcast({
            "type": "last_query",
            "query": self._last_query,
            "intent": self._last_intent,
            "latency_ms": self._last_latency_ms,
        }))

    def set_status(self, text: str) -> None:
        asyncio.ensure_future(self._broadcast({
            "type": "status_text",
            "text": text,
        }))

    def update_mic_level(self, value: float) -> None:
        pass

    def set_shutdown_callback(self, callback: Any) -> None:
        self._shutdown_callback = callback

    def set_unstick_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Async callback: stop TTS if needed and return to LISTENING (resume / unstick)."""
        self._unstick_callback = callback

    def set_mode_change_callback(self, callback: Callable[[str], None]) -> None:
        self._mode_change_callback = callback

    def set_personality_mode_callback(self, callback: Callable[[str], None]) -> None:
        self._personality_mode_callback = callback

    def set_text_input_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._text_input_callback = callback

    def attach_runtime_managers(
        self,
        brain_mode_mgr: Any,
        assistant_mode_mgr: Any,
        security_policy: Any,
    ) -> None:
        """Wire brain profile + assistant mode controls (localhost dashboard only)."""
        self._brain_mode_mgr = brain_mode_mgr
        self._assistant_mode_mgr = assistant_mode_mgr
        self._security_policy = security_policy

    async def broadcast_runtime_settings(
        self, brain_profile: str, assistant_mode: str,
    ) -> None:
        await self._broadcast({
            "type": "runtime_settings",
            "brain_profile": brain_profile,
            "assistant_mode": assistant_mode,
        })

    def broadcast_governor(self, throttled: bool) -> None:
        asyncio.ensure_future(self._broadcast({
            "type": "governor",
            "throttled": throttled,
        }))

    def broadcast_perf_mode(self, mode: str) -> None:
        asyncio.ensure_future(self._broadcast({
            "type": "perf_mode",
            "mode": mode,
        }))

    def broadcast_thinking_progress(self, elapsed_s: float, estimate_s: float) -> None:
        remaining = max(0, estimate_s - elapsed_s)
        asyncio.ensure_future(self._broadcast({
            "type": "thinking_progress",
            "elapsed_s": round(elapsed_s, 1),
            "estimate_s": round(estimate_s, 1),
            "remaining_s": round(remaining, 1),
        }))

    def broadcast_habits(self, habits: list[dict]) -> None:
        """Push active habits list to dashboard clients."""
        self._last_habits = habits
        asyncio.ensure_future(self._broadcast({
            "type": "habits_update",
            "habits": habits,
        }))

    def broadcast_autonomy_log(
        self, decision: str, detail: str = "", confidence: float = 0.0,
    ) -> None:
        """Push an autonomy decision log entry to dashboard clients."""
        asyncio.ensure_future(self._broadcast({
            "type": "autonomy_log",
            "decision": decision,
            "detail": detail,
            "confidence": round(confidence, 2),
            "timestamp": time.strftime("%H:%M:%S"),
        }))

    def broadcast_goals(self, goals: list[dict]) -> None:
        """Push goals data to dashboard clients."""
        asyncio.ensure_future(self._broadcast({
            "type": "goals_update",
            "goals": goals,
        }))

    def broadcast_predictions(self, predictions: list[dict]) -> None:
        """Push predictions to dashboard clients."""
        asyncio.ensure_future(self._broadcast({
            "type": "predictions_update",
            "predictions": predictions,
        }))

    def broadcast_profile(self, profile: dict) -> None:
        """Push user profile/energy data to dashboard clients."""
        asyncio.ensure_future(self._broadcast({
            "type": "profile_update",
            **profile,
        }))

    def broadcast_mode(self, mode_data: dict) -> None:
        """Push personality mode state to dashboard clients."""
        asyncio.ensure_future(self._broadcast({
            "type": "mode_update",
            **mode_data,
        }))

    async def on_state_changed(self, old: Any, new: Any, **_kw: Any) -> None:
        self.update_state(new.value)

    # ── Shutdown ────────────────────────────────────────────────────

    async def shutdown_async(self) -> None:
        if self._system_task:
            self._system_task.cancel()
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Web dashboard shut down")

    def shutdown(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.shutdown_async())
        except RuntimeError:
            pass
        logger.info("UI shut down")
