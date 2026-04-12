"""
ATOM -- Native macOS JARVIS UI.

Premium NSWindow with transparent title bar, NSVisualEffectView (Dark Vibrant),
and embedded WKWebView rendering the reactor-orb dashboard. Zero HTTP, zero
browser, zero port exposure -- pure AppKit + WebKit via multiprocessing.

Uses NSWindowStyleMaskTitled + FullSizeContentView with a transparent title
bar so the window is draggable like any native macOS app while keeping a
fully custom appearance.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state_manager import AtomState

logger = logging.getLogger("atom.ui.native")

WIN_W = 960
WIN_H = 700
_HTML_NAME = "native_dashboard.html"
_DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


# ═══════════════════════════════════════════════════════════════════════
#  AppKit Subprocess Worker
# ═══════════════════════════════════════════════════════════════════════

def _appkit_subprocess_worker(conn: mp.connection.Connection,
                              html_dir: str, html_name: str,
                              win_w: int, win_h: int) -> None:
    """Runs entirely isolated in its own process main thread."""
    try:
        import objc
        from AppKit import (
            NSApplication, NSWindow, NSBackingStoreBuffered,
            NSVisualEffectView, NSScreen,
            NSColor, NSApplicationActivationPolicyAccessory,
        )
        from Foundation import (
            NSObject, NSURL, NSMakeRect, NSRunLoop, NSDate,
        )
        from WebKit import (
            WKWebView, WKWebViewConfiguration, WKUserContentController,
        )

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        # ── Script Handler ──
        WKScriptMessageHandler = objc.protocolNamed("WKScriptMessageHandler")

        class ATOMScriptHandler(NSObject, protocols=[WKScriptMessageHandler]):
            @objc.typedSelector(b"v@:@@")
            def userContentController_didReceiveScriptMessage_(self, controller, message):
                try:
                    b = message.body()
                    if hasattr(b, "objectForKey_"):
                        action = str(b.objectForKey_("action"))
                    elif isinstance(b, dict):
                        action = str(b.get("action", ""))
                    else:
                        action = str(b)
                    conn.send({"type": "js_call", "action": action})
                except Exception:
                    pass

        handler = ATOMScriptHandler.alloc().init()

        # ── WKWebView Configuration ──
        wk_config = WKWebViewConfiguration.alloc().init()
        cc = WKUserContentController.alloc().init()
        cc.addScriptMessageHandler_name_(handler, "atomBridge")
        wk_config.setUserContentController_(cc)
        prefs = wk_config.preferences()
        prefs.setValue_forKey_(True, "developerExtrasEnabled")
        try:
            wk_config.setValue_forKey_(True, "allowFileAccessFromFileURLs")
            wk_config.setValue_forKey_(True, "allowUniversalAccessFromFileURLs")
            prefs.setValue_forKey_(True, "allowFileAccessFromFileURLs")
            prefs.setValue_forKey_(True, "allowUniversalAccessFromFileURLs")
        except Exception:
            pass

        # ── Window ──
        screen = NSScreen.mainScreen()
        sw = screen.frame().size.width
        sh = screen.frame().size.height
        frame = NSMakeRect((sw - win_w) / 2, (sh - win_h) / 2, win_w, win_h)

        TITLED = 1 << 0
        CLOSABLE = 1 << 1
        MINIATURIZABLE = 1 << 2
        RESIZABLE = 1 << 3
        FULL_SIZE_CONTENT = 1 << 15
        style = TITLED | CLOSABLE | MINIATURIZABLE | RESIZABLE | FULL_SIZE_CONTENT

        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False,
        )
        window.setTitlebarAppearsTransparent_(True)
        window.setTitleVisibility_(1)
        window.setMovableByWindowBackground_(True)

        for i in range(3):
            btn = window.standardWindowButton_(i)
            if btn:
                btn.setHidden_(True)

        window.setAlphaValue_(0.97)
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setHasShadow_(True)
        window.setTitle_("ATOM")
        window.setCollectionBehavior_(1 << 1)

        # ── Visual Effect (blur backdrop) ──
        content_frame = window.contentView().frame()
        blur = NSVisualEffectView.alloc().initWithFrame_(content_frame)
        blur.setMaterial_(4)
        blur.setBlendingMode_(0)
        blur.setState_(1)
        blur.setWantsLayer_(True)
        if blur.layer():
            blur.layer().setCornerRadius_(12.0)
            blur.layer().setMasksToBounds_(True)
        blur.setAutoresizingMask_(18)
        window.contentView().addSubview_(blur)

        # ── WebView ──
        webview = WKWebView.alloc().initWithFrame_configuration_(
            content_frame, wk_config,
        )
        webview.setAutoresizingMask_(18)
        webview.setValue_forKey_(False, "drawsBackground")
        if hasattr(webview, "setOpaque_"):
            webview.setOpaque_(False)
        blur.addSubview_(webview)

        # ── Load HTML ──
        dashboard_dir = Path(html_dir)
        html_path = dashboard_dir / html_name
        if not html_path.exists():
            conn.send({"type": "error", "msg": f"HTML not found: {html_path}"})
            os._exit(1)

        html_url = NSURL.fileURLWithPath_(str(html_path))
        base_url = NSURL.fileURLWithPath_(str(dashboard_dir))
        webview.loadFileURL_allowingReadAccessToURL_(html_url, base_url)

        window.makeKeyAndOrderFront_(None)
        app.activateIgnoringOtherApps_(True)

        conn.send({"type": "ready"})

        # ── Run Loop + Pipe Polling ──
        run_loop = NSRunLoop.currentRunLoop()
        while True:
            if conn.poll():
                msg = conn.recv()
                if msg.get("type") == "shutdown":
                    window.close()
                    break
                elif msg.get("type") == "eval_js":
                    try:
                        webview.evaluateJavaScript_completionHandler_(msg["code"], None)
                    except Exception:
                        pass

            pool = objc.autorelease_pool()
            try:
                run_loop.runMode_beforeDate_(
                    "kCFRunLoopDefaultMode",
                    NSDate.dateWithTimeIntervalSinceNow_(0.03),
                )
            finally:
                del pool

        os._exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            conn.send({"type": "error", "msg": traceback.format_exc()})
        except Exception:
            pass
        os._exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  NativeATOMWindow  — public API
# ═══════════════════════════════════════════════════════════════════════

class NativeATOMWindow:
    def __init__(self, mic_name: str = "Detecting...", config: dict | None = None) -> None:
        self._mic_name = mic_name
        self._config = dict(config or {})
        self._shutdown_callback: Callable | None = None
        self._process = None
        self._parent_conn = None
        self._stats_thread = None

    def start(self) -> None:
        logger.info("Spawning pure AppKit subprocess...")
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._parent_conn = parent_conn

        self._process = ctx.Process(
            target=_appkit_subprocess_worker,
            args=(child_conn, str(_DASHBOARD_DIR), _HTML_NAME, WIN_W, WIN_H),
            daemon=True,
        )
        self._process.start()

        deadline = time.monotonic() + 15.0
        ready = False
        while time.monotonic() < deadline:
            if self._parent_conn.poll(0.5):
                msg = self._parent_conn.recv()
                if msg.get("type") == "ready":
                    ready = True
                    break
                elif msg.get("type") == "error":
                    logger.error("UI subprocess error: %s", msg.get("msg", "unknown"))
                    return
            if not self._process.is_alive():
                logger.error("UI subprocess exited before ready signal")
                return

        if not ready:
            logger.error("UI subprocess timed out (15s) waiting for ready")
            self._process.terminate()
            return

        def _rx_thread():
            while self._process and self._process.is_alive():
                if self._parent_conn.poll(0.5):
                    try:
                        msg = self._parent_conn.recv()
                        if msg.get("type") == "js_call":
                            action = msg.get("action")
                            if action == "shutdown":
                                if self._shutdown_callback:
                                    self._shutdown_callback()
                                else:
                                    os._exit(0)
                    except EOFError:
                        break
        threading.Thread(target=_rx_thread, daemon=True).start()

        def _stat_loop():
            try:
                import psutil
                while self._process and self._process.is_alive():
                    stats = {
                        "cpu_percent": psutil.cpu_percent(),
                        "ram_percent": psutil.virtual_memory().percent,
                    }
                    bat = psutil.sensors_battery()
                    if bat:
                        stats["battery_percent"] = bat.percent
                        stats["battery_plugged"] = bat.power_plugged
                    self.broadcast_system_stats(stats)
                    time.sleep(2.0)
            except Exception:
                pass

        self._stats_thread = threading.Thread(target=_stat_loop, daemon=True)
        self._stats_thread.start()

        logger.info("Native macOS UI started (%dx%d)", WIN_W, WIN_H)

    def shutdown(self) -> None:
        if self._parent_conn:
            try:
                self._parent_conn.send({"type": "shutdown"})
            except Exception:
                pass
        if self._process:
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()

    def _eval_js(self, js_code: str) -> None:
        if self._parent_conn:
            try:
                self._parent_conn.send({"type": "eval_js", "code": js_code})
            except Exception:
                pass

    # ── API ─────────────────────────────────────

    def update_state(self, state_value: str) -> None:
        self._eval_js(f"updateState({json.dumps(state_value)});")

    def add_log(self, tag: str, message: str) -> None:
        self._eval_js(f"addLog({json.dumps(tag)}, {json.dumps(message)});")

    def show_hearing(self, text: str) -> None:
        display = text[:70] + "..." if len(text) > 70 else text
        self._eval_js(f"showHearing({json.dumps(display)});")

    def clear_hearing(self) -> None:
        self._eval_js("clearHearing();")

    def update_mic_level(self, value: float) -> None:
        clamped = max(0.0, min(1.0, value))
        self._eval_js(f"updateMicLevel({clamped});")

    def set_mic_name(self, name: str) -> None:
        self._eval_js(f"setMicName({json.dumps(name)});")

    def set_status(self, text: str) -> None:
        self._eval_js(f"setStatusText({json.dumps(text)});")

    def set_shutdown_callback(self, callback: Callable) -> None:
        self._shutdown_callback = callback

    def broadcast_autonomy_log(self, logs: list, **kwargs: Any) -> None:
        self._eval_js(f"if (window.updateAutonomy) window.updateAutonomy({json.dumps(json.dumps(logs))});")

    def broadcast_goals(self, goals: list) -> None:
        self._cached_goals = goals
        self._push_cognitive_state()

    def broadcast_predictions(self, predictions: list) -> None:
        self._cached_predictions = predictions
        self._push_cognitive_state()

    def _push_cognitive_state(self) -> None:
        data = {"goals": getattr(self, "_cached_goals", []), "predictions": getattr(self, "_cached_predictions", [])}
        self._eval_js(f"if (window.updateGoals) window.updateGoals({json.dumps(json.dumps(data))});")

    def broadcast_memory_snapshot(self, nodes: list) -> None:
        clean_nodes = [n.__dict__ if hasattr(n, "__dict__") else n for n in nodes]
        self._eval_js(f"if (window.updateMemorySnapshot) window.updateMemorySnapshot({json.dumps(json.dumps(clean_nodes))});")

    def broadcast_brain_path(self, path_name: str, latency: float) -> None:
        self._eval_js(f"if (window.updateBrain) window.updateBrain({json.dumps(path_name)}, {json.dumps(latency)});")

    def broadcast_system_stats(self, stats: dict) -> None:
        self._eval_js(f"if (window.updateSystemStats) window.updateSystemStats({json.dumps(json.dumps(stats))});")

    async def on_state_changed(self, old: Any, new: Any, **_kw: Any) -> None:
        self.update_state(new.value)

    # ── No-Ops ─────────────────────────────────────
    def attach_runtime_managers(self, *args: Any, **kwargs: Any) -> None: pass
    def set_init_info(self, **kwargs: Any) -> None: pass
    def set_mode_change_callback(self, callback: Callable) -> None: pass
    def set_personality_mode_callback(self, callback: Callable) -> None: pass
    def set_text_input_callback(self, callback: Callable) -> None: pass
    def set_unstick_callback(self, callback: Callable) -> None: pass
    def set_v7_health_provider(self, provider: Callable) -> None: pass
    def broadcast_perf_mode(self, mode: str, **_kw: Any) -> None: pass
    def broadcast_runtime_settings(self, *args: Any, **kwargs: Any) -> None: pass
    def broadcast_habits(self, habits: list) -> None: pass
    def broadcast_mode(self, mode_data: dict) -> None: pass
    def broadcast_governor(self, throttled: bool) -> None: pass
    def broadcast_thinking_progress(self, elapsed_s: float, estimate_s: float) -> None: pass
