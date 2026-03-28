"""
ATOM -- Premium dark UI with animated reactor orb.

880x800 frameless window with:
- Frosted dark glass aesthetic
- Large animated reactor orb with 5 pulsing rings + particle dots
- State label with glow effect beneath orb
- Scrollable conversation panel (voice I/O log)
- Mic level bar + bottom info strip
- Working minimize (to taskbar) and close (shutdown ATOM) buttons
- Auto-show on active states, hide on sleep
- Global hotkey indicator (Ctrl+Alt+A)
- Log trimming at 500 lines
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font as tkfont
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.state_manager import AtomState

logger = logging.getLogger("atom.ui")

STATE_CONFIG: dict[str, dict] = {
    "sleep":          {"colour": "#4a5568", "glow": "#1a1d24", "label": "SLEEP",     "status": "Inactive",       "accent": "#6B7280"},
    "idle":           {"colour": "#38bdf8", "glow": "#0c2d4a", "label": "IDLE",      "status": "Say 'Hey Atom'", "accent": "#7dd3fc"},
    "listening":      {"colour": "#00ffd0", "glow": "#003d32", "label": "LISTENING", "status": "Listening...",    "accent": "#00e6bb"},
    "thinking":       {"colour": "#ffb86c", "glow": "#3d2d00", "label": "THINKING",  "status": "Processing...",  "accent": "#ffa03e"},
    "speaking":       {"colour": "#a78bfa", "glow": "#2a1f50", "label": "SPEAKING",  "status": "Speaking...",    "accent": "#c4b5fd"},
    "error_recovery": {"colour": "#ff6b6b", "glow": "#3d1a1a", "label": "RECOVERY", "status": "Recovering...",  "accent": "#ff8a8a"},
}

WIN_W = 980
WIN_H = 700
BG = "#0a0e14"
BG_SURFACE = "#0f1319"
PANEL_BG = "#121820"
PANEL_BORDER = "#1c2433"
TITLE_BG = "#0a0e14"
ACCENT = "#00ffd0"
ACCENT_DIM = "#004d3d"
PURPLE = "#a78bfa"
ORANGE = "#ffb86c"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#b0bac5"
TEXT_DIM = "#8b949e"
TEXT_SUBTLE = "#3d4654"
ERROR_CLR = "#ff6b6b"
SYSTEM_CLR = "#64b5f6"
SUCCESS_CLR = "#66bb6a"
MAX_CONV_LINES = 500
TRIM_LINES = 100

ORB_SIZE = 120
ORB_CANVAS_SIZE = 260
PULSE_INTERVAL = 30
RING_COUNT = 6
PARTICLE_COUNT = 18


class FloatingIndicator:
    """Premium dark UI with animated reactor orb and conversation panel."""

    def __init__(self, mic_name: str = "Detecting...") -> None:
        self._mic_name = mic_name
        self._root: tk.Tk | None = None
        self._conv_text: tk.Text | None = None
        self._mic_bar: ttk.Progressbar | None = None
        self._state_label: tk.Label | None = None
        self._state_dot: tk.Canvas | None = None
        self._orb_canvas: tk.Canvas | None = None
        self._status_var: tk.StringVar | None = None
        self._orb_label_var: tk.StringVar | None = None
        self._mic_var: tk.StringVar | None = None
        self._current_state = "sleep"
        self._pulse_after_id: str | None = None
        self._pulse_phase = False
        self._is_pulsing = False
        self._minimized = False
        self._thinking_active = False
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._drag_data = {"x": 0, "y": 0}
        self._anim_angle = 0.0
        self._anim_after_id: str | None = None
        self._orb_rings: list[int] = []
        self._particles: list[dict] = []
        self._shutdown_callback = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_tk, daemon=True, name="ui")
        self._thread.start()
        self._ready.wait(timeout=5)
        logger.info("UI window started (%dx%d)", WIN_W, WIN_H)

    def _run_tk(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("ATOM")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg=BG)
        root.attributes("-alpha", 0.97)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - WIN_W) // 2
        y = (screen_h - WIN_H) // 2
        root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

        root.bind("<Map>", self._on_map_event)

        title_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        version_font = tkfont.Font(family="Segoe UI", size=9)
        state_font = tkfont.Font(family="Segoe UI Semibold", size=11, weight="bold")
        orb_label_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        orb_sub_font = tkfont.Font(family="Segoe UI", size=9)
        info_font = tkfont.Font(family="Segoe UI", size=8)
        btn_font = tkfont.Font(family="Segoe UI", size=11)
        try:
            conv_font = tkfont.Font(family="Cascadia Code", size=10)
            conv_font.actual()
        except Exception:
            conv_font = tkfont.Font(family="Consolas", size=10)
        conv_bold = tkfont.Font(family=conv_font.cget("family"), size=10, weight="bold")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Mic.Horizontal.TProgressbar",
                        troughcolor="#1a1f28", background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT,
                        bordercolor="#1a1f28", thickness=4)

        # --- Outer border frame for glass effect ---
        border_frame = tk.Frame(root, bg="#1c2433", bd=0)
        border_frame.pack(fill="both", expand=True, padx=1, pady=1)

        inner_frame = tk.Frame(border_frame, bg=BG)
        inner_frame.pack(fill="both", expand=True)

        # --- Title bar ---
        title_bar = tk.Frame(inner_frame, bg=TITLE_BG, height=44, cursor="fleur")
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        dot_canvas = tk.Canvas(title_bar, width=12, height=12,
                               bg=TITLE_BG, highlightthickness=0)
        dot_canvas.pack(side="left", padx=(20, 8), pady=16)
        dot_canvas.create_oval(1, 1, 11, 11, fill=ACCENT, outline="", tags="dot")
        self._state_dot = dot_canvas

        tk.Label(title_bar, text="A T O M", font=title_font,
                 fg=TEXT_PRIMARY, bg=TITLE_BG).pack(side="left")
        tk.Label(title_bar, text="  v13", font=version_font,
                 fg=TEXT_SUBTLE, bg=TITLE_BG).pack(side="left", pady=(5, 0))

        self._status_var = tk.StringVar(value="SLEEP")
        self._state_label = tk.Label(title_bar, textvariable=self._status_var,
                                     font=state_font, fg=TEXT_DIM, bg=TITLE_BG)
        self._state_label.pack(side="left", padx=(24, 0))

        btn_frame = tk.Frame(title_bar, bg=TITLE_BG)
        btn_frame.pack(side="right", padx=(0, 8))

        min_btn = tk.Label(btn_frame, text="\u2014", font=btn_font,
                           fg=TEXT_DIM, bg="#1a2030", cursor="hand2",
                           padx=14, pady=2)
        min_btn.pack(side="left", padx=(0, 4))
        min_btn.bind("<Button-1>", lambda e: self._minimize())
        min_btn.bind("<Enter>", lambda e: min_btn.configure(fg=TEXT_PRIMARY, bg="#2a3040"))
        min_btn.bind("<Leave>", lambda e: min_btn.configure(fg=TEXT_DIM, bg="#1a2030"))

        close_btn = tk.Label(btn_frame, text="\u2715", font=btn_font,
                             fg=TEXT_DIM, bg="#1a2030", cursor="hand2",
                             padx=14, pady=2)
        close_btn.pack(side="left")
        close_btn.bind("<Button-1>", lambda e: self._request_close())
        close_btn.bind("<Enter>", lambda e: close_btn.configure(fg="#ffffff", bg="#e81123"))
        close_btn.bind("<Leave>", lambda e: close_btn.configure(fg=TEXT_DIM, bg="#1a2030"))

        self._make_draggable(title_bar)

        # Separator
        tk.Frame(inner_frame, bg=PANEL_BORDER, height=1).pack(fill="x")

        # --- Orb section ---
        orb_section = tk.Frame(inner_frame, bg=BG, height=ORB_CANVAS_SIZE + 52)
        orb_section.pack(fill="x", pady=(4, 0))
        orb_section.pack_propagate(False)

        self._orb_canvas = tk.Canvas(
            orb_section, width=ORB_CANVAS_SIZE, height=ORB_CANVAS_SIZE,
            bg=BG, highlightthickness=0,
        )
        self._orb_canvas.pack()

        cx = ORB_CANVAS_SIZE // 2
        cy = ORB_CANVAS_SIZE // 2
        r = ORB_SIZE // 2

        # Outer rings (6 concentric, animated)
        self._orb_rings = []
        for i in range(RING_COUNT):
            ring_r = r + 12 + i * 8
            w = 2.0 - i * 0.2 if i < 4 else 1.0
            ring_id = self._orb_canvas.create_oval(
                cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                outline=ACCENT_DIM, width=w, tags="ring",
            )
            self._orb_rings.append(ring_id)

        # Orbiting particles (18 dots at varying distances)
        self._particles = []
        for j in range(PARTICLE_COUNT):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(r + 16, r + 58)
            px = cx + dist * math.cos(angle)
            py = cy + dist * math.sin(angle)
            pid = self._orb_canvas.create_oval(
                px - 2, py - 2, px + 2, py + 2,
                fill=ACCENT_DIM, outline="", tags="particle",
            )
            self._particles.append({
                "id": pid, "angle": angle, "dist": dist,
                "speed": random.uniform(0.004, 0.022),
                "size": random.uniform(1.5, 3.5),
                "orbit": j % 3,
            })

        # Three-layer glow (outer -> mid -> inner for depth)
        for g_offset, g_tag in [(10, "orb_glow_outer"), (5, "orb_glow_mid"), (0, "orb_glow")]:
            gr = r + g_offset
            self._orb_canvas.create_oval(
                cx - gr, cy - gr, cx + gr, cy + gr,
                fill=ACCENT_DIM, outline="", tags=g_tag,
            )

        # Core orb (the bright center)
        core_r = r - 3
        self._orb_canvas.create_oval(
            cx - core_r, cy - core_r, cx + core_r, cy + core_r,
            fill=ACCENT, outline="", tags="orb_core",
        )

        # Inner reactor crosshair lines (4 arcs for Tony Stark feel)
        self._reactor_arcs = []
        for arc_i in range(4):
            arc_r = r * 0.55
            arc_id = self._orb_canvas.create_arc(
                cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r,
                start=arc_i * 90 + 10, extent=70,
                style="arc", outline=ACCENT_DIM, width=2, tags="reactor_arc",
            )
            self._reactor_arcs.append(arc_id)

        # Inner dot (reactor center point)
        self._orb_canvas.create_oval(
            cx - 4, cy - 4, cx + 4, cy + 4,
            fill=ACCENT, outline="", tags="reactor_center",
        )

        # Label area below orb
        label_area = tk.Frame(orb_section, bg=BG)
        label_area.pack(pady=(4, 0))

        self._orb_label_var = tk.StringVar(value="SLEEP")
        tk.Label(label_area, textvariable=self._orb_label_var,
                 font=orb_label_font, fg=TEXT_DIM, bg=BG).pack()

        self._orb_sub_var = tk.StringVar(value="")
        tk.Label(label_area, textvariable=self._orb_sub_var,
                 font=orb_sub_font, fg=TEXT_SUBTLE, bg=BG).pack()

        # Separator
        tk.Frame(inner_frame, bg=PANEL_BORDER, height=1).pack(fill="x", padx=20, pady=(4, 0))

        # --- Conversation panel ---
        conv_outer = tk.Frame(inner_frame, bg=PANEL_BORDER, bd=0)
        conv_outer.pack(fill="both", expand=True, padx=18, pady=(8, 6))

        conv_frame = tk.Frame(conv_outer, bg=PANEL_BG, bd=0)
        conv_frame.pack(fill="both", expand=True, padx=1, pady=1)

        scrollbar = tk.Scrollbar(conv_frame, bg="#1a2030", troughcolor=PANEL_BG,
                                 activebackground=TEXT_SUBTLE, highlightthickness=0,
                                 bd=0, width=8)
        scrollbar.pack(side="right", fill="y")

        self._conv_text = tk.Text(
            conv_frame, bg=PANEL_BG, fg=TEXT_DIM, font=conv_font,
            wrap="word", bd=0, padx=14, pady=10,
            insertbackground=PANEL_BG, cursor="arrow",
            highlightthickness=0, state="disabled",
            spacing1=3, spacing3=3,
            yscrollcommand=scrollbar.set,
            undo=False, autoseparators=False, maxundo=0,
        )
        self._conv_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._conv_text.yview)

        self._conv_text.tag_configure("user_prefix", foreground=ACCENT, font=conv_bold)
        self._conv_text.tag_configure("user_msg", foreground="#b2ffe8")
        self._conv_text.tag_configure("atom_prefix", foreground=PURPLE, font=conv_bold)
        self._conv_text.tag_configure("atom_msg", foreground=TEXT_PRIMARY)
        self._conv_text.tag_configure("system", foreground=TEXT_DIM)
        self._conv_text.tag_configure("thinking", foreground=ORANGE)
        self._conv_text.tag_configure("error", foreground=ERROR_CLR)
        self._conv_text.tag_configure("timestamp", foreground=TEXT_SUBTLE)
        self._conv_text.tag_configure("info_tag", foreground=SYSTEM_CLR)
        self._conv_text.tag_configure("success", foreground=SUCCESS_CLR)

        # --- Bottom section ---
        tk.Frame(inner_frame, bg=PANEL_BORDER, height=1).pack(fill="x", padx=18)

        bottom = tk.Frame(inner_frame, bg=BG)
        bottom.pack(fill="x", padx=18, pady=(6, 8))

        mic_row = tk.Frame(bottom, bg=BG)
        mic_row.pack(fill="x", pady=(0, 4))
        tk.Label(mic_row, text="MIC", font=info_font, fg=TEXT_SUBTLE, bg=BG).pack(side="left", padx=(0, 6))
        self._mic_bar = ttk.Progressbar(
            mic_row, orient="horizontal", length=200, mode="determinate",
            style="Mic.Horizontal.TProgressbar", maximum=100,
        )
        self._mic_bar.pack(side="left", fill="x", expand=True)
        self._mic_bar["value"] = 0

        info_row = tk.Frame(bottom, bg=BG)
        info_row.pack(fill="x")

        self._mic_var = tk.StringVar(value=f"Mic: {self._mic_name}")
        tk.Label(info_row, textvariable=self._mic_var, font=info_font,
                 fg=TEXT_SUBTLE, bg=BG, anchor="w").pack(side="left")
        tk.Label(info_row, text="ATOM  |  Ctrl+Alt+A", font=info_font,
                 fg=TEXT_SUBTLE, bg=BG, anchor="e").pack(side="right")

        root.withdraw()
        self._ready.set()
        root.mainloop()

    # --- Draggable title bar ---

    def _make_draggable(self, widget: tk.Widget) -> None:
        def on_press(e):
            self._drag_data["x"] = e.x
            self._drag_data["y"] = e.y
        def on_drag(e):
            x = self._root.winfo_x() + e.x - self._drag_data["x"]
            y = self._root.winfo_y() + e.y - self._drag_data["y"]
            self._root.geometry(f"+{x}+{y}")
        widget.bind("<Button-1>", on_press)
        widget.bind("<B1-Motion>", on_drag)
        for child in widget.winfo_children():
            child.bind("<Button-1>", on_press)
            child.bind("<B1-Motion>", on_drag)

    def _minimize(self) -> None:
        """Minimize to taskbar by toggling overrideredirect off, then iconify."""
        if not self._root:
            return
        self._minimized = True
        self._root.overrideredirect(False)
        self._root.iconify()

    def _on_map_event(self, event=None) -> None:
        """Restore frameless look when window is un-minimized from taskbar."""
        if not self._root or not self._minimized:
            return
        self._minimized = False
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.lift()

    def _show_window(self) -> None:
        if not self._root:
            return
        if self._minimized:
            self._root.overrideredirect(True)
            self._minimized = False
        self._root.deiconify()
        self._root.attributes("-topmost", True)
        self._root.lift()

    def _hide_window(self) -> None:
        if self._root:
            self._root.withdraw()

    def attach_runtime_managers(self, *args, **kwargs) -> None:
        """No-op: WebDashboard uses this for assistant/brain toggles; desktop UI uses voice + config."""

    def set_shutdown_callback(self, callback) -> None:
        self._shutdown_callback = callback

    def _request_close(self) -> None:
        """Close button: trigger full ATOM shutdown."""
        if self._shutdown_callback:
            self._shutdown_callback()
        else:
            self.shutdown()
        import os
        os._exit(0)

    # --- Orb animation ---

    def _start_orb_anim(self) -> None:
        self._anim_angle = 0.0
        self._orb_anim_tick()

    def _stop_orb_anim(self) -> None:
        if self._anim_after_id is not None:
            try:
                self._root.after_cancel(self._anim_after_id)
            except Exception:
                pass
            self._anim_after_id = None

    def _orb_anim_tick(self) -> None:
        if self._root is None or self._orb_canvas is None:
            return

        self._anim_angle += 0.055
        a = self._anim_angle
        cx = ORB_CANVAS_SIZE // 2
        cy = ORB_CANVAS_SIZE // 2
        r = ORB_SIZE // 2
        state = self._current_state
        cfg = STATE_CONFIG.get(state, STATE_CONFIG["sleep"])
        colour = cfg["colour"]
        glow = cfg["glow"]
        accent = cfg.get("accent", colour)

        # --- Animate rings (6 concentric, phase-offset breathing) ---
        for i, ring_id in enumerate(self._orb_rings):
            phase = a + i * 0.65
            ring_r = r + 12 + i * 8
            if state == "listening":
                breathe = 1.0 + 0.08 * math.sin(phase * 1.1)
            elif state == "thinking":
                breathe = 1.0 + 0.14 * math.sin(phase * 2.4)
            elif state == "speaking":
                breathe = 1.0 + 0.06 * math.sin(phase * 1.6)
            else:
                breathe = 1.0 + 0.01 * math.sin(phase * 0.3)

            rr = ring_r * breathe
            intensity = max(0.10, 0.70 - i * 0.09 + 0.20 * math.sin(phase))
            ring_clr = self._blend(glow, colour, intensity)
            w = max(0.8, 2.2 - i * 0.25 + 0.4 * math.sin(phase))

            self._orb_canvas.coords(ring_id, cx - rr, cy - rr, cx + rr, cy + rr)
            self._orb_canvas.itemconfig(ring_id, outline=ring_clr, width=w)

        # --- Animate particles (orbit + wobble + fade) ---
        for p in self._particles:
            speed_mul = {"thinking": 3.5, "speaking": 2.2, "listening": 1.3}.get(state, 0.25)
            p["angle"] += p["speed"] * speed_mul

            orbit = p.get("orbit", 0)
            wobble_freq = 1.8 + orbit * 0.5
            wobble = 1.0 + 0.18 * math.sin(a * wobble_freq + p["angle"] * 2.5)
            dist = p["dist"] * wobble
            px = cx + dist * math.cos(p["angle"])
            py = cy + dist * math.sin(p["angle"])

            pulse = 0.7 + 0.5 * math.sin(a * 1.8 + p["angle"] * 1.3)
            sz = p["size"] * pulse

            bright = max(0.12, 0.35 + 0.55 * math.sin(a * 1.4 + p["angle"]))
            p_clr = self._blend(glow, accent, bright)

            self._orb_canvas.coords(p["id"], px - sz, py - sz, px + sz, py + sz)
            self._orb_canvas.itemconfig(p["id"], fill=p_clr)

        # --- Animate three glow layers ---
        for g_offset, g_tag, g_breathe_amp in [(10, "orb_glow_outer", 0.04), (5, "orb_glow_mid", 0.025), (0, "orb_glow", 0.015)]:
            gr = (r + g_offset) * (1.0 + g_breathe_amp * math.sin(a * 0.7 + g_offset))
            self._orb_canvas.coords(g_tag, cx - gr, cy - gr, cx + gr, cy + gr)
            glow_bright = 0.6 + 0.3 * math.sin(a * 0.5 + g_offset * 0.3)
            self._orb_canvas.itemconfig(g_tag, fill=self._blend(BG, glow, glow_bright))

        # --- Animate core orb ---
        if state == "listening":
            core_scale = 1.0 + 0.045 * math.sin(a * 1.3)
        elif state == "thinking":
            core_scale = 1.0 + 0.07 * math.sin(a * 3.0)
        elif state == "speaking":
            core_scale = 1.0 + 0.035 * math.sin(a * 2.0)
        else:
            core_scale = 1.0 + 0.008 * math.sin(a * 0.4)

        cr = (r - 3) * core_scale
        self._orb_canvas.coords("orb_core", cx - cr, cy - cr, cx + cr, cy + cr)
        core_bright = 0.85 + 0.15 * math.sin(a * 1.1)
        self._orb_canvas.itemconfig("orb_core", fill=self._blend(glow, colour, core_bright))

        # --- Animate reactor arcs (rotate slowly) ---
        if hasattr(self, "_reactor_arcs"):
            arc_speed = {"thinking": 2.5, "speaking": 1.5, "listening": 0.8}.get(state, 0.15)
            arc_offset = (a * arc_speed * 57.3) % 360
            for arc_i, arc_id in enumerate(self._reactor_arcs):
                start = arc_i * 90 + 10 + arc_offset
                arc_bright = 0.4 + 0.4 * math.sin(a * 1.5 + arc_i * 1.57)
                arc_clr = self._blend(glow, accent, arc_bright)
                self._orb_canvas.itemconfig(arc_id, start=start, outline=arc_clr)

        # --- Animate reactor center dot ---
        center_pulse = 3 + 2 * math.sin(a * 2.0)
        self._orb_canvas.coords("reactor_center",
                                cx - center_pulse, cy - center_pulse,
                                cx + center_pulse, cy + center_pulse)
        center_bright = 0.7 + 0.3 * math.sin(a * 2.5)
        self._orb_canvas.itemconfig("reactor_center", fill=self._blend(glow, colour, center_bright))

        try:
            self._anim_after_id = self._root.after(PULSE_INTERVAL, self._orb_anim_tick)
        except Exception:
            pass

    @staticmethod
    def _blend(c1: str, c2: str, t: float) -> str:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

    # --- Title bar dot pulse ---

    def _start_pulse(self) -> None:
        self._is_pulsing = True
        self._pulse_phase = False
        self._pulse_tick()

    def _stop_pulse(self) -> None:
        self._is_pulsing = False
        if self._pulse_after_id is not None:
            try:
                self._root.after_cancel(self._pulse_after_id)
            except Exception:
                pass
            self._pulse_after_id = None
        if self._state_dot:
            cfg = STATE_CONFIG.get(self._current_state, STATE_CONFIG["sleep"])
            self._state_dot.itemconfig("dot", fill=cfg["colour"])

    def _pulse_tick(self) -> None:
        if not self._is_pulsing or not self._root or not self._state_dot:
            return
        self._pulse_phase = not self._pulse_phase
        cfg = STATE_CONFIG.get(self._current_state, STATE_CONFIG["sleep"])
        colour = cfg["colour"] if self._pulse_phase else cfg.get("glow", "#1a1d24")
        self._state_dot.itemconfig("dot", fill=colour)
        try:
            self._pulse_after_id = self._root.after(500, self._pulse_tick)
        except Exception:
            pass

    # --- Mic level ---

    def update_mic_level(self, value: float) -> None:
        root = self._root
        if root is None or self._mic_bar is None:
            return
        try:
            level = max(0, min(100, int(value * 100)))
            root.after(0, lambda: self._mic_bar.configure(value=level))
        except RuntimeError:
            pass

    # --- Conversation panel ---

    def _insert_conv(self, *args_list: tuple) -> None:
        if self._conv_text is None:
            return
        self._conv_text.configure(state="normal")
        for text, tags in args_list:
            self._conv_text.insert("end", text, tags)
        self._conv_text.insert("end", "\n")
        self._trim_conv()
        self._conv_text.see("end")
        self._conv_text.configure(state="disabled")

    def _trim_conv(self) -> None:
        if self._conv_text is None:
            return
        line_count = int(self._conv_text.index("end-1c").split(".")[0])
        if line_count > MAX_CONV_LINES:
            self._conv_text.delete("1.0", f"{TRIM_LINES}.0")

    def _remove_thinking_placeholder(self) -> None:
        if not self._thinking_active or self._conv_text is None:
            return
        self._conv_text.configure(state="normal")
        search_result = self._conv_text.search("ATOM is thinking...", "1.0", "end")
        if search_result:
            line_start = search_result.split(".")[0] + ".0"
            line_end = str(int(search_result.split(".")[0]) + 1) + ".0"
            self._conv_text.delete(line_start, line_end)
        self._conv_text.configure(state="disabled")
        self._thinking_active = False

    # --- Public: add_log ---

    def add_log(self, tag: str, message: str) -> None:
        root = self._root
        if root is None:
            return
        try:
            root.after(0, self._route_log, tag, message)
        except RuntimeError:
            pass

    def _route_log(self, tag: str, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        ts_segment = (f"{ts}  ", ("timestamp",))

        if tag == "heard":
            self._remove_thinking_placeholder()
            self._insert_conv(
                ts_segment,
                ("YOU >> ", ("user_prefix",)),
                (message, ("user_msg",)),
            )
        elif tag == "action" and not message.startswith("[stream]") and not message.startswith("Thinking with local brain") and not message.startswith("Running:"):
            self._remove_thinking_placeholder()
            self._insert_conv(
                ts_segment,
                ("ATOM >> ", ("atom_prefix",)),
                (message, ("atom_msg",)),
            )
        elif tag == "action" and message.startswith("[stream]"):
            clean = message[len("[stream]"):].strip()
            if clean:
                self._remove_thinking_placeholder()
                self._insert_conv(
                    ts_segment,
                    ("ATOM >> ", ("atom_prefix",)),
                    (clean, ("atom_msg",)),
                )
        elif tag == "action" and message.startswith("Thinking with local brain"):
            self._thinking_active = True
            self._insert_conv(
                ts_segment,
                ("ATOM is thinking...", ("thinking",)),
            )
        elif tag == "action" and message.startswith("Running:"):
            self._insert_conv(
                ts_segment,
                ("[CMD] ", ("info_tag",)),
                (message, ("system",)),
            )
        elif tag == "intent":
            self._insert_conv(
                ts_segment,
                ("[INTENT] ", ("info_tag",)),
                (message, ("system",)),
            )
        elif tag == "error":
            self._insert_conv(
                ts_segment,
                ("[ERROR] ", ("error",)),
                (message, ("error",)),
            )
        elif tag == "info":
            self._insert_conv(
                ts_segment,
                (f"[INFO] {message}", ("system",)),
            )
        else:
            self._insert_conv(
                ts_segment,
                (f"[{tag.upper()}] {message}", ("system",)),
            )

    # --- State management ---

    def update_state(self, state_value: str) -> None:
        cfg = STATE_CONFIG.get(state_value, STATE_CONFIG["sleep"])
        root = self._root
        if root is None:
            return
        try:
            root.after(0, self._apply_state, cfg, state_value)
        except RuntimeError:
            pass

    def _apply_state(self, cfg: dict, state_value: str) -> None:
        old_state = self._current_state
        self._current_state = state_value
        colour = cfg["colour"]

        if state_value == "sleep":
            self._stop_pulse()
            self._stop_orb_anim()
            self._hide_window()
            return

        self._show_window()

        if self._state_dot:
            self._state_dot.itemconfig("dot", fill=colour)

        if self._status_var:
            self._status_var.set(f"{cfg['label']}  --  {cfg['status']}")

        if self._state_label:
            self._state_label.configure(fg=colour)

        if self._orb_label_var:
            self._orb_label_var.set(cfg["label"])

        if hasattr(self, "_orb_sub_var") and self._orb_sub_var:
            self._orb_sub_var.set(cfg["status"])

        if state_value == "listening":
            self._start_pulse()
        else:
            self._stop_pulse()

        if old_state == "sleep":
            self._start_orb_anim()

    def set_mic_name(self, name: str) -> None:
        self._mic_name = name
        root = self._root
        if root is None:
            return
        try:
            root.after(0, lambda: self._mic_var and self._mic_var.set(f"Mic: {name}"))
        except RuntimeError:
            pass

    def set_status(self, text: str) -> None:
        root = self._root
        if root is None:
            return
        try:
            root.after(0, lambda: self._status_var and self._status_var.set(text))
        except RuntimeError:
            pass

    def show_hearing(self, text: str) -> None:
        """Show live partial transcription under the orb."""
        root = self._root
        if root is None:
            return
        try:
            display = text[:60] + "..." if len(text) > 60 else text
            root.after(0, lambda: self._orb_sub_var and self._orb_sub_var.set(f'"{display}"'))
        except RuntimeError:
            pass

    def clear_hearing(self) -> None:
        """Clear the live hearing text after final STT."""
        root = self._root
        if root is None:
            return
        cfg = STATE_CONFIG.get(self._current_state, STATE_CONFIG["sleep"])
        try:
            root.after(0, lambda: self._orb_sub_var and self._orb_sub_var.set(cfg.get("status", "")))
        except RuntimeError:
            pass

    async def on_state_changed(self, old, new, **_kw) -> None:
        self.update_state(new.value)

    def shutdown(self) -> None:
        self._stop_pulse()
        self._stop_orb_anim()
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
        logger.info("UI shut down")
