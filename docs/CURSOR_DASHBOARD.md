# Open the ATOM dashboard inside Cursor

If the page **does not open** when you run `main.py` (policy blocks `webbrowser`, no default browser, etc.):

## Option A — Simple Browser (built into Cursor / VS Code)

1. Start ATOM: **Terminal** → `cd ATOM` → `py -3.11 main.py`
2. Wait until the log shows: `Web dashboard running at http://127.0.0.1:8765/`
3. **Ctrl+Shift+P** → type **Simple Browser: Show**
4. Enter: `http://127.0.0.1:8765/` → Enter

The preview opens in the editor area (you can drag the tab to the **left** side if you want it there).

## Option B — Tasks (this repo)

**Ctrl+Shift+B** or **Terminal → Run Task…**

- **ATOM: Run (main.py)** — starts the server (working directory is `Corporate/ATOM`).
- **ATOM: Open dashboard in browser** — runs `start http://127.0.0.1:8765/` (Windows default browser).

Run **Run** first, wait a few seconds, then **Open dashboard** if the tab did not appear.

## Checklist if it still fails

| Check | What to do |
|--------|------------|
| Port in use | Change `ui.web_port` in `config/settings.json` (e.g. `8766`) and use that port in the URL. |
| Wrong folder | Run from the **`ATOM`** folder (where `main.py` lives). |
| Firewall | Allow **Python** on private networks, or only localhost (127.0.0.1) — ATOM binds to localhost only. |
| `py` not found | Use full path to Python 3.11 or `python -3.11 main.py`. |
