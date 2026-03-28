"""
Heuristic next-query prediction for RAG prefetch (no extra LLM calls).

Uses repeat patterns, task continuation, time-of-day hints, and optional
timeline / memory graph context.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

logger = __import__("logging").getLogger("atom.cognition.predictor")

_MAX_OUT = 12


def _dedupe_preserve(candidates: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in candidates:
        t = (s or "").strip()[:400]
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= cap:
            break
    return out


def predict_next_queries(context: dict[str, Any]) -> list[str]:
    """Return predicted next user queries or short intent phrases for prefetch.

    context keys (all optional):
      - last_queries: list[str] recent user utterances (newest last)
      - active_task: str or dict describing current task
      - recent_actions: list[str] tool/action names or summaries
      - timeline_snippets: list[str] short labels from TimelineMemory
      - last_project: str from MemoryGraph helper
      - recent_entities: list[dict] from MemoryGraph helper
    """
    out: list[str] = []

    last_queries = list(context.get("last_queries") or [])
    if last_queries:
        q = (last_queries[-1] or "").strip()
        if len(q) >= 3:
            out.append(q)
            out.append(f"{q} more detail")
            low = q.lower()
            if any(w in low for w in ("how", "what", "why", "explain", "debug")):
                m = re.search(r"(how|what|why)\s+(?:do|does|is|are|to)?\s*(\w+)", low)
                if m:
                    out.append(f"explain {m.group(2)}")
            if "continue" in low or "next step" in low:
                out.append("what should I do next")

    task = context.get("active_task")
    if isinstance(task, str) and task.strip():
        t = task.strip()[:200]
        out.append(f"status of {t}")
        out.append(f"continue {t}")
    elif isinstance(task, dict):
        title = (task.get("title") or task.get("name") or "").strip()
        if title:
            out.append(f"continue {title[:120]}")
            out.append(f"progress on {title[:120]}")

    for a in (context.get("recent_actions") or [])[-5:]:
        if not a:
            continue
        s = str(a).strip()[:200]
        if s:
            out.append(f"after {s} what next")

    for snip in (context.get("timeline_snippets") or [])[-4:]:
        if snip:
            out.append(str(snip)[:200])

    proj = context.get("last_project")
    if isinstance(proj, str) and proj.strip():
        p = proj.strip()[:120]
        out.append(f"{p} context")
        out.append(f"files in {p}")

    for ent in (context.get("recent_entities") or [])[:5]:
        if isinstance(ent, dict):
            label = ent.get("label") or ent.get("id") or ""
            if label:
                out.append(str(label)[:200])

    hour = datetime.now().hour
    if 5 <= hour < 10:
        out.append("morning priorities")
        out.append("today schedule")
    elif 12 <= hour < 14:
        out.append("afternoon tasks")
    elif 18 <= hour < 23:
        out.append("end of day summary")

    if len(last_queries) >= 2:
        a, b = last_queries[-2], last_queries[-1]
        if a and b and a.strip() != b.strip():
            out.append(f"{a[:60]} then {b[:60]}")

    uniq = _dedupe_preserve(out, _MAX_OUT)
    feedback_engine = context.get("feedback_engine")
    if feedback_engine is not None:
        try:
            uniq = feedback_engine.reorder_predictions(uniq)
        except Exception:
            pass
    try:
        logger.info(
            "v7_prediction count=%d sample=%s",
            len(uniq),
            uniq[:3],
        )
    except Exception:
        pass
    return uniq
