"""
Persistent plan templates with rolling stats for reuse and calibration.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("atom.brain.plan_registry")

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "plan_registry.json"


class PlanRegistry:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else _DEFAULT_PATH
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            logger.warning("plan_registry.json not found at %s — using empty registry", self.path)
            self._data = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception as e:
            logger.error("Failed to load plan registry: %s", e)
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save plan registry: %s", e)

    def match_template(self, objective: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Return (template_id, template_dict) if objective matches a template."""
        ol = objective.lower()
        best_id: Optional[str] = None
        best_score = 0
        for tid, entry in self._data.items():
            if not isinstance(entry, dict):
                continue
            matches: List[str] = entry.get("match") or []
            score = 0
            for m in matches:
                if m.lower() in ol:
                    score += len(m)
            if score > best_score:
                best_score = score
                best_id = tid
        if best_id and best_score > 0:
            return best_id, self._data[best_id]
        return None

    def record_execution(self, template_id: str, success: bool, execution_time_s: float) -> None:
        if template_id not in self._data or not isinstance(self._data[template_id], dict):
            return
        t = self._data[template_id]
        runs = int(t.get("_runs", 0)) + 1
        successes = int(t.get("_successes", 0)) + (1 if success else 0)
        t["_runs"] = runs
        t["_successes"] = successes
        t["success_rate"] = round(successes / runs, 4)
        prev_avg = float(t.get("avg_time_s", execution_time_s or 1.0))
        t["avg_time_s"] = round(prev_avg + (execution_time_s - prev_avg) / runs, 2)
        self._save()

    def get(self, template_id: str) -> Optional[Dict[str, Any]]:
        v = self._data.get(template_id)
        return v if isinstance(v, dict) else None
