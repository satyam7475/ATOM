"""
ATOM -- Workflow Recording & Replay Engine.

"Watch what I do" -> Records a sequence of actions -> Names it -> Replays on command.

This gives ATOM JARVIS-level macro capability:
  - "Record workflow" -> starts recording all actions
  - "Stop recording" -> saves the workflow
  - "Run my morning workflow" -> replays the saved sequence

Workflows are persisted to logs/workflows.json and survive restarts.

Contract: CognitiveModuleContract (start, stop, persist)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.workflow")

_WORKFLOWS_FILE = Path("logs/workflows.json")
_MAX_WORKFLOWS = 50
_MAX_STEPS = 30


@dataclass
class WorkflowStep:
    """Single step in a recorded workflow."""
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    delay_ms: float = 0.0
    description: str = ""


@dataclass
class Workflow:
    """A named sequence of actions that can be replayed."""
    name: str
    steps: list[WorkflowStep] = field(default_factory=list)
    created_at: float = 0.0
    last_run_at: float = 0.0
    run_count: int = 0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [
                {"action": s.action, "args": s.args,
                 "delay_ms": s.delay_ms, "description": s.description}
                for s in self.steps
            ],
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "run_count": self.run_count,
            "description": self.description,
        }

    @staticmethod
    def from_dict(data: dict) -> Workflow:
        steps = [
            WorkflowStep(
                action=s["action"],
                args=s.get("args", {}),
                delay_ms=s.get("delay_ms", 0),
                description=s.get("description", ""),
            )
            for s in data.get("steps", [])
        ]
        return Workflow(
            name=data["name"],
            steps=steps,
            created_at=data.get("created_at", 0),
            last_run_at=data.get("last_run_at", 0),
            run_count=data.get("run_count", 0),
            description=data.get("description", ""),
        )


class WorkflowEngine:
    """Record, save, and replay action workflows."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = (config or {}).get("workflow", {})
        self._workflows: dict[str, Workflow] = {}
        self._recording = False
        self._record_buffer: list[WorkflowStep] = []
        self._record_name: str = ""
        self._last_action_time: float = 0.0
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if _WORKFLOWS_FILE.exists():
            try:
                data = json.loads(_WORKFLOWS_FILE.read_text(encoding="utf-8"))
                for wf_data in data.get("workflows", []):
                    wf = Workflow.from_dict(wf_data)
                    self._workflows[wf.name.lower()] = wf
                logger.info("Loaded %d workflows", len(self._workflows))
            except Exception:
                logger.debug("Workflow load failed", exc_info=True)

    def persist(self) -> None:
        if not self._dirty:
            return
        try:
            _WORKFLOWS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "workflows": [
                    wf.to_dict() for wf in self._workflows.values()
                ]
            }
            _WORKFLOWS_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
            self._dirty = False
        except Exception:
            logger.debug("Workflow persist failed", exc_info=True)

    def start_recording(self, name: str = "") -> str:
        if self._recording:
            return "Already recording a workflow, Boss. Say 'stop recording' first."
        self._recording = True
        self._record_buffer.clear()
        self._record_name = name or f"workflow_{int(time.time())}"
        self._last_action_time = time.time()
        logger.info("Started recording workflow: %s", self._record_name)
        return f"Recording workflow '{self._record_name}'. I'll capture your actions. Say 'stop recording' when done."

    def record_action(self, action: str, args: dict | None = None,
                      description: str = "") -> None:
        if not self._recording:
            return
        now = time.time()
        delay = (now - self._last_action_time) * 1000 if self._last_action_time else 0
        self._record_buffer.append(WorkflowStep(
            action=action,
            args=args or {},
            delay_ms=min(delay, 5000),
            description=description,
        ))
        self._last_action_time = now

        if len(self._record_buffer) >= _MAX_STEPS:
            self.stop_recording()

    def stop_recording(self) -> str:
        if not self._recording:
            return "Not recording right now, Boss."
        self._recording = False
        if not self._record_buffer:
            return "Nothing was recorded."

        wf = Workflow(
            name=self._record_name,
            steps=list(self._record_buffer),
            created_at=time.time(),
            description=f"Recorded workflow with {len(self._record_buffer)} steps",
        )
        self._workflows[wf.name.lower()] = wf
        self._dirty = True
        count = len(self._record_buffer)
        self._record_buffer.clear()
        logger.info("Workflow saved: '%s' (%d steps)", wf.name, count)
        return f"Workflow '{wf.name}' saved with {count} steps. Say 'run {wf.name}' to replay."

    def get_workflow(self, name: str) -> Workflow | None:
        return self._workflows.get(name.lower())

    def get_replay_steps(self, name: str) -> list[WorkflowStep] | None:
        wf = self.get_workflow(name)
        if wf is None:
            return None
        wf.last_run_at = time.time()
        wf.run_count += 1
        self._dirty = True
        return list(wf.steps)

    def list_workflows(self) -> str:
        if not self._workflows:
            return "No saved workflows yet. Say 'record workflow [name]' to create one."
        lines = ["Your saved workflows:"]
        for wf in sorted(self._workflows.values(), key=lambda w: w.last_run_at, reverse=True):
            lines.append(f"  - {wf.name} ({len(wf.steps)} steps, run {wf.run_count} times)")
        return " ".join(lines)

    def delete_workflow(self, name: str) -> str:
        key = name.lower()
        if key in self._workflows:
            del self._workflows[key]
            self._dirty = True
            return f"Workflow '{name}' deleted."
        return f"No workflow named '{name}' found."

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def workflow_count(self) -> int:
        return len(self._workflows)
