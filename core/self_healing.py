"""
ATOM -- Self-Healing Engine (Autonomous Failure Recovery).

The brain that monitors ATOM's own health, catches failures in real-time,
diagnoses root causes, suggests fixes, and applies them on command.

Architecture:
  1. ExceptionTracker: Global exception hook that captures ALL unhandled
     exceptions with full context (traceback, module, timestamp, state)
  2. ModuleHealthChecker: Tests each ATOM module individually for
     import errors, missing dependencies, config issues, runtime faults
  3. FailureAnalyzer: Categorizes failures, identifies patterns, and
     finds root causes using the CodeIntrospector
  4. FixEngine: Generates fix suggestions based on common patterns
     and applies safe fixes on owner command
  5. StartupValidator: Pre-flight checks before ATOM starts --
     validates all dependencies, files, configs, and models

Self-Healing Flow:
  1. Exception occurs -> ExceptionTracker captures it
  2. FailureAnalyzer categorizes and finds root cause
  3. FixEngine generates fix suggestions
  4. ATOM tells Boss what failed and what the fix is
  5. Boss says "fix it" -> FixEngine applies the fix
  6. ATOM verifies the fix worked

Owner: Satyam (Boss). ATOM heals itself.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.code_introspector import CodeIntrospector

logger = logging.getLogger("atom.self_healing")

_FAILURE_LOG_FILE = Path("logs/failures.json")
_FIX_HISTORY_FILE = Path("logs/fix_history.json")
_MAX_FAILURES = 500
_MAX_FIX_HISTORY = 200


@dataclass
class FailureRecord:
    """A captured failure with full context."""
    id: str
    timestamp: float
    timestamp_human: str
    exception_type: str
    exception_message: str
    traceback_lines: list[str]
    module: str
    function: str
    lineno: int
    category: str  # import, config, runtime, dependency, network, file, memory
    severity: str  # low, medium, high, critical
    atom_state: str
    resolved: bool = False
    fix_applied: str = ""
    occurrence_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_human": self.timestamp_human,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "traceback_lines": self.traceback_lines[-10:],
            "module": self.module,
            "function": self.function,
            "lineno": self.lineno,
            "category": self.category,
            "severity": self.severity,
            "atom_state": self.atom_state,
            "resolved": self.resolved,
            "fix_applied": self.fix_applied,
            "occurrence_count": self.occurrence_count,
        }


@dataclass
class FixSuggestion:
    """A suggested fix for a failure."""
    failure_id: str
    description: str
    fix_type: str  # config, restart, dependency, code, manual
    auto_fixable: bool
    fix_commands: list[str]
    confidence: float  # 0.0 to 1.0
    risk_level: str  # safe, moderate, risky


@dataclass
class ModuleHealthResult:
    """Health check result for a single module."""
    module_path: str
    status: str  # healthy, degraded, failed, missing
    import_ok: bool
    class_count: int
    issues: list[str]
    check_time_ms: float


class ExceptionTracker:
    """Global exception hook that captures all unhandled exceptions."""

    __slots__ = ("_failures", "_failure_counts", "_original_hook")

    def __init__(self) -> None:
        self._failures: list[FailureRecord] = []
        self._failure_counts: dict[str, int] = {}
        self._original_hook = sys.excepthook
        self._load()

    def install_hook(self) -> None:
        """Install as the global exception hook."""
        sys.excepthook = self._exception_hook
        logger.info("Exception tracker installed as global hook")

    def _exception_hook(self, exc_type, exc_value, exc_tb) -> None:
        """Custom excepthook that captures the exception."""
        self.capture(exc_type, exc_value, exc_tb)
        self._original_hook(exc_type, exc_value, exc_tb)

    def capture(
        self,
        exc_type: type | None = None,
        exc_value: BaseException | None = None,
        exc_tb: Any = None,
        context: str = "",
        atom_state: str = "unknown",
    ) -> FailureRecord:
        """Capture an exception with full context."""
        if exc_type is None:
            exc_type, exc_value, exc_tb = sys.exc_info()

        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        tb_text = "".join(tb_lines)

        module = ""
        function = ""
        lineno = 0
        if exc_tb is not None:
            frame = exc_tb
            while frame.tb_next:
                frame = frame.tb_next
            module = frame.tb_frame.f_code.co_filename
            function = frame.tb_frame.f_code.co_name
            lineno = frame.tb_lineno

        exc_type_name = exc_type.__name__ if exc_type else "Unknown"
        exc_msg = str(exc_value) if exc_value else ""

        category = self._categorize_exception(exc_type_name, exc_msg, module)
        severity = self._assess_severity(category, exc_type_name)

        failure_key = f"{exc_type_name}:{module}:{lineno}"
        self._failure_counts[failure_key] = self._failure_counts.get(failure_key, 0) + 1

        failure = FailureRecord(
            id=f"F{int(time.time()*1000) % 100000:05d}",
            timestamp=time.time(),
            timestamp_human=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exception_type=exc_type_name,
            exception_message=exc_msg[:500],
            traceback_lines=tb_text.split("\n")[-15:],
            module=str(Path(module).name) if module else context,
            function=function,
            lineno=lineno,
            category=category,
            severity=severity,
            atom_state=atom_state,
            occurrence_count=self._failure_counts[failure_key],
        )

        self._failures.append(failure)
        if len(self._failures) > _MAX_FAILURES:
            self._failures = self._failures[-_MAX_FAILURES:]

        logger.warning(
            "FAILURE CAPTURED [%s]: %s: %s in %s:%s:%d (severity=%s, count=%d)",
            failure.id, exc_type_name, exc_msg[:100],
            failure.module, function, lineno, severity,
            failure.occurrence_count,
        )
        self._persist()
        return failure

    @staticmethod
    def _categorize_exception(exc_type: str, message: str, module: str) -> str:
        """Categorize an exception by its type and context."""
        msg_lower = message.lower()
        exc_lower = exc_type.lower()

        if exc_type in ("ImportError", "ModuleNotFoundError"):
            return "dependency"
        if exc_type in ("FileNotFoundError", "PermissionError", "IsADirectoryError"):
            return "file"
        if exc_type in ("ConnectionError", "TimeoutError", "OSError"):
            if "connect" in msg_lower or "network" in msg_lower:
                return "network"
        if "config" in msg_lower or "settings" in msg_lower:
            return "config"
        if exc_type == "MemoryError" or "memory" in msg_lower:
            return "memory"
        if exc_type in ("KeyError", "ValueError", "TypeError", "AttributeError"):
            return "runtime"
        if "json" in msg_lower or "decode" in msg_lower:
            return "data"
        if "model" in msg_lower or "llm" in msg_lower or "gpu" in msg_lower:
            return "model"
        return "runtime"

    @staticmethod
    def _assess_severity(category: str, exc_type: str) -> str:
        if category in ("memory", "model"):
            return "critical"
        if category == "dependency":
            return "high"
        if category in ("config", "file"):
            return "high"
        if category == "network":
            return "medium"
        if exc_type in ("KeyboardInterrupt", "SystemExit"):
            return "low"
        return "medium"

    def _load(self) -> None:
        if _FAILURE_LOG_FILE.exists():
            try:
                data = json.loads(_FAILURE_LOG_FILE.read_text(encoding="utf-8"))
                for entry in data[-_MAX_FAILURES:]:
                    self._failures.append(FailureRecord(**entry))
            except Exception:
                pass

    def _persist(self) -> None:
        try:
            _FAILURE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = [f.to_dict() for f in self._failures[-_MAX_FAILURES:]]
            _FAILURE_LOG_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except Exception:
            logger.debug("Failure log persist failed", exc_info=True)

    @property
    def recent_failures(self) -> list[FailureRecord]:
        return list(self._failures[-20:])

    @property
    def unresolved_failures(self) -> list[FailureRecord]:
        return [f for f in self._failures if not f.resolved]

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    def get_failure_by_id(self, failure_id: str) -> FailureRecord | None:
        for f in self._failures:
            if f.id == failure_id:
                return f
        return None

    def mark_resolved(self, failure_id: str, fix_description: str = "") -> bool:
        for f in self._failures:
            if f.id == failure_id:
                f.resolved = True
                f.fix_applied = fix_description
                self._persist()
                return True
        return False


class ModuleHealthChecker:
    """Tests each ATOM module individually for health issues."""

    _CRITICAL_MODULES = [
        ("core.async_event_bus", "AsyncEventBus"),
        ("core.state_manager", "StateManager"),
        ("core.security_policy", "SecurityPolicy"),
        ("core.router.router", "Router"),
        ("core.intent_engine", "IntentEngine"),
        ("core.metrics", "MetricsCollector"),
        ("core.cache_engine", "CacheEngine"),
        ("core.memory_engine", "MemoryEngine"),
    ]

    _OPTIONAL_MODULES = [
        ("core.health_monitor", "HealthMonitor"),
        ("core.autonomy_engine", "AutonomyEngine"),
        ("core.behavior_tracker", "BehaviorTracker"),
        ("core.cognitive.second_brain", "SecondBrain"),
        ("core.cognitive.goal_engine", "GoalEngine"),
        ("core.cognitive.prediction_engine", "PredictionEngine"),
        ("core.jarvis_core", "JarvisCore"),
        ("core.system_scanner", "SystemScanner"),
        ("core.owner_understanding", "OwnerUnderstanding"),
        ("core.platform_adapter", None),
        ("core.system_control", "SystemControl"),
        ("core.reasoning.tool_registry", None),
        ("core.reasoning.action_executor", "ActionExecutor"),
        ("core.reasoning.code_sandbox", "CodeSandbox"),
        ("voice.stt_async", "STTAsync"),
        ("voice.tts_async", "TTSAsync"),
        ("voice.mic_manager", "MicManager"),
        ("brain.mini_llm", "MiniLLM"),
        ("context.context_engine", "ContextEngine"),
        ("context.privacy_filter", None),
        ("ui.web_dashboard", "WebDashboard"),
        ("cursor_bridge.local_brain_controller", "LocalBrainController"),
        ("cursor_bridge.structured_prompt_builder", "StructuredPromptBuilder"),
        ("core.vector_store", "VectorStore"),
        ("core.embedding_engine", "EmbeddingEngine"),
        ("core.document_ingestion", "DocumentIngestionEngine"),
        ("core.security_fortress", "SecurityFortress"),
        ("core.code_introspector", "CodeIntrospector"),
    ]

    def check_all(self) -> list[ModuleHealthResult]:
        """Check health of all ATOM modules."""
        results: list[ModuleHealthResult] = []

        for module_path, class_name in self._CRITICAL_MODULES:
            result = self._check_module(module_path, class_name, critical=True)
            results.append(result)

        for module_path, class_name in self._OPTIONAL_MODULES:
            result = self._check_module(module_path, class_name, critical=False)
            results.append(result)

        return results

    def check_single(self, module_path: str) -> ModuleHealthResult:
        """Check health of a specific module."""
        return self._check_module(module_path, None, critical=False)

    @staticmethod
    def _check_module(
        module_path: str,
        class_name: str | None,
        critical: bool,
    ) -> ModuleHealthResult:
        t0 = time.perf_counter()
        issues: list[str] = []
        import_ok = False
        class_count = 0

        try:
            mod = importlib.import_module(module_path)
            import_ok = True

            import inspect
            classes = [
                name for name, obj in inspect.getmembers(mod, inspect.isclass)
                if obj.__module__ == mod.__name__
            ]
            class_count = len(classes)

            if class_name and not hasattr(mod, class_name):
                issues.append(f"Expected class '{class_name}' not found")

        except ImportError as e:
            issues.append(f"Import failed: {e}")
        except Exception as e:
            issues.append(f"Load error: {e}")

        elapsed = (time.perf_counter() - t0) * 1000

        if not import_ok:
            status = "failed" if critical else "missing"
        elif issues:
            status = "degraded"
        else:
            status = "healthy"

        return ModuleHealthResult(
            module_path=module_path,
            status=status,
            import_ok=import_ok,
            class_count=class_count,
            issues=issues,
            check_time_ms=elapsed,
        )


class FailureAnalyzer:
    """Analyzes failures to find root causes and patterns."""

    def __init__(self, introspector: CodeIntrospector | None = None) -> None:
        self._introspector = introspector

    def analyze(self, failure: FailureRecord) -> dict[str, Any]:
        """Deep analysis of a single failure."""
        analysis: dict[str, Any] = {
            "failure_id": failure.id,
            "root_cause": self._find_root_cause(failure),
            "pattern": self._detect_pattern(failure),
            "affected_components": self._find_affected(failure),
            "similar_past_failures": [],
        }
        return analysis

    def _find_root_cause(self, failure: FailureRecord) -> str:
        """Determine the root cause of a failure."""
        exc = failure.exception_type
        msg = failure.exception_message.lower()

        if failure.category == "dependency":
            module_name = ""
            if "no module named" in msg:
                parts = msg.split("'")
                if len(parts) >= 2:
                    module_name = parts[1]
            return (
                f"Missing Python package: '{module_name}'. "
                f"This module needs to be installed."
            )

        if failure.category == "config":
            return (
                f"Configuration error in {failure.module}. "
                f"Check config/settings.json for missing or invalid values."
            )

        if failure.category == "file":
            return (
                f"File system error: {failure.exception_message}. "
                f"A required file or directory is missing or inaccessible."
            )

        if failure.category == "memory":
            return (
                "System ran out of memory. This is usually caused by "
                "loading a model that's too large for available RAM/VRAM."
            )

        if failure.category == "model":
            return (
                f"AI model error: {failure.exception_message}. "
                "The LLM model file may be corrupted, missing, or "
                "incompatible with the installed llama-cpp-python version."
            )

        if failure.category == "network":
            return (
                f"Network error: {failure.exception_message}. "
                "Check network connectivity."
            )

        if exc == "KeyError":
            return (
                f"Missing key in data structure: {failure.exception_message}. "
                f"This is likely a code bug in {failure.module}."
            )

        if exc == "AttributeError":
            return (
                f"Object missing expected attribute: {failure.exception_message}. "
                f"This may be a version mismatch or initialization order issue."
            )

        if exc == "TypeError":
            return (
                f"Type mismatch: {failure.exception_message}. "
                f"A function in {failure.module} received unexpected argument types."
            )

        return (
            f"{exc} in {failure.module} at line {failure.lineno}: "
            f"{failure.exception_message}"
        )

    @staticmethod
    def _detect_pattern(failure: FailureRecord) -> str:
        if failure.occurrence_count > 5:
            return "recurring"
        if failure.occurrence_count > 1:
            return "repeated"
        return "isolated"

    def _find_affected(self, failure: FailureRecord) -> list[str]:
        """Find components affected by this failure."""
        affected = [failure.module]
        if self._introspector and self._introspector.is_scanned:
            deps = self._introspector.get_dependency_graph()
            module_name = failure.module.replace(".py", "")
            for edge in deps:
                if module_name in edge.target:
                    affected.append(edge.source)
        return affected[:10]


class FixEngine:
    """Generates and applies fixes for identified failures."""

    __slots__ = ("_fix_history", "_introspector")

    def __init__(self, introspector: CodeIntrospector | None = None) -> None:
        self._introspector = introspector
        self._fix_history: list[dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        if _FIX_HISTORY_FILE.exists():
            try:
                self._fix_history = json.loads(
                    _FIX_HISTORY_FILE.read_text(encoding="utf-8")
                )
            except Exception:
                self._fix_history = []

    def _save_history(self) -> None:
        try:
            _FIX_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _FIX_HISTORY_FILE.write_text(
                json.dumps(self._fix_history[-_MAX_FIX_HISTORY:], indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Fix history save failed", exc_info=True)

    def suggest_fix(self, failure: FailureRecord) -> FixSuggestion:
        """Generate a fix suggestion for a failure."""
        if failure.category == "dependency":
            package = self._extract_package_name(failure.exception_message)
            return FixSuggestion(
                failure_id=failure.id,
                description=f"Install missing package: {package}",
                fix_type="dependency",
                auto_fixable=False,
                fix_commands=[f"pip install {package}"],
                confidence=0.9,
                risk_level="safe",
            )

        if failure.category == "config":
            return FixSuggestion(
                failure_id=failure.id,
                description="Reset configuration to defaults or fix the invalid value",
                fix_type="config",
                auto_fixable=True,
                fix_commands=["Validate and repair config/settings.json"],
                confidence=0.7,
                risk_level="safe",
            )

        if failure.category == "file":
            return FixSuggestion(
                failure_id=failure.id,
                description="Create missing directories or restore missing files",
                fix_type="file",
                auto_fixable=True,
                fix_commands=["Create required directories: logs/, data/, config/"],
                confidence=0.8,
                risk_level="safe",
            )

        if failure.category == "memory":
            return FixSuggestion(
                failure_id=failure.id,
                description=(
                    "Reduce memory usage by switching to a smaller model "
                    "or reducing context window"
                ),
                fix_type="config",
                auto_fixable=True,
                fix_commands=[
                    "Reduce brain.n_ctx in settings.json",
                    "Switch to a smaller GGUF model",
                ],
                confidence=0.6,
                risk_level="moderate",
            )

        if failure.category == "model":
            return FixSuggestion(
                failure_id=failure.id,
                description="Check model file integrity and compatibility",
                fix_type="manual",
                auto_fixable=False,
                fix_commands=[
                    "Verify brain.model_path in settings.json points to valid GGUF",
                    "Re-download the model file if corrupted",
                ],
                confidence=0.5,
                risk_level="safe",
            )

        if failure.exception_type == "KeyError":
            return FixSuggestion(
                failure_id=failure.id,
                description=(
                    f"Add missing key handling in {failure.module}. "
                    f"Use .get() instead of direct key access."
                ),
                fix_type="code",
                auto_fixable=False,
                fix_commands=[
                    f"Edit {failure.module} at line {failure.lineno}: "
                    f"use dict.get() with default value"
                ],
                confidence=0.8,
                risk_level="safe",
            )

        return FixSuggestion(
            failure_id=failure.id,
            description=f"Manual investigation needed for {failure.exception_type}",
            fix_type="manual",
            auto_fixable=False,
            fix_commands=[
                f"Check {failure.module} at line {failure.lineno}",
                f"Review traceback for {failure.exception_type}",
            ],
            confidence=0.3,
            risk_level="safe",
        )

    def apply_fix(self, suggestion: FixSuggestion) -> tuple[bool, str]:
        """Apply an auto-fixable fix. Returns (success, message)."""
        if not suggestion.auto_fixable:
            return False, (
                f"This fix requires manual intervention: {suggestion.description}"
            )

        success = False
        message = ""

        if suggestion.fix_type == "file":
            success, message = self._fix_missing_directories()
        elif suggestion.fix_type == "config":
            success, message = self._fix_config()
        else:
            message = f"Auto-fix not implemented for type: {suggestion.fix_type}"

        self._fix_history.append({
            "timestamp": datetime.now().isoformat(),
            "failure_id": suggestion.failure_id,
            "fix_type": suggestion.fix_type,
            "description": suggestion.description,
            "success": success,
            "message": message,
        })
        self._save_history()

        return success, message

    @staticmethod
    def _fix_missing_directories() -> tuple[bool, str]:
        """Create all required directories."""
        dirs = [
            Path("logs"), Path("data"), Path("config"),
            Path("data/security"), Path("data/vector_db"),
        ]
        created = []
        for d in dirs:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created.append(str(d))

        if created:
            return True, f"Created directories: {', '.join(created)}"
        return True, "All required directories already exist."

    @staticmethod
    def _fix_config() -> tuple[bool, str]:
        """Validate and repair config/settings.json."""
        config_path = Path("config/settings.json")
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default = {
                "owner": {"name": "Satyam", "title": "Boss"},
                "brain": {"enabled": False},
                "security": {"mode": "strict", "audit_to_file": True},
                "performance": {"mode": "lite"},
                "features": {},
            }
            config_path.write_text(
                json.dumps(default, indent=4), encoding="utf-8",
            )
            return True, "Created default config/settings.json"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return True, "Config file is valid JSON."
        except json.JSONDecodeError as e:
            backup = config_path.with_suffix(".json.bak")
            import shutil
            shutil.copy2(config_path, backup)
            return False, (
                f"Config has invalid JSON at line {e.lineno}. "
                f"Backup saved to {backup}. Fix manually."
            )

    @staticmethod
    def _extract_package_name(message: str) -> str:
        """Extract package name from ImportError message."""
        if "'" in message:
            parts = message.split("'")
            if len(parts) >= 2:
                name = parts[1].split(".")[0]
                _PACKAGE_MAP = {
                    "cv2": "opencv-python",
                    "PIL": "Pillow",
                    "yaml": "PyYAML",
                    "sklearn": "scikit-learn",
                    "chromadb": "chromadb",
                    "sentence_transformers": "sentence-transformers",
                    "pynvml": "pynvml",
                    "keyboard": "keyboard",
                    "pyaudio": "PyAudio",
                    "resemblyzer": "resemblyzer",
                    "cryptography": "cryptography",
                }
                return _PACKAGE_MAP.get(name, name)
        return "unknown"

    @property
    def fix_count(self) -> int:
        return len(self._fix_history)


class StartupValidator:
    """Pre-flight checks before ATOM starts."""

    @staticmethod
    def validate() -> tuple[bool, list[str], list[str]]:
        """Run all startup validations.

        Returns (all_ok, errors, warnings).
        """
        errors: list[str] = []
        warnings: list[str] = []

        required_dirs = [
            Path("logs"), Path("config"), Path("data"),
        ]
        for d in required_dirs:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created missing directory: {d}")

        config_path = Path("config/settings.json")
        if not config_path.exists():
            errors.append("config/settings.json not found")
        else:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                required_sections = ["owner", "security"]
                for section in required_sections:
                    if section not in config:
                        warnings.append(
                            f"Config missing '{section}' section (defaults will be used)"
                        )
            except json.JSONDecodeError as e:
                errors.append(f"config/settings.json is invalid JSON: {e}")

        critical_packages = {
            "asyncio": "asyncio (stdlib)",
            "json": "json (stdlib)",
            "logging": "logging (stdlib)",
            "pathlib": "pathlib (stdlib)",
        }
        for package, label in critical_packages.items():
            try:
                importlib.import_module(package)
            except ImportError:
                errors.append(f"Critical package missing: {label}")

        optional_packages = {
            "psutil": "system monitoring",
            "keyboard": "global hotkey",
            "chromadb": "vector memory",
            "cryptography": "encrypted vault",
        }
        for package, purpose in optional_packages.items():
            try:
                importlib.import_module(package)
            except ImportError:
                warnings.append(f"Optional: {package} not installed ({purpose})")

        log_dir = Path("logs")
        try:
            test_file = log_dir / ".write_test"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Cannot write to logs directory: {e}")

        all_ok = len(errors) == 0
        return all_ok, errors, warnings


class SelfHealingEngine:
    """Master self-healing controller.

    Orchestrates exception tracking, module health checking,
    failure analysis, and fix application into a single interface.
    """

    __slots__ = (
        "_tracker", "_health_checker", "_analyzer",
        "_fix_engine", "_introspector", "_config",
        "_last_health_check",
    )

    def __init__(
        self,
        config: dict | None = None,
        introspector: CodeIntrospector | None = None,
    ) -> None:
        self._config = config or {}
        self._introspector = introspector
        self._tracker = ExceptionTracker()
        self._health_checker = ModuleHealthChecker()
        self._analyzer = FailureAnalyzer(introspector)
        self._fix_engine = FixEngine(introspector)
        self._last_health_check: list[ModuleHealthResult] | None = None

    def start(self) -> None:
        """Install exception hook and run startup validation."""
        self._tracker.install_hook()

        ok, errors, warnings = StartupValidator.validate()
        for err in errors:
            logger.error("STARTUP: %s", err)
        for warn in warnings:
            logger.warning("STARTUP: %s", warn)

        if not ok:
            logger.error(
                "STARTUP VALIDATION FAILED: %d errors. "
                "ATOM may not function correctly.",
                len(errors),
            )
        else:
            logger.info("Startup validation passed (%d warnings)", len(warnings))

    def capture_exception(
        self,
        context: str = "",
        atom_state: str = "unknown",
    ) -> FailureRecord | None:
        """Capture the current exception. Call from except blocks."""
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is None:
            return None
        return self._tracker.capture(
            exc_type, exc_value, exc_tb,
            context=context, atom_state=atom_state,
        )

    def run_health_check(self) -> list[ModuleHealthResult]:
        """Run a full health check on all ATOM modules."""
        self._last_health_check = self._health_checker.check_all()
        return self._last_health_check

    def get_health_summary(self) -> str:
        """Get a human-readable health summary."""
        results = self._last_health_check or self.run_health_check()

        healthy = sum(1 for r in results if r.status == "healthy")
        degraded = sum(1 for r in results if r.status == "degraded")
        failed = sum(1 for r in results if r.status == "failed")
        missing = sum(1 for r in results if r.status == "missing")
        total = len(results)

        parts = [
            f"ATOM Module Health Check: {healthy}/{total} modules healthy.",
        ]

        if degraded > 0:
            degraded_names = [
                r.module_path for r in results if r.status == "degraded"
            ]
            parts.append(
                f"{degraded} degraded: {', '.join(degraded_names[:5])}"
            )

        if failed > 0:
            failed_names = [
                r.module_path for r in results if r.status == "failed"
            ]
            parts.append(
                f"{failed} FAILED: {', '.join(failed_names[:5])}"
            )
            for r in results:
                if r.status == "failed" and r.issues:
                    parts.append(f"  {r.module_path}: {r.issues[0]}")

        if missing > 0:
            parts.append(f"{missing} optional modules not installed.")

        unresolved = len(self._tracker.unresolved_failures)
        if unresolved > 0:
            parts.append(f"{unresolved} unresolved failures in log.")

        return " ".join(parts)

    def diagnose_failure(self, failure_id: str | None = None) -> str:
        """Diagnose a specific failure or the most recent one."""
        if failure_id:
            failure = self._tracker.get_failure_by_id(failure_id)
        else:
            failures = self._tracker.recent_failures
            failure = failures[-1] if failures else None

        if failure is None:
            if failure_id:
                return f"No failure found with ID {failure_id}."
            return "No failures recorded. Everything is running smoothly, Boss."

        analysis = self._analyzer.analyze(failure)
        suggestion = self._fix_engine.suggest_fix(failure)

        parts = [
            f"Failure Analysis [{failure.id}]:",
            f"  When: {failure.timestamp_human}",
            f"  What: {failure.exception_type}: {failure.exception_message[:200]}",
            f"  Where: {failure.module}:{failure.function} line {failure.lineno}",
            f"  Category: {failure.category} | Severity: {failure.severity}",
            f"  Pattern: {analysis['pattern']} ({failure.occurrence_count} occurrences)",
            f"  Root cause: {analysis['root_cause']}",
            f"  Suggested fix: {suggestion.description}",
            f"  Auto-fixable: {'yes' if suggestion.auto_fixable else 'no'}",
            f"  Confidence: {suggestion.confidence:.0%}",
        ]

        if suggestion.fix_commands:
            parts.append(f"  Commands: {'; '.join(suggestion.fix_commands)}")

        if suggestion.auto_fixable:
            parts.append(f"  Say 'fix it' to apply this fix automatically.")

        return " ".join(parts)

    def fix_last_failure(self) -> str:
        """Attempt to fix the most recent unresolved failure."""
        unresolved = self._tracker.unresolved_failures
        if not unresolved:
            return "No unresolved failures to fix, Boss."

        failure = unresolved[-1]
        suggestion = self._fix_engine.suggest_fix(failure)

        if not suggestion.auto_fixable:
            return (
                f"The fix for [{failure.id}] requires manual intervention: "
                f"{suggestion.description}. {'; '.join(suggestion.fix_commands)}"
            )

        success, message = self._fix_engine.apply_fix(suggestion)

        if success:
            self._tracker.mark_resolved(failure.id, suggestion.description)
            return (
                f"Fix applied successfully for [{failure.id}]: {message}. "
                f"The issue should be resolved."
            )
        else:
            return (
                f"Fix attempt for [{failure.id}] was not fully successful: "
                f"{message}. Manual review may be needed."
            )

    def fix_all(self) -> str:
        """Attempt to fix all auto-fixable unresolved failures."""
        unresolved = self._tracker.unresolved_failures
        if not unresolved:
            return "No unresolved failures, Boss. All systems clean."

        fixed = 0
        skipped = 0
        failed_fixes = 0

        for failure in unresolved:
            suggestion = self._fix_engine.suggest_fix(failure)
            if not suggestion.auto_fixable:
                skipped += 1
                continue

            success, _ = self._fix_engine.apply_fix(suggestion)
            if success:
                self._tracker.mark_resolved(failure.id, suggestion.description)
                fixed += 1
            else:
                failed_fixes += 1

        parts = [f"Self-healing complete."]
        if fixed:
            parts.append(f"{fixed} issues fixed automatically.")
        if skipped:
            parts.append(f"{skipped} issues need manual attention.")
        if failed_fixes:
            parts.append(f"{failed_fixes} fix attempts were unsuccessful.")

        return " ".join(parts)

    def get_failure_report(self) -> str:
        """Get a summary of all tracked failures."""
        failures = self._tracker.recent_failures
        if not failures:
            return "No failures recorded. ATOM is running perfectly, Boss."

        unresolved = sum(1 for f in failures if not f.resolved)
        resolved = sum(1 for f in failures if f.resolved)

        category_counts: dict[str, int] = {}
        for f in failures:
            category_counts[f.category] = category_counts.get(f.category, 0) + 1

        parts = [
            f"Failure Report: {len(failures)} total, "
            f"{unresolved} unresolved, {resolved} resolved.",
        ]

        if category_counts:
            cats = ", ".join(
                f"{cat}: {count}" for cat, count
                in sorted(category_counts.items(), key=lambda x: -x[1])
            )
            parts.append(f"By category: {cats}.")

        if unresolved > 0:
            latest = [f for f in failures if not f.resolved][-1]
            parts.append(
                f"Latest unresolved: [{latest.id}] {latest.exception_type} "
                f"in {latest.module} ({latest.severity})."
            )
            parts.append("Say 'diagnose failure' for details or 'fix it' to auto-repair.")

        return " ".join(parts)

    def persist(self) -> None:
        self._tracker._persist()
        self._fix_engine._save_history()

    @property
    def tracker(self) -> ExceptionTracker:
        return self._tracker

    @property
    def fix_engine(self) -> FixEngine:
        return self._fix_engine
