"""
ATOM -- Code Introspector (Self-Aware Source Code Analysis).

Gives ATOM the ability to read, parse, and understand its own source code.
This is the foundation for self-healing: ATOM must know its own structure
before it can diagnose and fix itself.

Capabilities:
  1. Source File Scanner: Discovers all ATOM Python source files
  2. AST Analyzer: Parses each file's Abstract Syntax Tree to extract
     classes, functions, imports, decorators, docstrings
  3. Module Dependency Mapper: Builds a full import dependency graph
  4. Event Bus Mapper: Traces bus.on() and bus.emit() calls to map
     the entire event wiring
  5. Architecture Explainer: Can describe any module's purpose, its
     classes, functions, and how it connects to the rest of ATOM
  6. Code Health Analyzer: Detects code smells, missing docstrings,
     overly complex functions, circular imports

Owner: Satyam (Boss). ATOM knows itself.
"""

from __future__ import annotations

import ast
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.code_introspector")

_ATOM_ROOT = Path(__file__).parent.parent


@dataclass
class FunctionInfo:
    """Metadata about a function or method."""
    name: str
    lineno: int
    end_lineno: int
    args: list[str]
    decorators: list[str]
    docstring: str
    is_async: bool
    complexity: int  # approximate cyclomatic complexity
    line_count: int


@dataclass
class ClassInfo:
    """Metadata about a class."""
    name: str
    lineno: int
    end_lineno: int
    bases: list[str]
    docstring: str
    methods: list[FunctionInfo]
    slots: list[str]


@dataclass
class ModuleInfo:
    """Full metadata about a Python module."""
    path: str
    relative_path: str
    docstring: str
    imports: list[str]
    from_imports: list[tuple[str, list[str]]]
    classes: list[ClassInfo]
    functions: list[FunctionInfo]
    global_vars: list[str]
    line_count: int
    event_subscriptions: list[str]
    event_emissions: list[str]
    last_analyzed: float = 0.0

    @property
    def all_class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    @property
    def all_function_names(self) -> list[str]:
        return [f.name for f in self.functions]


@dataclass
class DependencyEdge:
    """A dependency from one module to another."""
    source: str
    target: str
    imports: list[str]


@dataclass
class CodeHealthReport:
    """Health analysis of a module or the entire codebase."""
    total_files: int = 0
    total_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    missing_docstrings: list[str] = field(default_factory=list)
    complex_functions: list[str] = field(default_factory=list)
    large_files: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    score: float = 100.0


class CodeIntrospector:
    """ATOM's self-awareness engine -- reads and understands its own code.

    Usage:
        introspector = CodeIntrospector()
        introspector.scan()  # Analyze all source files

        # Get info about a specific module
        info = introspector.get_module("core/router/router.py")

        # Get full dependency graph
        deps = introspector.get_dependency_graph()

        # Get event bus wiring map
        events = introspector.get_event_map()

        # Explain a module in natural language
        explanation = introspector.explain_module("core/security_policy.py")

        # Get code health report
        health = introspector.get_code_health()
    """

    __slots__ = (
        "_atom_root", "_modules", "_dependency_graph",
        "_event_map", "_last_scan_time", "_scan_duration_ms",
    )

    def __init__(self, atom_root: Path | None = None) -> None:
        self._atom_root = atom_root or _ATOM_ROOT
        self._modules: dict[str, ModuleInfo] = {}
        self._dependency_graph: list[DependencyEdge] = []
        self._event_map: dict[str, dict[str, list[str]]] = {
            "subscriptions": {},
            "emissions": {},
        }
        self._last_scan_time: float = 0.0
        self._scan_duration_ms: float = 0.0

    def _get_source_files(self) -> list[Path]:
        """Discover all Python source files in the ATOM project."""
        files: list[Path] = []
        for f in self._atom_root.rglob("*.py"):
            rel = str(f.relative_to(self._atom_root))
            if any(skip in rel for skip in (
                "__pycache__", ".git", "logs", "data",
                "node_modules", ".env", ".venv",
            )):
                continue
            files.append(f)
        return sorted(files)

    def scan(self) -> int:
        """Scan and analyze all ATOM source files.

        Returns the number of files analyzed.
        """
        t0 = time.perf_counter()
        files = self._get_source_files()
        self._modules.clear()
        self._dependency_graph.clear()

        for filepath in files:
            try:
                module_info = self._analyze_file(filepath)
                self._modules[module_info.relative_path] = module_info
            except Exception as e:
                rel = str(filepath.relative_to(self._atom_root))
                logger.debug("Failed to analyze %s: %s", rel, e)

        self._build_dependency_graph()
        self._build_event_map()
        self._last_scan_time = time.time()
        self._scan_duration_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Code introspection complete: %d files, %d classes, %d functions "
            "in %.0fms",
            len(self._modules),
            sum(len(m.classes) for m in self._modules.values()),
            sum(len(m.functions) for m in self._modules.values()),
            self._scan_duration_ms,
        )
        return len(self._modules)

    def _analyze_file(self, filepath: Path) -> ModuleInfo:
        """Parse a single Python file and extract all metadata."""
        source = filepath.read_text(encoding="utf-8", errors="replace")
        rel_path = str(filepath.relative_to(self._atom_root))
        lines = source.split("\n")

        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            return ModuleInfo(
                path=str(filepath),
                relative_path=rel_path,
                docstring="[SYNTAX ERROR - could not parse]",
                imports=[], from_imports=[],
                classes=[], functions=[],
                global_vars=[], line_count=len(lines),
                event_subscriptions=[], event_emissions=[],
            )

        docstring = ast.get_docstring(tree) or ""
        imports = []
        from_imports = []
        classes = []
        functions = []
        global_vars = []
        event_subs = []
        event_emits = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                names = [alias.name for alias in node.names]
                from_imports.append((module_name, names))

            elif isinstance(node, (ast.ClassDef,)):
                classes.append(self._analyze_class(node))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._analyze_function(node))

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.append(target.id)

        event_subs, event_emits = self._extract_events(source)

        return ModuleInfo(
            path=str(filepath),
            relative_path=rel_path,
            docstring=docstring,
            imports=imports,
            from_imports=from_imports,
            classes=classes,
            functions=functions,
            global_vars=global_vars,
            line_count=len(lines),
            event_subscriptions=event_subs,
            event_emissions=event_emits,
            last_analyzed=time.time(),
        )

    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract metadata from a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.dump(base))

        methods = []
        slots = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._analyze_function(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "__slots__":
                        if isinstance(item.value, (ast.Tuple, ast.List)):
                            for elt in item.value.elts:
                                if isinstance(elt, ast.Constant):
                                    slots.append(str(elt.value))

        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            bases=bases,
            docstring=ast.get_docstring(node) or "",
            methods=methods,
            slots=slots,
        )

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        """Extract metadata from a function definition."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{ast.dump(dec)}")

        end_line = getattr(node, "end_lineno", node.lineno)
        line_count = end_line - node.lineno + 1

        complexity = self._estimate_complexity(node)

        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=end_line,
            args=args,
            decorators=decorators,
            docstring=ast.get_docstring(node) or "",
            is_async=isinstance(node, ast.AsyncFunctionDef),
            complexity=complexity,
            line_count=line_count,
        )

    @staticmethod
    def _estimate_complexity(node: ast.AST) -> int:
        """Estimate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For,
                                  ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.SetComp,
                                    ast.DictComp, ast.GeneratorExp)):
                complexity += 1
        return complexity

    @staticmethod
    def _extract_events(source: str) -> tuple[list[str], list[str]]:
        """Extract event bus subscriptions and emissions from source."""
        import re
        subs = re.findall(r'bus\.on\(["\'](\w+)["\']', source)
        subs += re.findall(r'_bus\.on\(["\'](\w+)["\']', source)
        subs += re.findall(r'self\._bus\.on\(["\'](\w+)["\']', source)

        emits = re.findall(r'bus\.emit(?:_fast|_long)?\(["\'](\w+)["\']', source)
        emits += re.findall(r'_bus\.emit(?:_fast|_long)?\(["\'](\w+)["\']', source)
        emits += re.findall(r'self\._bus\.emit(?:_fast|_long)?\(["\'](\w+)["\']', source)

        return list(set(subs)), list(set(emits))

    def _build_dependency_graph(self) -> None:
        """Build the import dependency graph."""
        self._dependency_graph = []
        for rel_path, module in self._modules.items():
            for imp_module, imp_names in module.from_imports:
                if imp_module.startswith(("core.", "voice.", "brain.",
                                         "context.", "ui.", "cursor_bridge.")):
                    self._dependency_graph.append(DependencyEdge(
                        source=rel_path,
                        target=imp_module.replace(".", "/") + ".py",
                        imports=imp_names,
                    ))

    def _build_event_map(self) -> None:
        """Build the event subscription/emission map across all modules."""
        subs: dict[str, list[str]] = {}
        emits: dict[str, list[str]] = {}

        for rel_path, module in self._modules.items():
            for event in module.event_subscriptions:
                subs.setdefault(event, []).append(rel_path)
            for event in module.event_emissions:
                emits.setdefault(event, []).append(rel_path)

        self._event_map = {"subscriptions": subs, "emissions": emits}

    # ── Query Interface ────────────────────────────────────────────────

    def get_module(self, relative_path: str) -> ModuleInfo | None:
        """Get detailed info about a specific module."""
        return self._modules.get(relative_path)

    def get_all_modules(self) -> dict[str, ModuleInfo]:
        return dict(self._modules)

    def get_dependency_graph(self) -> list[DependencyEdge]:
        return list(self._dependency_graph)

    def get_event_map(self) -> dict[str, dict[str, list[str]]]:
        return dict(self._event_map)

    def get_module_source(self, relative_path: str) -> str | None:
        """Read the actual source code of a module."""
        filepath = self._atom_root / relative_path
        if filepath.exists():
            try:
                return filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return None
        return None

    def search_code(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search across all source files for a string pattern."""
        import re
        results: list[dict[str, Any]] = []
        pattern = re.compile(re.escape(query), re.I)

        for rel_path, module in self._modules.items():
            source = self.get_module_source(rel_path)
            if source is None:
                continue
            for i, line in enumerate(source.split("\n"), 1):
                if pattern.search(line):
                    results.append({
                        "file": rel_path,
                        "line": i,
                        "content": line.strip()[:200],
                    })
                    if len(results) >= max_results:
                        return results
        return results

    def find_function(self, func_name: str) -> list[dict[str, Any]]:
        """Find all definitions of a function/method by name."""
        results: list[dict[str, Any]] = []
        for rel_path, module in self._modules.items():
            for func in module.functions:
                if func.name == func_name:
                    results.append({
                        "file": rel_path,
                        "function": func.name,
                        "line": func.lineno,
                        "is_async": func.is_async,
                        "args": func.args,
                        "docstring": func.docstring[:200],
                    })
            for cls in module.classes:
                for method in cls.methods:
                    if method.name == func_name:
                        results.append({
                            "file": rel_path,
                            "class": cls.name,
                            "method": method.name,
                            "line": method.lineno,
                            "is_async": method.is_async,
                            "args": method.args,
                            "docstring": method.docstring[:200],
                        })
        return results

    def find_class(self, class_name: str) -> list[dict[str, Any]]:
        """Find all definitions of a class by name."""
        results: list[dict[str, Any]] = []
        for rel_path, module in self._modules.items():
            for cls in module.classes:
                if cls.name == class_name:
                    results.append({
                        "file": rel_path,
                        "class": cls.name,
                        "line": cls.lineno,
                        "bases": cls.bases,
                        "methods": [m.name for m in cls.methods],
                        "slots": cls.slots,
                        "docstring": cls.docstring[:200],
                    })
        return results

    # ── Natural Language Explanation ────────────────────────────────────

    def explain_module(self, relative_path: str) -> str:
        """Generate a natural language explanation of a module."""
        module = self._modules.get(relative_path)
        if module is None:
            return f"Module '{relative_path}' not found in ATOM codebase."

        parts = [f"Module: {relative_path} ({module.line_count} lines)"]

        if module.docstring:
            doc_preview = module.docstring.split("\n")[0][:200]
            parts.append(f"Purpose: {doc_preview}")

        if module.classes:
            class_names = ", ".join(c.name for c in module.classes)
            parts.append(f"Classes: {class_names}")
            for cls in module.classes:
                method_names = ", ".join(m.name for m in cls.methods if not m.name.startswith("_"))
                if method_names:
                    parts.append(f"  {cls.name} methods: {method_names}")

        public_funcs = [f for f in module.functions if not f.name.startswith("_")]
        if public_funcs:
            func_names = ", ".join(f.name for f in public_funcs)
            parts.append(f"Functions: {func_names}")

        if module.event_subscriptions:
            parts.append(f"Listens to events: {', '.join(module.event_subscriptions)}")
        if module.event_emissions:
            parts.append(f"Emits events: {', '.join(module.event_emissions)}")

        deps = [e.target for e in self._dependency_graph if e.source == relative_path]
        if deps:
            parts.append(f"Depends on: {', '.join(deps[:8])}")

        dependents = [e.source for e in self._dependency_graph if e.target == relative_path]
        if dependents:
            parts.append(f"Used by: {', '.join(dependents[:8])}")

        return ". ".join(parts)

    def explain_architecture(self) -> str:
        """Generate a high-level architecture summary."""
        if not self._modules:
            return "No scan data. Run introspection first."

        total_files = len(self._modules)
        total_lines = sum(m.line_count for m in self._modules.values())
        total_classes = sum(len(m.classes) for m in self._modules.values())
        total_functions = sum(len(m.functions) for m in self._modules.values())

        categories: dict[str, int] = {}
        for rel_path in self._modules:
            cat = rel_path.split("/")[0] if "/" in rel_path else "root"
            categories[cat] = categories.get(cat, 0) + 1

        all_events = set()
        for subs in self._event_map.get("subscriptions", {}).keys():
            all_events.add(subs)
        for emits in self._event_map.get("emissions", {}).keys():
            all_events.add(emits)

        parts = [
            f"ATOM Architecture Summary:",
            f"  {total_files} source files, {total_lines:,} lines of code",
            f"  {total_classes} classes, {total_functions} functions",
            f"  {len(all_events)} unique events on the event bus",
            f"  {len(self._dependency_graph)} import dependencies",
            "Module categories:",
        ]
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            parts.append(f"  {cat}: {count} files")

        return " ".join(parts)

    # ── Code Health Analysis ────────────────────────────────────────────

    def get_code_health(self) -> CodeHealthReport:
        """Analyze code quality across the entire codebase."""
        report = CodeHealthReport()
        report.total_files = len(self._modules)
        report.total_lines = sum(m.line_count for m in self._modules.values())
        report.total_classes = sum(len(m.classes) for m in self._modules.values())
        report.total_functions = sum(len(m.functions) for m in self._modules.values())

        for rel_path, module in self._modules.items():
            if module.line_count > 500:
                report.large_files.append(
                    f"{rel_path} ({module.line_count} lines)"
                )

            if not module.docstring:
                report.missing_docstrings.append(rel_path)

            for cls in module.classes:
                if not cls.docstring:
                    report.missing_docstrings.append(
                        f"{rel_path}::{cls.name}"
                    )
                for method in cls.methods:
                    if method.complexity > 15:
                        report.complex_functions.append(
                            f"{rel_path}::{cls.name}.{method.name} "
                            f"(complexity={method.complexity})"
                        )

            for func in module.functions:
                if func.complexity > 15:
                    report.complex_functions.append(
                        f"{rel_path}::{func.name} "
                        f"(complexity={func.complexity})"
                    )

        penalties = 0.0
        if report.missing_docstrings:
            ratio = len(report.missing_docstrings) / max(1, report.total_files)
            penalties += min(20.0, ratio * 30.0)

        if report.complex_functions:
            penalties += min(15.0, len(report.complex_functions) * 2.0)

        if report.large_files:
            penalties += min(10.0, len(report.large_files) * 1.5)

        report.score = max(0.0, 100.0 - penalties)
        return report

    def format_code_health(self, report: CodeHealthReport | None = None) -> str:
        """Format code health report for voice/text output."""
        if report is None:
            report = self.get_code_health()

        parts = [
            f"ATOM Code Health Score: {report.score:.0f} out of 100.",
            f"{report.total_files} files, {report.total_lines:,} lines, "
            f"{report.total_classes} classes, {report.total_functions} functions.",
        ]

        if report.large_files:
            parts.append(
                f"{len(report.large_files)} large files: "
                f"{', '.join(report.large_files[:3])}"
            )

        if report.complex_functions:
            parts.append(
                f"{len(report.complex_functions)} complex functions: "
                f"{', '.join(report.complex_functions[:3])}"
            )

        if report.missing_docstrings:
            parts.append(
                f"{len(report.missing_docstrings)} modules/classes missing docstrings."
            )

        return " ".join(parts)

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def module_count(self) -> int:
        return len(self._modules)

    @property
    def is_scanned(self) -> bool:
        return self._last_scan_time > 0

    def get_scan_stats(self) -> dict[str, Any]:
        return {
            "files": len(self._modules),
            "lines": sum(m.line_count for m in self._modules.values()),
            "classes": sum(len(m.classes) for m in self._modules.values()),
            "functions": sum(len(m.functions) for m in self._modules.values()),
            "events": len(set(
                list(self._event_map.get("subscriptions", {}).keys()) +
                list(self._event_map.get("emissions", {}).keys())
            )),
            "dependencies": len(self._dependency_graph),
            "scan_duration_ms": self._scan_duration_ms,
            "last_scan": self._last_scan_time,
        }
