"""
ATOM — Cognitive Kernel (central brain coordinator).

The "brain of the brain": decides HOW to answer before answering.
Every query flows through the Cognitive Kernel, which picks the
optimal execution path based on query complexity, system state,
and available resources.

Execution paths (fastest → deepest):
    DIRECT  — Intent/quick-reply match. No LLM. Sub-5ms.
    CACHE   — Cached LLM response. No LLM. Sub-10ms.
    QUICK   — Fast brain (Qwen3-1.7B). 80-150ms.
    FULL    — Primary brain (Qwen3-4B thinking OFF). 300-600ms.
    DEEP    — Primary brain (Qwen3-4B thinking ON) + RAG. 800-2000ms.

System-aware routing:
    - Battery → prefer QUICK, skip RAG
    - Thermal pressure → degrade to QUICK
    - Memory pressure → skip RAG, reduce context
    - High error rate → circuit-break failing modules

Integrates with:
    IntentEngine, CacheEngine, QuickReplies, RuntimeModeResolver,
    InferenceGuard, SiliconGovernor, MetricsCollector, StateManager
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from core.fast_path import LatencyBudget
from core.rag.query_classifier import QueryComplexity, classify_query
from core.runtime.latency_controller import LatencyController

if TYPE_CHECKING:
    from core.async_event_bus import PriorityEventBus
    from core.cache_engine import CacheEngine
    from core.inference_guard import InferenceGuard
    from core.intent_engine import IntentEngine
    from core.metrics import MetricsCollector
    from core.silicon_governor import SiliconGovernor
    from core.state_manager import StateManager

logger = logging.getLogger("atom.cognitive_kernel")


# ── Execution paths ──────────────────────────────────────────────────

class ExecPath(str, Enum):
    DIRECT = "direct"
    CACHE = "cache"
    QUICK = "quick"
    FULL = "full"
    DEEP = "deep"


class CognitiveBudgetTier(str, Enum):
    COMMAND = "command"
    INFO = "info"
    SIMPLE = "simple"
    COMPLEX = "complex"
    CREATIVE = "creative"


@dataclass(frozen=True)
class _BudgetProfile:
    llm: str | bool
    use_rag: bool
    use_memory: bool
    budget_ms: float


_INFO_INTENTS = frozenset({
    "time", "date", "cpu", "ram", "battery", "disk",
    "system_info", "ip", "wifi", "uptime", "top_processes",
    "resource_report", "resource_trend", "app_history",
    "show_reminders", "self_diagnostic", "system_analyze",
    "self_check", "behavior_report", "status",
})

_INFO_HINTS = re.compile(
    r"\b("
    r"what\s+time|what\s+date|battery|cpu|ram|memory\s+usage|"
    r"disk\s+usage|uptime|system\s+status|system\s+info|ip\s+address|"
    r"wifi|resource\s+usage|diagnostic|self\s+check|how\s+are\s+you|"
    r"who\s+are\s+you"
    r")\b",
    re.I,
)
_CREATIVE_HINTS = re.compile(
    r"\b("
    r"brainstorm|idea|ideas|write|draft|rewrite|compose|story|poem|"
    r"lyrics|script|creative|imagine|invent|design|outline|pitch|proposal"
    r")\b",
    re.I,
)

_BUDGET_PROFILES: dict[CognitiveBudgetTier, _BudgetProfile] = {
    CognitiveBudgetTier.COMMAND: _BudgetProfile(
        llm=False,
        use_rag=False,
        use_memory=False,
        budget_ms=100.0,
    ),
    CognitiveBudgetTier.INFO: _BudgetProfile(
        llm=False,
        use_rag=False,
        use_memory=True,
        budget_ms=500.0,
    ),
    CognitiveBudgetTier.SIMPLE: _BudgetProfile(
        llm="small",
        use_rag=False,
        use_memory=True,
        budget_ms=1500.0,
    ),
    CognitiveBudgetTier.COMPLEX: _BudgetProfile(
        llm="large",
        use_rag=True,
        use_memory=True,
        budget_ms=5000.0,
    ),
    CognitiveBudgetTier.CREATIVE: _BudgetProfile(
        llm="large",
        use_rag=True,
        use_memory=True,
        budget_ms=10000.0,
    ),
}


# ── Query plan (output of routing decision) ─────────────────────────

@dataclass
class QueryPlan:
    """Describes how a query should be processed."""
    path: ExecPath
    model: str = "none"
    model_role: str = "none"
    runtime_mode: str = "SMART"
    use_rag: bool = False
    use_memory: bool = False
    thinking: bool = False
    budget_ms: float = 5000.0
    rag_budget_ms: float = 0.0
    reason: str = ""
    latency_reason: str = ""
    skip_llm: bool = False
    prompt_hint: str = ""
    reduce_context: bool = False
    memory_limit: int = 0
    history_turn_limit: int = 4
    requested_tier: str = CognitiveBudgetTier.SIMPLE.value
    budget_tier: str = CognitiveBudgetTier.SIMPLE.value
    base_budget_ms: float = 5000.0
    budget_allow_rag: bool = False
    budget_allow_memory: bool = False

    # Pre-resolved results (populated for DIRECT/CACHE paths)
    direct_response: str | None = None
    direct_intent: str | None = None
    direct_action: str | None = None
    direct_action_args: dict | None = None


# ── Latency budgets per path ─────────────────────────────────────────

_PATH_BUDGETS: dict[ExecPath, float] = {
    ExecPath.DIRECT: 50.0,
    ExecPath.CACHE: 100.0,
    ExecPath.QUICK: 1500.0,
    ExecPath.FULL: 5000.0,
    ExecPath.DEEP: 15000.0,
}


# ── Circuit breaker ──────────────────────────────────────────────────

@dataclass
class _CircuitState:
    failures: int = 0
    last_failure: float = 0.0
    open_until: float = 0.0

    _THRESHOLD = 3
    _COOLDOWN_S = 30.0

    @property
    def is_open(self) -> bool:
        if self.failures < self._THRESHOLD:
            return False
        return time.monotonic() < self.open_until

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure = time.monotonic()
        if self.failures >= self._THRESHOLD:
            self.open_until = time.monotonic() + self._COOLDOWN_S
            logger.warning(
                "Circuit breaker OPEN (failures=%d, cooldown=%.0fs)",
                self.failures, self._COOLDOWN_S,
            )

    def record_success(self) -> None:
        if self.failures > 0:
            self.failures = max(0, self.failures - 1)

    def reset(self) -> None:
        self.failures = 0
        self.open_until = 0.0


# ── System snapshot for routing decisions ────────────────────────────

@dataclass
class _SystemContext:
    """Point-in-time system state for routing."""
    memory_pct: float = 0.0
    cpu_pct: float = 0.0
    battery_pct: int = 100
    on_battery: bool = False
    thermal_pressure: str = "nominal"
    is_throttled: bool = False


# ── Cognitive Kernel ─────────────────────────────────────────────────

class CognitiveKernel:
    """Central coordinator for all intelligence decisions.

    Responsibilities:
        1. Query routing — pick the fastest viable path
        2. Model selection — small vs large, thinking ON/OFF
        3. Resource gating — respect memory/thermal/battery constraints
        4. Circuit breaking — bypass failing modules
        5. Latency budgets — enforce per-path time limits
        6. Metrics — record routing decisions for observability
    """

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        bus: "PriorityEventBus | None" = None,
        intent_engine: "IntentEngine | None" = None,
        cache_engine: "CacheEngine | None" = None,
        metrics: "MetricsCollector | None" = None,
        inference_guard: "InferenceGuard | None" = None,
        silicon_governor: "SiliconGovernor | None" = None,
        state_manager: "StateManager | None" = None,
    ) -> None:
        self._config = config or {}
        self._bus = bus
        self._intent = intent_engine
        self._cache = cache_engine
        self._metrics = metrics
        self._guard = inference_guard
        self._silicon = silicon_governor
        self._state = state_manager
        self._latency = LatencyController(self._config)

        ck = self._config.get("cognitive_kernel", {})
        self._quick_model = ck.get("quick_model", "qwen3-1.7b")
        self._full_model = ck.get("full_model", "qwen3-4b")
        self._deep_query_min_chars = int(ck.get("deep_query_min_chars", 120))
        self._simple_query_max_chars = int(ck.get("simple_query_max_chars", 50))
        self._battery_degrade = bool(ck.get("battery_degrade", True))
        self._thermal_degrade = bool(ck.get("thermal_degrade", True))
        self._memory_pressure_threshold = float(ck.get("memory_pressure_threshold", 85.0))
        self._rag_complexity_threshold = ck.get("rag_complexity_threshold", "complex")

        self._circuits: dict[str, _CircuitState] = {
            "intent": _CircuitState(),
            "cache": _CircuitState(),
            "llm_quick": _CircuitState(),
            "llm_full": _CircuitState(),
            "rag": _CircuitState(),
        }

        self._routing_counts: dict[str, int] = {p.value: 0 for p in ExecPath}
        self._budget_counts: dict[str, int] = {
            tier.value: 0 for tier in CognitiveBudgetTier
        }
        self._total_routed: int = 0

        if self._bus:
            self._bus.on("silicon_thermal_warn", self._on_thermal_warn)
            self._bus.on("silicon_memory_warn", self._on_memory_warn)

        logger.info(
            "CognitiveKernel: quick=%s, full=%s, deep_min=%d chars",
            self._quick_model, self._full_model, self._deep_query_min_chars,
        )

    # ── Public API ───────────────────────────────────────────────────

    def route(
        self,
        query: str,
        *,
        user_override: str | None = None,
        allow_cache: bool = True,
    ) -> QueryPlan:
        """Decide the optimal execution path for a query.

        This is the single entry point for all query processing decisions.
        Returns a QueryPlan describing how to handle the query.
        """
        t0 = time.perf_counter()
        query = (query or "").strip()

        if not query:
            plan = QueryPlan(
                path=ExecPath.DIRECT,
                skip_llm=True,
                reason="empty_query",
                direct_response="I didn't catch that, Boss.",
                budget_ms=_PATH_BUDGETS[ExecPath.DIRECT],
            )
            requested_tier = CognitiveBudgetTier.COMMAND
            plan = self._apply_budget_profile(plan, requested_tier=requested_tier)
            plan = self._apply_latency_policy(plan, query, self._get_system_context())
            self._record(plan, t0)
            return plan

        ctx = self._get_system_context()
        complexity = classify_query(query)

        # User can force a mode
        if user_override:
            plan = self._apply_override(query, user_override, ctx)
            requested_tier = self._classify_requested_tier(
                query,
                complexity=complexity,
            )
            plan = self._apply_budget_profile(plan, requested_tier=requested_tier)
            plan = self._apply_latency_policy(
                plan,
                query,
                ctx,
                complexity=complexity,
            )
            self._record(plan, t0)
            return plan

        # Path 1: DIRECT — intent match (sub-5ms, no LLM)
        plan = self._try_direct(query)
        if plan:
            requested_tier = self._classify_requested_tier(
                query,
                complexity=complexity,
                direct_intent=plan.direct_intent,
                direct_action=plan.direct_action,
            )
            plan = self._apply_budget_profile(plan, requested_tier=requested_tier)
            plan = self._apply_latency_policy(
                plan,
                query,
                ctx,
                complexity=complexity,
            )
            self._record(plan, t0)
            return plan

        # Path 2: CACHE — cached LLM response (sub-10ms)
        plan = self._try_cache(query) if allow_cache else None
        if plan:
            requested_tier = self._classify_requested_tier(
                query,
                complexity=complexity,
            )
            plan = self._apply_budget_profile(plan, requested_tier=requested_tier)
            plan = self._apply_latency_policy(
                plan,
                query,
                ctx,
                complexity=complexity,
            )
            self._record(plan, t0)
            return plan

        # Path 3-5: LLM routing based on complexity + system state
        requested_tier = self._classify_requested_tier(
            query,
            complexity=complexity,
        )
        plan = self._route_to_llm(query, ctx, complexity, requested_tier)
        plan = self._apply_budget_profile(plan, requested_tier=requested_tier)
        plan = self._apply_latency_policy(
            plan,
            query,
            ctx,
            complexity=complexity,
        )
        self._record(plan, t0)
        return plan

    def create_budget(self, plan: QueryPlan) -> LatencyBudget:
        """Create a latency budget tracker for the given plan."""
        return LatencyBudget(budget_ms=plan.budget_ms, label=plan.path.value)

    def record_outcome(self, path: str, success: bool) -> None:
        """Record whether a routed query succeeded or failed."""
        circuit = self._circuits.get(path)
        if circuit:
            if success:
                circuit.record_success()
            else:
                circuit.record_failure()

    def get_diagnostics(self) -> dict[str, Any]:
        """Return routing diagnostics for health monitoring."""
        total = self._total_routed or 1
        return {
            "total_routed": self._total_routed,
            "path_distribution": {
                path: {
                    "count": count,
                    "pct": round(count / total * 100, 1),
                }
                for path, count in self._routing_counts.items()
            },
            "budget_distribution": {
                tier: {
                    "count": count,
                    "pct": round(count / total * 100, 1),
                }
                for tier, count in self._budget_counts.items()
            },
            "llm_skip_rate_pct": round(
                (self._routing_counts.get("direct", 0) +
                 self._routing_counts.get("cache", 0)) / total * 100, 1,
            ),
            "circuits": {
                name: {
                    "failures": c.failures,
                    "is_open": c.is_open,
                }
                for name, c in self._circuits.items()
            },
        }

    # ── Path 1: DIRECT (intent + quick reply) ────────────────────────

    def _try_direct(self, query: str) -> QueryPlan | None:
        if self._circuits["intent"].is_open:
            return None

        # Quick reply (pattern match)
        try:
            from core.quick_replies import try_quick_reply
            reply = try_quick_reply(query, self._config)
            if reply:
                return QueryPlan(
                    path=ExecPath.DIRECT,
                    skip_llm=True,
                    reason="quick_reply",
                    direct_response=reply,
                    budget_ms=_PATH_BUDGETS[ExecPath.DIRECT],
                )
        except Exception:
            logger.debug("Quick reply check failed", exc_info=True)

        # Intent engine match
        if self._intent is None:
            return None
        try:
            result = self._intent.classify(query)
            if result.intent != "fallback" and result.intent != "empty":
                return QueryPlan(
                    path=ExecPath.DIRECT,
                    skip_llm=True,
                    reason=f"intent:{result.intent}",
                    direct_response=result.response,
                    direct_intent=result.intent,
                    direct_action=result.action,
                    direct_action_args=result.action_args,
                    budget_ms=_PATH_BUDGETS[ExecPath.DIRECT],
                )
        except Exception:
            self._circuits["intent"].record_failure()
            logger.debug("Intent classification failed", exc_info=True)

        return None

    # ── Path 2: CACHE ────────────────────────────────────────────────

    def _try_cache(self, query: str) -> QueryPlan | None:
        if self._cache is None or self._circuits["cache"].is_open:
            return None
        try:
            cached = self._cache.get(query)
            if cached:
                return QueryPlan(
                    path=ExecPath.CACHE,
                    skip_llm=True,
                    reason="cache_hit",
                    direct_response=cached,
                    budget_ms=_PATH_BUDGETS[ExecPath.CACHE],
                )
        except Exception:
            self._circuits["cache"].record_failure()
            logger.debug("Cache lookup failed", exc_info=True)
        return None

    # ── Paths 3-5: LLM routing ───────────────────────────────────────

    def _route_to_llm(
        self,
        query: str,
        ctx: _SystemContext,
        complexity: QueryComplexity,
        requested_tier: CognitiveBudgetTier,
    ) -> QueryPlan:
        qlen = len(query)

        # System-state degradation
        degraded = self._should_degrade(ctx)

        # Circuit breaker: if full LLM is broken, fall back to quick
        full_broken = self._circuits["llm_full"].is_open
        quick_broken = self._circuits["llm_quick"].is_open
        rag_broken = self._circuits["rag"].is_open

        # DEEP path: open-ended or long-form reasoning on a healthy system
        if not degraded and not full_broken and requested_tier is CognitiveBudgetTier.CREATIVE:
            return QueryPlan(
                path=ExecPath.DEEP,
                model=self._full_model,
                model_role="primary",
                runtime_mode="DEEP",
                use_rag=not rag_broken,
                use_memory=True,
                thinking=True,
                reason="complex_query",
                budget_ms=_PATH_BUDGETS[ExecPath.DEEP],
                prompt_hint=self._prompt_hint_for(ExecPath.DEEP),
            )

        # FULL path: substantive queries that need the primary model
        if (
            not degraded
            and not full_broken
            and requested_tier in {
                CognitiveBudgetTier.COMPLEX,
                CognitiveBudgetTier.CREATIVE,
            }
        ):
            use_rag = (
                not rag_broken
                and complexity == QueryComplexity.COMPLEX
            )
            return QueryPlan(
                path=ExecPath.FULL,
                model=self._full_model,
                model_role="primary",
                runtime_mode="SMART",
                use_rag=use_rag,
                use_memory=True,
                thinking=False,
                reason="moderate_query",
                budget_ms=_PATH_BUDGETS[ExecPath.FULL],
                prompt_hint=self._prompt_hint_for(ExecPath.FULL),
            )

        # QUICK path: simple queries, degraded state, or fallback
        if quick_broken:
            return QueryPlan(
                path=ExecPath.FULL,
                model=self._full_model,
                model_role="primary",
                runtime_mode="SMART",
                use_rag=False,
                use_memory=False,
                thinking=False,
                reason="quick_circuit_open_fallback",
                budget_ms=_PATH_BUDGETS[ExecPath.FULL],
                prompt_hint=self._prompt_hint_for(ExecPath.FULL),
            )

        reason = "simple_query"
        if degraded:
            reason = f"degraded:{self._degradation_reason(ctx)}"
        elif full_broken:
            reason = "full_circuit_open_fallback"

        return QueryPlan(
            path=ExecPath.QUICK,
            model=self._quick_model,
            model_role="fast",
            runtime_mode="FAST",
            use_rag=False,
            use_memory=qlen > 30,
            thinking=False,
            reason=reason,
            budget_ms=_PATH_BUDGETS[ExecPath.QUICK],
            prompt_hint=self._prompt_hint_for(ExecPath.QUICK, degraded=degraded),
        )

    def _classify_requested_tier(
        self,
        query: str,
        *,
        complexity: QueryComplexity,
        direct_intent: str | None = None,
        direct_action: str | None = None,
    ) -> CognitiveBudgetTier:
        low = (query or "").strip().lower()
        intent = str(direct_intent or "").strip().lower()
        action = str(direct_action or "").strip().lower()

        if action:
            return CognitiveBudgetTier.COMMAND
        if intent in _INFO_INTENTS:
            return CognitiveBudgetTier.INFO
        if _INFO_HINTS.search(low):
            return CognitiveBudgetTier.INFO
        if _CREATIVE_HINTS.search(low):
            return CognitiveBudgetTier.CREATIVE
        if complexity is QueryComplexity.COMPLEX and len(low) >= self._deep_query_min_chars:
            return CognitiveBudgetTier.CREATIVE
        if complexity is QueryComplexity.COMPLEX:
            return CognitiveBudgetTier.COMPLEX
        return CognitiveBudgetTier.SIMPLE

    @staticmethod
    def _applied_budget_tier(
        plan: QueryPlan,
        requested_tier: CognitiveBudgetTier,
    ) -> CognitiveBudgetTier:
        if plan.path is ExecPath.DIRECT:
            if requested_tier is CognitiveBudgetTier.INFO:
                return CognitiveBudgetTier.INFO
            return CognitiveBudgetTier.COMMAND
        if plan.path is ExecPath.CACHE:
            if requested_tier in {
                CognitiveBudgetTier.COMMAND,
                CognitiveBudgetTier.INFO,
            }:
                return requested_tier
            return CognitiveBudgetTier.SIMPLE
        if plan.path is ExecPath.QUICK:
            return CognitiveBudgetTier.SIMPLE
        if plan.path is ExecPath.FULL:
            return CognitiveBudgetTier.COMPLEX
        if plan.path is ExecPath.DEEP:
            return CognitiveBudgetTier.CREATIVE
        return requested_tier

    def _apply_budget_profile(
        self,
        plan: QueryPlan,
        *,
        requested_tier: CognitiveBudgetTier,
    ) -> QueryPlan:
        applied_tier = self._applied_budget_tier(plan, requested_tier)
        profile = _BUDGET_PROFILES[applied_tier]

        plan.requested_tier = requested_tier.value
        plan.budget_tier = applied_tier.value
        plan.base_budget_ms = float(profile.budget_ms)
        plan.budget_ms = float(profile.budget_ms)
        plan.budget_allow_memory = bool(profile.use_memory)
        plan.budget_allow_rag = bool(profile.use_rag)

        if plan.path in {ExecPath.DIRECT, ExecPath.CACHE}:
            plan.runtime_mode = "FAST"
            plan.use_rag = False
            return plan

        plan.use_memory = bool(plan.use_memory or profile.use_memory)
        plan.use_rag = bool(plan.use_rag and profile.use_rag)
        return plan

    # ── System-state awareness ───────────────────────────────────────

    def _get_system_context(self) -> _SystemContext:
        ctx = _SystemContext()
        if self._silicon:
            try:
                stats = self._silicon.get_stats()
                ctx.memory_pct = stats.memory_pct
                ctx.cpu_pct = stats.cpu_pct
                ctx.battery_pct = getattr(stats, "battery_pct", 100)
                ctx.on_battery = getattr(stats, "on_battery", False)
                ctx.thermal_pressure = getattr(stats, "thermal_pressure", "nominal")
                ctx.is_throttled = getattr(stats, "is_throttled", False)
            except Exception:
                logger.debug("Failed to read silicon stats", exc_info=True)
        return ctx

    def _should_degrade(self, ctx: _SystemContext) -> bool:
        if self._battery_degrade and ctx.on_battery and ctx.battery_pct < 20:
            return True
        if self._thermal_degrade and ctx.is_throttled:
            return True
        if ctx.memory_pct > self._memory_pressure_threshold:
            return True
        return False

    @staticmethod
    def _degradation_reason(ctx: _SystemContext) -> str:
        reasons = []
        if ctx.on_battery and ctx.battery_pct < 20:
            reasons.append(f"battery_{ctx.battery_pct}pct")
        if ctx.is_throttled:
            reasons.append(f"thermal_{ctx.thermal_pressure}")
        if ctx.memory_pct > 85:
            reasons.append(f"memory_{ctx.memory_pct:.0f}pct")
        return "+".join(reasons) or "unknown"

    @staticmethod
    def _prompt_hint_for(path: ExecPath, *, degraded: bool = False) -> str:
        if path == ExecPath.QUICK:
            if degraded:
                return (
                    "Respond directly and briefly. Prioritize latency and avoid "
                    "unnecessary reasoning unless the answer would otherwise be incorrect."
                )
            return (
                "Respond directly and briefly. Prefer the fastest correct answer "
                "and avoid unnecessary detail."
            )
        if path == ExecPath.DEEP:
            return (
                "This query needs deeper reasoning. Think carefully before answering, "
                "use available context or tools when helpful, and prioritize correctness "
                "over speed while keeping the final answer clear."
            )
        return (
            "Give a thoughtful but efficient answer. Use context when it materially "
            "improves the response, but stay concise."
        )

    def _apply_latency_policy(
        self,
        plan: QueryPlan,
        query: str,
        ctx: _SystemContext,
        *,
        complexity: QueryComplexity | None = None,
    ) -> QueryPlan:
        decision = self._latency.get_budget(
            query,
            path=plan.path.value,
            system_state=self._system_context_dict(ctx),
            complexity=complexity,
            base_budget_ms=plan.base_budget_ms,
            budget_tier=plan.budget_tier,
            skip_llm=plan.skip_llm,
            use_rag=plan.use_rag,
            use_memory=plan.use_memory,
            thinking=plan.thinking,
        )
        plan.budget_ms = float(decision.budget_ms)
        plan.rag_budget_ms = float(decision.rag_budget_ms)
        plan.latency_reason = decision.reason
        plan.reduce_context = bool(decision.reduce_context)
        plan.history_turn_limit = int(decision.history_turn_limit)
        plan.memory_limit = int(decision.memory_limit)
        if decision.skip_rag:
            plan.use_rag = False
            plan.rag_budget_ms = 0.0
        if not plan.use_memory:
            plan.memory_limit = 0
        if decision.reduce_context:
            extra_hint = "Keep context usage tight and prioritize only the highest-value evidence."
            plan.prompt_hint = f"{plan.prompt_hint} {extra_hint}".strip() if plan.prompt_hint else extra_hint
        return plan

    @staticmethod
    def _system_context_dict(ctx: _SystemContext) -> dict[str, Any]:
        return {
            "memory_pct": float(ctx.memory_pct),
            "cpu_pct": float(ctx.cpu_pct),
            "battery_pct": int(ctx.battery_pct),
            "on_battery": bool(ctx.on_battery),
            "thermal_pressure": str(ctx.thermal_pressure),
            "is_throttled": bool(ctx.is_throttled),
        }

    # ── User override ────────────────────────────────────────────────

    def _apply_override(
        self, query: str, override: str, ctx: _SystemContext,
    ) -> QueryPlan:
        mode = override.upper().strip()
        if mode == "FAST":
            return QueryPlan(
                path=ExecPath.QUICK,
                model=self._quick_model,
                model_role="fast",
                runtime_mode="FAST",
                reason="user_override:FAST",
                budget_ms=_PATH_BUDGETS[ExecPath.QUICK],
                prompt_hint=self._prompt_hint_for(ExecPath.QUICK, degraded=self._should_degrade(ctx)),
            )
        if mode == "DEEP":
            return QueryPlan(
                path=ExecPath.DEEP,
                model=self._full_model,
                model_role="primary",
                runtime_mode="DEEP",
                use_rag=True,
                use_memory=True,
                thinking=True,
                reason="user_override:DEEP",
                budget_ms=_PATH_BUDGETS[ExecPath.DEEP],
                prompt_hint=self._prompt_hint_for(ExecPath.DEEP),
            )
        # SMART / default → FULL
        return QueryPlan(
            path=ExecPath.FULL,
            model=self._full_model,
            model_role="primary",
            runtime_mode="SMART",
            use_rag=True,
            use_memory=True,
            thinking=False,
            reason=f"user_override:{mode}",
            budget_ms=_PATH_BUDGETS[ExecPath.FULL],
            prompt_hint=self._prompt_hint_for(ExecPath.FULL),
        )

    # ── Event handlers ───────────────────────────────────────────────

    def _on_thermal_warn(self, **_kw: Any) -> None:
        logger.info("Cognitive Kernel: thermal warning received, degrading routing")

    def _on_memory_warn(self, **_kw: Any) -> None:
        logger.info("Cognitive Kernel: memory warning received, degrading routing")

    # ── Metrics + recording ──────────────────────────────────────────

    def _record(self, plan: QueryPlan, t0: float) -> None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_routed += 1
        self._routing_counts[plan.path.value] = (
            self._routing_counts.get(plan.path.value, 0) + 1
        )
        self._budget_counts[plan.budget_tier] = (
            self._budget_counts.get(plan.budget_tier, 0) + 1
        )

        if self._metrics:
            self._metrics.record_latency("cognitive_kernel", elapsed_ms)

        if self._bus:
            self._bus.emit_fast(
                "cognitive_route",
                path=plan.path.value,
                requested_tier=plan.requested_tier,
                budget_tier=plan.budget_tier,
                model=plan.model,
                model_role=plan.model_role,
                runtime_mode=plan.runtime_mode,
                thinking=plan.thinking,
                reason=plan.reason,
                latency_reason=plan.latency_reason,
                base_budget_ms=plan.base_budget_ms,
                budget_ms=plan.budget_ms,
                rag_budget_ms=round(plan.rag_budget_ms, 1),
                budget_allow_memory=plan.budget_allow_memory,
                budget_allow_rag=plan.budget_allow_rag,
                reduce_context=plan.reduce_context,
                elapsed_ms=round(elapsed_ms, 2),
            )

        logger.info(
            "CK route: tier=%s requested=%s path=%s model=%s role=%s mode=%s rag=%s think=%s base=%.0fms budget=%.0fms rag_budget=%.0fms reason=%s latency=%s (%.1fms)",
            plan.budget_tier, plan.requested_tier, plan.path.value, plan.model,
            plan.model_role, plan.runtime_mode, plan.use_rag, plan.thinking,
            plan.base_budget_ms, plan.budget_ms, plan.rag_budget_ms,
            plan.reason, plan.latency_reason, elapsed_ms,
        )


__all__ = [
    "CognitiveKernel",
    "CognitiveBudgetTier",
    "ExecPath",
    "QueryPlan",
]
