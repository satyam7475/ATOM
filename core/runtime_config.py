"""
ATOM V6 — runtime flags (determinism, hot-path logging, in-process cognitive path).
Backward compatible: defaults preserve prior behavior unless env overrides.

ATOM V7: DegradationMode (FULL → LIMITED → SAFE) for graceful GPU/system degradation.
"""
import os
from enum import Enum


class SystemMode(str, Enum):
    NORMAL = "normal"
    CRITICAL = "critical"


class DegradationMode(str, Enum):
    """V7 graceful degradation: tools and background work reduced as pressure rises."""
    FULL = "full"
    LIMITED = "limited"
    SAFE = "safe"


class CognitionDeploymentMode(str, Enum):
    """FUSED = single-process low-latency; DISTRIBUTED = multi-process scaling."""
    FUSED = "fused"
    DISTRIBUTED = "distributed"


def get_cognition_deployment_mode(config: dict | None = None) -> CognitionDeploymentMode:
    raw = (os.environ.get("ATOM_DEPLOYMENT_MODE") or "").strip().lower()
    if raw == "fused":
        return CognitionDeploymentMode.FUSED
    if raw == "distributed":
        return CognitionDeploymentMode.DISTRIBUTED
    if config:
        v = (config.get("v7_gpu") or {}).get("deployment_mode", "distributed")
        if v == "fused":
            return CognitionDeploymentMode.FUSED
    return CognitionDeploymentMode.DISTRIBUTED


def is_fused_gpu_mode(config: dict | None = None) -> bool:
    """True when fused worker / single-process cognition path is selected."""
    if (config or {}).get("v7_gpu", {}).get("fused_gpu_worker"):
        return True
    return get_cognition_deployment_mode(config) == CognitionDeploymentMode.FUSED


_DEGRADATION_OVERRIDE: DegradationMode | None = None


def get_degradation_mode(config: dict | None = None) -> DegradationMode:
    """Current degradation tier (override > env ATOM_DEGRADATION > v7_gpu.degradation_default)."""
    global _DEGRADATION_OVERRIDE
    if _DEGRADATION_OVERRIDE is not None:
        return _DEGRADATION_OVERRIDE
    env = (os.environ.get("ATOM_DEGRADATION") or "").strip().lower()
    if env in ("full", "limited", "safe"):
        return DegradationMode(env)
    if config:
        raw = (config.get("v7_gpu") or {}).get("degradation_default", "full")
        if raw in ("full", "limited", "safe"):
            return DegradationMode(raw)
    return DegradationMode.FULL


def set_degradation_mode(mode: DegradationMode) -> None:
    """Hot-set degradation (e.g. from GPU watchdog or RecoveryManager)."""
    global _DEGRADATION_OVERRIDE
    _DEGRADATION_OVERRIDE = mode


def reset_degradation_mode() -> None:
    """Clear hot override so config/env apply again."""
    global _DEGRADATION_OVERRIDE
    _DEGRADATION_OVERRIDE = None


def get_system_mode() -> SystemMode:
    v = (os.environ.get("ATOM_SYSTEM_MODE") or "normal").strip().lower()
    if v == "critical":
        return SystemMode.CRITICAL
    return SystemMode.NORMAL


def is_critical_mode() -> bool:
    return get_system_mode() == SystemMode.CRITICAL


def use_inprocess_cognitive_path() -> bool:
    """When True, BrainOrchestrator uses LocalCognitivePipeline instead of ZMQ REQ for intent/context/decision."""
    return os.environ.get("ATOM_INPROCESS_COGNITIVE", "1").strip() not in ("0", "false", "no")


def hot_path_debug() -> bool:
    """Verbose logs on orchestrator / execution hot path (default off for latency)."""
    return os.environ.get("ATOM_DEBUG_HOT_PATH", "0").strip() in ("1", "true", "yes")


def v65_performance_defaults() -> dict:
    """Suggested config fragment for production hardening (embedding + workers)."""
    return {
        "performance": {
            "max_worker_threads": int(os.environ.get("ATOM_MAX_WORKERS", "4")),
        },
        "embedding": {
            "device": os.environ.get("ATOM_EMBED_DEVICE", "auto"),
        },
    }
