"""
ATOM OS -- Deployment profiles (corporate laptop vs personal workstation).

Maps high-level deployment.profile to startup guidance and optional UI badges.
Does not override user settings; only logs alignment hints for IT-safe operation.

Owner: Satyam
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger("atom.deployment")

PROFILE_CORPORATE_LAPTOP = "corporate_laptop"
PROFILE_PERSONAL = "personal"
PROFILE_PERSONAL_DESKTOP = "personal_desktop"
PROFILE_WORKSTATION = "workstation"


def audit_corporate_alignment(config: dict[str, Any]) -> list[str]:
    """Return human-readable warnings when config may conflict with corporate norms.

    Informational only — callers decide whether to log as WARNING or INFO.
    """
    warnings: list[str] = []
    feats = config.get("features", {}) or {}
    if feats.get("web_research"):
        warnings.append(
            "features.web_research is true — outbound web research; disable if policy forbids.",
        )
    if feats.get("online_weather"):
        warnings.append(
            "features.online_weather is true — uses network (wttr.in); disable if offline-only.",
        )

    vision = config.get("vision", {}) or {}
    if vision.get("enabled"):
        warnings.append(
            "vision.enabled is true — camera in use; confirm HR/IT policy before meetings.",
        )

    brain = config.get("brain", {}) or {}
    if int(brain.get("n_gpu_layers") or 0) > 0:
        warnings.append(
            "brain.n_gpu_layers > 0 — needs CUDA/GPU stack; often blocked on managed laptops.",
        )

    stt = config.get("stt", {}) or {}
    eng = (stt.get("engine") or "auto").lower()
    if eng not in ("auto", "macos_native", "faster_whisper", "google_online", "google"):
        warnings.append(
            f"stt.engine is {eng} — supported: auto, macos_native, faster_whisper, google_online, google.",
        )
    # Cloud STT engines (Google) ship audio off-machine, so flag them for
    # ALL profiles -- corporate especially, since that's a privacy / DLP
    # concern. The "macos_native is the only option here" warning is
    # different: that's host-specific runtime guidance, only useful when
    # the deployment profile is actually meant for this Mac.
    if eng in ("google_online", "google"):
        warnings.append(
            f"stt.engine is {eng} — uses Google Cloud Speech (network egress); "
            "prefer auto/macos_native/faster_whisper for offline-style operation.",
        )
    dep_profile_for_stt = (
        (config.get("deployment", {}) or {}).get("profile") or ""
    ).strip().lower()
    is_personal_profile = dep_profile_for_stt in (
        PROFILE_PERSONAL,
        PROFILE_PERSONAL_DESKTOP,
        PROFILE_WORKSTATION,
    )
    if (
        sys.platform == "darwin"
        and eng in ("faster_whisper",)
        and is_personal_profile
    ):
        warnings.append(
            f"stt.engine is {eng} — on macOS only Apple native STT runs; use auto or macos_native.",
        )

    tts = config.get("tts", {}) or {}
    if (tts.get("engine") or "sapi").lower() == "edge":
        warnings.append(
            "tts.engine is edge — neural TTS uses network; prefer sapi for air-gapped style.",
        )

    sec = config.get("security", {}) or {}
    if (sec.get("mode") or "strict").lower() != "strict":
        warnings.append(
            "security.mode is not strict — power/shell policy may be relaxed vs corporate baseline.",
        )

    ctrl = config.get("control", {}) or {}
    if (ctrl.get("assistant_mode") or "").lower() == "conversational":
        warnings.append(
            "control.assistant_mode is conversational — broader LLM exposure than hybrid/command_only.",
        )

    dep = config.get("deployment", {}) or {}
    prof = (dep.get("profile") or "").strip().lower()
    if not prof:
        warnings.append(
            'deployment.profile is unset — add "corporate_laptop" to enable this audit trail.',
        )

    return warnings


def log_deployment_bootstrap(config: dict[str, Any]) -> None:
    """Log active deployment profile and corporate alignment hints."""
    dep = config.get("deployment", {}) or {}
    profile = (dep.get("profile") or "unset").strip().lower() or "unset"
    if profile == PROFILE_PERSONAL_DESKTOP:
        profile = PROFILE_PERSONAL
    logger.info("Deployment profile: %s", profile)

    if profile == PROFILE_CORPORATE_LAPTOP:
        hints = audit_corporate_alignment(config)
        if not hints:
            logger.info("Corporate alignment: no policy warnings (current config).")
        for h in hints:
            logger.warning("[corporate] %s", h)
    elif profile == PROFILE_WORKSTATION:
        logger.info(
            "Workstation profile — tune brain.n_gpu_layers, n_ctx, and performance.mode "
            "for your GPU/RAM.",
        )
    elif profile == PROFILE_PERSONAL:
        logger.debug("Personal deployment — corporate audit skipped.")


def deployment_dashboard_badge(config: dict[str, Any]) -> tuple[str, bool]:
    """Label for the web dashboard top bar (empty string = hidden)."""
    dep = config.get("deployment", {}) or {}
    if not dep.get("dashboard_badge", True):
        return "", False

    p = (dep.get("profile") or "").strip().lower()
    labels = {
        PROFILE_CORPORATE_LAPTOP: "CORPORATE",
        PROFILE_WORKSTATION: "WORKSTATION",
        PROFILE_PERSONAL: "PERSONAL",
        PROFILE_PERSONAL_DESKTOP: "PERSONAL",
    }
    label = labels.get(p, "")
    return label, bool(label)
