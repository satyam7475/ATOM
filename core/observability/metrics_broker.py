"""ATOM Sprint N4 -- Prometheus-style metrics + structured /healthz.

ATOM already records a lot of internal counters (see ``core.metrics``)
but there's no single scrape endpoint a Prometheus / Grafana / curl
loop can hit. This module is the *broker*: any subsystem can register
a small "provider" callable that returns its current state (counters,
gauges, last-seen timestamps, errors). The broker can then render:

    * a Prometheus 0.0.4 text exposition (for /metrics)
    * a JSON health rollup (for /healthz)

Both formats are dependency-free. The module is intentionally tiny --
no new client lib, no scheduling -- because the data already exists in
the running subsystems; we only need a uniform shape.

Usage::

    broker = MetricsBroker()
    broker.register("screen_loop", screen_loop.metrics)
    broker.register("cloud_router", cloud_router.stats)
    broker.register("atom_room", room.snapshot)

Then in the aiohttp app::

    app.router.add_get("/metrics", broker.handle_metrics)
    app.router.add_get("/healthz", broker.handle_healthz)

Provider callables can return a flat dict[str, number|str|dict] OR a
:class:`ProviderSnapshot`. Numbers are emitted as gauges; strings are
emitted as gauge-with-label fallback; nested dicts are flattened with
underscore-joined names.

Owner: Boss (Satyam).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("atom.observability.metrics")


_LABEL_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe_label(name: str) -> str:
    """Normalise a key into a Prometheus-safe metric name fragment."""
    s = _LABEL_SAFE_RE.sub("_", str(name))
    if not s:
        return "x"
    if s[0].isdigit():
        s = "_" + s
    return s.lower()


def _flatten(
    prefix: str, blob: Any, out: dict[str, Any], depth: int = 0,
) -> None:
    if depth > 5:
        return
    if isinstance(blob, dict):
        for k, v in blob.items():
            key = f"{prefix}_{_safe_label(k)}" if prefix else _safe_label(k)
            _flatten(key, v, out, depth + 1)
    elif isinstance(blob, (list, tuple)):
        out[prefix + "_count"] = len(blob)
    else:
        out[prefix] = blob


# ── data ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ProviderSnapshot:
    """Optional richer return from a provider."""

    name: str
    healthy: bool = True
    error: str = ""
    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)


# ── broker ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _Registration:
    name: str
    provider: Callable[[], Any]
    enabled: bool = True


class MetricsBroker:
    """Registry + Prometheus-text / JSON renderer."""

    def __init__(self, *, namespace: str = "atom") -> None:
        self.namespace = _safe_label(namespace)
        self._providers: dict[str, _Registration] = {}
        self._created_at = time.time()

    def register(self, name: str, provider: Callable[[], Any]) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        self._providers[name] = _Registration(
            name=_safe_label(name), provider=provider,
        )
        logger.debug("MetricsBroker registered provider: %s", name)

    def unregister(self, name: str) -> None:
        self._providers.pop(_safe_label(name), None)

    def set_enabled(self, name: str, enabled: bool) -> None:
        reg = self._providers.get(_safe_label(name))
        if reg is not None:
            reg.enabled = bool(enabled)

    # ── snapshot collection ──────────────────────────────────────

    def collect(self) -> dict[str, dict[str, Any]]:
        """Pull a snapshot from every registered provider."""
        out: dict[str, dict[str, Any]] = {}
        for reg in self._providers.values():
            if not reg.enabled:
                out[reg.name] = {"_disabled": True}
                continue
            try:
                raw = reg.provider()
            except Exception as exc:
                logger.exception(
                    "MetricsBroker provider %s raised", reg.name,
                )
                out[reg.name] = {"_error": str(exc)[:200]}
                continue
            if isinstance(raw, ProviderSnapshot):
                out[reg.name] = {
                    "_healthy": raw.healthy,
                    "_error": raw.error,
                    "counters": raw.counters,
                    "gauges": raw.gauges,
                    "info": raw.info,
                }
            elif isinstance(raw, dict):
                out[reg.name] = raw
            else:
                out[reg.name] = {"value": raw}
        return out

    # ── Prometheus text rendering ────────────────────────────────

    def render_prometheus(self) -> str:
        snap = self.collect()
        ns = self.namespace
        lines: list[str] = [
            f"# ATOM metrics broker -- collected {time.time():.3f}",
            f"# uptime_seconds = {time.time() - self._created_at:.3f}",
        ]
        # broker uptime
        lines.append(f"# HELP {ns}_broker_uptime_seconds Broker uptime")
        lines.append(f"# TYPE {ns}_broker_uptime_seconds gauge")
        lines.append(
            f"{ns}_broker_uptime_seconds "
            f"{time.time() - self._created_at:.3f}",
        )

        for provider_name, blob in snap.items():
            if not isinstance(blob, dict):
                continue
            error = str(blob.get("_error") or "")
            healthy_tag = "1" if not error else "0"
            lines.append(
                f"# HELP {ns}_{provider_name}_healthy "
                f"1 = healthy, 0 = error",
            )
            lines.append(f"# TYPE {ns}_{provider_name}_healthy gauge")
            lines.append(
                f"{ns}_{provider_name}_healthy {healthy_tag}",
            )

            flat: dict[str, Any] = {}
            for top_key, value in blob.items():
                if top_key.startswith("_"):
                    continue
                _flatten(_safe_label(top_key), value, flat)

            for metric, value in flat.items():
                if isinstance(value, bool):
                    num = 1 if value else 0
                elif isinstance(value, (int, float)):
                    num = value
                elif value is None:
                    num = 0
                elif isinstance(value, str):
                    # try numeric coerce
                    try:
                        num = float(value)
                    except (TypeError, ValueError):
                        # emit as info label gauge
                        info_safe = _LABEL_SAFE_RE.sub(
                            "_", value[:40],
                        ) or "unknown"
                        info_metric = f"{ns}_{provider_name}_{metric}_info"
                        lines.append(f"# TYPE {info_metric} gauge")
                        lines.append(
                            f'{info_metric}{{value="{info_safe}"}} 1',
                        )
                        continue
                else:
                    continue
                full = f"{ns}_{provider_name}_{metric}"
                lines.append(f"# TYPE {full} gauge")
                lines.append(f"{full} {num}")

        return "\n".join(lines) + "\n"

    # ── /healthz JSON ────────────────────────────────────────────

    def render_healthz(self) -> dict[str, Any]:
        snap = self.collect()
        all_ok = True
        per_provider: dict[str, Any] = {}
        for name, blob in snap.items():
            ok = True
            if isinstance(blob, dict):
                if blob.get("_error"):
                    ok = False
                if blob.get("_healthy") is False:
                    ok = False
                if blob.get("_disabled"):
                    ok = False
            per_provider[name] = {
                "ok": ok,
                "snapshot": blob,
            }
            if not ok:
                all_ok = False
        return {
            "ok": all_ok,
            "ts": time.time(),
            "uptime_s": round(time.time() - self._created_at, 3),
            "providers": per_provider,
        }

    # ── aiohttp helpers (optional) ───────────────────────────────

    async def handle_metrics(self, _request: Any) -> Any:
        from aiohttp import web

        body = self.render_prometheus()
        return web.Response(
            text=body,
            content_type="text/plain",
            charset="utf-8",
        )

    async def handle_healthz(self, _request: Any) -> Any:
        from aiohttp import web

        return web.json_response(self.render_healthz())


__all__ = ["MetricsBroker", "ProviderSnapshot"]
