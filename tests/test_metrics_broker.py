"""Regression tests for Sprint N4 -- :class:`MetricsBroker`."""

from __future__ import annotations

import json

import pytest

from core.observability.metrics_broker import (
    MetricsBroker,
    ProviderSnapshot,
)


def test_broker_collects_dict_provider() -> None:
    b = MetricsBroker(namespace="atom_test")

    def provider() -> dict[str, object]:
        return {"persisted": 12, "errors": 0, "running_for_s": 1.5}

    b.register("screen_loop", provider)
    snap = b.collect()
    assert "screen_loop" in snap
    assert snap["screen_loop"]["persisted"] == 12


def test_broker_renders_prometheus_with_help_lines() -> None:
    b = MetricsBroker(namespace="atom_test")
    b.register(
        "router",
        lambda: {"local_count": 3, "cloud_count": 1, "ok": True},
    )
    text = b.render_prometheus()

    assert "atom_test_broker_uptime_seconds" in text
    assert "atom_test_router_local_count 3" in text
    assert "atom_test_router_cloud_count 1" in text
    assert "atom_test_router_healthy 1" in text
    # type lines must be there for prometheus parsers
    assert "# TYPE atom_test_router_local_count gauge" in text


def test_broker_marks_provider_unhealthy_on_exception() -> None:
    b = MetricsBroker(namespace="atom_test")

    def bad_provider() -> dict[str, object]:
        raise RuntimeError("kaboom")

    b.register("subsys", bad_provider)
    snap = b.collect()
    assert "_error" in snap["subsys"]
    text = b.render_prometheus()
    assert "atom_test_subsys_healthy 0" in text


def test_broker_supports_provider_snapshot() -> None:
    b = MetricsBroker(namespace="atom_test")

    def provider() -> ProviderSnapshot:
        return ProviderSnapshot(
            name="cloud",
            healthy=True,
            counters={"calls": 4},
            gauges={"latency_ms": 312.0},
            info={"provider": "gemini"},
        )

    b.register("cloud", provider)
    text = b.render_prometheus()
    assert "atom_test_cloud_counters_calls 4" in text
    assert "atom_test_cloud_gauges_latency_ms 312" in text


def test_healthz_aggregates_per_provider_status() -> None:
    b = MetricsBroker(namespace="atom_test")
    b.register("ok_one", lambda: {"x": 1})
    b.register("bad_one", lambda: (_ for _ in ()).throw(RuntimeError("nope")))

    health = b.render_healthz()
    assert health["ok"] is False
    assert health["providers"]["ok_one"]["ok"] is True
    assert health["providers"]["bad_one"]["ok"] is False


def test_unregister_removes_provider() -> None:
    b = MetricsBroker(namespace="x")
    b.register("p1", lambda: {"a": 1})
    b.unregister("p1")
    assert "p1" not in b.collect()


def test_disabled_provider_marked_disabled() -> None:
    b = MetricsBroker(namespace="x")
    b.register("p1", lambda: {"a": 1})
    b.set_enabled("p1", False)
    snap = b.collect()
    assert snap["p1"].get("_disabled") is True
    health = b.render_healthz()
    assert health["providers"]["p1"]["ok"] is False
