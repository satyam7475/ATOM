"""iPhone Shortcuts bridge: auth, payload, single-device, rate, port fallback.

These are unit tests (no real iPhone). Every request is crafted with
``aiohttp.test_utils`` so the bridge's HTTP contract is locked down
regardless of what the iOS Shortcut app does in the field.

Run: ``python3 -m pytest tests/test_iphone_bridge.py -v``
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cross_device.bridge_auth import (  # noqa: E402
    AuthAuditLog,
    generate_or_load_token,
    verify_token,
)
from core.cross_device.iphone_bridge import IPhoneBridge  # noqa: E402
from core.cross_device.trusted_device import (  # noqa: E402
    TrustedIPhoneRegistry,
    hash_device_id,
)


# ────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────

@pytest.fixture()
def atom_root(tmp_path: Path) -> Path:
    """Synthetic ATOM root with config/ + logs/ + data/ subdirs.

    Using a fresh tmp_path per test keeps the trusted-device state and
    the audit log isolated."""
    (tmp_path / "config").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    # Marker so _infer_atom_root fallbacks don't accidentally pick the
    # real ATOM checkout if the test suite is run from the repo.
    (tmp_path / "config" / "settings.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def captured_events() -> list[tuple[str, dict]]:
    """A shared list that ``fake_emit`` below writes into."""
    return []


@pytest.fixture()
def fake_emit(captured_events):
    def _emit(event: str, **data):
        captured_events.append((event, data))
    return _emit


def _bridge_config(root: Path, port: int = 0) -> dict:
    """Build a config dict pointing at *root* and an ephemeral port.

    ``port=0`` asks the OS for any free port (and the bridge's built-in
    fallback still exercises the +1/+2 branch if we start with a known
    busy port)."""
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    return {
        "cross_device": {
            "enabled": True,
            "bridge_port": port,
            "bind_host": "127.0.0.1",
            "token_path": str(root / "config" / "bridge_token"),
            "audit_log_path": str(root / "logs" / "atom_bridge_audit.jsonl"),
            "trusted_device_path": str(root / "data" / "trusted_iphone.json"),
            "port_file_path": str(root / "logs" / "atom_bridge.port"),
        },
    }


# ────────────────────────────────────────────
# bridge_auth
# ────────────────────────────────────────────

def test_token_minted_once_then_reused(atom_root: Path) -> None:
    p = atom_root / "config" / "bridge_token"
    t1 = generate_or_load_token(p)
    assert len(t1) >= 32
    t2 = generate_or_load_token(p)
    assert t1 == t2, "token must persist across reads"


def test_token_file_mode_0600(atom_root: Path) -> None:
    p = atom_root / "config" / "bridge_token"
    generate_or_load_token(p)
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600, f"token file must be 0600, got {oct(mode)}"


def test_verify_token_constant_time_semantics() -> None:
    t = "a" * 32
    assert verify_token(t, t) is True
    assert verify_token(t, "a" * 31) is False
    assert verify_token(t, "b" * 32) is False
    assert verify_token(t, None) is False
    assert verify_token("", t) is False


def test_audit_log_rolling_flood_threshold(atom_root: Path) -> None:
    log = AuthAuditLog(atom_root / "logs" / "atom_bridge_audit.jsonl")
    warned = []
    for i in range(9):
        warned.append(log.record_failure(
            source_ip="127.0.0.1", endpoint="/faceid", reason="bad_token"))
    assert not any(warned), "must not warn below threshold"
    triggered = log.record_failure(
        source_ip="127.0.0.1", endpoint="/faceid", reason="bad_token")
    assert triggered, "10th failure in the rolling window must warn"
    # Subsequent failure inside the same window must not re-warn.
    again = log.record_failure(
        source_ip="127.0.0.1", endpoint="/faceid", reason="bad_token")
    assert not again


# ────────────────────────────────────────────
# trusted_device
# ────────────────────────────────────────────

def test_trusted_device_first_handshake_wins(atom_root: Path) -> None:
    r = TrustedIPhoneRegistry(atom_root / "data" / "trusted_iphone.json")
    ok1, why1 = r.register_or_verify("iphone-abc", label="Boss iPhone")
    assert ok1 and why1 == "registered"
    ok2, why2 = r.register_or_verify("iphone-abc")
    assert ok2 and why2 == "already_registered"


def test_trusted_device_second_device_rejected(atom_root: Path) -> None:
    r = TrustedIPhoneRegistry(atom_root / "data" / "trusted_iphone.json")
    assert r.register_or_verify("iphone-boss")[0] is True
    ok, why = r.register_or_verify("iphone-housemate")
    assert ok is False
    assert why == "conflict"


def test_trusted_device_reset_restores_slot(atom_root: Path) -> None:
    p = atom_root / "data" / "trusted_iphone.json"
    r = TrustedIPhoneRegistry(p)
    r.register_or_verify("iphone-a")
    r.reset()
    assert not p.exists()
    ok, why = r.register_or_verify("iphone-b")
    assert ok is True and why == "registered"


def test_trusted_device_hash_is_stable_and_not_raw() -> None:
    h = hash_device_id("iphone-boss-15-pro")
    assert len(h) == 64
    assert "iphone" not in h, "hash must not leak raw device id"
    assert h == hash_device_id("iphone-boss-15-pro")


# ────────────────────────────────────────────
# IPhoneBridge HTTP surface (integration)
# ────────────────────────────────────────────

async def _start_bridge(root: Path, emit) -> IPhoneBridge:
    cfg = _bridge_config(root)
    b = IPhoneBridge(config=cfg, emit=emit, atom_root=root)
    started = await b.start()
    assert started, "bridge must start on a free port"
    assert b.actual_port is not None
    return b


@pytest.mark.asyncio
async def test_bridge_health_is_unauthed_and_200(atom_root, fake_emit) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{b.actual_port}/health") as r:
                assert r.status == 200
                body = await r.json()
                assert body == {"ok": True, "version": 1}
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_bridge_faceid_requires_token(atom_root, fake_emit) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/faceid",
                json={"device_id": "iphone-a", "verified": True},
            ) as r:
                assert r.status == 401

            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/faceid",
                headers={"X-ATOM-Token": "wrong"},
                json={"device_id": "iphone-a", "verified": True},
            ) as r:
                assert r.status == 401
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_bridge_faceid_good_token_emits_event(atom_root, fake_emit, captured_events) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/faceid",
                headers={"X-ATOM-Token": b.token},
                json={
                    "device_id": "iphone-boss",
                    "verified": True,
                    "label": "Boss iPhone",
                },
            ) as r:
                assert r.status == 200, await r.text()
    finally:
        await b.stop()

    events = [e for e in captured_events if e[0] == "iphone.faceid.verified"]
    assert len(events) == 1
    _, data = events[0]
    assert data["verified"] is True
    assert data["label"] == "Boss iPhone"
    assert "device_id" in data and data["device_id"] == "iphone-boss"


@pytest.mark.asyncio
async def test_bridge_presence_validates_state(atom_root, fake_emit, captured_events) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/presence",
                headers={"X-ATOM-Token": b.token},
                json={"device_id": "iphone-boss", "state": "at_desk"},
            ) as r:
                assert r.status == 200

            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/presence",
                headers={"X-ATOM-Token": b.token},
                json={"device_id": "iphone-boss", "state": "nonsense"},
            ) as r:
                assert r.status == 400
    finally:
        await b.stop()

    presence_events = [e for e in captured_events if e[0] == "iphone.presence.changed"]
    assert len(presence_events) == 1
    assert presence_events[0][1]["state"] == "at_desk"


@pytest.mark.asyncio
async def test_bridge_trigger_sanitizes_name(atom_root, fake_emit) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/trigger",
                headers={"X-ATOM-Token": b.token},
                json={"device_id": "iphone-boss", "name": ""},
            ) as r:
                assert r.status == 400

            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/trigger",
                headers={"X-ATOM-Token": b.token},
                json={
                    "device_id": "iphone-boss",
                    "name": "x" * 200,
                },
            ) as r:
                assert r.status == 400
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_bridge_single_device_lock(atom_root, fake_emit) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/faceid",
                headers={"X-ATOM-Token": b.token},
                json={"device_id": "iphone-boss", "verified": True},
            ) as r:
                assert r.status == 200

            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/faceid",
                headers={"X-ATOM-Token": b.token},
                json={"device_id": "iphone-housemate", "verified": True},
            ) as r:
                assert r.status == 409
                body = await r.json()
                assert body["error"] == "device_conflict"
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_bridge_rate_limit_second_request_throttled(atom_root, fake_emit) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            payload = {"device_id": "iphone-boss", "state": "at_desk"}
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/presence",
                headers={"X-ATOM-Token": b.token},
                json=payload,
            ) as r:
                assert r.status == 200

            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/presence",
                headers={"X-ATOM-Token": b.token},
                json=payload,
            ) as r:
                assert r.status == 429
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_bridge_port_fallback_when_preferred_busy(atom_root, fake_emit) -> None:
    """Binding the preferred port elsewhere forces the bridge to +1."""
    import aiohttp

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        busy_port = holder.getsockname()[1]
        holder.listen(1)

        cfg = _bridge_config(atom_root, port=busy_port)
        b = IPhoneBridge(config=cfg, emit=fake_emit, atom_root=atom_root)
        try:
            started = await b.start()
            assert started
            assert b.actual_port != busy_port
            async with aiohttp.ClientSession() as s:
                async with s.get(f"http://127.0.0.1:{b.actual_port}/health") as r:
                    assert r.status == 200
        finally:
            await b.stop()

    port_file = atom_root / "logs" / "atom_bridge.port"
    assert port_file.exists()
    assert int(port_file.read_text().strip()) == b.actual_port


@pytest.mark.asyncio
async def test_bridge_body_too_large_rejected(atom_root, fake_emit) -> None:
    import aiohttp
    b = await _start_bridge(atom_root, fake_emit)
    try:
        async with aiohttp.ClientSession() as s:
            huge = {"device_id": "iphone-boss", "name": "x", "args": {"blob": "a" * 200_000}}
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/trigger",
                headers={
                    "X-ATOM-Token": b.token,
                    "Content-Type": "application/json",
                },
                data=json.dumps(huge),
            ) as r:
                assert r.status in (400, 413), await r.text()
    finally:
        await b.stop()


# ────────────────────────────────────────────
# Schema validation for the new cross_device block
# ────────────────────────────────────────────

def test_cross_device_config_schema_validates() -> None:
    """The new cross_device block must validate against config_schema."""
    from core.config_schema import validate_config

    cfg = {
        "cross_device": {
            "enabled": True,
            "bridge_port": 8787,
            "bind_host": "127.0.0.1",
            "faceid_freshness_s": 300,
            "allow_origins": ["127.0.0.1"],
            "token_path": "config/bridge_token",
            "audit_log_path": "logs/atom_bridge_audit.jsonl",
            "trusted_device_path": "data/trusted_iphone.json",
            "port_file_path": "logs/atom_bridge.port",
        },
    }
    errors = validate_config(cfg)
    for err in errors:
        assert "cross_device" not in err, f"unexpected schema error: {err}"


def test_real_settings_json_has_cross_device_block() -> None:
    """settings.json ships a fully-formed cross_device block.

    The iPhone bridge defaulted off in the original Phase 1 sprint
    because the listener wasn't yet wired into ``main.py`` boot --
    enabling it would have done nothing.  Now that ``main.py`` starts
    the bridge (and prints a setup banner with the auto-generated
    token) we default it ON so a fresh boot is one Shortcuts run away
    from a paired iPhone.  Owners who don't have an iPhone (or simply
    don't want the listener) can flip ``enabled`` to false; nothing
    else needs to change.
    """
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "settings.json").read_text(encoding="utf-8"))
    cd = cfg.get("cross_device")
    assert isinstance(cd, dict), "cross_device block missing"
    # New default: ON. The bridge listens on loopback only, and stays
    # entirely passive until an authenticated /faceid POST arrives.
    assert cd.get("enabled") is True, (
        "default flipped to True so main.py boot starts the listener "
        "and shows the iPhone setup banner"
    )
    assert cd.get("bind_host") == "127.0.0.1", "bridge must default to loopback only"
    assert int(cd.get("bridge_port", 0)) >= 1024
    assert cd.get("bridge_port") != cfg.get("ui", {}).get("web_port"), \
        "bridge port must not clash with ui.web_port"


# ────────────────────────────────────────────
# Sprint P4.4 (Apr 26 2026): OpenAI-compat /v1/* shim
# ────────────────────────────────────────────

async def _fake_chat_stream_factory(text_chunks: list[str]):
    """Return an async generator that yields the supplied chunks then closes."""
    async def _gen(messages, **kwargs):
        for chunk in text_chunks:
            await __import__("asyncio").sleep(0)
            yield chunk, False
        yield "", True
    return _gen


@pytest.mark.asyncio
async def test_v1_models_requires_auth_and_returns_atom_local(
    atom_root, fake_emit,
) -> None:
    import aiohttp
    cfg = _bridge_config(atom_root)
    chat_stream = await _fake_chat_stream_factory(["hi"])
    b = IPhoneBridge(
        config=cfg,
        emit=fake_emit,
        atom_root=atom_root,
        chat_stream=chat_stream,
    )
    started = await b.start()
    assert started
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{b.actual_port}/v1/models") as r:
                assert r.status == 401
            async with s.get(
                f"http://127.0.0.1:{b.actual_port}/v1/models",
                headers={"X-ATOM-Token": b.token},
            ) as r:
                assert r.status == 200
                body = await r.json()
                assert body["object"] == "list"
                assert any(
                    m.get("id") == "atom-local" for m in body["data"]
                ), body
            async with s.get(
                f"http://127.0.0.1:{b.actual_port}/v1/models",
                headers={"Authorization": f"Bearer {b.token}"},
            ) as r:
                assert r.status == 200, "Bearer auth must work for Enchanted"
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_v1_chat_completions_non_stream(atom_root, fake_emit) -> None:
    import aiohttp
    cfg = _bridge_config(atom_root)
    chat_stream = await _fake_chat_stream_factory(
        ["Hi ", "Boss", "."]
    )
    b = IPhoneBridge(
        config=cfg,
        emit=fake_emit,
        atom_root=atom_root,
        chat_stream=chat_stream,
    )
    await b.start()
    try:
        async with aiohttp.ClientSession() as s:
            payload = {
                "model": "atom-local",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            }
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/v1/chat/completions",
                headers={"X-ATOM-Token": b.token},
                json=payload,
            ) as r:
                assert r.status == 200
                body = await r.json()
                assert body["object"] == "chat.completion"
                assert body["choices"][0]["message"]["content"] == (
                    "Hi Boss."
                )
                assert body["choices"][0]["finish_reason"] == "stop"
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_v1_chat_completions_stream_terminates_with_done(
    atom_root, fake_emit,
) -> None:
    import aiohttp
    cfg = _bridge_config(atom_root)
    chat_stream = await _fake_chat_stream_factory(
        ["Hello", " ", "Boss"]
    )
    b = IPhoneBridge(
        config=cfg,
        emit=fake_emit,
        atom_root=atom_root,
        chat_stream=chat_stream,
    )
    await b.start()
    try:
        async with aiohttp.ClientSession() as s:
            payload = {
                "model": "atom-local",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/v1/chat/completions",
                headers={"X-ATOM-Token": b.token},
                json=payload,
            ) as r:
                assert r.status == 200
                assert r.headers.get("Content-Type", "").startswith(
                    "text/event-stream"
                )
                body = await r.text()
        # SSE frames + a [DONE] terminator.
        assert "data: [DONE]" in body
        # We expect at least one delta with content="Hello".
        assert "Hello" in body
        # The framing uses "data: " prefix on every JSON chunk.
        assert "\"object\": \"chat.completion.chunk\"" in body
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_v1_chat_completions_no_handler_returns_503(
    atom_root, fake_emit,
) -> None:
    import aiohttp
    cfg = _bridge_config(atom_root)
    # No chat_stream provided -> /v1/chat/completions must NOT register.
    b = IPhoneBridge(config=cfg, emit=fake_emit, atom_root=atom_root)
    await b.start()
    try:
        async with aiohttp.ClientSession() as s:
            payload = {"messages": [{"role": "user", "content": "x"}]}
            async with s.post(
                f"http://127.0.0.1:{b.actual_port}/v1/chat/completions",
                headers={"X-ATOM-Token": b.token},
                json=payload,
            ) as r:
                # Without a handler the route is not registered -> 404.
                assert r.status in (404, 503), (
                    f"expected 404 or 503 without handler, got {r.status}"
                )
    finally:
        await b.stop()


# ────────────────────────────────────────────
# Sprint P4.6 (Apr 26 2026): /badge unified status surface
# ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_badge_no_provider_returns_unknown(atom_root, fake_emit) -> None:
    """``/badge`` is always registered. Without a status_provider it
    must return a stable shape (level=unknown), never 404 -- so menubar
    pollers that start before ATOM finishes wiring don't see a route
    flip-flop."""
    import aiohttp
    cfg = _bridge_config(atom_root)
    b = IPhoneBridge(config=cfg, emit=fake_emit, atom_root=atom_root)
    await b.start()
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{b.actual_port}/badge") as r:
                assert r.status == 200
                body = await r.json()
                assert "level" in body and "text" in body and "color" in body
                assert body["level"] == "unknown"
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_badge_uses_provider_after_late_wiring(
    atom_root, fake_emit,
) -> None:
    """Wire a status_provider AFTER ``start()`` -- mirrors how main.py
    attaches the HealthSnapshotBuilder rollup once subsystems finish
    coming online -- and confirm /badge picks it up live without
    needing a restart."""
    import aiohttp
    cfg = _bridge_config(atom_root)
    b = IPhoneBridge(config=cfg, emit=fake_emit, atom_root=atom_root)
    await b.start()

    state = {
        "ok": False,
        "status": "warn",
        "uptime_s": 12.5,
        "subsystems": {"stt": {"status": "degraded"}, "bus": {"status": "ok"}},
        "badge": {
            "ok": False,
            "level": "warn",
            "color": "amber",
            "text": "ATOM has 1 warning",
            "headline": "stt: degraded",
            "warnings": [{"name": "stt", "status": "degraded"}],
            "criticals": [],
            "subsystems_total": 2,
            "uptime_s": 12.5,
        },
    }

    def _provider() -> dict:
        return state

    b._status_provider = _provider
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{b.actual_port}/badge") as r:
                assert r.status == 200
                body = await r.json()
                assert body["level"] == "warn"
                assert body["text"] == "ATOM has 1 warning"
                assert body["headline"] == "stt: degraded"
                assert body["subsystems_total"] == 2
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_badge_provider_exception_renders_unknown(
    atom_root, fake_emit,
) -> None:
    """If the wired provider blows up mid-poll, the bridge must still
    return a 200 with level=unknown so the menubar app degrades
    gracefully instead of showing a transient 500 every 5 seconds."""
    import aiohttp
    cfg = _bridge_config(atom_root)
    b = IPhoneBridge(config=cfg, emit=fake_emit, atom_root=atom_root)
    await b.start()

    def _bad_provider() -> dict:
        raise RuntimeError("snapshot crashed mid-collect")

    b._status_provider = _bad_provider
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{b.actual_port}/badge") as r:
                assert r.status == 200
                body = await r.json()
                assert body["level"] == "unknown"
                assert "ATOM" in body["text"]
    finally:
        await b.stop()
