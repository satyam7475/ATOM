"""ATOM Sprint Ω8 -- consume external MCP servers from inside ATOM.

This is the mirror of :mod:`atom_mcp_server`: instead of exposing ATOM
to other clients, it lets ATOM call out to external MCP servers and
register their tools into the local :class:`ToolRegistry`. The moment a
server is registered, the LLM can pick its tools just like any built-in
ATOM action -- the existing :class:`ActionExecutor` security gate still
applies on every call.

Design constraints
------------------
* **stdio transport only** for the first cut. That covers the vast
  majority of community MCP servers (filesystem, git, github, sqlite,
  fetch, brave-search, postgres, etc.).
* **Lazy connect**: we don't open a process until the first tool from
  that server is actually called. This keeps boot fast and means a
  misconfigured server only fails its own calls, never the whole router.
* **Per-server timeout**: every external call has a hard wall-clock
  ceiling so a slow MCP server can never block ATOM's reasoning loop.
* **Sanitized tool names**: external tool names are prefixed
  ``<server>__<tool>`` so they can't collide with ATOM built-ins.

Wiring
------
1. ``config/mcp_servers.json`` lists the servers to load (see the
   bundled example).
2. :func:`load_mcp_servers_from_config` parses the file into
   :class:`MCPServerSpec` objects.
3. :class:`AtomMCPClient` registers each spec's tools into the
   :class:`ToolRegistry` and dispatches calls back to the right
   server.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.mcp.client")

if TYPE_CHECKING:
    from mcp import ClientSession

    from core.reasoning.tool_registry import ToolRegistry


_DEFAULT_CONFIG_PATH = Path("config") / "mcp_servers.json"
_DEFAULT_CALL_TIMEOUT_S = 8.0
_DEFAULT_LIST_TIMEOUT_S = 6.0


# ── Data shapes ─────────────────────────────────────────────────────


@dataclass(slots=True)
class MCPServerSpec:
    """One external MCP server to mount into ATOM."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    call_timeout_s: float = _DEFAULT_CALL_TIMEOUT_S
    list_timeout_s: float = _DEFAULT_LIST_TIMEOUT_S
    safety_level: str = "moderate"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPServerSpec:
        return cls(
            name=str(data["name"]).strip(),
            command=str(data["command"]).strip(),
            args=[str(a) for a in (data.get("args") or [])],
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            enabled=bool(data.get("enabled", True)),
            description=str(data.get("description") or ""),
            call_timeout_s=float(
                data.get("call_timeout_s", _DEFAULT_CALL_TIMEOUT_S),
            ),
            list_timeout_s=float(
                data.get("list_timeout_s", _DEFAULT_LIST_TIMEOUT_S),
            ),
            safety_level=str(data.get("safety_level") or "moderate"),
        )


def load_mcp_servers_from_config(
    path: str | Path | None = None,
) -> list[MCPServerSpec]:
    """Read ``config/mcp_servers.json`` (or a custom path)."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not p.exists():
        logger.info("MCP server config %s missing -- no external MCP servers", p)
        return []
    try:
        raw = json.loads(p.read_text())
    except Exception as exc:
        logger.warning("Failed to parse MCP server config %s: %s", p, exc)
        return []
    items = raw.get("servers") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        logger.warning("MCP server config %s: 'servers' must be a list", p)
        return []
    out: list[MCPServerSpec] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        try:
            spec = MCPServerSpec.from_dict(entry)
        except Exception as exc:
            logger.warning(
                "Skipping malformed MCP server entry %r: %s", entry, exc,
            )
            continue
        if not spec.enabled:
            continue
        out.append(spec)
    return out


# ── Client ─────────────────────────────────────────────────────────


class AtomMCPClient:
    """Manages a pool of external MCP server connections.

    Threading model: every :meth:`call` opens its own stdio session,
    runs the call, and tears it down. We deliberately *don't* keep
    persistent sessions in the first cut because the MCP stdio spec is
    process-per-client and any long-lived connection has to be owned
    by an asyncio task tree -- which complicates wiring with ATOM's
    sync action dispatcher. The cost is one ~50 ms process spawn per
    external call, which is dwarfed by the LLM's own latency.

    Future optimization: add an opt-in persistent session per server,
    guarded by a heartbeat watchdog and proper task-group teardown.
    """

    NAME_PREFIX = "mcp"
    """Tool names registered into the ATOM registry are prefixed
    ``mcp_<server>__<tool>`` so they never collide with built-ins."""

    def __init__(
        self,
        specs: list[MCPServerSpec],
        *,
        tool_registry: ToolRegistry,
    ) -> None:
        self._specs: dict[str, MCPServerSpec] = {s.name: s for s in specs}
        self._registry = tool_registry
        self._registered_tools: dict[str, str] = {}  # atom_name -> server
        self._registered_count = 0
        self._failures: dict[str, str] = {}

    # ── Public surface ────────────────────────────────────────

    @property
    def server_count(self) -> int:
        return len(self._specs)

    @property
    def registered_tool_count(self) -> int:
        return self._registered_count

    @property
    def failures(self) -> dict[str, str]:
        return dict(self._failures)

    def stats(self) -> dict[str, Any]:
        return {
            "server_count": self.server_count,
            "registered_tool_count": self.registered_tool_count,
            "failures": dict(self._failures),
            "servers": [
                {
                    "name": s.name,
                    "command": s.command,
                    "enabled": s.enabled,
                    "tools_registered": sum(
                        1 for srv in self._registered_tools.values()
                        if srv == s.name
                    ),
                }
                for s in self._specs.values()
            ],
        }

    async def discover_and_register_all(self) -> int:
        """Connect to every spec, list its tools, register them.

        Returns the total number of tools registered. Per-server
        failures are recorded in :attr:`failures` and don't prevent
        other servers from registering.
        """
        from core.reasoning.tool_registry import Tool, ToolParameter

        if not self._specs:
            logger.info("No external MCP servers configured.")
            return 0

        for spec in self._specs.values():
            try:
                tools = await asyncio.wait_for(
                    self._list_tools(spec),
                    timeout=spec.list_timeout_s,
                )
            except asyncio.TimeoutError:
                msg = f"list_tools timed out after {spec.list_timeout_s:.0f}s"
                self._failures[spec.name] = msg
                logger.warning("MCP server %s: %s", spec.name, msg)
                continue
            except Exception as exc:
                msg = f"list_tools failed: {exc}"
                self._failures[spec.name] = msg
                logger.warning("MCP server %s: %s", spec.name, msg)
                continue

            for mcp_tool in tools:
                atom_name = self._atom_tool_name(spec.name, mcp_tool.name)
                params = self._mcp_schema_to_atom_params(
                    getattr(mcp_tool, "inputSchema", None) or {},
                )
                description = (
                    f"[{spec.name}] {getattr(mcp_tool, 'description', '') or mcp_tool.name}"
                )

                handler = self._make_handler(spec, mcp_tool.name)

                self._registry.register(Tool(
                    name=atom_name,
                    description=description,
                    category=f"mcp.{spec.name}",
                    safety_level=spec.safety_level,
                    parameters=params,
                    handler=handler,
                ))
                self._registered_tools[atom_name] = spec.name
                self._registered_count += 1

        logger.info(
            "AtomMCPClient: registered %d external tools across %d servers",
            self._registered_count, self.server_count,
        )
        return self._registered_count

    # ── Internals ─────────────────────────────────────────────

    @staticmethod
    def _atom_tool_name(server: str, tool: str) -> str:
        clean_server = "".join(
            ch if ch.isalnum() else "_" for ch in server
        ).strip("_") or "server"
        clean_tool = "".join(
            ch if ch.isalnum() else "_" for ch in tool
        ).strip("_") or "tool"
        return f"{AtomMCPClient.NAME_PREFIX}_{clean_server}__{clean_tool}"

    @staticmethod
    def _mcp_schema_to_atom_params(
        schema: dict[str, Any],
    ) -> list[Any]:
        from core.reasoning.tool_registry import ToolParameter

        if not isinstance(schema, dict):
            return []
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or ())
        out: list[ToolParameter] = []
        for pname, pdef in properties.items():
            if not isinstance(pdef, dict):
                continue
            t = str(pdef.get("type") or "string")
            out.append(ToolParameter(
                name=str(pname),
                type=t,
                description=str(pdef.get("description") or "")[:240],
                required=str(pname) in required,
                default=pdef.get("default"),
                enum=list(pdef["enum"]) if isinstance(
                    pdef.get("enum"), list,
                ) else None,
            ))
        return out

    def _make_handler(self, spec: MCPServerSpec, tool_name: str):
        """Build a sync->async-bridging handler for ATOM's dispatcher."""

        async def _async_handler(**kwargs: Any) -> str:
            return await self.call(spec.name, tool_name, kwargs)

        # The ToolRegistry handler signature is sync (it returns
        # str | None). We expose this as an attribute so callers that
        # know about the async path can use it directly via
        # ``handler.__atom_async__`` -- the action_executor's async
        # dispatch path picks this up to avoid an event-loop dance.
        def _sync_shim(**kwargs: Any) -> str:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is None or not loop.is_running():
                return asyncio.run(_async_handler(**kwargs))
            # We're inside a running loop -- block on a fresh one in a
            # background thread. This path is rarely hit because the
            # async dispatcher prefers ``__atom_async__``.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(asyncio.run, _async_handler(**kwargs))
                return fut.result()

        _sync_shim.__atom_async__ = _async_handler  # type: ignore[attr-defined]
        return _sync_shim

    async def call(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        spec = self._specs.get(server_name)
        if spec is None:
            return f"[MCP ERROR] Unknown server: {server_name}"
        try:
            return await asyncio.wait_for(
                self._call_inner(spec, tool_name, arguments or {}),
                timeout=spec.call_timeout_s,
            )
        except asyncio.TimeoutError:
            return (
                f"[MCP ERROR] {server_name}.{tool_name} timed out after "
                f"{spec.call_timeout_s:.0f}s"
            )
        except Exception as exc:
            return f"[MCP ERROR] {server_name}.{tool_name}: {exc}"

    # ── stdio session lifecycle ─────────────────────────────

    @staticmethod
    async def _open_session(spec: MCPServerSpec, stack: AsyncExitStack):
        """Open a fresh stdio MCP session inside an exit stack."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_server as _stdio_server  # noqa: F401
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=spec.command,
            args=list(spec.args),
            env=dict(spec.env) if spec.env else None,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(
            ClientSession(read, write),
        )
        await session.initialize()
        return session

    async def _list_tools(self, spec: MCPServerSpec) -> list[Any]:
        async with AsyncExitStack() as stack:
            session = await self._open_session(spec, stack)
            response = await session.list_tools()
            return list(response.tools or ())

    async def _call_inner(
        self,
        spec: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        async with AsyncExitStack() as stack:
            session = await self._open_session(spec, stack)
            response = await session.call_tool(
                tool_name, arguments=dict(arguments),
            )
            return _flatten_tool_response(response)


def _flatten_tool_response(response: Any) -> str:
    """Squash an MCP CallToolResult into a single string for ATOM."""
    if response is None:
        return "[MCP] (no response)"
    parts: list[str] = []
    is_err = bool(getattr(response, "isError", False))
    for item in getattr(response, "content", None) or ():
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
            continue
        # Image / resource contents: emit a compact placeholder.
        kind = type(item).__name__
        parts.append(f"[{kind}]")
    body = "\n".join(parts).strip() or "(empty)"
    return f"[MCP ERROR] {body}" if is_err else body


__all__ = [
    "AtomMCPClient",
    "MCPServerSpec",
    "load_mcp_servers_from_config",
]
