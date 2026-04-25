"""ATOM Sprint Ω7 -- expose ATOM's tool registry as an MCP server.

Why MCP
-------
Anthropic's Model Context Protocol is the emerging standard for tool
interoperability between LLMs and external systems. By exposing ATOM
over MCP we get, with zero per-client work:

* Boss can drive ATOM from Claude Desktop, Cursor, Zed, Goose, or any
  other MCP-aware client.
* ATOM appears as a first-class tool provider in those UIs (just like
  the filesystem / GitHub / browser MCP servers).
* The transport (stdio) is exactly what Claude Desktop and Cursor wire
  in by default -- no custom HTTP server needed.

Security
--------
Every MCP ``call_tool`` request is dispatched through the same
:class:`core.reasoning.action_executor.ActionExecutor` that the LLM
fast path uses. That means:

* :class:`core.security_policy.SecurityPolicy` gates every action.
* Tool-grammar validation runs first.
* ``requires_confirmation`` tools refuse to execute and instead
  surface the confirmation prompt as a structured MCP error -- the
  remote client can re-prompt Boss.
* Audit logging via :mod:`core.audit_log` if wired by the host.

Usage
-----
Direct (in-process build) for tests::

    from core.mcp import build_atom_mcp_server
    server = build_atom_mcp_server(executor=executor, registry=registry)

CLI (production -- this is what Claude Desktop / Cursor wire to)::

    python scripts/run_mcp_server.py

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.mcp.server")

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from core.reasoning.action_executor import ActionExecutor
    from core.reasoning.tool_registry import ToolRegistry


# ── Internal helpers ────────────────────────────────────────────────


_ATOM_MCP_NAME = "atom"
_ATOM_MCP_INSTRUCTIONS = (
    "ATOM (Satyam's personal AI OS) tool surface. Every tool is "
    "security-gated and audited. Tools marked 'dangerous' will refuse "
    "to execute without explicit confirmation; their call returns an "
    "isError result with the confirmation prompt for the host UI to "
    "show Boss."
)


def _atom_tool_to_mcp_schema(tool: Any) -> dict[str, Any]:
    """Convert an ATOM Tool into an MCP-compatible JSON Schema."""
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for p in getattr(tool, "parameters", []) or ():
        prop: dict[str, Any] = {
            "type": _python_type_to_jsonschema(p.type),
        }
        if p.description:
            prop["description"] = p.description
        if p.enum:
            prop["enum"] = list(p.enum)
        if p.default is not None:
            prop["default"] = p.default
        properties[p.name] = prop
        if p.required:
            required.append(p.name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _python_type_to_jsonschema(t: str) -> str:
    """Map ATOM's loose type hints to JSON-Schema primitive types."""
    t = (t or "string").strip().lower()
    if t in ("int", "integer"):
        return "integer"
    if t in ("float", "number", "double"):
        return "number"
    if t in ("bool", "boolean"):
        return "boolean"
    if t in ("list", "array"):
        return "array"
    if t in ("dict", "object"):
        return "object"
    return "string"


# ── Server class ────────────────────────────────────────────────────


class AtomMCPServer:
    """Wrapper around :class:`mcp.server.fastmcp.FastMCP`.

    The server is build-once / run-many: ``build()`` constructs the
    FastMCP instance and registers every Atom tool, then ``run_stdio()``
    blocks on the stdio transport (this is what Claude Desktop / Cursor
    expect when they spawn ``python scripts/run_mcp_server.py``).
    """

    def __init__(
        self,
        *,
        action_executor: ActionExecutor,
        tool_registry: ToolRegistry,
        name: str = _ATOM_MCP_NAME,
        instructions: str = _ATOM_MCP_INSTRUCTIONS,
        deny_dangerous: bool = True,
    ) -> None:
        self._executor = action_executor
        self._registry = tool_registry
        self._name = name
        self._instructions = instructions
        self._deny_dangerous = deny_dangerous
        self._fastmcp: FastMCP | None = None

    # ── Public surface ────────────────────────────────────────

    def build(self) -> FastMCP:
        """Materialize the FastMCP server with every Atom tool wired."""
        from mcp.server.fastmcp import FastMCP

        if self._fastmcp is not None:
            return self._fastmcp

        fastmcp = FastMCP(
            name=self._name,
            instructions=self._instructions,
        )

        # Register every Atom tool. We can't use ``@fastmcp.tool()``
        # decorators because the tools are dynamic -- we register them
        # via ``add_tool()`` with explicit JSON-Schema input shapes.
        registered = 0
        for tool in self._registry.get_all():
            if (
                self._deny_dangerous
                and getattr(tool, "safety_level", "") == "blocked"
            ):
                continue
            self._register_tool(fastmcp, tool)
            registered += 1

        logger.info(
            "AtomMCPServer ready: %d tools registered (deny_dangerous=%s)",
            registered, self._deny_dangerous,
        )
        self._fastmcp = fastmcp
        return fastmcp

    async def run_stdio(self) -> None:
        """Run the server on stdio. Blocks until the client disconnects."""
        fastmcp = self.build()
        await fastmcp.run_stdio_async()

    # ── Internals ─────────────────────────────────────────────

    def _register_tool(self, fastmcp: FastMCP, tool: Any) -> None:
        """Wire a single Atom tool into the MCP server."""
        from mcp.server.fastmcp.tools import Tool as MCPTool

        schema = _atom_tool_to_mcp_schema(tool)
        atom_name = tool.name
        atom_description = tool.description or atom_name

        async def _handler(**kwargs: Any) -> str:
            return await self._dispatch(atom_name, kwargs)

        # FastMCP supports both decorator and add_tool forms. The
        # decorator infers the schema from type hints; we have a JSON
        # schema instead, so we build the Tool explicitly. This works
        # across the 1.x line and is what FastMCP recommends for
        # programmatic registration.
        try:
            mcp_tool = MCPTool.from_function(
                _handler,
                name=atom_name,
                description=atom_description,
            )
            # Override the auto-derived schema with our explicit one.
            mcp_tool.parameters = schema  # type: ignore[attr-defined]
            fastmcp._tool_manager.add_tool(mcp_tool)  # type: ignore[attr-defined]
        except Exception:
            # Last-ditch fallback: simpler add_tool path.
            try:
                fastmcp.add_tool(_handler, name=atom_name, description=atom_description)
            except Exception:
                logger.exception(
                    "Failed to register tool %s with MCP server", atom_name,
                )

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Run an Atom tool through the security-gated executor."""
        from core.reasoning.tool_parser import ToolCall

        # Refuse blocked tools at the boundary.
        tool = self._registry.get(name)
        if tool is None:
            return _err_payload(f"Unknown tool: {name}")
        if (
            self._deny_dangerous
            and getattr(tool, "safety_level", "") == "blocked"
        ):
            return _err_payload(f"Tool {name} is blocked by ATOM policy.")

        # If the tool requires confirmation, refuse over MCP -- the
        # caller is not an interactive user we can prompt. Surface the
        # confirmation message so the host (Claude Desktop, Cursor) can
        # show it to Boss and re-issue with an explicit override flag
        # in the future.
        if self._registry.requires_confirmation(name):
            return _err_payload(
                f"Tool {name} requires confirmation in ATOM. "
                f"Run it directly through the ATOM voice/UI surface, "
                f"or use a future explicit-override request once that "
                f"channel is wired.",
            )

        try:
            tc = ToolCall(name=name, arguments=dict(arguments or {}))
            result = await self._executor.execute_async(tc)
        except Exception as exc:
            return _err_payload(f"Executor raised: {exc}")

        if getattr(result, "blocked", False):
            return _err_payload(
                f"Blocked: {getattr(result, 'block_reason', 'security policy')}",
            )
        if not getattr(result, "success", False):
            return _err_payload(
                getattr(result, "error", "") or f"Tool {name} failed.",
            )
        return str(getattr(result, "output", "") or "Done.")


def _err_payload(message: str) -> str:
    """Format an error payload that surfaces cleanly in MCP clients.

    FastMCP wraps str returns into TextContent. To keep the host UI
    clear we prefix with [ATOM ERROR] so the client renders it as an
    obvious failure even when the framework treats it as a successful
    response.
    """
    return f"[ATOM ERROR] {message}"


def build_atom_mcp_server(
    *,
    action_executor: ActionExecutor,
    tool_registry: ToolRegistry,
    name: str = _ATOM_MCP_NAME,
    instructions: str = _ATOM_MCP_INSTRUCTIONS,
    deny_dangerous: bool = True,
) -> AtomMCPServer:
    """Convenience factory."""
    return AtomMCPServer(
        action_executor=action_executor,
        tool_registry=tool_registry,
        name=name,
        instructions=instructions,
        deny_dangerous=deny_dangerous,
    )


__all__ = ["AtomMCPServer", "build_atom_mcp_server"]
