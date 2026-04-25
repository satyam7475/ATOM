"""ATOM Sprint Ω7-Ω8 -- Model Context Protocol (MCP) integration.

This package exposes ATOM's :class:`core.reasoning.tool_registry.ToolRegistry`
as an MCP server (so external clients like Claude Desktop, Cursor, and
Zed can call ATOM's tools) AND lets ATOM consume external MCP servers
(filesystem, git, GitHub, browser, Slack, Postgres, ...).

Modules
-------
* :mod:`atom_mcp_server` -- run ATOM as an MCP server over stdio.
* :mod:`atom_mcp_client` -- connect ATOM to one or more external MCP
  servers and register their tools into the ATOM ToolRegistry.
"""

from __future__ import annotations

from .atom_mcp_client import (  # noqa: F401
    AtomMCPClient,
    MCPServerSpec,
    load_mcp_servers_from_config,
)
from .atom_mcp_server import (  # noqa: F401
    AtomMCPServer,
    build_atom_mcp_server,
)

__all__ = [
    "AtomMCPClient",
    "AtomMCPServer",
    "MCPServerSpec",
    "build_atom_mcp_server",
    "load_mcp_servers_from_config",
]
