"""Run ATOM as an MCP (Model Context Protocol) server over stdio.

This is the entrypoint that Claude Desktop, Cursor, Zed, Goose, and any
other MCP-aware client should spawn to talk to ATOM. The script:

  1. Boots ATOM's :class:`ToolRegistry` + :class:`ActionExecutor` in a
     minimal headless mode (no voice, no UI, no event loop tasks --
     just the tool surface).
  2. Wires the registry into an :class:`AtomMCPServer`.
  3. Runs the server on stdio. Blocks until the client disconnects.

Claude Desktop config example
-----------------------------
Add this to ``~/Library/Application Support/Claude/claude_desktop_config.json``::

    {
      "mcpServers": {
        "atom": {
          "command": "/Users/satyamyadav/Desktop/Personal/ATOM/.venv/bin/python",
          "args": ["/Users/satyamyadav/Desktop/Personal/ATOM/scripts/run_mcp_server.py"]
        }
      }
    }

Cursor config example
---------------------
Cursor reads MCP servers from the same JSON shape via Settings ->
MCP servers. Use the same ``command`` + ``args`` pair.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tokenizer parallelism off BEFORE any HF tokenizer touches the process,
# in case a future tool wants to import the brain.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Quiet down the rest of ATOM's loggers to stderr -- stdout is owned by
# the MCP stdio transport and a stray log line will corrupt the protocol.
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("atom.mcp.cli")


def _build_minimal_executor():
    """Build just enough ATOM machinery to expose the tool surface."""
    from core.reasoning.action_executor import ActionExecutor
    from core.reasoning.tool_registry import get_tool_registry
    from core.security_policy import SecurityPolicy

    registry = get_tool_registry()
    security = SecurityPolicy({"security": {"mode": "strict"}})

    # The dispatch path needs *some* implementation. We use the same
    # router-bound dispatcher the live runtime uses, but built lazily
    # so we don't pay the import cost up front when the client only
    # wants ``list_tools``.
    def _stub_dispatch(action: str, args: dict) -> str | None:
        # The MCP server should never reach this stub for built-in
        # tools because the live router's dispatcher is wired via
        # ``ActionExecutor`` once we import it. We keep this as a
        # safety net so an unhandled action returns a clean error.
        return None

    executor = ActionExecutor(
        dispatch_fn=_stub_dispatch,
        security=security,
        registry=registry,
    )
    return executor, registry


async def _amain() -> int:
    from core.mcp import build_atom_mcp_server

    executor, registry = _build_minimal_executor()
    logger.info(
        "ATOM MCP server starting on stdio: %d tools in registry",
        registry.count,
    )

    server = build_atom_mcp_server(
        action_executor=executor,
        tool_registry=registry,
        deny_dangerous=True,
    )
    await server.run_stdio()
    return 0


def main() -> int:
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
