"""MCP (Model Context Protocol) client -- connecting to external MCP
servers for tools a real model can invoke via `ask_with_tools`.

This is the one place in Sense's interop story that genuinely needs new
runtime infrastructure rather than composing what already exists: an MCP
tool has no Sense body at all -- invoking one means a JSON-RPC call to an
*external process* over `mcp`'s asyncio-native client SDK (an optional
dependency, imported lazily, same "use the real library, don't hand-roll
a protocol client" call already made for `inference(...)`'s own providers).

Why a background thread running one persistent event loop, not something
simpler: an MCP session is opened once and reused across many tool calls
-- calling `asyncio.run(...)` per call would tear down and re-spawn the
server subprocess every single call. The natural async-native fix (open
the connection inside an `async with` block via a callback) isn't
available either: Sense has no anonymous function / closure syntax, so
there's no way to write `with_mcp_server(cmd, args, fn(tools): ...)`. A
dedicated background thread bridged into the synchronous interpreter via
`asyncio.run_coroutine_threadsafe(...).result(timeout=...)` is the option
actually available given those two constraints.

`_McpBridge._lifetime` enters *and exits* both of `mcp`'s async context
managers (`stdio_client`, `ClientSession`) in one single, long-lived
asyncio task that blocks on a shutdown event -- found necessary directly
by prototyping this: `anyio`'s cancel scopes (used internally by both
context managers) require `__aenter__`/`__aexit__` to happen in the same
task, which scheduling a separate shutdown coroutine via
`run_coroutine_threadsafe` violates ("Attempted to exit cancel scope in a
different task than it was entered in"). Individual `list_tools()`/
`call_tool()` calls don't have this restriction -- they run as their own
short-lived tasks on the same loop while `_lifetime`'s task sits
suspended on the shutdown event, which is safe (asyncio is cooperative
single-threaded within one loop).
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class McpToolInfo:
    """One tool discovered from `list_tools()`, before it becomes a
    Sense-facing `values.McpTool` -- host-side, not a Sense value itself."""

    name: str
    description: str
    input_schema: dict[str, Any]


class McpBridge:
    """Owns one MCP server subprocess connection for its entire lifetime:
    a background daemon thread running a persistent asyncio event loop,
    bridged into synchronous calls. `start()` blocks (with a timeout)
    until the handshake and initial `list_tools()` complete; `call_tool`
    blocks until that one call's result comes back; `close()` blocks
    until the subprocess and background thread have actually stopped."""

    def __init__(self, command: str, args: list[str]):
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "the 'mcp' package is required for connect_mcp() -- "
                "install it with 'pip install mcp' (or 'pip install sense-lang[mcp]')"
            ) from exc
        self._command = command
        self._args = args
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._session: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout: float = 30.0) -> list[McpToolInfo]:
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(f"timed out connecting to MCP server '{self._command}' after {timeout}s")
        if self._error is not None:
            raise self._error
        return self._list_tools_sync(timeout)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._shutdown_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._lifetime())
        except BaseException as exc:  # noqa: BLE001 - handed back to the calling thread via start()/_ready
            self._error = exc
            self._ready.set()

    async def _lifetime(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self._command, args=self._args)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._shutdown_event.wait()

    def _list_tools_sync(self, timeout: float) -> list[McpToolInfo]:
        fut = asyncio.run_coroutine_threadsafe(self._session.list_tools(), self._loop)
        result = fut.result(timeout=timeout)
        return [
            McpToolInfo(name=t.name, description=t.description or "", input_schema=t.input_schema or {})
            for t in result.tools
        ]

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 30.0) -> str:
        """Invokes one remote tool and returns its result as a String --
        joining any `text`-type content blocks, same convention
        `AnthropicProvider.complete()` already uses for its own content
        blocks. Raises on a transport/protocol failure; a tool-level
        error (`CallToolResult.is_error`) is returned as text, not
        raised, matching how the tool's own error message is meant to be
        surfaced (the caller -- `Interpreter._call_mcp_tool` -- decides
        whether that should propagate)."""
        fut = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments), self._loop)
        result = fut.result(timeout=timeout)
        text = "".join(block.text for block in result.content if getattr(block, "type", None) == "text")
        if result.is_error:
            raise RuntimeError(text or f"MCP tool '{name}' failed with no error message")
        return text

    def close(self, timeout: float = 15.0) -> None:
        if self._loop is None or self._shutdown_event is None:
            return
        self._loop.call_soon_threadsafe(self._shutdown_event.set)
        self._thread.join(timeout=timeout)
