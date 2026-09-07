"""A tiny, real MCP server used as a test fixture -- run as a real
subprocess by tests/test_mcp_client.py (via `sys.executable
tests/mcp_test_server.py`), not mocked. Exercises the actual MCP
client/server protocol end to end: a real stdio handshake, real tool
discovery, real invocation.
"""

from mcp.server.mcpserver import MCPServer

server = MCPServer("calc")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@server.tool()
def always_fails() -> str:
    """A tool that always raises, for testing tool-level error handling."""
    raise ValueError("this tool always fails")


if __name__ == "__main__":
    server.run(transport="stdio")
