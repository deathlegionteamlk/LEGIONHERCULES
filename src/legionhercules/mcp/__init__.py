"""MCP (Model Context Protocol) module for LEGIONHERCULES.

MCP enables integration with external tool servers via the Model Context Protocol.
"""

from legionhercules.mcp.client import MCPClient
from legionhercules.mcp.tool_adapter import MCPToolAdapter


class MCPConnectionError(Exception):
    """Raised when MCP connection fails."""
    pass


__all__ = ["MCPClient", "MCPToolAdapter", "MCPConnectionError"]
