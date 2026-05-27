"""Adapter to convert MCP tools to LEGIONHERCULES tools."""

from __future__ import annotations

from typing import Any

from legionhercules.mcp.client import MCPClient
from legionhercules.tools.base import Tool, ToolResult
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class MCPToolAdapter:
    """Adapts MCP tools to LEGIONHERCULES tool format."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
    
    def adapt_tools(self) -> list[Tool]:
        """Convert MCP tools to LEGIONHERCULES Tool objects."""
        mcp_tools = self.mcp_client.list_tools()
        adapted_tools = []
        
        for mcp_tool in mcp_tools:
            tool = MCPToolWrapper(self.mcp_client, mcp_tool)
            adapted_tools.append(tool)
        
        return adapted_tools
    
    def register_with_registry(self, registry: Any) -> None:
        """Register adapted tools with a tool registry."""
        tools = self.adapt_tools()
        for tool in tools:
            registry.register_tool(tool)
            logger.debug(f"Registered MCP tool: {tool.name}")


class MCPToolWrapper(Tool):
    """Wrapper that adapts an MCP tool to LEGIONHERCULES Tool interface."""
    
    def __init__(self, mcp_client: MCPClient, mcp_tool_def: dict[str, Any]):
        self.mcp_client = mcp_client
        self.mcp_tool_def = mcp_tool_def
        
        # Extract tool definition
        self._name = mcp_tool_def.get("name", "unknown")
        self._description = mcp_tool_def.get("description", "")
        self._parameters = mcp_tool_def.get("inputSchema", {})
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool."""
        try:
            result = await self.mcp_client.call_tool(self._name, kwargs)
            
            # Extract content from result
            content = result.get("content", [])
            if content:
                if isinstance(content, list) and len(content) > 0:
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    output = "\n".join(text_parts)
                else:
                    output = str(content)
            else:
                output = "Tool executed successfully"
            
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self._name,
            "description": self._description,
            "parameters": self._parameters,
        }
