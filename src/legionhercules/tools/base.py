"""Base tool classes for LEGIONHERCULES."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
    
    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Get JSON schema for tool parameters."""
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_schema(),
        }


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def list_tools(self) -> list[str]:
        """List all tool names."""
        return list(self._tools.keys())
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.to_dict() for tool in self._tools.values()]
    
    def create_default_registry() -> ToolRegistry:
        """Create a registry with default tools."""
        from legionhercules.tools.file_tools import FileReadTool, FileWriteTool, FileEditTool
        from legionhercules.tools.bash_tool import BashTool
        from legionhercules.tools.web_search_tool import WebSearchTool
        
        registry = ToolRegistry()
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(FileEditTool())
        registry.register(BashTool())
        registry.register(WebSearchTool())
        
        return registry
