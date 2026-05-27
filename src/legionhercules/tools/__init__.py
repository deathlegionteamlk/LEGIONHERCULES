"""Tools module for LEGIONHERCULES - File operations, bash, web search."""

from legionhercules.tools.base import Tool, ToolResult, ToolRegistry
from legionhercules.tools.file_tools import FileReadTool, FileWriteTool, FileEditTool
from legionhercules.tools.bash_tool import BashTool
from legionhercules.tools.web_search_tool import WebSearchTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "FileReadTool",
    "FileWriteTool",
    "FileEditTool",
    "BashTool",
    "WebSearchTool",
]
