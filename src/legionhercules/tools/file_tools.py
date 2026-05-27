"""File operation tools for LEGIONHERCULES."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from legionhercules.tools.base import Tool, ToolResult
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class FileReadTool(Tool):
    """Tool for reading file contents."""
    
    def __init__(self):
        super().__init__(
            name="file_read",
            description="Read the contents of a file. Returns the file content as a string."
        )
    
    async def execute(self, path: str, offset: int = 0, limit: int = 0) -> ToolResult:
        """Read a file.
        
        Args:
            path: Path to the file to read
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read (0 = all)
        """
        try:
            file_path = Path(path).resolve()
            
            # Security check - prevent reading outside working directory
            cwd = Path.cwd()
            if not str(file_path).startswith(str(cwd)):
                return ToolResult(
                    success=False,
                    error=f"Cannot read file outside working directory: {path}"
                )
            
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}"
                )
            
            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Path is not a file: {path}"
                )
            
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Apply offset and limit
            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]
            
            result = '\n'.join(lines)
            
            logger.debug(f"Read file: {path} ({len(lines)} lines)")
            
            return ToolResult(
                success=True,
                output=result,
                metadata={
                    "path": str(file_path),
                    "size": len(content),
                    "lines": len(lines),
                    "offset": offset,
                    "limit": limit,
                }
            )
            
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return ToolResult(success=False, error=str(e))
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset to start from (0-indexed)",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum lines to read (0 = all)",
                    "default": 0
                }
            },
            "required": ["path"]
        }


class FileWriteTool(Tool):
    """Tool for writing file contents."""
    
    def __init__(self):
        super().__init__(
            name="file_write",
            description="Write content to a file. Creates the file if it doesn't exist."
        )
    
    async def execute(
        self,
        path: str,
        content: str,
        append: bool = False
    ) -> ToolResult:
        """Write content to a file.
        
        Args:
            path: Path to the file to write
            content: Content to write
            append: Whether to append to existing file
        """
        try:
            file_path = Path(path).resolve()
            
            # Security check
            cwd = Path.cwd()
            if not str(file_path).startswith(str(cwd)):
                return ToolResult(
                    success=False,
                    error=f"Cannot write file outside working directory: {path}"
                )
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            logger.debug(f"Wrote file: {path} ({len(content)} chars)")
            
            return ToolResult(
                success=True,
                output=f"File written successfully: {path}",
                metadata={
                    "path": str(file_path),
                    "size": len(content),
                    "append": append,
                }
            )
            
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return ToolResult(success=False, error=str(e))
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to existing file instead of overwriting",
                    "default": False
                }
            },
            "required": ["path", "content"]
        }


class FileEditTool(Tool):
    """Tool for editing file contents with search/replace."""
    
    def __init__(self):
        super().__init__(
            name="file_edit",
            description="Edit a file by searching for text and replacing it."
        )
    
    async def execute(
        self,
        path: str,
        old_string: str,
        new_string: str
    ) -> ToolResult:
        """Edit a file with search/replace.
        
        Args:
            path: Path to the file to edit
            old_string: Text to search for
            new_string: Text to replace with
        """
        try:
            file_path = Path(path).resolve()
            
            # Security check
            cwd = Path.cwd()
            if not str(file_path).startswith(str(cwd)):
                return ToolResult(
                    success=False,
                    error=f"Cannot edit file outside working directory: {path}"
                )
            
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}"
                )
            
            content = file_path.read_text(encoding='utf-8')
            
            # Count occurrences
            count = content.count(old_string)
            
            if count == 0:
                return ToolResult(
                    success=False,
                    error=f"String not found in file: {old_string[:50]}..."
                )
            
            if count > 1:
                return ToolResult(
                    success=False,
                    error=f"Multiple occurrences found ({count}). Please be more specific."
                )
            
            # Perform replacement
            new_content = content.replace(old_string, new_string, 1)
            
            # Write back
            file_path.write_text(new_content, encoding='utf-8')
            
            logger.debug(f"Edited file: {path}")
            
            return ToolResult(
                success=True,
                output=f"File edited successfully: {path}",
                metadata={
                    "path": str(file_path),
                    "replacements": 1,
                    "old_length": len(old_string),
                    "new_length": len(new_string),
                }
            )
            
        except Exception as e:
            logger.error(f"Error editing file {path}: {e}")
            return ToolResult(success=False, error=str(e))
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to search for (including whitespace)"
                },
                "new_string": {
                    "type": "string",
                    "description": "Text to replace with"
                }
            },
            "required": ["path", "old_string", "new_string"]
        }
