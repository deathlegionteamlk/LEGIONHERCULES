"""Bash command execution tool for LEGIONHERCULES."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any

from legionhercules.tools.base import Tool, ToolResult
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class BashTool(Tool):
    """Tool for executing bash commands."""
    
    # Dangerous commands that should be blocked
    DANGEROUS_COMMANDS = [
        "rm -rf /",
        "rm -rf /*",
        "> /dev/sda",
        "dd if=/dev/zero",
        "mkfs.",
        ":(){ :|:& };:",  # Fork bomb
        "chmod -R 777 /",
    ]
    
    def __init__(self, timeout: int = 60, working_dir: str = "."):
        super().__init__(
            name="bash",
            description="Execute a bash command in the shell. Returns stdout, stderr, and exit code."
        )
        self.timeout = timeout
        self.working_dir = Path(working_dir).resolve()
    
    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None
    ) -> ToolResult:
        """Execute a bash command.
        
        Args:
            command: The command to execute
            cwd: Working directory for the command
            env: Environment variables to set
            timeout: Timeout in seconds (overrides default)
        """
        try:
            # Security check
            if self._is_dangerous(command):
                return ToolResult(
                    success=False,
                    error="Command blocked for security reasons"
                )
            
            # Determine working directory
            work_dir = Path(cwd).resolve() if cwd else self.working_dir
            
            # Security check for working directory
            cwd_path = Path.cwd()
            if not str(work_dir).startswith(str(cwd_path)):
                return ToolResult(
                    success=False,
                    error=f"Cannot execute command outside working directory: {work_dir}"
                )
            
            cmd_timeout = timeout or self.timeout
            
            logger.debug(f"Executing command: {command[:100]}...")
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=cmd_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {cmd_timeout} seconds"
                )
            
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            success = process.returncode == 0
            
            # Truncate output if too long
            max_output = 10000
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n... (truncated)"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n... (truncated)"
            
            output = {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "exit_code": process.returncode,
            }
            
            logger.debug(f"Command completed: exit_code={process.returncode}")
            
            return ToolResult(
                success=success,
                output=output if success else stderr_str or stdout_str,
                error=None if success else stderr_str,
                metadata={
                    "command": command,
                    "exit_code": process.returncode,
                    "cwd": str(work_dir),
                    "stdout_length": len(stdout_str),
                    "stderr_length": len(stderr_str),
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _is_dangerous(self, command: str) -> bool:
        """Check if command contains dangerous patterns."""
        cmd_lower = command.lower().strip()
        
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous.lower() in cmd_lower:
                return True
        
        return False
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command",
                    "default": None
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 60
                }
            },
            "required": ["command"]
        }
