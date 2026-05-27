"""MCP (Model Context Protocol) client for LEGIONHERCULES."""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any, Optional

import httpx

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class MCPClient:
    """Client for connecting to MCP servers and discovering tools.
    
    Supports both stdio and HTTP transports.
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        command: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ):
        """Initialize MCP client.
        
        Args:
            server_url: HTTP URL for MCP server (for HTTP transport)
            command: Command to spawn MCP server (for stdio transport)
            env: Environment variables for stdio transport
        """
        self.server_url = server_url
        self.command = command
        self.env = env or {}
        self._process: Optional[subprocess.Popen] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._tools: list[dict[str, Any]] = []
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize connection to MCP server."""
        try:
            if self.server_url:
                # HTTP transport
                self._http_client = httpx.AsyncClient(base_url=self.server_url)
                await self._discover_tools_http()
            elif self.command:
                # stdio transport
                await self._start_stdio_server()
                await self._discover_tools_stdio()
            else:
                raise ValueError("Either server_url or command must be provided")
            
            self._initialized = True
            logger.info(f"MCP client initialized with {len(self._tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            return False
    
    async def _start_stdio_server(self) -> None:
        """Start MCP server via stdio."""
        env = {**dict(os.environ), **self.env} if hasattr(os, 'environ') else self.env
        
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        
        # Wait for server to be ready
        await asyncio.sleep(0.5)
        logger.info(f"Started MCP server: {' '.join(self.command)}")
    
    async def _discover_tools_http(self) -> None:
        """Discover tools via HTTP."""
        if not self._http_client:
            return
        
        try:
            response = await self._http_client.get("/tools")
            response.raise_for_status()
            data = response.json()
            self._tools = data.get("tools", [])
        except Exception as e:
            logger.error(f"Failed to discover tools via HTTP: {e}")
            self._tools = []
    
    async def _discover_tools_stdio(self) -> None:
        """Discover tools via stdio."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            return
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "legionhercules", "version": "0.1.0"},
            },
        }
        
        self._process.stdin.write(json.dumps(init_request) + "\n")
        self._process.stdin.flush()
        
        # Read response
        response_line = self._process.stdout.readline()
        response = json.loads(response_line)
        
        if "error" in response:
            logger.error(f"MCP initialize error: {response['error']}")
            return
        
        # Send tools/list request
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        }
        
        self._process.stdin.write(json.dumps(tools_request) + "\n")
        self._process.stdin.flush()
        
        response_line = self._process.stdout.readline()
        response = json.loads(response_line)
        
        if "result" in response:
            self._tools = response["result"].get("tools", [])
        else:
            logger.error(f"Failed to list tools: {response.get('error')}")
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on the MCP server."""
        if not self._initialized:
            raise RuntimeError("MCP client not initialized")
        
        if self.server_url:
            return await self._call_tool_http(tool_name, arguments)
        else:
            return await self._call_tool_stdio(tool_name, arguments)
    
    async def _call_tool_http(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call tool via HTTP."""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        
        payload = {
            "name": tool_name,
            "arguments": arguments,
        }
        
        response = await self._http_client.post("/tools/call", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _call_tool_stdio(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call tool via stdio."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("stdio server not running")
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        
        response_line = self._process.stdout.readline()
        response = json.loads(response_line)
        
        if "result" in response:
            return response["result"]
        else:
            raise RuntimeError(f"Tool call failed: {response.get('error')}")
    
    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from MCP server."""
        return self._tools.copy()
    
    async def close(self) -> None:
        """Close MCP client connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
        
        self._initialized = False
        logger.info("MCP client closed")
