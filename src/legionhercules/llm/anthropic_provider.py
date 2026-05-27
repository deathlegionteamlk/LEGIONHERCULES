"""Anthropic Claude API provider for LEGIONHERCULES."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from legionhercules.core.message import Message, MessageRole
from legionhercules.llm.base import LLMProvider, LLMResponse
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider.
    
    Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku.
    Get API key from: https://console.anthropic.com/
    """
    
    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the Anthropic provider."""
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
        
        logger.info(f"Anthropic provider initialized (model: {self.model})")
        self._initialized = True
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Anthropic."""
        if not self._client:
            await self.initialize()
        
        # Separate system message from conversation
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            elif msg.role == MessageRole.USER:
                conversation_messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                conversation_messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == MessageRole.TOOL:
                # Anthropic uses tool_result content block
                conversation_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "content": msg.content}]
                })
        
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": conversation_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_message:
            payload["system"] = system_message
        
        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
        
        try:
            response = await self._client.post("/messages", json=payload)
            response.raise_for_status()
            
            data = response.json()
            content_blocks = data.get("content", [])
            
            text_content = ""
            tool_calls = None
            
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    })
            
            return LLMResponse(
                content=text_content,
                tool_calls=tool_calls,
                metadata={
                    "model": data.get("model"),
                    "stop_reason": data.get("stop_reason"),
                },
                usage={
                    "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                },
            )
            
        except Exception as e:
            logger.error(f"Error in Anthropic chat: {e}")
            return LLMResponse(content=f"Error: {str(e)}", metadata={"error": str(e)})
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from prompt."""
        messages = [Message.user(prompt)]
        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
    
    async def list_models(self) -> list[str]:
        """List available Anthropic models."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
    
    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        if not self._client:
            try:
                await self.initialize()
            except Exception:
                return False
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False
    
    def _convert_tool_schema(self, tool: Any) -> dict[str, Any]:
        """Convert tool to Anthropic format."""
        tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
        return {
            "name": tool_dict.get("name", ""),
            "description": tool_dict.get("description", ""),
            "input_schema": tool_dict.get("parameters", {}),
        }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
