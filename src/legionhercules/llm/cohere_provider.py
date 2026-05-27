"""Cohere API provider for LEGIONHERCULES."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from legionhercules.core.message import Message, MessageRole
from legionhercules.llm.base import LLMProvider, LLMResponse
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class CohereProvider(LLMProvider):
    """Cohere API provider.
    
    Supports Command R+, Command R, Command, Command Light.
    Get API key from: https://cohere.com/
    """
    
    DEFAULT_BASE_URL = "https://api.cohere.com/v1"
    
    def __init__(
        self,
        model: str = "command-r-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the Cohere provider."""
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY environment variable.")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        
        logger.info(f"Cohere provider initialized (model: {self.model})")
        self._initialized = True
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Cohere."""
        if not self._client:
            await self.initialize()
        
        # Convert messages to Cohere format
        cohere_messages = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                cohere_messages.append({"role": "USER", "message": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                cohere_messages.append({"role": "CHATBOT", "message": msg.content})
        
        payload: dict[str, Any] = {
            "model": self.model,
            "message": cohere_messages[-1]["message"] if cohere_messages else "",
            "chat_history": cohere_messages[:-1] if len(cohere_messages) > 1 else [],
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
        
        try:
            response = await self._client.post("/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            text_content = data.get("text", "")
            tool_calls = None
            
            # Check for tool calls in response
            if "tool_calls" in data and data["tool_calls"]:
                tool_calls = []
                for tc in data["tool_calls"]:
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "arguments": tc.get("parameters", {}),
                    })
            
            return LLMResponse(
                content=text_content,
                tool_calls=tool_calls,
                metadata={
                    "finish_reason": data.get("finish_reason"),
                },
                usage=data.get("usage", {}),
            )
            
        except Exception as e:
            logger.error(f"Error in Cohere chat: {e}")
            return LLMResponse(content=f"Error: {str(e)}", metadata={"error": str(e)})
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from prompt."""
        if not self._client:
            await self.initialize()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = await self._client.post("/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                content=data.get("generations", [{}])[0].get("text", ""),
                metadata={},
            )
        except Exception as e:
            logger.error(f"Error in Cohere generate: {e}")
            return LLMResponse(content=f"Error: {str(e)}", metadata={"error": str(e)})
    
    async def list_models(self) -> list[str]:
        """List available Cohere models."""
        return [
            "command-r-plus",
            "command-r",
            "command",
            "command-light",
            "command-nightly",
            "command-light-nightly",
        ]
    
    async def health_check(self) -> bool:
        """Check if Cohere API is accessible."""
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
        """Convert tool to Cohere format."""
        tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
        return {
            "name": tool_dict.get("name", ""),
            "description": tool_dict.get("description", ""),
            "parameter_definitions": tool_dict.get("parameters", {}).get("properties", {}),
        }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
