"""Mistral AI API provider for LEGIONHERCULES."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from legionhercules.core.message import Message
from legionhercules.llm.base import LLMProvider, LLMResponse
from legionhercules.llm.message import ChatMessage
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class MistralProvider(LLMProvider):
    """Mistral AI API provider.
    
    Supports Mistral Large, Mistral Medium, Mistral Small, Codestral.
    Get API key from: https://console.mistral.ai/
    """
    
    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
    
    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the Mistral provider."""
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        
        logger.info(f"Mistral provider initialized (model: {self.model})")
        self._initialized = True
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Mistral."""
        if not self._client:
            await self.initialize()
        
        mistral_messages = []
        for msg in messages:
            chat_msg = ChatMessage.from_message(msg)
            mistral_messages.append(chat_msg.to_openai_dict())
        
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": mistral_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
            payload["tool_choice"] = "auto"
        
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            
            content = message.get("content", "")
            tool_calls = None
            
            if "tool_calls" in message:
                tool_calls = self._parse_tool_calls(message["tool_calls"])
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                metadata={
                    "model": data.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                },
                usage=data.get("usage"),
            )
            
        except Exception as e:
            logger.error(f"Error in Mistral chat: {e}")
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
        """List available Mistral models."""
        if not self._client:
            await self.initialize()
        
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
                "codestral-latest",
                "pixtral-large-latest",
            ]
    
    async def health_check(self) -> bool:
        """Check if Mistral API is accessible."""
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
        """Convert tool to Mistral format (OpenAI compatible)."""
        tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
        return {
            "type": "function",
            "function": {
                "name": tool_dict.get("name", ""),
                "description": tool_dict.get("description", ""),
                "parameters": tool_dict.get("parameters", {}),
            }
        }
    
    def _parse_tool_calls(self, tool_calls_data: list[dict]) -> list[dict[str, Any]]:
        """Parse tool calls from Mistral response."""
        parsed = []
        for tc in tool_calls_data:
            if tc.get("type") == "function":
                func = tc.get("function", {})
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                parsed.append({"name": func.get("name", ""), "arguments": arguments})
        return parsed
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
