"""OpenRouter API provider for LEGIONHERCULES - Access 500+ models from 60+ providers."""

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


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider - Universal access to 500+ models.
    
    OpenRouter provides unified access to models from OpenAI, Anthropic,
    Google, Meta, Mistral, and 60+ other providers with automatic failover.
    Get API key from: https://openrouter.ai/
    """
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.timeout = timeout
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "https://legionhercules.dev")
        self.site_name = site_name or os.environ.get("OPENROUTER_SITE_NAME", "LEGIONHERCULES")
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the OpenRouter provider."""
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")
        
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            }
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
            )
        
        logger.info(f"OpenRouter provider initialized (model: {self.model})")
        self._initialized = True
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages via OpenRouter."""
        if not self._client:
            await self.initialize()
        
        openrouter_messages = []
        for msg in messages:
            chat_msg = ChatMessage.from_message(msg)
            openrouter_messages.append(chat_msg.to_openai_dict())
        
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openrouter_messages,
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
                    "provider": data.get("provider", "unknown"),
                },
                usage=data.get("usage"),
            )
            
        except Exception as e:
            logger.error(f"Error in OpenRouter chat: {e}")
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
        """List available models via OpenRouter."""
        if not self._client:
            await self.initialize()
        
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return [
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-3-opus",
                "google/gemini-1.5-pro",
                "meta-llama/llama-3.1-70b-instruct",
                "mistralai/mistral-large",
                "deepseek/deepseek-chat",
            ]
    
    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
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
        """Convert tool to OpenRouter format (OpenAI compatible)."""
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
        """Parse tool calls from OpenRouter response."""
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
