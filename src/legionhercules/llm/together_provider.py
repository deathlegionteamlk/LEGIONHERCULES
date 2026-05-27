"""Together AI API provider for LEGIONHERCULES - 200+ open source models."""

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


class TogetherProvider(LLMProvider):
    """Together AI API provider - 200+ open source models.
    
    Together AI offers fast inference for open source models including
    Llama, Mixtral, Qwen, and many more with competitive pricing.
    Get API key from: https://api.together.xyz/
    """
    
    DEFAULT_BASE_URL = "https://api.together.xyz/v1"
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the Together AI provider."""
        if not self.api_key:
            raise ValueError("Together API key required. Set TOGETHER_API_KEY environment variable.")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        
        logger.info(f"Together AI provider initialized (model: {self.model})")
        self._initialized = True
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Together AI."""
        if not self._client:
            await self.initialize()
        
        together_messages = []
        for msg in messages:
            chat_msg = ChatMessage.from_message(msg)
            together_messages.append(chat_msg.to_openai_dict())
        
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": together_messages,
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
            logger.error(f"Error in Together AI chat: {e}")
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
        """List available Together AI models."""
        if not self._client:
            await self.initialize()
        
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            # Filter for chat completion models
            models = [m["id"] for m in data if m.get("type") == "chat"]
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return [
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "deepseek-ai/DeepSeek-V3",
                "deepseek-ai/DeepSeek-R1",
                "nvidia/Llama-3.1-Nemotron-70B-Instruct",
            ]
    
    async def health_check(self) -> bool:
        """Check if Together AI API is accessible."""
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
        """Convert tool to Together AI format (OpenAI compatible)."""
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
        """Parse tool calls from Together AI response."""
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
