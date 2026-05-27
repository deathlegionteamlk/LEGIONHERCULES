"""Perplexity AI API provider for LEGIONHERCULES."""

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


class PerplexityProvider(LLMProvider):
    """Perplexity AI provider with real-time web search capabilities."""

    DEFAULT_BASE_URL = "https://api.perplexity.ai"

    # Perplexity models with search capabilities
    MODELS = {
        "sonar": "lightweight model for simple tasks",
        "sonar-pro": "advanced model for complex queries",
        "sonar-reasoning": "enhanced reasoning with Chain of Thought",
        "sonar-deep-research": "deep research mode",
    }

    def __init__(
        self,
        model: str = "sonar-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        search_recency_filter: Optional[str] = None,  # month, week, day, hour
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        self.timeout = timeout
        self.search_recency_filter = search_recency_filter
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize the Perplexity provider."""
        if not self.api_key:
            raise ValueError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY environment variable."
            )

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

        logger.info(f"Perplexity provider initialized (model: {self.model})")
        self._initialized = True

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Perplexity with web search."""
        if not self._client:
            await self.initialize()

        perplexity_messages = []
        for msg in messages:
            chat_msg = ChatMessage.from_message(msg)
            perplexity_messages.append(chat_msg.to_openai_dict())

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": perplexity_messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Add search recency filter if specified
        if self.search_recency_filter:
            payload["search_recency_filter"] = self.search_recency_filter

        # Perplexity supports limited tool use
        if tools and "sonar-pro" in self.model:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools[:3]]  # Limit tools

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

            # Extract citations if available
            citations = data.get("citations", [])

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                metadata={
                    "model": data.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "citations": citations,
                    "has_search_results": len(citations) > 0,
                },
                usage=data.get("usage"),
            )

        except Exception as e:
            logger.error(f"Error in Perplexity chat: {e}")
            return LLMResponse(content=f"Error: {str(e)}", metadata={"error": str(e)})

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from prompt with web search."""
        messages = [Message.user(prompt)]
        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    async def search(
        self,
        query: str,
        recency_filter: Optional[str] = None,
    ) -> LLMResponse:
        """Perform a web search using Perplexity."""
        # Temporarily set recency filter
        original_filter = self.search_recency_filter
        if recency_filter:
            self.search_recency_filter = recency_filter

        try:
            messages = [
                Message.system("You are a helpful search assistant. Provide accurate, up-to-date information with sources."),
                Message.user(query),
            ]
            return await self.chat(messages)
        finally:
            self.search_recency_filter = original_filter

    async def list_models(self) -> list[str]:
        """List available Perplexity models."""
        return list(self.MODELS.keys())

    async def health_check(self) -> bool:
        """Check if Perplexity API is accessible."""
        if not self._client:
            try:
                await self.initialize()
            except Exception:
                return False
        try:
            # Simple health check via models endpoint
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def _convert_tool_schema(self, tool: Any) -> dict[str, Any]:
        """Convert tool to OpenAI format."""
        tool_dict = tool.to_dict() if hasattr(tool, "to_dict") else tool
        return {
            "type": "function",
            "function": {
                "name": tool_dict.get("name", ""),
                "description": tool_dict.get("description", ""),
                "parameters": tool_dict.get("parameters", {}),
            },
        }

    def _parse_tool_calls(self, tool_calls_data: list[dict]) -> list[dict[str, Any]]:
        """Parse tool calls from response."""
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
