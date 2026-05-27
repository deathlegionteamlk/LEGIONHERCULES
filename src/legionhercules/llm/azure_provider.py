"""Azure OpenAI API provider for LEGIONHERCULES."""

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


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider for enterprise deployments."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        timeout: float = 120.0,
    ):
        super().__init__(model, endpoint)
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize the Azure OpenAI provider."""
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY environment variable."
            )
        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT environment variable."
            )

        # Ensure endpoint doesn't end with trailing slash
        base_url = self.endpoint.rstrip("/")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self.timeout,
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json",
                },
            )

        logger.info(f"Azure OpenAI provider initialized (model: {self.model})")
        self._initialized = True

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Azure OpenAI."""
        if not self._client:
            await self.initialize()

        openai_messages = []
        for msg in messages:
            chat_msg = ChatMessage.from_message(msg)
            openai_messages.append(chat_msg.to_openai_dict())

        payload: dict[str, Any] = {
            "messages": openai_messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
            payload["tool_choice"] = "auto"

        try:
            url = f"/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
            response = await self._client.post(url, json=payload)
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
            logger.error(f"Error in Azure OpenAI chat: {e}")
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
        """List available Azure OpenAI models."""
        # Azure doesn't have a simple list models endpoint
        # Return common Azure OpenAI models
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-35-turbo",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

    async def health_check(self) -> bool:
        """Check if Azure OpenAI API is accessible."""
        if not self._client:
            try:
                await self.initialize()
            except Exception:
                return False
        try:
            # Try a simple models request
            response = await self._client.get(
                f"/openai/deployments?api-version={self.api_version}"
            )
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
