"""Google Gemini API provider for LEGIONHERCULES."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from legionhercules.core.message import Message, MessageRole
from legionhercules.llm.base import LLMProvider, LLMResponse
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class GoogleProvider(LLMProvider):
    """Google Gemini API provider.
    
    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro.
    Get API key from: https://ai.google.dev/
    """
    
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the Google provider."""
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        
        logger.info(f"Google provider initialized (model: {self.model})")
        self._initialized = True
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Gemini."""
        if not self._client:
            await self.initialize()
        
        # Convert to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == MessageRole.USER:
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == MessageRole.ASSISTANT:
                contents.append({"role": "model", "parts": [{"text": msg.content}]})
        
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        
        if tools:
            payload["tools"] = [{"functionDeclarations": [self._convert_tool_schema(t) for t in tools]}]
        
        try:
            url = f"/models/{self.model}:generateContent?key={self.api_key}"
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            candidates = data.get("candidates", [])
            
            if not candidates:
                return LLMResponse(content="", metadata={"error": "No response generated"})
            
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            
            text_content = ""
            tool_calls = None
            
            for part in content_parts:
                if "text" in part:
                    text_content += part["text"]
                elif "functionCall" in part:
                    if tool_calls is None:
                        tool_calls = []
                    fc = part["functionCall"]
                    tool_calls.append({
                        "name": fc.get("name", ""),
                        "arguments": fc.get("args", {}),
                    })
            
            usage = data.get("usageMetadata", {})
            
            return LLMResponse(
                content=text_content,
                tool_calls=tool_calls,
                metadata={
                    "finish_reason": candidate.get("finishReason"),
                },
                usage={
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0),
                },
            )
            
        except Exception as e:
            logger.error(f"Error in Google chat: {e}")
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
        """List available Gemini models."""
        return [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.0-pro",
        ]
    
    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        if not self._client:
            try:
                await self.initialize()
            except Exception:
                return False
        try:
            url = f"/models/{self.model}?key={self.api_key}"
            response = await self._client.get(url)
            return response.status_code == 200
        except Exception:
            return False
    
    def _convert_tool_schema(self, tool: Any) -> dict[str, Any]:
        """Convert tool to Gemini format."""
        tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
        return {
            "name": tool_dict.get("name", ""),
            "description": tool_dict.get("description", ""),
            "parameters": tool_dict.get("parameters", {}),
        }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
