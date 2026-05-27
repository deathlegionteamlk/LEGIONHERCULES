"""Ollama LLM provider for LEGIONHERCULES - completely free local inference."""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from legionhercules.core.message import Message, MessageRole
from legionhercules.llm.base import LLMProvider, LLMResponse
from legionhercules.llm.message import ChatMessage
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference.
    
    Ollama runs LLMs locally on your machine - completely free!
    Download from: https://ollama.com
    
    Example models:
    - llama3.2: Fast, efficient general purpose
    - llama3.1:8b: Good balance of speed and quality
    - codellama: Code-focused
    - mistral: Strong performance
    - phi3: Microsoft's efficient model
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the Ollama provider."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        
        # Check if Ollama is running
        if not await self.health_check():
            logger.warning(
                f"Ollama not available at {self.base_url}. "
                "Please ensure Ollama is installed and running."
            )
        else:
            logger.info(f"Ollama provider initialized (model: {self.model})")
        
        self._initialized = True
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Ollama.
        
        Note: Tool calling support varies by model in Ollama.
        Some models support native tool calling, others don't.
        """
        if not self._client:
            await self.initialize()
        
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            chat_msg = ChatMessage.from_message(msg)
            ollama_messages.append(chat_msg.to_dict())
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Add tools if provided (for models that support it)
        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
        
        try:
            logger.debug(f"Sending chat request to Ollama (model: {self.model})")
            
            response = await self._client.post(
                "/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract content
            content = data.get("message", {}).get("content", "")
            
            # Check for tool calls in response
            tool_calls = None
            if "tool_calls" in data.get("message", {}):
                tool_calls = self._parse_tool_calls(data["message"]["tool_calls"])
            
            # Try to extract tool calls from content if not in structured format
            if not tool_calls and tools:
                tool_calls = self._extract_tool_calls_from_content(content)
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                metadata={
                    "model": self.model,
                    "done": data.get("done", False),
                },
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                }
            )
            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return LLMResponse(
                content="Error: Cannot connect to Ollama. Please ensure Ollama is running.",
                metadata={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from prompt using Ollama."""
        if not self._client:
            await self.initialize()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            logger.debug(f"Sending generate request to Ollama (model: {self.model})")
            
            response = await self._client.post(
                "/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data.get("response", ""),
                metadata={
                    "model": self.model,
                    "done": data.get("done", False),
                },
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Ollama generate: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def list_models(self) -> list[str]:
        """List available models from Ollama."""
        if not self._client:
            await self.initialize()
        
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            
            logger.debug(f"Available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        if not self._client:
            try:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=5.0,
                )
            except Exception:
                return False
        
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def _convert_tool_schema(self, tool: Any) -> dict[str, Any]:
        """Convert tool to Ollama tool format."""
        if hasattr(tool, 'to_dict'):
            tool_dict = tool.to_dict()
        else:
            tool_dict = tool
        
        return {
            "type": "function",
            "function": {
                "name": tool_dict.get("name", ""),
                "description": tool_dict.get("description", ""),
                "parameters": tool_dict.get("parameters", {}),
            }
        }
    
    def _parse_tool_calls(self, tool_calls_data: list[dict]) -> list[dict[str, Any]]:
        """Parse tool calls from Ollama response."""
        parsed = []
        for tc in tool_calls_data:
            if "function" in tc:
                func = tc["function"]
                parsed.append({
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", {}),
                })
        return parsed
    
    def _extract_tool_calls_from_content(self, content: str) -> Optional[list[dict[str, Any]]]:
        """Try to extract tool calls from content if model doesn't support native tool calling.
        
        Looks for patterns like:
        ```tool
        {"name": "tool_name", "arguments": {...}}
        ```
        """
        import re
        
        tool_calls = []
        
        # Pattern for JSON tool calls in markdown code blocks
        pattern = r'```(?:tool|json)?\s*({[^}]+})\s*```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if "name" in data:
                    tool_calls.append({
                        "name": data["name"],
                        "arguments": data.get("arguments", data.get("params", {})),
                    })
            except json.JSONDecodeError:
                continue
        
        return tool_calls if tool_calls else None
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.
        
        This downloads the model if not already available.
        """
        if not self._client:
            await self.initialize()
        
        try:
            logger.info(f"Pulling model: {model_name}")
            
            response = await self._client.post(
                "/api/pull",
                json={"name": model_name, "stream": False}
            )
            response.raise_for_status()
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
