"""Replicate API provider for LEGIONHERCULES."""

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


class ReplicateProvider(LLMProvider):
    """Replicate provider for running open-source models in the cloud."""

    DEFAULT_BASE_URL = "https://api.replicate.com/v1"

    # Popular models on Replicate
    POPULAR_MODELS = {
        "meta/llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
        "meta/llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
        "meta/llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta/llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/mixtral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    }

    def __init__(
        self,
        model: str = "meta/meta-llama-3-70b-instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,  # Longer timeout for cold starts
    ):
        super().__init__(model, base_url or self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("REPLICATE_API_TOKEN")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize the Replicate provider."""
        if not self.api_key:
            raise ValueError(
                "Replicate API token required. Set REPLICATE_API_TOKEN environment variable."
            )

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                    "Prefer": "wait",  # Wait for completion
                },
            )

        logger.info(f"Replicate provider initialized (model: {self.model})")
        self._initialized = True

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages to Replicate."""
        if not self._client:
            await self.initialize()

        # Format messages for Llama-style chat
        prompt = self._format_messages(messages)

        payload: dict[str, Any] = {
            "version": self.model,
            "input": {
                "prompt": prompt,
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["input"]["max_tokens"] = max_tokens

        try:
            # Create prediction
            response = await self._client.post("/predictions", json=payload)
            response.raise_for_status()

            data = response.json()
            
            # Check if completed immediately or needs polling
            if data.get("status") == "succeeded":
                output = data.get("output", "")
                if isinstance(output, list):
                    output = "".join(output)
                
                return LLMResponse(
                    content=output,
                    metadata={
                        "model": self.model,
                        "status": "succeeded",
                        "metrics": data.get("metrics", {}),
                    },
                )
            else:
                # Poll for completion
                prediction_id = data.get("id")
                result = await self._poll_prediction(prediction_id)
                return result

        except Exception as e:
            logger.error(f"Error in Replicate chat: {e}")
            return LLMResponse(content=f"Error: {str(e)}", metadata={"error": str(e)})

    async def _poll_prediction(self, prediction_id: str) -> LLMResponse:
        """Poll for prediction completion."""
        import asyncio
        
        max_attempts = 60  # 5 minutes with 5-second intervals
        for attempt in range(max_attempts):
            try:
                response = await self._client.get(f"/predictions/{prediction_id}")
                response.raise_for_status()
                data = response.json()
                
                status = data.get("status")
                
                if status == "succeeded":
                    output = data.get("output", "")
                    if isinstance(output, list):
                        output = "".join(output)
                    
                    return LLMResponse(
                        content=output,
                        metadata={
                            "model": self.model,
                            "status": "succeeded",
                            "metrics": data.get("metrics", {}),
                        },
                    )
                elif status in ["failed", "canceled"]:
                    error = data.get("error", "Unknown error")
                    return LLMResponse(
                        content=f"Prediction failed: {error}",
                        metadata={"error": error, "status": status},
                    )
                
                # Wait before next poll
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error polling prediction: {e}")
                return LLMResponse(
                    content=f"Error polling: {str(e)}",
                    metadata={"error": str(e)},
                )
        
        return LLMResponse(
            content="Prediction timed out",
            metadata={"error": "timeout"},
        )

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages for Llama-style chat."""
        formatted = []
        for msg in messages:
            chat_msg = ChatMessage.from_message(msg)
            role = chat_msg.role
            content = chat_msg.content
            
            if role == "system":
                formatted.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                formatted.append(f"{content} [/INST]")
            elif role == "assistant":
                formatted.append(f" {content} </s><s>[INST] ")
        
        return "".join(formatted)

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
        """List available Replicate models."""
        return list(self.POPULAR_MODELS.keys())

    async def health_check(self) -> bool:
        """Check if Replicate API is accessible."""
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

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
