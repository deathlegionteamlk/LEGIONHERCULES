"""Base LLM provider classes for LEGIONHERCULES."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from legionhercules.core.message import Message


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: Optional[dict[str, int]] = None


class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, model: str, base_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        self._initialized = True
    
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat messages and get response."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass
