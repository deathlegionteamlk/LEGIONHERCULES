"""Message types for agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional


class MessageRole(Enum):
    """Message sender role."""
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()


@dataclass
class Message:
    """Represents a message in the conversation."""
    
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_results: Optional[list[dict[str, Any]]] = None
    
    @classmethod
    def system(cls, content: str, **metadata: Any) -> Message:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, metadata=metadata)
    
    @classmethod
    def user(cls, content: str, **metadata: Any) -> Message:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, metadata=metadata)
    
    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[list] = None, **metadata: Any) -> Message:
        """Create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            metadata=metadata
        )
    
    @classmethod
    def tool(cls, content: str, tool_results: Optional[list] = None, **metadata: Any) -> Message:
        """Create a tool message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_results=tool_results,
            metadata=metadata
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.name.lower(),
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
