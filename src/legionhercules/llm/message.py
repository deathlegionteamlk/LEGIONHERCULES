"""Message types for LLM communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChatMessage:
    """A chat message for LLM communication."""
    
    role: str
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.name:
            result["name"] = self.name
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )
    
    @classmethod
    def user(cls, content: str) -> ChatMessage:
        """Create a user message."""
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        """Create an assistant message."""
        return cls(role="assistant", content=content)
    
    @classmethod
    def system(cls, content: str) -> ChatMessage:
        """Create a system message."""
        return cls(role="system", content=content)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> ChatMessage:
        """Create a tool response message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    @classmethod
    def from_message(cls, msg) -> ChatMessage:
        """Convert a core Message to ChatMessage."""
        from legionhercules.core.message import MessageRole
        
        role_map = {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.TOOL: "tool",
        }
        
        role = role_map.get(msg.role, "user")
        
        return cls(
            role=role,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_results[0].get("tool_call_id") if msg.tool_results else None,
        )
