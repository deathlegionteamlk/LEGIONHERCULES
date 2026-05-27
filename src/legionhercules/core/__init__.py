"""Core module for LEGIONHERCULES agent orchestration."""

from legionhercules.core.orchestrator import AgentOrchestrator
from legionhercules.core.agent import Agent, AgentConfig
from legionhercules.core.task import Task, TaskResult, TaskStatus
from legionhercules.core.message import Message, MessageRole
from legionhercules.core.agent_loop import AgentLoop, AgentLoopStatus, AgentLoopResult, AdaptiveAgentLoop, ConvergenceDetector

__all__ = [
    "AgentOrchestrator",
    "Agent",
    "AgentConfig",
    "Task",
    "TaskResult",
    "TaskStatus",
    "Message",
    "MessageRole",
    "AgentLoop",
    "AgentLoopStatus",
    "AgentLoopResult",
    "AdaptiveAgentLoop",
    "ConvergenceDetector",
]
