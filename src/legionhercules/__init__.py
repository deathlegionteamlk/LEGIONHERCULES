"""LEGIONHERCULES - Autonomous CLI framework with parallel agent execution.

Developed by Death Legion Team Coders Demo X HEXA
"""

__version__ = "0.1.0"
__author__ = "Death Legion Team Coders Demo X HEXA"
__license__ = "MIT"

from legionhercules.core.orchestrator import AgentOrchestrator
from legionhercules.core.agent import Agent, AgentConfig
from legionhercules.core.task import Task, TaskResult

__all__ = [
    "AgentOrchestrator",
    "Agent",
    "AgentConfig",
    "Task",
    "TaskResult",
]
