"""Task management for LEGIONHERCULES."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, lower is higher priority
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[TaskResult] = None
    parent_task_id: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    
    # Execution
    func: Optional[Callable[..., Any]] = None
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    
    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self, result: TaskResult) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.result = result
        
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            result.duration_ms = duration * 1000
    
    def mark_cancelled(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (no pending dependencies)."""
        return self.status == TaskStatus.PENDING and len(self.dependencies) == 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None
