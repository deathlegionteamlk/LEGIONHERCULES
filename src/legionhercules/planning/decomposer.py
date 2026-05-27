"""Task decomposition for multi-step planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Set
from datetime import datetime
from enum import Enum
import uuid

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Task:
    """A single task in a plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[Task] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_effort: int = 1  # In arbitrary units (1-10)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "estimated_effort": self.estimated_effort,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Plan:
    """A complete plan with multiple tasks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    tasks: List[Task] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    @property
    def progress(self) -> float:
        """Calculate completion percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "is_complete": self.is_complete,
            "progress": self.progress,
            "metadata": self.metadata,
        }


class TaskDecomposer:
    """Decomposes complex goals into actionable tasks."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    async def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_depth: int = 3,
    ) -> Plan:
        """Decompose a goal into a plan with tasks."""
        logger.info(f"Decomposing goal: {goal[:100]}...")

        if self.llm_provider:
            return await self._decompose_with_llm(goal, context, max_depth)
        else:
            return self._decompose_heuristic(goal, context, max_depth)

    async def _decompose_with_llm(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        max_depth: int,
    ) -> Plan:
        """Use LLM to decompose goal into tasks."""
        prompt = f"""Decompose the following goal into a structured plan with specific tasks.

Goal: {goal}

Provide your response in this exact format:

TASK 1: [Task description]
- Priority: [CRITICAL/HIGH/MEDIUM/LOW]
- Dependencies: [none or list task numbers]
- Effort: [1-10]

TASK 2: [Task description]
...

Make tasks specific, actionable, and ordered logically."""

        try:
            response = await self.llm_provider.generate(prompt)
            tasks = self._parse_llm_response(response.content)
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            tasks = self._decompose_heuristic(goal, context, max_depth).tasks

        return Plan(
            goal=goal,
            tasks=tasks,
            context=context or {},
        )

    def _decompose_heuristic(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        max_depth: int,
    ) -> Plan:
        """Use heuristics to decompose goal."""
        tasks = []
        goal_lower = goal.lower()

        # Common patterns for task decomposition
        if "create" in goal_lower or "build" in goal_lower or "implement" in goal_lower:
            tasks = [
                Task(description="Analyze requirements and plan approach", priority=TaskPriority.HIGH, estimated_effort=2),
                Task(description="Design structure and components", priority=TaskPriority.HIGH, estimated_effort=3, dependencies=["0"]),
                Task(description="Implement core functionality", priority=TaskPriority.CRITICAL, estimated_effort=5, dependencies=["1"]),
                Task(description="Add error handling and edge cases", priority=TaskPriority.MEDIUM, estimated_effort=3, dependencies=["2"]),
                Task(description="Test and validate implementation", priority=TaskPriority.HIGH, estimated_effort=3, dependencies=["2", "3"]),
            ]
        elif "fix" in goal_lower or "debug" in goal_lower or "resolve" in goal_lower:
            tasks = [
                Task(description="Identify and reproduce the issue", priority=TaskPriority.CRITICAL, estimated_effort=2),
                Task(description="Analyze root cause", priority=TaskPriority.CRITICAL, estimated_effort=3, dependencies=["0"]),
                Task(description="Implement fix", priority=TaskPriority.CRITICAL, estimated_effort=3, dependencies=["1"]),
                Task(description="Test the fix", priority=TaskPriority.HIGH, estimated_effort=2, dependencies=["2"]),
                Task(description="Verify no regressions", priority=TaskPriority.MEDIUM, estimated_effort=2, dependencies=["3"]),
            ]
        elif "refactor" in goal_lower or "improve" in goal_lower or "optimize" in goal_lower:
            tasks = [
                Task(description="Analyze current implementation", priority=TaskPriority.HIGH, estimated_effort=2),
                Task(description="Identify improvement opportunities", priority=TaskPriority.HIGH, estimated_effort=2, dependencies=["0"]),
                Task(description="Plan refactoring approach", priority=TaskPriority.HIGH, estimated_effort=2, dependencies=["1"]),
                Task(description="Execute refactoring", priority=TaskPriority.CRITICAL, estimated_effort=5, dependencies=["2"]),
                Task(description="Verify functionality preserved", priority=TaskPriority.CRITICAL, estimated_effort=3, dependencies=["3"]),
            ]
        elif "research" in goal_lower or "investigate" in goal_lower or "learn" in goal_lower:
            tasks = [
                Task(description="Define research scope and questions", priority=TaskPriority.HIGH, estimated_effort=1),
                Task(description="Gather information from multiple sources", priority=TaskPriority.CRITICAL, estimated_effort=4, dependencies=["0"]),
                Task(description="Analyze and synthesize findings", priority=TaskPriority.CRITICAL, estimated_effort=3, dependencies=["1"]),
                Task(description="Document conclusions and recommendations", priority=TaskPriority.HIGH, estimated_effort=2, dependencies=["2"]),
            ]
        else:
            # Generic decomposition
            tasks = [
                Task(description="Understand requirements and context", priority=TaskPriority.HIGH, estimated_effort=2),
                Task(description="Plan approach and steps", priority=TaskPriority.HIGH, estimated_effort=2, dependencies=["0"]),
                Task(description="Execute main work", priority=TaskPriority.CRITICAL, estimated_effort=5, dependencies=["1"]),
                Task(description="Review and finalize", priority=TaskPriority.MEDIUM, estimated_effort=2, dependencies=["2"]),
            ]

        # Assign IDs and update dependencies
        for i, task in enumerate(tasks):
            task.id = str(i)
            task.dependencies = [str(int(d)) for d in task.dependencies if int(d) < len(tasks)]

        return Plan(goal=goal, tasks=tasks, context=context or {})

    def _parse_llm_response(self, response: str) -> List[Task]:
        """Parse LLM response into tasks."""
        tasks = []
        current_task = None

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("TASK"):
                if current_task:
                    tasks.append(current_task)
                current_task = Task(
                    description=line.split(":", 1)[1].strip() if ":" in line else line,
                )
            elif current_task and line.startswith("-"):
                if "Priority:" in line:
                    priority_str = line.split(":", 1)[1].strip().upper()
                    current_task.priority = TaskPriority[priority_str] if priority_str in TaskPriority.__members__ else TaskPriority.MEDIUM
                elif "Dependencies:" in line:
                    deps = line.split(":", 1)[1].strip()
                    if deps.lower() != "none":
                        current_task.dependencies = [d.strip() for d in deps.split(",")]
                elif "Effort:" in line:
                    try:
                        current_task.estimated_effort = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

        if current_task:
            tasks.append(current_task)

        # Assign IDs
        for i, task in enumerate(tasks):
            task.id = str(i)

        return tasks

    def get_ready_tasks(self, plan: Plan) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met)."""
        completed_ids = {t.id for t in plan.tasks if t.status == TaskStatus.COMPLETED}
        ready = []

        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in task.dependencies):
                    task.status = TaskStatus.READY
                    ready.append(task)

        return ready

    def get_execution_order(self, plan: Plan) -> List[Task]:
        """Get tasks in dependency-respecting execution order."""
        # Topological sort
        in_degree = {t.id: 0 for t in plan.tasks}
        dependents = {t.id: [] for t in plan.tasks}

        for task in plan.tasks:
            for dep in task.dependencies:
                if dep in dependents:
                    dependents[dep].append(task.id)
                in_degree[task.id] += 1

        # Start with tasks having no dependencies
        queue = [t for t in plan.tasks if in_degree[t.id] == 0]
        queue.sort(key=lambda t: t.priority.value)
        result = []

        while queue:
            task = queue.pop(0)
            result.append(task)

            for dependent_id in dependents[task.id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    dependent_task = next((t for t in plan.tasks if t.id == dependent_id), None)
                    if dependent_task:
                        queue.append(dependent_task)
                        queue.sort(key=lambda t: t.priority.value)

        return result

    def estimate_total_effort(self, plan: Plan) -> int:
        """Estimate total effort for plan completion."""
        return sum(t.estimated_effort for t in plan.tasks)

    def get_critical_path(self, plan: Plan) -> List[Task]:
        """Identify critical path (longest dependency chain)."""
        # Build adjacency list
        graph = {t.id: [] for t in plan.tasks}
        task_map = {t.id: t for t in plan.tasks}

        for task in plan.tasks:
            for dep in task.dependencies:
                if dep in graph:
                    graph[dep].append(task.id)

        # Calculate longest path using DP
        memo = {}

        def longest_path(task_id: str) -> int:
            if task_id in memo:
                return memo[task_id]
            task = task_map.get(task_id)
            if not task:
                return 0
            if not graph[task_id]:
                memo[task_id] = task.estimated_effort
                return task.estimated_effort
            max_path = max(longest_path(next_id) for next_id in graph[task_id])
            memo[task_id] = task.estimated_effort + max_path
            return memo[task_id]

        # Find starting points (no dependencies)
        starts = [t.id for t in plan.tasks if not t.dependencies]
        if not starts:
            return []

        # Find the longest path
        critical_start = max(starts, key=longest_path)

        # Reconstruct path
        path = []
        current = critical_start
        while current:
            path.append(task_map[current])
            next_tasks = graph[current]
            if not next_tasks:
                break
            current = max(next_tasks, key=lambda x: memo.get(x, 0))

        return path
