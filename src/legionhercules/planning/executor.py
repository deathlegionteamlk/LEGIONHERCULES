"""Plan executor for multi-step task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Awaitable
from datetime import datetime
from enum import Enum
import asyncio

from legionhercules.planning.decomposer import Plan, Task, TaskStatus, TaskDecomposer
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Execution mode for plans."""
    SEQUENTIAL = "sequential"  # Execute one task at a time
    PARALLEL = "parallel"      # Execute independent tasks in parallel
    ADAPTIVE = "adaptive"      # Adapt based on task dependencies


@dataclass
class ExecutionResult:
    """Result of plan execution."""
    plan_id: str
    success: bool
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    skipped_tasks: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get execution duration."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "skipped_tasks": self.skipped_tasks,
            "results": self.results,
            "errors": self.errors,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class PlanExecutor:
    """Executes plans with support for sequential and parallel execution."""

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        max_parallel_tasks: int = 5,
        task_timeout: Optional[int] = 300,
    ):
        self.mode = mode
        self.max_parallel_tasks = max_parallel_tasks
        self.task_timeout = task_timeout
        self.decomposer = TaskDecomposer()
        self._task_handlers: Dict[str, Callable[[Task], Awaitable[Any]]] = {}
        self._default_handler: Optional[Callable[[Task], Awaitable[Any]]] = None

    def register_task_handler(
        self,
        task_type: str,
        handler: Callable[[Task], Awaitable[Any]],
    ) -> None:
        """Register a handler for a specific task type."""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    def set_default_handler(
        self,
        handler: Callable[[Task], Awaitable[Any]],
    ) -> None:
        """Set default handler for tasks without specific handlers."""
        self._default_handler = handler

    async def execute(
        self,
        plan: Plan,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Task, ExecutionResult], Awaitable[None]]] = None,
    ) -> ExecutionResult:
        """Execute a plan."""
        logger.info(f"Starting execution of plan {plan.id}: {plan.goal}")
        
        result = ExecutionResult(
            plan_id=plan.id,
            success=True,
            metadata={"goal": plan.goal, "mode": self.mode.value},
        )

        try:
            if self.mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(plan, result, context, progress_callback)
            elif self.mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(plan, result, context, progress_callback)
            else:  # ADAPTIVE
                await self._execute_adaptive(plan, result, context, progress_callback)

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            result.success = False
            result.metadata["execution_error"] = str(e)

        finally:
            result.end_time = datetime.now()
            plan.completed_at = datetime.now()

        return result

    async def _execute_sequential(
        self,
        plan: Plan,
        result: ExecutionResult,
        context: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[Task, ExecutionResult], Awaitable[None]]],
    ) -> None:
        """Execute tasks sequentially in dependency order."""
        execution_order = self.decomposer.get_execution_order(plan)

        for task in execution_order:
            if task.status in [TaskStatus.SKIPPED, TaskStatus.FAILED]:
                continue

            success = await self._execute_task(task, context)
            
            if success:
                result.completed_tasks.append(task.id)
                result.results[task.id] = task.result
            else:
                result.failed_tasks.append(task.id)
                result.errors[task.id] = task.error or "Unknown error"
                result.success = False

            if progress_callback:
                await progress_callback(task, result)

    async def _execute_parallel(
        self,
        plan: Plan,
        result: ExecutionResult,
        context: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[Task, ExecutionResult], Awaitable[None]]],
    ) -> None:
        """Execute independent tasks in parallel."""
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        completed_ids = set()

        async def execute_with_limit(task: Task) -> Task:
            async with semaphore:
                await self._execute_task(task, context)
                return task

        while True:
            # Get tasks that are ready to execute
            ready_tasks = self.decomposer.get_ready_tasks(plan)
            
            if not ready_tasks:
                # Check if all tasks are done
                if all(t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED] for t in plan.tasks):
                    break
                # Wait a bit for dependencies
                await asyncio.sleep(0.1)
                continue

            # Execute ready tasks in parallel
            tasks_to_run = [t for t in ready_tasks if t.id not in completed_ids]
            if not tasks_to_run:
                break

            # Mark tasks as in progress
            for task in tasks_to_run:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()

            # Execute tasks
            done_tasks = await asyncio.gather(
                *[execute_with_limit(t) for t in tasks_to_run],
                return_exceptions=True,
            )

            # Process results
            for task in done_tasks:
                if isinstance(task, Exception):
                    logger.error(f"Task execution raised exception: {task}")
                    continue

                completed_ids.add(task.id)
                task.completed_at = datetime.now()

                if task.status == TaskStatus.COMPLETED:
                    result.completed_tasks.append(task.id)
                    result.results[task.id] = task.result
                elif task.status == TaskStatus.FAILED:
                    result.failed_tasks.append(task.id)
                    result.errors[task.id] = task.error or "Unknown error"
                    result.success = False

                if progress_callback:
                    await progress_callback(task, result)

    async def _execute_adaptive(
        self,
        plan: Plan,
        result: ExecutionResult,
        context: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[Task, ExecutionResult], Awaitable[None]]],
    ) -> None:
        """Execute tasks adaptively based on dependencies and priorities."""
        # Use parallel execution but respect dependencies
        await self._execute_parallel(plan, result, context, progress_callback)

    async def _execute_task(
        self,
        task: Task,
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """Execute a single task."""
        logger.info(f"Executing task {task.id}: {task.description}")
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            # Get handler for task
            handler = self._get_handler(task)
            
            if handler is None:
                raise ValueError(f"No handler registered for task: {task.description}")

            # Execute with timeout
            if self.task_timeout:
                task.result = await asyncio.wait_for(
                    handler(task),
                    timeout=self.task_timeout,
                )
            else:
                task.result = await handler(task)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info(f"Task {task.id} completed successfully")
            return True

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timed out after {self.task_timeout} seconds"
            logger.error(f"Task {task.id} timed out")
            return False

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.id} failed: {e}")
            return False

    def _get_handler(self, task: Task) -> Optional[Callable[[Task], Awaitable[Any]]]:
        """Get appropriate handler for task."""
        # Check for task type in metadata
        task_type = task.metadata.get("type", "")
        if task_type and task_type in self._task_handlers:
            return self._task_handlers[task_type]

        # Check for handler based on description keywords
        desc_lower = task.description.lower()
        for keyword, handler in self._task_handlers.items():
            if keyword.lower() in desc_lower:
                return handler

        # Return default handler
        return self._default_handler

    async def execute_with_recovery(
        self,
        plan: Plan,
        max_retries: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute plan with automatic retry on failure."""
        for attempt in range(max_retries):
            result = await self.execute(plan, context)
            
            if result.success:
                return result

            logger.warning(f"Plan execution failed (attempt {attempt + 1}/{max_retries})")
            
            # Reset failed tasks for retry
            for task in plan.tasks:
                if task.status == TaskStatus.FAILED:
                    task.status = TaskStatus.PENDING
                    task.error = None
                    task.result = None

            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return result

    def get_execution_stats(self, result: ExecutionResult) -> Dict[str, Any]:
        """Get statistics from execution result."""
        total_tasks = len(result.completed_tasks) + len(result.failed_tasks) + len(result.skipped_tasks)
        
        return {
            "total_tasks": total_tasks,
            "completed": len(result.completed_tasks),
            "failed": len(result.failed_tasks),
            "skipped": len(result.skipped_tasks),
            "success_rate": len(result.completed_tasks) / total_tasks if total_tasks > 0 else 0,
            "duration_seconds": result.duration_seconds,
            "tasks_per_second": total_tasks / result.duration_seconds if result.duration_seconds > 0 else 0,
        }
