"""Self-improving agent loop with reflection and iteration."""

from __future__ import annotations

import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Awaitable, Union
from datetime import datetime
from enum import Enum

from legionhercules.self_improve.reflection import ReflectionEngine, ReflectionResult
from legionhercules.self_improve.iteration import IterationManager, ImprovementCycle, IterationStatus
from legionhercules.memory.manager import MemoryManager
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class AgentLoopStatus(Enum):
    """Status of the agent loop."""
    IDLE = "idle"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    ITERATING = "iterating"
    CONVERGED = "converged"
    FAILED = "failed"


@dataclass
class AgentTask:
    """A task for the agent loop."""
    id: str
    description: str
    execute_fn: Callable[[], Awaitable[str]]
    validate_fn: Optional[Callable[[str], Awaitable[tuple[bool, str]]]] = None
    expected_outcome: Optional[str] = None
    max_iterations: int = 3
    convergence_threshold: float = 0.85
    context: Dict[str, Any] = field(default_factory=dict)
    status: AgentLoopStatus = AgentLoopStatus.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class AgentLoopResult:
    """Result of an agent loop execution."""
    task_id: str
    success: bool
    final_output: str
    iterations: int
    reflection_results: List[ReflectionResult]
    confidence_score: float
    duration_seconds: float
    status: AgentLoopStatus
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "final_output": self.final_output[:500] + "..." if len(self.final_output) > 500 else self.final_output,
            "iterations": self.iterations,
            "reflection_count": len(self.reflection_results),
            "confidence_score": self.confidence_score,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "metadata": self.metadata,
        }


class AgentLoop:
    """Self-improving agent loop with reflection and iteration."""

    def __init__(
        self,
        reflection_engine: Optional[ReflectionEngine] = None,
        iteration_manager: Optional[IterationManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_reflection: bool = True,
        enable_iteration: bool = True,
        default_max_iterations: int = 3,
        default_convergence_threshold: float = 0.85,
    ):
        self.reflection_engine = reflection_engine or ReflectionEngine()
        self.iteration_manager = iteration_manager or IterationManager(
            reflection_engine=self.reflection_engine,
        )
        self.memory_manager = memory_manager
        self.enable_reflection = enable_reflection
        self.enable_iteration = enable_iteration
        self.default_max_iterations = default_max_iterations
        self.default_convergence_threshold = default_convergence_threshold
        
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.task_history: List[Dict[str, Any]] = []

    async def execute(
        self,
        description: str,
        execute_fn: Callable[[], Awaitable[str]],
        validate_fn: Optional[Callable[[str], Awaitable[tuple[bool, str]]]] = None,
        expected_outcome: Optional[str] = None,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> AgentLoopResult:
        """Execute a task with self-improvement loop.
        
        Args:
            description: Task description
            execute_fn: Async function that executes the task
            validate_fn: Optional validation function returning (is_valid, feedback)
            expected_outcome: Expected outcome description
            max_iterations: Maximum improvement iterations
            convergence_threshold: Confidence threshold for convergence
            context: Additional context for the task
            task_id: Optional task ID (generated if not provided)
            
        Returns:
            AgentLoopResult with final output and execution metadata
        """
        task = AgentTask(
            id=task_id or self._generate_task_id(),
            description=description,
            execute_fn=execute_fn,
            validate_fn=validate_fn,
            expected_outcome=expected_outcome,
            max_iterations=max_iterations or self.default_max_iterations,
            convergence_threshold=convergence_threshold or self.default_convergence_threshold,
            context=context or {},
        )
        
        self.active_tasks[task.id] = task
        task.status = AgentLoopStatus.EXECUTING
        task.started_at = datetime.now()
        
        logger.info(f"Starting agent loop for task {task.id}: {description}")
        
        start_time = datetime.now()
        reflection_results: List[ReflectionResult] = []
        
        try:
            # Store task in memory if available
            if self.memory_manager:
                await self.memory_manager.store_memory(
                    content=f"Task: {description}",
                    metadata={"task_id": task.id, "type": "task_start"},
                )
            
            # Execute with iteration if enabled
            if self.enable_iteration and self.enable_reflection:
                cycle = await self.iteration_manager.improve(
                    task_id=task.id,
                    task_description=description,
                    execute_fn=execute_fn,
                    expected_outcome=expected_outcome,
                    context=context,
                )
                
                reflection_results = cycle.iterations
                final_output = cycle.final_output or ""
                iterations = cycle.iteration_count
                
                # Map iteration status to agent status
                if cycle.status == IterationStatus.CONVERGED:
                    task.status = AgentLoopStatus.CONVERGED
                elif cycle.status == IterationStatus.MAX_ITERATIONS:
                    task.status = AgentLoopStatus.CONVERGED
                elif cycle.status == IterationStatus.FAILED:
                    task.status = AgentLoopStatus.FAILED
                    
                # Get confidence from last reflection
                confidence = reflection_results[-1].confidence_score if reflection_results else 0.5
                
            elif self.enable_reflection:
                # Single execution with reflection only
                output = await execute_fn()
                reflection = await self.reflection_engine.reflect(
                    task_id=task.id,
                    task_description=description,
                    original_output=output,
                    expected_outcome=expected_outcome,
                )
                reflection_results = [reflection]
                final_output = reflection.improved_output or output
                iterations = 1
                confidence = reflection.confidence_score
                task.status = AgentLoopStatus.CONVERGED if confidence >= task.convergence_threshold else AgentLoopStatus.FAILED
                
            else:
                # Simple execution without reflection
                final_output = await execute_fn()
                iterations = 1
                confidence = 1.0
                reflection_results = []
                task.status = AgentLoopStatus.CONVERGED
            
            # Validate if validation function provided
            if validate_fn and final_output:
                is_valid, feedback = await validate_fn(final_output)
                if not is_valid:
                    task.status = AgentLoopStatus.FAILED
                    task.error = f"Validation failed: {feedback}"
                    confidence *= 0.5
            
            task.result = final_output
            task.completed_at = datetime.now()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = AgentLoopResult(
                task_id=task.id,
                success=task.status in [AgentLoopStatus.CONVERGED, AgentLoopStatus.IDLE],
                final_output=final_output,
                iterations=iterations,
                reflection_results=reflection_results,
                confidence_score=confidence,
                duration_seconds=duration,
                status=task.status,
                metadata={
                    "description": description,
                    "expected_outcome": expected_outcome,
                    "context": context,
                },
            )
            
            # Store result in memory
            if self.memory_manager:
                await self.memory_manager.store_memory(
                    content=f"Task completed: {description}\nResult: {final_output[:200]}",
                    metadata={
                        "task_id": task.id,
                        "type": "task_complete",
                        "success": result.success,
                        "confidence": confidence,
                    },
                )
            
            logger.info(f"Agent loop completed for task {task.id}: success={result.success}, iterations={iterations}")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent loop failed for task {task.id}: {e}")
            task.status = AgentLoopStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return AgentLoopResult(
                task_id=task.id,
                success=False,
                final_output="",
                iterations=0,
                reflection_results=reflection_results,
                confidence_score=0.0,
                duration_seconds=duration,
                status=AgentLoopStatus.FAILED,
                metadata={"error": str(e)},
            )
            
        finally:
            del self.active_tasks[task.id]
            self.completed_tasks.append(task)
            self.task_history.append(task.to_dict())

    async def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[AgentLoopResult]:
        """Execute multiple tasks with concurrency control.
        
        Args:
            tasks: List of task dictionaries with description, execute_fn, etc.
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of AgentLoopResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_limit(task_def: Dict[str, Any]) -> AgentLoopResult:
            async with semaphore:
                return await self.execute(**task_def)
        
        results = await asyncio.gather(
            *[execute_with_limit(task) for task in tasks],
            return_exceptions=True,
        )
        
        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Task failed with exception: {r}")
            else:
                valid_results.append(r)
        
        return valid_results

    async def reflect(
        self,
        task_id: str,
        task_description: str,
        original_output: str,
        expected_outcome: Optional[str] = None,
    ) -> ReflectionResult:
        """Perform reflection on a task output.
        
        Args:
            task_id: Task identifier
            task_description: Description of the task
            original_output: Output to reflect on
            expected_outcome: Expected outcome
            
        Returns:
            ReflectionResult with analysis and improvements
        """
        return await self.reflection_engine.reflect(
            task_id=task_id,
            task_description=task_description,
            original_output=original_output,
            expected_outcome=expected_outcome,
        )

    async def iterate(
        self,
        task_id: str,
        current_output: str,
        reflection: ReflectionResult,
    ) -> str:
        """Iterate on output based on reflection.
        
        Args:
            task_id: Task identifier
            current_output: Current output
            reflection: Reflection result with improvements
            
        Returns:
            Improved output
        """
        if reflection.improved_output:
            return reflection.improved_output
        return current_output

    def get_active_task(self, task_id: str) -> Optional[AgentTask]:
        """Get an active task by ID."""
        return self.active_tasks.get(task_id)

    def get_task_history(
        self,
        success_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get task execution history.
        
        Args:
            success_only: Filter to successful tasks only
            limit: Maximum number of tasks to return
            
        Returns:
            List of task dictionaries
        """
        history = self.task_history
        
        if success_only:
            history = [h for h in history if h.get("result") and not h.get("error")]
        
        history = sorted(history, key=lambda x: x.get("created_at", ""), reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get agent loop statistics."""
        if not self.task_history:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "success_rate": 0.0,
                "average_iterations": 0.0,
            }
        
        total = len(self.task_history)
        successful = sum(1 for t in self.task_history if t.get("result") and not t.get("error"))
        failed = total - successful
        
        # Get iteration stats from reflection engine
        reflection_stats = self.reflection_engine.get_improvement_stats()
        
        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "success_rate": successful / total if total > 0 else 0,
            "active_tasks": len(self.active_tasks),
            "reflection_stats": reflection_stats,
        }

    def clear_history(self) -> None:
        """Clear task history."""
        self.task_history.clear()
        self.completed_tasks.clear()
        self.reflection_engine.reflection_history.clear()
        self.iteration_manager.clear_history()
        logger.info("Agent loop history cleared")

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{uuid.uuid4().hex[:12]}"


class ConvergenceDetector:
    """Detects when further iterations won't improve results."""
    
    def __init__(
        self,
        threshold: float = 0.01,
        window_size: int = 3,
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.scores: List[float] = []
    
    def add_score(self, score: float) -> None:
        """Add a confidence score."""
        self.scores.append(score)
        if len(self.scores) > self.window_size:
            self.scores = self.scores[-self.window_size:]
    
    def has_converged(self) -> bool:
        """Check if scores have converged."""
        if len(self.scores) < self.window_size:
            return False
        
        # Check if improvement is below threshold
        deltas = [self.scores[i] - self.scores[i-1] for i in range(1, len(self.scores))]
        avg_delta = sum(deltas) / len(deltas)
        
        return abs(avg_delta) < self.threshold
    
    def reset(self) -> None:
        """Reset the detector."""
        self.scores.clear()


class AdaptiveAgentLoop(AgentLoop):
    """Agent loop with adaptive iteration limits based on task complexity."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_scores: Dict[str, float] = {}
    
    def estimate_complexity(self, description: str) -> float:
        """Estimate task complexity (0.0 to 1.0)."""
        # Simple heuristic based on description length and keywords
        complexity = 0.5
        
        # Longer descriptions tend to be more complex
        if len(description) > 200:
            complexity += 0.2
        
        # Keywords indicating complexity
        complex_keywords = [
            "complex", "difficult", "challenging", "advanced",
            "multi-step", "comprehensive", "detailed", "extensive",
        ]
        for keyword in complex_keywords:
            if keyword in description.lower():
                complexity += 0.1
        
        # Cap at 1.0
        return min(1.0, complexity)
    
    async def execute(
        self,
        description: str,
        execute_fn: Callable[[], Awaitable[str]],
        **kwargs,
    ) -> AgentLoopResult:
        """Execute with adaptive iteration limit."""
        complexity = self.estimate_complexity(description)
        
        # Adjust max iterations based on complexity
        if "max_iterations" not in kwargs:
            if complexity > 0.8:
                kwargs["max_iterations"] = 5
            elif complexity > 0.5:
                kwargs["max_iterations"] = 4
            else:
                kwargs["max_iterations"] = 2
        
        result = await super().execute(description, execute_fn, **kwargs)
        
        # Store complexity for learning
        self.complexity_scores[result.task_id] = complexity
        
        return result
