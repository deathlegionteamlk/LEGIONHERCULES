"""Iteration manager for self-improving agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Awaitable
from datetime import datetime
from enum import Enum
import asyncio

from legionhercules.self_improve.reflection import ReflectionEngine, ReflectionResult, ReflectionStatus
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class IterationStatus(Enum):
    """Status of improvement iteration."""
    IDLE = "idle"
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"


@dataclass
class ImprovementCycle:
    """A complete improvement cycle."""
    task_id: str
    task_description: str
    iterations: List[ReflectionResult] = field(default_factory=list)
    status: IterationStatus = IterationStatus.IDLE
    final_output: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    convergence_threshold: float = 0.85
    max_iterations: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get duration of cycle in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "iterations": [i.to_dict() for i in self.iterations],
            "status": self.status.value,
            "final_output": self.final_output,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "convergence_threshold": self.convergence_threshold,
            "max_iterations": self.max_iterations,
            "metadata": self.metadata,
        }


class IterationManager:
    """Manages iterative improvement cycles for tasks."""

    def __init__(
        self,
        reflection_engine: Optional[ReflectionEngine] = None,
        max_iterations: int = 3,
        convergence_threshold: float = 0.85,
        min_improvement_delta: float = 0.1,
    ):
        self.reflection_engine = reflection_engine or ReflectionEngine()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_improvement_delta = min_improvement_delta
        self.active_cycles: Dict[str, ImprovementCycle] = {}
        self.completed_cycles: List[ImprovementCycle] = []

    async def improve(
        self,
        task_id: str,
        task_description: str,
        execute_fn: Callable[[], Awaitable[str]],
        expected_outcome: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ImprovementCycle:
        """Execute task with iterative improvement."""
        cycle = ImprovementCycle(
            task_id=task_id,
            task_description=task_description,
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
            metadata=context or {},
        )
        
        self.active_cycles[task_id] = cycle
        cycle.status = IterationStatus.RUNNING

        try:
            # Initial execution
            logger.info(f"Starting improvement cycle for task {task_id}")
            current_output = await execute_fn()

            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"Iteration {iteration}/{self.max_iterations} for task {task_id}")

                # Reflect on current output
                reflection = await self.reflection_engine.reflect(
                    task_id=task_id,
                    task_description=task_description,
                    original_output=current_output,
                    expected_outcome=expected_outcome,
                    iteration=iteration,
                )

                cycle.iterations.append(reflection)

                # Check if converged
                if reflection.confidence_score >= self.convergence_threshold:
                    logger.info(f"Task {task_id} converged with confidence {reflection.confidence_score:.2f}")
                    cycle.status = IterationStatus.CONVERGED
                    cycle.final_output = reflection.improved_output or current_output
                    break

                # Check for improvement
                if iteration > 1:
                    prev_confidence = cycle.iterations[-2].confidence_score
                    improvement = reflection.confidence_score - prev_confidence
                    
                    if improvement < self.min_improvement_delta:
                        logger.info(f"Task {task_id} improvement stalled (delta: {improvement:.2f})")
                        cycle.status = IterationStatus.CONVERGED
                        cycle.final_output = reflection.improved_output or current_output
                        break

                # Update output for next iteration
                if reflection.improved_output:
                    current_output = reflection.improved_output

            else:
                # Max iterations reached
                logger.info(f"Task {task_id} reached max iterations")
                cycle.status = IterationStatus.MAX_ITERATIONS
                if cycle.iterations:
                    cycle.final_output = cycle.iterations[-1].improved_output or current_output
                else:
                    cycle.final_output = current_output

        except Exception as e:
            logger.error(f"Improvement cycle failed for task {task_id}: {e}")
            cycle.status = IterationStatus.FAILED
            cycle.metadata["error"] = str(e)
            cycle.final_output = current_output if 'current_output' in locals() else None

        finally:
            cycle.end_time = datetime.now()
            del self.active_cycles[task_id]
            self.completed_cycles.append(cycle)

        return cycle

    async def improve_with_feedback(
        self,
        task_id: str,
        task_description: str,
        execute_fn: Callable[[], Awaitable[str]],
        feedback_fn: Callable[[str], Awaitable[tuple[bool, str]]],
    ) -> ImprovementCycle:
        """Execute task with human-in-the-loop feedback."""
        cycle = ImprovementCycle(
            task_id=task_id,
            task_description=task_description,
            max_iterations=self.max_iterations,
        )
        
        self.active_cycles[task_id] = cycle
        cycle.status = IterationStatus.RUNNING

        try:
            current_output = await execute_fn()

            for iteration in range(1, self.max_iterations + 1):
                # Get feedback
                is_satisfied, feedback = await feedback_fn(current_output)
                
                if is_satisfied:
                    cycle.status = IterationStatus.CONVERGED
                    cycle.final_output = current_output
                    break

                # Use feedback to improve
                reflection = await self.reflection_engine.reflect(
                    task_id=task_id,
                    task_description=f"{task_description}\n\nFeedback: {feedback}",
                    original_output=current_output,
                    iteration=iteration,
                )

                cycle.iterations.append(reflection)

                if reflection.improved_output:
                    current_output = reflection.improved_output

            else:
                cycle.status = IterationStatus.MAX_ITERATIONS
                cycle.final_output = current_output

        except Exception as e:
            logger.error(f"Feedback improvement cycle failed: {e}")
            cycle.status = IterationStatus.FAILED
            cycle.metadata["error"] = str(e)

        finally:
            cycle.end_time = datetime.now()
            del self.active_cycles[task_id]
            self.completed_cycles.append(cycle)

        return cycle

    def get_active_cycle(self, task_id: str) -> Optional[ImprovementCycle]:
        """Get currently active improvement cycle."""
        return self.active_cycles.get(task_id)

    def get_completed_cycles(
        self,
        task_id: Optional[str] = None,
        status: Optional[IterationStatus] = None,
    ) -> List[ImprovementCycle]:
        """Get completed cycles with optional filtering."""
        cycles = self.completed_cycles
        
        if task_id:
            cycles = [c for c in cycles if c.task_id == task_id]
        
        if status:
            cycles = [c for c in cycles if c.status == status]
        
        return cycles

    def get_stats(self) -> Dict[str, Any]:
        """Get iteration statistics."""
        if not self.completed_cycles:
            return {}

        total = len(self.completed_cycles)
        converged = sum(1 for c in self.completed_cycles if c.status == IterationStatus.CONVERGED)
        max_iter = sum(1 for c in self.completed_cycles if c.status == IterationStatus.MAX_ITERATIONS)
        failed = sum(1 for c in self.completed_cycles if c.status == IterationStatus.FAILED)
        
        avg_iterations = sum(c.iteration_count for c in self.completed_cycles) / total
        avg_duration = sum(c.duration_seconds for c in self.completed_cycles) / total

        return {
            "total_cycles": total,
            "converged": converged,
            "max_iterations_reached": max_iter,
            "failed": failed,
            "convergence_rate": converged / total if total > 0 else 0,
            "average_iterations": avg_iterations,
            "average_duration_seconds": avg_duration,
            "active_cycles": len(self.active_cycles),
        }

    def clear_history(self) -> None:
        """Clear completed cycle history."""
        self.completed_cycles.clear()
        logger.info("Iteration history cleared")
