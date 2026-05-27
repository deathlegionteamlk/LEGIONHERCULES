"""Agent orchestrator with async task queue and parallel execution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Optional

from legionhercules.core.agent import Agent, AgentConfig
from legionhercules.core.task import Task, TaskResult, TaskStatus
from legionhercules.llm.base import LLMProvider
from legionhercules.tools.base import ToolRegistry
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """Orchestrates multiple agents with parallel task execution."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        max_concurrent_tasks: int = 5,
    ):
        self.llm_provider = llm_provider
        self.tools = tool_registry
        self.max_concurrent = max_concurrent_tasks
        
        # Task management
        self.tasks: dict[str, Task] = {}
        self.task_queue: asyncio.PriorityQueue[tuple[int, str]] = asyncio.PriorityQueue()
        self._task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Agent pool
        self.agents: dict[str, Agent] = {}
        self.agent_configs: dict[str, AgentConfig] = {}
        
        # Results aggregation
        self.results: dict[str, TaskResult] = {}
        self._running = False
        self._workers: list[asyncio.Task] = []
    
    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        await self.llm_provider.initialize()
        logger.info(f"Orchestrator initialized (max_concurrent={self.max_concurrent})")
    
    def register_agent(self, config: AgentConfig) -> str:
        """Register an agent configuration."""
        self.agent_configs[config.name] = config
        logger.info(f"Registered agent: {config.name}")
        return config.name
    
    async def create_agent(self, name: str) -> Agent:
        """Create an agent instance."""
        if name not in self.agent_configs:
            raise ValueError(f"Agent '{name}' not registered")
        
        if name not in self.agents:
            config = self.agent_configs[name]
            agent = Agent(config, self.llm_provider, self.tools)
            await agent.initialize()
            self.agents[name] = agent
        
        return self.agents[name]
    
    def create_task(
        self,
        description: str,
        agent_name: Optional[str] = None,
        priority: int = 5,
        dependencies: Optional[list[str]] = None,
        func: Optional[Callable[..., Any]] = None,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> Task:
        """Create and queue a new task."""
        task = Task(
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            func=func,
            args=args or (),
            kwargs=kwargs or {},
            context=context or {},
        )
        
        # Store agent assignment in context
        if agent_name:
            task.context["agent_name"] = agent_name
        
        self.tasks[task.id] = task
        
        # Add to queue if ready
        if task.is_ready:
            self.task_queue.put_nowait((priority, task.id))
        
        logger.info(f"Created task {task.id}: {description[:50]}...")
        return task
    
    async def execute_parallel(
        self,
        tasks: list[Task],
        aggregation_strategy: str = "all",
    ) -> dict[str, TaskResult]:
        """Execute multiple tasks in parallel."""
        logger.info(f"Executing {len(tasks)} tasks in parallel")
        
        # Queue all tasks
        for task in tasks:
            if task.is_ready:
                await self.task_queue.put((task.priority, task.id))
        
        # Start workers
        self._running = True
        workers = [
            asyncio.create_task(self._worker_loop())
            for _ in range(self.max_concurrent)
        ]
        
        # Wait for all tasks to complete
        await self._wait_for_completion(tasks)
        
        # Stop workers
        self._running = False
        for worker in workers:
            worker.cancel()
        
        # Collect results
        results = {task.id: self.results.get(task.id) for task in tasks}
        
        # Aggregate if needed
        if aggregation_strategy == "all":
            return results
        elif aggregation_strategy == "first_success":
            return self._aggregate_first_success(results)
        elif aggregation_strategy == "majority":
            return self._aggregate_majority(results)
        
        return results
    
    async def _worker_loop(self) -> None:
        """Worker loop for processing tasks."""
        while self._running:
            try:
                # Get task from queue with timeout
                priority, task_id = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                # Execute with semaphore
                async with self._task_semaphore:
                    await self._execute_task(task)
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        agent_name = task.context.get("agent_name", "default")
        
        try:
            # Get or create agent
            if agent_name not in self.agents:
                if agent_name in self.agent_configs:
                    await self.create_agent(agent_name)
                else:
                    # Create default agent
                    config = AgentConfig(name=agent_name)
                    self.register_agent(config)
                    await self.create_agent(agent_name)
            
            agent = self.agents[agent_name]
            
            # Execute task
            result = await agent.execute_task(task)
            self.results[task.id] = result
            
            # Update dependent tasks
            await self._update_dependencies(task)
            
        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {e}")
            result = TaskResult(success=False, error=str(e))
            task.mark_completed(result)
            self.results[task.id] = result
    
    async def _update_dependencies(self, completed_task: Task) -> None:
        """Update tasks that depend on the completed task."""
        for task in self.tasks.values():
            if completed_task.id in task.dependencies:
                task.dependencies.remove(completed_task.id)
                
                # If task is now ready, add to queue
                if task.is_ready:
                    await self.task_queue.put((task.priority, task.id))
    
    async def _wait_for_completion(self, tasks: list[Task]) -> None:
        """Wait for all tasks to complete."""
        pending_tasks = {t.id for t in tasks}
        
        while pending_tasks:
            completed = {tid for tid in pending_tasks if tid in self.results}
            pending_tasks -= completed
            
            if pending_tasks:
                await asyncio.sleep(0.1)
    
    def _aggregate_first_success(
        self,
        results: dict[str, TaskResult]
    ) -> dict[str, TaskResult]:
        """Return first successful result."""
        for task_id, result in results.items():
            if result and result.success:
                return {task_id: result}
        return results
    
    def _aggregate_majority(
        self,
        results: dict[str, TaskResult]
    ) -> dict[str, TaskResult]:
        """Aggregate results by majority vote."""
        # Simple implementation - can be extended
        successful = sum(1 for r in results.values() if r and r.success)
        total = len(results)
        
        majority_result = TaskResult(
            success=successful > total / 2,
            output=f"Majority vote: {successful}/{total} successful",
            metadata={"successful": successful, "total": total}
        )
        
        return {"majority": majority_result}
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def get_all_results(self) -> dict[str, TaskResult]:
        """Get all task results."""
        return self.results.copy()
    
    def reset(self) -> None:
        """Reset orchestrator state."""
        self.tasks.clear()
        self.results.clear()
        self.agents.clear()
        self.agent_configs.clear()
        # Clear queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("Orchestrator reset")
