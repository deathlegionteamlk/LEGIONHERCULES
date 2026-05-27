"""Subagent system for parallel task delegation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from legionhercules.core.agent import Agent, AgentConfig
from legionhercules.core.task import Task, TaskResult
from legionhercules.llm.base import LLMProvider
from legionhercules.tools.base import ToolRegistry
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""
    name: str
    description: str = ""
    parent_agent: Optional[str] = None
    scope: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 10
    timeout_seconds: float = 120.0


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    subagent_id: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Subagent:
    """A subagent that can be spawned for parallel task execution."""
    
    def __init__(
        self,
        config: SubagentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
    ):
        self.config = config
        self.id = str(uuid4())
        self.llm_provider = llm_provider
        self.tools = tool_registry
        self.agent: Optional[Agent] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the subagent."""
        agent_config = AgentConfig(
            name=self.config.name,
            description=self.config.description,
            system_prompt=self._build_system_prompt(),
            max_iterations=self.config.max_iterations,
            timeout_seconds=self.config.timeout_seconds,
        )
        
        self.agent = Agent(agent_config, self.llm_provider, self.tools)
        await self.agent.initialize()
        self._initialized = True
        
        logger.info(f"Subagent '{self.config.name}' initialized (id: {self.id})")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with scope information."""
        base_prompt = f"""You are a specialized subagent named '{self.config.name}'.
{self.config.description}

You are working within a scoped context. Focus only on your assigned task.
Report back results clearly and concisely to your parent agent.
"""
        if self.config.scope:
            base_prompt += f"\n\nScope:\n"
            for key, value in self.config.scope.items():
                base_prompt += f"- {key}: {value}\n"
        
        return base_prompt
    
    async def execute(self, task_description: str, context: Optional[dict] = None) -> SubagentResult:
        """Execute a task."""
        if not self._initialized:
            await self.initialize()
        
        if not self.agent:
            return SubagentResult(
                subagent_id=self.id,
                success=False,
                error="Subagent not initialized"
            )
        
        try:
            # Create task
            task = Task(
                description=task_description,
                context=context or {},
            )
            
            # Execute via agent
            result = await self.agent.execute_task(task)
            
            return SubagentResult(
                subagent_id=self.id,
                success=result.success,
                output=result.output if result.success else "",
                error=result.error if not result.success else None,
                metadata={
                    "iterations": result.metadata.get("iterations", 0),
                    "parent_agent": self.config.parent_agent,
                }
            )
            
        except Exception as e:
            logger.error(f"Subagent execution failed: {e}")
            return SubagentResult(
                subagent_id=self.id,
                success=False,
                error=str(e),
            )
    
    async def close(self) -> None:
        """Clean up subagent resources."""
        if self.agent:
            self.agent.reset()
            self.agent = None
        self._initialized = False


class SubagentManager:
    """Manages spawning and coordinating multiple subagents."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        max_subagents: int = 10,
    max_concurrent: Optional[int] = None,
        ):
        self.llm_provider = llm_provider
        self.tools = tool_registry
        self.max_subagents = max_subagents
        self.max_concurrent = max_concurrent or max_subagents
        self.subagents: dict[str, Subagent] = {}
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def spawn_subagent(
        self,
        name: str,
        description: str = "",
        scope: Optional[dict] = None,
        parent_agent: Optional[str] = None,
    ) -> Subagent:
        """Spawn a new subagent."""
        config = SubagentConfig(
            name=name,
            description=description,
            parent_agent=parent_agent,
            scope=scope or {},
        )
        
        subagent = Subagent(config, self.llm_provider, self.tools)
        await subagent.initialize()
        
        self.subagents[subagent.id] = subagent
        logger.info(f"Spawned subagent: {name} (id: {subagent.id})")
        
        return subagent
    
    async def delegate_task(
        self,
        subagent_id: str,
        task_description: str,
        context: Optional[dict] = None,
    ) -> SubagentResult:
        """Delegate a task to a subagent."""
        subagent = self.subagents.get(subagent_id)
        if not subagent:
            return SubagentResult(
                subagent_id=subagent_id,
                success=False,
                error=f"Subagent {subagent_id} not found"
            )
        
        async with self._semaphore:
            return await subagent.execute(task_description, context)
    
    async def delegate_parallel(
        self,
        tasks: list[tuple[str, str, Optional[dict]]],
    ) -> list[SubagentResult]:
        """Delegate multiple tasks to subagents in parallel.
        
        Args:
            tasks: List of (subagent_id, task_description, context) tuples
        
        Returns:
            List of SubagentResults in the same order as input
        """
        async def execute_with_semaphore(subagent_id: str, desc: str, ctx: Optional[dict]):
            async with self._semaphore:
                subagent = self.subagents.get(subagent_id)
                if not subagent:
                    return SubagentResult(
                        subagent_id=subagent_id,
                        success=False,
                        error=f"Subagent {subagent_id} not found"
                    )
                return await subagent.execute(desc, ctx)
        
        # Create tasks for parallel execution
        coroutines = [
            execute_with_semaphore(sid, desc, ctx)
            for sid, desc, ctx in tasks
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SubagentResult(
                    subagent_id=tasks[i][0],
                    success=False,
                    error=str(result),
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_subagent(self, subagent_id: str) -> Optional[Subagent]:
        """Get a subagent by ID."""
        return self.subagents.get(subagent_id)
    
    def list_subagents(self) -> list[dict[str, Any]]:
        """List all active subagents."""
        return [
            {
                "id": sid,
                "name": subagent.config.name,
                "description": subagent.config.description,
            }
            for sid, subagent in self.subagents.items()
        ]
    
    async def terminate_subagent(self, subagent_id: str) -> bool:
        """Terminate a subagent."""
        subagent = self.subagents.pop(subagent_id, None)
        if subagent:
            await subagent.close()
            logger.info(f"Terminated subagent: {subagent_id}")
            return True
        return False
    
    async def terminate_all(self) -> None:
        """Terminate all subagents."""
        for subagent in self.subagents.values():
            await subagent.close()
        self.subagents.clear()
        logger.info("All subagents terminated")