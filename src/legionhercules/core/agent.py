"""Agent implementation for LEGIONHERCULES."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from legionhercules.core.message import Message, MessageRole
from legionhercules.core.task import Task, TaskResult, TaskStatus
from legionhercules.llm.base import LLMProvider
from legionhercules.tools.base import Tool, ToolRegistry
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    
    name: str = "agent"
    description: str = ""
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = "You are a helpful AI assistant."
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 10
    timeout_seconds: float = 120.0


class Agent:
    """An autonomous agent that can execute tasks using tools and LLM."""
    
    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
    ):
        self.config = config
        self.llm = llm_provider
        self.tools = tool_registry
        self.messages: list[Message] = []
        self._initialized = False
        
        # Add system message
        self.messages.append(Message.system(config.system_prompt))
    
    async def initialize(self) -> None:
        """Initialize the agent."""
        if not self._initialized:
            await self.llm.initialize()
            self._initialized = True
            logger.info(f"Agent '{self.config.name}' initialized")
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task."""
        task.mark_running()
        logger.info(f"Agent '{self.config.name}' executing task: {task.description}")
        
        try:
            # Add task description as user message
            self.messages.append(Message.user(task.description))
            
            # Execute with tool loop
            result = await self._execute_with_tools(task)
            
            task.mark_completed(result)
            logger.info(f"Task {task.id} completed: success={result.success}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.id} failed: {error_msg}")
            result = TaskResult(success=False, error=error_msg)
            task.mark_completed(result)
            return result
    
    async def _execute_with_tools(self, task: Task) -> TaskResult:
        """Execute task with tool use loop."""
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            
            # Get LLM response
            response = await self.llm.chat(
                messages=self.messages,
                tools=self.tools.get_tools() if self.config.tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # Add assistant message
            self.messages.append(Message.assistant(
                content=response.content,
                tool_calls=response.tool_calls,
            ))
            
            # Check if we need to execute tools
            if not response.tool_calls:
                # No tool calls, we're done
                return TaskResult(
                    success=True,
                    output=response.content,
                    metadata={"iterations": iteration}
                )
            
            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                result = await self._execute_tool_call(tool_call)
                tool_results.append(result)
            
            # Add tool results as message
            tool_content = "\n".join([
                f"Tool: {r['tool']}\nResult: {r['result']}"
                for r in tool_results
            ])
            self.messages.append(Message.tool(
                content=tool_content,
                tool_results=tool_results
            ))
        
        # Max iterations reached
        return TaskResult(
            success=False,
            output=self.messages[-1].content if self.messages else "",
            error=f"Max iterations ({self.config.max_iterations}) reached",
            metadata={"iterations": iteration}
        )
    
    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call."""
        tool_name = tool_call.get("name", "")
        tool_input = tool_call.get("arguments", {})
        
        logger.debug(f"Executing tool: {tool_name} with args: {tool_input}")
        
        tool = self.tools.get_tool(tool_name)
        if not tool:
            return {
                "tool": tool_name,
                "result": f"Error: Tool '{tool_name}' not found",
                "success": False
            }
        
        try:
            result = await tool.execute(**tool_input)
            return {
                "tool": tool_name,
                "result": result.output if result.success else result.error,
                "success": result.success
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "result": f"Error: {str(e)}",
                "success": False
            }
    
    def reset(self) -> None:
        """Reset agent conversation history."""
        self.messages = [Message.system(self.config.system_prompt)]
        logger.debug(f"Agent '{self.config.name}' conversation reset")
    
    def get_conversation_history(self) -> list[Message]:
        """Get conversation history."""
        return self.messages.copy()
