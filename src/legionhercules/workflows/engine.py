"""Workflow automation system for LEGIONHERCULES."""

from __future__ import annotations

import yaml
import json
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Awaitable
from datetime import datetime
from enum import Enum
from pathlib import Path
import asyncio

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowStatus(Enum):
    """Status of a workflow."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    id: str
    name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "params": self.params,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Workflow:
    """A complete workflow definition."""
    id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "variables": self.variables,
            "triggers": self.triggers,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class WorkflowEngine:
    """Engine for executing workflows."""

    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.actions: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self._register_default_actions()

    def _register_default_actions(self) -> None:
        """Register default workflow actions."""
        self.register_action("shell", self._action_shell)
        self.register_action("python", self._action_python)
        self.register_action("delay", self._action_delay)
        self.register_action("notify", self._action_notify)
        self.register_action("condition", self._action_condition)
        self.register_action("parallel", self._action_parallel)

    def register_action(
        self,
        name: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Any]],
    ) -> None:
        """Register a workflow action handler."""
        self.actions[name] = handler
        logger.info(f"Registered workflow action: {name}")

    async def load_from_file(self, filepath: str) -> Optional[Workflow]:
        """Load workflow from YAML or JSON file."""
        path = Path(filepath)
        
        if not path.exists():
            logger.error(f"Workflow file not found: {filepath}")
            return None
        
        try:
            content = path.read_text()
            
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(content)
            elif path.suffix == '.json':
                data = json.loads(content)
            else:
                logger.error(f"Unsupported workflow file format: {path.suffix}")
                return None
            
            return self._parse_workflow(data)
            
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            return None

    def _parse_workflow(self, data: Dict[str, Any]) -> Workflow:
        """Parse workflow data into Workflow object."""
        steps = []
        for step_data in data.get("steps", []):
            step = WorkflowStep(
                id=step_data.get("id", f"step_{len(steps)}"),
                name=step_data.get("name", "Unnamed Step"),
                action=step_data.get("action", "shell"),
                params=step_data.get("params", {}),
                depends_on=step_data.get("depends_on", []),
                condition=step_data.get("condition"),
                max_retries=step_data.get("max_retries", 3),
                timeout=step_data.get("timeout"),
            )
            steps.append(step)
        
        return Workflow(
            id=data.get("id", f"workflow_{datetime.now().timestamp()}"),
            name=data.get("name", "Unnamed Workflow"),
            description=data.get("description", ""),
            steps=steps,
            variables=data.get("variables", {}),
            triggers=data.get("triggers", []),
            metadata=data.get("metadata", {}),
        )

    async def execute(
        self,
        workflow: Workflow,
        context: Optional[Dict[str, Any]] = None,
        step_callback: Optional[Callable[[WorkflowStep], Awaitable[None]]] = None,
    ) -> Workflow:
        """Execute a workflow."""
        logger.info(f"Starting workflow: {workflow.name}")
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        
        ctx = context or {}
        ctx.update(workflow.variables)
        
        try:
            # Get execution order respecting dependencies
            execution_order = self._get_execution_order(workflow)
            
            for step in execution_order:
                # Check if step should be skipped
                if step.status == StepStatus.SKIPPED:
                    continue
                
                # Evaluate condition
                if step.condition and not self._evaluate_condition(step.condition, ctx):
                    step.status = StepStatus.SKIPPED
                    logger.info(f"Skipping step {step.name} - condition not met")
                    continue
                
                # Execute step with retries
                success = await self._execute_step_with_retry(step, ctx, step_callback)
                
                if not success:
                    workflow.status = WorkflowStatus.FAILED
                    break
                
                # Store result in context
                ctx[f"step_{step.id}_result"] = step.result
                
                if step_callback:
                    await step_callback(step)
            
            else:
                workflow.status = WorkflowStatus.COMPLETED
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.metadata["error"] = str(e)
        
        finally:
            workflow.completed_at = datetime.now()
            if workflow.id in self.running_workflows:
                del self.running_workflows[workflow.id]
        
        return workflow

    async def execute_async(
        self,
        workflow: Workflow,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute workflow asynchronously and return workflow ID."""
        task = asyncio.create_task(self.execute(workflow, context))
        self.running_workflows[workflow.id] = task
        return workflow.id

    def _get_execution_order(self, workflow: Workflow) -> List[WorkflowStep]:
        """Get steps in dependency-respecting order."""
        # Build dependency graph
        in_degree = {s.id: 0 for s in workflow.steps}
        dependents = {s.id: [] for s in workflow.steps}
        step_map = {s.id: s for s in workflow.steps}
        
        for step in workflow.steps:
            for dep in step.depends_on:
                if dep in dependents:
                    dependents[dep].append(step.id)
                    in_degree[step.id] += 1
        
        # Topological sort
        queue = [step_map[sid] for sid, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            step = queue.pop(0)
            result.append(step)
            
            for dep_id in dependents[step.id]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(step_map[dep_id])
        
        return result

    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        callback: Optional[Callable[[WorkflowStep], Awaitable[None]]],
    ) -> bool:
        """Execute a step with retry logic."""
        for attempt in range(step.max_retries + 1):
            step.retry_count = attempt
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now()
            
            try:
                if step.action not in self.actions:
                    raise ValueError(f"Unknown action: {step.action}")
                
                # Substitute variables in params
                params = self._substitute_variables(step.params, context)
                
                # Execute with timeout
                if step.timeout:
                    step.result = await asyncio.wait_for(
                        self.actions[step.action](params),
                        timeout=step.timeout,
                    )
                else:
                    step.result = await self.actions[step.action](params)
                
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()
                step.error = None
                
                if callback:
                    await callback(step)
                
                return True
                
            except Exception as e:
                step.error = str(e)
                logger.warning(f"Step {step.name} failed (attempt {attempt + 1}): {e}")
                
                if attempt < step.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    step.status = StepStatus.FAILED
                    step.completed_at = datetime.now()
                    if callback:
                        await callback(step)
                    return False
        
        return False

    def _substitute_variables(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Substitute variables in parameters."""
        result = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                result[key] = context.get(var_name, value)
            elif isinstance(value, dict):
                result[key] = self._substitute_variables(value, context)
            elif isinstance(value, list):
                result[key] = [
                    self._substitute_variables({"_": v}, context)["_"] if isinstance(v, dict) else
                    context.get(v[1:], v) if isinstance(v, str) and v.startswith("$") else v
                    for v in value
                ]
            else:
                result[key] = value
        
        return result

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        try:
            # Simple evaluation - replace variables and eval
            for key, value in context.items():
                if isinstance(value, (int, float, bool, str)):
                    condition = condition.replace(f"${key}", str(value))
            
            # Safe evaluation
            return eval(condition, {"__builtins__": {}}, {})
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False

    # Default action handlers
    async def _action_shell(self, params: Dict[str, Any]) -> Any:
        """Execute shell command."""
        import subprocess
        
        command = params.get("command", "")
        cwd = params.get("cwd")
        env = params.get("env")
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    async def _action_python(self, params: Dict[str, Any]) -> Any:
        """Execute Python code."""
        code = params.get("code", "")
        
        # Create safe execution environment
        local_vars = {}
        exec(code, {"__builtins__": {}}, local_vars)
        
        return local_vars.get("result")

    async def _action_delay(self, params: Dict[str, Any]) -> Any:
        """Delay execution."""
        seconds = params.get("seconds", 1)
        await asyncio.sleep(seconds)
        return {"delayed": seconds}

    async def _action_notify(self, params: Dict[str, Any]) -> Any:
        """Send notification."""
        message = params.get("message", "")
        level = params.get("level", "info")
        
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
        
        return {"notified": True, "message": message}

    async def _action_condition(self, params: Dict[str, Any]) -> Any:
        """Evaluate condition."""
        condition = params.get("condition", "")
        return {"result": bool(condition)}

    async def _action_parallel(self, params: Dict[str, Any]) -> Any:
        """Execute steps in parallel."""
        steps = params.get("steps", [])
        
        async def run_step(step_params):
            action = step_params.get("action", "shell")
            if action in self.actions:
                return await self.actions[action](step_params.get("params", {}))
            return None
        
        results = await asyncio.gather(*[run_step(s) for s in steps])
        return {"results": results}

    def get_running_workflows(self) -> List[str]:
        """Get list of running workflow IDs."""
        return list(self.running_workflows.keys())

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.running_workflows:
            task = self.running_workflows[workflow_id]
            task.cancel()
            del self.running_workflows[workflow_id]
            return True
        return False
