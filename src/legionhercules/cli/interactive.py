"""Interactive session for LEGIONHERCULES CLI."""

from __future__ import annotations

import asyncio
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.status import Status

from legionhercules.core.agent import Agent, AgentConfig
from legionhercules.core.orchestrator import AgentOrchestrator
from legionhercules.core.task import Task
from legionhercules.llm.ollama_provider import OllamaProvider
from legionhercules.tools.base import ToolRegistry
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class InteractiveSession:
    """Interactive chat session with an agent."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        agent_name: str = "default",
        use_tools: bool = True,
    ):
        self.model = model
        self.agent_name = agent_name
        self.use_tools = use_tools
        self.console = Console()
        
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.agent: Optional[Agent] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the interactive session."""
        if self._initialized:
            return
        
        # Create LLM provider
        llm = OllamaProvider(model=self.model)
        
        # Create tool registry
        tools = ToolRegistry.create_default_registry() if self.use_tools else ToolRegistry()
        
        # Create orchestrator
        self.orchestrator = AgentOrchestrator(
            llm_provider=llm,
            tool_registry=tools,
            max_concurrent_tasks=3,
        )
        
        await self.orchestrator.initialize()
        
        # Register and create agent
        config = AgentConfig(
            name=self.agent_name,
            description="Interactive chat agent",
            model=self.model,
            system_prompt=self._get_system_prompt(),
            tools=tools.list_tools() if self.use_tools else [],
        )
        
        self.orchestrator.register_agent(config)
        self.agent = await self.orchestrator.create_agent(self.agent_name)
        
        self._initialized = True
        logger.info(f"Interactive session initialized (model: {self.model})")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are LEGIONHERCULES, an autonomous AI assistant.

You can help users with various tasks including:
- Reading and writing files
- Executing bash commands
- Searching the web
- Answering questions
- Writing code
- Analyzing data

When using tools:
1. Think about what tool is needed
2. Use the tool with proper parameters
3. Interpret the results
4. Provide helpful responses

Be concise but thorough in your responses."""
    
    async def run_interactive(self) -> None:
        """Run the interactive chat loop."""
        self.console.print()
        self.console.print(Panel(
            "[bold green]Interactive Mode[/bold green]\n"
            "Type your messages below. Commands:\n"
            "  [cyan]/quit[/cyan] or [cyan]/exit[/cyan] - Exit the session\n"
            "  [cyan]/clear[/cyan] - Clear conversation history\n"
            "  [cyan]/tools[/cyan] - Toggle tool usage\n"
            "  [cyan]/help[/cyan] - Show help",
            title="LEGIONHERCULES",
            border_style="green"
        ))
        self.console.print()
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                
                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    self.console.print("[dim]Goodbye![/dim]")
                    break
                
                elif user_input.lower() == '/clear':
                    if self.agent:
                        self.agent.reset()
                    self.console.print("[dim]Conversation history cleared.[/dim]")
                    continue
                
                elif user_input.lower() == '/tools':
                    self.use_tools = not self.use_tools
                    status = "enabled" if self.use_tools else "disabled"
                    self.console.print(f"[dim]Tools {status}.[/dim]")
                    continue
                
                elif user_input.lower() == '/help':
                    self._show_help()
                    continue
                
                elif not user_input.strip():
                    continue
                
                # Process message
                await self._process_message(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /quit to exit[/dim]")
                continue
            except EOFError:
                break
    
    async def _process_message(self, message: str) -> None:
        """Process a user message."""
        if not self.agent:
            self.console.print("[red]Agent not initialized![/red]")
            return
        
        # Show thinking indicator
        with Status("[bold green]Thinking...[/bold green]", spinner="dots"):
            try:
                # Create task for the message
                task = Task(
                    description=message,
                    context={"agent_name": self.agent_name}
                )
                
                # Execute task
                result = await self.agent.execute_task(task)
                
                if result.success:
                    # Display response
                    output = result.output or "No response"
                    self.console.print(Panel(
                        Markdown(output),
                        title="[bold green]Assistant[/bold green]",
                        border_style="green"
                    ))
                    
                    # Show metadata if verbose
                    if result.metadata:
                        duration = result.metadata.get('duration_ms', 0)
                        iterations = result.metadata.get('iterations', 0)
                        if duration or iterations:
                            self.console.print(
                                f"[dim]Duration: {duration:.0f}ms"
                                f"{f', Iterations: {iterations}' if iterations else ''}[/dim]",
                                style="dim"
                            )
                else:
                    self.console.print(Panel(
                        f"[red]Error:[/red] {result.error or 'Unknown error'}",
                        title="[bold red]Error[/bold red]",
                        border_style="red"
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.console.print(f"[red]Error: {e}[/red]")
    
    async def send_message(self, message: str) -> str:
        """Send a single message and return the response."""
        if not self.agent:
            await self.initialize()
        
        task = Task(
            description=message,
            context={"agent_name": self.agent_name}
        )
        
        result = await self.agent.execute_task(task)
        
        if result.success:
            return str(result.output or "")
        else:
            return f"Error: {result.error or 'Unknown error'}"
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
# LEGIONHERCULES Commands

## Interactive Commands
- `/quit` or `/exit` - Exit the session
- `/clear` - Clear conversation history
- `/tools` - Toggle tool usage on/off
- `/help` - Show this help

## Available Tools
When tools are enabled, you can:
- **file_read** - Read file contents
- **file_write** - Write to files
- **file_edit** - Edit files with search/replace
- **bash** - Execute bash commands
- **web_search** - Search the web

## Tips
- Be specific in your requests
- The agent can chain multiple tools
- Use `/clear` if the conversation gets too long
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="blue"))
