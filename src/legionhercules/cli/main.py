"""Main CLI entry point for LEGIONHERCULES."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from legionhercules import __version__
from legionhercules.cli.interactive import InteractiveSession
from legionhercules.config.manager import ConfigManager
from legionhercules.utils.logging import setup_logging

# Create Typer app
app = typer.Typer(
    name="legionhercules",
    help="LEGIONHERCULES - Autonomous CLI framework with parallel agent execution",
    add_completion=True,
)

console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"[bold cyan]LEGIONHERCULES[/bold cyan] version {__version__}")
        console.print("Developed by Death Legion Team Coders Demo X HEXA")
        raise typer.Exit()


def print_banner() -> None:
    """Print the LEGIONHERCULES banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██╗     ███████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗        ║
    ║   ██║     ██╔════╝██╔════╝ ██║██╔═══██╗████╗  ██║        ║
    ║   ██║     █████╗  ██║  ███╗██║██║   ██║██╔██╗ ██║        ║
    ║   ██║     ██╔══╝  ██║   ██║██║██║   ██║██║╚██╗██║        ║
    ║   ███████╗███████╗╚██████╔╝██║╚██████╔╝██║ ╚████║        ║
    ║   ╚══════╝╚══════╝ ╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═══╝        ║
    ║                                                           ║
    ║   ██╗  ██╗███████╗██████╗  ██████╗██╗   ██╗██╗     ███████╗███████╗ ║
    ║   ██║  ██║██╔════╝██╔══██╗██╔════╝██║   ██║██║     ██╔════╝██╔════╝ ║
    ║   ███████║█████╗  ██████╔╝██║     ██║   ██║██║     █████╗  ███████╗ ║
    ║   ██╔══██║██╔══╝  ██╔══██╗██║     ██║   ██║██║     ██╔══╝  ╚════██║ ║
    ║   ██║  ██║███████╗██║  ██║╚██████╗╚██████╔╝███████╗███████╗███████║ ║
    ║   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝ ║
    ║                                                           ║
    ║   [cyan]Autonomous CLI Framework with Parallel Agent Execution[/cyan]     ║
    ║   [dim]Completely Free - Powered by Local Ollama LLMs[/dim]         ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner)


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to config file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V",
        help="Enable verbose logging"
    ),
) -> None:
    """LEGIONHERCULES CLI framework."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)


@app.command()
def chat(
    message: Optional[str] = typer.Argument(
        None,
        help="Message to send (if not provided, enters interactive mode)"
    ),
    model: str = typer.Option(
        "llama3.2",
        "--model", "-m",
        help="Ollama model to use"
    ),
    agent: str = typer.Option(
        "default",
        "--agent", "-a",
        help="Agent configuration to use"
    ),
    no_tools: bool = typer.Option(
        False,
        "--no-tools",
        help="Disable tool usage"
    ),
) -> None:
    """Start a chat session with an agent."""
    print_banner()
    
    asyncio.run(_chat_async(message, model, agent, no_tools))


async def _chat_async(
    message: Optional[str],
    model: str,
    agent_name: str,
    no_tools: bool
) -> None:
    """Async chat handler."""
    session = InteractiveSession(
        model=model,
        agent_name=agent_name,
        use_tools=not no_tools,
    )
    
    await session.initialize()
    
    if message:
        # Single message mode
        response = await session.send_message(message)
        console.print(Panel(response, title="Assistant", border_style="green"))
    else:
        # Interactive mode
        await session.run_interactive()


@app.command()
def agents(
    list_all: bool = typer.Option(
        False, "--list", "-l",
        help="List all configured agents"
    ),
    create: Optional[str] = typer.Option(
        None, "--create",
        help="Create a new agent with given name"
    ),
) -> None:
    """Manage agents."""
    config = ConfigManager()
    
    if list_all:
        agents_list = config.list_agents()
        if agents_list:
            console.print("[bold]Configured Agents:[/bold]")
            for name in agents_list:
                console.print(f"  • {name}")
        else:
            console.print("[dim]No agents configured[/dim]")
    
    elif create:
        console.print(f"Creating agent: {create}")
        # TODO: Implement agent creation wizard
        console.print("[yellow]Agent creation wizard coming soon![/yellow]")
    
    else:
        console.print("Use --list to see agents or --create to create one")


@app.command()
def tools(
    list_all: bool = typer.Option(
        False, "--list", "-l",
        help="List all available tools"
    ),
) -> None:
    """Manage tools."""
    from legionhercules.tools.base import ToolRegistry
    
    registry = ToolRegistry.create_default_registry()
    
    if list_all:
        tool_names = registry.list_tools()
        console.print("[bold]Available Tools:[/bold]")
        for name in tool_names:
            tool = registry.get_tool(name)
            console.print(f"  • [cyan]{name}[/cyan]: {tool.description}")
    else:
        console.print("Use --list to see available tools")


@app.command()
def config(
    show: bool = typer.Option(
        False, "--show",
        help="Show current configuration"
    ),
    init: bool = typer.Option(
        False, "--init",
        help="Initialize configuration file"
    ),
) -> None:
    """Manage configuration."""
    config_manager = ConfigManager()
    
    if init:
        config_path = config_manager.init_config()
        console.print(f"[green]Configuration initialized at:[/green] {config_path}")
    
    elif show:
        settings = config_manager.load_config()
        console.print("[bold]Current Configuration:[/bold]")
        console.print(settings.model_dump_json(indent=2))
    
    else:
        console.print("Use --init to create config or --show to view current")


@app.command()
def models(
    list_all: bool = typer.Option(
        False, "--list", "-l",
        help="List available Ollama models"
    ),
    pull: Optional[str] = typer.Option(
        None, "--pull",
        help="Pull a model from Ollama"
    ),
) -> None:
    """Manage Ollama models."""
    from legionhercules.llm.ollama_provider import OllamaProvider
    
    provider = OllamaProvider()
    
    if list_all:
        asyncio.run(_list_models(provider))
    
    elif pull:
        asyncio.run(_pull_model(provider, pull))
    
    else:
        console.print("Use --list to see models or --pull <model> to download")


async def _list_models(provider: OllamaProvider) -> None:
    """List available models."""
    await provider.initialize()
    
    if not await provider.health_check():
        console.print("[red]Ollama is not running![/red]")
        console.print("Please install and start Ollama: https://ollama.com")
        return
    
    models_list = await provider.list_models()
    
    if models_list:
        console.print("[bold]Available Models:[/bold]")
        for model in models_list:
            console.print(f"  • {model}")
    else:
        console.print("[dim]No models found. Use --pull <model> to download one.[/dim]")
        console.print("\nRecommended models:")
        console.print("  • llama3.2 - Fast, efficient general purpose")
        console.print("  • llama3.1:8b - Good balance of speed and quality")
        console.print("  • codellama - Code-focused tasks")
        console.print("  • mistral - Strong performance")


async def _pull_model(provider: OllamaProvider, model_name: str) -> None:
    """Pull a model from Ollama."""
    await provider.initialize()
    
    if not await provider.health_check():
        console.print("[red]Ollama is not running![/red]")
        return
    
    console.print(f"Pulling model: {model_name}...")
    success = await provider.pull_model(model_name)
    
    if success:
        console.print(f"[green]Successfully pulled {model_name}![/green]")
    else:
        console.print(f"[red]Failed to pull {model_name}[/red]")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
