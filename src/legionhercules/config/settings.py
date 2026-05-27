"""Settings and configuration models for LEGIONHERCULES."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    """LLM provider settings."""
    
    provider: str = Field(default="ollama", description="LLM provider (ollama)")
    model: str = Field(default="llama3.2", description="Default model to use")
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: float = Field(default=120.0, gt=0)


class AgentSettings(BaseModel):
    """Agent configuration settings."""
    
    name: str = Field(default="default", description="Agent name")
    description: str = Field(default="", description="Agent description")
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System prompt for the agent"
    )
    max_iterations: int = Field(default=10, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    tools: list[str] = Field(default_factory=list, description="Enabled tools")


class OrchestratorSettings(BaseModel):
    """Orchestrator settings."""
    
    max_concurrent_tasks: int = Field(default=5, ge=1, le=20)
    task_timeout: float = Field(default=300.0, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    enable_parallel: bool = Field(default=True)


class LoggingSettings(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file: Optional[str] = Field(default=None, description="Log file path")
    use_rich: bool = Field(default=True, description="Use Rich for console logging")


class Settings(BaseModel):
    """Main settings model for LEGIONHERCULES."""
    
    # Version
    version: str = Field(default="0.1.0")
    
    # LLM settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    
    # Orchestrator settings
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    
    # Logging settings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Agents
    agents: list[AgentSettings] = Field(default_factory=list)
    
    # CLI settings
    cli: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
    
    def get_agent(self, name: str) -> Optional[AgentSettings]:
        """Get agent settings by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def add_agent(self, agent: AgentSettings) -> None:
        """Add or update agent settings."""
        # Remove existing agent with same name
        self.agents = [a for a in self.agents if a.name != agent.name]
        self.agents.append(agent)
    
    def remove_agent(self, name: str) -> bool:
        """Remove agent by name."""
        original_len = len(self.agents)
        self.agents = [a for a in self.agents if a.name != name]
        return len(self.agents) < original_len
    
    def to_yaml(self) -> str:
        """Convert settings to YAML string."""
        import yaml
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> Settings:
        """Load settings from YAML string."""
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls(**data)
    
    @classmethod
    def default(cls) -> Settings:
        """Create default settings."""
        return cls(
            llm=LLMSettings(),
            orchestrator=OrchestratorSettings(),
            logging=LoggingSettings(),
            agents=[
                AgentSettings(
                    name="default",
                    description="Default general-purpose agent",
                    system_prompt="You are a helpful AI assistant.",
                    tools=["file_read", "file_write", "file_edit", "bash", "web_search"],
                ),
                AgentSettings(
                    name="coder",
                    description="Code-focused agent",
                    system_prompt="""You are an expert programmer. You help write, review, and debug code.
You have access to file operations and bash commands to work with codebases.
Be concise and provide working code examples.""",
                    tools=["file_read", "file_write", "file_edit", "bash"],
                ),
                AgentSettings(
                    name="researcher",
                    description="Research-focused agent with web search",
                    system_prompt="""You are a research assistant. You help find information and answer questions.
You have access to web search to find current information.
Provide well-sourced answers with citations.""",
                    tools=["web_search", "file_read", "file_write"],
                ),
            ]
        )
