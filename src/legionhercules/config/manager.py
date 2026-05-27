"""Configuration manager for LEGIONHERCULES."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from legionhercules.config.settings import Settings
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Manages LEGIONHERCULES configuration."""
    
    CONFIG_DIR_NAME = ".legionhercules"
    CONFIG_FILE_NAME = "config.yaml"
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._settings: Optional[Settings] = None
    
    @property
    def config_dir(self) -> Path:
        """Get configuration directory."""
        if self.config_path:
            return self.config_path.parent
        
        # Check for project-local config first
        cwd = Path.cwd()
        local_config = cwd / self.CONFIG_DIR_NAME
        if local_config.exists():
            return local_config
        
        # Fall back to user home
        home = Path.home()
        return home / self.CONFIG_DIR_NAME
    
    @property
    def config_file(self) -> Path:
        """Get configuration file path."""
        if self.config_path:
            return self.config_path
        return self.config_dir / self.CONFIG_FILE_NAME
    
    def load_config(self) -> Settings:
        """Load configuration from file."""
        if self._settings:
            return self._settings
        
        if self.config_file.exists():
            try:
                logger.debug(f"Loading config from {self.config_file}")
                content = self.config_file.read_text()
                self._settings = Settings.from_yaml(content)
                return self._settings
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        # Return default settings
        self._settings = Settings.default()
        return self._settings
    
    def save_config(self, settings: Optional[Settings] = None) -> None:
        """Save configuration to file."""
        settings = settings or self._settings or Settings.default()
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config
        self.config_file.write_text(settings.to_yaml())
        logger.info(f"Configuration saved to {self.config_file}")
    
    def init_config(self) -> Path:
        """Initialize configuration file with defaults."""
        settings = Settings.default()
        self.save_config(settings)
        return self.config_file
    
    def get_settings(self) -> Settings:
        """Get current settings."""
        return self.load_config()
    
    def update_settings(self, settings: Settings) -> None:
        """Update and save settings."""
        self._settings = settings
        self.save_config(settings)
    
    def list_agents(self) -> list[str]:
        """List configured agent names."""
        settings = self.load_config()
        return [agent.name for agent in settings.agents]
    
    def get_agent_config(self, name: str) -> Optional[dict]:
        """Get agent configuration by name."""
        settings = self.load_config()
        agent = settings.get_agent(name)
        return agent.model_dump() if agent else None
    
    def add_agent(self, name: str, description: str = "", system_prompt: str = "", tools: Optional[list] = None) -> None:
        """Add a new agent configuration."""
        from legionhercules.config.settings import AgentSettings
        
        settings = self.load_config()
        agent = AgentSettings(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools or [],
        )
        settings.add_agent(agent)
        self.save_config(settings)
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent configuration."""
        settings = self.load_config()
        success = settings.remove_agent(name)
        if success:
            self.save_config(settings)
        return success
    
    def get_env_override(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value from environment variable."""
        env_key = f"LEGIONHERCULES_{key.upper()}"
        return os.environ.get(env_key, default)
    
    def config_exists(self) -> bool:
        """Check if configuration file exists."""
        return self.config_file.exists()


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get or create global config manager instance."""
    global _config_manager
    if _config_manager is None or config_path:
        _config_manager = ConfigManager(config_path)
    return _config_manager
