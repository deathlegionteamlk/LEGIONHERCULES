"""Plugin system with hot-reloading for LEGIONHERCULES."""

from __future__ import annotations

import os
import sys
import ast
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Callable, Type, Protocol
from datetime import datetime
from enum import Enum
import asyncio
import watchdog.observers
from watchdog.events import FileSystemEventHandler

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class PluginStatus(Enum):
    """Status of a plugin."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    DISABLED = "disabled"


class PluginInterface(Protocol):
    """Protocol that plugins must implement."""
    
    name: str
    version: str
    description: str
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        ...
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        ...
    
    def get_commands(self) -> Dict[str, Callable]:
        """Return commands provided by the plugin."""
        ...
    
    def get_hooks(self) -> Dict[str, List[Callable]]:
        """Return hooks provided by the plugin."""
        ...


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    description: str
    author: str = ""
    entry_point: str = ""
    dependencies: List[str] = field(default_factory=list)
    status: PluginStatus = PluginStatus.UNLOADED
    instance: Optional[Any] = None
    module: Optional[Any] = None
    loaded_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class PluginLoader:
    """Loads plugins from files."""

    @staticmethod
    def load_from_file(filepath: Path) -> Optional[PluginInfo]:
        """Load a plugin from a Python file."""
        logger.info(f"Loading plugin from: {filepath}")
        
        try:
            # Parse file for metadata
            source = filepath.read_text()
            tree = ast.parse(source)
            
            # Extract metadata from AST
            metadata = PluginLoader._extract_metadata(tree)
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                filepath.stem,
                filepath,
            )
            if not spec or not spec.loader:
                raise ImportError(f"Could not load spec from {filepath}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[filepath.stem] = module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = PluginLoader._find_plugin_class(module)
            if not plugin_class:
                raise ValueError(f"No plugin class found in {filepath}")
            
            # Create plugin info
            info = PluginInfo(
                name=metadata.get("name", filepath.stem),
                version=metadata.get("version", "0.1.0"),
                description=metadata.get("description", ""),
                author=metadata.get("author", ""),
                entry_point=str(filepath),
                dependencies=metadata.get("dependencies", []),
                module=module,
                metadata=metadata,
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to load plugin {filepath}: {e}")
            return None

    @staticmethod
    def _extract_metadata(tree: ast.AST) -> Dict[str, Any]:
        """Extract metadata from AST."""
        metadata = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in ["PLUGIN_NAME", "__plugin_name__"]:
                            if isinstance(node.value, ast.Constant):
                                metadata["name"] = node.value.value
                        elif target.id in ["PLUGIN_VERSION", "__version__"]:
                            if isinstance(node.value, ast.Constant):
                                metadata["version"] = node.value.value
                        elif target.id in ["PLUGIN_DESCRIPTION", "__doc__"]:
                            if isinstance(node.value, ast.Constant):
                                metadata["description"] = node.value.value
                        elif target.id == "PLUGIN_AUTHOR":
                            if isinstance(node.value, ast.Constant):
                                metadata["author"] = node.value.value
                        elif target.id == "PLUGIN_DEPENDENCIES":
                            if isinstance(node.value, ast.List):
                                metadata["dependencies"] = [
                                    elt.value for elt in node.value.elts
                                    if isinstance(elt, ast.Constant)
                                ]
        
        return metadata

    @staticmethod
    def _find_plugin_class(module: Any) -> Optional[Type]:
        """Find plugin class in module."""
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type):
                # Check if it looks like a plugin class
                if hasattr(obj, "name") and hasattr(obj, "version"):
                    return obj
        return None


class PluginManager:
    """Manages plugins with hot-reloading support."""

    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path.home() / ".legionhercules" / "plugins"
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, PluginInfo] = {}
        self.commands: Dict[str, Callable] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        
        self._observer: Optional[watchdog.observers.Observer] = None
        self._event_handler: Optional[PluginFileHandler] = None

    async def initialize(self) -> bool:
        """Initialize the plugin manager."""
        logger.info("Initializing plugin manager")
        
        # Load all plugins
        await self.discover_and_load()
        
        # Start file watcher
        self._start_watching()
        
        return True

    async def discover_and_load(self) -> List[PluginInfo]:
        """Discover and load all plugins."""
        loaded = []
        
        for filepath in self.plugins_dir.glob("*.py"):
            if filepath.name.startswith("_"):
                continue
            
            info = await self.load_plugin(filepath)
            if info:
                loaded.append(info)
        
        logger.info(f"Loaded {len(loaded)} plugins")
        return loaded

    async def load_plugin(self, filepath: Path) -> Optional[PluginInfo]:
        """Load a single plugin."""
        info = PluginLoader.load_from_file(filepath)
        if not info:
            return None
        
        # Check if already loaded
        if info.name in self.plugins:
            logger.warning(f"Plugin {info.name} already loaded, unloading first")
            await self.unload_plugin(info.name)
        
        info.status = PluginStatus.LOADING
        
        try:
            # Check dependencies
            for dep in info.dependencies:
                if dep not in self.plugins:
                    raise ImportError(f"Dependency {dep} not found")
            
            # Instantiate plugin
            plugin_class = PluginLoader._find_plugin_class(info.module)
            if plugin_class:
                info.instance = plugin_class()
                
                # Initialize
                if hasattr(info.instance, "initialize"):
                    success = await info.instance.initialize()
                    if not success:
                        raise RuntimeError("Plugin initialization failed")
                
                # Register commands
                if hasattr(info.instance, "get_commands"):
                    cmds = info.instance.get_commands()
                    for name, func in cmds.items():
                        self.commands[f"{info.name}.{name}"] = func
                
                # Register hooks
                if hasattr(info.instance, "get_hooks"):
                    hooks = info.instance.get_hooks()
                    for event, handlers in hooks.items():
                        if event not in self.hooks:
                            self.hooks[event] = []
                        self.hooks[event].extend(handlers)
            
            info.status = PluginStatus.LOADED
            info.loaded_at = datetime.now()
            self.plugins[info.name] = info
            
            logger.info(f"Plugin {info.name} v{info.version} loaded successfully")
            return info
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin {info.name}: {e}")
            info.status = PluginStatus.ERROR
            info.error_message = str(e)
            return info

    async def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name not in self.plugins:
            return False
        
        info = self.plugins[name]
        
        try:
            # Shutdown plugin
            if info.instance and hasattr(info.instance, "shutdown"):
                await info.instance.shutdown()
            
            # Unregister commands
            cmds_to_remove = [k for k in self.commands if k.startswith(f"{name}.")]
            for cmd in cmds_to_remove:
                del self.commands[cmd]
            
            # Unregister hooks
            for event, handlers in self.hooks.items():
                self.hooks[event] = [
                    h for h in handlers 
                    if not (hasattr(h, "__self__") and h.__self__ is info.instance)
                ]
            
            # Remove from loaded plugins
            del self.plugins[name]
            
            logger.info(f"Plugin {name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {name}: {e}")
            return False

    async def reload_plugin(self, name: str) -> Optional[PluginInfo]:
        """Reload a plugin."""
        if name not in self.plugins:
            return None
        
        info = self.plugins[name]
        filepath = Path(info.entry_point)
        
        await self.unload_plugin(name)
        return await self.load_plugin(filepath)

    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        return self.plugins.get(name)

    def get_loaded_plugins(self) -> List[PluginInfo]:
        """Get all loaded plugins."""
        return [
            info for info in self.plugins.values()
            if info.status == PluginStatus.LOADED
        ]

    async def execute_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Execute all handlers for a hook."""
        results = []
        
        if event in self.hooks:
            for handler in self.hooks[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(*args, **kwargs)
                    else:
                        result = handler(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook handler error for {event}: {e}")
        
        return results

    def _start_watching(self) -> None:
        """Start watching plugin files for changes."""
        try:
            self._event_handler = PluginFileHandler(self)
            self._observer = watchdog.observers.Observer()
            self._observer.schedule(
                self._event_handler,
                str(self.plugins_dir),
                recursive=False,
            )
            self._observer.start()
            logger.info(f"Started watching plugin directory: {self.plugins_dir}")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")

    def stop_watching(self) -> None:
        """Stop watching plugin files."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching plugin directory")

    async def shutdown(self) -> None:
        """Shutdown plugin manager."""
        self.stop_watching()
        
        # Unload all plugins
        for name in list(self.plugins.keys()):
            await self.unload_plugin(name)
        
        logger.info("Plugin manager shutdown complete")


class PluginFileHandler(FileSystemEventHandler):
    """Handles file system events for hot-reloading."""

    def __init__(self, manager: PluginManager):
        self.manager = manager
        self._debounce_timers: Dict[str, Any] = {}

    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(".py"):
            self._handle_change(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(".py"):
            self._handle_change(event.src_path)

    def _handle_change(self, filepath: str):
        """Handle file change with debouncing."""
        import threading
        
        # Cancel existing timer
        if filepath in self._debounce_timers:
            self._debounce_timers[filepath].cancel()
        
        # Create new timer
        timer = threading.Timer(1.0, self._reload_plugin, args=[filepath])
        self._debounce_timers[filepath] = timer
        timer.start()

    def _reload_plugin(self, filepath: str):
        """Reload the plugin."""
        path = Path(filepath)
        
        # Find plugin by filepath
        for name, info in self.manager.plugins.items():
            if info.entry_point == filepath:
                logger.info(f"Hot-reloading plugin: {name}")
                asyncio.create_task(self.manager.reload_plugin(name))
                break
        else:
            # New plugin
            logger.info(f"Loading new plugin: {path.name}")
            asyncio.create_task(self.manager.load_plugin(path))
