"""Checkpoint system for autonomous development."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentState:
    """Snapshot of agent state."""
    agent_id: str
    agent_name: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """A checkpoint for saving and restoring state."""
    id: str
    name: str
    timestamp: str
    agent_states: list[AgentState] = field(default_factory=list)
    file_snapshots: dict[str, str] = field(default_factory=dict)
    git_commit: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        name: str,
        agent_states: list[AgentState],
        file_snapshots: Optional[dict[str, str]] = None,
        git_commit: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Checkpoint:
        """Create a new checkpoint."""
        return cls(
            id=str(uuid4()),
            name=name,
            timestamp=datetime.utcnow().isoformat(),
            agent_states=agent_states,
            file_snapshots=file_snapshots or {},
            git_commit=git_commit,
            metadata=metadata or {},
        )


class CheckpointManager:
    """Manages checkpoints for autonomous development."""
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints (default: ~/.legionhercules/checkpoints)
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path.home() / ".legionhercules" / "checkpoints"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: dict[str, Checkpoint] = {}
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from disk."""
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    checkpoint = Checkpoint(**data)
                    self._checkpoints[checkpoint.id] = checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
    
    def save_checkpoint(
        self,
        session_id: str,
        conversation_history: list[dict],
        file_operations: list,
        agent_state: dict,
        metadata: Optional[dict] = None,
    ) -> Checkpoint:
        """Save a checkpoint (simplified interface for basic usage).
        
        Args:
            session_id: Session identifier
            conversation_history: List of conversation messages
            file_operations: List of file operations performed
            agent_state: Current agent state
            metadata: Additional metadata
            
        Returns:
            Created checkpoint
        """
        agent_states = [AgentState(
            agent_id=session_id,
            agent_name="default",
            conversation_history=conversation_history,
            metadata=agent_state,
        )]
        
        checkpoint = Checkpoint.create(
            name=f"checkpoint-{session_id}",
            agent_states=agent_states,
            metadata=metadata or {},
        )
        
        self._save_checkpoint(checkpoint)
        self._checkpoints[checkpoint.id] = checkpoint
        logger.info(f"Saved checkpoint: {checkpoint.id}")
        return checkpoint

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            
        Returns:
            Checkpoint if found, None otherwise
        """
        return self._checkpoints.get(checkpoint_id)

    def create_checkpoint(
        self,
        name: str,
        orchestrator: Any,
        files_to_snapshot: Optional[list[str]] = None,
        include_git: bool = True,
        metadata: Optional[dict] = None,
    ) -> Checkpoint:
        """Create a new checkpoint.
        
        Args:
            name: Checkpoint name
            orchestrator: AgentOrchestrator instance
            files_to_snapshot: List of file paths to snapshot
            include_git: Whether to include git commit info
            metadata: Additional metadata
        
        Returns:
            Created checkpoint
        """
        # Capture agent states
        agent_states = []
        if hasattr(orchestrator, 'agents'):
            for agent_id, agent in orchestrator.agents.items():
                state = AgentState(
                    agent_id=agent_id,
                    agent_name=agent.config.name,
                    conversation_history=[
                        {
                            "role": msg.role.value,
                            "content": msg.content,
                            "tool_calls": msg.tool_calls,
                        }
                        for msg in agent.messages
                    ],
                    metadata={
                        "config": {
                            "name": agent.config.name,
                            "description": agent.config.description,
                            "model": agent.config.model,
                            "temperature": agent.config.temperature,
                            "max_tokens": agent.config.max_tokens,
                            "system_prompt": agent.config.system_prompt,
                            "tools": agent.config.tools,
                            "max_iterations": agent.config.max_iterations,
                            "timeout_seconds": agent.config.timeout_seconds,
                        }
                    }
                )
                agent_states.append(state)
        
        # Capture file snapshots
        file_snapshots = {}
        if files_to_snapshot:
            for file_path in files_to_snapshot:
                try:
                    with open(file_path, 'r') as f:
                        file_snapshots[file_path] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to snapshot {file_path}: {e}")
        
        # Get git commit
        git_commit = None
        if include_git:
            git_commit = self._get_git_commit()
        
        # Create checkpoint
        checkpoint = Checkpoint.create(
            name=name,
            agent_states=agent_states,
            file_snapshots=file_snapshots,
            git_commit=git_commit,
            metadata=metadata or {},
        )
        
        # Save to disk
        self._save_checkpoint(checkpoint)
        self._checkpoints[checkpoint.id] = checkpoint
        
        logger.info(f"Created checkpoint: {name} (id: {checkpoint.id})")
        return checkpoint
    
    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint.id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    def restore_checkpoint(
        self,
        checkpoint_id: str,
        orchestrator: Any,
        restore_files: bool = True,
    ) -> bool:
        """Restore from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            orchestrator: AgentOrchestrator instance to restore
            restore_files: Whether to restore file snapshots
        
        Returns:
            True if successful
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
        
        try:
            # Restore agent states
            if hasattr(orchestrator, 'agents'):
                orchestrator.agents.clear()
                
                for state in checkpoint.agent_states:
                    # Recreate agent config
                    from legionhercules.core.agent import AgentConfig
                    config_data = state.metadata.get("config", {})
                    config = AgentConfig(**config_data)
                    
                    # Create agent
                    asyncio.create_task(orchestrator.create_agent(config.name))
                    # Note: In practice, this would need proper async handling
            
            # Restore files
            if restore_files and checkpoint.file_snapshots:
                for file_path, content in checkpoint.file_snapshots.items():
                    try:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(file_path, 'w') as f:
                            f.write(content)
                    except Exception as e:
                        logger.warning(f"Failed to restore {file_path}: {e}")
            
            logger.info(f"Restored checkpoint: {checkpoint.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints."""
        return [
            {
                "id": cp.id,
                "name": cp.name,
                "timestamp": cp.timestamp,
                "agent_count": len(cp.agent_states),
                "file_count": len(cp.file_snapshots),
                "git_commit": cp.git_commit,
            }
            for cp in sorted(
                self._checkpoints.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
        ]
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoint = self._checkpoints.pop(checkpoint_id, None)
        if checkpoint:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            try:
                checkpoint_file.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return True
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint file: {e}")
        return False
    
    def export_checkpoint(self, checkpoint_id: str, export_path: str) -> bool:
        """Export checkpoint to a file."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False
        
        try:
            with open(export_path, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export checkpoint: {e}")
            return False
    
    def import_checkpoint(self, import_path: str) -> Optional[Checkpoint]:
        """Import checkpoint from a file."""
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
                checkpoint = Checkpoint(**data)
                self._checkpoints[checkpoint.id] = checkpoint
                self._save_checkpoint(checkpoint)
                return checkpoint
        except Exception as e:
            logger.error(f"Failed to import checkpoint: {e}")
            return None
