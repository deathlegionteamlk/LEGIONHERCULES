"""Memory and context management with vector storage."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from legionhercules.utils.logging import get_logger
from legionhercules.memory.embeddings import EmbeddingProvider, LocalEmbeddingProvider
from legionhercules.memory.vector_store import VectorStore, ChromaVectorStore, InMemoryVectorStore

logger = get_logger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0  # 0.0 to 2.0, higher = more important
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SessionContext:
    """Context for a conversation session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    working_memory: List[MemoryEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.conversation_history.append(turn)
        self.last_activity = datetime.now()

    def get_recent_history(self, n: int = 10) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        return self.conversation_history[-n:] if n < len(self.conversation_history) else self.conversation_history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": [t.to_dict() for t in self.conversation_history],
            "working_memory": [m.to_dict() for m in self.working_memory],
            "metadata": self.metadata,
        }


class TokenCounter:
    """Count tokens in text for context window management."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self._encoder = None

        if HAS_TIKTOKEN:
            try:
                # Try to get encoder for the model
                if "gpt-4" in model or "gpt-3.5" in model:
                    self._encoder = tiktoken.encoding_for_model(model)
                else:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoder: {e}")

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            # Add tokens for message format
            total += 4  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            if "role" in msg:
                total += self.count(msg["role"])
            if "content" in msg:
                total += self.count(msg["content"])
            if "name" in msg:
                total += self.count(msg["name"])

        total += 2  # Every reply is primed with <|start|>assistant<|message|>
        return total


class MemoryManager:
    """Manages memory and context with vector storage."""

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        persist_directory: Optional[str] = None,
        max_context_tokens: int = 8000,
        context_window_ratio: float = 0.75,  # Use 75% for history, 25% for retrieved memories
    ):
        self.embedding_provider = embedding_provider or LocalEmbeddingProvider()
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=persist_directory
        )
        self.token_counter = TokenCounter()
        self.max_context_tokens = max_context_tokens
        self.context_window_ratio = context_window_ratio

        # Session management
        self.sessions: Dict[str, SessionContext] = {}
        self.current_session_id: Optional[str] = None

        # Memory policies
        self.retention_days = 30
        self.min_importance_threshold = 0.5
        self.max_memories_per_query = 10

        # Persistence
        self._memory_cache: Dict[str, MemoryEntry] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory manager."""
        if self._initialized:
            return

        # Initialize vector store
        if hasattr(self.vector_store, 'initialize'):
            await self.vector_store.initialize()

        self._initialized = True
        logger.info("MemoryManager initialized")

    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        session_id: Optional[str] = None,
    ) -> str:
        """Store a memory with embedding generation.

        Args:
            content: The text content to store
            metadata: Optional metadata dictionary
            importance: Importance score (0.0 to 2.0)
            session_id: Optional session ID for organization

        Returns:
            The memory ID
        """
        await self.initialize()

        # Generate embedding
        embedding = await self.embedding_provider.embed(content)

        # Create memory entry
        memory_id = self._generate_id(content)
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance,
        )

        # Add session info to metadata
        if session_id:
            entry.metadata["session_id"] = session_id
        elif self.current_session_id:
            entry.metadata["session_id"] = self.current_session_id

        # Store in vector database
        await self.vector_store.add(
            texts=[content],
            embeddings=[embedding],
            metadata=[entry.metadata],
            ids=[memory_id],
        )

        # Cache in memory
        self._memory_cache[memory_id] = entry

        logger.debug(f"Stored memory: {memory_id[:8]}...")
        return memory_id

    async def store_memories_batch(
        self,
        contents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        importance: Optional[List[float]] = None,
    ) -> List[str]:
        """Store multiple memories in batch.

        Args:
            contents: List of text contents to store
            metadata: Optional list of metadata dictionaries
            importance: Optional list of importance scores

        Returns:
            List of memory IDs
        """
        await self.initialize()

        if not contents:
            return []

        # Generate embeddings in batch
        embeddings = await self.embedding_provider.embed_batch(contents)

        # Create entries
        memory_ids = []
        entries = []

        for i, content in enumerate(contents):
            memory_id = self._generate_id(content)
            memory_ids.append(memory_id)

            entry = MemoryEntry(
                id=memory_id,
                content=content,
                embedding=embeddings[i],
                metadata=(metadata[i] if metadata and i < len(metadata) else {}),
                importance=(importance[i] if importance and i < len(importance) else 1.0),
            )

            if self.current_session_id:
                entry.metadata["session_id"] = self.current_session_id

            entries.append(entry)
            self._memory_cache[memory_id] = entry

        # Store in vector database
        await self.vector_store.add(
            texts=contents,
            embeddings=embeddings,
            metadata=[e.metadata for e in entries],
            ids=memory_ids,
        )

        logger.debug(f"Stored {len(memory_ids)} memories in batch")
        return memory_ids

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories using semantic search.

        Args:
            query: The search query
            top_k: Number of results to return
            min_relevance: Minimum relevance score (0.0 to 1.0)
            filter_dict: Optional filter for metadata
            session_id: Optional session ID to filter by

        Returns:
            List of memory results with relevance scores
        """
        await self.initialize()

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Add session filter if provided
        search_filter = filter_dict or {}
        if session_id:
            search_filter["session_id"] = session_id
        elif self.current_session_id:
            search_filter["session_id"] = self.current_session_id

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more to filter by relevance
            filter_dict=search_filter if search_filter else None,
        )

        # Filter by relevance and update access stats
        filtered_results = []
        for result in results:
            if result["score"] >= min_relevance:
                memory_id = result["id"]

                # Update access stats if in cache
                if memory_id in self._memory_cache:
                    entry = self._memory_cache[memory_id]
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    result["access_count"] = entry.access_count

                filtered_results.append(result)

        return filtered_results[:top_k]

    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        # Check cache first
        if memory_id in self._memory_cache:
            return self._memory_cache[memory_id]

        # TODO: Implement retrieval from vector store by ID
        return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        await self.initialize()

        # Remove from cache
        self._memory_cache.pop(memory_id, None)

        # Remove from vector store
        return await self.vector_store.delete([memory_id])

    async def clear_memories(self, session_id: Optional[str] = None) -> bool:
        """Clear memories, optionally filtered by session."""
        await self.initialize()

        if session_id:
            # Clear only session-specific memories
            ids_to_delete = [
                mid for mid, entry in self._memory_cache.items()
                if entry.metadata.get("session_id") == session_id
            ]
            for mid in ids_to_delete:
                self._memory_cache.pop(mid, None)
            if ids_to_delete:
                await self.vector_store.delete(ids_to_delete)
        else:
            # Clear all
            self._memory_cache.clear()
            await self.vector_store.clear()

        return True

    # Session Management

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        if session_id is None:
            session_id = self._generate_session_id()

        session = SessionContext(session_id=session_id)
        self.sessions[session_id] = session
        self.current_session_id = session_id

        logger.info(f"Created session: {session_id}")
        return session_id

    def set_session(self, session_id: str) -> bool:
        """Set the current session."""
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False

    def get_session(self, session_id: Optional[str] = None) -> Optional[SessionContext]:
        """Get a session by ID or current session."""
        sid = session_id or self.current_session_id
        return self.sessions.get(sid) if sid else None

    def add_conversation_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a turn to the current session's conversation history."""
        session = self.get_session()
        if session:
            session.add_turn(role, content, metadata)

    def get_conversation_history(
        self,
        n: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[ConversationTurn]:
        """Get conversation history for a session."""
        session = self.get_session(session_id)
        if not session:
            return []

        history = session.conversation_history
        if n:
            return history[-n:]
        return history

    # Context Window Optimization

    def optimize_context(
        self,
        conversation_history: List[ConversationTurn],
        retrieved_memories: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> Tuple[List[ConversationTurn], List[Dict[str, Any]]]:
        """Optimize context window by selecting most relevant content.

        Args:
            conversation_history: Full conversation history
            retrieved_memories: Retrieved relevant memories
            max_tokens: Maximum tokens allowed

        Returns:
            Tuple of (optimized_history, optimized_memories)
        """
        max_tokens = max_tokens or self.max_context_tokens
        history_budget = int(max_tokens * self.context_window_ratio)
        memory_budget = max_tokens - history_budget

        # Optimize conversation history (keep most recent)
        optimized_history = self._optimize_history(
            conversation_history, history_budget
        )

        # Optimize memories (keep most relevant)
        optimized_memories = self._optimize_memories(
            retrieved_memories, memory_budget
        )

        return optimized_history, optimized_memories

    def _optimize_history(
        self,
        history: List[ConversationTurn],
        token_budget: int,
    ) -> List[ConversationTurn]:
        """Optimize conversation history to fit token budget."""
        if not history:
            return []

        # Start with most recent messages
        optimized = []
        current_tokens = 0

        for turn in reversed(history):
            turn_tokens = self.token_counter.count(turn.content)

            if current_tokens + turn_tokens > token_budget:
                # Summarize older messages if needed
                if len(optimized) > 0:
                    summary = self._summarize_turns(list(reversed(optimized)))
                    return [ConversationTurn(role="system", content=summary)] + optimized
                break

            optimized.insert(0, turn)
            current_tokens += turn_tokens

        return optimized

    def _optimize_memories(
        self,
        memories: List[Dict[str, Any]],
        token_budget: int,
    ) -> List[Dict[str, Any]]:
        """Optimize memories to fit token budget."""
        if not memories:
            return []

        optimized = []
        current_tokens = 0

        for memory in memories:
            content = memory.get("text", "")
            memory_tokens = self.token_counter.count(content)

            if current_tokens + memory_tokens > token_budget:
                break

            optimized.append(memory)
            current_tokens += memory_tokens

        return optimized

    def _summarize_turns(self, turns: List[ConversationTurn]) -> str:
        """Create a summary of conversation turns."""
        # Simple summary - can be enhanced with LLM
        summary_parts = []
        for turn in turns[:3]:  # Summarize first few turns
            summary_parts.append(f"{turn.role}: {turn.content[:100]}...")

        return "Previous conversation summary:\n" + "\n".join(summary_parts)

    async def get_context_for_llm(
        self,
        query: Optional[str] = None,
        include_history: bool = True,
        include_memories: bool = True,
    ) -> Dict[str, Any]:
        """Get optimized context for LLM consumption.

        Returns:
            Dictionary with conversation_history and relevant_memories
        """
        context = {
            "conversation_history": [],
            "relevant_memories": [],
        }

        session = self.get_session()
        if not session:
            return context

        # Get conversation history
        if include_history:
            history = session.get_recent_history(20)
        else:
            history = []

        # Get relevant memories
        memories = []
        if include_memories and query:
            memories = await self.retrieve_memories(query, top_k=5)

        # Optimize context window
        optimized_history, optimized_memories = self.optimize_context(
            history, memories
        )

        context["conversation_history"] = [
            {"role": t.role, "content": t.content}
            for t in optimized_history
        ]
        context["relevant_memories"] = [
            {"content": m["text"], "relevance": m["score"]}
            for m in optimized_memories
        ]

        return context

    # Persistence

    async def persist(self, filepath: Optional[str] = None) -> bool:
        """Persist memory state to disk."""
        if filepath is None:
            filepath = str(Path.home() / ".legionhercules" / "memory_state.json")

        try:
            state = {
                "sessions": {
                    sid: s.to_dict() for sid, s in self.sessions.items()
                },
                "current_session_id": self.current_session_id,
                "cache_size": len(self._memory_cache),
                "persisted_at": datetime.now().isoformat(),
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"Memory state persisted to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist memory state: {e}")
            return False

    async def load(self, filepath: Optional[str] = None) -> bool:
        """Load memory state from disk."""
        if filepath is None:
            filepath = str(Path.home() / ".legionhercules" / "memory_state.json")

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore sessions
            for sid, sdata in state.get("sessions", {}).items():
                session = SessionContext(session_id=sid)
                session.created_at = datetime.fromisoformat(sdata["created_at"])
                session.last_activity = datetime.fromisoformat(sdata["last_activity"])
                session.metadata = sdata.get("metadata", {})
                self.sessions[sid] = session

            self.current_session_id = state.get("current_session_id")

            logger.info(f"Memory state loaded from {filepath}")
            return True
        except FileNotFoundError:
            logger.debug(f"No memory state file found at {filepath}")
            return False
        except Exception as e:
            logger.error(f"Failed to load memory state: {e}")
            return False

    # Utility Methods

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return hashlib.sha256(
            datetime.now().isoformat().encode()
        ).hexdigest()[:12]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        await self.initialize()

        return {
            "total_memories": await self.vector_store.count(),
            "cached_memories": len(self._memory_cache),
            "active_sessions": len(self.sessions),
            "current_session": self.current_session_id,
        }

    async def cleanup_old_memories(self, days: Optional[int] = None) -> int:
        """Clean up memories older than specified days."""
        days = days or self.retention_days
        cutoff = datetime.now() - timedelta(days=days)

        # Find old memories
        old_ids = [
            mid for mid, entry in self._memory_cache.items()
            if entry.timestamp < cutoff and entry.importance < 1.0
        ]

        if old_ids:
            await self.vector_store.delete(old_ids)
            for mid in old_ids:
                self._memory_cache.pop(mid, None)

        logger.info(f"Cleaned up {len(old_ids)} old memories")
        return len(old_ids)
