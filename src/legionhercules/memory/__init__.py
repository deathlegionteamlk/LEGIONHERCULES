"""Memory and context management module for LEGIONHERCULES."""

from legionhercules.memory.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
)

from legionhercules.memory.vector_store import (
    VectorStore,
    ChromaVectorStore,
    InMemoryVectorStore,
)

from legionhercules.memory.manager import (
    MemoryManager,
    MemoryEntry,
    ConversationTurn,
    SessionContext,
    TokenCounter,
)

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "VectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "MemoryManager",
    "MemoryEntry",
    "ConversationTurn",
    "SessionContext",
    "TokenCounter",
]
