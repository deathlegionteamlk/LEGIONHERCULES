"""Vector storage for memory system using ChromaDB."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, List, Dict
from datetime import datetime

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    async def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts with embeddings to the store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete items by ID."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all data."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total count of items."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector storage."""

    def __init__(
        self,
        collection_name: str = "legionhercules_memory",
        persist_directory: Optional[str] = None,
        embedding_dimension: int = 384,
    ):
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._client = None
        self._collection = None
        
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            self.persist_directory = Path.home() / ".legionhercules" / "chroma_db"
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts with embeddings to the store."""
        if self._collection is None:
            await self.initialize()

        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(text) for text in texts]

        # Add timestamps to metadata
        if metadata is None:
            metadata = [{} for _ in texts]
        
        for meta in metadata:
            meta["timestamp"] = datetime.now().isoformat()
            meta["dimension"] = self.embedding_dimension

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata,
            )
            logger.debug(f"Added {len(ids)} items to ChromaDB")
            return ids
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        if self._collection is None:
            await self.initialize()

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    })

            return formatted_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    async def delete(self, ids: List[str]) -> bool:
        """Delete items by ID."""
        if self._collection is None:
            await self.initialize()

        try:
            self._collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} items from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all data."""
        if self._collection is None:
            await self.initialize()

        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")
            return False

    async def count(self) -> int:
        """Get total count of items."""
        if self._collection is None:
            await self.initialize()

        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"Error counting ChromaDB items: {e}")
            return 0

    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    async def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts with embeddings."""
        if ids is None:
            ids = [hashlib.sha256(text.encode()).hexdigest()[:16] for text in texts]
        
        if metadata is None:
            metadata = [{} for _ in texts]

        for i, (id_, text, embedding, meta) in enumerate(zip(ids, texts, embeddings, metadata)):
            meta["timestamp"] = datetime.now().isoformat()
            self._data[id_] = {
                "text": text,
                "embedding": embedding,
                "metadata": meta,
            }

        return ids

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using cosine similarity."""
        import numpy as np

        if not self._data:
            return []

        # Calculate cosine similarity
        query_vec = np.array(query_embedding)
        similarities = []

        for id_, item in self._data.items():
            # Apply filter if provided
            if filter_dict:
                skip = False
                for key, value in filter_dict.items():
                    if item["metadata"].get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            vec = np.array(item["embedding"])
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            similarities.append((id_, similarity, item))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for id_, similarity, item in similarities[:top_k]:
            results.append({
                "id": id_,
                "text": item["text"],
                "metadata": item["metadata"],
                "distance": 1 - similarity,
                "score": similarity,
            })

        return results

    async def delete(self, ids: List[str]) -> bool:
        """Delete items by ID."""
        for id_ in ids:
            self._data.pop(id_, None)
        return True

    async def clear(self) -> bool:
        """Clear all data."""
        self._data.clear()
        return True

    async def count(self) -> int:
        """Get total count."""
        return len(self._data)
