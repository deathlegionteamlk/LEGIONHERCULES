"""Embedding providers for memory system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import hashlib
import json
import os
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Embed text into vector."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding using sentence-transformers or fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        super().__init__(dimension)
        self.model_name = model_name
        self._model = None
        self._cache_dir = Path.home() / ".legionhercules" / "embeddings_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def embed(self, text: str) -> List[float]:
        """Embed text using local model or fallback."""
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Try sentence-transformers
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, using fallback")
                return await self._fallback_embed(text)

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self._model.encode(text, convert_to_numpy=True)
            )
            result = embedding.tolist()
            self._cache_embedding(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return await self._fallback_embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                return [await self._fallback_embed(t) for t in texts]

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: self._model.encode(texts, convert_to_numpy=True)
            )
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [await self._fallback_embed(t) for t in texts]

    async def _fallback_embed(self, text: str) -> List[float]:
        """Simple hash-based fallback embedding."""
        # Create deterministic embedding from text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to floats in range [-1, 1]
        floats = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            val = int.from_bytes(chunk, 'little') / (2**31) - 1
            floats.append(val)
        
        # Pad or truncate to dimension
        if len(floats) < self.dimension:
            floats = floats * (self.dimension // len(floats) + 1)
        return floats[:self.dimension]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[List[float]]:
        """Get cached embedding."""
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _cache_embedding(self, key: str, embedding: List[float]) -> None:
        """Cache embedding."""
        cache_file = self._cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.debug(f"Failed to cache embedding: {e}")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API embeddings."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        super().__init__(dimension=1536 if "small" in model else 3072)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None

    async def embed(self, text: str) -> List[float]:
        """Embed text using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

        try:
            response = await self._client.post(
                "/embeddings",
                json={"model": self.model, "input": text}
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

        try:
            response = await self._client.post(
                "/embeddings",
                json={"model": self.model, "input": texts}
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise
