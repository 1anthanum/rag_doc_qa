"""
Embedding engine for converting text chunks to dense vectors.
Supports both local (sentence-transformers) and API-based models.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk enriched with its embedding vector."""
    chunk: Chunk
    embedding: np.ndarray

    @property
    def dimension(self) -> int:
        return self.embedding.shape[0]


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (n, dim) array."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (dim,) array."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality)
    Alternative:   all-mpnet-base-v2 (768-dim, higher quality)

    Usage:
        embedder = SentenceTransformerEmbedder()
        vectors = embedder.embed_texts(["Hello world", "How are you"])
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

        logger.info(
            f"Loaded embedding model: {model_name} "
            f"(dim={self._dimension}, device={self._model.device})"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query])[0]


class OpenAIEmbedder(BaseEmbedder):
    """
    API-based embedding using OpenAI's embedding endpoint.

    Requires OPENAI_API_KEY environment variable.

    Usage:
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install: pip install openai"
            )

        import os
        self.model = model
        self._client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

        # Dimension lookup for known models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = self._dimensions.get(model, 1536)

        logger.info(f"Using OpenAI embedding model: {model}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts via OpenAI API."""
        response = self._client.embeddings.create(
            input=texts,
            model=self.model,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


class EmbeddingEngine:
    """
    High-level engine that embeds Chunk objects.

    Orchestrates the embedder and returns EmbeddedChunk objects
    ready for vector store insertion.

    Usage:
        engine = EmbeddingEngine(provider="local")
        embedded_chunks = engine.embed_chunks(chunks)
    """

    PROVIDERS = {
        "local": SentenceTransformerEmbedder,
        "openai": OpenAIEmbedder,
    }

    def __init__(
        self,
        provider: str = "local",
        **kwargs,
    ):
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Choose from: {list(self.PROVIDERS.keys())}"
            )

        self.embedder = self.PROVIDERS[provider](**kwargs)
        self.provider = provider

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Embed a list of chunks and return EmbeddedChunk objects."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        embedded = [
            EmbeddedChunk(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]

        logger.info(
            f"Embedded {len(embedded)} chunks "
            f"(provider={self.provider}, dim={self.dimension})"
        )
        return embedded

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query for retrieval."""
        return self.embedder.embed_query(query)
