"""
Vector store implementations for document chunk storage and similarity search.
Supports FAISS (local, fast) and ChromaDB (persistent, metadata filtering).
"""

import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from ..ingestion.chunker import Chunk
from ..ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score."""

    chunk: Chunk
    score: float  # Higher = more similar


class VectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """Add embedded chunks to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search for most similar chunks."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the store to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the store from disk."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors from the store."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of stored vectors."""
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store for fast similarity search.

    Uses IndexFlatIP (inner product) with normalized vectors,
    which is equivalent to cosine similarity.

    Usage:
        store = FAISSVectorStore(dimension=384)
        store.add(embedded_chunks)
        results = store.search(query_vec, top_k=5)
        store.save("./index")
    """

    def __init__(self, dimension: int):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu required. Install: pip install faiss-cpu")

        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[Chunk] = []

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """Add embedded chunks to the FAISS index."""
        if not embedded_chunks:
            return

        embeddings = np.stack([ec.embedding for ec in embedded_chunks]).astype(
            np.float32
        )

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        self._index.add(embeddings)
        self._chunks.extend([ec.chunk for ec in embedded_chunks])

        logger.info(
            f"Added {len(embedded_chunks)} vectors to FAISS " f"(total: {self.size})"
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search for top-k most similar chunks."""
        if self.size == 0:
            return []

        # Normalize query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        top_k = min(top_k, self.size)
        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append(
                    SearchResult(chunk=self._chunks[idx], score=float(score))
                )

        return results

    def clear(self) -> None:
        """Remove all vectors from the FAISS index."""
        import faiss

        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks.clear()
        logger.info("Cleared FAISS index")

    def save(self, path: str) -> None:
        """Save FAISS index and chunk metadata to disk."""
        import faiss

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(save_dir / "index.faiss"))

        # Save chunk metadata
        chunks_data = []
        for chunk in self._chunks:
            chunks_data.append(
                {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                }
            )

        with open(save_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index ({self.size} vectors) to {path}")

    def load(self, path: str) -> None:
        """Load FAISS index and chunk metadata from disk."""
        import faiss

        save_dir = Path(path)

        self._index = faiss.read_index(str(save_dir / "index.faiss"))

        with open(save_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self._chunks = [
            Chunk(
                text=c["text"],
                chunk_id=c["chunk_id"],
                source=c["source"],
                start_char=c["start_char"],
                end_char=c["end_char"],
                metadata=c.get("metadata", {}),
            )
            for c in chunks_data
        ]

        logger.info(f"Loaded FAISS index ({self.size} vectors) from {path}")


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store with metadata filtering.

    Provides persistent storage and rich query capabilities
    including metadata-based filtering.

    Usage:
        store = ChromaVectorStore(
            collection_name="my_docs",
            persist_directory="./chroma_db"
        )
        store.add(embedded_chunks)
        results = store.search(query_vec, top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb required. Install: pip install chromadb")

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._chunks: List[Chunk] = []
        logger.info(
            f"ChromaDB collection '{collection_name}' "
            f"({self.size} existing documents)"
        )

    @property
    def size(self) -> int:
        return self._collection.count()

    def add(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """Add embedded chunks to ChromaDB."""
        if not embedded_chunks:
            return

        ids = [f"chunk_{ec.chunk.chunk_id}" for ec in embedded_chunks]
        documents = [ec.chunk.text for ec in embedded_chunks]
        embeddings = [ec.embedding.tolist() for ec in embedded_chunks]
        metadatas = [
            {
                "source": ec.chunk.source,
                "chunk_id": ec.chunk.chunk_id,
                **ec.chunk.metadata,
            }
            for ec in embedded_chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._chunks.extend([ec.chunk for ec in embedded_chunks])

        logger.info(
            f"Added {len(embedded_chunks)} documents to ChromaDB "
            f"(total: {self.size})"
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search ChromaDB for similar chunks."""
        if self.size == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.size),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = Chunk(
                text=doc,
                chunk_id=meta.get("chunk_id", 0),
                source=meta.get("source", ""),
                start_char=0,
                end_char=len(doc),
                metadata=meta,
            )
            # ChromaDB returns distance; convert to similarity
            score = 1.0 - distance
            search_results.append(SearchResult(chunk=chunk, score=score))

        return search_results

    def clear(self) -> None:
        """Remove all documents from the ChromaDB collection."""
        name = self._collection.name
        metadata = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )
        self._chunks.clear()
        logger.info(f"Cleared ChromaDB collection '{name}'")

    def save(self, path: str) -> None:
        """ChromaDB with PersistentClient auto-saves."""
        logger.info("ChromaDB auto-persists; no explicit save needed.")

    def load(self, path: str) -> None:
        """ChromaDB with PersistentClient auto-loads."""
        logger.info("ChromaDB auto-loads from persist_directory.")
