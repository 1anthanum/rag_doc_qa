"""
Digital Self Indexer — builds and persists a FAISS index from conversations.

Takes conversation JSON files, converts them to Chunks via ConversationLoader,
embeds them with EmbeddingEngine, stores in FAISSVectorStore, and optionally
builds a BM25 sparse index for hybrid search.

The index is persisted to disk using FAISSVectorStore.save(), which writes
both the FAISS index and chunk metadata (including conversation-specific
fields like roles, timestamps, turn indices) to a sidecar JSON file.

Usage:
    indexer = DigitalSelfIndexer.from_config("configs/digital_self.yaml")
    stats = indexer.index_directory("data/conversations/")
    indexer.save()
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..config import get as cfg
from ..ingestion.conversation_loader import ConversationLoader
from ..ingestion.embedder import EmbeddingEngine
from ..ingestion.chunker import Chunk
from ..retrieval import FAISSVectorStore, HybridRetriever

logger = logging.getLogger(__name__)


class DigitalSelfIndexer:
    """
    Builds a searchable index from conversation data.

    Reuses the existing RAG pipeline components:
        - ConversationLoader → List[Chunk]
        - EmbeddingEngine.embed_chunks() → List[EmbeddedChunk]
        - FAISSVectorStore.add() → FAISS index
        - HybridRetriever.index_sparse() → BM25 index (optional)
    """

    def __init__(
        self,
        loader: ConversationLoader,
        engine: EmbeddingEngine,
        store: FAISSVectorStore,
        persist_dir: str = "data/digital_self_index",
        hybrid_retriever: Optional[HybridRetriever] = None,
    ):
        self.loader = loader
        self.engine = engine
        self.store = store
        self.persist_dir = Path(persist_dir)
        self.hybrid_retriever = hybrid_retriever

        # Track indexed chunks for BM25 indexing
        self._all_chunks: List[Chunk] = []

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "DigitalSelfIndexer":
        """
        Create an indexer from a YAML config file.

        If config_path is None, uses default values. The config file is
        loaded via src.config which supports environment variable overrides.

        Args:
            config_path: Path to YAML config (e.g. "configs/digital_self.yaml").

        Returns:
            Configured DigitalSelfIndexer instance.
        """
        # Load config if provided
        if config_path:
            from ..config import load_config

            load_config(config_path)

        # Conversation loader
        conv_strategy = cfg("conversation.strategy", "turn_group")
        turns_per_chunk = cfg("conversation.turns_per_chunk", 4)
        overlap_turns = cfg("conversation.overlap_turns", 1)
        min_chunk_length = cfg("conversation.min_chunk_length", 50)

        loader = ConversationLoader(
            strategy=conv_strategy,
            turns_per_chunk=turns_per_chunk,
            overlap_turns=overlap_turns,
            min_chunk_length=min_chunk_length,
        )

        # Embedding engine
        emb_provider = cfg("ingestion.embedding.provider", "local")
        emb_model = cfg("ingestion.embedding.model", "all-MiniLM-L6-v2")
        engine = EmbeddingEngine(provider=emb_provider, model_name=emb_model)

        # Vector store
        store = FAISSVectorStore(dimension=engine.dimension)

        # Hybrid retriever (optional)
        hybrid_retriever = None
        use_hybrid = cfg("retrieval.mode", "simple") == "hybrid" or cfg(
            "retrieval.hybrid_search.enabled", False
        )
        if use_hybrid:
            hybrid_retriever = HybridRetriever(
                embedding_engine=engine,
                vector_store=store,
                top_k=cfg("retrieval.top_k", 5),
                dense_weight=cfg("retrieval.hybrid_search.dense_weight", 0.7),
                sparse_weight=cfg("retrieval.hybrid_search.sparse_weight", 0.3),
                rrf_k=cfg("retrieval.hybrid_search.rrf_k", 60),
            )

        persist_dir = cfg("index.persist_dir", "data/digital_self_index")

        return cls(
            loader=loader,
            engine=engine,
            store=store,
            persist_dir=persist_dir,
            hybrid_retriever=hybrid_retriever,
        )

    def index_file(self, file_path: str) -> Dict[str, Any]:
        """
        Index a single conversation file.

        Args:
            file_path: Path to conversation JSON file.

        Returns:
            Stats dict with keys: file, turns, chunks.
        """
        chunks = self.loader.load_and_chunk(file_path)

        if not chunks:
            logger.warning(f"No chunks produced from {file_path}")
            return {"file": file_path, "turns": 0, "chunks": 0}

        # Embed and add to vector store
        embedded = self.engine.embed_chunks(chunks)
        self.store.add(embedded)

        # Track for BM25
        self._all_chunks.extend(chunks)

        conv = self.loader.load_file(file_path)
        stats = {
            "file": file_path,
            "turns": conv.turn_count,
            "chunks": len(chunks),
        }
        logger.info(
            f"Indexed {file_path}: {stats['turns']} turns → {stats['chunks']} chunks"
        )
        return stats

    def index_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        Index all conversation JSON files in a directory.

        Args:
            dir_path: Path to directory containing JSON files.

        Returns:
            Aggregate stats dict.
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        total_files = 0
        total_turns = 0
        total_chunks = 0
        errors = []

        for json_file in sorted(path.glob("*.json")):
            try:
                stats = self.index_file(str(json_file))
                total_files += 1
                total_turns += stats["turns"]
                total_chunks += stats["chunks"]
            except Exception as e:
                logger.warning(f"Failed to index {json_file.name}: {e}")
                errors.append({"file": str(json_file), "error": str(e)})

        # Build BM25 index for hybrid search
        if self.hybrid_retriever and self._all_chunks:
            self.hybrid_retriever.index_sparse(self._all_chunks)
            logger.info(f"Built BM25 index with {len(self._all_chunks)} chunks")

        result = {
            "files": total_files,
            "turns": total_turns,
            "chunks": total_chunks,
            "index_size": self.store.size,
            "errors": errors,
        }
        logger.info(
            f"Indexing complete: {total_files} files, "
            f"{total_turns} turns → {total_chunks} chunks"
        )
        return result

    def save(self) -> str:
        """
        Persist the FAISS index and chunk metadata to disk.

        Uses FAISSVectorStore.save() which writes:
            - index.faiss: the FAISS vector index
            - chunks.json: all Chunk data including metadata

        Returns:
            Path to the persist directory.
        """
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(str(self.persist_dir))
        logger.info(f"Index saved to {self.persist_dir}")
        return str(self.persist_dir)

    def load(self) -> None:
        """
        Load a previously persisted index from disk.

        Restores both the FAISS index and chunk metadata.
        """
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {self.persist_dir}")

        self.store.load(str(self.persist_dir))
        logger.info(f"Index loaded from {self.persist_dir}: {self.store.size} chunks")

        # Rebuild BM25 from loaded chunks if hybrid
        if self.hybrid_retriever and self.store._chunks:
            self.hybrid_retriever.index_sparse(self.store._chunks)
            logger.info(
                f"Rebuilt BM25 index from {len(self.store._chunks)} loaded chunks"
            )
