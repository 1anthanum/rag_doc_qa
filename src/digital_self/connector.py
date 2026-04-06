"""
Digital Self RAG Connector — query interface for conversation-based RAG.

Provides a high-level API for querying a pre-built conversation index.
Supports behavioral HyDE (conversation-style hypothetical generation)
and all standard RAG features (hybrid search, CRAG, etc.).

Usage:
    connector = DigitalSelfConnector.from_config("configs/digital_self.yaml")
    connector.load_index()

    response = connector.query("What are my thoughts on remote work?")
    print(response.answer)
    print(response.sources)
"""

import logging
from typing import Optional

from ..config import get as cfg
from ..ingestion.embedder import EmbeddingEngine
from ..retrieval import FAISSVectorStore, Retriever, HybridRetriever
from ..retrieval.query_processor import QueryProcessor
from ..generation import RAGChain, AgenticRAGChain
from ..generation.llm_client import (
    LLMClient,
    OpenAIClient,
    OllamaClient,
    AnthropicClient,
)

logger = logging.getLogger(__name__)


class DigitalSelfConnector:
    """
    Query interface for the Digital Self conversation index.

    Wraps the standard RAG pipeline with conversation-specific defaults:
        - Behavioral HyDE for query optimization
        - Hybrid search (BM25 + dense) for conversational keyword matching
        - Metadata-aware source attribution

    The connector does NOT build the index — use DigitalSelfIndexer for that.
    It loads a pre-built index and provides query() for end-user interaction.
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        store: FAISSVectorStore,
        retriever,
        chain,
        persist_dir: str = "data/digital_self_index",
        hybrid_retriever: Optional[HybridRetriever] = None,
    ):
        self.engine = engine
        self.store = store
        self.retriever = retriever
        self.chain = chain
        self.persist_dir = persist_dir
        self.hybrid_retriever = hybrid_retriever
        self._loaded = False

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "DigitalSelfConnector":
        """
        Create a connector from config, ready for load_index() + query().

        Args:
            config_path: Path to YAML config. Uses defaults if None.

        Returns:
            DigitalSelfConnector instance.
        """
        if config_path:
            from ..config import load_config

            load_config(config_path)

        # Embedding engine
        emb_provider = cfg("ingestion.embedding.provider", "local")
        emb_model = cfg("ingestion.embedding.model", "all-MiniLM-L6-v2")
        engine = EmbeddingEngine(provider=emb_provider, model_name=emb_model)

        # Vector store
        store = FAISSVectorStore(dimension=engine.dimension)

        # LLM
        llm_provider = cfg("generation.provider", "anthropic")
        llm_model = cfg("generation.model", "claude-sonnet-4-20250514")
        if llm_provider == "anthropic":
            llm: LLMClient = AnthropicClient(model=llm_model)
        elif llm_provider == "ollama":
            llm = OllamaClient(model=llm_model)
        else:
            llm = OpenAIClient(model=llm_model)

        # Retriever
        top_k = cfg("retrieval.top_k", 5)
        use_hybrid = cfg("retrieval.mode", "simple") == "hybrid" or cfg(
            "retrieval.hybrid_search.enabled", False
        )

        hybrid_retriever = None
        if use_hybrid:
            retriever = HybridRetriever(
                embedding_engine=engine,
                vector_store=store,
                top_k=top_k,
                dense_weight=cfg("retrieval.hybrid_search.dense_weight", 0.7),
                sparse_weight=cfg("retrieval.hybrid_search.sparse_weight", 0.3),
                rrf_k=cfg("retrieval.hybrid_search.rrf_k", 60),
            )
            hybrid_retriever = retriever
        else:
            retriever = Retriever(
                embedding_engine=engine,
                vector_store=store,
                top_k=top_k,
            )

        # Query processor — behavioral HyDE for conversation domain
        query_processor = None
        qp_enabled = cfg("retrieval.query_processing.enabled", False)
        qp_strategy = cfg("retrieval.query_processing.strategy", "none")
        qp_domain = cfg("retrieval.query_processing.domain", None)

        if qp_enabled and qp_strategy != "none":
            query_processor = QueryProcessor(
                llm=llm,
                strategy=qp_strategy,
                domain=qp_domain,
            )

        # Chain
        agentic_enabled = cfg("agentic.enabled", False)
        if agentic_enabled:
            chain = AgenticRAGChain(
                retriever=retriever,
                llm=llm,
                query_processor=query_processor,
                max_correction_rounds=cfg("agentic.max_correction_rounds", 2),
                adaptive_retrieval=cfg("agentic.adaptive_retrieval", True),
            )
        else:
            chain = RAGChain(
                retriever=retriever,
                llm=llm,
                query_processor=query_processor,
            )

        persist_dir = cfg("index.persist_dir", "data/digital_self_index")

        return cls(
            engine=engine,
            store=store,
            retriever=retriever,
            chain=chain,
            persist_dir=persist_dir,
            hybrid_retriever=hybrid_retriever,
        )

    def load_index(self, persist_dir: Optional[str] = None) -> None:
        """
        Load a pre-built index from disk.

        Args:
            persist_dir: Override the configured persist directory.
        """
        path = persist_dir or self.persist_dir
        self.store.load(path)
        self._loaded = True

        logger.info(f"Loaded index from {path}: {self.store.size} chunks")

        # Rebuild BM25 if hybrid
        if self.hybrid_retriever and self.store._chunks:
            self.hybrid_retriever.index_sparse(self.store._chunks)
            logger.info(f"Rebuilt BM25 sparse index ({len(self.store._chunks)} chunks)")

    def query(
        self,
        question: str,
        mode: str = "qa",
        top_k: Optional[int] = None,
        temperature: float = 0.1,
    ):
        """
        Query the conversation knowledge base.

        Args:
            question: Natural language question.
            mode: Query mode — qa | summarize | compare | conversational.
            top_k: Override default top-k retrieval count.
            temperature: LLM temperature for generation.

        Returns:
            RAGResponse with answer, sources, retrieval results, and metadata.

        Raises:
            RuntimeError: If index hasn't been loaded yet.
        """
        if not self._loaded and self.store.size == 0:
            raise RuntimeError(
                "No index loaded. Call load_index() first, or use "
                "DigitalSelfIndexer to build an index."
            )

        kwargs = {
            "question": question,
            "mode": mode,
            "temperature": temperature,
        }
        if top_k is not None:
            kwargs["top_k"] = top_k

        return self.chain.query(**kwargs)

    @property
    def index_size(self) -> int:
        """Number of chunks in the loaded index."""
        return self.store.size

    @property
    def is_loaded(self) -> bool:
        """Whether an index has been loaded."""
        return self._loaded
