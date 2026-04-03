"""
Retriever: orchestrates query embedding + vector search + optional reranking.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from ..ingestion.embedder import EmbeddingEngine
from .vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Final retrieval result with context assembly."""

    query: str
    results: List[SearchResult]

    @property
    def context(self) -> str:
        """Assemble retrieved chunks into a single context string."""
        parts = []
        for i, r in enumerate(self.results, 1):
            source = r.chunk.metadata.get("filename", r.chunk.source)
            parts.append(
                f"[Source {i}: {source} | Score: {r.score:.3f}]\n" f"{r.chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    @property
    def sources(self) -> List[str]:
        """List unique source filenames."""
        seen = set()
        sources = []
        for r in self.results:
            src = r.chunk.metadata.get("filename", r.chunk.source)
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources


class Retriever:
    """
    High-level retriever combining embedding + search + reranking.

    Usage:
        retriever = Retriever(
            embedding_engine=engine,
            vector_store=store,
            top_k=5,
        )
        result = retriever.retrieve("What is attention mechanism?")
        print(result.context)
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
        rerank: bool = False,
        rerank_model: Optional[str] = None,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rerank = rerank
        self._reranker = None

        if rerank:
            self._init_reranker(rerank_model)

    def _init_reranker(self, model_name: Optional[str] = None):
        """Initialize cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning(
                "sentence-transformers needed for reranking. "
                "Falling back to no reranking."
            )
            self.rerank = False
            return

        model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._reranker = CrossEncoder(model_name)
        logger.info(f"Loaded reranker: {model_name}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's search query.
            top_k: Override instance top_k for this request (thread-safe).
            score_threshold: Override instance threshold for this request.

        Pipeline:
            1. Embed the query
            2. Search vector store for top candidates
            3. (Optional) Rerank with cross-encoder
            4. Filter by score threshold
            5. Return assembled result
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

        # Step 1: Embed query
        query_embedding = self.embedding_engine.embed_query(query)

        # Step 2: Vector search (fetch extra candidates if reranking)
        fetch_k = effective_top_k * 3 if self.rerank else effective_top_k
        search_results = self.vector_store.search(query_embedding, top_k=fetch_k)

        # Step 3: Optional reranking
        if self.rerank and self._reranker and search_results:
            search_results = self._rerank_results(query, search_results)

        # Step 4: Filter by threshold and limit
        filtered = [r for r in search_results if r.score >= effective_threshold][
            :effective_top_k
        ]

        logger.info(
            f"Retrieved {len(filtered)} chunks for query: "
            f"'{query[:50]}...' "
            f"(searched {len(search_results)}, "
            f"threshold={effective_threshold})"
        )

        return RetrievalResult(query=query, results=filtered)

    def _rerank_results(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using a cross-encoder model."""
        if not self._reranker:
            return results

        pairs = [(query, r.chunk.text) for r in results]
        scores = self._reranker.predict(pairs)

        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by reranked score (descending)
        results.sort(key=lambda r: r.score, reverse=True)
        return results
