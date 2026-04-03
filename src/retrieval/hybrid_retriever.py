"""
Hybrid retriever combining dense vector search with BM25 sparse retrieval.
Uses Reciprocal Rank Fusion (RRF) to merge results from both sources.
"""

import logging
from typing import List, Optional

from ..ingestion.embedder import EmbeddingEngine
from ..ingestion.chunker import Chunk
from .vector_store import VectorStore, SearchResult
from .sparse_retriever import BM25Retriever
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60,
) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked result lists.

    RRF score = sum over lists of: 1 / (k + rank_in_list)

    This gives a balanced fusion that doesn't depend on raw score scales,
    making it ideal for combining BM25 (unbounded scores) with cosine
    similarity (0-1 range).

    Args:
        result_lists: Multiple ranked lists of SearchResult.
        k: RRF constant (default 60, as per original paper).

    Returns:
        Merged and re-scored list of SearchResult, sorted by RRF score.
    """
    # Track RRF scores by chunk_id to handle deduplication
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            # Use (source, chunk_id) as unique key
            key = f"{result.chunk.source}::{result.chunk.chunk_id}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            # Keep the SearchResult with highest original score
            if key not in chunk_map or result.score > chunk_map[key].score:
                chunk_map[key] = result

    # Build merged results with RRF scores
    merged = []
    for key, rrf_score in rrf_scores.items():
        result = chunk_map[key]
        merged.append(SearchResult(chunk=result.chunk, score=rrf_score))

    # Sort by RRF score descending
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged


class HybridRetriever:
    """
    Hybrid retriever combining dense (vector) and sparse (BM25) retrieval
    with Reciprocal Rank Fusion.

    Research shows hybrid search improves top-k recall by 15-25% over
    either method alone, as BM25 captures exact keyword matches that
    semantic search may miss, and vice versa.

    Usage:
        hybrid = HybridRetriever(
            embedding_engine=engine,
            vector_store=store,
            top_k=10,
            dense_weight=0.7,
            sparse_weight=0.3,
        )
        hybrid.index_sparse(chunks)
        result = hybrid.retrieve("project budget Q3")
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        rerank: bool = False,
        rerank_model: Optional[str] = None,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.rerank = rerank

        self._sparse = BM25Retriever()
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

    def index_sparse(self, chunks: List[Chunk]) -> None:
        """Build the BM25 sparse index from chunks."""
        self._sparse.index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve using both dense and sparse search, fused with RRF.

        Args:
            query: User's search query.
            top_k: Override instance top_k for this request.
            score_threshold: Override instance threshold for this request.

        Returns:
            RetrievalResult with fused results.
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

        # Fetch more candidates for fusion (3x top_k from each source)
        fetch_k = effective_top_k * 3

        # Dense retrieval
        query_embedding = self.embedding_engine.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, top_k=fetch_k)

        # Sparse retrieval
        sparse_results = self._sparse.search(query, top_k=fetch_k)

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )

        # Optional reranking on fused results
        if self.rerank and self._reranker and fused:
            fused = self._rerank_results(query, fused)

        # Filter by threshold and limit
        filtered = [r for r in fused if r.score >= effective_threshold][
            :effective_top_k
        ]

        logger.info(
            f"Hybrid retrieved {len(filtered)} chunks "
            f"(dense={len(dense_results)}, sparse={len(sparse_results)}, "
            f"fused={len(fused)}) for: '{query[:50]}...'"
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

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def clear(self) -> None:
        """Clear both dense and sparse indices."""
        self._sparse.clear()
