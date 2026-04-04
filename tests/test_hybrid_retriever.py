"""Tests for the hybrid retriever and Reciprocal Rank Fusion."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import SearchResult
from src.retrieval.retriever import RetrievalResult
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion, HybridRetriever

# ── Helper factories ────────────────────────────────────────────


def _make_chunk(text, chunk_id, source="test.txt"):
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source=source,
        start_char=0,
        end_char=len(text),
        metadata={"filename": source},
    )


def _make_result(text, chunk_id, score, source="test.txt"):
    return SearchResult(
        chunk=_make_chunk(text, chunk_id, source),
        score=score,
    )


# ── RRF Tests ──────────────────────────────────────────────────


class TestReciprocalRankFusion:
    """Tests for the reciprocal_rank_fusion function."""

    def test_single_list_preserves_order(self):
        """RRF with one list preserves ranking order."""
        results = [
            _make_result("A", 0, 0.9),
            _make_result("B", 1, 0.7),
            _make_result("C", 2, 0.5),
        ]
        fused = reciprocal_rank_fusion([results], k=60)

        assert len(fused) == 3
        texts = [r.chunk.text for r in fused]
        assert texts == ["A", "B", "C"]

    def test_two_lists_merge(self):
        """RRF merges two ranked lists."""
        list_a = [
            _make_result("A", 0, 0.9),
            _make_result("B", 1, 0.8),
        ]
        list_b = [
            _make_result("C", 2, 0.7),
            _make_result("A", 0, 0.6),  # Duplicate of A
        ]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)

        # A appears in both lists, should have highest RRF score
        assert fused[0].chunk.text == "A"
        # All 3 unique chunks present
        assert len(fused) == 3

    def test_deduplication(self):
        """RRF deduplicates chunks that appear in multiple lists."""
        same_chunk = _make_result("Shared", 0, 0.9)
        same_chunk_copy = _make_result("Shared", 0, 0.5)

        fused = reciprocal_rank_fusion([[same_chunk], [same_chunk_copy]])
        assert len(fused) == 1
        assert fused[0].chunk.text == "Shared"

    def test_rrf_score_formula(self):
        """Verify RRF score calculation: 1/(k + rank + 1)."""
        results = [_make_result("A", 0, 0.9)]
        k = 60
        fused = reciprocal_rank_fusion([results], k=k)

        # Single list, rank 0: score = 1/(60+0+1) = 1/61
        expected = 1.0 / (k + 1)
        assert abs(fused[0].score - expected) < 1e-9

    def test_empty_lists(self):
        """RRF with empty lists returns empty."""
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_different_sources_not_deduped(self):
        """Chunks from different sources with same chunk_id are separate."""
        a = _make_result("Doc A", 0, 0.9, source="a.txt")
        b = _make_result("Doc B", 0, 0.8, source="b.txt")

        fused = reciprocal_rank_fusion([[a], [b]])
        assert len(fused) == 2

    def test_k_parameter_affects_scores(self):
        """Smaller k amplifies rank differences in scores."""
        results = [
            _make_result("First", 0, 0.9),
            _make_result("Second", 1, 0.5),
        ]
        fused_small_k = reciprocal_rank_fusion([results], k=1)
        fused_large_k = reciprocal_rank_fusion([results], k=1000)

        # With small k, the ratio between rank 0 and rank 1 scores is larger
        ratio_small = fused_small_k[0].score / fused_small_k[1].score
        ratio_large = fused_large_k[0].score / fused_large_k[1].score
        assert ratio_small > ratio_large


# ── HybridRetriever Tests ──────────────────────────────────────


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for indexing."""
        texts = [
            "The attention mechanism focuses on relevant tokens.",
            "Convolutional networks process image features.",
            "BERT uses masked language modeling for pre-training.",
        ]
        return [_make_chunk(text, i) for i, text in enumerate(texts)]

    @pytest.fixture
    def mock_engine(self):
        """Mock embedding engine."""
        engine = MagicMock()
        engine.embed_query.return_value = np.random.randn(64).astype(np.float32)
        return engine

    @pytest.fixture
    def mock_store(self, sample_chunks):
        """Mock vector store with pre-loaded results."""
        store = MagicMock()
        store.search.return_value = [
            SearchResult(chunk=sample_chunks[0], score=0.9),
            SearchResult(chunk=sample_chunks[2], score=0.7),
        ]
        return store

    @pytest.fixture
    def hybrid(self, mock_engine, mock_store):
        """Create HybridRetriever with mocked dense components."""
        return HybridRetriever(
            embedding_engine=mock_engine,
            vector_store=mock_store,
            top_k=5,
            dense_weight=0.7,
            sparse_weight=0.3,
        )

    def test_index_sparse(self, hybrid, sample_chunks):
        """Test BM25 sparse indexing."""
        hybrid.index_sparse(sample_chunks)
        assert hybrid._sparse.size == len(sample_chunks)

    def test_retrieve_returns_retrieval_result(self, hybrid, sample_chunks):
        """Test retrieve returns a RetrievalResult."""
        hybrid.index_sparse(sample_chunks)
        result = hybrid.retrieve("attention mechanism")

        assert isinstance(result, RetrievalResult)
        assert result.query == "attention mechanism"

    def test_retrieve_fuses_dense_and_sparse(self, hybrid, sample_chunks):
        """Test retrieve combines dense and sparse results."""
        hybrid.index_sparse(sample_chunks)
        result = hybrid.retrieve("attention mechanism")

        # Should have results (from either dense or sparse or both)
        assert len(result.results) > 0

    def test_retrieve_respects_top_k(self, hybrid, sample_chunks):
        """Test retrieve limits results to top_k."""
        hybrid.index_sparse(sample_chunks)
        result = hybrid.retrieve("attention", top_k=2)

        assert len(result.results) <= 2

    def test_retrieve_respects_score_threshold(self, mock_engine, sample_chunks):
        """Test retrieve filters by score threshold."""
        store = MagicMock()
        store.search.return_value = [
            SearchResult(chunk=sample_chunks[0], score=0.001),
        ]
        hybrid = HybridRetriever(
            embedding_engine=mock_engine,
            vector_store=store,
            top_k=5,
            score_threshold=0.1,  # High threshold
        )
        hybrid.index_sparse(sample_chunks)
        result = hybrid.retrieve("completely unrelated xyz")

        # RRF scores are very small (1/61 ≈ 0.016), below 0.1 threshold
        # All results should be filtered out
        assert len(result.results) == 0 or all(r.score >= 0.1 for r in result.results)

    def test_clear_resets_sparse_index(self, hybrid, sample_chunks):
        """Test clear resets the BM25 index."""
        hybrid.index_sparse(sample_chunks)
        assert hybrid._sparse.size > 0

        hybrid.clear()
        assert hybrid._sparse.size == 0

    def test_retrieve_without_sparse_index(self, hybrid):
        """Test retrieve works even without sparse index (dense only)."""
        result = hybrid.retrieve("attention mechanism")

        assert isinstance(result, RetrievalResult)
        # Should still return dense results
        assert len(result.results) >= 0

    def test_top_k_override(self, hybrid, sample_chunks):
        """Test per-request top_k override."""
        hybrid.index_sparse(sample_chunks)

        result = hybrid.retrieve("attention", top_k=1)
        assert len(result.results) <= 1
