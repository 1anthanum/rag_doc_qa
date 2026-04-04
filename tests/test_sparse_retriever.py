"""Tests for the BM25 sparse retriever module."""

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import SearchResult
from src.retrieval.sparse_retriever import BM25Retriever

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_chunks():
    """Create sample chunks for BM25 testing."""
    texts = [
        "The attention mechanism allows models to focus on relevant input tokens.",
        "Convolutional neural networks excel at image recognition and classification tasks.",
        "Recurrent networks process sequential data like time series and text.",
        "Transformers use self-attention instead of recurrence for parallelism.",
        "BERT is a bidirectional transformer pre-trained on masked language modeling.",
    ]
    return [
        Chunk(
            text=text,
            chunk_id=i,
            source="test.txt",
            start_char=0,
            end_char=len(text),
            metadata={"filename": "test.txt"},
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def bm25(sample_chunks):
    """Create a BM25 retriever with indexed sample data."""
    retriever = BM25Retriever()
    retriever.index(sample_chunks)
    return retriever


# ── BM25 Retriever Tests ───────────────────────────────────────


class TestBM25Retriever:
    """Tests for BM25Retriever."""

    def test_initial_state(self):
        """Test newly created retriever is empty."""
        retriever = BM25Retriever()
        assert retriever.size == 0

    def test_index_builds_corpus(self, bm25, sample_chunks):
        """Test indexing populates size correctly."""
        assert bm25.size == len(sample_chunks)

    def test_index_empty_chunks(self):
        """Test indexing with empty list produces zero-size index."""
        retriever = BM25Retriever()
        retriever.index([])
        assert retriever.size == 0

    def test_search_returns_results(self, bm25):
        """Test search returns relevant SearchResult objects."""
        results = bm25.search("attention mechanism transformer", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_are_positive(self, bm25):
        """Test all returned results have positive scores."""
        results = bm25.search("attention transformer", top_k=5)

        for result in results:
            assert result.score > 0

    def test_search_scores_are_descending(self, bm25):
        """Test results are ordered by descending score."""
        results = bm25.search("attention transformer", top_k=5)

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_keyword_relevance(self, bm25):
        """Test that keyword-matched chunks rank higher."""
        results = bm25.search("convolutional neural networks image", top_k=3)

        # The CNN chunk should appear in top results
        top_texts = [r.chunk.text for r in results]
        assert any("Convolutional" in t for t in top_texts)

    def test_search_no_index_returns_empty(self):
        """Test search on unindexed retriever returns empty list."""
        retriever = BM25Retriever()
        results = retriever.search("anything", top_k=5)
        assert results == []

    def test_search_empty_query_returns_empty(self, bm25):
        """Test search with empty query returns empty list."""
        results = bm25.search("", top_k=5)
        assert results == []

    def test_search_single_char_query_returns_empty(self, bm25):
        """Test single-character tokens are skipped by tokenizer."""
        # Single-char tokens are filtered, so query "a" yields no valid tokens
        results = bm25.search("a", top_k=5)
        assert results == []

    def test_top_k_limits_results(self, bm25):
        """Test top_k parameter limits returned results."""
        results = bm25.search("neural network model", top_k=2)
        assert len(results) <= 2

    def test_clear_resets_state(self, bm25):
        """Test clear empties the index."""
        assert bm25.size > 0
        bm25.clear()
        assert bm25.size == 0
        assert bm25.search("attention", top_k=5) == []

    def test_tokenizer_unicode_support(self):
        """Test tokenizer handles CJK characters."""
        retriever = BM25Retriever()
        tokens = retriever._tokenize("注意力机制是 transformers 的核心")
        assert len(tokens) > 0
        # "transformers" should be in tokens
        assert "transformers" in tokens

    def test_tokenizer_lowercases(self):
        """Test tokenizer converts to lowercase."""
        retriever = BM25Retriever()
        tokens = retriever._tokenize("ATTENTION Mechanism")
        assert "attention" in tokens
        assert "mechanism" in tokens

    def test_reindex_replaces_old_data(self, sample_chunks):
        """Test re-indexing replaces previous index."""
        retriever = BM25Retriever()
        retriever.index(sample_chunks)
        assert retriever.size == 5

        # BM25 IDF needs >= 3 documents to produce non-zero scores
        new_chunks = [
            Chunk(
                text="A completely different document about cooking recipes and kitchen techniques.",
                chunk_id=0,
                source="cooking.txt",
                start_char=0,
                end_char=70,
            ),
            Chunk(
                text="Another document discussing unrelated programming topics and software.",
                chunk_id=1,
                source="programming.txt",
                start_char=0,
                end_char=65,
            ),
            Chunk(
                text="Mathematics involves algebra calculus and geometry for problem solving.",
                chunk_id=2,
                source="math.txt",
                start_char=0,
                end_char=65,
            ),
        ]
        retriever.index(new_chunks)
        assert retriever.size == 3

        # Old data should be gone: searching for old content returns no results
        old_results = retriever.search("attention mechanism transformer", top_k=3)
        old_texts = [r.chunk.text for r in old_results]
        assert not any("attention" in t.lower() for t in old_texts)

        # New data should be searchable
        new_results = retriever.search("cooking recipes kitchen", top_k=3)
        assert len(new_results) > 0
        assert new_results[0].chunk.source == "cooking.txt"
