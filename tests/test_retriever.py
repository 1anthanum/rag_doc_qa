"""Tests for the retriever and vector store modules."""

import pytest
import numpy as np

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import EmbeddedChunk
from src.retrieval.vector_store import FAISSVectorStore, SearchResult
from src.retrieval.retriever import Retriever, RetrievalResult


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    texts = [
        "The attention mechanism allows models to focus on relevant input.",
        "Convolutional neural networks excel at image recognition tasks.",
        "Recurrent networks process sequential data like time series.",
        "Transformers use self-attention instead of recurrence.",
        "BERT is a bidirectional transformer pre-trained on masked language.",
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
def sample_embedded_chunks(sample_chunks):
    """Create embedded chunks with random embeddings."""
    dim = 64
    rng = np.random.RandomState(42)
    embedded = []
    for chunk in sample_chunks:
        vec = rng.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        embedded.append(EmbeddedChunk(chunk=chunk, embedding=vec))
    return embedded


@pytest.fixture
def faiss_store(sample_embedded_chunks):
    """Create a FAISS store pre-loaded with sample data."""
    store = FAISSVectorStore(dimension=64)
    store.add(sample_embedded_chunks)
    return store


# ── FAISS Vector Store Tests ────────────────────────────────────


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def test_add_and_size(self, sample_embedded_chunks):
        """Test adding chunks increases store size."""
        store = FAISSVectorStore(dimension=64)
        assert store.size == 0

        store.add(sample_embedded_chunks)
        assert store.size == len(sample_embedded_chunks)

    def test_add_empty(self):
        """Test adding empty list is a no-op."""
        store = FAISSVectorStore(dimension=64)
        store.add([])
        assert store.size == 0

    def test_search_returns_results(self, faiss_store):
        """Test search returns the correct number of results."""
        query = np.random.randn(64).astype(np.float32)
        results = faiss_store.search(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_scores(self, faiss_store):
        """Test search results have valid scores."""
        query = np.random.randn(64).astype(np.float32)
        results = faiss_store.search(query, top_k=3)

        # Scores should be descending (FAISS inner product)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store(self):
        """Test searching empty store returns empty list."""
        store = FAISSVectorStore(dimension=64)
        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, top_k=5)

        assert results == []

    def test_search_top_k_exceeds_size(self, faiss_store):
        """Test top_k larger than store size returns all results."""
        query = np.random.randn(64).astype(np.float32)
        results = faiss_store.search(query, top_k=100)

        assert len(results) == faiss_store.size

    def test_save_and_load(self, faiss_store, tmp_path):
        """Test saving and loading preserves data."""
        save_path = str(tmp_path / "test_index")
        faiss_store.save(save_path)

        loaded_store = FAISSVectorStore(dimension=64)
        loaded_store.load(save_path)

        assert loaded_store.size == faiss_store.size

        # Search should return same results
        query = np.random.randn(64).astype(np.float32)
        orig_results = faiss_store.search(query, top_k=3)
        loaded_results = loaded_store.search(query, top_k=3)

        assert len(orig_results) == len(loaded_results)
        for orig, loaded in zip(orig_results, loaded_results):
            assert orig.chunk.text == loaded.chunk.text
            assert abs(orig.score - loaded.score) < 1e-5


# ── RetrievalResult Tests ───────────────────────────────────────


class TestRetrievalResult:
    """Tests for RetrievalResult data class."""

    def test_context_assembly(self, sample_chunks):
        """Test context string is assembled from results."""
        results = [
            SearchResult(chunk=sample_chunks[0], score=0.9),
            SearchResult(chunk=sample_chunks[1], score=0.7),
        ]
        rr = RetrievalResult(query="test", results=results)

        context = rr.context
        assert "Source 1" in context
        assert "Source 2" in context
        assert sample_chunks[0].text in context

    def test_sources_deduplication(self):
        """Test sources property deduplicates filenames."""
        chunks = [
            Chunk(text="a", chunk_id=0, source="a.txt", start_char=0,
                  end_char=1, metadata={"filename": "a.txt"}),
            Chunk(text="b", chunk_id=1, source="a.txt", start_char=0,
                  end_char=1, metadata={"filename": "a.txt"}),
            Chunk(text="c", chunk_id=2, source="b.txt", start_char=0,
                  end_char=1, metadata={"filename": "b.txt"}),
        ]
        results = [SearchResult(chunk=c, score=0.5) for c in chunks]
        rr = RetrievalResult(query="test", results=results)

        assert rr.sources == ["a.txt", "b.txt"]

    def test_empty_results(self):
        """Test empty results produce empty context."""
        rr = RetrievalResult(query="test", results=[])
        assert rr.context == ""
        assert rr.sources == []
