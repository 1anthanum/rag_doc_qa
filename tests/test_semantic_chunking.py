"""Tests for semantic chunking in TextChunker."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.ingestion.loader import Document
from src.ingestion.chunker import TextChunker, ChunkingStrategy

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic embeddings."""
    embedder = MagicMock()

    def embed_texts(texts):
        """Generate embeddings where similar sentences get similar vectors."""
        rng = np.random.RandomState(42)
        dim = 64
        embeddings = []
        for i, text in enumerate(texts):
            vec = rng.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings)

    embedder.embed_texts.side_effect = embed_texts
    return embedder


@pytest.fixture
def topic_shift_embedder():
    """Embedder that simulates clear topic boundaries.

    Returns similar vectors within a topic group and dissimilar vectors
    across groups, so semantic chunking can detect boundaries.
    """
    embedder = MagicMock()

    def embed_texts(texts):
        dim = 64
        embeddings = []
        # Assign embeddings based on keyword clusters
        topic_a_vec = np.ones(dim, dtype=np.float32)
        topic_a_vec = topic_a_vec / np.linalg.norm(topic_a_vec)

        topic_b_vec = -np.ones(dim, dtype=np.float32)
        topic_b_vec = topic_b_vec / np.linalg.norm(topic_b_vec)

        for text in texts:
            if "attention" in text.lower() or "transformer" in text.lower():
                # Add small noise to base vector
                noise = np.random.RandomState(hash(text) % 2**31).randn(dim) * 0.1
                vec = topic_a_vec + noise.astype(np.float32)
            elif "cooking" in text.lower() or "recipe" in text.lower():
                noise = np.random.RandomState(hash(text) % 2**31).randn(dim) * 0.1
                vec = topic_b_vec + noise.astype(np.float32)
            else:
                vec = (
                    np.random.RandomState(hash(text) % 2**31)
                    .randn(dim)
                    .astype(np.float32)
                )
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings)

    embedder.embed_texts.side_effect = embed_texts
    return embedder


@pytest.fixture
def multi_topic_document():
    """Document with two distinct topic groups."""
    content = (
        "The attention mechanism is a key component of transformers. "
        "Self-attention computes queries, keys, and values from the same input. "
        "Multi-head attention runs multiple attention operations in parallel. "
        "The transformer architecture revolutionized natural language processing. "
        "Now let us discuss cooking techniques for beginners. "
        "A good recipe starts with fresh ingredients. "
        "Cooking at the right temperature ensures proper texture. "
        "Many recipes require precise timing and patience."
    )
    return Document(
        content=content,
        source="mixed_topics.txt",
        doc_type="txt",
        metadata={"filename": "mixed_topics.txt"},
    )


@pytest.fixture
def short_document():
    """Document too short for semantic chunking (< 3 sentences)."""
    return Document(
        content="Short text here. One more sentence.",
        source="short.txt",
        doc_type="txt",
        metadata={"filename": "short.txt"},
    )


# ── Semantic Chunking Tests ────────────────────────────────────


class TestSemanticChunking:
    """Tests for the SEMANTIC chunking strategy."""

    def test_semantic_strategy_requires_embedder(self):
        """Test that SEMANTIC strategy raises without embedder."""
        with pytest.raises(ValueError, match="requires an embedder"):
            TextChunker(
                chunk_size=500,
                overlap=50,
                strategy=ChunkingStrategy.SEMANTIC,
                embedder=None,
            )

    def test_semantic_strategy_accepts_embedder(self, mock_embedder):
        """Test SEMANTIC strategy initializes with embedder."""
        chunker = TextChunker(
            chunk_size=500,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=mock_embedder,
        )
        assert chunker.strategy == ChunkingStrategy.SEMANTIC

    def test_semantic_chunking_produces_chunks(
        self, topic_shift_embedder, multi_topic_document
    ):
        """Test semantic chunking produces non-empty chunks."""
        chunker = TextChunker(
            chunk_size=1000,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=topic_shift_embedder,
            semantic_threshold=0.5,
        )
        chunks = chunker.chunk_document(multi_topic_document)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0
            assert chunk.source == "mixed_topics.txt"

    def test_semantic_chunking_calls_embedder(
        self, mock_embedder, multi_topic_document
    ):
        """Test semantic chunking invokes the embedder."""
        chunker = TextChunker(
            chunk_size=1000,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=mock_embedder,
            semantic_threshold=0.5,
        )
        chunker.chunk_document(multi_topic_document)

        mock_embedder.embed_texts.assert_called_once()
        # Should be called with a list of sentences
        call_args = mock_embedder.embed_texts.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) >= 3

    def test_short_document_falls_back_to_recursive(
        self, mock_embedder, short_document
    ):
        """Test documents with < 3 sentences fall back to recursive split."""
        chunker = TextChunker(
            chunk_size=500,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=mock_embedder,
            semantic_threshold=0.5,
            min_chunk_size=5,  # Low threshold so short text is not filtered
        )
        chunks = chunker.chunk_document(short_document)

        # Should still produce chunks (via recursive fallback)
        assert len(chunks) >= 1
        # Embedder should NOT be called (fallback bypasses embedding)
        mock_embedder.embed_texts.assert_not_called()

    def test_semantic_threshold_affects_splits(
        self, topic_shift_embedder, multi_topic_document
    ):
        """Test that lower threshold produces fewer splits."""
        chunker_strict = TextChunker(
            chunk_size=2000,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=topic_shift_embedder,
            semantic_threshold=0.99,  # Very strict: almost always splits
        )
        chunker_loose = TextChunker(
            chunk_size=2000,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=topic_shift_embedder,
            semantic_threshold=0.01,  # Very loose: almost never splits
        )

        chunks_strict = chunker_strict.chunk_document(multi_topic_document)
        chunks_loose = chunker_loose.chunk_document(multi_topic_document)

        # Strict threshold should produce at least as many chunks as loose
        assert len(chunks_strict) >= len(chunks_loose)

    def test_semantic_chunks_have_valid_metadata(
        self, mock_embedder, multi_topic_document
    ):
        """Test semantic chunks carry correct metadata."""
        chunker = TextChunker(
            chunk_size=1000,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=mock_embedder,
        )
        chunks = chunker.chunk_document(multi_topic_document)

        for chunk in chunks:
            assert chunk.metadata.get("filename") == "mixed_topics.txt"
            assert chunk.metadata.get("doc_type") == "txt"

    def test_semantic_chunk_ids_monotonic(self, mock_embedder, multi_topic_document):
        """Test semantic chunk IDs are monotonically increasing."""
        chunker = TextChunker(
            chunk_size=1000,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=mock_embedder,
        )
        chunks = chunker.chunk_document(multi_topic_document)

        # IDs should be monotonically increasing (may have gaps due to
        # min_chunk_size filtering in chunk_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) > 0
        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1]

    def test_empty_document_returns_empty(self, mock_embedder):
        """Test semantic chunking on empty document."""
        doc = Document(content="", source="empty.txt", doc_type="txt")
        chunker = TextChunker(
            chunk_size=500,
            overlap=50,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=mock_embedder,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_large_group_gets_sub_split(self, mock_embedder):
        """Test that a semantically coherent group exceeding chunk_size is sub-split."""
        # Create a document where all sentences are semantically similar
        # but total length exceeds chunk_size
        long_content = (
            ". ".join(
                [
                    f"Sentence number {i} about machine learning research"
                    for i in range(50)
                ]
            )
            + "."
        )
        doc = Document(
            content=long_content,
            source="long.txt",
            doc_type="txt",
        )

        chunker = TextChunker(
            chunk_size=200,  # Small chunk size forces sub-splitting
            overlap=30,
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=mock_embedder,
            semantic_threshold=0.01,  # Loose: no semantic splits
        )
        chunks = chunker.chunk_document(doc)

        # With loose threshold, no semantic splits, but sub-splitting should
        # still break the single large group into multiple chunks
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 300  # Some slack for sentence boundaries


# ── Non-semantic strategies still work ─────────────────────────


class TestOtherStrategiesUnchanged:
    """Verify existing strategies are not broken by semantic additions."""

    def test_fixed_size_still_works(self):
        doc = Document(
            content="A" * 500,
            source="test.txt",
            doc_type="txt",
        )
        chunker = TextChunker(
            chunk_size=100,
            overlap=20,
            strategy=ChunkingStrategy.FIXED_SIZE,
            min_chunk_size=10,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 4

    def test_recursive_still_works(self):
        doc = Document(
            content="First paragraph content here.\n\nSecond paragraph with more text.\n\nThird paragraph.",
            source="test.txt",
            doc_type="txt",
        )
        chunker = TextChunker(
            chunk_size=100,
            overlap=10,
            strategy=ChunkingStrategy.RECURSIVE,
            min_chunk_size=10,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_embedder_param_ignored_for_non_semantic(self):
        """Test embedder param doesn't interfere with other strategies."""
        mock_emb = MagicMock()
        doc = Document(
            content="Simple text for chunking.",
            source="test.txt",
            doc_type="txt",
        )
        chunker = TextChunker(
            chunk_size=500,
            overlap=50,
            strategy=ChunkingStrategy.RECURSIVE,
            embedder=mock_emb,  # Passed but not used
            min_chunk_size=5,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        mock_emb.embed_texts.assert_not_called()
