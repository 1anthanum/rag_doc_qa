"""Tests for the text chunker module."""

import pytest
from src.ingestion.loader import Document
from src.ingestion.chunker import TextChunker, ChunkingStrategy


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        content=(
            "The attention mechanism is a key component of transformers. "
            "It allows the model to focus on relevant parts of the input.\n\n"
            "Self-attention computes queries, keys, and values from the same "
            "input sequence. The dot-product attention score determines how "
            "much each position attends to every other position.\n\n"
            "Multi-head attention runs multiple attention operations in "
            "parallel, allowing the model to capture different types of "
            "relationships simultaneously."
        ),
        source="test.txt",
        doc_type="txt",
        metadata={"filename": "test.txt"},
    )


@pytest.fixture
def long_document():
    """Create a long document for chunk boundary testing."""
    paragraphs = [
        f"Paragraph {i}. " + "This is filler text. " * 20
        for i in range(10)
    ]
    return Document(
        content="\n\n".join(paragraphs),
        source="long.txt",
        doc_type="txt",
        metadata={"filename": "long.txt"},
    )


class TestTextChunker:
    """Tests for TextChunker."""

    def test_fixed_size_chunking(self, sample_document):
        """Test fixed-size chunking produces valid chunks."""
        chunker = TextChunker(
            chunk_size=100,
            overlap=20,
            strategy=ChunkingStrategy.FIXED_SIZE,
        )
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) > 0
            assert chunk.source == "test.txt"

    def test_sentence_chunking(self, sample_document):
        """Test sentence-based chunking respects sentence boundaries."""
        chunker = TextChunker(
            chunk_size=200,
            overlap=30,
            strategy=ChunkingStrategy.SENTENCE,
        )
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) >= 1
        # Each chunk should end at a sentence boundary (period)
        for chunk in chunks:
            text = chunk.text.strip()
            assert text[-1] in ".!?", (
                f"Chunk should end at sentence boundary: '{text[-20:]}'"
            )

    def test_recursive_chunking(self, sample_document):
        """Test recursive chunking tries paragraph boundaries first."""
        chunker = TextChunker(
            chunk_size=200,
            overlap=30,
            strategy=ChunkingStrategy.RECURSIVE,
        )
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_ids_are_sequential(self, sample_document):
        """Test that chunk IDs are assigned sequentially."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_document(sample_document)

        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunk_metadata_preserved(self, sample_document):
        """Test that source metadata is propagated to chunks."""
        chunker = TextChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert chunk.metadata.get("filename") == "test.txt"

    def test_empty_document(self):
        """Test chunking an empty document returns no chunks."""
        doc = Document(content="", source="empty.txt", doc_type="txt")
        chunker = TextChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_small_document_single_chunk(self):
        """Test that a small document produces exactly one chunk."""
        doc = Document(
            content="Short text.",
            source="short.txt",
            doc_type="txt",
        )
        chunker = TextChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_chunk_documents_multiple(self, sample_document, long_document):
        """Test chunking multiple documents."""
        chunker = TextChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk_documents([sample_document, long_document])

        sources = {c.source for c in chunks}
        assert "test.txt" in sources
        assert "long.txt" in sources

    def test_min_chunk_size_filter(self):
        """Test that chunks below min size are filtered out."""
        doc = Document(
            content="Word. " * 5 + "\n\n" + "A" * 200,
            source="test.txt",
            doc_type="txt",
        )
        chunker = TextChunker(
            chunk_size=100,
            overlap=0,
            min_chunk_size=50,
            strategy=ChunkingStrategy.RECURSIVE,
        )
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            assert len(chunk.text.strip()) >= 50 or len(chunks) == 1
