"""
Text chunking strategies for RAG pipeline.
Splits documents into overlapping chunks optimized for embedding and retrieval.
"""

import re
import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .embedder import BaseEmbedder

from .loader import Document

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    """A text chunk with provenance metadata."""

    text: str
    chunk_id: int
    source: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (1 token ~ 4 chars for English)."""
        return len(self.text) // 4


class TextChunker:
    """
    Splits documents into overlapping text chunks.

    Three strategies are available:
        - fixed_size:  Split by character count with overlap
        - sentence:    Split on sentence boundaries
        - recursive:   Try paragraph → sentence → character boundaries

    Usage:
        chunker = TextChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk_document(document)
    """

    # Sentence boundary pattern
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff])")

    # Paragraph boundary pattern
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        min_chunk_size: int = 50,
        embedder: Optional["BaseEmbedder"] = None,
        semantic_threshold: float = 0.5,
    ):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self._embedder = embedder
        self.semantic_threshold = semantic_threshold

        if strategy == ChunkingStrategy.SEMANTIC and embedder is None:
            raise ValueError(
                "Semantic chunking requires an embedder. "
                "Pass embedder=... to TextChunker."
            )

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into chunks using the configured strategy."""
        text = document.content.strip()
        if not text:
            return []

        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            raw_chunks = self._fixed_size_split(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            raw_chunks = self._sentence_split(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            raw_chunks = self._recursive_split(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            raw_chunks = self._semantic_split(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        chunks = []
        for i, (chunk_text, start, end) in enumerate(raw_chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=chunk_text.strip(),
                        chunk_id=i,
                        source=document.source,
                        start_char=start,
                        end_char=end,
                        metadata={
                            "doc_type": document.doc_type,
                            "filename": document.metadata.get("filename", ""),
                        },
                    )
                )

        logger.info(
            f"Chunked '{document.metadata.get('filename', '?')}' → "
            f"{len(chunks)} chunks (strategy={self.strategy.value})"
        )
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents, maintaining global chunk IDs."""
        all_chunks = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            # Re-index globally
            for chunk in doc_chunks:
                chunk.chunk_id = len(all_chunks)
                all_chunks.append(chunk)
        return all_chunks

    # ── Splitting strategies ─────────────────────────────────────

    def _fixed_size_split(self, text: str) -> List[tuple]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append((text[start:end], start, end))
            start += self.chunk_size - self.overlap
        return chunks

    def _sentence_split(self, text: str) -> List[tuple]:
        """Split on sentence boundaries, grouping into chunk_size."""
        sentences = self.SENTENCE_PATTERN.split(text)
        chunks = []
        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    end = current_start + len(current_chunk)
                    chunks.append((current_chunk, current_start, end))
                    # Overlap: keep tail of previous chunk
                    overlap_text = current_chunk[-self.overlap :]
                    current_start = end - len(overlap_text)
                    current_chunk = overlap_text
            current_chunk += sentence

        if current_chunk.strip():
            end = current_start + len(current_chunk)
            chunks.append((current_chunk, current_start, end))

        return chunks

    def _recursive_split(self, text: str) -> List[tuple]:
        """
        Recursively split: try paragraphs first, then sentences,
        then fall back to fixed-size.
        """
        # First, split by paragraphs
        paragraphs = self.PARAGRAPH_PATTERN.split(text)

        chunks = []
        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                # Current paragraph would overflow
                if current_chunk:
                    end = current_start + len(current_chunk)
                    chunks.append((current_chunk, current_start, end))
                    current_start = end

                # If single paragraph exceeds chunk_size, split by sentences
                if len(para) > self.chunk_size:
                    sub_chunks = self._sentence_split(para)
                    for sub_text, sub_start, sub_end in sub_chunks:
                        chunks.append(
                            (
                                sub_text,
                                current_start + sub_start,
                                current_start + sub_end,
                            )
                        )
                    current_start += len(para)
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk.strip():
            end = current_start + len(current_chunk)
            chunks.append((current_chunk, current_start, end))

        return chunks

    def _semantic_split(self, text: str) -> List[tuple]:
        """
        Semantic chunking: split at points where topic/meaning shifts.

        Algorithm:
            1. Split text into sentences
            2. Embed each sentence
            3. Compute cosine similarity between adjacent sentence embeddings
            4. Split at points where similarity drops below threshold
            5. Group resulting segments, respecting max chunk_size

        This produces more coherent chunks than mechanical splitting because
        boundaries align with actual topic transitions in the text.
        """
        # Split into sentences
        sentence_ends = list(self.SENTENCE_PATTERN.finditer(text))
        sentences = []
        positions = []
        prev_end = 0

        for match in sentence_ends:
            sent = text[prev_end : match.start()].strip()
            if sent:
                sentences.append(sent)
                positions.append((prev_end, match.start()))
            prev_end = match.end()

        # Last sentence (after final split point)
        last = text[prev_end:].strip()
        if last:
            sentences.append(last)
            positions.append((prev_end, len(text)))

        # Fallback: if fewer than 3 sentences, use recursive split
        if len(sentences) < 3:
            return self._recursive_split(text)

        # Embed all sentences
        embeddings = self._embedder.embed_texts(sentences)

        # Compute cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                sim = float(np.dot(a, b) / (norm_a * norm_b))
            else:
                sim = 0.0
            similarities.append(sim)

        # Find split points where similarity drops below threshold
        split_indices = []
        for i, sim in enumerate(similarities):
            if sim < self.semantic_threshold:
                split_indices.append(i + 1)  # Split AFTER sentence i

        # Build groups of sentences
        groups = []
        prev_idx = 0
        for split_idx in split_indices:
            groups.append((prev_idx, split_idx))
            prev_idx = split_idx
        groups.append((prev_idx, len(sentences)))

        # Convert groups to chunks, respecting max chunk_size
        chunks = []
        for group_start, group_end in groups:
            group_text = " ".join(sentences[group_start:group_end])
            char_start = positions[group_start][0]
            char_end = positions[group_end - 1][1]

            if len(group_text) <= self.chunk_size:
                chunks.append((group_text, char_start, char_end))
            else:
                # Group too large: sub-split by sentences within the group
                sub_chunks = self._sentence_split(group_text)
                for sub_text, sub_s, sub_e in sub_chunks:
                    chunks.append((sub_text, char_start + sub_s, char_start + sub_e))

        logger.info(
            f"Semantic chunking: {len(sentences)} sentences → "
            f"{len(chunks)} chunks "
            f"(threshold={self.semantic_threshold})"
        )
        return chunks
