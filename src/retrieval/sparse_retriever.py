"""
BM25 sparse retriever for keyword-based document search.
Complements dense vector retrieval by excelling at exact keyword matching.
"""

import logging
import re
from typing import List

from ..ingestion.chunker import Chunk
from .vector_store import SearchResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25-based sparse retriever using rank_bm25.

    BM25 excels at exact keyword matching and is complementary to
    dense vector retrieval which captures semantic similarity.

    Usage:
        bm25 = BM25Retriever()
        bm25.index(chunks)
        results = bm25.search("budget allocation", top_k=10)
    """

    # Regex tokenizer for Latin/English text
    TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

    # CJK character ranges for language detection
    _CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

    # jieba availability (lazy-loaded)
    _jieba = None
    _jieba_checked = False

    def __init__(self):
        self._bm25 = None
        self._chunks: List[Chunk] = []
        self._corpus: List[List[str]] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    @classmethod
    def _get_jieba(cls):
        """Lazy-load jieba (optional dependency)."""
        if not cls._jieba_checked:
            try:
                import jieba

                jieba.setLogLevel(logging.WARNING)  # Suppress jieba init logs
                cls._jieba = jieba
                logger.info("jieba loaded — Chinese word segmentation enabled")
            except ImportError:
                cls._jieba = None
                logger.debug(
                    "jieba not installed — falling back to regex tokenizer "
                    "for Chinese text. Install: pip install jieba"
                )
            cls._jieba_checked = True
        return cls._jieba

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase tokens.

        Uses jieba for Chinese text (if available) and regex for English.
        Mixed text is handled by detecting CJK characters first.
        """
        has_cjk = bool(self._CJK_PATTERN.search(text))

        if has_cjk:
            jieba_mod = self._get_jieba()
            if jieba_mod is not None:
                # jieba word segmentation: proper Chinese tokenization
                return [
                    t.lower()
                    for t in jieba_mod.lcut(text)
                    if len(t.strip()) > 1
                    and not re.match(r"^[\s\W]+$", t)  # filter punctuation
                ]

        # Fallback: regex tokenizer (English or no-jieba Chinese)
        return [
            t.lower()
            for t in self.TOKEN_PATTERN.findall(text)
            if len(t) > 1  # Skip single-character tokens
        ]

    def index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 required for hybrid search. "
                "Install: pip install rank-bm25"
            )

        self._chunks = list(chunks)
        self._corpus = [self._tokenize(c.text) for c in self._chunks]

        if not self._corpus:
            self._bm25 = None
            logger.info("No chunks to index — BM25 index not built")
            return

        self._bm25 = BM25Okapi(self._corpus)

        logger.info(f"Built BM25 index with {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search the BM25 index for relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult ordered by BM25 score (descending).
        """
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    SearchResult(
                        chunk=self._chunks[idx],
                        score=float(scores[idx]),
                    )
                )

        logger.debug(f"BM25 found {len(results)} results for: '{query[:50]}...'")
        return results

    def clear(self) -> None:
        """Clear the BM25 index."""
        self._bm25 = None
        self._chunks.clear()
        self._corpus.clear()
