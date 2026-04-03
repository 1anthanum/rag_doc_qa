"""
Query processor for optimizing user queries before retrieval.
Supports rewriting, HyDE (Hypothetical Document Embeddings), and decomposition.
"""

import logging
from enum import Enum
from typing import List, Optional

from ..generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Available query processing strategies."""

    NONE = "none"
    REWRITE = "rewrite"
    HYDE = "hyde"
    DECOMPOSE = "decompose"


# ── Prompt templates for query processing ──────────────────────────

REWRITE_PROMPT = (
    "You are a search query optimizer. Rewrite the following user question "
    "to be more specific, detailed, and effective for document retrieval. "
    "Expand abbreviations, add relevant synonyms, and clarify ambiguous terms.\n\n"
    "Original question: {question}\n\n"
    "Rewritten query (respond with ONLY the rewritten query, nothing else):"
)

HYDE_PROMPT = (
    "You are a knowledgeable assistant. Given the following question, write "
    "a short, factual paragraph that would be a plausible answer found in "
    "a document. Do NOT say you don't know — generate a hypothetical answer "
    "as if it were written in a relevant document.\n\n"
    "Question: {question}\n\n"
    "Hypothetical document passage:"
)

DECOMPOSE_PROMPT = (
    "You are a research assistant. Break down the following complex question "
    "into 2-4 simpler sub-questions that, when answered together, would "
    "fully address the original question.\n\n"
    "Complex question: {question}\n\n"
    "Output each sub-question on a separate line, prefixed with a number. "
    "Example:\n"
    "1. What is X?\n"
    "2. How does X relate to Y?\n\n"
    "Sub-questions:"
)


class QueryProcessor:
    """
    Optimizes user queries before retrieval to improve recall and precision.

    Strategies:
        - none: Pass through unchanged
        - rewrite: LLM rewrites the query to be more specific
        - hyde: LLM generates a hypothetical answer, used as the search query
                (better alignment with document embeddings)
        - decompose: LLM splits complex queries into sub-questions

    Usage:
        processor = QueryProcessor(llm=llm_client, strategy="hyde")
        optimized = processor.process("What's the budget?")
        # optimized.queries contains the processed query/queries
    """

    def __init__(
        self,
        llm: LLMClient,
        strategy: str = "none",
        max_retries: int = 1,
    ):
        try:
            self.strategy = QueryStrategy(strategy)
        except ValueError:
            logger.warning(
                f"Unknown query strategy: {strategy}. Falling back to 'none'."
            )
            self.strategy = QueryStrategy.NONE

        self.llm = llm
        self.max_retries = max_retries

    def process(self, question: str) -> "QueryResult":
        """
        Process a query according to the configured strategy.

        Args:
            question: Original user question.

        Returns:
            QueryResult with processed query/queries.
        """
        if self.strategy == QueryStrategy.NONE:
            return QueryResult(
                original=question,
                queries=[question],
                strategy=self.strategy.value,
            )
        elif self.strategy == QueryStrategy.REWRITE:
            return self._rewrite(question)
        elif self.strategy == QueryStrategy.HYDE:
            return self._hyde(question)
        elif self.strategy == QueryStrategy.DECOMPOSE:
            return self._decompose(question)
        else:
            return QueryResult(
                original=question,
                queries=[question],
                strategy="none",
            )

    def rewrite_query(self, question: str) -> str:
        """
        Rewrite a single query using LLM. Used by CRAG for query refinement.

        Args:
            question: Original question.

        Returns:
            Rewritten query string.
        """
        result = self._rewrite(question)
        return result.queries[0]

    def _rewrite(self, question: str) -> "QueryResult":
        """Rewrite the query to be more search-friendly."""
        try:
            prompt = REWRITE_PROMPT.format(question=question)
            gen = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=256,
            )
            rewritten = gen.text.strip()

            logger.info(
                f"Query rewritten: '{question[:40]}...' → '{rewritten[:40]}...'"
            )
            return QueryResult(
                original=question,
                queries=[rewritten],
                strategy="rewrite",
            )
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}. Using original.")
            return QueryResult(
                original=question,
                queries=[question],
                strategy="rewrite",
            )

    def _hyde(self, question: str) -> "QueryResult":
        """
        HyDE: Generate a hypothetical document passage, then use it
        as the retrieval query. The hypothesis is closer in embedding
        space to actual document passages than the original question.
        """
        try:
            prompt = HYDE_PROMPT.format(question=question)
            gen = self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=256,
            )
            hypothesis = gen.text.strip()

            logger.info(
                f"HyDE generated hypothesis for: '{question[:40]}...' "
                f"({len(hypothesis)} chars)"
            )
            # Return both: hypothesis for embedding search,
            # original for BM25 keyword search
            return QueryResult(
                original=question,
                queries=[hypothesis],
                hyde_hypothesis=hypothesis,
                strategy="hyde",
            )
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Using original.")
            return QueryResult(
                original=question,
                queries=[question],
                strategy="hyde",
            )

    def _decompose(self, question: str) -> "QueryResult":
        """Decompose a complex question into sub-questions."""
        try:
            prompt = DECOMPOSE_PROMPT.format(question=question)
            gen = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
            )

            # Parse numbered sub-questions
            sub_questions = []
            for line in gen.text.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove leading number and punctuation
                    q = line.lstrip("0123456789.)-: ").strip()
                    if q:
                        sub_questions.append(q)

            if not sub_questions:
                sub_questions = [question]

            logger.info(
                f"Decomposed into {len(sub_questions)} sub-questions: "
                f"'{question[:40]}...'"
            )
            return QueryResult(
                original=question,
                queries=sub_questions,
                strategy="decompose",
            )
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}. Using original.")
            return QueryResult(
                original=question,
                queries=[question],
                strategy="decompose",
            )


class QueryResult:
    """Result of query processing."""

    def __init__(
        self,
        original: str,
        queries: List[str],
        strategy: str = "none",
        hyde_hypothesis: Optional[str] = None,
    ):
        self.original = original
        self.queries = queries
        self.strategy = strategy
        self.hyde_hypothesis = hyde_hypothesis

    @property
    def primary_query(self) -> str:
        """The main query to use for retrieval."""
        return self.queries[0] if self.queries else self.original

    @property
    def is_decomposed(self) -> bool:
        return self.strategy == "decompose" and len(self.queries) > 1
