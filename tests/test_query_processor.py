"""Tests for the query processor module."""

import pytest
from unittest.mock import MagicMock

from src.generation.llm_client import GenerationResult
from src.retrieval.query_processor import (
    QueryProcessor,
    QueryResult,
    QueryStrategy,
)

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock()
    llm.generate.return_value = GenerationResult(
        text="What are the specific budget allocations for Q3?",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 50, "completion_tokens": 15},
    )
    return llm


@pytest.fixture
def none_processor(mock_llm):
    """QueryProcessor with strategy=none."""
    return QueryProcessor(llm=mock_llm, strategy="none")


@pytest.fixture
def rewrite_processor(mock_llm):
    """QueryProcessor with strategy=rewrite."""
    return QueryProcessor(llm=mock_llm, strategy="rewrite")


@pytest.fixture
def hyde_processor(mock_llm):
    """QueryProcessor with strategy=hyde."""
    mock_llm.generate.return_value = GenerationResult(
        text="The Q3 budget allocated $500K to engineering and $200K to marketing.",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 60, "completion_tokens": 30},
    )
    return QueryProcessor(llm=mock_llm, strategy="hyde")


@pytest.fixture
def decompose_processor(mock_llm):
    """QueryProcessor with strategy=decompose."""
    mock_llm.generate.return_value = GenerationResult(
        text="1. What was the total Q3 budget?\n2. How was the budget allocated across departments?\n3. Were there any budget overruns?",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 70, "completion_tokens": 40},
    )
    return QueryProcessor(llm=mock_llm, strategy="decompose")


# ── QueryStrategy Tests ────────────────────────────────────────


class TestQueryStrategy:
    """Tests for QueryStrategy enum."""

    def test_valid_strategies(self):
        assert QueryStrategy("none") == QueryStrategy.NONE
        assert QueryStrategy("rewrite") == QueryStrategy.REWRITE
        assert QueryStrategy("hyde") == QueryStrategy.HYDE
        assert QueryStrategy("decompose") == QueryStrategy.DECOMPOSE

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            QueryStrategy("invalid")


# ── QueryResult Tests ──────────────────────────────────────────


class TestQueryResult:
    """Tests for QueryResult data class."""

    def test_primary_query_returns_first(self):
        qr = QueryResult(
            original="What is X?",
            queries=["Rewritten: what is X specifically?"],
            strategy="rewrite",
        )
        assert qr.primary_query == "Rewritten: what is X specifically?"

    def test_primary_query_fallback_to_original(self):
        qr = QueryResult(original="What is X?", queries=[], strategy="none")
        assert qr.primary_query == "What is X?"

    def test_is_decomposed_true(self):
        qr = QueryResult(
            original="Complex Q",
            queries=["Sub Q1", "Sub Q2"],
            strategy="decompose",
        )
        assert qr.is_decomposed is True

    def test_is_decomposed_false_single_query(self):
        qr = QueryResult(
            original="Simple Q",
            queries=["Simple Q"],
            strategy="decompose",
        )
        assert qr.is_decomposed is False

    def test_is_decomposed_false_wrong_strategy(self):
        qr = QueryResult(
            original="Q",
            queries=["A", "B"],
            strategy="rewrite",
        )
        assert qr.is_decomposed is False

    def test_hyde_hypothesis_stored(self):
        qr = QueryResult(
            original="What is X?",
            queries=["Hypothetical passage about X."],
            strategy="hyde",
            hyde_hypothesis="Hypothetical passage about X.",
        )
        assert qr.hyde_hypothesis == "Hypothetical passage about X."


# ── QueryProcessor Tests ───────────────────────────────────────


class TestQueryProcessor:
    """Tests for QueryProcessor."""

    def test_none_strategy_passthrough(self, none_processor):
        """Test none strategy returns original query unchanged."""
        result = none_processor.process("What is the budget?")

        assert result.original == "What is the budget?"
        assert result.queries == ["What is the budget?"]
        assert result.strategy == "none"

    def test_none_strategy_does_not_call_llm(self, none_processor, mock_llm):
        """Test none strategy doesn't invoke LLM."""
        none_processor.process("Any question")
        mock_llm.generate.assert_not_called()

    def test_rewrite_calls_llm(self, rewrite_processor, mock_llm):
        """Test rewrite strategy calls LLM."""
        result = rewrite_processor.process("What's the budget?")

        mock_llm.generate.assert_called_once()
        assert result.strategy == "rewrite"
        assert len(result.queries) == 1
        assert result.queries[0] != result.original

    def test_rewrite_preserves_original(self, rewrite_processor):
        """Test rewrite keeps original query in result."""
        result = rewrite_processor.process("What's the budget?")
        assert result.original == "What's the budget?"

    def test_hyde_generates_hypothesis(self, hyde_processor):
        """Test HyDE generates a hypothetical document passage."""
        result = hyde_processor.process("What was Q3 spending?")

        assert result.strategy == "hyde"
        assert result.hyde_hypothesis is not None
        assert len(result.hyde_hypothesis) > 0
        assert result.queries[0] == result.hyde_hypothesis

    def test_decompose_produces_sub_questions(self, decompose_processor):
        """Test decompose splits into multiple sub-questions."""
        result = decompose_processor.process("Tell me everything about Q3 budget")

        assert result.strategy == "decompose"
        assert len(result.queries) >= 2
        assert result.is_decomposed is True

    def test_decompose_parses_numbered_lines(self, decompose_processor):
        """Test decompose correctly parses numbered output."""
        result = decompose_processor.process("Complex question")

        # Each sub-question should be a clean string (no leading numbers)
        for q in result.queries:
            assert not q[0].isdigit()

    def test_rewrite_query_method(self, rewrite_processor, mock_llm):
        """Test the rewrite_query convenience method."""
        mock_llm.generate.return_value = GenerationResult(
            text="Optimized query text",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 30, "completion_tokens": 10},
        )
        rewritten = rewrite_processor.rewrite_query("Original query")
        assert rewritten == "Optimized query text"

    def test_invalid_strategy_falls_back_to_none(self, mock_llm):
        """Test unknown strategy string falls back to NONE."""
        processor = QueryProcessor(llm=mock_llm, strategy="nonexistent")
        assert processor.strategy == QueryStrategy.NONE

    def test_llm_failure_returns_original(self, mock_llm):
        """Test LLM error falls back to original query."""
        mock_llm.generate.side_effect = RuntimeError("API error")
        processor = QueryProcessor(llm=mock_llm, strategy="rewrite")

        result = processor.process("Some question")
        assert result.queries == ["Some question"]
        assert result.strategy == "rewrite"

    def test_hyde_failure_returns_original(self, mock_llm):
        """Test HyDE LLM error falls back to original query."""
        mock_llm.generate.side_effect = RuntimeError("API error")
        processor = QueryProcessor(llm=mock_llm, strategy="hyde")

        result = processor.process("Some question")
        assert result.queries == ["Some question"]

    def test_decompose_failure_returns_original(self, mock_llm):
        """Test decompose LLM error falls back to original query."""
        mock_llm.generate.side_effect = RuntimeError("API error")
        processor = QueryProcessor(llm=mock_llm, strategy="decompose")

        result = processor.process("Some question")
        assert result.queries == ["Some question"]

    def test_decompose_empty_output_returns_original(self, mock_llm):
        """Test decompose with no parseable sub-questions returns original."""
        mock_llm.generate.return_value = GenerationResult(
            text="",  # Empty response
            model="gpt-4o-mini",
            usage={"prompt_tokens": 30, "completion_tokens": 0},
        )
        processor = QueryProcessor(llm=mock_llm, strategy="decompose")
        result = processor.process("Complex question")

        assert result.queries == ["Complex question"]
