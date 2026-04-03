"""
RAG Chain: end-to-end orchestration of retrieval + generation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from ..retrieval.retriever import Retriever, RetrievalResult
from ..retrieval.hybrid_retriever import HybridRetriever
from .llm_client import LLMClient, GenerationResult
from .prompt_templates import RAGPrompt

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG response with full provenance."""

    answer: str
    query: str
    sources: List[str]
    retrieval: RetrievalResult
    generation: GenerationResult
    mode: str

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": self.sources,
            "mode": self.mode,
            "model": self.generation.model,
            "num_chunks_used": len(self.retrieval.results),
            "token_usage": self.generation.usage,
        }


class RAGChain:
    """
    End-to-end RAG pipeline: Query → Retrieve → Generate → Response.

    Supports multiple modes:
        - qa:             Standard question answering
        - summarize:      Summarize retrieved content
        - compare:        Compare information across sources
        - conversational: Multi-turn with chat history

    Usage:
        chain = RAGChain(retriever=retriever, llm=llm_client)
        response = chain.query("What is attention mechanism?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(
        self,
        retriever: Union[Retriever, HybridRetriever],
        llm: LLMClient,
        mode: str = "qa",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        query_processor=None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.query_processor = query_processor
        self._chat_history: List[dict] = []

    def query(
        self,
        question: str,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            question: User's question
            mode: Override the default mode for this query
            temperature: Override default temperature (thread-safe)
            max_tokens: Override default max_tokens (thread-safe)
            top_k: Override retriever top_k (thread-safe)

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        current_mode = mode or self.mode
        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Step 0 (optional): Query optimization
        search_query = question
        if self.query_processor:
            try:
                qr = self.query_processor.process(question)
                if qr.is_decomposed:
                    # Multi-query: retrieve for each sub-question, merge results
                    retrieval_result = self._multi_query_retrieve(
                        qr.queries, top_k=top_k
                    )
                else:
                    search_query = qr.primary_query
                    retrieval_result = self.retriever.retrieve(
                        search_query, top_k=top_k
                    )
                logger.info(
                    f"Query processed ({qr.strategy}): "
                    f"'{question[:40]}...' → '{search_query[:40]}...'"
                )
            except Exception as e:
                logger.warning(f"Query processing failed: {e}. Using original query.")
                retrieval_result = self.retriever.retrieve(question, top_k=top_k)
        else:
            # Step 1: Retrieve relevant chunks
            retrieval_result = self.retriever.retrieve(question, top_k=top_k)

        if not retrieval_result.results:
            return RAGResponse(
                answer="I could not find any relevant information "
                "in the provided documents.",
                query=question,
                sources=[],
                retrieval=retrieval_result,
                generation=GenerationResult(text="", model="none", usage={}),
                mode=current_mode,
            )

        # Step 2: Build prompt from template
        template = RAGPrompt.get_template(current_mode)

        if current_mode == "conversational":
            history_str = self._format_chat_history()
            prompt = template.format(
                context=retrieval_result.context,
                question=question,
                chat_history=history_str,
            )
        elif current_mode in ("qa", "compare"):
            prompt = template.format(
                context=retrieval_result.context,
                question=question,
            )
        else:  # summarize
            prompt = template.format(
                context=retrieval_result.context,
                question=question,
            )

        # Step 3: Generate answer
        gen_result = self.llm.generate(
            prompt=prompt,
            system_prompt=template.system_prompt,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
        )

        # Step 4: Update chat history if conversational
        if current_mode == "conversational":
            self._chat_history.append({"role": "user", "content": question})
            self._chat_history.append({"role": "assistant", "content": gen_result.text})

        logger.info(
            f"RAG query complete: mode={current_mode}, "
            f"chunks={len(retrieval_result.results)}, "
            f"tokens={gen_result.usage.get('total_tokens', '?')}"
        )

        return RAGResponse(
            answer=gen_result.text,
            query=question,
            sources=retrieval_result.sources,
            retrieval=retrieval_result,
            generation=gen_result,
            mode=current_mode,
        )

    def _multi_query_retrieve(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve for multiple sub-queries and merge results (dedup by chunk_id).
        Used when QueryProcessor decomposes a complex question.
        """
        from ..retrieval.vector_store import SearchResult

        seen_keys = set()
        all_results: List[SearchResult] = []

        for q in queries:
            result = self.retriever.retrieve(q, top_k=top_k)
            for sr in result.results:
                key = f"{sr.chunk.source}::{sr.chunk.chunk_id}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(sr)

        # Sort by score descending and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        effective_top_k = top_k if top_k is not None else self.retriever.top_k
        all_results = all_results[:effective_top_k]

        return RetrievalResult(
            query=" | ".join(queries),
            results=all_results,
        )

    def clear_history(self):
        """Clear conversation history."""
        self._chat_history.clear()

    def _format_chat_history(self, max_turns: int = 5) -> str:
        """Format recent chat history for the prompt."""
        recent = self._chat_history[-(max_turns * 2) :]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines) if lines else "(No previous conversation)"
