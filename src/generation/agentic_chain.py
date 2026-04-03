"""
Agentic RAG Chain: self-correcting retrieval with relevance evaluation.

Implements Corrective RAG (CRAG):
    1. Retrieve documents
    2. Evaluate relevance via LLM
    3. If irrelevant → rewrite query → re-retrieve (up to N rounds)
    4. Generate final answer from best retrieved context

Also supports:
    - Adaptive retrieval (skip retrieval for follow-up questions)
    - Multi-hop query decomposition (via QueryProcessor)
"""

import logging
from typing import Optional, Union

from ..retrieval.retriever import Retriever, RetrievalResult
from ..retrieval.hybrid_retriever import HybridRetriever
from .chain import RAGChain, RAGResponse
from .llm_client import LLMClient, GenerationResult
from .prompt_templates import RAGPrompt

logger = logging.getLogger(__name__)


class AgenticRAGChain(RAGChain):
    """
    Self-correcting RAG chain that evaluates retrieval quality
    and retries with rewritten queries when results are irrelevant.

    Inherits all functionality from RAGChain and adds:
        - Relevance evaluation after retrieval
        - Automatic query rewriting on poor results
        - Adaptive retrieval (skip for conversational follow-ups)

    Usage:
        chain = AgenticRAGChain(
            retriever=retriever,
            llm=llm_client,
            max_correction_rounds=2,
        )
        response = chain.query("What were Q3 milestones?")
    """

    def __init__(
        self,
        retriever: Union[Retriever, HybridRetriever],
        llm: LLMClient,
        mode: str = "qa",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        query_processor=None,
        max_correction_rounds: int = 2,
        relevance_threshold: float = 0.5,
        adaptive_retrieval: bool = True,
    ):
        super().__init__(
            retriever=retriever,
            llm=llm,
            mode=mode,
            temperature=temperature,
            max_tokens=max_tokens,
            query_processor=query_processor,
        )
        self.max_correction_rounds = max_correction_rounds
        self.relevance_threshold = relevance_threshold
        self.adaptive_retrieval = adaptive_retrieval

    def query(
        self,
        question: str,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute the agentic RAG pipeline with self-correction.

        Pipeline:
            1. (Adaptive) Check if retrieval is needed
            2. Retrieve → Evaluate relevance → Correct if needed
            3. Generate answer from best context
        """
        current_mode = mode or self.mode
        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Step 0: Adaptive retrieval check (for conversational mode)
        if (
            self.adaptive_retrieval
            and current_mode == "conversational"
            and self._chat_history
        ):
            needs_retrieval = self._needs_retrieval(question)
            if not needs_retrieval:
                logger.info("Adaptive retrieval: skipping retrieval for follow-up")
                return self._generate_without_retrieval(
                    question,
                    current_mode,
                    effective_temperature,
                    effective_max_tokens,
                )

        # Step 1: Corrective retrieval loop
        retrieval_result = self._corrective_retrieve(question, top_k=top_k)

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

        # Step 2: Generate answer (same as base RAGChain)
        template = RAGPrompt.get_template(current_mode)

        if current_mode == "conversational":
            history_str = self._format_chat_history()
            prompt = template.format(
                context=retrieval_result.context,
                question=question,
                chat_history=history_str,
            )
        else:
            prompt = template.format(
                context=retrieval_result.context,
                question=question,
            )

        gen_result = self.llm.generate(
            prompt=prompt,
            system_prompt=template.system_prompt,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
        )

        # Update chat history
        if current_mode == "conversational":
            self._chat_history.append({"role": "user", "content": question})
            self._chat_history.append({"role": "assistant", "content": gen_result.text})

        logger.info(
            f"Agentic RAG complete: mode={current_mode}, "
            f"chunks={len(retrieval_result.results)}"
        )

        return RAGResponse(
            answer=gen_result.text,
            query=question,
            sources=retrieval_result.sources,
            retrieval=retrieval_result,
            generation=gen_result,
            mode=current_mode,
        )

    def _corrective_retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Corrective RAG loop: retrieve → evaluate → rewrite → re-retrieve.

        Exits when:
            - Results are judged RELEVANT
            - Max correction rounds exhausted
            - No results found at all
        """
        current_query = question

        # First, apply query processor if available
        if self.query_processor:
            try:
                qr = self.query_processor.process(question)
                if qr.is_decomposed:
                    return self._multi_query_retrieve(qr.queries, top_k=top_k)
                current_query = qr.primary_query
            except Exception as e:
                logger.warning(f"Query processing failed: {e}")

        for round_num in range(self.max_correction_rounds + 1):
            # Retrieve
            retrieval_result = self.retriever.retrieve(current_query, top_k=top_k)

            if not retrieval_result.results:
                logger.info(f"CRAG round {round_num}: no results found")
                break

            # Evaluate relevance (skip on last round — just use what we have)
            if round_num < self.max_correction_rounds:
                relevance = self._evaluate_relevance(question, retrieval_result.context)

                if relevance == "relevant":
                    logger.info(f"CRAG round {round_num}: results judged RELEVANT")
                    return retrieval_result

                if relevance == "irrelevant":
                    logger.info(
                        f"CRAG round {round_num}: results judged IRRELEVANT, "
                        f"rewriting query..."
                    )
                    current_query = self._rewrite_query(question)
                else:
                    # Ambiguous — use current results but try to augment
                    logger.info(
                        f"CRAG round {round_num}: results judged AMBIGUOUS, "
                        f"attempting refinement..."
                    )
                    current_query = self._rewrite_query(question)
            else:
                logger.info("CRAG: max rounds reached, using current results")

        return retrieval_result

    def _evaluate_relevance(self, question: str, context: str) -> str:
        """
        Use LLM to judge whether retrieved context is relevant to the question.

        Returns: "relevant", "ambiguous", or "irrelevant"
        """
        template = RAGPrompt.RELEVANCE_EVAL
        prompt = template.format(question=question, context=context)

        try:
            result = self.llm.generate(
                prompt=prompt,
                system_prompt=template.system_prompt,
                temperature=0.0,
                max_tokens=16,
            )
            judgment = result.text.strip().lower()

            if "relevant" in judgment and "irrelevant" not in judgment:
                return "relevant"
            elif "irrelevant" in judgment:
                return "irrelevant"
            else:
                return "ambiguous"

        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return "ambiguous"  # Default to ambiguous on error

    def _rewrite_query(self, question: str) -> str:
        """Use LLM to rewrite a query for better retrieval."""
        template = RAGPrompt.QUERY_REWRITE
        prompt = template.format(question=question)

        try:
            result = self.llm.generate(
                prompt=prompt,
                system_prompt=template.system_prompt,
                temperature=0.3,
                max_tokens=256,
            )
            rewritten = result.text.strip()
            logger.info(f"CRAG rewrite: '{question[:40]}' → '{rewritten[:40]}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return question

    def _needs_retrieval(self, question: str) -> bool:
        """
        Adaptive retrieval: determine if the question needs document search
        or can be answered from conversation context alone.
        """
        template = RAGPrompt.NEEDS_RETRIEVAL
        history_str = self._format_chat_history()
        prompt = template.format(
            chat_history=history_str,
            question=question,
        )

        try:
            result = self.llm.generate(
                prompt=prompt,
                system_prompt=template.system_prompt,
                temperature=0.0,
                max_tokens=16,
            )
            decision = result.text.strip().upper()
            return "RETRIEVE" in decision
        except Exception as e:
            logger.warning(f"Adaptive retrieval check failed: {e}")
            return True  # Default: always retrieve

    def _generate_without_retrieval(
        self,
        question: str,
        mode: str,
        temperature: float,
        max_tokens: int,
    ) -> RAGResponse:
        """Generate answer using only chat history (no retrieval)."""
        history_str = self._format_chat_history()
        prompt = (
            f"Previous conversation:\n{history_str}\n\n"
            f"User: {question}\n"
            f"Assistant:"
        )

        gen_result = self.llm.generate(
            prompt=prompt,
            system_prompt=RAGPrompt.SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._chat_history.append({"role": "user", "content": question})
        self._chat_history.append({"role": "assistant", "content": gen_result.text})

        return RAGResponse(
            answer=gen_result.text,
            query=question,
            sources=[],
            retrieval=RetrievalResult(query=question, results=[]),
            generation=gen_result,
            mode=mode,
        )
