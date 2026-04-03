"""
Prompt templates for RAG generation.
Separates prompt engineering from generation logic.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution."""

    template: str
    system_prompt: Optional[str] = None

    def format(self, **kwargs) -> str:
        """Fill in template variables."""
        return self.template.format(**kwargs)


class RAGPrompt:
    """
    Pre-built prompt templates for RAG use cases.

    Templates enforce grounded answering: the model must use
    provided context and acknowledge when information is insufficient.
    """

    SYSTEM_PROMPT = (
        "You are a precise, helpful research assistant. "
        "Answer questions based ONLY on the provided context. "
        "If the context doesn't contain enough information, "
        "say so clearly. Always cite which source(s) you used."
    )

    # ── Standard Q&A ─────────────────────────────────────────────

    QA = PromptTemplate(
        system_prompt=SYSTEM_PROMPT,
        template=(
            "Context:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Question: {question}\n\n"
            "Instructions: Answer the question using ONLY the context above. "
            "If the answer is not in the context, say "
            '"I cannot find this information in the provided documents." '
            "Cite the source number(s) used."
        ),
    )

    # ── Summarization ────────────────────────────────────────────

    SUMMARIZE = PromptTemplate(
        system_prompt=SYSTEM_PROMPT,
        template=(
            "Context:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "User request: {question}\n\n"
            "Task: Provide a concise summary of the key points from "
            "the context above, guided by the user's request. "
            "Organize by topic if multiple sources "
            "cover different areas. Cite source numbers."
        ),
    )

    # ── Comparison ───────────────────────────────────────────────

    COMPARE = PromptTemplate(
        system_prompt=SYSTEM_PROMPT,
        template=(
            "Context:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Question: {question}\n\n"
            "Instructions: Compare and contrast the relevant information "
            "from the context. Highlight agreements, disagreements, and "
            "unique contributions from each source. Cite source numbers."
        ),
    )

    # ── Multi-turn conversation ──────────────────────────────────

    CONVERSATIONAL = PromptTemplate(
        system_prompt=(
            "You are a helpful research assistant engaged in a "
            "multi-turn conversation about documents. "
            "Use the provided context to answer. "
            "Maintain conversation continuity."
        ),
        template=(
            "Previous conversation:\n"
            "{chat_history}\n\n"
            "Context:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "User: {question}\n"
            "Assistant:"
        ),
    )

    # ── Agentic RAG: Relevance Evaluation ─────────────────────────

    RELEVANCE_EVAL = PromptTemplate(
        system_prompt=(
            "You are a relevance judge. Evaluate whether the retrieved "
            "context is relevant to the user's question."
        ),
        template=(
            "Question: {question}\n\n"
            "Retrieved Context:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Evaluate the relevance of the context to the question. "
            "Respond with exactly ONE of these words:\n"
            "- RELEVANT: The context contains information that can answer the question\n"
            "- AMBIGUOUS: The context is partially related but may not fully answer\n"
            "- IRRELEVANT: The context does not contain useful information\n\n"
            "Your evaluation (one word only):"
        ),
    )

    # ── Agentic RAG: Query Rewrite for Correction ──────────────

    QUERY_REWRITE = PromptTemplate(
        system_prompt=(
            "You are a search query optimizer. The previous search did not "
            "return relevant results. Rewrite the query to improve retrieval."
        ),
        template=(
            "Original question: {question}\n\n"
            "The previous search returned irrelevant results. "
            "Rewrite this question using different keywords, synonyms, "
            "or alternative phrasing to improve document retrieval. "
            "Be specific and detailed.\n\n"
            "Rewritten query (respond with ONLY the rewritten query):"
        ),
    )

    # ── Agentic RAG: Adaptive Retrieval Check ──────────────────

    NEEDS_RETRIEVAL = PromptTemplate(
        system_prompt=(
            "You are a helpful assistant that determines whether a question "
            "requires document retrieval or can be answered from conversation context."
        ),
        template=(
            "Chat history:\n{chat_history}\n\n"
            "New question: {question}\n\n"
            "Does this question require searching the document index, or can it "
            "be answered from the conversation context alone?\n"
            "Respond with exactly: RETRIEVE or NO_RETRIEVE"
        ),
    )

    @classmethod
    def get_template(cls, mode: str = "qa") -> PromptTemplate:
        """Get a prompt template by mode name."""
        templates = {
            "qa": cls.QA,
            "summarize": cls.SUMMARIZE,
            "compare": cls.COMPARE,
            "conversational": cls.CONVERSATIONAL,
        }
        if mode not in templates:
            raise ValueError(
                f"Unknown mode: {mode}. " f"Available: {list(templates.keys())}"
            )
        return templates[mode]
