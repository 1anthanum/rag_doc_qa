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
            "\"I cannot find this information in the provided documents.\" "
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
                f"Unknown mode: {mode}. "
                f"Available: {list(templates.keys())}"
            )
        return templates[mode]
