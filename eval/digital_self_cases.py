"""
Evaluation dataset for Digital Self (conversation-based) RAG.

Contains sample conversations and behavioral test cases to validate that:
    1. ConversationLoader correctly chunks dialogue data
    2. Behavioral HyDE generates conversation-style hypotheticals
    3. Retrieval finds relevant conversation turns (not just keywords)
    4. Generated answers reflect the user's actual stated views/preferences

Categories:
    - preference: Questions about user preferences (food, tools, etc.)
    - opinion: Questions about opinions/viewpoints
    - experience: Questions about past experiences or decisions
    - behavior: Questions about habits or patterns
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BehavioralEvalCase:
    """A single behavioral evaluation test case."""

    question: str
    answer_keywords: List[str]  # Keywords the answer must contain
    chunk_keywords: List[str]  # Keywords that should appear in retrieved chunks
    difficulty: str = "medium"  # easy | medium | hard
    category: str = "preference"  # preference | opinion | experience | behavior


# ── Sample Conversations ──────────────────────────────────────

CONV_TECH_PREFERENCES = [
    {
        "role": "user",
        "content": "I've been thinking about switching from VS Code to Neovim. "
        "I love the speed but I'm worried about the learning curve.",
        "timestamp": "2024-06-15T09:30:00",
    },
    {
        "role": "assistant",
        "content": "That's a common concern. Many developers find the initial "
        "2-3 weeks challenging but then become significantly faster. "
        "Would you like some tips for a gradual transition?",
        "timestamp": "2024-06-15T09:30:15",
    },
    {
        "role": "user",
        "content": "Yes! I already use vim keybindings in VS Code so I'm "
        "somewhat familiar. My main concern is losing the LSP and "
        "debugging integration. I use Python and TypeScript daily.",
        "timestamp": "2024-06-15T09:31:00",
    },
    {
        "role": "assistant",
        "content": "Great news — Neovim has excellent LSP support via "
        "nvim-lspconfig, and for debugging there's nvim-dap. "
        "For Python and TypeScript specifically, pyright and "
        "typescript-language-server work well.",
        "timestamp": "2024-06-15T09:31:30",
    },
    {
        "role": "user",
        "content": "That sounds promising. I think I'll try it for a side "
        "project first rather than switching cold turkey for work. "
        "I don't want to slow down during deadlines.",
        "timestamp": "2024-06-15T09:32:00",
    },
    {
        "role": "assistant",
        "content": "Smart approach. You could also use the VS Code Neovim "
        "extension as an intermediate step.",
        "timestamp": "2024-06-15T09:32:15",
    },
]

CONV_WORK_PHILOSOPHY = [
    {
        "role": "user",
        "content": "I've been doing more code reviews lately and I'm finding "
        "that being too strict makes junior devs afraid to submit PRs. "
        "But being too lenient lets bugs through.",
        "timestamp": "2024-07-20T14:00:00",
    },
    {
        "role": "assistant",
        "content": "That's a real tension. What's your current approach to "
        "balancing rigor with encouragement?",
        "timestamp": "2024-07-20T14:00:15",
    },
    {
        "role": "user",
        "content": "I try to separate 'must fix' from 'suggestion'. I prefix "
        "blocking comments with 'BLOCKER:' and optional feedback with "
        "'nit:' or 'consider:'. It helps set expectations.",
        "timestamp": "2024-07-20T14:01:00",
    },
    {
        "role": "assistant",
        "content": "That's a solid framework. The explicit prefixes remove "
        "ambiguity and help reviewees prioritize.",
        "timestamp": "2024-07-20T14:01:20",
    },
    {
        "role": "user",
        "content": "Yeah, and I always start with something positive. Even if "
        "the PR needs major rework, I'll acknowledge the approach or "
        "the effort. Nobody learns from pure criticism.",
        "timestamp": "2024-07-20T14:02:00",
    },
    {
        "role": "assistant",
        "content": "Starting with positives creates psychological safety. "
        "Research shows people are more receptive to constructive "
        "feedback when they feel their contributions are valued.",
        "timestamp": "2024-07-20T14:02:30",
    },
    {
        "role": "user",
        "content": "Exactly. I think mentoring is the most important part of "
        "a senior developer's job. Writing code is just the baseline.",
        "timestamp": "2024-07-20T14:03:00",
    },
]

CONV_PROJECT_DECISIONS = [
    {
        "role": "user",
        "content": "For our new microservice, I'm debating between FastAPI and "
        "Express. The team knows both, but our infra is mostly Python.",
        "timestamp": "2024-08-10T11:00:00",
    },
    {
        "role": "assistant",
        "content": "Since your infrastructure is Python-based, FastAPI would "
        "minimize context switching and share dependencies. What are "
        "the service's requirements?",
        "timestamp": "2024-08-10T11:00:20",
    },
    {
        "role": "user",
        "content": "It's a REST API for real-time data processing. High "
        "throughput, low latency. We need WebSocket support too. "
        "FastAPI's async support is great but Express has more "
        "WebSocket battle-testing in production.",
        "timestamp": "2024-08-10T11:01:00",
    },
    {
        "role": "assistant",
        "content": "FastAPI handles WebSockets natively and its async "
        "performance is comparable to Node.js for I/O-bound tasks. "
        "For CPU-bound processing, Python might need worker pools.",
        "timestamp": "2024-08-10T11:01:30",
    },
    {
        "role": "user",
        "content": "Good point. I think I'll go with FastAPI. Keeping the stack "
        "uniform matters more than marginal performance differences. "
        "Plus our monitoring and deployment are all set up for Python.",
        "timestamp": "2024-08-10T11:02:00",
    },
    {
        "role": "assistant",
        "content": "Solid reasoning. Operational simplicity often outweighs "
        "micro-benchmarks. You can always optimize hot paths later.",
        "timestamp": "2024-08-10T11:02:20",
    },
]


# ── Test Cases ────────────────────────────────────────────────

BEHAVIORAL_EVAL_CASES: List[BehavioralEvalCase] = [
    # --- Easy: answer directly stated in a single conversation ---
    BehavioralEvalCase(
        question="What editor is the user thinking about switching to?",
        answer_keywords=["Neovim", "vim"],
        chunk_keywords=["Neovim", "VS Code", "switching"],
        difficulty="easy",
        category="preference",
    ),
    BehavioralEvalCase(
        question="What programming languages does the user work with?",
        answer_keywords=["Python", "TypeScript"],
        chunk_keywords=["Python", "TypeScript", "daily"],
        difficulty="easy",
        category="preference",
    ),
    BehavioralEvalCase(
        question="What framework did the user choose for the new microservice?",
        answer_keywords=["FastAPI"],
        chunk_keywords=["FastAPI", "microservice"],
        difficulty="easy",
        category="experience",
    ),
    # --- Medium: requires synthesis from conversation context ---
    BehavioralEvalCase(
        question="How does the user approach code reviews?",
        answer_keywords=["BLOCKER", "positive"],
        chunk_keywords=["code review", "BLOCKER", "nit", "positive"],
        difficulty="medium",
        category="behavior",
    ),
    BehavioralEvalCase(
        question="What is the user's philosophy on mentoring?",
        answer_keywords=["mentor", "senior"],
        chunk_keywords=["mentoring", "senior", "important"],
        difficulty="medium",
        category="opinion",
    ),
    BehavioralEvalCase(
        question="Why did the user choose FastAPI over Express?",
        answer_keywords=["Python", "stack", "uniform"],
        chunk_keywords=["FastAPI", "Express", "infrastructure", "Python"],
        difficulty="medium",
        category="experience",
    ),
    # --- Hard: cross-conversation reasoning ---
    BehavioralEvalCase(
        question="What does the user value most in technical decisions?",
        answer_keywords=["simplicity"],
        chunk_keywords=["uniform", "operational", "gradual"],
        difficulty="hard",
        category="opinion",
    ),
    BehavioralEvalCase(
        question="How does the user handle learning new tools?",
        answer_keywords=["gradual", "side project"],
        chunk_keywords=["side project", "cold turkey", "gradual"],
        difficulty="hard",
        category="behavior",
    ),
]


# ── Conversation registry ─────────────────────────────────────

CONVERSATIONS = {
    "tech_preferences.json": CONV_TECH_PREFERENCES,
    "work_philosophy.json": CONV_WORK_PHILOSOPHY,
    "project_decisions.json": CONV_PROJECT_DECISIONS,
}
