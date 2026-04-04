"""
Evaluation dataset: documents + questions + ground truth annotations.

Each test case includes:
    - A question
    - Ground-truth keywords that a correct answer MUST contain
    - Ground-truth chunk keywords: terms that should appear in the retrieved chunks
    - Difficulty level: easy (direct keyword match), medium (requires inference),
      hard (requires cross-paragraph reasoning or ambiguous phrasing)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalCase:
    """A single evaluation test case."""

    question: str
    answer_keywords: List[str]  # Keywords the answer must contain
    chunk_keywords: List[str]  # Keywords that should appear in retrieved chunks
    difficulty: str = "medium"  # easy | medium | hard
    category: str = "factual"  # factual | comparison | reasoning | ambiguous


# ── Documents ────────────────────────────────────────────────────

DOC_TRANSFORMER = """\
# Attention Is All You Need — Key Concepts

## Introduction
The Transformer architecture was introduced by Vaswani et al. in 2017, \
replacing recurrence entirely with self-attention. Unlike RNNs and LSTMs \
that process tokens sequentially, Transformers handle all positions in \
parallel. This parallelization dramatically reduces training time — the \
base Transformer trained in 12 hours on 8 GPUs, compared to weeks for \
comparable RNN models.

## Self-Attention Mechanism
Self-attention computes a weighted sum of value vectors (V), where the \
weights come from the compatibility between query (Q) and key (K) vectors. \
The formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. The \
scaling factor sqrt(d_k) prevents the dot products from becoming too large, \
which would push softmax into regions with extremely small gradients.

Multi-head attention runs h parallel attention functions (h=8 in the base \
model), each operating on d_model/h dimensions. This allows the model to \
attend to information from different representation subspaces simultaneously.

## Positional Encoding
Since Transformers have no built-in notion of sequence order, positional \
encodings are added to the input embeddings. The original paper used \
sinusoidal functions: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)). \
Learned positional embeddings performed equally well in experiments, but \
sinusoidal encodings can extrapolate to longer sequences than seen during \
training.

## Architecture Details
The encoder uses 6 identical layers. Each layer has two sub-layers: \
multi-head self-attention and a position-wise feed-forward network \
(two linear transformations with ReLU activation, inner dimension 2048). \
Residual connections and layer normalization wrap each sub-layer: \
LayerNorm(x + Sublayer(x)).

The decoder also uses 6 layers, adding a third sub-layer: cross-attention \
over the encoder output. The decoder self-attention is masked to prevent \
positions from attending to future tokens (autoregressive property).

## Training Configuration
The model was trained using Adam optimizer with a custom learning rate \
schedule: warmup over 4000 steps, then decay proportional to the inverse \
square root of the step number. Label smoothing (epsilon=0.1) improved \
BLEU scores despite slightly hurting perplexity. Dropout of 0.1 was \
applied to attention weights and feed-forward sub-layers.

## Results and Comparison
On WMT 2014 English-German translation, the big Transformer achieved \
28.4 BLEU — exceeding all previous models including ensembles. On \
English-French, it achieved 41.0 BLEU while requiring only 25% of the \
training cost of the previous state-of-the-art.

The base model has 65 million parameters (d_model=512, h=8, N=6). \
The big model has 213 million parameters (d_model=1024, h=16, N=6). \
Even the base model outperformed all previous single models.
"""

DOC_RAG = """\
# Retrieval-Augmented Generation (RAG) — Overview

## What is RAG?
RAG (Retrieval-Augmented Generation) combines a retrieval system with a \
language model. Instead of relying solely on parametric knowledge stored \
in model weights, RAG first retrieves relevant documents from an external \
knowledge base, then conditions the generation on both the query and the \
retrieved context. This was first proposed by Lewis et al. (2020) at \
Facebook AI Research.

## Why RAG Matters
Large language models hallucinate: they generate plausible-sounding but \
factually incorrect information. RAG mitigates this by grounding \
generation in actual source documents. It also enables easy knowledge \
updates (just add new documents) without expensive model retraining. \
For enterprise use, RAG provides attribution — every answer can be \
traced back to its source documents.

## Dense Retrieval
Dense retrieval encodes both queries and documents as dense vectors \
using neural encoders (e.g., DPR, Contriever, or sentence-transformers). \
Retrieval is performed via approximate nearest neighbor search (FAISS, \
ScaNN, or Annoy). The key advantage over BM25 is semantic matching: \
"automobile" retrieves "car" documents. The downside is that dense \
retrieval can miss exact keyword matches — "GPT-4o" might not retrieve \
documents mentioning that specific model name.

## Sparse Retrieval (BM25)
BM25 is a bag-of-words ranking function based on TF-IDF. It excels at \
exact keyword matching and requires no neural computation. BM25's formula \
considers term frequency, inverse document frequency, and document length \
normalization. It remains a strong baseline and outperforms dense \
retrieval on keyword-heavy queries (model names, error codes, IDs).

## Hybrid Search
Hybrid search combines dense and sparse retrieval. The most popular \
fusion method is Reciprocal Rank Fusion (RRF), which scores each \
document as: RRF_score = sum(1 / (k + rank_i)) across all result lists. \
RRF is parameter-free (k=60 by convention) and doesn't require score \
normalization, making it more robust than CombSUM or learned weights.

Research by Ma et al. (2023) showed hybrid search improves recall by \
15-25% compared to dense-only, especially on technical documents with \
acronyms and domain-specific terminology.

## Advanced Techniques
HyDE (Hypothetical Document Embeddings) generates a hypothetical answer \
first, then uses it as the retrieval query. This bridges the semantic gap \
between short questions and long document passages.

Corrective RAG (CRAG) by Yan et al. (2024) adds a self-evaluation step: \
after retrieval, an LLM judges whether the context is relevant. If not, \
the query is rewritten and retrieval is retried. This catch-and-correct \
loop reduces "garbage in, garbage out" failures.

Query decomposition breaks complex questions into simpler sub-questions, \
retrieves for each, and merges the results. This is effective for \
multi-hop reasoning questions that span multiple document sections.

## Chunking Strategies
Documents must be split into chunks for embedding. Fixed-size chunking \
(every N tokens) is simple but can split sentences or paragraphs mid-thought. \
Recursive chunking tries paragraph → sentence → character boundaries. \
Semantic chunking uses embedding similarity to find natural topic \
boundaries — adjacent sentences with low cosine similarity indicate a \
topic shift and create a split point.

Chunk size typically ranges from 256 to 1024 tokens. Smaller chunks are \
more precise but lose surrounding context. Overlap (typically 10-15% of \
chunk size) ensures important boundary information isn't lost.
"""

DOC_EVALUATION = """\
# Evaluating RAG Systems

## Key Metrics
RAG evaluation involves both retrieval and generation metrics. Retrieval \
is measured by Recall@K (fraction of relevant documents found in top K), \
Precision@K (fraction of top K results that are relevant), and MRR \
(Mean Reciprocal Rank — reciprocal of the rank of the first relevant result).

Generation quality is harder to measure. Common approaches include \
keyword overlap with gold answers, BERTScore (embedding-based similarity), \
and LLM-as-judge (using GPT-4 to grade answers on a 1-5 scale). Each \
has tradeoffs: keyword overlap is fast but misses paraphrasing, BERTScore \
is semantic but noisy, LLM-as-judge is accurate but expensive and \
non-deterministic.

## Ragas Framework
Ragas (Retrieval-Augmented Generation Assessment) provides four core \
metrics: faithfulness (is the answer grounded in context?), answer \
relevance (does it address the question?), context relevance (are \
retrieved chunks useful?), and context recall (do chunks cover the \
ground truth?). These can be computed automatically with LLM-based \
evaluation.

## Common Failure Modes
The most common RAG failure is "retrieval miss" — the right document \
exists but wasn't retrieved. This typically happens with keyword-heavy \
queries or when chunk boundaries split a relevant paragraph.

"Context poisoning" occurs when irrelevant but high-scoring chunks \
contaminate the context, leading the LLM to generate an answer from \
wrong information. This is particularly dangerous because the LLM may \
present incorrect information confidently.

"Lost in the middle" refers to the tendency of LLMs to focus on the \
beginning and end of the context, ignoring relevant information in the \
middle. Placing the most relevant chunk first mitigates this.

## End-to-End Evaluation Best Practices
The gold standard is human evaluation: domain experts rate answer \
correctness, completeness, and relevance. For automated evaluation, \
a combination of retrieval metrics (Recall@5) and generation metrics \
(keyword + LLM-judge) provides the best coverage.

Important: always evaluate the full pipeline end-to-end, not just \
individual components. A system with 80% retrieval recall and good \
generation often outperforms one with 95% recall but poor prompting. \
The components interact in non-obvious ways.
"""

# ── Test Cases ───────────────────────────────────────────────────

EVAL_CASES: List[EvalCase] = [
    # --- Easy: direct keyword match, answer in a single chunk ---
    EvalCase(
        question="What is the formula for self-attention?",
        answer_keywords=["softmax", "QK", "sqrt", "d_k"],
        chunk_keywords=["softmax", "query", "key", "value"],
        difficulty="easy",
        category="factual",
    ),
    EvalCase(
        question="How many parameters does the base Transformer have?",
        answer_keywords=["65 million", "d_model=512"],
        chunk_keywords=["65 million", "base model"],
        difficulty="easy",
        category="factual",
    ),
    EvalCase(
        question="What BLEU score did the Transformer achieve on English-German?",
        answer_keywords=["28.4"],
        chunk_keywords=["28.4", "BLEU", "English-German"],
        difficulty="easy",
        category="factual",
    ),
    # --- Medium: requires understanding, may span chunks ---
    EvalCase(
        question="Why is BM25 sometimes better than dense retrieval?",
        answer_keywords=["keyword", "exact"],
        chunk_keywords=["BM25", "keyword", "dense"],
        difficulty="medium",
        category="reasoning",
    ),
    EvalCase(
        question="How does RRF handle the score normalization problem?",
        answer_keywords=["rank"],
        chunk_keywords=["RRF", "rank", "normalization"],
        difficulty="medium",
        category="reasoning",
    ),
    EvalCase(
        question="What is the purpose of the scaling factor in self-attention?",
        answer_keywords=["gradient", "softmax"],
        chunk_keywords=["sqrt", "d_k", "softmax", "gradient"],
        difficulty="medium",
        category="reasoning",
    ),
    EvalCase(
        question="Compare fixed-size chunking with semantic chunking.",
        answer_keywords=["topic", "boundary"],
        chunk_keywords=["fixed-size", "semantic", "chunk"],
        difficulty="medium",
        category="comparison",
    ),
    # --- Hard: ambiguous phrasing, cross-document reasoning ---
    EvalCase(
        question="What approaches exist to reduce hallucination in LLMs?",
        answer_keywords=["RAG", "retrieval", "ground"],
        chunk_keywords=["hallucinate", "RAG", "ground"],
        difficulty="hard",
        category="reasoning",
    ),
    EvalCase(
        question="How do you evaluate if a RAG system's retrieval is working well?",
        answer_keywords=["recall", "precision"],
        chunk_keywords=["Recall", "Precision", "MRR"],
        difficulty="hard",
        category="reasoning",
    ),
    EvalCase(
        question="What is the difference between HyDE and query rewriting?",
        answer_keywords=["hypothetical"],
        chunk_keywords=["HyDE", "hypothetical", "rewrite"],
        difficulty="hard",
        category="comparison",
    ),
    EvalCase(
        question="What causes garbage-in garbage-out in RAG?",
        answer_keywords=["irrelevant", "retrieval"],
        chunk_keywords=["CRAG", "irrelevant", "retrieval"],
        difficulty="hard",
        category="ambiguous",
    ),
    EvalCase(
        question="Why can attention models handle longer sequences than RNNs?",
        answer_keywords=["parallel"],
        chunk_keywords=["parallel", "sequential", "RNN"],
        difficulty="hard",
        category="reasoning",
    ),
]


DOCUMENTS = {
    "transformer.md": DOC_TRANSFORMER,
    "rag_overview.md": DOC_RAG,
    "evaluation.md": DOC_EVALUATION,
}
