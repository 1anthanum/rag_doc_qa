#!/usr/bin/env python3
"""
RAG Document Q&A — Interactive Demo

A self-contained demo that walks through the full pipeline step by step,
printing intermediate results so you can see exactly how each component works.

Usage:
    # With OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    python demo.py

    # Fully local (Ollama + local embeddings)
    LLM_PROVIDER=ollama python demo.py

    # Custom document
    python demo.py --file path/to/your/document.pdf

    # Enable advanced features
    python demo.py --hybrid --hyde --crag
"""

import os
import time
import argparse
import textwrap
import logging

logging.basicConfig(level=logging.WARNING)

# ── Helpers ───────────────────────────────────────────────────────

DIVIDER = "─" * 60
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BLUE}{BOLD}{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}{RESET}\n")


def step(number: int, title: str) -> None:
    print(f"{GREEN}[Step {number}]{RESET} {BOLD}{title}{RESET}")


def info(label: str, value) -> None:
    print(f"  {CYAN}{label}:{RESET} {value}")


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


# ── Sample document (used when no file is provided) ──────────────

SAMPLE_DOCUMENT = """\
# Attention Is All You Need — Summary

## Introduction
The Transformer architecture, introduced by Vaswani et al. in 2017, \
revolutionized natural language processing by replacing recurrence with \
self-attention mechanisms. Unlike RNNs and LSTMs, Transformers process \
all positions in a sequence simultaneously, enabling massive parallelization.

## Self-Attention Mechanism
Self-attention computes a weighted sum of value vectors, where the weights \
are determined by the compatibility between query and key vectors. The \
formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. Multi-head \
attention runs several attention functions in parallel, allowing the model \
to attend to information from different representation subspaces.

## Architecture Details
The Transformer uses an encoder-decoder structure. The encoder consists of \
6 identical layers, each with multi-head self-attention and a feed-forward \
network. The decoder adds a third sub-layer for cross-attention over the \
encoder output. Residual connections and layer normalization are applied \
around each sub-layer. Positional encodings (sinusoidal functions) are \
added to input embeddings to inject sequence order information.

## Training and Results
The model was trained on WMT 2014 English-German and English-French \
translation tasks. It achieved 28.4 BLEU on English-German (a new \
state-of-the-art) and 41.0 BLEU on English-French, while requiring \
significantly less training time than previous models. The base model \
has 65 million parameters; the big model has 213 million.

## Impact
The Transformer became the foundation for BERT, GPT, T5, and virtually \
all modern large language models. Its self-attention mechanism proved \
superior to recurrence for capturing long-range dependencies, and its \
parallelizable architecture enabled training on much larger datasets.
"""


def create_sample_file(tmp_dir: str) -> str:
    """Write the sample document to a temp file and return its path."""
    path = os.path.join(tmp_dir, "transformer_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_DOCUMENT)
    return path


# ── Pipeline construction ────────────────────────────────────────


def build_pipeline(args):
    """Build the RAG pipeline based on CLI flags."""
    from src.config import get as cfg
    from src.ingestion import DocumentLoader, TextChunker, ChunkingStrategy
    from src.ingestion.embedder import EmbeddingEngine
    from src.retrieval import FAISSVectorStore, Retriever, HybridRetriever
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import RAGChain, AgenticRAGChain
    from src.generation.llm_client import OpenAIClient, OllamaClient

    # LLM
    provider = cfg("generation.provider", "openai")
    model = cfg("generation.model", "gpt-4o-mini")
    if provider == "ollama":
        llm = OllamaClient(model=model)
    else:
        llm = OpenAIClient(model=model)

    # Embedder
    emb_provider = cfg("ingestion.embedding.provider", "local")
    emb_model = cfg("ingestion.embedding.model", "all-MiniLM-L6-v2")
    engine = EmbeddingEngine(provider=emb_provider, model_name=emb_model)

    # Chunker
    strategy = ChunkingStrategy.RECURSIVE
    chunker_kw = {"chunk_size": 512, "overlap": 64, "strategy": strategy}

    # Vector store
    store = FAISSVectorStore(dimension=engine.dimension)

    # Retriever
    if args.hybrid:
        retriever = HybridRetriever(
            embedding_engine=engine,
            vector_store=store,
            top_k=5,
            dense_weight=0.7,
            sparse_weight=0.3,
            rrf_k=60,
        )
    else:
        retriever = Retriever(
            embedding_engine=engine,
            vector_store=store,
            top_k=5,
        )

    # Query processor
    qp = None
    if args.hyde:
        qp = QueryProcessor(llm=llm, strategy="hyde")
    elif args.rewrite:
        qp = QueryProcessor(llm=llm, strategy="rewrite")

    # Chain
    if args.crag:
        chain = AgenticRAGChain(
            retriever=retriever,
            llm=llm,
            query_processor=qp,
            max_correction_rounds=2,
            adaptive_retrieval=True,
        )
    else:
        chain = RAGChain(
            retriever=retriever,
            llm=llm,
            query_processor=qp,
        )

    return {
        "loader": DocumentLoader(),
        "chunker": TextChunker(**chunker_kw),
        "engine": engine,
        "store": store,
        "retriever": retriever,
        "chain": chain,
        "llm_info": f"{provider}/{model}",
        "emb_info": f"{emb_provider}/{emb_model}",
    }


# ── Demo steps ───────────────────────────────────────────────────


def demo_ingest(pipeline, file_path: str):
    """Step 1-3: Load → Chunk → Embed → Index."""
    loader = pipeline["loader"]
    chunker = pipeline["chunker"]
    engine = pipeline["engine"]
    store = pipeline["store"]
    retriever = pipeline["retriever"]

    # Load
    step(1, "Load Document")
    doc = loader.load_file(file_path)
    info("Source", doc.source)
    info("Length", f"{len(doc.content):,} characters")
    print(f"  {dim('Preview: ' + doc.content[:120].replace(chr(10), ' ') + '...')}")

    # Chunk
    step(2, "Chunk Document")
    t0 = time.time()
    chunks = chunker.chunk_documents([doc])
    dt = time.time() - t0
    info("Strategy", chunker.strategy.value)
    info("Chunks", f"{len(chunks)} (chunk_size=512, overlap=64)")
    info("Time", f"{dt:.3f}s")
    print()
    for i, c in enumerate(chunks[:3]):
        preview = c.text[:80].replace("\n", " ")
        print(f"  {DIM}chunk[{c.chunk_id}]{RESET} ({len(c.text)} chars): {preview}...")
    if len(chunks) > 3:
        print(f"  {dim(f'  ... and {len(chunks) - 3} more chunks')}")

    # Embed
    step(3, "Embed & Index")
    t0 = time.time()
    embedded = engine.embed_chunks(chunks)
    store.add(embedded)
    dt = time.time() - t0
    info("Model", pipeline["emb_info"])
    info("Dimension", engine.dimension)
    info("Indexed", f"{store.size} chunks in {dt:.3f}s")

    # BM25 index for hybrid
    from src.retrieval.hybrid_retriever import HybridRetriever

    if isinstance(retriever, HybridRetriever):
        retriever.index_sparse(chunks)
        info("BM25", f"Built sparse index ({len(chunks)} documents)")

    return chunks


def demo_retrieve(pipeline, question: str, chunks):
    """Step 4: Retrieve relevant chunks."""
    retriever = pipeline["retriever"]
    from src.retrieval.hybrid_retriever import HybridRetriever

    step(4, "Retrieve")
    info("Question", question)

    is_hybrid = isinstance(retriever, HybridRetriever)
    info("Mode", "Hybrid (BM25 + Dense + RRF)" if is_hybrid else "Dense (FAISS)")

    t0 = time.time()
    result = retriever.retrieve(question)
    dt = time.time() - t0

    info("Results", f"{len(result.results)} chunks in {dt:.3f}s")
    print()

    for i, r in enumerate(result.results):
        preview = r.chunk.text[:100].replace("\n", " ")
        score_bar = "█" * int(r.score * 20) + "░" * (20 - int(r.score * 20))
        print(f"  {YELLOW}#{i+1}{RESET} score={r.score:.4f} [{score_bar}]")
        print(f"     {dim(preview + '...')}")

    return result


def demo_generate(pipeline, question: str, args):
    """Step 5: Generate answer via RAG chain."""
    chain = pipeline["chain"]

    step(5, "Generate Answer")
    info("LLM", pipeline["llm_info"])
    info("Mode", "Corrective RAG (CRAG)" if args.crag else "Standard RAG")

    if pipeline["chain"].query_processor:
        strategy = pipeline["chain"].query_processor.strategy.value
        info("Query Optimization", strategy.upper())

    print()
    t0 = time.time()
    response = chain.query(question=question, mode="qa")
    dt = time.time() - t0

    # Print answer
    print(f"  {GREEN}{BOLD}Answer:{RESET}")
    wrapped = textwrap.fill(
        response.answer, width=76, initial_indent="  ", subsequent_indent="  "
    )
    print(wrapped)

    print()
    info("Sources", ", ".join(response.sources) if response.sources else "N/A")
    info("Tokens", response.generation.usage)
    info("Time", f"{dt:.1f}s")

    return response


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RAG Document Q&A — Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to a document (PDF, TXT, MD, DOCX). Uses built-in sample if omitted.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Question to ask. Defaults to a sample question about the document.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search (BM25 + dense vectors + RRF fusion).",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Enable HyDE (Hypothetical Document Embeddings) query optimization.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Enable query rewriting optimization.",
    )
    parser.add_argument(
        "--crag",
        action="store_true",
        help="Enable Corrective RAG (CRAG) self-correction loop.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all advanced features (hybrid + HyDE + CRAG).",
    )
    args = parser.parse_args()

    if args.all:
        args.hybrid = True
        args.hyde = True
        args.crag = True

    # Header
    header("RAG Document Q&A — Demo")

    features = []
    if args.hybrid:
        features.append("Hybrid Search")
    if args.hyde:
        features.append("HyDE")
    elif args.rewrite:
        features.append("Query Rewrite")
    if args.crag:
        features.append("CRAG")
    info("Features", ", ".join(features) if features else "Standard (baseline)")
    print()

    # Build pipeline
    print(f"{DIM}Initializing pipeline...{RESET}")
    t_total = time.time()
    pipeline = build_pipeline(args)
    print(f"{DIM}Pipeline ready ({time.time() - t_total:.1f}s){RESET}")
    print()

    # Ingest
    import tempfile

    if args.file:
        file_path = args.file
    else:
        tmp_dir = tempfile.mkdtemp()
        file_path = create_sample_file(tmp_dir)
        print(f"{DIM}Using built-in sample: Transformer architecture summary{RESET}")

    print(DIVIDER)
    chunks = demo_ingest(pipeline, file_path)

    # Retrieve
    question = (
        args.question or "How does self-attention work and why is it better than RNNs?"
    )
    print()
    print(DIVIDER)
    demo_retrieve(pipeline, question, chunks)

    # Generate
    print()
    print(DIVIDER)
    demo_generate(pipeline, question, args)

    # Summary
    print()
    header("Done")
    dt_total = time.time() - t_total
    info("Total time", f"{dt_total:.1f}s")
    info(
        "Pipeline",
        f"{'Hybrid' if args.hybrid else 'Dense'} retrieval → "
        f"{'HyDE' if args.hyde else 'Rewrite' if args.rewrite else 'No'} query opt → "
        f"{'CRAG' if args.crag else 'Standard'} generation",
    )

    # Interactive mode
    print(f"\n{DIVIDER}")
    print(f"{BOLD}Interactive mode{RESET} — type a question (or 'quit' to exit)\n")

    while True:
        try:
            q = input(f"{CYAN}Question:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q or q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        t0 = time.time()
        resp = pipeline["chain"].query(question=q, mode="qa")
        dt = time.time() - t0

        print(f"\n{GREEN}{BOLD}Answer:{RESET}")
        print(
            textwrap.fill(
                resp.answer, width=76, initial_indent="  ", subsequent_indent="  "
            )
        )
        print(f"\n  {DIM}({len(resp.retrieval.results)} chunks | {dt:.1f}s){RESET}\n")


if __name__ == "__main__":
    main()
