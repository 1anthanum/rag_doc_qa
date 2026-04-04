#!/usr/bin/env python3
"""
RAG Benchmark: automated end-to-end evaluation across configurations.

Compares retrieval quality and answer accuracy across different pipeline
configurations (baseline, hybrid, hybrid+HyDE, hybrid+HyDE+CRAG).

Usage:
    export OPENAI_API_KEY="sk-..."
    python -m eval.benchmark                    # Run all configs
    python -m eval.benchmark --configs baseline hybrid   # Specific configs
    python -m eval.benchmark --retrieval-only   # Skip generation (no LLM cost)
    python -m eval.benchmark --output report.md # Save report to file

Retrieval-only mode evaluates chunk recall without calling the LLM,
making it free and fast — useful for tuning retrieval parameters.
"""

import os
import sys
import time
import json
import logging
import argparse
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.dataset import DOCUMENTS, EVAL_CASES, EvalCase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Configuration Profiles ───────────────────────────────────────

CONFIGS = {
    "baseline": {
        "label": "Baseline (Dense only)",
        "hybrid": False,
        "query_strategy": "none",
        "agentic": False,
    },
    "hybrid": {
        "label": "Hybrid (BM25 + Dense + RRF)",
        "hybrid": True,
        "query_strategy": "none",
        "agentic": False,
    },
    "hybrid_hyde": {
        "label": "Hybrid + HyDE",
        "hybrid": True,
        "query_strategy": "hyde",
        "agentic": False,
    },
    "hybrid_hyde_crag": {
        "label": "Hybrid + HyDE + CRAG",
        "hybrid": True,
        "query_strategy": "hyde",
        "agentic": True,
    },
}


# ── Metrics ──────────────────────────────────────────────────────


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics for a single query."""

    chunk_recall: float  # Fraction of expected keywords found in chunks
    top1_relevant: bool  # Was the top-1 chunk relevant?
    num_results: int
    latency_ms: float


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics for a single query."""

    keyword_coverage: float  # Fraction of expected answer keywords found
    answer_length: int
    latency_ms: float


@dataclass
class CaseResult:
    """Full evaluation result for a single case."""

    question: str
    difficulty: str
    category: str
    retrieval: RetrievalMetrics
    generation: Optional[GenerationMetrics] = None


@dataclass
class ConfigResult:
    """Aggregated results for a single configuration."""

    config_name: str
    config_label: str
    cases: List[CaseResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def avg_chunk_recall(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.retrieval.chunk_recall for c in self.cases) / len(self.cases)

    @property
    def avg_top1_hit(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.retrieval.top1_relevant) / len(self.cases)

    @property
    def avg_keyword_coverage(self) -> float:
        gen_cases = [c for c in self.cases if c.generation is not None]
        if not gen_cases:
            return 0.0
        return sum(c.generation.keyword_coverage for c in gen_cases) / len(gen_cases)

    @property
    def avg_retrieval_latency(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.retrieval.latency_ms for c in self.cases) / len(self.cases)

    @property
    def avg_total_latency(self) -> float:
        gen_cases = [c for c in self.cases if c.generation is not None]
        if not gen_cases:
            return self.avg_retrieval_latency
        return sum(
            c.retrieval.latency_ms + c.generation.latency_ms for c in gen_cases
        ) / len(gen_cases)

    def by_difficulty(self, diff: str) -> "ConfigResult":
        """Filter results by difficulty level."""
        filtered = ConfigResult(self.config_name, self.config_label)
        filtered.cases = [c for c in self.cases if c.difficulty == diff]
        return filtered


# ── Pipeline Builder ─────────────────────────────────────────────


def build_pipeline(config: dict, retrieval_only: bool = False):
    """Build a RAG pipeline from a configuration profile."""
    from src.ingestion import DocumentLoader, TextChunker, ChunkingStrategy
    from src.ingestion.embedder import EmbeddingEngine
    from src.retrieval import FAISSVectorStore, Retriever, HybridRetriever

    # Use config module for defaults
    from src.config import get as cfg

    # LLM: only needed for generation, query processing, or agentic mode
    needs_llm = (
        not retrieval_only
        or config.get("query_strategy", "none") != "none"
        or config.get("agentic", False)
    )

    llm = None
    if needs_llm:
        from src.generation.llm_client import OpenAIClient, OllamaClient

        provider = cfg("generation.provider", "openai")
        model = cfg("generation.model", "gpt-4o-mini")
        if provider == "ollama":
            llm = OllamaClient(model=model)
        else:
            llm = OpenAIClient(model=model)

    emb_provider = cfg("ingestion.embedding.provider", "local")
    emb_model = cfg("ingestion.embedding.model", "all-MiniLM-L6-v2")
    engine = EmbeddingEngine(provider=emb_provider, model_name=emb_model)

    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=512, overlap=64, strategy=ChunkingStrategy.RECURSIVE
    )
    store = FAISSVectorStore(dimension=engine.dimension)

    if config["hybrid"]:
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

    chain = None
    qp = None
    if llm is not None:
        from src.retrieval.query_processor import QueryProcessor
        from src.generation import RAGChain, AgenticRAGChain

        strategy = config.get("query_strategy", "none")
        if strategy != "none":
            qp = QueryProcessor(llm=llm, strategy=strategy)

        if config.get("agentic"):
            chain = AgenticRAGChain(
                retriever=retriever,
                llm=llm,
                query_processor=qp,
                max_correction_rounds=2,
                adaptive_retrieval=False,  # Disable for eval (no chat history)
            )
        else:
            chain = RAGChain(retriever=retriever, llm=llm, query_processor=qp)

    return {
        "loader": loader,
        "chunker": chunker,
        "engine": engine,
        "store": store,
        "retriever": retriever,
        "chain": chain,
        "llm": llm,
    }


# ── Evaluation Logic ─────────────────────────────────────────────


def ingest_documents(pipeline: dict) -> None:
    """Load, chunk, embed, and index all evaluation documents."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    loader = pipeline["loader"]
    chunker = pipeline["chunker"]
    engine = pipeline["engine"]
    store = pipeline["store"]
    retriever = pipeline["retriever"]

    all_chunks = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for filename, content in DOCUMENTS.items():
            path = Path(tmp_dir) / filename
            path.write_text(content, encoding="utf-8")
            doc = loader.load_file(str(path))
            chunks = chunker.chunk_documents([doc])
            all_chunks.extend(chunks)

    embedded = engine.embed_chunks(all_chunks)
    store.add(embedded)

    if isinstance(retriever, HybridRetriever):
        retriever.index_sparse(all_chunks)

    return all_chunks


def evaluate_retrieval(
    pipeline: dict, case: EvalCase
) -> Tuple[RetrievalMetrics, object]:
    """Evaluate retrieval quality for a single test case."""
    retriever = pipeline["retriever"]

    t0 = time.time()
    result = retriever.retrieve(case.question)
    latency = (time.time() - t0) * 1000

    # Compute chunk recall: how many expected keywords appear in retrieved chunks?
    all_chunk_text = " ".join(r.chunk.text for r in result.results).lower()
    hits = sum(1 for kw in case.chunk_keywords if kw.lower() in all_chunk_text)
    chunk_recall = hits / len(case.chunk_keywords) if case.chunk_keywords else 1.0

    # Top-1 relevance: does the top chunk contain at least one expected keyword?
    top1_relevant = False
    if result.results:
        top1_text = result.results[0].chunk.text.lower()
        top1_relevant = any(kw.lower() in top1_text for kw in case.chunk_keywords)

    metrics = RetrievalMetrics(
        chunk_recall=chunk_recall,
        top1_relevant=top1_relevant,
        num_results=len(result.results),
        latency_ms=latency,
    )
    return metrics, result


def evaluate_generation(pipeline: dict, case: EvalCase) -> GenerationMetrics:
    """Evaluate end-to-end generation quality for a single test case."""
    chain = pipeline["chain"]

    t0 = time.time()
    response = chain.query(question=case.question, mode="qa")
    latency = (time.time() - t0) * 1000

    # Keyword coverage: how many expected answer keywords appear in the answer?
    answer_lower = response.answer.lower()
    hits = sum(1 for kw in case.answer_keywords if kw.lower() in answer_lower)
    keyword_coverage = hits / len(case.answer_keywords) if case.answer_keywords else 1.0

    return GenerationMetrics(
        keyword_coverage=keyword_coverage,
        answer_length=len(response.answer),
        latency_ms=latency,
    )


def run_evaluation(
    config_name: str,
    config: dict,
    retrieval_only: bool = False,
    verbose: bool = False,
) -> ConfigResult:
    """Run full evaluation for a single configuration."""
    result = ConfigResult(config_name=config_name, config_label=config["label"])

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {config['label']}")
        print(f"{'─' * 60}")

    # Build pipeline
    pipeline = build_pipeline(config, retrieval_only=retrieval_only)

    # Ingest documents
    if verbose:
        print("  Indexing documents...", end=" ", flush=True)
    t0 = time.time()
    ingest_documents(pipeline)
    if verbose:
        print(f"done ({time.time() - t0:.1f}s)")

    # Evaluate each case
    t_total = time.time()
    for i, case in enumerate(EVAL_CASES):
        if verbose:
            tag = f"[{case.difficulty[0].upper()}]"
            print(f"  {tag} Q{i+1}: {case.question[:50]}...", end=" ", flush=True)

        # Retrieval
        ret_metrics, ret_result = evaluate_retrieval(pipeline, case)

        # Generation (optional)
        gen_metrics = None
        if not retrieval_only:
            gen_metrics = evaluate_generation(pipeline, case)

        case_result = CaseResult(
            question=case.question,
            difficulty=case.difficulty,
            category=case.category,
            retrieval=ret_metrics,
            generation=gen_metrics,
        )
        result.cases.append(case_result)

        if verbose:
            recall_str = f"recall={ret_metrics.chunk_recall:.0%}"
            if gen_metrics:
                kw_str = f"answer={gen_metrics.keyword_coverage:.0%}"
                print(f"{recall_str} | {kw_str}")
            else:
                print(recall_str)

    result.total_latency_ms = (time.time() - t_total) * 1000
    return result


# ── Report Generation ────────────────────────────────────────────


def format_pct(value: float) -> str:
    """Format a float as percentage string."""
    return f"{value * 100:.1f}%"


def generate_report(
    results: Dict[str, ConfigResult],
    retrieval_only: bool = False,
) -> str:
    """Generate a Markdown comparison report."""
    lines = []
    lines.append("# RAG Benchmark Report\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Test cases**: {len(EVAL_CASES)}")
    lines.append(f"**Documents**: {len(DOCUMENTS)}")
    lines.append(f"**Mode**: {'Retrieval only' if retrieval_only else 'End-to-end'}")
    lines.append("")

    # ── Summary table ────────────────────────────────────────────
    lines.append("## Summary\n")

    if retrieval_only:
        lines.append(
            "| Configuration | Chunk Recall | Top-1 Hit Rate | Retrieval Latency |"
        )
        lines.append(
            "|---------------|-------------|----------------|-------------------|"
        )
        for name, r in results.items():
            lines.append(
                f"| {r.config_label} "
                f"| {format_pct(r.avg_chunk_recall)} "
                f"| {format_pct(r.avg_top1_hit)} "
                f"| {r.avg_retrieval_latency:.0f}ms |"
            )
    else:
        lines.append(
            "| Configuration | Chunk Recall | Top-1 Hit | Answer Keywords | Avg Latency |"
        )
        lines.append(
            "|---------------|-------------|-----------|-----------------|-------------|"
        )
        for name, r in results.items():
            lines.append(
                f"| {r.config_label} "
                f"| {format_pct(r.avg_chunk_recall)} "
                f"| {format_pct(r.avg_top1_hit)} "
                f"| {format_pct(r.avg_keyword_coverage)} "
                f"| {r.avg_total_latency:.0f}ms |"
            )

    lines.append("")

    # ── Breakdown by difficulty ──────────────────────────────────
    lines.append("## By Difficulty\n")

    for diff in ["easy", "medium", "hard"]:
        lines.append(f"### {diff.capitalize()}\n")

        if retrieval_only:
            lines.append("| Configuration | Chunk Recall | Top-1 Hit Rate |")
            lines.append("|---------------|-------------|----------------|")
        else:
            lines.append(
                "| Configuration | Chunk Recall | Top-1 Hit | Answer Keywords |"
            )
            lines.append(
                "|---------------|-------------|-----------|-----------------|"
            )

        for name, r in results.items():
            subset = r.by_difficulty(diff)
            if not subset.cases:
                continue

            if retrieval_only:
                lines.append(
                    f"| {r.config_label} "
                    f"| {format_pct(subset.avg_chunk_recall)} "
                    f"| {format_pct(subset.avg_top1_hit)} |"
                )
            else:
                lines.append(
                    f"| {r.config_label} "
                    f"| {format_pct(subset.avg_chunk_recall)} "
                    f"| {format_pct(subset.avg_top1_hit)} "
                    f"| {format_pct(subset.avg_keyword_coverage)} |"
                )

        lines.append("")

    # ── Per-question detail ──────────────────────────────────────
    lines.append("## Per-Question Detail\n")

    config_names = list(results.keys())
    header_cols = " | ".join(results[n].config_label for n in config_names)

    lines.append(f"| # | Question | Difficulty | {header_cols} |")
    sep_cols = " | ".join("---" for _ in config_names)
    lines.append(f"|---|----------|------------|{sep_cols}|")

    for i, case in enumerate(EVAL_CASES):
        q_short = case.question[:40] + ("..." if len(case.question) > 40 else "")
        cols = []
        for name in config_names:
            cr = results[name].cases[i]
            recall = f"{cr.retrieval.chunk_recall:.0%}"
            if cr.generation:
                kw = f"{cr.generation.keyword_coverage:.0%}"
                cols.append(f"R:{recall} A:{kw}")
            else:
                cols.append(f"R:{recall}")

        cols_str = " | ".join(cols)
        lines.append(f"| {i+1} | {q_short} | {case.difficulty} | {cols_str} |")

    lines.append("")

    # ── Improvement delta ────────────────────────────────────────
    if len(results) >= 2:
        lines.append("## Improvement vs Baseline\n")

        baseline_name = list(results.keys())[0]
        baseline = results[baseline_name]

        lines.append(
            "| Configuration | Chunk Recall Δ | Top-1 Δ |"
            + (" Answer Δ |" if not retrieval_only else "")
        )
        lines.append(
            "|---------------|---------------|---------|"
            + ("---------|" if not retrieval_only else "")
        )

        for name, r in results.items():
            if name == baseline_name:
                continue
            cr_delta = r.avg_chunk_recall - baseline.avg_chunk_recall
            t1_delta = r.avg_top1_hit - baseline.avg_top1_hit
            row = f"| {r.config_label} | {cr_delta:+.1%} | {t1_delta:+.1%} |"
            if not retrieval_only:
                kw_delta = r.avg_keyword_coverage - baseline.avg_keyword_coverage
                row += f" {kw_delta:+.1%} |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RAG Benchmark — evaluate pipeline configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=None,
        help="Configurations to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval (no LLM calls, no cost).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save report to file (Markdown). Defaults to stdout.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save raw results to JSON file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress during evaluation.",
    )
    args = parser.parse_args()

    config_names = args.configs or list(CONFIGS.keys())

    # Check if LLM is needed
    if not args.retrieval_only:
        needs_llm = any(
            CONFIGS[c].get("query_strategy", "none") != "none"
            or CONFIGS[c].get("agentic", False)
            for c in config_names
        )
        if needs_llm or not args.retrieval_only:
            provider = os.environ.get("LLM_PROVIDER", "openai")
            if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
                print(
                    "Warning: OPENAI_API_KEY not set. Use --retrieval-only for free eval,"
                )
                print("or set LLM_PROVIDER=ollama for local LLM.")
                print()

    print("=" * 60)
    print("  RAG Benchmark")
    print("=" * 60)
    print(f"  Configs: {', '.join(config_names)}")
    print(f"  Cases:   {len(EVAL_CASES)}")
    print(f"  Mode:    {'Retrieval only' if args.retrieval_only else 'End-to-end'}")
    print("=" * 60)

    # Run evaluations
    results = {}
    for name in config_names:
        config = CONFIGS[name]
        result = run_evaluation(
            config_name=name,
            config=config,
            retrieval_only=args.retrieval_only,
            verbose=args.verbose,
        )
        results[name] = result

        print(
            f"\n  ✓ {config['label']}: "
            f"chunk_recall={format_pct(result.avg_chunk_recall)}, "
            f"top1_hit={format_pct(result.avg_top1_hit)}"
            + (
                f", answer_kw={format_pct(result.avg_keyword_coverage)}"
                if not args.retrieval_only
                else ""
            )
        )

    # Generate report
    report = generate_report(results, retrieval_only=args.retrieval_only)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n  Report saved to: {args.output}")
    else:
        print("\n")
        print(report)

    # Save JSON
    if args.json:
        json_data = {}
        for name, r in results.items():
            json_data[name] = {
                "config_label": r.config_label,
                "avg_chunk_recall": r.avg_chunk_recall,
                "avg_top1_hit": r.avg_top1_hit,
                "avg_keyword_coverage": r.avg_keyword_coverage,
                "avg_retrieval_latency_ms": r.avg_retrieval_latency,
                "avg_total_latency_ms": r.avg_total_latency,
                "total_latency_ms": r.total_latency_ms,
                "cases": [
                    {
                        "question": c.question,
                        "difficulty": c.difficulty,
                        "chunk_recall": c.retrieval.chunk_recall,
                        "top1_relevant": c.retrieval.top1_relevant,
                        "keyword_coverage": (
                            c.generation.keyword_coverage if c.generation else None
                        ),
                    }
                    for c in r.cases
                ],
            }
        Path(args.json).write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  JSON saved to: {args.json}")


if __name__ == "__main__":
    main()
