#!/usr/bin/env python3
"""
CLI tool for indexing documents into the RAG vector store.

Usage:
    python scripts/index_documents.py ./docs/ --save-index ./index
    python scripts/index_documents.py paper.pdf --provider openai
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import DocumentLoader, TextChunker, ChunkingStrategy
from src.ingestion.embedder import EmbeddingEngine
from src.retrieval import FAISSVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index documents for RAG Q&A")
    parser.add_argument(
        "input",
        help="Path to a file or directory of documents",
    )
    parser.add_argument(
        "--save-index",
        default="./index",
        help="Directory to save the FAISS index (default: ./index)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Chunk overlap in characters (default: 64)",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "sentence", "recursive"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openai"],
        default="local",
        help="Embedding provider (default: local)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model name (default: auto)",
    )
    args = parser.parse_args()

    # Initialize components
    loader = DocumentLoader()
    strategy = ChunkingStrategy[args.strategy.upper()]
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        strategy=strategy,
    )
    engine = EmbeddingEngine(
        provider=args.provider,
        model_name=args.model,
    )
    store = FAISSVectorStore(dimension=engine.dimension)

    # Load documents
    input_path = Path(args.input)
    if input_path.is_dir():
        documents = loader.load_directory(str(input_path))
    elif input_path.is_file():
        documents = [loader.load_file(str(input_path))]
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    logger.info(f"Loaded {len(documents)} documents")

    # Chunk
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Embed
    embedded = engine.embed_chunks(chunks)
    logger.info(f"Generated {len(embedded)} embeddings")

    # Store
    store.add(embedded)
    store.save(args.save_index)
    logger.info(f"Index saved to {args.save_index} ({store.size} vectors)")


if __name__ == "__main__":
    main()
