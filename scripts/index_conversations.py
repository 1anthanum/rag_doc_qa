#!/usr/bin/env python3
"""
Build a Digital Self FAISS index from conversation JSON files.

Usage:
    # Index a directory of conversation files
    python scripts/index_conversations.py --input data/conversations/

    # Index a single file
    python scripts/index_conversations.py --input chat_export.json

    # With custom config
    python scripts/index_conversations.py --input data/conversations/ \\
        --config configs/digital_self.yaml

    # Override output directory
    python scripts/index_conversations.py --input data/conversations/ \\
        --output data/my_index/
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Add project root to path so imports work when run as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── ANSI colors ──────────────────────────────────────────────────

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


def info(label: str, value) -> None:
    print(f"  {CYAN}{label}:{RESET} {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Digital Self conversation index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a conversation JSON file or directory of JSON files.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/digital_self.yaml",
        help="Path to YAML config file (default: configs/digital_self.yaml).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Override output directory for the index.",
    )
    args = parser.parse_args()

    header("Digital Self — Build Conversation Index")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Load config
    config_path = args.config if Path(args.config).exists() else None
    if config_path:
        info("Config", config_path)
    else:
        info("Config", "defaults (config file not found)")

    # Build indexer
    from src.digital_self.indexer import DigitalSelfIndexer

    t0 = time.time()
    indexer = DigitalSelfIndexer.from_config(config_path)

    if args.output:
        indexer.persist_dir = Path(args.output)

    info("Embedding model", f"{indexer.engine.provider}/{indexer.engine.model_name}")
    info("Output directory", str(indexer.persist_dir))
    print()

    # Index
    if input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
        info("Input directory", str(input_path))
        info("JSON files found", len(json_files))
        print()

        if not json_files:
            logger.error("No JSON files found in directory.")
            sys.exit(1)

        stats = indexer.index_directory(str(input_path))
    else:
        info("Input file", str(input_path))
        print()
        stats = indexer.index_file(str(input_path))
        stats = {
            "files": 1,
            "turns": stats["turns"],
            "chunks": stats["chunks"],
            "index_size": indexer.store.size,
            "errors": [],
        }

    # Save
    print(f"\n{DIM}Saving index...{RESET}")
    save_path = indexer.save()
    dt = time.time() - t0

    # Summary
    header("Indexing Complete")
    info("Files indexed", stats["files"])
    info("Total turns", stats["turns"])
    info("Total chunks", stats["chunks"])
    info("Index size", f"{stats['index_size']} vectors")
    info("Saved to", save_path)
    info("Time", f"{dt:.1f}s")

    if stats.get("errors"):
        print(f"\n  {YELLOW}Warnings:{RESET}")
        for err in stats["errors"]:
            print(f"    - {err['file']}: {err['error']}")

    print(f"\n  {GREEN}Ready for querying via DigitalSelfConnector.{RESET}")
    print(f"  {DIM}Example:{RESET}")
    print("    from src.digital_self import DigitalSelfConnector")
    print(f'    connector = DigitalSelfConnector.from_config("{args.config}")')
    print("    connector.load_index()")
    print('    response = connector.query("What are my views on X?")')
    print()


if __name__ == "__main__":
    main()
