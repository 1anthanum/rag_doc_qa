"""
Digital Self RAG — conversation-based personal knowledge system.

This module provides an adapter layer that connects the core RAG pipeline
to conversation data (chat exports, dialogue logs). It reuses the existing
ingestion/retrieval/generation infrastructure without modifying it.

Components:
    - ConversationLoader: JSON → Chunk conversion (in src.ingestion)
    - DigitalSelfIndexer: Build and persist FAISS index from conversations
    - DigitalSelfConnector: Query interface with behavioral HyDE support
"""

from .indexer import DigitalSelfIndexer
from .connector import DigitalSelfConnector

__all__ = ["DigitalSelfIndexer", "DigitalSelfConnector"]
